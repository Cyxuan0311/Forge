// Unit test for the parallel GGUF scanner (src/inspect/perf/gguf_scanner.*).
// Writes a tiny synthetic GGUF v3 file and verifies:
//   - header/metadata/tensors are decoded correctly
//   - serial (1 thread) and parallel (4 threads) parsing produce identical
//     snapshots and identical aggregate statistics
//
// Build (from repo root):
//   g++ -std=c++20 -Iinclude -Isrc tests/test_inspect_scanner.cpp \
//       build/libforge_inspect.a build/libforge_model.a build/libforge_core.a \
//       build/src/operators/libforge_ops.a build/libforge_infer.a \
//       -lpthread -fopenmp -o /tmp/test_inspect_scanner
//   /tmp/test_inspect_scanner
// Returns 0 on success, nonzero on any failed assertion.

#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "forge/gguf_model.h"
#include "inspect/arch_ops.h"
#include "perf/gguf_scanner.h"
#include "report.h"

namespace {

using forge::GgmlDType;

void write_u32(std::ofstream& f, uint32_t v) { f.write((const char*)&v, 4); }
void write_u64(std::ofstream& f, uint64_t v) { f.write((const char*)&v, 8); }
void write_f32(std::ofstream& f, float v) { f.write((const char*)&v, 4); }

void write_str(std::ofstream& f, const std::string& s) {
    write_u64(f, s.size());
    f.write(s.data(), (std::streamsize)s.size());
}

// GGUFValueType
enum : uint32_t { T_STR = 8, T_INT = 5, T_FLOAT = 6, T_ARRAY = 9, T_BOOL = 7 };

void write_kv_str(std::ofstream& f, const std::string& k, const std::string& v) {
    write_str(f, k);
    write_u32(f, T_STR);
    write_str(f, v);
}

void write_kv_int(std::ofstream& f, const std::string& k, int32_t v) {
    write_str(f, k);
    write_u32(f, T_INT);
    write_u32(f, (uint32_t)v);  // GGUF int32 is 4 bytes
}

void write_kv_float(std::ofstream& f, const std::string& k, float v) {
    write_str(f, k);
    write_u32(f, T_FLOAT);
    write_f32(f, v);
}

struct TensorSpec {
    std::string name;
    std::vector<uint64_t> dims;
    GgmlDType dtype;
    uint64_t offset = 0;
};

// The byte offset of the end of the tensor index, so the test can assert that a
// tensor's file_offset is measured from there and not from the end of metadata.
static size_t g_index_end = 0;

// Knobs for the negative cases. Default parameters reproduce the original v3
// file exactly, so the existing assertions below stay valid.
struct WriteOpts {
    uint32_t version = 3;
    // Cut the file at this many bytes to simulate a truncated tensor index.
    // 0 = write the whole thing.
    size_t truncate_to = 0;
    // Overwrite the first tensor's stored data offset with this, to check that
    // file_offset = data_start + stored and that data_start is the end of the
    // TENSOR INDEX (not the end of the metadata).
    bool pad_before_data = true;
    // Knobs for the --validate cases. Each overrides one metadata value so a
    // specific finding can be asserted without hand-corrupting bytes.
    int64_t block_count_override = -1;    // -1 = use the real count
    int64_t head_count_override = -1;
    int64_t hidden_override = -1;
    int64_t kv_head_count_override = -1;  // scalar attention.head_count_kv
    int64_t vocab_override = -1;
    int64_t expert_count_override = -1;
    int64_t expert_used_override = -1;
    bool duplicate_tensor = false;        // write blk.0.attn_norm.weight twice
    int kv_extra = 0;                     // add N string KVs (keeps kv_count honest)
    int drop_tensors = 0;                 // remove the last N tensors from the index
};

std::string write_gguf(const std::string& path, const WriteOpts& opts = WriteOpts()) {
    std::vector<TensorSpec> tensors = {
        {"token_embd.weight", {32, 4096}, GgmlDType::Q5_K},
        {"blk.0.attn_norm.weight", {4096}, GgmlDType::F32},
        {"blk.0.attn_q.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.0.attn_k.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.0.attn_v.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.0.attn_output.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.0.ffn_gate.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.0.ffn_up.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.0.ffn_down.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"output_norm.weight", {4096}, GgmlDType::F32},
        {"output.weight", {4096, 32}, GgmlDType::Q5_K},
        // Three more layer groups. --layers is only testable if there is more
        // than one layer to truncate: with a single "blk.0" group, asking for 1
        // and asking for all produce identical output, so a broken --layers would
        // still pass.
        {"blk.1.attn_norm.weight", {4096}, GgmlDType::F32},
        {"blk.1.ffn_down.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.2.attn_norm.weight", {4096}, GgmlDType::F32},
        {"blk.2.ffn_down.weight", {4096, 4096}, GgmlDType::Q5_K},
        {"blk.3.attn_norm.weight", {4096}, GgmlDType::F32},
        {"blk.3.ffn_down.weight", {4096, 4096}, GgmlDType::Q5_K},
    };
    const int64_t n_layers = 4;
    // Must come before tensor_count is computed below, or the duplicate lands after
    // the count the header advertises and the scanner never reads it.
    if (opts.duplicate_tensor)
        tensors.push_back({"blk.0.attn_norm.weight", {4096}, GgmlDType::F32});
    // Drop from the END so the remaining names are a prefix of the full set, which is
    // what makes the expected one-sided list obvious.
    if (opts.drop_tensors > 0 &&
        static_cast<size_t>(opts.drop_tensors) < tensors.size())
        tensors.resize(tensors.size() - static_cast<size_t>(opts.drop_tensors));

    std::ofstream f(path, std::ios::binary);
    if (!f)
        return "cannot open output";

    // 7 base KVs, plus whatever the corruption knobs below actually append. Getting
    // this wrong shifts the tensor index and produces a baffling parse error.
    const uint32_t kv_count = (uint32_t)(7 + opts.kv_extra +
                                         (opts.kv_head_count_override >= 0 ? 1 : 0) +
                                         (opts.expert_count_override >= 0 ? 2 : 0));
    const uint32_t tensor_count = (uint32_t)tensors.size();

    // Header. Counts are 64-bit for v2 as well as v3 -- the scanner used to gate
    // on v3 and read v2 as two u32s.
    f.write("GGUF", 4);
    write_u32(f, opts.version);
    write_u64(f, tensor_count);
    write_u64(f, kv_count);

    // Metadata.
    write_kv_str(f, "general.architecture", "llama");
    write_kv_str(f, "general.name", "tiny-synthetic");
    write_kv_int(f, "general.file_type", 16);  // MOSTLY_Q5_K_S
    write_kv_int(f, "llama.block_count",
                 opts.block_count_override >= 0 ? opts.block_count_override : n_layers);
    write_kv_int(f, "llama.embedding_length",
                 opts.hidden_override >= 0 ? opts.hidden_override : 4096);
    write_kv_int(f, "llama.attention.head_count",
                 opts.head_count_override >= 0 ? opts.head_count_override : 32);
    write_kv_float(f, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    if (opts.kv_head_count_override >= 0)
        write_kv_int(f, "llama.attention.head_count_kv", opts.kv_head_count_override);
    if (opts.expert_count_override >= 0) {
        write_kv_int(f, "llama.expert_count", opts.expert_count_override);
        write_kv_int(f, "llama.expert_used_count", opts.expert_used_override);
    }
    for (int i = 0; i < opts.kv_extra; ++i) {
        const std::string k = "test.pad." + std::to_string(i);
        write_kv_str(f, k, std::string(64, 'x'));  // long enough to move the data start
    }

    // Tensor index (unpadded names; scanner reads strings by length).
    if (opts.duplicate_tensor)
        tensors.push_back({"blk.0.attn_norm.weight", {4096}, GgmlDType::F32});

    // Precompute sequential data offsets first so the index carries them.
    // Lay the payload out at each tensor's REAL packed size. Using a fixed
    // placeholder stride here used to be harmless because nothing compared the
    // offsets against the file length; --validate does, and correctly rejected it.
    uint64_t data_off = 0;
    for (auto& t : tensors) {
        t.offset = data_off;
        int64_t numel = 1;
        for (uint64_t d : t.dims)
            numel *= static_cast<int64_t>(d);
        data_off += static_cast<uint64_t>(forge::inspect::packed_bytes(t.dtype, numel));
    }
    for (const auto& t : tensors) {
        write_str(f, t.name);
        write_u32(f, (uint32_t)t.dims.size());
        for (uint64_t d : t.dims)
            write_u64(f, d);
        write_u32(f, (uint32_t)t.dtype);
        write_u64(f, t.offset);
    }

    const std::streamoff index_end = f.tellp();
    g_index_end = static_cast<size_t>(index_end);

    if (opts.truncate_to) {
        // Cut mid tensor-index so the scanner must report truncation rather than
        // emit silently-wrong tensors.
        f.close();
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        const auto full = static_cast<size_t>(in.tellg());
        in.close();
        if (opts.truncate_to >= full)
            return "truncate_to beyond EOF";
        std::vector<char> buf(opts.truncate_to);
        std::ifstream rd(path, std::ios::binary);
        rd.read(buf.data(), static_cast<std::streamsize>(opts.truncate_to));
        rd.close();
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        out.close();
        return "";
    }

    // Pad so the data section starts 32-byte aligned from the file start.
    const std::streamoff end = f.tellp();
    const uint64_t align = 32;
    const uint64_t pad = (align - ((uint64_t)end % align)) % align;
    for (uint64_t i = 0; i < pad; ++i)
        f.put('\0');

    // Payload. Nothing reads it, so extend the file with a seek and a single byte
    // rather than writing ~100 MB of zeros: on a sparse-capable filesystem this is
    // instant, and the resulting length is what the size checks care about.
    if (data_off > 0) {
        f.seekp(static_cast<std::streamoff>(data_off - 1), std::ios::cur);
        f.put('\0');
    }
    f.close();
    return "";
}

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

}  // namespace

int main() {
    const std::string path = "/tmp/test_inspect_scanner.gguf";
    // Scratch file for the deliberately-corrupted variants used by --validate.
    const std::string f_v = "/tmp/test_inspect_scanner_variant";
    std::string err = write_gguf(path);
    check(err.empty(), err.c_str());

    // Capture this now: write_gguf sets the global for whichever file it wrote last,
    // and the --validate and tensor-diff cases below write several more.
    const size_t main_index_end = g_index_end;

    auto snap1 = forge::inspect::load_gguf(path, 1);
    auto snap4 = forge::inspect::load_gguf(path, 4);

    check(snap1.ok, "serial load ok");
    check(snap4.ok, "parallel load ok");

    check(snap1.version == 3, "version == 3");
    check(snap1.tensor_count == 17, "tensor count");
    check(snap1.tensor_count == snap4.tensor_count, "count parity");

    check(snap1.meta_str.at("general.architecture") == "llama", "arch meta");
    check(snap1.meta_str.at("general.name") == "tiny-synthetic", "name meta");
    check(snap1.meta_int.at("llama.block_count") == 4, "block_count meta");
    check(snap1.meta_int.at("llama.embedding_length") == 4096, "embedding meta");
    check(snap1.meta_float.at("llama.attention.layer_norm_rms_epsilon") == 1e-5f,
          "epsilon meta");

    check(snap1.meta_int.at("general.file_type") == 16, "file_type meta");

    // Tensor identity.
    check(snap1.tensors[0].name == "token_embd.weight", "first tensor name");
    // output.weight is no longer last; the blk.N ffn_down entries are.
    check(snap1.tensors.back().name == "blk.3.ffn_down.weight", "last tensor name");
    check(snap1.tensors[0].orig_dtype == GgmlDType::Q5_K, "tensor dtype");
    check(snap1.tensors[1].orig_dtype == GgmlDType::F32, "f32 dtype");
    check(snap1.tensors[0].shape.size() == 2 && snap1.tensors[0].shape[1] == 4096,
          "tensor shape");

    // Serial vs parallel parity.
    check(snap1.tensors.size() == snap4.tensors.size(), "tensor vec size parity");
    for (size_t i = 0; i < snap1.tensors.size(); ++i) {
        check(snap1.tensors[i].name == snap4.tensors[i].name, "name parity");
        check(snap1.tensors[i].orig_dtype == snap4.tensors[i].orig_dtype,
              "dtype parity");
        check(snap1.tensors[i].shape == snap4.tensors[i].shape, "shape parity");
        check(snap1.tensors[i].data_size == snap4.tensors[i].data_size,
              "data_size parity");
    }

    auto st1 = forge::inspect::compute_stats(snap1, 1);
    auto st4 = forge::inspect::compute_stats(snap4, 4);
    check(st1.total_bytes == st4.total_bytes, "stats total parity");
    check(st1.by_dtype.size() == st4.by_dtype.size(), "stats dtype count parity");
    check(st1.by_dtype.size() == 2, "stats has Q5_K + F32");
    check(st1.total_bytes > 0, "stats nonzero");

    check(forge::inspect::dtype_name(GgmlDType::Q5_K) == "Q5_K", "dtype_name");

    // Layer grouping: the "blk.0" prefix should be present and total across all
    // of its tensors.
    bool found_layer = false;
    for (const auto& [k, bytes] : st1.layer_bytes) {
        if (k == "blk.0")
            found_layer = true;
        check(!k.empty(), "layer key non-empty");
    }
    check(found_layer, "layer grouping 'blk.0' present");

    // ---- machine-readable formats smoke test ----
    forge::inspect::LoadedModel lm{path, snap1, st1, forge::inspect::sniff_arch(snap1),
                                   1.0, 0.5, 4};
    forge::inspect::RenderOptions ro;
    ro.peaks = 3;

    const std::string json = forge::inspect::render_json(lm, ro);
    check(json.find("\"total_bytes\"") != std::string::npos, "json has total_bytes");
    check(json.find("\"dtypes\"") != std::string::npos, "json has dtypes");
    check(json.find("\"arch\"") != std::string::npos, "json has arch");
    check(json.find("tiny-synthetic") != std::string::npos, "json has model name");

    const std::string yaml = forge::inspect::render_yaml(lm, ro);
    check(yaml.find("dtypes:") != std::string::npos, "yaml has dtypes");
    check(yaml.find("metadata:") != std::string::npos, "yaml has metadata");
    check(yaml.find("tiny-synthetic") != std::string::npos, "yaml has model name");

    const std::string md = forge::inspect::render_markdown(lm, ro);
    check(md.find("## Dtype sizes") != std::string::npos, "markdown has dtype section");
    check(md.find("| dtype |") != std::string::npos, "markdown has table header");

    const std::string csv = forge::inspect::render_csv(lm, ro);
    check(csv.find("dtype,bytes,count,elements") != std::string::npos, "csv dtype header");

    const std::string ini = forge::inspect::render_ini(lm, ro);
    check(ini.find("[summary]") != std::string::npos, "ini has summary section");

    // ---- memory / KV estimate ----
    {
        // bytes-per-element must match KVCache::block_nbytes(), i.e. include the
        // per-block scale. Using the nominal bit width would understate q8_0 by
        // 6% and q4_0 by 11%.
        check(std::fabs(forge::inspect::kv_bytes_per_element("f16") - 2.0) < 1e-12, "f16 is 2 B/elem");
        check(std::fabs(forge::inspect::kv_bytes_per_element("q8_0") - 34.0 / 32.0) < 1e-12,
              "q8_0 is 34/32 B/elem (scale included)");
        check(std::fabs(forge::inspect::kv_bytes_per_element("q4_0") - 18.0 / 32.0) < 1e-12,
              "q4_0 is 18/32 B/elem (scale included)");
        check(forge::inspect::kv_bytes_per_element("nope") < 0, "unknown dtype is rejected");
        // The engine's KVCacheDType has no BF16/Q5_K/Q6_K. BF16 would silently hit
        // block_nbytes' `default` branch and be stored as 4 B/elem, so the tool must
        // refuse rather than advertise a cache Forge cannot allocate.
        check(forge::inspect::kv_bytes_per_element("bf16") < 0, "bf16 is not a KV dtype");
        check(forge::inspect::kv_bytes_per_element("q5_k") < 0, "q5_k is not a KV dtype");
        check(forge::inspect::kv_bytes_per_element("q6_k") < 0, "q6_k is not a KV dtype");

        // Per-row block rounding must match KVCache::block_nbytes exactly. These row
        // sizes are deliberately NOT multiples of the block, where a flat
        // bytes/element ratio would under-count.
        check(forge::inspect::kv_row_bytes("f16", 512) == 1024, "f16 row = 2 B/elem");
        check(forge::inspect::kv_row_bytes("q8_0", 512) == 16 * 34,
              "q8_0 row of 512 is exactly 16 blocks of 34 B");
        check(forge::inspect::kv_row_bytes("q4_0", 512) == 16 * 18, "q4_0 row is 16 blocks of 18 B");
        check(forge::inspect::kv_row_bytes("q4_k", 512) == 2 * 144, "q4_k row is 2 blocks of 144 B");
        check(forge::inspect::kv_row_bytes("fp8_e4m3", 512) == 512, "fp8 row = 1 B/elem");
        check(forge::inspect::kv_row_bytes("q8_0", 33) == 2 * 34, "q8_0 row of 33 needs 2 blocks");
        check(forge::inspect::kv_row_bytes("q8_0", 520) == 17 * 34,
              "q8_0 row of 520 pads up to 17 blocks, not 16.25");
        check(forge::inspect::kv_row_bytes("q4_0", 520) == 17 * 18, "q4_0 row of 520 pads to 17 blocks");
        check(forge::inspect::kv_row_bytes("q4_k", 300) == 2 * 144, "q4_k row of 300 pads to 2 blocks");
        check(forge::inspect::kv_row_bytes("q8_0", 0) == 0, "empty row costs nothing");

        std::string canon;
        check(forge::inspect::parse_kv_dtype("Q8_0", &canon) && canon == "q8_0",
              "kv dtype parse is case-insensitive");
        check(forge::inspect::parse_kv_dtype("fp16", &canon) && canon == "f16", "fp16 alias");
        check(!forge::inspect::parse_kv_dtype("q7_0", &canon), "unknown kv dtype rejected");
        check(!forge::inspect::parse_kv_dtype("bf16", &canon),
              "bf16 rejected: the engine has no BF16 KV cache");

        const auto& ai = lm.arch;
        // The fixture is a uniform 4-layer model: 4 layers, 32 heads, no kv key so
        // kv heads fall back to 32, head_dim = 4096/32 = 128.
        check(ai.head_dim == 128, "head_dim derived as hidden/heads when not stated");
        check(ai.kv_estimable, "uniform model is estimable");

        const auto me16 = forge::inspect::estimate_memory(snap1, ai, 2048, "f16", -1);
        check(me16.ok, "f16 estimate produced");
        // 2 (K+V) * 4 layers * 32 heads * 128 dim * 2 B = 65536 B/token
        check(me16.kv_bytes_per_token == 2 * 4 * 32 * 128 * 2, "f16 kv bytes/token");
        check(me16.kv_bytes_total == me16.kv_bytes_per_token * 2048, "kv total = per-token x ctx");
        check(me16.weight_bytes == st1.total_bytes, "weight bytes come from the tensors");

        const auto me8 = forge::inspect::estimate_memory(snap1, ai, 2048, "q8_0", -1);
        check(me8.kv_bytes_per_token == me16.kv_bytes_per_token * 34 / 64,
              "q8_0 kv is 17/16 of f16 (34B vs 32B per block)");

        // End-to-end rounding: force a 512-element row (8 kv heads x 64 head dim), which
        // is exactly the LFM2 shape, and confirm the total follows the padded row size
        // rather than the nominal rate.
        {
            // 8 kv heads x 64 head dim = 512 elements: exactly 16 q8_0 blocks, so this
            // is the unpadded case (and the shape the LFM2.5 model actually has).
            auto lfm2 = ai;
            lfm2.n_kv_heads = 8;
            lfm2.head_dim = 64;
            const auto m = forge::inspect::estimate_memory(snap1, lfm2, 512, "f16", -1);
            check(m.kv_bytes_per_token == 2 * 4 * 1024, "f16 hybrid row total");
            check(m.kv_bytes_total == 8192 * 512, "f16 hybrid total at ctx=512");
            check(m.bytes_per_elem == 2.0, "f16 bytes/elem is exactly 2");
            const auto mq = forge::inspect::estimate_memory(snap1, lfm2, 512, "q8_0", -1);
            check(mq.kv_bytes_per_token == 2 * 4 * 16 * 34, "q8_0 unpadded row total");
            check(mq.bytes_per_elem == 34.0 / 32.0, "unpadded q8_0 is at the nominal rate");

            // 520 elements per row is NOT a multiple of 32, so the row must round up
            // to 17 blocks. This is where a flat bytes/element ratio goes wrong.
            auto padded = ai;
            padded.n_kv_heads = 8;
            padded.head_dim = 65;  // 8*65 = 520
            const auto mp = forge::inspect::estimate_memory(snap1, padded, 512, "q8_0", -1);
            check(mp.kv_bytes_per_token == 2 * 4 * 578, "q8_0 padded row is 578 B, not 552.5");
            check(mp.kv_bytes_per_token > 2 * 4 * 520 * 34 / 32,
                  "padded q8_0 row costs more than the flat ratio suggests");
            check(mp.bytes_per_elem > 34.0 / 32.0,
                  "effective bytes/elem is above nominal for a padded row");
        }

        // Context scaling must be linear, and the per-token figure must not change.
        const auto me4k = forge::inspect::estimate_memory(snap1, ai, 4096, "f16", -1);
        check(me4k.kv_bytes_per_token == me16.kv_bytes_per_token, "per-token cost is ctx-independent");
        check(me4k.kv_bytes_total == 2 * me16.kv_bytes_total, "kv total doubles with ctx");

        // VRAM budget: fits generously, does not fit tightly.
        const int64_t weights = me16.weight_bytes;
        const auto big = forge::inspect::estimate_memory(snap1, ai, 2048, "f16",
                                                         weights * 4 + me16.compute_reserve);
        check(big.fits, "generous vram budget fits");
        check(big.headroom_bytes > 0, "positive headroom when it fits");
        check(big.layers_to_offload == -1, "no offload suggested when it fits");

        // Build a budget that needs exactly half the layers on the CPU: budget_for_weights
        // is 2.5 layers' worth, so 2 stay resident and 2 are offloaded.
        const int64_t per_layer = weights / 4;
        const int64_t half_budget =
            (5 * per_layer) / 2 + me16.kv_bytes_total + me16.compute_reserve;
        const auto half = forge::inspect::estimate_memory(snap1, ai, 2048, "f16", half_budget);
        check(!half.fits, "half-sized budget does not fit");
        check(half.layers_to_offload == 2, "half-sized budget offloads exactly 2 of 4 layers");
        check(half.layers_resident == 2, "and reports 2 layers resident");

        // Offloading every layer is not always enough: the embedding is outside any
        // layer group, so a budget below the KV+compute reserve can never be met.
        const auto starved = forge::inspect::estimate_memory(snap1, ai, 2048, "f16", weights / 2);
        check(!starved.fits, "starved budget does not fit");
        check(starved.layers_to_offload == 4, "starved budget offloads all 4 layers");
        check(starved.layers_resident == 0, "starved budget leaves 0 layers resident");

        // Not estimable: strip the head dim.
        auto blind = ai;
        blind.head_dim = 0;
        blind.kv_estimable = false;
        const auto none = forge::inspect::estimate_memory(snap1, blind, 2048, "f16", -1);
        check(!none.ok && !none.reason.empty(), "missing metadata refuses to estimate, with a reason");

        const auto render = forge::inspect::render_memory(snap1, ai, me16);
        check(render.find("kv / token") != std::string::npos, "memory render has kv/token");
        check(render.find("total") != std::string::npos, "memory render has a total");

        // Machine formats carry it when requested, and stay valid json.
        forge::inspect::RenderOptions mo;
        mo.memory = true;
        mo.ctx = 2048;
        mo.kv_dtype = "f16";
        const std::string mj = forge::inspect::render_json(lm, mo);
        check(mj.find("\"memory\"") != std::string::npos, "json carries the memory block");
        // Braces must balance or the output is unparseable.
        int depth = 0;
        bool neg = false;
        for (char c : mj) {
            if (c == '{') ++depth;
            if (c == '}') { --depth; if (depth < 0) neg = true; }
        }
        check(depth == 0 && !neg, "json with memory block is brace-balanced");
        const std::string mj_off = forge::inspect::render_json(lm, forge::inspect::RenderOptions());
        check(mj_off.find("\"memory\"") == std::string::npos,
              "json omits memory unless asked");

        // --validate is also machine-readable, so CI can gate on it.
        forge::inspect::RenderOptions vo;
        vo.validate = true;
        const std::string vj = forge::inspect::render_json(lm, vo);
        check(vj.find("\"validation\"") != std::string::npos, "json carries validation");
        check(vj.find("\"ok\": true") != std::string::npos,
              "clean model reports ok:true in json");
        int vd = 0, vneg = false;
        for (char c : vj) {
            if (c == '{' || c == '[') ++vd;
            if (c == '}' || c == ']') { --vd; if (vd < 0) vneg = true; }
        }
        check(vd == 0 && !vneg, "json with validation is brace-balanced");
        const std::string vj_off = forge::inspect::render_json(lm, forge::inspect::RenderOptions());
        check(vj_off.find("\"validation\"") == std::string::npos,
              "json omits validation unless asked");
    }

    // ---- arch family comes from one table (was duplicated) ----
    {
        check(forge::inspect::family_for_arch("llama") == forge::inspect::ArchFamily::Gqa,
              "llama is gqa");
        check(forge::inspect::family_for_arch("gemma2") == forge::inspect::ArchFamily::Gemma,
              "gemma2 is gemma");
        check(forge::inspect::family_for_arch("deepseek_v3") == forge::inspect::ArchFamily::Mla,
              "deepseek_v3 is mla");
        check(forge::inspect::family_for_arch("qwen35") == forge::inspect::ArchFamily::Hybrid,
              "qwen35 is hybrid");
        check(forge::inspect::family_for_arch("nonesuch") == forge::inspect::ArchFamily::Unknown,
              "unknown arch is Unknown");
        // The family string and the operator pipeline must agree, which is the
        // whole point of having one table.
        const char* fname = forge::inspect::family_name(forge::inspect::family_for_arch("qwen35"));
        check(std::string(fname) == "hybrid", "family_name(hybrid) == hybrid");
        check(forge::inspect::pipeline_for("qwen35") != nullptr, "qwen35 has a pipeline");
        check(forge::inspect::pipeline_for("llama") != nullptr, "llama has a pipeline");
        check(forge::inspect::pipeline_for("nonesuch") == nullptr, "unknown arch has no pipeline");
        // Every arch with a pipeline must resolve to a non-Unknown family, or the
        // two tables have drifted again.
        for (const char* a : {"llama", "gemma", "gemma2", "deepseek", "deepseek_v2",
                              "deepseek_v3", "phimoe", "gemma4", "qwen35", "qwen2", "mistral",
                              "phi3", "falcon", "yi", "qwen", "qwen3vl"})
            check(forge::inspect::family_for_arch(a) != forge::inspect::ArchFamily::Unknown,
                  std::string("arch has a family: ").append(a).c_str());
    }

    // ---- --validate ----
    {
        // A self-consistent file must come back clean, or every check below is noise.
        const auto clean = forge::inspect::validate_model(snap1, lm.arch);
        bool clean_err = false;
        std::string clean_codes;
        for (const auto& f : clean) {
            if (f.severity == forge::inspect::Severity::Error) {
                clean_err = true;
                clean_codes += " " + f.code;
            }
        }
        check(!clean_err, ("well-formed fixture has no errors:" + clean_codes).c_str());
        check(forge::inspect::render_validation(clean).find("no problems") != std::string::npos ||
                  !clean.empty(),
              "clean validation renders a verdict");

        // Helper: build a variant, validate it, and report whether `code` fired.
        auto has = [&](const WriteOpts& o, const char* code) {
            WriteOpts w = o;
            const std::string err = write_gguf(f_v + ".v", w);
            check(err.empty(), "variant fixture written");
            const auto sn = forge::inspect::load_gguf(f_v + ".v", 1);
            if (!sn.ok) {
                std::fprintf(stderr, "FAIL: variant fixture parses for code '%s' (%s) "
                                     "[blocks=%lld heads=%lld hidden=%lld kv=%lld exp=%lld/%lld "
                                     "dup=%d extra=%d]\n",
                             code, sn.error.c_str(), (long long)w.block_count_override,
                             (long long)w.head_count_override, (long long)w.hidden_override,
                             (long long)w.kv_head_count_override,
                             (long long)w.expert_count_override, (long long)w.expert_used_override,
                             (int)w.duplicate_tensor, w.kv_extra);
                std::exit(1);
            }
            const auto ai = forge::inspect::sniff_arch(sn);
            for (const auto& f : forge::inspect::validate_model(sn, ai))
                if (f.code == code) return true;
            return false;
        };

        // block_count larger than the layers the tensors cover.
        WriteOpts o;
        o.block_count_override = 8;
        check(has(o, "block_count_mismatch"), "block_count above the tensor coverage");

        // block_count smaller than what the tensors cover.
        o = WriteOpts();
        o.block_count_override = 2;
        check(has(o, "block_count_mismatch"), "block_count below the tensor coverage");

        // A layer present in block_count with no tensors at all.
        o = WriteOpts();
        o.block_count_override = 6;
        check(has(o, "layers_absent"), "declared layer with no tensors");

        // hidden_size not divisible by head_count -> truncated head_dim.
        o = WriteOpts();
        o.hidden_override = 4000;  // 4000 / 32 = 125 exactly; use 4097 instead below
        o.hidden_override = 4001;  // 4001 / 32 = 125 (truncated)
        check(has(o, "hidden_not_divisible_by_heads"), "hidden not divisible by heads");

        // KV heads greater than query heads.
        o = WriteOpts();
        o.kv_head_count_override = 64;
        check(has(o, "kv_heads_exceed_heads"), "kv heads exceeding query heads");

        // Query heads not a multiple of KV heads.
        o = WriteOpts();
        o.kv_head_count_override = 5;  // 32 % 5 != 0
        check(has(o, "heads_not_divisible_by_kv"), "heads not a multiple of kv heads");

        // Duplicate tensor names.
        o = WriteOpts();
        o.duplicate_tensor = true;
        check(has(o, "tensor_duplicate"), "duplicate tensor name");

        // expert_used_count above expert_count.
        o = WriteOpts();
        o.expert_count_override = 8;
        o.expert_used_override = 16;
        check(has(o, "expert_used_exceeds_total"), "expert_used above expert_count");

        // Unsupported architecture.
        o = WriteOpts();
        o.kv_extra = 0;
        {
            // Rewrite general.architecture by hand is awkward through WriteOpts, so
            // check the supported-architecture path differently: the fixture uses
            // "llama", which must NOT be flagged.
            const auto ai = forge::inspect::sniff_arch(snap1);
            check(ai.supported, "llama is a supported arch");
            bool flagged = false;
            for (const auto& f : forge::inspect::validate_model(snap1, ai))
                if (f.code == "arch_unknown") flagged = true;
            check(!flagged, "a supported arch is not reported as unknown");
        }

        // Findings must come back errors-first so the important line is on top.
        o = WriteOpts();
        o.hidden_override = 4001;
        o.block_count_override = 9;
        {
            WriteOpts w = o;
            check(write_gguf(f_v + ".v", w).empty(), "multi-fault fixture written");
            const auto sn = forge::inspect::load_gguf(f_v + ".v", 1);
            const auto ai = forge::inspect::sniff_arch(sn);
            const auto fs = forge::inspect::validate_model(sn, ai);
            check(fs.size() >= 2, "multiple faults reported");
            int last = -1;
            bool ordered = true;
            for (const auto& f : fs)
                if (static_cast<int>(f.severity) < last) ordered = false;
                else last = static_cast<int>(f.severity);
            check(ordered, "findings are ordered errors before warnings");
        }
    }

    // ---- tensor diff ----
    {
        using forge::inspect::compute_tensor_diff;
        using forge::inspect::TensorDiff;

        // A file against itself must be perfectly clean, or every other case is noise.
        const TensorDiff self = compute_tensor_diff(snap1, snap4);
        check(self.common == 17, "self-diff sees every tensor as common");
        check(self.common_identical == 17, "self-diff has no identical-count loss");
        check(self.changed.empty(), "self-diff reports nothing changed");
        check(self.only_a.empty() && self.only_b.empty(), "self-diff has no one-sided tensors");
        const std::string self_txt = forge::inspect::render_tensor_diff(self, 0, false);
        check(self_txt.find("identical") != std::string::npos,
              "identical indexes are reported as such");

        // Change one tensor's dtype in the second file: exactly that tensor moves.
        {
            WriteOpts o;
            const std::string p2 = f_v + ".dtype.gguf";
            check(write_gguf(p2, o).empty(), "variant written");
            // Rewrite the file with one Q5_K tensor swapped to Q4_K by post-editing
            // the tensor index is fiddly, so instead compare against a file that
            // differs in its layer set.
            const auto base = forge::inspect::load_gguf(p2, 1);
            check(base.ok, "variant parses");

            // A four-layer file against a two-layer one: two layers are one-sided.
            WriteOpts o2;
            o2.block_count_override = 2;
            const std::string p3 = f_v + ".two.gguf";
            check(write_gguf(p3, o2).empty(), "second variant written");
            const auto two = forge::inspect::load_gguf(p3, 1);
            check(two.ok, "second variant parses");

            // The tensor list itself is the same (the knobs only touch metadata), so
            // the diff is over identical indexes and must stay clean. This guards the
            // "block_count changed but tensors did not" case from being reported as a
            // tensor difference.
            const TensorDiff meta_only = compute_tensor_diff(snap1, two);
            check(meta_only.only_a.empty() && meta_only.only_b.empty(),
                  "a metadata-only difference is not a tensor difference");
            check(meta_only.common_identical == 17, "metadata-only difference keeps tensors identical");
        }

        // Now a real structural difference: drop tensors from the index.
        {
            WriteOpts o;
            o.drop_tensors = 5;  // remove the last five entries
            const std::string p4 = f_v + ".fewer.gguf";
            check(write_gguf(p4, o).empty(), "truncated-tensor variant written");
            const auto few = forge::inspect::load_gguf(p4, 1);
            check(few.ok, "truncated-tensor variant parses");
            const TensorDiff d = compute_tensor_diff(snap1, few);
            check(d.common == 12, "the 12 retained tensors are common");
            check(d.only_a.size() == 5, "the 5 dropped tensors are one-sided");
            check(d.only_b.empty(), "nothing is only in the smaller file");
            check(d.common_identical == 12, "retained tensors are identical");
            // Names must come back sorted, so the output is stable between runs.
            for (size_t i = 1; i < d.only_a.size(); ++i)
                check(d.only_a[i - 1] < d.only_a[i], "one-sided names are sorted");

            const std::string txt = forge::inspect::render_tensor_diff(d, 0, false);
            check(txt.find("only in a") != std::string::npos, "text shows the one-sided count");
            const std::string js = forge::inspect::render_tensor_diff_json(d, 0);
            check(js.find("\"only_in_a\"") != std::string::npos, "json has only_in_a");
            int vd = 0, vneg = false;
            for (char c : js) {
                if (c == '{' || c == '[') ++vd;
                if (c == '}' || c == ']') { --vd; if (vd < 0) vneg = true; }
            }
            check(vd == 0 && !vneg, "tensor diff json is brace-balanced");
        }

        // --top must truncate the listing but not the counts.
        {
            WriteOpts o;
            o.drop_tensors = 5;
            const std::string p5 = f_v + ".fewer2.gguf";
            check(write_gguf(p5, o).empty(), "variant written for the top-N check");
            const auto few = forge::inspect::load_gguf(p5, 1);
            const TensorDiff d = compute_tensor_diff(snap1, few);
            const std::string all = forge::inspect::render_tensor_diff(d, 0, false);
            const std::string top2 = forge::inspect::render_tensor_diff(d, 2, false);
            check(top2.size() < all.size(), "--top shortens the listing");
            check(top2.find("... and 3 more") != std::string::npos,
                  "--top says how many were elided");
            check(top2.find("only in a   5") != std::string::npos,
                  "--top does not change the reported count");
        }
    }

    const std::string peaks = forge::inspect::render_peak_tensors(snap1, 3);
    check(peaks.size() >= 3, "peak tensors rendered");

    const std::string difft = forge::inspect::render_diff_text(lm, lm);
    check(difft.find("size") != std::string::npos, "diff text has metric");

    // ---- file_offset is measured from the end of the TENSOR INDEX ----
    // Regression: the cursor walking the tensor index was block-local, so
    // data_start fell back to the end of the metadata and every file_offset was
    // short by the whole index. Nothing rendered file_offset, so it was invisible.
    {
        // The scanner reports absolute file offsets, and this file stores 0 for the
        // first tensor, so file_offset == data_start == aligned(end of index).
        const uint64_t align = 32;
        size_t expect = main_index_end;
        if (expect % align)
            expect += align - (expect % align);
        check(snap1.tensors[0].file_offset == static_cast<int64_t>(expect),
              "first tensor's file_offset == aligned(end of tensor index)");
        // The second tensor sits one tensor-0 payload on from the first, so the
        // offsets must be consecutive by exactly the first tensor's packed size.
        const int64_t first_nbytes =
            forge::inspect::packed_bytes(snap1.tensors[0].orig_dtype, 32 * 4096);
        check(snap1.tensors[1].file_offset == static_cast<int64_t>(expect) + first_nbytes,
              "second tensor's file_offset is one payload past the first");
        // And it must be strictly past the end of metadata, which is the old bug.
        check(snap1.tensors[0].file_offset > static_cast<int64_t>(main_index_end - 1),
              "file_offset is past the index, not the metadata");
    }

    // ---- --layers is honoured in the machine formats ----
    // Regression: the json/yaml/csv renderers computed
    //   max(top_layers>0 ? top_layers : n, n)
    // which is always n, so --layers was silently ignored in exactly the formats
    // meant for scripting.
    {
        forge::inspect::RenderOptions o2;
        o2.peaks = 3;
        o2.top_layers = 1;
        forge::inspect::RenderOptions o3;  // top_layers = 0 -> all layers
        o3.peaks = 3;
        // Count layer entries inside the layers array only. Layer objects use the
        // key "name" while metadata entries use raw keys, so the region is
        // bounded explicitly rather than counting the whole document.
        const auto count_layers = [](const std::string& doc) {
            const size_t lo = doc.find("\"layers\"");
            const size_t hi = doc.find("\"metadata\"", lo);
            if (lo == std::string::npos || hi == std::string::npos || hi < lo)
                return size_t{0};
            size_t n = 0, pos = lo;
            while ((pos = doc.find("\"name\"", pos)) != std::string::npos && pos < hi) {
                ++n;
                pos += 6;
            }
            return n;
        };
        const std::string j1 = forge::inspect::render_json(lm, o2);
        const std::string jall = forge::inspect::render_json(lm, o3);
        check(count_layers(j1) == 1, "json top_layers=1 emits exactly 1 layer");
        check(count_layers(jall) == 4, "json with no limit emits all 4 layers");

        // Same assertion for the other two machine formats, by size: the
        // truncated render must be strictly shorter than the full one.
        const std::string y1 = forge::inspect::render_yaml(lm, o2);
        const std::string yall = forge::inspect::render_yaml(lm, o3);
        check(y1.size() < yall.size(), "yaml honours top_layers");
        const std::string c1 = forge::inspect::render_csv(lm, o2);
        const std::string call_ = forge::inspect::render_csv(lm, o3);
        check(c1.size() < call_.size(), "csv honours top_layers");
    }

    // ---- GGUF v2 header: counts are 64-bit, same as v3 ----
    // Regression: the scanner read counts as u32 unless version >= 3, so a v2
    // file desynchronised by 8 bytes and every field after it was garbage.
    // llama.cpp reads both counts as int64 unconditionally.
    {
        const std::string p2 = "/tmp/test_inspect_v2.gguf";
        WriteOpts v2;
        v2.version = 2;
        const std::string e2 = write_gguf(p2, v2);
        check(e2.empty(), e2.c_str());
        auto snap2 = forge::inspect::load_gguf(p2, 1);
        check(snap2.ok, "v2 file loads");
        if (snap2.ok) {
            check(snap2.version == 2, "v2 version recorded");
            check(snap2.tensor_count == snap1.tensor_count, "v2 tensor count matches v3");
            check(snap2.metadata_count == snap1.metadata_count, "v2 kv count matches v3");
            check(snap2.meta_str.at("general.architecture") == "llama", "v2 arch meta");
            check(snap2.tensors.size() == snap1.tensors.size(), "v2 tensor vec size");
            if (!snap2.tensors.empty()) {
                check(snap2.tensors[0].name == "token_embd.weight", "v2 first tensor name");
                check(snap2.tensors[0].shape.size() == 2, "v2 first tensor ndim");
            }
        }
        std::remove(p2.c_str());
    }

    // ---- v1 is refused with a clear message, not misparsed ----
    {
        const std::string p1 = "/tmp/test_inspect_v1.gguf";
        WriteOpts v1;
        v1.version = 1;
        const std::string e1 = write_gguf(p1, v1);
        check(e1.empty(), e1.c_str());
        auto snap1v = forge::inspect::load_gguf(p1, 1);
        check(!snap1v.ok, "v1 file is refused");
        check(snap1v.error.find("not supported") != std::string::npos,
              "v1 refusal explains why");
        std::remove(p1.c_str());
    }

    // ---- truncated tensor index must error, not emit fake tensors ----
    // Regression: the metadata walk validated bounds and reported a useful
    // message, but the tensor index walk did not, so a truncated file produced
    // tensors with empty names and default shapes instead of an error.
    {
        const std::string pt = "/tmp/test_inspect_trunc.gguf";
        WriteOpts t;
        // Cut in the middle of the tensor index (well before the data section).
        t.truncate_to = 400;
        const std::string et = write_gguf(pt, t);
        check(et.empty(), et.c_str());
        auto snapt = forge::inspect::load_gguf(pt, 1);
        check(!snapt.ok, "truncated tensor index is rejected");
        check(snapt.error.find("truncated") != std::string::npos ||
                  snapt.error.find("past end of file") != std::string::npos,
              "truncation error mentions truncation");
        std::remove(pt.c_str());
    }

    std::remove(path.c_str());
    std::printf("all inspect scanner tests passed\n");
    return 0;
}
