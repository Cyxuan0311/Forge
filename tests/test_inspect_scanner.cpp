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
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "forge/gguf_model.h"
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

    std::ofstream f(path, std::ios::binary);
    if (!f)
        return "cannot open output";

    const uint32_t kv_count = 7;
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
    write_kv_int(f, "llama.block_count", n_layers);
    write_kv_int(f, "llama.embedding_length", 4096);
    write_kv_int(f, "llama.attention.head_count", 32);
    write_kv_float(f, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    // Tensor index (unpadded names; scanner reads strings by length).
    // Precompute sequential data offsets first so the index carries them.
    uint64_t data_off = 0;
    for (auto& t : tensors) {
        t.offset = data_off;
        data_off += 1024;  // arbitrary placeholder strides
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

    // Payload (not read by the scanner; placeholder bytes).
    for (uint64_t i = 0; i < data_off; ++i)
        f.put('\0');
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
    std::string err = write_gguf(path);
    check(err.empty(), err.c_str());

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
        size_t expect = g_index_end;
        if (expect % align)
            expect += align - (expect % align);
        check(snap1.tensors[0].file_offset == static_cast<int64_t>(expect),
              "first tensor's file_offset == aligned(end of tensor index)");
        // The second tensor stores 1024, so it must land exactly one stride on.
        check(snap1.tensors[1].file_offset ==
                  static_cast<int64_t>(expect) + 1024,
              "second tensor's file_offset is one stride past the first");
        // And it must be strictly past the end of metadata, which is the old bug.
        check(snap1.tensors[0].file_offset > static_cast<int64_t>(g_index_end - 1),
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
