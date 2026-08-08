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

std::string write_gguf(const std::string& path) {
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
    };
    const int64_t n_layers = 1;

    std::ofstream f(path, std::ios::binary);
    if (!f)
        return "cannot open output";

    const uint32_t kv_count = 7;
    const uint32_t tensor_count = (uint32_t)tensors.size();

    // Header (GGUF v3).
    f.write("GGUF", 4);
    write_u32(f, 3);
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
    check(snap1.tensor_count == 11, "tensor count");
    check(snap1.tensor_count == snap4.tensor_count, "count parity");

    check(snap1.meta_str.at("general.architecture") == "llama", "arch meta");
    check(snap1.meta_str.at("general.name") == "tiny-synthetic", "name meta");
    check(snap1.meta_int.at("llama.block_count") == 1, "block_count meta");
    check(snap1.meta_int.at("llama.embedding_length") == 4096, "embedding meta");
    check(snap1.meta_float.at("llama.attention.layer_norm_rms_epsilon") == 1e-5f,
          "epsilon meta");

    check(snap1.meta_int.at("general.file_type") == 16, "file_type meta");

    // Tensor identity.
    check(snap1.tensors[0].name == "token_embd.weight", "first tensor name");
    check(snap1.tensors.back().name == "output.weight", "last tensor name");
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

    std::remove(path.c_str());
    std::printf("all inspect scanner tests passed\n");
    return 0;
}
