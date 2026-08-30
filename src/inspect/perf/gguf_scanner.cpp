#include "perf/gguf_scanner.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <unordered_map>

#include "../core/platform.h"
#include "perf/thread_pool.h"

namespace forge::inspect {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;

// GGML tensor type -> human-readable name (superset of what Forge loads).
const char* ggml_dtype_name(GgmlDType dt) {
    switch (dt) {
    case GgmlDType::F32:     return "F32";
    case GgmlDType::F16:     return "F16";
    case GgmlDType::Q4_0:    return "Q4_0";
    case GgmlDType::Q4_1:    return "Q4_1";
    case GgmlDType::Q5_0:    return "Q5_0";
    case GgmlDType::Q5_1:    return "Q5_1";
    case GgmlDType::Q8_0:    return "Q8_0";
    case GgmlDType::Q8_1:    return "Q8_1";
    case GgmlDType::Q2_K:    return "Q2_K";
    case GgmlDType::Q3_K:    return "Q3_K";
    case GgmlDType::Q4_K:    return "Q4_K";
    case GgmlDType::Q5_K:    return "Q5_K";
    case GgmlDType::Q6_K:    return "Q6_K";
    case GgmlDType::Q8_K:    return "Q8_K";
    case GgmlDType::IQ2_XXS: return "IQ2_XXS";
    case GgmlDType::IQ2_XS:  return "IQ2_XS";
    case GgmlDType::IQ3_XXS: return "IQ3_XXS";
    case GgmlDType::IQ1_S:   return "IQ1_S";
    case GgmlDType::IQ4_NL:  return "IQ4_NL";
    case GgmlDType::IQ3_S:   return "IQ3_S";
    case GgmlDType::IQ2_S:   return "IQ2_S";
    case GgmlDType::IQ4_XS:  return "IQ4_XS";
    case GgmlDType::IQ1_M:   return "IQ1_M";
    case GgmlDType::BF16:    return "BF16";
    case GgmlDType::I8:      return "I8";
    case GgmlDType::I16:     return "I16";
    case GgmlDType::I32:     return "I32";
    case GgmlDType::I64:     return "I64";
    case GgmlDType::F64:     return "F64";
    default:                 return "UNKNOWN";
    }
}

struct BlockLayout {
    int64_t block_elements = 1;  // elements per quantization block
    int64_t block_size = 0;      // bytes per block (0 = plain element type)
    int64_t elem_size = 0;       // plain element size (non-quantized)
};

BlockLayout ggml_layout(GgmlDType dt) {
    BlockLayout l;
    switch (dt) {
    case GgmlDType::F32:   l.elem_size = 4; break;
    case GgmlDType::F16:   l.elem_size = 2; break;
    case GgmlDType::BF16:  l.elem_size = 2; break;
    case GgmlDType::I8:    l.elem_size = 1; break;
    case GgmlDType::I16:   l.elem_size = 2; break;
    case GgmlDType::I32:   l.elem_size = 4; break;
    case GgmlDType::I64:   l.elem_size = 8; break;
    case GgmlDType::F64:   l.elem_size = 8; break;
    case GgmlDType::Q4_0:  l = {32, 18, 0}; break;
    case GgmlDType::Q4_1:  l = {32, 20, 0}; break;
    case GgmlDType::Q5_0:  l = {32, 22, 0}; break;
    case GgmlDType::Q5_1:  l = {32, 24, 0}; break;
    case GgmlDType::Q8_0:  l = {32, 34, 0}; break;
    case GgmlDType::Q8_1:  l = {32, 36, 0}; break;
    case GgmlDType::Q2_K:  l = {256, 84, 0}; break;
    case GgmlDType::Q3_K:  l = {256, 110, 0}; break;
    case GgmlDType::Q4_K:  l = {256, 144, 0}; break;
    case GgmlDType::Q5_K:  l = {256, 176, 0}; break;
    case GgmlDType::Q6_K:  l = {256, 210, 0}; break;
    case GgmlDType::Q8_K:  l = {256, 292, 0}; break;
    case GgmlDType::IQ2_XXS: l = {256, 66, 0}; break;
    case GgmlDType::IQ2_XS:  l = {256, 74, 0}; break;
    case GgmlDType::IQ3_XXS: l = {256, 110, 0}; break;
    case GgmlDType::IQ1_S:   l = {256, 34, 0}; break;
    case GgmlDType::IQ4_NL:  l = {32, 18, 0}; break;
    case GgmlDType::IQ3_S:   l = {256, 110, 0}; break;
    case GgmlDType::IQ2_S:   l = {256, 82, 0}; break;
    case GgmlDType::IQ4_XS:  l = {256, 90, 0}; break;
    case GgmlDType::IQ1_M:   l = {256, 40, 0}; break;
    default: break;
    }
    return l;
}

// Number of on-disk bytes for `numel` elements of a given GGML type.
int64_t packed_bytes(GgmlDType dt, int64_t numel) {
    const BlockLayout l = ggml_layout(dt);
    if (l.block_size > 0) {
        const int64_t n_blocks = (numel + l.block_elements - 1) / l.block_elements;
        return n_blocks * l.block_size;
    }
    return numel * l.elem_size;
}

bool is_quantized_dtype(GgmlDType dt) { return ggml_layout(dt).block_size > 0; }

// Read a length-prefixed GGUF string at [off]; returns "" on OOB.
std::string read_gguf_string(const uint8_t* data, size_t size, size_t& off) {
    if (off + 8 > size)
        return "";
    uint64_t len;
    std::memcpy(&len, data + off, 8);
    off += 8;
    if (len > size - off)
        return "";
    std::string s(reinterpret_cast<const char*>(data + off), len);
    off += len;
    return s;
}

// Byte length of a KV value payload located at `off` (after the type tag).
// Pure arithmetic (no allocations), clamped so pass 1 can never walk past EOF.
size_t kv_skip(const uint8_t* data, size_t size, size_t off, uint32_t vtype) {
    const size_t avail = off < size ? size - off : 0;
    auto peek_u64 = [&](size_t at) -> uint64_t {
        if (at + 8 > size)
            return 0;
        uint64_t v;
        std::memcpy(&v, data + at, 8);
        return v;
    };
    switch (vtype) {
        case 0: case 1: case 7: return 1;                       // u8/i8/bool
        case 2: case 3: return 2;                               // u16/i16
        case 4: case 5: case 6: return 4;                       // u32/i32/f32
        case 10: case 11: case 12: return 8;                    // u64/i64/f64
        case 8: {                                               // string
            if (avail < 8)
                return 0;
            const uint64_t len = peek_u64(off);
            const uint64_t room = avail - 8;
            return 8 + (len > room ? room : len);
        }
        case 9: {                                               // array
            if (avail < 12)
                return 0;
            uint32_t arr_type;
            std::memcpy(&arr_type, data + off, 4);
            const uint64_t n = peek_u64(off + 4);
            size_t p = off + 12;
            if (arr_type == 8) {                                // string array
                size_t total = 0;
                for (uint64_t i = 0; i < n && p + total + 8 <= size; ++i)
                    total += 8 + peek_u64(p + total);
                return (p - off) + total;
            }
            if (arr_type == 9) {                                // array of arrays
                size_t cur = p;
                for (uint64_t i = 0; i < n; ++i) {
                    const size_t adv = kv_skip(data, size, cur, 9);
                    if (adv == 0)
                        return (size > off) ? (size - off) : 0;
                    cur += adv;
                }
                return cur - off;
            }
            // Fixed-width elements. bool (7) is 1 byte, NOT 8 — treating it as
            // 8 desynchronised the walk and made later keys decode as garbage.
            const size_t elem = (arr_type <= 1 || arr_type == 7) ? 1
                                : (arr_type <= 3)               ? 2
                                : (arr_type <= 6)               ? 4
                                : (arr_type <= 12)              ? 8
                                                                : 4;
            uint64_t count = n;
            if (count > (avail - 12) / elem)
                count = (avail - 12) / elem;
            return 12 + count * elem;
        }
        default:
            return 8;
    }
}

}  // namespace

// Human-readable name for a GGML tensor type, e.g. "Q4_K".
std::string dtype_name(GgmlDType dt) {
    return ggml_dtype_name(dt);
}

GgufSnapshot load_gguf(const std::string& path, int threads, ThreadPool* pool) {
    GgufSnapshot snap;

    int fd = forge_open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        snap.ok = false;
        snap.error = "cannot open file '" + path + "': " + std::strerror(errno);
        return snap;
    }

    forge_stat_t sb;
    if (forge_fstat(fd, &sb) < 0) {
        forge_close(fd);
        snap.ok = false;
        snap.error = "cannot stat file '" + path + "': " + std::strerror(errno);
        return snap;
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);
    snap.file_size = file_size;

    if (file_size == 0) {
        forge_close(fd);
        snap.ok = false;
        snap.error = "file is empty: " + path;
        return snap;
    }

    void* mapped = forge_mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    forge_close(fd);
    if (mapped == FORGE_MAP_FAILED) {
        snap.ok = false;
        snap.error = "mmap failed: " + std::string(std::strerror(errno));
        return snap;
    }
    // Cold-cache readahead hint. Only the header + tensor index (~a few MB) is
    // read, so prefetch that range after its end is known (see below); the
    // mmap range we actually touch is small relative to model payloads.
    auto munmap_guard = [&]() { forge_munmap(mapped, file_size); };

    const uint8_t* data = static_cast<const uint8_t*>(mapped);
    size_t size = file_size;
    auto fail = [&](const std::string& msg) {
        snap.ok = false;
        snap.error = msg;
        munmap_guard();
        return snap;
    };
    if (size < 8) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "file too small to be a GGUF (%zu bytes)", size);
        return fail(buf);
    }

    uint32_t magic;
    std::memcpy(&magic, data, 4);
    if (magic != GGUF_MAGIC) {
        char buf[96];
        std::snprintf(buf, sizeof(buf),
                      "not a GGUF file (bad magic 0x%08x, expected 0x46554747)", magic);
        return fail(buf);
    }

    uint32_t version;
    std::memcpy(&version, data + 4, 4);
    snap.version = version;

    uint64_t tensor_count, metadata_count;
    size_t offset = 8;
    if (version >= 3) {
        if (offset + 16 > size)
            return fail("truncated GGUF header (need 16 bytes for tensor/metadata counts)");
        std::memcpy(&tensor_count, data + offset, 8);
        offset += 8;
        std::memcpy(&metadata_count, data + offset, 8);
        offset += 8;
    } else {
        if (offset + 8 > size)
            return fail("truncated GGUF header (need 8 bytes for tensor/metadata counts)");
        uint32_t tc, mc;
        std::memcpy(&tc, data + offset, 4);
        offset += 4;
        std::memcpy(&mc, data + offset, 4);
        offset += 4;
        tensor_count = tc;
        metadata_count = mc;
    }
    snap.tensor_count = tensor_count;
    snap.metadata_count = metadata_count;

    // ---- Metadata KV pairs (two-pass: offsets, then parallel decode) ----

    // Pass 0 (sequential, arithmetic only): locate each KV's start offset.
    std::vector<size_t> kv_offsets(metadata_count);
    {
        size_t off = offset;
        uint64_t i = 0;
        for (; i < metadata_count; ++i) {
            kv_offsets[i] = off;
            if (off + 8 > size)
                break;
            uint64_t klen;
            std::memcpy(&klen, data + off, 8);
            off += 8;
            if (klen > size - off)
                break;
            off += static_cast<size_t>(klen);
            if (off + 4 > size)
                break;
            uint32_t vtype;
            std::memcpy(&vtype, data + off, 4);
            off += 4;
            off += kv_skip(data, size, off, vtype);
        }
        if (i < metadata_count) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "truncated metadata at offset %zu (file size %zu, expected %llu "
                          "KV entries)",
                          kv_offsets[i], size, (unsigned long long)metadata_count);
            return fail(buf);
        }
        offset = off;
    }

    // Pass 1 (parallel): decode each KV into per-chunk partial maps.
    const size_t kv_threads = (threads > 0) ? static_cast<size_t>(threads) : 1;
    struct KvPart {
        std::unordered_map<std::string, std::string> meta_str;
        std::unordered_map<std::string, int64_t> meta_int;
        std::unordered_map<std::string, double> meta_float;
        std::unordered_map<std::string, std::vector<int32_t>> meta_int_array;
    };
    const size_t n_kv_parts =
        metadata_count == 0 ? 1 : std::min<size_t>(kv_threads, metadata_count);
    const size_t per_kv = (metadata_count + n_kv_parts - 1) / n_kv_parts;
    std::vector<KvPart> parts(n_kv_parts);
    auto decode_kv = [&](size_t c) {
        KvPart& part = parts[c];
        const size_t begin = c * per_kv;
        const size_t end = std::min<size_t>(metadata_count, begin + per_kv);
        size_t off = begin < metadata_count ? kv_offsets[begin] : 0;
        for (size_t k = begin; k < end; ++k) {
            size_t entry = off;
            std::string key = read_gguf_string(data, size, off);
            uint32_t vtype = 0;
            if (off + 4 <= size) {
                std::memcpy(&vtype, data + off, 4);
                off += 4;
            }
            switch (vtype) {
            case 0: { uint8_t v; if (off + 1 > size) break; std::memcpy(&v, data + off, 1); off += 1; part.meta_int[key] = v; break; }
            case 1: { int8_t v; if (off + 1 > size) break; std::memcpy(&v, data + off, 1); off += 1; part.meta_int[key] = v; break; }
            case 2: { uint16_t v; if (off + 2 > size) break; std::memcpy(&v, data + off, 2); off += 2; part.meta_int[key] = v; break; }
            case 3: { int16_t v; if (off + 2 > size) break; std::memcpy(&v, data + off, 2); off += 2; part.meta_int[key] = v; break; }
            case 4: { uint32_t v; if (off + 4 > size) break; std::memcpy(&v, data + off, 4); off += 4; part.meta_int[key] = v; break; }
            case 5: { int32_t v; if (off + 4 > size) break; std::memcpy(&v, data + off, 4); off += 4; part.meta_int[key] = v; break; }
            case 6: { float v; if (off + 4 > size) break; std::memcpy(&v, data + off, 4); off += 4; part.meta_float[key] = v; break; }
            case 7: { bool v; if (off + 1 > size) break; std::memcpy(&v, data + off, 1); off += 1; part.meta_int[key] = v ? 1 : 0; break; }
            case 8: { std::string v = read_gguf_string(data, size, off); part.meta_str[key] = v; break; }
            case 9: {
                if (off + 12 > size) break;
                uint32_t arr_type;
                std::memcpy(&arr_type, data + off, 4);
                off += 4;
                uint64_t arr_len;
                std::memcpy(&arr_len, data + off, 8);
                off += 8;
                if (arr_type == 5) {
                    std::vector<int32_t> arr(arr_len);
                    for (uint64_t j = 0; j < arr_len; ++j) {
                        if (off + 4 > size) break;
                        std::memcpy(&arr[j], data + off, 4);
                        off += 4;
                    }
                    part.meta_int_array[key] = std::move(arr);
                } else if (arr_type == 7) {
                    std::vector<int32_t> arr(arr_len);
                    for (uint64_t j = 0; j < arr_len; ++j) {
                        if (off + 1 > size) break;
                        bool v;
                        std::memcpy(&v, data + off, 1);
                        off += 1;
                        arr[j] = v ? 1 : 0;
                    }
                    part.meta_int_array[key] = std::move(arr);
                } else if (arr_type == 8) {
                    for (uint64_t j = 0; j < arr_len; ++j)
                        read_gguf_string(data, size, off);
                } else {
                    size_t elem = (arr_type <= 1) ? 1 : (arr_type <= 3) ? 2 : (arr_type <= 6) ? 4 : (arr_type <= 12) ? 8 : 4;
                    off += static_cast<size_t>(arr_len) * elem;
                }
                break;
            }
            case 10: { uint64_t v; if (off + 8 > size) break; std::memcpy(&v, data + off, 8); off += 8; part.meta_int[key] = static_cast<int64_t>(v); break; }
            case 11: { int64_t v; if (off + 8 > size) break; std::memcpy(&v, data + off, 8); off += 8; part.meta_int[key] = v; break; }
            case 12: { double v; if (off + 8 > size) break; std::memcpy(&v, data + off, 8); off += 8; part.meta_float[key] = v; break; }
            default:
                off += 8;
                break;
            }
            (void)entry;
        }
    };
    if (kv_threads > 1 && metadata_count > 64) {
        ThreadPool& p = pool ? *pool : cached_pool(kv_threads);
        p.parallel_for(size_t{0}, parts.size(), decode_kv);
    } else {
        for (size_t c = 0; c < parts.size(); ++c)
            decode_kv(c);
    }
    // Merge partial maps.
    for (const auto& part : parts) {
        for (const auto& [k, v] : part.meta_str)
            snap.meta_str[k] = v;
        for (const auto& [k, v] : part.meta_int)
            snap.meta_int[k] = v;
        for (const auto& [k, v] : part.meta_float)
            snap.meta_float[k] = v;
        for (const auto& [k, v] : part.meta_int_array)
            snap.meta_int_array[k] = v;
    }

    // ---- Tensor index section ----
    const size_t info_start = offset;

    // Pass 1 (sequential): locate each record's start offset. Pure arithmetic.
    std::vector<size_t> record_offsets(tensor_count);
    {
        size_t off = info_start;
        for (uint64_t i = 0; i < tensor_count; ++i) {
            record_offsets[i] = off;
            size_t name_len_off = off;
            uint64_t name_len = 0;
            if (name_len_off + 8 <= size)
                std::memcpy(&name_len, data + name_len_off, 8);
            off = name_len_off + 8 + name_len;
            uint32_t ndim = 0;
            if (off + 4 <= size)
                std::memcpy(&ndim, data + off, 4);
            off += 4;
            off += static_cast<size_t>(ndim) * 8;  // dims
            off += 4 + 8;                          // dtype + offset
        }
    }

    uint64_t alignment = 32;
    auto it_align = snap.meta_int.find("general.alignment");
    if (it_align != snap.meta_int.end())
        alignment = static_cast<uint64_t>(it_align->second);

    size_t data_start = record_offsets.empty() ? info_start : offset;
    if (data_start % alignment != 0)
        data_start += alignment - (data_start % alignment);

    // Prefetch the region we will read (header + tensor index) on cold cache.
    forge_prefetch(mapped, std::min(file_size, data_start));

    snap.tensors.resize(tensor_count);

    // Pass 2 (parallel): decode each record + compute packed size.
    const size_t n_threads = (threads > 0) ? static_cast<size_t>(threads) : 1;
    auto decode_one = [&](size_t i) {
        GgufLoadedTensor& t = snap.tensors[i];
        size_t off = record_offsets[i];
        std::string name = read_gguf_string(data, size, off);
        t.name = name;

        uint32_t ndim = 0;
        if (off + 4 <= size) {
            std::memcpy(&ndim, data + off, 4);
            off += 4;
        }
        t.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            uint64_t dim = 0;
            if (off + 8 <= size) {
                std::memcpy(&dim, data + off, 8);
                off += 8;
            }
            t.shape[d] = static_cast<int64_t>(dim);
        }
        uint32_t dtype_val = 0;
        if (off + 4 <= size) {
            std::memcpy(&dtype_val, data + off, 4);
            off += 4;
        }
        t.orig_dtype = static_cast<GgmlDType>(dtype_val);
        // Report dtype: keep DataType FP32 as placeholder; authoritative dtype is orig_dtype.
        t.dtype = DataType::FP32;
        int64_t file_offset = 0;
        if (off + 8 <= size) {
            std::memcpy(&file_offset, data + off, 8);
            off += 8;
        }
        t.file_offset = static_cast<int64_t>(data_start) + file_offset;
        t.is_gguf_layout = true;

        int64_t numel = 1;
        for (int64_t d : t.shape)
            numel *= d;
        t.data_size = packed_bytes(t.orig_dtype, numel);
    };

    if (n_threads > 1 && tensor_count > 64) {
        ThreadPool& p = pool ? *pool : cached_pool(n_threads);
        p.parallel_for(size_t{0}, tensor_count, decode_one);
    } else {
        for (size_t i = 0; i < tensor_count; ++i)
            decode_one(i);
    }

    munmap_guard();
    snap.ok = true;
    return snap;
}

TensorStats compute_stats(const GgufSnapshot& snap, int threads, ThreadPool* pool) {
    TensorStats stats;
    const size_t n = snap.tensors.size();
    if (n == 0)
        return stats;

    const size_t n_threads = (threads > 0) ? static_cast<size_t>(threads) : 1;
    const size_t n_chunks = std::min(n, n_threads);

    struct Chunk {
        std::unordered_map<uint32_t, int64_t> count;
        std::unordered_map<uint32_t, int64_t> bytes;
        std::unordered_map<uint32_t, int64_t> elements;
        std::unordered_map<std::string, int64_t> layers;
        int64_t total_bytes = 0;
        int64_t total_elements = 0;
        int64_t max_bytes = 0;
        std::string max_name;
    };
    std::vector<Chunk> chunks(n_chunks);

    const size_t per = (n + n_chunks - 1) / n_chunks;
    auto run_chunk = [&](size_t c) {
        Chunk& ch = chunks[c];
        const size_t begin = c * per;
        const size_t end = std::min(n, begin + per);
        for (size_t i = begin; i < end; ++i) {
            const GgufLoadedTensor& t = snap.tensors[i];
            const uint32_t key = static_cast<uint32_t>(t.orig_dtype);
            int64_t numel = 1;
            for (int64_t d : t.shape)
                numel *= d;
            ch.count[key] += 1;
            ch.bytes[key] += t.data_size;
            ch.elements[key] += numel;
            ch.total_bytes += t.data_size;
            ch.total_elements += numel;
            // Layer grouping helpers: recognize "<prefix>.layers.<idx>" (born
            // from the "model.layers" style) and "<prefix>.blk.<idx>" (older
            // gguf convention) and group per full layer-subtree prefix.
            auto group_key = [&t](const std::string& marker) -> std::string {
                const size_t p = t.name.find(marker);
                if (p == std::string::npos)
                    return std::string();
                const size_t rest = p + marker.size();
                size_t endp = t.name.find('.', rest);
                const size_t last = endp == std::string::npos ? t.name.size() : endp;
                return t.name.substr(0, last);
            };
            for (const char* m : {"model.layers.", "blk.", "layers."}) {
                const std::string prefix = group_key(m);
                if (!prefix.empty()) {
                    ch.layers[prefix] += t.data_size;
                    break;
                }
            }
            if (t.data_size > ch.max_bytes) {
                ch.max_bytes = t.data_size;
                ch.max_name = t.name;
            }
        }
    };

    if (n_chunks > 1) {
        ThreadPool& p = pool ? *pool : cached_pool(n_chunks);
        p.parallel_for(size_t{0}, n_chunks, run_chunk);
    } else {
        for (size_t c = 0; c < n_chunks; ++c)
            run_chunk(c);
    }

    // Merge chunks
    std::unordered_map<uint32_t, TensorStats::DtypeStat> merged;
    for (const auto& ch : chunks) {
        for (auto& [k, cnt] : ch.count) {
            auto& d = merged[k];
            d.dtype = static_cast<GgmlDType>(k);
            d.count += cnt;
        }
        for (auto& [k, b] : ch.bytes)
            merged[k].bytes += b;
        for (auto& [k, e] : ch.elements)
            merged[k].elements += e;
        stats.total_bytes += ch.total_bytes;
        stats.total_elements += ch.total_elements;
        if (ch.max_bytes > stats.peak_tensor_bytes) {
            stats.peak_tensor_bytes = ch.max_bytes;
            stats.largest_tensor_name = ch.max_name;
        }
        for (auto& [k, v] : ch.layers)
            stats.layer_bytes.emplace_back(k, v);
    }

    for (auto& [k, d] : merged)
        d.name = ggml_dtype_name(d.dtype);

    // Sort by bytes desc
    std::vector<TensorStats::DtypeStat> vec;
    vec.reserve(merged.size());
    for (auto& [k, d] : merged)
        vec.push_back(d);
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        if (a.bytes != b.bytes)
            return a.bytes > b.bytes;
        return a.dtype < b.dtype;
    });
    stats.by_dtype = std::move(vec);

    // Merge duplicate layer prefixes (parallel chunks may overlap)
    std::sort(stats.layer_bytes.begin(), stats.layer_bytes.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<std::string, int64_t>> merged_layers;
    for (auto& kv : stats.layer_bytes) {
        if (!merged_layers.empty() && merged_layers.back().first == kv.first)
            merged_layers.back().second += kv.second;
        else
            merged_layers.push_back(kv);
    }
    std::sort(merged_layers.begin(), merged_layers.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    stats.layer_bytes = std::move(merged_layers);

    return stats;
}

}  // namespace forge::inspect
