"""
GGUF file format reader.

Parses GGUF v2+ binary files: magic, header, metadata KV pairs,
and tensor index (name, dims, dtype, offset).
"""

import struct
import numpy as np

GGUF_MAGIC = 0x46554747

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15

GGML_TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0",
    GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_Q8_K: "Q8_K",
}


def ggml_type_name(t):
    return GGML_TYPE_NAMES.get(t, f"UNKNOWN({t})")


def read_gguf_string(data, offset):
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset : offset + length].decode("utf-8")
    offset += length
    return s, offset


def read_gguf_value(data, offset, vtype):
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack_from("<B", data, offset)[0], offset + 1
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack_from("<B", data, offset)[0] != 0, offset + 1
    elif vtype == GGUF_TYPE_STRING:
        return read_gguf_string(data, offset)
    elif vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        arr_len = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        arr = []
        for _ in range(arr_len):
            val, offset = read_gguf_value(data, offset, elem_type)
            arr.append(val)
        return arr, offset
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def parse_gguf(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    offset = 0
    magic = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file (magic: {hex(magic)})")

    version = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if version < 2:
        raise ValueError(f"Unsupported GGUF version: {version}")

    tensor_count = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    metadata_kv_count = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    metadata = {}
    for _ in range(metadata_kv_count):
        key, offset = read_gguf_string(data, offset)
        vtype = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        value, offset = read_gguf_value(data, offset, vtype)
        metadata[key] = value

    tensors_info = []
    for _ in range(tensor_count):
        name, offset = read_gguf_string(data, offset)
        n_dims = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dims = [struct.unpack_from("<Q", data, offset + 8 * i)[0] for i in range(n_dims)]
        offset += 8 * n_dims
        dtype_ggml = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tensor_offset = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        tensors_info.append(
            {
                "name": name,
                "dims": dims,
                "dtype": dtype_ggml,
                "offset": tensor_offset,
            }
        )

    alignment = metadata.get("general.alignment", 32)
    data_offset = (offset + alignment - 1) & ~(alignment - 1)

    for t in tensors_info:
        t["data_offset"] = data_offset + t["offset"]

    return metadata, tensors_info, data


def get_tensor_byte_size(dims, dtype):
    numel = int(np.prod(dims))
    if dtype == GGML_TYPE_F32:
        return numel * 4
    elif dtype == GGML_TYPE_F16:
        return numel * 2
    elif dtype == GGML_TYPE_Q4_0:
        return ((numel + 31) // 32) * 18
    elif dtype == GGML_TYPE_Q8_0:
        return ((numel + 31) // 32) * 34
    elif dtype == GGML_TYPE_Q4_1:
        return ((numel + 31) // 32) * 20
    elif dtype == GGML_TYPE_Q6_K:
        return ((numel + 255) // 256) * 210
    elif dtype == GGML_TYPE_Q5_K:
        return ((numel + 255) // 256) * 176
    elif dtype == GGML_TYPE_Q4_K:
        return ((numel + 255) // 256) * 144
    elif dtype == GGML_TYPE_Q3_K:
        return ((numel + 255) // 256) * 110
    elif dtype == GGML_TYPE_Q2_K:
        return ((numel + 255) // 256) * 84
    elif dtype == GGML_TYPE_Q5_0:
        return ((numel + 31) // 32) * 22
    elif dtype == GGML_TYPE_Q5_1:
        return ((numel + 31) // 32) * 24
    else:
        return numel * 4
