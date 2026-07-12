"""
Forge binary format constants and serialization helpers.

Shared by convert.py and create_test_model.py to avoid duplicating
the .ninf header/tensor info layout.
"""

import struct
import json
import os

NINF_MAGIC = b'NINFMODL'
NINF_VERSION = 0x00010000
HEADER_SIZE = 32
TENSOR_INFO_SIZE = 120
TENSOR_NAME_PADDING = 64
DATA_ALIGNMENT = 64

NINF_DTYPE_FP32 = 0
NINF_DTYPE_FP16 = 1
NINF_DTYPE_Q4_0 = 2
NINF_DTYPE_Q4_1 = 3

DTYPE_NAMES = {0: 'FP32', 1: 'FP16', 2: 'Q4_0', 3: 'Q4_1'}

HEADER_FMT = '=8sIIIIQ'
TENSOR_INFO_FMT = '=64sII4QQQ'


def pack_header(meta_offset, meta_size, tensor_count, data_size):
    return struct.pack(HEADER_FMT,
                       NINF_MAGIC, NINF_VERSION,
                       meta_offset, meta_size,
                       tensor_count, data_size)


def pack_tensor_info(name, dtype, shape, offset, size_bytes):
    name_bytes = name.encode('utf-8')
    name_padded = name_bytes + b'\x00' * (TENSOR_NAME_PADDING - len(name_bytes))
    ndim = len(shape)
    dims = list(shape) + [0] * (4 - ndim)
    info = struct.pack(TENSOR_INFO_FMT,
                       name_padded, dtype, ndim,
                       dims[0], dims[1], dims[2], dims[3],
                       offset, size_bytes)
    assert len(info) == TENSOR_INFO_SIZE
    return info


def compute_data_offset(tensor_info_end):
    return (tensor_info_end + DATA_ALIGNMENT - 1) & ~(DATA_ALIGNMENT - 1)


def build_ninf(meta_dict, tensors, output_path):
    meta_json = json.dumps(meta_dict, separators=(',', ':'))
    header_size = HEADER_SIZE
    meta_offset = header_size
    meta_size = len(meta_json)
    tensor_info_start = meta_offset + meta_size
    tensor_info_total = len(tensors) * TENSOR_INFO_SIZE
    data_start = compute_data_offset(tensor_info_start + tensor_info_total)

    current = data_start
    offsets = []
    sizes = []
    for _, _, _, raw in tensors:
        sz = len(raw)
        offsets.append(current)
        sizes.append(sz)
        current += sz

    total_data_size = current - data_start

    with open(output_path, 'wb') as f:
        f.write(pack_header(meta_offset, meta_size, len(tensors), total_data_size))
        f.write(meta_json.encode('utf-8'))
        for i, (name, dtype, shape, raw) in enumerate(tensors):
            f.write(pack_tensor_info(name, dtype, shape, offsets[i], sizes[i]))
        padding = data_start - (tensor_info_start + tensor_info_total)
        if padding > 0:
            f.write(b'\x00' * padding)
        for _, _, _, raw in tensors:
            f.write(raw)

    return os.path.getsize(output_path)


def read_ninf_header(filepath):
    with open(filepath, 'rb') as f:
        header_data = f.read(HEADER_SIZE)
        magic, version, meta_off, meta_sz, tensor_cnt, data_sz = \
            struct.unpack_from(HEADER_FMT, header_data, 0)
        magic = magic.rstrip(b'\x00').decode()
        if magic != 'NINFMODL':
            raise ValueError(f'Not a .ninf file (magic: {magic})')

        f.seek(meta_off)
        meta_json = json.loads(f.read(meta_sz).decode('utf-8'))

        tensor_info_start = meta_off + meta_sz
        f.seek(tensor_info_start)
        tensors = []
        for _ in range(tensor_cnt):
            raw = f.read(TENSOR_INFO_SIZE)
            name_b, dtype, ndim, d0, d1, d2, d3, offset, size = \
                struct.unpack_from(TENSOR_INFO_FMT, raw, 0)
            name = name_b.rstrip(b'\x00').decode('utf-8')
            shape = tuple(int(d) for d in [d0, d1, d2, d3][:ndim])
            tensors.append({
                'name': name, 'dtype': dtype, 'shape': shape,
                'offset': offset, 'size': size,
            })

        return {
            'magic': magic,
            'version': version,
            'tensor_count': tensor_cnt,
            'data_size': data_sz,
            'metadata': meta_json,
            'tensors': tensors,
        }
