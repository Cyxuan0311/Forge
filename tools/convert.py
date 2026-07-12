"""
GGUF → .ninf conversion orchestration.

Importable function `convert_gguf_to_ninf(gguf_path, ninf_path, target_dtype)`.
"""

import numpy as np

from tools.format import NINF_DTYPE_FP32, NINF_DTYPE_Q4_0, NINF_DTYPE_Q4_1, build_ninf
from tools.gguf_reader import parse_gguf, get_tensor_byte_size, ggml_type_name, \
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0
from tools.dequant import dequantize
from tools.name_mapping import map_gguf_name


def convert_gguf_to_ninf(gguf_path, ninf_path, target_dtype='q4_0'):
    metadata, tensors_info, data = parse_gguf(gguf_path)

    arch = metadata.get('general.architecture', 'llama')
    print(f'\nArchitecture: {arch}')

    vocab_size = metadata.get(f'{arch}.vocab_size', 0)
    hidden_dim = metadata.get(f'{arch}.embedding_length', 4096)
    intermediate_dim = metadata.get(f'{arch}.feed_forward_length', 11008)
    num_layers = metadata.get(f'{arch}.block_count', 32)
    num_heads = metadata.get(f'{arch}.attention.head_count', 32)
    num_kv_heads = metadata.get(f'{arch}.attention.head_count_kv', num_heads)
    head_dim = hidden_dim // num_heads

    for t in tensors_info:
        if t['name'] == 'token_embd.weight' and len(t['dims']) >= 2:
            vocab_size = t['dims'][1]
            if hidden_dim == 0:
                hidden_dim = t['dims'][0]
            break

    if vocab_size == 0:
        vocab_size = 32000

    print(f'Model config: vocab={vocab_size}, hidden={hidden_dim}, '
          f'intermediate={intermediate_dim}, layers={num_layers}, '
          f'heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}')

    ninf_tensors = []

    for t in tensors_info:
        gguf_name = t['name']
        dims = t['dims']
        dtype_ggml = t['dtype']
        data_offset = t['data_offset']

        name = map_gguf_name(gguf_name)

        if gguf_name.endswith('.bias'):
            continue

        if gguf_name == 'output.weight' and dtype_ggml not in (
                GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0):
            print(f'  Skipping {gguf_name} ({ggml_type_name(dtype_ggml)}) '
                  f'- will use tied embedding weights')
            continue

        byte_size = get_tensor_byte_size(dims, dtype_ggml)
        raw = data[data_offset:data_offset + byte_size]

        # Transpose 2D dims
        out_dims = [dims[1], dims[0]] if len(dims) == 2 else list(dims)

        if dtype_ggml == GGML_TYPE_Q4_0:
            ninf_tensors.append((name, NINF_DTYPE_Q4_0, out_dims, raw))
        elif dtype_ggml == GGML_TYPE_Q4_1:
            ninf_tensors.append((name, NINF_DTYPE_Q4_1, out_dims, raw))
        elif dtype_ggml == GGML_TYPE_F32:
            ninf_tensors.append((name, NINF_DTYPE_FP32, out_dims, raw))
        elif dtype_ggml == GGML_TYPE_F16:
            numel = int(np.prod(dims))
            fp32_data = np.frombuffer(raw, dtype=np.float16, count=numel).astype(np.float32)
            ninf_tensors.append((name, NINF_DTYPE_FP32, out_dims, fp32_data.tobytes()))
        else:
            # Dequantize to FP32 for all other types
            numel = int(np.prod(dims))
            print(f'  Dequantizing {ggml_type_name(dtype_ggml)} tensor '
                  f'{name} to FP32')
            fp32_data = dequantize(data, data_offset, dtype_ggml, dims)
            ninf_tensors.append((name, NINF_DTYPE_FP32, out_dims, fp32_data.tobytes()))

        dtype_out = 'Q4_0' if ninf_tensors[-1][1] == NINF_DTYPE_Q4_0 else \
                    'Q4_1' if ninf_tensors[-1][1] == NINF_DTYPE_Q4_1 else 'FP32'
        print(f'  {name}: {ggml_type_name(dtype_ggml)} -> {dtype_out} '
              f'shape={ninf_tensors[-1][2]}')

    print(f'\nTotal tensors to write: {len(ninf_tensors)}')

    meta_dict = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'intermediate_dim': intermediate_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'model_type': arch,
    }

    file_size = build_ninf(meta_dict, ninf_tensors, ninf_path)
    print('\nConversion complete!')
    print(f'  Output: {ninf_path}')
    print(f'  File size: {file_size / (1024**3):.2f} GB')
    print(f'  Tensors: {len(ninf_tensors)}')
