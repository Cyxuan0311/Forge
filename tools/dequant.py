"""
Block-wise dequantization for GGML quantization formats.

All functions accept (raw_bytes, offset, numel) and return a flat
float32 NumPy array.  K-quant variants use per-block vectorisation
to avoid Python loops over blocks, while element unpacking still
iterates within a single block (32 or 256 elements).
"""

import numpy as np

from tools.gguf_reader import GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, \
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, \
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, \
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K

Q4_0_BLOCK_SIZE = 18
Q4_0_BLOCK_ELEMS = 32
Q4_1_BLOCK_SIZE = 20
Q5_0_BLOCK_SIZE = 22
Q5_1_BLOCK_SIZE = 24
Q8_0_BLOCK_SIZE = 34
Q2_K_BLOCK_SIZE = 84
Q3_K_BLOCK_SIZE = 110
Q4_K_BLOCK_SIZE = 144
Q5_K_BLOCK_SIZE = 176
Q6_K_BLOCK_SIZE = 210
QK_K = 256


def dequantize(data, offset, dtype, dims):
    numel = int(np.prod(dims))
    dispatch = {
        GGML_TYPE_Q4_0: dequant_q4_0,
        GGML_TYPE_Q4_1: dequant_q4_1,
        GGML_TYPE_Q5_0: dequant_q5_0,
        GGML_TYPE_Q5_1: dequant_q5_1,
        GGML_TYPE_Q8_0: dequant_q8_0,
        GGML_TYPE_Q6_K: dequant_q6_k,
        GGML_TYPE_Q5_K: dequant_q5_k,
        GGML_TYPE_Q4_K: dequant_q4_k,
        GGML_TYPE_Q3_K: dequant_q3_k,
        GGML_TYPE_Q2_K: dequant_q2_k,
    }
    fn = dispatch.get(dtype)
    if fn is None:
        raise ValueError(f'Unsupported dtype for dequant: {dtype}')
    return fn(data, offset, numel)


# ---------------------------------------------------------------------------
# Helper: FP16 stored as 2 bytes → float32
# ---------------------------------------------------------------------------

def _f16(val):
    return float(np.frombuffer(val, dtype=np.float16)[0])


# ---------------------------------------------------------------------------
# Q4_0  – 18 bytes/block, 32 elements
# ---------------------------------------------------------------------------

def dequant_q4_0(data, offset, numel):
    n_blocks = (numel + Q4_0_BLOCK_ELEMS - 1) // Q4_0_BLOCK_ELEMS
    raw = data[offset:offset + n_blocks * Q4_0_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q4_0_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q4_0_BLOCK_SIZE)

    scale_bits = blocks[:, 0:2].copy().view(np.uint16).flatten()
    sign = (scale_bits >> 15) & 1
    exp = (scale_bits >> 10) & 0x1F
    mant = scale_bits & 0x3FF

    scale = np.where(
        exp == 0,
        np.where(mant == 0, 0.0, mant.astype(np.float32) / 1024.0 * (2.0 ** -14)),
        (1.0 + mant.astype(np.float32) / 1024.0) * (2.0 ** (exp.astype(np.float32) - 15)),
    )
    scale = np.where(sign.astype(bool), -scale, scale)

    qs = blocks[:, 2:18]
    dequant = np.zeros((n_blocks, 32), dtype=np.float32)
    for j in range(16):
        lo = (qs[:, j] & 0x0F).astype(np.int8)
        hi = ((qs[:, j] >> 4) & 0x0F).astype(np.int8)
        lo = np.where(lo >= 8, lo - 16, lo).astype(np.float32)
        hi = np.where(hi >= 8, hi - 16, hi).astype(np.float32)
        dequant[:, j * 2] = lo
        dequant[:, j * 2 + 1] = hi

    return (dequant * scale[:, np.newaxis]).flatten()[:numel]


# ---------------------------------------------------------------------------
# Q4_1  – 20 bytes/block, 32 elements
# ---------------------------------------------------------------------------

def dequant_q4_1(data, offset, numel):
    n_blocks = (numel + Q4_0_BLOCK_ELEMS - 1) // Q4_0_BLOCK_ELEMS
    raw = data[offset:offset + n_blocks * Q4_1_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q4_1_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q4_1_BLOCK_SIZE)

    d_vals = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).flatten()
    m_vals = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()
    qs = blocks[:, 4:20]

    dequant = np.zeros((n_blocks, 32), dtype=np.float32)
    for j in range(16):
        lo = (qs[:, j] & 0x0F).astype(np.float32)
        hi = ((qs[:, j] >> 4) & 0x0F).astype(np.float32)
        dequant[:, j] = lo
        dequant[:, j + 16] = hi

    return (dequant * d_vals[:, np.newaxis] + m_vals[:, np.newaxis]).flatten()[:numel]


# ---------------------------------------------------------------------------
# Q5_0  – 22 bytes/block, 32 elements
# ---------------------------------------------------------------------------

def dequant_q5_0(data, offset, numel):
    n_blocks = (numel + Q4_0_BLOCK_ELEMS - 1) // Q4_0_BLOCK_ELEMS
    raw = data[offset:offset + n_blocks * Q5_0_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q5_0_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q5_0_BLOCK_SIZE)

    scale_bits = blocks[:, 0:2].copy().view(np.uint16).flatten()
    sign = (scale_bits >> 15) & 1
    exp = (scale_bits >> 10) & 0x1F
    mant = scale_bits & 0x3FF
    scale = np.where(
        exp == 0,
        np.where(mant == 0, 0.0, mant.astype(np.float32) / 1024.0 * (2.0 ** -14)),
        (1.0 + mant.astype(np.float32) / 1024.0) * (2.0 ** (exp.astype(np.float32) - 15)),
    )
    scale = np.where(sign.astype(bool), -scale, scale)

    qh = blocks[:, 2:6]
    qs = blocks[:, 6:22]
    dequant = np.zeros((n_blocks, 32), dtype=np.float32)
    for j in range(32):
        h_bit = (qh[:, j // 8] >> (j % 8)) & 1
        lo = (qs[:, j // 2] >> (4 * (j % 2))) & 0x0F
        q_val = (lo | (h_bit.astype(np.uint8) << 4)).astype(np.float32)
        dequant[:, j] = (q_val - 16.0) * scale

    return dequant.flatten()[:numel]


# ---------------------------------------------------------------------------
# Q5_1  – 24 bytes/block, 32 elements
# ---------------------------------------------------------------------------

def dequant_q5_1(data, offset, numel):
    n_blocks = (numel + Q4_0_BLOCK_ELEMS - 1) // Q4_0_BLOCK_ELEMS
    raw = data[offset:offset + n_blocks * Q5_1_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q5_1_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q5_1_BLOCK_SIZE)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).flatten()
    m = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()
    qh = blocks[:, 4:8]
    qs = blocks[:, 8:24]

    dequant = np.zeros((n_blocks, 32), dtype=np.float32)
    for j in range(32):
        h_bit = (qh[:, j // 8] >> (j % 8)) & 1
        lo = (qs[:, j // 2] >> (4 * (j % 2))) & 0x0F
        q_val = (lo | (h_bit.astype(np.uint8) << 4)).astype(np.float32)
        dequant[:, j] = d * q_val + m

    return dequant.flatten()[:numel]


# ---------------------------------------------------------------------------
# Q8_0  – 34 bytes/block, 32 elements
# ---------------------------------------------------------------------------

def dequant_q8_0(data, offset, numel):
    n_blocks = (numel + Q4_0_BLOCK_ELEMS - 1) // Q4_0_BLOCK_ELEMS
    raw = data[offset:offset + n_blocks * Q8_0_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q8_0_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q8_0_BLOCK_SIZE)

    scale_bits = blocks[:, 0:2].copy().view(np.uint16).flatten()
    sign = (scale_bits >> 15) & 1
    exp = (scale_bits >> 10) & 0x1F
    mant = scale_bits & 0x3FF
    scale = np.where(
        exp == 0,
        np.where(mant == 0, 0.0, mant.astype(np.float32) / 1024.0 * (2.0 ** -14)),
        (1.0 + mant.astype(np.float32) / 1024.0) * (2.0 ** (exp.astype(np.float32) - 15)),
    )
    scale = np.where(sign.astype(bool), -scale, scale)

    qs = blocks[:, 2:34].astype(np.int8)
    return (qs * scale[:, np.newaxis]).flatten()[:numel]


# ---------------------------------------------------------------------------
# Q6_K  – 210 bytes/block, 256 elements
# ---------------------------------------------------------------------------

def dequant_q6_k(data, offset, numel):
    n_blocks = (numel + QK_K - 1) // QK_K
    raw = data[offset:offset + n_blocks * Q6_K_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q6_K_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q6_K_BLOCK_SIZE)

    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    sc = blocks[:, 192:208]
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32).flatten()

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for j8 in range(8):
        for elem in range(32):
            idx = j8 * 32 + elem
            lo = (ql[:, j8 * 16 + elem // 2] >> (4 * (elem % 2))) & 0x0F
            hi = (qh[:, j8 * 8 + elem // 4] >> (2 * (elem % 4))) & 3
            q_val = (lo | (hi << 4)).astype(np.int32) - 32

            sc_idx = j8 * 2 + (elem // 16)
            sc_shift = 4 * (sc_idx % 2)
            sc_val = ((sc[:, sc_idx] >> sc_shift) & 0x3F).astype(np.int32) - 32

            result[:, idx] = d * sc_val * q_val

    return result.flatten()[:numel]


# ---------------------------------------------------------------------------
# Q5_K  – 176 bytes/block, 256 elements
# ---------------------------------------------------------------------------

def dequant_q5_k(data, offset, numel):
    n_blocks = (numel + QK_K - 1) // QK_K
    raw = data[offset:offset + n_blocks * Q5_K_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q5_K_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q5_K_BLOCK_SIZE)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).flatten()
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()
    sc = blocks[:, 4:12]
    m = blocks[:, 12:20]
    qh = blocks[:, 20:52]
    qs = blocks[:, 52:180]

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for j in range(QK_K):
        j8 = j // 32
        sc_val = ((sc[:, j8] >> (4 * (j8 % 2))) & 0x0F).astype(np.float32)
        m_val = ((m[:, j8] >> (4 * (j8 % 2))) & 0x0F).astype(np.float32)
        h_bit = (qh[:, j // 8] >> (j % 8)) & 1
        lo = (qs[:, j8 * 32 + (j % 32)] >> (4 * (j % 2))) & 0x0F
        q_val = (lo | (h_bit.astype(np.uint8) << 4)).astype(np.float32)
        result[:, j] = d * sc_val * (q_val - 16.0) - dmin * m_val

    return result.flatten()[:numel]


# ---------------------------------------------------------------------------
# Q4_K  – 144 bytes/block, 256 elements
# ---------------------------------------------------------------------------

def dequant_q4_k(data, offset, numel):
    n_blocks = (numel + QK_K - 1) // QK_K
    raw = data[offset:offset + n_blocks * Q4_K_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q4_K_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q4_K_BLOCK_SIZE)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).flatten()
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()
    sc = blocks[:, 4:12]
    m = blocks[:, 12:20]
    qs = blocks[:, 20:148]

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for j in range(QK_K):
        j8 = j // 32
        sc_val = ((sc[:, j8] >> (4 * (j8 % 2))) & 0x0F).astype(np.float32)
        m_val = ((m[:, j8] >> (4 * (j8 % 2))) & 0x0F).astype(np.float32)
        q_idx = j8 * 32 + (j % 32)
        q_val = ((qs[:, q_idx] >> (4 * (j % 2))) & 0x0F).astype(np.float32)
        result[:, j] = d * sc_val * (q_val - 8.0) - dmin * m_val

    return result.flatten()[:numel]


# ---------------------------------------------------------------------------
# Q3_K  – 110 bytes/block, 256 elements
# ---------------------------------------------------------------------------

def dequant_q3_k(data, offset, numel):
    n_blocks = (numel + QK_K - 1) // QK_K
    raw = data[offset:offset + n_blocks * Q3_K_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q3_K_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q3_K_BLOCK_SIZE)

    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    sc = blocks[:, 96:108]
    d = blocks[:, 108:110].copy().view(np.float16).astype(np.float32).flatten()

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for j in range(QK_K):
        j8 = j // 32
        sc_val = ((sc[:, j8 // 4] >> (2 * (j8 % 4))) & 3).astype(np.float32)
        m_val = ((hmask[:, j // 8] >> (j % 8)) & 1).astype(np.float32)
        q_idx = j8 * 32 + (j % 32)
        q_val = ((qs[:, q_idx] >> (2 * (j % 4))) & 3).astype(np.float32)
        result[:, j] = d * sc_val * (q_val - 4.0 * (1.0 - m_val))

    return result.flatten()[:numel]


# ---------------------------------------------------------------------------
# Q2_K  – 84 bytes/block, 256 elements
# ---------------------------------------------------------------------------

def dequant_q2_k(data, offset, numel):
    n_blocks = (numel + QK_K - 1) // QK_K
    raw = data[offset:offset + n_blocks * Q2_K_BLOCK_SIZE]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q2_K_BLOCK_SIZE)
    blocks = blocks.reshape(n_blocks, Q2_K_BLOCK_SIZE)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).flatten()
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()
    scales = blocks[:, 4:20]
    qs = blocks[:, 20:84]

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for j in range(QK_K):
        j8 = j // 32
        sc_val = ((scales[:, j8 // 2] >> (4 * (j8 % 2))) & 0x0F).astype(np.float32)
        m_val = ((scales[:, 8 + j8 // 2] >> (4 * (j8 % 2))) & 0x0F).astype(np.float32)
        q_idx = j8 * 32 + (j % 32)
        q_val = ((qs[:, q_idx] >> (2 * (j % 4))) & 3).astype(np.float32)
        result[:, j] = d * sc_val * (q_val - 2.0) - dmin * m_val

    return result.flatten()[:numel]
