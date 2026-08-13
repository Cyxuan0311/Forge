static inline float dot_iq3_s_q8_K_avx2(const uint8_t* iq3s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;

    // Sign expansion masks (ported from llama.cpp ggml-cpu/arch/x86/quants.c:3384)
    static const uint8_t k_mask1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    // Union to access __m256i as 16 uint32 indices for grid lookup
    union {
        __m256i  vec[2];
        uint32_t index[16];
    } idx;

    __m256 accumf = _mm256_setzero_ps();

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq3s_row + (size_t)bi * 110;
        const float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr)) * q8[bi].d;
        const uint8_t* qs = block_ptr + 2;
        const uint8_t* qh = block_ptr + 66;
        const uint16_t* signs = reinterpret_cast<const uint16_t*>(block_ptr + 74);
        const uint8_t* scales = block_ptr + 106;
        const int8_t* q8d = q8[bi].qs;

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8d); q8d += 32;

            // Build 16 grid indices: qs[0..15] + (qh << shift & 256)
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)qs));
            qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32 + 0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32 + 1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // Grid lookup: 8 uint32 grid values per __m256i
            const __m256i q2_1 = _mm256_set_epi32(
                iq3s_grid[idx.index[7]],  iq3s_grid[idx.index[6]],
                iq3s_grid[idx.index[5]],  iq3s_grid[idx.index[4]],
                iq3s_grid[idx.index[3]],  iq3s_grid[idx.index[2]],
                iq3s_grid[idx.index[1]],  iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(
                iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]],
                iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]],
                iq3s_grid[idx.index[9]],  iq3s_grid[idx.index[8]]);

            // Sign expansion via shuffle+cmpeq (no branches, no stack arrays)
            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t)signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t)signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);

            const uint16_t ls1 = scales[ib32 / 2] & 0xf;
            const uint16_t ls2 = scales[ib32 / 2] >> 4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }

    return cpu::hsum_avx2(accumf);
}

