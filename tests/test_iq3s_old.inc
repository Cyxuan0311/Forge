static inline float dot_iq3_s_q8_K_avx2_old(const uint8_t* iq3s_row,
                                        const cpu::block_q8_K* q8, int nb) {
    constexpr int QK_K = 256;
    constexpr int IQ3_S_BLOCK_SIZE = 110;

    float acc = 0.0f;

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* block_ptr = iq3s_row + (size_t)bi * IQ3_S_BLOCK_SIZE;
        float d = cpu::fp16_to_float_scalar(*reinterpret_cast<const uint16_t*>(block_ptr));
        const uint8_t* qs = block_ptr + 2;
        const uint8_t* qh = block_ptr + 2 + 64;
        const uint8_t* signs = block_ptr + 2 + 64 + 8;
        const uint8_t* scales = block_ptr + 2 + 64 + 8 + 32;

        const cpu::block_q8_K* y = q8 + bi;
        const float d_q8 = y->d;
        const int8_t* q8d = y->qs;

        float block_acc = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1_f = (float)(1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2_f = (float)(1 + 2 * (scales[ib32 / 2] >> 4));

            for (int half = 0; half < 2; ++half) {
                const float db_f = (half == 0) ? db1_f : db2_f;

                alignas(32) uint8_t grid_vals[32];
                alignas(32) uint8_t sign_mask[32];
                for (int l = 0; l < 4; ++l) {
                    const int idx1 = qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256);
                    const int idx2 = qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256);
                    memcpy(grid_vals + l * 8 + 0, &iq3s_grid[idx1], 4);
                    memcpy(grid_vals + l * 8 + 4, &iq3s_grid[idx2], 4);
                    const uint8_t s = signs[l];
                    for (int j = 0; j < 8; ++j)
                        sign_mask[l * 8 + j] = (s & kmask_iq2xs[j]) ? 0xFF : 0x00;
                }

                const __m256i grid_vec = _mm256_load_si256((const __m256i*)grid_vals);
                const __m256i q8_vec = _mm256_loadu_si256((const __m256i*)(q8d + ib32 * 32 + half * 32));
                const __m256i sign_vec = _mm256_load_si256((const __m256i*)sign_mask);

                const __m256i zero = _mm256_setzero_si256();
                const __m256i q8_neg = _mm256_sub_epi8(zero, q8_vec);
                const __m256i q8_signed = _mm256_blendv_epi8(q8_vec, q8_neg, sign_vec);

                const __m256i p16 = _mm256_maddubs_epi16(grid_vec, q8_signed);
                const __m256i p32 = _mm256_madd_epi16(_mm256_set1_epi16(1), p16);

                const __m128i lo = _mm256_extracti128_si256(p32, 0);
                const __m128i hi = _mm256_extracti128_si256(p32, 1);
                __m128i h = _mm_hadd_epi32(lo, hi);
                h = _mm_hadd_epi32(h, h);
                h = _mm_hadd_epi32(h, h);
                const int s_all = _mm_cvtsi128_si32(h);

                block_acc += (float)s_all * db_f;

                qs += 8;
                signs += 4;
            }
            qh += 2;
        }

        acc += d * d_q8 * block_acc;
    }

    return acc;
}
