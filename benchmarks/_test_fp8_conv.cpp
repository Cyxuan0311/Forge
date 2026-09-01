#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "forge/fp8_utils.h"

using namespace forge;

int main() {
    float vals[] = {1.0f,    -2.5f,    0.5f,  100.0f, 0.01f,   -0.333f, 3.14159f,
                    240.0f,  0.0f,     1e-6f, 1.999f, -1.999f, 0.0039f, -0.0039f,
                    123.25f, -123.25f, 42.0f, -42.0f, 7.5f,    -7.5f};
    int n = sizeof(vals) / sizeof(vals[0]);
    double max_err_e4 = 0, max_rel_e4 = 0, sum_rel_e4 = 0;
    double max_err_e5 = 0, max_rel_e5 = 0, sum_rel_e5 = 0;
    for (int i = 0; i < n; ++i) {
        float v = vals[i];
        uint8_t b4 = fp32_to_fp8_e4m3(v);
        float r4 = fp8_e4m3_to_fp32(b4);
        double e4 = fabs((double)r4 - (double)v);
        double rel4 = e4 / (fabs((double)v) + 1e-9);
        if (e4 > max_err_e4)
            max_err_e4 = e4;
        if (rel4 > max_rel_e4)
            max_rel_e4 = rel4;
        sum_rel_e4 += rel4;

        uint8_t b5 = fp32_to_fp8_e5m2(v);
        float r5 = fp8_e5m2_to_fp32(b5);
        double e5 = fabs((double)r5 - (double)v);
        double rel5 = e5 / (fabs((double)v) + 1e-9);
        if (e5 > max_err_e5)
            max_err_e5 = e5;
        if (rel5 > max_rel_e5)
            max_rel_e5 = rel5;
        sum_rel_e5 += rel5;

        printf("v=%.5f  e4m3=0x%02X->%.5f(err=%.5f)  e5m2=0x%02X->%.5f(err=%.5f)\n", v, b4, r4, e4,
               b5, r5, e5);
    }
    printf("\nE4M3: max_abs_err=%.6f mean_rel=%.4f%% max_rel=%.4f%%\n", max_err_e4,
           100.0 * sum_rel_e4 / n, 100.0 * max_rel_e4);
    printf("E5M2: max_abs_err=%.6f mean_rel=%.4f%% max_rel=%.4f%%\n", max_err_e5,
           100.0 * sum_rel_e5 / n, 100.0 * max_rel_e5);
    return 0;
}
