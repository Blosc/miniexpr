/*
 * Benchmark sin**2 + cos**2 for float32/float64 with varying block sizes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "miniexpr.h"
#include "minctest.h"

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef struct {
    const char *name;
    me_dtype dtype;
    size_t elem_size;
} dtype_info_t;

static void fill_data(void *data, const dtype_info_t *info, int nitems) {
    if (info->dtype == ME_FLOAT32) {
        float *f = (float *)data;
        for (int i = 0; i < nitems; i++) {
            f[i] = (float)(i % 1024) * 0.001f + 0.1f;
        }
    } else {
        double *d = (double *)data;
        for (int i = 0; i < nitems; i++) {
            d[i] = (double)(i % 1024) * 0.001 + 0.1;
        }
    }
}

static double run_me(const me_expr *expr, const void **vars, void *out,
                     int nitems, int iterations) {
    ME_EVAL_CHECK(expr, vars, 1, out, nitems);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        ME_EVAL_CHECK(expr, vars, 1, out, nitems);
    }
    return (get_time() - start) / iterations;
}

static double run_c(const void *data, void *out, int nitems,
                    const dtype_info_t *info, int iterations) {
    double start = get_time();
    volatile double sink = 0.0;

    for (int it = 0; it < iterations; it++) {
        if (info->dtype == ME_FLOAT32) {
            const float *a = (const float *)data;
            float *o = (float *)out;
            for (int i = 0; i < nitems; i++) {
                float s = sinf(a[i]);
                float c = cosf(a[i]);
                o[i] = s * s + c * c;
            }
            sink += o[nitems - 1];
        } else {
            const double *a = (const double *)data;
            double *o = (double *)out;
            for (int i = 0; i < nitems; i++) {
                double s = sin(a[i]);
                double c = cos(a[i]);
                o[i] = s * s + c * c;
            }
            sink += o[nitems - 1];
        }
    }

    if (sink == 0.123456789) {
        printf(".");
    }

    return (get_time() - start) / iterations;
}

static void benchmark_dtype(const dtype_info_t *info, const int *blocks, int nblocks) {
    int max_block = 0;
    for (int i = 0; i < nblocks; i++) {
        if (blocks[i] > max_block) {
            max_block = blocks[i];
        }
    }

    void *data = malloc((size_t)max_block * info->elem_size);
    void *out = malloc((size_t)max_block * info->elem_size);
    if (!data || !out) {
        printf("Allocation failed for %s\n", info->name);
        free(data);
        free(out);
        return;
    }

    fill_data(data, info, max_block);

    me_variable vars[] = {{"a", info->dtype, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sin(a) ** 2 + cos(a) ** 2",
                             vars, 1, info->dtype, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile sin/cos expression for %s (err=%d)\n", info->name, err);
        free(data);
        free(out);
        return;
    }

    const void *var_ptrs[] = {data};

    printf("\n========================================\n");
    printf("sin**2 + cos**2 (%s)\n", info->name);
    printf("========================================\n");
    printf("BlockKiB ME_U10    ME_U35  ME_SCAL       C\n");
    me_set_sincos_simd(1);
    me_set_sincos_ulp(10);
    const char *backend_u10 = me_get_sincos_backend();
    printf("Backend U10: %s\n", backend_u10);
    me_set_sincos_ulp(35);
    const char *backend_u35 = me_get_sincos_backend();
    printf("Backend U35: %s\n", backend_u35);
    if (strcmp(backend_u10, backend_u35) == 0) {
        printf("Note: backend did not change between U10 and U35\n");
    }
    me_set_sincos_ulp(10);

    for (int i = 0; i < nblocks; i++) {
        int nitems = blocks[i];
        int iterations = (nitems < 65536) ? 20 : 8;
        me_set_sincos_simd(1);
        me_set_sincos_ulp(10);
        double me_time_u10 = run_me(expr, var_ptrs, out, nitems, iterations);
        me_set_sincos_ulp(35);
        double me_time_u35 = run_me(expr, var_ptrs, out, nitems, iterations);
        me_set_sincos_simd(0);
        double me_scalar_time = run_me(expr, var_ptrs, out, nitems, iterations);
        double c_time = run_c(data, out, nitems, info, iterations);
        double data_gb = (double)(nitems * info->elem_size * 2ULL) / 1e9;
        double me_gbps_u10 = data_gb / me_time_u10;
        double me_gbps_u35 = data_gb / me_time_u35;
        double me_scalar_gbps = data_gb / me_scalar_time;
        double c_gbps = data_gb / c_time;

        int kib = (int)((nitems * info->elem_size) / 1024);
        printf("%6d  %7.2f  %7.2f  %7.2f  %7.2f\n",
               kib, me_gbps_u10, me_gbps_u35, me_scalar_gbps, c_gbps);
    }

    me_set_sincos_simd(1);
    me_set_sincos_ulp(10);
    me_free(expr);
    free(data);
    free(out);
}

int main(void) {
    const dtype_info_t infos[] = {
        {"float32", ME_FLOAT32, sizeof(float)},
        {"float64", ME_FLOAT64, sizeof(double)}
    };
    const int blocks[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    const int nblocks = (int)(sizeof(blocks) / sizeof(blocks[0]));

    printf("========================================\n");
    printf("MiniExpr sin/cos Benchmark (Block Sizes)\n");
    printf("========================================\n");
    printf("Expression: sin(a)**2 + cos(a)**2\n");

    for (size_t i = 0; i < sizeof(infos) / sizeof(infos[0]); i++) {
        benchmark_dtype(&infos[i], blocks, nblocks);
    }

    printf("\n========================================\n");
    printf("Benchmark complete\n");
    printf("========================================\n");

    return 0;
}
