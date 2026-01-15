/*
 * Benchmark exp throughput for float32/float64 with varying block sizes.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "miniexpr.h"

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
    const double min = -5.0;
    const double max = 5.0;
    const double step = (max - min) / (double)(nitems ? nitems : 1);

    if (info->dtype == ME_FLOAT32) {
        float *f = (float *)data;
        for (int i = 0; i < nitems; i++) {
            f[i] = (float)(min + step * (double)i);
        }
    } else {
        double *d = (double *)data;
        for (int i = 0; i < nitems; i++) {
            d[i] = min + step * (double)i;
        }
    }
}

static double run_me(const me_expr *expr, const void **vars, void *out,
                     int nitems, int iterations, const me_eval_params *params) {
    int rc = me_eval(expr, vars, 1, out, nitems, params);
    if (rc != ME_EVAL_SUCCESS) {
        fprintf(stderr, "me_eval failed: %d\n", rc);
        exit(1);
    }

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        rc = me_eval(expr, vars, 1, out, nitems, params);
        if (rc != ME_EVAL_SUCCESS) {
            fprintf(stderr, "me_eval failed: %d\n", rc);
            exit(1);
        }
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
                o[i] = 2.0f * expf(a[i]);
            }
            sink += o[nitems - 1];
        } else {
            const double *a = (const double *)data;
            double *o = (double *)out;
            for (int i = 0; i < nitems; i++) {
                o[i] = 2.0 * exp(a[i]);
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
    const int max_block = blocks[nblocks - 1];
    void *data = malloc((size_t)max_block * info->elem_size);
    void *out = malloc((size_t)max_block * info->elem_size);
    if (!data || !out) {
        printf("Allocation failed for %s\n", info->name);
        free(data);
        free(out);
        return;
    }

    fill_data(data, info, max_block);

    me_variable vars[] = {{"x", info->dtype, data}};
    int err = 0;
    me_expr *expr = NULL;
    const char *expr_text = "2 * exp(x)";
    int rc_expr = me_compile(expr_text, vars, 1, info->dtype, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile expression for %s (err=%d)\n", info->name, err);
        free(data);
        free(out);
        return;
    }

    const void *var_ptrs[] = {data};

    printf("\n========================================\n");
    printf("2 * exp(x) (%s)\n", info->name);
    printf("========================================\n");
    printf("BlockKiB ME_U10    ME_U35  ME_SCAL       C\n");

    me_eval_params params_u10 = ME_EVAL_PARAMS_DEFAULTS;
    params_u10.simd_ulp_mode = ME_SIMD_ULP_1;
    me_eval_params params_u35 = ME_EVAL_PARAMS_DEFAULTS;
    params_u35.simd_ulp_mode = ME_SIMD_ULP_3_5;
    me_eval_params params_scalar = ME_EVAL_PARAMS_DEFAULTS;
    params_scalar.disable_simd = true;

    for (int i = 0; i < nblocks; i++) {
        int nitems = blocks[i];
        int iterations = (nitems < 65536) ? 20 : 8;
        double me_time_u10 = run_me(expr, var_ptrs, out, nitems, iterations, &params_u10);
        double me_time_u35 = run_me(expr, var_ptrs, out, nitems, iterations, &params_u35);
        double me_scalar_time = run_me(expr, var_ptrs, out, nitems, iterations, &params_scalar);
        double c_time = run_c(data, out, nitems, info, iterations);
        double data_gb = (double)(nitems * info->elem_size) / 1e9;
        double me_gbps_u10 = data_gb / me_time_u10;
        double me_gbps_u35 = data_gb / me_time_u35;
        double me_scalar_gbps = data_gb / me_scalar_time;
        double c_gbps = data_gb / c_time;

        int kib = (int)((nitems * info->elem_size) / 1024);
        printf("%6d  %7.2f  %7.2f  %7.2f  %7.2f\n",
               kib, me_gbps_u10, me_gbps_u35, me_scalar_gbps, c_gbps);
    }

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
    printf("MiniExpr exp Benchmark (Block Sizes)\n");
    printf("========================================\n");
    printf("Expression: 2 * exp(x)\n");

    for (size_t i = 0; i < sizeof(infos) / sizeof(infos[0]); i++) {
        benchmark_dtype(&infos[i], blocks, nblocks);
    }

    printf("\n========================================\n");
    printf("Benchmark complete\n");
    printf("========================================\n");
    return 0;
}
