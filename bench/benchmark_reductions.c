/*
 * Benchmark reductions (sum) for int32 and float32.
 * Compares MiniExpr sum(x) against a pure C loop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include "miniexpr.h"
#include "minctest.h"



typedef struct {
    double me_time;
    double c_time;
    double me_gbps;
    double c_gbps;
} bench_result_t;

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static bench_result_t benchmark_sum_int32(size_t total_elems, int iterations) {
    printf("\n=== sum(int32) ===\n");

    int32_t *data = malloc(total_elems * sizeof(int32_t));
    if (!data) {
        printf("Allocation failed for int32 data\n");
        bench_result_t empty = {0};
        return empty;
    }

    for (size_t i = 0; i < total_elems; i++) {
        data[i] = (int32_t)(i % 97);
    }

    me_variable vars[] = {{"x", ME_INT32, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile sum(x) for int32 (err=%d)\n", err);
        free(data);
        bench_result_t empty = {0};
        return empty;
    }

    const void *var_ptrs[] = {data};
    int64_t output = 0;

    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, (int)total_elems);

    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, (int)total_elems);
    }
    double me_time = (get_time() - start) / iterations;

    volatile int64_t sink = 0;
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        int64_t acc = 0;
        for (size_t i = 0; i < total_elems; i++) {
            acc += data[i];
        }
        sink = acc;
    }
    double c_time = (get_time() - start) / iterations;

    double gb = (double)(total_elems * sizeof(int32_t)) / 1e9;
    printf("MiniExpr: %.4f s (%.2f GB/s)\n", me_time, gb / me_time);
    printf("Pure C : %.4f s (%.2f GB/s)\n", c_time, gb / c_time);
    printf("Result check (MiniExpr): %lld\n", (long long)output);
    printf("Result check (C):        %lld\n", (long long)sink);

    me_free(expr);
    free(data);
    bench_result_t result = {
        .me_time = me_time,
        .c_time = c_time,
        .me_gbps = gb / me_time,
        .c_gbps = gb / c_time
    };
    return result;
}

static bench_result_t benchmark_sum_float32(size_t total_elems, int iterations) {
    printf("\n=== sum(float32) ===\n");

    float *data = malloc(total_elems * sizeof(float));
    if (!data) {
        printf("Allocation failed for float32 data\n");
        bench_result_t empty = {0};
        return empty;
    }

    for (size_t i = 0; i < total_elems; i++) {
        data[i] = (float)(i % 97) * 0.25f;
    }

    me_variable vars[] = {{"x", ME_FLOAT32, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile sum(x) for float32 (err=%d)\n", err);
        free(data);
        bench_result_t empty = {0};
        return empty;
    }

    const void *var_ptrs[] = {data};
    float output = 0.0f;

    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, (int)total_elems);

    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, (int)total_elems);
    }
    double me_time = (get_time() - start) / iterations;

    volatile float sink = 0.0f;
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        float acc = 0.0f;
        for (size_t i = 0; i < total_elems; i++) {
            acc += data[i];
        }
        sink = acc;
    }
    double c_time = (get_time() - start) / iterations;

    double gb = (double)(total_elems * sizeof(float)) / 1e9;
    printf("MiniExpr: %.4f s (%.2f GB/s)\n", me_time, gb / me_time);
    printf("Pure C : %.4f s (%.2f GB/s)\n", c_time, gb / c_time);
    printf("Result check (MiniExpr): %.6f\n", output);
    printf("Result check (C):        %.6f\n", sink);

    me_free(expr);
    free(data);
    bench_result_t result = {
        .me_time = me_time,
        .c_time = c_time,
        .me_gbps = gb / me_time,
        .c_gbps = gb / c_time
    };
    return result;
}

int main(void) {
    printf("========================================\n");
    printf("MiniExpr Reduction Benchmark (sum)\n");
    printf("========================================\n");

    const size_t sizes_mb[] = {1, 2, 4, 8, 16};
    const int iterations = 4;
    const size_t num_sizes = sizeof(sizes_mb) / sizeof(sizes_mb[0]);

    bench_result_t int_results[10];
    bench_result_t float_results[10];

    printf("Iterations: %d\n", iterations);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t bytes = sizes_mb[i] * 1024 * 1024;
        size_t total_elems = bytes / sizeof(int32_t);

        printf("\n--- Working set: %zu MB (%zu elements) ---\n", sizes_mb[i], total_elems);
        int_results[i] = benchmark_sum_int32(total_elems, iterations);
        float_results[i] = benchmark_sum_float32(total_elems, iterations);
    }

    printf("\n========================================\n");
    printf("Summary (GB/s)\n");
    printf("========================================\n");
    printf("Size(MB)  Int32 ME   Int32 C    F32 ME     F32 C\n");
    for (size_t i = 0; i < num_sizes; i++) {
        printf("%7zu  %9.2f  %9.2f  %9.2f  %9.2f\n",
               sizes_mb[i],
               int_results[i].me_gbps,
               int_results[i].c_gbps,
               float_results[i].me_gbps,
               float_results[i].c_gbps);
    }

    printf("========================================\n");
    printf("Benchmark complete!\n");
    printf("========================================\n");

    return 0;
}
