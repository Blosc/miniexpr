/*
 * Benchmark for boolean logical operators.
 *
 * Measures throughput for &, |, ^, and ~ on boolean arrays.
 *
 * Usage: ./benchmark_logical_bool
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "miniexpr.h"
#include "minctest.h"


#define TOTAL_SIZE (10 * 1024 * 1024)
#define WARMUP_ITERS 2
#define BENCH_ITERS 10

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

typedef struct {
    const char *expr;
    double throughput_bool;
} bench_result_t;

static void benchmark_logical(const char *expr_str, const me_variable *vars, int num_vars,
                              const void **ptrs, size_t n, bench_result_t *result) {
    int err;
    double start, elapsed;
    bool *output = malloc(n * sizeof(bool));

    if (!output) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        result->throughput_bool = 0.0;
        return;
    }

    me_expr *expr = NULL;
    if (me_compile(expr_str, vars, num_vars, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
        fprintf(stderr, "Failed to compile %s: error %d\n", expr_str, err);
        free(output);
        result->throughput_bool = 0.0;
        return;
    }

    for (int i = 0; i < WARMUP_ITERS; i++) {
        ME_EVAL_CHECK(expr, ptrs, num_vars, output, n);
    }

    start = get_time_sec();
    for (int i = 0; i < BENCH_ITERS; i++) {
        ME_EVAL_CHECK(expr, ptrs, num_vars, output, n);
    }
    elapsed = get_time_sec() - start;
    result->expr = expr_str;
    result->throughput_bool = (n * BENCH_ITERS / elapsed) / 1e6;

    me_free(expr);
    free(output);
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  Boolean Logical Operators Benchmark\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("Configuration:\n");
    printf("  - Dataset size: %d elements (%.1f MB per array)\n",
           TOTAL_SIZE, TOTAL_SIZE * sizeof(bool) / (1024.0 * 1024.0));
    printf("  - Warmup iterations: %d\n", WARMUP_ITERS);
    printf("  - Benchmark iterations: %d\n", BENCH_ITERS);
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    bool *a = malloc(TOTAL_SIZE * sizeof(bool));
    bool *b = malloc(TOTAL_SIZE * sizeof(bool));

    if (!a || !b) {
        fprintf(stderr, "Failed to allocate input arrays\n");
        free(a);
        free(b);
        return 1;
    }

    for (size_t i = 0; i < TOTAL_SIZE; i++) {
        a[i] = (i & 1) == 0;
        b[i] = (i % 3) == 0;
    }

    me_variable vars_ab[] = {{"a", ME_BOOL}, {"b", ME_BOOL}};
    me_variable vars_a[] = {{"a", ME_BOOL}};
    const void *ptrs_ab[] = {a, b};
    const void *ptrs_a[] = {a};

    bench_result_t results[4];
    benchmark_logical("a & b", vars_ab, 2, ptrs_ab, TOTAL_SIZE, &results[0]);
    benchmark_logical("a | b", vars_ab, 2, ptrs_ab, TOTAL_SIZE, &results[1]);
    benchmark_logical("a ^ b", vars_ab, 2, ptrs_ab, TOTAL_SIZE, &results[2]);
    benchmark_logical("~a", vars_a, 1, ptrs_a, TOTAL_SIZE, &results[3]);

    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("Results:\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("%-12s  %14s\n", "Expression", "Bool (Me/s)");
    printf("───────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < 4; i++) {
        printf("%-12s  %14.2f\n", results[i].expr, results[i].throughput_bool);
    }
    printf("═══════════════════════════════════════════════════════════════════════\n");

    free(a);
    free(b);

    return 0;
}
