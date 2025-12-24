/*
 * Benchmark for comparison operations with boolean output
 *
 * Tests various comparison expressions and measures performance:
 *   - Simple comparisons: a < b, a == b
 *   - Complex comparisons: a**2 == (a + b), sqrt(a) < b
 *   - Compares ME_BOOL output vs ME_FLOAT64 output
 *
 * This benchmark evaluates the overhead of type conversion when
 * outputting boolean results from floating-point comparisons.
 *
 * Usage: ./benchmark_comparisons
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "miniexpr.h"

/* Configuration */
#define TOTAL_SIZE (10 * 1024 * 1024)  /* 10M elements */
#define WARMUP_ITERS 2
#define BENCH_ITERS 10

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

typedef struct {
    const char *name;
    const char *expr;
    int num_vars;
    double throughput_bool;   /* Melems/sec with ME_BOOL output */
    double throughput_f64;    /* Melems/sec with ME_FLOAT64 output */
} bench_result_t;

/*
 * Benchmark a comparison expression with both bool and float64 output
 */
static void benchmark_comparison(const char *name, const char *expr_str,
                                  double *a, double *b, double *c,
                                  int num_vars, size_t n,
                                  bench_result_t *result) {
    int err;
    double start, elapsed;

    result->name = name;
    result->expr = expr_str;
    result->num_vars = num_vars;

    /* Setup variable definitions with explicit types */
    me_variable vars2[] = {{"a", ME_FLOAT64}, {"b", ME_FLOAT64}};
    me_variable vars3[] = {{"a", ME_FLOAT64}, {"b", ME_FLOAT64}, {"c", ME_FLOAT64}};
    me_variable *vars = (num_vars == 2) ? vars2 : vars3;

    const void *ptrs2[] = {a, b};
    const void *ptrs3[] = {a, b, c};
    const void **ptrs = (num_vars == 2) ? ptrs2 : ptrs3;

    /* Allocate output buffers */
    bool *result_bool = malloc(n * sizeof(bool));
    double *result_f64 = malloc(n * sizeof(double));

    if (!result_bool || !result_f64) {
        fprintf(stderr, "Failed to allocate result buffers\n");
        free(result_bool);
        free(result_f64);
        return;
    }

    /*
     * Benchmark 1: ME_BOOL output
     */
    me_expr *expr_bool = me_compile(expr_str, vars, num_vars, ME_BOOL, &err);
    if (!expr_bool) {
        fprintf(stderr, "Failed to compile %s with ME_BOOL: error %d\n", name, err);
        free(result_bool);
        free(result_f64);
        return;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        me_eval(expr_bool, ptrs, num_vars, result_bool, n);
    }

    /* Timed iterations */
    start = get_time_sec();
    for (int i = 0; i < BENCH_ITERS; i++) {
        me_eval(expr_bool, ptrs, num_vars, result_bool, n);
    }
    elapsed = get_time_sec() - start;
    result->throughput_bool = (n * BENCH_ITERS / elapsed) / 1e6;

    me_free(expr_bool);

    /*
     * Benchmark 2: ME_FLOAT64 output (for comparison)
     */
    me_expr *expr_f64 = me_compile(expr_str, vars, num_vars, ME_FLOAT64, &err);
    if (!expr_f64) {
        fprintf(stderr, "Failed to compile %s with ME_FLOAT64: error %d\n", name, err);
        free(result_bool);
        free(result_f64);
        return;
    }

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        me_eval(expr_f64, ptrs, num_vars, result_f64, n);
    }

    /* Timed iterations */
    start = get_time_sec();
    for (int i = 0; i < BENCH_ITERS; i++) {
        me_eval(expr_f64, ptrs, num_vars, result_f64, n);
    }
    elapsed = get_time_sec() - start;
    result->throughput_f64 = (n * BENCH_ITERS / elapsed) / 1e6;

    me_free(expr_f64);

    /* Verify results match (spot check) */
    int mismatches = 0;
    for (size_t i = 0; i < n && mismatches < 5; i += n / 10) {
        bool b_val = result_bool[i];
        bool f_val = (result_f64[i] != 0.0);
        if (b_val != f_val) {
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "Warning: %d mismatches in %s\n", mismatches, name);
    }

    free(result_bool);
    free(result_f64);
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  Comparison Operations Benchmark\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("Configuration:\n");
    printf("  - Dataset size: %d elements (%.1f MB per array)\n",
           TOTAL_SIZE, TOTAL_SIZE * sizeof(double) / (1024.0 * 1024.0));
    printf("  - Warmup iterations: %d\n", WARMUP_ITERS);
    printf("  - Benchmark iterations: %d\n", BENCH_ITERS);
    printf("  - Comparing ME_BOOL vs ME_FLOAT64 output types\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    /* Allocate and initialize input arrays */
    double *a = malloc(TOTAL_SIZE * sizeof(double));
    double *b = malloc(TOTAL_SIZE * sizeof(double));
    double *c = malloc(TOTAL_SIZE * sizeof(double));

    if (!a || !b || !c) {
        fprintf(stderr, "Failed to allocate input arrays\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    /* Initialize with varied data to exercise different comparison outcomes */
    for (size_t i = 0; i < TOTAL_SIZE; i++) {
        a[i] = (double)(i % 1000) / 100.0;           /* 0.00 to 9.99 */
        b[i] = (double)((i + 500) % 1000) / 100.0;   /* Offset pattern */
        /* c[i] such that a**2 == a + c is sometimes true */
        c[i] = a[i] * a[i] - a[i];                   /* c = a² - a, so a² == a + c */
    }

    /* Define benchmarks */
    bench_result_t results[10];
    int num_benchmarks = 0;

    printf("Running benchmarks...\n\n");

    /* Simple comparisons */
    benchmark_comparison("a < b", "a < b", a, b, c, 2, TOTAL_SIZE, &results[num_benchmarks++]);
    benchmark_comparison("a <= b", "a <= b", a, b, c, 2, TOTAL_SIZE, &results[num_benchmarks++]);
    benchmark_comparison("a == b", "a == b", a, b, c, 2, TOTAL_SIZE, &results[num_benchmarks++]);
    benchmark_comparison("a != b", "a != b", a, b, c, 2, TOTAL_SIZE, &results[num_benchmarks++]);

    /* Comparisons with arithmetic */
    benchmark_comparison("a + b < c", "a + b < c", a, b, c, 3, TOTAL_SIZE, &results[num_benchmarks++]);
    benchmark_comparison("a * b == c", "a * b == c", a, b, c, 3, TOTAL_SIZE, &results[num_benchmarks++]);

    /* Comparisons with power operations */
    benchmark_comparison("a**2 < b", "a**2 < b", a, b, c, 2, TOTAL_SIZE, &results[num_benchmarks++]);
    benchmark_comparison("a**2 + b**2 < c", "a**2 + b**2 < c", a, b, c, 3, TOTAL_SIZE, &results[num_benchmarks++]);

    /* Complex comparisons */
    benchmark_comparison("sqrt(a) < b", "sqrt(a) < b", a, b, c, 2, TOTAL_SIZE, &results[num_benchmarks++]);
    benchmark_comparison("a**2 + b**2 < c**2", "a**2 + b**2 < c**2", a, b, c, 3, TOTAL_SIZE, &results[num_benchmarks++]);

    /* Print results table */
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("Results:\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("%-22s  %12s  %12s  %10s\n",
           "Expression", "Bool (Me/s)", "F64 (Me/s)", "Ratio");
    printf("───────────────────────────────────────────────────────────────────────\n");

    double total_bool = 0, total_f64 = 0;
    int valid_count = 0;
    for (int i = 0; i < num_benchmarks; i++) {
        double ratio = results[i].throughput_bool / results[i].throughput_f64;
        /* Handle potential inf/nan values */
        if (results[i].throughput_bool > 1e9 || results[i].throughput_f64 > 1e9) {
            printf("%-22s  %12s  %12s  %10s\n",
                   results[i].expr, "error", "error", "N/A");
        } else {
            printf("%-22s  %12.2f  %12.2f  %9.2fx\n",
                   results[i].expr,
                   results[i].throughput_bool,
                   results[i].throughput_f64,
                   ratio);
            total_bool += results[i].throughput_bool;
            total_f64 += results[i].throughput_f64;
            valid_count++;
        }
    }

    printf("───────────────────────────────────────────────────────────────────────\n");
    if (valid_count > 0) {
        double avg_bool = total_bool / valid_count;
        double avg_f64 = total_f64 / valid_count;
        printf("%-22s  %12.2f  %12.2f  %9.2fx\n",
               "AVERAGE", avg_bool, avg_f64, avg_bool / avg_f64);
    }
    printf("═══════════════════════════════════════════════════════════════════════\n");

    /* Memory bandwidth analysis */
    printf("\nMemory Analysis (for simple 'a < b'):\n");
    printf("  - Input:  2 × %.1f MB = %.1f MB read\n",
           TOTAL_SIZE * sizeof(double) / 1e6,
           2 * TOTAL_SIZE * sizeof(double) / 1e6);
    printf("  - Output (bool): %.1f MB written\n",
           TOTAL_SIZE * sizeof(bool) / 1e6);
    printf("  - Output (f64):  %.1f MB written\n",
           TOTAL_SIZE * sizeof(double) / 1e6);

    double bw_bool = results[0].throughput_bool * (2 * sizeof(double) + sizeof(bool)) / 1000.0;
    double bw_f64 = results[0].throughput_f64 * (3 * sizeof(double)) / 1000.0;
    printf("  - Bandwidth (bool): %.2f GB/s\n", bw_bool);
    printf("  - Bandwidth (f64):  %.2f GB/s\n", bw_f64);

    printf("\nKey Observations:\n");
    printf("  - ME_BOOL output computes in float64, then converts to bool\n");
    printf("  - Ratio > 1.0 means bool output is faster (less memory written)\n");
    printf("  - Ratio < 1.0 means conversion overhead exceeds memory savings\n");
    printf("  - Complex expressions amortize conversion overhead better\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    free(a);
    free(b);
    free(c);

    return 0;
}
