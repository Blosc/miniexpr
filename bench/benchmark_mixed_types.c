#include "../src/miniexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define WARMUP_ITERS 5
#define BENCH_ITERS 5

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

typedef enum {
    EXPR_ADD,
    EXPR_SQRT_HYPOT
} ExprType;

void benchmark_mixed_types(const char *expr_str, ExprType expr_type, int nitems) {
    // Setup variables with different types (int64 and float32)
    int64_t *a_i64 = malloc(nitems * sizeof(int64_t));
    float *b_f32 = malloc(nitems * sizeof(float));
    double *result = malloc(nitems * sizeof(double));
    double *result_pure_c = malloc(nitems * sizeof(double));

    for (int i = 0; i < nitems; i++) {
        a_i64[i] = i + 1;
        b_f32[i] = (float) (i + 2);
    }

    // MiniExpr with type promotion
    // For mixed types, use ME_AUTO to infer the output type
    me_variable vars[] = {
        {"a", ME_INT64, a_i64},
        {"b", ME_FLOAT32, b_f32}
    };

    int err;
    me_expr *expr = me_compile(expr_str, vars, 2, result, nitems, ME_AUTO, &err);

    if (!expr) {
        printf("  ERROR: Failed to compile '%s'\n", expr_str);
        free(a_i64);
        free(b_f32);
        free(result);
        free(result_pure_c);
        return;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        me_eval(expr);
    }

    // Benchmark MiniExpr
    double t_start = get_time();
    for (int i = 0; i < BENCH_ITERS; i++) {
        me_eval(expr);
    }
    double t_expr = get_time() - t_start;

    // Pure C equivalent
    t_start = get_time();
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        if (expr_type == EXPR_ADD) {
            for (int i = 0; i < nitems; i++) {
                result_pure_c[i] = (double) a_i64[i] + (double) b_f32[i];
            }
        } else {
            // EXPR_SQRT_HYPOT
            for (int i = 0; i < nitems; i++) {
                double a_d = (double) a_i64[i];
                double b_d = (double) b_f32[i];
                result_pure_c[i] = sqrt(a_d * a_d + b_d * b_d);
            }
        }
    }
    double t_pure_c = get_time() - t_start;

    // Prevent optimization
    volatile double sum = result_pure_c[nitems - 1];
    (void) sum;

    // Calculate GFLOPS
    double ops_per_iter;
    if (expr_type == EXPR_ADD) {
        ops_per_iter = nitems * 3.0; // 2 conversions + 1 add
    } else {
        // EXPR_SQRT_HYPOT
        ops_per_iter = nitems * 7.0; // 2 conversions + 2 muls + 1 add + 1 sqrt + overhead
    }

    double gflops_expr = (BENCH_ITERS * ops_per_iter / t_expr) / 1e9;
    double gflops_pure_c = (BENCH_ITERS * ops_per_iter / t_pure_c) / 1e9;

    printf("%10d | %8.3f | %8.3f | %6.2f%%\n",
           nitems, gflops_expr, gflops_pure_c,
           (gflops_expr / gflops_pure_c) * 100.0);

    me_free(expr);
    free(a_i64);
    free(b_f32);
    free(result);
    free(result_pure_c);
}

int main() {
    printf("=== Mixed Type Performance Benchmark ===\n");
    printf("Variables: a (int64) + b (float32) -> output (float64)\n");
    printf("This tests automatic type promotion overhead\n\n");

    int sizes[] = {1000, 100000};

    // Test 1: Simple addition
    printf("Expression: a + b\n");
    printf("Vector Size | MiniExpr | Pure C   | Relative\n");
    printf("            | (GFLOPS) | (GFLOPS) | Speed   \n");
    printf("-----------------------------------------------\n");
    for (int i = 0; i < 2; i++) {
        benchmark_mixed_types("a + b", EXPR_ADD, sizes[i]);
    }

    printf("\n");

    // Test 2: Complex expression with sqrt
    printf("Expression: sqrt(a*a + b*b)\n");
    printf("Vector Size | MiniExpr | Pure C   | Relative\n");
    printf("            | (GFLOPS) | (GFLOPS) | Speed   \n");
    printf("-----------------------------------------------\n");
    for (int i = 0; i < 2; i++) {
        benchmark_mixed_types("sqrt(a*a + b*b)", EXPR_SQRT_HYPOT, sizes[i]);
    }

    printf("\n");
    return 0;
}
