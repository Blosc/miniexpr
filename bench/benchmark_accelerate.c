/* Benchmark: Accelerate vs Pure C comparison */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "../src/miniexpr.h"

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void benchmark_expression(const char *name, const char *expr_str,
                         int nitems, int iterations) {
    // Allocate arrays
    double *a = malloc(nitems * sizeof(double));
    double *b = malloc(nitems * sizeof(double));
    double *result = malloc(nitems * sizeof(double));

    // Initialize
    for (int i = 0; i < nitems; i++) {
        a[i] = (double)i * 0.1 + 1.0;
        b[i] = (double)(nitems - i) * 0.05 + 1.0;
    }

    // Compile expression
    me_variable vars[] = {{"a", ME_AUTO, a}, {"b", ME_AUTO, b}};
    int err;
    me_expr *expr = me_compile(expr_str, vars, 2, result, nitems, ME_FLOAT64, &err);

    if (!expr) {
        printf("Failed to compile '%s'\n", expr_str);
        free(a);
        free(b);
        free(result);
        return;
    }

    // Warmup
    for (int i = 0; i < 5; i++) {
        me_eval(expr);
    }

    // Benchmark
    double t_start = get_time();
    for (int i = 0; i < iterations; i++) {
        me_eval(expr);
    }
    double t_elapsed = get_time() - t_start;

    // Calculate throughput
    double ops_per_elem;
    if (strcmp(name, "Pythagorean") == 0) {
        ops_per_elem = 5.0;  // 2 muls + 1 add + 1 sqrt + overhead
    } else if (strcmp(name, "Trigonometric") == 0) {
        ops_per_elem = 3.0;  // sin + cos + mul
    } else if (strcmp(name, "Complex Math") == 0) {
        ops_per_elem = 8.0;  // sin + cos + mul + exp + log + add + div
    } else {
        ops_per_elem = 3.0;  // Default estimate
    }

    double total_ops = (double)nitems * iterations * ops_per_elem;
    double gflops = total_ops / t_elapsed / 1e9;
    double throughput_gb = (double)nitems * iterations * sizeof(double) * 3.0 / t_elapsed / 1e9;

    printf("%-20s: %8.4f s  (%6.2f GFLOPS, %6.2f GB/s)\n",
           name, t_elapsed, gflops, throughput_gb);

    me_free(expr);
    free(a);
    free(b);
    free(result);
}

int main() {
    printf("========================================\n");
    printf("Accelerate vs Pure C Benchmark\n");
    printf("========================================\n");
#ifdef __APPLE__
    printf("Platform: macOS (Accelerate ENABLED)\n");
#else
    printf("Platform: Other (Pure C loops)\n");
#endif
    printf("\n");

    const int sizes[] = {100000, 1000000, 10000000};
    const char *size_names[] = {"100K", "1M", "10M"};

    for (int s = 0; s < 3; s++) {
        int nitems = sizes[s];
        int iterations = (nitems <= 1000000) ? 1000 : 100;

        printf("Vector Size: %s elements (%d iterations)\n", size_names[s], iterations);
        printf("========================================\n");

        // Test 1: Pythagorean formula (sqrt(a*a + b*b))
        benchmark_expression("Pythagorean", "sqrt(a*a + b*b)", nitems, iterations);

        // Test 2: Trigonometric (sin(a) * cos(b))
        benchmark_expression("Trigonometric", "sin(a) * cos(b)", nitems, iterations);

        // Test 3: Complex mathematical expression
        benchmark_expression("Complex Math", "sin(a) * cos(b) + exp(a/10) / log(b)",
                           nitems, iterations);

        // Test 4: Simple arithmetic (baseline)
        benchmark_expression("Simple Add", "a + b", nitems, iterations);

        // Test 5: Polynomial
        benchmark_expression("Polynomial", "a*a*a + 2*a*a + 3*a + 4", nitems, iterations);

        printf("\n");
    }

    printf("========================================\n");
    printf("Notes:\n");
#ifdef __APPLE__
    printf("- Using Apple Accelerate framework\n");
    printf("- vDSP for basic ops (+, -, *, /)\n");
    printf("- vForce for math functions (sin, cos, sqrt, exp, log)\n");
    printf("\nTo compare with pure C:\n");
    printf("  Rebuild without Accelerate and re-run\n");
#else
    printf("- Using pure C loops with compiler auto-vectorization\n");
    printf("- To test Accelerate: compile on macOS\n");
#endif
    printf("========================================\n");

    return 0;
}
