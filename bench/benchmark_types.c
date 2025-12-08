/* Multi-type benchmark for MiniExpr */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "miniexpr.h"

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void benchmark_expression(const char *type_name, me_dtype dtype, int use_double,
                          const char *expr_str, const char *expr_name) {
    const int sizes[] = {100000, 1000000};
    int size_count = 2;

    for (int s = 0; s < size_count; s++) {
        int n = sizes[s];
        const int iterations = (n < 100000) ? 1000 : 100;

        // Allocate arrays based on type
        void *a, *b, *result, *native_result;
        size_t elem_size = use_double ? sizeof(double) : sizeof(float);

        a = malloc(n * elem_size);
        b = malloc(n * elem_size);
        result = malloc(n * elem_size);
        native_result = malloc(n * elem_size);

        // Initialize data
        if (use_double) {
            double *ad = (double *) a, *bd = (double *) b;
            for (int i = 0; i < n; i++) {
                ad[i] = i * 0.1;
                bd[i] = (n - i) * 0.1;
            }
        } else {
            float *af = (float *) a, *bf = (float *) b;
            for (int i = 0; i < n; i++) {
                af[i] = i * 0.1f;
                bf[i] = (n - i) * 0.1f;
            }
        }

        me_variable vars[] = {{"a", a}, {"b", b}};
        int err;

        printf("\n--- Vector size: %d, iterations: %d ---\n", n, iterations);
        printf("Expression: %s\n", expr_name);

        // Compile expression
        me_expr *expr = me_compile(expr_str, vars, 2, result, n, dtype, &err);

        if (!expr) {
            printf("ERROR: Failed to compile expression (error at %d)\n", err);
            continue;
        }

        // Determine which expression type for native C (OUTSIDE iteration loop!)
        int expr_type = 0; // 0=sqrt, 1=a+5
        if (strcmp(expr_str, "a+5") == 0) {
            expr_type = 1;
        }

        // Benchmark native C
        double start = get_time();
        if (use_double) {
            double *ad = (double *) a, *bd = (double *) b, *rd = (double *) native_result;
            if (expr_type == 1) {
                // a+5
                for (int iter = 0; iter < iterations; iter++) {
                    for (int i = 0; i < n; i++) {
                        rd[i] = ad[i] + 5.0;
                    }
                }
            } else {
                // sqrt(a*a+b*b)
                for (int iter = 0; iter < iterations; iter++) {
                    for (int i = 0; i < n; i++) {
                        rd[i] = sqrt(ad[i] * ad[i] + bd[i] * bd[i]);
                    }
                }
            }
        } else {
            float *af = (float *) a, *bf = (float *) b, *rf = (float *) native_result;
            if (expr_type == 1) {
                // a+5
                for (int iter = 0; iter < iterations; iter++) {
                    for (int i = 0; i < n; i++) {
                        rf[i] = af[i] + 5.0f;
                    }
                }
            } else {
                // sqrt(a*a+b*b)
                for (int iter = 0; iter < iterations; iter++) {
                    for (int i = 0; i < n; i++) {
                        rf[i] = sqrtf(af[i] * af[i] + bf[i] * bf[i]);
                    }
                }
            }
        }
        double native_time = get_time() - start;

        // Benchmark MiniExpr
        start = get_time();
        for (int iter = 0; iter < iterations; iter++) {
            me_eval(expr);
        }
        double me_time = get_time() - start;

        // Count operations based on expression
        long long ops;
        if (strcmp(expr_str, "sqrt(a*a+b*b)") == 0) {
            ops = (long long) iterations * n * 6; // 2 muls, 1 add, 1 sqrt, 2 loads
        } else if (strcmp(expr_str, "a+5") == 0) {
            ops = (long long) iterations * n * 2; // 1 add, 1 load
        } else {
            ops = (long long) iterations * n; // Default
        }

        printf("Native C:     %.4f s  (%.2f GFLOPS)  [baseline]\n",
               native_time, (ops / native_time) / 1e9);
        printf("MiniExpr:     %.4f s  (%.2f GFLOPS)  %.2fx slower\n",
               me_time, (ops / me_time) / 1e9, me_time / native_time);

        // Verify correctness
        int correct = 1;
        if (use_double) {
            double *rd = (double *) result, *nd = (double *) native_result;
            for (int i = 0; i < (n < 10 ? n : 10); i++) {
                if (fabs(rd[i] - nd[i]) > 1e-10) {
                    correct = 0;
                    printf("MISMATCH at %d: te=%.6f native=%.6f\n", i, rd[i], nd[i]);
                }
            }
        } else {
            float *rf = (float *) result, *nf = (float *) native_result;
            for (int i = 0; i < (n < 10 ? n : 10); i++) {
                if (fabsf(rf[i] - nf[i]) > 1e-5f) {
                    correct = 0;
                    printf("MISMATCH at %d: te=%.6f native=%.6f\n", i, rf[i], nf[i]);
                }
            }
        }
        printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

        me_free(expr);
        free(a);
        free(b);
        free(result);
        free(native_result);
    }
}

void benchmark_type(const char *type_name, me_dtype dtype, int use_double) {
    printf("\n========================================\n");
    printf("Testing with %s type\n", type_name);
    printf("========================================\n");

    // Test simple expression
    printf("\n>>> Simple Expression: a+5\n");
    benchmark_expression(type_name, dtype, use_double, "a+5", "a+5");

    // Test complex expression
    printf("\n>>> Complex Expression: sqrt(a*a+b*b)\n");
    benchmark_expression(type_name, dtype, use_double, "sqrt(a*a+b*b)", "sqrt(a*a+b*b)");
}

int main() {
    printf("MiniExpr Multi-Type Benchmark\n");
    printf("===============================\n");
    printf("(Testing with vector sizes: 100K, 1M)\n");

    // Benchmark float
    benchmark_type("float (32-bit)", ME_FLOAT32, 0);

    // Benchmark double
    benchmark_type("double (64-bit)", ME_FLOAT64, 1);

    printf("\n\nBenchmark complete!\n");
    return 0;
}
