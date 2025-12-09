/* Benchmark key C99 types */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <sys/time.h>
#include "miniexpr.h"

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    const int n = 1000000;
    const int iter = 100;

    printf("MiniExpr C99 Type Benchmark (1M elements, 100 iterations)\n");
    printf("==========================================================\n\n");

    printf("Simple Expression (a+5):\n");
    printf("------------------------\n");

    /* INT32 - Simple */
    {
        int32_t *a = malloc(n * sizeof(int32_t));
        int32_t *result = malloc(n * sizeof(int32_t));
        for (int i = 0; i < n; i++) a[i] = i;

        me_variable vars[] = {{"a", ME_AUTO, a}};
        int err;
        me_expr *expr = me_compile("a+5", vars, 1, result, n, ME_INT32, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("int32_t:       %.4f s  (%.2f GFLOPS)\n",
               elapsed, (2.0 * n * iter / elapsed) / 1e9);

        me_free(expr);
        free(a);
        free(result);
    }

    /* UINT64 - Simple */
    {
        uint64_t *a = malloc(n * sizeof(uint64_t));
        uint64_t *result = malloc(n * sizeof(uint64_t));
        for (int i = 0; i < n; i++) a[i] = i;

        me_variable vars[] = {{"a", ME_AUTO, a}};
        int err;
        me_expr *expr = me_compile("a+5", vars, 1, result, n, ME_UINT64, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("uint64_t:      %.4f s  (%.2f GFLOPS)\n",
               elapsed, (2.0 * n * iter / elapsed) / 1e9);

        me_free(expr);
        free(a);
        free(result);
    }

    /* FLOAT - Simple */
    {
        float *a = malloc(n * sizeof(float));
        float *result = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) a[i] = i * 0.1f;

        me_variable vars[] = {{"a", ME_AUTO, a}};
        int err;
        me_expr *expr = me_compile("a+5", vars, 1, result, n, ME_FLOAT32, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("float:         %.4f s  (%.2f GFLOPS)\n",
               elapsed, (2.0 * n * iter / elapsed) / 1e9);

        me_free(expr);
        free(a);
        free(result);
    }

    /* DOUBLE - Simple */
    {
        double *a = malloc(n * sizeof(double));
        double *result = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) a[i] = i * 0.1;

        me_variable vars[] = {{"a", ME_AUTO, a}};
        int err;
        me_expr *expr = me_compile("a+5", vars, 1, result, n, ME_FLOAT64, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("double:        %.4f s  (%.2f GFLOPS)\n",
               elapsed, (2.0 * n * iter / elapsed) / 1e9);

        me_free(expr);
        free(a);
        free(result);
    }

    /* COMPLEX64 - Simple */
    {
        float complex *a = malloc(n * sizeof(float complex));
        float complex *result = malloc(n * sizeof(float complex));
        for (int i = 0; i < n; i++) a[i] = (float) i + (float) i * I;

        me_variable vars[] = {{"a", ME_AUTO, a}};
        int err;
        me_expr *expr = me_compile("a+5", vars, 1, result, n, ME_COMPLEX64, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("complex64:     %.4f s  (%.2f GFLOPS)\n",
               elapsed, (4.0 * n * iter / elapsed) / 1e9); // 2 adds (real+imag)

        me_free(expr);
        free(a);
        free(result);
    }

    printf("\nComplex Expression (sqrt(a*a+b*b)):\n");
    printf("-----------------------------------\n");

    /* INT32 - Complex */
    {
        int32_t *a = malloc(n * sizeof(int32_t));
        int32_t *b = malloc(n * sizeof(int32_t));
        int32_t *result = malloc(n * sizeof(int32_t));
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        me_variable vars[] = {{"a", ME_AUTO, a}, {"b", ME_AUTO, b}};
        int err;
        me_expr *expr = me_compile("sqrt(a*a+b*b)", vars, 2, result, n, ME_INT32, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("int32_t:       %.4f s  (%.2f GFLOPS)\n",
               elapsed, (6.0 * n * iter / elapsed) / 1e9); // 2 muls, 1 add, 1 sqrt

        me_free(expr);
        free(a);
        free(b);
        free(result);
    }

    /* UINT64 - Complex */
    {
        uint64_t *a = malloc(n * sizeof(uint64_t));
        uint64_t *b = malloc(n * sizeof(uint64_t));
        uint64_t *result = malloc(n * sizeof(uint64_t));
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        me_variable vars[] = {{"a", ME_AUTO, a}, {"b", ME_AUTO, b}};
        int err;
        me_expr *expr = me_compile("sqrt(a*a+b*b)", vars, 2, result, n, ME_UINT64, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("uint64_t:      %.4f s  (%.2f GFLOPS)\n",
               elapsed, (6.0 * n * iter / elapsed) / 1e9);

        me_free(expr);
        free(a);
        free(b);
        free(result);
    }

    /* FLOAT - Complex */
    {
        float *a = malloc(n * sizeof(float));
        float *b = malloc(n * sizeof(float));
        float *result = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.1f;
            b[i] = (n - i) * 0.1f;
        }

        me_variable vars[] = {{"a", ME_AUTO, a}, {"b", ME_AUTO, b}};
        int err;
        me_expr *expr = me_compile("sqrt(a*a+b*b)", vars, 2, result, n, ME_FLOAT32, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("float:         %.4f s  (%.2f GFLOPS)\n",
               elapsed, (6.0 * n * iter / elapsed) / 1e9);

        me_free(expr);
        free(a);
        free(b);
        free(result);
    }

    /* DOUBLE - Complex */
    {
        double *a = malloc(n * sizeof(double));
        double *b = malloc(n * sizeof(double));
        double *result = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.1;
            b[i] = (n - i) * 0.1;
        }

        me_variable vars[] = {{"a", ME_AUTO, a}, {"b", ME_AUTO, b}};
        int err;
        me_expr *expr = me_compile("sqrt(a*a+b*b)", vars, 2, result, n, ME_FLOAT64, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("double:        %.4f s  (%.2f GFLOPS)\n",
               elapsed, (6.0 * n * iter / elapsed) / 1e9);

        me_free(expr);
        free(a);
        free(b);
        free(result);
    }

    /* COMPLEX64 - Complex (magnitude) */
    {
        float complex *a = malloc(n * sizeof(float complex));
        float complex *b = malloc(n * sizeof(float complex));
        float complex *result = malloc(n * sizeof(float complex));
        for (int i = 0; i < n; i++) {
            a[i] = (float) i * 0.1f + (float) i * 0.1f * I;
            b[i] = (float) (n - i) * 0.1f + (float) (n - i) * 0.1f * I;
        }

        me_variable vars[] = {{"a", ME_AUTO, a}, {"b", ME_AUTO, b}};
        int err;
        me_expr *expr = me_compile("sqrt(a*a+b*b)", vars, 2, result, n, ME_COMPLEX64, &err);

        double start = get_time();
        for (int i = 0; i < iter; i++) me_eval(expr);
        double elapsed = get_time() - start;

        printf("complex64:     %.4f s  (%.2f GFLOPS)\n",
               elapsed, (24.0 * n * iter / elapsed) / 1e9); // Complex ops count more

        me_free(expr);
        free(a);
        free(b);
        free(result);
    }

    printf("\n✅ All types benchmarked successfully!\n");

    return 0;
}
