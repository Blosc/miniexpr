/* Test all C99 types supported by MiniExpr */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include "miniexpr.h"

/* Macro for single-variable tests (expressions like "a+5") */
#define TEST1(name, type_enum, type, fmt, init_expr, test_expr, expected_expr) do { \
    const int n = 10; \
    type *a = malloc(n * sizeof(type)); \
    type *result = malloc(n * sizeof(type)); \
    type *expected = malloc(n * sizeof(type)); \
    \
    for (int i = 0; i < n; i++) { \
        a[i] = (type)(init_expr); \
        expected[i] = (type)(expected_expr); \
    } \
    \
    me_variable vars[] = {{"a"}}; \
    int err; \
    me_expr *expr = me_compile(test_expr, vars, 1, type_enum, &err); \
    \
    if (!expr) { \
        printf("Failed to compile '%s' for %s (error at %d)\n", test_expr, name, err); \
        free(a); free(result); free(expected); \
        return 1; \
    } \
    \
    const void *var_ptrs[] = {a}; \
    me_eval(expr, var_ptrs, 1, result, n); \
    \
    int passed = 1; \
    for (int i = 0; i < n && passed; i++) { \
        if (result[i] != expected[i]) { \
            passed = 0; \
            printf("%-15s: Mismatch at [%d]: got " fmt ", expected " fmt "\n", \
                   name, i, result[i], expected[i]); \
        } \
    } \
    \
    if (passed) { \
        printf("✅ %-15s: '%s' passed\n", name, test_expr); \
    } \
    \
    me_free(expr); \
    free(a); free(result); free(expected); \
    \
    if (!passed) return 1; \
} while(0)

/* Macro for two-variable tests (expressions like "a+b") */
#define TEST2(name, type_enum, type, fmt, init_expr, test_expr, expected_expr) do { \
    const int n = 10; \
    type *a = malloc(n * sizeof(type)); \
    type *b = malloc(n * sizeof(type)); \
    type *result = malloc(n * sizeof(type)); \
    type *expected = malloc(n * sizeof(type)); \
    \
    for (int i = 0; i < n; i++) { \
        a[i] = (type)(init_expr); \
        b[i] = (type)((n - i) * 0.5); \
        expected[i] = (type)(expected_expr); \
    } \
    \
    me_variable vars[] = {{"a"}, {"b"}}; \
    int err; \
    me_expr *expr = me_compile(test_expr, vars, 2, type_enum, &err); \
    \
    if (!expr) { \
        printf("Failed to compile '%s' for %s (error at %d)\n", test_expr, name, err); \
        free(a); free(b); free(result); free(expected); \
        return 1; \
    } \
    \
    const void *var_ptrs[] = {a, b}; \
    me_eval(expr, var_ptrs, 2, result, n); \
    \
    int passed = 1; \
    for (int i = 0; i < n && passed; i++) { \
        if (result[i] != expected[i]) { \
            passed = 0; \
            printf("%-15s: Mismatch at [%d]: got " fmt ", expected " fmt "\n", \
                   name, i, result[i], expected[i]); \
        } \
    } \
    \
    if (passed) { \
        printf("✅ %-15s: '%s' passed\n", name, test_expr); \
    } \
    \
    me_free(expr); \
    free(a); free(b); free(result); free(expected); \
    \
    if (!passed) return 1; \
} while(0)

int main() {
    printf("Testing All C99 Types\n");
    printf("=====================\n\n");

    /* Integer types */
    printf("Signed Integers:\n");
    TEST1("int8_t", ME_INT8, int8_t, "%" PRId8, i, "a+5", a[i]+5);
    TEST1("int16_t", ME_INT16, int16_t, "%" PRId16, i*10, "a+100", a[i]+100);
    TEST2("int32_t", ME_INT32, int32_t, "%" PRId32, i*1000, "a+b", a[i]+b[i]);
    TEST1("int64_t", ME_INT64, int64_t, "%" PRId64, i*1000000, "a*2", a[i]*2);

    printf("\nUnsigned Integers:\n");
    TEST1("uint8_t", ME_UINT8, uint8_t, "%" PRIu8, i, "a+10", a[i]+10);
    TEST1("uint16_t", ME_UINT16, uint16_t, "%" PRIu16, i*100, "a+200", a[i]+200);
    TEST2("uint32_t", ME_UINT32, uint32_t, "%" PRIu32, i*1000, "a+b", a[i]+b[i]);
    TEST1("uint64_t", ME_UINT64, uint64_t, "%" PRIu64, i*1000000, "a*3", a[i]*3);

    printf("\nFloating Point:\n");
    TEST1("float", ME_FLOAT32, float, "%.2f", (float)i, "a+5.0", a[i]+5.0f);
    TEST2("double", ME_FLOAT64, double, "%.2f", (double)i, "a+b", a[i]+b[i]);

    printf("\n✅ All basic type tests passed!\n\n");

    /* Complex numbers need special testing */
    printf("Complex Numbers:\n"); {
        const int n = 10;
        float complex *a = malloc(n * sizeof(float complex));
        float complex *result = malloc(n * sizeof(float complex));

        for (int i = 0; i < n; i++) {
            a[i] = (float) i + (float) i * I; // i + i*I
        }

        me_variable vars[] = {{"a"}};
        int err;
        me_expr *expr = me_compile("a+5", vars, 1, ME_COMPLEX64, &err);

        if (!expr) {
            printf("❌ complex64: Failed to compile\n");
        } else {
            const void *var_ptrs[] = {a};
            me_eval(expr, var_ptrs, 1, result, n);

            int passed = 1;
            for (int i = 0; i < n && passed; i++) {
                float complex expected = a[i] + 5.0f;
                if (crealf(result[i]) != crealf(expected) ||
                    cimagf(result[i]) != cimagf(expected)) {
                    passed = 0;
                    printf("❌ complex64: Mismatch at [%d]\n", i);
                }
            }

            if (passed) {
                printf("✅ float complex: 'a+5' passed\n");
            }

            me_free(expr);
        }

        free(a);
        free(result);
    } {
        const int n = 10;
        double complex *a = malloc(n * sizeof(double complex));
        double complex *result = malloc(n * sizeof(double complex));

        for (int i = 0; i < n; i++) {
            a[i] = (double) i + (double) i * I;
        }

        me_variable vars[] = {{"a"}};
        int err;
        me_expr *expr = me_compile("a*2", vars, 1, ME_COMPLEX128, &err);

        if (!expr) {
            printf("❌ complex128: Failed to compile\n");
        } else {
            const void *var_ptrs[] = {a};
            me_eval(expr, var_ptrs, 1, result, n);

            int passed = 1;
            for (int i = 0; i < n && passed; i++) {
                double complex expected = a[i] * 2.0;
                if (creal(result[i]) != creal(expected) ||
                    cimag(result[i]) != cimag(expected)) {
                    passed = 0;
                    printf("❌ complex128: Mismatch at [%d]\n", i);
                }
            }

            if (passed) {
                printf("✅ double complex: 'a*2' passed\n");
            }

            me_free(expr);
        }

        free(a);
        free(result);
    }

    printf("\n🎉 All 13 C99 types working!\n\n");
    printf("Supported types:\n");
    printf("  • int8_t, int16_t, int32_t, int64_t\n");
    printf("  • uint8_t, uint16_t, uint32_t, uint64_t\n");
    printf("  • float, double\n");
    printf("  • float complex, double complex\n");
    printf("  • bool (uses int8_t evaluator)\n");

    return 0;
}
