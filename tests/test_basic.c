/*
 * Basic smoke test for miniexpr API
 *
 * This test provides quick validation of core functionality.
 * It intentionally overlaps with other tests to serve as a simple
 * sanity check and documentation reference.
 *
 * For comprehensive testing, see:
 *   - test_all_types.c (all 13 C99 data types)
 *   - test_operators.c (bitwise, logical, comparison operators)
 *   - test_mixed_types.c (type promotion and ME_AUTO)
 *   - test_chunked_eval.c (chunked processing patterns)
 *   - test_threadsafe_chunk.c (parallel thread safety)
 *
 * This test is intentionally simple and readable for
 * documentation purposes and quick development iteration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../src/miniexpr.h"
#include "minctest.h"



#define TEST_SIZE 100

int test_simple_expression() {
    printf("\n=== Test 1: Simple expression (a + b) ===\n");

    double *a = malloc(TEST_SIZE * sizeof(double));
    double *b = malloc(TEST_SIZE * sizeof(double));
    double *result = malloc(TEST_SIZE * sizeof(double));

    for (int i = 0; i < TEST_SIZE; i++) {
        a[i] = i * 1.0;
        b[i] = i * 2.0;
    }

    me_variable vars[] = {{"a"}, {"b"}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a + b", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: Compilation error at position %d\n", err);
        free(a);
        free(b);
        free(result);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, TEST_SIZE);

    int passed = 1;
    for (int i = 0; i < TEST_SIZE; i++) {
        double expected = a[i] + b[i];
        if (fabs(result[i] - expected) > 1e-10) {
            printf("  ❌ FAILED at [%d]: got %.2f, expected %.2f\n", i, result[i], expected);
            passed = 0;
            break;
        }
    }

    if (passed) {
        printf("  ✅ PASSED\n");
    }

    me_free(expr);
    free(a);
    free(b);
    free(result);
    return passed ? 0 : 1;
}

int test_complex_expression() {
    printf("\n=== Test 2: Complex expression (sqrt(a*a + b*b)) ===\n");

    double *a = malloc(TEST_SIZE * sizeof(double));
    double *b = malloc(TEST_SIZE * sizeof(double));
    double *result = malloc(TEST_SIZE * sizeof(double));

    for (int i = 0; i < TEST_SIZE; i++) {
        a[i] = i * 0.3;
        b[i] = i * 0.4;
    }

    me_variable vars[] = {{"a"}, {"b"}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: Compilation error\n");
        free(a);
        free(b);
        free(result);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, TEST_SIZE);

    int passed = 1;
    for (int i = 0; i < TEST_SIZE; i++) {
        double expected = sqrt(a[i] * a[i] + b[i] * b[i]);
        if (fabs(result[i] - expected) > 1e-10) {
            printf("  ❌ FAILED at [%d]: got %.6f, expected %.6f\n", i, result[i], expected);
            passed = 0;
            break;
        }
    }

    if (passed) {
        printf("  ✅ PASSED\n");
    }

    me_free(expr);
    free(a);
    free(b);
    free(result);
    return passed ? 0 : 1;
}

int test_integer_types() {
    printf("\n=== Test 3: Integer types (int32_t) ===\n");

    int32_t *a = malloc(TEST_SIZE * sizeof(int32_t));
    int32_t *b = malloc(TEST_SIZE * sizeof(int32_t));
    int32_t *result = malloc(TEST_SIZE * sizeof(int32_t));

    for (int i = 0; i < TEST_SIZE; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    me_variable vars[] = {{"a"}, {"b"}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a + b", vars, 2, ME_INT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: Compilation error\n");
        free(a);
        free(b);
        free(result);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, TEST_SIZE);

    int passed = 1;
    for (int i = 0; i < TEST_SIZE; i++) {
        int32_t expected = a[i] + b[i];
        if (result[i] != expected) {
            printf("  ❌ FAILED at [%d]: got %d, expected %d\n", i, result[i], expected);
            passed = 0;
            break;
        }
    }

    if (passed) {
        printf("  ✅ PASSED\n");
    }

    me_free(expr);
    free(a);
    free(b);
    free(result);
    return passed ? 0 : 1;
}

int test_mixed_types() {
    printf("\n=== Test 4: Mixed types (int32 + float64) ===\n");

    int32_t *a = malloc(TEST_SIZE * sizeof(int32_t));
    double *b = malloc(TEST_SIZE * sizeof(double));
    double *result = malloc(TEST_SIZE * sizeof(double));

    for (int i = 0; i < TEST_SIZE; i++) {
        a[i] = i;
        b[i] = i * 0.5;
    }

    me_variable vars[] = {{"a", ME_INT32}, {"b", ME_FLOAT64}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a + b", vars, 2, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: Compilation error\n");
        free(a);
        free(b);
        free(result);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, TEST_SIZE);

    int passed = 1;
    for (int i = 0; i < TEST_SIZE; i++) {
        double expected = a[i] + b[i];
        if (fabs(result[i] - expected) > 1e-10) {
            printf("  ❌ FAILED at [%d]: got %.2f, expected %.2f\n", i, result[i], expected);
            passed = 0;
            break;
        }
    }

    if (passed) {
        printf("  ✅ PASSED (inferred type: %d)\n", me_get_dtype(expr));
    }

    me_free(expr);
    free(a);
    free(b);
    free(result);
    return passed ? 0 : 1;
}

int test_fac_ln() {
    printf("\n=== Test 5: fac and ln ===\n");

    const int n = 10;
    double *a = malloc((size_t)n * sizeof(double));
    double *fac_out = malloc((size_t)n * sizeof(double));
    double *ln_out = malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) {
        a[i] = (double)i;
    }

    int err;
    me_expr *fac_expr = NULL;
    me_expr *ln_expr = NULL;
    me_variable vars[] = {{"a"}};

    if (me_compile("fac(a)", vars, 1, ME_FLOAT64, &err, &fac_expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: fac compilation error at position %d\n", err);
        free(a);
        free(fac_out);
        free(ln_out);
        return 1;
    }
    if (me_compile("ln(a + 1)", vars, 1, ME_FLOAT64, &err, &ln_expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: ln compilation error at position %d\n", err);
        me_free(fac_expr);
        free(a);
        free(fac_out);
        free(ln_out);
        return 1;
    }

    const void *var_ptrs[] = {a};
    ME_EVAL_CHECK(fac_expr, var_ptrs, 1, fac_out, n);
    ME_EVAL_CHECK(ln_expr, var_ptrs, 1, ln_out, n);

    int passed = 1;
    for (int i = 0; i < n; i++) {
        double expected_fac = 1.0;
        for (int j = 1; j <= i; j++) {
            expected_fac *= (double)j;
        }
        double expected_ln = log(a[i] + 1.0);
        if (fabs(fac_out[i] - expected_fac) > 1e-10) {
            printf("  ❌ FAILED fac at [%d]: got %.6f, expected %.6f\n",
                   i, fac_out[i], expected_fac);
            passed = 0;
            break;
        }
        if (fabs(ln_out[i] - expected_ln) > 1e-10) {
            printf("  ❌ FAILED ln at [%d]: got %.6f, expected %.6f\n",
                   i, ln_out[i], expected_ln);
            passed = 0;
            break;
        }
    }

    if (passed) {
        printf("  ✅ PASSED\n");
    }

    me_free(fac_expr);
    me_free(ln_expr);
    free(a);
    free(fac_out);
    free(ln_out);
    return passed ? 0 : 1;
}

int main() {
    printf("========================================\n");
    printf("MiniExpr Basic Functionality Tests\n");
    printf("========================================\n");

    int failures = 0;
    failures += test_simple_expression();
    failures += test_complex_expression();
    failures += test_integer_types();
    failures += test_mixed_types();
    failures += test_fac_ln();

    printf("\n========================================\n");
    if (failures == 0) {
        printf("✅ All tests passed!\n");
    } else {
        printf("❌ %d test(s) failed\n", failures);
    }
    printf("========================================\n");

    return failures;
}
