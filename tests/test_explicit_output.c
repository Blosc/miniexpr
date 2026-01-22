/* Test explicit variable types with explicit output dtype
 *
 * This test verifies that when both variable types and output dtype
 * are explicitly specified, the behavior is correct:
 * - Variables keep their types during computation
 * - Result is cast to the specified output dtype
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "../src/miniexpr.h"
#include "minctest.h"



#define VECTOR_SIZE 10

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("\nTesting: %s\n", name); \
    tests_run++;

#define ASSERT_EQ(expected, actual, idx, fmt) \
    if ((expected) != (actual)) { \
        printf("  FAIL at [%d]: expected " fmt ", got " fmt "\n", idx, expected, actual); \
        tests_failed++; \
        return; \
    }

void test_mixed_types_float32_output() {
    TEST("Mixed types (INT32 + FLOAT64) with FLOAT32 output");

    int32_t a[VECTOR_SIZE];
    double b[VECTOR_SIZE];
    float result[VECTOR_SIZE];
    float expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i * 10;
        b[i] = i * 0.5;
        // Expected: INT32 + FLOAT64 promotes to FLOAT64, then cast to FLOAT32
        expected[i] = (float)((double)a[i] + b[i]);
    }

    me_variable vars[] = {
        {"a", ME_INT32},
        {"b", ME_FLOAT64}
    };

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a + b", vars, 2, ME_FLOAT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_FLOAT32) {
        printf("  FAIL: output dtype should be ME_FLOAT32, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float diff = fabsf(result[i] - expected[i]);
        if (diff > 1e-5f) {
            printf("  FAIL at [%d]: expected %.6f, got %.6f (diff: %.6f)\n",
                   i, expected[i], result[i], diff);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match (within float32 tolerance)\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

void test_float32_vars_float64_output() {
    TEST("FLOAT32 variables with FLOAT64 output");

    float x[VECTOR_SIZE];
    float y[VECTOR_SIZE];
    double result[VECTOR_SIZE];
    double expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        x[i] = (float)(i * 1.5f);
        y[i] = (float)(i * 0.5f);
        // Expected: FLOAT32 + FLOAT32 = FLOAT32, then cast to FLOAT64
        expected[i] = (double)(x[i] + y[i]);
    }

    me_variable vars[] = {
        {"x", ME_FLOAT32},
        {"y", ME_FLOAT32}
    };

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_FLOAT64) {
        printf("  FAIL: output dtype should be ME_FLOAT64, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {x, y};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        double diff = fabs(result[i] - expected[i]);
        if (diff > 1e-9) {
            printf("  FAIL at [%d]: expected %.9f, got %.9f (diff: %.9f)\n",
                   i, expected[i], result[i], diff);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match (within float64 tolerance)\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

void test_float32_with_constant_float64_output() {
    TEST("FLOAT32 variable + constant with FLOAT64 output");

    float a[VECTOR_SIZE];
    double result[VECTOR_SIZE];
    double expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (float)i;
        // Expected: FLOAT32 + FLOAT32 constant = FLOAT32, then cast to FLOAT64
        // Constant 3.0 is typed as FLOAT32 (NumPy convention), so computation is FLOAT32
        expected[i] = (double)(a[i] + 3.0f);
    }

    me_variable vars[] = {{"a", ME_FLOAT32}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a + 3.0", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_FLOAT64) {
        printf("  FAIL: output dtype should be ME_FLOAT64, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        double diff = fabs(result[i] - expected[i]);
        if (diff > 1e-9) {
            printf("  FAIL at [%d]: expected %.9f, got %.9f (diff: %.9f)\n",
                   i, expected[i], result[i], diff);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match (within float64 tolerance)\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

void test_comparison_explicit_bool_output() {
    TEST("Comparison with explicit BOOL output");

    int32_t a[VECTOR_SIZE];
    int32_t b[VECTOR_SIZE];
    bool result[VECTOR_SIZE];
    bool expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i * 2;
        b[i] = i * 2 + 1;
        expected[i] = (a[i] > b[i]);
    }

    me_variable vars[] = {
        {"a", ME_INT32},
        {"b", ME_INT32}
    };

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a > b", vars, 2, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_BOOL) {
        printf("  FAIL: output dtype should be ME_BOOL, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (result[i] != expected[i]) {
            printf("  FAIL at [%d]: expected %d, got %d\n",
                   i, (int)expected[i], (int)result[i]);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

int main() {
    printf("========================================================================\n");
    printf("TEST: Explicit Variable Types with Explicit Output Dtype\n");
    printf("========================================================================\n");
    printf("This test verifies that when both variable types and output dtype\n");
    printf("are explicitly specified:\n");
    printf("  - Variables keep their types during computation\n");
    printf("  - Result is correctly cast to the specified output dtype\n");
    printf("========================================================================\n");

    test_mixed_types_float32_output();
    test_float32_vars_float64_output();
    test_float32_with_constant_float64_output();
    test_comparison_explicit_bool_output();

    printf("\n========================================================================\n");
    printf("Test Summary\n");
    printf("========================================================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);
    printf("========================================================================\n");

    if (tests_failed == 0) {
        printf("✅ ALL TESTS PASSED\n");
    } else {
        printf("❌ SOME TESTS FAILED\n");
    }

    return (tests_failed == 0) ? 0 : 1;
}

