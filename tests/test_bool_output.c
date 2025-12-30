/* Test that comparison operations output bool arrays */
#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>



#define VECTOR_SIZE 10

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

#define ASSERT_BOOL_EQ(expected, actual, idx) \
    if ((expected) != (actual)) { \
        printf("  FAIL at [%d]: expected %d, got %d\n", idx, (int)(expected), (int)(actual)); \
        tests_failed++; \
        return; \
    }

void test_comparison_bool_output() {
    TEST("a1 ** 2 == (a1 + a2) -> bool output");

    double a1[VECTOR_SIZE] = {2.0, 3.0, 4.0, 5.0, 1.0, 0.0, -2.0, 6.0, 2.5, 3.5};
    double a2[VECTOR_SIZE] = {2.0, 6.0, 12.0, 20.0, 0.0, 0.0, 6.0, 30.0, 3.75, 8.75};
    bool result[VECTOR_SIZE] = {0};

    // Explicitly specify variable dtypes as ME_FLOAT64
    me_variable vars[] = {{"a1", ME_FLOAT64}, {"a2", ME_FLOAT64}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a1 ** 2 == (a1 + a2)", vars, 2, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    // Check that the expression dtype is ME_BOOL
    me_dtype dtype = me_get_dtype(expr);
    printf("  Expression dtype: %d (expected ME_BOOL=%d)\n", dtype, ME_BOOL);
    if (dtype != ME_BOOL) {
        printf("  FAIL: dtype should be ME_BOOL\n");
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a1, a2};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double left = a1[i] * a1[i];
        double right = a1[i] + a2[i];
        bool expected = (left == right);  // All should be true in this test data
        printf("  [%d] %.1f ** 2 == (%.1f + %.1f) -> expected: %d, got: %d\n",
               i, a1[i], a1[i], a2[i], (int)expected, (int)result[i]);
        ASSERT_BOOL_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_auto_dtype_comparison() {
    TEST("a1 < a2 with ME_AUTO -> should infer ME_BOOL");

    double a1[VECTOR_SIZE] = {1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 9.0, 0.0};
    double a2[VECTOR_SIZE] = {2.0, 4.0, 4.0, 6.0, 3.0, 7.0, 5.0, 5.0, 10.0, 1.0};
    bool result[VECTOR_SIZE] = {0};

    // For ME_AUTO output, must specify explicit variable dtypes
    me_variable vars[] = {{"a1", ME_FLOAT64}, {"a2", ME_FLOAT64}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a1 < a2", vars, 2, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    // Check that the expression dtype is ME_BOOL when inferred
    me_dtype dtype = me_get_dtype(expr);
    printf("  Expression dtype: %d (expected ME_BOOL=%d)\n", dtype, ME_BOOL);
    if (dtype != ME_BOOL) {
        printf("  FAIL: dtype should be inferred as ME_BOOL for comparison\n");
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a1, a2};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        bool expected = (a1[i] < a2[i]);
        printf("  [%d] %.1f < %.1f -> expected: %d, got: %d\n",
               i, a1[i], a2[i], (int)expected, (int)result[i]);
        ASSERT_BOOL_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

int main() {
    printf("=== Testing Bool Output for Comparisons ===\n\n");

    test_comparison_bool_output();
    test_auto_dtype_comparison();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
