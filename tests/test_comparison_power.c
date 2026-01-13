/* Test comparisons with power operations */
#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#define VECTOR_SIZE 10
#define TOLERANCE 1e-6

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

#define ASSERT_EQ(expected, actual, idx) \
    if (fabs((expected) - (actual)) > TOLERANCE) { \
        printf("  FAIL at [%d]: expected %.6f, got %.6f\n", idx, (double)(expected), (double)(actual)); \
        tests_failed++; \
        return; \
    }

void test_power_equality_comparison() {
    TEST("a1 ** 2 == (a1 + a2)");

    double a1[VECTOR_SIZE] = {2.0, 3.0, 4.0, 5.0, 1.0, 0.0, -2.0, 6.0, 2.5, 3.5};
    double a2[VECTOR_SIZE] = {2.0, 6.0, 12.0, 20.0, 0.0, 0.0, 6.0, 30.0, 3.75, 8.75};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a1"}, {"a2"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a1 ** 2 == (a1 + a2)", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a1, a2};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double left = a1[i] * a1[i];
        double right = a1[i] + a2[i];
        double expected = (fabs(left - right) < TOLERANCE) ? 1.0 : 0.0;
        printf("  [%d] %.6f ** 2 (%.6f) == (%.6f + %.6f) (%.6f) -> expected: %.0f, got: %.6f\n",
               i, a1[i], left, a1[i], a2[i], right, expected, result[i]);
        ASSERT_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_power_less_than_comparison() {
    TEST("a1 ** 2 < a2");

    double a1[VECTOR_SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 0.5, 10.0};
    double a2[VECTOR_SIZE] = {2.0, 5.0, 8.0, 15.0, 30.0, 1.0, 10.0, 15.0, 1.0, 50.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a1"}, {"a2"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a1 ** 2 < a2", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a1, a2};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double left = a1[i] * a1[i];
        double expected = (left < a2[i]) ? 1.0 : 0.0;
        printf("  [%d] %.6f ** 2 (%.6f) < %.6f -> expected: %.0f, got: %.6f\n",
               i, a1[i], left, a2[i], expected, result[i]);
        ASSERT_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_power_greater_equal_comparison() {
    TEST("a1 ** 3 >= a2");

    double a1[VECTOR_SIZE] = {2.0, 3.0, 1.0, 4.0, 2.0, 1.5, 2.5, 0.0, -2.0, 3.0};
    double a2[VECTOR_SIZE] = {8.0, 27.0, 0.0, 100.0, 8.0, 3.0, 20.0, 0.0, -8.0, 30.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a1"}, {"a2"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a1 ** 3 >= a2", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a1, a2};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double left = a1[i] * a1[i] * a1[i];
        double expected = (left >= a2[i]) ? 1.0 : 0.0;
        printf("  [%d] %.6f ** 3 (%.6f) >= %.6f -> expected: %.0f, got: %.6f\n",
               i, a1[i], left, a2[i], expected, result[i]);
        ASSERT_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_complex_power_comparison() {
    TEST("(a1 ** 2 + a2 ** 2) == a3");

    double a1[VECTOR_SIZE] = {3.0, 4.0, 5.0, 1.0, 0.0, 2.0, 6.0, 8.0, 1.5, 2.5};
    double a2[VECTOR_SIZE] = {4.0, 3.0, 12.0, 1.0, 0.0, 2.0, 8.0, 6.0, 2.0, 6.0};
    double a3[VECTOR_SIZE] = {25.0, 25.0, 169.0, 2.0, 0.0, 8.0, 100.0, 100.0, 6.25, 42.25};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a1"}, {"a2"}, {"a3"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("(a1 ** 2 + a2 ** 2) == a3", vars, 3, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a1, a2, a3};
    ME_EVAL_CHECK(expr, var_ptrs, 3, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double left = a1[i] * a1[i] + a2[i] * a2[i];
        double expected = (fabs(left - a3[i]) < TOLERANCE) ? 1.0 : 0.0;
        printf("  [%d] (%.6f ** 2 + %.6f ** 2) (%.6f) == %.6f -> expected: %.0f, got: %.6f\n",
               i, a1[i], a2[i], left, a3[i], expected, result[i]);
        ASSERT_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_power_cube_equality() {
    TEST("a1 ** 3 == (a1 * a1 * a1)");

    double a1[VECTOR_SIZE] = {2.0, -3.0, 1.5, 0.0, 4.0, -1.0, 2.5, -2.0, 3.5, 5.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a1"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a1 ** 3 == (a1 * a1 * a1)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a1};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double left = a1[i] * a1[i] * a1[i];
        double expected = (fabs(left - left) < TOLERANCE) ? 1.0 : 0.0;
        printf("  [%d] %.6f ** 3 (%.6f) == (%.6f * %.6f * %.6f) (%.6f) -> expected: %.0f, got: %.6f\n",
               i, a1[i], left, a1[i], a1[i], a1[i], left, expected, result[i]);
        ASSERT_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

int main() {
    printf("=== Testing Comparison Operators with Power Operations ===\n\n");

    test_power_equality_comparison();
    test_power_less_than_comparison();
    test_power_greater_equal_comparison();
    test_complex_power_comparison();
    test_power_cube_equality();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
