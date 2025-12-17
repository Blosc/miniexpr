/* Test two-parameter mathematical functions */
#include "../src/miniexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define VECTOR_SIZE 10
#define TOLERANCE 1e-9

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

#define ASSERT_NEAR(expected, actual, idx) \
    if (fabs((expected) - (actual)) > TOLERANCE) { \
        printf("  FAIL at [%d]: expected %.10f, got %.10f (diff: %.2e)\n", \
               idx, (double)(expected), (double)(actual), fabs((expected) - (actual))); \
        tests_failed++; \
        return; \
    }

void test_atan2() {
    TEST("atan2(y, x) - two-argument arctangent");

    double y[VECTOR_SIZE] = {1.0, 0.0, -1.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5};
    double x[VECTOR_SIZE] = {1.0, 1.0, 1.0, -1.0, -1.0, 3.0, -3.0, 0.8, -0.8, 2.5};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"y"}, {"x"}};

    int err;
    me_expr *expr = me_compile("atan2(y, x)", vars, 2, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {y, x};
    me_eval(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = atan2(y[i], x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_pow() {
    TEST("pow(base, exponent) - power function");

    double base[VECTOR_SIZE] = {2.0, 3.0, 4.0, 5.0, 2.5, 1.5, 10.0, 0.5, 8.0, 3.5};
    double exp[VECTOR_SIZE] = {3.0, 2.0, 0.5, 3.0, 2.0, 3.0, -1.0, 2.0, 1.0/3.0, 2.5};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"base"}, {"exp"}};

    int err;
    me_expr *expr = me_compile("pow(base, exp)", vars, 2, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {base, exp};
    me_eval(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = pow(base[i], exp[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_ncr() {
    TEST("ncr(n, r) - combinations (n choose r)");

    double n[VECTOR_SIZE] = {5.0, 10.0, 8.0, 6.0, 7.0, 9.0, 4.0, 12.0, 15.0, 20.0};
    double r[VECTOR_SIZE] = {2.0, 3.0, 3.0, 2.0, 3.0, 4.0, 2.0, 5.0, 7.0, 10.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"n"}, {"r"}};

    int err;
    me_expr *expr = me_compile("ncr(n, r)", vars, 2, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {n, r};
    me_eval(expr, var_ptrs, 2, result, VECTOR_SIZE);

    /* Expected values calculated manually:
     * ncr(5,2) = 10, ncr(10,3) = 120, ncr(8,3) = 56, ncr(6,2) = 15,
     * ncr(7,3) = 35, ncr(9,4) = 126, ncr(4,2) = 6, ncr(12,5) = 792,
     * ncr(15,7) = 6435, ncr(20,10) = 184756
     */
    double expected[VECTOR_SIZE] = {10.0, 120.0, 56.0, 15.0, 35.0, 126.0, 6.0, 792.0, 6435.0, 184756.0};

    for (int i = 0; i < VECTOR_SIZE; i++) {
        ASSERT_NEAR(expected[i], result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_npr() {
    TEST("npr(n, r) - permutations");

    double n[VECTOR_SIZE] = {5.0, 10.0, 8.0, 6.0, 7.0, 9.0, 4.0, 12.0, 10.0, 8.0};
    double r[VECTOR_SIZE] = {2.0, 3.0, 3.0, 2.0, 3.0, 4.0, 2.0, 5.0, 5.0, 4.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"n"}, {"r"}};

    int err;
    me_expr *expr = me_compile("npr(n, r)", vars, 2, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {n, r};
    me_eval(expr, var_ptrs, 2, result, VECTOR_SIZE);

    /* Expected values calculated manually:
     * npr(5,2) = 20, npr(10,3) = 720, npr(8,3) = 336, npr(6,2) = 30,
     * npr(7,3) = 210, npr(9,4) = 3024, npr(4,2) = 12, npr(12,5) = 95040,
     * npr(10,5) = 30240, npr(8,4) = 1680
     */
    double expected[VECTOR_SIZE] = {20.0, 720.0, 336.0, 30.0, 210.0, 3024.0, 12.0, 95040.0, 30240.0, 1680.0};

    for (int i = 0; i < VECTOR_SIZE; i++) {
        ASSERT_NEAR(expected[i], result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_mixed_expression() {
    TEST("mixed expression with two-param functions: pow(x, 2) + atan2(y, x)");

    double x[VECTOR_SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5};
    double y[VECTOR_SIZE] = {1.0, 1.0, 2.0, 2.0, 3.0, 1.2, 2.2, 3.2, 4.2, 5.2};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}, {"y"}};

    int err;
    me_expr *expr = me_compile("pow(x, 2) + atan2(y, x)", vars, 2, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x, y};
    me_eval(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = pow(x[i], 2.0) + atan2(y[i], x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_nested_two_param() {
    TEST("nested two-param functions: pow(2, pow(x, y))");

    double x[VECTOR_SIZE] = {1.0, 2.0, 1.5, 2.0, 1.0, 2.0, 1.2, 1.8, 2.5, 1.5};
    double y[VECTOR_SIZE] = {2.0, 1.0, 2.0, 2.0, 3.0, 1.5, 2.0, 1.5, 1.0, 2.5};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}, {"y"}};

    int err;
    me_expr *expr = me_compile("pow(2, pow(x, y))", vars, 2, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x, y};
    me_eval(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = pow(2.0, pow(x[i], y[i]));
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_two_param_with_mixed_types() {
    TEST("pow() with mixed types (int32 and float64)");

    int32_t base[VECTOR_SIZE] = {2, 3, 4, 5, 2, 3, 10, 2, 8, 3};
    double exp[VECTOR_SIZE] = {3.0, 2.0, 0.5, 3.0, 2.5, 3.5, -1.0, 4.0, 1.0/3.0, 2.2};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"base", ME_INT32}, {"exp", ME_FLOAT64}};

    int err;
    me_expr *expr = me_compile("pow(base, exp)", vars, 2, ME_AUTO, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {base, exp};
    me_eval(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = pow((double)base[i], exp[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

int main() {
    printf("=== Testing Two-Parameter Mathematical Functions ===\n\n");

    test_atan2();
    test_pow();
    test_ncr();
    test_npr();
    test_mixed_expression();
    test_nested_two_param();
    test_two_param_with_mixed_types();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
