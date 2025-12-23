/* Test inverse hyperbolic functions (asinh, acosh, atanh) and their aliases */
#include "../src/miniexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void test_asinh() {
    TEST("asinh(x) - inverse hyperbolic sine");

    // Test values: asinh is defined for all real numbers
    double x[VECTOR_SIZE] = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = me_compile("asinh(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = asinh(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_asinh_alias() {
    TEST("asinh(x) vs arcsinh(x) - alias test");

    double x[VECTOR_SIZE] = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0};
    double result_a[VECTOR_SIZE] = {0};
    double result_arc[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr_a = me_compile("asinh(x)", vars, 1, ME_FLOAT64, &err);
    me_expr *expr_arc = me_compile("arcsinh(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr_a || !expr_arc) {
        printf("  FAIL: compilation error\n");
        tests_failed++;
        if (expr_a) me_free(expr_a);
        if (expr_arc) me_free(expr_arc);
        return;
    }

    const void *var_ptrs[] = {x};
    me_eval(expr_a, var_ptrs, 1, result_a, VECTOR_SIZE);
    me_eval(expr_arc, var_ptrs, 1, result_arc, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
            printf("  FAIL at [%d]: asinh=%.10f, arcsinh=%.10f\n", i, result_a[i], result_arc[i]);
            tests_failed++;
            me_free(expr_a);
            me_free(expr_arc);
            return;
        }
    }

    me_free(expr_a);
    me_free(expr_arc);
    printf("  PASS\n");
}

void test_acosh() {
    TEST("acosh(x) - inverse hyperbolic cosine");

    // Test values: acosh is defined for x >= 1
    double x[VECTOR_SIZE] = {1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0, 1000.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = me_compile("acosh(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = acosh(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_acosh_alias() {
    TEST("acosh(x) vs arccosh(x) - alias test");

    double x[VECTOR_SIZE] = {1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0, 1000.0};
    double result_a[VECTOR_SIZE] = {0};
    double result_arc[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr_a = me_compile("acosh(x)", vars, 1, ME_FLOAT64, &err);
    me_expr *expr_arc = me_compile("arccosh(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr_a || !expr_arc) {
        printf("  FAIL: compilation error\n");
        tests_failed++;
        if (expr_a) me_free(expr_a);
        if (expr_arc) me_free(expr_arc);
        return;
    }

    const void *var_ptrs[] = {x};
    me_eval(expr_a, var_ptrs, 1, result_a, VECTOR_SIZE);
    me_eval(expr_arc, var_ptrs, 1, result_arc, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
            printf("  FAIL at [%d]: acosh=%.10f, arccosh=%.10f\n", i, result_a[i], result_arc[i]);
            tests_failed++;
            me_free(expr_a);
            me_free(expr_arc);
            return;
        }
    }

    me_free(expr_a);
    me_free(expr_arc);
    printf("  PASS\n");
}

void test_atanh() {
    TEST("atanh(x) - inverse hyperbolic tangent");

    // Test values: atanh is defined for |x| < 1
    double x[VECTOR_SIZE] = {-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = me_compile("atanh(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = atanh(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_atanh_alias() {
    TEST("atanh(x) vs arctanh(x) - alias test");

    double x[VECTOR_SIZE] = {-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999};
    double result_a[VECTOR_SIZE] = {0};
    double result_arc[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr_a = me_compile("atanh(x)", vars, 1, ME_FLOAT64, &err);
    me_expr *expr_arc = me_compile("arctanh(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr_a || !expr_arc) {
        printf("  FAIL: compilation error\n");
        tests_failed++;
        if (expr_a) me_free(expr_a);
        if (expr_arc) me_free(expr_arc);
        return;
    }

    const void *var_ptrs[] = {x};
    me_eval(expr_a, var_ptrs, 1, result_a, VECTOR_SIZE);
    me_eval(expr_arc, var_ptrs, 1, result_arc, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
            printf("  FAIL at [%d]: atanh=%.10f, arctanh=%.10f\n", i, result_a[i], result_arc[i]);
            tests_failed++;
            me_free(expr_a);
            me_free(expr_arc);
            return;
        }
    }

    me_free(expr_a);
    me_free(expr_arc);
    printf("  PASS\n");
}

void test_inverse_hyperbolic_roundtrip() {
    TEST("Roundtrip test: asinh(sinh(x)) ≈ x");

    double x[VECTOR_SIZE] = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = me_compile("asinh(sinh(x))", vars, 1, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        ASSERT_NEAR(x[i], result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

int main() {
    printf("=== Testing Inverse Hyperbolic Functions ===\n\n");

    test_asinh();
    test_asinh_alias();
    test_acosh();
    test_acosh_alias();
    test_atanh();
    test_atanh_alias();
    test_inverse_hyperbolic_roundtrip();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}

