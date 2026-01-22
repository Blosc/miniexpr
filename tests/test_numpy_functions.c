/* Test NumPy-compatible functions: expm1, log1p, log2, logaddexp, round, sign, square, trunc */
#include "../src/miniexpr.h"
#include "minctest.h"
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

void test_expm1() {
    TEST("expm1(x) - exp(x) - 1, more accurate for small x");

    // Test values including small values where expm1 is more accurate
    double x[VECTOR_SIZE] = {-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 10.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("expm1(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = expm1(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_log1p() {
    TEST("log1p(x) - log(1 + x), more accurate for small x");

    // Test values including small values where log1p is more accurate
    double x[VECTOR_SIZE] = {-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("log1p(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = log1p(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_log2() {
    TEST("log2(x) - base-2 logarithm");

    double x[VECTOR_SIZE] = {0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1024.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("log2(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = log2(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_logaddexp() {
    TEST("logaddexp(a, b) - log(exp(a) + exp(b)), numerically stable");

    double a[VECTOR_SIZE] = {1.0, 2.0, 0.0, -1.0, 10.0, -5.0, 100.0, -100.0, 0.5, -0.5};
    double b[VECTOR_SIZE] = {2.0, 1.0, 0.0, -2.0, 5.0, -3.0, 50.0, -50.0, -0.5, 0.5};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("logaddexp(a, b)", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        // Reference implementation (same as in miniexpr.c)
        double max_val = (a[i] > b[i]) ? a[i] : b[i];
        double min_val = (a[i] > b[i]) ? b[i] : a[i];
        double expected;
        if (a[i] == b[i]) {
            expected = a[i] + log1p(1.0);  // log(2*exp(a)) = a + log(2)
        } else {
            expected = max_val + log1p(exp(min_val - max_val));
        }
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_expm1_small_values() {
    TEST("expm1(x) - accuracy test for very small values");

    // Very small values where expm1 is significantly more accurate
    double x[5] = {1e-10, 1e-8, 1e-6, 1e-4, 1e-2};
    double result[5] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("expm1(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, 5);

    for (int i = 0; i < 5; i++) {
        double expected = expm1(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_log1p_small_values() {
    TEST("log1p(x) - accuracy test for very small values");

    // Very small values where log1p is significantly more accurate
    double x[5] = {1e-10, 1e-8, 1e-6, 1e-4, 1e-2};
    double result[5] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("log1p(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, 5);

    for (int i = 0; i < 5; i++) {
        double expected = log1p(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_logaddexp_extreme_values() {
    TEST("logaddexp(a, b) - test with extreme values");

    // Test cases that would overflow with naive exp(a) + exp(b)
    double a[5] = {700.0, -700.0, 100.0, -100.0, 50.0};
    double b[5] = {700.0, -700.0, 50.0, -50.0, 100.0};
    double result[5] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("logaddexp(a, b)", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, 5);

    for (int i = 0; i < 5; i++) {
        // Reference implementation
        double max_val = (a[i] > b[i]) ? a[i] : b[i];
        double min_val = (a[i] > b[i]) ? b[i] : a[i];
        double expected;
        if (a[i] == b[i]) {
            expected = a[i] + log1p(1.0);
        } else {
            expected = max_val + log1p(exp(min_val - max_val));
        }
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_mixed_expressions() {
    TEST("Mixed expressions with new functions");

    double x[5] = {0.1, 0.5, 1.0, 2.0, 10.0};
    double result[5] = {0};

    me_variable vars[] = {{"x"}};
    int err;

    // Test: log1p(expm1(x)) should equal x (for x > -1)
    me_expr *expr1 = NULL;
    int rc_expr1 = me_compile("log1p(expm1(x))", vars, 1, ME_FLOAT64, &err, &expr1);
    if (rc_expr1 != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error for log1p(expm1(x)) at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr1, var_ptrs, 1, result, 5);

    for (int i = 0; i < 5; i++) {
        // log1p(expm1(x)) should equal x (within numerical precision)
        double expected = x[i];
        ASSERT_NEAR(expected, result[i], i);
    }
    me_free(expr1);

    // Test: log2(x) = log(x) / log(2)
    me_expr *expr2 = NULL;
    int rc_expr2 = me_compile("log2(x)", vars, 1, ME_FLOAT64, &err, &expr2);
    me_expr *expr3 = NULL;
    int rc_expr3 = me_compile("log(x) / log(2)", vars, 1, ME_FLOAT64, &err, &expr3);
    if (!expr2 || !expr3) {
        printf("  FAIL: compilation error for log2 comparison\n");
        if (rc_expr2 == ME_COMPILE_SUCCESS) me_free(expr2);
        if (rc_expr3 == ME_COMPILE_SUCCESS) me_free(expr3);
        tests_failed++;
        return;
    }

    double result2[5] = {0};
    double result3[5] = {0};
    ME_EVAL_CHECK(expr2, var_ptrs, 1, result2, 5);
    ME_EVAL_CHECK(expr3, var_ptrs, 1, result3, 5);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(result2[i], result3[i], i);
    }
    me_free(expr2);
    me_free(expr3);

    printf("  PASS\n");
}

void test_round_func() {
    TEST("round(x) - round to nearest integer");

    double x[VECTOR_SIZE] = {1.4, 1.5, 1.6, -1.4, -1.5, -1.6, 2.5, -2.5, 0.0, 3.14159};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("round(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = round(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_sign() {
    TEST("sign(x) - sign function (-1, 0, or 1)");

    double x[VECTOR_SIZE] = {-5.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 5.0, 100.0, -100.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sign(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected;
        if (x[i] > 0.0) expected = 1.0;
        else if (x[i] < 0.0) expected = -1.0;
        else expected = 0.0;
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_square() {
    TEST("square(x) - x * x");

    double x[VECTOR_SIZE] = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, -10.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("square(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = x[i] * x[i];
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_trunc_func() {
    TEST("trunc(x) - truncate towards zero");

    double x[VECTOR_SIZE] = {1.4, 1.5, 1.6, -1.4, -1.5, -1.6, 2.7, -2.7, 0.0, 3.14159};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("trunc(x)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = trunc(x[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_square_vs_pow() {
    TEST("square(x) == pow(x, 2)");

    double x[5] = {-3.0, -1.0, 0.0, 1.0, 5.0};
    double result1[5] = {0};
    double result2[5] = {0};

    me_variable vars[] = {{"x"}};

    int err;
    me_expr *expr1 = NULL;
    int rc_expr1 = me_compile("square(x)", vars, 1, ME_FLOAT64, &err, &expr1);
    me_expr *expr2 = NULL;
    int rc_expr2 = me_compile("pow(x, 2)", vars, 1, ME_FLOAT64, &err, &expr2);

    if (!expr1 || !expr2) {
        printf("  FAIL: compilation error\n");
        if (rc_expr1 == ME_COMPILE_SUCCESS) me_free(expr1);
        if (rc_expr2 == ME_COMPILE_SUCCESS) me_free(expr2);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {x};
    ME_EVAL_CHECK(expr1, var_ptrs, 1, result1, 5);
    ME_EVAL_CHECK(expr2, var_ptrs, 1, result2, 5);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(result1[i], result2[i], i);
    }

    me_free(expr1);
    me_free(expr2);
    printf("  PASS\n");
}

void test_real_imag_on_real_inputs() {
#if defined(_WIN32) || defined(_WIN64)
    TEST("real/imag(x) on real inputs follow NumPy semantics");
    printf("  SKIP: real/imag tests are disabled on Windows\n");
    return;
#endif
    TEST("real/imag(x) on real inputs follow NumPy semantics");

    double x[VECTOR_SIZE] = {-3.0, -1.5, -0.0, 0.0, 0.5, 1.0, 2.5, 10.0, -10.0, 3.14159};
    double real_result[VECTOR_SIZE] = {0};
    double imag_result[VECTOR_SIZE] = {0};

    me_variable vars64[] = {{"x"}};
    int err;

    /* real(x) with float64 input -> float64 output, identity */
    me_expr *expr_real64 = NULL;
    int rc_expr_real64 = me_compile("real(x)", vars64, 1, ME_FLOAT64, &err, &expr_real64);
    if (rc_expr_real64 != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error for real(x) float64 at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs64[] = {x};
    ME_EVAL_CHECK(expr_real64, var_ptrs64, 1, real_result, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        ASSERT_NEAR(x[i], real_result[i], i);
    }
    me_free(expr_real64);

    /* imag(x) with float64 input -> float64 output, all zeros */
    me_expr *expr_imag64 = NULL;
    int rc_expr_imag64 = me_compile("imag(x)", vars64, 1, ME_FLOAT64, &err, &expr_imag64);
    if (rc_expr_imag64 != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error for imag(x) float64 at position %d\n", err);
        tests_failed++;
        return;
    }

    ME_EVAL_CHECK(expr_imag64, var_ptrs64, 1, imag_result, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        ASSERT_NEAR(0.0, imag_result[i], i);
    }
    me_free(expr_imag64);

    /* Now test float32 inputs */
    float xf[VECTOR_SIZE];
    float real_result_f[VECTOR_SIZE] = {0};
    float imag_result_f[VECTOR_SIZE] = {0};
    for (int i = 0; i < VECTOR_SIZE; i++) {
        xf[i] = (float)x[i];
    }

    me_variable vars32[] = {{"x", ME_FLOAT32}};

    me_expr *expr_real32 = NULL;
    int rc_expr_real32 = me_compile("real(x)", vars32, 1, ME_FLOAT32, &err, &expr_real32);
    if (rc_expr_real32 != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error for real(x) float32 at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs32[] = {xf};
    ME_EVAL_CHECK(expr_real32, var_ptrs32, 1, real_result_f, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        ASSERT_NEAR(xf[i], real_result_f[i], i);
    }
    me_free(expr_real32);

    me_expr *expr_imag32 = NULL;
    int rc_expr_imag32 = me_compile("imag(x)", vars32, 1, ME_FLOAT32, &err, &expr_imag32);
    if (rc_expr_imag32 != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error for imag(x) float32 at position %d\n", err);
        tests_failed++;
        return;
    }

    ME_EVAL_CHECK(expr_imag32, var_ptrs32, 1, imag_result_f, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        ASSERT_NEAR(0.0, imag_result_f[i], i);
    }
    me_free(expr_imag32);

    printf("  PASS\n");
}

void test_where_basic() {
    TEST("where(cond, x, y) - basic NumPy-like behavior");

    double x[VECTOR_SIZE]  = {0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 10.0, 20.0, -5.0, 7.0};
    double y[VECTOR_SIZE]  = {10.0, 9.0, 8.0, 7.0, -10.0, -9.0, 0.0, -1.0, 5.0, -7.0};
    double c[VECTOR_SIZE]  = {0.0, 1.0, 0.0, 1.0,  0.0,  1.0,  1.0,  0.0, 0.0, 1.0};
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"c"}, {"x"}, {"y"}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("where(c, x, y)", vars, 3, ME_FLOAT64, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {c, x, y};
    ME_EVAL_CHECK(expr, var_ptrs, 3, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = (c[i] != 0.0) ? x[i] : y[i];
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

int main() {
    printf("=== Testing NumPy-Compatible Functions ===\n\n");

    test_expm1();
    test_log1p();
    test_log2();
    test_logaddexp();
    test_expm1_small_values();
    test_log1p_small_values();
    test_logaddexp_extreme_values();
    test_mixed_expressions();
    test_round_func();
    test_sign();
    test_square();
    test_trunc_func();
    test_square_vs_pow();
    test_where_basic();
    test_real_imag_on_real_inputs();
    test_real_imag_on_real_inputs();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
