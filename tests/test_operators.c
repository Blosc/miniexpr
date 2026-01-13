/* Test new bitwise, logical, and comparison operators */
#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
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
        printf("  FAIL at [%d]: expected %.2f, got %.2f\n", idx, (double)(expected), (double)(actual)); \
        tests_failed++; \
        return; \
    }

#define ASSERT_EQ_INT(expected, actual, idx) \
    if ((expected) != (actual)) { \
        printf("  FAIL at [%d]: expected %lld, got %lld\n", idx, (long long)(expected), (long long)(actual)); \
        tests_failed++; \
        return; \
    }

void test_bitwise_and_int() {
    TEST("bitwise AND on integers");

    int32_t a[VECTOR_SIZE] = {15, 7, 255, 128, 0, 12, 3, 6, 9, 31};
    int32_t b[VECTOR_SIZE] = {7, 15, 15, 64, 0, 10, 1, 2, 3, 16};
    int32_t result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a & b", vars, 2, ME_INT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        int32_t expected = a[i] & b[i];
        ASSERT_EQ_INT(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_bitwise_or_int() {
    TEST("bitwise OR on integers");

    int32_t a[VECTOR_SIZE] = {8, 4, 1, 0, 7, 12, 3, 6, 9, 16};
    int32_t b[VECTOR_SIZE] = {4, 8, 2, 0, 8, 10, 1, 2, 3, 32};
    int32_t result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a | b", vars, 2, ME_INT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        int32_t expected = a[i] | b[i];
        ASSERT_EQ_INT(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_bitwise_xor_int() {
    TEST("bitwise XOR on integers");

    int32_t a[VECTOR_SIZE] = {15, 7, 255, 128, 0, 12, 3, 6, 9, 31};
    int32_t b[VECTOR_SIZE] = {7, 15, 15, 64, 0, 10, 1, 2, 3, 16};
    int32_t result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a ^ b", vars, 2, ME_INT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        int32_t expected = a[i] ^ b[i];
        ASSERT_EQ_INT(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_bit_shift_left() {
    TEST("left shift on integers");

    int32_t a[VECTOR_SIZE] = {1, 2, 3, 4, 5, 8, 16, 32, 64, 128};
    int32_t b[VECTOR_SIZE] = {1, 2, 3, 1, 2, 1, 1, 1, 1, 1};
    int32_t result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a << b", vars, 2, ME_INT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        int32_t expected = a[i] << b[i];
        ASSERT_EQ_INT(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_bit_shift_right() {
    TEST("right shift on integers");

    int32_t a[VECTOR_SIZE] = {128, 64, 32, 16, 8, 4, 2, 1, 255, 1024};
    int32_t b[VECTOR_SIZE] = {1, 2, 3, 1, 2, 1, 1, 1, 4, 3};
    int32_t result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a >> b", vars, 2, ME_INT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        int32_t expected = a[i] >> b[i];
        ASSERT_EQ_INT(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_comparison_eq_float() {
    TEST("equality comparison on floats");

    float a[VECTOR_SIZE] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    float b[VECTOR_SIZE] = {1.0f, 3.0f, 3.0f, 5.0f, 5.0f, 1.5f, 2.0f, 3.5f, 4.0f, 5.5f};
    float result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a == b", vars, 2, ME_FLOAT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        float expected = (a[i] == b[i]) ? 1.0f : 0.0f;
        ASSERT_EQ(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_comparison_lt_int() {
    TEST("less-than comparison on integers");

    int32_t a[VECTOR_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int32_t b[VECTOR_SIZE] = {2, 2, 4, 3, 5, 7, 6, 9, 8, 10};
    int32_t result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a < b", vars, 2, ME_INT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        int32_t expected = (a[i] < b[i]) ? 1 : 0;
        ASSERT_EQ_INT(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_logical_bool() {
    TEST("logical operations on booleans");

    bool a[VECTOR_SIZE] = {true, false, true, false, true, false, true, false, true, false};
    bool b[VECTOR_SIZE] = {true, true, false, false, true, true, false, false, true, true};
    bool result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"a"}, {"b"}};

    // Test AND
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a & b", vars, 2, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: AND compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        bool expected = a[i] && b[i];
        if (result[i] != expected) {
            printf("  FAIL AND at [%d]: expected %d, got %d\n", i, expected, result[i]);
            tests_failed++;
            me_free(expr);
            return;
        }
    }

    me_free(expr);

    // Test OR
    expr = NULL;
    rc_expr = me_compile("a | b", vars, 2, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: OR compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs_or[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs_or, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        bool expected = a[i] || b[i];
        if (result[i] != expected) {
            printf("  FAIL OR at [%d]: expected %d, got %d\n", i, expected, result[i]);
            tests_failed++;
            me_free(expr);
            return;
        }
    }

    me_free(expr);

    // Test XOR
    expr = NULL;
    rc_expr = me_compile("a ^ b", vars, 2, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: XOR compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs_xor[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs_xor, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        bool expected = a[i] != b[i];
        if (result[i] != expected) {
            printf("  FAIL XOR at [%d]: expected %d, got %d\n", i, expected, result[i]);
            tests_failed++;
            me_free(expr);
            return;
        }
    }

    me_free(expr);

    // Test NOT
    me_variable vars_not[] = {{"a"}};
    expr = NULL;
    rc_expr = me_compile("~a", vars_not, 1, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: NOT compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs_not[] = {a};
    ME_EVAL_CHECK(expr, var_ptrs_not, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        bool expected = !a[i];
        if (result[i] != expected) {
            printf("  FAIL NOT at [%d]: expected %d, got %d\n", i, expected, result[i]);
            tests_failed++;
            me_free(expr);
            return;
        }
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_logical_comparisons() {
    TEST("logical ops on comparison results");

    float o0[VECTOR_SIZE] = {0.2f, 0.6f, 1.2f, 0.4f, 0.9f, 0.1f, 0.8f, 0.0f, 0.51f, 0.49f};
    int32_t o1[VECTOR_SIZE] = {9999, 10001, 10000, 15000, 5000, 20000, 10002, 42, 10001, 10000};
    bool result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"o0", ME_FLOAT32}, {"o1", ME_INT32}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("((o0 > 0.5) & (o1 > 10000))", vars, 2, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {o0, o1};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        bool expected = (o0[i] > 0.5f) && (o1[i] > 10000);
        if (result[i] != expected) {
            printf("  FAIL at [%d]: expected %d, got %d\n", i, expected, result[i]);
            tests_failed++;
            me_free(expr);
            return;
        }
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_unary_bool_funcs() {
    TEST("unary funcs on bool");

    bool a[VECTOR_SIZE] = {true, false, true, true, false, false, true, false, true, false};
    bool result[VECTOR_SIZE] = {0};
    const char *exprs[] = {"abs(a)", "ceil(a)", "floor(a)", "trunc(a)", "square(a)"};
    const int expr_count = (int)(sizeof(exprs) / sizeof(exprs[0]));

    me_variable vars[] = {{"a", ME_BOOL}};
    const void *var_ptrs[] = {a};

    for (int e = 0; e < expr_count; e++) {
        int err;
        me_expr *expr = NULL;
        int rc_expr = me_compile(exprs[e], vars, 1, ME_BOOL, &err, &expr);

        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL: %s compilation error at position %d\n", exprs[e], err);
            tests_failed++;
            return;
        }

        ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (result[i] != a[i]) {
                printf("  FAIL %s at [%d]: expected %d, got %d\n", exprs[e], i, a[i], result[i]);
                tests_failed++;
                me_free(expr);
                return;
            }
        }

        me_free(expr);
    }

    printf("  PASS\n");
}

int main() {
    printf("=== Testing New Operators ===\n\n");

    test_bitwise_and_int();
    test_bitwise_or_int();
    test_bitwise_xor_int();
    test_bit_shift_left();
    test_bit_shift_right();
    test_comparison_eq_float();
    test_comparison_lt_int();
    test_logical_bool();
    test_logical_comparisons();
    test_unary_bool_funcs();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
