/* Tests for ME_STRING comparisons and string predicates */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "../src/miniexpr.h"
#include "minctest.h"



#define NAMES_COUNT 4

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

static void assert_bool_array(const bool *actual, const bool *expected, int n, const char *label) {
    for (int i = 0; i < n; i++) {
        if (actual[i] != expected[i]) {
            printf("  FAIL %s at [%d]: expected %d, got %d\n",
                   label, i, (int)expected[i], (int)actual[i]);
            tests_failed++;
            return;
        }
    }
    printf("  PASS %s\n", label);
}

static void test_string_compare_literal(void) {
    TEST("name == \"alpha\"");

    uint32_t names[NAMES_COUNT][8] = {
        {'a','l','p','h','a',0,0,0},
        {'b','e','t','a',0,0,0,0},
        {'a','l','p',0,0,0,0,0},
        {0,0,0,0,0,0,0,0}
    };

    me_variable_ex vars[] = {
        {"name", ME_STRING, names, ME_VARIABLE, NULL, sizeof(names[0])}
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile_ex("name == \"alpha\"", vars, 1, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    bool result[NAMES_COUNT] = {0};
    const void *var_ptrs[] = {names};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, NAMES_COUNT);

    const bool expected[NAMES_COUNT] = {true, false, false, false};
    assert_bool_array(result, expected, NAMES_COUNT, "name == \"alpha\"");

    me_free(expr);
}

static void test_string_predicates(void) {
    TEST("startswith/contains with or");

    uint32_t names[NAMES_COUNT][8] = {
        {'a','l','p','h','a',0,0,0},
        {'b','e','t','a',0,0,0,0},
        {'a','l','p',0,0,0,0,0},
        {0,0,0,0,0,0,0,0}
    };

    me_variable_ex vars[] = {
        {"name", ME_STRING, names, ME_VARIABLE, NULL, sizeof(names[0])}
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile_ex("startswith(name, \"alp\") or contains(name, \"et\")",
                           vars, 1, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    bool result[NAMES_COUNT] = {0};
    const void *var_ptrs[] = {names};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, NAMES_COUNT);

    const bool expected[NAMES_COUNT] = {true, true, true, false};
    assert_bool_array(result, expected, NAMES_COUNT, "startswith(...) or contains(...)");

    me_free(expr);
}

static void test_string_compare_itemsize(void) {
    TEST("string compare with different itemsize");

    uint32_t left[NAMES_COUNT][3] = {
        {'a',0,0},
        {'b',0,0},
        {'c',0,0},
        {0,0,0}
    };
    uint32_t right[NAMES_COUNT][5] = {
        {'a',0,0,0,0},
        {'x',0,0,0,0},
        {'c',0,0,0,0},
        {0,0,0,0,0}
    };

    me_variable_ex vars[] = {
        {"left", ME_STRING, left, ME_VARIABLE, NULL, sizeof(left[0])},
        {"right", ME_STRING, right, ME_VARIABLE, NULL, sizeof(right[0])}
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile_ex("left == right", vars, 2, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    bool result[NAMES_COUNT] = {0};
    const void *var_ptrs[] = {left, right};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, NAMES_COUNT);

    const bool expected[NAMES_COUNT] = {true, false, true, true};
    assert_bool_array(result, expected, NAMES_COUNT, "left == right");

    me_free(expr);
}

static void test_invalid_string_usage(void) {
    TEST("invalid string usage");

    uint32_t names[NAMES_COUNT][4] = {
        {'a',0,0,0},
        {'b',0,0,0},
        {'c',0,0,0},
        {0,0,0,0}
    };
    double values[NAMES_COUNT] = {1.0, 2.0, 3.0, 4.0};

    me_variable_ex vars_bad_size[] = {
        {"name", ME_STRING, names, ME_VARIABLE, NULL, 0}
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile_ex("name == \"a\"", vars_bad_size, 1, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_ERR_INVALID_ARG_TYPE) {
        printf("  FAIL: expected invalid arg type for itemsize=0, got %d\n", rc);
        tests_failed++;
    }

    me_variable_ex vars_mixed[] = {
        {"name", ME_STRING, names, ME_VARIABLE, NULL, sizeof(names[0])},
        {"x", ME_FLOAT64, values, ME_VARIABLE, NULL, 0}
    };

    rc = me_compile_ex("name == x", vars_mixed, 2, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_ERR_INVALID_ARG_TYPE) {
        printf("  FAIL: expected invalid arg type for string/numeric compare, got %d\n", rc);
        tests_failed++;
    }
    printf("  PASS invalid string usage\n");
}

int main(void) {
    printf("=== Testing ME_STRING operations ===\n\n");

    test_string_compare_literal();
    test_string_predicates();
    test_string_compare_itemsize();
    test_invalid_string_usage();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
