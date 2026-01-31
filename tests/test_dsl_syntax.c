/*
 * DSL syntax tests for miniexpr.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../src/miniexpr.h"
#include "minctest.h"

static int check_all_close(const double *out, const double *expected, int n, double tol) {
    for (int i = 0; i < n; i++) {
        double diff = out[i] - expected[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) {
            printf("  ❌ FAILED: idx %d got %.12f expected %.12f\n", i, out[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

static int test_assign_and_result_stmt(void) {
    printf("\n=== DSL Test 1: assignments + final expr ===\n");

    double a[8];
    double b[8];
    double out[8];
    double expected[8];

    for (int i = 0; i < 8; i++) {
        a[i] = (double)i;
        b[i] = (double)(i + 1);
        expected[i] = (a[i] + b[i]) * 2.0;
    }

    const char *src =
        "temp = a + b;\n"
        "temp * 2";

    me_variable vars[] = {{"a", ME_FLOAT64}, {"b", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    const void *inputs[] = {a, b};
    if (me_eval(expr, inputs, 2, out, 8, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    int rc = check_all_close(out, expected, 8, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_loop_break_continue(void) {
    printf("\n=== DSL Test 2: for loop + break/continue ===\n");

    double out[4];
    double expected_break[4];
    double expected_continue[4];

    for (int i = 0; i < 4; i++) {
        expected_break[i] = 3.0;   // 0 + 1 + 2 then break
        expected_continue[i] = 5.0; // 0 + 2 + 3
    }

    const char *src_break =
        "sum = 0;\n"
        "for i in range(5) {\n"
        "  sum = sum + i;\n"
        "  break if i == 2;\n"
        "}\n"
        "result = sum;\n";

    const char *src_continue =
        "sum = 0;\n"
        "for i in range(4) {\n"
        "  continue if i == 1;\n"
        "  sum = sum + i;\n"
        "}\n"
        "result = sum;\n";

    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(src_break, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: break compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: break eval error\n");
        me_free(expr);
        return 1;
    }
    int rc = check_all_close(out, expected_break, 4, 1e-12);
    me_free(expr);
    if (rc != 0) {
        return rc;
    }

    expr = NULL;
    if (me_compile(src_continue, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: continue compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: continue eval error\n");
        me_free(expr);
        return 1;
    }
    rc = check_all_close(out, expected_continue, 4, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_nd_indices(void) {
    printf("\n=== DSL Test 3: ND indices ===\n");

    const char *src = "result = _i0 + _i1";
    int64_t shape[2] = {2, 3};
    int32_t chunks[2] = {2, 3};
    int32_t blocks[2] = {2, 3};

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile_nd(src, NULL, 0, ME_FLOAT64, 2, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double out[6];
    if (me_eval_nd(expr, NULL, 0, out, 6, 0, 0, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    double expected[6] = {0, 1, 2, 1, 2, 3};
    int rc = check_all_close(out, expected, 6, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_nd_padding(void) {
    printf("\n=== DSL Test 4: ND padding in blocks ===\n");

    const char *src = "result = _i0 + _i1";
    int64_t shape[2] = {3, 5};
    int32_t chunks[2] = {2, 4};
    int32_t blocks[2] = {2, 3};

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile_nd(src, NULL, 0, ME_FLOAT64, 2, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double out[6];
    if (me_eval_nd(expr, NULL, 0, out, 6, 1, 0, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    double expected[6] = {4, 0, 0, 5, 0, 0};
    int rc = check_all_close(out, expected, 6, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_nd_large_block(void) {
    printf("\n=== DSL Test 5: ND larger block ===\n");

    const char *src = "result = _i0 + _i1";
    int64_t shape[2] = {6, 7};
    int32_t chunks[2] = {4, 4};
    int32_t blocks[2] = {2, 2};

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile_nd(src, NULL, 0, ME_FLOAT64, 2, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double out[4];
    if (me_eval_nd(expr, NULL, 0, out, 4, 2, 1, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    double expected[4] = {6, 7, 7, 8};
    int rc = check_all_close(out, expected, 4, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_nd_3d_indices_padding(void) {
    printf("\n=== DSL Test 6: 3D indices + padding + _n* + _ndim ===\n");

    const char *src = "result = _i0 + _i1 + _i2 + _n0 + _n1 + _n2 + _ndim";
    int64_t shape[3] = {3, 4, 5};
    int32_t chunks[3] = {2, 3, 4};
    int32_t blocks[3] = {2, 2, 3};

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile_nd(src, NULL, 0, ME_FLOAT64, 3, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double out[12];
    if (me_eval_nd(expr, NULL, 0, out, 12, 0, 3, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    double expected[12] = {20, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0};
    int rc = check_all_close(out, expected, 12, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_nested_loops_and_conditionals(void) {
    printf("\n=== DSL Test 7: nested loops + mixed-type conditions ===\n");

    const char *src =
        "sum = 0;\n"
        "for i in range(3) {\n"
        "  for j in range(4) {\n"
        "    continue if ((i + 0.5) > 1.0) & (j < 2);\n"
        "    sum = sum + i + j;\n"
        "  }\n"
        "}\n"
        "result = sum;\n";

    double out[5];
    double expected[5];
    for (int i = 0; i < 5; i++) {
        expected[i] = 22.0;
    }

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 5, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }
    int rc = check_all_close(out, expected, 5, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_break_any_condition(void) {
    printf("\n=== DSL Test 8: break with array condition (any) ===\n");

    const char *src =
        "sum = 0;\n"
        "for i in range(5) {\n"
        "  sum = sum + i;\n"
        "  break if x > 0;\n"
        "}\n"
        "result = sum;\n";

    double x[4] = {-1.0, 2.0, -3.0, 0.0};
    double out[4];
    double expected[4];
    for (int i = 0; i < 4; i++) {
        expected[i] = 0.0;
    }

    me_variable vars[] = {{"x", ME_FLOAT64}};
    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    const void *inputs[] = {x};
    if (me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    int rc = check_all_close(out, expected, 4, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

int main(void) {
    int fail = 0;
    fail |= test_assign_and_result_stmt();
    fail |= test_loop_break_continue();
    fail |= test_nd_indices();
    fail |= test_nd_padding();
    fail |= test_nd_large_block();
    fail |= test_nd_3d_indices_padding();
    fail |= test_nested_loops_and_conditionals();
    fail |= test_break_any_condition();
    return fail;
}
