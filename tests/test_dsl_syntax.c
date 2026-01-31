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

int main(void) {
    int fail = 0;
    fail |= test_assign_and_result_stmt();
    fail |= test_loop_break_continue();
    fail |= test_nd_indices();
    return fail;
}
