/*
 * DSL user-defined function tests for miniexpr.
 */

#include <stdio.h>
#include <stdlib.h>
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

static double clamp01(double x) {
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

static double scale(void *ctx, double x) {
    return (*(double *)ctx) * x;
}

static int test_udf_function(void) {
    printf("\n=== DSL UDF Test 1: function ===\n");

    double x[] = {-0.5, 0.0, 0.25, 1.0, 1.5};
    double out[5];
    double expected[] = {0.0, 0.0, 0.25, 1.0, 1.0};

    const char *src = "result = clamp01(x)";
    me_variable_ex vars[] = {
        {"x", ME_FLOAT64, NULL, 0, NULL, 0},
        {"clamp01", ME_FLOAT64, clamp01, ME_FUNCTION1 | ME_FLAG_PURE, NULL, 0},
    };

    int err = 0;
    me_expr *expr = NULL;
    if (me_compile_ex(src, vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    const void *inputs[] = {x};
    if (me_eval(expr, inputs, 1, out, 5, NULL) != ME_EVAL_SUCCESS) {
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

static int test_udf_closure(void) {
    printf("\n=== DSL UDF Test 2: closure ===\n");

    double x[] = {1.0, 2.0, 3.0, 4.0};
    double out[4];
    double expected[] = {2.5, 5.0, 7.5, 10.0};
    double factor = 2.5;

    const char *src = "result = scale(x)";
    me_variable_ex vars[] = {
        {"x", ME_FLOAT64, NULL, 0, NULL, 0},
        {"scale", ME_FLOAT64, scale, ME_CLOSURE1 | ME_FLAG_PURE, &factor, 0},
    };

    int err = 0;
    me_expr *expr = NULL;
    if (me_compile_ex(src, vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
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

static int test_udf_invalid(void) {
    printf("\n=== DSL UDF Test 3: invalid registrations ===\n");

    int err = 0;
    me_expr *expr = NULL;

    me_variable_ex vars_builtin[] = {
        {"x", ME_FLOAT64, NULL, 0, NULL, 0},
        {"sum", ME_FLOAT64, clamp01, ME_FUNCTION1 | ME_FLAG_PURE, NULL, 0},
    };

    if (me_compile_ex("result = sum(x)", vars_builtin, 2, ME_FLOAT64, &err, &expr)
        == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: builtin name accepted\n");
        me_free(expr);
        return 1;
    }

    expr = NULL;
    me_variable_ex vars_reserved[] = {
        {"x", ME_FLOAT64, NULL, 0, NULL, 0},
        {"result", ME_FLOAT64, clamp01, ME_FUNCTION1 | ME_FLAG_PURE, NULL, 0},
    };

    if (me_compile_ex("result = result(x)", vars_reserved, 2, ME_FLOAT64, &err, &expr)
        == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: reserved name accepted\n");
        me_free(expr);
        return 1;
    }

    expr = NULL;
    me_variable_ex vars_auto[] = {
        {"x", ME_FLOAT64, NULL, 0, NULL, 0},
        {"clamp01", ME_AUTO, clamp01, ME_FUNCTION1 | ME_FLAG_PURE, NULL, 0},
    };

    if (me_compile_ex("result = clamp01(x)", vars_auto, 2, ME_FLOAT64, &err, &expr)
        == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: ME_AUTO return dtype accepted\n");
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

int main(void) {
    int fail = 0;
    fail |= test_udf_function();
    fail |= test_udf_closure();
    fail |= test_udf_invalid();
    return fail;
}
