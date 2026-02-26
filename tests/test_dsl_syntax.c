/*
 * DSL syntax tests for miniexpr.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
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

static char *dup_env_value(const char *name) {
    const char *value = getenv(name);
    if (!value) {
        return NULL;
    }
    size_t n = strlen(value) + 1;
    char *copy = malloc(n);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, value, n);
    return copy;
}

static void restore_env_value(const char *name, const char *value) {
    if (!name) {
        return;
    }
    if (value) {
        (void)setenv(name, value, 1);
    }
    else {
        (void)unsetenv(name);
    }
}

static int compile_eval_double(const char *src, me_variable *vars, int nvars,
                               const void **inputs, int nitems, double *out) {
    if (!src || !out || nitems <= 0) {
        return 1;
    }
    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, vars, nvars, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }
    int rc = me_eval(expr, inputs, nvars, out, nitems, NULL);
    me_free(expr);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error (%d)\n", rc);
        return 1;
    }
    return 0;
}

static int compile_eval_int64(const char *src, me_variable *vars, int nvars,
                              const void **inputs, int nitems, int64_t *out) {
    if (!src || !out || nitems <= 0) {
        return 1;
    }
    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, vars, nvars, ME_INT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }
    int rc = me_eval(expr, inputs, nvars, out, nitems, NULL);
    me_free(expr);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error (%d)\n", rc);
        return 1;
    }
    return 0;
}

static int test_assign_and_result_stmt(void) {
    printf("\n=== DSL Test 1: assignments + return ===\n");

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
        "def kernel(a, b):\n"
        "    temp = a + b\n"
        "    return temp * 2\n";

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
        "def kernel():\n"
        "    sum = 0\n"
        "    for i in range(5):\n"
        "        sum = sum + i\n"
        "        if i == 2:\n"
        "            break\n"
        "    return sum\n";

    const char *src_continue =
        "def kernel():\n"
        "    sum = 0\n"
        "    for i in range(4):\n"
        "        if i == 1:\n"
        "            continue\n"
        "        sum = sum + i\n"
        "    return sum\n";

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

static int test_range_start_stop_step(void) {
    printf("\n=== DSL Test 2b: range() start/stop/step forms ===\n");

    const char *src_start_stop =
        "def kernel():\n"
        "    acc = 0\n"
        "    for i in range(2, 6):\n"
        "        acc = acc + i\n"
        "    return acc\n";
    const char *src_step =
        "def kernel():\n"
        "    acc = 0\n"
        "    for i in range(1, 8, 3):\n"
        "        acc = acc + i\n"
        "    return acc\n";
    const char *src_neg_step =
        "def kernel():\n"
        "    acc = 0\n"
        "    for i in range(5, -2, -2):\n"
        "        acc = acc + i\n"
        "    return acc\n";
    const char *src_zero_step =
        "def kernel():\n"
        "    for i in range(0, 5, 0):\n"
        "        return i\n"
        "    return 0\n";
    const char *src_parity =
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(1, 8, 3):\n"
        "        if x > i:\n"
        "            acc = acc + i\n"
        "    return acc\n";

    int err = 0;
    me_expr *expr = NULL;
    double out[4] = {0.0, 0.0, 0.0, 0.0};

    if (me_compile(src_start_stop, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: range(start, stop) compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: range(start, stop) eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    double expected_start_stop[4] = {14.0, 14.0, 14.0, 14.0};
    if (check_all_close(out, expected_start_stop, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: range(start, stop) output mismatch\n");
        return 1;
    }

    expr = NULL;
    if (me_compile(src_step, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: range(start, stop, step) compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: range(start, stop, step) eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    double expected_step[4] = {12.0, 12.0, 12.0, 12.0};
    if (check_all_close(out, expected_step, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: range(start, stop, step) output mismatch\n");
        return 1;
    }

    expr = NULL;
    if (me_compile(src_neg_step, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: range() negative-step compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: range() negative-step eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    double expected_neg_step[4] = {8.0, 8.0, 8.0, 8.0};
    if (check_all_close(out, expected_neg_step, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: range() negative-step output mismatch\n");
        return 1;
    }

    expr = NULL;
    if (me_compile(src_zero_step, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: zero-step range compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_ERR_INVALID_ARG) {
        printf("  ❌ FAILED: zero-step range should fail at runtime\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {0.0, 2.0, 5.0, 10.0};
    const void *inputs[] = {x};
    double out_interp[4] = {0.0, 0.0, 0.0, 0.0};
    double out_jit_enabled[4] = {0.0, 0.0, 0.0, 0.0};
    char *saved_jit = dup_env_value("ME_DSL_JIT");

    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src_parity, vars, 1, inputs, 4, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src_parity, vars, 1, inputs, 4, out_jit_enabled) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    double expected_parity[4] = {0.0, 1.0, 5.0, 12.0};
    if (check_all_close(out_interp, expected_parity, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: range parity unexpected interpreter output\n");
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        if (out_interp[i] != out_jit_enabled[i]) {
            printf("  ❌ FAILED: range parity mismatch at %d (interp=%.17g jit=%.17g)\n",
                   i, out_interp[i], out_jit_enabled[i]);
            return 1;
        }
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_while_loop_semantics(void) {
    printf("\n=== DSL Test 2c: while loop semantics ===\n");

    const char *src_scalar =
        "def kernel():\n"
        "    i = 0\n"
        "    acc = 0\n"
        "    while i < 5:\n"
        "        acc = acc + i\n"
        "        i = i + 1\n"
        "    return acc\n";
    const char *src_elementwise =
        "def kernel(x):\n"
        "    i = 0\n"
        "    acc = 0\n"
        "    while i < 5:\n"
        "        if x > i:\n"
        "            acc = acc + 1\n"
        "            i = i + 1\n"
        "            continue\n"
        "        break\n"
        "    return acc\n";

    int err = 0;
    me_expr *expr = NULL;
    double out[4] = {0.0, 0.0, 0.0, 0.0};

    if (me_compile(src_scalar, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: while scalar compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: while scalar eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    double expected_scalar[4] = {10.0, 10.0, 10.0, 10.0};
    if (check_all_close(out, expected_scalar, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: while scalar output mismatch\n");
        return 1;
    }

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {-1.0, 2.0, 5.0, 0.0};
    const void *inputs[] = {x};
    double out_interp[4] = {0.0, 0.0, 0.0, 0.0};
    double out_jit_enabled[4] = {0.0, 0.0, 0.0, 0.0};
    char *saved_jit = dup_env_value("ME_DSL_JIT");

    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src_elementwise, vars, 1, inputs, 4, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src_elementwise, vars, 1, inputs, 4, out_jit_enabled) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    double expected_elementwise[4] = {0.0, 2.0, 5.0, 0.0};
    if (check_all_close(out_interp, expected_elementwise, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: while element-wise output mismatch\n");
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        if (out_interp[i] != out_jit_enabled[i]) {
            printf("  ❌ FAILED: while parity mismatch at %d (interp=%.17g jit=%.17g)\n",
                   i, out_interp[i], out_jit_enabled[i]);
            return 1;
        }
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_while_iteration_cap(void) {
    printf("\n=== DSL Test 2d: while iteration cap ===\n");

    const char *src =
        "def kernel():\n"
        "    i = 0\n"
        "    while 1:\n"
        "        i = i + 1\n"
        "    return i\n";

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    char *saved_cap = dup_env_value("ME_DSL_WHILE_MAX_ITERS");
    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        free(saved_cap);
        return 1;
    }
    if (setenv("ME_DSL_WHILE_MAX_ITERS", "32", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_WHILE_MAX_ITERS failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        free(saved_cap);
        return 1;
    }

    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: while-cap compile error at %d\n", err);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_WHILE_MAX_ITERS", saved_cap);
        free(saved_jit);
        free(saved_cap);
        return 1;
    }
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    int rc_eval = me_eval(expr, NULL, 0, out, 4, NULL);
    me_free(expr);

    restore_env_value("ME_DSL_JIT", saved_jit);
    restore_env_value("ME_DSL_WHILE_MAX_ITERS", saved_cap);
    free(saved_jit);
    free(saved_cap);

    if (rc_eval != ME_EVAL_ERR_INVALID_ARG) {
        printf("  ❌ FAILED: expected while-cap runtime error, got %d\n", rc_eval);
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_invalid_conditionals(void) {
    printf("\n=== DSL Test 3: invalid conditionals ===\n");

    const char *src_deprecated =
        "def kernel():\n"
        "    for i in range(2):\n"
        "        break if i == 0\n"
        "    return 0\n";

    const char *src_non_scalar =
        "def kernel(x):\n"
        "    if x > 0:\n"
        "        return 1\n"
        "    else:\n"
        "        return 0\n";

    const char *src_return_mismatch =
        "def kernel(x):\n"
        "    if any(x > 0):\n"
        "        return x > 0\n"
        "    return x\n";

    const char *src_missing_return =
        "def kernel(x):\n"
        "    if any(x > 0):\n"
        "        return 1\n";

    const char *src_missing_def =
        "temp = x + 1\n"
        "return temp\n";

    const char *src_signature_mismatch =
        "def kernel(x, y):\n"
        "    return x + y\n";

    const char *src_signature_order =
        "def kernel(y, x):\n"
        "    return x + 2 * y\n";

    const char *src_new_local =
        "def kernel(x):\n"
        "    if x > 0:\n"
        "        y = 2\n"
        "        return y\n"
        "    else:\n"
        "        z = 3\n"
        "        return z\n";

    const char *src_element_return_in_loop =
        "def kernel(x):\n"
        "    for i in range(4):\n"
        "        if x == i:\n"
        "            return i + 10\n"
        "    return -1\n";

    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(src_deprecated, NULL, 0, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: deprecated break if syntax accepted\n");
        me_free(expr);
        return 1;
    }

    expr = NULL;
    me_variable vars[] = {{"x", ME_FLOAT64}};
    me_variable vars_order[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};
    if (me_compile(src_non_scalar, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: non-reduction conditional was rejected\n");
        return 1;
    }
    me_free(expr);

    expr = NULL;
    if (me_compile(src_return_mismatch, vars, 1, ME_AUTO, &err, &expr) == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: mismatched return dtypes accepted\n");
        me_free(expr);
        return 1;
    }

    expr = NULL;
    if (me_compile(src_missing_return, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: missing return path rejected at compile time\n");
        return 1;
    }
    double x_missing[4] = {-1.0, -2.0, -3.0, -4.0};
    double out_missing[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs_missing[] = {x_missing};
    int rc_missing = me_eval(expr, inputs_missing, 1, out_missing, 4, NULL);
    me_free(expr);
    if (rc_missing != ME_EVAL_ERR_INVALID_ARG) {
        printf("  ❌ FAILED: missing return path should be runtime eval error (got %d)\n", rc_missing);
        return 1;
    }

    expr = NULL;
    if (me_compile(src_missing_def, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: missing def accepted\n");
        me_free(expr);
        return 1;
    }

    expr = NULL;
    if (me_compile(src_signature_mismatch, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: signature mismatch accepted\n");
        me_free(expr);
        return 1;
    }

    expr = NULL;
    if (me_compile(src_signature_order, vars_order, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: signature order rejected\n");
        return 1;
    }
    double x_vals[4] = {1.0, 2.0, 3.0, 4.0};
    double y_vals[4] = {10.0, 20.0, 30.0, 40.0};
    double out_order[4];
    const void *inputs_order[] = {x_vals, y_vals};
    if (me_eval(expr, inputs_order, 2, out_order, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: signature order eval error\n");
        me_free(expr);
        return 1;
    }
    double expected_order[4] = {21.0, 42.0, 63.0, 84.0};
    if (check_all_close(out_order, expected_order, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: signature order eval mismatch\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    expr = NULL;
    if (me_compile(src_new_local, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: new local inside branch should be accepted\n");
        return 1;
    }
    double x_new_local[4] = {-1.0, 2.0, 0.0, -3.0};
    double out_new_local[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs_new_local[] = {x_new_local};
    if (me_eval(expr, inputs_new_local, 1, out_new_local, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: new-local-branch eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    double expected_new_local[4] = {3.0, 2.0, 3.0, 3.0};
    if (check_all_close(out_new_local, expected_new_local, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: new-local-branch output mismatch\n");
        return 1;
    }

    expr = NULL;
    if (me_compile(src_element_return_in_loop, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: return inside loop should be accepted\n");
        return 1;
    }
    double x_return_loop[4] = {0.0, 1.0, 3.0, 5.0};
    double out_return_loop[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs_return_loop[] = {x_return_loop};
    if (me_eval(expr, inputs_return_loop, 1, out_return_loop, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: return-inside-loop eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    double expected_return_loop[4] = {10.0, 11.0, 13.0, -1.0};
    if (check_all_close(out_return_loop, expected_return_loop, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: return-inside-loop output mismatch\n");
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_string_truthiness(void) {
    printf("\n=== DSL Test 21: string truthiness in conditions ===\n");

    const uint32_t names[4][8] = {
        {'a', 'l', 'p', 'h', 'a', 0, 0, 0},
        {'b', 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {'z', 0, 0, 0, 0, 0, 0, 0}
    };
    const char *src =
        "def kernel(name):\n"
        "    if name:\n"
        "        return 1\n"
        "    return 0\n";

    me_variable vars[] = {
        {"name", ME_STRING, names, ME_VARIABLE, NULL, sizeof(names[0])}
    };
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs[] = {names};
    int rc_eval = me_eval(expr, inputs, 1, out, 4, NULL);
    me_free(expr);
    if (rc_eval != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error (%d)\n", rc_eval);
        return 1;
    }

    double expected[4] = {1.0, 1.0, 0.0, 1.0};
    int rc = check_all_close(out, expected, 4, 1e-12);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_if_elif_else(void) {
    printf("\n=== DSL Test 3b: if/elif/else ===\n");

    const char *src =
        "def kernel(x):\n"
        "    if any(x > 0):\n"
        "        return 1\n"
        "    elif any(x < 0):\n"
        "        return 2\n"
        "    else:\n"
        "        return 3\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double out[4];
    me_expr *expr = NULL;
    int err = 0;

    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double x_case1[4] = {-1.0, 2.0, -3.0, 0.0};
    const void *vars_case1[] = {x_case1};
    if (me_eval(expr, vars_case1, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error (case 1)\n");
        me_free(expr);
        return 1;
    }
    double expected_case1[4] = {1.0, 1.0, 1.0, 1.0};
    if (check_all_close(out, expected_case1, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: unexpected output (case 1)\n");
        me_free(expr);
        return 1;
    }

    double x_case2[4] = {-1.0, -2.0, -3.0, -4.0};
    const void *vars_case2[] = {x_case2};
    if (me_eval(expr, vars_case2, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error (case 2)\n");
        me_free(expr);
        return 1;
    }
    double expected_case2[4] = {2.0, 2.0, 2.0, 2.0};
    if (check_all_close(out, expected_case2, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: unexpected output (case 2)\n");
        me_free(expr);
        return 1;
    }

    double x_case3[4] = {0.0, 0.0, 0.0, 0.0};
    const void *vars_case3[] = {x_case3};
    if (me_eval(expr, vars_case3, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error (case 3)\n");
        me_free(expr);
        return 1;
    }
    double expected_case3[4] = {3.0, 3.0, 3.0, 3.0};
    int rc = check_all_close(out, expected_case3, 4, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_nd_indices(void) {
    printf("\n=== DSL Test 4: ND indices ===\n");

    const char *src =
        "def kernel():\n"
        "    return _i0 + _i1\n";
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

static int test_flat_idx(void) {
    printf("\n=== DSL Test 4b: global linear index ===\n");

    const char *src_linear =
        "def kernel():\n"
        "    return _flat_idx\n";

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src_linear, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: linear compile error at %d\n", err);
        return 1;
    }

    double out_linear[4] = {0.0, 0.0, 0.0, 0.0};
    if (me_eval(expr, NULL, 0, out_linear, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: linear eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    double expected_linear[4] = {0.0, 1.0, 2.0, 3.0};
    if (check_all_close(out_linear, expected_linear, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: linear output mismatch\n");
        return 1;
    }

    const char *src_nd =
        "def kernel():\n"
        "    return _flat_idx\n";
    int64_t shape[2] = {3, 5};
    int32_t chunks[2] = {2, 4};
    int32_t blocks[2] = {2, 3};

    expr = NULL;
    err = 0;
    if (me_compile_nd(src_nd, NULL, 0, ME_FLOAT64, 2, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: ND compile error at %d\n", err);
        return 1;
    }

    double out_nd[6] = {0, 0, 0, 0, 0, 0};
    if (me_eval_nd(expr, NULL, 0, out_nd, 6, 1, 0, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: ND eval error\n");
        me_free(expr);
        return 1;
    }

    double expected_nd[6] = {4, 0, 0, 9, 0, 0};
    int rc = check_all_close(out_nd, expected_nd, 6, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_reserved_index_vars_jit_parity(void) {
    printf("\n=== DSL Test 4c: reserved index vars interpreter/JIT parity ===\n");

    const char *src =
        "def kernel():\n"
        "    return _flat_idx + _i0 + _n0 + _ndim\n";
    double out_interp[6] = {0, 0, 0, 0, 0, 0};
    double out_jit[6] = {0, 0, 0, 0, 0, 0};
    double expected[6] = {7, 9, 11, 13, 15, 17};

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, NULL, 0, NULL, 6, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, NULL, 0, NULL, 6, out_jit) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    for (int i = 0; i < 6; i++) {
        if (out_interp[i] != out_jit[i]) {
            printf("  ❌ FAILED: mismatch at %d (interp=%.17g jit=%.17g)\n",
                   i, out_interp[i], out_jit[i]);
            return 1;
        }
    }
    if (check_all_close(out_interp, expected, 6, 1e-12) != 0) {
        printf("  ❌ FAILED: unexpected output for reserved index vars parity\n");
        return 1;
    }

    const char *src_nd =
        "def kernel():\n"
        "    return _flat_idx + _i0 + _i1 + _n0 + _n1 + _ndim\n";
    int64_t shape[2] = {3, 5};
    int32_t chunks[2] = {2, 4};
    int32_t blocks[2] = {2, 3};
    double out_nd_interp[6] = {0, 0, 0, 0, 0, 0};
    double out_nd_jit[6] = {0, 0, 0, 0, 0, 0};
    double expected_nd[6] = {18, 0, 0, 24, 0, 0};

    saved_jit = dup_env_value("ME_DSL_JIT");
    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed (ND)\n");
        free(saved_jit);
        return 1;
    }
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile_nd(src_nd, NULL, 0, ME_FLOAT64, 2, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: ND interp compile error at %d\n", err);
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (me_eval_nd(expr, NULL, 0, out_nd_interp, 6, 1, 0, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: ND interp eval error\n");
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    me_free(expr);

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed (ND)\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    err = 0;
    expr = NULL;
    if (me_compile_nd(src_nd, NULL, 0, ME_FLOAT64, 2, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: ND jit compile error at %d\n", err);
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (me_eval_nd(expr, NULL, 0, out_nd_jit, 6, 1, 0, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: ND jit eval error\n");
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    me_free(expr);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    for (int i = 0; i < 6; i++) {
        if (out_nd_interp[i] != out_nd_jit[i]) {
            printf("  ❌ FAILED: ND mismatch at %d (interp=%.17g jit=%.17g)\n",
                   i, out_nd_interp[i], out_nd_jit[i]);
            return 1;
        }
    }
    if (check_all_close(out_nd_interp, expected_nd, 6, 1e-12) != 0) {
        printf("  ❌ FAILED: unexpected ND output for reserved index vars parity\n");
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_reserved_index_vars_env_gate(void) {
    printf("\n=== DSL Test 4d: reserved index vars env gate ===\n");

    const char *src_plain =
        "def kernel(x):\n"
        "    return x + 1\n";
    const char *src_reserved =
        "def kernel():\n"
        "    return _flat_idx + 1\n";

    double x[4] = {0.0, 1.0, 2.0, 3.0};
    const void *plain_inputs[] = {x};
    me_variable vars[] = {{"x", ME_FLOAT64}};

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    char *saved_index_vars = dup_env_value("ME_DSL_JIT_INDEX_VARS");
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }

    me_expr *expr_plain = NULL;
    int err = 0;
    if (me_compile(src_plain, vars, 1, ME_FLOAT64, &err, &expr_plain) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: plain compile error at %d\n", err);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    bool baseline_has_jit = me_expr_has_jit_kernel(expr_plain);
    double out_plain[4] = {0.0, 0.0, 0.0, 0.0};
    if (me_eval(expr_plain, plain_inputs, 1, out_plain, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: plain eval error\n");
        me_free(expr_plain);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    me_free(expr_plain);

    if (setenv("ME_DSL_JIT_INDEX_VARS", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT_INDEX_VARS=0 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }

    me_expr *expr_gate_off = NULL;
    err = 0;
    if (me_compile(src_reserved, NULL, 0, ME_FLOAT64, &err, &expr_gate_off) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: reserved compile (gate=0) error at %d\n", err);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    if (me_expr_has_jit_kernel(expr_gate_off)) {
        printf("  ❌ FAILED: reserved kernel unexpectedly JIT-compiled with gate=0\n");
        me_free(expr_gate_off);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    double out_gate_off[4] = {0.0, 0.0, 0.0, 0.0};
    if (me_eval(expr_gate_off, NULL, 0, out_gate_off, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: reserved eval (gate=0) error\n");
        me_free(expr_gate_off);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    me_free(expr_gate_off);

    if (setenv("ME_DSL_JIT_INDEX_VARS", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT_INDEX_VARS=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }

    me_expr *expr_gate_on = NULL;
    err = 0;
    if (me_compile(src_reserved, NULL, 0, ME_FLOAT64, &err, &expr_gate_on) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: reserved compile (gate=1) error at %d\n", err);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    if (baseline_has_jit && !me_expr_has_jit_kernel(expr_gate_on)) {
        printf("  ❌ FAILED: reserved kernel did not JIT-compile with gate=1\n");
        me_free(expr_gate_on);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    double out_gate_on[4] = {0.0, 0.0, 0.0, 0.0};
    if (me_eval(expr_gate_on, NULL, 0, out_gate_on, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: reserved eval (gate=1) error\n");
        me_free(expr_gate_on);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        free(saved_jit);
        free(saved_index_vars);
        return 1;
    }
    me_free(expr_gate_on);

    restore_env_value("ME_DSL_JIT", saved_jit);
    restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
    free(saved_jit);
    free(saved_index_vars);

    double expected_reserved[4] = {1.0, 2.0, 3.0, 4.0};
    if (check_all_close(out_gate_off, expected_reserved, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: gate=0 output mismatch\n");
        return 1;
    }
    if (check_all_close(out_gate_on, expected_reserved, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: gate=1 output mismatch\n");
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_nd_padding(void) {
    printf("\n=== DSL Test 5: ND padding in blocks ===\n");

    const char *src =
        "def kernel():\n"
        "    return _i0 + _i1\n";
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
    printf("\n=== DSL Test 6: ND larger block ===\n");

    const char *src =
        "def kernel():\n"
        "    return _i0 + _i1\n";
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
    printf("\n=== DSL Test 7: 3D indices + padding + _n* + _ndim ===\n");

    const char *src =
        "def kernel():\n"
        "    return _i0 + _i1 + _i2 + _n0 + _n1 + _n2 + _ndim\n";
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
    printf("\n=== DSL Test 8: nested loops + mixed-type conditions ===\n");

    const char *src =
        "def kernel():\n"
        "    sum = 0\n"
        "    for i in range(3):\n"
        "        for j in range(4):\n"
        "            if any(((i + 0.5) > 1.0) & (j < 2)):\n"
        "                continue\n"
        "            sum = sum + i + j\n"
        "    return sum\n";

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
    printf("\n=== DSL Test 9: break with array condition (any) ===\n");

    const char *src =
        "def kernel(x):\n"
        "    sum = 0\n"
        "    for i in range(5):\n"
        "        sum = sum + i\n"
        "        if any(x > 0):\n"
        "            break\n"
        "    return sum\n";

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

static int test_dsl_function_calls(void) {
    printf("\n=== DSL Test 10: assorted function calls ===\n");

    const char *src =
        "def kernel(a, b, d):\n"
        "    t0 = sin(a) + cos(a)\n"
        "    t1 = expm1(b) + log1p(abs(b))\n"
        "    t2 = sqrt(abs(a)) + hypot(a, b)\n"
        "    t3 = atan2(a, b) + pow(a, 2)\n"
        "    t4 = floor(d) + ceil(d) + trunc(d) + round(d)\n"
        "    return t0 + t1 + t2 + t3 + t4\n";

    double a[8] = {0.0, 0.5, 1.0, -1.5, 2.0, -2.5, 3.0, -3.5};
    float b[8] = {1.0f, -0.5f, 2.0f, -2.0f, 0.25f, -0.25f, 4.0f, -4.0f};
    double d[8] = {0.2, -0.7, 1.4, -1.6, 2.2, -2.8, 3.0, -3.2};

    double expected[8];
    for (int i = 0; i < 8; i++) {
        double av = a[i];
        double bv = (double)b[i];
        double dv = d[i];
        double t0 = sin(av) + cos(av);
        double t1 = expm1(bv) + log1p(fabs(bv));
        double t2 = sqrt(fabs(av)) + hypot(av, bv);
        double t3 = atan2(av, bv) + pow(av, 2.0);
        double t4 = floor(dv) + ceil(dv) + trunc(dv) + round(dv);
        expected[i] = t0 + t1 + t2 + t3 + t4;
    }

    me_variable vars[] = {{"a", ME_FLOAT64}, {"b", ME_FLOAT32}, {"d", ME_FLOAT64}};
    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, vars, 3, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    const void *inputs[] = {a, b, d};
    double out[8];
    if (me_eval(expr, inputs, 3, out, 8, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    int rc = check_all_close(out, expected, 8, 1e-6);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_black_scholes_dsl_kernel_support(void) {
    printf("\n=== DSL Test 10b: Black-Scholes kernel support ===\n");

    const char *src =
        "def kernel(S, X, T, R, V):\n"
        "    A1 = 0.31938153\n"
        "    A2 = -0.356563782\n"
        "    A3 = 1.781477937\n"
        "    A4 = -1.821255978\n"
        "    A5 = 1.330274429\n"
        "    RSQRT2PI = 0.39894228040143267793994605993438\n"
        "\n"
        "    sqrtT = sqrt(T)\n"
        "    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)\n"
        "    d2 = d1 - V * sqrtT\n"
        "    K = 1.0 / (1.0 + 0.2316419 * abs(d1))\n"
        "\n"
        "    ret_val = (RSQRT2PI * exp(-0.5 * d1 * d1) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
        "    cndd1 = ret_val\n"
        "    if d1 > 0:\n"
        "        cndd1 = 1.0 - ret_val\n"
        "    else:\n"
        "        cndd1 = ret_val\n"
        "\n"
        "    K = 1.0 / (1.0 + 0.2316419 * abs(d2))\n"
        "    ret_val = (RSQRT2PI * exp(-0.5 * d2 * d2) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
        "    if d2 > 0:\n"
        "        cndd2 = 1.0 - ret_val\n"
        "    else:\n"
        "        cndd2 = ret_val\n"
        "\n"
        "    expRT = exp((-1.0 * R) * T)\n"
        "    callResult = (S * cndd1 - X * expRT * cndd2)\n"
        "    return callResult\n";

    const int n = 8;
    double S[8] = {100.0, 105.0, 110.0, 95.0, 120.0, 80.0, 150.0, 60.0};
    double X[8] = {100.0, 100.0, 115.0, 90.0, 110.0, 85.0, 140.0, 65.0};
    double T[8] = {1.0, 0.5, 2.0, 1.5, 0.25, 3.0, 0.75, 1.2};
    double R[8] = {0.05, 0.03, 0.04, 0.02, 0.01, 0.06, 0.025, 0.015};
    double V[8] = {0.2, 0.25, 0.3, 0.18, 0.35, 0.22, 0.28, 0.32};
    const void *inputs[] = {S, X, T, R, V};

    double expected[8];
    for (int i = 0; i < n; i++) {
        double A1 = 0.31938153;
        double A2 = -0.356563782;
        double A3 = 1.781477937;
        double A4 = -1.821255978;
        double A5 = 1.330274429;
        double RSQRT2PI = 0.39894228040143267793994605993438;

        double sqrtT = sqrt(T[i]);
        double d1 = (log(S[i] / X[i]) + (R[i] + 0.5 * V[i] * V[i]) * T[i]) / (V[i] * sqrtT);
        double d2 = d1 - V[i] * sqrtT;
        double K = 1.0 / (1.0 + 0.2316419 * fabs(d1));
        double ret_val = RSQRT2PI * exp(-0.5 * d1 * d1) *
                         (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
        double cndd1 = (d1 > 0.0) ? (1.0 - ret_val) : ret_val;

        K = 1.0 / (1.0 + 0.2316419 * fabs(d2));
        ret_val = RSQRT2PI * exp(-0.5 * d2 * d2) *
                  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
        double cndd2 = (d2 > 0.0) ? (1.0 - ret_val) : ret_val;

        double expRT = exp((-1.0 * R[i]) * T[i]);
        expected[i] = S[i] * cndd1 - X[i] * expRT * cndd2;
    }

    me_variable vars[] = {
        {"S", ME_FLOAT64},
        {"X", ME_FLOAT64},
        {"T", ME_FLOAT64},
        {"R", ME_FLOAT64},
        {"V", ME_FLOAT64}
    };
    double out_interp[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double out_jit[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    char *saved_jit = dup_env_value("ME_DSL_JIT");

    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 5, inputs, n, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 5, inputs, n, out_jit) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    if (check_all_close(out_interp, expected, n, 1e-6) != 0) {
        printf("  ❌ FAILED: unexpected interpreter output\n");
        return 1;
    }
    if (check_all_close(out_jit, out_interp, n, 1e-10) != 0) {
        printf("  ❌ FAILED: interpreter/JIT mismatch\n");
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_black_scholes_dsl_kernel_specialized_constants_support(void) {
    printf("\n=== DSL Test 10c: Black-Scholes specialized-constants kernel support ===\n");

    const char *src =
        "def kernel(S, X, T):\n"
        "    A1 = 0.31938153\n"
        "    A2 = -0.356563782\n"
        "    A3 = 1.781477937\n"
        "    A4 = -1.821255978\n"
        "    A5 = 1.330274429\n"
        "    RSQRT2PI = 0.39894228040143267793994605993438\n"
        "\n"
        "    sqrtT = sqrt(T)\n"
        "    d1 = (log(S / X) + (0.1 + 0.5 * 1.0 * 1.0) * T) / (1.0 * sqrtT)\n"
        "    d2 = d1 - 1.0 * sqrtT\n"
        "    K = 1.0 / (1.0 + 0.2316419 * abs(d1))\n"
        "\n"
        "    ret_val = (RSQRT2PI * exp(-0.5 * d1 * d1) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
        "    cndd1 = ret_val\n"
        "    if d1 > 0:\n"
        "        cndd1 = 1.0 - ret_val\n"
        "    else:\n"
        "        cndd1 = ret_val\n"
        "\n"
        "    K = 1.0 / (1.0 + 0.2316419 * abs(d2))\n"
        "    ret_val = (RSQRT2PI * exp(-0.5 * d2 * d2) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
        "    if d2 > 0:\n"
        "        cndd2 = 1.0 - ret_val\n"
        "    else:\n"
        "        cndd2 = ret_val\n"
        "\n"
        "    expRT = exp((-1.0 * 0.1) * T)\n"
        "    callResult = (S * cndd1 - X * expRT * cndd2)\n"
        "    return callResult\n";

    const int n = 8;
    double S[8] = {100.0, 105.0, 110.0, 95.0, 120.0, 80.0, 150.0, 60.0};
    double X[8] = {100.0, 100.0, 115.0, 90.0, 110.0, 85.0, 140.0, 65.0};
    double T[8] = {1.0, 0.5, 2.0, 1.5, 0.25, 3.0, 0.75, 1.2};
    const void *inputs[] = {S, X, T};

    double expected[8];
    for (int i = 0; i < n; i++) {
        double A1 = 0.31938153;
        double A2 = -0.356563782;
        double A3 = 1.781477937;
        double A4 = -1.821255978;
        double A5 = 1.330274429;
        double RSQRT2PI = 0.39894228040143267793994605993438;

        double sqrtT = sqrt(T[i]);
        double d1 = (log(S[i] / X[i]) + (0.1 + 0.5 * 1.0 * 1.0) * T[i]) / (1.0 * sqrtT);
        double d2 = d1 - 1.0 * sqrtT;
        double K = 1.0 / (1.0 + 0.2316419 * fabs(d1));
        double ret_val = RSQRT2PI * exp(-0.5 * d1 * d1) *
                         (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
        double cndd1 = (d1 > 0.0) ? (1.0 - ret_val) : ret_val;

        K = 1.0 / (1.0 + 0.2316419 * fabs(d2));
        ret_val = RSQRT2PI * exp(-0.5 * d2 * d2) *
                  (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
        double cndd2 = (d2 > 0.0) ? (1.0 - ret_val) : ret_val;

        double expRT = exp((-1.0 * 0.1) * T[i]);
        expected[i] = S[i] * cndd1 - X[i] * expRT * cndd2;
    }

    me_variable vars[] = {
        {"S", ME_FLOAT64},
        {"X", ME_FLOAT64},
        {"T", ME_FLOAT64}
    };
    double out_interp[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double out_jit[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    char *saved_jit = dup_env_value("ME_DSL_JIT");

    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 3, inputs, n, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 3, inputs, n, out_jit) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    if (check_all_close(out_interp, expected, n, 1e-6) != 0) {
        printf("  ❌ FAILED: unexpected interpreter output\n");
        return 1;
    }
    if (check_all_close(out_jit, out_interp, n, 1e-10) != 0) {
        printf("  ❌ FAILED: interpreter/JIT mismatch\n");
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_loop_condition_policy(void) {
    printf("\n=== DSL Test 11: loop condition policy ===\n");

    const char *src =
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(3):\n"
        "        if x > 0:\n"
        "            acc = acc + 1\n"
        "    return acc\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {-1.0, 2.0, -3.0, 0.0};
    double out[4];
    double expected[4] = {0.0, 3.0, 0.0, 0.0};
    const void *inputs[] = {x};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }
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

static int test_elementwise_break(void) {
    printf("\n=== DSL Test 12: element-wise break ===\n");

    const char *src =
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(5):\n"
        "        if x > i:\n"
        "            acc = acc + 1\n"
        "        else:\n"
        "            break\n"
        "    return acc\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {-1.0, 2.0, 5.0, 0.0};
    double out[4];
    double expected[4] = {0.0, 2.0, 5.0, 0.0};

    int err = 0;
    me_expr *expr = NULL;
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

static int test_reduction_any_remains_global(void) {
    printf("\n=== DSL Test 13: reduction any() remains global ===\n");

    const char *src =
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(4):\n"
        "        if any(x > 0):\n"
        "            acc = acc + 1\n"
        "            break\n"
        "    return acc\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {-1.0, 2.0, -3.0, 0.0};
    double out[4];
    double expected[4] = {1.0, 1.0, 1.0, 1.0};

    int err = 0;
    me_expr *expr = NULL;
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

static int test_interpreter_jit_parity(void) {
    printf("\n=== DSL Test 14: interpreter/JIT parity ===\n");

    const char *src =
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(8):\n"
        "        if i == 0:\n"
        "            continue\n"
        "        if x > i:\n"
        "            acc = acc + i\n"
        "        else:\n"
        "            break\n"
        "    return acc\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {0.0, 2.0, 5.0, 10.0};
    const void *inputs[] = {x};
    double out_interp[4] = {0.0, 0.0, 0.0, 0.0};
    double out_jit[4] = {0.0, 0.0, 0.0, 0.0};

    char *saved_jit = dup_env_value("ME_DSL_JIT");

    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 1, inputs, 4, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 1, inputs, 4, out_jit) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    for (int i = 0; i < 4; i++) {
        if (out_interp[i] != out_jit[i]) {
            printf("  ❌ FAILED: mismatch at %d (interp=%.17g jit=%.17g)\n",
                   i, out_interp[i], out_jit[i]);
            return 1;
        }
    }

    double expected[4] = {0.0, 1.0, 10.0, 28.0};
    if (check_all_close(out_interp, expected, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: unexpected parity output\n");
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_unknown_me_pragma_rejected(void) {
    printf("\n=== DSL Test 15: unknown me pragma rejected ===\n");

    const char *src =
        "# me:bogus=element\n"
        "def kernel(x):\n"
        "    return x\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: unknown me pragma should be rejected\n");
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_return_inside_loop_interpreter_jit_parity(void) {
    printf("\n=== DSL Test 16: return-in-loop interpreter/JIT parity ===\n");

    const char *src =
        "def kernel(x):\n"
        "    for i in range(4):\n"
        "        if x == i:\n"
        "            return i + 10\n"
        "    return -1\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {0.0, 1.0, 3.0, 5.0};
    const void *inputs[] = {x};
    double out_interp[4] = {0.0, 0.0, 0.0, 0.0};
    double out_jit[4] = {0.0, 0.0, 0.0, 0.0};
    double expected[4] = {10.0, 11.0, 13.0, -1.0};

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 1, inputs, 4, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 1, inputs, 4, out_jit) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    if (check_all_close(out_interp, expected, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: unexpected interpreter output\n");
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        if (out_interp[i] != out_jit[i]) {
            printf("  ❌ FAILED: mismatch at %d (interp=%.17g jit=%.17g)\n",
                   i, out_interp[i], out_jit[i]);
            return 1;
        }
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_missing_return_runtime_error_with_jit_enabled(void) {
    printf("\n=== DSL Test 17: missing return runtime error with JIT enabled ===\n");

    const char *src =
        "def kernel(x):\n"
        "    if any(x > 0):\n"
        "        return 1\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {-1.0, -2.0, -3.0, -4.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs[] = {x};

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        free(saved_jit);
        return 1;
    }

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }

    int rc_eval = me_eval(expr, inputs, 1, out, 4, NULL);
    me_free(expr);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    if (rc_eval != ME_EVAL_ERR_INVALID_ARG) {
        printf("  ❌ FAILED: expected runtime missing-return error, got %d\n", rc_eval);
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_branch_local_decls_interpreter_jit_parity(void) {
    printf("\n=== DSL Test 18: branch-local declarations interpreter/JIT parity ===\n");

    const char *src =
        "def kernel(x):\n"
        "    if x > 0:\n"
        "        y = x + 10\n"
        "        return y\n"
        "    else:\n"
        "        z = x - 10\n"
        "        return z\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {-2.0, 0.0, 3.0, 5.0};
    const void *inputs[] = {x};
    double out_interp[4] = {0.0, 0.0, 0.0, 0.0};
    double out_jit[4] = {0.0, 0.0, 0.0, 0.0};
    double expected[4] = {-12.0, -10.0, 13.0, 15.0};

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 1, inputs, 4, out_interp) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    if (compile_eval_double(src, vars, 1, inputs, 4, out_jit) != 0) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        return 1;
    }
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);

    if (check_all_close(out_interp, expected, 4, 1e-12) != 0) {
        printf("  ❌ FAILED: unexpected branch-local interpreter output\n");
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        if (out_interp[i] != out_jit[i]) {
            printf("  ❌ FAILED: mismatch at %d (interp=%.17g jit=%.17g)\n",
                   i, out_interp[i], out_jit[i]);
            return 1;
        }
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_fp_pragma_modes(void) {
    printf("\n=== DSL Test 19: fp pragma modes ===\n");

    const char *src_strict =
        "# me:fp=strict\n"
        "def kernel(x):\n"
        "    return x * x + 1\n";
    const char *src_contract =
        "# me:fp=contract\n"
        "def kernel(x):\n"
        "    return x * x + 1\n";
    const char *src_fast =
        "# me:fp=fast\n"
        "def kernel(x):\n"
        "    return x * x + 1\n";
    const char *src_invalid =
        "# me:fp=ultra\n"
        "def kernel(x):\n"
        "    return x\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double x[4] = {1.0, 2.0, -3.0, 0.5};
    const void *inputs[] = {x};
    double out[4];
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(src_strict, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: strict fp pragma compile error at %d\n", err);
        return 1;
    }
    if (me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: strict fp pragma eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    expr = NULL;
    if (me_compile(src_contract, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: contract fp pragma compile error at %d\n", err);
        return 1;
    }
    me_free(expr);

    expr = NULL;
    if (me_compile(src_fast, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: fast fp pragma compile error at %d\n", err);
        return 1;
    }
    me_free(expr);

    expr = NULL;
    if (me_compile(src_invalid, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: invalid fp pragma compiled successfully\n");
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_dsl_print_stmt(void) {
    printf("\n=== DSL Test 20: print statement ===\n");

    const char *src =
        "def kernel():\n"
        "    print(\"value = {}\", 1 + 2)\n"
        "    print(\"sum =\", 1 + 2)\n"
        "    print(1 + 2)\n"
        "    print(\"sum =\", 1 + 2, 3 + 4)\n"
        "    return 0\n";

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double out[4];
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }

    double expected[4] = {0.0, 0.0, 0.0, 0.0};
    int rc = check_all_close(out, expected, 4, 1e-12);
    me_free(expr);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

static int test_cast_intrinsics(void) {
    printf("\n=== DSL Test 22: cast intrinsics int/float/bool ===\n");

    {
        const char *src =
            "def kernel():\n"
            "    return float(3)\n";
        me_expr *expr = NULL;
        int err = 0;
        if (me_compile(src, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: float() compile error at %d\n", err);
            return 1;
        }
        double out[4] = {-1.0, -1.0, -1.0, -1.0};
        if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
            printf("  ❌ FAILED: float() eval error\n");
            me_free(expr);
            return 1;
        }
        me_free(expr);
        for (int i = 0; i < 4; i++) {
            if (out[i] != 3.0) {
                printf("  ❌ FAILED: float() mismatch at %d got %.12f expected 3.0\n", i, out[i]);
                return 1;
            }
        }
    }

    {
        const char *src =
            "def kernel():\n"
            "    return int(3.9)\n";
        me_expr *expr = NULL;
        int err = 0;
        if (me_compile(src, NULL, 0, ME_INT64, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: int() compile error at %d\n", err);
            return 1;
        }
        int64_t out[4] = {-1, -1, -1, -1};
        if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
            printf("  ❌ FAILED: int() eval error\n");
            me_free(expr);
            return 1;
        }
        me_free(expr);
        for (int i = 0; i < 4; i++) {
            if (out[i] != 3) {
                printf("  ❌ FAILED: int() mismatch at %d got %lld expected 3\n",
                       i, (long long)out[i]);
                return 1;
            }
        }
    }

    {
        const char *src =
            "def kernel(x):\n"
            "    return bool(x)\n";
        me_variable vars[] = {{"x", ME_FLOAT64}};
        double x[4] = {0.0, -1.5, 0.0, 2.0};
        const void *inputs[] = {x};
        me_expr *expr = NULL;
        int err = 0;
        if (me_compile(src, vars, 1, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: bool() compile error at %d\n", err);
            return 1;
        }
        bool out[4] = {false, false, false, false};
        if (me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
            printf("  ❌ FAILED: bool() eval error\n");
            me_free(expr);
            return 1;
        }
        me_free(expr);
        bool expected[4] = {false, true, false, true};
        for (int i = 0; i < 4; i++) {
            if (out[i] != expected[i]) {
                printf("  ❌ FAILED: bool() mismatch at %d got %d expected %d\n",
                       i, (int)out[i], (int)expected[i]);
                return 1;
            }
        }
    }

    {
        const char *src_bad0 =
            "def kernel(x):\n"
            "    return float()\n";
        const char *src_bad2 =
            "def kernel(x):\n"
            "    return float(x, x)\n";
        const char *src_unknown =
            "def kernel(x):\n"
            "    return floaty(x)\n";
        me_variable vars[] = {{"x", ME_FLOAT64}};
        int err = 0;
        me_expr *expr = NULL;

        if (me_compile(src_bad0, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: float() with zero args unexpectedly compiled\n");
            me_free(expr);
            return 1;
        }
        if (me_compile(src_bad2, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: float(x, x) unexpectedly compiled\n");
            me_free(expr);
            return 1;
        }
        if (me_compile(src_unknown, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: floaty(x) unexpectedly compiled\n");
            me_free(expr);
            return 1;
        }
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_cast_conversion_runtime_parity(void) {
    printf("\n=== DSL Test 23: cast conversion runtime parity ===\n");

    int rc = 1;
    char *saved_jit = dup_env_value("ME_DSL_JIT");

    {
        const char *src =
            "def kernel():\n"
            "    return int(3.9)\n";
        int64_t out_interp[4] = {0, 0, 0, 0};
        int64_t out_jit[4] = {0, 0, 0, 0};

        if (setenv("ME_DSL_JIT", "0", 1) != 0) {
            printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
            goto cleanup;
        }
        if (compile_eval_int64(src, NULL, 0, NULL, 4, out_interp) != 0) {
            goto cleanup;
        }

        if (setenv("ME_DSL_JIT", "1", 1) != 0) {
            printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
            goto cleanup;
        }
        if (compile_eval_int64(src, NULL, 0, NULL, 4, out_jit) != 0) {
            goto cleanup;
        }

        for (int i = 0; i < 4; i++) {
            if (out_interp[i] != out_jit[i] || out_jit[i] != 3) {
                printf("  ❌ FAILED: int(3.9) mismatch at %d (interp=%lld jit=%lld)\n",
                       i, (long long)out_interp[i], (long long)out_jit[i]);
                goto cleanup;
            }
        }
    }

    {
        const char *src =
            "def kernel(x):\n"
            "    return float(int(x)) + bool(x)\n";
        me_variable vars[] = {{"x", ME_FLOAT64}};
        double x[6] = {0.0, 0.2, 1.0, 1.9, 2.0, 3.2};
        double out_interp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double out_jit[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double expected[6] = {0.0, 1.0, 2.0, 2.0, 3.0, 4.0};
        const void *inputs[] = {x};

        if (setenv("ME_DSL_JIT", "0", 1) != 0) {
            printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
            goto cleanup;
        }
        if (compile_eval_double(src, vars, 1, inputs, 6, out_interp) != 0) {
            goto cleanup;
        }

        if (setenv("ME_DSL_JIT", "1", 1) != 0) {
            printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
            goto cleanup;
        }
        if (compile_eval_double(src, vars, 1, inputs, 6, out_jit) != 0) {
            goto cleanup;
        }

        for (int i = 0; i < 6; i++) {
            if (out_interp[i] != out_jit[i] || out_jit[i] != expected[i]) {
                printf("  ❌ FAILED: float(int(x))+bool(x) mismatch at %d (interp=%.17g jit=%.17g exp=%.17g)\n",
                       i, out_interp[i], out_jit[i], expected[i]);
                goto cleanup;
            }
        }
    }

    {
        const char *src =
            "def kernel(x):\n"
            "    return int(x)\n";
        me_variable vars[] = {{"x", ME_FLOAT32}};
        float x[6] = {0.0f, -1.9f, 2.2f, 3.9f, 255.8f, -1024.5f};
        int64_t out_interp[6] = {0, 0, 0, 0, 0, 0};
        int64_t out_jit[6] = {0, 0, 0, 0, 0, 0};
        int64_t expected[6] = {0, -1, 2, 3, 255, -1024};
        const void *inputs[] = {x};

        if (setenv("ME_DSL_JIT", "0", 1) != 0) {
            printf("  ❌ FAILED: setenv ME_DSL_JIT=0 failed\n");
            goto cleanup;
        }
        if (compile_eval_int64(src, vars, 1, inputs, 6, out_interp) != 0) {
            goto cleanup;
        }

        if (setenv("ME_DSL_JIT", "1", 1) != 0) {
            printf("  ❌ FAILED: setenv ME_DSL_JIT=1 failed\n");
            goto cleanup;
        }
        if (compile_eval_int64(src, vars, 1, inputs, 6, out_jit) != 0) {
            goto cleanup;
        }

        for (int i = 0; i < 6; i++) {
            if (out_interp[i] != out_jit[i] || out_jit[i] != expected[i]) {
                printf("  ❌ FAILED: float32->int64 cast mismatch at %d (interp=%lld jit=%lld exp=%lld)\n",
                       i, (long long)out_interp[i], (long long)out_jit[i], (long long)expected[i]);
                goto cleanup;
            }
        }
    }

    rc = 0;
    printf("  ✅ PASSED\n");

cleanup:
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);
    return rc;
}

static int test_compound_assignment_desugar(void) {
    printf("\n=== DSL Test 24: compound assignment desugaring ===\n");

    const char *src =
        "def kernel():\n"
        "    i = 10\n"
        "    i += 5\n"
        "    i -= 3\n"
        "    i *= 2\n"
        "    i /= 4\n"
        "    i //= 2\n"
        "    return i\n";
    const char *src_floor =
        "def kernel():\n"
        "    i = -3\n"
        "    i //= 2\n"
        "    return i\n";

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compile error at %d\n", err);
        return 1;
    }

    double out[4] = {-1.0, -1.0, -1.0, -1.0};
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: eval error\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    me_expr *expr_floor = NULL;
    err = 0;
    if (me_compile(src_floor, NULL, 0, ME_FLOAT64, &err, &expr_floor) != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: floor compile error at %d\n", err);
        return 1;
    }
    double out_floor[4] = {-1.0, -1.0, -1.0, -1.0};
    if (me_eval(expr_floor, NULL, 0, out_floor, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  ❌ FAILED: floor eval error\n");
        me_free(expr_floor);
        return 1;
    }
    me_free(expr_floor);

    double expected[4] = {3.0, 3.0, 3.0, 3.0};
    double expected_floor[4] = {-2.0, -2.0, -2.0, -2.0};
    int rc = check_all_close(out, expected, 4, 1e-12);
    rc |= check_all_close(out_floor, expected_floor, 4, 1e-12);
    if (rc == 0) {
        printf("  ✅ PASSED\n");
    }
    return rc;
}

int main(void) {
    int fail = 0;
    fail |= test_assign_and_result_stmt();
    fail |= test_loop_break_continue();
    fail |= test_range_start_stop_step();
    fail |= test_while_loop_semantics();
    fail |= test_while_iteration_cap();
    fail |= test_invalid_conditionals();
    fail |= test_if_elif_else();
    fail |= test_nd_indices();
    fail |= test_flat_idx();
    fail |= test_reserved_index_vars_jit_parity();
    fail |= test_reserved_index_vars_env_gate();
    fail |= test_nd_padding();
    fail |= test_nd_large_block();
    fail |= test_nd_3d_indices_padding();
    fail |= test_nested_loops_and_conditionals();
    fail |= test_break_any_condition();
    fail |= test_dsl_function_calls();
    fail |= test_black_scholes_dsl_kernel_support();
    fail |= test_black_scholes_dsl_kernel_specialized_constants_support();
    fail |= test_loop_condition_policy();
    fail |= test_elementwise_break();
    fail |= test_reduction_any_remains_global();
    fail |= test_interpreter_jit_parity();
    fail |= test_unknown_me_pragma_rejected();
    fail |= test_return_inside_loop_interpreter_jit_parity();
    fail |= test_missing_return_runtime_error_with_jit_enabled();
    fail |= test_branch_local_decls_interpreter_jit_parity();
    fail |= test_fp_pragma_modes();
    fail |= test_dsl_print_stmt();
    fail |= test_string_truthiness();
    fail |= test_cast_intrinsics();
    fail |= test_cast_conversion_runtime_parity();
    fail |= test_compound_assignment_desugar();
    return fail;
}
