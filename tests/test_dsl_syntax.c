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

    me_variable_ex vars[] = {
        {"name", ME_STRING, names, ME_VARIABLE, NULL, sizeof(names[0])}
    };
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile_ex(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
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
    fail |= test_nd_padding();
    fail |= test_nd_large_block();
    fail |= test_nd_3d_indices_padding();
    fail |= test_nested_loops_and_conditionals();
    fail |= test_break_any_condition();
    fail |= test_dsl_function_calls();
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
    return fail;
}
