#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "miniexpr.h"

static int nearly_equal(double a, double b, double tol) {
    double diff = fabs(a - b);
    if (diff <= tol) return 1;
    double scale = fmax(fabs(a), fabs(b));
    return diff <= tol * scale;
}

static int nearly_equal_f(float a, float b, float tol) {
    float diff = fabsf(a - b);
    if (diff <= tol) return 1;
    float scale = fmaxf(fabsf(a), fabsf(b));
    return diff <= tol * scale;
}

static int test_identity(int n) {
    double *input = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!input || !out) {
        printf("Allocation failed\n");
        free(input);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        input[i] = (double)i * 0.001 - 1.0;
    }

    me_variable vars[] = {{"x", ME_FLOAT64, input}};
    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile("sin(x) * sin(x) + cos(x) * cos(x)", vars, 1, ME_FLOAT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("Failed to compile sin/cos expression (err=%d)\n", err);
        free(input);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {input};
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("Evaluation failed (err=%d)\n", eval_rc);
        me_free(expr);
        free(input);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = 1.0;
        if (!nearly_equal(out[i], expected, 1e-12)) {
            if (failures < 5) {
                printf("Identity mismatch at %d: got %.15f expected %.15f\n", i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(input);
    free(out);
    if (failures) {
        printf("Identity FAIL: %d mismatches\n", failures);
        return 1;
    }

    printf("Identity PASS\n");
    return 0;
}

static int test_sin_cos_f64(int n) {
    double *input = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!input || !out) {
        printf("Allocation failed\n");
        free(input);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        input[i] = (double)i * 0.0007 - 0.9;
    }

    me_variable vars[] = {{"x", ME_FLOAT64, input}};
    int err = 0;
    me_expr *expr_sin = NULL;
    me_expr *expr_cos = NULL;

    if (me_compile("sin(x)", vars, 1, ME_FLOAT64, &err, &expr_sin) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile sin(x) (err=%d)\n", err);
        free(input);
        free(out);
        return 1;
    }
    if (me_compile("cos(x)", vars, 1, ME_FLOAT64, &err, &expr_cos) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile cos(x) (err=%d)\n", err);
        me_free(expr_sin);
        free(input);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {input};
    int eval_rc = me_eval(expr_sin, var_ptrs, 1, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("sin(x) eval failed (err=%d)\n", eval_rc);
        me_free(expr_sin);
        me_free(expr_cos);
        free(input);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = sin(input[i]);
        if (!nearly_equal(out[i], expected, 1e-12)) {
            if (failures < 5) {
                printf("sin mismatch at %d: got %.15f expected %.15f\n", i, out[i], expected);
            }
            failures++;
        }
    }

    eval_rc = me_eval(expr_cos, var_ptrs, 1, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("cos(x) eval failed (err=%d)\n", eval_rc);
        me_free(expr_sin);
        me_free(expr_cos);
        free(input);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        double expected = cos(input[i]);
        if (!nearly_equal(out[i], expected, 1e-12)) {
            if (failures < 5) {
                printf("cos mismatch at %d: got %.15f expected %.15f\n", i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr_sin);
    me_free(expr_cos);
    free(input);
    free(out);

    if (failures) {
        printf("Trig f64 FAIL: %d mismatches\n", failures);
        return 1;
    }

    printf("Trig f64 PASS\n");
    return 0;
}

static int test_sin_cos_f32(int n) {
    float *input = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!input || !out) {
        printf("Allocation failed\n");
        free(input);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        input[i] = (float)i * 0.0007f - 0.9f;
    }

    me_variable vars[] = {{"x", ME_FLOAT32, input}};
    int err = 0;
    me_expr *expr_sin = NULL;
    me_expr *expr_cos = NULL;

    if (me_compile("sin(x)", vars, 1, ME_FLOAT32, &err, &expr_sin) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile sin(x) f32 (err=%d)\n", err);
        free(input);
        free(out);
        return 1;
    }
    if (me_compile("cos(x)", vars, 1, ME_FLOAT32, &err, &expr_cos) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile cos(x) f32 (err=%d)\n", err);
        me_free(expr_sin);
        free(input);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {input};
    int eval_rc = me_eval(expr_sin, var_ptrs, 1, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("sin(x) f32 eval failed (err=%d)\n", eval_rc);
        me_free(expr_sin);
        me_free(expr_cos);
        free(input);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = sinf(input[i]);
        if (!nearly_equal_f(out[i], expected, 1e-6f)) {
            if (failures < 5) {
                printf("sin f32 mismatch at %d: got %.7f expected %.7f\n", i, out[i], expected);
            }
            failures++;
        }
    }

    eval_rc = me_eval(expr_cos, var_ptrs, 1, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("cos(x) f32 eval failed (err=%d)\n", eval_rc);
        me_free(expr_sin);
        me_free(expr_cos);
        free(input);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        float expected = cosf(input[i]);
        if (!nearly_equal_f(out[i], expected, 1e-6f)) {
            if (failures < 5) {
                printf("cos f32 mismatch at %d: got %.7f expected %.7f\n", i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr_sin);
    me_free(expr_cos);
    free(input);
    free(out);

    if (failures) {
        printf("Trig f32 FAIL: %d mismatches\n", failures);
        return 1;
    }

    printf("Trig f32 PASS\n");
    return 0;
}

int main(void) {
    printf("=== Testing trig functions ===\n");

    int rc = 0;
    rc |= test_identity(1024);
    rc |= test_sin_cos_f64(2048);
    rc |= test_sin_cos_f32(2048);

    if (rc == 0) {
        printf("PASS\n");
        return 0;
    }

    printf("FAIL\n");
    return 1;
}
