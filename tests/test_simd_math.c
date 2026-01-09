/*
 * SIMD math tests for functions accelerated via SLEEF.
 */
#include <math.h>
#include <stdbool.h>
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

static void fill_input_range_f64(double *input, int n, double min_val, double max_val) {
    double span = max_val - min_val;
    for (int i = 0; i < n; i++) {
        double t = (n > 1) ? (double)i / (double)(n - 1) : 0.0;
        input[i] = min_val + span * t;
    }
}

static void fill_input_range_f32(float *input, int n, float min_val, float max_val) {
    float span = max_val - min_val;
    for (int i = 0; i < n; i++) {
        float t = (n > 1) ? (float)i / (float)(n - 1) : 0.0f;
        input[i] = min_val + span * t;
    }
}

static int run_unary_f64(const char *name, double (*func)(double), int n,
                         bool simd_enabled, double min_val, double max_val, double tol) {
    double *input = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!input || !out) {
        printf("Allocation failed\n");
        free(input);
        free(out);
        return 1;
    }

    fill_input_range_f64(input, n, min_val, max_val);

    me_variable vars[] = {{"x", ME_FLOAT64, input}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_text[32];

    snprintf(expr_text, sizeof(expr_text), "%s(x)", name);
    if (me_compile(expr_text, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s (err=%d)\n", expr_text, err);
        free(input);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {input};
    me_disable_simd(!simd_enabled);
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        free(input);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = func(input[i]);
        if (!nearly_equal(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s mismatch at %d: got %.15f expected %.15f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(input);
    free(out);
    me_disable_simd(false);

    if (failures) {
        printf("%s f64 %s FAIL: %d mismatches\n",
               expr_text, simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("%s f64 %s PASS\n", expr_text, simd_enabled ? "SIMD" : "scalar");
    return 0;
}

static int run_unary_f32(const char *name, float (*func)(float), int n,
                         bool simd_enabled, float min_val, float max_val, float tol) {
    float *input = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!input || !out) {
        printf("Allocation failed\n");
        free(input);
        free(out);
        return 1;
    }

    fill_input_range_f32(input, n, min_val, max_val);

    me_variable vars[] = {{"x", ME_FLOAT32, input}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_text[32];

    snprintf(expr_text, sizeof(expr_text), "%s(x)", name);
    if (me_compile(expr_text, vars, 1, ME_FLOAT32, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s f32 (err=%d)\n", expr_text, err);
        free(input);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {input};
    me_disable_simd(!simd_enabled);
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s f32 eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        free(input);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = func(input[i]);
        if (!nearly_equal_f(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s f32 mismatch at %d: got %.7f expected %.7f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(input);
    free(out);
    me_disable_simd(false);

    if (failures) {
        printf("%s f32 %s FAIL: %d mismatches\n",
               expr_text, simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("%s f32 %s PASS\n", expr_text, simd_enabled ? "SIMD" : "scalar");
    return 0;
}

static int run_binary_f64(const char *name, double (*func)(double, double), int n,
                          bool simd_enabled, double a_min, double a_max,
                          double b_min, double b_max, double tol) {
    double *a = (double *)malloc((size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!a || !b || !out) {
        printf("Allocation failed\n");
        free(a);
        free(b);
        free(out);
        return 1;
    }

    fill_input_range_f64(a, n, a_min, a_max);
    fill_input_range_f64(b, n, b_min, b_max);

    me_variable vars[] = {{"a", ME_FLOAT64, a}, {"b", ME_FLOAT64, b}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_text[48];

    snprintf(expr_text, sizeof(expr_text), "%s(a, b)", name);
    if (me_compile(expr_text, vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s (err=%d)\n", expr_text, err);
        free(a);
        free(b);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    me_disable_simd(!simd_enabled);
    int eval_rc = me_eval(expr, var_ptrs, 2, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        free(a);
        free(b);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = func(a[i], b[i]);
        if (!nearly_equal(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s mismatch at %d: got %.15f expected %.15f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(a);
    free(b);
    free(out);
    me_disable_simd(false);

    if (failures) {
        printf("%s f64 %s FAIL: %d mismatches\n",
               expr_text, simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("%s f64 %s PASS\n", expr_text, simd_enabled ? "SIMD" : "scalar");
    return 0;
}

static int run_binary_f32(const char *name, float (*func)(float, float), int n,
                          bool simd_enabled, float a_min, float a_max,
                          float b_min, float b_max, float tol) {
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!a || !b || !out) {
        printf("Allocation failed\n");
        free(a);
        free(b);
        free(out);
        return 1;
    }

    fill_input_range_f32(a, n, a_min, a_max);
    fill_input_range_f32(b, n, b_min, b_max);

    me_variable vars[] = {{"a", ME_FLOAT32, a}, {"b", ME_FLOAT32, b}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_text[48];

    snprintf(expr_text, sizeof(expr_text), "%s(a, b)", name);
    if (me_compile(expr_text, vars, 2, ME_FLOAT32, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s f32 (err=%d)\n", expr_text, err);
        free(a);
        free(b);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    me_disable_simd(!simd_enabled);
    int eval_rc = me_eval(expr, var_ptrs, 2, out, n);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s f32 eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        free(a);
        free(b);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = func(a[i], b[i]);
        if (!nearly_equal_f(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s f32 mismatch at %d: got %.7f expected %.7f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(a);
    free(b);
    free(out);
    me_disable_simd(false);

    if (failures) {
        printf("%s f32 %s FAIL: %d mismatches\n",
               expr_text, simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("%s f32 %s PASS\n", expr_text, simd_enabled ? "SIMD" : "scalar");
    return 0;
}

static int run_unary_pair_f64(const char *name, double (*func)(double), int n,
                              double min_val, double max_val, double tol) {
    int failures = 0;
    failures += run_unary_f64(name, func, n, false, min_val, max_val, tol);
    failures += run_unary_f64(name, func, n, true, min_val, max_val, tol);
    return failures;
}

static int run_unary_pair_f32(const char *name, float (*func)(float), int n,
                              float min_val, float max_val, float tol) {
    int failures = 0;
    failures += run_unary_f32(name, func, n, false, min_val, max_val, tol);
    failures += run_unary_f32(name, func, n, true, min_val, max_val, tol);
    return failures;
}

static int run_binary_pair_f64(const char *name, double (*func)(double, double), int n,
                               double a_min, double a_max, double b_min, double b_max,
                               double tol) {
    int failures = 0;
    failures += run_binary_f64(name, func, n, false, a_min, a_max, b_min, b_max, tol);
    failures += run_binary_f64(name, func, n, true, a_min, a_max, b_min, b_max, tol);
    return failures;
}

static int run_binary_pair_f32(const char *name, float (*func)(float, float), int n,
                               float a_min, float a_max, float b_min, float b_max,
                               float tol) {
    int failures = 0;
    failures += run_binary_f32(name, func, n, false, a_min, a_max, b_min, b_max, tol);
    failures += run_binary_f32(name, func, n, true, a_min, a_max, b_min, b_max, tol);
    return failures;
}

int main(void) {
    int failures = 0;
    const int n = 1024;

    failures += run_unary_pair_f64("abs", fabs, n, -10.0, 10.0, 1e-12);
    failures += run_unary_pair_f64("exp", exp, n, -5.0, 5.0, 1e-12);
    failures += run_unary_pair_f64("expm1", expm1, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("log", log, n, 0.1, 10.0, 1e-12);
    failures += run_unary_pair_f64("log10", log10, n, 0.1, 10.0, 1e-12);
    failures += run_unary_pair_f64("log1p", log1p, n, -0.9, 10.0, 1e-12);
    failures += run_unary_pair_f64("log2", log2, n, 0.1, 10.0, 1e-12);
    failures += run_unary_pair_f64("sqrt", sqrt, n, 0.0, 100.0, 1e-12);
    failures += run_unary_pair_f64("sinh", sinh, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("cosh", cosh, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("tanh", tanh, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("acosh", acosh, n, 1.0, 10.0, 1e-12);
    failures += run_unary_pair_f64("asinh", asinh, n, -5.0, 5.0, 1e-12);
    failures += run_unary_pair_f64("atanh", atanh, n, -0.9, 0.9, 1e-12);
    failures += run_unary_pair_f64("ceil", ceil, n, -3.5, 3.5, 1e-12);
    failures += run_unary_pair_f64("floor", floor, n, -3.5, 3.5, 1e-12);
    failures += run_unary_pair_f64("round", round, n, -3.5, 3.5, 1e-12);
    failures += run_unary_pair_f64("trunc", trunc, n, -3.5, 3.5, 1e-12);

    failures += run_binary_pair_f64("pow", pow, n, 0.1, 4.0, -2.0, 2.0, 1e-11);

    failures += run_unary_pair_f32("abs", fabsf, n, -10.0f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("exp", expf, n, -5.0f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("expm1", expm1f, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("log", logf, n, 0.1f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("log10", log10f, n, 0.1f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("log1p", log1pf, n, -0.9f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("log2", log2f, n, 0.1f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("sqrt", sqrtf, n, 0.0f, 100.0f, 1e-5f);
    failures += run_unary_pair_f32("sinh", sinhf, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("cosh", coshf, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("tanh", tanhf, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("acosh", acoshf, n, 1.0f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("asinh", asinhf, n, -5.0f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("atanh", atanhf, n, -0.9f, 0.9f, 1e-5f);
    failures += run_unary_pair_f32("ceil", ceilf, n, -3.5f, 3.5f, 1e-5f);
    failures += run_unary_pair_f32("floor", floorf, n, -3.5f, 3.5f, 1e-5f);
    failures += run_unary_pair_f32("round", roundf, n, -3.5f, 3.5f, 1e-5f);
    failures += run_unary_pair_f32("trunc", truncf, n, -3.5f, 3.5f, 1e-5f);

    failures += run_binary_pair_f32("pow", powf, n, 0.1f, 4.0f, -2.0f, 2.0f, 1e-5f);

    return failures ? 1 : 0;
}
