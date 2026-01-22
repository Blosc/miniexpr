/*
 * SIMD math tests for functions accelerated via SLEEF.
 */
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "functions-simd.h"
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

static double exp10_ref(double x) {
    return pow(10.0, x);
}

static float exp10_ref_f(float x) {
    return powf(10.0f, x);
}

static double sinpi_ref(double x) {
    const double pi = 3.14159265358979323846;
    return sin(pi * x);
}

static float sinpi_ref_f(float x) {
    const float pi = 3.14159265358979323846f;
    return sinf(pi * x);
}

static double cospi_ref(double x) {
    const double pi = 3.14159265358979323846;
    return cos(pi * x);
}

static float cospi_ref_f(float x) {
    const float pi = 3.14159265358979323846f;
    return cosf(pi * x);
}

static double ldexp_ref(double x, double exp) {
    return ldexp(x, (int)exp);
}

static float ldexp_ref_f(float x, float exp) {
    return ldexpf(x, (int)exp);
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
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, &eval_params);
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
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, &eval_params);
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
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 2, out, n, &eval_params);
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
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 2, out, n, &eval_params);
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

    if (failures) {
        printf("%s f32 %s FAIL: %d mismatches\n",
               expr_text, simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("%s f32 %s PASS\n", expr_text, simd_enabled ? "SIMD" : "scalar");
    return 0;
}

static int run_ternary_f64(const char *name, double (*func)(double, double, double), int n,
                           bool simd_enabled, double a_min, double a_max,
                           double b_min, double b_max, double c_min, double c_max,
                           double tol) {
    double *a = (double *)malloc((size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *c = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!a || !b || !c || !out) {
        printf("Allocation failed\n");
        free(a);
        free(b);
        free(c);
        free(out);
        return 1;
    }

    fill_input_range_f64(a, n, a_min, a_max);
    fill_input_range_f64(b, n, b_min, b_max);
    fill_input_range_f64(c, n, c_min, c_max);

    me_variable vars[] = {{"a", ME_FLOAT64, a}, {"b", ME_FLOAT64, b}, {"c", ME_FLOAT64, c}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_text[64];

    snprintf(expr_text, sizeof(expr_text), "%s(a, b, c)", name);
    if (me_compile(expr_text, vars, 3, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s (err=%d)\n", expr_text, err);
        free(a);
        free(b);
        free(c);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {a, b, c};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 3, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        free(a);
        free(b);
        free(c);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = func(a[i], b[i], c[i]);
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
    free(c);
    free(out);

    if (failures) {
        printf("%s f64 %s FAIL: %d mismatches\n",
               expr_text, simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("%s f64 %s PASS\n", expr_text, simd_enabled ? "SIMD" : "scalar");
    return 0;
}

static int run_ternary_f32(const char *name, float (*func)(float, float, float), int n,
                           bool simd_enabled, float a_min, float a_max,
                           float b_min, float b_max, float c_min, float c_max,
                           float tol) {
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    float *c = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!a || !b || !c || !out) {
        printf("Allocation failed\n");
        free(a);
        free(b);
        free(c);
        free(out);
        return 1;
    }

    fill_input_range_f32(a, n, a_min, a_max);
    fill_input_range_f32(b, n, b_min, b_max);
    fill_input_range_f32(c, n, c_min, c_max);

    me_variable vars[] = {{"a", ME_FLOAT32, a}, {"b", ME_FLOAT32, b}, {"c", ME_FLOAT32, c}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_text[64];

    snprintf(expr_text, sizeof(expr_text), "%s(a, b, c)", name);
    if (me_compile(expr_text, vars, 3, ME_FLOAT32, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s f32 (err=%d)\n", expr_text, err);
        free(a);
        free(b);
        free(c);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {a, b, c};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 3, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s f32 eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        free(a);
        free(b);
        free(c);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = func(a[i], b[i], c[i]);
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
    free(c);
    free(out);

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

static int run_ternary_pair_f64(const char *name, double (*func)(double, double, double), int n,
                                double a_min, double a_max, double b_min, double b_max,
                                double c_min, double c_max, double tol) {
    int failures = 0;
    failures += run_ternary_f64(name, func, n, false, a_min, a_max, b_min, b_max, c_min, c_max, tol);
    failures += run_ternary_f64(name, func, n, true, a_min, a_max, b_min, b_max, c_min, c_max, tol);
    return failures;
}

static int run_ternary_pair_f32(const char *name, float (*func)(float, float, float), int n,
                                float a_min, float a_max, float b_min, float b_max,
                                float c_min, float c_max, float tol) {
    int failures = 0;
    failures += run_ternary_f32(name, func, n, false, a_min, a_max, b_min, b_max, c_min, c_max, tol);
    failures += run_ternary_f32(name, func, n, true, a_min, a_max, b_min, b_max, c_min, c_max, tol);
    return failures;
}

static int run_binary_const_f64(const char *expr_text, double (*func)(double, double),
                                const double *a, double b, double *out, int n,
                                bool simd_enabled, double tol) {
    me_variable vars[] = {{"a", ME_FLOAT64, (void*)a}};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(expr_text, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s (err=%d)\n", expr_text, err);
        return 1;
    }

    const void *var_ptrs[] = {a};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = func(a[i], b);
        if (!nearly_equal(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s mismatch at %d: got %.15f expected %.15f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    return failures;
}

static int run_binary_const_f32(const char *expr_text, float (*func)(float, float),
                                const float *a, float b, float *out, int n,
                                bool simd_enabled, float tol) {
    me_variable vars[] = {{"a", ME_FLOAT32, (void*)a}};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(expr_text, vars, 1, ME_FLOAT32, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s f32 (err=%d)\n", expr_text, err);
        return 1;
    }

    const void *var_ptrs[] = {a};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s f32 eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = func(a[i], b);
        if (!nearly_equal_f(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s f32 mismatch at %d: got %.7f expected %.7f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    return failures;
}

static int run_ternary_const_f64(const char *expr_text, double (*func)(double, double, double),
                                 const double *a, double b, double c, double *out, int n,
                                 bool simd_enabled, double tol) {
    me_variable vars[] = {{"a", ME_FLOAT64, (void*)a}};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(expr_text, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s (err=%d)\n", expr_text, err);
        return 1;
    }

    const void *var_ptrs[] = {a};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = func(a[i], b, c);
        if (!nearly_equal(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s mismatch at %d: got %.15f expected %.15f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    return failures;
}

static int run_ternary_const_f32(const char *expr_text, float (*func)(float, float, float),
                                 const float *a, float b, float c, float *out, int n,
                                 bool simd_enabled, float tol) {
    me_variable vars[] = {{"a", ME_FLOAT32, (void*)a}};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile(expr_text, vars, 1, ME_FLOAT32, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s f32 (err=%d)\n", expr_text, err);
        return 1;
    }

    const void *var_ptrs[] = {a};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s f32 eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = func(a[i], b, c);
        if (!nearly_equal_f(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s f32 mismatch at %d: got %.7f expected %.7f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    return failures;
}

static int run_nan_edge_cases(void) {
    int failures = 0;
    double nan = NAN;
    double out64[4] = {0};
    float out32[4] = {0};
    double a64[] = {nan, 1.0, nan, -2.0};
    double b64[] = {2.0, nan, nan, 3.0};
    float a32[] = {NAN, 1.0f, NAN, -2.0f};
    float b32[] = {2.0f, NAN, NAN, 3.0f};

    me_variable vars64[] = {{"a", ME_FLOAT64, a64}, {"b", ME_FLOAT64, b64}};
    me_variable vars32[] = {{"a", ME_FLOAT32, a32}, {"b", ME_FLOAT32, b32}};
    const void *ptrs64[] = {a64, b64};
    const void *ptrs32[] = {a32, b32};
    me_expr *expr = NULL;
    int err = 0;

    if (me_compile("fmax(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
            for (int i = 0; i < 4; i++) {
                double expected = fmax(a64[i], b64[i]);
                if ((isnan(expected) && !isnan(out64[i])) ||
                    (!isnan(expected) && expected != out64[i])) {
                    printf("fmax NaN edge case failed (f64)\n");
                    failures++;
                    break;
                }
            }
        } else {
            failures++;
        }
        me_free(expr);
    } else {
        failures++;
    }

    expr = NULL;
    if (me_compile("fmin(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
            for (int i = 0; i < 4; i++) {
                double expected = fmin(a64[i], b64[i]);
                if ((isnan(expected) && !isnan(out64[i])) ||
                    (!isnan(expected) && expected != out64[i])) {
                    printf("fmin NaN edge case failed (f64)\n");
                    failures++;
                    break;
                }
            }
        } else {
            failures++;
        }
        me_free(expr);
    } else {
        failures++;
    }

    expr = NULL;
    if (me_compile("fmax(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
        if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
            for (int i = 0; i < 4; i++) {
                float expected = fmaxf(a32[i], b32[i]);
                if ((isnan(expected) && !isnan(out32[i])) ||
                    (!isnan(expected) && expected != out32[i])) {
                    printf("fmax NaN edge case failed (f32)\n");
                    failures++;
                    break;
                }
            }
        } else {
            failures++;
        }
        me_free(expr);
    } else {
        failures++;
    }

    expr = NULL;
    if (me_compile("fmin(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
        if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
            for (int i = 0; i < 4; i++) {
                float expected = fminf(a32[i], b32[i]);
                if ((isnan(expected) && !isnan(out32[i])) ||
                    (!isnan(expected) && expected != out32[i])) {
                    printf("fmin NaN edge case failed (f32)\n");
                    failures++;
                    break;
                }
            }
        } else {
            failures++;
        }
        me_free(expr);
    } else {
        failures++;
    }

    return failures;
}

static int run_additional_edge_cases(void) {
    int failures = 0;

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {1.0, -0.0, INFINITY, -INFINITY};
        double b64[] = {-2.0, 3.0, 5.0, -7.0};
        float a32[] = {1.0f, -0.0f, INFINITY, -INFINITY};
        float b32[] = {-2.0f, 3.0f, 5.0f, -7.0f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}, {"b", ME_FLOAT64, b64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}, {"b", ME_FLOAT32, b32}};
        const void *ptrs64[] = {a64, b64};
        const void *ptrs32[] = {a32, b32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("copysign(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = copysign(a64[i], b64[i]);
                    if (memcmp(&expected, &out64[i], sizeof(double)) != 0) {
                        printf("copysign edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("copysign(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = copysignf(a32[i], b32[i]);
                    if (memcmp(&expected, &out32[i], sizeof(float)) != 0) {
                        printf("copysign edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {0.0, -0.0, 1.0, -1.0};
        double b64[] = {INFINITY, -INFINITY, INFINITY, -INFINITY};
        float a32[] = {0.0f, -0.0f, 1.0f, -1.0f};
        float b32[] = {INFINITY, -INFINITY, INFINITY, -INFINITY};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}, {"b", ME_FLOAT64, b64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}, {"b", ME_FLOAT32, b32}};
        const void *ptrs64[] = {a64, b64};
        const void *ptrs32[] = {a32, b32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("nextafter(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = nextafter(a64[i], b64[i]);
                    double abs_expected = fabs(expected);
                    if (abs_expected > 0.0 && abs_expected < DBL_MIN) {
                        if (out64[i] == 0.0) {
                            continue;
                        }
                    }
                    if (expected == 0.0 && out64[i] == 0.0) {
                        continue;
                    }
                    if (isinf(expected) && isinf(out64[i]) && (signbit(expected) == signbit(out64[i]))) {
                        continue;
                    }
                    if (memcmp(&expected, &out64[i], sizeof(double)) != 0) {
                        printf("nextafter edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
#ifdef _WIN32
        (void)out32;
#else
        if (me_compile("nextafter(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = nextafterf(a32[i], b32[i]);
                    float abs_expected = fabsf(expected);
                    if (abs_expected > 0.0f && abs_expected < FLT_MIN) {
                        if (out32[i] == 0.0f) {
                            continue;
                        }
                    }
                    if (expected == 0.0f && out32[i] == 0.0f) {
                        continue;
                    }
                    if (isinf(expected) && isinf(out32[i]) && (signbit(expected) == signbit(out32[i]))) {
                        continue;
                    }
                    if (memcmp(&expected, &out32[i], sizeof(float)) != 0) {
                        printf("nextafter edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
#endif
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {1.0, -1.0, 5.0, -5.0};
        double b64[] = {0.0, 0.0, 2.0, -2.0};
        float a32[] = {1.0f, -1.0f, 5.0f, -5.0f};
        float b32[] = {0.0f, 0.0f, 2.0f, -2.0f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}, {"b", ME_FLOAT64, b64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}, {"b", ME_FLOAT32, b32}};
        const void *ptrs64[] = {a64, b64};
        const void *ptrs32[] = {a32, b32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("remainder(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = remainder(a64[i], b64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (!isnan(expected) && expected != out64[i])) {
                        printf("remainder edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("remainder(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = remainderf(a32[i], b32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (!isnan(expected) && expected != out32[i])) {
                        printf("remainder edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    return failures;
}

static int run_more_edge_cases(void) {
    int failures = 0;

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {5.5, -5.5, 5.5, -5.5};
        double b64[] = {2.0, 2.0, -2.0, -2.0};
        float a32[] = {5.5f, -5.5f, 5.5f, -5.5f};
        float b32[] = {2.0f, 2.0f, -2.0f, -2.0f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}, {"b", ME_FLOAT64, b64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}, {"b", ME_FLOAT32, b32}};
        const void *ptrs64[] = {a64, b64};
        const void *ptrs32[] = {a32, b32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("fmod(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = fmod(a64[i], b64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (!isnan(expected) && expected != out64[i])) {
                        printf("fmod edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("fmod(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = fmodf(a32[i], b32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (!isnan(expected) && expected != out32[i])) {
                        printf("fmod edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {INFINITY, 3.0, NAN, 0.0};
        double b64[] = {4.0, INFINITY, 2.0, NAN};
        float a32[] = {INFINITY, 3.0f, NAN, 0.0f};
        float b32[] = {4.0f, INFINITY, 2.0f, NAN};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}, {"b", ME_FLOAT64, b64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}, {"b", ME_FLOAT32, b32}};
        const void *ptrs64[] = {a64, b64};
        const void *ptrs32[] = {a32, b32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("hypot(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = hypot(a64[i], b64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (!isnan(expected) && expected != out64[i])) {
                        printf("hypot edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("hypot(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = hypotf(a32[i], b32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (!isnan(expected) && expected != out32[i])) {
                        printf("hypot edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[6] = {0};
        float out32[6] = {0};
        double a64[] = {0.5, 1.5, 2.5, -0.5, -1.5, -2.5};
        float a32[] = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("rint(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 6, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 6; i++) {
                    double expected = rint(a64[i]);
                    if (expected != out64[i]) {
                        printf("rint edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("rint(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 6, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 6; i++) {
                    float expected = rintf(a32[i]);
                    if (expected != out32[i]) {
                        printf("rint edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    return failures;
}

static int run_edge_overflow_cases(void) {
    int failures = 0;

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {1.0, -1.0, NAN, 2.0};
        double b64[] = {NAN, 2.0, 3.0, NAN};
        float a32[] = {1.0f, -1.0f, NAN, 2.0f};
        float b32[] = {NAN, 2.0f, 3.0f, NAN};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}, {"b", ME_FLOAT64, b64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}, {"b", ME_FLOAT32, b32}};
        const void *ptrs64[] = {a64, b64};
        const void *ptrs32[] = {a32, b32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("fdim(a, b)", vars64, 2, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 2, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = fdim(a64[i], b64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (!isnan(expected) && expected != out64[i])) {
                        printf("fdim NaN edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("fdim(a, b)", vars32, 2, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 2, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = fdimf(a32[i], b32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (!isnan(expected) && expected != out32[i])) {
                        printf("fdim NaN edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {1.0, 2.0, -3.0, NAN};
        float a32[] = {1.0f, 2.0f, -3.0f, NAN};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("lgamma(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = lgamma(a64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (!isnan(expected) && expected != out64[i])) {
                        printf("lgamma edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("lgamma(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = lgammaf(a32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (!isnan(expected) && expected != out32[i])) {
                        printf("lgamma edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[3] = {0};
        float out32[3] = {0};
        double a64[] = {1e3, -1e3, NAN};
        float a32[] = {100.0f, -100.0f, NAN};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("exp2(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 3, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 3; i++) {
                    double expected = exp2(a64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (!isnan(expected) && expected != out64[i])) {
                        printf("exp2 overflow edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("exp2(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 3, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 3; i++) {
                    float expected = exp2f(a32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (!isnan(expected) && expected != out32[i])) {
                        printf("exp2 overflow edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[3] = {0};
        float out32[3] = {0};
        double a64[] = {400.0, -400.0, NAN};
        float a32[] = {50.0f, -50.0f, NAN};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("exp10(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 3, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 3; i++) {
                    double expected = exp10_ref(a64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (!isnan(expected) && expected != out64[i])) {
                        printf("exp10 overflow edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("exp10(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 3, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 3; i++) {
                    float expected = exp10_ref_f(a32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (!isnan(expected) && expected != out32[i])) {
                        printf("exp10 overflow edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    return failures;
}

static int run_more_math_edges(void) {
    int failures = 0;

    {
        double out64[5] = {0};
        float out32[5] = {0};
        double a64[] = {0.0, 1.0, -1.0, 0.5, -0.5};
        float a32[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("sinpi(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 5, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 5; i++) {
                    double expected = sinpi_ref(a64[i]);
                    if (!nearly_equal(out64[i], expected, 1e-12)) {
                        printf("sinpi edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("sinpi(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 5, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 5; i++) {
                    float expected = sinpi_ref_f(a32[i]);
                    if (!nearly_equal_f(out32[i], expected, 1e-5f)) {
                        printf("sinpi edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[5] = {0};
        float out32[5] = {0};
        double a64[] = {0.0, 1.0, -1.0, 0.5, -0.5};
        float a32[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("cospi(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 5, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 5; i++) {
                    double expected = cospi_ref(a64[i]);
                    if (!nearly_equal(out64[i], expected, 1e-12)) {
                        printf("cospi edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("cospi(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 5, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 5; i++) {
                    float expected = cospi_ref_f(a32[i]);
                    if (!nearly_equal_f(out32[i], expected, 1e-5f)) {
                        printf("cospi edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {0.0, 1.0, 5.0, 10.0};
        float a32[] = {0.0f, 1.0f, 5.0f, 10.0f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("erf(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = erf(a64[i]);
                    if (!nearly_equal(out64[i], expected, 1e-12)) {
                        printf("erf edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("erf(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = erff(a32[i]);
                    if (!nearly_equal_f(out32[i], expected, 1e-5f)) {
                        printf("erf edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {0.0, 1.0, 5.0, 10.0};
        float a32[] = {0.0f, 1.0f, 5.0f, 10.0f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("erfc(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = erfc(a64[i]);
                    if (!nearly_equal(out64[i], expected, 1e-12)) {
                        printf("erfc edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("erfc(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = erfcf(a32[i]);
                    if (!nearly_equal_f(out32[i], expected, 1e-5f)) {
                        printf("erfc edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {1.0, 2.0, 0.5, -0.5};
        float a32[] = {1.0f, 2.0f, 0.5f, -0.5f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("tgamma(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = tgamma(a64[i]);
                    if (!nearly_equal(out64[i], expected, 1e-12)) {
                        printf("tgamma edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("tgamma(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = tgammaf(a32[i]);
                    if (!nearly_equal_f(out32[i], expected, 1e-5f)) {
                        printf("tgamma edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    {
        double out64[4] = {0};
        float out32[4] = {0};
        double a64[] = {-2.0, -1.0, 0.0, 1.0};
        float a32[] = {-2.0f, -1.0f, 0.0f, 1.0f};

        me_variable vars64[] = {{"a", ME_FLOAT64, a64}};
        me_variable vars32[] = {{"a", ME_FLOAT32, a32}};
        const void *ptrs64[] = {a64};
        const void *ptrs32[] = {a32};
        me_expr *expr = NULL;
        int err = 0;

        if (me_compile("tgamma(a)", vars64, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs64, 1, out64, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    double expected = tgamma(a64[i]);
                    if ((isnan(expected) && !isnan(out64[i])) ||
                        (isinf(expected) && !isinf(out64[i])) ||
                        (!isnan(expected) && !isinf(expected) && expected != out64[i])) {
                        printf("tgamma pole edge case failed (f64)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }

        expr = NULL;
        if (me_compile("tgamma(a)", vars32, 1, ME_FLOAT32, &err, &expr) == ME_COMPILE_SUCCESS) {
            if (me_eval(expr, ptrs32, 1, out32, 4, NULL) == ME_EVAL_SUCCESS) {
                for (int i = 0; i < 4; i++) {
                    float expected = tgammaf(a32[i]);
                    if ((isnan(expected) && !isnan(out32[i])) ||
                        (isinf(expected) && !isinf(out32[i])) ||
                        (!isnan(expected) && !isinf(expected) && expected != out32[i])) {
                        printf("tgamma pole edge case failed (f32)\n");
                        failures++;
                        break;
                    }
                }
            } else {
                failures++;
            }
            me_free(expr);
        } else {
            failures++;
        }
    }

    return failures;
}

static int test_simd_init(void) {
    double data[] = {0.1, 0.2, 0.3, 0.4};
    double out[4] = {0};
    const void *vars[] = {data};
    me_variable v[] = {{"x", ME_FLOAT64, data}};
    me_expr *expr = NULL;
    int err = 0;

    me_simd_reset_for_tests();
    if (me_simd_initialized_for_tests() != 0) {
        printf("SIMD init state should be 0 before eval\n");
        return 1;
    }

    if (me_compile("sin(x) + cos(x)", v, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile simd init test (err=%d)\n", err);
        return 1;
    }

    if (me_eval(expr, vars, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("me_eval failed in simd init test\n");
        me_free(expr);
        return 1;
    }

    if (me_simd_initialized_for_tests() == 0) {
        printf("SIMD init state should be 1 after eval\n");
        me_free(expr);
        return 1;
    }

    me_free(expr);
    return 0;
}

int main(void) {
    int failures = 0;
    const int n = 1024;

    failures += test_simd_init();
    failures += run_unary_pair_f64("abs", fabs, n, -10.0, 10.0, 1e-12);
    failures += run_unary_pair_f64("exp", exp, n, -5.0, 5.0, 1e-12);
    failures += run_unary_pair_f64("expm1", expm1, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("exp2", exp2, n, -5.0, 5.0, 1e-12);
    failures += run_unary_pair_f64("exp10", exp10_ref, n, -2.0, 2.0, 1e-12);
    failures += run_unary_pair_f64("log", log, n, 0.1, 10.0, 1e-12);
    failures += run_unary_pair_f64("log10", log10, n, 0.1, 10.0, 1e-12);
    failures += run_unary_pair_f64("log1p", log1p, n, -0.9, 10.0, 1e-12);
    failures += run_unary_pair_f64("log2", log2, n, 0.1, 10.0, 1e-12);
    failures += run_unary_pair_f64("sqrt", sqrt, n, 0.0, 100.0, 1e-12);
    failures += run_unary_pair_f64("cbrt", cbrt, n, -10.0, 10.0, 1e-12);
    failures += run_unary_pair_f64("erf", erf, n, -2.0, 2.0, 1e-12);
    failures += run_unary_pair_f64("erfc", erfc, n, -2.0, 2.0, 1e-12);
    failures += run_unary_pair_f64("sinpi", sinpi_ref, n, -2.0, 2.0, 1e-12);
    failures += run_unary_pair_f64("cospi", cospi_ref, n, -2.0, 2.0, 1e-12);
    failures += run_unary_pair_f64("sinh", sinh, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("cosh", cosh, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("tanh", tanh, n, -3.0, 3.0, 1e-12);
    failures += run_unary_pair_f64("tan", tan, n, -1.0, 1.0, 1e-12);
    failures += run_unary_pair_f64("asin", asin, n, -0.9, 0.9, 1e-12);
    failures += run_unary_pair_f64("acos", acos, n, -0.9, 0.9, 1e-12);
    failures += run_unary_pair_f64("atan", atan, n, -5.0, 5.0, 1e-12);
    failures += run_unary_pair_f64("acosh", acosh, n, 1.0, 10.0, 1e-12);
    failures += run_unary_pair_f64("asinh", asinh, n, -5.0, 5.0, 1e-12);
    failures += run_unary_pair_f64("atanh", atanh, n, -0.9, 0.9, 1e-12);
    failures += run_unary_pair_f64("tgamma", tgamma, n, 0.5, 5.0, 1e-12);
    failures += run_unary_pair_f64("lgamma", lgamma, n, 0.5, 5.0, 1e-12);
    failures += run_unary_pair_f64("rint", rint, n, -5.0, 5.0, 1e-12);
    failures += run_unary_pair_f64("ceil", ceil, n, -3.5, 3.5, 1e-12);
    failures += run_unary_pair_f64("floor", floor, n, -3.5, 3.5, 1e-12);
    failures += run_unary_pair_f64("round", round, n, -3.5, 3.5, 1e-12);
    failures += run_unary_pair_f64("trunc", trunc, n, -3.5, 3.5, 1e-12);

    failures += run_binary_pair_f64("pow", pow, n, 0.1, 4.0, -2.0, 2.0, 1e-11);
    failures += run_binary_pair_f64("atan2", atan2, n, -3.0, 3.0, -3.0, 3.0, 1e-12);
    failures += run_binary_pair_f64("copysign", copysign, n, -5.0, 5.0, -5.0, 5.0, 0.0);
    failures += run_binary_pair_f64("fdim", fdim, n, -5.0, 5.0, -5.0, 5.0, 1e-12);
    failures += run_binary_pair_f64("fmax", fmax, n, -5.0, 5.0, -5.0, 5.0, 0.0);
    failures += run_binary_pair_f64("fmin", fmin, n, -5.0, 5.0, -5.0, 5.0, 0.0);
    failures += run_binary_pair_f64("fmod", fmod, n, -5.0, 5.0, 0.5, 5.0, 1e-12);
    failures += run_binary_pair_f64("hypot", hypot, n, -3.0, 3.0, -3.0, 3.0, 1e-12);
    failures += run_binary_pair_f64("ldexp", ldexp_ref, n, -5.0, 5.0, -4.0, 4.0, 1e-12);
    failures += run_binary_pair_f64("nextafter", nextafter, n, -2.0, 2.0, -2.0, 2.0, 0.0);
    failures += run_binary_pair_f64("remainder", remainder, n, -5.0, 5.0, 0.5, 5.0, 1e-12);
    failures += run_ternary_pair_f64("fma", fma, n, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, 1e-11);

    failures += run_unary_pair_f32("abs", fabsf, n, -10.0f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("exp", expf, n, -5.0f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("expm1", expm1f, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("exp2", exp2f, n, -5.0f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("exp10", exp10_ref_f, n, -2.0f, 2.0f, 1e-5f);
    failures += run_unary_pair_f32("log", logf, n, 0.1f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("log10", log10f, n, 0.1f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("log1p", log1pf, n, -0.9f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("log2", log2f, n, 0.1f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("sqrt", sqrtf, n, 0.0f, 100.0f, 1e-5f);
    failures += run_unary_pair_f32("cbrt", cbrtf, n, -10.0f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("erf", erff, n, -2.0f, 2.0f, 1e-5f);
    failures += run_unary_pair_f32("erfc", erfcf, n, -2.0f, 2.0f, 1e-5f);
    failures += run_unary_pair_f32("sinpi", sinpi_ref_f, n, -2.0f, 2.0f, 1e-5f);
    failures += run_unary_pair_f32("cospi", cospi_ref_f, n, -2.0f, 2.0f, 1e-5f);
    failures += run_unary_pair_f32("sinh", sinhf, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("cosh", coshf, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("tanh", tanhf, n, -3.0f, 3.0f, 1e-5f);
    failures += run_unary_pair_f32("tan", tanf, n, -1.0f, 1.0f, 1e-5f);
    failures += run_unary_pair_f32("asin", asinf, n, -0.9f, 0.9f, 1e-5f);
    failures += run_unary_pair_f32("acos", acosf, n, -0.9f, 0.9f, 1e-5f);
    failures += run_unary_pair_f32("atan", atanf, n, -5.0f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("acosh", acoshf, n, 1.0f, 10.0f, 1e-5f);
    failures += run_unary_pair_f32("asinh", asinhf, n, -5.0f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("atanh", atanhf, n, -0.9f, 0.9f, 1e-5f);
    failures += run_unary_pair_f32("tgamma", tgammaf, n, 0.5f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("lgamma", lgammaf, n, 0.5f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("rint", rintf, n, -5.0f, 5.0f, 1e-5f);
    failures += run_unary_pair_f32("ceil", ceilf, n, -3.5f, 3.5f, 1e-5f);
    failures += run_unary_pair_f32("floor", floorf, n, -3.5f, 3.5f, 1e-5f);
    failures += run_unary_pair_f32("round", roundf, n, -3.5f, 3.5f, 1e-5f);
    failures += run_unary_pair_f32("trunc", truncf, n, -3.5f, 3.5f, 1e-5f);

    failures += run_binary_pair_f32("pow", powf, n, 0.1f, 4.0f, -2.0f, 2.0f, 1e-5f);
    failures += run_binary_pair_f32("atan2", atan2f, n, -3.0f, 3.0f, -3.0f, 3.0f, 1e-5f);
    failures += run_binary_pair_f32("copysign", copysignf, n, -5.0f, 5.0f, -5.0f, 5.0f, 0.0f);
    failures += run_binary_pair_f32("fdim", fdimf, n, -5.0f, 5.0f, -5.0f, 5.0f, 1e-5f);
    failures += run_binary_pair_f32("fmax", fmaxf, n, -5.0f, 5.0f, -5.0f, 5.0f, 0.0f);
    failures += run_binary_pair_f32("fmin", fminf, n, -5.0f, 5.0f, -5.0f, 5.0f, 0.0f);
    failures += run_binary_pair_f32("fmod", fmodf, n, -5.0f, 5.0f, 0.5f, 5.0f, 1e-5f);
    failures += run_binary_pair_f32("hypot", hypotf, n, -3.0f, 3.0f, -3.0f, 3.0f, 1e-5f);
    failures += run_binary_pair_f32("ldexp", ldexp_ref_f, n, -5.0f, 5.0f, -4.0f, 4.0f, 1e-5f);
    failures += run_binary_pair_f32("nextafter", nextafterf, n, -2.0f, 2.0f, -2.0f, 2.0f, 0.0f);
    failures += run_binary_pair_f32("remainder", remainderf, n, -5.0f, 5.0f, 0.5f, 5.0f, 1e-5f);
    failures += run_ternary_pair_f32("fma", fmaf, n, -5.0f, 5.0f, -5.0f, 5.0f, -5.0f, 5.0f, 1e-5f);

    {
        double a64[16];
        float a32[16];
        double out64[16];
        float out32[16];
        for (int i = 0; i < 16; i++) {
            a64[i] = -4.0 + 0.5 * i;
            a32[i] = (float)a64[i];
        }

        failures += run_binary_const_f64("ldexp(a, 3.7)", ldexp_ref, a64, 3.7, out64, 16, false, 1e-12);
        failures += run_binary_const_f64("ldexp(a, 3.7)", ldexp_ref, a64, 3.7, out64, 16, true, 1e-12);
        failures += run_binary_const_f32("ldexp(a, 3.7)", ldexp_ref_f, a32, 3.7f, out32, 16, false, 1e-5f);
        failures += run_binary_const_f32("ldexp(a, 3.7)", ldexp_ref_f, a32, 3.7f, out32, 16, true, 1e-5f);

        failures += run_ternary_const_f64("fma(a, 2.5, -1.25)", fma, a64, 2.5, -1.25, out64, 16, false, 1e-12);
        failures += run_ternary_const_f64("fma(a, 2.5, -1.25)", fma, a64, 2.5, -1.25, out64, 16, true, 1e-12);
        failures += run_ternary_const_f32("fma(a, 2.5, -1.25)", fmaf, a32, 2.5f, -1.25f, out32, 16, false, 1e-5f);
        failures += run_ternary_const_f32("fma(a, 2.5, -1.25)", fmaf, a32, 2.5f, -1.25f, out32, 16, true, 1e-5f);
    }

    failures += run_nan_edge_cases();
    failures += run_additional_edge_cases();
    failures += run_more_edge_cases();
    failures += run_edge_overflow_cases();
    failures += run_more_math_edges();

    return failures ? 1 : 0;
}
