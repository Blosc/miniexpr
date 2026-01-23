#include <math.h>
#include <stdbool.h>
#include <stdint.h>
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
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, NULL);
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

static int run_trig_f64(const char *name, double (*func)(double), int n,
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

static int run_trig_f32(const char *name, float (*func)(float), int n,
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

static int run_trig_i32_inverse(const char *name, double (*func)(double), int n,
                                int32_t value, double tol) {
    int32_t *input = (int32_t *)malloc((size_t)(2 * n) * sizeof(int32_t));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!input || !out) {
        printf("Allocation failed\n");
        free(input);
        free(out);
        return 1;
    }

    for (int i = 0; i < 2 * n; i++) {
        input[i] = value;
    }

    me_variable vars[] = {{"x", ME_INT32, input}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_text[32];

    snprintf(expr_text, sizeof(expr_text), "%s(x)", name);
    if (me_compile(expr_text, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s i32 (err=%d)\n", expr_text, err);
        free(input);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {input};
    int eval_rc = me_eval(expr, var_ptrs, 1, out, n, NULL);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("%s i32 eval failed (err=%d)\n", expr_text, eval_rc);
        me_free(expr);
        free(input);
        free(out);
        return 1;
    }

    int failures = 0;
    double expected = func((double)value);
    for (int i = 0; i < n; i++) {
        if (!nearly_equal(out[i], expected, tol)) {
            if (failures < 5) {
                printf("%s i32 mismatch at %d: got %.15f expected %.15f\n",
                       expr_text, i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(input);
    free(out);
    if (failures) {
        printf("%s i32 FAIL: %d mismatches\n", expr_text, failures);
        return 1;
    }

    printf("%s i32 PASS\n", expr_text);
    return 0;
}

static int run_atan2_f64(int n, bool simd_enabled) {
    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!x || !y || !out) {
        printf("Allocation failed\n");
        free(x);
        free(y);
        free(out);
        return 1;
    }

    fill_input_range_f64(x, n, -1.0, 1.0);
    fill_input_range_f64(y, n, 1.0, -1.0);

    me_variable vars[] = {{"y", ME_FLOAT64, y}, {"x", ME_FLOAT64, x}};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile("atan2(y, x)", vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile atan2(y, x) (err=%d)\n", err);
        free(x);
        free(y);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {y, x};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 2, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("atan2(y, x) eval failed (err=%d)\n", eval_rc);
        me_free(expr);
        free(x);
        free(y);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        double expected = atan2(y[i], x[i]);
        if (!nearly_equal(out[i], expected, 1e-12)) {
            if (failures < 5) {
                printf("atan2(y, x) mismatch at %d: got %.15f expected %.15f\n",
                       i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(x);
    free(y);
    free(out);
    if (failures) {
        printf("atan2(y, x) f64 %s FAIL: %d mismatches\n",
               simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("atan2(y, x) f64 %s PASS\n", simd_enabled ? "SIMD" : "scalar");
    return 0;
}

static int run_atan2_f32(int n, bool simd_enabled) {
    float *x = (float *)malloc((size_t)n * sizeof(float));
    float *y = (float *)malloc((size_t)n * sizeof(float));
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!x || !y || !out) {
        printf("Allocation failed\n");
        free(x);
        free(y);
        free(out);
        return 1;
    }

    fill_input_range_f32(x, n, -1.0f, 1.0f);
    fill_input_range_f32(y, n, 1.0f, -1.0f);

    me_variable vars[] = {{"y", ME_FLOAT32, y}, {"x", ME_FLOAT32, x}};
    int err = 0;
    me_expr *expr = NULL;

    if (me_compile("atan2(y, x)", vars, 2, ME_FLOAT32, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile atan2(y, x) f32 (err=%d)\n", err);
        free(x);
        free(y);
        free(out);
        return 1;
    }

    const void *var_ptrs[] = {y, x};
    me_eval_params eval_params = ME_EVAL_PARAMS_DEFAULTS;
    eval_params.disable_simd = !simd_enabled;
    int eval_rc = me_eval(expr, var_ptrs, 2, out, n, &eval_params);
    if (eval_rc != ME_EVAL_SUCCESS) {
        printf("atan2(y, x) f32 eval failed (err=%d)\n", eval_rc);
        me_free(expr);
        free(x);
        free(y);
        free(out);
        return 1;
    }

    int failures = 0;
    for (int i = 0; i < n; i++) {
        float expected = atan2f(y[i], x[i]);
        if (!nearly_equal_f(out[i], expected, 1e-5f)) {
            if (failures < 5) {
                printf("atan2(y, x) f32 mismatch at %d: got %.7f expected %.7f\n",
                       i, out[i], expected);
            }
            failures++;
        }
    }

    me_free(expr);
    free(x);
    free(y);
    free(out);
    if (failures) {
        printf("atan2(y, x) f32 %s FAIL: %d mismatches\n",
               simd_enabled ? "SIMD" : "scalar", failures);
        return 1;
    }

    printf("atan2(y, x) f32 %s PASS\n", simd_enabled ? "SIMD" : "scalar");
    return 0;
}

int main(void) {
    typedef struct {
        const char *name;
        double (*func)(double);
    } trig_f64_t;
    typedef struct {
        const char *name;
        float (*func)(float);
    } trig_f32_t;

    const trig_f64_t f64_tests[] = {
        {"sin", sin},
        {"cos", cos},
        {"tan", tan},
        {"asin", asin},
        {"acos", acos},
        {"atan", atan}
    };
    const trig_f32_t f32_tests[] = {
        {"sin", sinf},
        {"cos", cosf},
        {"tan", tanf},
        {"asin", asinf},
        {"acos", acosf},
        {"atan", atanf}
    };

    printf("=== Testing trig functions ===\n");

    int rc = 0;
    rc |= test_identity(1024);
    rc |= run_trig_i32_inverse("arccos", acos, 1024, -1, 1e-12);
    rc |= run_trig_i32_inverse("arcsin", asin, 1024, -1, 1e-12);

    for (size_t i = 0; i < sizeof(f64_tests) / sizeof(f64_tests[0]); i++) {
        double min_val = (f64_tests[i].func == asin || f64_tests[i].func == acos) ? -1.0 : -0.9;
        double max_val = (f64_tests[i].func == asin || f64_tests[i].func == acos) ? 1.0 : 0.9;
        rc |= run_trig_f64(f64_tests[i].name, f64_tests[i].func, 2048, true, min_val, max_val, 1e-12);
        rc |= run_trig_f64(f64_tests[i].name, f64_tests[i].func, 2048, false, min_val, max_val, 1e-12);
    }

    for (size_t i = 0; i < sizeof(f32_tests) / sizeof(f32_tests[0]); i++) {
        float min_val = (f32_tests[i].func == asinf || f32_tests[i].func == acosf) ? -1.0f : -0.9f;
        float max_val = (f32_tests[i].func == asinf || f32_tests[i].func == acosf) ? 1.0f : 0.9f;
        rc |= run_trig_f32(f32_tests[i].name, f32_tests[i].func, 2048, true, min_val, max_val, 1e-5f);
        rc |= run_trig_f32(f32_tests[i].name, f32_tests[i].func, 2048, false, min_val, max_val, 1e-5f);
    }

    rc |= run_atan2_f64(2048, true);
    rc |= run_atan2_f64(2048, false);
    rc |= run_atan2_f32(2048, true);
    rc |= run_atan2_f32(2048, false);

    if (rc == 0) {
        printf("PASS\n");
        return 0;
    }

    printf("FAIL\n");
    return 1;
}
