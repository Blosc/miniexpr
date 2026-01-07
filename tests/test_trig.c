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

static int run_trig_f64(const char *name, double (*func)(double), int n, bool simd_enabled) {
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
        if (!nearly_equal(out[i], expected, 1e-12)) {
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

static int run_trig_f32(const char *name, float (*func)(float), int n, bool simd_enabled) {
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
        if (!nearly_equal_f(out[i], expected, 1e-5f)) {
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
        {"tan", tan}
    };
    const trig_f32_t f32_tests[] = {
        {"sin", sinf},
        {"cos", cosf},
        {"tan", tanf}
    };

    printf("=== Testing trig functions ===\n");

    int rc = 0;
    rc |= test_identity(1024);

    for (size_t i = 0; i < sizeof(f64_tests) / sizeof(f64_tests[0]); i++) {
        rc |= run_trig_f64(f64_tests[i].name, f64_tests[i].func, 2048, true);
        rc |= run_trig_f64(f64_tests[i].name, f64_tests[i].func, 2048, false);
    }

    for (size_t i = 0; i < sizeof(f32_tests) / sizeof(f32_tests[0]); i++) {
        rc |= run_trig_f32(f32_tests[i].name, f32_tests[i].func, 2048, true);
        rc |= run_trig_f32(f32_tests[i].name, f32_tests[i].func, 2048, false);
    }

    if (rc == 0) {
        printf("PASS\n");
        return 0;
    }

    printf("FAIL\n");
    return 1;
}
