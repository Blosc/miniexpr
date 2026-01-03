/*
 * Tests for sum(), prod(), min(), and max() reductions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <complex.h>
#include "../src/miniexpr.h"
#include "minctest.h"



#if defined(_MSC_VER)
static inline float _Complex make_c64(float real, float imag) {
    union { float _Complex c; _Fcomplex m; } u;
    u.m = _FCbuild(real, imag);
    return u.c;
}
static inline float crealf_compat(float _Complex z) {
    union { float _Complex c; _Fcomplex m; } u;
    u.c = z;
    return u.m._Val[0];
}
static inline float cimagf_compat(float _Complex z) {
    union { float _Complex c; _Fcomplex m; } u;
    u.c = z;
    return u.m._Val[1];
}
#define MAKE_C64(real, imag) make_c64((real), (imag))
#define CREALF(z) crealf_compat((z))
#define CIMAGF(z) cimagf_compat((z))
#else
#define MAKE_C64(real, imag) CMPLXF((real), (imag))
#define CREALF(z) crealf(z)
#define CIMAGF(z) cimagf(z)
#endif

static int test_sum_int64() {
    printf("\n=== sum(int32) -> int64 ===\n");

    int32_t data[] = {1, 2, 3, 4};
    int64_t output = 0;

    me_variable vars[] = {{"x", ME_INT32, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    if (me_get_dtype(expr) != ME_INT64) {
        printf("  ❌ FAILED: expected dtype ME_INT64, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 4);

    if (output != 10) {
        printf("  ❌ FAILED: expected 10, got %lld\n", (long long)output);
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_sum_uint64() {
    printf("\n=== sum(uint32) -> uint64 ===\n");

    uint32_t data[] = {1, 2, 3, 4};
    uint64_t output = 0;

    me_variable vars[] = {{"x", ME_UINT32, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    if (me_get_dtype(expr) != ME_UINT64) {
        printf("  ❌ FAILED: expected dtype ME_UINT64, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 4);

    if (output != 10) {
        printf("  ❌ FAILED: expected 10, got %llu\n", (unsigned long long)output);
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_sum_float32() {
    printf("\n=== sum(float32) -> float32 ===\n");

    float data[] = {1.0f, 2.0f, 3.0f};
    float output = 0.0f;

    me_variable vars[] = {{"x", ME_FLOAT32, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    if (me_get_dtype(expr) != ME_FLOAT32) {
        printf("  ❌ FAILED: expected dtype ME_FLOAT32, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 3);

    if (fabsf(output - 6.0f) > 1e-6f) {
        printf("  ❌ FAILED: expected 6, got %.6f\n", output);
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_sum_single_output_chunk() {
    printf("\n=== sum(int32) output chunk size 1 ===\n");

    int32_t data[] = {1, 2, 3, 4};
    struct {
        int64_t output;
        int64_t guard;
    } buffer = {0, 0x1122334455667788LL};

    me_variable vars[] = {{"x", ME_INT32, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &buffer.output, 4);

    if (buffer.output != 10) {
        printf("  ❌ FAILED: expected 10, got %lld\n", (long long)buffer.output);
        me_free(expr);
        return 1;
    }

    if (buffer.guard != 0x1122334455667788LL) {
        printf("  ❌ FAILED: output chunk wrote past single element\n");
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_prod_complex64() {
    printf("\n=== prod(complex64) -> complex64 ===\n");

    float _Complex data[] = {MAKE_C64(1.0f, 2.0f), MAKE_C64(3.0f, -1.0f)};
    float _Complex output = MAKE_C64(0.0f, 0.0f);

    me_variable vars[] = {{"x", ME_COMPLEX64, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
#if defined(_WIN32) || defined(_WIN64)
        printf("  ✅ PASSED (complex not supported on Windows)\n");
        return 0;
#else
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
#endif
    }

    if (me_get_dtype(expr) != ME_COMPLEX64) {
        printf("  ❌ FAILED: expected dtype ME_COMPLEX64, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 2);

    float _Complex expected = MAKE_C64(1.0f, 2.0f) * MAKE_C64(3.0f, -1.0f);
    if (fabsf(CREALF(output) - CREALF(expected)) > 1e-6f ||
        fabsf(CIMAGF(output) - CIMAGF(expected)) > 1e-6f) {
        printf("  ❌ FAILED: expected (%.6f, %.6f), got (%.6f, %.6f)\n",
               CREALF(expected), CIMAGF(expected), CREALF(output), CIMAGF(output));
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_min_max_int32() {
    printf("\n=== min/max(int32) -> int32 ===\n");

    int32_t data[] = {3, 1, 4, 2};
    int32_t output = 0;

    me_variable vars[] = {{"x", ME_INT32, data}};
    int err = 0;
    me_expr *expr = NULL;

    int rc_expr = me_compile("min(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_INT32) {
        printf("  ❌ FAILED: expected dtype ME_INT32, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 4);
    if (output != 1) {
        printf("  ❌ FAILED: min expected 1, got %d\n", output);
        me_free(expr);
        return 1;
    }
    me_free(expr);

    output = 0;
    rc_expr = me_compile("max(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_INT32) {
        printf("  ❌ FAILED: expected dtype ME_INT32, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 4);
    if (output != 4) {
        printf("  ❌ FAILED: max expected 4, got %d\n", output);
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_min_max_float32() {
    printf("\n=== min/max(float32) -> float32 ===\n");

    float data[] = {3.5f, -1.0f, 2.0f};
    float output = 0.0f;

    me_variable vars[] = {{"x", ME_FLOAT32, data}};
    int err = 0;
    me_expr *expr = NULL;

    int rc_expr = me_compile("min(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_FLOAT32) {
        printf("  ❌ FAILED: expected dtype ME_FLOAT32, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 3);
    if (fabsf(output - (-1.0f)) > 1e-6f) {
        printf("  ❌ FAILED: min expected -1.0, got %.6f\n", output);
        me_free(expr);
        return 1;
    }
    me_free(expr);

    output = 0.0f;
    rc_expr = me_compile("max(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_FLOAT32) {
        printf("  ❌ FAILED: expected dtype ME_FLOAT32, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 3);
    if (fabsf(output - 3.5f) > 1e-6f) {
        printf("  ❌ FAILED: max expected 3.5, got %.6f\n", output);
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_min_max_float32_nan() {
    printf("\n=== min/max(float32) NaN ===\n");

    float data[] = {1.0f, NAN, 2.0f};
    float output = 0.0f;

    me_variable vars[] = {{"x", ME_FLOAT32, data}};
    int err = 0;
    me_expr *expr = NULL;

    int rc_expr = me_compile("min(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    const void *var_ptrs[] = {data};
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 3);
    if (!isnan(output)) {
        printf("  ❌ FAILED: min expected NaN, got %.6f\n", output);
        me_free(expr);
        return 1;
    }
    me_free(expr);

    output = 0.0f;
    rc_expr = me_compile("max(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 3);
    if (!isnan(output)) {
        printf("  ❌ FAILED: max expected NaN, got %.6f\n", output);
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_any_all_bool() {
    printf("\n=== any/all(bool) -> bool ===\n");

    bool data_any[] = {false, false, true};
    bool data_all[] = {true, true, true};
    bool output = false;

    me_variable vars_any[] = {{"x", ME_BOOL, data_any}};
    me_variable vars_all[] = {{"x", ME_BOOL, data_all}};
    int err = 0;
    me_expr *expr = NULL;

    int rc_expr = me_compile("any(x)", vars_any, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_BOOL) {
        printf("  ❌ FAILED: expected dtype ME_BOOL, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    const void *var_ptrs_any[] = {data_any};
    ME_EVAL_CHECK(expr, var_ptrs_any, 1, &output, 3);
    if (!output) {
        printf("  ❌ FAILED: any expected true, got false\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    output = false;
    rc_expr = me_compile("all(x)", vars_all, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_BOOL) {
        printf("  ❌ FAILED: expected dtype ME_BOOL, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    const void *var_ptrs_all[] = {data_all};
    ME_EVAL_CHECK(expr, var_ptrs_all, 1, &output, 3);
    if (!output) {
        printf("  ❌ FAILED: all expected true, got false\n");
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_any_all_int32() {
    printf("\n=== any/all(int32) -> bool ===\n");

    int32_t data_any[] = {0, 0, 5};
    int32_t data_all[] = {1, 2, 3};
    bool output = false;

    me_variable vars_any[] = {{"x", ME_INT32, data_any}};
    me_variable vars_all[] = {{"x", ME_INT32, data_all}};
    int err = 0;
    me_expr *expr = NULL;

    int rc_expr = me_compile("any(x)", vars_any, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_BOOL) {
        printf("  ❌ FAILED: expected dtype ME_BOOL, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    const void *var_ptrs_any[] = {data_any};
    ME_EVAL_CHECK(expr, var_ptrs_any, 1, &output, 3);
    if (!output) {
        printf("  ❌ FAILED: any expected true, got false\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    output = false;
    rc_expr = me_compile("all(x)", vars_all, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    if (me_get_dtype(expr) != ME_BOOL) {
        printf("  ❌ FAILED: expected dtype ME_BOOL, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }
    const void *var_ptrs_all[] = {data_all};
    ME_EVAL_CHECK(expr, var_ptrs_all, 1, &output, 3);
    if (!output) {
        printf("  ❌ FAILED: all expected true, got false\n");
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_reduction_expression_args() {
    printf("\n=== Reduction expressions ===\n");

    double data[] = {1.0, 2.0, 3.0};
    me_variable vars[] = {{"x", ME_FLOAT64, data}};
    const void *var_ptrs[] = {data};
    int err = 0;
    me_expr *expr = NULL;

    double sum_out = 0.0;
    int rc_expr = me_compile("sum(x + 1)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    ME_EVAL_CHECK(expr, var_ptrs, 1, &sum_out, 3);
    if (fabs(sum_out - 9.0) > 1e-12) {
        printf("  ❌ FAILED: expected sum(x + 1) = 9, got %.6f\n", sum_out);
        me_free(expr);
        return 1;
    }
    me_free(expr);

    double output[3] = {0.0, 0.0, 0.0};
    rc_expr = me_compile("x + sum(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    ME_EVAL_CHECK(expr, var_ptrs, 1, output, 3);

    double expected_sum = 6.0;
    for (int i = 0; i < 3; i++) {
        double expected = data[i] + expected_sum;
        if (fabs(output[i] - expected) > 1e-12) {
            printf("  ❌ FAILED: expected %.6f, got %.6f\n", expected, output[i]);
            me_free(expr);
            return 1;
        }
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_reduction_expression_multi_vars() {
    printf("\n=== Reduction expressions (multi-var) ===\n");

    int32_t x[] = {1, 2, 3};
    double y[] = {4.5, 5.5, 6.5};
    me_variable vars[] = {{"x", ME_INT32, x}, {"y", ME_FLOAT64, y}};
    const void *var_ptrs[] = {x, y};
    int err = 0;
    me_expr *expr = NULL;

    double sum_out = 0.0;
    int rc_expr = me_compile("sum(x + y + 2.5)", vars, 2, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    ME_EVAL_CHECK(expr, var_ptrs, 2, &sum_out, 3);
    if (fabs(sum_out - 30.0) > 1e-12) {
        printf("  ❌ FAILED: expected sum(x + y + 2.5) = 30, got %.6f\n", sum_out);
        me_free(expr);
        return 1;
    }
    me_free(expr);

    double output[3] = {0.0, 0.0, 0.0};
    rc_expr = me_compile("x + sum(x + y + 2.5) + 1.5", vars, 2, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }
    ME_EVAL_CHECK(expr, var_ptrs, 2, output, 3);

    double expected_sum = 30.0;
    for (int i = 0; i < 3; i++) {
        double expected = (double)x[i] + expected_sum + 1.5;
        if (fabs(output[i] - expected) > 1e-12) {
            printf("  ❌ FAILED: expected %.6f, got %.6f\n", expected, output[i]);
            me_free(expr);
            return 1;
        }
    }

    printf("  ✅ PASSED\n");
    me_free(expr);
    return 0;
}

static int test_reduction_errors() {
    printf("\n=== Reduction validation errors ===\n");

    double data[] = {1.0, 2.0};
    me_variable vars[] = {{"x", ME_FLOAT64, data}};
    int err = 0;

    me_expr *expr = NULL;
    int rc_expr = me_compile("sum(sum(x))", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: expected sum(sum(x)) to be rejected\n");
        me_free(expr);
        return 1;
    }

    rc_expr = me_compile("sum(x + sum(x))", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr == ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: expected sum(x + sum(x)) to be rejected\n");
        me_free(expr);
        return 1;
    }

    printf("  ✅ PASSED\n");
    return 0;
}

static int test_empty_inputs() {
    printf("\n=== Reduction empty inputs ===\n");

    int failures = 0;

    int32_t i32_data[1] = {0};
    uint32_t u32_data[1] = {0};
    float f32_data[1] = {0.0f};
    float _Complex c64_data[1] = {MAKE_C64(0.0f, 0.0f)};

    {
        int err = 0;
        int64_t output = -1;
        me_variable vars[] = {{"x", ME_INT32, i32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: sum(int32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {i32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (output != 0) {
            printf("  ❌ FAILED: sum(int32) empty expected 0, got %lld\n", (long long)output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        int32_t output = -1;
        me_variable vars[] = {{"x", ME_INT32, i32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("min(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: min(int32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {i32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (output != INT32_MAX) {
            printf("  ❌ FAILED: min(int32) empty expected %d, got %d\n", INT32_MAX, output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        bool output = false;
        bool b_data[1] = {false};
        me_variable vars[] = {{"x", ME_BOOL, b_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("any(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: any(bool) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {b_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (output) {
            printf("  ❌ FAILED: any(bool) empty expected false, got true\n");
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        bool output = false;
        bool b_data[1] = {false};
        me_variable vars[] = {{"x", ME_BOOL, b_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("all(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: all(bool) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {b_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (!output) {
            printf("  ❌ FAILED: all(bool) empty expected true, got false\n");
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        int32_t output = -1;
        me_variable vars[] = {{"x", ME_INT32, i32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("max(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: max(int32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {i32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (output != INT32_MIN) {
            printf("  ❌ FAILED: max(int32) empty expected %d, got %d\n", INT32_MIN, output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        int64_t output = -1;
        me_variable vars[] = {{"x", ME_INT32, i32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: prod(int32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {i32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (output != 1) {
            printf("  ❌ FAILED: prod(int32) empty expected 1, got %lld\n", (long long)output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        uint64_t output = 0;
        me_variable vars[] = {{"x", ME_UINT32, u32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: sum(uint32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {u32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (output != 0) {
            printf("  ❌ FAILED: sum(uint32) empty expected 0, got %llu\n", (unsigned long long)output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        uint64_t output = 0;
        me_variable vars[] = {{"x", ME_UINT32, u32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: prod(uint32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {u32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (output != 1) {
            printf("  ❌ FAILED: prod(uint32) empty expected 1, got %llu\n", (unsigned long long)output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        float output = -1.0f;
        me_variable vars[] = {{"x", ME_FLOAT32, f32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: sum(float32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {f32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (fabsf(output - 0.0f) > 1e-6f) {
            printf("  ❌ FAILED: sum(float32) empty expected 0, got %.6f\n", output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        float output = -1.0f;
        me_variable vars[] = {{"x", ME_FLOAT32, f32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("min(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: min(float32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {f32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (!isinf(output) || output < 0.0f) {
            printf("  ❌ FAILED: min(float32) empty expected +inf, got %.6f\n", output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        float output = -1.0f;
        me_variable vars[] = {{"x", ME_FLOAT32, f32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("max(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: max(float32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {f32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (!isinf(output) || output > 0.0f) {
            printf("  ❌ FAILED: max(float32) empty expected -inf, got %.6f\n", output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        float output = -1.0f;
        me_variable vars[] = {{"x", ME_FLOAT32, f32_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  ❌ FAILED: prod(float32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {f32_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (fabsf(output - 1.0f) > 1e-6f) {
            printf("  ❌ FAILED: prod(float32) empty expected 1, got %.6f\n", output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        float _Complex output = MAKE_C64(-1.0f, 0.0f);
        me_variable vars[] = {{"x", ME_COMPLEX64, c64_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
#if defined(_WIN32) || defined(_WIN64)
            printf("  ✅ PASSED (complex not supported on Windows)\n");
            return 0;
#else
            printf("  ❌ FAILED: sum(complex64) compile error %d\n", err);
            return 1;
#endif
        }
        const void *var_ptrs[] = {c64_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (fabsf(CREALF(output)) > 1e-6f || fabsf(CIMAGF(output)) > 1e-6f) {
            printf("  ❌ FAILED: sum(complex64) empty expected 0, got (%.6f, %.6f)\n",
                   CREALF(output), CIMAGF(output));
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        float _Complex output = MAKE_C64(0.0f, 0.0f);
        me_variable vars[] = {{"x", ME_COMPLEX64, c64_data}};
        me_expr *expr = NULL;
        int rc_expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
#if defined(_WIN32) || defined(_WIN64)
            printf("  ✅ PASSED (complex not supported on Windows)\n");
            return 0;
#else
            printf("  ❌ FAILED: prod(complex64) compile error %d\n", err);
            return 1;
#endif
        }
        const void *var_ptrs[] = {c64_data};
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, 0);
        if (fabsf(CREALF(output) - 1.0f) > 1e-6f || fabsf(CIMAGF(output)) > 1e-6f) {
            printf("  ❌ FAILED: prod(complex64) empty expected 1, got (%.6f, %.6f)\n",
                   CREALF(output), CIMAGF(output));
            failures++;
        }
        me_free(expr);
    }

    if (failures == 0) {
        printf("  ✅ PASSED\n");
    }
    return failures == 0 ? 0 : 1;
}

int main(void) {
    int failures = 0;

    failures += test_sum_int64();
    failures += test_sum_uint64();
    failures += test_sum_float32();
    failures += test_sum_single_output_chunk();
    failures += test_prod_complex64();
    failures += test_min_max_int32();
    failures += test_min_max_float32();
    failures += test_min_max_float32_nan();
    failures += test_any_all_bool();
    failures += test_any_all_int32();
    failures += test_reduction_expression_args();
    failures += test_reduction_expression_multi_vars();
    failures += test_reduction_errors();
    failures += test_empty_inputs();

    if (failures == 0) {
        printf("\n✅ All reduction tests passed!\n");
    } else {
        printf("\n❌ Reduction tests failed: %d\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
