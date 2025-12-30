/*
 * Tests for sum() and prod() reductions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include "../src/miniexpr.h"
#include "me_complex.h"

#define MAKE_C64(real, imag) ME_C64_BUILD((real), (imag))
#define CREALF(z) ME_CREALF((z))
#define CIMAGF(z) ME_CIMAGF((z))

static int test_sum_int64() {
    printf("\n=== sum(int32) -> int64 ===\n");

    int32_t data[] = {1, 2, 3, 4};
    int64_t output = 0;

    me_variable vars[] = {{"x", ME_INT32, data}};
    int err = 0;
    me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
    if (!expr) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    if (me_get_dtype(expr) != ME_INT64) {
        printf("  ❌ FAILED: expected dtype ME_INT64, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    me_eval(expr, var_ptrs, 1, &output, 4);

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
    me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
    if (!expr) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    if (me_get_dtype(expr) != ME_UINT64) {
        printf("  ❌ FAILED: expected dtype ME_UINT64, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    me_eval(expr, var_ptrs, 1, &output, 4);

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
    me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
    if (!expr) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    if (me_get_dtype(expr) != ME_FLOAT32) {
        printf("  ❌ FAILED: expected dtype ME_FLOAT32, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    me_eval(expr, var_ptrs, 1, &output, 3);

    if (fabsf(output - 6.0f) > 1e-6f) {
        printf("  ❌ FAILED: expected 6, got %.6f\n", output);
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
    me_expr *expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err);
    if (!expr) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 1;
    }

    if (me_get_dtype(expr) != ME_COMPLEX64) {
        printf("  ❌ FAILED: expected dtype ME_COMPLEX64, got %d\n", me_get_dtype(expr));
        me_free(expr);
        return 1;
    }

    const void *var_ptrs[] = {data};
    me_eval(expr, var_ptrs, 1, &output, 2);

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

static int test_reduction_errors() {
    printf("\n=== Reduction validation errors ===\n");

    double data[] = {1.0, 2.0};
    me_variable vars[] = {{"x", ME_FLOAT64, data}};
    int err = 0;

    me_expr *expr = me_compile("sum(x + 1)", vars, 1, ME_AUTO, &err);
    if (expr) {
        printf("  ❌ FAILED: expected sum(x + 1) to be rejected\n");
        me_free(expr);
        return 1;
    }

    expr = me_compile("x + sum(x)", vars, 1, ME_AUTO, &err);
    if (expr) {
        printf("  ❌ FAILED: expected x + sum(x) to be rejected\n");
        me_free(expr);
        return 1;
    }

    expr = me_compile("sum(x, x)", vars, 1, ME_AUTO, &err);
    if (expr) {
        printf("  ❌ FAILED: expected sum(x, x) to be rejected\n");
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
        me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: sum(int32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {i32_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
        if (output != 0) {
            printf("  ❌ FAILED: sum(int32) empty expected 0, got %lld\n", (long long)output);
            failures++;
        }
        me_free(expr);
    }

    {
        int err = 0;
        int64_t output = -1;
        me_variable vars[] = {{"x", ME_INT32, i32_data}};
        me_expr *expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: prod(int32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {i32_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
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
        me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: sum(uint32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {u32_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
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
        me_expr *expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: prod(uint32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {u32_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
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
        me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: sum(float32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {f32_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
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
        me_expr *expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: prod(float32) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {f32_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
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
        me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: sum(complex64) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {c64_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
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
        me_expr *expr = me_compile("prod(x)", vars, 1, ME_AUTO, &err);
        if (!expr) {
            printf("  ❌ FAILED: prod(complex64) compile error %d\n", err);
            return 1;
        }
        const void *var_ptrs[] = {c64_data};
        me_eval(expr, var_ptrs, 1, &output, 0);
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
    failures += test_prod_complex64();
    failures += test_reduction_errors();
    failures += test_empty_inputs();

    if (failures == 0) {
        printf("\n✅ All reduction tests passed!\n");
    } else {
        printf("\n❌ Reduction tests failed: %d\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
