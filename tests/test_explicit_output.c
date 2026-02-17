/* Test explicit variable types with explicit output dtype
 *
 * This test verifies that when both variable types and output dtype
 * are explicitly specified, the behavior is correct:
 * - Variables keep their types during computation
 * - Result is cast to the specified output dtype
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include "../src/miniexpr.h"
#include "minctest.h"



#define VECTOR_SIZE 10

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("\nTesting: %s\n", name); \
    tests_run++;

#define ASSERT_EQ(expected, actual, idx, fmt) \
    if ((expected) != (actual)) { \
        printf("  FAIL at [%d]: expected " fmt ", got " fmt "\n", idx, expected, actual); \
        tests_failed++; \
        return; \
    }

void test_mixed_types_float32_output() {
    TEST("Mixed types (INT32 + FLOAT64) with FLOAT32 output");

    int32_t a[VECTOR_SIZE];
    double b[VECTOR_SIZE];
    float result[VECTOR_SIZE];
    float expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i * 10;
        b[i] = i * 0.5;
        // Expected: INT32 + FLOAT64 promotes to FLOAT64, then cast to FLOAT32
        expected[i] = (float)((double)a[i] + b[i]);
    }

    me_variable vars[] = {
        {"a", ME_INT32},
        {"b", ME_FLOAT64}
    };

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a + b", vars, 2, ME_FLOAT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_FLOAT32) {
        printf("  FAIL: output dtype should be ME_FLOAT32, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float diff = fabsf(result[i] - expected[i]);
        if (diff > 1e-5f) {
            printf("  FAIL at [%d]: expected %.6f, got %.6f (diff: %.6f)\n",
                   i, expected[i], result[i], diff);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match (within float32 tolerance)\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

void test_float32_vars_float64_output() {
    TEST("FLOAT32 variables with FLOAT64 output");

    float x[VECTOR_SIZE];
    float y[VECTOR_SIZE];
    double result[VECTOR_SIZE];
    double expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        x[i] = (float)(i * 1.5f);
        y[i] = (float)(i * 0.5f);
        // Expected: FLOAT32 + FLOAT32 = FLOAT32, then cast to FLOAT64
        expected[i] = (double)(x[i] + y[i]);
    }

    me_variable vars[] = {
        {"x", ME_FLOAT32},
        {"y", ME_FLOAT32}
    };

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_FLOAT64) {
        printf("  FAIL: output dtype should be ME_FLOAT64, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {x, y};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        double diff = fabs(result[i] - expected[i]);
        if (diff > 1e-9) {
            printf("  FAIL at [%d]: expected %.9f, got %.9f (diff: %.9f)\n",
                   i, expected[i], result[i], diff);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match (within float64 tolerance)\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

void test_float32_with_constant_float64_output() {
    TEST("FLOAT32 variable + constant with FLOAT64 output");

    float a[VECTOR_SIZE];
    double result[VECTOR_SIZE];
    double expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (float)i;
        // Expected: FLOAT32 + FLOAT32 constant = FLOAT32, then cast to FLOAT64
        // Constant 3.0 is typed as FLOAT32 (NumPy convention), so computation is FLOAT32
        expected[i] = (double)(a[i] + 3.0f);
    }

    me_variable vars[] = {{"a", ME_FLOAT32}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a + 3.0", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_FLOAT64) {
        printf("  FAIL: output dtype should be ME_FLOAT64, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        double diff = fabs(result[i] - expected[i]);
        if (diff > 1e-9) {
            printf("  FAIL at [%d]: expected %.9f, got %.9f (diff: %.9f)\n",
                   i, expected[i], result[i], diff);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match (within float64 tolerance)\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

void test_comparison_explicit_bool_output() {
    TEST("Comparison with explicit BOOL output");

    int32_t a[VECTOR_SIZE];
    int32_t b[VECTOR_SIZE];
    bool result[VECTOR_SIZE];
    bool expected[VECTOR_SIZE];

    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i * 2;
        b[i] = i * 2 + 1;
        expected[i] = (a[i] > b[i]);
    }

    me_variable vars[] = {
        {"a", ME_INT32},
        {"b", ME_INT32}
    };

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("a > b", vars, 2, ME_BOOL, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    if (output_dtype != ME_BOOL) {
        printf("  FAIL: output dtype should be ME_BOOL, got %d\n", output_dtype);
        tests_failed++;
        me_free(expr);
        return;
    }

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, var_ptrs, 2, result, VECTOR_SIZE);

    int passed = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (result[i] != expected[i]) {
            printf("  FAIL at [%d]: expected %d, got %d\n",
                   i, (int)expected[i], (int)result[i]);
            passed = 0;
        }
    }

    if (passed) {
        printf("  PASS: All %d values match\n", VECTOR_SIZE);
    } else {
        tests_failed++;
    }

    me_free(expr);
}

void test_integer_output_conversions() {
    TEST("Integer conversions with explicit output dtype");

    int passed = 1;
    int err;
    me_expr *expr = NULL;

    {
        int64_t x[VECTOR_SIZE] = {-1000000LL, -129LL, -1LL, 0LL, 1LL, 127LL, 128LL, 255LL, 32767LL, 1000000LL};
        int64_t y[VECTOR_SIZE] = {0};
        int32_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_INT64}, {"y", ME_INT64}};

        int rc_expr = me_compile("x + y", vars, 2, ME_INT32, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL int64->int32: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL int64->int32: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    int32_t expected = (int32_t)x[i];
                    if (out[i] != expected) {
                        printf("  FAIL int64->int32 at [%d]: expected %lld, got %lld\n",
                               i, (long long)expected, (long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        int32_t x[VECTOR_SIZE] = {-300, -1, 0, 1, 127, 128, 255, 256, 511, 1000};
        int32_t y[VECTOR_SIZE] = {0};
        uint8_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_INT32}, {"y", ME_INT32}};

        int rc_expr = me_compile("x + y", vars, 2, ME_UINT8, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL int32->uint8: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL int32->uint8: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    uint8_t expected = (uint8_t)x[i];
                    if (out[i] != expected) {
                        printf("  FAIL int32->uint8 at [%d]: expected %llu, got %llu\n",
                               i, (unsigned long long)expected, (unsigned long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        uint64_t x[VECTOR_SIZE] = {0ULL, 1ULL, 2ULL, 42ULL, 255ULL, 1024ULL, 2048ULL, 4096ULL, 12345ULL, 32767ULL};
        uint64_t y[VECTOR_SIZE] = {0};
        int16_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_UINT64}, {"y", ME_UINT64}};

        int rc_expr = me_compile("x + y", vars, 2, ME_INT16, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL uint64->int16: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL uint64->int16: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    int16_t expected = (int16_t)x[i];
                    if (out[i] != expected) {
                        printf("  FAIL uint64->int16 at [%d]: expected %lld, got %lld\n",
                               i, (long long)expected, (long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        int16_t x[VECTOR_SIZE] = {-32768, -1024, -1, 0, 1, 2, 127, 255, 1024, 32767};
        int16_t y[VECTOR_SIZE] = {0};
        uint32_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_INT16}, {"y", ME_INT16}};

        int rc_expr = me_compile("x + y", vars, 2, ME_UINT32, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL int16->uint32: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL int16->uint32: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    uint32_t expected = (uint32_t)x[i];
                    if (out[i] != expected) {
                        printf("  FAIL int16->uint32 at [%d]: expected %llu, got %llu\n",
                               i, (unsigned long long)expected, (unsigned long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
    }

    if (passed) {
        printf("  PASS: Integer conversion outputs match expected casts\n");
    } else {
        tests_failed++;
    }
}

void test_numeric_output_conversions() {
    TEST("Additional numeric conversions with explicit output dtype");

    int passed = 1;
    int err;
    me_expr *expr = NULL;

    {
        double x[VECTOR_SIZE] = {-3.9, -2.1, -1.0, 0.0, 1.2, 2.8, 42.0, 127.9, 128.1, 1000.4};
        double y[VECTOR_SIZE] = {0};
        int32_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};

        int rc_expr = me_compile("x + y", vars, 2, ME_INT32, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL float64->int32: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL float64->int32: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    int32_t expected = (int32_t)(x[i] + y[i]);
                    if (out[i] != expected) {
                        printf("  FAIL float64->int32 at [%d]: expected %lld, got %lld\n",
                               i, (long long)expected, (long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        float x[VECTOR_SIZE] = {0.0f, 0.9f, 1.1f, 2.9f, 127.5f, 128.5f, 255.9f, 256.1f, 1024.7f, 4095.9f};
        float y[VECTOR_SIZE] = {0};
        uint16_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_FLOAT32}, {"y", ME_FLOAT32}};

        int rc_expr = me_compile("x + y", vars, 2, ME_UINT16, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL float32->uint16: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL float32->uint16: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    uint16_t expected = (uint16_t)(x[i] + y[i]);
                    if (out[i] != expected) {
                        printf("  FAIL float32->uint16 at [%d]: expected %llu, got %llu\n",
                               i, (unsigned long long)expected, (unsigned long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        uint64_t x[VECTOR_SIZE] = {0ULL, 1ULL, 2ULL, 42ULL, 255ULL, 1024ULL, 65535ULL, 1048576ULL, 1234567ULL, 16777215ULL};
        uint64_t y[VECTOR_SIZE] = {0};
        float out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_UINT64}, {"y", ME_UINT64}};

        int rc_expr = me_compile("x + y", vars, 2, ME_FLOAT32, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL uint64->float32: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL uint64->float32: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    float expected = (float)(x[i] + y[i]);
                    if (fabsf(out[i] - expected) > 1e-6f) {
                        printf("  FAIL uint64->float32 at [%d]: expected %.9g, got %.9g\n",
                               i, expected, out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        uint32_t x[VECTOR_SIZE] = {0U, 1U, 2U, 42U, 255U, 1024U, 65535U, 1000000U, 1234567U, 16777215U};
        uint32_t y[VECTOR_SIZE] = {0};
        float out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_UINT32}, {"y", ME_UINT32}};

        int rc_expr = me_compile("x + y", vars, 2, ME_FLOAT32, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL uint32->float32: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL uint32->float32: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    float expected = (float)(x[i] + y[i]);
                    if (fabsf(out[i] - expected) > 1e-6f) {
                        printf("  FAIL uint32->float32 at [%d]: expected %.9g, got %.9g\n",
                               i, expected, out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
    }

    if (passed) {
        printf("  PASS: Additional numeric conversion outputs match expected casts\n");
    } else {
        tests_failed++;
    }
}

void test_complex_output_conversions() {
    TEST("Complex-to-real and complex narrowing conversions");

    int passed = 1;
    int err;
    me_expr *expr = NULL;

    {
        float _Complex x[VECTOR_SIZE] = {
            -3.5f + 1.0f * I, -2.0f - 4.0f * I, -1.0f + 2.0f * I, 0.0f + 3.0f * I, 1.0f - 1.0f * I,
            2.25f + 0.5f * I, 42.0f + 5.0f * I, 127.75f - 7.0f * I, 128.5f + 8.0f * I, 1000.0f - 2.0f * I
        };
        float _Complex y[VECTOR_SIZE] = {0};
        double out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_COMPLEX64}, {"y", ME_COMPLEX64}};

        int rc_expr = me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL complex64->float64: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL complex64->float64: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    double expected = (double)crealf(x[i] + y[i]);
                    if (fabs(out[i] - expected) > 1e-12) {
                        printf("  FAIL complex64->float64 at [%d]: expected %.17g, got %.17g\n",
                               i, expected, out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        float _Complex x[VECTOR_SIZE] = {
            -300.0f + 0.25f * I, -1.0f - 2.0f * I, 0.0f + 1.0f * I, 1.0f + 0.5f * I, 127.9f - 1.0f * I,
            128.1f + 2.0f * I, 255.0f - 3.0f * I, 256.0f + 4.0f * I, 511.7f - 5.0f * I, 1000.2f + 6.0f * I
        };
        float _Complex y[VECTOR_SIZE] = {0};
        int32_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_COMPLEX64}, {"y", ME_COMPLEX64}};

        int rc_expr = me_compile("x + y", vars, 2, ME_INT32, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL complex64->int32: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL complex64->int32: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    int32_t expected = (int32_t)crealf(x[i] + y[i]);
                    if (out[i] != expected) {
                        printf("  FAIL complex64->int32 at [%d]: expected %lld, got %lld\n",
                               i, (long long)expected, (long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        double _Complex x[VECTOR_SIZE] = {
            -3.5 + 1.25 * I, -2.0 - 4.5 * I, -1.0 + 2.75 * I, 0.0 + 3.0 * I, 1.0 - 1.5 * I,
            2.25 + 0.5 * I, 42.0 + 5.5 * I, 127.75 - 7.125 * I, 128.5 + 8.25 * I, 1000.0 - 2.875 * I
        };
        double _Complex y[VECTOR_SIZE] = {0};
        float out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_COMPLEX128}, {"y", ME_COMPLEX128}};

        int rc_expr = me_compile("x + y", vars, 2, ME_FLOAT32, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL complex128->float32: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL complex128->float32: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    float expected = (float)creal(x[i] + y[i]);
                    if (fabsf(out[i] - expected) > 1e-6f) {
                        printf("  FAIL complex128->float32 at [%d]: expected %.9g, got %.9g\n",
                               i, expected, out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        double _Complex x[VECTOR_SIZE] = {
            0.0 + 1.0 * I, 1.0 + 2.0 * I, 2.0 + 3.0 * I, 42.0 + 4.0 * I, 255.0 + 5.0 * I,
            1024.0 + 6.0 * I, 2048.0 + 7.0 * I, 4096.0 + 8.0 * I, 12345.0 + 9.0 * I, 32767.0 + 10.0 * I
        };
        double _Complex y[VECTOR_SIZE] = {0};
        uint16_t out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_COMPLEX128}, {"y", ME_COMPLEX128}};

        int rc_expr = me_compile("x + y", vars, 2, ME_UINT16, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL complex128->uint16: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL complex128->uint16: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    uint16_t expected = (uint16_t)creal(x[i] + y[i]);
                    if (out[i] != expected) {
                        printf("  FAIL complex128->uint16 at [%d]: expected %llu, got %llu\n",
                               i, (unsigned long long)expected, (unsigned long long)out[i]);
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
        expr = NULL;
    }

    {
        double _Complex x[VECTOR_SIZE] = {
            -3.5 + 1.25 * I, -2.0 - 4.5 * I, -1.0 + 2.75 * I, 0.0 + 3.0 * I, 1.0 - 1.5 * I,
            2.25 + 0.5 * I, 42.0 + 5.5 * I, 127.75 - 7.125 * I, 128.5 + 8.25 * I, 1000.0 - 2.875 * I
        };
        double _Complex y[VECTOR_SIZE] = {0};
        float _Complex out[VECTOR_SIZE];
        me_variable vars[] = {{"x", ME_COMPLEX128}, {"y", ME_COMPLEX128}};

        int rc_expr = me_compile("x + y", vars, 2, ME_COMPLEX64, &err, &expr);
        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("  FAIL complex128->complex64: compilation error at position %d\n", err);
            passed = 0;
        } else {
            const void *var_ptrs[] = {x, y};
            int rc_eval = me_eval(expr, var_ptrs, 2, out, VECTOR_SIZE, NULL);
            if (rc_eval != ME_EVAL_SUCCESS) {
                printf("  FAIL complex128->complex64: eval error %d\n", rc_eval);
                passed = 0;
            } else {
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    float _Complex expected = (float _Complex)(x[i] + y[i]);
                    float dre = fabsf(crealf(out[i]) - crealf(expected));
                    float dim = fabsf(cimagf(out[i]) - cimagf(expected));
                    if (dre > 1e-6f || dim > 1e-6f) {
                        printf("  FAIL complex128->complex64 at [%d]: expected (%.9g, %.9g), got (%.9g, %.9g)\n",
                               i, crealf(expected), cimagf(expected), crealf(out[i]), cimagf(out[i]));
                        passed = 0;
                    }
                }
            }
        }
        me_free(expr);
    }

    if (passed) {
        printf("  PASS: Complex conversion outputs match expected casts\n");
    } else {
        tests_failed++;
    }
}

int main() {
    printf("========================================================================\n");
    printf("TEST: Explicit Variable Types with Explicit Output Dtype\n");
    printf("========================================================================\n");
    printf("This test verifies that when both variable types and output dtype\n");
    printf("are explicitly specified:\n");
    printf("  - Variables keep their types during computation\n");
    printf("  - Result is correctly cast to the specified output dtype\n");
    printf("========================================================================\n");

    test_mixed_types_float32_output();
    test_float32_vars_float64_output();
    test_float32_with_constant_float64_output();
    test_comparison_explicit_bool_output();
    test_integer_output_conversions();
    test_numeric_output_conversions();
    test_complex_output_conversions();

    printf("\n========================================================================\n");
    printf("Test Summary\n");
    printf("========================================================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);
    printf("========================================================================\n");

    if (tests_failed == 0) {
        printf("✅ ALL TESTS PASSED\n");
    } else {
        printf("❌ SOME TESTS FAILED\n");
    }

    return (tests_failed == 0) ? 0 : 1;
}
