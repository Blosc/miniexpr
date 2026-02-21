/*
 * Test dtype-mismatched math functions: int64 input → float64 output.
 *
 * Reproduces the python-blosc2 failure where arcsinh(int64_array)
 * with float64 output produces truncated (integer) values instead of
 * the correct floating-point results.
 *
 * On macOS/Linux this works correctly; the bug manifests on Windows
 * (clang-cl builds).
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define N 20
#define TOL 1e-6

static int failures = 0;

#define CHECK_NEAR(expected, actual, idx, label) \
    do { \
        double _e = (double)(expected); \
        double _a = (double)(actual); \
        double _d = fabs(_e - _a); \
        if (_d > TOL) { \
            printf("  FAIL %s [%d]: expected %.10f, got %.10f (diff %.2e)\n", \
                   label, idx, _e, _a, _d); \
            failures++; \
        } \
    } while (0)

/* ------------------------------------------------------------------ */
/* Test 1: arcsinh(int64) → float64 output                           */
/* ------------------------------------------------------------------ */
static void test_arcsinh_int64_to_float64(void) {
    printf("Test: arcsinh(int64) -> float64 output\n");

    int64_t x[N];
    double result[N];
    for (int i = 0; i < N; i++) {
        x[i] = (int64_t)(i + 1);  /* 1, 2, 3, ... 20 */
    }

    me_variable vars[] = {{"x", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("arcsinh(x)", vars, 1, ME_FLOAT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {x};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        double expected = asinh((double)x[i]);
        CHECK_NEAR(expected, result[i], i, "arcsinh(int64)->f64");
    }

    me_free(expr);
    printf("  %s\n", failures == 0 ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 2: sin(int64) → float64 output                               */
/* ------------------------------------------------------------------ */
static void test_sin_int64_to_float64(void) {
    printf("Test: sin(int64) -> float64 output\n");
    int prev_failures = failures;

    int64_t x[N];
    double result[N];
    for (int i = 0; i < N; i++) {
        x[i] = (int64_t)(i + 1);
    }

    me_variable vars[] = {{"x", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("sin(x)", vars, 1, ME_FLOAT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {x};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        double expected = sin((double)x[i]);
        CHECK_NEAR(expected, result[i], i, "sin(int64)->f64");
    }

    me_free(expr);
    printf("  %s\n", failures == prev_failures ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 3: sqrt(int32) → float64 output                              */
/* ------------------------------------------------------------------ */
static void test_sqrt_int32_to_float64(void) {
    printf("Test: sqrt(int32) -> float64 output\n");
    int prev_failures = failures;

    int32_t x[N];
    double result[N];
    for (int i = 0; i < N; i++) {
        x[i] = (int32_t)(i + 1);
    }

    me_variable vars[] = {{"x", ME_INT32}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("sqrt(x)", vars, 1, ME_FLOAT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {x};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        double expected = sqrt((double)x[i]);
        CHECK_NEAR(expected, result[i], i, "sqrt(int32)->f64");
    }

    me_free(expr);
    printf("  %s\n", failures == prev_failures ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 4: float32 input + float64 output  (a + 0.5)                 */
/* ------------------------------------------------------------------ */
static void test_float32_expr_to_float64(void) {
    printf("Test: (float32 + 0.5) -> float64 output\n");
    int prev_failures = failures;

    float x[N];
    double result[N];
    for (int i = 0; i < N; i++) {
        x[i] = (float)i * 0.1f;
    }

    me_variable vars[] = {{"x", ME_FLOAT32}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("arcsinh(x)", vars, 1, ME_FLOAT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {x};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        double expected = asinh((double)x[i]);
        CHECK_NEAR(expected, result[i], i, "arcsinh(f32)->f64");
    }

    me_free(expr);
    printf("  %s\n", failures == prev_failures ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 5: int64 + float32 mixed binary → float64 output             */
/* ------------------------------------------------------------------ */
static void test_mixed_int64_float32_to_float64(void) {
    printf("Test: int64 + float32 -> float64 output\n");
    int prev_failures = failures;

    int64_t a[N];
    float b[N];
    double result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int64_t)(i + 1);
        b[i] = 0.5f;
    }

    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_FLOAT32}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + b", vars, 2, ME_FLOAT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, ptrs, 2, result, N);

    for (int i = 0; i < N; i++) {
        double expected = (double)a[i] + (double)b[i];
        CHECK_NEAR(expected, result[i], i, "int64+f32->f64");
    }

    me_free(expr);
    printf("  %s\n", failures == prev_failures ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 6: arcsinh(int64) via me_eval_nd (blosc2 code path)          */
/* ------------------------------------------------------------------ */
static void test_arcsinh_int64_to_float64_nd(void) {
    printf("Test: arcsinh(int64) -> float64 via me_eval_nd\n");
    int prev_failures = failures;

    const int rows = 4, cols = 5;
    const int total = rows * cols;
    int64_t x[20];
    double result[20];
    for (int i = 0; i < total; i++) {
        x[i] = (int64_t)(i + 1);
    }

    int64_t shape[] = {rows, cols};
    int32_t chunks[] = {rows, cols};
    int32_t blocks[] = {2, 5};

    me_variable vars[] = {{"x", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile_nd("arcsinh(x)", vars, 1, ME_FLOAT64, 2,
                           shape, chunks, blocks, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {x};
    int block_nitems = blocks[0] * blocks[1];
    /* Evaluate block (0,0) via me_eval_nd */
    rc = me_eval_nd(expr, ptrs, 1, result, block_nitems, 0, 0, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL: me_eval_nd returned %d\n", rc);
        failures++;
        me_free(expr);
        return;
    }

    for (int i = 0; i < block_nitems; i++) {
        double expected = asinh((double)x[i]);
        CHECK_NEAR(expected, result[i], i, "arcsinh(int64)->f64 nd");
    }

    me_free(expr);
    printf("  %s\n", failures == prev_failures ? "PASS" : "FAILED");
}

int main(void) {
    printf("=== Dtype-mismatch math tests ===\n\n");

    test_arcsinh_int64_to_float64();
    test_sin_int64_to_float64();
    test_sqrt_int32_to_float64();
    test_float32_expr_to_float64();
    test_mixed_int64_float32_to_float64();
    test_arcsinh_int64_to_float64_nd();

    printf("\n=== %s: %d failure(s) ===\n",
           failures == 0 ? "ALL PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
