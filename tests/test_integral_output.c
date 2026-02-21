/*
 * Test integral output dtypes: int64, int32, bool outputs.
 *
 * On Windows (clang-cl), miniexpr has been reported to produce
 * wrong results when the output dtype is integral.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define N 20

static int failures = 0;

/* ------------------------------------------------------------------ */
/* Test 1: int64 + int64 -> int64 output (basic arithmetic)           */
/* ------------------------------------------------------------------ */
static void test_int64_add_int64(void) {
    printf("Test: int64 + int64 -> int64 output\n");
    int prev = failures;

    int64_t a[N], b[N], result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int64_t)(i + 1);
        b[i] = (int64_t)(i * 2);
    }

    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + b", vars, 2, ME_INT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, ptrs, 2, result, N);

    for (int i = 0; i < N; i++) {
        int64_t expected = a[i] + b[i];
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %lld, got %lld\n", i,
                   (long long)expected, (long long)result[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 2: int64 * int64 -> int64 output                              */
/* ------------------------------------------------------------------ */
static void test_int64_mul_int64(void) {
    printf("Test: int64 * int64 -> int64 output\n");
    int prev = failures;

    int64_t a[N], b[N], result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int64_t)(i + 1);
        b[i] = (int64_t)(i + 3);
    }

    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a * b", vars, 2, ME_INT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, ptrs, 2, result, N);

    for (int i = 0; i < N; i++) {
        int64_t expected = a[i] * b[i];
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %lld, got %lld\n", i,
                   (long long)expected, (long long)result[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 3: int32 + int32 -> int32 output                              */
/* ------------------------------------------------------------------ */
static void test_int32_add_int32(void) {
    printf("Test: int32 + int32 -> int32 output\n");
    int prev = failures;

    int32_t a[N], b[N], result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int32_t)(i + 1);
        b[i] = (int32_t)(i * 3);
    }

    me_variable vars[] = {{"a", ME_INT32}, {"b", ME_INT32}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + b", vars, 2, ME_INT32, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, ptrs, 2, result, N);

    for (int i = 0; i < N; i++) {
        int32_t expected = a[i] + b[i];
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %d, got %d\n", i, expected, result[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 4: float64 expression -> int64 output (truncation)            */
/* ------------------------------------------------------------------ */
static void test_float64_to_int64(void) {
    printf("Test: float64 expr -> int64 output (truncation)\n");
    int prev = failures;

    double a[N];
    int64_t result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (double)i + 0.7;
    }

    me_variable vars[] = {{"a", ME_FLOAT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + 0.1", vars, 1, ME_INT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        int64_t expected = (int64_t)(a[i] + 0.1);
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %lld, got %lld\n", i,
                   (long long)expected, (long long)result[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 5: float32 -> bool output (nonzero check)                     */
/* ------------------------------------------------------------------ */
static void test_float32_to_bool(void) {
    printf("Test: float32 -> bool output (x != 0)\n");
    int prev = failures;

    float a[N];
    int8_t result[N];  /* ME_BOOL is stored as int8 */
    for (int i = 0; i < N; i++) {
        a[i] = (i % 3 == 0) ? 0.0f : (float)(i + 1);
    }

    me_variable vars[] = {{"a", ME_FLOAT32}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a != 0", vars, 1, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        int8_t expected = (a[i] != 0.0f) ? 1 : 0;
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %d, got %d (a=%.1f)\n",
                   i, expected, result[i], a[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 6: int64 arithmetic -> int64 via me_eval_nd                   */
/* ------------------------------------------------------------------ */
static void test_int64_add_nd(void) {
    printf("Test: int64 + int64 -> int64 via me_eval_nd\n");
    int prev = failures;

    const int rows = 4, cols = 5;
    const int total = rows * cols;
    int64_t a[20], b[20], result[20];
    for (int i = 0; i < total; i++) {
        a[i] = (int64_t)(i + 1);
        b[i] = (int64_t)(i * 2);
    }

    int64_t shape[] = {rows, cols};
    int32_t chunks[] = {rows, cols};
    int32_t blocks[] = {2, 5};

    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile_nd("a + b", vars, 2, ME_INT64, 2,
                           shape, chunks, blocks, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a, b};
    int block_nitems = blocks[0] * blocks[1];
    rc = me_eval_nd(expr, ptrs, 2, result, block_nitems, 0, 0, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL: me_eval_nd returned %d\n", rc);
        failures++;
        me_free(expr);
        return;
    }

    for (int i = 0; i < block_nitems; i++) {
        int64_t expected = a[i] + b[i];
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %lld, got %lld\n", i,
                   (long long)expected, (long long)result[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 7: int64 expression with constant -> int64 output             */
/* ------------------------------------------------------------------ */
static void test_int64_expr_with_constant(void) {
    printf("Test: int64 * 3 + 1 -> int64 output\n");
    int prev = failures;

    int64_t a[N], result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int64_t)(i + 1);
    }

    me_variable vars[] = {{"a", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a * 3 + 1", vars, 1, ME_INT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        int64_t expected = a[i] * 3 + 1;
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %lld, got %lld\n", i,
                   (long long)expected, (long long)result[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

/* ------------------------------------------------------------------ */
/* Test 8: int64 comparison -> bool output                            */
/* ------------------------------------------------------------------ */
static void test_int64_comparison_to_bool(void) {
    printf("Test: int64 > 10 -> bool output\n");
    int prev = failures;

    int64_t a[N];
    int8_t result[N];
    for (int i = 0; i < N; i++) {
        a[i] = (int64_t)(i + 1);
    }

    me_variable vars[] = {{"a", ME_INT64}};
    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a > 10", vars, 1, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile error %d at pos %d\n", rc, err);
        failures++;
        return;
    }

    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, result, N);

    for (int i = 0; i < N; i++) {
        int8_t expected = (a[i] > 10) ? 1 : 0;
        if (result[i] != expected) {
            printf("  FAIL [%d]: expected %d, got %d (a=%lld)\n",
                   i, expected, result[i], (long long)a[i]);
            failures++;
        }
    }

    me_free(expr);
    printf("  %s\n", failures == prev ? "PASS" : "FAILED");
}

int main(void) {
    printf("=== Integral output dtype tests ===\n\n");

    test_int64_add_int64();
    test_int64_mul_int64();
    test_int32_add_int32();
    test_float64_to_int64();
    test_float32_to_bool();
    test_int64_add_nd();
    test_int64_expr_with_constant();
    test_int64_comparison_to_bool();

    printf("\n=== %s: %d failure(s) ===\n",
           failures == 0 ? "ALL PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
