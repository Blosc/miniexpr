/* Test that both acos/arccos naming conventions work */
#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#define VECTOR_SIZE 5
#define TOLERANCE 1e-9

int main() {
    printf("=== Testing Inverse Trigonometric Function Aliases ===\n\n");

    double x[VECTOR_SIZE] = {0.0, 0.5, 0.707, 0.866, 1.0};
    double result_a[VECTOR_SIZE] = {0};
    double result_arc[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};
    int err;
    int tests_passed = 0;
    int tests_total = 0;

    // Test acos vs arccos
    tests_total++;
    printf("Test 1: acos(x) vs arccos(x)\n");
    me_expr *expr_acos = NULL;
    int rc_expr_acos = me_compile("acos(x)", vars, 1, ME_FLOAT64, &err, &expr_acos);
    me_expr *expr_arccos = NULL;
    int rc_expr_arccos = me_compile("arccos(x)", vars, 1, ME_FLOAT64, &err, &expr_arccos);

    if (rc_expr_acos != ME_COMPILE_SUCCESS || rc_expr_arccos != ME_COMPILE_SUCCESS ||
        !expr_acos || !expr_arccos) {
        printf("  FAIL: Compilation failed (acos=%d, arccos=%d, err=%d)\n",
               rc_expr_acos, rc_expr_arccos, err);
    } else {
        const void *var_ptrs[] = {x};
        ME_EVAL_CHECK(expr_acos, var_ptrs, 1, result_a, VECTOR_SIZE);
        ME_EVAL_CHECK(expr_arccos, var_ptrs, 1, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_acos);
        me_free(expr_arccos);
    }

    // Test asin vs arcsin
    tests_total++;
    printf("\nTest 2: asin(x) vs arcsin(x)\n");
    me_expr *expr_asin = NULL;
    int rc_expr_asin = me_compile("asin(x)", vars, 1, ME_FLOAT64, &err, &expr_asin);
    me_expr *expr_arcsin = NULL;
    int rc_expr_arcsin = me_compile("arcsin(x)", vars, 1, ME_FLOAT64, &err, &expr_arcsin);

    if (rc_expr_asin != ME_COMPILE_SUCCESS || rc_expr_arcsin != ME_COMPILE_SUCCESS ||
        !expr_asin || !expr_arcsin) {
        printf("  FAIL: Compilation failed (asin=%d, arcsin=%d, err=%d)\n",
               rc_expr_asin, rc_expr_arcsin, err);
    } else {
        const void *var_ptrs[] = {x};
        ME_EVAL_CHECK(expr_asin, var_ptrs, 1, result_a, VECTOR_SIZE);
        ME_EVAL_CHECK(expr_arcsin, var_ptrs, 1, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_asin);
        me_free(expr_arcsin);
    }

    // Test atan vs arctan
    tests_total++;
    printf("\nTest 3: atan(x) vs arctan(x)\n");
    me_expr *expr_atan = NULL;
    int rc_expr_atan = me_compile("atan(x)", vars, 1, ME_FLOAT64, &err, &expr_atan);
    me_expr *expr_arctan = NULL;
    int rc_expr_arctan = me_compile("arctan(x)", vars, 1, ME_FLOAT64, &err, &expr_arctan);

    if (rc_expr_atan != ME_COMPILE_SUCCESS || rc_expr_arctan != ME_COMPILE_SUCCESS ||
        !expr_atan || !expr_arctan) {
        printf("  FAIL: Compilation failed (atan=%d, arctan=%d, err=%d)\n",
               rc_expr_atan, rc_expr_arctan, err);
    } else {
        const void *var_ptrs[] = {x};
        ME_EVAL_CHECK(expr_atan, var_ptrs, 1, result_a, VECTOR_SIZE);
        ME_EVAL_CHECK(expr_arctan, var_ptrs, 1, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_atan);
        me_free(expr_arctan);
    }

    // Test atan2 vs arctan2
    tests_total++;
    printf("\nTest 4: atan2(y, x) vs arctan2(y, x)\n");
    double y[VECTOR_SIZE] = {1.0, 0.5, 0.707, 0.866, 0.0};
    me_variable vars2[] = {{"y"}, {"x"}};
    me_expr *expr_atan2 = NULL;
    int rc_expr_atan2 = me_compile("atan2(y, x)", vars2, 2, ME_FLOAT64, &err, &expr_atan2);
    me_expr *expr_arctan2 = NULL;
    int rc_expr_arctan2 = me_compile("arctan2(y, x)", vars2, 2, ME_FLOAT64, &err, &expr_arctan2);

    if (rc_expr_atan2 != ME_COMPILE_SUCCESS || rc_expr_arctan2 != ME_COMPILE_SUCCESS ||
        !expr_atan2 || !expr_arctan2) {
        printf("  FAIL: Compilation failed (atan2=%d, arctan2=%d, err=%d)\n",
               rc_expr_atan2, rc_expr_arctan2, err);
    } else {
        const void *var_ptrs[] = {y, x};
        ME_EVAL_CHECK(expr_atan2, var_ptrs, 2, result_a, VECTOR_SIZE);
        ME_EVAL_CHECK(expr_arctan2, var_ptrs, 2, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_atan2);
        me_free(expr_arctan2);
    }

    // Test arccos on int32 (replicating python-blosc2 failure on Windows)
    tests_total++;
    printf("\nTest 5: arccos(int32) - linspace-like integer values\n");

    // Simulate linspace(0.01, 0.99, 10) for int32
    // When converting floats 0.01..0.99 to int32, they all become 0
    int32_t x_int32[10];
    for (int i = 0; i < 10; i++) {
        double val = 0.01 + (0.99 - 0.01) * i / 9.0;
        x_int32[i] = (int32_t)val;  // All will be 0
    }

    printf("  Input int32 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", x_int32[i]);
    }
    printf("\n");

    double result_int32[10] = {0};
    me_variable vars_int32[] = {{"x", ME_INT32, x_int32}};
    me_expr *expr_arccos_int32 = NULL;
    int rc_arccos_int32 = me_compile("arccos(x)", vars_int32, 1, ME_AUTO, &err, &expr_arccos_int32);

    if (rc_arccos_int32 != ME_COMPILE_SUCCESS || !expr_arccos_int32) {
        printf("  FAIL: Compilation failed (rc=%d, err=%d)\n", rc_arccos_int32, err);
    } else {
        const void *var_ptrs_int32[] = {x_int32};
        ME_EVAL_CHECK(expr_arccos_int32, var_ptrs_int32, 1, result_int32, 10);

        printf("  Output values: ");
        int has_nan = 0;
        for (int i = 0; i < 10; i++) {
            if (isnan(result_int32[i])) {
                printf("nan ");
                has_nan = 1;
            } else {
                printf("%.6f ", result_int32[i]);
            }
        }
        printf("\n");

        // arccos(0) should be π/2 ≈ 1.570796
        int all_valid = 1;
        for (int i = 0; i < 10; i++) {
            if (isnan(result_int32[i])) {
                all_valid = 0;
                break;
            }
            // Expected: arccos(0) = 1.570796...
            double expected = acos((double)x_int32[i]);
            if (fabs(result_int32[i] - expected) > 1e-6) {
                all_valid = 0;
                break;
            }
        }

        if (all_valid && !has_nan) {
            printf("  PASS: All values are valid (no unexpected NaN)\n");
            tests_passed++;
        } else {
            printf("  FAIL: Found unexpected NaN values or incorrect results\n");
            printf("  This replicates the python-blosc2 Windows CI failure!\n");
        }

        me_free(expr_arccos_int32);
    }

    // Test arccos on int32 with chunked evaluation (chunk size = 3)
    tests_total++;
    printf("\nTest 6: arccos(int32) - chunked evaluation (size 3)\n");

    me_expr *expr_arccos_chunked = NULL;
    int rc_arccos_chunked = me_compile("arccos(x)", vars_int32, 1, ME_AUTO, &err, &expr_arccos_chunked);

    if (rc_arccos_chunked != ME_COMPILE_SUCCESS || !expr_arccos_chunked) {
        printf("  FAIL: Compilation failed (rc=%d, err=%d)\n", rc_arccos_chunked, err);
    } else {
        double result_chunked[10] = {0};
        const void *var_ptrs_int32[] = {x_int32};

        // Evaluate in chunks of 3 (simulating python-blosc2's chunkshape)
        printf("  Evaluating in chunks of 3:\n");
        for (int chunk_start = 0; chunk_start < 10; chunk_start += 3) {
            int chunk_size = (chunk_start + 3 <= 10) ? 3 : (10 - chunk_start);
            int32_t *chunk_input = &x_int32[chunk_start];
            double *chunk_output = &result_chunked[chunk_start];
            const void *chunk_vars[] = {chunk_input};

            ME_EVAL_CHECK(expr_arccos_chunked, chunk_vars, 1, chunk_output, chunk_size);

            printf("    Chunk [%d:%d]: ", chunk_start, chunk_start + chunk_size - 1);
            for (int i = 0; i < chunk_size; i++) {
                if (isnan(chunk_output[i])) {
                    printf("nan ");
                } else {
                    printf("%.6f ", chunk_output[i]);
                }
            }
            printf("\n");
        }

        printf("  Final result: ");
        int has_nan_chunked = 0;
        for (int i = 0; i < 10; i++) {
            if (isnan(result_chunked[i])) {
                printf("nan ");
                has_nan_chunked = 1;
            } else {
                printf("%.6f ", result_chunked[i]);
            }
        }
        printf("\n");

        int all_valid_chunked = 1;
        for (int i = 0; i < 10; i++) {
            if (isnan(result_chunked[i])) {
                all_valid_chunked = 0;
                break;
            }
            double expected = acos((double)x_int32[i]);
            if (fabs(result_chunked[i] - expected) > 1e-6) {
                all_valid_chunked = 0;
                break;
            }
        }

        if (all_valid_chunked && !has_nan_chunked) {
            printf("  PASS: Chunked evaluation produces valid results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Chunked evaluation produced unexpected NaN\n");
            printf("  This matches the python-blosc2 Windows CI failure pattern!\n");
        }

        me_free(expr_arccos_chunked);
    }

    printf("\n=== Test Summary ===\n");
    printf("Tests passed: %d/%d\n", tests_passed, tests_total);

    return (tests_passed == tests_total) ? 0 : 1;
}
