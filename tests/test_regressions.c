#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "miniexpr.h"
#include "minctest.h"



#define SMALL_SIZE 10
#define LARGE_SIZE 100
#define CHUNK_SIZE 5

// ============================================================================
// ARCTAN2 ARRAY-SCALAR TESTS
// ============================================================================

int test_arctan2_with_scalar_constant(const char *description, int size, float scalar_value) {
    printf("\n%s\n", description);
    printf("======================================================================\n");

    float *input = malloc(size * sizeof(float));
    float max_val = (size == SMALL_SIZE) ? 5.0f : 10.0f;
    for (int i = 0; i < size; i++) {
        input[i] = max_val * i / (size - 1);
    }

    char expr_str[256];
    snprintf(expr_str, sizeof(expr_str), "arctan2(x, %.1f)", scalar_value);
    printf("Expression: %s\n", expr_str);
    printf("Array size: %d elements\n", size);

    me_variable vars[] = {{"x", ME_FLOAT32}};
    int err;
    // Use ME_AUTO - following NumPy conventions, float constants match variable type (FLOAT32)
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, vars, 1, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("✗ COMPILATION FAILED with error code: %d\n", err);
        free(input);
        return 0;
    }

    // Following NumPy conventions, float constants match variable type, so result is FLOAT32
    const void *var_ptrs[] = {input};
    float *result = malloc(size * sizeof(float));
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, size);

    float *expected = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        expected[i] = atan2f(input[i], scalar_value);
    }

    int passed = 1;
    float max_diff = 0.0f;
#ifdef __FAST_MATH__
    const float tolerance = 1e-4f;
#else
    const float tolerance = 1e-5f;
#endif
    for (int i = 0; i < size; i++) {
        float diff = fabsf(result[i] - expected[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            passed = 0;
        }
    }

    printf("Result (first 5):   ");
    for (int i = 0; i < 5 && i < size; i++) {
        printf("%.6f ", result[i]);
    }
    printf("...\n");

    printf("Expected (first 5): ");
    for (int i = 0; i < 5 && i < size; i++) {
        printf("%.6f ", expected[i]);
    }
    printf("...\n");

    if (passed) {
        printf("Status: ✓ PASS\n");
    } else {
        printf("Status: ✗ FAIL (max diff: %.9f)\n", max_diff);
    }

    free(input);
    free(result);
    free(expected);
    me_free(expr);

    return passed;
}

int test_arctan2_with_two_arrays(const char *description, int size, float scalar_value) {
    printf("\n%s\n", description);
    printf("======================================================================\n");

    float *input1 = malloc(size * sizeof(float));
    float *input2 = malloc(size * sizeof(float));
    float max_val = (size == SMALL_SIZE) ? 5.0f : 10.0f;
    for (int i = 0; i < size; i++) {
        input1[i] = max_val * i / (size - 1);
        input2[i] = scalar_value;
    }

    printf("Expression: arctan2(x, y)\n");
    printf("Array size: %d elements\n", size);
    printf("y array: all elements = %.1f\n", scalar_value);

    me_variable vars[] = {{"x", ME_FLOAT32}, {"y", ME_FLOAT32}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("arctan2(x, y)", vars, 2, ME_FLOAT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("✗ COMPILATION FAILED with error code: %d\n", err);
        free(input1);
        free(input2);
        return 0;
    }

    const void *var_ptrs[] = {input1, input2};
    float *result = malloc(size * sizeof(float));

    ME_EVAL_CHECK(expr, var_ptrs, 2, result, size);

    float *expected = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        expected[i] = atan2f(input1[i], input2[i]);
    }

    int passed = 1;
    float max_diff = 0.0f;
#ifdef __FAST_MATH__
    const float tolerance = 1e-4f;
#else
    const float tolerance = 1e-5f;
#endif
    for (int i = 0; i < size; i++) {
        float diff = fabsf(result[i] - expected[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            passed = 0;
        }
    }

    printf("Result (first 5):   ");
    for (int i = 0; i < 5 && i < size; i++) {
        printf("%.6f ", result[i]);
    }
    printf("...\n");

    printf("Expected (first 5): ");
    for (int i = 0; i < 5 && i < size; i++) {
        printf("%.6f ", expected[i]);
    }
    printf("...\n");

    if (passed) {
        printf("Status: ✓ PASS\n");
    } else {
        printf("Status: ✗ FAIL (max diff: %.9f)\n", max_diff);
    }

    free(input1);
    free(input2);
    free(result);
    free(expected);
    me_free(expr);

    return passed;
}

// ============================================================================
// ARCTAN2 BUG TESTS (MIXED ARRAY/SCALAR)
// ============================================================================

int test_arctan2_array_scalar_f64(const char *description, const char *expr_str,
                                   double *data, double scalar, int invert) {
    printf("\n%s\n", description);

    me_variable vars[] = {{"y", ME_FLOAT64}, {"x", ME_FLOAT64}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, invert ? vars + 1 : vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: Compilation error %d\n", err);
        return 0;
    }

    const void *var_ptrs[] = {data};
    double result[CHUNK_SIZE];

    ME_EVAL_CHECK(expr, var_ptrs, 1, result, CHUNK_SIZE);

    printf("  Results:\n");
    int passed = 1;
    for (int i = 0; i < CHUNK_SIZE; i++) {
        double expected = invert ? atan2(scalar, data[i]) : atan2(data[i], scalar);
        printf("    %s = %.6f (expected: %.6f)", expr_str, result[i], expected);
        if (fabs(result[i] - expected) > 1e-10) {
            printf(" ❌ MISMATCH\n");
            passed = 0;
        } else {
            printf(" ✓\n");
        }
    }

    if (passed) {
        printf("  ✅ PASSED\n");
    } else {
        printf("  ❌ FAILED\n");
    }

    me_free(expr);
    return passed;
}

int test_pow_array_scalar_f64(const char *description, const char *expr_str,
                               double *data, double scalar, int invert) {
    printf("\n%s\n", description);

    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: Compilation error %d\n", err);
        return 0;
    }

    const void *var_ptrs[] = {data};
    double result[CHUNK_SIZE];

    ME_EVAL_CHECK(expr, var_ptrs, 1, result, CHUNK_SIZE);

    printf("  Results:\n");
    int passed = 1;
    for (int i = 0; i < CHUNK_SIZE; i++) {
        double expected = invert ? pow(scalar, data[i]) : pow(data[i], scalar);
        printf("    %s = %.6f (expected: %.6f)", expr_str, result[i], expected);
        if (fabs(result[i] - expected) > 1e-10) {
            printf(" ❌ MISMATCH\n");
            passed = 0;
        } else {
            printf(" ✓\n");
        }
    }

    if (passed) {
        printf("  ✅ PASSED\n");
    } else {
        printf("  ❌ FAILED\n");
    }

    me_free(expr);
    return passed;
}

// ============================================================================
// ARCTAN2 COMPLEX EXPRESSION TESTS
// ============================================================================

int test_arctan2_complex_expr(const char *description, const char *expr_str,
                               double *x_data, double *y_data,
                               double (*expected_fn)(double, double)) {
    printf("\n%s\n", description);

    me_variable vars[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, vars, 2, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: Compilation error %d\n", err);
        return 0;
    }

    const void *var_ptrs[] = {x_data, y_data};
    double result[CHUNK_SIZE];

    ME_EVAL_CHECK(expr, var_ptrs, 2, result, CHUNK_SIZE);

    printf("  Results:\n");
    int passed = 1;
    for (int i = 0; i < CHUNK_SIZE; i++) {
        double expected = expected_fn(x_data[i], y_data[i]);
        printf("    x=%.1f, y=%.1f: %.6f (expected: %.6f)",
               x_data[i], y_data[i], result[i], expected);
        if (fabs(result[i] - expected) > 1e-10) {
            printf(" ❌ MISMATCH\n");
            passed = 0;
        } else {
            printf(" ✓\n");
        }
    }

    if (passed) {
        printf("  ✅ PASSED\n");
    } else {
        printf("  ❌ FAILED\n");
    }

    me_free(expr);
    return passed;
}

double arctan2_x_plus_y_1(double x, double y) { return atan2(x + y, 1.0); }
double arctan2_1_x_plus_y(double x, double y) { return atan2(1.0, x + y); }

// ============================================================================
// CONSTANT TYPE INFERENCE TESTS
// ============================================================================

int test_constant_type_f32(const char *description, const char *expr_str,
                            float (*expected_fn)(float)) {
    printf("\n%s\n", description);
    printf("=================================================================\n");
    printf("Expression: %s\n", expr_str);
    printf("Variable dtype: ME_FLOAT32, Output dtype: ME_AUTO\n");
    printf("Note: Following NumPy conventions, float constants match variable type, so result will be FLOAT32\n");

    me_variable vars[] = {{"a", ME_FLOAT32}};
    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, vars, 1, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("❌ COMPILATION FAILED (error %d)\n", err);
        return 0;
    }

    me_dtype actual_dtype = me_get_dtype(expr);
    printf("Actual result dtype: %s\n",
           actual_dtype == ME_FLOAT32 ? "ME_FLOAT32" :
           actual_dtype == ME_FLOAT64 ? "ME_FLOAT64" : "OTHER");

    // Following NumPy conventions, float constants match variable type, so result should be FLOAT32
    if (actual_dtype != ME_FLOAT32) {
        printf("⚠️  Type mismatch! Expected ME_FLOAT32 (NumPy convention), got %d\n", actual_dtype);
        me_free(expr);
        return 0;
    }

    float input[SMALL_SIZE];
    for (int i = 0; i < SMALL_SIZE; i++) input[i] = (float)i;

    const void *var_ptrs[] = {input};
    float result[SMALL_SIZE];  // Use float buffer for FLOAT32 result

    ME_EVAL_CHECK(expr, var_ptrs, 1, result, SMALL_SIZE);

    int passed = 1;
    printf("\nFirst 5 results:\n");
    printf("Index  Input      Result     Expected   Status\n");

    for (int i = 0; i < 5; i++) {
        float expected = expected_fn(input[i]);
        float diff = fabsf(result[i] - expected);
        char status = (diff < 1e-5f) ? 'Y' : 'N';
        printf("%-6d %-10.2f %-10.6f %-10.6f %c\n", i, input[i], result[i], expected, status);
        if (diff >= 1e-5f) passed = 0;
    }

    printf("\nStatus: ");
    if (passed) {
        printf("✅ PASS\n");
    } else {
        printf("❌ FAIL\n");
    }

    me_free(expr);
    return passed;
}

float add_3_f32(float x) { return x + 3.0f; }
float pow_2_f32(float x) { return powf(x, 2.0f); }
float arctan2_3_f32(float x) { return atan2f(x, 3.0f); }

// ============================================================================
// SCALAR CONSTANT BUG TESTS
// ============================================================================

int test_scalar_constant(const char *description, const char *expr_str,
                          float (*expected_fn)(float)) {
    printf("\n%s\n", description);
    printf("====================================\n");
    printf("Testing: %s\n", expr_str);

    me_variable vars[] = {{"a", ME_FLOAT32}};
    int err;
    // Use ME_AUTO - following NumPy conventions, float constants match variable type (FLOAT32)
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, vars, 1, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ COMPILATION FAILED (error %d)\n", err);
        return 0;
    }

    // Following NumPy conventions, float constants match variable type, so result is FLOAT32
    float input[SMALL_SIZE];
    for (int i = 0; i < SMALL_SIZE; i++) input[i] = (float)i;

    const void *var_ptrs[] = {input};
    float result[SMALL_SIZE];
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, SMALL_SIZE);

    int passed = 1;
    printf("  Input     Result    Expected  Status\n");
    for (int i = 0; i < SMALL_SIZE; i++) {
        float expected = expected_fn(input[i]);
        float result_val = result[i];
        float diff = fabsf(result_val - expected);
        char status = (diff < 1e-5f) ? 'Y' : 'N';

        printf("  %8.3f  %8.3f  %8.3f  %c", input[i], result_val, expected, status);
        if (diff >= 1e-5f) {
            printf(" (diff: %.6f)", diff);
            passed = 0;
        }
        printf("\n");
    }

    me_free(expr);

    if (passed) {
        printf("  ✅ PASS\n");
    } else {
        printf("  ❌ FAIL\n");
    }

    return passed;
}

float mul_5_f32(float x) { return x * 5.0f; }
float sub_2_f32(float x) { return x - 2.0f; }
float div_4_f32(float x) { return x / 4.0f; }

// ============================================================================
// LARGE INT64 + FLOAT CONSTANT TEST
// ============================================================================

int test_int64_large_constant(const char *description, int size) {
    printf("\n%s\n", description);
    printf("======================================================================\n");

    // Create an int64 array of the requested size and fill with small increasing values.
    // Using small integers ensures integer -> floating conversions are not lossy for the
    // integer operand itself; the potential issue comes from the large floating constant.
    long long *input = malloc(size * sizeof(long long));
    if (!input) {
        printf("  ❌ FAILED: malloc failed\n");
        return 0;
    }
    for (int i = 0; i < size; i++) {
        input[i] = (long long)i;
    }

    // Compile expression. We use ME_AUTO so mixed-type rules are applied and the
    // compiler decides the result dtype.
    me_variable vars[] = {{"a", ME_INT64}};
    int err;
    const char *expr_str = "(a + 90000.00001) + 1";
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, vars, 1, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ COMPILATION FAILED (error %d)\n", err);
        free(input);
        return 0;
    }

    me_dtype out_dtype = me_get_dtype(expr);
    printf("  Compiled expression: %s\n", expr_str);
    printf("  Inferred output dtype: %d\n", out_dtype);

    int passed = 1;
    double max_diff = 0.0;

    // Evaluate depending on inferred output dtype. The expression contains a
    // floating-point constant, so a floating output dtype is expected.
    const void *var_ptrs[] = {input};

    if (out_dtype == ME_FLOAT64) {
        double *result = malloc(size * sizeof(double));
        if (!result) {
            printf("  ❌ FAILED: malloc for result failed\n");
            me_free(expr);
            free(input);
            return 0;
        }
        ME_EVAL_CHECK(expr, var_ptrs, 1, result, size);

        // Compute expected values using double arithmetic.
        for (int i = 0; i < size; i++) {
            double expected = ((double)input[i] + 90000.00001) + 1.0;
            double diff = fabs(result[i] - expected);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-9) passed = 0; // tight tolerance for this check
        }

        printf("  Result (first 5):   ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.9f ", result[i]);
        printf("...\n");

        printf("  Expected (first 5): ");
        for (int i = 0; i < 5 && i < size; i++) {
            double expected = ((double)input[i] + 90000.00001) + 1.0;
            printf("%.9f ", expected);
        }
        printf("...\n");

        free(result);

    } else if (out_dtype == ME_FLOAT32) {
        float *result = malloc(size * sizeof(float));
        if (!result) {
            printf("  ❌ FAILED: malloc for result failed\n");
            me_free(expr);
            free(input);
            return 0;
        }
        ME_EVAL_CHECK(expr, var_ptrs, 1, result, size);

        for (int i = 0; i < size; i++) {
            float expected = (float)(((double)input[i] + 90000.00001) + 1.0);
            float diff = fabsf(result[i] - expected);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-5f) passed = 0; // looser tolerance for float32
        }

        printf("  Result (first 5):   ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.7f ", result[i]);
        printf("...\n");

        printf("  Expected (first 5): ");
        for (int i = 0; i < 5 && i < size; i++) {
            float expected = (float)(((double)input[i] + 90000.00001) + 1.0);
            printf("%.7f ", expected);
        }
        printf("...\n");

        free(result);

    } else {
        // Unexpected output dtype: try to evaluate into a double buffer and compare
        // raw integer or other outputs as a conservative fallback.
        printf("  ⚠️  Unexpected output dtype (%d). Attempting double evaluation for comparison.\n", out_dtype);
        double *result = malloc(size * sizeof(double));
        if (!result) {
            printf("  ❌ FAILED: malloc for fallback result failed\n");
            me_free(expr);
            free(input);
            return 0;
        }
        ME_EVAL_CHECK(expr, var_ptrs, 1, result, size);

        for (int i = 0; i < size; i++) {
            double expected = ((double)input[i] + 90000.00001) + 1.0;
            double diff = fabs(result[i] - expected);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-9) passed = 0;
        }

        free(result);
    }

    me_free(expr);
    free(input);

    if (passed) {
        printf("  ✅ PASS\n");
    } else {
        printf("  ❌ FAIL (max diff: %.12f)\n", max_diff);
    }

    // The caller expects this test to surface the reported problem; return the
    // actual pass/fail so it shows up in the overall summary. The external app
    // that reported the issue used this expression and observed incorrect
    // behaviour, so a failure here indicates the bug is present.
    return passed;
}

// ============================================================================
// FLOAT32 ARRAY + FLOAT64 CONSTANTS TEST
// ============================================================================

int test_float32_array_float64_constants(const char *description, int size) {
    printf("\n%s\n", description);
    printf("======================================================================\n");

    // Create a float32 array for o0
    float *input = malloc(size * sizeof(float));
    if (!input) {
        printf("  ❌ FAILED: malloc failed\n");
        return 0;
    }
    // Fill with some test values
    for (int i = 0; i < size; i++) {
        input[i] = (float)(i * 0.1f);
    }

    // Compile expression with ME_AUTO to let type inference work
    me_variable vars[] = {{"o0", ME_FLOAT32}};
    int err;
    const char *expr_str = "((o0 + 1067.3366832990887) + 0.2901221513748169)";
    me_expr *expr = NULL;
    int rc_expr = me_compile(expr_str, vars, 1, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ COMPILATION FAILED (error %d)\n", err);
        free(input);
        return 0;
    }

    me_dtype out_dtype = me_get_dtype(expr);
    printf("  Compiled expression: %s\n", expr_str);
    printf("  Inferred output dtype: %s\n",
           out_dtype == ME_FLOAT32 ? "ME_FLOAT32" :
           out_dtype == ME_FLOAT64 ? "ME_FLOAT64" : "OTHER");
    printf("  Expected output dtype: ME_FLOAT32 (NumPy convention - constants match variable type)\n");
    printf("  Note: High-precision constants will be converted to FLOAT32, may lose precision\n");

    int passed = 1;
    float max_diff = 0.0f;
#ifdef __FAST_MATH__
    const float tolerance = 1e-4f;
#else
    const float tolerance = 1e-5f;
#endif

    const void *var_ptrs[] = {input};

    // Following NumPy conventions, float constants match variable type, so output should be ME_FLOAT32
    if (out_dtype == ME_FLOAT32) {
        float *result = malloc(size * sizeof(float));
        if (!result) {
            printf("  ❌ FAILED: malloc for result failed\n");
            me_free(expr);
            free(input);
            return 0;
        }
        ME_EVAL_CHECK(expr, var_ptrs, 1, result, size);

        // Compute expected values using float32 arithmetic (NumPy behavior - constants converted to float32)
        for (int i = 0; i < size; i++) {
            float expected = ((float)input[i] + (float)1067.3366832990887) + (float)0.2901221513748169;
            float diff = fabsf(result[i] - expected);
            if (diff > max_diff) max_diff = diff;
            if (diff > tolerance) passed = 0; // tolerance for float32
        }

        printf("  Result (first 5):   ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.7f ", result[i]);
        printf("...\n");

        printf("  Expected (first 5): ");
        for (int i = 0; i < 5 && i < size; i++) {
            float expected = ((float)input[i] + (float)1067.3366832990887) + (float)0.2901221513748169;
            printf("%.7f ", expected);
        }
        printf("...\n");

        free(result);

    } else {
        printf("  ❌ FAILED: Expected output dtype ME_FLOAT32 (NumPy convention), got %d\n", out_dtype);
        me_free(expr);
        free(input);
        return 0;
    }

    me_free(expr);
    free(input);

    if (passed) {
        printf("  ✅ PASS\n");
    } else {
        printf("  ❌ FAIL (max diff: %.9f)\n", max_diff);
    }

    return passed;
}

// ============================================================================
// CONJ WITH REAL INPUT SHOULD PRESERVE FLOAT32
// ============================================================================
int test_conj_real_preserves_dtype() {
    printf("\nTest: conj(a) with float32 input returns float32 output\n");
    printf("======================================================================\n");

    float a[SMALL_SIZE] = {1.0f, -2.5f, 3.75f, -4.125f, 0.0f, 5.5f, -6.25f, 7.0f, -8.5f, 9.0f};
    float result[SMALL_SIZE] = {0};

    me_variable vars[] = {{"a", ME_FLOAT32}};

    int err;
    me_expr *expr = NULL;
    int rc_expr = me_compile("conj(a)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ COMPILATION FAILED at position %d\n", err);
        return 0;
    }

    const void *var_ptrs[] = {a};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, SMALL_SIZE);

    int passed = 1;
    float max_diff = 0.0f;
    for (int i = 0; i < SMALL_SIZE; i++) {
        float expected = a[i]; // conj on real input is identity; stays float32
        float diff = fabsf(expected - result[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6f) {
            passed = 0;
        }
    }

    if (passed) {
        printf("  ✅ PASS\n");
    } else {
        printf("  ❌ FAIL (max diff: %.9f)\n", max_diff);
    }

    me_free(expr);
    return passed;
}

// ============================================================================
// REDUCTION + COMPARISON (EXPLICIT OUTPUT DTYPE)
// ============================================================================

static int test_sum_comparison_explicit_output(void) {
    printf("\n=== Regression: sum(x != 0) with explicit output dtype ===\n");

    int32_t data[] = {0, 1, 2, 0, 3, 0, 4, 5};
    const int nitems = (int)(sizeof(data) / sizeof(data[0]));
    me_variable vars[] = {{"x", ME_INT32}};
    const void *var_ptrs[] = {data};
    int err = 0;
    me_expr *expr = NULL;

    int rc_expr = me_compile("sum(x != 0)", vars, 1, ME_INT64, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 0;
    }

    int64_t output = 0;
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, nitems);

    int passed = 1;
    if (output != 5) {
        printf("  ❌ FAILED: expected 5, got %lld\n", (long long)output);
        passed = 0;
    } else {
        printf("  ✅ PASS\n");
    }

    me_free(expr);
    return passed;
}

static int test_log_int_promotes_output(void) {
    printf("\nTest: log(int) promotes to float output (ME_AUTO)\n");
    printf("======================================================================\n");

    int32_t data[] = {1, 2, 3, 4, 5};
    const int nitems = (int)(sizeof(data) / sizeof(data[0]));
    me_variable vars[] = {{"x", ME_INT32}};
    const void *var_ptrs[] = {data};
    int err = 0;
    me_expr *expr = NULL;

    int rc_expr = me_compile("log(x)", vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  ❌ FAILED: compilation error %d\n", err);
        return 0;
    }

    me_dtype dtype = me_get_dtype(expr);
    if (dtype != ME_FLOAT64) {
        printf("  ❌ FAILED: expected output dtype ME_FLOAT64, got %d\n", dtype);
        me_free(expr);
        return 0;
    }

    double output[nitems];
    ME_EVAL_CHECK(expr, var_ptrs, 1, output, nitems);

    int passed = 1;
    double max_diff = 0.0;
    const double tolerance = 1e-12;
    for (int i = 0; i < nitems; i++) {
        double expected = log((double)data[i]);
        double diff = fabs(output[i] - expected);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            passed = 0;
        }
    }

    if (passed) {
        printf("  ✅ PASS\n");
    } else {
        printf("  ❌ FAIL (max diff: %.12g)\n", max_diff);
    }

    me_free(expr);
    return passed;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    printf("========================================================================\n");
    printf("MINIEXPR REGRESSION TEST SUITE\n");
    printf("========================================================================\n");
    printf("This combines tests from multiple bug fix verification modules:\n");
    printf("  - arctan2 with array/scalar operands\n");
    printf("  - arctan2 with complex expressions\n");
    printf("  - constant type inference\n");
    printf("  - scalar constant operations\n");
    printf("  - conj on real inputs preserves dtype\n");
    printf("  - sum(comparison) with explicit output dtype\n");
    printf("  - log(int) promotes to float output\n");
    printf("========================================================================\n");

    int total = 0;
    int passed = 0;

    // ========================================================================
    // ARCTAN2 ARRAY-SCALAR TESTS
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 1: ARCTAN2 ARRAY-SCALAR TESTS\n");
    printf("========================================================================\n");

    total++;
    if (test_arctan2_with_scalar_constant("Test 1.1: Small array (10 elements) + scalar 3.0",
                                           SMALL_SIZE, 3.0f)) passed++;

    total++;
    if (test_arctan2_with_two_arrays("Test 1.2: Two arrays (same data as Test 1.1)",
                                      SMALL_SIZE, 3.0f)) passed++;

    total++;
    if (test_arctan2_with_scalar_constant("Test 1.3: Larger array (100 elements) + scalar 3.0",
                                           LARGE_SIZE, 3.0f)) passed++;

    total++;
    if (test_arctan2_with_scalar_constant("Test 1.4: Small array + scalar 0.5",
                                           SMALL_SIZE, 0.5f)) passed++;

    total++;
    if (test_arctan2_with_scalar_constant("Test 1.5: Small array + scalar 10.0",
                                           SMALL_SIZE, 10.0f)) passed++;

    // ========================================================================
    // REDUCTION + COMPARISON TEST
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 2: REDUCTION + COMPARISON\n");
    printf("========================================================================\n");

    total++;
    if (test_sum_comparison_explicit_output()) passed++;

    total++;
    if (test_log_int_promotes_output()) passed++;

    // ========================================================================
    // ARCTAN2 BUG TESTS
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 3: ARCTAN2 MIXED ARRAY/SCALAR OPERAND TESTS\n");
    printf("========================================================================\n");

    double y_data[CHUNK_SIZE] = {0.0, 1.0, -1.0, 2.0, -2.0};
    double x_data[CHUNK_SIZE] = {1.0, 2.0, -1.0, 0.5, -2.0};

    total++;
    if (test_arctan2_array_scalar_f64("Test 2.1: arctan2(y, 1.0) where y is an array",
                                      "arctan2(y, 1.0)", y_data, 1.0, 0)) passed++;

    total++;
    if (test_arctan2_array_scalar_f64("Test 2.2: arctan2(1.0, x) where x is an array",
                                      "arctan2(1.0, x)", x_data, 1.0, 1)) passed++;

    double x_data2[CHUNK_SIZE] = {1.0, 2.0, 3.0, -2.0, 0.5};
    total++;
    if (test_pow_array_scalar_f64("Test 2.3: pow(x, 2.0) where x is an array",
                                  "pow(x, 2.0)", x_data2, 2.0, 0)) passed++;

    double x_data3[CHUNK_SIZE] = {0.0, 1.0, 2.0, 3.0, -1.0};
    total++;
    if (test_pow_array_scalar_f64("Test 2.4: pow(2.0, x) where x is an array",
                                  "pow(2.0, x)", x_data3, 2.0, 1)) passed++;

    // ========================================================================
    // ARCTAN2 COMPLEX EXPRESSION TESTS
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 4: ARCTAN2 WITH COMPLEX EXPRESSIONS\n");
    printf("========================================================================\n");

    double x_data4[CHUNK_SIZE] = {0.0, 1.0, 2.0, -1.0, 0.5};
    double y_data4[CHUNK_SIZE] = {0.0, 0.0, -1.0, 1.0, 0.5};

    total++;
    if (test_arctan2_complex_expr("Test 3.1: arctan2(x+y, 1.0)",
                                  "arctan2(x+y, 1.0)", x_data4, y_data4,
                                  arctan2_x_plus_y_1)) passed++;

    double x_data5[CHUNK_SIZE] = {1.0, 2.0, -1.0, 0.5, -2.0};
    double y_data5[CHUNK_SIZE] = {0.0, -1.0, 1.0, 0.5, 1.0};

    total++;
    if (test_arctan2_complex_expr("Test 3.2: arctan2(1.0, x+y)",
                                  "arctan2(1.0, x+y)", x_data5, y_data5,
                                  arctan2_1_x_plus_y)) passed++;

    // ========================================================================
    // CONSTANT TYPE INFERENCE TESTS
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 5: CONSTANT TYPE INFERENCE TESTS\n");
    printf("========================================================================\n");

    total++;
    if (test_constant_type_f32("Test 4.1: FLOAT32 variable + constant, output=ME_AUTO",
                               "a + 3.0", add_3_f32)) passed++;

    total++;
    if (test_constant_type_f32("Test 4.2: FLOAT32 variable ** constant, output=ME_AUTO",
                               "a ** 2.0", pow_2_f32)) passed++;

    total++;
    if (test_constant_type_f32("Test 4.3: FLOAT32 in arctan2(a, constant), output=ME_AUTO",
                               "arctan2(a, 3.0)", arctan2_3_f32)) passed++;

    // ========================================================================
    // SCALAR CONSTANT BUG TESTS
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 6: SCALAR CONSTANT OPERATIONS\n");
    printf("========================================================================\n");

    total++;
    if (test_scalar_constant("Test 5.1: a + 3", "a + 3", add_3_f32)) passed++;

    total++;
    if (test_scalar_constant("Test 5.2: a ** 2", "a ** 2", pow_2_f32)) passed++;

    total++;
    if (test_scalar_constant("Test 5.3: arctan2(a, 3.0)", "arctan2(a, 3.0)", arctan2_3_f32)) passed++;

    total++;
    if (test_scalar_constant("Test 5.4: a * 5", "a * 5", mul_5_f32)) passed++;

    total++;
    if (test_scalar_constant("Test 5.5: a - 2", "a - 2", sub_2_f32)) passed++;

    total++;
    if (test_scalar_constant("Test 5.6: a / 4", "a / 4", div_4_f32)) passed++;

    // ========================================================================
    // SECTION 7: LARGE INT64 + FLOAT CONSTANT (expected to fail)
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 7: LARGE INT64 + FLOAT CONSTANT\n");
    printf("========================================================================\n");

    total++;
    if (test_int64_large_constant("Test 6.1: (a + 90000.00001) + 1 where a is int64[1000]",
                                  1000)) passed++;

    // ========================================================================
    // SECTION 8: FLOAT32 ARRAY + FLOAT64 CONSTANTS
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 8: FLOAT32 ARRAY + FLOAT64 CONSTANTS\n");
    printf("========================================================================\n");

    total++;
    if (test_float32_array_float64_constants("Test 7.1: ((o0 + 1067.3366832990887) + 0.2901221513748169) where o0 is float32 array",
                                             SMALL_SIZE)) passed++;

    // ========================================================================
    // SECTION 9: CONJ ON REAL INPUTS (FLOAT32)
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("SECTION 9: CONJ ON REAL INPUTS (FLOAT32)\n");
    printf("========================================================================\n");

    total++;
    if (test_conj_real_preserves_dtype()) passed++;

    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    printf("\n\n========================================================================\n");
    printf("FINAL RESULTS\n");
    printf("========================================================================\n");
    printf("Total tests: %d\n", total);
    printf("Passed:      %d\n", passed);
    printf("Failed:      %d\n", total - passed);
    printf("========================================================================\n");

    if (passed == total) {
        printf("✅ ALL TESTS PASSED\n");
    } else {
        printf("❌ SOME TESTS FAILED\n");
    }

    return (passed == total) ? 0 : 1;
}
