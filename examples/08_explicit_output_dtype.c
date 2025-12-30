/*
 * Example 8: Explicit Variable Types with Explicit Output Dtype
 *
 * Demonstrates how to specify both explicit variable types and an explicit
 * output dtype. This is useful when you want to:
 * - Keep variable types during computation (heterogeneous types)
 * - Cast the final result to a specific output type
 *
 * This is different from ME_AUTO, which infers the output type from the
 * expression computation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "../src/miniexpr.h"
#include "minctest.h"



int main() {
    printf("=== Example 8: Explicit Variable Types with Explicit Output Dtype ===\n\n");

    const int n = 5;

    // Example 1: Mixed types (INT32 + FLOAT64) with explicit FLOAT32 output
    printf("Example 1: Mixed types with explicit output casting\n");
    printf("----------------------------------------------------\n");
    printf("Expression: a + b\n");
    printf("Types: a=INT32, b=FLOAT64, output=FLOAT32\n");
    printf("Behavior: Variables keep their types, result is cast to FLOAT32\n\n");

    int32_t a[] = {10, 20, 30, 40, 50};
    double b[] = {1.5, 2.5, 3.5, 4.5, 5.5};
    float result_f32[5];

    me_variable vars[] = {
        {"a", ME_INT32},
        {"b", ME_FLOAT64}
    };

    int error;
    // Explicit variable types + explicit output dtype
    me_expr *expr1 = me_compile("a + b", vars, 2, ME_FLOAT32, &error);

    if (!expr1) {
        printf("ERROR: Failed to compile at position %d\n", error);
        return 1;
    }

    me_dtype output_dtype = me_get_dtype(expr1);
    printf("Output dtype: %s\n",
           output_dtype == ME_FLOAT32 ? "ME_FLOAT32 ✓" : "OTHER ✗");
    printf("\n");

    const void *var_ptrs[] = {a, b};
    ME_EVAL_CHECK(expr1, var_ptrs, 2, result_f32, n);

    printf("Results (computed in FLOAT64, cast to FLOAT32):\n");
    printf("   a    b      a+b (float32)\n");
    printf("  ---  ---   ---------------\n");
    for (int i = 0; i < n; i++) {
        printf("  %3d  %.1f   %8.2f\n", a[i], b[i], result_f32[i]);
    }

    me_free(expr1);

    // Example 2: FLOAT32 variables with FLOAT64 output
    printf("\n\nExample 2: FLOAT32 variables with FLOAT64 output\n");
    printf("----------------------------------------------------\n");
    printf("Expression: x * 2.5 + y\n");
    printf("Types: x=FLOAT32, y=FLOAT32, output=FLOAT64\n");
    printf("Behavior: Variables stay FLOAT32, result is cast to FLOAT64\n\n");

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
    double result_f64[5];

    me_variable vars2[] = {
        {"x", ME_FLOAT32},
        {"y", ME_FLOAT32}
    };

    // FLOAT32 variables but FLOAT64 output
    me_expr *expr2 = me_compile("x * 2.5 + y", vars2, 2, ME_FLOAT64, &error);

    if (!expr2) {
        printf("ERROR: Failed to compile at position %d\n", error);
        return 1;
    }

    output_dtype = me_get_dtype(expr2);
    printf("Output dtype: %s\n",
           output_dtype == ME_FLOAT64 ? "ME_FLOAT64 ✓" : "OTHER ✗");
    printf("\n");

    const void *var_ptrs2[] = {x, y};
    ME_EVAL_CHECK(expr2, var_ptrs2, 2, result_f64, n);

    printf("Results (computed in FLOAT32, cast to FLOAT64):\n");
    printf("   x    y      x*2.5+y (float64)\n");
    printf("  ---  ---   -------------------\n");
    for (int i = 0; i < n; i++) {
        printf("  %.1f  %.1f   %12.6f\n", x[i], y[i], result_f64[i]);
    }

    me_free(expr2);

    // Example 3: Comparison with explicit output
    printf("\n\nExample 3: Comparison with explicit output dtype\n");
    printf("----------------------------------------------------\n");
    printf("Expression: a > b\n");
    printf("Types: a=INT32, b=INT32, output=BOOL\n");
    printf("Behavior: Comparison computed, result is BOOL\n\n");

    int32_t a2[] = {10, 5, 15, 8, 20};
    int32_t b2[] = {5, 10, 10, 8, 15};
    bool result_bool[5];

    me_variable vars3[] = {
        {"a", ME_INT32},
        {"b", ME_INT32}
    };

    // Explicit BOOL output for comparison
    me_expr *expr3 = me_compile("a > b", vars3, 2, ME_BOOL, &error);

    if (!expr3) {
        printf("ERROR: Failed to compile at position %d\n", error);
        return 1;
    }

    output_dtype = me_get_dtype(expr3);
    printf("Output dtype: %s\n",
           output_dtype == ME_BOOL ? "ME_BOOL ✓" : "OTHER ✗");
    printf("\n");

    const void *var_ptrs3[] = {a2, b2};
    ME_EVAL_CHECK(expr3, var_ptrs3, 2, result_bool, n);

    printf("Results:\n");
    printf("   a    b      a > b\n");
    printf("  ---  ---   -------\n");
    for (int i = 0; i < n; i++) {
        printf("  %3d  %3d   %s\n", a2[i], b2[i], result_bool[i] ? "true" : "false");
    }

    me_free(expr3);

    printf("\n✅ Examples complete!\n");
    printf("\nKey takeaway: When you specify both variable types and output dtype,\n");
    printf("variables keep their types during computation, and the result is cast\n");
    printf("to your specified output type. This is useful for:\n");
    printf("  - Memory efficiency (compute in FLOAT32, output as needed)\n");
    printf("  - Type safety (explicit control over output type)\n");
    printf("  - Heterogeneous inputs with specific output requirements\n");

    return 0;
}

