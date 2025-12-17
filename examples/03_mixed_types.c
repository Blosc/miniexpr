/*
 * Example 3: Mixed Types with Automatic Inference
 *
 * Demonstrates type promotion and ME_AUTO for automatic type inference
 * when mixing different data types (int32 and float64).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../src/miniexpr.h"

int main() {
    printf("=== Mixed Types Example ===\n");
    printf("Expression: a + b\n");
    printf("Types: a=int32, b=float64\n\n");

    // Input arrays with different types
    const int n = 5;
    int32_t a[] = {1, 2, 3, 4, 5}; // Integer
    double b[] = {0.1, 0.2, 0.3, 0.4, 0.5}; // Float
    double result[5];

    // Define variables with explicit types
    me_variable vars[] = {
        {"a", ME_INT32},
        {"b", ME_FLOAT64}
    };

    // Compile with ME_AUTO to infer output type from inputs
    int error;
    me_expr *expr = me_compile("a + b", vars, 2, ME_AUTO, &error);

    if (!expr) {
        printf("ERROR: Failed to compile at position %d\n", error);
        return 1;
    }

    printf("Inferred result type: ");
    me_dtype result_dtype = me_get_dtype(expr);
    switch (result_dtype) {
        case ME_INT32: printf("ME_INT32\n");
            break;
        case ME_FLOAT64: printf("ME_FLOAT64\n");
            break;
        default: printf("%d\n", result_dtype);
    }
    printf("\n");

    // Prepare variable pointers
    const void *var_ptrs[] = {a, b};

    // Evaluate (integers will be promoted to float64)
    me_eval(expr, var_ptrs, 2, result, n);

    // Display results
    printf("Results (int32 promoted to float64):\n");
    printf("   a    b      a+b\n");
    printf("  ---  ---   ------\n");
    for (int i = 0; i < n; i++) {
        printf("  %3d  %.1f   %6.1f\n", a[i], b[i], result[i]);
    }

    // Cleanup
    me_free(expr);

    printf("\n✅ Mixed type promotion complete!\n");
    return 0;
}
