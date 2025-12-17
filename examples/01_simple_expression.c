/*
 * Example 1: Simple Expression
 *
 * Demonstrates basic usage with a simple arithmetic expression.
 * Computes: (x + y) * 2 for arrays of values.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

int main() {
    printf("=== Simple Expression Example ===\n");
    printf("Expression: (x + y) * 2\n\n");

    // Input data
    const int n = 5;
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    double result[5];

    // Define variables (just names)
    me_variable vars[] = {{"x"}, {"y"}};

    // Compile the expression
    int error;
    me_expr *expr = me_compile("(x + y) * 2", vars, 2, ME_FLOAT64, &error);

    if (!expr) {
        printf("ERROR: Failed to compile expression at position %d\n", error);
        return 1;
    }

    // Prepare variable pointers
    const void *var_ptrs[] = {x, y};

    // Evaluate
    me_eval(expr, var_ptrs, 2, result, n);

    // Display results
    printf("Results:\n");
    printf("  x     y     (x+y)*2\n");
    printf("----  ----  ---------\n");
    for (int i = 0; i < n; i++) {
        printf("%4.0f  %4.0f  %9.0f\n", x[i], y[i], result[i]);
    }

    // Cleanup
    me_free(expr);

    printf("\n✅ Simple expression evaluation complete!\n");
    return 0;
}
