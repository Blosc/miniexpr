/*
 * Example 6: Using me_print() for Debugging
 *
 * Demonstrates how to use me_print() to visualize expression trees
 * for debugging and understanding how expressions are parsed.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

void compile_and_print(const char *expression, me_variable *vars, int nvars, me_dtype dtype) {
    printf("\nExpression: %s\n", expression);
    printf("Tree structure:\n");

    int error;
    me_expr *expr = me_compile_chunk(expression, vars, nvars, dtype, &error);

    if (!expr) {
        printf("  ERROR: Failed to compile at position %d\n", error);
        return;
    }

    me_print(expr);
    me_free(expr);
}

int main() {
    printf("=== Expression Tree Visualization Example ===\n");
    printf("Using me_print() to see how expressions are parsed\n");

    me_variable vars1[] = {{"x"}};
    me_variable vars2[] = {{"x"}, {"y"}};
    me_variable vars3[] = {{"a"}, {"b"}, {"c"}};

    // Simple arithmetic
    compile_and_print("x + 5", vars1, 1, ME_FLOAT64);

    // Two variables
    compile_and_print("x * y", vars2, 2, ME_FLOAT64);

    // Nested operations
    compile_and_print("(a + b) * c", vars3, 3, ME_FLOAT64);

    // Function calls
    compile_and_print("sqrt(x)", vars1, 1, ME_FLOAT64);

    // Complex expression
    compile_and_print("sin(x) + cos(x)", vars1, 1, ME_FLOAT64);

    // Multiple levels of nesting
    compile_and_print("sqrt(x*x + y*y)", vars2, 2, ME_FLOAT64);

    printf("\n=== Tree Node Legend ===\n");
    printf("f0, f1, f2, ...  - Functions with N arguments\n");
    printf("bound <address>  - Variable reference\n");
    printf("<number>         - Constant value\n");
    printf("\nThe tree is displayed in pre-order traversal\n");
    printf("with indentation showing nesting level.\n");

    printf("\n✅ Tree visualization complete!\n");
    printf("Use me_print() to debug complex expressions!\n");

    return 0;
}
