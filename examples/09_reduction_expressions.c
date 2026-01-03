/*
 * Example 9: Reductions in Expressions
 *
 * Demonstrates reductions over expressions and using reductions inside
 * larger expressions.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"
#include "minctest.h"


int main() {
    printf("=== Reduction Expressions Example ===\n\n");

    const int n = 4;
    double x[] = {1.0, 2.0, 3.0, 4.0};

    me_variable vars[] = {{"x", ME_FLOAT64, x}};
    const void* var_ptrs[] = {x};

    {
        printf("Expression: sum(x + 1)\n");
        double sum_out = 0.0;
        int error = 0;
        me_expr* expr = NULL;

        if (me_compile("sum(x + 1)", vars, 1, ME_AUTO, &error, &expr) != ME_COMPILE_SUCCESS) {
            printf("ERROR: Failed to compile expression at position %d\n", error);
            return 1;
        }

        ME_EVAL_CHECK(expr, var_ptrs, 1, &sum_out, n);
        printf("Result: %.2f\n\n", sum_out);

        me_free(expr);
    }

    {
        printf("Expression: x + sum(x)\n");
        double result[4] = {0};
        int error = 0;
        me_expr* expr = NULL;

        if (me_compile("x + sum(x)", vars, 1, ME_AUTO, &error, &expr) != ME_COMPILE_SUCCESS) {
            printf("ERROR: Failed to compile expression at position %d\n", error);
            return 1;
        }

        ME_EVAL_CHECK(expr, var_ptrs, 1, result, n);

        printf("Results:\n");
        printf("  x     x+sum(x)\n");
        printf("----  ---------\n");
        for (int i = 0; i < n; i++) {
            printf("%4.0f  %9.0f\n", x[i], result[i]);
        }

        me_free(expr);
    }

    printf("\n✅ Reduction expression evaluation complete!\n");
    return 0;
}
