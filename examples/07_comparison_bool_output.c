/**
 * Example 07: Comparison Operators with Boolean Output
 *
 * This example demonstrates how to use comparison operators (==, <, >, <=, >=, !=)
 * in expressions and get boolean (true/false) results.
 *
 * Key concepts:
 * - Comparisons with arithmetic expressions (e.g., a**2 == b)
 * - Getting bool output arrays from comparisons
 * - Using ME_AUTO to infer ME_BOOL for comparison expressions
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "miniexpr.h"
#include "minctest.h"


#define N 10

int main() {
    printf("=== Example 07: Comparison Operators with Boolean Output ===\n\n");

    /* Sample data where a**2 == (a + b) for all elements */
    double a[] = {2.0, 3.0, 4.0, 5.0, 1.0, 0.0, -2.0, 6.0, 2.5, 3.5};
    double b[] = {2.0, 6.0, 12.0, 20.0, 0.0, 0.0, 6.0, 30.0, 3.75, 8.75};

    int err;

    /*
     * Example 1: Comparison with explicit ME_BOOL output
     *
     * When you want bool output from a comparison expression,
     * specify explicit variable dtypes and use ME_BOOL for output.
     */
    printf("Example 1: a ** 2 == (a + b) with ME_BOOL output\n");
    printf("--------------------------------------------------\n");
    {
        bool result[N];

        /* Specify that input variables are ME_FLOAT64 */
        me_variable vars[] = {
            {"a", ME_FLOAT64},
            {"b", ME_FLOAT64}
        };

        /* Request ME_BOOL output */
        me_expr* expr = NULL;
        int rc_expr = me_compile("a ** 2 == (a + b)", vars, 2, ME_BOOL, &err, &expr);

        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }

        const void* ptrs[] = {a, b};
        ME_EVAL_CHECK(expr, ptrs, 2, result, N);

        printf("  a     | a**2   | a+b    | a**2 == (a+b)\n");
        printf("  ------|--------|--------|---------------\n");
        for (int i = 0; i < N; i++) {
            printf("  %5.1f | %6.2f | %6.2f | %s\n",
                   a[i], a[i] * a[i], a[i] + b[i],
                   result[i] ? "true" : "false");
        }

        me_free(expr);
    }

    /*
     * Example 2: Using ME_AUTO to infer ME_BOOL
     *
     * When you use ME_AUTO for output dtype, miniexpr automatically
     * infers ME_BOOL for comparison expressions.
     */
    printf("\nExample 2: a < b with ME_AUTO (auto-infers ME_BOOL)\n");
    printf("----------------------------------------------------\n");
    {
        bool result[N];
        double x[] = {1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 9.0, 0.0};
        double y[] = {2.0, 4.0, 4.0, 6.0, 3.0, 7.0, 5.0, 5.0, 10.0, 1.0};

        /* For ME_AUTO output, variables must have explicit dtypes */
        me_variable vars[] = {
            {"x", ME_FLOAT64},
            {"y", ME_FLOAT64}
        };

        /* Use ME_AUTO - will infer ME_BOOL for comparison */
        me_expr* expr = NULL;
        int rc_expr = me_compile("x < y", vars, 2, ME_AUTO, &err, &expr);

        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }

        /* Verify the inferred type is ME_BOOL */
        me_dtype inferred = me_get_dtype(expr);
        printf("  Inferred output dtype: %s\n\n",
               inferred == ME_BOOL ? "ME_BOOL" : "other");

        const void* ptrs[] = {x, y};
        ME_EVAL_CHECK(expr, ptrs, 2, result, N);

        printf("  x     | y     | x < y\n");
        printf("  ------|-------|-------\n");
        for (int i = 0; i < N; i++) {
            printf("  %5.1f | %5.1f | %s\n",
                   x[i], y[i], result[i] ? "true" : "false");
        }

        me_free(expr);
    }

    /*
     * Example 3: Complex comparison with power operations
     *
     * Demonstrates Pythagorean theorem check: a² + b² == c²
     */
    printf("\nExample 3: Pythagorean theorem check (a**2 + b**2 == c**2)\n");
    printf("-----------------------------------------------------------\n");
    {
        bool result[5];

        /* Classic Pythagorean triples and some non-triples */
        double side_a[] = {3.0, 5.0, 8.0, 7.0, 9.0};
        double side_b[] = {4.0, 12.0, 15.0, 24.0, 12.0};
        double side_c[] = {5.0, 13.0, 17.0, 25.0, 16.0};

        me_variable vars[] = {
            {"a", ME_FLOAT64},
            {"b", ME_FLOAT64},
            {"c", ME_FLOAT64}
        };

        me_expr* expr = NULL;
        int rc_expr = me_compile("a**2 + b**2 == c**2", vars, 3, ME_BOOL, &err, &expr);

        if (rc_expr != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }

        const void* ptrs[] = {side_a, side_b, side_c};
        ME_EVAL_CHECK(expr, ptrs, 3, result, 5);

        printf("  a  | b   | c   | a²+b² | c²    | Is Pythagorean?\n");
        printf("  ---|-----|-----|-------|-------|----------------\n");
        for (int i = 0; i < 5; i++) {
            double a2_b2 = side_a[i] * side_a[i] + side_b[i] * side_b[i];
            double c2 = side_c[i] * side_c[i];
            printf("  %2.0f | %3.0f | %3.0f | %5.0f | %5.0f | %s\n",
                   side_a[i], side_b[i], side_c[i],
                   a2_b2, c2, result[i] ? "YES" : "no");
        }

        me_free(expr);
    }

    /*
     * Example 4: Multiple comparison operators
     *
     * Shows different comparison operators available.
     */
    printf("\nExample 4: Various comparison operators\n");
    printf("-----------------------------------------\n");
    {
        double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        bool result[5];

        me_variable vars[] = {{"x", ME_FLOAT64}};
        const void* ptrs[] = {vals};

        const char* operators[] = {"x < 3", "x <= 3", "x == 3", "x >= 3", "x > 3", "x != 3"};

        printf("  Values: [1, 2, 3, 4, 5]\n\n");
        printf("  Expression | Results\n");
        printf("  -----------|--------------------\n");

        for (int op = 0; op < 6; op++) {
            me_expr* expr = NULL;
            int rc_expr = me_compile(operators[op], vars, 1, ME_BOOL, &err, &expr);
            if (rc_expr == ME_COMPILE_SUCCESS) {
                ME_EVAL_CHECK(expr, ptrs, 1, result, 5);
                printf("  %-10s | ", operators[op]);
                for (int i = 0; i < 5; i++) {
                    printf("%s ", result[i] ? "T" : "F");
                }
                printf("\n");
                me_free(expr);
            }
        }
    }

    printf("\n=== Summary ===\n");
    printf("- Use explicit variable dtypes (e.g., ME_FLOAT64) with ME_BOOL output\n");
    printf("- Or use ME_AUTO output which auto-infers ME_BOOL for comparisons\n");
    printf("- Comparisons compute in the input type, then convert to bool\n");
    printf("- Available operators: ==, !=, <, <=, >, >=\n");

    return 0;
}
