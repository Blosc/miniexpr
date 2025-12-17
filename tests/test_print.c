/* Test and demonstration of me_print() debugging function */

#include <stdio.h>
#include "../src/miniexpr.h"

int main() {
    printf("========================================\n");
    printf("Testing me_print() - Expression Tree Visualization\n");
    printf("========================================\n\n");

    // Example 1: Simple expression
    printf("1. Simple expression: x + 5\n");
    printf("   Expression tree:\n");
    me_variable vars1[] = {{"x"}};
    int err;
    me_expr *expr1 = me_compile("x + 5", vars1, 1, ME_FLOAT64, &err);
    if (expr1) {
        me_print(expr1);
        me_free(expr1);
    }
    printf("\n");

    // Example 2: Complex nested expression
    printf("2. Complex expression: (a + b) * (c - d)\n");
    printf("   Expression tree:\n");
    me_variable vars2[] = {{"a"}, {"b"}, {"c"}, {"d"}};
    me_expr *expr2 = me_compile("(a + b) * (c - d)", vars2, 4, ME_FLOAT64, &err);
    if (expr2) {
        me_print(expr2);
        me_free(expr2);
    }
    printf("\n");

    // Example 3: Expression with functions
    printf("3. Expression with functions: sqrt(x*x + y*y)\n");
    printf("   Expression tree:\n");
    me_variable vars3[] = {{"x"}, {"y"}};
    me_expr *expr3 = me_compile("sqrt(x*x + y*y)", vars3, 2, ME_FLOAT64, &err);
    if (expr3) {
        me_print(expr3);
        me_free(expr3);
    }
    printf("\n");

    // Example 4: Show that evaluation still works
    printf("4. Actual evaluation of: x + y\n");
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    double result[3];

    me_variable vars4[] = {{"x"}, {"y"}};
    me_expr *expr4 = me_compile("x + y", vars4, 2, ME_FLOAT64, &err);
    if (expr4) {
        printf("   Tree structure:\n");
        me_print(expr4);

        const void *var_ptrs[] = {x, y};
        me_eval(expr4, var_ptrs, 2, result, 3);

        printf("   Evaluation results:\n");
        for (int i = 0; i < 3; i++) {
            printf("   x[%d]=%.1f + y[%d]=%.1f = %.1f\n", i, x[i], i, y[i], result[i]);
        }
        me_free(expr4);
    }
    printf("\n");

    printf("========================================\n");
    printf("me_print() helps you visualize the\n");
    printf("expression tree structure for debugging.\n");
    printf("========================================\n");

    return 0;
}
