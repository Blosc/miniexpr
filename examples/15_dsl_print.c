/*
 * Example: DSL print() debugging
 */
#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"



int main(void) {
    double x[6] = {1.0, 2.0, 3.0, -1.0, 0.5, 4.0};
    double out[6] = {0};

    const char *src =
        "print(\"sum = {}\", sum(x))\n"
        "print(\"sum =\", sum(x))\n"
        "print(\"min = {} max = {}\", min(x), max(x))\n"
        "print(\"sum and max =\", sum(x), max(x))\n"
        "result = x * 2\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("compile error at %d\n", err);
        return 1;
    }

    const void *inputs[] = {x};
    if (me_eval(expr, inputs, 1, out, 6, NULL) != ME_EVAL_SUCCESS) {
        printf("eval error\n");
        me_free(expr);
        return 1;
    }

    printf("result:");
    for (int i = 0; i < 6; i++) {
        printf(" %.1f", out[i]);
    }
    printf("\n");

    me_free(expr);
    return 0;
}
