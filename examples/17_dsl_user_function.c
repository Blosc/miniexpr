/*
 * Example 17: DSL User-defined Function
 *
 * Demonstrates registering a custom C function for DSL expressions.
 */

#include <stdio.h>
#include "../src/miniexpr.h"

static double clamp01(double x) {
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

static void print_array(const char *name, const double *arr, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f%s", arr[i], i < n - 1 ? ", " : "");
    }
    printf("]\n");
}

int main(void) {
    printf("=== DSL User-defined Function Example ===\n\n");

    const char *dsl_source =
        "def kernel(x):\n"
        "    return clamp01(x)\n";
    printf("DSL Program:\n%s\n\n", dsl_source);

    double x[] = {-0.5, 0.0, 0.25, 1.0, 1.5};
    int n = (int)(sizeof(x) / sizeof(x[0]));
    double out[5];

    me_variable vars[] = {
        {"x", ME_FLOAT64, NULL, 0, NULL, 0},
        {"clamp01", ME_FLOAT64, clamp01, ME_FUNCTION1 | ME_FLAG_PURE, NULL, 0},
    };

    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(dsl_source, vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Compile error at %d\n", err);
        return 1;
    }

    const void *inputs[] = {x};
    if (me_eval(expr, inputs, 1, out, n, NULL) != ME_EVAL_SUCCESS) {
        printf("Eval error\n");
        me_free(expr);
        return 1;
    }

    print_array("x", x, n);
    print_array("clamp01(x)", out, n);

    me_free(expr);
    return 0;
}
