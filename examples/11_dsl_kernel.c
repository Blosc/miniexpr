/*
 * Example 11: DSL Multi-Statement Kernel
 *
 * Demonstrates DSL features including temporary variables,
 * element-wise conditionals, and index access.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/miniexpr.h"
#include "../src/dsl_parser.h"

void print_array(const char *name, const double *arr, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n && i < 8; i++) {
        printf("%.4f%s", arr[i], i < n - 1 && i < 7 ? ", " : "");
    }
    if (n > 8) printf(", ...");
    printf("]\n");
}

int main() {
    printf("=== DSL Multi-Statement Kernel Example ===\n\n");

    /* Example 1: Temporary variables for polynomial evaluation
     * Computes: x^3 - 2*x^2 + 3*x - 1 using Horner's method
     */
    printf("--- Example 1: Polynomial with temporaries ---\n");
    {
        const char *dsl_source =
            "def kernel(x):\n"
            "    t1 = 1.0 * x - 2.0\n"
            "    t2 = t1 * x + 3.0\n"
            "    return t2 * x - 1.0";

        printf("DSL Program:\n%s\n\n", dsl_source);

        me_dsl_error error;
        me_dsl_program *prog = me_dsl_parse(dsl_source, &error);
        if (!prog) {
            printf("Parse error at line %d, col %d: %s\n",
                   error.line, error.column, error.message);
            return 1;
        }
        printf("Parsed %d statements successfully.\n", prog->block.nstmts);

        /* For evaluation, we'd compile each sub-expression with me_compile
         * and evaluate with me_eval. Here we just demonstrate parsing. */

        /* Manual evaluation for demonstration */
        double x[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        double result[5];
        int n = 5;

        for (int i = 0; i < n; i++) {
            double t1 = 1.0 * x[i] - 2.0;
            double t2 = t1 * x[i] + 3.0;
            result[i] = t2 * x[i] - 1.0;
        }

        print_array("x", x, n);
        print_array("result", result, n);
        printf("(x^3 - 2x^2 + 3x - 1)\n\n");

        me_dsl_program_free(prog);
    }

    /* Example 2: Conditional with where()
     * Clamps values to [0, 1] range
     */
    printf("--- Example 2: Conditional clamping ---\n");
    {
        const char *dsl_source =
            "def kernel(x):\n"
            "    clamped = where(x < 0, 0, where(x > 1, 1, x))\n"
            "    return clamped";

        printf("DSL Program:\n%s\n\n", dsl_source);

        me_dsl_error error;
        me_dsl_program *prog = me_dsl_parse(dsl_source, &error);
        if (!prog) {
            printf("Parse error: %s\n", error.message);
            return 1;
        }
        printf("Parsed %d statements successfully.\n", prog->block.nstmts);

        /* Manual evaluation for demonstration */
        double x[] = {-0.5, 0.0, 0.3, 0.7, 1.0, 1.5};
        double result[6];
        int n = 6;

        for (int i = 0; i < n; i++) {
            result[i] = x[i] < 0 ? 0 : (x[i] > 1 ? 1 : x[i]);
        }

        print_array("x", x, n);
        print_array("clamped", result, n);
        printf("\n");

        me_dsl_program_free(prog);
    }

    /* Example 3: Sin^2 + Cos^2 identity
     * Should always equal 1.0
     */
    printf("--- Example 3: Trigonometric identity ---\n");
    {
        const char *dsl_source =
            "def kernel(x):\n"
            "    s = sin(x) ** 2\n"
            "    c = cos(x) ** 2\n"
            "    return s + c";

        printf("DSL Program:\n%s\n\n", dsl_source);

        me_dsl_error error;
        me_dsl_program *prog = me_dsl_parse(dsl_source, &error);
        if (!prog) {
            printf("Parse error: %s\n", error.message);
            return 1;
        }
        printf("Parsed %d statements successfully.\n", prog->block.nstmts);

        /* Evaluate using miniexpr directly for this simple case */
        double x[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.14159};
        double result[6];
        int n = 6;

        me_variable vars[] = {{"x", ME_FLOAT64, NULL}};
        int err;
        me_expr *expr = NULL;

        if (me_compile("sin(x)**2 + cos(x)**2", vars, 1, ME_FLOAT64, &err, &expr)
            == ME_COMPILE_SUCCESS) {
            const void *ptrs[] = {x};
            me_eval(expr, ptrs, 1, result, n, NULL);
            me_free(expr);

            print_array("x", x, n);
            print_array("sin²+cos²", result, n);
            printf("(All values should be 1.0)\n\n");
        }

        me_dsl_program_free(prog);
    }

    /* Example 4: Complex formula with multiple operations */
    printf("--- Example 4: Damped oscillation ---\n");
    {
        const char *dsl_source =
            "def kernel(amplitude, t):\n"
            "    decay = exp(-0.1 * t)\n"
            "    oscillation = sin(2 * 3.14159 * t)\n"
            "    return amplitude * decay * oscillation";

        printf("DSL Program:\n%s\n\n", dsl_source);

        me_dsl_error error;
        me_dsl_program *prog = me_dsl_parse(dsl_source, &error);
        if (!prog) {
            printf("Parse error: %s\n", error.message);
            return 1;
        }
        printf("Parsed %d statements successfully.\n", prog->block.nstmts);

        /* Manual evaluation */
        double amplitude = 1.0;
        double t[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0};
        double result[8];
        int n = 8;

        for (int i = 0; i < n; i++) {
            double decay = exp(-0.1 * t[i]);
            double oscillation = sin(2 * 3.14159 * t[i]);
            result[i] = amplitude * decay * oscillation;
        }

        print_array("t", t, n);
        print_array("y(t)", result, n);

        me_dsl_program_free(prog);
    }

    printf("\n✅ DSL examples complete!\n");
    return 0;
}
