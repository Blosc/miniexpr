/*
 * Example 18: DSL if/elif/else
 *
 * Demonstrates scalar conditionals in DSL kernels, including
 * return branches and flow-only loop control.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

static void print_array(const char *name, const double *arr, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n && i < 8; i++) {
        printf("%.2f%s", arr[i], i < n - 1 && i < 7 ? ", " : "");
    }
    if (n > 8) {
        printf(", ...");
    }
    printf("]\n");
}

static int run_result_branches(void) {
    printf("--- Example 1: if/elif/else return branches ---\n");

    const char *dsl_source =
        "def kernel(x):\n"
        "    if any(x > 0):\n"
        "        return 1\n"
        "    elif any(x < 0):\n"
        "        return 2\n"
        "    else:\n"
        "        return 3\n";

    me_variable vars[] = {{"x", ME_FLOAT64}};
    me_expr *expr = NULL;
    int err = 0;

    if (me_compile(dsl_source, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Compile error at %d\n", err);
        return 1;
    }

    double out[4];
    double x_case1[4] = {-1.0, 2.0, -3.0, 0.0};
    const void *vars_case1[] = {x_case1};
    if (me_eval(expr, vars_case1, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("Eval error (case 1)\n");
        me_free(expr);
        return 1;
    }
    print_array("x (case 1)", x_case1, 4);
    print_array("result", out, 4);

    double x_case2[4] = {-1.0, -2.0, -3.0, -4.0};
    const void *vars_case2[] = {x_case2};
    if (me_eval(expr, vars_case2, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("Eval error (case 2)\n");
        me_free(expr);
        return 1;
    }
    print_array("x (case 2)", x_case2, 4);
    print_array("result", out, 4);

    double x_case3[4] = {0.0, 0.0, 0.0, 0.0};
    const void *vars_case3[] = {x_case3};
    if (me_eval(expr, vars_case3, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("Eval error (case 3)\n");
        me_free(expr);
        return 1;
    }
    print_array("x (case 3)", x_case3, 4);
    print_array("result", out, 4);

    me_free(expr);
    return 0;
}

static int run_flow_only_loop(void) {
    printf("\n--- Example 2: flow-only loop control ---\n");

    const char *dsl_source =
        "def kernel():\n"
        "    sum = 0\n"
        "    for i in range(10):\n"
        "        if i == 3:\n"
        "            continue\n"
        "        elif i == 7:\n"
        "            break\n"
        "        sum = sum + i\n"
        "    return sum\n";

    me_expr *expr = NULL;
    int err = 0;

    if (me_compile(dsl_source, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Compile error at %d\n", err);
        return 1;
    }

    double out[4];
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("Eval error\n");
        me_free(expr);
        return 1;
    }
    print_array("result", out, 4);
    printf("Expected: 18.00 (0 + 1 + 2 + 4 + 5 + 6)\n");

    me_free(expr);
    return 0;
}

int main(void) {
    printf("=== DSL if/elif/else Example ===\n\n");

    if (run_result_branches() != 0) {
        return 1;
    }
    if (run_flow_only_loop() != 0) {
        return 1;
    }

    printf("\n✅ DSL if/elif/else example complete!\n");
    return 0;
}
