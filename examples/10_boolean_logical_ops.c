/**
 * Example 10: Boolean Logical Operators
 *
 * Demonstrates logical behavior of and/or/not (and their symbol forms) when used
 * with boolean arrays. For boolean inputs, these operators follow NumPy-style
 * logical semantics.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include "miniexpr.h"
#include "minctest.h"


#define N 8

static void print_bool_row(const char *label, const bool *data) {
    printf("  %-7s |", label);
    for (int i = 0; i < N; i++) {
        printf(" %c", data[i] ? 'T' : 'F');
    }
    printf("\n");
}

int main() {
    printf("=== Example 10: Boolean Logical Operators ===\n\n");

    bool a[N] = {true, false, true, false, true, false, true, false};
    bool b[N] = {true, true, false, false, true, true, false, false};
    bool result[N] = {0};

    me_variable vars_ab[] = {{"a", ME_BOOL}, {"b", ME_BOOL}};
    me_variable vars_a[] = {{"a", ME_BOOL}};
    int err;

    printf("Example 1: Logical ops on boolean arrays\n");
    printf("----------------------------------------\n");
    {
        me_expr *expr = NULL;

        printf("  Index   | 0 1 2 3 4 5 6 7\n");
        printf("  --------+----------------\n");
        print_bool_row("a", a);
        print_bool_row("b", b);

        expr = NULL;
        if (me_compile("a and b", vars_ab, 2, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }
        const void *ptrs_and[] = {a, b};
        ME_EVAL_CHECK(expr, ptrs_and, 2, result, N);
        print_bool_row("a and b", result);
        me_free(expr);

        expr = NULL;
        if (me_compile("a or b", vars_ab, 2, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }
        const void *ptrs_or[] = {a, b};
        ME_EVAL_CHECK(expr, ptrs_or, 2, result, N);
        print_bool_row("a or b", result);
        me_free(expr);

        expr = NULL;
        if (me_compile("a ^ b", vars_ab, 2, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }
        const void *ptrs_xor[] = {a, b};
        ME_EVAL_CHECK(expr, ptrs_xor, 2, result, N);
        print_bool_row("a ^ b", result);
        me_free(expr);

        expr = NULL;
        if (me_compile("not a", vars_a, 1, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }
        const void *ptrs_not[] = {a};
        ME_EVAL_CHECK(expr, ptrs_not, 1, result, N);
        print_bool_row("not a", result);
        me_free(expr);
    }

    printf("\nExample 2: Combine comparison masks with and/or\n");
    printf("------------------------------------------------\n");
    {
        float o0[N] = {0.2f, 0.6f, 1.2f, 0.4f, 0.9f, 0.1f, 0.8f, 0.0f};
        int32_t o1[N] = {9999, 10001, 10000, 15000, 5000, 20000, 10002, 42};
        bool mask[N] = {0};

        me_variable vars[] = {{"o0", ME_FLOAT32}, {"o1", ME_INT32}};
        me_expr *expr = NULL;

        if (me_compile("o0 > 0.5 and o1 > 10000 or o1 == 42", vars, 2, ME_BOOL, &err, &expr)
            != ME_COMPILE_SUCCESS) {
            printf("Compilation error at position %d\n", err);
            return 1;
        }

        const void *ptrs[] = {o0, o1};
        ME_EVAL_CHECK(expr, ptrs, 2, mask, N);

        printf("  idx |  o0  |  o1   | o0 > 0.5 and o1 > 10000 or o1 == 42\n");
        printf("  ----+------+-------+-----------------------------------------\n");
        for (int i = 0; i < N; i++) {
            printf("  %3d | %4.2f | %5d | %s\n",
                   i, o0[i], o1[i], mask[i] ? "true" : "false");
        }

        me_free(expr);
    }

    return 0;
}
