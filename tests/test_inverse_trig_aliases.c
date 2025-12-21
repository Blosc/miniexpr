/* Test that both acos/arccos naming conventions work */
#include "../src/miniexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_SIZE 5
#define TOLERANCE 1e-9

int main() {
    printf("=== Testing Inverse Trigonometric Function Aliases ===\n\n");

    double x[VECTOR_SIZE] = {0.0, 0.5, 0.707, 0.866, 1.0};
    double result_a[VECTOR_SIZE] = {0};
    double result_arc[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"x"}};
    int err;
    int tests_passed = 0;
    int tests_total = 0;

    // Test acos vs arccos
    tests_total++;
    printf("Test 1: acos(x) vs arccos(x)\n");
    me_expr *expr_acos = me_compile("acos(x)", vars, 1, ME_FLOAT64, &err);
    me_expr *expr_arccos = me_compile("arccos(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr_acos || !expr_arccos) {
        printf("  FAIL: Compilation failed\n");
    } else {
        const void *var_ptrs[] = {x};
        me_eval(expr_acos, var_ptrs, 1, result_a, VECTOR_SIZE);
        me_eval(expr_arccos, var_ptrs, 1, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_acos);
        me_free(expr_arccos);
    }

    // Test asin vs arcsin
    tests_total++;
    printf("\nTest 2: asin(x) vs arcsin(x)\n");
    me_expr *expr_asin = me_compile("asin(x)", vars, 1, ME_FLOAT64, &err);
    me_expr *expr_arcsin = me_compile("arcsin(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr_asin || !expr_arcsin) {
        printf("  FAIL: Compilation failed\n");
    } else {
        const void *var_ptrs[] = {x};
        me_eval(expr_asin, var_ptrs, 1, result_a, VECTOR_SIZE);
        me_eval(expr_arcsin, var_ptrs, 1, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_asin);
        me_free(expr_arcsin);
    }

    // Test atan vs arctan
    tests_total++;
    printf("\nTest 3: atan(x) vs arctan(x)\n");
    me_expr *expr_atan = me_compile("atan(x)", vars, 1, ME_FLOAT64, &err);
    me_expr *expr_arctan = me_compile("arctan(x)", vars, 1, ME_FLOAT64, &err);

    if (!expr_atan || !expr_arctan) {
        printf("  FAIL: Compilation failed\n");
    } else {
        const void *var_ptrs[] = {x};
        me_eval(expr_atan, var_ptrs, 1, result_a, VECTOR_SIZE);
        me_eval(expr_arctan, var_ptrs, 1, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_atan);
        me_free(expr_arctan);
    }

    // Test atan2 vs arctan2
    tests_total++;
    printf("\nTest 4: atan2(x, x) vs arctan2(x, x)\n");
    me_expr *expr_atan2 = me_compile("atan2(x, x)", vars, 1, ME_FLOAT64, &err);
    me_expr *expr_arctan2 = me_compile("arctan2(x, x)", vars, 1, ME_FLOAT64, &err);

    if (!expr_atan2 || !expr_arctan2) {
        printf("  FAIL: Compilation failed\n");
    } else {
        const void *var_ptrs[] = {x};
        me_eval(expr_atan2, var_ptrs, 1, result_a, VECTOR_SIZE);
        me_eval(expr_arctan2, var_ptrs, 1, result_arc, VECTOR_SIZE);

        int match = 1;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            if (fabs(result_a[i] - result_arc[i]) > TOLERANCE) {
                match = 0;
                break;
            }
        }

        if (match) {
            printf("  PASS: Both produce identical results\n");
            tests_passed++;
        } else {
            printf("  FAIL: Results differ\n");
        }

        me_free(expr_atan2);
        me_free(expr_arctan2);
    }

    printf("\n=== Test Summary ===\n");
    printf("Tests passed: %d/%d\n", tests_passed, tests_total);

    return (tests_passed == tests_total) ? 0 : 1;
}
