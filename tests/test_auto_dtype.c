/* Test ME_AUTO dtype and ME_BOOL to ensure no interference */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "miniexpr.h"

#define VECTOR_SIZE 10

int main() {
    printf("Testing ME_AUTO and ME_BOOL\n");
    printf("===========================\n\n");

    int tests_passed = 0;
    int tests_total = 0;

    /* Test 1: ME_AUTO (value 0) for automatic type inference */
    {
        tests_total++;
        printf("Test 1: ME_AUTO for automatic type inference\n");

        int32_t a[VECTOR_SIZE];
        int32_t b[VECTOR_SIZE];
        int32_t result[VECTOR_SIZE];

        for (int i = 0; i < VECTOR_SIZE; i++) {
            a[i] = i;
            b[i] = i * 2;
        }

        /* Use ME_AUTO (0) to let the compiler infer the type from variables */
        me_variable vars[] = {
            {"a", ME_AUTO, a},  // Explicitly use ME_AUTO
            {"b", ME_AUTO, b}
        };

        int err;
        me_expr *expr = me_compile("a + b", vars, 2, result, VECTOR_SIZE, ME_INT32, &err);

        if (!expr) {
            printf("  ❌ FAIL: Compilation error at position %d\n", err);
        } else {
            me_eval(expr);

            bool passed = true;
            for (int i = 0; i < VECTOR_SIZE && passed; i++) {
                if (result[i] != a[i] + b[i]) {
                    printf("  ❌ FAIL: Mismatch at [%d]: got %d, expected %d\n",
                           i, result[i], a[i] + b[i]);
                    passed = false;
                }
            }

            if (passed) {
                printf("  ✅ PASS: ME_AUTO works correctly\n");
                tests_passed++;
            }

            me_free(expr);
        }
    }

    /* Test 2: ME_BOOL (value 1 after ME_AUTO) */
    {
        tests_total++;
        printf("\nTest 2: ME_BOOL operations\n");

        bool a[VECTOR_SIZE] = {true, false, true, false, true, false, true, false, true, false};
        bool b[VECTOR_SIZE] = {true, true, false, false, true, true, false, false, true, true};
        bool result[VECTOR_SIZE] = {0};

        me_variable vars[] = {
            {"a", ME_AUTO, a},
            {"b", ME_AUTO, b}
        };

        int err;
        me_expr *expr = me_compile("a & b", vars, 2, result, VECTOR_SIZE, ME_BOOL, &err);

        if (!expr) {
            printf("  ❌ FAIL: Compilation error at position %d\n", err);
        } else {
            me_eval(expr);

            bool passed = true;
            for (int i = 0; i < VECTOR_SIZE && passed; i++) {
                bool expected = a[i] && b[i];
                if (result[i] != expected) {
                    printf("  ❌ FAIL: Mismatch at [%d]: got %d, expected %d\n",
                           i, result[i], expected);
                    passed = false;
                }
            }

            if (passed) {
                printf("  ✅ PASS: ME_BOOL works correctly\n");
                tests_passed++;
            }

            me_free(expr);
        }
    }

    /* Test 3: Verify ME_AUTO and ME_BOOL have different values */
    {
        tests_total++;
        printf("\nTest 3: ME_AUTO != ME_BOOL\n");

        if (ME_AUTO != ME_BOOL) {
            printf("  ✅ PASS: ME_AUTO (%d) != ME_BOOL (%d)\n", ME_AUTO, ME_BOOL);
            tests_passed++;
        } else {
            printf("  ❌ FAIL: ME_AUTO (%d) == ME_BOOL (%d) - conflict!\n", ME_AUTO, ME_BOOL);
        }
    }

    /* Test 4: Verify ME_AUTO is 0 */
    {
        tests_total++;
        printf("\nTest 4: ME_AUTO value is 0\n");

        if (ME_AUTO == 0) {
            printf("  ✅ PASS: ME_AUTO == 0\n");
            tests_passed++;
        } else {
            printf("  ❌ FAIL: ME_AUTO == %d (expected 0)\n", ME_AUTO);
        }
    }

    /* Test 5: Mixed - explicit ME_BOOL with ME_AUTO inference */
    {
        tests_total++;
        printf("\nTest 5: Mixed ME_BOOL variable with ME_AUTO for other var\n");

        bool a[VECTOR_SIZE];
        int32_t b[VECTOR_SIZE];
        int32_t result[VECTOR_SIZE];

        for (int i = 0; i < VECTOR_SIZE; i++) {
            a[i] = (i % 2 == 0);
            b[i] = i * 10;
        }

        me_variable vars[] = {
            {"a", ME_BOOL, a},    // Explicit ME_BOOL
            {"b", ME_INT32, b}    // Explicit ME_INT32
        };

        int err;
        // Simple expression: cast bool to int and add - use ME_AUTO to infer result
        me_expr *expr = me_compile("a + b", vars, 2, result, VECTOR_SIZE, ME_AUTO, &err);

        if (!expr) {
            printf("  ❌ FAIL: Compilation error at position %d\n", err);
        } else {
            me_eval(expr);

            bool passed = true;
            for (int i = 0; i < VECTOR_SIZE && passed; i++) {
                int32_t expected = a[i] + b[i];
                if (result[i] != expected) {
                    printf("  ❌ FAIL: Mismatch at [%d]: got %d, expected %d\n",
                           i, result[i], expected);
                    passed = false;
                }
            }

            if (passed) {
                printf("  ✅ PASS: Mixed ME_BOOL and ME_AUTO work together\n");
                tests_passed++;
            }

            me_free(expr);
        }
    }

    printf("\n");
    printf("===============================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_total);
    printf("===============================\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
