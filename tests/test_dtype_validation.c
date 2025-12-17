/* Test dtype validation rules for me_compile */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "miniexpr.h"

#define VECTOR_SIZE 10

int main() {
    printf("Testing dtype Validation Rules\n");
    printf("===============================\n\n");

    int32_t a[VECTOR_SIZE];
    int32_t b[VECTOR_SIZE];
    int32_t result[VECTOR_SIZE];
    int err;

    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    /* Test 1: Valid - All variables ME_AUTO, specific output dtype */
    printf("Test 1: All vars ME_AUTO, output dtype = ME_INT32\n");
    {
        me_variable vars[] = {{"a", ME_AUTO}, {"b", ME_AUTO}};
        me_expr *expr = me_compile("a + b", vars, 2, ME_INT32, &err);

        if (expr) {
            printf("  ✅ PASS: Compilation succeeded\n");
            printf("  Result dtype: %d\n", me_get_dtype(expr));
            me_free(expr);
        } else {
            printf("  ❌ FAIL: Should have succeeded\n");
        }
    }

    /* Test 2: Valid - All variables have explicit dtypes, output dtype = ME_AUTO */
    printf("\nTest 2: All vars have dtypes, output dtype = ME_AUTO\n");
    {
        me_variable vars[] = {{"a", ME_INT32}, {"b", ME_INT32}};
        me_expr *expr = me_compile("a + b", vars, 2, ME_AUTO, &err);

        if (expr) {
            printf("  ✅ PASS: Compilation succeeded\n");
            printf("  Inferred result dtype: %d (ME_INT32=%d)\n", me_get_dtype(expr), ME_INT32);
            me_free(expr);
        } else {
            printf("  ❌ FAIL: Should have succeeded\n");
        }
    }

    /* Test 3: Invalid - Mix of ME_AUTO and explicit dtypes with specific output */
    printf("\nTest 3: INVALID - Mixed var dtypes with specific output\n");
    {
        me_variable vars[] = {{"a", ME_INT32}, {"b", ME_AUTO}};
        me_expr *expr = me_compile("a + b", vars, 2, ME_INT32, &err);

        if (!expr) {
            printf("  ✅ PASS: Correctly rejected (error=%d)\n", err);
        } else {
            printf("  ❌ FAIL: Should have been rejected\n");
            me_free(expr);
        }
    }

    /* Test 4: Invalid - All explicit dtypes but non-ME_AUTO output */
    printf("\nTest 4: INVALID - Explicit var dtypes with specific output\n");
    {
        me_variable vars[] = {{"a", ME_INT32}, {"b", ME_INT32}};
        me_expr *expr = me_compile("a + b", vars, 2, ME_INT32, &err);

        if (!expr) {
            printf("  ✅ PASS: Correctly rejected (error=%d)\n", err);
        } else {
            printf("  ❌ FAIL: Should have been rejected\n");
            me_free(expr);
        }
    }

    /* Test 5: Invalid - All ME_AUTO variables with ME_AUTO output */
    printf("\nTest 5: INVALID - All ME_AUTO vars with ME_AUTO output\n");
    {
        me_variable vars[] = {{"a", ME_AUTO}, {"b", ME_AUTO}};
        me_expr *expr = me_compile("a + b", vars, 2, ME_AUTO, &err);

        if (!expr) {
            printf("  ✅ PASS: Correctly rejected (error=%d)\n", err);
        } else {
            printf("  ❌ FAIL: Should have been rejected\n");
            me_free(expr);
        }
    }

    printf("\n===============================\n");
    printf("Validation Rules Summary:\n");
    printf("1. output=ME_AUTO    → all vars must have explicit dtypes\n");
    printf("2. output=<specific> → all vars must be ME_AUTO\n");
    printf("3. No mixing allowed!\n");
    printf("===============================\n");

    return 0;
}
