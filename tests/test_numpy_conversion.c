/* Test NumPy conversion functions */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "../src/miniexpr_numpy.h"

int main() {
    printf("Testing NumPy <-> MiniExpr dtype conversions\n");
    printf("==============================================\n\n");

    int tests_passed = 0;
    int tests_total = 0;

    // Test 1: Round-trip conversion for all supported types
    printf("Test 1: Round-trip conversions\n");
    {
        me_dtype types[] = {
            ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
            ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64,
            ME_FLOAT32, ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
        };
        const char* names[] = {
            "ME_BOOL", "ME_INT8", "ME_INT16", "ME_INT32", "ME_INT64",
            "ME_UINT8", "ME_UINT16", "ME_UINT32", "ME_UINT64",
            "ME_FLOAT32", "ME_FLOAT64", "ME_COMPLEX64", "ME_COMPLEX128"
        };

        int all_passed = 1;
        for (int i = 0; i < 13; i++) {
            tests_total++;
            int numpy_num = me_dtype_to_numpy(types[i]);
            me_dtype back = me_dtype_from_numpy(numpy_num);

            if (back != types[i]) {
                printf("  ❌ FAIL: %s -> numpy=%d -> %d (expected %d)\n",
                       names[i], numpy_num, back, types[i]);
                all_passed = 0;
            } else {
                tests_passed++;
            }
        }

        if (all_passed) {
            printf("  ✅ PASS: All 13 types convert correctly\n");
        }
    }

    // Test 2: ME_AUTO returns -1 for NumPy
    printf("\nTest 2: ME_AUTO handling\n");
    {
        tests_total++;
        int numpy_num = me_dtype_to_numpy(ME_AUTO);
        if (numpy_num == -1) {
            printf("  ✅ PASS: ME_AUTO -> -1 (no NumPy equivalent)\n");
            tests_passed++;
        } else {
            printf("  ❌ FAIL: ME_AUTO -> %d (expected -1)\n", numpy_num);
        }
    }

    // Test 3: Specific NumPy type numbers
    printf("\nTest 3: Known NumPy type numbers\n");
    {
        struct {
            int numpy_num;
            me_dtype expected;
            const char* name;
        } cases[] = {
            {0, ME_BOOL, "NPY_BOOL"},
            {1, ME_INT8, "NPY_BYTE"},
            {2, ME_UINT8, "NPY_UBYTE"},
            {7, ME_INT64, "NPY_LONGLONG"},
            {11, ME_FLOAT32, "NPY_FLOAT"},
            {12, ME_FLOAT64, "NPY_DOUBLE"},
            {14, ME_COMPLEX64, "NPY_CFLOAT"},
            {15, ME_COMPLEX128, "NPY_CDOUBLE"},
        };

        int all_passed = 1;
        for (int i = 0; i < 8; i++) {
            tests_total++;
            me_dtype result = me_dtype_from_numpy(cases[i].numpy_num);
            if (result != cases[i].expected) {
                printf("  ❌ FAIL: numpy %d (%s) -> %d (expected %d)\n",
                       cases[i].numpy_num, cases[i].name, result, cases[i].expected);
                all_passed = 0;
            } else {
                tests_passed++;
            }
        }

        if (all_passed) {
            printf("  ✅ PASS: All known NumPy types map correctly\n");
        }
    }

    // Test 4: Unsupported NumPy types
    printf("\nTest 4: Unsupported NumPy types\n");
    {
        int unsupported[] = {9, 10, 13, 99};  // float16, longdouble, clongdouble, invalid
        const char* names[] = {"float16", "longdouble", "clongdouble", "invalid"};

        int all_passed = 1;
        for (int i = 0; i < 4; i++) {
            tests_total++;
            me_dtype result = me_dtype_from_numpy(unsupported[i]);
            int supported = me_numpy_type_supported(unsupported[i]);

            if (result != -1 || supported != 0) {
                printf("  ❌ FAIL: numpy %d (%s) -> %d, supported=%d (expected -1, 0)\n",
                       unsupported[i], names[i], result, supported);
                all_passed = 0;
            } else {
                tests_passed++;
            }
        }

        if (all_passed) {
            printf("  ✅ PASS: Unsupported types return -1\n");
        }
    }

    // Test 5: Type name function
    printf("\nTest 5: Type name function\n");
    {
        tests_total++;
        const char* name = me_numpy_type_name(7);  // int64
        if (strcmp(name, "int64") == 0) {
            printf("  ✅ PASS: me_numpy_type_name(7) = '%s'\n", name);
            tests_passed++;
        } else {
            printf("  ❌ FAIL: me_numpy_type_name(7) = '%s' (expected 'int64')\n", name);
        }
    }

    // Summary
    printf("\n");
    printf("==============================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_total);
    printf("==============================================\n");

    if (tests_passed == tests_total) {
        printf("\n✅ All NumPy conversion tests passed!\n");
        printf("\nUsage in Python bindings:\n");
        printf("  me_dtype dtype = me_dtype_from_numpy(array.dtype.num);\n");
        printf("  int numpy_num = me_dtype_to_numpy(expr->dtype);\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed\n");
        return 1;
    }
}
