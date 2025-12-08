#include "../src/miniexpr.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

int main() {
    int err;
    
    /* Test basic power operator */
    printf("Testing ** power operator...\n");
    
    /* Simple power */
    double output1[1];
    me_expr *expr1 = me_compile("2**3", NULL, 0, output1, 1, ME_FLOAT64, &err);
    assert(expr1 != NULL);
    me_eval(expr1);
    printf("2**3 = %f (expected 8.0)\n", output1[0]);
    assert(fabs(output1[0] - 8.0) < 1e-10);
    me_free(expr1);
    
    /* Negative exponent */
    double output2[1];
    me_expr *expr2 = me_compile("2**-2", NULL, 0, output2, 1, ME_FLOAT64, &err);
    assert(expr2 != NULL);
    me_eval(expr2);
    printf("2**-2 = %f (expected 0.25)\n", output2[0]);
    assert(fabs(output2[0] - 0.25) < 1e-10);
    me_free(expr2);
    
    /* Fractional exponent */
    double output3[1];
    me_expr *expr3 = me_compile("4**0.5", NULL, 0, output3, 1, ME_FLOAT64, &err);
    assert(expr3 != NULL);
    me_eval(expr3);
    printf("4**0.5 = %f (expected 2.0)\n", output3[0]);
    assert(fabs(output3[0] - 2.0) < 1e-10);
    me_free(expr3);
    
    /* Test with variables */
    double a_data[] = {2.0, 3.0, 4.0};
    double b_data[] = {3.0, 2.0, 0.5};
    double output[3];
    
    me_variable vars[] = {
        {"a", a_data, ME_VARIABLE, 0},
        {"b", b_data, ME_VARIABLE, 0}
    };
    
    me_expr *expr4 = me_compile("a**b", vars, 2, output, 3, ME_FLOAT64, &err);
    assert(expr4 != NULL);
    me_eval(expr4);
    printf("a**b with vectors:\n");
    printf("  2**3 = %f (expected 8.0)\n", output[0]);
    printf("  3**2 = %f (expected 9.0)\n", output[1]);
    printf("  4**0.5 = %f (expected 2.0)\n", output[2]);
    assert(fabs(output[0] - 8.0) < 1e-10);
    assert(fabs(output[1] - 9.0) < 1e-10);
    assert(fabs(output[2] - 2.0) < 1e-10);
    me_free(expr4);
    
    /* Test associativity - right associative like Python */
    double output5[1];
    me_expr *expr5 = me_compile("2**3**2", NULL, 0, output5, 1, ME_FLOAT64, &err);
    assert(expr5 != NULL);
    me_eval(expr5);
    printf("2**3**2 = %f (expected 512.0 for right-assoc, 64.0 for left-assoc)\n", output5[0]);
    #ifdef ME_POW_FROM_RIGHT
    assert(fabs(output5[0] - 512.0) < 1e-10);  // 2**(3**2) = 2**9 = 512
    #else
    assert(fabs(output5[0] - 64.0) < 1e-10);   // (2**3)**2 = 8**2 = 64
    #endif
    me_free(expr5);
    
    /* Test that ^ is XOR for integers, not power */
    int32_t int_a[] = {5, 12};
    int32_t int_b[] = {3, 10};
    int32_t int_out[2];
    
    me_variable int_vars[] = {
        {"a", int_a, ME_VARIABLE, 0},
        {"b", int_b, ME_VARIABLE, 0}
    };
    
    me_expr *expr6 = me_compile("a^b", int_vars, 2, int_out, 2, ME_INT32, &err);
    assert(expr6 != NULL);
    me_eval(expr6);
    printf("XOR test: 5^3 = %d (expected 6), 12^10 = %d (expected 6)\n", int_out[0], int_out[1]);
    assert(int_out[0] == 6);  // 5 XOR 3 = 6
    assert(int_out[1] == 6);  // 12 XOR 10 = 6
    me_free(expr6);
    
    printf("All ** operator tests passed!\n");
    return 0;
}
