/* Test complex functions: conj, imag, and real */
#include "../src/miniexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

#define VECTOR_SIZE 10
#define TOLERANCE 1e-9

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

#define ASSERT_NEAR(expected, actual, idx) \
    if (fabs((expected) - (actual)) > TOLERANCE) { \
        printf("  FAIL at [%d]: expected %.10f, got %.10f (diff: %.2e)\n", \
               idx, (double)(expected), (double)(actual), fabs((expected) - (actual))); \
        tests_failed++; \
        return; \
    }

#define ASSERT_COMPLEX_NEAR(expected, actual, idx) \
    { \
        double diff_real = fabs(creal(expected) - creal(actual)); \
        double diff_imag = fabs(cimag(expected) - cimag(actual)); \
        if (diff_real > TOLERANCE || diff_imag > TOLERANCE) { \
            printf("  FAIL at [%d]: expected (%.10f%+.10fi), got (%.10f%+.10fi) (diff: %.2e)\n", \
                   idx, creal(expected), cimag(expected), creal(actual), cimag(actual), \
                   (diff_real > diff_imag) ? diff_real : diff_imag); \
            tests_failed++; \
            return; \
        } \
    }

void test_conj_c64() {
    TEST("conj(z) - complex conjugate for float complex");

    float complex z[VECTOR_SIZE] = {
        1.0f + 2.0f*I,
        -1.0f + 2.0f*I,
        1.0f - 2.0f*I,
        -1.0f - 2.0f*I,
        0.0f + 1.0f*I,
        0.0f - 1.0f*I,
        3.5f + 0.0f*I,
        -3.5f + 0.0f*I,
        0.0f + 0.0f*I,
        2.5f + 3.7f*I
    };
    float complex result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX64}};

    int err;
    me_expr *expr = me_compile("conj(z)", vars, 1, ME_COMPLEX64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        float complex expected = conjf(z[i]);
        ASSERT_COMPLEX_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_conj_c128() {
    TEST("conj(z) - complex conjugate for double complex");

    double complex z[VECTOR_SIZE] = {
        1.0 + 2.0*I,
        -1.0 + 2.0*I,
        1.0 - 2.0*I,
        -1.0 - 2.0*I,
        0.0 + 1.0*I,
        0.0 - 1.0*I,
        3.5 + 0.0*I,
        -3.5 + 0.0*I,
        0.0 + 0.0*I,
        2.5 + 3.7*I
    };
    double complex result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX128}};

    int err;
    me_expr *expr = me_compile("conj(z)", vars, 1, ME_COMPLEX128, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double complex expected = conj(z[i]);
        ASSERT_COMPLEX_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_imag_c64() {
    TEST("imag(z) - imaginary part for float complex");

    float complex z[VECTOR_SIZE] = {
        1.0f + 2.0f*I,
        -1.0f + 2.0f*I,
        1.0f - 2.0f*I,
        -1.0f - 2.0f*I,
        0.0f + 1.0f*I,
        0.0f - 1.0f*I,
        3.5f + 0.0f*I,
        -3.5f + 0.0f*I,
        0.0f + 0.0f*I,
        2.5f + 3.7f*I
    };
    float result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX64}};

    int err;
    me_expr *expr = me_compile("imag(z)", vars, 1, ME_FLOAT32, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        float expected = cimagf(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_imag_c128() {
    TEST("imag(z) - imaginary part for double complex");

    double complex z[VECTOR_SIZE] = {
        1.0 + 2.0*I,
        -1.0 + 2.0*I,
        1.0 - 2.0*I,
        -1.0 - 2.0*I,
        0.0 + 1.0*I,
        0.0 - 1.0*I,
        3.5 + 0.0*I,
        -3.5 + 0.0*I,
        0.0 + 0.0*I,
        2.5 + 3.7*I
    };
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX128}};

    int err;
    me_expr *expr = me_compile("imag(z)", vars, 1, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = cimag(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_conj_identity() {
    TEST("conj(conj(z)) == z - double conjugation identity");

    double complex z[5] = {
        1.0 + 2.0*I,
        -1.0 + 2.0*I,
        2.5 + 3.7*I,
        -3.5 + 4.2*I,
        0.0 + 0.0*I
    };
    double complex result[5] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX128}};

    int err;
    me_expr *expr = me_compile("conj(conj(z))", vars, 1, ME_COMPLEX128, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, 5);

    for (int i = 0; i < 5; i++) {
        ASSERT_COMPLEX_NEAR(z[i], result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_imag_auto_dtype() {
    TEST("imag(z) with ME_AUTO output dtype");

    double complex z[5] = {
        1.0 + 2.0*I,
        -1.0 + 2.0*I,
        2.5 + 3.7*I,
        -3.5 + 4.2*I,
        0.0 + 0.0*I
    };
    double result[5] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX128}};

    int err;
    me_expr *expr = me_compile("imag(z)", vars, 1, ME_AUTO, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, 5);

    for (int i = 0; i < 5; i++) {
        double expected = cimag(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_c64() {
    TEST("real(z) - real part for float complex");

    float complex z[VECTOR_SIZE] = {
        1.0f + 2.0f*I,
        -1.0f + 2.0f*I,
        1.0f - 2.0f*I,
        -1.0f - 2.0f*I,
        0.0f + 1.0f*I,
        0.0f - 1.0f*I,
        3.5f + 0.0f*I,
        -3.5f + 0.0f*I,
        0.0f + 0.0f*I,
        2.5f + 3.7f*I
    };
    float result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX64}};

    int err;
    me_expr *expr = me_compile("real(z)", vars, 1, ME_FLOAT32, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        float expected = crealf(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_c128() {
    TEST("real(z) - real part for double complex");

    double complex z[VECTOR_SIZE] = {
        1.0 + 2.0*I,
        -1.0 + 2.0*I,
        1.0 - 2.0*I,
        -1.0 - 2.0*I,
        0.0 + 1.0*I,
        0.0 - 1.0*I,
        3.5 + 0.0*I,
        -3.5 + 0.0*I,
        0.0 + 0.0*I,
        2.5 + 3.7*I
    };
    double result[VECTOR_SIZE] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX128}};

    int err;
    me_expr *expr = me_compile("real(z)", vars, 1, ME_FLOAT64, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        double expected = creal(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_auto_dtype() {
    TEST("real(z) with ME_AUTO output dtype");

    double complex z[5] = {
        1.0 + 2.0*I,
        -1.0 + 2.0*I,
        2.5 + 3.7*I,
        -3.5 + 4.2*I,
        0.0 + 0.0*I
    };
    double result[5] = {0};

    me_variable vars[] = {{"z", ME_COMPLEX128}};

    int err;
    me_expr *expr = me_compile("real(z)", vars, 1, ME_AUTO, &err);

    if (!expr) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    me_eval(expr, var_ptrs, 1, result, 5);

    for (int i = 0; i < 5; i++) {
        double expected = creal(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

int main() {
    printf("=== Testing Complex Functions (conj, imag, real) ===\n\n");

    test_conj_c64();
    test_conj_c128();
    test_imag_c64();
    test_imag_c128();
    test_real_c64();
    test_real_c128();
    test_conj_identity();
    test_imag_auto_dtype();
    test_real_auto_dtype();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}

