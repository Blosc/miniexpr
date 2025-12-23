/* Test complex functions: conj, imag, and real */
#include "../src/miniexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

#if defined(_MSC_VER) && defined(__clang__)
// On Windows with clang-cl, I is defined as _Fcomplex struct
// We need the proper _Complex constant instead
#ifdef I
#undef I
#endif
#define I (1.0fi)  // Use the imaginary constant literal
#endif

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

#if defined(_MSC_VER) && defined(__clang__)
#define CREAL(x) __builtin_creal(x)
#define CIMAG(x) __builtin_cimag(x)
#define CREALF(x) __builtin_crealf(x)
#define CIMAGF(x) __builtin_cimagf(x)
#define CONJ(x) __builtin_conj(x)
#define CONJF(x) __builtin_conjf(x)
#else
#define CREAL(x) creal(x)
#define CIMAG(x) cimag(x)
#define CREALF(x) crealf(x)
#define CIMAGF(x) cimagf(x)
#define CONJ(x) conj(x)
#define CONJF(x) conjf(x)
#endif

#define ASSERT_COMPLEX_NEAR(expected, actual, idx) \
    { \
        double diff_real = fabs(CREAL(expected) - CREAL(actual)); \
        double diff_imag = fabs(CIMAG(expected) - CIMAG(actual)); \
        if (diff_real > TOLERANCE || diff_imag > TOLERANCE) { \
            printf("  FAIL at [%d]: expected (%.10f%+.10fi), got (%.10f%+.10fi) (diff: %.2e)\n", \
                   idx, CREAL(expected), CIMAG(expected), CREAL(actual), CIMAG(actual), \
                   (diff_real > diff_imag) ? diff_real : diff_imag); \
            tests_failed++; \
            return; \
        } \
    }

void test_conj_c64() {
    TEST("conj(z) - complex conjugate for float complex");

    float _Complex z[VECTOR_SIZE] = {
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
    float _Complex result[VECTOR_SIZE] = {0};

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
        float _Complex expected = CONJF(z[i]);
        ASSERT_COMPLEX_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_conj_c128() {
    TEST("conj(z) - complex conjugate for double complex");

    double _Complex z[VECTOR_SIZE] = {
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
    double _Complex result[VECTOR_SIZE] = {0};

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
        double _Complex expected = CONJ(z[i]);
        ASSERT_COMPLEX_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_imag_c64() {
    TEST("imag(z) - imaginary part for float complex");

    float _Complex z[VECTOR_SIZE] = {
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
        float expected = CIMAGF(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_imag_c128() {
    TEST("imag(z) - imaginary part for double complex");

    double _Complex z[VECTOR_SIZE] = {
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
        double expected = CIMAG(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_conj_identity() {
    TEST("conj(conj(z)) == z - double conjugation identity");

    double _Complex z[5] = {
        1.0 + 2.0*I,
        -1.0 + 2.0*I,
        2.5 + 3.7*I,
        -3.5 + 4.2*I,
        0.0 + 0.0*I
    };
    double _Complex result[5] = {0};

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

    double _Complex z[5] = {
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
        double expected = CIMAG(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_c64() {
    TEST("real(z) - real part for float complex");

    float _Complex z[VECTOR_SIZE] = {
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
        float expected = CREALF(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_c128() {
    TEST("real(z) - real part for double complex");

    double _Complex z[VECTOR_SIZE] = {
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
        double expected = CREAL(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_auto_dtype() {
    TEST("real(z) with ME_AUTO output dtype");

    double _Complex z[5] = {
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
        double expected = CREAL(z[i]);
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

