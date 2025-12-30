/* Test complex functions: conj, imag, and real */
#include "../src/miniexpr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include "me_complex.h"

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
        double diff_real = fabs(ME_CREAL(expected) - ME_CREAL(actual)); \
        double diff_imag = fabs(ME_CIMAG(expected) - ME_CIMAG(actual)); \
        if (diff_real > TOLERANCE || diff_imag > TOLERANCE) { \
            printf("  FAIL at [%d]: expected (%.10f%+.10fi), got (%.10f%+.10fi) (diff: %.2e)\n", \
                   idx, ME_CREAL(expected), ME_CIMAG(expected), ME_CREAL(actual), ME_CIMAG(actual), \
                   (diff_real > diff_imag) ? diff_real : diff_imag); \
            tests_failed++; \
            return; \
        } \
    }

void test_conj_c64() {
    TEST("conj(z) - complex conjugate for float complex");

    float _Complex z[VECTOR_SIZE] = {
        ME_C64_BUILD(1.0f, 2.0f),
        ME_C64_BUILD(-1.0f, 2.0f),
        ME_C64_BUILD(1.0f, -2.0f),
        ME_C64_BUILD(-1.0f, -2.0f),
        ME_C64_BUILD(0.0f, 1.0f),
        ME_C64_BUILD(0.0f, -1.0f),
        ME_C64_BUILD(3.5f, 0.0f),
        ME_C64_BUILD(-3.5f, 0.0f),
        ME_C64_BUILD(0.0f, 0.0f),
        ME_C64_BUILD(2.5f, 3.7f)
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
        float _Complex expected = ME_CONJF(z[i]);
        ASSERT_COMPLEX_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_conj_c128() {
    TEST("conj(z) - complex conjugate for double complex");

    double _Complex z[VECTOR_SIZE] = {
        ME_C128_BUILD(1.0, 2.0),
        ME_C128_BUILD(-1.0, 2.0),
        ME_C128_BUILD(1.0, -2.0),
        ME_C128_BUILD(-1.0, -2.0),
        ME_C128_BUILD(0.0, 1.0),
        ME_C128_BUILD(0.0, -1.0),
        ME_C128_BUILD(3.5, 0.0),
        ME_C128_BUILD(-3.5, 0.0),
        ME_C128_BUILD(0.0, 0.0),
        ME_C128_BUILD(2.5, 3.7)
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
        double _Complex expected = ME_CONJ(z[i]);
        ASSERT_COMPLEX_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_imag_c64() {
    TEST("imag(z) - imaginary part for float complex");

    float _Complex z[VECTOR_SIZE] = {
        ME_C64_BUILD(1.0f, 2.0f),
        ME_C64_BUILD(-1.0f, 2.0f),
        ME_C64_BUILD(1.0f, -2.0f),
        ME_C64_BUILD(-1.0f, -2.0f),
        ME_C64_BUILD(0.0f, 1.0f),
        ME_C64_BUILD(0.0f, -1.0f),
        ME_C64_BUILD(3.5f, 0.0f),
        ME_C64_BUILD(-3.5f, 0.0f),
        ME_C64_BUILD(0.0f, 0.0f),
        ME_C64_BUILD(2.5f, 3.7f)
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
        float expected = ME_CIMAGF(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_imag_c128() {
    TEST("imag(z) - imaginary part for double complex");

    double _Complex z[VECTOR_SIZE] = {
        ME_C128_BUILD(1.0, 2.0),
        ME_C128_BUILD(-1.0, 2.0),
        ME_C128_BUILD(1.0, -2.0),
        ME_C128_BUILD(-1.0, -2.0),
        ME_C128_BUILD(0.0, 1.0),
        ME_C128_BUILD(0.0, -1.0),
        ME_C128_BUILD(3.5, 0.0),
        ME_C128_BUILD(-3.5, 0.0),
        ME_C128_BUILD(0.0, 0.0),
        ME_C128_BUILD(2.5, 3.7)
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
        double expected = ME_CIMAG(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_conj_identity() {
    TEST("conj(conj(z)) == z - double conjugation identity");

    double _Complex z[5] = {
        ME_C128_BUILD(1.0, 2.0),
        ME_C128_BUILD(-1.0, 2.0),
        ME_C128_BUILD(2.5, 3.7),
        ME_C128_BUILD(-3.5, 4.2),
        ME_C128_BUILD(0.0, 0.0)
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
        ME_C128_BUILD(1.0, 2.0),
        ME_C128_BUILD(-1.0, 2.0),
        ME_C128_BUILD(2.5, 3.7),
        ME_C128_BUILD(-3.5, 4.2),
        ME_C128_BUILD(0.0, 0.0)
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
        double expected = ME_CIMAG(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_c64() {
    TEST("real(z) - real part for float complex");

    float _Complex z[VECTOR_SIZE] = {
        ME_C64_BUILD(1.0f, 2.0f),
        ME_C64_BUILD(-1.0f, 2.0f),
        ME_C64_BUILD(1.0f, -2.0f),
        ME_C64_BUILD(-1.0f, -2.0f),
        ME_C64_BUILD(0.0f, 1.0f),
        ME_C64_BUILD(0.0f, -1.0f),
        ME_C64_BUILD(3.5f, 0.0f),
        ME_C64_BUILD(-3.5f, 0.0f),
        ME_C64_BUILD(0.0f, 0.0f),
        ME_C64_BUILD(2.5f, 3.7f)
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
        float expected = ME_CREALF(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_c128() {
    TEST("real(z) - real part for double complex");

    double _Complex z[VECTOR_SIZE] = {
        ME_C128_BUILD(1.0, 2.0),
        ME_C128_BUILD(-1.0, 2.0),
        ME_C128_BUILD(1.0, -2.0),
        ME_C128_BUILD(-1.0, -2.0),
        ME_C128_BUILD(0.0, 1.0),
        ME_C128_BUILD(0.0, -1.0),
        ME_C128_BUILD(3.5, 0.0),
        ME_C128_BUILD(-3.5, 0.0),
        ME_C128_BUILD(0.0, 0.0),
        ME_C128_BUILD(2.5, 3.7)
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
        double expected = ME_CREAL(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_real_auto_dtype() {
    TEST("real(z) with ME_AUTO output dtype");

    double _Complex z[5] = {
        ME_C128_BUILD(1.0, 2.0),
        ME_C128_BUILD(-1.0, 2.0),
        ME_C128_BUILD(2.5, 3.7),
        ME_C128_BUILD(-3.5, 4.2),
        ME_C128_BUILD(0.0, 0.0)
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
        double expected = ME_CREAL(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

void test_windows_complex_helpers() {
#if defined(_WIN32) || defined(_WIN64)
    TEST("Windows complex helper round-trip");

    float _Complex zf = ME_C64_BUILD(1.0f, -2.5f);
    if (fabsf(ME_CREALF(zf) - 1.0f) > TOLERANCE ||
        fabsf(ME_CIMAGF(zf) + 2.5f) > TOLERANCE) {
        printf("  FAIL: float helpers mismatch (real=%.6f imag=%.6f)\n",
               (double)ME_CREALF(zf), (double)ME_CIMAGF(zf));
        tests_failed++;
        return;
    }

    double _Complex zd = ME_C128_BUILD(-3.25, 4.75);
    if (fabs(ME_CREAL(zd) + 3.25) > TOLERANCE ||
        fabs(ME_CIMAG(zd) - 4.75) > TOLERANCE) {
        printf("  FAIL: double helpers mismatch (real=%.6f imag=%.6f)\n",
               ME_CREAL(zd), ME_CIMAG(zd));
        tests_failed++;
        return;
    }

    float _Complex zf_conj = ME_CONJF(zf);
    if (fabsf(ME_CIMAGF(zf_conj) - 2.5f) > TOLERANCE) {
        printf("  FAIL: conj helper mismatch (imag=%.6f)\n", (double)ME_CIMAGF(zf_conj));
        tests_failed++;
        return;
    }

    double _Complex zd_conj = ME_CONJ(zd);
    if (fabs(ME_CIMAG(zd_conj) + 4.75) > TOLERANCE) {
        printf("  FAIL: conj helper mismatch (imag=%.6f)\n", ME_CIMAG(zd_conj));
        tests_failed++;
        return;
    }

    printf("  PASS\n");
#endif
}

int main() {
    printf("=== Testing Complex Functions (conj, imag, real) ===\n\n");

    test_windows_complex_helpers();
    test_conj_c64();
    test_conj_c128();
    test_conj_identity();
    test_imag_c64();
    test_imag_c128();
    test_real_c64();
    test_real_c128();
    test_imag_auto_dtype();
    test_real_auto_dtype();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
