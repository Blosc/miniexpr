/* Test complex functions: conj, imag, and real */
#include "../src/miniexpr.h"
#include "minctest.h"
#ifdef ME_TEST_HAS_ARTIFACT
#include "../src/miniexpr_artifact.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "complex_compat.h"
#include <stdint.h>
#include <string.h>
#include "../src/me_complex.h"



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
    me_expr *expr = NULL;
    int rc_expr = me_compile("conj(z)", vars, 1, ME_COMPLEX64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("conj(z)", vars, 1, ME_COMPLEX128, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("imag(z)", vars, 1, ME_FLOAT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("imag(z)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("conj(conj(z))", vars, 1, ME_COMPLEX128, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, 5);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("imag(z)", vars, 1, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, 5);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("real(z)", vars, 1, ME_FLOAT32, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("real(z)", vars, 1, ME_FLOAT64, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);

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
    me_expr *expr = NULL;
    int rc_expr = me_compile("real(z)", vars, 1, ME_AUTO, &err, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error at position %d\n", err);
        tests_failed++;
        return;
    }

    const void *var_ptrs[] = {z};
    ME_EVAL_CHECK(expr, var_ptrs, 1, result, 5);

    for (int i = 0; i < 5; i++) {
        double expected = CREAL(z[i]);
        ASSERT_NEAR(expected, result[i], i);
    }

    me_free(expr);
    printf("  PASS\n");
}

/* A scalar operand used to send divide through the generic binary fallback,
   which narrows through double and silently dropped the imaginary part. */
void test_scalar_complex_divide() {
    TEST("scalar / complex and complex / scalar keep the imaginary part");

    double _Complex z[VECTOR_SIZE] = {
        3.0 + 4.0*I,  1.0 + 1.0*I,  -2.0 + 0.5*I, 0.25 - 3.0*I, 5.0 + 0.0*I,
        -1.5 - 2.5*I, 0.0 + 2.0*I,  7.0 - 1.0*I,  -0.5 + 0.5*I, 2.5 + 3.7*I
    };
    double _Complex result[VECTOR_SIZE] = {0};
    me_variable vars[] = {{"z", ME_COMPLEX128}};
    const void *var_ptrs[] = {z};

    const char *sources[] = {"1.0 / z", "z / 2.0"};
    for (int s = 0; s < 2; s++) {
        int err;
        me_expr *expr = NULL;
        if (me_compile(sources[s], vars, 1, ME_COMPLEX128, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("  FAIL: compilation error at position %d\n", err);
            tests_failed++;
            return;
        }
        ME_EVAL_CHECK(expr, var_ptrs, 1, result, VECTOR_SIZE);
        for (int i = 0; i < VECTOR_SIZE; i++) {
            double _Complex expected = (s == 0) ? 1.0 / z[i] : z[i] / 2.0;
            ASSERT_COMPLEX_NEAR(expected, result[i], i);
        }
        me_free(expr);
    }
    printf("  PASS\n");
}

/* Unlike C99 complex multiplication, NumPy does not repair NaN components
 * after multiplying infinities. Cover vector and scalar operand paths. */
static void test_multiply_nonfinite(void) {
    TEST("complex multiplication nonfinite components");
    const double parts[][2] = {
        {INFINITY, -INFINITY}, {INFINITY, 1}, {1, INFINITY},
        {NAN, 1}, {1, NAN}, {0, -0.0}, {-0.0, 0}, {2, -3}
    };
    const char *sources[] = {"z * w", "z * 2", "2 * z"};
    for (int precision = 0; precision < 2; precision++) {
        me_dtype dtype = precision ? ME_COMPLEX128 : ME_COMPLEX64;
        size_t width = precision ? sizeof(double) : sizeof(float);
        double z128[64][2], w128[64][2], out128[64][2];
        float z64[64][2], w64[64][2], out64[64][2];
        for (int i = 0; i < 64; i++) {
            for (int c = 0; c < 2; c++) {
                z128[i][c] = parts[i / 8][c];
                w128[i][c] = parts[i % 8][c];
                z64[i][c] = (float)z128[i][c];
                w64[i][c] = (float)w128[i][c];
            }
        }
        me_variable vars[] = {{"z", dtype}, {"w", dtype}};
        const void *inputs[] = {precision ? (void *)z128 : (void *)z64,
                                precision ? (void *)w128 : (void *)w64};
        void *output = precision ? (void *)out128 : (void *)out64;
        for (int s = 0; s < 3; s++) {
            int err;
            me_expr *expr = NULL;
            int nvars = s ? 1 : 2;
            if (me_compile(sources[s], vars, nvars, dtype, &err, &expr) != ME_COMPILE_SUCCESS) {
                tests_failed++;
                return;
            }
            ME_EVAL_CHECK(expr, inputs, nvars, output, 64);
            me_free(expr);
            for (int i = 0; i < 64; i++) {
                double ar = z128[i][0], ai = z128[i][1];
                double br = s ? 2.0 : w128[i][0], bi = s ? 0.0 : w128[i][1];
                /* Separate products also prevent contraction in the reference. */
                volatile double ac = ar * br, bd = ai * bi;
                volatile double ad = ar * bi, bc = ai * br;
                double expected[] = {ac - bd, ad + bc};
                for (int c = 0; c < 2; c++) {
                    double actual = width == sizeof(double) ? out128[i][c] : out64[i][c];
                    double value = expected[c];
                    if (!((isnan(value) && isnan(actual)) ||
                          (value == actual && (value != 0 || !!signbit(value) == !!signbit(actual))))) {
                        printf("  FAIL: %s dtype=%d lane=%d component=%d expected=%g actual=%g\n",
                               sources[s], dtype, i, c, value, actual);
                        tests_failed++;
                        return;
                    }
                }
            }
        }
    }
    printf("  PASS\n");
}

/* The complex libm wrappers (me_cexp/me_csqrt/me_clog/me_cpow/me_cabs) must
 * hand values across the complex ABI without corrupting components. This is
 * the failure mode a _Dcomplex-vs-_Complex mismatch would produce, so
 * cross-check the wrapper results against the same values rebuilt from
 * component-only libm (exp, cos, sin, hypot, atan2), for finite inputs
 * (including signed zeros) and nonfinite specials. */
static int abi_close(double expected, double actual, double tol) {
    if (isnan(expected)) return isnan(actual);
    if (isinf(expected)) return actual == expected;
    return fabs(expected - actual) <= tol * (1.0 + fabs(expected));
}

/* Call the wrappers directly: abs(z) uses hypot on clang-cl, and exp/log
 * expression support depends on the optional artifact adapter. memcpy avoids
 * changing signed zeros or nonfinite components while constructing values. */
static void test_direct_libm_abi(void) {
    TEST("direct complex libm ABI, both precisions and signed zeros");
    static const double values[][2] = {
        {1.0, 2.0}, {-0.5, 1.5}, {-3.0, -4.0},
        {4.0, -0.0}, {2.0, 0.0},
        {INFINITY, 0.0}, {-INFINITY, 0.0}, {NAN, 0.0}
    };
    const char *names[] = {"exp", "sqrt", "log", "pow", "abs"};
    for (int precision = 0; precision < 2; precision++) {
        for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
            double x = values[i][0], y = values[i][1];
            double magnitude = hypot(x, y);
            double expected[5][2] = {{0}};
            if (i < 5) {
                expected[0][0] = exp(x) * cos(y);
                expected[0][1] = exp(x) * sin(y);
                expected[1][0] = sqrt((magnitude + x) / 2.0);
                expected[1][1] = copysign(sqrt((magnitude - x) / 2.0), y);
                expected[2][0] = log(magnitude);
                expected[2][1] = atan2(y, x);
                expected[3][0] = exp(3.0 * log(magnitude)) * cos(3.0 * atan2(y, x));
                expected[3][1] = exp(3.0 * log(magnitude)) * sin(3.0 * atan2(y, x));
            }
            else if (i == 5) {
                expected[0][0] = expected[1][0] = expected[2][0] = INFINITY;
            }
            else if (i == 6) {
                expected[1][1] = INFINITY;
                expected[2][0] = INFINITY;
                expected[2][1] = atan2(0.0, -1.0);
            }
            else {
                expected[0][0] = expected[1][0] = expected[2][0] = NAN;
                expected[1][1] = expected[2][1] = NAN;
            }
            expected[4][0] = magnitude;
            for (int p = 0; p < 5; p++) {
                /* Nonfinite complex powers have platform-specific semantics. */
                if (i >= 5 && p == 3) continue;
                /* Apple libm differs from Annex G for sqrt(-inf + i0). */
                if (i == 6 && p == 1) continue;
                double actual[2] = {0};
                if (precision) {
                    double _Complex z, w, result;
                    double exponent[2] = {3.0, 0.0};
                    memcpy(&z, values[i], sizeof(z));
                    memcpy(&w, exponent, sizeof(w));
                    if (p == 4) {
                        actual[0] = me_cabs(z);
                    }
                    else {
                        result = p == 0 ? me_cexp(z) : p == 1 ? me_csqrt(z) :
                                 p == 2 ? me_clog(z) : me_cpow(z, w);
                        memcpy(actual, &result, sizeof(result));
                    }
                }
                else {
                    float parts[2] = {(float)x, (float)y}, exponent[2] = {3.0f, 0.0f};
                    float _Complex z, w, result;
                    memcpy(&z, parts, sizeof(z));
                    memcpy(&w, exponent, sizeof(w));
                    if (p == 4) {
                        actual[0] = me_cabsf(z);
                    }
                    else {
                        result = p == 0 ? me_cexpf(z) : p == 1 ? me_csqrtf(z) :
                                 p == 2 ? me_clogf(z) : me_cpowf(z, w);
                        memcpy(parts, &result, sizeof(result));
                        actual[0] = parts[0];
                        actual[1] = parts[1];
                    }
                }
                for (int c = 0; c < (p == 4 ? 1 : 2); c++) {
                    double reference = expected[p][c];
                    /* Check zero signs for sqrt/log branch cuts and abs.
                     * Apple cexpf loses the imaginary -0; polar power is also
                     * only an approximate reference. */
                    int zero_ok = p == 0 || p == 3 || reference != 0.0 ||
                        (actual[c] == 0.0 && !!signbit(reference) == !!signbit(actual[c]));
                    if (!abi_close(reference, actual[c], precision ? 1e-9 : 5e-6) || !zero_ok) {
                        printf("  FAIL: %s precision=%d case=%zu component=%d expected=%.17g actual=%.17g\n",
                               names[p], precision, i, c, reference, actual[c]);
                        tests_failed++;
                        return;
                    }
                }
            }
        }
    }
    printf("  PASS\n");
}

static void test_libm_abi(void) {
    TEST("complex libm ABI: wrapper results vs component-only libm");
    static const double cases[][2] = {
        {1.0, 2.0}, {-0.5, 1.5}, {2.0, -0.25}, {-3.0, 4.0},
        {0.5, 0.0}, {1.0, -0.0}, {-3.0, -0.0},
        {-0.0, 1.25}, {-0.0, -1.25},
        {0.0, -0.0}, {-0.0, 0.0}, {0.0, 1.5}, {0.0, -1.5},
        {-3.0, 0.0}
    };
    const int ncases = (int)(sizeof(cases) / sizeof(*cases));
    /* sqrt/pow/abs are the complex functions me_compile accepts; exp/log only
     * exist in the portable interpreter and are checked below. */
    static const struct {
        const char *expr;
        bool is_real_output;
        double tol;
    } probes[] = {
        {"sqrt(z)", false, 1e-9},
        {"pow(z, 3)", false, 1e-9},
        {"abs(z)", true, 1e-9},
    };
    const int nprobes = (int)(sizeof(probes) / sizeof(*probes));
    {
        double z[3][2] = {
            {NAN, 0.0}, {INFINITY, 4.0}, {INFINITY, -INFINITY},
        };
        /* Reference the host hypot(): cabs must agree with the real-libm
         * magnitude of the same components on every platform. */
        double lr[3] = {hypot(z[0][0], z[0][1]), hypot(z[1][0], z[1][1]), hypot(z[2][0], z[2][1])};
        double li[3] = {0.0, 0.0, 0.0};
        me_variable vars[] = {{"z", ME_COMPLEX128}};
        for (int i = 0; i < 3; i++) {
            const void *inputs[] = {z[i]};
            me_expr *expr = NULL;
            if (me_compile("abs(z)", vars, 1, ME_AUTO, 0, &expr) != ME_COMPILE_SUCCESS) {
                printf("  FAIL: abs(z) compile error\n");
                tests_failed++;
                return;
            }
            double out[2] = {0.0, 0.0};
            ME_EVAL_CHECK(expr, inputs, 1, out, 1);
            me_free(expr);
            /* abs() is real-valued: only the magnitude lane is written. */
            if (!abi_close(lr[i], out[0], 1e-9)) {
                printf("  FAIL: %s nonfinite case %d: expected (%.17g, %.17g), got (%.17g, %.17g)\n",
                       "abs(z)", i, lr[i], li[i], out[0], out[1]);
                tests_failed++;
                return;
            }
        }
    }
    for (int precision = 0; precision < 2; precision++) {
        double z128[64][2];
        float z64[64][2];
        for (int i = 0; i < ncases; i++) {
            z128[i][0] = cases[i][0];
            z128[i][1] = cases[i][1];
            z64[i][0] = (float)cases[i][0];
            z64[i][1] = (float)cases[i][1];
        }
        me_variable vars[] = {{"z", precision ? ME_COMPLEX128 : ME_COMPLEX64}};
        const void *inputs[] = {precision ? (void *)z128 : (void *)z64};
        for (int p = 0; p < nprobes; p++) {
            me_dtype out_kind = probes[p].is_real_output ?
                (precision ? ME_FLOAT64 : ME_FLOAT32) :
                (precision ? ME_COMPLEX128 : ME_COMPLEX64);
            me_expr *expr = NULL;
            int rc_expr = me_compile(probes[p].expr, vars, 1, out_kind, 0, &expr);
            if (rc_expr != ME_COMPILE_SUCCESS) {
                printf("  FAIL: %s (precision %d) compile error %d\n",
                       probes[p].expr, precision, rc_expr);
                tests_failed++;
                return;
            }
            double out128[128];
            float out64[128];
            void *output = precision ? (void *)out128 : (void *)out64;
            ME_EVAL_CHECK(expr, inputs, 1, output, ncases);
            me_free(expr);
            for (int i = 0; i < ncases; i++) {
                double x = cases[i][0], y = cases[i][1];
                double expected[2] = {0.0, 0.0};
                double magnitude = hypot(x, y);
                if (!strcmp(probes[p].expr, "sqrt(z)")) {
                    expected[0] = sqrt((magnitude + x) / 2.0);
                    expected[1] = copysign(sqrt((magnitude - x) / 2.0), y);
                }
                else if (!strcmp(probes[p].expr, "pow(z, 3)")) {
                    /* pow = exp(w * log z) with w = 3 + 0i */
                    double lr = log(magnitude), li = atan2(y, x);
                    expected[0] = exp(3.0 * lr) * cos(3.0 * li);
                    expected[1] = exp(3.0 * lr) * sin(3.0 * li);
                }
                else {
                    expected[0] = magnitude;
                }
                for (int c = 0; c < (probes[p].is_real_output ? 1 : 2); c++) {
                    int lane = probes[p].is_real_output ? i : 2 * i + c;
                    double actual = precision ? out128[lane] : out64[lane];
                    double value = expected[c];
                    /* Float precision results carry a few float-ulp pipeline
                     * noise over the double-precision reference. */
                    double tol = precision ? probes[p].tol : probes[p].tol * 500.0;
                    if (!abi_close(value, actual, tol)) {
                        printf("  FAIL: %s (precision %d) case %d component %d: "
                               "expected %.17g, got %.17g\n",
                               probes[p].expr, precision, i, c, value, actual);
                        tests_failed++;
                        return;
                    }
                }
            }
        }
    }

    /* Nonfinite specials (double precision). exp/log of complex live in the
     * portable interpreter; reach their wrappers through the artifact JSON
     * loader when the artifact adapter is linked. */
#ifdef ME_TEST_HAS_ARTIFACT
    {
        static const double values[][2] = {
            {INFINITY, 0.0},   /* exp(inf + i0)  = inf + i0       */
            {-INFINITY, 0.0},  /* exp(-inf + i0) = 0 + i0         */
            {INFINITY, 0.0},   /* log(inf + i0)  = inf + i0       */
            {-INFINITY, 0.0},  /* log(-inf + i0) = inf + i * pi   */
            {0.0, 0.0},        /* log(0 + i0)    = -inf + i0      */
        };
        const double lr[] = {INFINITY, 0.0, INFINITY, INFINITY, -INFINITY};
        const double li[] = {0.0, 0.0, 0.0, M_PI, 0.0};
        const char *sources[] = {
            "def k(z):\\n    return exp(z)\\n",
            "def k(z):\\n    return exp(z)\\n",
            "def k(z):\\n    return log(z)\\n",
            "def k(z):\\n    return log(z)\\n",
            "def k(z):\\n    return log(z)\\n",
        };
        double z[5][2], out[5][2];
        const int nspecials = (int)(sizeof(values) / sizeof(*values));
        size_t count = (size_t)nspecials;
        int64_t shape = nspecials, origin = 0, extent = nspecials;
        me_artifact_eval_descriptor descriptor = {
            .struct_size = sizeof(descriptor), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
            .nitems = count, .output_capacity = count * sizeof(double complex), .ndim = 1,
            .logical_shape = &shape, .block_origin = &origin, .block_extent = &extent
        };
        for (int i = 0; i < nspecials; i++) {
            z[i][0] = values[i][0];
            z[i][1] = values[i][1];
        }
        me_artifact_buffer input = {"z", ME_COMPLEX128, sizeof(double complex), z,
                                    count * sizeof(double complex)};
        for (int i = 0; i < nspecials; i++) {
            char json[4096];
            int n = snprintf(json, sizeof(json),
                "{\"schema_version\":\"0.1\",\"language\":{\"name\":\"menudet\",\"version\":\"0.1\"},"
                "\"requires\":[\"numeric\",\"nd-context\"],"
                "\"source\":\"%s\",\"entry_point\":\"k\","
                "\"inputs\":[{\"name\":\"z\",\"dtype\":\"complex128\"}],\"constants\":[],"
                "\"output\":{\"dtype\":\"complex128\",\"contract\":\"elementwise\"},"
                "\"context\":{\"ndim\":1},"
                "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
                sources[i]);
            if (n <= 0 || (size_t)n >= sizeof(json)) {
                printf("  FAIL: encoding artifact source\n");
                tests_failed++;
                return;
            }
            me_artifact *artifact = NULL;
            me_artifact_error error;
            /* Interpreter only: the JIT backend rejects complex kernels. */
            if (me_artifact_load(json, (size_t)n, ME_JIT_OFF, &artifact, &error)) {
                printf("  FAIL: artifact load: %s\n", error.message);
                tests_failed++;
                return;
            }
            if (me_artifact_eval_ex(artifact, &input, 1, out, &descriptor, &error)) {
                printf("  FAIL: artifact eval: %s\n", error.message);
                tests_failed++;
                me_artifact_free(artifact);
                return;
            }
            me_artifact_free(artifact);
            if (!abi_close(lr[i], out[i][0], 1e-9) || !abi_close(li[i], out[i][1], 1e-9)) {
                printf("  FAIL: nonfinite case %d: expected (%.17g, %.17g), got (%.17g, %.17g)\n",
                       i, lr[i], li[i], out[i][0], out[i][1]);
                tests_failed++;
                return;
            }
        }
    }
#endif


    printf("  PASS\n");
}

int main() {
    printf("=== Testing Complex Functions (conj, imag, real) ===\n\n");

    test_conj_c64();
    test_conj_c128();
    test_conj_identity();
    test_imag_c64();
    test_imag_c128();
    test_real_c64();
    test_real_c128();
    test_imag_auto_dtype();
    test_real_auto_dtype();
    test_scalar_complex_divide();
    test_multiply_nonfinite();
    test_libm_abi();
    test_direct_libm_abi();

    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_run - tests_failed);
    printf("Tests failed: %d\n", tests_failed);

    return (tests_failed == 0) ? 0 : 1;
}
