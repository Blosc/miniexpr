/*
 * Complex powers follow NumPy (npy_cpow).  A constant base (2 ** z) used to
 * be computed with the real pow() of the real parts, which dropped the
 * imaginary part.  The C library's cpow() alone also gives NaN for 0 ** 0 and
 * 0 ** z, and inexact results for integer exponents ((1+1j) ** 4 is not -4
 * exactly); NumPy returns 1 for x ** 0, 0 or NaN for a zero base, and
 * multiplies repeatedly for integer exponents with |n| < 100.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include "complex_compat.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define N 9

static int failures = 0;

typedef struct {
    const char *expr;
    /* Expected value for base/exponent element i, NULL: compare with libm cpow(). */
    double _Complex (*ref)(double _Complex z, double _Complex w);
} pow_case;

static double _Complex make(double re, double im) {
    double parts[2] = {re, im};
    double _Complex z;
    memcpy(&z, parts, sizeof(z));
    return z;
}

/* Gaussian integers and dyadic values: their small integer powers are exact. */
static void inputs(double _Complex *z, double _Complex *w) {
    const double zr[N] = {1, -1, 0, 2, 0, 2, -3, 0.5, 1};
    const double zi[N] = {1, 0, 0, 0, 1, -1, 2, 0.25, 0};
    const double wr[N] = {0.5, 2.5, 0, 1, -0.5, 1, 0, -2, 0.25};
    const double wi[N] = {0.25, -1, 0, 1, 0, 0, 1, 0.5, 0};
    for (int i = 0; i < N; i++) {
        z[i] = make(zr[i], zi[i]);
        w[i] = make(wr[i], wi[i]);
    }
}

static double _Complex mul(double _Complex a, double _Complex b) {
    return make(creal(a) * creal(b) - cimag(a) * cimag(b), creal(a) * cimag(b) + cimag(a) * creal(b));
}

static double _Complex ipow(double _Complex z, int n) {
    double _Complex r = make(1, 0);
    for (int k = 0; k < (n < 0 ? -n : n); k++) r = mul(r, z);
    if (n >= 0) return r;
    /* 1 / r; exact for the inverses used below (powers of 2, i, 1 + i). */
    double d = creal(r) * creal(r) + cimag(r) * cimag(r);
    return make(creal(r) / d, -cimag(r) / d);
}

static double _Complex r_zero_exp(double _Complex z, double _Complex w) { (void)z; (void)w; return make(1, 0); }
static double _Complex r_zero_base(double _Complex z, double _Complex w) {
    (void)z;
    if (creal(w) == 0 && cimag(w) == 0) return make(1, 0);
    return creal(w) > 0 ? make(0, 0) : make(NAN, NAN);
}
static double _Complex r_sq(double _Complex z, double _Complex w) { (void)w; return ipow(z, 2); }
static double _Complex r_cube(double _Complex z, double _Complex w) { (void)w; return ipow(z, 3); }
static double _Complex r_p4(double _Complex z, double _Complex w) { (void)w; return ipow(z, 4); }
static double _Complex r_p7(double _Complex z, double _Complex w) { (void)w; return ipow(z, 7); }
/* Negative exponents, used on bases with exact inverses ((0 - 2) folds to -2). */
static double _Complex r_inv_sq(double _Complex z, double _Complex w) { (void)w; return ipow(z, -2); }
static double _Complex r_inv_cube(double _Complex z, double _Complex w) { (void)w; return ipow(z, -3); }
static double _Complex r_inv(double _Complex z, double _Complex w) { (void)w; return ipow(z, -1); }

static const pow_case cases[] = {
    {"z ** 0", r_zero_exp},
    {"0 ** w", r_zero_base},
    {"z ** 2", r_sq},
    {"z ** 3", r_cube},
    {"z ** 4", r_p4},
    {"z ** 7", r_p7},
    /* Non-integer exponents go to the C library's cpow(), as in NumPy. */
    {"2 ** w", NULL},
    {"0.5 ** w", NULL},
    {"2 ** (w + 1)", NULL},
    {"z ** 2.5", NULL},
    {"z ** w", NULL},
};

/* The base and exponent of a libm case (z, w and the literals above). */
static void libm_operands(const char *expr, double _Complex z, double _Complex w, double _Complex *a,
                          double _Complex *b) {
    *a = z;
    *b = w;
    if (!strcmp(expr, "2 ** w")) *a = 2;
    else if (!strcmp(expr, "0.5 ** w")) *a = 0.5;
    else if (!strcmp(expr, "2 ** (w + 1)")) *a = 2, *b = w + 1;
    else if (!strcmp(expr, "z ** 2.5")) *b = 2.5;
}

static bool same(double got, double want, double rtol) {
    if (isnan(want)) return isnan(got);
    if (rtol == 0) return got == want;
    return fabs(got - want) <= rtol * fabs(want) + 1e-300;
}

static void run(const char *expr, me_dtype dt, bool disable_simd, double _Complex (*ref)(double _Complex, double _Complex),
                const double _Complex *z, const double _Complex *w, int nitems, bool exact) {
    float _Complex zf[N], wf[N], of[N];
    double _Complex od[N];
    for (int i = 0; i < nitems; i++) {
        zf[i] = (float _Complex)z[i];
        wf[i] = (float _Complex)w[i];
    }
    const bool uses_z = strchr(expr, 'z') != NULL, uses_w = strchr(expr, 'w') != NULL;
    me_variable vars[2];
    const void *ptrs[2];
    int nvars = 0;
    if (uses_z) {
        vars[nvars] = (me_variable){"z", dt};
        ptrs[nvars++] = dt == ME_COMPLEX64 ? (const void *)zf : (const void *)z;
    }
    if (uses_w) {
        vars[nvars] = (me_variable){"w", dt};
        ptrs[nvars++] = dt == ME_COMPLEX64 ? (const void *)wf : (const void *)w;
    }
    int err = 0;
    me_expr *e = NULL;
    if (me_compile(expr, vars, nvars, dt, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s' (dtype %d)\n", expr, dt);
        failures++;
        return;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.disable_simd = disable_simd;
    if (me_eval(e, ptrs, nvars, dt == ME_COMPLEX64 ? (void *)of : (void *)od, nitems, &params) != ME_EVAL_SUCCESS) {
        printf("  FAIL: eval '%s'\n", expr);
        failures++;
        me_free(e);
        return;
    }
    me_free(e);
    for (int i = 0; i < nitems; i++) {
        double _Complex want;
        if (ref) {
            want = ref(z[i], w[i]);
        }
        else {
            double _Complex a, b;
            libm_operands(expr, z[i], w[i], &a, &b);
            if ((creal(b) == 0 && cimag(b) == 0) || (creal(a) == 0 && cimag(a) == 0)) {
                want = r_zero_base(a, b);
            }
            else if (dt == ME_COMPLEX64) {
                /* What NumPy calls for complex64: cpowf() of the float operands. */
                want = cpowf((float _Complex)a, (float _Complex)b);
            }
            else {
                want = cpow(a, b);
            }
        }
        double _Complex got = dt == ME_COMPLEX64 ? (double _Complex)of[i] : od[i];
        const double rtol = exact ? 0 : (dt == ME_COMPLEX64 ? 1e-6 : 1e-14);
        if (!same(creal(got), creal(want), rtol) || !same(cimag(got), cimag(want), rtol)) {
            printf("  FAIL: '%s' (dtype %d%s) [%d] z=(%g,%g) w=(%g,%g): got (%.17g,%.17g), expected (%.17g,%.17g)\n",
                   expr, dt, disable_simd ? ", no SIMD" : "", i, creal(z[i]), cimag(z[i]), creal(w[i]), cimag(w[i]),
                   creal(got), cimag(got), creal(want), cimag(want));
            failures++;
            return;
        }
    }
}

/* A DSL local assigned from itself (w = 2 ** w) evaluates with the output
 * buffer as the operand: the constant must not be broadcast over it first.
 * Compared with the same expression into a separate output. */
static void check_in_place(me_dtype dt) {
    static const char *dsl[] = {"    w = 2 ** w\n", "    w = 2 / w\n", "    w = w / 4\n"};
    static const char *expr[] = {"2 ** z", "2 / z", "z / 4"};
    double _Complex z[N], w_unused[N];
    inputs(z, w_unused);
    float _Complex zf[N];
    for (int i = 0; i < N; i++) zf[i] = (float _Complex)z[i];
    const void *ptrs[] = {dt == ME_COMPLEX64 ? (const void *)zf : (const void *)z};
    me_variable vars[] = {{"z", dt}};
    for (int k = 0; k < 3; k++) {
        char source[128];
        snprintf(source, sizeof(source), "def k(z):\n    w = z + 0\n%s    return w\n", dsl[k]);
        double _Complex got[N], want[N];
        int64_t shape[] = {N};
        int32_t grid[] = {N};
        int err = 0;
        me_expr *e = NULL, *f = NULL;
        if (me_compile_nd_jit(source, vars, 1, dt, 1, shape, grid, grid, ME_JIT_OFF, &err, &e) != ME_COMPILE_SUCCESS ||
            me_compile(expr[k], vars, 1, dt, &err, &f) != ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile in-place '%s'\n", expr[k]);
            failures++;
            me_free(e);
            me_free(f);
            continue;
        }
        ME_EVAL_CHECK(e, ptrs, 1, got, N);
        ME_EVAL_CHECK(f, ptrs, 1, want, N);
        const size_t bytes = dt == ME_COMPLEX64 ? sizeof(float _Complex) : sizeof(double _Complex);
        for (int i = 0; i < N; i++) {
            double g[2], h[2];
            if (dt == ME_COMPLEX64) {
                float gf[2], hf[2];
                memcpy(gf, (char *)got + i * bytes, bytes);
                memcpy(hf, (char *)want + i * bytes, bytes);
                g[0] = gf[0]; g[1] = gf[1]; h[0] = hf[0]; h[1] = hf[1];
            } else {
                memcpy(g, (char *)got + i * bytes, bytes);
                memcpy(h, (char *)want + i * bytes, bytes);
            }
            if (!same(g[0], h[0], 0) || !same(g[1], h[1], 0)) {
                printf("  FAIL: in-place '%s' (dtype %d) [%d] = (%g%+gj), expected (%g%+gj)\n", expr[k], dt, i, g[0],
                       g[1], h[0], h[1]);
                failures++;
                break;
            }
        }
        me_free(e);
        me_free(f);
    }
}

int main(void) {
    printf("Test: complex powers follow NumPy\n");
    check_in_place(ME_COMPLEX64);
    check_in_place(ME_COMPLEX128);
    double _Complex z[N], w[N];
    inputs(z, w);
    static const me_dtype dtypes[] = {ME_COMPLEX64, ME_COMPLEX128};
    for (int d = 0; d < 2; d++) {
        for (int simd = 0; simd < 2; simd++) {
            for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
                run(cases[k].expr, dtypes[d], simd, cases[k].ref, z, w, N, cases[k].ref != NULL);
            }
            /* Bases whose inverse powers are exact: 2, i, 1 + i, -1. */
            double _Complex zb[N] = {make(2, 0), make(0, 1), make(1, 1), make(-1, 0)}, wb[N] = {0};
            run("z ** (0 - 2)", dtypes[d], simd, r_inv_sq, zb, wb, 4, true);
            run("z ** (0 - 3)", dtypes[d], simd, r_inv_cube, zb, wb, 4, true);
            run("z ** (0 - 1)", dtypes[d], simd, r_inv, zb, wb, 4, true);
        }
    }
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All complex power tests passed\n");
    return 0;
}
