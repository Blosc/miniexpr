/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/
/* ABI-safe wrappers for the C99 complex function set. Internal header, not
 * installed. Under clang-cl, <complex.h> resolves to the UCRT header, which
 * declares csqrt/cexp/clog/cpow/... in terms of the _Fcomplex/_Dcomplex
 * struct types ({T _Val[2]}), not native _Complex. Calling them with clang's
 * native _Complex types (e.g. __builtin_cexp lowering to a "cexp" libcall)
 * crosses a libm boundary with unverified ABI; the wrappers below always
 * convert through the UCRT struct types. */
#ifndef ME_COMPLEX_H
#define ME_COMPLEX_H

#include <complex.h>
#include <string.h>

typedef float _Complex float_complex;
typedef double _Complex double_complex;

/* Arithmetic stays native under clang-cl. Multiplication deliberately follows
 * NumPy's component-wise semantics, without C99 infinity recovery. Copying
 * components avoids introducing NaNs through real + I * imag construction. */
#define add_c64(a, b) ((a) + (b))
#define sub_c64(a, b) ((a) - (b))
#define neg_c64(a) (-(a))
#define div_c64(a, b) ((a) / (b))
#define add_c128(a, b) ((a) + (b))
#define sub_c128(a, b) ((a) - (b))
#define neg_c128(a) (-(a))
#define div_c128(a, b) ((a) / (b))

static inline float _Complex mul_c64(float _Complex a, float _Complex b) {
    float av[2], bv[2], parts[2];
    float _Complex result;
    memcpy(av, &a, sizeof(av));
    memcpy(bv, &b, sizeof(bv));
    parts[0] = av[0] * bv[0] - av[1] * bv[1];
    parts[1] = av[0] * bv[1] + av[1] * bv[0];
    memcpy(&result, parts, sizeof(result));
    return result;
}

static inline double _Complex mul_c128(double _Complex a, double _Complex b) {
    double av[2], bv[2], parts[2];
    double _Complex result;
    memcpy(av, &a, sizeof(av));
    memcpy(bv, &b, sizeof(bv));
    parts[0] = av[0] * bv[0] - av[1] * bv[1];
    parts[1] = av[0] * bv[1] + av[1] * bv[0];
    memcpy(&result, parts, sizeof(result));
    return result;
}

#if defined(_MSC_VER)
/* clang-cl (_MSC_VER + __clang__): accessors and conjugation need no libcall
 * across the ABI boundary, so they stay native; everything else converts. */
#define me_cimagf(z) __builtin_cimagf(z)
#define me_cimag(z) __builtin_cimag(z)
#define me_crealf(z) __builtin_crealf(z)
#define me_creal(z) __builtin_creal(z)
#define me_conjf(z) __builtin_conjf(z)
#define me_conj(z) __builtin_conj(z)

static inline float _Complex me_cpowf(float _Complex a, float _Complex b) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ub, ur;
    ua.c = a;
    ub.c = b;
    ur.m = cpowf(ua.m, ub.m);
    return ur.c;
}
static inline double _Complex me_cpow(double _Complex a, double _Complex b) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ub, ur;
    ua.c = a;
    ub.c = b;
    ur.m = cpow(ua.m, ub.m);
    return ur.c;
}
static inline float me_cabsf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } u;
    u.c = a;
    return cabsf(u.m);
}
static inline double me_cabs(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } u;
    u.c = a;
    return cabs(u.m);
}

/* me_c<name>f(...) converts to _Fcomplex, calls the UCRT <name>f, and
 * converts back; the double variant does the same with _Dcomplex. */
#define ME_COMPLEX_CALL_F(fn) \
    static inline float _Complex me_##fn(float _Complex a) { \
        union { \
            float _Complex c; \
            _Fcomplex m; \
        } u, r; \
        u.c = a; \
        r.m = fn(u.m); \
        return r.c; \
    }
#define ME_COMPLEX_CALL_D(fn) \
    static inline double _Complex me_##fn(double _Complex a) { \
        union { \
            double _Complex c; \
            _Dcomplex m; \
        } u, r; \
        u.c = a; \
        r.m = fn(u.m); \
        return r.c; \
    }
ME_COMPLEX_CALL_F(csqrtf)
ME_COMPLEX_CALL_F(cexpf)
ME_COMPLEX_CALL_F(clogf)
ME_COMPLEX_CALL_F(csinf)
ME_COMPLEX_CALL_F(ccosf)
ME_COMPLEX_CALL_F(ctanf)
ME_COMPLEX_CALL_F(casinf)
ME_COMPLEX_CALL_F(cacosf)
ME_COMPLEX_CALL_F(catanf)
ME_COMPLEX_CALL_F(csinhf)
ME_COMPLEX_CALL_F(ccoshf)
ME_COMPLEX_CALL_F(ctanhf)
ME_COMPLEX_CALL_F(casinhf)
ME_COMPLEX_CALL_F(cacoshf)
ME_COMPLEX_CALL_F(catanhf)
ME_COMPLEX_CALL_D(csqrt)
ME_COMPLEX_CALL_D(cexp)
ME_COMPLEX_CALL_D(clog)
ME_COMPLEX_CALL_D(csin)
ME_COMPLEX_CALL_D(ccos)
ME_COMPLEX_CALL_D(ctan)
ME_COMPLEX_CALL_D(casin)
ME_COMPLEX_CALL_D(cacos)
ME_COMPLEX_CALL_D(catan)
ME_COMPLEX_CALL_D(csinh)
ME_COMPLEX_CALL_D(ccosh)
ME_COMPLEX_CALL_D(ctanh)
ME_COMPLEX_CALL_D(casinh)
ME_COMPLEX_CALL_D(cacosh)
ME_COMPLEX_CALL_D(catanh)
#undef ME_COMPLEX_CALL_F
#undef ME_COMPLEX_CALL_D
#else
#define me_cpowf cpowf
#define me_cpow cpow
#define me_csqrtf csqrtf
#define me_csqrt csqrt
#define me_cexpf cexpf
#define me_cexp cexp
#define me_clogf clogf
#define me_clog clog
#define me_cabsf cabsf
#define me_cabs cabs
#define me_csinf csinf
#define me_csin csin
#define me_ccosf ccosf
#define me_ccos ccos
#define me_ctanf ctanf
#define me_ctan ctan
#define me_casinf casinf
#define me_casin casin
#define me_cacosf cacosf
#define me_cacos cacos
#define me_catanf catanf
#define me_catan catan
#define me_csinhf csinhf
#define me_csinh csinh
#define me_ccoshf ccoshf
#define me_ccosh ccosh
#define me_ctanhf ctanhf
#define me_ctanh ctanh
#define me_casinhf casinhf
#define me_casinh casinh
#define me_cacoshf cacoshf
#define me_cacosh cacosh
#define me_catanhf catanhf
#define me_catanh catanh
#define me_cimagf cimagf
#define me_cimag cimag
#define me_crealf crealf
#define me_creal creal
#define me_conjf conjf
#define me_conj conj
#endif

#endif /* ME_COMPLEX_H */
