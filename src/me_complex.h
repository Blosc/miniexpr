#ifndef ME_COMPLEX_H
#define ME_COMPLEX_H

#include <complex.h>

#if defined(_WIN32) || defined(_WIN64)
#include <string.h>

typedef struct { float re; float im; } me_c64_pair;
typedef struct { double re; double im; } me_c128_pair;

static inline float _Complex me_c64_build(float real, float imag) {
    me_c64_pair p = {real, imag};
    float _Complex z;
    memcpy(&z, &p, sizeof(p));
    return z;
}

static inline double _Complex me_c128_build(double real, double imag) {
    me_c128_pair p = {real, imag};
    double _Complex z;
    memcpy(&z, &p, sizeof(p));
    return z;
}

static inline float me_crealf(float _Complex a) {
    me_c64_pair p;
    memcpy(&p, &a, sizeof(p));
    return p.re;
}

static inline double me_creal(double _Complex a) {
    me_c128_pair p;
    memcpy(&p, &a, sizeof(p));
    return p.re;
}

static inline float me_cimagf(float _Complex a) {
    me_c64_pair p;
    memcpy(&p, &a, sizeof(p));
    return p.im;
}

static inline double me_cimag(double _Complex a) {
    me_c128_pair p;
    memcpy(&p, &a, sizeof(p));
    return p.im;
}

static inline float _Complex me_conjf(float _Complex a) {
    return me_c64_build(me_crealf(a), -me_cimagf(a));
}

static inline double _Complex me_conj(double _Complex a) {
    return me_c128_build(me_creal(a), -me_cimag(a));
}

#if defined(_MSC_VER) && !defined(__clang__)
static inline float _Complex me_cpowf(float _Complex a, float _Complex b) {
    union { float _Complex c; _Fcomplex m; } ua, ub, ur;
    ua.c = a;
    ub.c = b;
    ur.m = cpowf(ua.m, ub.m);
    return ur.c;
}

static inline double _Complex me_cpow(double _Complex a, double _Complex b) {
    union { double _Complex c; _Dcomplex m; } ua, ub, ur;
    ua.c = a;
    ub.c = b;
    ur.m = cpow(ua.m, ub.m);
    return ur.c;
}

static inline float _Complex me_csqrtf(float _Complex a) {
    union { float _Complex c; _Fcomplex m; } ua, ur;
    ua.c = a;
    ur.m = csqrtf(ua.m);
    return ur.c;
}

static inline double _Complex me_csqrt(double _Complex a) {
    union { double _Complex c; _Dcomplex m; } ua, ur;
    ua.c = a;
    ur.m = csqrt(ua.m);
    return ur.c;
}

static inline float _Complex me_cexpf(float _Complex a) {
    union { float _Complex c; _Fcomplex m; } ua, ur;
    ua.c = a;
    ur.m = cexpf(ua.m);
    return ur.c;
}

static inline double _Complex me_cexp(double _Complex a) {
    union { double _Complex c; _Dcomplex m; } ua, ur;
    ua.c = a;
    ur.m = cexp(ua.m);
    return ur.c;
}

static inline float _Complex me_clogf(float _Complex a) {
    union { float _Complex c; _Fcomplex m; } ua, ur;
    ua.c = a;
    ur.m = clogf(ua.m);
    return ur.c;
}

static inline double _Complex me_clog(double _Complex a) {
    union { double _Complex c; _Dcomplex m; } ua, ur;
    ua.c = a;
    ur.m = clog(ua.m);
    return ur.c;
}

static inline float me_cabsf(float _Complex a) {
    union { float _Complex c; _Fcomplex m; } ua;
    ua.c = a;
    return cabsf(ua.m);
}

static inline double me_cabs(double _Complex a) {
    union { double _Complex c; _Dcomplex m; } ua;
    ua.c = a;
    return cabs(ua.m);
}
#elif defined(__clang__)
#define me_cpowf __builtin_cpowf
#define me_cpow __builtin_cpow
#define me_csqrtf __builtin_csqrtf
#define me_csqrt __builtin_csqrt
#define me_cexpf __builtin_cexpf
#define me_cexp __builtin_cexp
#define me_clogf __builtin_clogf
#define me_clog __builtin_clog
#define me_cabsf __builtin_cabsf
#define me_cabs __builtin_cabs
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
#endif

#elif defined(__clang__)
#define me_c64_build(real, imag) __builtin_complex((float)(real), (float)(imag))
#define me_c128_build(real, imag) __builtin_complex((double)(real), (double)(imag))
#define me_crealf __builtin_crealf
#define me_creal __builtin_creal
#define me_cimagf __builtin_cimagf
#define me_cimag __builtin_cimag
#define me_cpowf __builtin_cpowf
#define me_cpow __builtin_cpow
#define me_csqrtf __builtin_csqrtf
#define me_csqrt __builtin_csqrt
#define me_cexpf __builtin_cexpf
#define me_cexp __builtin_cexp
#define me_clogf __builtin_clogf
#define me_clog __builtin_clog
#define me_cabsf __builtin_cabsf
#define me_cabs __builtin_cabs
#else
#define me_c64_build(real, imag) CMPLXF((real), (imag))
#define me_c128_build(real, imag) CMPLX((real), (imag))
#define me_crealf crealf
#define me_creal creal
#define me_cimagf cimagf
#define me_cimag cimag
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
#endif
#if !defined(_WIN32) && !defined(_WIN64)
static inline float _Complex me_conjf(float _Complex a) {
    return me_c64_build(me_crealf(a), -me_cimagf(a));
}

static inline double _Complex me_conj(double _Complex a) {
    return me_c128_build(me_creal(a), -me_cimag(a));
}
#endif

#define ME_C64_BUILD(real, imag) me_c64_build((real), (imag))
#define ME_C128_BUILD(real, imag) me_c128_build((real), (imag))
#define ME_CREALF(z) me_crealf((z))
#define ME_CIMAGF(z) me_cimagf((z))
#define ME_CREAL(z) me_creal((z))
#define ME_CIMAG(z) me_cimag((z))
#define ME_CONJF(z) me_conjf((z))
#define ME_CONJ(z) me_conj((z))

#endif
