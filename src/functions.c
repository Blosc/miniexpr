/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

// Loosely based on https://github.com/CodePlea/tinyexpr. License follows:
// SPDX-License-Identifier: Zlib
/*
 * TINYEXPR - Tiny recursive descent parser and evaluation engine in C
 *
 * Copyright (c) 2015-2020 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgement in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* COMPILE TIME OPTIONS */

/* Exponentiation associativity:
For a**b**c = (a**b)**c and -a**b = (-a)**b do nothing.
For a**b**c = a**(b**c) and -a**b = -(a**b) uncomment the next line.*/
/* #define ME_POW_FROM_RIGHT */

/* Logarithms
For log = natural log do nothing (NumPy compatible)
For log = base 10 log comment the next line. */
#define ME_NAT_LOG

#include "functions.h"
#include "functions-simd.h"
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#if defined(__SSE2__) || defined(__SSE__) || defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#define IVDEP
#else
#define IVDEP _Pragma("GCC ivdep")
#endif

#include <complex.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define float_complex _Fcomplex
#define double_complex _Dcomplex
// And it doesn't support standard operators for them in C
static inline _Fcomplex add_c64(_Fcomplex a, _Fcomplex b) {
    return _FCbuild(crealf(a) + crealf(b), cimagf(a) + cimagf(b));
}
static inline _Fcomplex sub_c64(_Fcomplex a, _Fcomplex b) {
    return _FCbuild(crealf(a) - crealf(b), cimagf(a) - cimagf(b));
}
static inline _Fcomplex neg_c64(_Fcomplex a) { return _FCbuild(-crealf(a), -cimagf(a)); }
static inline _Fcomplex mul_c64(_Fcomplex a, _Fcomplex b) {
    return _FCbuild(crealf(a) * crealf(b) - cimagf(a) * cimagf(b), crealf(a) * cimagf(b) + cimagf(a) * crealf(b));
}
static inline _Fcomplex div_c64(_Fcomplex a, _Fcomplex b) {
    float denom = crealf(b) * crealf(b) + cimagf(b) * cimagf(b);
    return _FCbuild((crealf(a) * crealf(b) + cimagf(a) * cimagf(b)) / denom,
                    (cimagf(a) * crealf(b) - crealf(a) * cimagf(b)) / denom);
}
static inline _Dcomplex add_c128(_Dcomplex a, _Dcomplex b) { return _Cbuild(creal(a) + creal(b), cimag(a) + cimag(b)); }
static inline _Dcomplex sub_c128(_Dcomplex a, _Dcomplex b) { return _Cbuild(creal(a) - creal(b), cimag(a) - cimag(b)); }
static inline _Dcomplex neg_c128(_Dcomplex a) { return _Cbuild(-creal(a), -cimag(a)); }
static inline _Dcomplex mul_c128(_Dcomplex a, _Dcomplex b) {
    return _Cbuild(creal(a) * creal(b) - cimag(a) * cimag(b), creal(a) * cimag(b) + cimag(a) * creal(b));
}
static inline _Dcomplex div_c128(_Dcomplex a, _Dcomplex b) {
    double denom = creal(b) * creal(b) + cimag(b) * cimag(b);
    return _Cbuild((creal(a) * creal(b) + cimag(a) * cimag(b)) / denom,
                   (cimag(a) * creal(b) - creal(a) * cimag(b)) / denom);
}
#else
#define float_complex float _Complex
#define double_complex double _Complex
#define add_c64(a, b) ((a) + (b))
#define sub_c64(a, b) ((a) - (b))
#define neg_c64(a) (-(a))
#define mul_c64(a, b) ((a) * (b))
#define div_c64(a, b) ((a) / (b))
#define add_c128(a, b) ((a) + (b))
#define sub_c128(a, b) ((a) - (b))
#define neg_c128(a) (-(a))
#define mul_c128(a, b) ((a) * (b))
#define div_c128(a, b) ((a) / (b))
#endif

#if defined(_MSC_VER) && !defined(__clang__)
/* Wrappers for complex functions to handle MSVC's _Fcomplex/_Dcomplex */
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
static inline float _Complex me_csqrtf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = csqrtf(ua.m);
    return ur.c;
}
static inline double _Complex me_csqrt(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = csqrt(ua.m);
    return ur.c;
}
static inline float _Complex me_cexpf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = cexpf(ua.m);
    return ur.c;
}
static inline double _Complex me_cexp(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = cexp(ua.m);
    return ur.c;
}
static inline float _Complex me_clogf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = clogf(ua.m);
    return ur.c;
}
static inline double _Complex me_clog(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = clog(ua.m);
    return ur.c;
}
static inline float me_cabsf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua;
    ua.c = a;
    return cabsf(ua.m);
}
static inline double me_cabs(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua;
    ua.c = a;
    return cabs(ua.m);
}
static inline float me_cimagf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua;
    ua.c = a;
    return cimagf(ua.m);
}
static inline double me_cimag(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua;
    ua.c = a;
    return cimag(ua.m);
}
static inline float me_crealf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua;
    ua.c = a;
    return crealf(ua.m);
}
static inline double me_creal(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua;
    ua.c = a;
    return creal(ua.m);
}
static inline float _Complex me_conjf(float _Complex a) {
    union {
        float _Complex c;
        _Fcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = conjf(ua.m);
    return ur.c;
}
static inline double _Complex me_conj(double _Complex a) {
    union {
        double _Complex c;
        _Dcomplex m;
    } ua, ur;
    ua.c = a;
    ur.m = conj(ua.m);
    return ur.c;
}
#else
#if defined(_MSC_VER) && defined(__clang__)
#define me_cimagf __builtin_cimagf
#define me_cimag __builtin_cimag
#define me_crealf __builtin_crealf
#define me_creal __builtin_creal
#define me_conjf __builtin_conjf
#define me_conj __builtin_conj
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
#define me_cimagf cimagf
#define me_cimag cimag
#define me_crealf crealf
#define me_creal creal
#define me_conjf conjf
#define me_conj conj
#endif
#endif

/* Type-specific cast and comparison macros to handle MSVC complex structs */
#define TO_TYPE_bool(x) (bool)(x)
#define TO_TYPE_i8(x) (int8_t)(x)
#define TO_TYPE_i16(x) (int16_t)(x)
#define TO_TYPE_i32(x) (int32_t)(x)
#define TO_TYPE_i64(x) (int64_t)(x)
#define TO_TYPE_u8(x) (uint8_t)(x)
#define TO_TYPE_u16(x) (uint16_t)(x)
#define TO_TYPE_u32(x) (uint32_t)(x)
#define TO_TYPE_u64(x) (uint64_t)(x)
#define TO_TYPE_f32(x) (float)(x)
#define TO_TYPE_f64(x) (double)(x)

#define FROM_TYPE_bool(x) (double)(x)
#define FROM_TYPE_i8(x) (double)(x)
#define FROM_TYPE_i16(x) (double)(x)
#define FROM_TYPE_i32(x) (double)(x)
#define FROM_TYPE_i64(x) (double)(x)
#define FROM_TYPE_u8(x) (double)(x)
#define FROM_TYPE_u16(x) (double)(x)
#define FROM_TYPE_u32(x) (double)(x)
#define FROM_TYPE_u64(x) (double)(x)
#define FROM_TYPE_f32(x) (double)(x)
#define FROM_TYPE_f64(x) (double)(x)

#define IS_NONZERO_bool(x) (x)
#define IS_NONZERO_i8(x) ((x) != 0)
#define IS_NONZERO_i16(x) ((x) != 0)
#define IS_NONZERO_i32(x) ((x) != 0)
#define IS_NONZERO_i64(x) ((x) != 0)
#define IS_NONZERO_u8(x) ((x) != 0)
#define IS_NONZERO_u16(x) ((x) != 0)
#define IS_NONZERO_u32(x) ((x) != 0)
#define IS_NONZERO_u64(x) ((x) != 0)
#define IS_NONZERO_f32(x) ((x) != 0.0f)
#define IS_NONZERO_f64(x) ((x) != 0.0)

#if defined(_MSC_VER) && !defined(__clang__)
#define TO_TYPE_c64(x) _FCbuild((float)(x), 0.0f)
#define TO_TYPE_c128(x) _Cbuild((double)(x), 0.0)
#define FROM_TYPE_c64(x) (double)crealf(x)
#define FROM_TYPE_c128(x) (double)creal(x)
#define IS_NONZERO_c64(x) (crealf(x) != 0.0f || cimagf(x) != 0.0f)
#define IS_NONZERO_c128(x) (creal(x) != 0.0 || cimag(x) != 0.0)

/* Helper macros for complex-to-complex conversions */
#define CONV_c64_to_c128(x) _Cbuild((double)crealf(x), (double)cimagf(x))
#define TO_TYPE_c128_from_c64(x) CONV_c64_to_c128(x)
#else
#define TO_TYPE_c64(x) (float_complex)(x)
#define TO_TYPE_c128(x) (double_complex)(x)
#define FROM_TYPE_c64(x) (double)me_crealf(x)
#define FROM_TYPE_c128(x) (double)me_creal(x)
#define IS_NONZERO_c64(x) (me_crealf(x) != 0.0f || me_cimagf(x) != 0.0f)
#define IS_NONZERO_c128(x) (me_creal(x) != 0.0 || me_cimag(x) != 0.0)
#define TO_TYPE_c128_from_c64(x) (double_complex)(x)
#endif

#include <assert.h>

#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

/* Portable complex number construction for MSVC compatibility */
static double _Complex me_cmplx(double re, double im) {
#if defined(_MSC_VER)
    double _Complex v;
    __real__ v = re;
    __imag__ v = im;
    return v;
#else
    return re + im * I;
#endif
}



#if defined(_WIN32) || defined(_WIN64)
#endif

static bool is_integer_dtype(me_dtype dt) {
    return dt >= ME_INT8 && dt <= ME_UINT64;
}

static bool is_float_dtype(me_dtype dt) {
    return dt == ME_FLOAT32 || dt == ME_FLOAT64;
}

static bool is_complex_dtype(me_dtype dt) {
    return dt == ME_COMPLEX64 || dt == ME_COMPLEX128;
}

static double sum_reduce(double x);
static double mean_reduce(double x);
static double prod_reduce(double x);
static double any_reduce(double x);
static double all_reduce(double x);
double min_reduce(double x);
double max_reduce(double x);

me_dtype reduction_output_dtype(me_dtype dt, const void* func) {
    if (func == (void*)any_reduce || func == (void*)all_reduce) {
        return ME_BOOL;
    }
    if (func == (void*)mean_reduce) {
        if (dt == ME_COMPLEX64 || dt == ME_COMPLEX128) {
            return ME_COMPLEX128;
        }
        return ME_FLOAT64;
    }
    if (func == (void*)sum_reduce || func == (void*)prod_reduce) {
        if (dt == ME_BOOL) {
            return ME_INT64;
        }
        if (dt >= ME_UINT8 && dt <= ME_UINT64) {
            return ME_UINT64;
        }
        if (dt >= ME_INT8 && dt <= ME_INT64) {
            return ME_INT64;
        }
    }
    return dt;
}

me_reduce_kind reduction_kind(const void* func) {
    if (func == (void*)sum_reduce) return ME_REDUCE_SUM;
    if (func == (void*)mean_reduce) return ME_REDUCE_MEAN;
    if (func == (void*)prod_reduce) return ME_REDUCE_PROD;
    if (func == (void*)min_reduce) return ME_REDUCE_MIN;
    if (func == (void*)max_reduce) return ME_REDUCE_MAX;
    if (func == (void*)any_reduce) return ME_REDUCE_ANY;
    if (func == (void*)all_reduce) return ME_REDUCE_ALL;
    return ME_REDUCE_NONE;
}

/* Get size of a type in bytes */
size_t dtype_size(me_dtype dtype) {
    switch (dtype) {
    case ME_BOOL: return sizeof(bool);
    case ME_INT8: return sizeof(int8_t);
    case ME_INT16: return sizeof(int16_t);
    case ME_INT32: return sizeof(int32_t);
    case ME_INT64: return sizeof(int64_t);
    case ME_UINT8: return sizeof(uint8_t);
    case ME_UINT16: return sizeof(uint16_t);
    case ME_UINT32: return sizeof(uint32_t);
    case ME_UINT64: return sizeof(uint64_t);
    case ME_FLOAT32: return sizeof(float);
    case ME_FLOAT64: return sizeof(double);
    case ME_COMPLEX64: return sizeof(float _Complex);
    case ME_COMPLEX128: return sizeof(double _Complex);
    case ME_STRING: return 0;
    default: return 0;
    }
}

static bool is_reduction_function(const void* func) {
    return func == (void*)sum_reduce || func == (void*)mean_reduce ||
        func == (void*)prod_reduce || func == (void*)min_reduce ||
        func == (void*)max_reduce || func == (void*)any_reduce ||
        func == (void*)all_reduce;
}

bool is_reduction_node(const me_expr* n) {
    return n && IS_FUNCTION(n->type) && ARITY(n->type) == 1 &&
        is_reduction_function(n->function);
}

static bool contains_reduction(const me_expr* n) {
    if (!n) return false;
    if (is_reduction_node(n)) return true;
    if (IS_FUNCTION(n->type) || IS_CLOSURE(n->type)) {
        const int arity = ARITY(n->type);
        for (int i = 0; i < arity; i++) {
            if (contains_reduction((const me_expr*)n->parameters[i])) return true;
        }
    }
    return false;
}

static void private_eval(const me_expr* n);
static void eval_reduction(const me_expr* n, int output_nitems);
static double pi(void) { return 3.14159265358979323846; }
static double e(void) { return 2.71828182845904523536; }

/* Wrapper for expm1: exp(x) - 1, more accurate for small x */
static double expm1_wrapper(double x) { return expm1(x); }

/* Wrapper for log1p: log(1 + x), more accurate for small x */
static double log1p_wrapper(double x) { return log1p(x); }

/* Wrapper for log2: base-2 logarithm */
static double log2_wrapper(double x) { return log2(x); }

/* Wrapper for exp10: base-10 exponent */
static double exp10_wrapper(double x) { return pow(10.0, x); }

/* Wrapper for sinpi: sin(pi * x) */
static double sinpi_wrapper(double x) { return sin(pi() * x); }

/* Wrapper for cospi: cos(pi * x) */
static double cospi_wrapper(double x) { return cos(pi() * x); }

/* Wrapper for ldexp: x * 2^exp with integer exponent */
static double ldexp_wrapper(double x, double exp) { return ldexp(x, (int)exp); }

/* logaddexp: log(exp(a) + exp(b)), numerically stable */
static double logaddexp(double a, double b) {
    if (a == b) {
        return a + log1p(1.0); // log(2*exp(a)) = a + log(2)
    }
    double max_val = (a > b) ? a : b;
    double min_val = (a > b) ? b : a;
    return max_val + log1p(exp(min_val - max_val));
}

/* Forward declarations for complex operations */
/* (Already declared above) */

/* Wrapper functions for complex operations (for function pointer compatibility) */
/* These are placeholders - actual implementation is in vector functions */
static double conj_wrapper(double x) { return x; }

double imag_wrapper(double x) {
    (void)x;
    return 0.0;
}

/* Wrapper for round: round to nearest integer */
static double round_wrapper(double x) { return round(x); }

/* sign: returns -1.0, 0.0, or 1.0 based on sign of x */
static double sign(double x) {
    if (isnan(x)) return NAN;
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

/* square: x * x */
static double square(double x) { return x * x; }

/* Wrapper for trunc: truncate towards zero */
static double trunc_wrapper(double x) { return trunc(x); }

bool is_float_math_function(const void* func) {
    return func == (void*)acos ||
        func == (void*)acosh ||
        func == (void*)asin ||
        func == (void*)asinh ||
        func == (void*)atan ||
        func == (void*)atanh ||
        func == (void*)cbrt ||
        func == (void*)cos ||
        func == (void*)cosh ||
        func == (void*)cospi_wrapper ||
        func == (void*)erf ||
        func == (void*)erfc ||
        func == (void*)exp ||
        func == (void*)exp10_wrapper ||
        func == (void*)exp2 ||
        func == (void*)expm1_wrapper ||
        func == (void*)lgamma ||
        func == (void*)log ||
        func == (void*)log10 ||
        func == (void*)log1p_wrapper ||
        func == (void*)log2_wrapper ||
        func == (void*)sin ||
        func == (void*)sinh ||
        func == (void*)sinpi_wrapper ||
        func == (void*)sqrt ||
        func == (void*)tan ||
        func == (void*)tanh ||
        func == (void*)tgamma;
}

/* Scalar helper for where(), used only in generic slow path */
double where_scalar(double c, double x, double y) {
    return (c != 0.0) ? x : y;
}

double real_wrapper(double x) { return x; }

static double str_startswith(double a, double b);
static double str_endswith(double a, double b);
static double str_contains(double a, double b);

static double fac(double a) {
    /* simplest version of fac */
    if (a < 0.0)
        return NAN;
    if (a > UINT_MAX)
        return INFINITY;
    unsigned int ua = (unsigned int)(a);
    unsigned long int result = 1, i;
    for (i = 1; i <= ua; i++) {
        if (i > ULONG_MAX / result)
            return INFINITY;
        result *= i;
    }
    return (double)result;
}

static double ncr(double n, double r) {
    if (n < 0.0 || r < 0.0 || n < r) return NAN;
    if (n > UINT_MAX || r > UINT_MAX) return INFINITY;
    unsigned long int un = (unsigned int)(n), ur = (unsigned int)(r), i;
    unsigned long int result = 1;
    if (ur > un / 2) ur = un - ur;
    for (i = 1; i <= ur; i++) {
        if (result > ULONG_MAX / (un - ur + i))
            return INFINITY;
        result *= un - ur + i;
        result /= i;
    }
    return result;
}

static double npr(double n, double r) { return ncr(n, r) * fac(r); }

#ifdef _MSC_VER
#pragma function (ceil)
#pragma function (floor)
#endif

static const me_variable_ex functions[] = {
    /* must be in alphabetical order */
    /* Format: {name, dtype, address, type, context} */
    {"abs", 0, fabs, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"acos", 0, acos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"acosh", 0, acosh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"all", 0, all_reduce, ME_FUNCTION1, 0},
    {"any", 0, any_reduce, ME_FUNCTION1, 0},
    {"arccos", 0, acos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arccosh", 0, acosh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arcsin", 0, asin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arcsinh", 0, asinh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arctan", 0, atan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"arctan2", 0, atan2, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"arctanh", 0, atanh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"asin", 0, asin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"asinh", 0, asinh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"atan", 0, atan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"atan2", 0, atan2, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"atanh", 0, atanh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"cbrt", 0, cbrt, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"ceil", 0, ceil, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"conj", 0, conj_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"contains", 0, str_contains, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"copysign", 0, copysign, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"cos", 0, cos, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"cosh", 0, cosh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"cospi", 0, cospi_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"e", 0, e, ME_FUNCTION0 | ME_FLAG_PURE, 0},
    {"endswith", 0, str_endswith, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"erf", 0, erf, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"erfc", 0, erfc, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"exp", 0, exp, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"exp10", 0, exp10_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"exp2", 0, exp2, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"expm1", 0, expm1_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"fac", 0, fac, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"fdim", 0, fdim, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"floor", 0, floor, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"fma", 0, fma, ME_FUNCTION3 | ME_FLAG_PURE, 0},
    {"fmax", 0, fmax, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"fmin", 0, fmin, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"fmod", 0, fmod, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"hypot", 0, hypot, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"imag", 0, imag_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"ldexp", 0, ldexp_wrapper, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"lgamma", 0, lgamma, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"ln", 0, log, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#ifdef ME_NAT_LOG
    {"log", 0, log, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#else
    {"log", 0, log10, ME_FUNCTION1 | ME_FLAG_PURE, 0},
#endif
    {"log10", 0, log10, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"log1p", 0, log1p_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"log2", 0, log2_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"logaddexp", 0, logaddexp, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"max", 0, max_reduce, ME_FUNCTION1, 0},
    {"mean", 0, mean_reduce, ME_FUNCTION1, 0},
    {"min", 0, min_reduce, ME_FUNCTION1, 0},
    {"ncr", 0, ncr, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"nextafter", 0, nextafter, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"npr", 0, npr, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"pi", 0, pi, ME_FUNCTION0 | ME_FLAG_PURE, 0},
    {"pow", 0, pow, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"prod", 0, prod_reduce, ME_FUNCTION1, 0},
    {"real", 0, real_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"remainder", 0, remainder, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"rint", 0, rint, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"round", 0, round_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sign", 0, sign, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sin", 0, sin, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sinh", 0, sinh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sinpi", 0, sinpi_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"sqrt", 0, sqrt, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"square", 0, square, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"startswith", 0, str_startswith, ME_FUNCTION2 | ME_FLAG_PURE, 0},
    {"sum", 0, sum_reduce, ME_FUNCTION1, 0},
    {"tan", 0, tan, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"tanh", 0, tanh, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"tgamma", 0, tgamma, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"trunc", 0, trunc_wrapper, ME_FUNCTION1 | ME_FLAG_PURE, 0},
    {"where", 0, where_scalar, ME_FUNCTION3 | ME_FLAG_PURE, 0},
    {0, 0, 0, 0, 0}
};

static const me_variable_ex* find_builtin(const char* name, int len) {
    int imin = 0;
    int imax = sizeof(functions) / sizeof(me_variable_ex) - 2;

    /*Binary search.*/
    while (imax >= imin) {
        const int i = (imin + ((imax - imin) / 2));
        int c = strncmp(name, functions[i].name, len);
        if (!c) c = '\0' - functions[i].name[len];
        if (c == 0) {
            return functions + i;
        }
        else if (c > 0) {
            imin = i + 1;
        }
        else {
            imax = i - 1;
        }
    }

    return 0;
}

bool me_is_builtin_function_name(const char* name, size_t len) {
    if (!name || len == 0) {
        return false;
    }
    return find_builtin(name, (int)len) != NULL;
}

static const me_variable_ex* find_lookup(const state* s, const char* name, int len) {
    int iters;
    const me_variable_ex* var;
    if (!s->lookup) return 0;

    for (var = s->lookup, iters = s->lookup_len; iters; ++var, --iters) {
        if (strncmp(name, var->name, len) == 0 && var->name[len] == '\0') {
            return var;
        }
    }
    return 0;
}


static double add(double a, double b) { return a + b; }
static double sub(double a, double b) { return a - b; }
static double mul(double a, double b) { return a * b; }
static double divide(double a, double b) { return a / b; }
static double negate(double a) { return -a; }
static volatile double sum_salt = 0.0;
static volatile double mean_salt = 0.0;
static volatile double prod_salt = 1.0;
static volatile double min_salt = 0.0;
static volatile double max_salt = 0.0;
static volatile double any_salt = 0.0;
static volatile double all_salt = 0.0;
static double sum_reduce(double x) { return x + sum_salt; }
static double mean_reduce(double x) { return x + mean_salt; }
static double prod_reduce(double x) { return x * prod_salt; }
static double any_reduce(double x) { return x + any_salt; }
static double all_reduce(double x) { return x * (1.0 + all_salt); }
double min_reduce(double x) { return x + min_salt; }
double max_reduce(double x) { return x - max_salt; }

static float reduce_min_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256 vmin = _mm256_set1_ps(INFINITY);
    __m256 vnan = _mm256_setzero_ps();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        vmin = _mm256_min_ps(vmin, v);
    }
    __m128 low = _mm256_castps256_ps128(vmin);
    __m128 high = _mm256_extractf128_ps(vmin, 1);
    __m128 min128 = _mm_min_ps(low, high);
    __m128 tmp = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    tmp = _mm_min_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm256_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif defined(__SSE__)
    int i = 0;
    __m128 vmin = _mm_set1_ps(INFINITY);
    __m128 vnan = _mm_setzero_ps();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        vnan = _mm_or_ps(vnan, _mm_cmpunord_ps(v, v));
        vmin = _mm_min_ps(vmin, v);
    }
    __m128 tmp = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
    tmp = _mm_min_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    float32x4_t vmin = vdupq_n_f32(INFINITY);
    uint32x4_t vnan = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t eq = vceqq_f32(v, v);
        vnan = vorrq_u32(vnan, vmvnq_u32(eq));
        vmin = vminq_f32(vmin, v);
    }
#if defined(__aarch64__)
    float acc = vminvq_f32(vmin);
#else
    float32x2_t min2 = vmin_f32(vget_low_f32(vmin), vget_high_f32(vmin));
    min2 = vpmin_f32(min2, min2);
    float acc = vget_lane_f32(min2, 0);
#endif
    uint32x2_t nan2 = vorr_u32(vget_low_u32(vnan), vget_high_u32(vnan));
    nan2 = vpadd_u32(nan2, nan2);
    if (vget_lane_u32(nan2, 0)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#else
    float acc = data[0];
    for (int i = 0; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#endif
}

static float reduce_max_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return -INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    __m256 vnan = _mm256_setzero_ps();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        vmax = _mm256_max_ps(vmax, v);
    }
    __m128 low = _mm256_castps256_ps128(vmax);
    __m128 high = _mm256_extractf128_ps(vmax, 1);
    __m128 max128 = _mm_max_ps(low, high);
    __m128 tmp = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    tmp = _mm_max_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm256_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif defined(__SSE__)
    int i = 0;
    __m128 vmax = _mm_set1_ps(-INFINITY);
    __m128 vnan = _mm_setzero_ps();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        vnan = _mm_or_ps(vnan, _mm_cmpunord_ps(v, v));
        vmax = _mm_max_ps(vmax, v);
    }
    __m128 tmp = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
    tmp = _mm_max_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1));
    float acc = _mm_cvtss_f32(tmp);
    if (_mm_movemask_ps(vnan)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    float32x4_t vmax = vdupq_n_f32(-INFINITY);
    uint32x4_t vnan = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t eq = vceqq_f32(v, v);
        vnan = vorrq_u32(vnan, vmvnq_u32(eq));
        vmax = vmaxq_f32(vmax, v);
    }
#if defined(__aarch64__)
    float acc = vmaxvq_f32(vmax);
#else
    float32x2_t max2 = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    max2 = vpmax_f32(max2, max2);
    float acc = vget_lane_f32(max2, 0);
#endif
    uint32x2_t nan2 = vorr_u32(vget_low_u32(vnan), vget_high_u32(vnan));
    nan2 = vpadd_u32(nan2, nan2);
    if (vget_lane_u32(nan2, 0)) return NAN;
    for (; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#else
    float acc = data[0];
    for (int i = 0; i < nitems; i++) {
        float v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#endif
}

static double reduce_min_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vmin = _mm256_set1_pd(INFINITY);
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vmin = _mm256_min_pd(vmin, v);
    }
    __m128d low = _mm256_castpd256_pd128(vmin);
    __m128d high = _mm256_extractf128_pd(vmin, 1);
    __m128d min128 = _mm_min_pd(low, high);
    min128 = _mm_min_sd(min128, _mm_unpackhi_pd(min128, min128));
    double acc = _mm_cvtsd_f64(min128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vmin = _mm_set1_pd(INFINITY);
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vmin = _mm_min_pd(vmin, v);
    }
    vmin = _mm_min_sd(vmin, _mm_unpackhi_pd(vmin, vmin));
    double acc = _mm_cvtsd_f64(vmin);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vmin = vdupq_n_f64(INFINITY);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vmin = vminq_f64(vmin, v);
    }
    double acc = vminvq_f64(vmin);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#else
    double acc = data[0];
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v < acc) acc = v;
    }
    return acc;
#endif
}

static double reduce_max_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return -INFINITY;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vmax = _mm256_set1_pd(-INFINITY);
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vmax = _mm256_max_pd(vmax, v);
    }
    __m128d low = _mm256_castpd256_pd128(vmax);
    __m128d high = _mm256_extractf128_pd(vmax, 1);
    __m128d max128 = _mm_max_pd(low, high);
    max128 = _mm_max_sd(max128, _mm_unpackhi_pd(max128, max128));
    double acc = _mm_cvtsd_f64(max128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vmax = _mm_set1_pd(-INFINITY);
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vmax = _mm_max_pd(vmax, v);
    }
    vmax = _mm_max_sd(vmax, _mm_unpackhi_pd(vmax, vmax));
    double acc = _mm_cvtsd_f64(vmax);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vmax = vdupq_n_f64(-INFINITY);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vmax = vmaxq_f64(vmax, v);
    }
    double acc = vmaxvq_f64(vmax);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#else
    double acc = data[0];
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        if (v != v) return v;
        if (v > acc) acc = v;
    }
    return acc;
#endif
}

static int32_t reduce_min_int32(const int32_t* data, int nitems) {
    if (nitems <= 0) return INT32_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi32(INT32_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epi32(vmin, v);
    }
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    int32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__SSE4_1__)
    int i = 0;
    __m128i vmin = _mm_set1_epi32(INT32_MAX);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(data + i));
        vmin = _mm_min_epi32(vmin, v);
    }
    int32_t tmp[4];
    _mm_storeu_si128((__m128i*)tmp, vmin);
    int32_t acc = tmp[0];
    for (int j = 1; j < 4; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int32x4_t vmin = vdupq_n_s32(INT32_MAX);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        vmin = vminq_s32(vmin, v);
    }
#if defined(__aarch64__)
    int32_t acc = vminvq_s32(vmin);
#else
    int32x2_t min2 = vmin_s32(vget_low_s32(vmin), vget_high_s32(vmin));
    min2 = vpmin_s32(min2, min2);
    int32_t acc = vget_lane_s32(min2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    int32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static int32_t reduce_max_int32(const int32_t* data, int nitems) {
    if (nitems <= 0) return INT32_MIN;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_set1_epi32(INT32_MIN);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epi32(vmax, v);
    }
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    int32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__SSE4_1__)
    int i = 0;
    __m128i vmax = _mm_set1_epi32(INT32_MIN);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i*)(data + i));
        vmax = _mm_max_epi32(vmax, v);
    }
    int32_t tmp[4];
    _mm_storeu_si128((__m128i*)tmp, vmax);
    int32_t acc = tmp[0];
    for (int j = 1; j < 4; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int32x4_t vmax = vdupq_n_s32(INT32_MIN);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        vmax = vmaxq_s32(vmax, v);
    }
#if defined(__aarch64__)
    int32_t acc = vmaxvq_s32(vmax);
#else
    int32x2_t max2 = vmax_s32(vget_low_s32(vmax), vget_high_s32(vmax));
    max2 = vpmax_s32(max2, max2);
    int32_t acc = vget_lane_s32(max2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    int32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static int8_t reduce_min_int8(const int8_t* data, int nitems) {
    if (nitems <= 0) return INT8_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi8(INT8_MAX);
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epi8(vmin, v);
    }
    int8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    int8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int8x16_t vmin = vdupq_n_s8(INT8_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        int8x16_t v = vld1q_s8(data + i);
        vmin = vminq_s8(vmin, v);
    }
#if defined(__aarch64__)
    int8_t acc = vminvq_s8(vmin);
#else
    int8x8_t min8 = vmin_s8(vget_low_s8(vmin), vget_high_s8(vmin));
    min8 = vpmin_s8(min8, min8);
    min8 = vpmin_s8(min8, min8);
    int8_t acc = vget_lane_s8(min8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    int8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static int8_t reduce_max_int8(const int8_t* data, int nitems) {
    if (nitems <= 0) return INT8_MIN;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_set1_epi8(INT8_MIN);
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epi8(vmax, v);
    }
    int8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    int8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int8x16_t vmax = vdupq_n_s8(INT8_MIN);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        int8x16_t v = vld1q_s8(data + i);
        vmax = vmaxq_s8(vmax, v);
    }
#if defined(__aarch64__)
    int8_t acc = vmaxvq_s8(vmax);
#else
    int8x8_t max8 = vmax_s8(vget_low_s8(vmax), vget_high_s8(vmax));
    max8 = vpmax_s8(max8, max8);
    max8 = vpmax_s8(max8, max8);
    int8_t acc = vget_lane_s8(max8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    int8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static int16_t reduce_min_int16(const int16_t* data, int nitems) {
    if (nitems <= 0) return INT16_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi16(INT16_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epi16(vmin, v);
    }
    int16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    int16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int16x8_t vmin = vdupq_n_s16(INT16_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        int16x8_t v = vld1q_s16(data + i);
        vmin = vminq_s16(vmin, v);
    }
#if defined(__aarch64__)
    int16_t acc = vminvq_s16(vmin);
#else
    int16x4_t min4 = vmin_s16(vget_low_s16(vmin), vget_high_s16(vmin));
    min4 = vpmin_s16(min4, min4);
    min4 = vpmin_s16(min4, min4);
    int16_t acc = vget_lane_s16(min4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    int16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static int16_t reduce_max_int16(const int16_t* data, int nitems) {
    if (nitems <= 0) return INT16_MIN;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_set1_epi16(INT16_MIN);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epi16(vmax, v);
    }
    int16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    int16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    int16x8_t vmax = vdupq_n_s16(INT16_MIN);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        int16x8_t v = vld1q_s16(data + i);
        vmax = vmaxq_s16(vmax, v);
    }
#if defined(__aarch64__)
    int16_t acc = vmaxvq_s16(vmax);
#else
    int16x4_t max4 = vmax_s16(vget_low_s16(vmax), vget_high_s16(vmax));
    max4 = vpmax_s16(max4, max4);
    max4 = vpmax_s16(max4, max4);
    int16_t acc = vget_lane_s16(max4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    int16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static int64_t reduce_min_int64(const int64_t* data, int nitems) {
    if (nitems <= 0) return INT64_MAX;
    int64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
}

static int64_t reduce_max_int64(const int64_t* data, int nitems) {
    if (nitems <= 0) return INT64_MIN;
    int64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
}

static uint8_t reduce_min_uint8(const uint8_t* data, int nitems) {
    if (nitems <= 0) return UINT8_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi8((char)UINT8_MAX);
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epu8(vmin, v);
    }
    uint8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    uint8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint8x16_t vmin = vdupq_n_u8(UINT8_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        uint8x16_t v = vld1q_u8(data + i);
        vmin = vminq_u8(vmin, v);
    }
#if defined(__aarch64__)
    uint8_t acc = vminvq_u8(vmin);
#else
    uint8x8_t min8 = vmin_u8(vget_low_u8(vmin), vget_high_u8(vmin));
    min8 = vpmin_u8(min8, min8);
    min8 = vpmin_u8(min8, min8);
    uint8_t acc = vget_lane_u8(min8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    uint8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static uint8_t reduce_max_uint8(const uint8_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_setzero_si256();
    const int limit = nitems & ~31;
    for (; i < limit; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epu8(vmax, v);
    }
    uint8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    uint8_t acc = tmp[0];
    for (int j = 1; j < 32; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint8x16_t vmax = vdupq_n_u8(0);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        uint8x16_t v = vld1q_u8(data + i);
        vmax = vmaxq_u8(vmax, v);
    }
#if defined(__aarch64__)
    uint8_t acc = vmaxvq_u8(vmax);
#else
    uint8x8_t max8 = vmax_u8(vget_low_u8(vmax), vget_high_u8(vmax));
    max8 = vpmax_u8(max8, max8);
    max8 = vpmax_u8(max8, max8);
    uint8_t acc = vget_lane_u8(max8, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    uint8_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static uint16_t reduce_min_uint16(const uint16_t* data, int nitems) {
    if (nitems <= 0) return UINT16_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi16((short)UINT16_MAX);
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epu16(vmin, v);
    }
    uint16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    uint16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint16x8_t vmin = vdupq_n_u16(UINT16_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        uint16x8_t v = vld1q_u16(data + i);
        vmin = vminq_u16(vmin, v);
    }
#if defined(__aarch64__)
    uint16_t acc = vminvq_u16(vmin);
#else
    uint16x4_t min4 = vmin_u16(vget_low_u16(vmin), vget_high_u16(vmin));
    min4 = vpmin_u16(min4, min4);
    min4 = vpmin_u16(min4, min4);
    uint16_t acc = vget_lane_u16(min4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    uint16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static uint16_t reduce_max_uint16(const uint16_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_setzero_si256();
    const int limit = nitems & ~15;
    for (; i < limit; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epu16(vmax, v);
    }
    uint16_t tmp[16];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    uint16_t acc = tmp[0];
    for (int j = 1; j < 16; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint16x8_t vmax = vdupq_n_u16(0);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        uint16x8_t v = vld1q_u16(data + i);
        vmax = vmaxq_u16(vmax, v);
    }
#if defined(__aarch64__)
    uint16_t acc = vmaxvq_u16(vmax);
#else
    uint16x4_t max4 = vmax_u16(vget_low_u16(vmax), vget_high_u16(vmax));
    max4 = vpmax_u16(max4, max4);
    max4 = vpmax_u16(max4, max4);
    uint16_t acc = vget_lane_u16(max4, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    uint16_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static uint32_t reduce_min_uint32(const uint32_t* data, int nitems) {
    if (nitems <= 0) return UINT32_MAX;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmin = _mm256_set1_epi32((int)UINT32_MAX);
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmin = _mm256_min_epu32(vmin, v);
    }
    uint32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmin);
    uint32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] < acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint32x4_t vmin = vdupq_n_u32(UINT32_MAX);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        vmin = vminq_u32(vmin, v);
    }
#if defined(__aarch64__)
    uint32_t acc = vminvq_u32(vmin);
#else
    uint32x2_t min2 = vmin_u32(vget_low_u32(vmin), vget_high_u32(vmin));
    min2 = vpmin_u32(min2, min2);
    uint32_t acc = vget_lane_u32(min2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#else
    uint32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
#endif
}

static uint32_t reduce_max_uint32(const uint32_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i vmax = _mm256_setzero_si256();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        vmax = _mm256_max_epu32(vmax, v);
    }
    uint32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, vmax);
    uint32_t acc = tmp[0];
    for (int j = 1; j < 8; j++) {
        if (tmp[j] > acc) acc = tmp[j];
    }
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int i = 0;
    uint32x4_t vmax = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        vmax = vmaxq_u32(vmax, v);
    }
#if defined(__aarch64__)
    uint32_t acc = vmaxvq_u32(vmax);
#else
    uint32x2_t max2 = vmax_u32(vget_low_u32(vmax), vget_high_u32(vmax));
    max2 = vpmax_u32(max2, max2);
    uint32_t acc = vget_lane_u32(max2, 0);
#endif
    for (; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#else
    uint32_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
#endif
}

static uint64_t reduce_min_uint64(const uint64_t* data, int nitems) {
    if (nitems <= 0) return UINT64_MAX;
    uint64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] < acc) acc = data[i];
    }
    return acc;
}

static uint64_t reduce_max_uint64(const uint64_t* data, int nitems) {
    if (nitems <= 0) return 0;
    uint64_t acc = data[0];
    for (int i = 1; i < nitems; i++) {
        if (data[i] > acc) acc = data[i];
    }
    return acc;
}

static double reduce_prod_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return 1.0;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vprod0 = _mm256_set1_pd(1.0);
    __m256d vprod1 = _mm256_set1_pd(1.0);
    int nan_mask = 0;
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        nan_mask |= _mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m256d vlo = _mm256_cvtps_pd(vlow);
        __m256d vhi = _mm256_cvtps_pd(vhigh);
        vprod0 = _mm256_mul_pd(vprod0, vlo);
        vprod1 = _mm256_mul_pd(vprod1, vhi);
    }
    __m256d vprod = _mm256_mul_pd(vprod0, vprod1);
    __m128d low = _mm256_castpd256_pd128(vprod);
    __m128d high = _mm256_extractf128_pd(vprod, 1);
    __m128d prod128 = _mm_mul_pd(low, high);
    prod128 = _mm_mul_sd(prod128, _mm_unpackhi_pd(prod128, prod128));
    double acc = _mm_cvtsd_f64(prod128);
    if (nan_mask) return NAN;
    for (; i < nitems; i++) {
        double v = (double)data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vprod0 = _mm_set1_pd(1.0);
    __m128d vprod1 = _mm_set1_pd(1.0);
    int nan_mask = 0;
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        nan_mask |= _mm_movemask_ps(_mm_cmpunord_ps(v, v));
        __m128 vhigh = _mm_movehl_ps(v, v);
        __m128d vlo = _mm_cvtps_pd(v);
        __m128d vhi = _mm_cvtps_pd(vhigh);
        vprod0 = _mm_mul_pd(vprod0, vlo);
        vprod1 = _mm_mul_pd(vprod1, vhi);
    }
    __m128d prod128 = _mm_mul_pd(vprod0, vprod1);
    prod128 = _mm_mul_sd(prod128, _mm_unpackhi_pd(prod128, prod128));
    double acc = _mm_cvtsd_f64(prod128);
    if (nan_mask) return NAN;
    for (; i < nitems; i++) {
        double v = (double)data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vprod0 = vdupq_n_f64(1.0);
    float64x2_t vprod1 = vdupq_n_f64(1.0);
    uint32x4_t vnan = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t eq = vceqq_f32(v, v);
        vnan = vorrq_u32(vnan, veorq_u32(eq, vdupq_n_u32(~0U)));
        float64x2_t vlo = vcvt_f64_f32(vget_low_f32(v));
        float64x2_t vhi = vcvt_f64_f32(vget_high_f32(v));
        vprod0 = vmulq_f64(vprod0, vlo);
        vprod1 = vmulq_f64(vprod1, vhi);
    }
    float64x2_t vprod = vmulq_f64(vprod0, vprod1);
    double acc = vgetq_lane_f64(vprod, 0) * vgetq_lane_f64(vprod, 1);
    uint32x4_t nan_or = vorrq_u32(vnan, vextq_u32(vnan, vnan, 2));
    nan_or = vorrq_u32(nan_or, vextq_u32(nan_or, nan_or, 1));
    if (vgetq_lane_u32(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = (double)data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#else
    double acc = 1.0;
    for (int i = 0; i < nitems; i++) {
        double v = (double)data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#endif
}

static double reduce_prod_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return 1.0;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vprod = _mm256_set1_pd(1.0);
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vprod = _mm256_mul_pd(vprod, v);
    }
    __m128d low = _mm256_castpd256_pd128(vprod);
    __m128d high = _mm256_extractf128_pd(vprod, 1);
    __m128d prod128 = _mm_mul_pd(low, high);
    prod128 = _mm_mul_sd(prod128, _mm_unpackhi_pd(prod128, prod128));
    double acc = _mm_cvtsd_f64(prod128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vprod = _mm_set1_pd(1.0);
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vprod = _mm_mul_pd(vprod, v);
    }
    vprod = _mm_mul_sd(vprod, _mm_unpackhi_pd(vprod, vprod));
    double acc = _mm_cvtsd_f64(vprod);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vprod = vdupq_n_f64(1.0);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vprod = vmulq_f64(vprod, v);
    }
    double acc = vgetq_lane_f64(vprod, 0) * vgetq_lane_f64(vprod, 1);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#else
    double acc = 1.0;
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        acc *= v;
        if (v != v) return v;
    }
    return acc;
#endif
}

static double reduce_sum_float32_nan_safe(const float* data, int nitems) {
    if (nitems <= 0) return 0.0;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vsum0 = _mm256_setzero_pd();
    __m256d vsum1 = _mm256_setzero_pd();
    int nan_mask = 0;
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        nan_mask |= _mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m256d vlo = _mm256_cvtps_pd(vlow);
        __m256d vhi = _mm256_cvtps_pd(vhigh);
        vsum0 = _mm256_add_pd(vsum0, vlo);
        vsum1 = _mm256_add_pd(vsum1, vhi);
    }
    __m256d vsum = _mm256_add_pd(vsum0, vsum1);
    __m128d low = _mm256_castpd256_pd128(vsum);
    __m128d high = _mm256_extractf128_pd(vsum, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128));
    double acc = _mm_cvtsd_f64(sum128);
    if (nan_mask) return NAN;
    for (; i < nitems; i++) {
        double v = (double)data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vsum0 = _mm_setzero_pd();
    __m128d vsum1 = _mm_setzero_pd();
    int nan_mask = 0;
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m128 v = _mm_loadu_ps(data + i);
        nan_mask |= _mm_movemask_ps(_mm_cmpunord_ps(v, v));
        __m128 vhigh = _mm_movehl_ps(v, v);
        __m128d vlo = _mm_cvtps_pd(v);
        __m128d vhi = _mm_cvtps_pd(vhigh);
        vsum0 = _mm_add_pd(vsum0, vlo);
        vsum1 = _mm_add_pd(vsum1, vhi);
    }
    __m128d sum128 = _mm_add_pd(vsum0, vsum1);
    sum128 = _mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128));
    double acc = _mm_cvtsd_f64(sum128);
    if (nan_mask) return NAN;
    for (; i < nitems; i++) {
        double v = (double)data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vsum0 = vdupq_n_f64(0.0);
    float64x2_t vsum1 = vdupq_n_f64(0.0);
    uint32x4_t vnan = vdupq_n_u32(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t eq = vceqq_f32(v, v);
        vnan = vorrq_u32(vnan, veorq_u32(eq, vdupq_n_u32(~0U)));
        float64x2_t vlo = vcvt_f64_f32(vget_low_f32(v));
        float64x2_t vhi = vcvt_f64_f32(vget_high_f32(v));
        vsum0 = vaddq_f64(vsum0, vlo);
        vsum1 = vaddq_f64(vsum1, vhi);
    }
    double acc = vaddvq_f64(vaddq_f64(vsum0, vsum1));
    uint32x4_t nan_or = vorrq_u32(vnan, vextq_u32(vnan, vnan, 2));
    nan_or = vorrq_u32(nan_or, vextq_u32(nan_or, nan_or, 1));
    if (vgetq_lane_u32(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = (double)data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#else
    double acc = 0.0;
    for (int i = 0; i < nitems; i++) {
        double v = (double)data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#endif
}


static double reduce_sum_float64_nan_safe(const double* data, int nitems) {
    if (nitems <= 0) return 0.0;
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    __m256d vsum = _mm256_setzero_pd();
    __m256d vnan = _mm256_setzero_pd();
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        vnan = _mm256_or_pd(vnan, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
        vsum = _mm256_add_pd(vsum, v);
    }
    __m128d low = _mm256_castpd256_pd128(vsum);
    __m128d high = _mm256_extractf128_pd(vsum, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128));
    double acc = _mm_cvtsd_f64(sum128);
    if (_mm256_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#elif defined(__SSE2__)
    int i = 0;
    __m128d vsum = _mm_setzero_pd();
    __m128d vnan = _mm_setzero_pd();
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        vnan = _mm_or_pd(vnan, _mm_cmpunord_pd(v, v));
        vsum = _mm_add_pd(vsum, v);
    }
    vsum = _mm_add_sd(vsum, _mm_unpackhi_pd(vsum, vsum));
    double acc = _mm_cvtsd_f64(vsum);
    if (_mm_movemask_pd(vnan)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    float64x2_t vsum = vdupq_n_f64(0.0);
    uint64x2_t vnan = vdupq_n_u64(0);
    const int limit = nitems & ~1;
    for (; i < limit; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        uint64x2_t eq = vceqq_f64(v, v);
        vnan = vorrq_u64(vnan, veorq_u64(eq, vdupq_n_u64(~0ULL)));
        vsum = vaddq_f64(vsum, v);
    }
    double acc = vaddvq_f64(vsum);
    uint64x2_t nan_or = vorrq_u64(vnan, vextq_u64(vnan, vnan, 1));
    if (vgetq_lane_u64(nan_or, 0)) return NAN;
    for (; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#else
    double acc = 0.0;
    for (int i = 0; i < nitems; i++) {
        double v = data[i];
        acc += v;
        if (v != v) return v;
    }
    return acc;
#endif
}

static int64_t reduce_sum_int32(const int32_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        __m128i vlow = _mm256_castsi256_si128(v);
        __m128i vhigh = _mm256_extracti128_si256(v, 1);
        __m256i vlow64 = _mm256_cvtepi32_epi64(vlow);
        __m256i vhigh64 = _mm256_cvtepi32_epi64(vhigh);
        acc0 = _mm256_add_epi64(acc0, vlow64);
        acc1 = _mm256_add_epi64(acc1, vhigh64);
    }
    acc0 = _mm256_add_epi64(acc0, acc1);
    int64_t tmp[4];
    _mm256_storeu_si256((__m256i *)tmp, acc0);
    int64_t acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < nitems; i++) {
        acc += data[i];
    }
    return acc;
#else
    int64_t acc = 0;
    for (int i = 0; i < nitems; i++) {
        acc += data[i];
    }
    return acc;
#endif
}

static uint64_t reduce_sum_uint32(const uint32_t* data, int nitems) {
    if (nitems <= 0) return 0;
#if defined(__AVX2__)
    int i = 0;
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    const int limit = nitems & ~7;
    for (; i < limit; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        __m128i vlow = _mm256_castsi256_si128(v);
        __m128i vhigh = _mm256_extracti128_si256(v, 1);
        __m256i vlow64 = _mm256_cvtepu32_epi64(vlow);
        __m256i vhigh64 = _mm256_cvtepu32_epi64(vhigh);
        acc0 = _mm256_add_epi64(acc0, vlow64);
        acc1 = _mm256_add_epi64(acc1, vhigh64);
    }
    acc0 = _mm256_add_epi64(acc0, acc1);
    uint64_t tmp[4];
    _mm256_storeu_si256((__m256i *)tmp, acc0);
    uint64_t acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < nitems; i++) {
        acc += data[i];
    }
    return acc;
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    int i = 0;
    uint64x2_t acc0 = vdupq_n_u64(0);
    uint64x2_t acc1 = vdupq_n_u64(0);
    const int limit = nitems & ~3;
    for (; i < limit; i += 4) {
        uint32x4_t v = vld1q_u32(data + i);
        uint64x2_t lo = vmovl_u32(vget_low_u32(v));
        uint64x2_t hi = vmovl_u32(vget_high_u32(v));
        acc0 = vaddq_u64(acc0, lo);
        acc1 = vaddq_u64(acc1, hi);
    }
    uint64x2_t accv = vaddq_u64(acc0, acc1);
    uint64_t acc = vgetq_lane_u64(accv, 0) + vgetq_lane_u64(accv, 1);
    for (; i < nitems; i++) {
        acc += data[i];
    }
    return acc;
#else
    uint64_t acc = 0;
    for (int i = 0; i < nitems; i++) {
        acc += data[i];
    }
    return acc;
#endif
}

static double comma(double a, double b) {
    (void)a;
    return b;
}

/* Bitwise operators (for integer types; bool uses logical semantics) */
static double bit_and(double a, double b) { return (double)((int64_t)a & (int64_t)b); }
static double bit_or(double a, double b) { return (double)((int64_t)a | (int64_t)b); }
static double bit_xor(double a, double b) { return (double)((int64_t)a ^ (int64_t)b); }
static double bit_not(double a) { return (double)(~(int64_t)a); }
static double bit_shl(double a, double b) { return (double)((int64_t)a << (int64_t)b); }
static double bit_shr(double a, double b) { return (double)((int64_t)a >> (int64_t)b); }

/* Comparison operators (return 1.0 for true, 0.0 for false) */
static double cmp_eq(double a, double b) { return a == b ? 1.0 : 0.0; }
static double cmp_ne(double a, double b) { return a != b ? 1.0 : 0.0; }
static double cmp_lt(double a, double b) { return a < b ? 1.0 : 0.0; }
static double cmp_le(double a, double b) { return a <= b ? 1.0 : 0.0; }
static double cmp_gt(double a, double b) { return a > b ? 1.0 : 0.0; }
static double cmp_ge(double a, double b) { return a >= b ? 1.0 : 0.0; }

/* Keep these stubs distinct to avoid identical code folding (ICF) on some linkers. */
static double str_startswith(double a, double b) {
    (void)a;
    (void)b;
    return 1.0;
}

static double str_endswith(double a, double b) {
    (void)a;
    (void)b;
    return 2.0;
}

static double str_contains(double a, double b) {
    (void)a;
    (void)b;
    return 3.0;
}

static bool is_comparison_function(const void* func) {
    return func == (void*)cmp_eq || func == (void*)cmp_ne ||
        func == (void*)cmp_lt || func == (void*)cmp_le ||
        func == (void*)cmp_gt || func == (void*)cmp_ge;
}

static bool is_string_function(const void* func) {
    return func == (void*)str_startswith || func == (void*)str_endswith ||
        func == (void*)str_contains;
}

bool is_comparison_node(const me_expr* n) {
    return n && IS_FUNCTION(n->type) && ARITY(n->type) == 2 &&
        is_comparison_function(n->function);
}

static bool is_string_node(const me_expr* n) {
    if (!n) return false;
    if (TYPE_MASK(n->type) == ME_STRING_CONSTANT) return true;
    return TYPE_MASK(n->type) == ME_VARIABLE && n->dtype == ME_STRING;
}

static size_t string_len_u32(const uint32_t* s, size_t max_units) {
    size_t len = 0;
    if (!s) return 0;
    while (len < max_units && s[len] != 0) {
        len++;
    }
    return len;
}

static bool string_view_at(const me_expr* n, int idx, const uint32_t** data, size_t* len) {
    if (!n || !data || !len) return false;
    if (TYPE_MASK(n->type) == ME_STRING_CONSTANT) {
        *data = (const uint32_t*)n->bound;
        *len = n->str_len;
        return *data != NULL;
    }
    if (TYPE_MASK(n->type) == ME_VARIABLE && n->dtype == ME_STRING) {
        if (n->itemsize == 0 || (n->itemsize % sizeof(uint32_t)) != 0) {
            return false;
        }
        const char* base = (const char*)n->bound + (size_t)idx * n->itemsize;
        const uint32_t* s = (const uint32_t*)base;
        size_t max_units = n->itemsize / sizeof(uint32_t);
        *data = s;
        *len = string_len_u32(s, max_units);
        return true;
    }
    return false;
}

static bool string_equals(const uint32_t* a, size_t alen, const uint32_t* b, size_t blen) {
    if (alen != blen) return false;
    if (alen == 0) return true;
    return memcmp(a, b, alen * sizeof(uint32_t)) == 0;
}

static bool string_starts_with(const uint32_t* s, size_t slen, const uint32_t* prefix, size_t plen) {
    if (plen > slen) return false;
    if (plen == 0) return true;
    return memcmp(s, prefix, plen * sizeof(uint32_t)) == 0;
}

static bool string_ends_with(const uint32_t* s, size_t slen, const uint32_t* suffix, size_t plen) {
    if (plen > slen) return false;
    if (plen == 0) return true;
    return memcmp(s + (slen - plen), suffix, plen * sizeof(uint32_t)) == 0;
}

static bool string_contains(const uint32_t* s, size_t slen, const uint32_t* needle, size_t nlen) {
    if (nlen == 0) return true;
    if (nlen > slen) return false;
    for (size_t i = 0; i + nlen <= slen; i++) {
        if (memcmp(s + i, needle, nlen * sizeof(uint32_t)) == 0) {
            return true;
        }
    }
    return false;
}

static bool contains_string_node(const me_expr* n) {
    if (!n) return false;
    if ((n->flags & ME_EXPR_FLAG_HAS_STRING_VALID) != 0) {
        return (n->flags & ME_EXPR_FLAG_HAS_STRING) != 0;
    }

    bool has_string = false;
    if (is_string_node(n)) {
        has_string = true;
    }
    else if (IS_FUNCTION(n->type) || IS_CLOSURE(n->type)) {
        const int arity = ARITY(n->type);
        for (int i = 0; i < arity; i++) {
            if (contains_string_node((const me_expr*)n->parameters[i])) {
                has_string = true;
                break;
            }
        }
    }

    me_expr* mut = (me_expr*)n;
    if (has_string) {
        mut->flags |= ME_EXPR_FLAG_HAS_STRING;
    }
    else {
        mut->flags &= ~ME_EXPR_FLAG_HAS_STRING;
    }
    mut->flags |= ME_EXPR_FLAG_HAS_STRING_VALID;
    return has_string;
}

static bool validate_string_usage_node(const me_expr* n) {
    if (!n) return true;
    if (is_string_node(n)) return true;

    if (IS_FUNCTION(n->type) || IS_CLOSURE(n->type)) {
        const int arity = ARITY(n->type);

        if (is_reduction_node(n)) {
            const me_expr* arg = (const me_expr*)n->parameters[0];
            if (arg && contains_string_node(arg)) {
                return false;
            }
        }

        if (is_string_function(n->function)) {
            if (arity != 2) return false;
            const me_expr* left = (const me_expr*)n->parameters[0];
            const me_expr* right = (const me_expr*)n->parameters[1];
            return is_string_node(left) && is_string_node(right);
        }

        if (is_comparison_node(n)) {
            const me_expr* left = (const me_expr*)n->parameters[0];
            const me_expr* right = (const me_expr*)n->parameters[1];
            const bool left_str = is_string_node(left);
            const bool right_str = is_string_node(right);
            if (left_str || right_str) {
                if (!left_str || !right_str) {
                    return false;
                }
                return n->function == (void*)cmp_eq || n->function == (void*)cmp_ne;
            }
        }

        for (int i = 0; i < arity; i++) {
            const me_expr* child = (const me_expr*)n->parameters[i];
            if (is_string_node(child)) {
                return false;
            }
            if (!validate_string_usage_node(child)) {
                return false;
            }
        }
    }

    return true;
}

bool validate_string_usage(const me_expr* n) {
    if (!validate_string_usage_node(n)) {
        return false;
    }
    if (infer_output_type(n) == ME_STRING) {
        return false;
    }
    return true;
}

/* Logical operators (for bool type) - short-circuit via OR/AND */
static double logical_and(double a, double b) { return ((int)a) && ((int)b) ? 1.0 : 0.0; }
static double logical_or(double a, double b) { return ((int)a) || ((int)b) ? 1.0 : 0.0; }
static double logical_not(double a) { return !(int)a ? 1.0 : 0.0; }
static double logical_xor(double a, double b) { return ((int)a) != ((int)b) ? 1.0 : 0.0; }

static bool is_logical_function(const void* func) {
    return func == (void*)logical_and || func == (void*)logical_or ||
        func == (void*)logical_not || func == (void*)logical_xor;
}

typedef void (*convert_func_t)(const void*, void*, int);
static convert_func_t get_convert_func(me_dtype from, me_dtype to);

static void vec_and_bool(const bool* a, const bool* b, bool* out, int n);
static void vec_or_bool(const bool* a, const bool* b, bool* out, int n);
static void vec_xor_bool(const bool* a, const bool* b, bool* out, int n);
static void vec_not_bool(const bool* a, bool* out, int n);

static void promote_logical_bool(me_expr* node) {
    if (!node || node->dtype != ME_BOOL) return;

    if (node->function == bit_and) {
        node->function = logical_and;
    }
    else if (node->function == bit_or) {
        node->function = logical_or;
    }
    else if (node->function == bit_xor) {
        node->function = logical_xor;
    }
    else if (node->function == bit_not) {
        node->function = logical_not;
    }
}

static bool eval_operand_to_type(me_expr* expr, me_dtype eval_type, int nitems,
                                 const void** data, void** temp,
                                 bool* is_const, double* const_val) {
    if (!expr || !data || !temp || !is_const || !const_val) return false;

    *data = NULL;
    *temp = NULL;
    *is_const = false;
    *const_val = 0.0;

    if (expr->type == ME_STRING_CONSTANT || expr->dtype == ME_STRING) {
        return false;
    }

    if (expr->type == ME_CONSTANT) {
        *is_const = true;
        *const_val = expr->value;
        return true;
    }

    if (expr->type == ME_VARIABLE) {
        if (expr->dtype == eval_type) {
            *data = expr->bound;
            return true;
        }

        void* buffer = malloc((size_t)nitems * dtype_size(eval_type));
        if (!buffer) return false;

        convert_func_t conv = get_convert_func(expr->dtype, eval_type);
        if (!conv) {
            free(buffer);
            return false;
        }

        conv(expr->bound, buffer, nitems);
        *data = buffer;
        *temp = buffer;
        return true;
    }

    void* buffer = malloc((size_t)nitems * dtype_size(eval_type));
    if (!buffer) return false;

    void* saved_output = expr->output;
    me_dtype saved_dtype = expr->dtype;
    int saved_nitems = expr->nitems;

    expr->output = buffer;
    expr->dtype = eval_type;
    expr->nitems = nitems;
    private_eval(expr);

    expr->output = saved_output;
    expr->dtype = saved_dtype;
    expr->nitems = saved_nitems;

    *data = buffer;
    *temp = buffer;
    return true;
}

static bool compare_to_bool_output(const me_expr* n, me_dtype eval_type,
                                   const void* ldata, const void* rdata,
                                   bool lconst, bool rconst,
                                   double lval, double rval,
                                   bool* out, int nitems) {
    if (!n || !out) return false;

    if (eval_type == ME_COMPLEX64 || eval_type == ME_COMPLEX128) {
        return false;
    }

    const void* func = n->function;

#define CMP_LOOP_OP(TYPE, OP) \
    do { \
        const TYPE* lptr = (const TYPE*)ldata; \
        const TYPE* rptr = (const TYPE*)rdata; \
        TYPE lc = (TYPE)lval; \
        TYPE rc = (TYPE)rval; \
        for (int i = 0; i < nitems; i++) { \
            TYPE a = lconst ? lc : lptr[i]; \
            TYPE b = rconst ? rc : rptr[i]; \
            out[i] = (a OP b); \
        } \
    } while (0)

#define CMP_SWITCH(TYPE) \
    do { \
        if (func == (void*)cmp_eq) { CMP_LOOP_OP(TYPE, ==); } \
        else if (func == (void*)cmp_ne) { CMP_LOOP_OP(TYPE, !=); } \
        else if (func == (void*)cmp_lt) { CMP_LOOP_OP(TYPE, <); } \
        else if (func == (void*)cmp_le) { CMP_LOOP_OP(TYPE, <=); } \
        else if (func == (void*)cmp_gt) { CMP_LOOP_OP(TYPE, >); } \
        else if (func == (void*)cmp_ge) { CMP_LOOP_OP(TYPE, >=); } \
        else { return false; } \
    } while (0)

    switch (eval_type) {
    case ME_BOOL:
        CMP_SWITCH(bool);
        break;
    case ME_INT8:
        CMP_SWITCH(int8_t);
        break;
    case ME_INT16:
        CMP_SWITCH(int16_t);
        break;
    case ME_INT32:
        CMP_SWITCH(int32_t);
        break;
    case ME_INT64:
        CMP_SWITCH(int64_t);
        break;
    case ME_UINT8:
        CMP_SWITCH(uint8_t);
        break;
    case ME_UINT16:
        CMP_SWITCH(uint16_t);
        break;
    case ME_UINT32:
        CMP_SWITCH(uint32_t);
        break;
    case ME_UINT64:
        CMP_SWITCH(uint64_t);
        break;
    case ME_FLOAT32:
        CMP_SWITCH(float);
        break;
    case ME_FLOAT64:
        CMP_SWITCH(double);
        break;
    default:
        return false;
    }

#undef CMP_SWITCH
#undef CMP_LOOP_OP
    return true;
}

static bool eval_string_predicate(const me_expr* n, bool* out, int nitems) {
    if (!n || !out) return false;
    if (!IS_FUNCTION(n->type) || ARITY(n->type) != 2) return false;

    const me_expr* left = (const me_expr*)n->parameters[0];
    const me_expr* right = (const me_expr*)n->parameters[1];
    if (!is_string_node(left) || !is_string_node(right)) {
        return false;
    }

    bool is_cmp = is_comparison_node(n);
    bool is_func = is_string_function(n->function);
    if (!is_cmp && !is_func) {
        return false;
    }

    const uint32_t* lconst = NULL;
    const uint32_t* rconst = NULL;
    size_t lconst_len = 0;
    size_t rconst_len = 0;
    const bool left_const = (TYPE_MASK(left->type) == ME_STRING_CONSTANT);
    const bool right_const = (TYPE_MASK(right->type) == ME_STRING_CONSTANT);

    if (left_const) {
        lconst = (const uint32_t*)left->bound;
        lconst_len = left->str_len;
    }
    if (right_const) {
        rconst = (const uint32_t*)right->bound;
        rconst_len = right->str_len;
    }

    for (int i = 0; i < nitems; i++) {
        const uint32_t* ldata = NULL;
        const uint32_t* rdata = NULL;
        size_t llen = 0;
        size_t rlen = 0;

        if (left_const) {
            ldata = lconst;
            llen = lconst_len;
        }
        else if (!string_view_at(left, i, &ldata, &llen)) {
            return false;
        }

        if (right_const) {
            rdata = rconst;
            rlen = rconst_len;
        }
        else if (!string_view_at(right, i, &rdata, &rlen)) {
            return false;
        }

        bool result = false;
        if (is_cmp) {
            if (n->function == (void*)cmp_eq) {
                result = string_equals(ldata, llen, rdata, rlen);
            }
            else if (n->function == (void*)cmp_ne) {
                result = !string_equals(ldata, llen, rdata, rlen);
            }
            else {
                return false;
            }
        }
        else {
            if (n->function == (void*)str_startswith) {
                result = string_starts_with(ldata, llen, rdata, rlen);
            }
            else if (n->function == (void*)str_endswith) {
                result = string_ends_with(ldata, llen, rdata, rlen);
            }
            else if (n->function == (void*)str_contains) {
                result = string_contains(ldata, llen, rdata, rlen);
            }
            else {
                return false;
            }
        }

        out[i] = result;
    }

    return true;
}

static bool eval_bool_expr(me_expr* n) {
    if (!n || !n->output) return false;

    if (n->type == ME_CONSTANT) {
        bool val = (bool)(n->value != 0.0);
        bool* out = (bool*)n->output;
        for (int i = 0; i < n->nitems; i++) {
            out[i] = val;
        }
        return true;
    }

    if (n->type == ME_VARIABLE) {
        bool* out = (bool*)n->output;
        if (n->dtype == ME_STRING) {
            return false;
        }
        if (n->dtype == ME_BOOL) {
            const bool* src = (const bool*)n->bound;
            for (int i = 0; i < n->nitems; i++) {
                out[i] = src[i];
            }
            return true;
        }

        convert_func_t conv = get_convert_func(n->dtype, ME_BOOL);
        if (!conv) return false;
        conv(n->bound, out, n->nitems);
        return true;
    }

    if (IS_FUNCTION(n->type) && is_comparison_node(n)) {
        if (eval_string_predicate(n, (bool*)n->output, n->nitems)) {
            return true;
        }
        me_expr* left = (me_expr*)n->parameters[0];
        me_expr* right = (me_expr*)n->parameters[1];
        if (!left || !right) return false;

        me_dtype eval_type = infer_result_type(n);
        const void* ldata = NULL;
        const void* rdata = NULL;
        void* ltemp = NULL;
        void* rtemp = NULL;
        bool lconst = false;
        bool rconst = false;
        double lval = 0.0;
        double rval = 0.0;

        if (!eval_operand_to_type(left, eval_type, n->nitems, &ldata, &ltemp, &lconst, &lval) ||
            !eval_operand_to_type(right, eval_type, n->nitems, &rdata, &rtemp, &rconst, &rval)) {
            free(ltemp);
            free(rtemp);
            return false;
        }

        bool ok = compare_to_bool_output(n, eval_type, ldata, rdata,
                                         lconst, rconst, lval, rval,
                                         (bool*)n->output, n->nitems);

        free(ltemp);
        free(rtemp);
        return ok;
    }

    if (IS_FUNCTION(n->type) && is_string_function(n->function)) {
        return eval_string_predicate(n, (bool*)n->output, n->nitems);
    }

    if (IS_FUNCTION(n->type) && is_logical_function(n->function)) {
        const int arity = ARITY(n->type);
        if (arity == 1 && n->function == (void*)logical_not) {
            me_expr* arg = (me_expr*)n->parameters[0];
            if (!arg) return false;
            if (!arg->output) {
                arg->output = malloc((size_t)n->nitems * sizeof(bool));
                if (!arg->output) return false;
            }
            arg->nitems = n->nitems;
            if (!eval_bool_expr(arg)) return false;
            vec_not_bool((const bool*)arg->output, (bool*)n->output, n->nitems);
            return true;
        }

        if (arity == 2) {
            me_expr* left = (me_expr*)n->parameters[0];
            me_expr* right = (me_expr*)n->parameters[1];
            if (!left || !right) return false;

            if (!left->output) {
                left->output = malloc((size_t)n->nitems * sizeof(bool));
                if (!left->output) return false;
            }
            if (!right->output) {
                right->output = malloc((size_t)n->nitems * sizeof(bool));
                if (!right->output) return false;
            }
            left->nitems = n->nitems;
            right->nitems = n->nitems;

            if (!eval_bool_expr(left) || !eval_bool_expr(right)) return false;

            if (n->function == (void*)logical_and) {
                vec_and_bool((const bool*)left->output, (const bool*)right->output,
                             (bool*)n->output, n->nitems);
            }
            else if (n->function == (void*)logical_or) {
                vec_or_bool((const bool*)left->output, (const bool*)right->output,
                            (bool*)n->output, n->nitems);
            }
            else if (n->function == (void*)logical_xor) {
                vec_xor_bool((const bool*)left->output, (const bool*)right->output,
                             (bool*)n->output, n->nitems);
            }
            else {
                return false;
            }
            return true;
        }
    }

    if (IS_FUNCTION(n->type) && is_string_function(n->function)) {
        return eval_string_predicate(n, (bool*)n->output, n->nitems);
    }

    return false;
}

static bool is_identifier_start(char c) {
    return isalpha((unsigned char)c) || c == '_';
}

static bool is_identifier_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static void skip_whitespace(state* s) {
    while (*s->next && isspace((unsigned char)*s->next)) {
        s->next++;
    }
}

static void read_number_token(state* s) {
    const char* start = s->next;
    s->value = strtod(s->next, (char**)&s->next);
    s->type = TOK_NUMBER;

    // Determine if it is a floating point or integer constant
    bool is_float = false;
    for (const char* p = start; p < s->next; p++) {
        if (*p == '.' || *p == 'e' || *p == 'E') {
            is_float = true;
            break;
        }
    }

    if (is_float) {
        // Match NumPy conventions: float constants match target_dtype when it's a float type
        // This ensures FLOAT32 arrays + float constants -> FLOAT32 (NumPy behavior)
        if (s->target_dtype == ME_FLOAT32) {
            s->dtype = ME_FLOAT32;
        }
        else {
            s->dtype = ME_FLOAT64;
        }
    }
    else {
        // For integers, we use a heuristic
        if (s->value > INT_MAX || s->value < INT_MIN) {
            s->dtype = ME_INT64;
        }
        else {
            // Use target_dtype if it's an integer type, otherwise default to INT32
            if (is_integer_dtype(s->target_dtype)) {
                s->dtype = s->target_dtype;
            }
            else {
                s->dtype = ME_INT32;
            }
        }
    }
}

static bool read_hex_codepoint(const char** p, int digits, uint32_t* out) {
    uint32_t value = 0;
    for (int i = 0; i < digits; i++) {
        char c = (*p)[i];
        uint32_t v;
        if (c >= '0' && c <= '9') v = (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v = (uint32_t)(10 + c - 'a');
        else if (c >= 'A' && c <= 'F') v = (uint32_t)(10 + c - 'A');
        else return false;
        value = (value << 4) | v;
    }
    *p += digits;
    if (value > 0x10FFFFu || (value >= 0xD800u && value <= 0xDFFFu)) {
        return false;
    }
    *out = value;
    return true;
}

static bool read_utf8_codepoint(const char** p, uint32_t* out) {
    const unsigned char* s = (const unsigned char*)*p;
    if (s[0] < 0x80) {
        *out = s[0];
        (*p)++;
        return true;
    }
    if ((s[0] & 0xE0) == 0xC0) {
        if ((s[1] & 0xC0) != 0x80) return false;
        uint32_t cp = ((uint32_t)(s[0] & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
        if (cp < 0x80) return false;
        *out = cp;
        *p += 2;
        return true;
    }
    if ((s[0] & 0xF0) == 0xE0) {
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return false;
        uint32_t cp = ((uint32_t)(s[0] & 0x0F) << 12) |
                      ((uint32_t)(s[1] & 0x3F) << 6) |
                      (uint32_t)(s[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800u && cp <= 0xDFFFu)) return false;
        *out = cp;
        *p += 3;
        return true;
    }
    if ((s[0] & 0xF8) == 0xF0) {
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) return false;
        uint32_t cp = ((uint32_t)(s[0] & 0x07) << 18) |
                      ((uint32_t)(s[1] & 0x3F) << 12) |
                      ((uint32_t)(s[2] & 0x3F) << 6) |
                      (uint32_t)(s[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFFu) return false;
        *out = cp;
        *p += 4;
        return true;
    }
    return false;
}

static void read_string_token(state* s) {
    const char quote = *s->next;
    const char* p = s->next + 1;
    size_t cap = 16;
    size_t len = 0;
    uint32_t* buf = malloc(cap * sizeof(uint32_t));
    if (!buf) {
        s->type = TOK_ERROR;
        return;
    }

    bool closed = false;
    while (*p) {
        if (*p == quote) {
            p++;
            closed = true;
            break;
        }

        uint32_t cp = 0;
        if (*p == '\\') {
            p++;
            if (!*p) {
                free(buf);
                s->type = TOK_ERROR;
                return;
            }
            char esc = *p++;
            switch (esc) {
            case '\\': cp = '\\'; break;
            case '"': cp = '"'; break;
            case '\'': cp = '\''; break;
            case 'n': cp = '\n'; break;
            case 't': cp = '\t'; break;
            case 'u':
                if (!read_hex_codepoint(&p, 4, &cp)) {
                    free(buf);
                    s->type = TOK_ERROR;
                    return;
                }
                break;
            case 'U':
                if (!read_hex_codepoint(&p, 8, &cp)) {
                    free(buf);
                    s->type = TOK_ERROR;
                    return;
                }
                break;
            default:
                free(buf);
                s->type = TOK_ERROR;
                return;
            }
        }
        else {
            if (!read_utf8_codepoint(&p, &cp)) {
                free(buf);
                s->type = TOK_ERROR;
                return;
            }
        }

        if (len + 1 >= cap) {
            size_t next_cap = cap * 2;
            uint32_t* next_buf = realloc(buf, next_cap * sizeof(uint32_t));
            if (!next_buf) {
                free(buf);
                s->type = TOK_ERROR;
                return;
            }
            buf = next_buf;
            cap = next_cap;
        }
        buf[len++] = cp;
    }

    if (!closed) {
        free(buf);
        s->type = TOK_ERROR;
        return;
    }

    buf[len++] = 0;
    s->str_data = buf;
    s->str_len = len - 1;
    s->type = TOK_STRING;
    s->next = p;
}

static void read_identifier_token(state* s) {
    const char* start = s->next;
    while (is_identifier_char(*s->next)) {
        s->next++;
    }

    size_t len = (size_t)(s->next - start);
    if (len == 3 && strncmp(start, "and", len) == 0) {
        s->type = TOK_LOGICAL_AND;
        s->function = logical_and;
        s->itemsize = 0;
        return;
    }
    if (len == 2 && strncmp(start, "or", len) == 0) {
        s->type = TOK_LOGICAL_OR;
        s->function = logical_or;
        s->itemsize = 0;
        return;
    }
    if (len == 3 && strncmp(start, "not", len) == 0) {
        s->type = TOK_LOGICAL_NOT;
        s->function = logical_not;
        s->itemsize = 0;
        return;
    }

    const me_variable_ex* var = find_lookup(s, start, s->next - start);
    if (!var) {
        var = find_builtin(start, s->next - start);
    }

    if (!var) {
        s->type = TOK_ERROR;
        return;
    }

    switch (TYPE_MASK(var->type)) {
    case ME_VARIABLE:
        s->type = TOK_VARIABLE;
        s->bound = var->address;
        s->dtype = var->dtype;
        s->itemsize = var->itemsize;
        break;

    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        s->context = var->context;
    /* Falls through. */
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
        s->type = var->type;
        s->function = var->address;
        s->dtype = var->dtype;
        s->itemsize = 0;
        break;
    }
}

typedef struct {
    const char* literal;
    int token_type;
    me_fun2 function;
} operator_spec;

static bool handle_multi_char_operator(state* s) {
    static const operator_spec multi_ops[] = {
        {"**", TOK_POW, pow},
        {"&&", TOK_LOGICAL_AND, logical_and},
        {"||", TOK_LOGICAL_OR, logical_or},
        {"<<", TOK_SHIFT, bit_shl},
        {">>", TOK_SHIFT, bit_shr},
        {"==", TOK_COMPARE, cmp_eq},
        {"!=", TOK_COMPARE, cmp_ne},
        {"<=", TOK_COMPARE, cmp_le},
        {">=", TOK_COMPARE, cmp_ge},
    };

    for (size_t i = 0; i < sizeof(multi_ops) / sizeof(multi_ops[0]); i++) {
        const operator_spec* op = &multi_ops[i];
        size_t len = strlen(op->literal);
        if (strncmp(s->next, op->literal, len) == 0) {
            s->type = op->token_type;
            s->function = op->function;
            s->next += len;
            return true;
        }
    }
    return false;
}

static void handle_single_char_operator(state* s, char c) {
    s->next++;
    switch (c) {
    case '+': s->type = TOK_INFIX;
        s->function = add;
        break;
    case '-': s->type = TOK_INFIX;
        s->function = sub;
        break;
    case '*': s->type = TOK_INFIX;
        s->function = mul;
        break;
    case '/': s->type = TOK_INFIX;
        s->function = divide;
        break;
    case '%': s->type = TOK_INFIX;
        s->function = fmod;
        break;
    case '&': s->type = TOK_BITWISE;
        s->function = bit_and;
        break;
    case '|': s->type = TOK_BITWISE;
        s->function = bit_or;
        break;
    case '^': s->type = TOK_BITWISE;
        s->function = bit_xor;
        break;
    case '~': s->type = TOK_BITWISE;
        s->function = bit_not;
        break;
    case '!': s->type = TOK_LOGICAL_NOT;
        s->function = logical_not;
        break;
    case '<': s->type = TOK_COMPARE;
        s->function = cmp_lt;
        break;
    case '>': s->type = TOK_COMPARE;
        s->function = cmp_gt;
        break;
    case '(': s->type = TOK_OPEN;
        break;
    case ')': s->type = TOK_CLOSE;
        break;
    case ',': s->type = TOK_SEP;
        break;
    default: s->type = TOK_ERROR;
        break;
    }
}

static void read_operator_token(state* s) {
    if (handle_multi_char_operator(s)) {
        return;
    }

    if (!*s->next) {
        s->type = TOK_END;
        return;
    }

    handle_single_char_operator(s, *s->next);
}

void next_token(state* s) {
    s->type = TOK_NULL;

    do {
        skip_whitespace(s);

        if (!*s->next) {
            s->type = TOK_END;
            return;
        }

        if (s->next[0] == '"' || s->next[0] == '\'') {
            read_string_token(s);
        }
        else if ((s->next[0] >= '0' && s->next[0] <= '9') || s->next[0] == '.') {
            read_number_token(s);
        }
        else if (is_identifier_start(s->next[0])) {
            read_identifier_token(s);
        }
        else {
            read_operator_token(s);
        }
    }
    while (s->type == TOK_NULL);
}


me_expr* list(state* s);

static me_expr* expr(state* s);

static me_expr* power(state* s);

static me_expr* shift_expr(state* s);

static me_expr* bitwise_and(state* s);

static me_expr* bitwise_xor(state* s);

static me_expr* bitwise_or(state* s);

static me_expr* comparison(state* s);

static me_expr* logical_not_expr(state* s);

static me_expr* logical_and_expr(state* s);

static me_expr* logical_or_expr(state* s);


static me_expr* base(state* s) {
    /* <base>      =    <constant> | <variable> | <function-0> {"(" ")"} | <function-1> <power> | <function-X> "(" <expr> {"," <expr>} ")" | "(" <list> ")" */
    me_expr* ret = NULL;
    int arity;

    switch (s->type) {
    case TOK_NUMBER:
        ret = new_expr(ME_CONSTANT, 0);
        CHECK_NULL(ret);

        ret->value = s->value;
        // Use inferred type for constants (floating point vs integer)
        if (s->target_dtype == ME_AUTO) {
            ret->dtype = s->dtype;
        }
        else {
            // If target_dtype is integer but constant is float/complex, we must use float/complex
            if (is_integer_dtype(s->target_dtype)) {
                if (is_float_dtype(s->dtype) || is_complex_dtype(s->dtype)) {
                    ret->dtype = s->dtype;
                }
                else if (is_integer_dtype(s->dtype) && dtype_size(s->dtype) > dtype_size(s->target_dtype)) {
                    // Use larger integer type if needed
                    ret->dtype = s->dtype;
                }
                else {
                    ret->dtype = s->target_dtype;
                }
            }
            else {
                // For float/complex target types, use target_dtype to match NumPy conventions
                // Float constants are typed based on target_dtype (FLOAT32 or FLOAT64)
                // This ensures FLOAT32 arrays + float constants -> FLOAT32 (NumPy behavior)
                ret->dtype = s->target_dtype;
            }
        }
        next_token(s);
        break;

    case TOK_STRING:
        ret = new_expr(ME_STRING_CONSTANT, 0);
        CHECK_NULL(ret);

        ret->bound = s->str_data;
        ret->dtype = ME_STRING;
        ret->input_dtype = ME_STRING;
        ret->itemsize = (s->str_len + 1) * sizeof(uint32_t);
        ret->str_len = s->str_len;
        ret->flags |= ME_EXPR_FLAG_OWNS_STRING;
        s->str_data = NULL;
        s->str_len = 0;
        next_token(s);
        break;

    case TOK_VARIABLE:
        ret = new_expr(ME_VARIABLE, 0);
        CHECK_NULL(ret);

        ret->bound = s->bound;
        ret->dtype = s->dtype; // Set the variable's type
        ret->input_dtype = s->dtype;
        ret->itemsize = s->itemsize;
        next_token(s);
        break;

    case TOK_OPEN:
        next_token(s);
        ret = list(s);
        CHECK_NULL(ret);

        if (s->type != TOK_CLOSE) {
            s->type = TOK_ERROR;
        }
        else {
            next_token(s);
        }
        break;

    default:
        break;
    }

    if (ret) {
        return ret;
    }

    if (IS_FUNCTION(s->type) || IS_CLOSURE(s->type)) {
        me_dtype func_dtype = s->dtype;
        switch (TYPE_MASK(s->type)) {
        case ME_FUNCTION0:
        case ME_CLOSURE0:
            ret = new_expr(s->type, 0);
            CHECK_NULL(ret);

            ret->function = s->function;
            if (func_dtype != ME_AUTO) {
                ret->dtype = func_dtype;
                ret->flags |= ME_EXPR_FLAG_EXPLICIT_DTYPE;
            }
            if (IS_CLOSURE(s->type)) ret->parameters[0] = s->context;
            next_token(s);
            if (s->type == TOK_OPEN) {
                next_token(s);
                if (s->type != TOK_CLOSE) {
                    s->type = TOK_ERROR;
                }
                else {
                    next_token(s);
                }
            }
            break;

        case ME_FUNCTION1:
        case ME_CLOSURE1:
            ret = new_expr(s->type, 0);
            CHECK_NULL(ret);

            ret->function = s->function;
            if (func_dtype != ME_AUTO) {
                ret->dtype = func_dtype;
                ret->flags |= ME_EXPR_FLAG_EXPLICIT_DTYPE;
            }
            if (IS_CLOSURE(s->type)) ret->parameters[1] = s->context;
            next_token(s);
            ret->parameters[0] = power(s);
            CHECK_NULL(ret->parameters[0], me_free(ret));
            break;

        case ME_FUNCTION2:
        case ME_FUNCTION3:
        case ME_FUNCTION4:
        case ME_FUNCTION5:
        case ME_FUNCTION6:
        case ME_FUNCTION7:
        case ME_CLOSURE2:
        case ME_CLOSURE3:
        case ME_CLOSURE4:
        case ME_CLOSURE5:
        case ME_CLOSURE6:
        case ME_CLOSURE7:
            arity = ARITY(s->type);

            ret = new_expr(s->type, 0);
            CHECK_NULL(ret);

            ret->function = s->function;
            if (func_dtype != ME_AUTO) {
                ret->dtype = func_dtype;
                ret->flags |= ME_EXPR_FLAG_EXPLICIT_DTYPE;
            }
            if (IS_CLOSURE(s->type)) ret->parameters[arity] = s->context;
            next_token(s);

            if (s->type != TOK_OPEN) {
                s->type = TOK_ERROR;
            }
            else {
                int i;
                for (i = 0; i < arity; i++) {
                    next_token(s);
                    /* Allow full comparison expressions inside multi-arg function calls. */
                    ret->parameters[i] = comparison(s);
                    CHECK_NULL(ret->parameters[i], me_free(ret));

                    if (s->type != TOK_SEP) {
                        break;
                    }
                }
                if (s->type != TOK_CLOSE || i != arity - 1) {
                    s->type = TOK_ERROR;
                }
                else {
                    next_token(s);
                }
            }

            if (is_string_function(ret->function)) {
                ret->dtype = ME_BOOL;
            }

            break;

        default:
            ret = new_expr(0, 0);
            CHECK_NULL(ret);

            s->type = TOK_ERROR;
            ret->value = NAN;
            break;
        }
        return ret;
    }

    ret = new_expr(0, 0);
    CHECK_NULL(ret);

    s->type = TOK_ERROR;
    ret->value = NAN;
    return ret;
}


static me_expr* power(state* s) {
    /* <power>     =    {("-" | "+" | "~")} <base> */
    if (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* inner = power(s);
        CHECK_NULL(inner);

        if (t == add) {
            return inner;
        }

        me_expr* ret = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, inner);
        CHECK_NULL(ret, me_free(inner));

        ret->function = negate;
        return ret;
    }

    if (s->type == TOK_BITWISE && s->function == bit_not) {
        next_token(s);
        me_expr* inner = power(s);
        CHECK_NULL(inner);

        me_expr* ret = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, inner);
        CHECK_NULL(ret, me_free(inner));

        ret->function = bit_not;
        ret->dtype = inner->dtype;
        promote_logical_bool(ret);
        return ret;
    }

    return base(s);
}

#ifdef ME_POW_FROM_RIGHT
static me_expr* factor(state* s) {
    /* <factor>    =    <power> {"**" <factor>}  (right associative) */
    me_expr* ret = power(s);
    CHECK_NULL(ret);

    if (s->type == TOK_POW) {
        me_fun2 t = s->function;
        next_token(s);
        me_expr* f = factor(s); /* Right associative: recurse */
        CHECK_NULL(f, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = t;
        apply_type_promotion(ret);
    }

    return ret;
}
#else
static me_expr* factor(state* s) {
    /* <factor>    =    <power> {"**" <power>}  (left associative) */
    me_expr* ret = power(s);
    CHECK_NULL(ret);

    while (s->type == TOK_POW) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* f = power(s);
        CHECK_NULL(f, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
    }

    return ret;
}
#endif


static me_expr* term(state* s) {
    /* <term>      =    <factor> {("*" | "/" | "%") <factor>} */
    me_expr* ret = factor(s);
    CHECK_NULL(ret);

    while (s->type == TOK_INFIX && (s->function == mul || s->function == divide || s->function == fmod)) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* f = factor(s);
        CHECK_NULL(f, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, f);
        CHECK_NULL(ret, me_free(f), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr* expr(state* s) {
    /* <expr>      =    <term> {("+" | "-") <term>} */
    me_expr* ret = term(s);
    CHECK_NULL(ret);

    while (s->type == TOK_INFIX && (s->function == add || s->function == sub)) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* te = term(s);
        CHECK_NULL(te, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, te);
        CHECK_NULL(ret, me_free(te), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret); // Apply type promotion
    }

    return ret;
}


static me_expr* shift_expr(state* s) {
    /* <shift_expr> =    <expr> {("<<" | ">>") <expr>} */
    me_expr* ret = expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_SHIFT) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* e = expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
    }

    return ret;
}


static me_expr* bitwise_and(state* s) {
    /* <bitwise_and> =    <shift_expr> {"&" <shift_expr>} */
    me_expr* ret = shift_expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && s->function == bit_and) {
        next_token(s);
        me_expr* e = shift_expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = bit_and;
        apply_type_promotion(ret);
        promote_logical_bool(ret);
    }

    return ret;
}


static me_expr* bitwise_xor(state* s) {
    /* <bitwise_xor> =    <bitwise_and> {"^" <bitwise_and>} */
    /* Note: ^ is XOR for integers and logical XOR for bools. Use ** for power */
    me_expr* ret = bitwise_and(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && s->function == bit_xor) {
        next_token(s);
        me_expr* e = bitwise_and(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = bit_xor;
        apply_type_promotion(ret);
        promote_logical_bool(ret);
    }

    return ret;
}


static me_expr* bitwise_or(state* s) {
    /* <bitwise_or> =    <bitwise_xor> {"|" <bitwise_xor>} */
    me_expr* ret = bitwise_xor(s);
    CHECK_NULL(ret);

    while (s->type == TOK_BITWISE && (s->function == bit_or)) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* e = bitwise_xor(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
        promote_logical_bool(ret);
    }

    return ret;
}


static me_expr* comparison(state* s) {
    /* <comparison> =    <bitwise_or> {("<" | ">" | "<=" | ">=" | "==" | "!=") <bitwise_or>} */
    me_expr* ret = bitwise_or(s);
    CHECK_NULL(ret);

    while (s->type == TOK_COMPARE) {
        me_fun2 t = (me_fun2)s->function;
        next_token(s);
        me_expr* e = bitwise_or(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = (void*)t;
        apply_type_promotion(ret);
        /* Comparisons always return bool */
        ret->dtype = ME_BOOL;
    }

    return ret;
}

static me_expr* logical_not_expr(state* s) {
    /* <logical_not> = ["not" | "!"] <logical_not> | <comparison> */
    if (s->type == TOK_LOGICAL_NOT) {
        next_token(s);
        me_expr* inner = logical_not_expr(s);
        CHECK_NULL(inner);

        me_expr* ret = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, inner);
        CHECK_NULL(ret, me_free(inner));

        ret->function = logical_not;
        ret->dtype = ME_BOOL;
        return ret;
    }

    return comparison(s);
}

static me_expr* logical_and_expr(state* s) {
    /* <logical_and> = <logical_not> {("and" | "&&") <logical_not>} */
    me_expr* ret = logical_not_expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_LOGICAL_AND) {
        next_token(s);
        me_expr* e = logical_not_expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = logical_and;
        apply_type_promotion(ret);
        ret->dtype = ME_BOOL;
    }

    return ret;
}

static me_expr* logical_or_expr(state* s) {
    /* <logical_or> = <logical_and> {("or" | "||") <logical_and>} */
    me_expr* ret = logical_and_expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_LOGICAL_OR) {
        next_token(s);
        me_expr* e = logical_and_expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = logical_or;
        apply_type_promotion(ret);
        ret->dtype = ME_BOOL;
    }

    return ret;
}

me_expr* list(state* s) {
    /* <list>      =    <logical_or> {"," <logical_or>} */
    me_expr* ret = logical_or_expr(s);
    CHECK_NULL(ret);

    while (s->type == TOK_SEP) {
        next_token(s);
        me_expr* e = logical_or_expr(s);
        CHECK_NULL(e, me_free(ret));

        me_expr* prev = ret;
        ret = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, ret, e);
        CHECK_NULL(ret, me_free(e), me_free(prev));

        ret->function = comma;
        apply_type_promotion(ret);
    }

    return ret;
}


#define ME_FUN(...) ((double(*)(__VA_ARGS__))n->function)
#define M(e) me_eval_scalar(n->parameters[e])

static double me_eval_scalar(const me_expr* n) {
    if (!n) return NAN;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT: return n->value;
    case ME_STRING_CONSTANT: return NAN;
    case ME_VARIABLE:
        if (n->dtype == ME_STRING) return NAN;
        return *(const double*)n->bound;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
        switch (ARITY(n->type)) {
        case 0: return ME_FUN(void)();
        case 1: return ME_FUN(double)(M(0));
        case 2: return ME_FUN(double, double)(M(0), M(1));
        case 3: return ME_FUN(double, double, double)(M(0), M(1), M(2));
        case 4: return ME_FUN(double, double, double, double)(M(0), M(1), M(2), M(3));
        case 5: return ME_FUN(double, double, double, double, double)(M(0), M(1), M(2), M(3), M(4));
        case 6: return ME_FUN(double, double, double, double, double, double)(
                M(0), M(1), M(2), M(3), M(4), M(5));
        case 7: return ME_FUN(double, double, double, double, double, double, double)(
                M(0), M(1), M(2), M(3), M(4), M(5), M(6));
        default: return NAN;
        }

    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        switch (ARITY(n->type)) {
        case 0: return ME_FUN(void*)(n->parameters[0]);
        case 1: return ME_FUN(void*, double)(n->parameters[1], M(0));
        case 2: return ME_FUN(void*, double, double)(n->parameters[2], M(0), M(1));
        case 3: return ME_FUN(void*, double, double, double)(n->parameters[3], M(0), M(1), M(2));
        case 4: return ME_FUN(void*, double, double, double, double)(n->parameters[4], M(0), M(1), M(2), M(3));
        case 5: return ME_FUN(void*, double, double, double, double, double)(
                n->parameters[5], M(0), M(1), M(2), M(3), M(4));
        case 6: return ME_FUN(void*, double, double, double, double, double, double)(
                n->parameters[6], M(0), M(1), M(2), M(3), M(4), M(5));
        case 7: return ME_FUN(void*, double, double, double, double, double, double, double)(
                n->parameters[7], M(0), M(1), M(2), M(3), M(4), M(5), M(6));
        default: return NAN;
        }

    default: return NAN;
    }
}

#undef ME_FUN
#undef M

/* Specialized vector operations for better performance */
static void vec_add(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void vec_sub(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void vec_mul(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void vec_div(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static void vec_add_scalar(const double* a, double b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b;
    }
}

static void vec_mul_scalar(const double* a, double b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b;
    }
}

static void vec_pow(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = pow(a[i], b[i]);
    }
}

static void vec_pow_scalar(const double* a, double b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = pow(a[i], b);
    }
}

static void vec_sqrt(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sqrt(a[i]);
    }
}

static void vec_negate(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = -a[i];
    }
}

/* ============================================================================
 * FLOAT32 VECTOR OPERATIONS
 * ============================================================================ */

static void vec_add_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void vec_sub_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void vec_mul_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void vec_div_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static void vec_add_scalar_f32(const float* a, float b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b;
    }
}

static void vec_mul_scalar_f32(const float* a, float b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b;
    }
}

static void vec_pow_f32(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = powf(a[i], b[i]);
    }
}

static void vec_pow_scalar_f32(const float* a, float b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = powf(a[i], b);
    }
}

static void vec_sqrt_f32(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sqrtf(a[i]);
    }
}

static void vec_negame_f32(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = -a[i];
    }
}

/* ============================================================================
 * INTEGER VECTOR OPERATIONS (int8_t through uint64_t)
 * ============================================================================ */

/* Macros to generate integer vector operations */
#define DEFINE_INT_VEC_OPS(SUFFIX, TYPE) \
static void vec_add_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] + b[i]; \
} \
static void vec_sub_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] - b[i]; \
} \
static void vec_mul_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] * b[i]; \
} \
static void vec_div_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (b[i] != 0) ? (a[i] / b[i]) : 0; \
} \
static void vec_add_scalar_##SUFFIX(const TYPE *a, TYPE b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] + b; \
} \
static void vec_mul_scalar_##SUFFIX(const TYPE *a, TYPE b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] * b; \
} \
static void vec_pow_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (TYPE)pow((double)a[i], (double)b[i]); \
} \
static void vec_pow_scalar_##SUFFIX(const TYPE *a, TYPE b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (TYPE)pow((double)a[i], (double)b); \
} \
static void vec_sqrt_##SUFFIX(const TYPE *a, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = (TYPE)sqrt((double)a[i]); \
} \
static void vec_negame_##SUFFIX(const TYPE *a, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = -a[i]; \
} \
static void vec_and_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] & b[i]; \
} \
static void vec_or_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] | b[i]; \
} \
static void vec_xor_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] ^ b[i]; \
} \
static void vec_not_##SUFFIX(const TYPE *a, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = ~a[i]; \
} \
static void vec_shl_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] << b[i]; \
} \
static void vec_shr_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    _Pragma("GCC ivdep") \
    for (i = 0; i < n; i++) out[i] = a[i] >> b[i]; \
}

/* Generate ops for all integer types */
DEFINE_INT_VEC_OPS(i8, int8_t)
DEFINE_INT_VEC_OPS(i16, int16_t)
DEFINE_INT_VEC_OPS(i32, int32_t)
DEFINE_INT_VEC_OPS(i64, int64_t)
DEFINE_INT_VEC_OPS(u8, uint8_t)
DEFINE_INT_VEC_OPS(u16, uint16_t)
DEFINE_INT_VEC_OPS(u32, uint32_t)
DEFINE_INT_VEC_OPS(u64, uint64_t)

/* Boolean logical operations */
static void vec_and_bool(const bool* a, const bool* b, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = a[i] & b[i];
}

static void vec_or_bool(const bool* a, const bool* b, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = a[i] | b[i];
}

static void vec_xor_bool(const bool* a, const bool* b, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = a[i] != b[i];
}

static void vec_not_bool(const bool* a, bool* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = !a[i];
}

/* Comparison operations - generate for all numeric types */
/* Note: These return bool arrays, but we'll store them as the same type for simplicity */
#define DEFINE_COMPARE_OPS(SUFFIX, TYPE) \
static void vec_cmp_eq_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] == b[i]) ? 1 : 0; \
} \
static void vec_cmp_ne_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] != b[i]) ? 1 : 0; \
} \
static void vec_cmp_lt_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] < b[i]) ? 1 : 0; \
} \
static void vec_cmp_le_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] <= b[i]) ? 1 : 0; \
} \
static void vec_cmp_gt_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] > b[i]) ? 1 : 0; \
} \
static void vec_cmp_ge_##SUFFIX(const TYPE *a, const TYPE *b, TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = (a[i] >= b[i]) ? 1 : 0; \
}

/* Generate comparison ops for all types */
DEFINE_COMPARE_OPS(i8, int8_t)
DEFINE_COMPARE_OPS(i16, int16_t)
DEFINE_COMPARE_OPS(i32, int32_t)
DEFINE_COMPARE_OPS(i64, int64_t)
DEFINE_COMPARE_OPS(u8, uint8_t)
DEFINE_COMPARE_OPS(u16, uint16_t)
DEFINE_COMPARE_OPS(u32, uint32_t)
DEFINE_COMPARE_OPS(u64, uint64_t)
DEFINE_COMPARE_OPS(f32, float)
DEFINE_COMPARE_OPS(f64, double)

/* Complex operations */
static void vec_add_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c64(a[i], b[i]);
}

static void vec_sub_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = sub_c64(a[i], b[i]);
}

static void vec_mul_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c64(a[i], b[i]);
}

static void vec_div_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = div_c64(a[i], b[i]);
}

static void vec_add_scalar_c64(const float _Complex* a, float _Complex b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c64(a[i], b);
}

static void vec_mul_scalar_c64(const float _Complex* a, float _Complex b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c64(a[i], b);
}

static void vec_pow_c64(const float _Complex* a, const float _Complex* b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpowf(a[i], b[i]);
}

static void vec_pow_scalar_c64(const float _Complex* a, float _Complex b, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpowf(a[i], b);
}

static void vec_sqrt_c64(const float _Complex* a, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_csqrtf(a[i]);
}

static void vec_negame_c64(const float _Complex* a, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = neg_c64(a[i]);
}

static void vec_conj_c64(const float _Complex* a, float _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_conjf(a[i]);
}

static void vec_imag_c64(const float _Complex* a, float* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cimagf(a[i]);
}

static void vec_add_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c128(a[i], b[i]);
}

static void vec_sub_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = sub_c128(a[i], b[i]);
}

static void vec_mul_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c128(a[i], b[i]);
}

static void vec_div_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = div_c128(a[i], b[i]);
}

static void vec_add_scalar_c128(const double _Complex* a, double _Complex b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = add_c128(a[i], b);
}

static void vec_mul_scalar_c128(const double _Complex* a, double _Complex b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = mul_c128(a[i], b);
}

static void vec_pow_c128(const double _Complex* a, const double _Complex* b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpow(a[i], b[i]);
}

static void vec_pow_scalar_c128(const double _Complex* a, double _Complex b, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cpow(a[i], b);
}

static void vec_sqrt_c128(const double _Complex* a, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_csqrt(a[i]);
}

static void vec_negame_c128(const double _Complex* a, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = neg_c128(a[i]);
}

static void vec_conj_c128(const double _Complex* a, double _Complex* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_conj(a[i]);
}

static void vec_imag_c128(const double _Complex* a, double* out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = me_cimag(a[i]);
}

/* ============================================================================
 * TYPE CONVERSION FUNCTIONS
 * ============================================================================
 * These functions convert between different data types for mixed-type expressions.
 */

#define DEFINE_VEC_CONVERT(FROM_SUFFIX, TO_SUFFIX, FROM_TYPE, TO_TYPE) \
static void vec_convert_##FROM_SUFFIX##_to_##TO_SUFFIX(const FROM_TYPE *in, TO_TYPE *out, int n) { \
    int i; \
    IVDEP \
    for (i = 0; i < n; i++) out[i] = TO_TYPE_##TO_SUFFIX(in[i]); \
}


/* Generate all conversion functions */
/* Conversions FROM bool TO other types */
DEFINE_VEC_CONVERT(bool, i8, bool, int8_t)
DEFINE_VEC_CONVERT(bool, i16, bool, int16_t)
DEFINE_VEC_CONVERT(bool, i32, bool, int32_t)
DEFINE_VEC_CONVERT(bool, i64, bool, int64_t)
DEFINE_VEC_CONVERT(bool, u8, bool, uint8_t)
DEFINE_VEC_CONVERT(bool, u16, bool, uint16_t)
DEFINE_VEC_CONVERT(bool, u32, bool, uint32_t)
DEFINE_VEC_CONVERT(bool, u64, bool, uint64_t)
DEFINE_VEC_CONVERT(bool, f32, bool, float)
DEFINE_VEC_CONVERT(bool, f64, bool, double)

/* Conversions FROM other types TO bool */
DEFINE_VEC_CONVERT(i8, bool, int8_t, bool)
DEFINE_VEC_CONVERT(i16, bool, int16_t, bool)
DEFINE_VEC_CONVERT(i32, bool, int32_t, bool)
DEFINE_VEC_CONVERT(i64, bool, int64_t, bool)
DEFINE_VEC_CONVERT(u8, bool, uint8_t, bool)
DEFINE_VEC_CONVERT(u16, bool, uint16_t, bool)
DEFINE_VEC_CONVERT(u32, bool, uint32_t, bool)
DEFINE_VEC_CONVERT(u64, bool, uint64_t, bool)
DEFINE_VEC_CONVERT(f32, bool, float, bool)
DEFINE_VEC_CONVERT(f64, bool, double, bool)
DEFINE_VEC_CONVERT(f64, f32, double, float)

static void vec_convert_c64_to_bool(const float _Complex *in, bool *out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = IS_NONZERO_c64(in[i]);
}

static void vec_convert_c128_to_bool(const double _Complex *in, bool *out, int n) {
    int i;
    IVDEP
    for (i = 0; i < n; i++) out[i] = IS_NONZERO_c128(in[i]);
}

DEFINE_VEC_CONVERT(i8, i16, int8_t, int16_t)
DEFINE_VEC_CONVERT(i8, i32, int8_t, int32_t)
DEFINE_VEC_CONVERT(i8, i64, int8_t, int64_t)
DEFINE_VEC_CONVERT(i8, f32, int8_t, float)
DEFINE_VEC_CONVERT(i8, f64, int8_t, double)

DEFINE_VEC_CONVERT(i16, i32, int16_t, int32_t)
DEFINE_VEC_CONVERT(i16, i64, int16_t, int64_t)
DEFINE_VEC_CONVERT(i16, f32, int16_t, float)
DEFINE_VEC_CONVERT(i16, f64, int16_t, double)

DEFINE_VEC_CONVERT(i32, i64, int32_t, int64_t)
DEFINE_VEC_CONVERT(i32, f32, int32_t, float)
DEFINE_VEC_CONVERT(i32, f64, int32_t, double)

DEFINE_VEC_CONVERT(i64, f64, int64_t, double)

DEFINE_VEC_CONVERT(u8, u16, uint8_t, uint16_t)
DEFINE_VEC_CONVERT(u8, u32, uint8_t, uint32_t)
DEFINE_VEC_CONVERT(u8, u64, uint8_t, uint64_t)
DEFINE_VEC_CONVERT(u8, i16, uint8_t, int16_t)
DEFINE_VEC_CONVERT(u8, i32, uint8_t, int32_t)
DEFINE_VEC_CONVERT(u8, i64, uint8_t, int64_t)
DEFINE_VEC_CONVERT(u8, f32, uint8_t, float)
DEFINE_VEC_CONVERT(u8, f64, uint8_t, double)

DEFINE_VEC_CONVERT(u16, u32, uint16_t, uint32_t)
DEFINE_VEC_CONVERT(u16, u64, uint16_t, uint64_t)
DEFINE_VEC_CONVERT(u16, i32, uint16_t, int32_t)
DEFINE_VEC_CONVERT(u16, i64, uint16_t, int64_t)
DEFINE_VEC_CONVERT(u16, f32, uint16_t, float)
DEFINE_VEC_CONVERT(u16, f64, uint16_t, double)

DEFINE_VEC_CONVERT(u32, u64, uint32_t, uint64_t)
DEFINE_VEC_CONVERT(u32, i64, uint32_t, int64_t)
DEFINE_VEC_CONVERT(u32, f64, uint32_t, double)

DEFINE_VEC_CONVERT(u64, f64, uint64_t, double)

DEFINE_VEC_CONVERT(f32, f64, float, double)
DEFINE_VEC_CONVERT(f32, c64, float, float _Complex)
DEFINE_VEC_CONVERT(f32, c128, float, double _Complex)

DEFINE_VEC_CONVERT(f64, c128, double, double _Complex)

DEFINE_VEC_CONVERT(c64, c128, float _Complex, double _Complex)

/* Function to get conversion function pointer */
static convert_func_t get_convert_func(me_dtype from, me_dtype to) {
    /* Return conversion function for a specific type pair */
    if (from == to) return NULL; // No conversion needed
    if (from == ME_STRING || to == ME_STRING) return NULL;

#define CONV_CASE(FROM, TO, FROM_S, TO_S) \
        if (from == FROM && to == TO) return (convert_func_t)vec_convert_##FROM_S##_to_##TO_S;

    CONV_CASE(ME_BOOL, ME_INT8, bool, i8)
    CONV_CASE(ME_BOOL, ME_INT16, bool, i16)
    CONV_CASE(ME_BOOL, ME_INT32, bool, i32)
    CONV_CASE(ME_BOOL, ME_INT64, bool, i64)
    CONV_CASE(ME_BOOL, ME_UINT8, bool, u8)
    CONV_CASE(ME_BOOL, ME_UINT16, bool, u16)
    CONV_CASE(ME_BOOL, ME_UINT32, bool, u32)
    CONV_CASE(ME_BOOL, ME_UINT64, bool, u64)
    CONV_CASE(ME_BOOL, ME_FLOAT32, bool, f32)
    CONV_CASE(ME_BOOL, ME_FLOAT64, bool, f64)

    CONV_CASE(ME_INT8, ME_BOOL, i8, bool)
    CONV_CASE(ME_INT16, ME_BOOL, i16, bool)
    CONV_CASE(ME_INT32, ME_BOOL, i32, bool)
    CONV_CASE(ME_INT64, ME_BOOL, i64, bool)
    CONV_CASE(ME_UINT8, ME_BOOL, u8, bool)
    CONV_CASE(ME_UINT16, ME_BOOL, u16, bool)
    CONV_CASE(ME_UINT32, ME_BOOL, u32, bool)
    CONV_CASE(ME_UINT64, ME_BOOL, u64, bool)
    CONV_CASE(ME_FLOAT32, ME_BOOL, f32, bool)
    CONV_CASE(ME_FLOAT64, ME_BOOL, f64, bool)
    if (from == ME_COMPLEX64 && to == ME_BOOL) return (convert_func_t)vec_convert_c64_to_bool;
    if (from == ME_COMPLEX128 && to == ME_BOOL) return (convert_func_t)vec_convert_c128_to_bool;

    CONV_CASE(ME_INT8, ME_INT16, i8, i16)
    CONV_CASE(ME_INT8, ME_INT32, i8, i32)
    CONV_CASE(ME_INT8, ME_INT64, i8, i64)
    CONV_CASE(ME_INT8, ME_FLOAT32, i8, f32)
    CONV_CASE(ME_INT8, ME_FLOAT64, i8, f64)

    CONV_CASE(ME_INT16, ME_INT32, i16, i32)
    CONV_CASE(ME_INT16, ME_INT64, i16, i64)
    CONV_CASE(ME_INT16, ME_FLOAT32, i16, f32)
    CONV_CASE(ME_INT16, ME_FLOAT64, i16, f64)

    CONV_CASE(ME_INT32, ME_INT64, i32, i64)
    CONV_CASE(ME_INT32, ME_FLOAT32, i32, f32)
    CONV_CASE(ME_INT32, ME_FLOAT64, i32, f64)

    CONV_CASE(ME_INT64, ME_FLOAT64, i64, f64)

    CONV_CASE(ME_UINT8, ME_UINT16, u8, u16)
    CONV_CASE(ME_UINT8, ME_UINT32, u8, u32)
    CONV_CASE(ME_UINT8, ME_UINT64, u8, u64)
    CONV_CASE(ME_UINT8, ME_INT16, u8, i16)
    CONV_CASE(ME_UINT8, ME_INT32, u8, i32)
    CONV_CASE(ME_UINT8, ME_INT64, u8, i64)
    CONV_CASE(ME_UINT8, ME_FLOAT32, u8, f32)
    CONV_CASE(ME_UINT8, ME_FLOAT64, u8, f64)

    CONV_CASE(ME_UINT16, ME_UINT32, u16, u32)
    CONV_CASE(ME_UINT16, ME_UINT64, u16, u64)
    CONV_CASE(ME_UINT16, ME_INT32, u16, i32)
    CONV_CASE(ME_UINT16, ME_INT64, u16, i64)
    CONV_CASE(ME_UINT16, ME_FLOAT32, u16, f32)
    CONV_CASE(ME_UINT16, ME_FLOAT64, u16, f64)

    CONV_CASE(ME_UINT32, ME_UINT64, u32, u64)
    CONV_CASE(ME_UINT32, ME_INT64, u32, i64)
    CONV_CASE(ME_UINT32, ME_FLOAT64, u32, f64)

    CONV_CASE(ME_UINT64, ME_FLOAT64, u64, f64)

    CONV_CASE(ME_FLOAT32, ME_FLOAT64, f32, f64)
    CONV_CASE(ME_FLOAT32, ME_COMPLEX64, f32, c64)
    CONV_CASE(ME_FLOAT32, ME_COMPLEX128, f32, c128)

    CONV_CASE(ME_FLOAT64, ME_FLOAT32, f64, f32)
    CONV_CASE(ME_FLOAT64, ME_COMPLEX128, f64, c128)

    CONV_CASE(ME_COMPLEX64, ME_COMPLEX128, c64, c128)

#undef CONV_CASE

    return NULL; // Unsupported conversion
}


typedef double (*me_fun1)(double);

typedef float (*me_fun1_f32)(float);

/* Template for type-specific evaluator */
#define DEFINE_ME_EVAL(SUFFIX, TYPE, VEC_ADD, VEC_SUB, VEC_MUL, VEC_DIV, VEC_POW, \
    VEC_ADD_SCALAR, VEC_MUL_SCALAR, VEC_POW_SCALAR, \
    VEC_SQRT, VEC_SIN, VEC_COS, VEC_TAN, \
    VEC_ASIN, VEC_ACOS, VEC_ATAN, \
    VEC_EXP, VEC_LOG, VEC_LOG10, VEC_LOG1P, VEC_LOG2, VEC_EXPM1, VEC_EXP2, VEC_EXP10, \
    VEC_SINH, VEC_COSH, VEC_TANH, \
    VEC_ABS, VEC_CEIL, VEC_FLOOR, VEC_ROUND, VEC_TRUNC, \
    VEC_ACOSH, VEC_ASINH, VEC_ATANH, VEC_CBRT, VEC_ERF, VEC_ERFC, VEC_SINPI, VEC_COSPI, \
    VEC_TGAMMA, VEC_LGAMMA, VEC_RINT, \
    VEC_COPYSIGN, VEC_FDIM, VEC_FMAX, VEC_FMIN, VEC_FMOD, VEC_HYPOT, VEC_LDEXP, \
    VEC_NEXTAFTER, VEC_REMAINDER, VEC_FMA, \
    VEC_NEGATE, \
    SQRT_FUNC, SIN_FUNC, COS_FUNC, TAN_FUNC, ASIN_FUNC, ACOS_FUNC, ATAN_FUNC, \
    EXP_FUNC, LOG_FUNC, LOG10_FUNC, LOG1P_FUNC, LOG2_FUNC, EXPM1_FUNC, EXP2_FUNC, EXP10_FUNC, \
    SINH_FUNC, COSH_FUNC, TANH_FUNC, ACOSH_FUNC, ASINH_FUNC, ATANH_FUNC, CBRT_FUNC, ERF_FUNC, ERFC_FUNC, \
    SINPI_FUNC, COSPI_FUNC, TGAMMA_FUNC, LGAMMA_FUNC, RINT_FUNC, \
    COPYSIGN_FUNC, FDIM_FUNC, FMAX_FUNC, FMIN_FUNC, FMOD_FUNC, HYPOT_FUNC, LDEXP_FUNC, \
    NEXTAFTER_FUNC, REMAINDER_FUNC, FMA_FUNC, \
    CEIL_FUNC, FLOOR_FUNC, ROUND_FUNC, TRUNC_FUNC, FABS_FUNC, POW_FUNC, \
    VEC_CONJ, HAS_VEC_TAN, HAS_VEC_ASIN, HAS_VEC_ACOS, HAS_VEC_ATAN, HAS_VEC_MATH, \
    VEC_ATAN2, ATAN2_FUNC, HAS_VEC_ATAN2) \
static void me_eval_##SUFFIX(const me_expr *n) { \
    if (!n || !n->output) return; \
    if (is_reduction_node(n)) { \
        eval_reduction(n, n->nitems); \
        return; \
    } \
    if (n->nitems <= 0) return; \
    \
    int i, j; \
    const int arity = ARITY(n->type); \
    TYPE *output = (TYPE*)n->output; \
    \
    switch(TYPE_MASK(n->type)) { \
        case ME_CONSTANT: \
            { \
                TYPE val = TO_TYPE_##SUFFIX(n->value); \
                for (i = 0; i < n->nitems; i++) { \
                    output[i] = val; \
                } \
            } \
            break; \
            \
        case ME_VARIABLE: \
            { \
                const TYPE *src = (const TYPE*)n->bound; \
                for (i = 0; i < n->nitems; i++) { \
                    output[i] = src[i]; \
                } \
            } \
            break; \
        \
        case ME_FUNCTION0: case ME_FUNCTION1: case ME_FUNCTION2: case ME_FUNCTION3: \
        case ME_FUNCTION4: case ME_FUNCTION5: case ME_FUNCTION6: case ME_FUNCTION7: \
        case ME_CLOSURE0: case ME_CLOSURE1: case ME_CLOSURE2: case ME_CLOSURE3: \
        case ME_CLOSURE4: case ME_CLOSURE5: case ME_CLOSURE6: case ME_CLOSURE7: \
            { \
            /* Check if this node is a conversion node (arity=1, function=NULL) */ \
            int is_conv_node = (arity == 1 && IS_FUNCTION(n->type) && n->function == NULL); \
            \
            if (is_conv_node) { \
                /* Conversion node: evaluate child with native dtype, then convert */ \
                me_expr *source = (me_expr*)n->parameters[0]; \
                me_dtype source_dtype = n->input_dtype; \
                size_t source_size = dtype_size(source_dtype); \
                \
                if (source->type != ME_CONSTANT && source->type != ME_VARIABLE) { \
                    if (!source->output) { \
                        source->output = malloc(n->nitems * source_size); \
                        source->nitems = n->nitems; \
                    } \
                    /* Evaluate source with its native dtype via private_eval */ \
                    private_eval(source); \
                } \
                \
                /* Perform the conversion */ \
                const void *src_data = (source->type == ME_CONSTANT) ? NULL : \
                    (source->type == ME_VARIABLE) ? source->bound : source->output; \
                if (src_data) { \
                    convert_func_t conv = get_convert_func(source_dtype, n->dtype); \
                    if (conv) { \
                        conv(src_data, output, n->nitems); \
                    } \
                } \
                break; \
            } \
            \
            if (IS_FUNCTION(n->type) && arity == 2 && \
                (is_comparison_node(n) || is_string_function(n->function))) { \
                me_expr *left = (me_expr*)n->parameters[0]; \
                me_expr *right = (me_expr*)n->parameters[1]; \
                if (is_string_node(left) && is_string_node(right)) { \
                    for (i = 0; i < n->nitems; i++) { \
                        const uint32_t *ldata = NULL; \
                        const uint32_t *rdata = NULL; \
                        size_t llen = 0; \
                        size_t rlen = 0; \
                        if (!string_view_at(left, i, &ldata, &llen) || \
                            !string_view_at(right, i, &rdata, &rlen)) { \
                            return; \
                        } \
                        bool result = false; \
                        if (is_comparison_node(n)) { \
                            if (n->function == (void*)cmp_eq) { \
                                result = string_equals(ldata, llen, rdata, rlen); \
                            } else if (n->function == (void*)cmp_ne) { \
                                result = !string_equals(ldata, llen, rdata, rlen); \
                            } else { \
                                return; \
                            } \
                        } else { \
                            if (n->function == (void*)str_startswith) { \
                                result = string_starts_with(ldata, llen, rdata, rlen); \
                            } else if (n->function == (void*)str_endswith) { \
                                result = string_ends_with(ldata, llen, rdata, rlen); \
                            } else if (n->function == (void*)str_contains) { \
                                result = string_contains(ldata, llen, rdata, rlen); \
                            } else { \
                                return; \
                            } \
                        } \
                        output[i] = result ? TO_TYPE_##SUFFIX(1) : TO_TYPE_##SUFFIX(0); \
                    } \
                    break; \
                } \
            } \
            \
            for (j = 0; j < arity; j++) { \
                me_expr *param = (me_expr*)n->parameters[j]; \
                if (param->type != ME_CONSTANT && param->type != ME_VARIABLE) { \
                    /* Check if param is a conversion node - if so, let it keep its dtype */ \
                    int param_is_conv = (ARITY(param->type) == 1 && IS_FUNCTION(param->type) && param->function == NULL); \
                    if (!param->output) { \
                        if (param_is_conv) { \
                            /* Conversion node: allocate for target dtype */ \
                            param->output = malloc(n->nitems * sizeof(TYPE)); \
                        } else { \
                            param->output = malloc(n->nitems * sizeof(TYPE)); \
                            param->dtype = n->dtype; \
                        } \
                        param->nitems = n->nitems; \
                    } \
                    if (param_is_conv) { \
                        /* Evaluate conversion node - it will handle its child's dtype */ \
                        me_eval_##SUFFIX(param); \
                    } else { \
                        me_eval_##SUFFIX(param); \
                    } \
                } \
            } \
            \
            if (arity == 2 && IS_FUNCTION(n->type)) { \
                me_expr *left = (me_expr*)n->parameters[0]; \
                me_expr *right = (me_expr*)n->parameters[1]; \
                \
                const TYPE *ldata = (left->type == ME_CONSTANT) ? NULL : \
                                   (left->type == ME_VARIABLE) ? (const TYPE*)left->bound : (const TYPE*)left->output; \
                const TYPE *rdata = (right->type == ME_CONSTANT) ? NULL : \
                                    (right->type == ME_VARIABLE) ? (const TYPE*)right->bound : (const TYPE*)right->output; \
                \
                me_fun2 func = (me_fun2)n->function; \
                \
                if (func == add) { \
                    if (ldata && rdata) { \
                        VEC_ADD(ldata, rdata, output, n->nitems); \
                    } else if (ldata && right->type == ME_CONSTANT) { \
                        VEC_ADD_SCALAR(ldata, TO_TYPE_##SUFFIX(right->value), output, n->nitems); \
                    } else if (left->type == ME_CONSTANT && rdata) { \
                        VEC_ADD_SCALAR(rdata, TO_TYPE_##SUFFIX(left->value), output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == sub) { \
                    if (ldata && rdata) { \
                        VEC_SUB(ldata, rdata, output, n->nitems); \
                    } else if (ldata && right->type == ME_CONSTANT) { \
                        TYPE b = TO_TYPE_##SUFFIX(right->value); \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = ldata[i] - b; \
                        } \
                    } else if (left->type == ME_CONSTANT && rdata) { \
                        TYPE a = TO_TYPE_##SUFFIX(left->value); \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = a - rdata[i]; \
                        } \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == mul) { \
                    if (ldata && rdata) { \
                        VEC_MUL(ldata, rdata, output, n->nitems); \
                    } else if (ldata && right->type == ME_CONSTANT) { \
                        VEC_MUL_SCALAR(ldata, TO_TYPE_##SUFFIX(right->value), output, n->nitems); \
                    } else if (left->type == ME_CONSTANT && rdata) { \
                        VEC_MUL_SCALAR(rdata, TO_TYPE_##SUFFIX(left->value), output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == divide) { \
                    if (ldata && rdata) { \
                        VEC_DIV(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)pow) { \
                    if (ldata && rdata) { \
                        VEC_POW(ldata, rdata, output, n->nitems); \
                    } else if (ldata && right->type == ME_CONSTANT) { \
                        VEC_POW_SCALAR(ldata, TO_TYPE_##SUFFIX(right->value), output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)ATAN2_FUNC) { \
                    if (ldata && rdata && HAS_VEC_ATAN2) { \
                        VEC_ATAN2(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)copysign) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_COPYSIGN(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)fdim) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_FDIM(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)fmax) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_FMAX(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)fmin) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_FMIN(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)fmod) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_FMOD(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)hypot) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_HYPOT(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)LDEXP_FUNC) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_LDEXP(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)nextafter) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_NEXTAFTER(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else if (func == (me_fun2)remainder) { \
                    if (ldata && rdata && HAS_VEC_MATH) { \
                        VEC_REMAINDER(ldata, rdata, output, n->nitems); \
                    } else { \
                        goto general_case_binary_##SUFFIX; \
                    } \
                } else { \
                    general_case_binary_##SUFFIX: \
                    for (i = 0; i < n->nitems; i++) { \
                        double a = (left->type == ME_CONSTANT) ? left->value : \
                                  FROM_TYPE_##SUFFIX(ldata[i]); \
                        double b = (right->type == ME_CONSTANT) ? right->value : \
                                  FROM_TYPE_##SUFFIX(rdata[i]); \
                        output[i] = TO_TYPE_##SUFFIX(func(a, b)); \
                    } \
                } \
            } else if (arity == 3 && IS_FUNCTION(n->type) && n->function == (void*)fma) { \
                me_expr *xexpr = (me_expr*)n->parameters[0]; \
                me_expr *yexpr = (me_expr*)n->parameters[1]; \
                me_expr *zexpr = (me_expr*)n->parameters[2]; \
                \
                const TYPE *xdata = (xexpr->type == ME_CONSTANT) ? NULL : \
                                   (xexpr->type == ME_VARIABLE) ? (const TYPE*)xexpr->bound : (const TYPE*)xexpr->output; \
                const TYPE *ydata = (yexpr->type == ME_CONSTANT) ? NULL : \
                                   (yexpr->type == ME_VARIABLE) ? (const TYPE*)yexpr->bound : (const TYPE*)yexpr->output; \
                const TYPE *zdata = (zexpr->type == ME_CONSTANT) ? NULL : \
                                   (zexpr->type == ME_VARIABLE) ? (const TYPE*)zexpr->bound : (const TYPE*)zexpr->output; \
                \
                if (HAS_VEC_MATH && xdata && ydata && zdata) { \
                    VEC_FMA(xdata, ydata, zdata, output, n->nitems); \
                } else { \
                    for (i = 0; i < n->nitems; i++) { \
                        double a = (xexpr->type == ME_CONSTANT) ? xexpr->value : \
                                  FROM_TYPE_##SUFFIX(xdata[i]); \
                        double b = (yexpr->type == ME_CONSTANT) ? yexpr->value : \
                                  FROM_TYPE_##SUFFIX(ydata[i]); \
                        double c = (zexpr->type == ME_CONSTANT) ? zexpr->value : \
                                  FROM_TYPE_##SUFFIX(zdata[i]); \
                        output[i] = TO_TYPE_##SUFFIX(FMA_FUNC(a, b, c)); \
                    } \
                } \
            } else if (arity == 3 && IS_FUNCTION(n->type) && n->function == (void*)where_scalar) { \
                /* where(cond, x, y) – NumPy-like semantics: cond != 0 selects x else y */ \
                me_expr *cond = (me_expr*)n->parameters[0]; \
                me_expr *xexpr = (me_expr*)n->parameters[1]; \
                me_expr *yexpr = (me_expr*)n->parameters[2]; \
                \
                const TYPE *cdata = (const TYPE*)((cond->type == ME_VARIABLE) ? cond->bound : cond->output); \
                const TYPE *xdata = (const TYPE*)((xexpr->type == ME_VARIABLE) ? xexpr->bound : xexpr->output); \
                const TYPE *ydata = (const TYPE*)((yexpr->type == ME_VARIABLE) ? yexpr->bound : yexpr->output); \
                \
                for (i = 0; i < n->nitems; i++) { \
                    output[i] = (IS_NONZERO_##SUFFIX(cdata[i])) ? xdata[i] : ydata[i]; \
                } \
            } \
            else if (arity == 1 && IS_FUNCTION(n->type)) { \
                me_expr *arg = (me_expr*)n->parameters[0]; \
                \
                const TYPE *adata = (arg->type == ME_CONSTANT) ? NULL : \
                                   (arg->type == ME_VARIABLE) ? (const TYPE*)arg->bound : (const TYPE*)arg->output; \
                void *arg_temp = NULL; \
                /* Convert variable inputs to node dtype to avoid reinterpreting buffers. */ \
                if (arg->type == ME_VARIABLE && arg->dtype != n->dtype) { \
                    convert_func_t conv = get_convert_func(arg->dtype, n->dtype); \
                    if (conv) { \
                        arg_temp = malloc((size_t)n->nitems * sizeof(TYPE)); \
                        if (arg_temp) { \
                            conv(arg->bound, arg_temp, n->nitems); \
                            adata = (const TYPE*)arg_temp; \
                        } \
                    } \
                } \
                \
                const void *func_ptr = n->function; \
                \
                if (func_ptr == (void*)sqrt) { \
                    if (adata) VEC_SQRT(adata, output, n->nitems); \
                } else if (func_ptr == (void*)sin) { \
                    if (adata) VEC_SIN(adata, output, n->nitems); \
                } else if (func_ptr == (void*)cos) { \
                    if (adata) VEC_COS(adata, output, n->nitems); \
                } else if (func_ptr == (void*)tan) { \
                    if (adata) { \
                        if (HAS_VEC_TAN) { \
                            VEC_TAN(adata, output, n->nitems); \
                        } else { \
                            for (i = 0; i < n->nitems; i++) { \
                                output[i] = TO_TYPE_##SUFFIX(TAN_FUNC(FROM_TYPE_##SUFFIX(adata[i]))); \
                            } \
                        } \
                    } \
                } else if (func_ptr == (void*)ASIN_FUNC) { \
                    if (adata) { \
                        if (HAS_VEC_ASIN) { \
                            VEC_ASIN(adata, output, n->nitems); \
                        } else { \
                            for (i = 0; i < n->nitems; i++) { \
                                output[i] = TO_TYPE_##SUFFIX(ASIN_FUNC(FROM_TYPE_##SUFFIX(adata[i]))); \
                            } \
                        } \
                    } \
                } else if (func_ptr == (void*)ACOS_FUNC) { \
                    if (adata) { \
                        if (HAS_VEC_ACOS) { \
                            VEC_ACOS(adata, output, n->nitems); \
                        } else { \
                            for (i = 0; i < n->nitems; i++) { \
                                output[i] = TO_TYPE_##SUFFIX(ACOS_FUNC(FROM_TYPE_##SUFFIX(adata[i]))); \
                            } \
                        } \
                    } \
                } else if (func_ptr == (void*)ATAN_FUNC) { \
                    if (adata) { \
                        if (HAS_VEC_ATAN) { \
                            VEC_ATAN(adata, output, n->nitems); \
                        } else { \
                            for (i = 0; i < n->nitems; i++) { \
                                output[i] = TO_TYPE_##SUFFIX(ATAN_FUNC(FROM_TYPE_##SUFFIX(adata[i]))); \
                            } \
                        } \
                    } \
                } else if (HAS_VEC_MATH && func_ptr == (void*)exp) { \
                    if (adata) VEC_EXP(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)log) { \
                    if (adata) VEC_LOG(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)log10) { \
                    if (adata) VEC_LOG10(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)log1p_wrapper) { \
                    if (adata) VEC_LOG1P(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)log2_wrapper) { \
                    if (adata) VEC_LOG2(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)expm1_wrapper) { \
                    if (adata) VEC_EXPM1(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)sinh) { \
                    if (adata) VEC_SINH(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)cosh) { \
                    if (adata) VEC_COSH(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)tanh) { \
                    if (adata) VEC_TANH(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)exp2) { \
                    if (adata) VEC_EXP2(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)exp10_wrapper) { \
                    if (adata) VEC_EXP10(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)acosh) { \
                    if (adata) VEC_ACOSH(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)asinh) { \
                    if (adata) VEC_ASINH(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)atanh) { \
                    if (adata) VEC_ATANH(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)cbrt) { \
                    if (adata) VEC_CBRT(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)erf) { \
                    if (adata) VEC_ERF(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)erfc) { \
                    if (adata) VEC_ERFC(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)sinpi_wrapper) { \
                    if (adata) VEC_SINPI(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)cospi_wrapper) { \
                    if (adata) VEC_COSPI(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)tgamma) { \
                    if (adata) VEC_TGAMMA(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)lgamma) { \
                    if (adata) VEC_LGAMMA(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)rint) { \
                    if (adata) VEC_RINT(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)fabs) { \
                    if (adata) VEC_ABS(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)ceil) { \
                    if (adata) VEC_CEIL(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)floor) { \
                    if (adata) VEC_FLOOR(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)round_wrapper) { \
                    if (adata) VEC_ROUND(adata, output, n->nitems); \
                } else if (HAS_VEC_MATH && func_ptr == (void*)trunc_wrapper) { \
                    if (adata) VEC_TRUNC(adata, output, n->nitems); \
                } else if (func_ptr == (void*)negate) { \
                    if (adata) VEC_NEGATE(adata, output, n->nitems); \
                } else if (func_ptr == (void*)imag_wrapper) { \
                    /* NumPy semantics: imag(real) == 0 with same dtype */ \
                    if (adata) { \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = TO_TYPE_##SUFFIX(0); \
                        } \
                    } \
                } else if (func_ptr == (void*)real_wrapper) { \
                    /* NumPy semantics: real(real) == real with same dtype */ \
                    if (adata) { \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = adata[i]; \
                        } \
                    } \
                } else if (func_ptr == (void*)conj_wrapper) { \
                    if (adata) VEC_CONJ(adata, output, n->nitems); \
                } else { \
                    me_fun1 func = (me_fun1)func_ptr; \
                    if (arg->type == ME_CONSTANT) { \
                        TYPE val = TO_TYPE_##SUFFIX(func(arg->value)); \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = val; \
                        } \
                    } else { \
                        for (i = 0; i < n->nitems; i++) { \
                            output[i] = TO_TYPE_##SUFFIX(func(FROM_TYPE_##SUFFIX(adata[i]))); \
                        } \
                    } \
                } \
                if (arg_temp) { \
                    free(arg_temp); \
                } \
            } \
            else { \
                for (i = 0; i < n->nitems; i++) { \
                    double args[7]; \
                    \
                    for (j = 0; j < arity; j++) { \
                        me_expr *param = (me_expr*)n->parameters[j]; \
                        const TYPE *pdata = (const TYPE*)((param->type == ME_VARIABLE) ? param->bound : param->output); \
                        if (param->type == ME_CONSTANT) { \
                            args[j] = param->value; \
                        } else { \
                            args[j] = FROM_TYPE_##SUFFIX(pdata[i]); \
                        } \
                    } \
                    \
                    if (IS_FUNCTION(n->type)) { \
                        switch(arity) { \
                            case 0: output[i] = TO_TYPE_##SUFFIX(((double(*)(void))n->function)()); break; \
                            case 3: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double))n->function)(args[0], args[1], args[2])); break; \
                            case 4: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double))n->function)(args[0], args[1], args[2], args[3])); break; \
                            case 5: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4])); break; \
                            case 6: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4], args[5])); break; \
                            case 7: output[i] = TO_TYPE_##SUFFIX(((double(*)(double,double,double,double,double,double,double))n->function)(args[0], args[1], args[2], args[3], args[4], args[5], args[6])); break; \
                        } \
                    } else if (IS_CLOSURE(n->type)) { \
                        void *context = n->parameters[arity]; \
                        switch(arity) { \
                            case 0: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*))n->function)(context)); break; \
                            case 1: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double))n->function)(context, args[0])); break; \
                            case 2: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double))n->function)(context, args[0], args[1])); break; \
                            case 3: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double))n->function)(context, args[0], args[1], args[2])); break; \
                            case 4: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3])); break; \
                            case 5: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4])); break; \
                            case 6: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4], args[5])); break; \
                            case 7: output[i] = TO_TYPE_##SUFFIX(((double(*)(void*,double,double,double,double,double,double,double))n->function)(context, args[0], args[1], args[2], args[3], args[4], args[5], args[6])); break; \
                        } \
                    } \
                } \
            } \
            } \
            break; \
        \
        default: \
            for (i = 0; i < n->nitems; i++) { \
                output[i] = TO_TYPE_##SUFFIX(NAN); \
            } \
            break; \
    } \
}

/* Vector operation macros - expand to inline loops */
#define vec_add(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow(a, b, out, n) vec_pow_dispatch((a), (b), (out), (n))
#define vec_add_scalar(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = pow((a)[_i], (b)); } while(0)
#define vec_sqrt(a, out, n) vec_sqrt_dispatch((a), (out), (n))
#define vec_sin(a, out, n) vec_sin_dispatch((a), (out), (n))
#define vec_cos(a, out, n) vec_cos_dispatch((a), (out), (n))
#define vec_asin(a, out, n) vec_asin_dispatch((a), (out), (n))
#define vec_acos(a, out, n) vec_acos_dispatch((a), (out), (n))
#define vec_atan(a, out, n) vec_atan_dispatch((a), (out), (n))
#define vec_tan(a, out, n) vec_tan_dispatch((a), (out), (n))
#define vec_exp(a, out, n) vec_exp_dispatch((a), (out), (n))
#define vec_expm1(a, out, n) vec_expm1_dispatch((a), (out), (n))
#define vec_exp2(a, out, n) vec_exp2_dispatch((a), (out), (n))
#define vec_exp10(a, out, n) vec_exp10_dispatch((a), (out), (n))
#define vec_log(a, out, n) vec_log_dispatch((a), (out), (n))
#define vec_log10(a, out, n) vec_log10_dispatch((a), (out), (n))
#define vec_log1p(a, out, n) vec_log1p_dispatch((a), (out), (n))
#define vec_log2(a, out, n) vec_log2_dispatch((a), (out), (n))
#define vec_sinh(a, out, n) vec_sinh_dispatch((a), (out), (n))
#define vec_cosh(a, out, n) vec_cosh_dispatch((a), (out), (n))
#define vec_tanh(a, out, n) vec_tanh_dispatch((a), (out), (n))
#define vec_abs(a, out, n) vec_abs_dispatch((a), (out), (n))
#define vec_ceil(a, out, n) vec_ceil_dispatch((a), (out), (n))
#define vec_floor(a, out, n) vec_floor_dispatch((a), (out), (n))
#define vec_round(a, out, n) vec_round_dispatch((a), (out), (n))
#define vec_trunc(a, out, n) vec_trunc_dispatch((a), (out), (n))
#define vec_acosh(a, out, n) vec_acosh_dispatch((a), (out), (n))
#define vec_asinh(a, out, n) vec_asinh_dispatch((a), (out), (n))
#define vec_atanh(a, out, n) vec_atanh_dispatch((a), (out), (n))
#define vec_cbrt(a, out, n) vec_cbrt_dispatch((a), (out), (n))
#define vec_erf(a, out, n) vec_erf_dispatch((a), (out), (n))
#define vec_erfc(a, out, n) vec_erfc_dispatch((a), (out), (n))
#define vec_sinpi(a, out, n) vec_sinpi_dispatch((a), (out), (n))
#define vec_cospi(a, out, n) vec_cospi_dispatch((a), (out), (n))
#define vec_tgamma(a, out, n) vec_tgamma_dispatch((a), (out), (n))
#define vec_lgamma(a, out, n) vec_lgamma_dispatch((a), (out), (n))
#define vec_rint(a, out, n) vec_rint_dispatch((a), (out), (n))
#define vec_copysign(a, b, out, n) vec_copysign_dispatch((a), (b), (out), (n))
#define vec_fdim(a, b, out, n) vec_fdim_dispatch((a), (b), (out), (n))
#define vec_fmax(a, b, out, n) vec_fmax_dispatch((a), (b), (out), (n))
#define vec_fmin(a, b, out, n) vec_fmin_dispatch((a), (b), (out), (n))
#define vec_fmod(a, b, out, n) vec_fmod_dispatch((a), (b), (out), (n))
#define vec_hypot(a, b, out, n) vec_hypot_dispatch((a), (b), (out), (n))
#define vec_ldexp(a, b, out, n) vec_ldexp_dispatch((a), (b), (out), (n))
#define vec_nextafter(a, b, out, n) vec_nextafter_dispatch((a), (b), (out), (n))
#define vec_remainder(a, b, out, n) vec_remainder_dispatch((a), (b), (out), (n))
#define vec_fma(a, b, c, out, n) vec_fma_dispatch((a), (b), (c), (out), (n))
#define vec_fma_unused(a, b, c, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i]; } while(0)
#define vec_negate(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)
#define vec_copy(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i]; } while(0)

#define vec_add_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_f32(a, b, out, n) vec_pow_f32_dispatch((a), (b), (out), (n))
#define vec_add_scalar_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_f32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = powf((a)[_i], (b)); } while(0)
#define vec_sqrt_f32(a, out, n) vec_sqrt_f32_dispatch((a), (out), (n))
#define vec_sin_f32(a, out, n) vec_sin_f32_dispatch((a), (out), (n))
#define vec_cos_f32(a, out, n) vec_cos_f32_dispatch((a), (out), (n))
#define vec_asin_f32(a, out, n) vec_asin_f32_dispatch((a), (out), (n))
#define vec_acos_f32(a, out, n) vec_acos_f32_dispatch((a), (out), (n))
#define vec_atan_f32(a, out, n) vec_atan_f32_dispatch((a), (out), (n))
#define vec_tan_f32(a, out, n) vec_tan_f32_dispatch((a), (out), (n))
#define vec_exp_f32(a, out, n) vec_exp_f32_dispatch((a), (out), (n))
#define vec_expm1_f32(a, out, n) vec_expm1_f32_dispatch((a), (out), (n))
#define vec_exp2_f32(a, out, n) vec_exp2_f32_dispatch((a), (out), (n))
#define vec_exp10_f32(a, out, n) vec_exp10_f32_dispatch((a), (out), (n))
#define vec_log_f32(a, out, n) vec_log_f32_dispatch((a), (out), (n))
#define vec_log10_f32(a, out, n) vec_log10_f32_dispatch((a), (out), (n))
#define vec_log1p_f32(a, out, n) vec_log1p_f32_dispatch((a), (out), (n))
#define vec_log2_f32(a, out, n) vec_log2_f32_dispatch((a), (out), (n))
#define vec_sinh_f32(a, out, n) vec_sinh_f32_dispatch((a), (out), (n))
#define vec_cosh_f32(a, out, n) vec_cosh_f32_dispatch((a), (out), (n))
#define vec_tanh_f32(a, out, n) vec_tanh_f32_dispatch((a), (out), (n))
#define vec_abs_f32(a, out, n) vec_abs_f32_dispatch((a), (out), (n))
#define vec_ceil_f32(a, out, n) vec_ceil_f32_dispatch((a), (out), (n))
#define vec_floor_f32(a, out, n) vec_floor_f32_dispatch((a), (out), (n))
#define vec_round_f32(a, out, n) vec_round_f32_dispatch((a), (out), (n))
#define vec_trunc_f32(a, out, n) vec_trunc_f32_dispatch((a), (out), (n))
#define vec_acosh_f32(a, out, n) vec_acosh_f32_dispatch((a), (out), (n))
#define vec_asinh_f32(a, out, n) vec_asinh_f32_dispatch((a), (out), (n))
#define vec_atanh_f32(a, out, n) vec_atanh_f32_dispatch((a), (out), (n))
#define vec_cbrt_f32(a, out, n) vec_cbrt_f32_dispatch((a), (out), (n))
#define vec_erf_f32(a, out, n) vec_erf_f32_dispatch((a), (out), (n))
#define vec_erfc_f32(a, out, n) vec_erfc_f32_dispatch((a), (out), (n))
#define vec_sinpi_f32(a, out, n) vec_sinpi_f32_dispatch((a), (out), (n))
#define vec_cospi_f32(a, out, n) vec_cospi_f32_dispatch((a), (out), (n))
#define vec_tgamma_f32(a, out, n) vec_tgamma_f32_dispatch((a), (out), (n))
#define vec_lgamma_f32(a, out, n) vec_lgamma_f32_dispatch((a), (out), (n))
#define vec_rint_f32(a, out, n) vec_rint_f32_dispatch((a), (out), (n))
#define vec_copysign_f32(a, b, out, n) vec_copysign_f32_dispatch((a), (b), (out), (n))
#define vec_fdim_f32(a, b, out, n) vec_fdim_f32_dispatch((a), (b), (out), (n))
#define vec_fmax_f32(a, b, out, n) vec_fmax_f32_dispatch((a), (b), (out), (n))
#define vec_fmin_f32(a, b, out, n) vec_fmin_f32_dispatch((a), (b), (out), (n))
#define vec_fmod_f32(a, b, out, n) vec_fmod_f32_dispatch((a), (b), (out), (n))
#define vec_hypot_f32(a, b, out, n) vec_hypot_f32_dispatch((a), (b), (out), (n))
#define vec_ldexp_f32(a, b, out, n) vec_ldexp_f32_dispatch((a), (b), (out), (n))
#define vec_nextafter_f32(a, b, out, n) vec_nextafter_f32_dispatch((a), (b), (out), (n))
#define vec_remainder_f32(a, b, out, n) vec_remainder_f32_dispatch((a), (b), (out), (n))
#define vec_fma_f32(a, b, c, out, n) vec_fma_f32_dispatch((a), (b), (c), (out), (n))
#define vec_negame_f32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_atan2(a, b, out, n) vec_atan2_dispatch((a), (b), (out), (n))
#define vec_atan2_f32(a, b, out, n) vec_atan2_f32_dispatch((a), (b), (out), (n))

#define vec_add_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int8_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int8_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int8_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int16_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int16_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int16_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int32_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int32_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int32_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int64_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_i64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int64_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_i64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (int64_t)sqrt((a)[_i]); } while(0)
#define vec_negame_i64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint8_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u8(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint8_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint8_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u8(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint16_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u16(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint16_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint16_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u16(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint32_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u32(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint32_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint32_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u32(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#define vec_add_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint64_t)pow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_u64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint64_t)pow((a)[_i], (b)); } while(0)
#define vec_sqrt_u64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (uint64_t)sqrt((a)[_i]); } while(0)
#define vec_negame_u64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)

#if defined(_MSC_VER) && !defined(__clang__)
#define vec_add_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c64((a)[_i], (b)[_i]); } while(0)
#define vec_sub_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sub_c64((a)[_i], (b)[_i]); } while(0)
#define vec_mul_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c64((a)[_i], (b)[_i]); } while(0)
#define vec_div_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = div_c64((a)[_i], (b)[_i]); } while(0)
#define vec_pow_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpowf((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c64((a)[_i], (b)); } while(0)
#define vec_mul_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c64((a)[_i], (b)); } while(0)
#define vec_pow_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpowf((a)[_i], (b)); } while(0)
#define vec_sqrt_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = csqrtf((a)[_i]); } while(0)
#define vec_negame_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = neg_c64((a)[_i]); } while(0)
#define vec_conj_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = conjf((a)[_i]); } while(0)
#define vec_imag_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimagf((a)[_i]); } while(0)
#define vec_real_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_crealf((a)[_i]); } while(0)
#define vec_conj_noop(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i]; } while(0)

#define vec_add_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c128((a)[_i], (b)[_i]); } while(0)
#define vec_sub_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = sub_c128((a)[_i], (b)[_i]); } while(0)
#define vec_mul_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c128((a)[_i], (b)[_i]); } while(0)
#define vec_div_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = div_c128((a)[_i], (b)[_i]); } while(0)
#define vec_pow_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = add_c128((a)[_i], (b)); } while(0)
#define vec_mul_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = mul_c128((a)[_i], (b)); } while(0)
#define vec_pow_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = cpow((a)[_i], (b)); } while(0)
#define vec_sqrt_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = csqrt((a)[_i]); } while(0)
#define vec_negame_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = neg_c128((a)[_i]); } while(0)
#define vec_conj_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = conj((a)[_i]); } while(0)
#define vec_imag_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimag((a)[_i]); } while(0)
#define vec_real_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_creal((a)[_i]); } while(0)
#else
#define vec_add_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpowf((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_c64(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpowf((a)[_i], (b)); } while(0)
#define vec_sqrt_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_csqrtf((a)[_i]); } while(0)
#define vec_negame_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)
#define vec_conj_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_conjf((a)[_i]); } while(0)
#define vec_imag_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimagf((a)[_i]); } while(0)
#define vec_real_c64(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_crealf((a)[_i]); } while(0)
#define vec_conj_noop(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i]; } while(0)

#define vec_add_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b)[_i]; } while(0)
#define vec_sub_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] - (b)[_i]; } while(0)
#define vec_mul_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b)[_i]; } while(0)
#define vec_div_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] / (b)[_i]; } while(0)
#define vec_pow_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpow((a)[_i], (b)[_i]); } while(0)
#define vec_add_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] + (b); } while(0)
#define vec_mul_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = (a)[_i] * (b); } while(0)
#define vec_pow_scalar_c128(a, b, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cpow((a)[_i], (b)); } while(0)
#define vec_sqrt_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_csqrt((a)[_i]); } while(0)
#define vec_negame_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = -(a)[_i]; } while(0)
#define vec_conj_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_conj((a)[_i]); } while(0)
#define vec_imag_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_cimag((a)[_i]); } while(0)
#define vec_real_c128(a, out, n) do { for (int _i = 0; _i < (n); _i++) (out)[_i] = me_creal((a)[_i]); } while(0)
#endif

/* Generate float32 evaluator */
DEFINE_ME_EVAL(f32, float,
               vec_add_f32, vec_sub_f32, vec_mul_f32, vec_div_f32, vec_pow_f32,
               vec_add_scalar_f32, vec_mul_scalar_f32, vec_pow_scalar_f32,
               vec_sqrt_f32, vec_sin_f32_cached, vec_cos_f32_cached, vec_tan_f32,
               vec_asin_f32, vec_acos_f32, vec_atan_f32,
               vec_exp_f32, vec_log_f32, vec_log10_f32, vec_log1p_f32, vec_log2_f32, vec_expm1_f32,
               vec_exp2_f32, vec_exp10_f32,
               vec_sinh_f32, vec_cosh_f32, vec_tanh_f32,
               vec_abs_f32, vec_ceil_f32, vec_floor_f32, vec_round_f32, vec_trunc_f32,
               vec_acosh_f32, vec_asinh_f32, vec_atanh_f32,
               vec_cbrt_f32, vec_erf_f32, vec_erfc_f32, vec_sinpi_f32, vec_cospi_f32,
               vec_tgamma_f32, vec_lgamma_f32, vec_rint_f32,
               vec_copysign_f32, vec_fdim_f32, vec_fmax_f32, vec_fmin_f32, vec_fmod_f32, vec_hypot_f32, vec_ldexp_f32,
               vec_nextafter_f32, vec_remainder_f32, vec_fma_f32,
               vec_negame_f32,
               sqrtf, sinf, cosf, tanf, asinf, acosf, atanf,
               expf, logf, log10f, log1pf, log2f, expm1f, exp2f, exp10_wrapper,
               sinhf, coshf, tanhf, acoshf, asinhf, atanhf, cbrtf, erff, erfcf,
               sinpi_wrapper, cospi_wrapper, tgammaf, lgammaf, rintf,
               copysignf, fdimf, fmaxf, fminf, fmodf, hypotf, ldexp_wrapper,
               nextafterf, remainderf, fmaf,
               ceilf, floorf, roundf, truncf, fabsf, powf,
               vec_copy, 1, 1, 1, 1, 1,
               vec_atan2_f32, atan2f, 1)

/* Generate float64 (double) evaluator */
DEFINE_ME_EVAL(f64, double,
               vec_add, vec_sub, vec_mul, vec_div, vec_pow,
               vec_add_scalar, vec_mul_scalar, vec_pow_scalar,
               vec_sqrt, vec_sin_cached, vec_cos_cached, vec_tan,
               vec_asin, vec_acos, vec_atan,
               vec_exp, vec_log, vec_log10, vec_log1p, vec_log2, vec_expm1,
               vec_exp2, vec_exp10,
               vec_sinh, vec_cosh, vec_tanh,
               vec_abs, vec_ceil, vec_floor, vec_round, vec_trunc,
               vec_acosh, vec_asinh, vec_atanh,
               vec_cbrt, vec_erf, vec_erfc, vec_sinpi, vec_cospi,
               vec_tgamma, vec_lgamma, vec_rint,
               vec_copysign, vec_fdim, vec_fmax, vec_fmin, vec_fmod, vec_hypot, vec_ldexp,
               vec_nextafter, vec_remainder, vec_fma,
               vec_negate,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_copy, 1, 1, 1, 1, 1,
               vec_atan2, atan2, 1)

/* Generate integer evaluators - sin/cos cast to double and back */
DEFINE_ME_EVAL(i8, int8_t,
               vec_add_i8, vec_sub_i8, vec_mul_i8, vec_div_i8, vec_pow_i8,
               vec_add_scalar_i8, vec_mul_scalar_i8, vec_pow_scalar_i8,
               vec_sqrt_i8, vec_sqrt_i8, vec_sqrt_i8, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_i8, vec_add_i8, vec_add_i8, vec_add_i8, vec_add_i8, vec_add_i8, vec_add_i8,
               vec_add_i8, vec_add_i8, vec_fma_unused,
               vec_negame_i8,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_i8, atan2, 0)

DEFINE_ME_EVAL(i16, int16_t,
               vec_add_i16, vec_sub_i16, vec_mul_i16, vec_div_i16, vec_pow_i16,
               vec_add_scalar_i16, vec_mul_scalar_i16, vec_pow_scalar_i16,
               vec_sqrt_i16, vec_sqrt_i16, vec_sqrt_i16, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_i16, vec_add_i16, vec_add_i16, vec_add_i16, vec_add_i16, vec_add_i16, vec_add_i16,
               vec_add_i16, vec_add_i16, vec_fma_unused,
               vec_negame_i16,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_i16, atan2, 0)

DEFINE_ME_EVAL(i32, int32_t,
               vec_add_i32, vec_sub_i32, vec_mul_i32, vec_div_i32, vec_pow_i32,
               vec_add_scalar_i32, vec_mul_scalar_i32, vec_pow_scalar_i32,
               vec_sqrt_i32, vec_sqrt_i32, vec_sqrt_i32, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_i32, vec_add_i32, vec_add_i32, vec_add_i32, vec_add_i32, vec_add_i32, vec_add_i32,
               vec_add_i32, vec_add_i32, vec_fma_unused,
               vec_negame_i32,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_i32, atan2, 0)

DEFINE_ME_EVAL(i64, int64_t,
               vec_add_i64, vec_sub_i64, vec_mul_i64, vec_div_i64, vec_pow_i64,
               vec_add_scalar_i64, vec_mul_scalar_i64, vec_pow_scalar_i64,
               vec_sqrt_i64, vec_sqrt_i64, vec_sqrt_i64, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_i64, vec_add_i64, vec_add_i64, vec_add_i64, vec_add_i64, vec_add_i64, vec_add_i64,
               vec_add_i64, vec_add_i64, vec_fma_unused,
               vec_negame_i64,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_i64, atan2, 0)

DEFINE_ME_EVAL(u8, uint8_t,
               vec_add_u8, vec_sub_u8, vec_mul_u8, vec_div_u8, vec_pow_u8,
               vec_add_scalar_u8, vec_mul_scalar_u8, vec_pow_scalar_u8,
               vec_sqrt_u8, vec_sqrt_u8, vec_sqrt_u8, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_u8, vec_add_u8, vec_add_u8, vec_add_u8, vec_add_u8, vec_add_u8, vec_add_u8,
               vec_add_u8, vec_add_u8, vec_fma_unused,
               vec_negame_u8,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_u8, atan2, 0)

DEFINE_ME_EVAL(u16, uint16_t,
               vec_add_u16, vec_sub_u16, vec_mul_u16, vec_div_u16, vec_pow_u16,
               vec_add_scalar_u16, vec_mul_scalar_u16, vec_pow_scalar_u16,
               vec_sqrt_u16, vec_sqrt_u16, vec_sqrt_u16, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_u16, vec_add_u16, vec_add_u16, vec_add_u16, vec_add_u16, vec_add_u16, vec_add_u16,
               vec_add_u16, vec_add_u16, vec_fma_unused,
               vec_negame_u16,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_u16, atan2, 0)

DEFINE_ME_EVAL(u32, uint32_t,
               vec_add_u32, vec_sub_u32, vec_mul_u32, vec_div_u32, vec_pow_u32,
               vec_add_scalar_u32, vec_mul_scalar_u32, vec_pow_scalar_u32,
               vec_sqrt_u32, vec_sqrt_u32, vec_sqrt_u32, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_u32, vec_add_u32, vec_add_u32, vec_add_u32, vec_add_u32, vec_add_u32, vec_add_u32,
               vec_add_u32, vec_add_u32, vec_fma_unused,
               vec_negame_u32,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_u32, atan2, 0)

DEFINE_ME_EVAL(u64, uint64_t,
               vec_add_u64, vec_sub_u64, vec_mul_u64, vec_div_u64, vec_pow_u64,
               vec_add_scalar_u64, vec_mul_scalar_u64, vec_pow_scalar_u64,
               vec_sqrt_u64, vec_sqrt_u64, vec_sqrt_u64, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_u64, vec_add_u64, vec_add_u64, vec_add_u64, vec_add_u64, vec_add_u64, vec_add_u64,
               vec_add_u64, vec_add_u64, vec_fma_unused,
               vec_negame_u64,
               sqrt, sin, cos, tan, asin, acos, atan,
               exp, log, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, fabs, pow,
               vec_conj_noop, 0, 0, 0, 0, 0,
               vec_add_u64, atan2, 0)

/* Generate complex evaluators */
DEFINE_ME_EVAL(c64, float _Complex,
               vec_add_c64, vec_sub_c64, vec_mul_c64, vec_div_c64, vec_pow_c64,
               vec_add_scalar_c64, vec_mul_scalar_c64, vec_pow_scalar_c64,
               vec_sqrt_c64, vec_sqrt_c64, vec_sqrt_c64, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_c64, vec_add_c64, vec_add_c64, vec_add_c64, vec_add_c64, vec_add_c64, vec_add_c64,
               vec_add_c64, vec_add_c64, vec_fma_unused,
               vec_negame_c64,
               me_csqrtf, me_csqrtf, me_csqrtf, tan, asin, acos, atan,
               me_cexpf, me_clogf, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, me_cabsf, me_cpowf,
               vec_conj_c64, 0, 0, 0, 0, 0,
               vec_add_c64, atan2, 0)

DEFINE_ME_EVAL(c128, double _Complex,
               vec_add_c128, vec_sub_c128, vec_mul_c128, vec_div_c128, vec_pow_c128,
               vec_add_scalar_c128, vec_mul_scalar_c128, vec_pow_scalar_c128,
               vec_sqrt_c128, vec_sqrt_c128, vec_sqrt_c128, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy, vec_copy,
               vec_copy, vec_copy, vec_copy,
               vec_add_c128, vec_add_c128, vec_add_c128, vec_add_c128, vec_add_c128, vec_add_c128, vec_add_c128,
               vec_add_c128, vec_add_c128, vec_fma_unused,
               vec_negame_c128,
               me_csqrt, me_csqrt, me_csqrt, tan, asin, acos, atan,
               me_cexp, me_clog, log10, log1p, log2, expm1, exp2, exp10_wrapper,
               sinh, cosh, tanh, acosh, asinh, atanh, cbrt, erf, erfc,
               sinpi_wrapper, cospi_wrapper, tgamma, lgamma, rint,
               copysign, fdim, fmax, fmin, fmod, hypot, ldexp_wrapper,
               nextafter, remainder, fma,
               ceil, floor, round, trunc, me_cabs, me_cpow,
               vec_conj_c128, 0, 0, 0, 0, 0,
               vec_add_c128, atan2, 0)

/* Public API - dispatches to correct type-specific evaluator */
/* Structure to track promoted variables */
typedef struct {
    void* promoted_data; // Temporary buffer for promoted data
    me_dtype original_type;
    bool needs_free;
} promoted_var_t;

/* Helper to save original variable bindings */
static void save_variable_bindings(const me_expr* node,
                                   const void** original_bounds,
                                   me_dtype* original_types,
                                   int* save_idx) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        original_bounds[*save_idx] = node->bound;
        original_types[*save_idx] = node->dtype;
        (*save_idx)++;
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            // Skip conversion nodes - they handle their own type conversion
            if (IS_FUNCTION(node->type) && ARITY(node->type) == 1 && node->function == NULL) {
                break;
            }
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_variable_bindings((const me_expr*)node->parameters[i],
                                       original_bounds, original_types, save_idx);
            }
            break;
        }
    }
}

/* Recursively promote variables in expression tree */
static void promote_variables_in_tree(me_expr* n, me_dtype target_type,
                                      promoted_var_t* promotions, int* promo_count,
                                      int nitems) {
    if (!n) return;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        // Constants are promoted on-the-fly during evaluation
        break;

    case ME_VARIABLE:
        if (n->dtype != target_type) {
            // Need to promote this variable
            void* promoted = malloc(nitems * dtype_size(target_type));
            if (promoted) {
                convert_func_t conv = get_convert_func(n->dtype, target_type);
                if (conv) {
                    conv(n->bound, promoted, nitems);

                    // Track this promotion for later cleanup
                    promotions[*promo_count].promoted_data = promoted;
                    promotions[*promo_count].original_type = n->dtype;
                    promotions[*promo_count].needs_free = true;
                    (*promo_count)++;

                    // Temporarily replace bound pointer
                    n->bound = promoted;
                    n->dtype = target_type;
                }
                else {
                    free(promoted);
                }
            }
        }
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            // Skip conversion nodes - they handle their own type conversion
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && n->function == NULL) {
                break;
            }
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                promote_variables_in_tree((me_expr*)n->parameters[i], target_type,
                                          promotions, promo_count, nitems);
            }
            break;
        }
    }
}

/* Restore original variable bindings after promotion */
static void restore_variables_in_tree(me_expr* n, const void** original_bounds,
                                      const me_dtype* original_types, int* restore_idx) {
    if (!n) return;

    switch (TYPE_MASK(n->type)) {
    case ME_VARIABLE:
        if (original_bounds[*restore_idx] != NULL) {
            n->bound = original_bounds[*restore_idx];
            n->dtype = original_types[*restore_idx];
            (*restore_idx)++;
        }
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            // Skip conversion nodes - they handle their own type conversion
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && n->function == NULL) {
                break;
            }
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                restore_variables_in_tree((me_expr*)n->parameters[i], original_bounds, original_types, restore_idx);
            }
            break;
        }
    }
}

/* Check if all variables in tree match target type */
static bool all_variables_match_type(const me_expr* n, me_dtype target_type) {
    if (!n) return true;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return true; // Constants are always OK

    case ME_VARIABLE:
        return n->dtype == target_type;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            // Skip conversion nodes - they handle their own type conversion
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && n->function == NULL) {
                return true;
            }
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!all_variables_match_type((const me_expr*)n->parameters[i], target_type)) {
                    return false;
                }
            }
            return true;
        }
    }

    return true;
}

static void broadcast_reduction_output(void* output, me_dtype dtype, int output_nitems) {
    if (!output || output_nitems <= 1) return;
    switch (dtype) {
    case ME_BOOL:
        {
            bool val = ((bool*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((bool*)output)[i] = val;
            }
            break;
        }
    case ME_INT8:
        {
            int8_t val = ((int8_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int8_t*)output)[i] = val;
            }
            break;
        }
    case ME_INT16:
        {
            int16_t val = ((int16_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int16_t*)output)[i] = val;
            }
            break;
        }
    case ME_INT32:
        {
            int32_t val = ((int32_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int32_t*)output)[i] = val;
            }
            break;
        }
    case ME_INT64:
        {
            int64_t val = ((int64_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((int64_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT8:
        {
            uint8_t val = ((uint8_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint8_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT16:
        {
            uint16_t val = ((uint16_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint16_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT32:
        {
            uint32_t val = ((uint32_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint32_t*)output)[i] = val;
            }
            break;
        }
    case ME_UINT64:
        {
            uint64_t val = ((uint64_t*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((uint64_t*)output)[i] = val;
            }
            break;
        }
    case ME_FLOAT32:
        {
            float val = ((float*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((float*)output)[i] = val;
            }
            break;
        }
    case ME_FLOAT64:
        {
            double val = ((double*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((double*)output)[i] = val;
            }
            break;
        }
    case ME_COMPLEX64:
        {
            float _Complex val = ((float _Complex*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((float _Complex*)output)[i] = val;
            }
            break;
        }
    case ME_COMPLEX128:
        {
            double _Complex val = ((double _Complex*)output)[0];
            for (int i = 1; i < output_nitems; i++) {
                ((double _Complex*)output)[i] = val;
            }
            break;
        }
    default:
        break;
    }
}

static void eval_reduction(const me_expr* n, int output_nitems) {
    if (!n || !n->output || !is_reduction_node(n)) return;
    if (output_nitems <= 0) return;

    me_expr* arg = (me_expr*)n->parameters[0];
    if (!arg) return;

    const int nitems = n->nitems;
    me_dtype arg_type = arg->dtype;
    if (arg->type != ME_CONSTANT && arg->type != ME_VARIABLE) {
        arg_type = infer_output_type(arg);
        const bool is_comparison_bool = (arg_type == ME_BOOL) && is_comparison_node(arg);
        me_dtype eval_type = is_comparison_bool ? infer_result_type(arg) : arg_type;
        if (nitems > 0) {
            if (is_comparison_bool) {
                void* eval_output = malloc((size_t)nitems * dtype_size(eval_type));
                if (!eval_output) return;
                arg->output = eval_output;
                arg->nitems = nitems;
                arg->dtype = eval_type;
                private_eval(arg);

                void* bool_output = malloc((size_t)nitems * sizeof(bool));
                if (!bool_output) {
                    free(eval_output);
                    return;
                }
                convert_func_t conv = get_convert_func(eval_type, ME_BOOL);
                if (!conv) {
                    arg->output = NULL;
                    arg->dtype = arg_type;
                    free(eval_output);
                    free(bool_output);
                    return;
                }
                conv(eval_output, bool_output, nitems);
                free(eval_output);

                arg->output = bool_output;
                arg->dtype = ME_BOOL;
            }
            else {
                if (!arg->output) {
                    arg->output = malloc((size_t)nitems * dtype_size(arg_type));
                    if (!arg->output) return;
                }
                arg->nitems = nitems;
                arg->dtype = arg_type;
                private_eval(arg);
            }
        }
    }
    me_dtype result_type = reduction_output_dtype(arg_type, n->function);
    me_dtype output_type = n->dtype;
    bool is_prod = n->function == (void*)prod_reduce;
    bool is_mean = n->function == (void*)mean_reduce;
    bool is_min = n->function == (void*)min_reduce;
    bool is_max = n->function == (void*)max_reduce;
    bool is_any = n->function == (void*)any_reduce;
    bool is_all = n->function == (void*)all_reduce;

    void* write_ptr = n->output;
    void* temp_output = NULL;
    if (output_type != result_type) {
        temp_output = malloc((size_t)output_nitems * dtype_size(result_type));
        if (!temp_output) return;
        write_ptr = temp_output;
    }

    if (arg->type == ME_CONSTANT) {
        double val = arg->value;
        if (is_mean) {
            if (result_type == ME_COMPLEX128) {
                double _Complex acc = (nitems == 0) ? me_cmplx(NAN, NAN) :
                    (double _Complex)val;
                ((double _Complex*)write_ptr)[0] = acc;
            }
            else {
                double acc = (nitems == 0) ? NAN : val;
                ((double*)write_ptr)[0] = acc;
            }
        }
        else if (is_any || is_all) {
            bool acc = is_all;
            if (nitems == 0) {
                acc = is_all;
            }
            else {
                switch (arg_type) {
                case ME_BOOL:
                    acc = val != 0.0;
                    break;
                case ME_INT8:
                case ME_INT16:
                case ME_INT32:
                case ME_INT64:
                case ME_UINT8:
                case ME_UINT16:
                case ME_UINT32:
                case ME_UINT64:
                case ME_FLOAT32:
                case ME_FLOAT64:
                    acc = val != 0.0;
                    break;
                case ME_COMPLEX64:
                case ME_COMPLEX128:
                    acc = val != 0.0;
                    break;
                default:
                    acc = false;
                    break;
                }
            }
            ((bool*)write_ptr)[0] = acc;
        }
        else if (is_min || is_max) {
            switch (arg_type) {
            case ME_BOOL:
                {
                    bool acc = is_min;
                    if (nitems > 0) {
                        acc = (bool)val;
                    }
                    ((bool*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT8:
                {
                    int8_t acc = (int8_t)(is_min ? INT8_MAX : INT8_MIN);
                    if (nitems > 0) acc = (int8_t)val;
                    ((int8_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT16:
                {
                    int16_t acc = (int16_t)(is_min ? INT16_MAX : INT16_MIN);
                    if (nitems > 0) acc = (int16_t)val;
                    ((int16_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT32:
                {
                    int32_t acc = (int32_t)(is_min ? INT32_MAX : INT32_MIN);
                    if (nitems > 0) acc = (int32_t)val;
                    ((int32_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_INT64:
                {
                    int64_t acc = is_min ? INT64_MAX : INT64_MIN;
                    if (nitems > 0) acc = (int64_t)val;
                    ((int64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT8:
                {
                    uint8_t acc = is_min ? UINT8_MAX : 0;
                    if (nitems > 0) acc = (uint8_t)val;
                    ((uint8_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT16:
                {
                    uint16_t acc = is_min ? UINT16_MAX : 0;
                    if (nitems > 0) acc = (uint16_t)val;
                    ((uint16_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT32:
                {
                    uint32_t acc = is_min ? UINT32_MAX : 0;
                    if (nitems > 0) acc = (uint32_t)val;
                    ((uint32_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT64:
                {
                    uint64_t acc = is_min ? UINT64_MAX : 0;
                    if (nitems > 0) acc = (uint64_t)val;
                    ((uint64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT32:
                {
                    float acc = is_min ? INFINITY : -INFINITY;
                    if (nitems > 0) acc = (float)val;
                    ((float*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT64:
                {
                    double acc = is_min ? INFINITY : -INFINITY;
                    if (nitems > 0) acc = val;
                    ((double*)write_ptr)[0] = acc;
                    break;
                }
            case ME_COMPLEX64:
                {
                    ((float _Complex*)write_ptr)[0] = (float _Complex)0.0f;
                    break;
                }
            case ME_COMPLEX128:
                {
                    ((double _Complex*)write_ptr)[0] = (double _Complex)0.0;
                    break;
                }
            default:
                break;
            }
        }
        else {
            switch (arg_type) {
            case ME_BOOL:
            case ME_INT8:
            case ME_INT16:
            case ME_INT32:
            case ME_INT64:
                {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        int64_t v = (int64_t)val;
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = (int64_t)val * (int64_t)nitems;
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_UINT8:
            case ME_UINT16:
            case ME_UINT32:
            case ME_UINT64:
                {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        uint64_t v = (uint64_t)val;
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = (uint64_t)val * (uint64_t)nitems;
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT32:
                {
                    float acc = is_prod ? 1.0f : 0.0f;
                    if (nitems == 0) {
                        acc = is_prod ? 1.0f : 0.0f;
                    }
                    else if (is_prod) {
                        float v = (float)val;
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = (float)val * (float)nitems;
                    }
                    ((float*)write_ptr)[0] = acc;
                    break;
                }
            case ME_FLOAT64:
                {
                    double acc = is_prod ? 1.0 : 0.0;
                    if (nitems == 0) {
                        acc = is_prod ? 1.0 : 0.0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= val;
                    }
                    else {
                        acc = val * (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                    break;
                }
            case ME_COMPLEX64:
                {
                    float _Complex acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                    float _Complex v = (float _Complex)val;
                    if (nitems == 0) {
                        acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = v * (float)nitems;
                    }
                    ((float _Complex*)write_ptr)[0] = acc;
                    break;
                }
            case ME_COMPLEX128:
                {
                    double _Complex acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                    double _Complex v = (double _Complex)val;
                    if (nitems == 0) {
                        acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= v;
                    }
                    else {
                        acc = v * (double)nitems;
                    }
                    ((double _Complex*)write_ptr)[0] = acc;
                    break;
                }
            default:
                break;
            }
        }
    }
    else {
        const void* saved_bound = arg->bound;
        int saved_type = arg->type;
        if (arg->type != ME_VARIABLE) {
            ((me_expr*)arg)->bound = arg->output;
            ((me_expr*)arg)->type = ME_VARIABLE;
        }
        switch (arg_type) {
        case ME_BOOL:
            {
                const bool* data = (const bool*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        int64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i] ? 1 : 0;
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i]) { acc = true; break; }
                            }
                            else {
                                if (!data[i]) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    bool acc = is_min;
                    if (nitems > 0) {
                        acc = data[0];
                        for (int i = 1; i < nitems; i++) {
                            acc = is_min ? (acc && data[i]) : (acc || data[i]);
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i] ? 1 : 0;
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i] ? 1 : 0;
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT8:
            {
                const int8_t* data = (const int8_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        int64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int8_t acc = is_min ? reduce_min_int8(data, nitems) :
                        reduce_max_int8(data, nitems);
                    ((int8_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT16:
            {
                const int16_t* data = (const int16_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        int64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int16_t acc = is_min ? reduce_min_int16(data, nitems) :
                        reduce_max_int16(data, nitems);
                    ((int16_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT32:
            {
                const int32_t* data = (const int32_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        int64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int32_t acc = is_min ? reduce_min_int32(data, nitems) :
                        reduce_max_int32(data, nitems);
                    ((int32_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        acc = reduce_sum_int32(data, nitems);
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_INT64:
            {
                const int64_t* data = (const int64_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        int64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    int64_t acc = is_min ? reduce_min_int64(data, nitems) :
                        reduce_max_int64(data, nitems);
                    ((int64_t*)write_ptr)[0] = acc;
                }
                else {
                    int64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((int64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT8:
            {
                const uint8_t* data = (const uint8_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        uint64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint8_t acc = is_min ? reduce_min_uint8(data, nitems) :
                        reduce_max_uint8(data, nitems);
                    ((uint8_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT16:
            {
                const uint16_t* data = (const uint16_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        uint64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint16_t acc = is_min ? reduce_min_uint16(data, nitems) :
                        reduce_max_uint16(data, nitems);
                    ((uint16_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT32:
            {
                const uint32_t* data = (const uint32_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        uint64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint32_t acc = is_min ? reduce_min_uint32(data, nitems) :
                        reduce_max_uint32(data, nitems);
                    ((uint32_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        acc = reduce_sum_uint32(data, nitems);
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_UINT64:
            {
                const uint64_t* data = (const uint64_t*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        uint64_t sum = 0;
                        for (int i = 0; i < nitems; i++) {
                            sum += data[i];
                        }
                        acc = (double)sum / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else if (is_min || is_max) {
                    uint64_t acc = is_min ? reduce_min_uint64(data, nitems) :
                        reduce_max_uint64(data, nitems);
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                else {
                    uint64_t acc = is_prod ? 1 : 0;
                    if (nitems == 0) {
                        acc = is_prod ? 1 : 0;
                    }
                    else if (is_prod) {
                        for (int i = 0; i < nitems; i++) acc *= data[i];
                    }
                    else {
                        for (int i = 0; i < nitems; i++) acc += data[i];
                    }
                    ((uint64_t*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_FLOAT32:
            {
                const float* data = (const float*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        acc = reduce_sum_float32_nan_safe(data, nitems) / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0.0f) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0.0f) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else {
                    if (nitems == 0) {
                        float acc = 0.0f;
                        if (is_min) acc = INFINITY;
                        else if (is_max) acc = -INFINITY;
                        else acc = is_prod ? 1.0f : 0.0f;
                        ((float*)write_ptr)[0] = acc;
                    }
                    else if (is_min) {
                        float acc = reduce_min_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = acc;
                    }
                    else if (is_max) {
                        float acc = reduce_max_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = acc;
                    }
                    else if (is_prod) {
                        /* Accumulate float32 sum/prod in float64 for better precision. */
                        double acc = reduce_prod_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = (float)acc;
                    }
                    else {
                        double acc = reduce_sum_float32_nan_safe(data, nitems);
                        ((float*)write_ptr)[0] = (float)acc;
                    }
                }
                break;
            }
        case ME_FLOAT64:
            {
                const double* data = (const double*)arg->bound;
                if (is_mean) {
                    double acc = NAN;
                    if (nitems > 0) {
                        acc = reduce_sum_float64_nan_safe(data, nitems) / (double)nitems;
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                else if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            if (is_any) {
                                if (data[i] != 0.0) { acc = true; break; }
                            }
                            else {
                                if (data[i] == 0.0) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                }
                else {
                    double acc = 0.0;
                    if (nitems == 0) {
                        if (is_min) acc = INFINITY;
                        else if (is_max) acc = -INFINITY;
                        else acc = is_prod ? 1.0 : 0.0;
                    }
                    else if (is_min) {
                        acc = reduce_min_float64_nan_safe(data, nitems);
                    }
                    else if (is_max) {
                        acc = reduce_max_float64_nan_safe(data, nitems);
                    }
                    else if (is_prod) {
                        acc = reduce_prod_float64_nan_safe(data, nitems);
                    }
                    else {
                        acc = reduce_sum_float64_nan_safe(data, nitems);
                    }
                    ((double*)write_ptr)[0] = acc;
                }
                break;
            }
        case ME_COMPLEX64:
            {
                const float _Complex* data = (const float _Complex*)arg->bound;
                if (is_mean) {
                    double _Complex acc = me_cmplx(NAN, NAN);
                    if (nitems > 0) {
                        acc = (double _Complex)0.0;
                        for (int i = 0; i < nitems; i++) {
                            acc += (double _Complex)data[i];
                        }
                        acc /= (double)nitems;
                    }
                    ((double _Complex*)write_ptr)[0] = acc;
                    break;
                }
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            bool nonzero = IS_NONZERO_c64(data[i]);
                            if (is_any) {
                                if (nonzero) { acc = true; break; }
                            }
                            else {
                                if (!nonzero) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                    break;
                }
                if (is_min || is_max) {
                    ((float _Complex*)write_ptr)[0] = (float _Complex)0.0f;
                    break;
                }
                float _Complex acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                if (nitems == 0) {
                    acc = is_prod ? (float _Complex)1.0f : (float _Complex)0.0f;
                }
                else if (is_prod) {
                    for (int i = 0; i < nitems; i++) acc *= data[i];
                }
                else {
                    for (int i = 0; i < nitems; i++) acc += data[i];
                }
                ((float _Complex*)write_ptr)[0] = acc;
                break;
            }
        case ME_COMPLEX128:
            {
                const double _Complex* data = (const double _Complex*)arg->bound;
                if (is_mean) {
                    double _Complex acc = me_cmplx(NAN, NAN);
                    if (nitems > 0) {
                        acc = (double _Complex)0.0;
                        for (int i = 0; i < nitems; i++) acc += data[i];
                        acc /= (double)nitems;
                    }
                    ((double _Complex*)write_ptr)[0] = acc;
                    break;
                }
                if (is_any || is_all) {
                    bool acc = is_all;
                    if (nitems > 0) {
                        acc = is_all;
                        for (int i = 0; i < nitems; i++) {
                            bool nonzero = IS_NONZERO_c128(data[i]);
                            if (is_any) {
                                if (nonzero) { acc = true; break; }
                            }
                            else {
                                if (!nonzero) { acc = false; break; }
                            }
                        }
                    }
                    ((bool*)write_ptr)[0] = acc;
                    break;
                }
                if (is_min || is_max) {
                    ((double _Complex*)write_ptr)[0] = (double _Complex)0.0;
                    break;
                }
                double _Complex acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                if (nitems == 0) {
                    acc = is_prod ? (double _Complex)1.0 : (double _Complex)0.0;
                }
                else if (is_prod) {
                    for (int i = 0; i < nitems; i++) acc *= data[i];
                }
                else {
                    for (int i = 0; i < nitems; i++) acc += data[i];
                }
                ((double _Complex*)write_ptr)[0] = acc;
                break;
            }
        default:
            break;
        }
        if (saved_type != ME_VARIABLE) {
            ((me_expr*)arg)->bound = saved_bound;
            ((me_expr*)arg)->type = saved_type;
        }
    }

    {
        me_dtype write_type = temp_output ? result_type : output_type;
        broadcast_reduction_output(write_ptr, write_type, output_nitems);
    }

    if (temp_output) {
        convert_func_t conv = get_convert_func(result_type, output_type);
        if (conv) {
            conv(temp_output, n->output, output_nitems);
        }
        free(temp_output);
    }
}

static void private_eval(const me_expr* n) {
    if (!n) return;

    if (is_reduction_node(n)) {
        eval_reduction(n, 1);
        return;
    }

    // Special case: imag(), real(), abs() return real from complex input
    if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
        if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper ||
            n->function == (void*)fabs) {
            me_expr* arg = (me_expr*)n->parameters[0];
            me_dtype arg_type = infer_result_type(arg);

            if (n->function == (void*)fabs && arg_type == ME_COMPLEX64) {
                if (!arg->output) {
                    arg->output = malloc(n->nitems * sizeof(float _Complex));
                    arg->nitems = n->nitems;
                    ((me_expr*)arg)->dtype = ME_COMPLEX64;
                }
                me_eval_c64(arg);

                const float _Complex* cdata = (const float _Complex*)arg->output;
                float* output = (float*)n->output;
                for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                    float re = __builtin_crealf(cdata[i]);
                    float im = __builtin_cimagf(cdata[i]);
                    output[i] = hypotf(re, im);
#else
                    output[i] = cabsf(cdata[i]);
#endif
                }
                return;
            }
            else if (n->function == (void*)fabs && arg_type == ME_COMPLEX128) {
                if (!arg->output) {
                    arg->output = malloc(n->nitems * sizeof(double _Complex));
                    arg->nitems = n->nitems;
                    ((me_expr*)arg)->dtype = ME_COMPLEX128;
                }
                me_eval_c128(arg);

                const double _Complex* cdata = (const double _Complex*)arg->output;
                double* output = (double*)n->output;
                for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                    double re = __builtin_creal(cdata[i]);
                    double im = __builtin_cimag(cdata[i]);
                    output[i] = hypot(re, im);
#else
                    output[i] = cabs(cdata[i]);
#endif
                }
                return;
            }

            if (arg_type == ME_COMPLEX64) {
                // Evaluate argument as complex64
                if (!arg->output) {
                    arg->output = malloc(n->nitems * sizeof(float _Complex));
                    arg->nitems = n->nitems;
                    ((me_expr*)arg)->dtype = ME_COMPLEX64;
                }
                me_eval_c64(arg);

                // Extract real/imaginary part to float32 output
                const float _Complex* cdata = (const float _Complex*)arg->output;
                float* output = (float*)n->output;
                if (n->function == (void*)imag_wrapper) {
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_cimagf(cdata[i]);
#else
                        output[i] = cimagf(cdata[i]);
#endif
                    }
                }
                else { // real_wrapper
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_crealf(cdata[i]);
#else
                        output[i] = crealf(cdata[i]);
#endif
                    }
                }
                return;
            }
            else if (arg_type == ME_COMPLEX128) {
                // Evaluate argument as complex128
                if (!arg->output) {
                    arg->output = malloc(n->nitems * sizeof(double _Complex));
                    arg->nitems = n->nitems;
                    ((me_expr*)arg)->dtype = ME_COMPLEX128;
                }
                me_eval_c128(arg);

                // Extract real/imaginary part to float64 output
                const double _Complex* cdata = (const double _Complex*)arg->output;
                double* output = (double*)n->output;
                if (n->function == (void*)imag_wrapper) {
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_cimag(cdata[i]);
#else
                        output[i] = cimag(cdata[i]);
#endif
                    }
                }
                else { // real_wrapper
                    for (int i = 0; i < n->nitems; i++) {
#if defined(_MSC_VER) && defined(__clang__)
                        output[i] = __builtin_creal(cdata[i]);
#else
                        output[i] = creal(cdata[i]);
#endif
                    }
                }
                return;
            }
            // If not complex, fall through to normal evaluation
        }
    }

    // Infer the result type from the expression tree
    me_dtype result_type = infer_result_type(n);
    const bool has_string = contains_string_node(n);

    // Fast path for boolean expressions: compute directly into bool output
    if (n->dtype == ME_BOOL && infer_output_type(n) == ME_BOOL) {
        if (!has_string) {
            promoted_var_t promotions[ME_MAX_VARS];
            int promo_count = 0;

            const void* original_bounds[ME_MAX_VARS];
            me_dtype original_types[ME_MAX_VARS];
            int save_idx = 0;

            save_variable_bindings(n, original_bounds, original_types, &save_idx);
            promote_variables_in_tree((me_expr*)n, result_type, promotions, &promo_count, n->nitems);

            if (!eval_bool_expr((me_expr*)n)) {
                int restore_idx = 0;
                restore_variables_in_tree((me_expr*)n, original_bounds, original_types, &restore_idx);
                for (int i = 0; i < promo_count; i++) {
                    if (promotions[i].needs_free) {
                        free(promotions[i].promoted_data);
                    }
                }
                goto fallback_eval;
            }

            int restore_idx = 0;
            restore_variables_in_tree((me_expr*)n, original_bounds, original_types, &restore_idx);
            for (int i = 0; i < promo_count; i++) {
                if (promotions[i].needs_free) {
                    free(promotions[i].promoted_data);
                }
            }
            return;
        }

        if (eval_bool_expr((me_expr*)n)) {
            return;
        }
        goto fallback_eval;
    }

    if (has_string) {
        switch (n->dtype) {
        case ME_BOOL: me_eval_i8(n);
            return;
        case ME_INT8: me_eval_i8(n);
            return;
        case ME_INT16: me_eval_i16(n);
            return;
        case ME_INT32: me_eval_i32(n);
            return;
        case ME_INT64: me_eval_i64(n);
            return;
        case ME_UINT8: me_eval_u8(n);
            return;
        case ME_UINT16: me_eval_u16(n);
            return;
        case ME_UINT32: me_eval_u32(n);
            return;
        case ME_UINT64: me_eval_u64(n);
            return;
        case ME_FLOAT32: me_eval_f32(n);
            return;
        case ME_FLOAT64: me_eval_f64(n);
            return;
        case ME_COMPLEX64: me_eval_c64(n);
            return;
        case ME_COMPLEX128: me_eval_c128(n);
            return;
        default:
            return;
        }
    }

fallback_eval:
    ;
    // If all variables already match result type, use fast path
    bool all_match = all_variables_match_type(n, result_type);
    if (result_type == n->dtype && all_match) {
        // Fast path: no promotion needed
        if (n->dtype == ME_AUTO) {
            fprintf(stderr, "FATAL: ME_AUTO dtype in evaluation. This is a bug.\n");
#ifdef NDEBUG
            abort(); // Release build: terminate immediately
#else
            assert(0 && "ME_AUTO should be resolved during compilation"); // Debug: trigger debugger
#endif
        }
        switch (n->dtype) {
        case ME_BOOL: me_eval_i8(n);
            break;
        case ME_INT8: me_eval_i8(n);
            break;
        case ME_INT16: me_eval_i16(n);
            break;
        case ME_INT32: me_eval_i32(n);
            break;
        case ME_INT64: me_eval_i64(n);
            break;
        case ME_UINT8: me_eval_u8(n);
            break;
        case ME_UINT16: me_eval_u16(n);
            break;
        case ME_UINT32: me_eval_u32(n);
            break;
        case ME_UINT64: me_eval_u64(n);
            break;
        case ME_FLOAT32: me_eval_f32(n);
            break;
        case ME_FLOAT64: me_eval_f64(n);
            break;
        case ME_COMPLEX64: me_eval_c64(n);
            break;
        case ME_COMPLEX128: me_eval_c128(n);
            break;
        default:
            fprintf(stderr, "FATAL: Invalid dtype %d in evaluation.\n", n->dtype);
#ifdef NDEBUG
            abort(); // Release build: terminate immediately
#else
            assert(0 && "Invalid dtype"); // Debug: trigger debugger
#endif
        }
        return;
    }

    // Slow path: need to promote variables
    // Allocate tracking structures (max ME_MAX_VARS variables)
    promoted_var_t promotions[ME_MAX_VARS];
    int promo_count = 0;

    // Save original variable bindings
    const void* original_bounds[ME_MAX_VARS];
    me_dtype original_types[ME_MAX_VARS];
    int save_idx = 0;

    save_variable_bindings(n, original_bounds, original_types, &save_idx);

    // Promote variables
    promote_variables_in_tree((me_expr*)n, result_type, promotions, &promo_count, n->nitems);

    // Check if we need output type conversion (e.g., computation in float64, output in bool)
    me_dtype saved_dtype = n->dtype;
    void* original_output = n->output;
    void* temp_output = NULL;

    if (saved_dtype != result_type) {
        // Allocate temp buffer for computation
        temp_output = malloc(n->nitems * dtype_size(result_type));
        if (temp_output) {
            ((me_expr*)n)->output = temp_output;
        }
    }

    // Update expression type for evaluation
    ((me_expr*)n)->dtype = result_type;

    // Evaluate with promoted types
    if (result_type == ME_AUTO) {
        fprintf(stderr, "FATAL: ME_AUTO result type in evaluation. This is a bug.\n");
#ifdef NDEBUG
        abort(); // Release build: terminate immediately
#else
        assert(0 && "ME_AUTO should be resolved during compilation"); // Debug: trigger debugger
#endif
    }
    switch (result_type) {
    case ME_BOOL: me_eval_i8(n);
        break;
    case ME_INT8: me_eval_i8(n);
        break;
    case ME_INT16: me_eval_i16(n);
        break;
    case ME_INT32: me_eval_i32(n);
        break;
    case ME_INT64: me_eval_i64(n);
        break;
    case ME_UINT8: me_eval_u8(n);
        break;
    case ME_UINT16: me_eval_u16(n);
        break;
    case ME_UINT32: me_eval_u32(n);
        break;
    case ME_UINT64: me_eval_u64(n);
        break;
    case ME_FLOAT32: me_eval_f32(n);
        break;
    case ME_FLOAT64: me_eval_f64(n);
        break;
    case ME_COMPLEX64: me_eval_c64(n);
        break;
    case ME_COMPLEX128: me_eval_c128(n);
        break;
    default:
        fprintf(stderr, "FATAL: Invalid result type %d in evaluation.\n", result_type);
#ifdef NDEBUG
        abort(); // Release build: terminate immediately
#else
        assert(0 && "Invalid dtype"); // Debug: trigger debugger
#endif
    }

    // If we used a temp buffer, convert to final output type
    if (temp_output) {
        convert_func_t conv = get_convert_func(result_type, saved_dtype);
        if (conv) {
            conv(temp_output, original_output, n->nitems);
        }
        // Restore original output pointer
        ((me_expr*)n)->output = original_output;
        free(temp_output);
    }

    // Restore original variable bindings
    int restore_idx = 0;
    restore_variables_in_tree((me_expr*)n, original_bounds, original_types, &restore_idx);

    // Restore expression type
    ((me_expr*)n)->dtype = saved_dtype;

    // Free promoted buffers
    for (int i = 0; i < promo_count; i++) {
        if (promotions[i].needs_free) {
            free(promotions[i].promoted_data);
        }
    }
}

/* Helper to update variable bindings and nitems in tree */
static void save_nitems_in_tree(const me_expr* node, int* nitems_array, int* idx) {
    if (!node) return;
    nitems_array[(*idx)++] = node->nitems;

    switch (TYPE_MASK(node->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_nitems_in_tree((const me_expr*)node->parameters[i], nitems_array, idx);
            }
            break;
        }
    default:
        break;
    }
}

static void restore_nitems_in_tree(me_expr* node, const int* nitems_array, int* idx) {
    if (!node) return;
    node->nitems = nitems_array[(*idx)++];

    switch (TYPE_MASK(node->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                restore_nitems_in_tree((me_expr*)node->parameters[i], nitems_array, idx);
            }
            break;
        }
    default:
        break;
    }
}

/* Helper to free intermediate output buffers */
static void free_intermediate_buffers(me_expr* node) {
    if (!node) return;

    switch (TYPE_MASK(node->type)) {
    case ME_CONSTANT:
    case ME_VARIABLE:
        // These don't have intermediate buffers
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                me_expr* param = (me_expr*)node->parameters[i];
                free_intermediate_buffers(param);

                // Free intermediate buffer (but not for root or variables/constants)
                if (param->type != ME_CONSTANT && param->type != ME_VARIABLE && param->output) {
                    free(param->output);
                    param->output = NULL;
                }
            }
            break;
        }
    }
}

/* Helper to save original variable bindings with their pointers */
static void save_variable_metadata(const me_expr* node, const void** var_pointers, size_t* var_sizes, int* var_count) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        // Check if this pointer is already in the list
        for (int i = 0; i < *var_count; i++) {
            if (var_pointers[i] == node->bound) return; // Already saved
        }
        var_pointers[*var_count] = node->bound;
        if (node->dtype == ME_STRING && node->itemsize > 0) {
            var_sizes[*var_count] = node->itemsize;
        }
        else {
            var_sizes[*var_count] = dtype_size(node->input_dtype);
        }
        (*var_count)++;
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                save_variable_metadata((const me_expr*)node->parameters[i], var_pointers, var_sizes, var_count);
            }
            break;
        }
    }
}

static int count_variable_nodes(const me_expr* node) {
    if (!node) return 0;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        return 1;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            int count = 0;
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                count += count_variable_nodes((const me_expr*)node->parameters[i]);
            }
            return count;
        }
    }
    return 0;
}

static void collect_variable_nodes(me_expr* node, const void** var_pointers, int n_vars,
                                   me_expr** var_nodes, int* var_indices, int* node_count) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        {
            int idx = -1;
            for (int i = 0; i < n_vars; i++) {
                if (node->bound == var_pointers[i]) {
                    idx = i;
                    break;
                }
            }
            if (idx >= 0) {
                var_nodes[*node_count] = node;
                var_indices[*node_count] = idx;
                (*node_count)++;
            }
            break;
        }
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                collect_variable_nodes((me_expr*)node->parameters[i], var_pointers, n_vars,
                                       var_nodes, var_indices, node_count);
            }
            break;
        }
    }
}

/* Helper to update variable bindings by matching original pointers */
static void update_vars_by_pointer(me_expr* node, const void** old_pointers, const void** new_pointers, int n_vars) {
    if (!node) return;
    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        // Find which variable this is and update to new pointer
        for (int i = 0; i < n_vars; i++) {
            if (node->bound == old_pointers[i]) {
                node->bound = new_pointers[i];
                break;
            }
        }
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                update_vars_by_pointer((me_expr*)node->parameters[i], old_pointers, new_pointers, n_vars);
            }
            break;
        }
    }
}

/* Helper to update variable bindings and nitems in tree */
static void update_variable_bindings(me_expr* node, const void** new_bounds, int* var_idx, int new_nitems) {
    if (!node) return;

    // Update nitems for all nodes to handle intermediate buffers
    if (new_nitems > 0) {
        node->nitems = new_nitems;
    }

    switch (TYPE_MASK(node->type)) {
    case ME_VARIABLE:
        if (new_bounds && *var_idx >= 0) {
            node->bound = new_bounds[*var_idx];
            (*var_idx)++;
        }
        break;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                update_variable_bindings((me_expr*)node->parameters[i], new_bounds, var_idx, new_nitems);
            }
            break;
        }
    }
}

/* Evaluate compiled expression with new variable and output pointers */
static me_expr* clone_expr(const me_expr* src) {
    if (!src) return NULL;

    const int arity = ARITY(src->type);
    const int psize = sizeof(void*) * arity;
    const int size = (sizeof(me_expr) - sizeof(void*)) + psize + (IS_CLOSURE(src->type) ? sizeof(void*) : 0);
    me_expr* clone = malloc(size);
    if (!clone) return NULL;

    // Copy the entire structure
    memcpy(clone, src, size);

    // Clone children recursively
    if (arity > 0) {
        for (int i = 0; i < arity; i++) {
            clone->parameters[i] = clone_expr((const me_expr*)src->parameters[i]);
            if (src->parameters[i] && !clone->parameters[i]) {
                // Clone failed, clean up
                for (int j = 0; j < i; j++) {
                    me_free((me_expr*)clone->parameters[j]);
                }
                free(clone);
                return NULL;
            }
        }
    }

    // Don't clone output buffer - it will be set by caller
    // Don't clone bytecode - not needed for clones
    clone->output = NULL;
    clone->bytecode = NULL;
    clone->ncode = 0;
    clone->dsl_program = NULL;
    if (TYPE_MASK(clone->type) == ME_STRING_CONSTANT) {
        clone->flags &= ~ME_EXPR_FLAG_OWNS_STRING;
    }

    return clone;
}

/* Thread-safe chunked evaluation using expression cloning.
 * This function is safe to call from multiple threads simultaneously,
 * even on the same expression object. Each call creates a temporary
 * clone of the expression tree to avoid race conditions. */
int me_eval(const me_expr* expr, const void** vars_block,
            int n_vars, void* output_block, int block_nitems,
            const me_eval_params* params) {
    if (!expr) return ME_EVAL_ERR_NULL_EXPR;
    if (expr->dtype == ME_STRING) return ME_EVAL_ERR_INVALID_ARG;
    if (expr->dsl_program) {
        return me_eval_dsl_program(expr, vars_block, n_vars, output_block, block_nitems, params);
    }

    // Verify variable count matches
    const void* original_var_pointers[ME_MAX_VARS];
    size_t var_sizes[ME_MAX_VARS];
    int actual_var_count = 0;
    save_variable_metadata(expr, original_var_pointers, var_sizes, &actual_var_count);
    if (actual_var_count > ME_MAX_VARS) {
        fprintf(stderr, "Error: Expression uses %d variables, exceeds ME_MAX_VARS=%d\n",
                actual_var_count, ME_MAX_VARS);
        return ME_EVAL_ERR_TOO_MANY_VARS;
    }

    if (actual_var_count != n_vars) {
        return ME_EVAL_ERR_VAR_MISMATCH;
    }

    // Check if using synthetic addresses by testing if all pointers are synthetic
    // Only sort if synthetic addresses are detected to restore declaration order
    int uses_synthetic = 0;
    if (actual_var_count >= 1) {
        uses_synthetic = 1;
        for (int i = 0; i < actual_var_count && uses_synthetic; i++) {
            if (!is_synthetic_address(original_var_pointers[i])) {
                uses_synthetic = 0;
            }
        }
    }

    if (uses_synthetic) {
        // Sort original_var_pointers (and var_sizes) by pointer value to ensure
        // consistent order matching the declaration order (synthetic addresses are sequential)
        for (int i = 0; i < actual_var_count - 1; i++) {
            for (int j = i + 1; j < actual_var_count; j++) {
                if (original_var_pointers[i] > original_var_pointers[j]) {
                    // Swap pointers
                    const void* tmp_ptr = original_var_pointers[i];
                    original_var_pointers[i] = original_var_pointers[j];
                    original_var_pointers[j] = tmp_ptr;
                    // Swap sizes
                    size_t tmp_size = var_sizes[i];
                    var_sizes[i] = var_sizes[j];
                    var_sizes[j] = tmp_size;
                }
            }
        }
    }

    // Clone the expression tree
    me_expr* clone = clone_expr(expr);
    if (!clone) return ME_EVAL_ERR_OOM;

    me_simd_params_state simd_state;
    me_simd_params_push(params, &simd_state);

    const int eval_block_nitems = ME_EVAL_BLOCK_NITEMS;
    int status = ME_EVAL_SUCCESS;

    if (!ME_EVAL_ENABLE_BLOCKING || block_nitems <= eval_block_nitems || contains_reduction(clone)) {
        // Update clone's variable bindings
        update_vars_by_pointer(clone, original_var_pointers, vars_block, n_vars);

        // Update clone's nitems throughout the tree
        int update_idx = 0;
        update_variable_bindings(clone, NULL, &update_idx, block_nitems);

        // Set output pointer
        clone->output = output_block;

        // Evaluate the clone
        me_sincos_eval_start();
        private_eval(clone);
    }
    else if (is_reduction_node(clone)) {
        // Reductions operate on the full chunk; avoid block processing.
        update_vars_by_pointer(clone, original_var_pointers, vars_block, n_vars);

        int update_idx = 0;
        update_variable_bindings(clone, NULL, &update_idx, block_nitems);

        clone->output = output_block;
        me_sincos_eval_start();
        private_eval(clone);
    }
    else {
        const size_t output_item_size = dtype_size(clone->dtype);
        const int max_var_nodes = count_variable_nodes(clone);
        me_expr** var_nodes = NULL;
        int* var_indices = NULL;
        int var_node_count = 0;

        if (max_var_nodes > 0) {
            var_nodes = malloc((size_t)max_var_nodes * sizeof(*var_nodes));
            var_indices = malloc((size_t)max_var_nodes * sizeof(*var_indices));
            if (!var_nodes || !var_indices) {
                free(var_nodes);
                free(var_indices);
                status = ME_EVAL_ERR_OOM;
                goto cleanup;
            }
            collect_variable_nodes(clone, original_var_pointers, n_vars,
                                   var_nodes, var_indices, &var_node_count);
        }

#if defined(__clang__)
#pragma clang loop unroll_count(4)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 4
#endif
        for (int offset = 0; offset < block_nitems; offset += eval_block_nitems) {
            int current = eval_block_nitems;
            if (offset + current > block_nitems) {
                current = block_nitems - offset;
            }

            const void* block_vars[ME_MAX_VARS];
            for (int i = 0; i < n_vars; i++) {
                const unsigned char* base = (const unsigned char*)vars_block[i];
                block_vars[i] = base + (size_t)offset * var_sizes[i];
            }

            for (int i = 0; i < var_node_count; i++) {
                var_nodes[i]->bound = block_vars[var_indices[i]];
            }

            int update_idx = 0;
            update_variable_bindings(clone, NULL, &update_idx, current);

            clone->output = (unsigned char*)output_block + (size_t)offset * output_item_size;
            me_sincos_eval_start();
            private_eval(clone);
        }

        free(var_nodes);
        free(var_indices);
    }

cleanup:
    // Free the clone (including any intermediate buffers it allocated)
    me_simd_params_pop(&simd_state);
    me_free(clone);
    return status;
}


void optimize(me_expr* n) {
    /* Evaluates as much as possible. */
    if (!n) return;
    if (n->type == ME_CONSTANT) return;
    if (n->type == ME_STRING_CONSTANT) return;
    if (n->type == ME_VARIABLE) return;

    /* Only optimize out functions flagged as pure. */
    if (IS_PURE(n->type)) {
        const int arity = ARITY(n->type);
        int known = 1;
        int i;
        for (i = 0; i < arity; ++i) {
            optimize(n->parameters[i]);
            if (((me_expr*)(n->parameters[i]))->type != ME_CONSTANT) {
                known = 0;
            }
        }
        if (IS_FUNCTION(n->type) && arity == 2 && n->function == (void*)pow) {
            me_expr *right = (me_expr*)n->parameters[1];
            if (right && right->type == ME_CONSTANT && right->value == 2.0) {
                /* Fast path: rewrite x**2 (or pow(x,2)) to x*x to avoid scalar pow in eval. */
                me_expr *left = (me_expr*)n->parameters[0];
                me_expr *left_clone = clone_expr(left);
                if (left_clone) {
                    me_free(right);
                    n->parameters[1] = left_clone;
                    n->function = mul;
                    apply_type_promotion(n);
                    known = 0;
                }
            } else if (right && right->type == ME_CONSTANT && right->value == 3.0) {
                /* Fast path: rewrite x**3 (or pow(x,3)) to x*x*x to avoid scalar pow in eval. */
                me_expr *left = (me_expr*)n->parameters[0];
                me_expr *left_clone = clone_expr(left);
                me_expr *left_clone2 = clone_expr(left);
                if (left_clone && left_clone2) {
                    me_expr *inner = NEW_EXPR(ME_FUNCTION2 | ME_FLAG_PURE, left, left_clone);
                    if (inner) {
                        inner->function = mul;
                        apply_type_promotion(inner);
                        me_free(right);
                        n->parameters[0] = inner;
                        n->parameters[1] = left_clone2;
                        n->function = mul;
                        apply_type_promotion(n);
                        known = 0;
                    } else {
                        me_free(left_clone);
                        me_free(left_clone2);
                    }
                } else {
                    me_free(left_clone);
                    me_free(left_clone2);
                }
            }
        }
        if (known) {
            const double value = me_eval_scalar(n);
            me_free_parameters(n);
            n->type = ME_CONSTANT;
            n->value = value;
        }
    }
}

bool has_complex_node(const me_expr* n) {
    if (!n) return false;
    if (n->dtype == ME_COMPLEX64 || n->dtype == ME_COMPLEX128) return true;
    const int arity = ARITY(n->type);
    for (int i = 0; i < arity; i++) {
        if (has_complex_node((const me_expr*)n->parameters[i])) return true;
    }
    return false;
}

bool has_complex_input(const me_expr* n) {
    if (!n) return false;
    if (n->input_dtype == ME_COMPLEX64 || n->input_dtype == ME_COMPLEX128) return true;
    const int arity = ARITY(n->type);
    for (int i = 0; i < arity; i++) {
        if (has_complex_input((const me_expr*)n->parameters[i])) return true;
    }
    return false;
}

bool has_complex_input_types(const me_expr* n) {
    if (!n) return false;
    if (n->dtype == ME_COMPLEX64 || n->dtype == ME_COMPLEX128 ||
        n->input_dtype == ME_COMPLEX64 || n->input_dtype == ME_COMPLEX128) {
        return true;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
    case ME_VARIABLE:
        return n->dtype == ME_COMPLEX64 || n->dtype == ME_COMPLEX128 ||
            n->input_dtype == ME_COMPLEX64 || n->input_dtype == ME_COMPLEX128;
    default:
        break;
    }

    const int arity = ARITY(n->type);
    for (int i = 0; i < arity; i++) {
        if (has_complex_input_types((const me_expr*)n->parameters[i])) return true;
    }
    return false;
}

static bool is_complex_supported_function(const me_expr* n) {
    if (!n || !IS_FUNCTION(n->type)) return true;
    if (is_reduction_node(n)) return true;
    if (is_comparison_node(n)) return false;

    const int arity = ARITY(n->type);
    const void* func = n->function;

    if (arity == 1) {
        return func == (void*)negate ||
            func == (void*)sqrt ||
            func == (void*)conj_wrapper ||
            func == (void*)real_wrapper ||
            func == (void*)imag_wrapper ||
            func == (void*)fabs;
    }
    if (arity == 2) {
        return func == add ||
            func == sub ||
            func == mul ||
            func == divide ||
            func == (void*)pow;
    }

    return false;
}

bool has_unsupported_complex_function(const me_expr* n) {
    if (!n) return false;
    if (IS_FUNCTION(n->type) && !is_complex_supported_function(n)) return true;

    const int arity = ARITY(n->type);
    for (int i = 0; i < arity; i++) {
        if (has_unsupported_complex_function((const me_expr*)n->parameters[i])) return true;
    }
    return false;
}
me_cmp_kind comparison_kind(const void* func) {
    if (func == (void*)cmp_eq) return ME_CMP_EQ;
    if (func == (void*)cmp_ne) return ME_CMP_NE;
    if (func == (void*)cmp_lt) return ME_CMP_LT;
    if (func == (void*)cmp_le) return ME_CMP_LE;
    if (func == (void*)cmp_gt) return ME_CMP_GT;
    if (func == (void*)cmp_ge) return ME_CMP_GE;
    return ME_CMP_NONE;
}
