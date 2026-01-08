/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "functions.h"
#include "functions-simd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_MSC_VER)
#define ME_THREAD_LOCAL __declspec(thread)
#else
#define ME_THREAD_LOCAL __thread
#endif

static ME_THREAD_LOCAL unsigned long long me_eval_cookie = 0;

#ifndef ME_USE_SLEEF
#define ME_USE_SLEEF 1
#endif

#if ME_USE_SLEEF && (defined(__clang__) || defined(__GNUC__)) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64))
#define ME_ENABLE_SLEEF_SIMD 1
#else
#define ME_ENABLE_SLEEF_SIMD 0
#endif

#if ME_ENABLE_SLEEF_SIMD
#if defined(__clang__) || defined(__GNUC__)
#define ME_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define ME_AVX2_TARGET
#endif
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wmacro-redefined"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wshadow"
#endif
/* Limit warning suppression to the SLEEF include block below. */

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#endif
#ifndef __AVX__
#define ME_SLEEF_UNDEF_AVX
#define __AVX__ 1
#endif
#ifndef __AVX2__
#define ME_SLEEF_UNDEF_AVX2
#define __AVX2__ 1
#endif
#ifndef __FMA__
#define ME_SLEEF_UNDEF_FMA
#define __FMA__ 1
#endif
#define ENABLE_AVX2
#include "src/common/common.c"
#include "src/libm/rempitab.c"
#include "src/libm/sleefsimddp.c"
#include "src/libm/sleefsimdsp.c"
#undef ENABLE_AVX2
#if defined(ME_SLEEF_UNDEF_FMA)
#undef __FMA__
#undef ME_SLEEF_UNDEF_FMA
#endif
#if defined(ME_SLEEF_UNDEF_AVX2)
#undef __AVX2__
#undef ME_SLEEF_UNDEF_AVX2
#endif
#if defined(ME_SLEEF_UNDEF_AVX)
#undef __AVX__
#undef ME_SLEEF_UNDEF_AVX
#endif
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ENABLE_ADVSIMD
#include "src/libm/rempitab.c"
#include "src/libm/sleefsimddp.c"
#include "src/libm/sleefsimdsp.c"
#undef ENABLE_ADVSIMD
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

typedef void (*me_vec_unary_f64)(const double* a, double* out, int n);
typedef void (*me_vec_unary_f32)(const float* a, float* out, int n);
typedef void (*me_vec_binary_f64)(const double* a, const double* b, double* out, int n);
typedef void (*me_vec_binary_f32)(const float* a, const float* b, float* out, int n);
typedef void (*me_vec_sincos_f64)(const double* a, double* sin_out, double* cos_out, int n);
typedef void (*me_vec_sincos_f32)(const float* a, float* sin_out, float* cos_out, int n);

/* Default to u35 for SIMD transcendentals; use me_set_simd_ulp_mode(ME_SIMD_ULP_1) for u10. */
static int me_simd_use_u35 = 1;

typedef struct {
    const void *key;
    int nitems;
    unsigned long long cookie;
    double *sin_buf;
    double *cos_buf;
    int cap;
} me_sincos_cache_f64;

typedef struct {
    const void *key;
    int nitems;
    unsigned long long cookie;
    float *sin_buf;
    float *cos_buf;
    int cap;
} me_sincos_cache_f32;

static ME_THREAD_LOCAL me_sincos_cache_f64 me_sincos_cache_dp = {0};
static ME_THREAD_LOCAL me_sincos_cache_f32 me_sincos_cache_sp = {0};

static void vec_sin_scalar(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static void vec_cos_scalar(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static void vec_sin_f32_scalar(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static void vec_cos_f32_scalar(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static void vec_asin_scalar(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = asin(a[i]);
    }
}

static void vec_acos_scalar(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = acos(a[i]);
    }
}

static void vec_atan_scalar(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = atan(a[i]);
    }
}

static void vec_atan2_scalar(const double* a, const double* b, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = atan2(a[i], b[i]);
    }
}

static void vec_asin_f32_scalar(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = asinf(a[i]);
    }
}

static void vec_acos_f32_scalar(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = acosf(a[i]);
    }
}

static void vec_atan_f32_scalar(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = atanf(a[i]);
    }
}

static void vec_atan2_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = atan2f(a[i], b[i]);
    }
}

static void vec_tan_scalar(const double* a, double* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = tan(a[i]);
    }
}

static void vec_tan_f32_scalar(const float* a, float* out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        out[i] = tanf(a[i]);
    }
}

static void vec_sincos_scalar(const double* a, double* sin_out, double* cos_out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        sin_out[i] = sin(a[i]);
        cos_out[i] = cos(a[i]);
    }
}

static void vec_sincos_f32_scalar(const float* a, float* sin_out, float* cos_out, int n) {
    int i;
#pragma GCC ivdep
    for (i = 0; i < n; i++) {
        sin_out[i] = sinf(a[i]);
        cos_out[i] = cosf(a[i]);
    }
}

#if ME_ENABLE_SLEEF_SIMD && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
static ME_AVX2_TARGET void vec_sincos_avx2(const double* a, double* sin_out, double* cos_out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble2 r = me_simd_use_u35 ? xsincos(v) : xsincos_u1(v);
        vstoreu_v_p_vd(sin_out + i, vd2getx_vd_vd2(r));
        vstoreu_v_p_vd(cos_out + i, vd2gety_vd_vd2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sin(a[i]);
        cos_out[i] = cos(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sincos_f32_avx2(const float* a, float* sin_out, float* cos_out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat2 r = me_simd_use_u35 ? xsincosf(v) : xsincosf_u1(v);
        vstoreu_v_p_vf(sin_out + i, vf2getx_vf_vf2(r));
        vstoreu_v_p_vf(cos_out + i, vf2gety_vf_vf2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sinf(a[i]);
        cos_out[i] = cosf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sin_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xsin(v) : xsin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static ME_AVX2_TARGET void vec_cos_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xcos(v) : xcos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sin_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xsinf(v) : xsinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_cos_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xcosf(v) : xcosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_asin_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xasin(v) : xasin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asin(a[i]);
    }
}

static ME_AVX2_TARGET void vec_acos_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xacos(v) : xacos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acos(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xatan(v) : xatan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan2_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble vy = vloadu_vd_p(a + i);
        vdouble vx = vloadu_vd_p(b + i);
        vdouble r = me_simd_use_u35 ? xatan2(vy, vx) : xatan2_u1(vy, vx);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_asin_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xasinf(v) : xasinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_acos_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xacosf(v) : xacosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acosf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xatanf(v) : xatanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan2_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat vy = vloadu_vf_p(a + i);
        vfloat vx = vloadu_vf_p(b + i);
        vfloat r = me_simd_use_u35 ? xatan2f(vy, vx) : xatan2f_u1(vy, vx);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2f(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_tan_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xtan(v) : xtan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tan(a[i]);
    }
}

static ME_AVX2_TARGET void vec_tan_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xtanf(v) : xtanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanf(a[i]);
    }
}
#endif

#if ME_ENABLE_SLEEF_SIMD && (defined(__aarch64__) || defined(_M_ARM64))
static void vec_sincos_advsimd(const double* a, double* sin_out, double* cos_out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble2 r = me_simd_use_u35 ? xsincos(v) : xsincos_u1(v);
        vstoreu_v_p_vd(sin_out + i, vd2getx_vd_vd2(r));
        vstoreu_v_p_vd(cos_out + i, vd2gety_vd_vd2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sin(a[i]);
        cos_out[i] = cos(a[i]);
    }
}

static void vec_sincos_f32_advsimd(const float* a, float* sin_out, float* cos_out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat2 r = me_simd_use_u35 ? xsincosf(v) : xsincosf_u1(v);
        vstoreu_v_p_vf(sin_out + i, vf2getx_vf_vf2(r));
        vstoreu_v_p_vf(cos_out + i, vf2gety_vf_vf2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sinf(a[i]);
        cos_out[i] = cosf(a[i]);
    }
}

static void vec_sin_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xsin(v) : xsin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static void vec_cos_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xcos(v) : xcos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static void vec_sin_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xsinf(v) : xsinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static void vec_cos_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xcosf(v) : xcosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static void vec_asin_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xasin(v) : xasin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asin(a[i]);
    }
}

static void vec_acos_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xacos(v) : xacos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acos(a[i]);
    }
}

static void vec_atan_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xatan(v) : xatan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan(a[i]);
    }
}

static void vec_atan2_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble vy = vloadu_vd_p(a + i);
        vdouble vx = vloadu_vd_p(b + i);
        vdouble r = me_simd_use_u35 ? xatan2(vy, vx) : xatan2_u1(vy, vx);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2(a[i], b[i]);
    }
}

static void vec_asin_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xasinf(v) : xasinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinf(a[i]);
    }
}

static void vec_acos_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xacosf(v) : xacosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acosf(a[i]);
    }
}

static void vec_atan_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xatanf(v) : xatanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanf(a[i]);
    }
}

static void vec_atan2_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat vy = vloadu_vf_p(a + i);
        vfloat vx = vloadu_vf_p(b + i);
        vfloat r = me_simd_use_u35 ? xatan2f(vy, vx) : xatan2f_u1(vy, vx);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2f(a[i], b[i]);
    }
}

static void vec_tan_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = me_simd_use_u35 ? xtan(v) : xtan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tan(a[i]);
    }
}

static void vec_tan_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = me_simd_use_u35 ? xtanf(v) : xtanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanf(a[i]);
    }
}
#endif

static me_vec_unary_f64 vec_sin_impl = vec_sin_scalar;
static me_vec_unary_f64 vec_cos_impl = vec_cos_scalar;
static me_vec_unary_f64 vec_tan_impl = vec_tan_scalar;
static me_vec_unary_f64 vec_asin_impl = vec_asin_scalar;
static me_vec_unary_f64 vec_acos_impl = vec_acos_scalar;
static me_vec_unary_f64 vec_atan_impl = vec_atan_scalar;
static me_vec_binary_f64 vec_atan2_impl = vec_atan2_scalar;
static me_vec_unary_f32 vec_sin_f32_impl = vec_sin_f32_scalar;
static me_vec_unary_f32 vec_cos_f32_impl = vec_cos_f32_scalar;
static me_vec_unary_f32 vec_tan_f32_impl = vec_tan_f32_scalar;
static me_vec_unary_f32 vec_asin_f32_impl = vec_asin_f32_scalar;
static me_vec_unary_f32 vec_acos_f32_impl = vec_acos_f32_scalar;
static me_vec_unary_f32 vec_atan_f32_impl = vec_atan_f32_scalar;
static me_vec_binary_f32 vec_atan2_f32_impl = vec_atan2_f32_scalar;
static me_vec_sincos_f64 vec_sincos_impl = vec_sincos_scalar;
static me_vec_sincos_f32 vec_sincos_f32_impl = vec_sincos_f32_scalar;
static int me_simd_initialized = 0;
static int me_simd_enabled = 1;
static const char *me_simd_backend = "scalar";

void me_sincos_eval_start(void) {
    me_eval_cookie++;
}

void vec_sin_cached(const double* a, double* out, int n) {
    if (!a || !out || n <= 0) {
        return;
    }
    me_sincos_cache_f64 *cache = &me_sincos_cache_dp;
    if (cache->cookie != me_eval_cookie || cache->key != a || cache->nitems != n) {
        if (cache->cap < n) {
            int new_cap = n;
            double *sin_buf = realloc(cache->sin_buf, (size_t)new_cap * sizeof(double));
            double *cos_buf = realloc(cache->cos_buf, (size_t)new_cap * sizeof(double));
            if (!sin_buf || !cos_buf) {
                free(sin_buf);
                free(cos_buf);
                cache->sin_buf = NULL;
                cache->cos_buf = NULL;
                cache->cap = 0;
                vec_sin_scalar(a, out, n);
                return;
            }
            cache->sin_buf = sin_buf;
            cache->cos_buf = cos_buf;
            cache->cap = new_cap;
        }
        vec_sincos_impl(a, cache->sin_buf, cache->cos_buf, n);
        cache->cookie = me_eval_cookie;
        cache->key = a;
        cache->nitems = n;
    }
    memcpy(out, cache->sin_buf, (size_t)n * sizeof(double));
}

void vec_cos_cached(const double* a, double* out, int n) {
    if (!a || !out || n <= 0) {
        return;
    }
    me_sincos_cache_f64 *cache = &me_sincos_cache_dp;
    if (cache->cookie != me_eval_cookie || cache->key != a || cache->nitems != n) {
        if (cache->cap < n) {
            int new_cap = n;
            double *sin_buf = realloc(cache->sin_buf, (size_t)new_cap * sizeof(double));
            double *cos_buf = realloc(cache->cos_buf, (size_t)new_cap * sizeof(double));
            if (!sin_buf || !cos_buf) {
                free(sin_buf);
                free(cos_buf);
                cache->sin_buf = NULL;
                cache->cos_buf = NULL;
                cache->cap = 0;
                vec_cos_scalar(a, out, n);
                return;
            }
            cache->sin_buf = sin_buf;
            cache->cos_buf = cos_buf;
            cache->cap = new_cap;
        }
        vec_sincos_impl(a, cache->sin_buf, cache->cos_buf, n);
        cache->cookie = me_eval_cookie;
        cache->key = a;
        cache->nitems = n;
    }
    memcpy(out, cache->cos_buf, (size_t)n * sizeof(double));
}

void vec_sin_f32_cached(const float* a, float* out, int n) {
    if (!a || !out || n <= 0) {
        return;
    }
    me_sincos_cache_f32 *cache = &me_sincos_cache_sp;
    if (cache->cookie != me_eval_cookie || cache->key != a || cache->nitems != n) {
        if (cache->cap < n) {
            int new_cap = n;
            float *sin_buf = realloc(cache->sin_buf, (size_t)new_cap * sizeof(float));
            float *cos_buf = realloc(cache->cos_buf, (size_t)new_cap * sizeof(float));
            if (!sin_buf || !cos_buf) {
                free(sin_buf);
                free(cos_buf);
                cache->sin_buf = NULL;
                cache->cos_buf = NULL;
                cache->cap = 0;
                vec_sin_f32_scalar(a, out, n);
                return;
            }
            cache->sin_buf = sin_buf;
            cache->cos_buf = cos_buf;
            cache->cap = new_cap;
        }
        vec_sincos_f32_impl(a, cache->sin_buf, cache->cos_buf, n);
        cache->cookie = me_eval_cookie;
        cache->key = a;
        cache->nitems = n;
    }
    memcpy(out, cache->sin_buf, (size_t)n * sizeof(float));
}

void vec_cos_f32_cached(const float* a, float* out, int n) {
    if (!a || !out || n <= 0) {
        return;
    }
    me_sincos_cache_f32 *cache = &me_sincos_cache_sp;
    if (cache->cookie != me_eval_cookie || cache->key != a || cache->nitems != n) {
        if (cache->cap < n) {
            int new_cap = n;
            float *sin_buf = realloc(cache->sin_buf, (size_t)new_cap * sizeof(float));
            float *cos_buf = realloc(cache->cos_buf, (size_t)new_cap * sizeof(float));
            if (!sin_buf || !cos_buf) {
                free(sin_buf);
                free(cos_buf);
                cache->sin_buf = NULL;
                cache->cos_buf = NULL;
                cache->cap = 0;
                vec_cos_f32_scalar(a, out, n);
                return;
            }
            cache->sin_buf = sin_buf;
            cache->cos_buf = cos_buf;
            cache->cap = new_cap;
        }
        vec_sincos_f32_impl(a, cache->sin_buf, cache->cos_buf, n);
        cache->cookie = me_eval_cookie;
        cache->key = a;
        cache->nitems = n;
    }
    memcpy(out, cache->cos_buf, (size_t)n * sizeof(float));
}

static int me_cpu_supports_avx2(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#if defined(_MSC_VER)
    int info[4] = {0, 0, 0, 0};
    __cpuidex(info, 1, 0);
    const int osxsave = (info[2] & (1 << 27)) != 0;
    const int avx = (info[2] & (1 << 28)) != 0;
    const int fma = (info[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) {
        return 0;
    }
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return 0;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int a = 0, b = 0, c = 0, d = 0;
    __asm__ __volatile__("cpuid"
                         : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                         : "a"(1), "c"(0));
    const int osxsave = (c & (1u << 27)) != 0;
    const int avx = (c & (1u << 28)) != 0;
    const int fma = (c & (1u << 12)) != 0;
    if (!osxsave || !avx || !fma) {
        return 0;
    }
    unsigned int xcr0_lo = 0, xcr0_hi = 0;
    __asm__ __volatile__("xgetbv"
                         : "=a"(xcr0_lo), "=d"(xcr0_hi)
                         : "c"(0));
    if ((xcr0_lo & 0x6u) != 0x6u) {
        return 0;
    }
    __asm__ __volatile__("cpuid"
                         : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                         : "a"(7), "c"(0));
    return (b & (1u << 5)) != 0;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

static int me_cpu_supports_advsimd(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return 1;
#else
    return 0;
#endif
}

static void me_init_simd(void) {
    if (me_simd_initialized) {
        return;
    }
    me_simd_initialized = 1;

    if (!me_simd_enabled) {
        vec_sin_impl = vec_sin_scalar;
        vec_cos_impl = vec_cos_scalar;
        vec_tan_impl = vec_tan_scalar;
        vec_asin_impl = vec_asin_scalar;
        vec_acos_impl = vec_acos_scalar;
        vec_atan_impl = vec_atan_scalar;
        vec_atan2_impl = vec_atan2_scalar;
        vec_sin_f32_impl = vec_sin_f32_scalar;
        vec_cos_f32_impl = vec_cos_f32_scalar;
        vec_tan_f32_impl = vec_tan_f32_scalar;
        vec_asin_f32_impl = vec_asin_f32_scalar;
        vec_acos_f32_impl = vec_acos_f32_scalar;
        vec_atan_f32_impl = vec_atan_f32_scalar;
        vec_atan2_f32_impl = vec_atan2_f32_scalar;
        vec_sincos_impl = vec_sincos_scalar;
        vec_sincos_f32_impl = vec_sincos_f32_scalar;
        me_simd_backend = "scalar";
        return;
    }

    /* Use SLEEF SIMD kernels when the CPU supports them. */
#if ME_ENABLE_SLEEF_SIMD && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
    if (me_cpu_supports_avx2()) {
        vec_sin_impl = vec_sin_avx2;
        vec_cos_impl = vec_cos_avx2;
        vec_tan_impl = vec_tan_avx2;
        vec_asin_impl = vec_asin_avx2;
        vec_acos_impl = vec_acos_avx2;
        vec_atan_impl = vec_atan_avx2;
        vec_atan2_impl = vec_atan2_avx2;
        vec_sin_f32_impl = vec_sin_f32_avx2;
        vec_cos_f32_impl = vec_cos_f32_avx2;
        vec_tan_f32_impl = vec_tan_f32_avx2;
        vec_asin_f32_impl = vec_asin_f32_avx2;
        vec_acos_f32_impl = vec_acos_f32_avx2;
        vec_atan_f32_impl = vec_atan_f32_avx2;
        vec_atan2_f32_impl = vec_atan2_f32_avx2;
        vec_sincos_impl = vec_sincos_avx2;
        vec_sincos_f32_impl = vec_sincos_f32_avx2;
        me_simd_backend = me_simd_use_u35 ? "avx2-u35" : "avx2-u10";
        return;
    }
#endif

#if ME_ENABLE_SLEEF_SIMD && (defined(__aarch64__) || defined(_M_ARM64))
    if (me_cpu_supports_advsimd()) {
        vec_sin_impl = vec_sin_advsimd;
        vec_cos_impl = vec_cos_advsimd;
        vec_tan_impl = vec_tan_advsimd;
        vec_asin_impl = vec_asin_advsimd;
        vec_acos_impl = vec_acos_advsimd;
        vec_atan_impl = vec_atan_advsimd;
        vec_atan2_impl = vec_atan2_advsimd;
        vec_sin_f32_impl = vec_sin_f32_advsimd;
        vec_cos_f32_impl = vec_cos_f32_advsimd;
        vec_tan_f32_impl = vec_tan_f32_advsimd;
        vec_asin_f32_impl = vec_asin_f32_advsimd;
        vec_acos_f32_impl = vec_acos_f32_advsimd;
        vec_atan_f32_impl = vec_atan_f32_advsimd;
        vec_atan2_f32_impl = vec_atan2_f32_advsimd;
        vec_sincos_impl = vec_sincos_advsimd;
        vec_sincos_f32_impl = vec_sincos_f32_advsimd;
        me_simd_backend = me_simd_use_u35 ? "advsimd-u35" : "advsimd-u10";
    }
#endif

    if (me_simd_backend[0] == '\0') {
        me_simd_backend = "scalar";
    }
}

void vec_sin_dispatch(const double* a, double* out, int n) {
    me_init_simd();
    vec_sin_impl(a, out, n);
}

void vec_cos_dispatch(const double* a, double* out, int n) {
    me_init_simd();
    vec_cos_impl(a, out, n);
}

void vec_asin_dispatch(const double* a, double* out, int n) {
    me_init_simd();
    vec_asin_impl(a, out, n);
}

void vec_acos_dispatch(const double* a, double* out, int n) {
    me_init_simd();
    vec_acos_impl(a, out, n);
}

void vec_atan_dispatch(const double* a, double* out, int n) {
    me_init_simd();
    vec_atan_impl(a, out, n);
}

void vec_tan_dispatch(const double* a, double* out, int n) {
    me_init_simd();
    vec_tan_impl(a, out, n);
}

void vec_atan2_dispatch(const double* a, const double* b, double* out, int n) {
    me_init_simd();
    vec_atan2_impl(a, b, out, n);
}

void vec_sin_f32_dispatch(const float* a, float* out, int n) {
    me_init_simd();
    vec_sin_f32_impl(a, out, n);
}

void vec_cos_f32_dispatch(const float* a, float* out, int n) {
    me_init_simd();
    vec_cos_f32_impl(a, out, n);
}

void vec_asin_f32_dispatch(const float* a, float* out, int n) {
    me_init_simd();
    vec_asin_f32_impl(a, out, n);
}

void vec_acos_f32_dispatch(const float* a, float* out, int n) {
    me_init_simd();
    vec_acos_f32_impl(a, out, n);
}

void vec_atan_f32_dispatch(const float* a, float* out, int n) {
    me_init_simd();
    vec_atan_f32_impl(a, out, n);
}

void vec_tan_f32_dispatch(const float* a, float* out, int n) {
    me_init_simd();
    vec_tan_f32_impl(a, out, n);
}

void vec_atan2_f32_dispatch(const float* a, const float* b, float* out, int n) {
    me_init_simd();
    vec_atan2_f32_impl(a, b, out, n);
}

void me_disable_simd(int disabled) {
    if (!disabled) {
        me_simd_enabled = 1;
        me_simd_initialized = 0;
        me_init_simd();
    } else {
        me_simd_enabled = 0;
        me_simd_initialized = 1;
        vec_sin_impl = vec_sin_scalar;
        vec_cos_impl = vec_cos_scalar;
        vec_tan_impl = vec_tan_scalar;
        vec_sin_f32_impl = vec_sin_f32_scalar;
        vec_cos_f32_impl = vec_cos_f32_scalar;
        vec_tan_f32_impl = vec_tan_f32_scalar;
        vec_sincos_impl = vec_sincos_scalar;
        vec_sincos_f32_impl = vec_sincos_f32_scalar;
        me_simd_backend = "scalar";
    }
}

void me_set_simd_ulp_mode(me_simd_ulp_mode mode) {
    me_simd_use_u35 = (mode == ME_SIMD_ULP_3_5) ? 1 : 0;
    if (me_simd_enabled) {
        me_simd_initialized = 0;
        me_init_simd();
    }
}

const char *me_get_simd_backend(void) {
    me_init_simd();
    return me_simd_backend;
}
