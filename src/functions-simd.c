/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "functions.h"
#include "functions-simd.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define IVDEP
#else
#define IVDEP _Pragma("GCC ivdep")
#endif

#if defined(_MSC_VER)
#define ME_THREAD_LOCAL __declspec(thread)
#else
#define ME_THREAD_LOCAL __thread
#endif

static ME_THREAD_LOCAL unsigned long long me_eval_cookie = 0;
static ME_THREAD_LOCAL int me_simd_force_scalar = 0;
static ME_THREAD_LOCAL int me_simd_use_u35_override = -1;
static void me_init_simd(void);

#ifndef ME_USE_SLEEF
#define ME_USE_SLEEF 1
#endif

#ifndef ME_DSL_TRACE_DEFAULT
#define ME_DSL_TRACE_DEFAULT 0
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
#pragma clang diagnostic ignored "-Wunused-function"
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
typedef void (*me_vec_ternary_f64)(const double* a, const double* b, const double* c, double* out, int n);
typedef void (*me_vec_ternary_f32)(const float* a, const float* b, const float* c, float* out, int n);
typedef void (*me_vec_sincos_f64)(const double* a, double* sin_out, double* cos_out, int n);
typedef void (*me_vec_sincos_f32)(const float* a, float* sin_out, float* cos_out, int n);

/* Default to ME_SIMD_ULP_DEFAULT_MODE for SIMD transcendentals. */
static int me_simd_use_u35_default = (ME_SIMD_ULP_DEFAULT_MODE == ME_SIMD_ULP_3_5) ? 1 : 0;
static const double me_pi = 3.14159265358979323846;
static const float me_pif = 3.14159265358979323846f;

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

static int me_simd_use_u35_active(void) {
    if (me_simd_use_u35_override >= 0) {
        return me_simd_use_u35_override;
    }
    return me_simd_use_u35_default;
}

void me_simd_params_push(const me_eval_params *params, me_simd_params_state *state) {
    state->force_scalar = me_simd_force_scalar;
    state->override_u35 = me_simd_use_u35_override;

    if (!params) {
        me_simd_force_scalar = 0;
        me_simd_use_u35_override = -1;
        return;
    }

    me_simd_force_scalar = params->disable_simd ? 1 : 0;
    switch (params->simd_ulp_mode) {
    case ME_SIMD_ULP_1:
        me_simd_use_u35_override = 0;
        break;
    case ME_SIMD_ULP_3_5:
        me_simd_use_u35_override = 1;
        break;
    case ME_SIMD_ULP_DEFAULT:
    default:
        me_simd_use_u35_override = -1;
        break;
    }
}

void me_simd_params_pop(const me_simd_params_state *state) {
    me_simd_force_scalar = state->force_scalar;
    me_simd_use_u35_override = state->override_u35;
}

static void vec_sin_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static void vec_cos_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static void vec_sin_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static void vec_cos_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static void vec_asin_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = asin(a[i]);
    }
}

static void vec_acos_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = acos(a[i]);
    }
}

static void vec_atan_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = atan(a[i]);
    }
}

static void vec_atan2_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = atan2(a[i], b[i]);
    }
}

static void vec_asin_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = asinf(a[i]);
    }
}

static void vec_acos_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = acosf(a[i]);
    }
}

static void vec_atan_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = atanf(a[i]);
    }
}

static void vec_atan2_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = atan2f(a[i], b[i]);
    }
}

static void vec_tan_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = tan(a[i]);
    }
}

static void vec_tan_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = tanf(a[i]);
    }
}

static void vec_abs_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fabs(a[i]);
    }
}

static void vec_abs_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fabsf(a[i]);
    }
}

static void vec_exp_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = exp(a[i]);
    }
}

static void vec_exp_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = expf(a[i]);
    }
}

static void vec_expm1_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = expm1(a[i]);
    }
}

static void vec_expm1_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = expm1f(a[i]);
    }
}

static void vec_log_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = log(a[i]);
    }
}

static void vec_log_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = logf(a[i]);
    }
}

static void vec_log10_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = log10(a[i]);
    }
}

static void vec_log10_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = log10f(a[i]);
    }
}

static void vec_log1p_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = log1p(a[i]);
    }
}

static void vec_log1p_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = log1pf(a[i]);
    }
}

static void vec_log2_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = log2(a[i]);
    }
}

static void vec_log2_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = log2f(a[i]);
    }
}

static void vec_sqrt_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sqrt(a[i]);
    }
}

static void vec_sqrt_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sqrtf(a[i]);
    }
}

static void vec_sinh_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sinh(a[i]);
    }
}

static void vec_sinh_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sinhf(a[i]);
    }
}

static void vec_cosh_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = cosh(a[i]);
    }
}

static void vec_cosh_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = coshf(a[i]);
    }
}

static void vec_tanh_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = tanh(a[i]);
    }
}

static void vec_tanh_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = tanhf(a[i]);
    }
}

static void vec_acosh_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = acosh(a[i]);
    }
}

static void vec_acosh_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = acoshf(a[i]);
    }
}

static void vec_asinh_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = asinh(a[i]);
    }
}

static void vec_asinh_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = asinhf(a[i]);
    }
}

static void vec_atanh_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = atanh(a[i]);
    }
}

static void vec_atanh_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = atanhf(a[i]);
    }
}

static void vec_ceil_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = ceil(a[i]);
    }
}

static void vec_ceil_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = ceilf(a[i]);
    }
}

static void vec_floor_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = floor(a[i]);
    }
}

static void vec_floor_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = floorf(a[i]);
    }
}

static void vec_round_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = round(a[i]);
    }
}

static void vec_round_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = roundf(a[i]);
    }
}

static void vec_trunc_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = trunc(a[i]);
    }
}

static void vec_trunc_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = truncf(a[i]);
    }
}

static void vec_exp2_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = exp2(a[i]);
    }
}

static void vec_exp2_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = exp2f(a[i]);
    }
}

static void vec_exp10_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = pow(10.0, a[i]);
    }
}

static void vec_exp10_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = powf(10.0f, a[i]);
    }
}

static void vec_cbrt_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = cbrt(a[i]);
    }
}

static void vec_cbrt_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = cbrtf(a[i]);
    }
}

static void vec_erf_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = erf(a[i]);
    }
}

static void vec_erf_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = erff(a[i]);
    }
}

static void vec_erfc_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = erfc(a[i]);
    }
}

static void vec_erfc_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = erfcf(a[i]);
    }
}

static void vec_sinpi_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sin(me_pi * a[i]);
    }
}

static void vec_sinpi_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = sinf(me_pif * a[i]);
    }
}

static void vec_cospi_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = cos(me_pi * a[i]);
    }
}

static void vec_cospi_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = cosf(me_pif * a[i]);
    }
}

static void vec_tgamma_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = tgamma(a[i]);
    }
}

static void vec_tgamma_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = tgammaf(a[i]);
    }
}

static void vec_lgamma_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = lgamma(a[i]);
    }
}

static void vec_lgamma_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = lgammaf(a[i]);
    }
}

static void vec_rint_scalar(const double* a, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = rint(a[i]);
    }
}

static void vec_rint_f32_scalar(const float* a, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = rintf(a[i]);
    }
}

static void vec_copysign_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = copysign(a[i], b[i]);
    }
}

static void vec_copysign_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = copysignf(a[i], b[i]);
    }
}

static void vec_fdim_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fdim(a[i], b[i]);
    }
}

static void vec_fdim_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fdimf(a[i], b[i]);
    }
}

static void vec_fmax_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fmax(a[i], b[i]);
    }
}

static void vec_fmax_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fmaxf(a[i], b[i]);
    }
}

static void vec_fmin_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fmin(a[i], b[i]);
    }
}

static void vec_fmin_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fminf(a[i], b[i]);
    }
}

static void vec_fmod_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fmod(a[i], b[i]);
    }
}

static void vec_fmod_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fmodf(a[i], b[i]);
    }
}

static void vec_hypot_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = hypot(a[i], b[i]);
    }
}

static void vec_hypot_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = hypotf(a[i], b[i]);
    }
}

static void vec_ldexp_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = ldexp(a[i], (int)b[i]);
    }
}

static void vec_ldexp_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = ldexpf(a[i], (int)b[i]);
    }
}

static void vec_nextafter_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = nextafter(a[i], b[i]);
    }
}

static void vec_nextafter_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = nextafterf(a[i], b[i]);
    }
}

static void vec_remainder_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = remainder(a[i], b[i]);
    }
}

static void vec_remainder_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = remainderf(a[i], b[i]);
    }
}

static void vec_fma_scalar(const double* a, const double* b, const double* c, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fma(a[i], b[i], c[i]);
    }
}

static void vec_fma_f32_scalar(const float* a, const float* b, const float* c, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = fmaf(a[i], b[i], c[i]);
    }
}

static void vec_pow_scalar(const double* a, const double* b, double* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = pow(a[i], b[i]);
    }
}

static void vec_pow_f32_scalar(const float* a, const float* b, float* out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        out[i] = powf(a[i], b[i]);
    }
}

static void vec_sincos_scalar(const double* a, double* sin_out, double* cos_out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        sin_out[i] = sin(a[i]);
        cos_out[i] = cos(a[i]);
    }
}

static void vec_sincos_f32_scalar(const float* a, float* sin_out, float* cos_out, int n) {
    int i;
IVDEP
    for (i = 0; i < n; i++) {
        sin_out[i] = sinf(a[i]);
        cos_out[i] = cosf(a[i]);
    }
}

#if ME_ENABLE_SLEEF_SIMD && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
static ME_AVX2_TARGET void vec_sincos_avx2(const double* a, double* sin_out, double* cos_out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble2 r = use_u35 ? xsincos(v) : xsincos_u1(v);
        vstoreu_v_p_vd(sin_out + i, vd2getx_vd_vd2(r));
        vstoreu_v_p_vd(cos_out + i, vd2gety_vd_vd2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sin(a[i]);
        cos_out[i] = cos(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sincos_f32_avx2(const float* a, float* sin_out, float* cos_out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat2 r = use_u35 ? xsincosf(v) : xsincosf_u1(v);
        vstoreu_v_p_vf(sin_out + i, vf2getx_vf_vf2(r));
        vstoreu_v_p_vf(cos_out + i, vf2gety_vf_vf2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sinf(a[i]);
        cos_out[i] = cosf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sin_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xsin(v) : xsin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static ME_AVX2_TARGET void vec_cos_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xcos(v) : xcos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sin_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xsinf(v) : xsinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_cos_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xcosf(v) : xcosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_asin_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xasin(v) : xasin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asin(a[i]);
    }
}

static ME_AVX2_TARGET void vec_acos_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xacos(v) : xacos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acos(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xatan(v) : xatan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan2_avx2(const double* a, const double* b, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble vy = vloadu_vd_p(a + i);
        vdouble vx = vloadu_vd_p(b + i);
        vdouble r = use_u35 ? xatan2(vy, vx) : xatan2_u1(vy, vx);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_asin_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xasinf(v) : xasinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_acos_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xacosf(v) : xacosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acosf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xatanf(v) : xatanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atan2_f32_avx2(const float* a, const float* b, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat vy = vloadu_vf_p(a + i);
        vfloat vx = vloadu_vf_p(b + i);
        vfloat r = use_u35 ? xatan2f(vy, vx) : xatan2f_u1(vy, vx);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2f(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_tan_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xtan(v) : xtan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tan(a[i]);
    }
}

static ME_AVX2_TARGET void vec_tan_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xtanf(v) : xtanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_abs_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xfabs(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fabs(a[i]);
    }
}

static ME_AVX2_TARGET void vec_abs_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xfabsf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fabsf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_exp_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xexp(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = exp(a[i]);
    }
}

static ME_AVX2_TARGET void vec_exp_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xexpf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = expf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_expm1_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xexpm1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = expm1(a[i]);
    }
}

static ME_AVX2_TARGET void vec_expm1_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xexpm1f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = expm1f(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xlog(v) : xlog_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xlogf(v) : xlogf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = logf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log10_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlog10(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log10(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log10_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlog10f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log10f(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log1p_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlog1p(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log1p(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log1p_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlog1pf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log1pf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log2_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlog2(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log2(a[i]);
    }
}

static ME_AVX2_TARGET void vec_log2_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlog2f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log2f(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sqrt_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xsqrt(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sqrt(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sqrt_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xsqrtf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sqrtf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sinh_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xsinh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinh(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sinh_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xsinhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinhf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_cosh_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xcosh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosh(a[i]);
    }
}

static ME_AVX2_TARGET void vec_cosh_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xcoshf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = coshf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_tanh_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xtanh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanh(a[i]);
    }
}

static ME_AVX2_TARGET void vec_tanh_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xtanhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanhf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_acosh_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xacosh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acosh(a[i]);
    }
}

static ME_AVX2_TARGET void vec_acosh_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xacoshf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acoshf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_asinh_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xasinh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinh(a[i]);
    }
}

static ME_AVX2_TARGET void vec_asinh_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xasinhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinhf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atanh_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xatanh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanh(a[i]);
    }
}

static ME_AVX2_TARGET void vec_atanh_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xatanhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanhf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_ceil_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xceil(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ceil(a[i]);
    }
}

static ME_AVX2_TARGET void vec_ceil_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xceilf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ceilf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_floor_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xfloor(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = floor(a[i]);
    }
}

static ME_AVX2_TARGET void vec_floor_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xfloorf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = floorf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_round_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xround(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = round(a[i]);
    }
}

static ME_AVX2_TARGET void vec_round_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xroundf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = roundf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_trunc_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xtrunc(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = trunc(a[i]);
    }
}

static ME_AVX2_TARGET void vec_trunc_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xtruncf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = truncf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_pow_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xpow(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = pow(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_pow_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xpowf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = powf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_exp2_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xexp2_u35(v) : xexp2(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = exp2(a[i]);
    }
}

static ME_AVX2_TARGET void vec_exp2_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xexp2f_u35(v) : xexp2f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = exp2f(a[i]);
    }
}

static ME_AVX2_TARGET void vec_exp10_avx2(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xexp10_u35(v) : xexp10(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = pow(10.0, a[i]);
    }
}

static ME_AVX2_TARGET void vec_exp10_f32_avx2(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xexp10f_u35(v) : xexp10f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = powf(10.0f, a[i]);
    }
}

static ME_AVX2_TARGET void vec_cbrt_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xcbrt_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cbrt(a[i]);
    }
}

static ME_AVX2_TARGET void vec_cbrt_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xcbrtf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cbrtf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_erf_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xerf_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_erf_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xerff_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erff(a[i]);
    }
}

static ME_AVX2_TARGET void vec_erfc_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xerfc_u15(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erfc(a[i]);
    }
}

static ME_AVX2_TARGET void vec_erfc_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xerfcf_u15(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erfcf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_sinpi_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xsinpi_u05(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sin(me_pi * a[i]);
    }
}

static ME_AVX2_TARGET void vec_sinpi_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xsinpif_u05(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinf(me_pif * a[i]);
    }
}

static ME_AVX2_TARGET void vec_cospi_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xcospi_u05(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cos(me_pi * a[i]);
    }
}

static ME_AVX2_TARGET void vec_cospi_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xcospif_u05(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosf(me_pif * a[i]);
    }
}

static ME_AVX2_TARGET void vec_tgamma_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xtgamma_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tgamma(a[i]);
    }
}

static ME_AVX2_TARGET void vec_tgamma_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xtgammaf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tgammaf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_lgamma_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlgamma_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = lgamma(a[i]);
    }
}

static ME_AVX2_TARGET void vec_lgamma_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlgammaf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = lgammaf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_rint_avx2(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xrint(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = rint(a[i]);
    }
}

static ME_AVX2_TARGET void vec_rint_f32_avx2(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xrintf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = rintf(a[i]);
    }
}

static ME_AVX2_TARGET void vec_copysign_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xcopysign(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = copysign(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_copysign_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xcopysignf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = copysignf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fdim_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfdim(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fdim(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fdim_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfdimf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fdimf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fmax_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfmax(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmax(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fmax_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfmaxf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmaxf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fmin_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfmin(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmin(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fmin_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfminf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fminf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fmod_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfmod(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmod(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fmod_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfmodf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmodf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_hypot_avx2(const double* a, const double* b, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = use_u35 ? xhypot_u35(va, vb) : xhypot_u05(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = hypot(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_hypot_f32_avx2(const float* a, const float* b, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = use_u35 ? xhypotf_u35(va, vb) : xhypotf_u05(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = hypotf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_ldexp_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble vb_int = xtrunc(vb);
        vint q = vrint_vi_vd(vb_int);
        vdouble r = xldexp(va, q);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ldexp(a[i], (int)b[i]);
    }
}

static ME_AVX2_TARGET void vec_ldexp_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat vb_int = xtruncf(vb);
        vint2 q = vrint_vi2_vf(vb_int);
        vfloat r = xldexpf(va, q);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ldexpf(a[i], (int)b[i]);
    }
}

static ME_AVX2_TARGET void vec_nextafter_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xnextafter(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = nextafter(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_nextafter_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xnextafterf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = nextafterf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_remainder_avx2(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xremainder(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = remainder(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_remainder_f32_avx2(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xremainderf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = remainderf(a[i], b[i]);
    }
}

static ME_AVX2_TARGET void vec_fma_avx2(const double* a, const double* b, const double* c, double* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble vc = vloadu_vd_p(c + i);
        vdouble r = xfma(va, vb, vc);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fma(a[i], b[i], c[i]);
    }
}

static ME_AVX2_TARGET void vec_fma_f32_avx2(const float* a, const float* b, const float* c, float* out, int n) {
    int i = 0;
    const int limit = n & ~7;
    for (; i < limit; i += 8) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat vc = vloadu_vf_p(c + i);
        vfloat r = xfmaf(va, vb, vc);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmaf(a[i], b[i], c[i]);
    }
}
#endif

#if ME_ENABLE_SLEEF_SIMD && (defined(__aarch64__) || defined(_M_ARM64))
static void vec_sincos_advsimd(const double* a, double* sin_out, double* cos_out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble2 r = use_u35 ? xsincos(v) : xsincos_u1(v);
        vstoreu_v_p_vd(sin_out + i, vd2getx_vd_vd2(r));
        vstoreu_v_p_vd(cos_out + i, vd2gety_vd_vd2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sin(a[i]);
        cos_out[i] = cos(a[i]);
    }
}

static void vec_sincos_f32_advsimd(const float* a, float* sin_out, float* cos_out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat2 r = use_u35 ? xsincosf(v) : xsincosf_u1(v);
        vstoreu_v_p_vf(sin_out + i, vf2getx_vf_vf2(r));
        vstoreu_v_p_vf(cos_out + i, vf2gety_vf_vf2(r));
    }
    for (; i < n; i++) {
        sin_out[i] = sinf(a[i]);
        cos_out[i] = cosf(a[i]);
    }
}

static void vec_sin_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xsin(v) : xsin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sin(a[i]);
    }
}

static void vec_cos_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xcos(v) : xcos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cos(a[i]);
    }
}

static void vec_sin_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xsinf(v) : xsinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinf(a[i]);
    }
}

static void vec_cos_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xcosf(v) : xcosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosf(a[i]);
    }
}

static void vec_asin_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xasin(v) : xasin_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asin(a[i]);
    }
}

static void vec_acos_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xacos(v) : xacos_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acos(a[i]);
    }
}

static void vec_atan_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xatan(v) : xatan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan(a[i]);
    }
}

static void vec_atan2_advsimd(const double* a, const double* b, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble vy = vloadu_vd_p(a + i);
        vdouble vx = vloadu_vd_p(b + i);
        vdouble r = use_u35 ? xatan2(vy, vx) : xatan2_u1(vy, vx);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2(a[i], b[i]);
    }
}

static void vec_asin_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xasinf(v) : xasinf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinf(a[i]);
    }
}

static void vec_acos_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xacosf(v) : xacosf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acosf(a[i]);
    }
}

static void vec_atan_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xatanf(v) : xatanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanf(a[i]);
    }
}

static void vec_atan2_f32_advsimd(const float* a, const float* b, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat vy = vloadu_vf_p(a + i);
        vfloat vx = vloadu_vf_p(b + i);
        vfloat r = use_u35 ? xatan2f(vy, vx) : xatan2f_u1(vy, vx);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atan2f(a[i], b[i]);
    }
}

static void vec_tan_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xtan(v) : xtan_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tan(a[i]);
    }
}

static void vec_tan_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xtanf(v) : xtanf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanf(a[i]);
    }
}

static void vec_abs_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xfabs(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fabs(a[i]);
    }
}

static void vec_abs_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xfabsf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fabsf(a[i]);
    }
}

static void vec_exp_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xexp(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = exp(a[i]);
    }
}

static void vec_exp_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xexpf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = expf(a[i]);
    }
}

static void vec_expm1_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xexpm1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = expm1(a[i]);
    }
}

static void vec_expm1_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xexpm1f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = expm1f(a[i]);
    }
}

static void vec_log_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xlog(v) : xlog_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log(a[i]);
    }
}

static void vec_log_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xlogf(v) : xlogf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = logf(a[i]);
    }
}

static void vec_log10_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlog10(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log10(a[i]);
    }
}

static void vec_log10_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlog10f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log10f(a[i]);
    }
}

static void vec_log1p_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlog1p(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log1p(a[i]);
    }
}

static void vec_log1p_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlog1pf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log1pf(a[i]);
    }
}

static void vec_log2_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlog2(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log2(a[i]);
    }
}

static void vec_log2_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlog2f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = log2f(a[i]);
    }
}

static void vec_sqrt_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xsqrt(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sqrt(a[i]);
    }
}

static void vec_sqrt_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xsqrtf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sqrtf(a[i]);
    }
}

static void vec_sinh_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xsinh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinh(a[i]);
    }
}

static void vec_sinh_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xsinhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinhf(a[i]);
    }
}

static void vec_cosh_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xcosh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosh(a[i]);
    }
}

static void vec_cosh_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xcoshf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = coshf(a[i]);
    }
}

static void vec_tanh_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xtanh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanh(a[i]);
    }
}

static void vec_tanh_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xtanhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tanhf(a[i]);
    }
}

static void vec_acosh_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xacosh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acosh(a[i]);
    }
}

static void vec_acosh_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xacoshf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = acoshf(a[i]);
    }
}

static void vec_asinh_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xasinh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinh(a[i]);
    }
}

static void vec_asinh_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xasinhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = asinhf(a[i]);
    }
}

static void vec_atanh_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xatanh(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanh(a[i]);
    }
}

static void vec_atanh_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xatanhf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = atanhf(a[i]);
    }
}

static void vec_ceil_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xceil(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ceil(a[i]);
    }
}

static void vec_ceil_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xceilf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ceilf(a[i]);
    }
}

static void vec_floor_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xfloor(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = floor(a[i]);
    }
}

static void vec_floor_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xfloorf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = floorf(a[i]);
    }
}

static void vec_round_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xround(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = round(a[i]);
    }
}

static void vec_round_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xroundf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = roundf(a[i]);
    }
}

static void vec_trunc_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xtrunc(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = trunc(a[i]);
    }
}

static void vec_trunc_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xtruncf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = truncf(a[i]);
    }
}

static void vec_pow_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xpow(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = pow(a[i], b[i]);
    }
}

static void vec_pow_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xpowf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = powf(a[i], b[i]);
    }
}

static void vec_exp2_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xexp2_u35(v) : xexp2(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = exp2(a[i]);
    }
}

static void vec_exp2_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xexp2f_u35(v) : xexp2f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = exp2f(a[i]);
    }
}

static void vec_exp10_advsimd(const double* a, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = use_u35 ? xexp10_u35(v) : xexp10(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = pow(10.0, a[i]);
    }
}

static void vec_exp10_f32_advsimd(const float* a, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = use_u35 ? xexp10f_u35(v) : xexp10f(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = powf(10.0f, a[i]);
    }
}

static void vec_cbrt_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xcbrt_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cbrt(a[i]);
    }
}

static void vec_cbrt_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xcbrtf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cbrtf(a[i]);
    }
}

static void vec_erf_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xerf_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erf(a[i]);
    }
}

static void vec_erf_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xerff_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erff(a[i]);
    }
}

static void vec_erfc_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xerfc_u15(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erfc(a[i]);
    }
}

static void vec_erfc_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xerfcf_u15(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = erfcf(a[i]);
    }
}

static void vec_sinpi_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xsinpi_u05(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sin(me_pi * a[i]);
    }
}

static void vec_sinpi_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xsinpif_u05(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = sinf(me_pif * a[i]);
    }
}

static void vec_cospi_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xcospi_u05(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cos(me_pi * a[i]);
    }
}

static void vec_cospi_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xcospif_u05(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = cosf(me_pif * a[i]);
    }
}

static void vec_tgamma_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xtgamma_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tgamma(a[i]);
    }
}

static void vec_tgamma_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xtgammaf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = tgammaf(a[i]);
    }
}

static void vec_lgamma_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xlgamma_u1(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = lgamma(a[i]);
    }
}

static void vec_lgamma_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xlgammaf_u1(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = lgammaf(a[i]);
    }
}

static void vec_rint_advsimd(const double* a, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble v = vloadu_vd_p(a + i);
        vdouble r = xrint(v);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = rint(a[i]);
    }
}

static void vec_rint_f32_advsimd(const float* a, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat v = vloadu_vf_p(a + i);
        vfloat r = xrintf(v);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = rintf(a[i]);
    }
}

static void vec_copysign_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xcopysign(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = copysign(a[i], b[i]);
    }
}

static void vec_copysign_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xcopysignf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = copysignf(a[i], b[i]);
    }
}

static void vec_fdim_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfdim(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fdim(a[i], b[i]);
    }
}

static void vec_fdim_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfdimf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fdimf(a[i], b[i]);
    }
}

static void vec_fmax_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfmax(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmax(a[i], b[i]);
    }
}

static void vec_fmax_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfmaxf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmaxf(a[i], b[i]);
    }
}

static void vec_fmin_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfmin(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmin(a[i], b[i]);
    }
}

static void vec_fmin_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfminf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fminf(a[i], b[i]);
    }
}

static void vec_fmod_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xfmod(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmod(a[i], b[i]);
    }
}

static void vec_fmod_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xfmodf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmodf(a[i], b[i]);
    }
}

static void vec_hypot_advsimd(const double* a, const double* b, double* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = use_u35 ? xhypot_u35(va, vb) : xhypot_u05(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = hypot(a[i], b[i]);
    }
}

static void vec_hypot_f32_advsimd(const float* a, const float* b, float* out, int n) {
    const int use_u35 = me_simd_use_u35_active();
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = use_u35 ? xhypotf_u35(va, vb) : xhypotf_u05(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = hypotf(a[i], b[i]);
    }
}

static void vec_ldexp_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble vb_int = xtrunc(vb);
        vint q = vrint_vi_vd(vb_int);
        vdouble r = xldexp(va, q);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ldexp(a[i], (int)b[i]);
    }
}

static void vec_ldexp_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat vb_int = xtruncf(vb);
        vint2 q = vrint_vi2_vf(vb_int);
        vfloat r = xldexpf(va, q);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = ldexpf(a[i], (int)b[i]);
    }
}

static void vec_nextafter_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xnextafter(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = nextafter(a[i], b[i]);
    }
}

static void vec_nextafter_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xnextafterf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = nextafterf(a[i], b[i]);
    }
}

static void vec_remainder_advsimd(const double* a, const double* b, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble r = xremainder(va, vb);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = remainder(a[i], b[i]);
    }
}

static void vec_remainder_f32_advsimd(const float* a, const float* b, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat r = xremainderf(va, vb);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = remainderf(a[i], b[i]);
    }
}

static void vec_fma_advsimd(const double* a, const double* b, const double* c, double* out, int n) {
    int i = 0;
    const int limit = n & ~1;
    for (; i < limit; i += 2) {
        vdouble va = vloadu_vd_p(a + i);
        vdouble vb = vloadu_vd_p(b + i);
        vdouble vc = vloadu_vd_p(c + i);
        vdouble r = xfma(va, vb, vc);
        vstoreu_v_p_vd(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fma(a[i], b[i], c[i]);
    }
}

static void vec_fma_f32_advsimd(const float* a, const float* b, const float* c, float* out, int n) {
    int i = 0;
    const int limit = n & ~3;
    for (; i < limit; i += 4) {
        vfloat va = vloadu_vf_p(a + i);
        vfloat vb = vloadu_vf_p(b + i);
        vfloat vc = vloadu_vf_p(c + i);
        vfloat r = xfmaf(va, vb, vc);
        vstoreu_v_p_vf(out + i, r);
    }
    for (; i < n; i++) {
        out[i] = fmaf(a[i], b[i], c[i]);
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
static me_vec_unary_f64 vec_abs_impl = vec_abs_scalar;
static me_vec_unary_f64 vec_exp_impl = vec_exp_scalar;
static me_vec_unary_f64 vec_expm1_impl = vec_expm1_scalar;
static me_vec_unary_f64 vec_log_impl = vec_log_scalar;
static me_vec_unary_f64 vec_log10_impl = vec_log10_scalar;
static me_vec_unary_f64 vec_log1p_impl = vec_log1p_scalar;
static me_vec_unary_f64 vec_log2_impl = vec_log2_scalar;
static me_vec_unary_f64 vec_sqrt_impl = vec_sqrt_scalar;
static me_vec_unary_f64 vec_sinh_impl = vec_sinh_scalar;
static me_vec_unary_f64 vec_cosh_impl = vec_cosh_scalar;
static me_vec_unary_f64 vec_tanh_impl = vec_tanh_scalar;
static me_vec_unary_f64 vec_acosh_impl = vec_acosh_scalar;
static me_vec_unary_f64 vec_asinh_impl = vec_asinh_scalar;
static me_vec_unary_f64 vec_atanh_impl = vec_atanh_scalar;
static me_vec_unary_f64 vec_ceil_impl = vec_ceil_scalar;
static me_vec_unary_f64 vec_floor_impl = vec_floor_scalar;
static me_vec_unary_f64 vec_round_impl = vec_round_scalar;
static me_vec_unary_f64 vec_trunc_impl = vec_trunc_scalar;
static me_vec_unary_f64 vec_exp2_impl = vec_exp2_scalar;
static me_vec_unary_f64 vec_exp10_impl = vec_exp10_scalar;
static me_vec_unary_f64 vec_cbrt_impl = vec_cbrt_scalar;
static me_vec_unary_f64 vec_erf_impl = vec_erf_scalar;
static me_vec_unary_f64 vec_erfc_impl = vec_erfc_scalar;
static me_vec_unary_f64 vec_sinpi_impl = vec_sinpi_scalar;
static me_vec_unary_f64 vec_cospi_impl = vec_cospi_scalar;
static me_vec_unary_f64 vec_tgamma_impl = vec_tgamma_scalar;
static me_vec_unary_f64 vec_lgamma_impl = vec_lgamma_scalar;
static me_vec_unary_f64 vec_rint_impl = vec_rint_scalar;
static me_vec_binary_f64 vec_pow_impl = vec_pow_scalar;
static me_vec_binary_f64 vec_copysign_impl = vec_copysign_scalar;
static me_vec_binary_f64 vec_fdim_impl = vec_fdim_scalar;
static me_vec_binary_f64 vec_fmax_impl = vec_fmax_scalar;
static me_vec_binary_f64 vec_fmin_impl = vec_fmin_scalar;
static me_vec_binary_f64 vec_fmod_impl = vec_fmod_scalar;
static me_vec_binary_f64 vec_hypot_impl = vec_hypot_scalar;
static me_vec_binary_f64 vec_ldexp_impl = vec_ldexp_scalar;
static me_vec_binary_f64 vec_nextafter_impl = vec_nextafter_scalar;
static me_vec_binary_f64 vec_remainder_impl = vec_remainder_scalar;
static me_vec_ternary_f64 vec_fma_impl = vec_fma_scalar;
static me_vec_unary_f32 vec_sin_f32_impl = vec_sin_f32_scalar;
static me_vec_unary_f32 vec_cos_f32_impl = vec_cos_f32_scalar;
static me_vec_unary_f32 vec_tan_f32_impl = vec_tan_f32_scalar;
static me_vec_unary_f32 vec_asin_f32_impl = vec_asin_f32_scalar;
static me_vec_unary_f32 vec_acos_f32_impl = vec_acos_f32_scalar;
static me_vec_unary_f32 vec_atan_f32_impl = vec_atan_f32_scalar;
static me_vec_binary_f32 vec_atan2_f32_impl = vec_atan2_f32_scalar;
static me_vec_unary_f32 vec_abs_f32_impl = vec_abs_f32_scalar;
static me_vec_unary_f32 vec_exp_f32_impl = vec_exp_f32_scalar;
static me_vec_unary_f32 vec_expm1_f32_impl = vec_expm1_f32_scalar;
static me_vec_unary_f32 vec_log_f32_impl = vec_log_f32_scalar;
static me_vec_unary_f32 vec_log10_f32_impl = vec_log10_f32_scalar;
static me_vec_unary_f32 vec_log1p_f32_impl = vec_log1p_f32_scalar;
static me_vec_unary_f32 vec_log2_f32_impl = vec_log2_f32_scalar;
static me_vec_unary_f32 vec_sqrt_f32_impl = vec_sqrt_f32_scalar;
static me_vec_unary_f32 vec_sinh_f32_impl = vec_sinh_f32_scalar;
static me_vec_unary_f32 vec_cosh_f32_impl = vec_cosh_f32_scalar;
static me_vec_unary_f32 vec_tanh_f32_impl = vec_tanh_f32_scalar;
static me_vec_unary_f32 vec_acosh_f32_impl = vec_acosh_f32_scalar;
static me_vec_unary_f32 vec_asinh_f32_impl = vec_asinh_f32_scalar;
static me_vec_unary_f32 vec_atanh_f32_impl = vec_atanh_f32_scalar;
static me_vec_unary_f32 vec_ceil_f32_impl = vec_ceil_f32_scalar;
static me_vec_unary_f32 vec_floor_f32_impl = vec_floor_f32_scalar;
static me_vec_unary_f32 vec_round_f32_impl = vec_round_f32_scalar;
static me_vec_unary_f32 vec_trunc_f32_impl = vec_trunc_f32_scalar;
static me_vec_unary_f32 vec_exp2_f32_impl = vec_exp2_f32_scalar;
static me_vec_unary_f32 vec_exp10_f32_impl = vec_exp10_f32_scalar;
static me_vec_unary_f32 vec_cbrt_f32_impl = vec_cbrt_f32_scalar;
static me_vec_unary_f32 vec_erf_f32_impl = vec_erf_f32_scalar;
static me_vec_unary_f32 vec_erfc_f32_impl = vec_erfc_f32_scalar;
static me_vec_unary_f32 vec_sinpi_f32_impl = vec_sinpi_f32_scalar;
static me_vec_unary_f32 vec_cospi_f32_impl = vec_cospi_f32_scalar;
static me_vec_unary_f32 vec_tgamma_f32_impl = vec_tgamma_f32_scalar;
static me_vec_unary_f32 vec_lgamma_f32_impl = vec_lgamma_f32_scalar;
static me_vec_unary_f32 vec_rint_f32_impl = vec_rint_f32_scalar;
static me_vec_binary_f32 vec_pow_f32_impl = vec_pow_f32_scalar;
static me_vec_binary_f32 vec_copysign_f32_impl = vec_copysign_f32_scalar;
static me_vec_binary_f32 vec_fdim_f32_impl = vec_fdim_f32_scalar;
static me_vec_binary_f32 vec_fmax_f32_impl = vec_fmax_f32_scalar;
static me_vec_binary_f32 vec_fmin_f32_impl = vec_fmin_f32_scalar;
static me_vec_binary_f32 vec_fmod_f32_impl = vec_fmod_f32_scalar;
static me_vec_binary_f32 vec_hypot_f32_impl = vec_hypot_f32_scalar;
static me_vec_binary_f32 vec_ldexp_f32_impl = vec_ldexp_f32_scalar;
static me_vec_binary_f32 vec_nextafter_f32_impl = vec_nextafter_f32_scalar;
static me_vec_binary_f32 vec_remainder_f32_impl = vec_remainder_f32_scalar;
static me_vec_ternary_f32 vec_fma_f32_impl = vec_fma_f32_scalar;
static me_vec_sincos_f64 vec_sincos_impl = vec_sincos_scalar;
static me_vec_sincos_f32 vec_sincos_f32_impl = vec_sincos_f32_scalar;
static int me_simd_initialized = 0;
static int me_simd_enabled = 1;
static const char *me_simd_backend = "scalar";

static int me_dsl_trace_enabled(void) {
    const char *env = getenv("ME_DSL_TRACE");
    if (!env || env[0] == '\0') {
        return (ME_DSL_TRACE_DEFAULT != 0) ? 1 : 0;
    }
    return (strcmp(env, "0") != 0) ? 1 : 0;
}

static void me_dsl_tracef(const char *fmt, ...) {
    if (!fmt || !me_dsl_trace_enabled()) {
        return;
    }
    fprintf(stderr, "[me-dsl] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void me_dsl_trace_simd_init(int use_u35) {
    me_dsl_tracef("simd init: backend=%s sleef=%s simd=%s ulp=%s",
                  me_simd_backend,
#if ME_USE_SLEEF
                  "on",
#else
                  "off",
#endif
#if ME_ENABLE_SLEEF_SIMD
                  "on",
#else
                  "off",
#endif
                  use_u35 ? "u35" : "u10");
}

void me_sincos_eval_start(void) {
    me_eval_cookie++;
}

void vec_sin_cached(const double* a, double* out, int n) {
    if (!a || !out || n <= 0) {
        return;
    }
    if (me_simd_force_scalar) {
        vec_sin_scalar(a, out, n);
        return;
    }
    me_init_simd();
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
    if (me_simd_force_scalar) {
        vec_cos_scalar(a, out, n);
        return;
    }
    me_init_simd();
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
    if (me_simd_force_scalar) {
        vec_sin_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
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
    if (me_simd_force_scalar) {
        vec_cos_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
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

#if ME_ENABLE_SLEEF_SIMD && (defined(__aarch64__) || defined(_M_ARM64))
static int me_cpu_supports_advsimd(void) {
    return 1;
}
#endif

static void me_init_simd(void) {
    const int use_u35 = me_simd_use_u35_active();
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
        vec_abs_impl = vec_abs_scalar;
        vec_exp_impl = vec_exp_scalar;
        vec_expm1_impl = vec_expm1_scalar;
        vec_log_impl = vec_log_scalar;
        vec_log10_impl = vec_log10_scalar;
        vec_log1p_impl = vec_log1p_scalar;
        vec_log2_impl = vec_log2_scalar;
        vec_sqrt_impl = vec_sqrt_scalar;
        vec_sinh_impl = vec_sinh_scalar;
        vec_cosh_impl = vec_cosh_scalar;
        vec_tanh_impl = vec_tanh_scalar;
        vec_acosh_impl = vec_acosh_scalar;
        vec_asinh_impl = vec_asinh_scalar;
        vec_atanh_impl = vec_atanh_scalar;
        vec_ceil_impl = vec_ceil_scalar;
        vec_floor_impl = vec_floor_scalar;
        vec_round_impl = vec_round_scalar;
        vec_trunc_impl = vec_trunc_scalar;
        vec_exp2_impl = vec_exp2_scalar;
        vec_exp10_impl = vec_exp10_scalar;
        vec_cbrt_impl = vec_cbrt_scalar;
        vec_erf_impl = vec_erf_scalar;
        vec_erfc_impl = vec_erfc_scalar;
        vec_sinpi_impl = vec_sinpi_scalar;
        vec_cospi_impl = vec_cospi_scalar;
        vec_tgamma_impl = vec_tgamma_scalar;
        vec_lgamma_impl = vec_lgamma_scalar;
        vec_rint_impl = vec_rint_scalar;
        vec_pow_impl = vec_pow_scalar;
        vec_copysign_impl = vec_copysign_scalar;
        vec_fdim_impl = vec_fdim_scalar;
        vec_fmax_impl = vec_fmax_scalar;
        vec_fmin_impl = vec_fmin_scalar;
        vec_fmod_impl = vec_fmod_scalar;
        vec_hypot_impl = vec_hypot_scalar;
        vec_ldexp_impl = vec_ldexp_scalar;
        vec_nextafter_impl = vec_nextafter_scalar;
        vec_remainder_impl = vec_remainder_scalar;
        vec_fma_impl = vec_fma_scalar;
        vec_sin_f32_impl = vec_sin_f32_scalar;
        vec_cos_f32_impl = vec_cos_f32_scalar;
        vec_tan_f32_impl = vec_tan_f32_scalar;
        vec_asin_f32_impl = vec_asin_f32_scalar;
        vec_acos_f32_impl = vec_acos_f32_scalar;
        vec_atan_f32_impl = vec_atan_f32_scalar;
        vec_atan2_f32_impl = vec_atan2_f32_scalar;
        vec_abs_f32_impl = vec_abs_f32_scalar;
        vec_exp_f32_impl = vec_exp_f32_scalar;
        vec_expm1_f32_impl = vec_expm1_f32_scalar;
        vec_log_f32_impl = vec_log_f32_scalar;
        vec_log10_f32_impl = vec_log10_f32_scalar;
        vec_log1p_f32_impl = vec_log1p_f32_scalar;
        vec_log2_f32_impl = vec_log2_f32_scalar;
        vec_sqrt_f32_impl = vec_sqrt_f32_scalar;
        vec_sinh_f32_impl = vec_sinh_f32_scalar;
        vec_cosh_f32_impl = vec_cosh_f32_scalar;
        vec_tanh_f32_impl = vec_tanh_f32_scalar;
        vec_acosh_f32_impl = vec_acosh_f32_scalar;
        vec_asinh_f32_impl = vec_asinh_f32_scalar;
        vec_atanh_f32_impl = vec_atanh_f32_scalar;
        vec_ceil_f32_impl = vec_ceil_f32_scalar;
        vec_floor_f32_impl = vec_floor_f32_scalar;
        vec_round_f32_impl = vec_round_f32_scalar;
        vec_trunc_f32_impl = vec_trunc_f32_scalar;
        vec_exp2_f32_impl = vec_exp2_f32_scalar;
        vec_exp10_f32_impl = vec_exp10_f32_scalar;
        vec_cbrt_f32_impl = vec_cbrt_f32_scalar;
        vec_erf_f32_impl = vec_erf_f32_scalar;
        vec_erfc_f32_impl = vec_erfc_f32_scalar;
        vec_sinpi_f32_impl = vec_sinpi_f32_scalar;
        vec_cospi_f32_impl = vec_cospi_f32_scalar;
        vec_tgamma_f32_impl = vec_tgamma_f32_scalar;
        vec_lgamma_f32_impl = vec_lgamma_f32_scalar;
        vec_rint_f32_impl = vec_rint_f32_scalar;
        vec_pow_f32_impl = vec_pow_f32_scalar;
        vec_copysign_f32_impl = vec_copysign_f32_scalar;
        vec_fdim_f32_impl = vec_fdim_f32_scalar;
        vec_fmax_f32_impl = vec_fmax_f32_scalar;
        vec_fmin_f32_impl = vec_fmin_f32_scalar;
        vec_fmod_f32_impl = vec_fmod_f32_scalar;
        vec_hypot_f32_impl = vec_hypot_f32_scalar;
        vec_ldexp_f32_impl = vec_ldexp_f32_scalar;
        vec_nextafter_f32_impl = vec_nextafter_f32_scalar;
        vec_remainder_f32_impl = vec_remainder_f32_scalar;
        vec_fma_f32_impl = vec_fma_f32_scalar;
        vec_sincos_impl = vec_sincos_scalar;
        vec_sincos_f32_impl = vec_sincos_f32_scalar;
        me_simd_backend = "scalar";
        me_dsl_trace_simd_init(use_u35);
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
        vec_abs_impl = vec_abs_avx2;
        vec_exp_impl = vec_exp_avx2;
        vec_expm1_impl = vec_expm1_avx2;
        vec_log_impl = vec_log_avx2;
        vec_log10_impl = vec_log10_avx2;
        vec_log1p_impl = vec_log1p_avx2;
        vec_log2_impl = vec_log2_avx2;
        vec_sqrt_impl = vec_sqrt_avx2;
        vec_sinh_impl = vec_sinh_avx2;
        vec_cosh_impl = vec_cosh_avx2;
        vec_tanh_impl = vec_tanh_avx2;
        vec_acosh_impl = vec_acosh_avx2;
        vec_asinh_impl = vec_asinh_avx2;
        vec_atanh_impl = vec_atanh_avx2;
        vec_ceil_impl = vec_ceil_avx2;
        vec_floor_impl = vec_floor_avx2;
        vec_round_impl = vec_round_avx2;
        vec_trunc_impl = vec_trunc_avx2;
        vec_exp2_impl = vec_exp2_avx2;
        vec_exp10_impl = vec_exp10_avx2;
        vec_cbrt_impl = vec_cbrt_avx2;
        vec_erf_impl = vec_erf_avx2;
        vec_erfc_impl = vec_erfc_avx2;
        vec_sinpi_impl = vec_sinpi_avx2;
        vec_cospi_impl = vec_cospi_avx2;
        vec_tgamma_impl = vec_tgamma_avx2;
        vec_lgamma_impl = vec_lgamma_avx2;
        vec_rint_impl = vec_rint_avx2;
        vec_pow_impl = vec_pow_avx2;
        vec_copysign_impl = vec_copysign_avx2;
        vec_fdim_impl = vec_fdim_avx2;
        vec_fmax_impl = vec_fmax_avx2;
        vec_fmin_impl = vec_fmin_avx2;
        vec_fmod_impl = vec_fmod_avx2;
        vec_hypot_impl = vec_hypot_avx2;
        vec_ldexp_impl = vec_ldexp_avx2;
        vec_nextafter_impl = vec_nextafter_avx2;
#ifdef _WIN32
        /* Windows SIMD nextafter diverges in edge cases; keep scalar for determinism. */
        vec_nextafter_impl = vec_nextafter_scalar;
#endif
        vec_remainder_impl = vec_remainder_avx2;
        vec_fma_impl = vec_fma_avx2;
        vec_sin_f32_impl = vec_sin_f32_avx2;
        vec_cos_f32_impl = vec_cos_f32_avx2;
        vec_tan_f32_impl = vec_tan_f32_avx2;
        vec_asin_f32_impl = vec_asin_f32_avx2;
        vec_acos_f32_impl = vec_acos_f32_avx2;
        vec_atan_f32_impl = vec_atan_f32_avx2;
        vec_atan2_f32_impl = vec_atan2_f32_avx2;
        vec_abs_f32_impl = vec_abs_f32_avx2;
        vec_exp_f32_impl = vec_exp_f32_avx2;
        vec_expm1_f32_impl = vec_expm1_f32_avx2;
        vec_log_f32_impl = vec_log_f32_avx2;
        vec_log10_f32_impl = vec_log10_f32_avx2;
        vec_log1p_f32_impl = vec_log1p_f32_avx2;
        vec_log2_f32_impl = vec_log2_f32_avx2;
        vec_sqrt_f32_impl = vec_sqrt_f32_avx2;
        vec_sinh_f32_impl = vec_sinh_f32_avx2;
        vec_cosh_f32_impl = vec_cosh_f32_avx2;
        vec_tanh_f32_impl = vec_tanh_f32_avx2;
        vec_acosh_f32_impl = vec_acosh_f32_avx2;
        vec_asinh_f32_impl = vec_asinh_f32_avx2;
        vec_atanh_f32_impl = vec_atanh_f32_avx2;
        vec_ceil_f32_impl = vec_ceil_f32_avx2;
        vec_floor_f32_impl = vec_floor_f32_avx2;
        vec_round_f32_impl = vec_round_f32_avx2;
        vec_trunc_f32_impl = vec_trunc_f32_avx2;
        vec_exp2_f32_impl = vec_exp2_f32_avx2;
        vec_exp10_f32_impl = vec_exp10_f32_avx2;
        vec_cbrt_f32_impl = vec_cbrt_f32_avx2;
        vec_erf_f32_impl = vec_erf_f32_avx2;
        vec_erfc_f32_impl = vec_erfc_f32_avx2;
        vec_sinpi_f32_impl = vec_sinpi_f32_avx2;
        vec_cospi_f32_impl = vec_cospi_f32_avx2;
        vec_tgamma_f32_impl = vec_tgamma_f32_avx2;
        vec_lgamma_f32_impl = vec_lgamma_f32_avx2;
        vec_rint_f32_impl = vec_rint_f32_avx2;
        vec_pow_f32_impl = vec_pow_f32_avx2;
        vec_copysign_f32_impl = vec_copysign_f32_avx2;
        vec_fdim_f32_impl = vec_fdim_f32_avx2;
        vec_fmax_f32_impl = vec_fmax_f32_avx2;
        vec_fmin_f32_impl = vec_fmin_f32_avx2;
        vec_fmod_f32_impl = vec_fmod_f32_avx2;
        vec_hypot_f32_impl = vec_hypot_f32_avx2;
        vec_ldexp_f32_impl = vec_ldexp_f32_avx2;
        vec_nextafter_f32_impl = vec_nextafter_f32_avx2;
#ifdef _WIN32
        /* Windows SIMD nextafter diverges in edge cases; keep scalar for determinism. */
        vec_nextafter_f32_impl = vec_nextafter_f32_scalar;
#endif
        vec_remainder_f32_impl = vec_remainder_f32_avx2;
        vec_fma_f32_impl = vec_fma_f32_avx2;
        vec_sincos_impl = vec_sincos_avx2;
        vec_sincos_f32_impl = vec_sincos_f32_avx2;
        me_simd_backend = use_u35 ? "avx2-u35" : "avx2-u10";
        me_dsl_trace_simd_init(use_u35);
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
        vec_abs_impl = vec_abs_advsimd;
        vec_exp_impl = vec_exp_advsimd;
        vec_expm1_impl = vec_expm1_advsimd;
        vec_log_impl = vec_log_advsimd;
        vec_log10_impl = vec_log10_advsimd;
        vec_log1p_impl = vec_log1p_advsimd;
        vec_log2_impl = vec_log2_advsimd;
        vec_sqrt_impl = vec_sqrt_advsimd;
        vec_sinh_impl = vec_sinh_advsimd;
        vec_cosh_impl = vec_cosh_advsimd;
        vec_tanh_impl = vec_tanh_advsimd;
        vec_acosh_impl = vec_acosh_advsimd;
        vec_asinh_impl = vec_asinh_advsimd;
        vec_atanh_impl = vec_atanh_advsimd;
        vec_ceil_impl = vec_ceil_advsimd;
        vec_floor_impl = vec_floor_advsimd;
        vec_round_impl = vec_round_advsimd;
        vec_trunc_impl = vec_trunc_advsimd;
        vec_exp2_impl = vec_exp2_advsimd;
        vec_exp10_impl = vec_exp10_advsimd;
        vec_cbrt_impl = vec_cbrt_advsimd;
        vec_erf_impl = vec_erf_advsimd;
        vec_erfc_impl = vec_erfc_advsimd;
        vec_sinpi_impl = vec_sinpi_advsimd;
        vec_cospi_impl = vec_cospi_advsimd;
        vec_tgamma_impl = vec_tgamma_advsimd;
        vec_lgamma_impl = vec_lgamma_advsimd;
        vec_rint_impl = vec_rint_advsimd;
        vec_pow_impl = vec_pow_advsimd;
        vec_copysign_impl = vec_copysign_advsimd;
        vec_fdim_impl = vec_fdim_advsimd;
        vec_fmax_impl = vec_fmax_advsimd;
        vec_fmin_impl = vec_fmin_advsimd;
        vec_fmod_impl = vec_fmod_advsimd;
        vec_hypot_impl = vec_hypot_advsimd;
        vec_ldexp_impl = vec_ldexp_advsimd;
        vec_nextafter_impl = vec_nextafter_advsimd;
#ifdef _WIN32
        /* Windows SIMD nextafter diverges in edge cases; keep scalar for determinism. */
        vec_nextafter_impl = vec_nextafter_scalar;
#endif
        vec_remainder_impl = vec_remainder_advsimd;
        vec_fma_impl = vec_fma_advsimd;
        vec_sin_f32_impl = vec_sin_f32_advsimd;
        vec_cos_f32_impl = vec_cos_f32_advsimd;
        vec_tan_f32_impl = vec_tan_f32_advsimd;
        vec_asin_f32_impl = vec_asin_f32_advsimd;
        vec_acos_f32_impl = vec_acos_f32_advsimd;
        vec_atan_f32_impl = vec_atan_f32_advsimd;
        vec_atan2_f32_impl = vec_atan2_f32_advsimd;
        vec_abs_f32_impl = vec_abs_f32_advsimd;
        vec_exp_f32_impl = vec_exp_f32_advsimd;
        vec_expm1_f32_impl = vec_expm1_f32_advsimd;
        vec_log_f32_impl = vec_log_f32_advsimd;
        vec_log10_f32_impl = vec_log10_f32_advsimd;
        vec_log1p_f32_impl = vec_log1p_f32_advsimd;
        vec_log2_f32_impl = vec_log2_f32_advsimd;
        vec_sqrt_f32_impl = vec_sqrt_f32_advsimd;
        vec_sinh_f32_impl = vec_sinh_f32_advsimd;
        vec_cosh_f32_impl = vec_cosh_f32_advsimd;
        vec_tanh_f32_impl = vec_tanh_f32_advsimd;
        vec_acosh_f32_impl = vec_acosh_f32_advsimd;
        vec_asinh_f32_impl = vec_asinh_f32_advsimd;
        vec_atanh_f32_impl = vec_atanh_f32_advsimd;
        vec_ceil_f32_impl = vec_ceil_f32_advsimd;
        vec_floor_f32_impl = vec_floor_f32_advsimd;
        vec_round_f32_impl = vec_round_f32_advsimd;
        vec_trunc_f32_impl = vec_trunc_f32_advsimd;
        vec_exp2_f32_impl = vec_exp2_f32_advsimd;
        vec_exp10_f32_impl = vec_exp10_f32_advsimd;
        vec_cbrt_f32_impl = vec_cbrt_f32_advsimd;
        vec_erf_f32_impl = vec_erf_f32_advsimd;
        vec_erfc_f32_impl = vec_erfc_f32_advsimd;
        vec_sinpi_f32_impl = vec_sinpi_f32_advsimd;
        vec_cospi_f32_impl = vec_cospi_f32_advsimd;
        vec_tgamma_f32_impl = vec_tgamma_f32_advsimd;
        vec_lgamma_f32_impl = vec_lgamma_f32_advsimd;
        vec_rint_f32_impl = vec_rint_f32_advsimd;
        vec_pow_f32_impl = vec_pow_f32_advsimd;
        vec_copysign_f32_impl = vec_copysign_f32_advsimd;
        vec_fdim_f32_impl = vec_fdim_f32_advsimd;
        vec_fmax_f32_impl = vec_fmax_f32_advsimd;
        vec_fmin_f32_impl = vec_fmin_f32_advsimd;
        vec_fmod_f32_impl = vec_fmod_f32_advsimd;
        vec_hypot_f32_impl = vec_hypot_f32_advsimd;
        vec_ldexp_f32_impl = vec_ldexp_f32_advsimd;
        vec_nextafter_f32_impl = vec_nextafter_f32_advsimd;
#ifdef _WIN32
        /* Windows SIMD nextafter diverges in edge cases; keep scalar for determinism. */
        vec_nextafter_f32_impl = vec_nextafter_f32_scalar;
#endif
        vec_remainder_f32_impl = vec_remainder_f32_advsimd;
        vec_fma_f32_impl = vec_fma_f32_advsimd;
        vec_sincos_impl = vec_sincos_advsimd;
        vec_sincos_f32_impl = vec_sincos_f32_advsimd;
        me_simd_backend = use_u35 ? "advsimd-u35" : "advsimd-u10";
    }
#endif

    if (me_simd_backend[0] == '\0') {
        me_simd_backend = "scalar";
    }
    me_dsl_trace_simd_init(use_u35);
}

int me_simd_initialized_for_tests(void) {
    return me_simd_initialized;
}

void me_simd_reset_for_tests(void) {
    me_simd_initialized = 0;
    me_simd_backend = "scalar";
}

void vec_sin_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_sin_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sin_impl(a, out, n);
}

void vec_cos_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_cos_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cos_impl(a, out, n);
}

void vec_asin_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_asin_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_asin_impl(a, out, n);
}

void vec_acos_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_acos_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_acos_impl(a, out, n);
}

void vec_atan_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_atan_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_atan_impl(a, out, n);
}

void vec_tan_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_tan_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_tan_impl(a, out, n);
}

void vec_atan2_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_atan2_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_atan2_impl(a, b, out, n);
}

void vec_sin_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_sin_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sin_f32_impl(a, out, n);
}

void vec_cos_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_cos_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cos_f32_impl(a, out, n);
}

void vec_asin_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_asin_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_asin_f32_impl(a, out, n);
}

void vec_acos_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_acos_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_acos_f32_impl(a, out, n);
}

void vec_atan_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_atan_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_atan_f32_impl(a, out, n);
}

void vec_tan_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_tan_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_tan_f32_impl(a, out, n);
}

void vec_atan2_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_atan2_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_atan2_f32_impl(a, b, out, n);
}

void vec_abs_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_abs_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_abs_impl(a, out, n);
}

void vec_exp_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_exp_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_exp_impl(a, out, n);
}

void vec_expm1_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_expm1_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_expm1_impl(a, out, n);
}

void vec_log_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_log_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log_impl(a, out, n);
}

void vec_log10_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_log10_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log10_impl(a, out, n);
}

void vec_log1p_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_log1p_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log1p_impl(a, out, n);
}

void vec_log2_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_log2_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log2_impl(a, out, n);
}

void vec_sqrt_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_sqrt_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sqrt_impl(a, out, n);
}

void vec_sinh_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_sinh_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sinh_impl(a, out, n);
}

void vec_cosh_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_cosh_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cosh_impl(a, out, n);
}

void vec_tanh_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_tanh_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_tanh_impl(a, out, n);
}

void vec_acosh_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_acosh_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_acosh_impl(a, out, n);
}

void vec_asinh_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_asinh_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_asinh_impl(a, out, n);
}

void vec_atanh_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_atanh_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_atanh_impl(a, out, n);
}

void vec_ceil_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_ceil_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_ceil_impl(a, out, n);
}

void vec_floor_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_floor_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_floor_impl(a, out, n);
}

void vec_round_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_round_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_round_impl(a, out, n);
}

void vec_trunc_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_trunc_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_trunc_impl(a, out, n);
}

void vec_pow_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_pow_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_pow_impl(a, b, out, n);
}

void vec_exp2_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_exp2_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_exp2_impl(a, out, n);
}

void vec_exp10_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_exp10_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_exp10_impl(a, out, n);
}

void vec_cbrt_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_cbrt_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cbrt_impl(a, out, n);
}

void vec_erf_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_erf_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_erf_impl(a, out, n);
}

void vec_erfc_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_erfc_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_erfc_impl(a, out, n);
}

void vec_sinpi_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_sinpi_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sinpi_impl(a, out, n);
}

void vec_cospi_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_cospi_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cospi_impl(a, out, n);
}

void vec_tgamma_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_tgamma_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_tgamma_impl(a, out, n);
}

void vec_lgamma_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_lgamma_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_lgamma_impl(a, out, n);
}

void vec_rint_dispatch(const double* a, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_rint_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_rint_impl(a, out, n);
}

void vec_copysign_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_copysign_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_copysign_impl(a, b, out, n);
}

void vec_fdim_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_fdim_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fdim_impl(a, b, out, n);
}

void vec_fmax_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_fmax_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fmax_impl(a, b, out, n);
}

void vec_fmin_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_fmin_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fmin_impl(a, b, out, n);
}

void vec_fmod_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_fmod_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fmod_impl(a, b, out, n);
}

void vec_hypot_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_hypot_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_hypot_impl(a, b, out, n);
}

void vec_ldexp_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_ldexp_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_ldexp_impl(a, b, out, n);
}

void vec_nextafter_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_nextafter_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_nextafter_impl(a, b, out, n);
}

void vec_remainder_dispatch(const double* a, const double* b, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_remainder_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_remainder_impl(a, b, out, n);
}

void vec_fma_dispatch(const double* a, const double* b, const double* c, double* out, int n) {
    if (me_simd_force_scalar) {
        vec_fma_scalar(a, b, c, out, n);
        return;
    }
    me_init_simd();
    vec_fma_impl(a, b, c, out, n);
}

void vec_abs_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_abs_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_abs_f32_impl(a, out, n);
}

void vec_exp_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_exp_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_exp_f32_impl(a, out, n);
}

void vec_expm1_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_expm1_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_expm1_f32_impl(a, out, n);
}

void vec_log_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_log_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log_f32_impl(a, out, n);
}

void vec_log10_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_log10_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log10_f32_impl(a, out, n);
}

void vec_log1p_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_log1p_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log1p_f32_impl(a, out, n);
}

void vec_log2_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_log2_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_log2_f32_impl(a, out, n);
}

void vec_sqrt_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_sqrt_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sqrt_f32_impl(a, out, n);
}

void vec_sinh_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_sinh_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sinh_f32_impl(a, out, n);
}

void vec_cosh_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_cosh_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cosh_f32_impl(a, out, n);
}

void vec_tanh_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_tanh_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_tanh_f32_impl(a, out, n);
}

void vec_acosh_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_acosh_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_acosh_f32_impl(a, out, n);
}

void vec_asinh_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_asinh_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_asinh_f32_impl(a, out, n);
}

void vec_atanh_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_atanh_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_atanh_f32_impl(a, out, n);
}

void vec_ceil_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_ceil_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_ceil_f32_impl(a, out, n);
}

void vec_floor_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_floor_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_floor_f32_impl(a, out, n);
}

void vec_round_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_round_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_round_f32_impl(a, out, n);
}

void vec_trunc_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_trunc_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_trunc_f32_impl(a, out, n);
}

void vec_pow_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_pow_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_pow_f32_impl(a, b, out, n);
}

void vec_exp2_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_exp2_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_exp2_f32_impl(a, out, n);
}

void vec_exp10_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_exp10_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_exp10_f32_impl(a, out, n);
}

void vec_cbrt_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_cbrt_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cbrt_f32_impl(a, out, n);
}

void vec_erf_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_erf_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_erf_f32_impl(a, out, n);
}

void vec_erfc_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_erfc_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_erfc_f32_impl(a, out, n);
}

void vec_sinpi_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_sinpi_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_sinpi_f32_impl(a, out, n);
}

void vec_cospi_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_cospi_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_cospi_f32_impl(a, out, n);
}

void vec_tgamma_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_tgamma_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_tgamma_f32_impl(a, out, n);
}

void vec_lgamma_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_lgamma_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_lgamma_f32_impl(a, out, n);
}

void vec_rint_f32_dispatch(const float* a, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_rint_f32_scalar(a, out, n);
        return;
    }
    me_init_simd();
    vec_rint_f32_impl(a, out, n);
}

void vec_copysign_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_copysign_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_copysign_f32_impl(a, b, out, n);
}

void vec_fdim_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_fdim_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fdim_f32_impl(a, b, out, n);
}

void vec_fmax_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_fmax_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fmax_f32_impl(a, b, out, n);
}

void vec_fmin_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_fmin_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fmin_f32_impl(a, b, out, n);
}

void vec_fmod_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_fmod_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_fmod_f32_impl(a, b, out, n);
}

void vec_hypot_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_hypot_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_hypot_f32_impl(a, b, out, n);
}

void vec_ldexp_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_ldexp_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_ldexp_f32_impl(a, b, out, n);
}

void vec_nextafter_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_nextafter_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_nextafter_f32_impl(a, b, out, n);
}

void vec_remainder_f32_dispatch(const float* a, const float* b, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_remainder_f32_scalar(a, b, out, n);
        return;
    }
    me_init_simd();
    vec_remainder_f32_impl(a, b, out, n);
}

void vec_fma_f32_dispatch(const float* a, const float* b, const float* c, float* out, int n) {
    if (me_simd_force_scalar) {
        vec_fma_f32_scalar(a, b, c, out, n);
        return;
    }
    me_init_simd();
    vec_fma_f32_impl(a, b, c, out, n);
}
