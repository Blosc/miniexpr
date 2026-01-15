/*********************************************************************
Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_FUNCTIONS_SIMD_H
#define MINIEXPR_FUNCTIONS_SIMD_H

#include "miniexpr.h"

typedef struct {
    int force_scalar;
    int override_u35;
} me_simd_params_state;

void me_simd_params_push(const me_eval_params *params, me_simd_params_state *state);
void me_simd_params_pop(const me_simd_params_state *state);
void me_sincos_eval_start(void);

void vec_sin_dispatch(const double* a, double* out, int n);
void vec_cos_dispatch(const double* a, double* out, int n);
void vec_asin_dispatch(const double* a, double* out, int n);
void vec_acos_dispatch(const double* a, double* out, int n);
void vec_atan_dispatch(const double* a, double* out, int n);
void vec_tan_dispatch(const double* a, double* out, int n);
void vec_atan2_dispatch(const double* a, const double* b, double* out, int n);
void vec_abs_dispatch(const double* a, double* out, int n);
void vec_exp_dispatch(const double* a, double* out, int n);
void vec_expm1_dispatch(const double* a, double* out, int n);
void vec_exp2_dispatch(const double* a, double* out, int n);
void vec_exp10_dispatch(const double* a, double* out, int n);
void vec_log_dispatch(const double* a, double* out, int n);
void vec_log10_dispatch(const double* a, double* out, int n);
void vec_log1p_dispatch(const double* a, double* out, int n);
void vec_log2_dispatch(const double* a, double* out, int n);
void vec_sqrt_dispatch(const double* a, double* out, int n);
void vec_sinh_dispatch(const double* a, double* out, int n);
void vec_cosh_dispatch(const double* a, double* out, int n);
void vec_tanh_dispatch(const double* a, double* out, int n);
void vec_acosh_dispatch(const double* a, double* out, int n);
void vec_asinh_dispatch(const double* a, double* out, int n);
void vec_atanh_dispatch(const double* a, double* out, int n);
void vec_ceil_dispatch(const double* a, double* out, int n);
void vec_floor_dispatch(const double* a, double* out, int n);
void vec_round_dispatch(const double* a, double* out, int n);
void vec_trunc_dispatch(const double* a, double* out, int n);
void vec_pow_dispatch(const double* a, const double* b, double* out, int n);
void vec_cbrt_dispatch(const double* a, double* out, int n);
void vec_erf_dispatch(const double* a, double* out, int n);
void vec_erfc_dispatch(const double* a, double* out, int n);
void vec_sinpi_dispatch(const double* a, double* out, int n);
void vec_cospi_dispatch(const double* a, double* out, int n);
void vec_tgamma_dispatch(const double* a, double* out, int n);
void vec_lgamma_dispatch(const double* a, double* out, int n);
void vec_rint_dispatch(const double* a, double* out, int n);
void vec_copysign_dispatch(const double* a, const double* b, double* out, int n);
void vec_fdim_dispatch(const double* a, const double* b, double* out, int n);
void vec_fmax_dispatch(const double* a, const double* b, double* out, int n);
void vec_fmin_dispatch(const double* a, const double* b, double* out, int n);
void vec_fmod_dispatch(const double* a, const double* b, double* out, int n);
void vec_hypot_dispatch(const double* a, const double* b, double* out, int n);
void vec_ldexp_dispatch(const double* a, const double* b, double* out, int n);
void vec_nextafter_dispatch(const double* a, const double* b, double* out, int n);
void vec_remainder_dispatch(const double* a, const double* b, double* out, int n);
void vec_fma_dispatch(const double* a, const double* b, const double* c, double* out, int n);

void vec_sin_f32_dispatch(const float* a, float* out, int n);
void vec_cos_f32_dispatch(const float* a, float* out, int n);
void vec_asin_f32_dispatch(const float* a, float* out, int n);
void vec_acos_f32_dispatch(const float* a, float* out, int n);
void vec_atan_f32_dispatch(const float* a, float* out, int n);
void vec_tan_f32_dispatch(const float* a, float* out, int n);
void vec_atan2_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_abs_f32_dispatch(const float* a, float* out, int n);
void vec_exp_f32_dispatch(const float* a, float* out, int n);
void vec_expm1_f32_dispatch(const float* a, float* out, int n);
void vec_exp2_f32_dispatch(const float* a, float* out, int n);
void vec_exp10_f32_dispatch(const float* a, float* out, int n);
void vec_log_f32_dispatch(const float* a, float* out, int n);
void vec_log10_f32_dispatch(const float* a, float* out, int n);
void vec_log1p_f32_dispatch(const float* a, float* out, int n);
void vec_log2_f32_dispatch(const float* a, float* out, int n);
void vec_sqrt_f32_dispatch(const float* a, float* out, int n);
void vec_sinh_f32_dispatch(const float* a, float* out, int n);
void vec_cosh_f32_dispatch(const float* a, float* out, int n);
void vec_tanh_f32_dispatch(const float* a, float* out, int n);
void vec_acosh_f32_dispatch(const float* a, float* out, int n);
void vec_asinh_f32_dispatch(const float* a, float* out, int n);
void vec_atanh_f32_dispatch(const float* a, float* out, int n);
void vec_ceil_f32_dispatch(const float* a, float* out, int n);
void vec_floor_f32_dispatch(const float* a, float* out, int n);
void vec_round_f32_dispatch(const float* a, float* out, int n);
void vec_trunc_f32_dispatch(const float* a, float* out, int n);
void vec_pow_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_cbrt_f32_dispatch(const float* a, float* out, int n);
void vec_erf_f32_dispatch(const float* a, float* out, int n);
void vec_erfc_f32_dispatch(const float* a, float* out, int n);
void vec_sinpi_f32_dispatch(const float* a, float* out, int n);
void vec_cospi_f32_dispatch(const float* a, float* out, int n);
void vec_tgamma_f32_dispatch(const float* a, float* out, int n);
void vec_lgamma_f32_dispatch(const float* a, float* out, int n);
void vec_rint_f32_dispatch(const float* a, float* out, int n);
void vec_copysign_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_fdim_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_fmax_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_fmin_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_fmod_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_hypot_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_ldexp_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_nextafter_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_remainder_f32_dispatch(const float* a, const float* b, float* out, int n);
void vec_fma_f32_dispatch(const float* a, const float* b, const float* c, float* out, int n);

void vec_sin_cached(const double* a, double* out, int n);
void vec_cos_cached(const double* a, double* out, int n);
void vec_sin_f32_cached(const float* a, float* out, int n);
void vec_cos_f32_cached(const float* a, float* out, int n);

#endif
