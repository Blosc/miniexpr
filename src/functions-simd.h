/*********************************************************************
Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_FUNCTIONS_SIMD_H
#define MINIEXPR_FUNCTIONS_SIMD_H

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

void vec_sin_cached(const double* a, double* out, int n);
void vec_cos_cached(const double* a, double* out, int n);
void vec_sin_f32_cached(const float* a, float* out, int n);
void vec_cos_f32_cached(const float* a, float* out, int n);

#endif
