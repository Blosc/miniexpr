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

void vec_sin_f32_dispatch(const float* a, float* out, int n);
void vec_cos_f32_dispatch(const float* a, float* out, int n);
void vec_asin_f32_dispatch(const float* a, float* out, int n);
void vec_acos_f32_dispatch(const float* a, float* out, int n);
void vec_atan_f32_dispatch(const float* a, float* out, int n);
void vec_tan_f32_dispatch(const float* a, float* out, int n);
void vec_atan2_f32_dispatch(const float* a, const float* b, float* out, int n);

void vec_sin_cached(const double* a, double* out, int n);
void vec_cos_cached(const double* a, double* out, int n);
void vec_sin_f32_cached(const float* a, float* out, int n);
void vec_cos_f32_cached(const float* a, float* out, int n);

#endif
