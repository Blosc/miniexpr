/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2021  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  Accelerate Framework Optimizations for MiniExpr (macOS only)

  This file provides optimized implementations of vectorized operations
  using Apple's Accelerate framework on macOS.
**********************************************************************/

#ifndef MINIEXPR_ACCELERATE_H
#define MINIEXPR_ACCELERATE_H

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

/* Float64 (double) operations using vDSP and vForce */

static inline void vec_add_accel_f64(const double *a, const double *b, double *out, int n) {
    vDSP_vaddD(a, 1, b, 1, out, 1, n);
}

static inline void vec_sub_accel_f64(const double *a, const double *b, double *out, int n) {
    vDSP_vsubD(b, 1, a, 1, out, 1, n);  // Note: vDSP_vsubD computes a[i] - b[i]
}

static inline void vec_mul_accel_f64(const double *a, const double *b, double *out, int n) {
    vDSP_vmulD(a, 1, b, 1, out, 1, n);
}

static inline void vec_div_accel_f64(const double *a, const double *b, double *out, int n) {
    vDSP_vdivD(b, 1, a, 1, out, 1, n);  // Note: vDSP_vdivD computes a[i] / b[i]
}

static inline void vec_add_scalar_accel_f64(const double *a, double b, double *out, int n) {
    vDSP_vsaddD(a, 1, &b, out, 1, n);
}

static inline void vec_mul_scalar_accel_f64(const double *a, double b, double *out, int n) {
    vDSP_vsmulD(a, 1, &b, out, 1, n);
}

static inline void vec_sqrt_accel_f64(const double *a, double *out, int n) {
    int n_int = n;
    vvsqrt(out, a, &n_int);
}

static inline void vec_sin_accel_f64(const double *a, double *out, int n) {
    int n_int = n;
    vvsin(out, a, &n_int);
}

static inline void vec_cos_accel_f64(const double *a, double *out, int n) {
    int n_int = n;
    vvcos(out, a, &n_int);
}

static inline void vec_exp_accel_f64(const double *a, double *out, int n) {
    int n_int = n;
    vvexp(out, a, &n_int);
}

static inline void vec_log_accel_f64(const double *a, double *out, int n) {
    int n_int = n;
    vvlog(out, a, &n_int);
}

static inline void vec_pow_accel_f64(const double *a, const double *b, double *out, int n) {
    int n_int = n;
    vvpow(out, b, a, &n_int);  // Note: vvpow(out, exp, base, n) computes base^exp
}

static inline void vec_negate_accel_f64(const double *a, double *out, int n) {
    vDSP_vnegD(a, 1, out, 1, n);
}

/* Float32 (float) operations using vDSP and vForce */

static inline void vec_add_accel_f32(const float *a, const float *b, float *out, int n) {
    vDSP_vadd(a, 1, b, 1, out, 1, n);
}

static inline void vec_sub_accel_f32(const float *a, const float *b, float *out, int n) {
    vDSP_vsub(b, 1, a, 1, out, 1, n);
}

static inline void vec_mul_accel_f32(const float *a, const float *b, float *out, int n) {
    vDSP_vmul(a, 1, b, 1, out, 1, n);
}

static inline void vec_div_accel_f32(const float *a, const float *b, float *out, int n) {
    vDSP_vdiv(b, 1, a, 1, out, 1, n);
}

static inline void vec_add_scalar_accel_f32(const float *a, float b, float *out, int n) {
    vDSP_vsadd(a, 1, &b, out, 1, n);
}

static inline void vec_mul_scalar_accel_f32(const float *a, float b, float *out, int n) {
    vDSP_vsmul(a, 1, &b, out, 1, n);
}

static inline void vec_sqrt_accel_f32(const float *a, float *out, int n) {
    int n_int = n;
    vvsqrtf(out, a, &n_int);
}

static inline void vec_sin_accel_f32(const float *a, float *out, int n) {
    int n_int = n;
    vvsinf(out, a, &n_int);
}

static inline void vec_cos_accel_f32(const float *a, float *out, int n) {
    int n_int = n;
    vvcosf(out, a, &n_int);
}

static inline void vec_exp_accel_f32(const float *a, float *out, int n) {
    int n_int = n;
    vvexpf(out, a, &n_int);
}

static inline void vec_log_accel_f32(const float *a, float *out, int n) {
    int n_int = n;
    vvlogf(out, a, &n_int);
}

static inline void vec_pow_accel_f32(const float *a, const float *b, float *out, int n) {
    int n_int = n;
    vvpowf(out, b, a, &n_int);
}

static inline void vec_negate_accel_f32(const float *a, float *out, int n) {
    vDSP_vneg(a, 1, out, 1, n);
}

#endif /* __APPLE__ */

#endif /* MINIEXPR_ACCELERATE_H */
