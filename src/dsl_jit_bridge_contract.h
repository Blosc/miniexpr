/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_JIT_BRIDGE_CONTRACT_H
#define MINIEXPR_DSL_JIT_BRIDGE_CONTRACT_H

/* Frozen runtime math bridge symbol/signature contract (bridge ABI v1). */
#define ME_DSL_JIT_BRIDGE_SYMBOL_CONTRACT(X) \
    X(me_jit_abs, dsl_jit_bridge_abs, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_abs(double);") \
    X(me_jit_sin, dsl_jit_bridge_sin, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_sin(double);") \
    X(me_jit_cos, dsl_jit_bridge_cos, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_cos(double);") \
    X(me_jit_exp, dsl_jit_bridge_exp, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_exp(double);") \
    X(me_jit_log, dsl_jit_bridge_log, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_log(double);") \
    X(me_jit_sqrt, dsl_jit_bridge_sqrt, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_sqrt(double);") \
    X(me_jit_exp10, dsl_jit_bridge_exp10, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_exp10(double);") \
    X(me_jit_sinpi, dsl_jit_bridge_sinpi, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_sinpi(double);") \
    X(me_jit_cospi, dsl_jit_bridge_cospi, me_dsl_jit_sig_scalar_unary_fn, "extern double me_jit_cospi(double);") \
    X(me_jit_logaddexp, dsl_jit_bridge_logaddexp, me_dsl_jit_sig_scalar_binary_fn, "extern double me_jit_logaddexp(double, double);") \
    X(me_jit_where, dsl_jit_bridge_where, me_dsl_jit_sig_scalar_ternary_fn, "extern double me_jit_where(double, double, double);") \
    X(me_jit_vec_sin_f64, dsl_jit_bridge_vec_sin_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_sin_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_cos_f64, dsl_jit_bridge_vec_cos_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_cos_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_exp_f64, dsl_jit_bridge_vec_exp_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_exp_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_log_f64, dsl_jit_bridge_vec_log_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_log_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_exp10_f64, dsl_jit_bridge_vec_exp10_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_exp10_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_sinpi_f64, dsl_jit_bridge_vec_sinpi_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_sinpi_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_cospi_f64, dsl_jit_bridge_vec_cospi_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_cospi_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_atan2_f64, dsl_jit_bridge_vec_atan2_f64, me_dsl_jit_sig_vec_binary_f64_fn, "extern void me_jit_vec_atan2_f64(const double *, const double *, double *, int64_t);") \
    X(me_jit_vec_hypot_f64, dsl_jit_bridge_vec_hypot_f64, me_dsl_jit_sig_vec_binary_f64_fn, "extern void me_jit_vec_hypot_f64(const double *, const double *, double *, int64_t);") \
    X(me_jit_vec_pow_f64, dsl_jit_bridge_vec_pow_f64, me_dsl_jit_sig_vec_binary_f64_fn, "extern void me_jit_vec_pow_f64(const double *, const double *, double *, int64_t);") \
    X(me_jit_vec_fmax_f64, dsl_jit_bridge_vec_fmax_f64, me_dsl_jit_sig_vec_binary_f64_fn, "extern void me_jit_vec_fmax_f64(const double *, const double *, double *, int64_t);") \
    X(me_jit_vec_fmin_f64, dsl_jit_bridge_vec_fmin_f64, me_dsl_jit_sig_vec_binary_f64_fn, "extern void me_jit_vec_fmin_f64(const double *, const double *, double *, int64_t);") \
    X(me_jit_vec_expm1_f64, dsl_jit_bridge_vec_expm1_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_expm1_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_log10_f64, dsl_jit_bridge_vec_log10_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_log10_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_sinh_f64, dsl_jit_bridge_vec_sinh_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_sinh_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_cosh_f64, dsl_jit_bridge_vec_cosh_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_cosh_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_tanh_f64, dsl_jit_bridge_vec_tanh_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_tanh_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_asinh_f64, dsl_jit_bridge_vec_asinh_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_asinh_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_acosh_f64, dsl_jit_bridge_vec_acosh_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_acosh_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_atanh_f64, dsl_jit_bridge_vec_atanh_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_atanh_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_sin_f32, dsl_jit_bridge_vec_sin_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_sin_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_cos_f32, dsl_jit_bridge_vec_cos_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_cos_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_exp_f32, dsl_jit_bridge_vec_exp_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_exp_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_log_f32, dsl_jit_bridge_vec_log_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_log_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_exp10_f32, dsl_jit_bridge_vec_exp10_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_exp10_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_sinpi_f32, dsl_jit_bridge_vec_sinpi_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_sinpi_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_cospi_f32, dsl_jit_bridge_vec_cospi_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_cospi_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_atan2_f32, dsl_jit_bridge_vec_atan2_f32, me_dsl_jit_sig_vec_binary_f32_fn, "extern void me_jit_vec_atan2_f32(const float *, const float *, float *, int64_t);") \
    X(me_jit_vec_hypot_f32, dsl_jit_bridge_vec_hypot_f32, me_dsl_jit_sig_vec_binary_f32_fn, "extern void me_jit_vec_hypot_f32(const float *, const float *, float *, int64_t);") \
    X(me_jit_vec_pow_f32, dsl_jit_bridge_vec_pow_f32, me_dsl_jit_sig_vec_binary_f32_fn, "extern void me_jit_vec_pow_f32(const float *, const float *, float *, int64_t);") \
    X(me_jit_vec_fmax_f32, dsl_jit_bridge_vec_fmax_f32, me_dsl_jit_sig_vec_binary_f32_fn, "extern void me_jit_vec_fmax_f32(const float *, const float *, float *, int64_t);") \
    X(me_jit_vec_fmin_f32, dsl_jit_bridge_vec_fmin_f32, me_dsl_jit_sig_vec_binary_f32_fn, "extern void me_jit_vec_fmin_f32(const float *, const float *, float *, int64_t);") \
    X(me_jit_vec_expm1_f32, dsl_jit_bridge_vec_expm1_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_expm1_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_log10_f32, dsl_jit_bridge_vec_log10_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_log10_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_sinh_f32, dsl_jit_bridge_vec_sinh_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_sinh_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_cosh_f32, dsl_jit_bridge_vec_cosh_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_cosh_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_tanh_f32, dsl_jit_bridge_vec_tanh_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_tanh_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_asinh_f32, dsl_jit_bridge_vec_asinh_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_asinh_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_acosh_f32, dsl_jit_bridge_vec_acosh_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_acosh_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_atanh_f32, dsl_jit_bridge_vec_atanh_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_atanh_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_abs_f64, dsl_jit_bridge_vec_abs_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_abs_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_sqrt_f64, dsl_jit_bridge_vec_sqrt_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_sqrt_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_log1p_f64, dsl_jit_bridge_vec_log1p_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_log1p_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_exp2_f64, dsl_jit_bridge_vec_exp2_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_exp2_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_log2_f64, dsl_jit_bridge_vec_log2_f64, me_dsl_jit_sig_vec_unary_f64_fn, "extern void me_jit_vec_log2_f64(const double *, double *, int64_t);") \
    X(me_jit_vec_abs_f32, dsl_jit_bridge_vec_abs_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_abs_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_sqrt_f32, dsl_jit_bridge_vec_sqrt_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_sqrt_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_log1p_f32, dsl_jit_bridge_vec_log1p_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_log1p_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_exp2_f32, dsl_jit_bridge_vec_exp2_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_exp2_f32(const float *, float *, int64_t);") \
    X(me_jit_vec_log2_f32, dsl_jit_bridge_vec_log2_f32, me_dsl_jit_sig_vec_unary_f32_fn, "extern void me_jit_vec_log2_f32(const float *, float *, int64_t);")

#endif
