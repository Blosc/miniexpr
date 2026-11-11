/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_MATH_DISPATCH_H
#define MINIEXPR_DSL_MATH_DISPATCH_H

#include "dsl_parser.h"

#include <stdbool.h>

/* Direct Menudet/libm operations that may be redirected to an accelerated
 * library. Menudet-defined operations (sinpi/cospi, floor-mod remainder,
 * ties-even rounding, min/max) intentionally stay on their semantic helpers. */
typedef enum {
    ME_MATHOP_ACOS = 0,
    ME_MATHOP_ACOSH,
    ME_MATHOP_ASIN,
    ME_MATHOP_ASINH,
    ME_MATHOP_ATAN,
    ME_MATHOP_ATANH,
    ME_MATHOP_CBRT,
    ME_MATHOP_COS,
    ME_MATHOP_COSH,
    ME_MATHOP_ERF,
    ME_MATHOP_ERFC,
    ME_MATHOP_EXP,
    ME_MATHOP_EXP2,
    ME_MATHOP_EXPM1,
    ME_MATHOP_LGAMMA,
    ME_MATHOP_LOG,
    ME_MATHOP_LOG10,
    ME_MATHOP_LOG1P,
    ME_MATHOP_LOG2,
    ME_MATHOP_SIN,
    ME_MATHOP_SINH,
    ME_MATHOP_SQRT,
    ME_MATHOP_TAN,
    ME_MATHOP_TANH,
    ME_MATHOP_TGAMMA,
    ME_MATHOP_FABS,
    ME_MATHOP_CEIL,
    ME_MATHOP_FLOOR,
    ME_MATHOP_TRUNC,
    ME_MATHOP_ATAN2,
    ME_MATHOP_COPYSIGN,
    ME_MATHOP_FDIM,
    ME_MATHOP_HYPOT,
    ME_MATHOP_NEXTAFTER,
    ME_MATHOP_REMAINDER,
    ME_MATHOP_FMOD,
    ME_MATHOP_POW,
    ME_MATHOP_COUNT
} me_math_op;

typedef void (*me_math_unary_f64)(const double *, double *, int);
typedef void (*me_math_unary_f32)(const float *, float *, int);
typedef void (*me_math_binary_f64)(const double *, const double *, double *, int);
typedef void (*me_math_binary_f32)(const float *, const float *, float *, int);

/* Immutable per-kernel table. A NULL entry means that the operation keeps its
 * system/libm semantics for this backend (documented system fallback). */
typedef struct {
    me_dsl_math_backend backend;
    me_dsl_sleef_ulp ulp;
    bool accelerated;
    me_math_unary_f64 unary_f64[ME_MATHOP_COUNT];
    me_math_unary_f32 unary_f32[ME_MATHOP_COUNT];
    me_math_binary_f64 binary_f64[ME_MATHOP_COUNT];
    me_math_binary_f32 binary_f32[ME_MATHOP_COUNT];
} me_math_dispatch;

me_math_op me_math_op_from_name(const char *name);
bool me_math_dispatch_init(me_math_dispatch *dispatch, me_dsl_math_backend backend,
                           me_dsl_sleef_ulp ulp);
bool me_math_dispatch_has_unary(const me_math_dispatch *dispatch, me_math_op op);
bool me_math_dispatch_has_binary(const me_math_dispatch *dispatch, me_math_op op);

/* Invocation-scoped active policy. The pointer is borrowed and must outlive the
 * evaluation; concurrent threads may select different kernels independently. */
const me_math_dispatch *me_math_dispatch_current(void);
const me_math_dispatch *me_math_dispatch_set(const me_math_dispatch *dispatch);

/* Scalar bridge for interpreter/JIT callbacks. Accelerated backends use bounded
 * lane packing (documented fallback where scalar SLEEF entry points are absent). */
bool me_math_scalar_unary_f32(me_math_op op, float x, float *out);
bool me_math_scalar_unary_f64(me_math_op op, double x, double *out);
bool me_math_scalar_binary_f32(me_math_op op, float x, float y, float *out);
bool me_math_scalar_binary_f64(me_math_op op, double x, double y, double *out);

#endif
