/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_PORTABLE_TYPES_H
#define MINIEXPR_DSL_PORTABLE_TYPES_H

#include "miniexpr.h"
#include <stdint.h>

/* Internal 1.0 typing foundation. Not used by full DSL.
 * ME_AUTO reports an unsupported pair; never means output-context typing. */
me_dtype dsl_portable_numeric_promote(me_dtype left, me_dtype right);
me_dtype dsl_portable_division_dtype(me_dtype left, me_dtype right);
/* NumPy 2.x strong dtype promotion; independent of operand values/output. */
me_dtype dsl_numpy_numeric_promote(me_dtype left, me_dtype right);
bool dsl_numpy_can_cast(me_dtype from, me_dtype to, const char *policy);

typedef enum {
    ME_PORTABLE_NUMERIC_OK = 0,
    ME_PORTABLE_NUMERIC_TYPE,
    ME_PORTABLE_NUMERIC_RANGE,
    ME_PORTABLE_NUMERIC_ZERO,
    ME_PORTABLE_NUMERIC_SHIFT
} me_portable_numeric_status;

typedef enum {
    ME_PORTABLE_ADD,
    ME_PORTABLE_SUB,
    ME_PORTABLE_MUL,
    ME_PORTABLE_FLOORDIV,
    ME_PORTABLE_MOD,
    ME_PORTABLE_SHL,
    ME_PORTABLE_SHR,
    ME_PORTABLE_POW,
    ME_PORTABLE_AND,
    ME_PORTABLE_OR,
    ME_PORTABLE_XOR
} me_portable_integer_op;

/* Values must already have been promoted. Validate input ranges as well as the
 * result. On error, leave *out unchanged. No operation invokes C signed UB. */
me_portable_numeric_status dsl_portable_signed_op(me_dtype dtype, me_portable_integer_op op,
                                                 int64_t left, int64_t right, int64_t *out);
me_portable_numeric_status dsl_portable_unsigned_op(me_dtype dtype, me_portable_integer_op op,
                                                    uint64_t left, uint64_t right, uint64_t *out);
me_portable_numeric_status dsl_numpy_signed_op(me_dtype dtype, me_portable_integer_op op,
                                              int64_t left, int64_t right, int64_t *out);
me_portable_numeric_status dsl_numpy_unsigned_op(me_dtype dtype, me_portable_integer_op op,
                                                uint64_t left, uint64_t right, uint64_t *out);
int64_t dsl_numpy_signed_bits(me_dtype dtype, uint64_t raw);
uint64_t dsl_numpy_unsigned_bits(me_dtype dtype, uint64_t raw);
me_portable_numeric_status dsl_portable_float_to_signed(me_dtype dtype, double value, int64_t *out);
me_portable_numeric_status dsl_portable_float_to_unsigned(me_dtype dtype, double value, uint64_t *out);
me_portable_numeric_status dsl_portable_signed_to_signed(me_dtype dtype, int64_t value, int64_t *out);
me_portable_numeric_status dsl_portable_unsigned_to_signed(me_dtype dtype, uint64_t value, int64_t *out);
me_portable_numeric_status dsl_portable_signed_to_unsigned(me_dtype dtype, int64_t value, uint64_t *out);
me_portable_numeric_status dsl_portable_unsigned_to_unsigned(me_dtype dtype, uint64_t value, uint64_t *out);
/* Exact three-way comparison without a floating common type. */
int dsl_portable_compare_signed_unsigned(int64_t left, uint64_t right);

#endif
