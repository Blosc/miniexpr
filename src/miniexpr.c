/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025  Blosc Development Team <blosc@blosc.org>
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
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>
#include <assert.h>

/* ND metadata attached to compiled expressions (used by me_eval_nd). */
typedef struct {
    int ndims;
    /* Layout: shape[ndims], chunkshape[ndims], blockshape[ndims] (all int64_t). */
    int64_t data[1];
} me_nd_info;

static int64_t ceil_div64(int64_t a, int64_t b) {
    return (b == 0) ? 0 : (a + b - 1) / b;
}

#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

/* Type promotion table following NumPy rules */
/* Note: ME_AUTO (0) should never appear in type promotion, so we index from 1 */
static const me_dtype type_promotion_table[13][13] = {
    /* Rows: left operand, Columns: right operand */
    /* BOOL,  INT8,    INT16,   INT32,   INT64,   UINT8,   UINT16,  UINT32,  UINT64,  FLOAT32, FLOAT64, COMPLEX64, COMPLEX128 */
    {
        ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* BOOL */
    {
        ME_INT8, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* INT8 */
    {
        ME_INT16, ME_INT16, ME_INT16, ME_INT32, ME_INT64, ME_INT32, ME_INT32, ME_INT64, ME_FLOAT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* INT16 */
    {
        ME_INT32, ME_INT32, ME_INT32, ME_INT32, ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* INT32 */
    {
        ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* INT64 */
    {
        ME_UINT8, ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* UINT8 */
    {
        ME_UINT16, ME_INT32, ME_INT32, ME_INT64, ME_FLOAT64, ME_UINT16, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* UINT16 */
    {
        ME_UINT32, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_UINT32, ME_UINT32, ME_UINT32, ME_UINT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* UINT32 */
    {
        ME_UINT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_UINT64, ME_UINT64, ME_UINT64, ME_UINT64,
        ME_FLOAT64, ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* UINT64 */
    {
        ME_FLOAT32, ME_FLOAT32, ME_FLOAT32, ME_FLOAT64, ME_FLOAT64, ME_FLOAT32, ME_FLOAT32, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT32, ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* FLOAT32 */
    {
        ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* FLOAT64 */
    {
        ME_COMPLEX64, ME_COMPLEX64, ME_COMPLEX64, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX64,
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX128
    }, /* COMPLEX64 */
    {
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128,
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128
    } /* COMPLEX128 */
};

/* Promote two types according to NumPy rules */
static me_dtype promote_types(me_dtype a, me_dtype b) {
    // ME_AUTO should have been resolved during compilation
    if (a == ME_AUTO || b == ME_AUTO) {
        fprintf(stderr, "FATAL: ME_AUTO in type promotion (a=%d, b=%d). This is a bug.\n", a, b);
#ifdef NDEBUG
        abort(); // Release build: terminate immediately
#else
        assert(0 && "ME_AUTO should be resolved during compilation"); // Debug: trigger debugger
#endif
    }

    // Adjust indices since table starts at ME_BOOL (index 1), not ME_AUTO (index 0)
    int a_idx = a - 1;
    int b_idx = b - 1;
    if (a_idx >= 0 && a_idx < 13 && b_idx >= 0 && b_idx < 13) {
        return type_promotion_table[a_idx][b_idx];
    }
    fprintf(stderr, "WARNING: Invalid dtype in type promotion (a=%d, b=%d). Falling back to FLOAT64.\n", a, b);
    return ME_FLOAT64; // Fallback for out-of-range types
}

static bool is_integral_or_bool(me_dtype dtype) {
    return dtype == ME_BOOL || (dtype >= ME_INT8 && dtype <= ME_UINT64);
}

static bool is_valid_dtype(me_dtype dtype) {
    return dtype >= ME_AUTO && dtype <= ME_COMPLEX128;
}

static me_dtype promote_float_math_result(me_dtype param_type) {
    if (param_type == ME_COMPLEX64 || param_type == ME_COMPLEX128) {
        return param_type;
    }
    if (param_type == ME_FLOAT32) {
        return ME_FLOAT32;
    }
    if (param_type == ME_FLOAT64) {
        return ME_FLOAT64;
    }
    if (is_integral_or_bool(param_type)) {
        return ME_FLOAT64;
    }
    return param_type;
}


static bool contains_reduction(const me_expr* n) {
    if (!n) return false;
    if (is_reduction_node(n)) return true;

    switch (TYPE_MASK(n->type)) {
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
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (contains_reduction((const me_expr*)n->parameters[i])) {
                    return true;
                }
            }
            return false;
        }
    default:
        return false;
    }
}

// Synthetic addresses for ordinal matching (when user provides NULL addresses)
static char synthetic_var_addresses[ME_MAX_VARS];

static bool output_is_scalar(const me_expr* n) {
    if (!n) return true;
    if (is_reduction_node(n)) return true;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return true;
    case ME_VARIABLE:
        return false;
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
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!output_is_scalar((const me_expr*)n->parameters[i])) {
                    return false;
                }
            }
            return true;
        }
    default:
        return true;
    }
}

typedef union {
    bool b;
    int64_t i64;
    uint64_t u64;
    float f32;
    double f64;
    float _Complex c64;
    double _Complex c128;
} me_scalar;

static void write_scalar(void* out, me_dtype out_type, me_dtype in_type, const me_scalar* v) {
    if (out_type == in_type) {
        switch (out_type) {
        case ME_BOOL: *(bool*)out = v->b; return;
        case ME_INT8: *(int8_t*)out = (int8_t)v->i64; return;
        case ME_INT16: *(int16_t*)out = (int16_t)v->i64; return;
        case ME_INT32: *(int32_t*)out = (int32_t)v->i64; return;
        case ME_INT64: *(int64_t*)out = v->i64; return;
        case ME_UINT8: *(uint8_t*)out = (uint8_t)v->u64; return;
        case ME_UINT16: *(uint16_t*)out = (uint16_t)v->u64; return;
        case ME_UINT32: *(uint32_t*)out = (uint32_t)v->u64; return;
        case ME_UINT64: *(uint64_t*)out = v->u64; return;
        case ME_FLOAT32: *(float*)out = v->f32; return;
        case ME_FLOAT64: *(double*)out = v->f64; return;
        case ME_COMPLEX64: *(float _Complex*)out = v->c64; return;
        case ME_COMPLEX128: *(double _Complex*)out = v->c128; return;
        default: return;
        }
    }

    switch (out_type) {
    case ME_BOOL:
        switch (in_type) {
        case ME_BOOL: *(bool*)out = v->b; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(bool*)out = v->i64 != 0; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(bool*)out = v->u64 != 0; break;
        case ME_FLOAT32: *(bool*)out = v->f32 != 0.0f; break;
        case ME_FLOAT64: *(bool*)out = v->f64 != 0.0; break;
        case ME_COMPLEX64: *(bool*)out = (crealf(v->c64) != 0.0f || cimagf(v->c64) != 0.0f); break;
        case ME_COMPLEX128: *(bool*)out = (creal(v->c128) != 0.0 || cimag(v->c128) != 0.0); break;
        default: *(bool*)out = false; break;
        }
        break;
    case ME_INT8:
    case ME_INT16:
    case ME_INT32:
    case ME_INT64:
        switch (in_type) {
        case ME_BOOL: *(int64_t*)out = v->b ? 1 : 0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(int64_t*)out = v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(int64_t*)out = (int64_t)v->u64; break;
        case ME_FLOAT32: *(int64_t*)out = (int64_t)v->f32; break;
        case ME_FLOAT64: *(int64_t*)out = (int64_t)v->f64; break;
        default: *(int64_t*)out = 0; break;
        }
        break;
    case ME_UINT8:
    case ME_UINT16:
    case ME_UINT32:
    case ME_UINT64:
        switch (in_type) {
        case ME_BOOL: *(uint64_t*)out = v->b ? 1 : 0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(uint64_t*)out = (uint64_t)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(uint64_t*)out = v->u64; break;
        case ME_FLOAT32: *(uint64_t*)out = (uint64_t)v->f32; break;
        case ME_FLOAT64: *(uint64_t*)out = (uint64_t)v->f64; break;
        default: *(uint64_t*)out = 0; break;
        }
        break;
    case ME_FLOAT32:
        switch (in_type) {
        case ME_BOOL: *(float*)out = v->b ? 1.0f : 0.0f; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(float*)out = (float)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(float*)out = (float)v->u64; break;
        case ME_FLOAT32: *(float*)out = v->f32; break;
        case ME_FLOAT64: *(float*)out = (float)v->f64; break;
        default: *(float*)out = 0.0f; break;
        }
        break;
    case ME_FLOAT64:
        switch (in_type) {
        case ME_BOOL: *(double*)out = v->b ? 1.0 : 0.0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(double*)out = (double)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(double*)out = (double)v->u64; break;
        case ME_FLOAT32: *(double*)out = (double)v->f32; break;
        case ME_FLOAT64: *(double*)out = v->f64; break;
        default: *(double*)out = 0.0; break;
        }
        break;
    case ME_COMPLEX64:
        switch (in_type) {
        case ME_COMPLEX64: *(float _Complex*)out = v->c64; break;
        case ME_COMPLEX128: *(float _Complex*)out = (float _Complex)v->c128; break;
        case ME_FLOAT32: *(float _Complex*)out = v->f32 + 0.0f * I; break;
        case ME_FLOAT64: *(float _Complex*)out = (float)v->f64 + 0.0f * I; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(float _Complex*)out = (float)v->i64 + 0.0f * I; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(float _Complex*)out = (float)v->u64 + 0.0f * I; break;
        case ME_BOOL: *(float _Complex*)out = (v->b ? 1.0f : 0.0f) + 0.0f * I; break;
        default: *(float _Complex*)out = 0.0f + 0.0f * I; break;
        }
        break;
    case ME_COMPLEX128:
        switch (in_type) {
        case ME_COMPLEX64: *(double _Complex*)out = (double _Complex)v->c64; break;
        case ME_COMPLEX128: *(double _Complex*)out = v->c128; break;
        case ME_FLOAT32: *(double _Complex*)out = (double)v->f32 + 0.0 * I; break;
        case ME_FLOAT64: *(double _Complex*)out = v->f64 + 0.0 * I; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(double _Complex*)out = (double)v->i64 + 0.0 * I; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(double _Complex*)out = (double)v->u64 + 0.0 * I; break;
        case ME_BOOL: *(double _Complex*)out = (v->b ? 1.0 : 0.0) + 0.0 * I; break;
        default: *(double _Complex*)out = 0.0 + 0.0 * I; break;
        }
        break;
    default:
        break;
    }
}

static void read_scalar(const void* in, me_dtype in_type, me_scalar* v) {
    switch (in_type) {
    case ME_BOOL: v->b = *(const bool*)in; break;
    case ME_INT8: v->i64 = *(const int8_t*)in; break;
    case ME_INT16: v->i64 = *(const int16_t*)in; break;
    case ME_INT32: v->i64 = *(const int32_t*)in; break;
    case ME_INT64: v->i64 = *(const int64_t*)in; break;
    case ME_UINT8: v->u64 = *(const uint8_t*)in; break;
    case ME_UINT16: v->u64 = *(const uint16_t*)in; break;
    case ME_UINT32: v->u64 = *(const uint32_t*)in; break;
    case ME_UINT64: v->u64 = *(const uint64_t*)in; break;
    case ME_FLOAT32: v->f32 = *(const float*)in; break;
    case ME_FLOAT64: v->f64 = *(const double*)in; break;
    case ME_COMPLEX64: v->c64 = *(const float _Complex*)in; break;
    case ME_COMPLEX128: v->c128 = *(const double _Complex*)in; break;
    default: break;
    }
}

static bool read_as_bool(const void* base, int64_t off, me_dtype type, bool* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off]; return true;
    case ME_INT8: *out = ((const int8_t*)base)[off] != 0; return true;
    case ME_INT16: *out = ((const int16_t*)base)[off] != 0; return true;
    case ME_INT32: *out = ((const int32_t*)base)[off] != 0; return true;
    case ME_INT64: *out = ((const int64_t*)base)[off] != 0; return true;
    case ME_UINT8: *out = ((const uint8_t*)base)[off] != 0; return true;
    case ME_UINT16: *out = ((const uint16_t*)base)[off] != 0; return true;
    case ME_UINT32: *out = ((const uint32_t*)base)[off] != 0; return true;
    case ME_UINT64: *out = ((const uint64_t*)base)[off] != 0; return true;
    case ME_FLOAT32: *out = ((const float*)base)[off] != 0.0f; return true;
    case ME_FLOAT64: *out = ((const double*)base)[off] != 0.0; return true;
    default:
        return false;
    }
}

static bool read_as_int64(const void* base, int64_t off, me_dtype type, int64_t* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1 : 0; return true;
    case ME_INT8: *out = ((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = ((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = ((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = ((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = (int64_t)((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = (int64_t)((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = (int64_t)((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = (int64_t)((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = (int64_t)((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = (int64_t)((const double*)base)[off]; return true;
    default:
        return false;
    }
}

static bool read_as_uint64(const void* base, int64_t off, me_dtype type, uint64_t* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1 : 0; return true;
    case ME_INT8: *out = (uint64_t)((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = (uint64_t)((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = (uint64_t)((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = (uint64_t)((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = ((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = ((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = ((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = ((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = (uint64_t)((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = (uint64_t)((const double*)base)[off]; return true;
    default:
        return false;
    }
}

static bool read_as_float(const void* base, int64_t off, me_dtype type, float* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1.0f : 0.0f; return true;
    case ME_INT8: *out = (float)((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = (float)((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = (float)((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = (float)((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = (float)((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = (float)((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = (float)((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = (float)((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = ((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = (float)((const double*)base)[off]; return true;
    default:
        return false;
    }
}

static bool read_as_double(const void* base, int64_t off, me_dtype type, double* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1.0 : 0.0; return true;
    case ME_INT8: *out = (double)((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = (double)((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = (double)((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = (double)((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = (double)((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = (double)((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = (double)((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = (double)((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = (double)((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = ((const double*)base)[off]; return true;
    default:
        return false;
    }
}

static bool cmp_int64(me_cmp_kind cmp, int64_t a, int64_t b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static bool cmp_uint64(me_cmp_kind cmp, uint64_t a, uint64_t b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static bool cmp_float(me_cmp_kind cmp, float a, float b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static bool cmp_double(me_cmp_kind cmp, double a, double b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static me_cmp_kind invert_cmp_kind(me_cmp_kind cmp) {
    switch (cmp) {
    case ME_CMP_LT: return ME_CMP_GT;
    case ME_CMP_LE: return ME_CMP_GE;
    case ME_CMP_GT: return ME_CMP_LT;
    case ME_CMP_GE: return ME_CMP_LE;
    case ME_CMP_EQ: return ME_CMP_EQ;
    case ME_CMP_NE: return ME_CMP_NE;
    default: return ME_CMP_NONE;
    }
}

static bool reduce_strided_variable(const me_expr* expr, const void** vars_block, int n_vars,
                                    const int64_t* valid_len, const int64_t* stride, int nd,
                                    int64_t valid_items, void* output_block) {
    if (!expr || !is_reduction_node(expr) || valid_items <= 0) {
        return false;
    }
    const me_expr* arg = (const me_expr*)expr->parameters[0];
    if (!arg || TYPE_MASK(arg->type) != ME_VARIABLE || !is_synthetic_address(arg->bound)) {
        return false;
    }
    int idx = (int)((const char*)arg->bound - synthetic_var_addresses);
    if (idx < 0 || idx >= n_vars) {
        return false;
    }

    const me_reduce_kind rkind = reduction_kind(expr->function);
    if (rkind == ME_REDUCE_NONE) {
        return false;
    }

    const me_dtype arg_type = infer_result_type(arg);
    const me_dtype result_type = reduction_output_dtype(arg_type, expr->function);
    const me_dtype output_type = expr->dtype;

    int64_t indices[64] = {0};
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) total_iters *= valid_len[i];

    me_scalar acc;
    switch (result_type) {
    case ME_BOOL: acc.b = (rkind == ME_REDUCE_ALL); break;
    case ME_INT64: acc.i64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
    case ME_UINT64: acc.u64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
    case ME_FLOAT32: acc.f64 = (rkind == ME_REDUCE_PROD) ? 1.0 : 0.0; break;
    case ME_FLOAT64: acc.f64 = (rkind == ME_REDUCE_PROD) ? 1.0 : 0.0; break;
    case ME_COMPLEX64: acc.c64 = (rkind == ME_REDUCE_PROD) ? (float _Complex)1.0f : (float _Complex)0.0f; break;
    case ME_COMPLEX128: acc.c128 = (rkind == ME_REDUCE_PROD) ? (double _Complex)1.0 : (double _Complex)0.0; break;
    default: break;
    }

    const unsigned char* base = (const unsigned char*)vars_block[idx];
    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }

        switch (arg_type) {
        case ME_BOOL: {
            bool v = ((const bool*)base)[off];
            if (rkind == ME_REDUCE_ANY) { if (v) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!v) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v ? 1 : 0;
            else acc.i64 += v ? 1 : 0;
            break;
        }
        case ME_INT8: {
            int8_t v = ((const int8_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int8_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int8_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT16: {
            int16_t v = ((const int16_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int16_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int16_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT32: {
            int32_t v = ((const int32_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int32_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int32_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT64: {
            int64_t v = ((const int64_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_UINT8: {
            uint8_t v = ((const uint8_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint8_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint8_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT16: {
            uint16_t v = ((const uint16_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint16_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint16_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT32: {
            uint32_t v = ((const uint32_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint32_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint32_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT64: {
            uint64_t v = ((const uint64_t*)base)[off];
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_FLOAT32: {
            float v = ((const float*)base)[off];
            if (v != v) { acc.f64 = NAN; goto done_reduce; }
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (float)acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (float)acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.f64 *= (double)v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0.0f) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0.0f) { acc.b = false; goto done_reduce; } }
            else acc.f64 += (double)v;
            break;
        }
        case ME_FLOAT64: {
            double v = ((const double*)base)[off];
            if (v != v) { acc.f64 = NAN; goto done_reduce; }
            if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.f64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0.0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0.0) { acc.b = false; goto done_reduce; } }
            else acc.f64 += v;
            break;
        }
        case ME_COMPLEX64: {
            float _Complex v = ((const float _Complex*)base)[off];
            bool nonzero = (crealf(v) != 0.0f || cimagf(v) != 0.0f);
            if (rkind == ME_REDUCE_ANY) { if (nonzero) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!nonzero) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.c64 *= v;
            else acc.c64 += v;
            break;
        }
        case ME_COMPLEX128: {
            double _Complex v = ((const double _Complex*)base)[off];
            bool nonzero = (creal(v) != 0.0 || cimag(v) != 0.0);
            if (rkind == ME_REDUCE_ANY) { if (nonzero) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!nonzero) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.c128 *= v;
            else acc.c128 += v;
            break;
        }
        default:
            break;
        }

        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) break;
            indices[i] = 0;
        }
    }

done_reduce:
    if (result_type == ME_FLOAT32) {
        acc.f32 = (float)acc.f64;
    }
    write_scalar(output_block, output_type, result_type, &acc);
    return true;
}

static bool reduce_strided_predicate(const me_expr* expr, const void** vars_block, int n_vars,
                                     const int64_t* valid_len, const int64_t* stride, int nd,
                                     int64_t valid_items, void* output_block) {
    if (!expr || !is_reduction_node(expr) || valid_items <= 0) {
        return false;
    }
    const me_expr* arg = (const me_expr*)expr->parameters[0];
    if (!arg || !is_comparison_node(arg)) {
        return false;
    }

    me_reduce_kind rkind = reduction_kind(expr->function);
    if (!(rkind == ME_REDUCE_ANY || rkind == ME_REDUCE_ALL)) {
        /* Keep only any/all predicate reductions; sum(x == c) uses pack path. */
        return false;
    }

    const me_expr* left = (const me_expr*)arg->parameters[0];
    const me_expr* right = (const me_expr*)arg->parameters[1];
    if (!left || !right) {
        return false;
    }

    const me_expr* var_node = NULL;
    const me_expr* const_node = NULL;
    bool const_on_left = false;

    if (TYPE_MASK(left->type) == ME_VARIABLE && right->type == ME_CONSTANT) {
        var_node = left;
        const_node = right;
    }
    else if (TYPE_MASK(right->type) == ME_VARIABLE && left->type == ME_CONSTANT) {
        var_node = right;
        const_node = left;
        const_on_left = true;
    }
    else {
        return false;
    }

    if (!is_synthetic_address(var_node->bound)) {
        return false;
    }
    int idx = (int)((const char*)var_node->bound - synthetic_var_addresses);
    if (idx < 0 || idx >= n_vars) {
        return false;
    }

    me_cmp_kind cmp = comparison_kind(arg->function);
    if (cmp == ME_CMP_NONE) {
        return false;
    }
    if (const_on_left) {
        cmp = invert_cmp_kind(cmp);
        if (cmp == ME_CMP_NONE) {
            return false;
        }
    }

    const me_dtype eval_type = infer_result_type(arg);
    if (eval_type == ME_COMPLEX64 || eval_type == ME_COMPLEX128) {
        return false;
    }

    const me_dtype output_type = expr->dtype;
    const me_dtype result_type = reduction_output_dtype(ME_BOOL, expr->function);

    int64_t indices[64] = {0};
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) total_iters *= valid_len[i];

    me_scalar acc;
    acc.b = (rkind == ME_REDUCE_ALL);

    const unsigned char* base = (const unsigned char*)vars_block[idx];
    const double cval = const_node->value;

    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }

        bool pred = false;
        switch (eval_type) {
        case ME_BOOL: {
            bool v = false;
            if (!read_as_bool(base, off, var_node->input_dtype, &v)) return false;
            bool c = (cval != 0.0);
            pred = cmp_int64(cmp, v ? 1 : 0, c ? 1 : 0);
            break;
        }
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: {
            int64_t v = 0;
            if (!read_as_int64(base, off, var_node->input_dtype, &v)) return false;
            int64_t c = (int64_t)cval;
            pred = cmp_int64(cmp, v, c);
            break;
        }
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: {
            uint64_t v = 0;
            if (!read_as_uint64(base, off, var_node->input_dtype, &v)) return false;
            uint64_t c = (uint64_t)cval;
            pred = cmp_uint64(cmp, v, c);
            break;
        }
        case ME_FLOAT32: {
            float v = 0.0f;
            if (!read_as_float(base, off, var_node->input_dtype, &v)) return false;
            float c = (float)cval;
            pred = cmp_float(cmp, v, c);
            break;
        }
        case ME_FLOAT64: {
            double v = 0.0;
            if (!read_as_double(base, off, var_node->input_dtype, &v)) return false;
            pred = cmp_double(cmp, v, cval);
            break;
        }
        default:
            return false;
        }

        if (rkind == ME_REDUCE_ANY) {
            if (pred) { acc.b = true; goto done_pred; }
        }
        else if (rkind == ME_REDUCE_ALL) {
            if (!pred) { acc.b = false; goto done_pred; }
        }

        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) break;
            indices[i] = 0;
        }
    }

done_pred:
    write_scalar(output_block, output_type, result_type, &acc);
    return true;
}
static bool reduction_usage_is_valid(const me_expr* n) {
    if (!n) return true;
    if (is_reduction_node(n)) {
        me_expr* arg = (me_expr*)n->parameters[0];
        if (!arg) return false;
        if (contains_reduction(arg)) return false;
        me_dtype arg_type = infer_output_type(arg);
        if (n->function == (void*)min_reduce || n->function == (void*)max_reduce) {
            if (arg_type == ME_COMPLEX64 || arg_type == ME_COMPLEX128) {
                return false;
            }
        }
        return true;
    }

    switch (TYPE_MASK(n->type)) {
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
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!reduction_usage_is_valid((const me_expr*)n->parameters[i])) {
                    return false;
                }
            }
            return true;
        }
    default:
        return true;
    }
}

/* Infer computation type from expression tree (for evaluation) */
me_dtype infer_result_type(const me_expr* n) {
    if (!n) return ME_FLOAT64;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return n->dtype;

    case ME_VARIABLE:
        return n->dtype;

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
            if (is_reduction_node(n)) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                return reduction_output_dtype(param_type, n->function);
            }
            // Special case: imag() and real() return real type from complex input
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
                if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper) {
                    me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    else if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    // If input is not complex, return as-is (shouldn't happen, but be safe)
                    return param_type;
                }
                if (n->function == (void*)fabs) {
                    me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    return param_type;
                }
            }

            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && is_float_math_function(n->function)) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                return promote_float_math_result(param_type);
            }

            // For comparisons with ME_BOOL output, we still need to infer the
            // computation type from operands (e.g., float64 for float inputs).
            // Don't return ME_BOOL early - let the operand types determine
            // the computation type.

            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Infer logical output type from expression tree (for compilation with ME_AUTO) */
me_dtype infer_output_type(const me_expr* n) {
    if (!n) return ME_FLOAT64;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return n->dtype;

    case ME_VARIABLE:
        return n->dtype;

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
            if (is_reduction_node(n)) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                return reduction_output_dtype(param_type, n->function);
            }
            // Special case: imag() and real() return real type from complex input
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
                if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper) {
                    me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    else if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    // If input is not complex, return as-is (shouldn't happen, but be safe)
                    return param_type;
                }
                if (n->function == (void*)fabs) {
                    me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    return param_type;
                }
            }

            // Special case: where(cond, x, y) -> promote(x, y), regardless of cond type.
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 3 &&
                n->function == (void*)where_scalar) {
                me_dtype x_type = infer_output_type((const me_expr*)n->parameters[1]);
                me_dtype y_type = infer_output_type((const me_expr*)n->parameters[2]);
                return promote_types(x_type, y_type);
            }

            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && is_float_math_function(n->function)) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                return promote_float_math_result(param_type);
            }

            // If this node is a comparison (dtype == ME_BOOL set during parsing),
            // the output type is ME_BOOL
            if (n->dtype == ME_BOOL) {
                return ME_BOOL;
            }

            // Otherwise, infer from operands
            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Apply type promotion to a binary operation node */
static me_expr* create_conversion_node(me_expr* source, me_dtype target_dtype) {
    /* Create a unary conversion node that converts source to target_dtype */
    me_expr* conv = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, source);
    if (conv) {
        conv->function = NULL; // Mark as conversion
        conv->dtype = target_dtype;
        conv->input_dtype = source->dtype;
    }
    return conv;
}

void apply_type_promotion(me_expr* node) {
    if (!node || ARITY(node->type) < 2) return;

    me_expr* left = (me_expr*)node->parameters[0];
    me_expr* right = (me_expr*)node->parameters[1];

    if (left && right) {
        me_dtype left_type = left->dtype;
        me_dtype right_type = right->dtype;
        me_dtype promoted = promote_types(left_type, right_type);

        // Store the promoted output type
        node->dtype = promoted;

        // Insert conversion nodes if needed for nested expressions with different dtype
        if (left_type != promoted && TYPE_MASK(left->type) >= ME_FUNCTION0) {
            me_expr *conv_left = create_conversion_node(left, promoted);
            if (conv_left) {
                node->parameters[0] = conv_left;
            }
        }

        if (right_type != promoted && TYPE_MASK(right->type) >= ME_FUNCTION0) {
            me_expr *conv_right = create_conversion_node(right, promoted);
            if (conv_right) {
                node->parameters[1] = conv_right;
            }
        }
    }
}

/* Check for mixed-type nested expressions (currently not supported) */
static int check_mixed_type_nested(const me_expr* node, me_dtype parent_dtype) {
    if (!node) return 0;

    switch (TYPE_MASK(node->type)) {
    case ME_CONSTANT:
    case ME_VARIABLE:
        return 0;

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
        /* Skip reduction nodes - they handle their own type conversions */
        if (is_reduction_node(node)) {
            return 0;
        }

        /* Skip comparison nodes - they naturally have different output type (bool) than operands */
        if (is_comparison_node(node)) {
            return 0;
        }

        /* Only check binary operations (arity 2) for mixed-type nested expressions */
        /* Unary operations are fine */
        const int arity = ARITY(node->type);
        if (arity == 2 && IS_FUNCTION(node->type)) {
            me_expr* left = (me_expr*)node->parameters[0];
            me_expr* right = (me_expr*)node->parameters[1];

            /* If either operand is a nested expression (not constant/variable) with different dtype, flag it */
            if (left && TYPE_MASK(left->type) >= ME_FUNCTION0 &&
                left->dtype != ME_AUTO && node->dtype != ME_AUTO &&
                left->dtype != node->dtype) {
                return 1;
            }
            if (right && TYPE_MASK(right->type) >= ME_FUNCTION0 &&
                right->dtype != ME_AUTO && node->dtype != ME_AUTO &&
                right->dtype != node->dtype) {
                return 1;
            }
        }

        /* Recursively check children */
        for (int i = 0; i < arity; i++) {
            if (check_mixed_type_nested((const me_expr*)node->parameters[i], node->dtype)) {
                return 1;
            }
        }
        break;
    }

    return 0;
}

me_expr* new_expr(const int type, const me_expr* parameters[]) {
    const int arity = ARITY(type);
    const int psize = sizeof(void*) * arity;
    const int size = (sizeof(me_expr) - sizeof(void*)) + psize + (IS_CLOSURE(type) ? sizeof(void*) : 0);
    me_expr* ret = malloc(size);
    CHECK_NULL(ret);

    memset(ret, 0, size);
    if (arity && parameters) {
        memcpy(ret->parameters, parameters, psize);
    }
    ret->type = type;
    ret->bound = 0;
    ret->output = NULL;
    ret->nitems = 0;
    ret->dtype = ME_FLOAT64; // Default to double
    ret->bytecode = NULL;
    ret->ncode = 0;
    return ret;
}


void me_free_parameters(me_expr* n) {
    if (!n) return;
    switch (TYPE_MASK(n->type)) {
    case ME_FUNCTION7:
    case ME_CLOSURE7:
        if (n->parameters[6] && ((me_expr*)n->parameters[6])->output &&
            ((me_expr*)n->parameters[6])->output != n->output) {
            free(((me_expr*)n->parameters[6])->output);
        }
        me_free(n->parameters[6]);
    case ME_FUNCTION6:
    case ME_CLOSURE6:
        if (n->parameters[5] && ((me_expr*)n->parameters[5])->output &&
            ((me_expr*)n->parameters[5])->output != n->output) {
            free(((me_expr*)n->parameters[5])->output);
        }
        me_free(n->parameters[5]);
    case ME_FUNCTION5:
    case ME_CLOSURE5:
        if (n->parameters[4] && ((me_expr*)n->parameters[4])->output &&
            ((me_expr*)n->parameters[4])->output != n->output) {
            free(((me_expr*)n->parameters[4])->output);
        }
        me_free(n->parameters[4]);
    case ME_FUNCTION4:
    case ME_CLOSURE4:
        if (n->parameters[3] && ((me_expr*)n->parameters[3])->output &&
            ((me_expr*)n->parameters[3])->output != n->output) {
            free(((me_expr*)n->parameters[3])->output);
        }
        me_free(n->parameters[3]);
    case ME_FUNCTION3:
    case ME_CLOSURE3:
        if (n->parameters[2] && ((me_expr*)n->parameters[2])->output &&
            ((me_expr*)n->parameters[2])->output != n->output) {
            free(((me_expr*)n->parameters[2])->output);
        }
        me_free(n->parameters[2]);
    case ME_FUNCTION2:
    case ME_CLOSURE2:
        if (n->parameters[1] && ((me_expr*)n->parameters[1])->output &&
            ((me_expr*)n->parameters[1])->output != n->output) {
            free(((me_expr*)n->parameters[1])->output);
        }
        me_free(n->parameters[1]);
    case ME_FUNCTION1:
    case ME_CLOSURE1:
        if (n->parameters[0] && ((me_expr*)n->parameters[0])->output &&
            ((me_expr*)n->parameters[0])->output != n->output) {
            free(((me_expr*)n->parameters[0])->output);
        }
        me_free(n->parameters[0]);
    }
}


void me_free(me_expr* n) {
    if (!n) return;
    me_free_parameters(n);
    if (n->bytecode) {
        free(n->bytecode);
    }
    free(n);
}

static int private_compile(const char* expression, const me_variable* variables, int var_count,
                           void* output, int nitems, me_dtype dtype, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!expression || !out || var_count < 0) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    if (dtype != ME_AUTO && !is_valid_dtype(dtype)) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG_TYPE;
    }

    if (variables && var_count > 0) {
        for (int i = 0; i < var_count; i++) {
            if (!is_valid_dtype(variables[i].dtype)) {
                if (error) *error = -1;
                return ME_COMPILE_ERR_INVALID_ARG_TYPE;
            }
        }
    }

    // Validate dtype usage: either all vars are ME_AUTO (use dtype), or dtype is ME_AUTO (use var dtypes)
    if (variables && var_count > 0) {
        int auto_count = 0;
        int specified_count = 0;

        for (int i = 0; i < var_count; i++) {
            if (variables[i].dtype == ME_AUTO) {
                auto_count++;
            }
            else {
                specified_count++;
            }
        }

        // Check the two valid modes
        if (dtype == ME_AUTO) {
            // Mode 1: Output dtype is ME_AUTO, all variables must have explicit dtypes
            if (auto_count > 0) {
                fprintf(
                    stderr,
                    "Error: When output dtype is ME_AUTO, all variable dtypes must be specified (not ME_AUTO)\n");
                if (error) *error = -1;
                return ME_COMPILE_ERR_VAR_UNSPECIFIED;
            }
        }
        else {
            // Mode 2: Output dtype is specified
            // Two sub-modes: all ME_AUTO (homogeneous), or all explicit (heterogeneous with conversion)
            if (auto_count > 0 && specified_count > 0) {
                // Mixed mode not allowed
                fprintf(stderr, "Error: Variable dtypes must be all ME_AUTO or all explicitly specified\n");
                if (error) *error = -1;
                return ME_COMPILE_ERR_VAR_MIXED;
            }
        }
    }

    // Create a copy of variables with dtype filled in (if not already set)
    me_variable* vars_copy = NULL;
    if (variables && var_count > 0) {
        vars_copy = malloc(var_count * sizeof(me_variable));
        if (!vars_copy) {
            if (error) *error = -1;
            return ME_COMPILE_ERR_OOM;
        }
        for (int i = 0; i < var_count; i++) {
            vars_copy[i] = variables[i];
            // If dtype not set (ME_AUTO), use the provided dtype
            if (vars_copy[i].dtype == ME_AUTO && vars_copy[i].type == 0) {
                vars_copy[i].dtype = dtype;
                vars_copy[i].type = ME_VARIABLE;
            }
        }
    }

    state s;
    s.start = s.next = expression;
    s.lookup = vars_copy ? vars_copy : variables;
    s.lookup_len = var_count;
    // When dtype is ME_AUTO, infer target dtype from variables to avoid type mismatch
    if (dtype != ME_AUTO) {
        s.target_dtype = dtype;
    }
    else if (variables && var_count > 0) {
        // Use the first variable's dtype as the target for constants
        // This prevents type promotion issues when mixing float32 vars with float64 constants
        s.target_dtype = variables[0].dtype;
    }
    else {
        s.target_dtype = ME_AUTO;
    }

    next_token(&s);
    me_expr* root = list(&s);

    if (root == NULL) {
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_OOM;
    }

    if (contains_reduction(root) && !reduction_usage_is_valid(root)) {
        me_free(root);
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_REDUCTION_INVALID;
    }

    bool any_complex_vars = false;
    if (variables && var_count > 0) {
        const me_variable* vars_check = vars_copy ? vars_copy : variables;
        for (int i = 0; i < var_count; i++) {
            if (vars_check[i].dtype == ME_COMPLEX64 || vars_check[i].dtype == ME_COMPLEX128) {
                any_complex_vars = true;
                break;
            }
        }
    }

    if ((any_complex_vars || has_complex_input_types(root)) && has_unsupported_complex_function(root)) {
        me_free(root);
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_INVALID_ARG_TYPE;
    }

#if defined(_WIN32) || defined(_WIN64)
    {
        const me_variable* vars_check = vars_copy ? vars_copy : variables;
        bool complex_vars = false;
        if (vars_check) {
            for (int i = 0; i < var_count; i++) {
                if (vars_check[i].dtype == ME_COMPLEX64 || vars_check[i].dtype == ME_COMPLEX128) {
                    complex_vars = true;
                    break;
                }
            }
        }
        if (complex_vars ||
            dtype == ME_COMPLEX64 || dtype == ME_COMPLEX128 ||
            has_complex_node(root) || has_complex_input(root)) {
            fprintf(stderr, "Error: Complex expressions are not supported on Windows (no C99 complex ABI)\n");
            me_free(root);
            if (error) *error = -1;
            if (vars_copy) free(vars_copy);
            return ME_COMPILE_ERR_COMPLEX_UNSUPPORTED;
        }
    }
#endif

    if (s.type != TOK_END) {
        me_free(root);
        if (error) {
            *error = (s.next - s.start);
            if (*error == 0) *error = 1;
        }
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_PARSE;
    }
    else {
        optimize(root);
        root->output = output;
        root->nitems = nitems;

        // If dtype is ME_AUTO, infer from expression; otherwise use provided dtype
        if (dtype == ME_AUTO) {
            root->dtype = infer_output_type(root);
        }
        else {
            // User explicitly requested a dtype - use it (will cast if needed)
            root->dtype = dtype;
        }

        // Mixed-type nested expressions now handled via conversion nodes
        // (see apply_type_promotion which inserts conversion nodes when needed)

        if (error) *error = 0;
        if (vars_copy) free(vars_copy);
        *out = root;
        return ME_COMPILE_SUCCESS;
    }
}

// Check if a pointer is a synthetic address
int is_synthetic_address(const void* ptr) {
    const char* p = (const char*)ptr;
    return (p >= synthetic_var_addresses && p < synthetic_var_addresses + ME_MAX_VARS);
}

int me_compile(const char* expression, const me_variable* variables,
               int var_count, me_dtype dtype, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!out) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    // For chunked evaluation, we compile without specific output/nitems
    // If variables have NULL addresses, assign synthetic unique addresses for ordinal matching
    me_variable* vars_copy = NULL;
    int needs_synthetic = 0;

    if (variables && var_count > 0) {
        // Check if any variables have NULL addresses
        for (int i = 0; i < var_count; i++) {
            if (variables[i].address == NULL) {
                needs_synthetic = 1;
                break;
            }
        }

        if (needs_synthetic) {
            // Create copy with synthetic addresses
            vars_copy = malloc(var_count * sizeof(me_variable));
            if (!vars_copy) {
                if (error) *error = -1;
                return ME_COMPILE_ERR_OOM;
            }

            for (int i = 0; i < var_count; i++) {
                vars_copy[i] = variables[i];
                if (vars_copy[i].address == NULL) {
                    // Use address in synthetic array (each index is unique)
                    vars_copy[i].address = &synthetic_var_addresses[i];
                }
            }

            int status = private_compile(expression, vars_copy, var_count, NULL, 0, dtype, error, out);
            free(vars_copy);
            return status;
        }
    }

    // No NULL addresses, use variables as-is
    return private_compile(expression, variables, var_count, NULL, 0, dtype, error, out);
}

int me_compile_nd(const char* expression, const me_variable* variables,
                  int var_count, me_dtype dtype, int ndims,
                  const int64_t* shape, const int32_t* chunkshape,
                  const int32_t* blockshape, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!expression || !out || ndims <= 0 || !shape || !chunkshape || !blockshape) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    for (int i = 0; i < ndims; i++) {
        if (chunkshape[i] <= 0 || blockshape[i] <= 0) {
            if (error) *error = -1;
            return ME_COMPILE_ERR_INVALID_ARG;
        }
    }

    me_expr* expr = NULL;
    int rc = me_compile(expression, variables, var_count, dtype, error, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        return rc;
    }

    const size_t extra_items = (size_t)(3 * ndims - 1);
    const size_t info_size = sizeof(me_nd_info) + extra_items * sizeof(int64_t);
    me_nd_info* info = malloc(info_size);
    if (!info) {
        me_free(expr);
        if (error) *error = -1;
        return ME_COMPILE_ERR_OOM;
    }

    info->ndims = ndims;
    int64_t* ptr = info->data;
    for (int i = 0; i < ndims; i++) {
        ptr[i] = shape[i];
    }
    ptr += ndims;
    for (int i = 0; i < ndims; i++) {
        ptr[i] = (int64_t)chunkshape[i];
    }
    ptr += ndims;
    for (int i = 0; i < ndims; i++) {
        ptr[i] = (int64_t)blockshape[i];
    }

    expr->bytecode = info;
    *out = expr;
    return rc;
}

static void pn(const me_expr* n, int depth) {
    int i, arity;
    printf("%*s", depth, "");

    if (!n) {
        printf("NULL\n");
        return;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT: printf("%f\n", n->value);
        break;
    case ME_VARIABLE: printf("bound %p\n", n->bound);
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
        arity = ARITY(n->type);
        printf("f%d", arity);
        for (i = 0; i < arity; i++) {
            printf(" %p", n->parameters[i]);
        }
        printf("\n");
        for (i = 0; i < arity; i++) {
            pn(n->parameters[i], depth + 1);
        }
        break;
    }
}

void me_print(const me_expr* n) {
    pn(n, 0);
}

me_dtype me_get_dtype(const me_expr* expr) {
    return expr ? expr->dtype : ME_AUTO;
}

static int compute_valid_items(const me_expr* expr, int64_t nchunk, int64_t nblock,
                               int chunk_nitems, int64_t* valid_items, int64_t* padded_items) {
    if (!expr || !valid_items || !padded_items) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const me_nd_info* info = (const me_nd_info*)expr->bytecode;
    if (!info || info->ndims <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const int nd = info->ndims;
    if (nd > 64) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    const int64_t* shape = info->data;
    const int64_t* chunkshape = shape + nd;
    const int64_t* blockshape = chunkshape + nd;

    int64_t total_chunks = 1;
    int64_t total_blocks = 1;
    int64_t padded = 1;

    for (int i = 0; i < nd; i++) {
        if (chunkshape[i] <= 0 || blockshape[i] <= 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        const int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        const int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        if (nchunks_d <= 0 || nblocks_d <= 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        if (total_chunks > LLONG_MAX / nchunks_d || total_blocks > LLONG_MAX / nblocks_d) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        total_chunks *= nchunks_d;
        total_blocks *= nblocks_d;
        if (padded > LLONG_MAX / blockshape[i]) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        padded *= blockshape[i];
    }

    if (nchunk < 0 || nchunk >= total_chunks || nblock < 0 || nblock >= total_blocks) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (chunk_nitems > 0 && (int64_t)chunk_nitems < padded) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t chunk_idx[64];
    int64_t block_idx[64];

    int64_t tmp = nchunk;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
        tmp /= nchunks_d;
    }

    tmp = nblock;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
        tmp /= nblocks_d;
    }

    int64_t valid = 1;
    for (int i = 0; i < nd; i++) {
        const int64_t chunk_start = chunk_idx[i] * chunkshape[i];
        if (shape[i] <= chunk_start) {
            valid = 0;
            break;
        }
        int64_t chunk_len = shape[i] - chunk_start;
        if (chunk_len > chunkshape[i]) {
            chunk_len = chunkshape[i];
        }

        const int64_t block_start = block_idx[i] * blockshape[i];
        if (block_start >= chunk_len) {
            valid = 0;
            break;
        }
        const int64_t remain = chunk_len - block_start;
        const int64_t len = (remain < blockshape[i]) ? remain : blockshape[i];
        if (valid > LLONG_MAX / len) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        valid *= len;
    }

    if (chunk_nitems > 0 && valid > (int64_t)chunk_nitems) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    *valid_items = valid;
    *padded_items = padded;
    return ME_EVAL_SUCCESS;
}

static int collect_var_sizes(const me_expr* expr, size_t* var_sizes, int n_vars) {
    if (!expr || !var_sizes || n_vars <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    for (int i = 0; i < n_vars; i++) {
        var_sizes[i] = 0;
    }

    /* DFS to collect sizes from variable nodes (synthetic address index). */
    const me_expr* stack[256];
    int top = 0;
    stack[top++] = expr;
    while (top) {
        const me_expr* n = stack[--top];
        if (!n) continue;
        if (TYPE_MASK(n->type) == ME_VARIABLE && is_synthetic_address(n->bound)) {
            int idx = (int)((const char*)n->bound - synthetic_var_addresses);
            if (idx >= 0 && idx < n_vars && var_sizes[idx] == 0) {
                var_sizes[idx] = dtype_size(n->input_dtype);
            }
        }
        else if (IS_FUNCTION(n->type) || IS_CLOSURE(n->type)) {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity && top < 256; i++) {
                stack[top++] = (const me_expr*)n->parameters[i];
            }
        }
    }

    for (int i = 0; i < n_vars; i++) {
        if (var_sizes[i] == 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
    }
    return ME_EVAL_SUCCESS;
}

int me_eval_nd(const me_expr* expr, const void** vars_block,
               int n_vars, void* output_block, int block_nitems,
               int64_t nchunk, int64_t nblock, const me_eval_params* params) {
    if (!expr) {
        return ME_EVAL_ERR_NULL_EXPR;
    }
    if (!output_block || block_nitems <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t valid_items = 0;
    int64_t padded_items = 0;
    int rc = compute_valid_items(expr, nchunk, nblock, block_nitems, &valid_items, &padded_items);
    if (rc != ME_EVAL_SUCCESS) {
        return rc;
    }
    if (valid_items > INT_MAX) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const size_t item_size = dtype_size(me_get_dtype(expr));
    if (item_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const bool is_reduction_output = contains_reduction(expr) && output_is_scalar(expr);

    /* Fast path: no padding needed (valid == padded), single call. */
    if (valid_items == padded_items) {
        if (valid_items == 0) {
            /* Scalar outputs only write the first item. */
            if (is_reduction_output) {
                memset(output_block, 0, item_size);
            }
            else {
                memset(output_block, 0, (size_t)padded_items * item_size);
            }
            return ME_EVAL_SUCCESS;
        }
        return me_eval(expr, vars_block, n_vars, output_block, (int)valid_items, params);
    }

    const me_nd_info* info = (const me_nd_info*)expr->bytecode;
    const int nd = info->ndims;
    const int64_t* shape = info->data;
    const int64_t* chunkshape = shape + nd;
    const int64_t* blockshape = chunkshape + nd;

    size_t var_sizes[ME_MAX_VARS];
    rc = collect_var_sizes(expr, var_sizes, n_vars);
    if (rc != ME_EVAL_SUCCESS) {
        return rc;
    }

    /* Compute per-dim lengths for this chunk/block. */
    int64_t chunk_idx[64];
    int64_t block_idx[64];
    int64_t valid_len[64];
    if (nd > 64) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t tmp = nchunk;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
        tmp /= nchunks_d;
    }

    tmp = nblock;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
        tmp /= nblocks_d;
    }

    for (int i = 0; i < nd; i++) {
        const int64_t chunk_start = chunk_idx[i] * chunkshape[i];
        int64_t chunk_len = shape[i] - chunk_start;
        if (chunk_len > chunkshape[i]) {
            chunk_len = chunkshape[i];
        }
        const int64_t block_start = block_idx[i] * blockshape[i];
        if (block_start >= chunk_len) {
            valid_len[i] = 0;
        }
        else {
            int64_t len = chunk_len - block_start;
            if (len > blockshape[i]) {
                len = blockshape[i];
            }
            valid_len[i] = len;
        }
    }

    /* Strides inside padded block (C-order). */
    int64_t stride[64];
    stride[nd - 1] = 1;
    for (int i = nd - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * blockshape[i + 1];
    }

    /* Pack → single eval → scatter */
    if (valid_items == 0) {
        if (is_reduction_output) {
            memset(output_block, 0, item_size);
        }
        else {
            memset(output_block, 0, (size_t)padded_items * item_size);
        }
        return ME_EVAL_SUCCESS;
    }

    bool allow_repeat_reduce = false;
    me_reduce_kind rkind = ME_REDUCE_NONE;
    if (is_reduction_output && is_reduction_node(expr)) {
        rkind = reduction_kind(expr->function);
        if (rkind == ME_REDUCE_ANY || rkind == ME_REDUCE_ALL) {
            allow_repeat_reduce = true;
        }
        else if (rkind == ME_REDUCE_SUM) {
            const me_expr* arg = (const me_expr*)expr->parameters[0];
            if (arg && TYPE_MASK(arg->type) == ME_VARIABLE) {
                allow_repeat_reduce = true;
            }
        }
    }

    /* Decide whether repeat-eval is applicable, and precompute run layout. */
    int split_dim = -2;
    int64_t run_len = 0;
    int64_t total_runs = 0;
    bool repeat_eval_selected = false;
    if (!is_reduction_output || allow_repeat_reduce) {
        split_dim = nd - 2;
        run_len = valid_len[nd - 1];
        bool can_extend = (valid_len[nd - 1] == blockshape[nd - 1]);
        for (int i = nd - 2; i >= 0; i--) {
            if (can_extend && valid_len[i] == blockshape[i]) {
                if (run_len > LLONG_MAX / blockshape[i]) {
                    split_dim = -2;
                    break;
                }
                run_len *= blockshape[i];
                split_dim = i - 1;
            }
            else {
                break;
            }
        }

        if (split_dim >= -1 && run_len > 0 && run_len <= INT_MAX) {
            total_runs = 1;
            bool overflow = false;
            if (split_dim >= 0) {
                for (int i = 0; i <= split_dim; i++) {
                    if (total_runs > LLONG_MAX / valid_len[i]) {
                        overflow = true;
                        break;
                    }
                    total_runs *= valid_len[i];
                }
            }
            if (!overflow) {
                if (!is_reduction_output) {
                    repeat_eval_selected = true;
                }
                else if (rkind == ME_REDUCE_SUM) {
                    repeat_eval_selected = (total_runs <= 16);
                }
                else {
                    repeat_eval_selected = allow_repeat_reduce;
                }
            }
        }
    }

    /* Reduction fast paths (skip when repeat-eval is selected). */
    if (is_reduction_output && !repeat_eval_selected) {
        if (reduce_strided_predicate(expr, vars_block, n_vars, valid_len, stride, nd,
                                     valid_items, output_block)) {
            return ME_EVAL_SUCCESS;
        }
        if (reduce_strided_variable(expr, vars_block, n_vars, valid_len, stride, nd,
                                    valid_items, output_block)) {
            return ME_EVAL_SUCCESS;
        }
    }

    /* Repeat me_eval on contiguous valid runs instead of packing. */
    if (repeat_eval_selected) {
        const void* run_ptrs[ME_MAX_VARS];
        if (is_reduction_output) {
            me_scalar acc;
            bool acc_init = (rkind != ME_REDUCE_MIN && rkind != ME_REDUCE_MAX);
            const me_dtype output_type = expr->dtype;
            switch (output_type) {
            case ME_BOOL: acc.b = (rkind == ME_REDUCE_ALL); break;
            case ME_INT8:
            case ME_INT16:
            case ME_INT32:
            case ME_INT64: acc.i64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
            case ME_UINT8:
            case ME_UINT16:
            case ME_UINT32:
            case ME_UINT64: acc.u64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
            case ME_FLOAT32:
            case ME_FLOAT64: acc.f64 = (rkind == ME_REDUCE_PROD) ? 1.0 : 0.0; break;
            case ME_COMPLEX64: acc.c64 = (rkind == ME_REDUCE_PROD) ? (float _Complex)1.0f : (float _Complex)0.0f; break;
            case ME_COMPLEX128: acc.c128 = (rkind == ME_REDUCE_PROD) ? (double _Complex)1.0 : (double _Complex)0.0; break;
            default: acc_init = false; break;
            }

            int64_t indices[64] = {0};
            bool done = false;
            for (int64_t run = 0; run < total_runs && !done; run++) {
                int64_t off = 0;
                if (split_dim >= 0) {
                    for (int i = 0; i <= split_dim; i++) {
                        off += indices[i] * stride[i];
                    }
                }
                for (int v = 0; v < n_vars; v++) {
                    run_ptrs[v] = (const unsigned char*)vars_block[v] + (size_t)off * var_sizes[v];
                }
                me_scalar run_out;
                rc = me_eval(expr, run_ptrs, n_vars, &run_out, (int)run_len, params);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }

                me_scalar run_val;
                read_scalar(&run_out, output_type, &run_val);

                if (!acc_init) {
                    if (output_type == ME_FLOAT32) {
                        acc.f64 = (double)run_val.f32;
                    }
                    else if (output_type == ME_FLOAT64) {
                        acc.f64 = run_val.f64;
                    }
                    else {
                        acc = run_val;
                    }
                    acc_init = true;
                }
                else {
                    switch (rkind) {
                    case ME_REDUCE_SUM:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: acc.i64 += run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: acc.u64 += run_val.u64; break;
                        case ME_FLOAT32: acc.f64 += (double)run_val.f32; break;
                        case ME_FLOAT64: acc.f64 += run_val.f64; break;
                        case ME_COMPLEX64: acc.c64 += run_val.c64; break;
                        case ME_COMPLEX128: acc.c128 += run_val.c128; break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_PROD:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: acc.i64 *= run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: acc.u64 *= run_val.u64; break;
                        case ME_FLOAT32: acc.f64 *= (double)run_val.f32; break;
                        case ME_FLOAT64: acc.f64 *= run_val.f64; break;
                        case ME_COMPLEX64: acc.c64 *= run_val.c64; break;
                        case ME_COMPLEX128: acc.c128 *= run_val.c128; break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_MIN:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: if (run_val.i64 < acc.i64) acc.i64 = run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: if (run_val.u64 < acc.u64) acc.u64 = run_val.u64; break;
                        case ME_FLOAT32:
                            if (run_val.f32 != run_val.f32) { acc.f64 = NAN; done = true; }
                            else if (run_val.f32 < (float)acc.f64) acc.f64 = (double)run_val.f32;
                            break;
                        case ME_FLOAT64:
                            if (run_val.f64 != run_val.f64) { acc.f64 = NAN; done = true; }
                            else if (run_val.f64 < acc.f64) acc.f64 = run_val.f64;
                            break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_MAX:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: if (run_val.i64 > acc.i64) acc.i64 = run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: if (run_val.u64 > acc.u64) acc.u64 = run_val.u64; break;
                        case ME_FLOAT32:
                            if (run_val.f32 != run_val.f32) { acc.f64 = NAN; done = true; }
                            else if (run_val.f32 > (float)acc.f64) acc.f64 = (double)run_val.f32;
                            break;
                        case ME_FLOAT64:
                            if (run_val.f64 != run_val.f64) { acc.f64 = NAN; done = true; }
                            else if (run_val.f64 > acc.f64) acc.f64 = run_val.f64;
                            break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_ANY:
                        if (output_type == ME_BOOL) {
                            acc.b = acc.b || run_val.b;
                            if (acc.b) done = true;
                        }
                        break;
                    case ME_REDUCE_ALL:
                        if (output_type == ME_BOOL) {
                            acc.b = acc.b && run_val.b;
                            if (!acc.b) done = true;
                        }
                        break;
                    default:
                        break;
                    }
                }

                if (split_dim >= 0) {
                    for (int i = split_dim; i >= 0; i--) {
                        indices[i]++;
                        if (indices[i] < valid_len[i]) break;
                        indices[i] = 0;
                    }
                }
            }

            if (output_type == ME_FLOAT32) {
                acc.f32 = (float)acc.f64;
            }
            write_scalar(output_block, output_type, output_type, &acc);
            return ME_EVAL_SUCCESS;
        }
        else {
            memset(output_block, 0, (size_t)padded_items * item_size);
            int64_t indices[64] = {0};
            for (int64_t run = 0; run < total_runs; run++) {
                int64_t off = 0;
                if (split_dim >= 0) {
                    for (int i = 0; i <= split_dim; i++) {
                        off += indices[i] * stride[i];
                    }
                }
                for (int v = 0; v < n_vars; v++) {
                    run_ptrs[v] = (const unsigned char*)vars_block[v] + (size_t)off * var_sizes[v];
                }
                void* out_ptr = (unsigned char*)output_block + (size_t)off * item_size;
                rc = me_eval(expr, run_ptrs, n_vars, out_ptr, (int)run_len, params);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }
                if (split_dim >= 0) {
                    for (int i = split_dim; i >= 0; i--) {
                        indices[i]++;
                        if (indices[i] < valid_len[i]) break;
                        indices[i] = 0;
                    }
                }
            }
            return ME_EVAL_SUCCESS;
        }
    }

    void* packed_vars[ME_MAX_VARS];
    for (int v = 0; v < n_vars; v++) {
        packed_vars[v] = malloc((size_t)valid_items * var_sizes[v]);
        if (!packed_vars[v]) {
            for (int u = 0; u < v; u++) free(packed_vars[u]);
            return ME_EVAL_ERR_OOM;
        }
    }
    void* packed_out = NULL;
    if (!is_reduction_output) {
        packed_out = malloc((size_t)valid_items * item_size);
        if (!packed_out) {
            for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
            return ME_EVAL_ERR_OOM;
        }
    }

    /* Pack valid elements */
    int64_t indices[64] = {0};
    int64_t write_idx = 0;
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) total_iters *= valid_len[i];
    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }
        for (int v = 0; v < n_vars; v++) {
            const unsigned char* src = (const unsigned char*)vars_block[v] + (size_t)off * var_sizes[v];
            memcpy((unsigned char*)packed_vars[v] + (size_t)write_idx * var_sizes[v], src, var_sizes[v]);
        }
        write_idx++;
        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) break;
            indices[i] = 0;
        }
    }

    if (is_reduction_output) {
        rc = me_eval(expr, (const void**)packed_vars, n_vars, output_block, (int)valid_items, params);
        if (rc != ME_EVAL_SUCCESS) {
            for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
            return rc;
        }
    }
    else {
        rc = me_eval(expr, (const void**)packed_vars, n_vars, packed_out, (int)valid_items, params);
        if (rc != ME_EVAL_SUCCESS) {
            for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
            free(packed_out);
            return rc;
        }

        /* Scatter back and zero padding */
        memset(output_block, 0, (size_t)padded_items * item_size);
        indices[0] = 0;
        for (int i = 1; i < nd; i++) indices[i] = 0;
        write_idx = 0;
        for (int64_t it = 0; it < total_iters; it++) {
            int64_t off = 0;
            for (int i = 0; i < nd; i++) {
                off += indices[i] * stride[i];
            }
            unsigned char* dst = (unsigned char*)output_block + (size_t)off * item_size;
            memcpy(dst, (unsigned char*)packed_out + (size_t)write_idx * item_size, item_size);
            write_idx++;
            for (int i = nd - 1; i >= 0; i--) {
                indices[i]++;
                if (indices[i] < valid_len[i]) break;
                indices[i] = 0;
            }
        }
    }

    for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
    free(packed_out);

    return ME_EVAL_SUCCESS;
}

int me_nd_valid_nitems(const me_expr* expr, int64_t nchunk, int64_t nblock, int64_t* valid_nitems) {
    if (!valid_nitems) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    int64_t padded = 0;
    int rc = compute_valid_items(expr, nchunk, nblock, -1, valid_nitems, &padded);
    return rc;
}
