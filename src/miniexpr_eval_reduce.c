/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "miniexpr_internal.h"
#include "dsl_eval_internal.h"

#include <math.h>

static float _Complex me_cmplxf(float re, float im) {
#if defined(_MSC_VER)
    float _Complex v;
    __real__ v = re;
    __imag__ v = im;
    return v;
#else
    return re + im * I;
#endif
}

static double _Complex me_cmplx(double re, double im) {
#if defined(_MSC_VER)
    double _Complex v;
    __real__ v = re;
    __imag__ v = im;
    return v;
#else
    return re + im * I;
#endif
}

void write_scalar(void *out, me_dtype out_type, me_dtype in_type, const me_scalar *v) {
    if (out_type == in_type) {
        switch (out_type) {
        case ME_BOOL: *(bool *)out = v->b; return;
        case ME_INT8: *(int8_t *)out = (int8_t)v->i64; return;
        case ME_INT16: *(int16_t *)out = (int16_t)v->i64; return;
        case ME_INT32: *(int32_t *)out = (int32_t)v->i64; return;
        case ME_INT64: *(int64_t *)out = v->i64; return;
        case ME_UINT8: *(uint8_t *)out = (uint8_t)v->u64; return;
        case ME_UINT16: *(uint16_t *)out = (uint16_t)v->u64; return;
        case ME_UINT32: *(uint32_t *)out = (uint32_t)v->u64; return;
        case ME_UINT64: *(uint64_t *)out = v->u64; return;
        case ME_FLOAT32: *(float *)out = v->f32; return;
        case ME_FLOAT64: *(double *)out = v->f64; return;
        case ME_COMPLEX64: *(float _Complex *)out = v->c64; return;
        case ME_COMPLEX128: *(double _Complex *)out = v->c128; return;
        default: return;
        }
    }

    switch (out_type) {
    case ME_BOOL:
        switch (in_type) {
        case ME_BOOL: *(bool *)out = v->b; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(bool *)out = v->i64 != 0; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(bool *)out = v->u64 != 0; break;
        case ME_FLOAT32: *(bool *)out = v->f32 != 0.0f; break;
        case ME_FLOAT64: *(bool *)out = v->f64 != 0.0; break;
        case ME_COMPLEX64: *(bool *)out = (me_crealf(v->c64) != 0.0f || me_cimagf(v->c64) != 0.0f); break;
        case ME_COMPLEX128: *(bool *)out = (me_creal(v->c128) != 0.0 || me_cimag(v->c128) != 0.0); break;
        default: *(bool *)out = false; break;
        }
        break;
    case ME_INT8:
    case ME_INT16:
    case ME_INT32:
    case ME_INT64:
        switch (in_type) {
        case ME_BOOL: *(int64_t *)out = v->b ? 1 : 0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(int64_t *)out = v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(int64_t *)out = (int64_t)v->u64; break;
        case ME_FLOAT32: *(int64_t *)out = (int64_t)v->f32; break;
        case ME_FLOAT64: *(int64_t *)out = (int64_t)v->f64; break;
        default: *(int64_t *)out = 0; break;
        }
        break;
    case ME_UINT8:
    case ME_UINT16:
    case ME_UINT32:
    case ME_UINT64:
        switch (in_type) {
        case ME_BOOL: *(uint64_t *)out = v->b ? 1 : 0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(uint64_t *)out = (uint64_t)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(uint64_t *)out = v->u64; break;
        case ME_FLOAT32: *(uint64_t *)out = (uint64_t)v->f32; break;
        case ME_FLOAT64: *(uint64_t *)out = (uint64_t)v->f64; break;
        default: *(uint64_t *)out = 0; break;
        }
        break;
    case ME_FLOAT32:
        switch (in_type) {
        case ME_BOOL: *(float *)out = v->b ? 1.0f : 0.0f; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(float *)out = (float)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(float *)out = (float)v->u64; break;
        case ME_FLOAT32: *(float *)out = v->f32; break;
        case ME_FLOAT64: *(float *)out = (float)v->f64; break;
        default: *(float *)out = 0.0f; break;
        }
        break;
    case ME_FLOAT64:
        switch (in_type) {
        case ME_BOOL: *(double *)out = v->b ? 1.0 : 0.0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(double *)out = (double)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(double *)out = (double)v->u64; break;
        case ME_FLOAT32: *(double *)out = (double)v->f32; break;
        case ME_FLOAT64: *(double *)out = v->f64; break;
        default: *(double *)out = 0.0; break;
        }
        break;
    case ME_COMPLEX64:
        switch (in_type) {
        case ME_COMPLEX64: *(float _Complex *)out = v->c64; break;
        case ME_COMPLEX128: *(float _Complex *)out = (float _Complex)v->c128; break;
        case ME_FLOAT32: *(float _Complex *)out = me_cmplxf(v->f32, 0.0f); break;
        case ME_FLOAT64: *(float _Complex *)out = me_cmplxf((float)v->f64, 0.0f); break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(float _Complex *)out = me_cmplxf((float)v->i64, 0.0f); break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(float _Complex *)out = me_cmplxf((float)v->u64, 0.0f); break;
        case ME_BOOL: *(float _Complex *)out = me_cmplxf(v->b ? 1.0f : 0.0f, 0.0f); break;
        default: *(float _Complex *)out = me_cmplxf(0.0f, 0.0f); break;
        }
        break;
    case ME_COMPLEX128:
        switch (in_type) {
        case ME_COMPLEX64: *(double _Complex *)out = (double _Complex)v->c64; break;
        case ME_COMPLEX128: *(double _Complex *)out = v->c128; break;
        case ME_FLOAT32: *(double _Complex *)out = me_cmplx((double)v->f32, 0.0); break;
        case ME_FLOAT64: *(double _Complex *)out = me_cmplx(v->f64, 0.0); break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(double _Complex *)out = me_cmplx((double)v->i64, 0.0); break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(double _Complex *)out = me_cmplx((double)v->u64, 0.0); break;
        case ME_BOOL: *(double _Complex *)out = me_cmplx(v->b ? 1.0 : 0.0, 0.0); break;
        default: *(double _Complex *)out = me_cmplx(0.0, 0.0); break;
        }
        break;
    default:
        break;
    }
}

void read_scalar(const void *in, me_dtype in_type, me_scalar *v) {
    switch (in_type) {
    case ME_BOOL: v->b = *(const bool *)in; break;
    case ME_INT8: v->i64 = *(const int8_t *)in; break;
    case ME_INT16: v->i64 = *(const int16_t *)in; break;
    case ME_INT32: v->i64 = *(const int32_t *)in; break;
    case ME_INT64: v->i64 = *(const int64_t *)in; break;
    case ME_UINT8: v->u64 = *(const uint8_t *)in; break;
    case ME_UINT16: v->u64 = *(const uint16_t *)in; break;
    case ME_UINT32: v->u64 = *(const uint32_t *)in; break;
    case ME_UINT64: v->u64 = *(const uint64_t *)in; break;
    case ME_FLOAT32: v->f32 = *(const float *)in; break;
    case ME_FLOAT64: v->f64 = *(const double *)in; break;
    case ME_COMPLEX64: v->c64 = *(const float _Complex *)in; break;
    case ME_COMPLEX128: v->c128 = *(const double _Complex *)in; break;
    default: break;
    }
}

static bool read_as_bool(const void *base, int64_t off, me_dtype type, bool *out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool *)base)[off]; return true;
    case ME_INT8: *out = ((const int8_t *)base)[off] != 0; return true;
    case ME_INT16: *out = ((const int16_t *)base)[off] != 0; return true;
    case ME_INT32: *out = ((const int32_t *)base)[off] != 0; return true;
    case ME_INT64: *out = ((const int64_t *)base)[off] != 0; return true;
    case ME_UINT8: *out = ((const uint8_t *)base)[off] != 0; return true;
    case ME_UINT16: *out = ((const uint16_t *)base)[off] != 0; return true;
    case ME_UINT32: *out = ((const uint32_t *)base)[off] != 0; return true;
    case ME_UINT64: *out = ((const uint64_t *)base)[off] != 0; return true;
    case ME_FLOAT32: *out = ((const float *)base)[off] != 0.0f; return true;
    case ME_FLOAT64: *out = ((const double *)base)[off] != 0.0; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_int64(const void *base, int64_t off, me_dtype type, int64_t *out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool *)base)[off] ? 1 : 0; return true;
    case ME_INT8: *out = ((const int8_t *)base)[off]; return true;
    case ME_INT16: *out = ((const int16_t *)base)[off]; return true;
    case ME_INT32: *out = ((const int32_t *)base)[off]; return true;
    case ME_INT64: *out = ((const int64_t *)base)[off]; return true;
    case ME_UINT8: *out = (int64_t)((const uint8_t *)base)[off]; return true;
    case ME_UINT16: *out = (int64_t)((const uint16_t *)base)[off]; return true;
    case ME_UINT32: *out = (int64_t)((const uint32_t *)base)[off]; return true;
    case ME_UINT64: *out = (int64_t)((const uint64_t *)base)[off]; return true;
    case ME_FLOAT32: *out = (int64_t)((const float *)base)[off]; return true;
    case ME_FLOAT64: *out = (int64_t)((const double *)base)[off]; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_uint64(const void *base, int64_t off, me_dtype type, uint64_t *out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool *)base)[off] ? 1 : 0; return true;
    case ME_INT8: *out = (uint64_t)((const int8_t *)base)[off]; return true;
    case ME_INT16: *out = (uint64_t)((const int16_t *)base)[off]; return true;
    case ME_INT32: *out = (uint64_t)((const int32_t *)base)[off]; return true;
    case ME_INT64: *out = (uint64_t)((const int64_t *)base)[off]; return true;
    case ME_UINT8: *out = ((const uint8_t *)base)[off]; return true;
    case ME_UINT16: *out = ((const uint16_t *)base)[off]; return true;
    case ME_UINT32: *out = ((const uint32_t *)base)[off]; return true;
    case ME_UINT64: *out = ((const uint64_t *)base)[off]; return true;
    case ME_FLOAT32: *out = (uint64_t)((const float *)base)[off]; return true;
    case ME_FLOAT64: *out = (uint64_t)((const double *)base)[off]; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_float(const void *base, int64_t off, me_dtype type, float *out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool *)base)[off] ? 1.0f : 0.0f; return true;
    case ME_INT8: *out = (float)((const int8_t *)base)[off]; return true;
    case ME_INT16: *out = (float)((const int16_t *)base)[off]; return true;
    case ME_INT32: *out = (float)((const int32_t *)base)[off]; return true;
    case ME_INT64: *out = (float)((const int64_t *)base)[off]; return true;
    case ME_UINT8: *out = (float)((const uint8_t *)base)[off]; return true;
    case ME_UINT16: *out = (float)((const uint16_t *)base)[off]; return true;
    case ME_UINT32: *out = (float)((const uint32_t *)base)[off]; return true;
    case ME_UINT64: *out = (float)((const uint64_t *)base)[off]; return true;
    case ME_FLOAT32: *out = ((const float *)base)[off]; return true;
    case ME_FLOAT64: *out = (float)((const double *)base)[off]; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_double(const void *base, int64_t off, me_dtype type, double *out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool *)base)[off] ? 1.0 : 0.0; return true;
    case ME_INT8: *out = (double)((const int8_t *)base)[off]; return true;
    case ME_INT16: *out = (double)((const int16_t *)base)[off]; return true;
    case ME_INT32: *out = (double)((const int32_t *)base)[off]; return true;
    case ME_INT64: *out = (double)((const int64_t *)base)[off]; return true;
    case ME_UINT8: *out = (double)((const uint8_t *)base)[off]; return true;
    case ME_UINT16: *out = (double)((const uint16_t *)base)[off]; return true;
    case ME_UINT32: *out = (double)((const uint32_t *)base)[off]; return true;
    case ME_UINT64: *out = (double)((const uint64_t *)base)[off]; return true;
    case ME_FLOAT32: *out = (double)((const float *)base)[off]; return true;
    case ME_FLOAT64: *out = ((const double *)base)[off]; return true;
    case ME_STRING: return false;
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

bool reduce_strided_variable(const me_expr *expr, const void **vars_block, int n_vars,
                             const int64_t *valid_len, const int64_t *stride, int nd,
                             int64_t valid_items, void *output_block) {
    if (!expr || !is_reduction_node(expr) || valid_items <= 0) {
        return false;
    }
    const me_expr *arg = (const me_expr *)expr->parameters[0];
    if (!arg || TYPE_MASK(arg->type) != ME_VARIABLE || !is_synthetic_address(arg->bound)) {
        return false;
    }
    int idx = (int)((const char *)arg->bound - synthetic_var_addresses);
    if (idx < 0 || idx >= n_vars) {
        return false;
    }

    const me_reduce_kind rkind = reduction_kind(expr->function);
    if (rkind == ME_REDUCE_NONE) {
        return false;
    }
    const bool is_mean = (rkind == ME_REDUCE_MEAN);

    const me_dtype arg_type = infer_result_type(arg);
    const me_dtype result_type = reduction_output_dtype(arg_type, expr->function);
    const me_dtype output_type = expr->dtype;

    int64_t indices[64] = {0};
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) {
        total_iters *= valid_len[i];
    }

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

    const unsigned char *base = (const unsigned char *)vars_block[idx];
    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }

        switch (arg_type) {
        case ME_BOOL: {
            bool v = ((const bool *)base)[off];
            if (is_mean) {
                acc.f64 += v ? 1.0 : 0.0;
            }
            else if (rkind == ME_REDUCE_ANY) { if (v) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!v) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v ? 1 : 0;
            else acc.i64 += v ? 1 : 0;
            break;
        }
        case ME_INT8: {
            int8_t v = ((const int8_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int8_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int8_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT16: {
            int16_t v = ((const int16_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int16_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int16_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT32: {
            int32_t v = ((const int32_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int32_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int32_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT64: {
            int64_t v = ((const int64_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_UINT8: {
            uint8_t v = ((const uint8_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint8_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint8_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT16: {
            uint16_t v = ((const uint16_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint16_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint16_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT32: {
            uint32_t v = ((const uint32_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint32_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint32_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT64: {
            uint64_t v = ((const uint64_t *)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_FLOAT32: {
            float v = ((const float *)base)[off];
            if (v != v) { acc.f64 = NAN; goto done_reduce; }
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (float)acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (float)acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.f64 *= (double)v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0.0f) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0.0f) { acc.b = false; goto done_reduce; } }
            else acc.f64 += (double)v;
            break;
        }
        case ME_FLOAT64: {
            double v = ((const double *)base)[off];
            if (v != v) { acc.f64 = NAN; goto done_reduce; }
            if (is_mean) { acc.f64 += v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.f64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0.0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0.0) { acc.b = false; goto done_reduce; } }
            else acc.f64 += v;
            break;
        }
        case ME_COMPLEX64: {
            float _Complex v = ((const float _Complex *)base)[off];
            bool nonzero = (me_crealf(v) != 0.0f || me_cimagf(v) != 0.0f);
            if (is_mean) { acc.c128 += (double _Complex)v; }
            else if (rkind == ME_REDUCE_ANY) { if (nonzero) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!nonzero) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.c64 *= v;
            else acc.c64 += v;
            break;
        }
        case ME_COMPLEX128: {
            double _Complex v = ((const double _Complex *)base)[off];
            bool nonzero = (me_creal(v) != 0.0 || me_cimag(v) != 0.0);
            if (is_mean) { acc.c128 += v; }
            else if (rkind == ME_REDUCE_ANY) { if (nonzero) { acc.b = true; goto done_reduce; } }
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
            if (indices[i] < valid_len[i]) {
                break;
            }
            indices[i] = 0;
        }
    }

done_reduce:
    if (is_mean) {
        if (result_type == ME_COMPLEX128) {
            acc.c128 /= (double)valid_items;
        }
        else {
            acc.f64 /= (double)valid_items;
        }
    }
    if (result_type == ME_FLOAT32) {
        acc.f32 = (float)acc.f64;
    }
    write_scalar(output_block, output_type, result_type, &acc);
    return true;
}

bool reduce_strided_predicate(const me_expr *expr, const void **vars_block, int n_vars,
                              const int64_t *valid_len, const int64_t *stride, int nd,
                              int64_t valid_items, void *output_block) {
    if (!expr || !is_reduction_node(expr) || valid_items <= 0) {
        return false;
    }
    const me_expr *arg = (const me_expr *)expr->parameters[0];
    if (!arg || !is_comparison_node(arg)) {
        return false;
    }

    me_reduce_kind rkind = reduction_kind(expr->function);
    if (!(rkind == ME_REDUCE_ANY || rkind == ME_REDUCE_ALL)) {
        return false;
    }

    const me_expr *left = (const me_expr *)arg->parameters[0];
    const me_expr *right = (const me_expr *)arg->parameters[1];
    if (!left || !right) {
        return false;
    }

    const me_expr *var_node = NULL;
    const me_expr *const_node = NULL;
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
    int idx = (int)((const char *)var_node->bound - synthetic_var_addresses);
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
    for (int i = 0; i < nd; i++) {
        total_iters *= valid_len[i];
    }

    me_scalar acc;
    acc.b = (rkind == ME_REDUCE_ALL);

    const unsigned char *base = (const unsigned char *)vars_block[idx];
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
            if (indices[i] < valid_len[i]) {
                break;
            }
            indices[i] = 0;
        }
    }

done_pred:
    write_scalar(output_block, output_type, result_type, &acc);
    return true;
}
