/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "miniexpr_internal.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int64_t ceil_div64(int64_t a, int64_t b) {
    return (b == 0) ? 0 : (a + b - 1) / b;
}

static double _Complex me_nd_cmplx(double re, double im) {
#if defined(_MSC_VER)
    double _Complex v;
    __real__ v = re;
    __imag__ v = im;
    return v;
#else
    return re + im * I;
#endif
}

int me_nd_compute_valid_items(const me_expr *expr, int64_t nchunk, int64_t nblock,
                              int chunk_nitems, int64_t *valid_items, int64_t *padded_items) {
    if (!expr || !valid_items || !padded_items) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const me_nd_info *info = (const me_nd_info *)expr->bytecode;
    if (!info || info->ndims <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const int nd = info->ndims;
    if (nd > 64) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    const int64_t *shape = info->data;
    const int64_t *chunkshape = shape + nd;
    const int64_t *blockshape = chunkshape + nd;

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

static int collect_var_sizes(const me_expr *expr, size_t *var_sizes, int n_vars) {
    if (!expr || !var_sizes || n_vars <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    for (int i = 0; i < n_vars; i++) {
        var_sizes[i] = 0;
    }

    /* DFS to collect sizes from variable nodes (synthetic address index). */
    const me_expr *stack[256];
    int top = 0;
    stack[top++] = expr;
    while (top) {
        const me_expr *n = stack[--top];
        if (!n) {
            continue;
        }
        if (TYPE_MASK(n->type) == ME_VARIABLE && is_synthetic_address(n->bound)) {
            int idx = (int)((const char *)n->bound - synthetic_var_addresses);
            if (idx >= 0 && idx < n_vars && var_sizes[idx] == 0) {
                if ((n->dtype == ME_STRING || n->input_dtype == ME_STRING) && n->itemsize > 0) {
                    var_sizes[idx] = n->itemsize;
                }
                else {
                    var_sizes[idx] = dtype_size(n->input_dtype);
                }
            }
        }
        else if (IS_FUNCTION(n->type) || IS_CLOSURE(n->type)) {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity && top < 256; i++) {
                stack[top++] = (const me_expr *)n->parameters[i];
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

int me_eval_nd_classic(const me_expr *expr, const void **vars_block,
                       int n_vars, void *output_block, int block_nitems,
                       int64_t nchunk, int64_t nblock, const me_eval_params *params) {
    if (!expr) {
        return ME_EVAL_ERR_NULL_EXPR;
    }
    if (!output_block || block_nitems <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t valid_items = 0;
    int64_t padded_items = 0;
    int rc = me_nd_compute_valid_items(expr, nchunk, nblock, block_nitems, &valid_items, &padded_items);
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

    if (valid_items == padded_items) {
        if (valid_items == 0) {
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

    const me_nd_info *info = (const me_nd_info *)expr->bytecode;
    const int nd = info->ndims;
    const int64_t *shape = info->data;
    const int64_t *chunkshape = shape + nd;
    const int64_t *blockshape = chunkshape + nd;

    size_t var_sizes[ME_MAX_VARS];
    rc = collect_var_sizes(expr, var_sizes, n_vars);
    if (rc != ME_EVAL_SUCCESS) {
        return rc;
    }

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

    int64_t stride[64];
    stride[nd - 1] = 1;
    for (int i = nd - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * blockshape[i + 1];
    }

    if (valid_items == 0) {
        if (is_reduction_output) {
            if (is_reduction_node(expr) && reduction_kind(expr->function) == ME_REDUCE_MEAN) {
                const me_expr *arg = (const me_expr *)expr->parameters[0];
                me_dtype arg_type = arg ? infer_result_type(arg) : ME_FLOAT64;
                me_dtype result_type = reduction_output_dtype(arg_type, expr->function);
                me_scalar acc;
                if (result_type == ME_COMPLEX128) {
                    acc.c128 = me_nd_cmplx(NAN, NAN);
                }
                else {
                    acc.f64 = NAN;
                }
                write_scalar(output_block, expr->dtype, result_type, &acc);
            }
            else {
                memset(output_block, 0, item_size);
            }
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
            const me_expr *arg = (const me_expr *)expr->parameters[0];
            if (arg && TYPE_MASK(arg->type) == ME_VARIABLE) {
                allow_repeat_reduce = true;
            }
        }
    }

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

    if (repeat_eval_selected) {
        const void *run_ptrs[ME_MAX_VARS];
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
                    run_ptrs[v] = (const unsigned char *)vars_block[v] + (size_t)off * var_sizes[v];
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
                            if (run_val.f32 != run_val.f32) {
                                acc.f64 = NAN;
                                done = true;
                            }
                            else if (run_val.f32 < (float)acc.f64) {
                                acc.f64 = (double)run_val.f32;
                            }
                            break;
                        case ME_FLOAT64:
                            if (run_val.f64 != run_val.f64) {
                                acc.f64 = NAN;
                                done = true;
                            }
                            else if (run_val.f64 < acc.f64) {
                                acc.f64 = run_val.f64;
                            }
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
                            if (run_val.f32 != run_val.f32) {
                                acc.f64 = NAN;
                                done = true;
                            }
                            else if (run_val.f32 > (float)acc.f64) {
                                acc.f64 = (double)run_val.f32;
                            }
                            break;
                        case ME_FLOAT64:
                            if (run_val.f64 != run_val.f64) {
                                acc.f64 = NAN;
                                done = true;
                            }
                            else if (run_val.f64 > acc.f64) {
                                acc.f64 = run_val.f64;
                            }
                            break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_ANY:
                        if (output_type == ME_BOOL) {
                            acc.b = acc.b || run_val.b;
                            if (acc.b) {
                                done = true;
                            }
                        }
                        break;
                    case ME_REDUCE_ALL:
                        if (output_type == ME_BOOL) {
                            acc.b = acc.b && run_val.b;
                            if (!acc.b) {
                                done = true;
                            }
                        }
                        break;
                    default:
                        break;
                    }
                }

                if (split_dim >= 0) {
                    for (int i = split_dim; i >= 0; i--) {
                        indices[i]++;
                        if (indices[i] < valid_len[i]) {
                            break;
                        }
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
                    run_ptrs[v] = (const unsigned char *)vars_block[v] + (size_t)off * var_sizes[v];
                }
                void *out_ptr = (unsigned char *)output_block + (size_t)off * item_size;
                rc = me_eval(expr, run_ptrs, n_vars, out_ptr, (int)run_len, params);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }
                if (split_dim >= 0) {
                    for (int i = split_dim; i >= 0; i--) {
                        indices[i]++;
                        if (indices[i] < valid_len[i]) {
                            break;
                        }
                        indices[i] = 0;
                    }
                }
            }
            return ME_EVAL_SUCCESS;
        }
    }

    void *packed_vars[ME_MAX_VARS];
    for (int v = 0; v < n_vars; v++) {
        packed_vars[v] = malloc((size_t)valid_items * var_sizes[v]);
        if (!packed_vars[v]) {
            for (int u = 0; u < v; u++) {
                free(packed_vars[u]);
            }
            return ME_EVAL_ERR_OOM;
        }
    }
    void *packed_out = NULL;
    if (!is_reduction_output) {
        packed_out = malloc((size_t)valid_items * item_size);
        if (!packed_out) {
            for (int v = 0; v < n_vars; v++) {
                free(packed_vars[v]);
            }
            return ME_EVAL_ERR_OOM;
        }
    }

    int64_t indices[64] = {0};
    int64_t write_idx = 0;
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) {
        total_iters *= valid_len[i];
    }
    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }
        for (int v = 0; v < n_vars; v++) {
            const unsigned char *src = (const unsigned char *)vars_block[v] + (size_t)off * var_sizes[v];
            memcpy((unsigned char *)packed_vars[v] + (size_t)write_idx * var_sizes[v], src, var_sizes[v]);
        }
        write_idx++;
        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) {
                break;
            }
            indices[i] = 0;
        }
    }

    if (is_reduction_output) {
        rc = me_eval(expr, (const void **)packed_vars, n_vars, output_block, (int)valid_items, params);
        if (rc != ME_EVAL_SUCCESS) {
            for (int v = 0; v < n_vars; v++) {
                free(packed_vars[v]);
            }
            return rc;
        }
    }
    else {
        rc = me_eval(expr, (const void **)packed_vars, n_vars, packed_out, (int)valid_items, params);
        if (rc != ME_EVAL_SUCCESS) {
            for (int v = 0; v < n_vars; v++) {
                free(packed_vars[v]);
            }
            free(packed_out);
            return rc;
        }

        memset(output_block, 0, (size_t)padded_items * item_size);
        indices[0] = 0;
        for (int i = 1; i < nd; i++) {
            indices[i] = 0;
        }
        write_idx = 0;
        for (int64_t it = 0; it < total_iters; it++) {
            int64_t off = 0;
            for (int i = 0; i < nd; i++) {
                off += indices[i] * stride[i];
            }
            unsigned char *dst = (unsigned char *)output_block + (size_t)off * item_size;
            memcpy(dst, (unsigned char *)packed_out + (size_t)write_idx * item_size, item_size);
            write_idx++;
            for (int i = nd - 1; i >= 0; i--) {
                indices[i]++;
                if (indices[i] < valid_len[i]) {
                    break;
                }
                indices[i] = 0;
            }
        }
    }

    for (int v = 0; v < n_vars; v++) {
        free(packed_vars[v]);
    }
    free(packed_out);

    return ME_EVAL_SUCCESS;
}

int me_nd_valid_nitems(const me_expr *expr, int64_t nchunk, int64_t nblock, int64_t *valid_nitems) {
    if (!valid_nitems) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    int64_t padded = 0;
    return me_nd_compute_valid_items(expr, nchunk, nblock, -1, valid_nitems, &padded);
}
