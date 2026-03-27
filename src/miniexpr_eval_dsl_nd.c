/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "miniexpr_internal.h"
#include "dsl_eval_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int64_t ceil_div64(int64_t a, int64_t b) {
    return (b == 0) ? 0 : (a + b - 1) / b;
}

static int64_t dsl_i64_mul_wrap_local(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a * (uint64_t)b);
}

/* ND synthesis context layout v2:
   [0] = ndim
   [1 .. 1+ndim-1] = shape[d]
   [1+ndim .. 1+2*ndim-1] = shape_stride[d]
   [1+2*ndim .. 1+3*ndim-1] = base_idx[d]
   [1+3*ndim .. 1+4*ndim-1] = iter_len[d]
   [1+4*ndim] = abi version (2)
   [1+4*ndim+1] = flags (bit0: seq contiguous walk)
   [1+4*ndim+2] = global_linear_base (wrap-safe int64 arithmetic) */
static bool dsl_build_nd_synth_ctx(int nd,
                                   const int64_t *shape,
                                   const int64_t *shape_stride,
                                   const int64_t *base_idx,
                                   const int64_t *iter_len,
                                   int64_t *out_ctx,
                                   size_t out_ctx_len) {
    if (!shape || !shape_stride || !base_idx || !iter_len || !out_ctx || out_ctx_len == 0) {
        return false;
    }
    memset(out_ctx, 0, out_ctx_len * sizeof(*out_ctx));
    if (nd <= 0 || nd > ME_DSL_MAX_NDIM) {
        return false;
    }
    const size_t tail = (size_t)(1 + 4 * nd);
    const size_t need = tail + 3;
    if (out_ctx_len < need) {
        return false;
    }
    out_ctx[0] = (int64_t)nd;
    for (int d = 0; d < nd; d++) {
        out_ctx[1 + d] = shape[d];
        out_ctx[1 + nd + d] = shape_stride[d];
        out_ctx[1 + 2 * nd + d] = base_idx[d];
        out_ctx[1 + 3 * nd + d] = iter_len[d];
    }

    bool seq = true;
    for (int d = nd - 1; d >= 1; d--) {
        if (base_idx[d] != 0 || iter_len[d] != shape[d]) {
            seq = false;
            break;
        }
    }
    int64_t glin_base = 0;
    for (int d = 0; d < nd; d++) {
        glin_base = dsl_i64_addmul_wrap(glin_base, base_idx[d], shape_stride[d]);
    }
    out_ctx[tail] = (int64_t)ME_DSL_JIT_SYNTH_ND_CTX_V2_VERSION;
    out_ctx[tail + 1] = seq ? (int64_t)ME_DSL_ND_CTX_FLAG_SEQ : 0;
    out_ctx[tail + 2] = glin_base;
    return true;
}

int me_eval_nd_dsl(const me_expr *expr, const void **vars_block,
                   int n_vars, void *output_block, int block_nitems,
                   int64_t nchunk, int64_t nblock,
                   const me_eval_params *params) {
    if (!expr || !expr->dsl_program) {
        return ME_EVAL_ERR_NULL_EXPR;
    }
    const me_dsl_compiled_program *program = (const me_dsl_compiled_program *)expr->dsl_program;
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

    const me_nd_info *info = (const me_nd_info *)expr->bytecode;
    if (!info || info->ndims <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    const int nd = info->ndims;
    const int64_t *shape = info->data;
    const int64_t *chunkshape = shape + nd;
    const int64_t *blockshape = chunkshape + nd;

    const size_t item_size = dtype_size(me_get_dtype(expr));
    if (item_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    if (valid_items == 0) {
        if (program->output_is_scalar) {
            memset(output_block, 0, item_size);
        }
        else {
            memset(output_block, 0, (size_t)padded_items * item_size);
        }
        return ME_EVAL_SUCCESS;
    }

    int64_t chunk_idx[64];
    int64_t block_idx[64];
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

    int64_t base_idx[64];
    for (int i = 0; i < nd; i++) {
        int64_t chunk_term = dsl_i64_mul_wrap_local(chunk_idx[i], chunkshape[i]);
        base_idx[i] = dsl_i64_addmul_wrap(chunk_term, block_idx[i], blockshape[i]);
    }
    int64_t shape_stride[64];
    shape_stride[nd - 1] = 1;
    for (int i = nd - 2; i >= 0; i--) {
        shape_stride[i] = dsl_i64_mul_wrap_local(shape_stride[i + 1], shape[i + 1]);
    }

    int64_t *idx_buffers[ME_DSL_MAX_NDIM];
    for (int i = 0; i < ME_DSL_MAX_NDIM; i++) {
        idx_buffers[i] = NULL;
    }
    int64_t *global_linear_idx_buffer = NULL;
    int64_t nd_synth_ctx[ME_DSL_JIT_SYNTH_ND_CTX_WORDS];
    const int64_t *nd_synth_ctx_ptr = NULL;
    const bool prefer_nd_synth_fast =
        program->jit_synth_reserved_nd &&
        program->jit_kernel_fn &&
        !me_eval_jit_disabled(params);

    if (valid_items == padded_items) {
        bool nd_ctx_ready = false;
        bool use_nd_synth_fast = false;
        if (program->jit_synth_reserved_nd && prefer_nd_synth_fast) {
            nd_ctx_ready = dsl_build_nd_synth_ctx(nd, shape, shape_stride, base_idx, blockshape,
                                                  nd_synth_ctx,
                                                  sizeof(nd_synth_ctx) / sizeof(nd_synth_ctx[0]));
            use_nd_synth_fast = nd_ctx_ready;
        }
        bool need_reserved_indices =
            ((program->uses_i_mask != 0) || program->uses_flat_idx) &&
            !use_nd_synth_fast;
        if (need_reserved_indices) {
            for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
                if (program->uses_i_mask & (1 << d)) {
                    idx_buffers[d] = malloc((size_t)valid_items * sizeof(int64_t));
                    if (!idx_buffers[d]) {
                        for (int j = 0; j < ME_DSL_MAX_NDIM; j++) {
                            free(idx_buffers[j]);
                        }
                        return ME_EVAL_ERR_OOM;
                    }
                }
            }
            if (program->uses_flat_idx) {
                global_linear_idx_buffer = malloc((size_t)valid_items * sizeof(int64_t));
                if (!global_linear_idx_buffer) {
                    for (int j = 0; j < ME_DSL_MAX_NDIM; j++) {
                        free(idx_buffers[j]);
                    }
                    return ME_EVAL_ERR_OOM;
                }
            }
            for (int d = nd; d < ME_DSL_MAX_NDIM; d++) {
                if (idx_buffers[d]) {
                    dsl_fill_i64(idx_buffers[d], (int)valid_items, 0);
                }
            }
            int64_t indices[64] = {0};
            int64_t total_iters = padded_items;
            for (int64_t it = 0; it < total_iters; it++) {
                for (int d = 0; d < ME_DSL_MAX_NDIM && d < nd; d++) {
                    if (idx_buffers[d]) {
                        idx_buffers[d][it] = base_idx[d] + indices[d];
                    }
                }
                if (global_linear_idx_buffer) {
                    int64_t global_idx = 0;
                    for (int d = 0; d < nd; d++) {
                        int64_t coord = dsl_i64_add_wrap(base_idx[d], indices[d]);
                        global_idx = dsl_i64_addmul_wrap(global_idx, coord, shape_stride[d]);
                    }
                    global_linear_idx_buffer[it] = global_idx;
                }
                for (int i = nd - 1; i >= 0; i--) {
                    indices[i]++;
                    if (indices[i] < blockshape[i]) {
                        break;
                    }
                    indices[i] = 0;
                }
            }
        }

        if (program->jit_synth_reserved_nd) {
            if (!nd_ctx_ready) {
                nd_ctx_ready = dsl_build_nd_synth_ctx(nd, shape, shape_stride, base_idx, blockshape,
                                                      nd_synth_ctx,
                                                      sizeof(nd_synth_ctx) / sizeof(nd_synth_ctx[0]));
            }
            if (nd_ctx_ready) {
                nd_synth_ctx_ptr = nd_synth_ctx;
            }
        }

        rc = dsl_eval_program(program, vars_block, n_vars, output_block,
                              (int)valid_items, params, nd, shape, idx_buffers,
                              global_linear_idx_buffer, nd_synth_ctx_ptr);
        for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
            free(idx_buffers[d]);
        }
        free(global_linear_idx_buffer);
        return rc;
    }

    int64_t valid_len[64];
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

    size_t var_sizes[ME_MAX_VARS];
    for (int v = 0; v < n_vars; v++) {
        var_sizes[v] = dtype_size(program->vars.dtypes[v]);
        if (var_sizes[v] == 0) {
            return ME_EVAL_ERR_INVALID_ARG;
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
    if (!program->output_is_scalar) {
        packed_out = malloc((size_t)valid_items * item_size);
        if (!packed_out) {
            for (int v = 0; v < n_vars; v++) {
                free(packed_vars[v]);
            }
            return ME_EVAL_ERR_OOM;
        }
    }

    bool nd_ctx_ready = false;
    bool use_nd_synth_fast = false;
    if (program->jit_synth_reserved_nd && prefer_nd_synth_fast) {
        nd_ctx_ready = dsl_build_nd_synth_ctx(nd, shape, shape_stride, base_idx, valid_len,
                                              nd_synth_ctx,
                                              sizeof(nd_synth_ctx) / sizeof(nd_synth_ctx[0]));
        use_nd_synth_fast = nd_ctx_ready;
    }

    if (!use_nd_synth_fast) {
        for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
            if (program->uses_i_mask & (1 << d)) {
                idx_buffers[d] = malloc((size_t)valid_items * sizeof(int64_t));
                if (!idx_buffers[d]) {
                    for (int v = 0; v < n_vars; v++) {
                        free(packed_vars[v]);
                    }
                    free(packed_out);
                    for (int j = 0; j < ME_DSL_MAX_NDIM; j++) {
                        free(idx_buffers[j]);
                    }
                    return ME_EVAL_ERR_OOM;
                }
            }
        }
        if (program->uses_flat_idx) {
            global_linear_idx_buffer = malloc((size_t)valid_items * sizeof(int64_t));
            if (!global_linear_idx_buffer) {
                for (int v = 0; v < n_vars; v++) {
                    free(packed_vars[v]);
                }
                free(packed_out);
                for (int j = 0; j < ME_DSL_MAX_NDIM; j++) {
                    free(idx_buffers[j]);
                }
                return ME_EVAL_ERR_OOM;
            }
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
        for (int d = 0; d < ME_DSL_MAX_NDIM && d < nd; d++) {
            if (idx_buffers[d]) {
                idx_buffers[d][write_idx] = base_idx[d] + indices[d];
            }
        }
        if (global_linear_idx_buffer) {
            int64_t global_idx = 0;
            for (int d = 0; d < nd; d++) {
                int64_t coord = dsl_i64_add_wrap(base_idx[d], indices[d]);
                global_idx = dsl_i64_addmul_wrap(global_idx, coord, shape_stride[d]);
            }
            global_linear_idx_buffer[write_idx] = global_idx;
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

    void *dsl_out = program->output_is_scalar ? malloc((size_t)valid_items * item_size) : packed_out;
    if (!dsl_out) {
        for (int v = 0; v < n_vars; v++) {
            free(packed_vars[v]);
        }
        free(packed_out);
        for (int j = 0; j < ME_DSL_MAX_NDIM; j++) {
            free(idx_buffers[j]);
        }
        free(global_linear_idx_buffer);
        return ME_EVAL_ERR_OOM;
    }

    if (program->jit_synth_reserved_nd) {
        if (!nd_ctx_ready) {
            nd_ctx_ready = dsl_build_nd_synth_ctx(nd, shape, shape_stride, base_idx, valid_len,
                                                  nd_synth_ctx,
                                                  sizeof(nd_synth_ctx) / sizeof(nd_synth_ctx[0]));
        }
        if (nd_ctx_ready) {
            nd_synth_ctx_ptr = nd_synth_ctx;
        }
    }

    rc = dsl_eval_program(program, (const void **)packed_vars, n_vars, dsl_out,
                          (int)valid_items, params, nd, shape, idx_buffers,
                          global_linear_idx_buffer, nd_synth_ctx_ptr);
    if (rc != ME_EVAL_SUCCESS) {
        for (int v = 0; v < n_vars; v++) {
            free(packed_vars[v]);
        }
        if (program->output_is_scalar) {
            free(dsl_out);
        }
        free(packed_out);
        for (int j = 0; j < ME_DSL_MAX_NDIM; j++) {
            free(idx_buffers[j]);
        }
        free(global_linear_idx_buffer);
        return rc;
    }

    if (program->output_is_scalar) {
        memcpy(output_block, dsl_out, item_size);
        free(dsl_out);
    }
    else {
        memset(output_block, 0, (size_t)padded_items * item_size);
        memset(indices, 0, sizeof(indices));
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
    if (packed_out) {
        free(packed_out);
    }
    for (int j = 0; j < ME_DSL_MAX_NDIM; j++) {
        free(idx_buffers[j]);
    }
    free(global_linear_idx_buffer);

    return ME_EVAL_SUCCESS;
}
