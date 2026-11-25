/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_GRID_INTERNAL_H
#define MINIEXPR_GRID_INTERNAL_H

#include "miniexpr.h"
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ME_ND_MAX_DIMS 64

static inline int64_t me_grid_div_ceil(int64_t extent, int64_t step) {
    return step <= 0 ? 0 : extent / step + (extent % step != 0);
}

static inline int64_t me_grid_axis_window(int64_t shape_d, int64_t chunk_d,
    int64_t block_d, int64_t chunk_idx, int64_t block_idx, int64_t *origin_out) {
    int64_t start = chunk_idx * chunk_d;
    int64_t relative = block_idx * block_d;
    int64_t remaining = shape_d > start ? shape_d - start : 0;
    if (origin_out) {
        *origin_out = relative > remaining ? shape_d : start + relative;
    }
    int64_t valid = remaining > relative ? remaining - relative : 0;
    int64_t chunk_valid = chunk_d > relative ? chunk_d - relative : 0;
    if (valid > chunk_valid) valid = chunk_valid;
    if (valid > block_d) valid = block_d;
    return valid;
}

static inline void me_grid_copy_runs(int ndim, const int64_t *blocks,
    const int64_t *extent, const size_t *strides, size_t valid_nitems,
    void *compact, void *physical, size_t width, bool gather) {
    if (ndim <= 0 || !valid_nitems) return;
    int axis = ndim - 1;
    size_t run = (size_t)extent[axis];
    /* Merge a contiguous suffix, then advance outer coordinates as an odometer.
     * No per-lane division/modulo, and one memcpy per contiguous run. */
    while (axis > 0 && extent[axis] == blocks[axis]) {
        run *= (size_t)extent[--axis];
    }
    size_t coordinates[ME_ND_MAX_DIMS] = {0};
    size_t physical_offset = 0;
    for (size_t offset = 0; offset < valid_nitems; offset += run) {
        unsigned char *packed = (unsigned char *)compact + offset * width;
        unsigned char *padded = (unsigned char *)physical + physical_offset * width;
        if (gather) memcpy(packed, padded, run * width);
        else memcpy(padded, packed, run * width);
        for (int d = axis - 1; d >= 0; d--) {
            physical_offset += strides[d];
            if (++coordinates[d] < (size_t)extent[d]) break;
            physical_offset -= coordinates[d] * strides[d];
            coordinates[d] = 0;
        }
    }
}

static inline int me_nd_locate_block(int nd, const int64_t *shape,
    const int64_t *chunkshape, const int64_t *blockshape,
    int64_t nchunk, int64_t nblock, int chunk_nitems,
    int64_t *valid_items, int64_t *padded_items,
    int64_t *origin_out, int64_t *valid_len_out, size_t *stride_out) {
    if (nd <= 0 || nd > ME_ND_MAX_DIMS || !shape || !chunkshape || !blockshape ||
        !valid_items || !padded_items) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    int64_t nchunks_axis[ME_ND_MAX_DIMS];
    int64_t nblocks_axis[ME_ND_MAX_DIMS];
    int64_t total_chunks = 1;
    int64_t total_blocks = 1;
    int64_t padded = 1;
    for (int d = 0; d < nd; d++) {
        if (shape[d] <= 0 || chunkshape[d] <= 0 || blockshape[d] <= 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        int64_t nchunks_d = me_grid_div_ceil(shape[d], chunkshape[d]);
        int64_t nblocks_d = me_grid_div_ceil(chunkshape[d], blockshape[d]);
        if (nchunks_d <= 0 || nblocks_d <= 0 ||
            total_chunks > LLONG_MAX / nchunks_d ||
            total_blocks > LLONG_MAX / nblocks_d ||
            padded > LLONG_MAX / blockshape[d]) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        nchunks_axis[d] = nchunks_d;
        nblocks_axis[d] = nblocks_d;
        total_chunks *= nchunks_d;
        total_blocks *= nblocks_d;
        padded *= blockshape[d];
    }
    if (nchunk < 0 || nchunk >= total_chunks || nblock < 0 || nblock >= total_blocks ||
        (chunk_nitems > 0 && (int64_t)chunk_nitems < padded) ||
        (stride_out && (uint64_t)padded > SIZE_MAX)) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    int64_t valid = 1;
    size_t stride = 1;
    for (int d = nd - 1; d >= 0; d--) {
        int64_t chunk = nchunk % nchunks_axis[d];
        int64_t block = nblock % nblocks_axis[d];
        nchunk /= nchunks_axis[d];
        nblock /= nblocks_axis[d];
        int64_t len = me_grid_axis_window(shape[d], chunkshape[d], blockshape[d],
            chunk, block, origin_out ? &origin_out[d] : NULL);
        if (valid_len_out) valid_len_out[d] = len;
        if (stride_out) {
            stride_out[d] = stride;
            stride *= (size_t)blockshape[d];
        }
        if (len && valid > LLONG_MAX / len) {
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

#endif
