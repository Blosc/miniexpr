/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "miniexpr_grid.h"
#include "miniexpr_grid_internal.h"
#include "miniexpr_kernel_internal.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct me_grid {
    int ndim;
    int64_t shape[ME_GRID_MAX_NDIM];
    int64_t chunks[ME_GRID_MAX_NDIM];
    int64_t blocks[ME_GRID_MAX_NDIM];
    int64_t chunk_grid[ME_GRID_MAX_NDIM];
    int64_t block_grid[ME_GRID_MAX_NDIM];
    size_t strides[ME_GRID_MAX_NDIM];
    size_t padded;
    int64_t nchunks;
    int64_t nblocks;
};

static me_status grid_error(me_error *error, me_status status, const char *message) {
    if (error) {
        memset(error, 0, sizeof(*error));
        error->status = status;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}

void me_grid_free(me_grid *grid) {
    free(grid);
}

me_status me_grid_create(const me_grid_spec *spec, me_grid **out, me_error *error) {
    if (out) *out = NULL;
    if (error) memset(error, 0, sizeof(*error));
    if (!out || !spec || spec->struct_size < ME_GRID_SPEC_V1_SIZE || spec->ndim < 0 ||
        spec->ndim > ME_GRID_MAX_NDIM ||
        (spec->ndim && (!spec->shape || !spec->chunkshape || !spec->blockshape))) {
        return grid_error(error, ME_ERR_INVALID_ARG, "invalid grid descriptor");
    }
    me_grid *grid = calloc(1, sizeof(*grid));
    if (!grid) return grid_error(error, ME_ERR_OOM, "out of memory creating grid");
    grid->ndim = spec->ndim;
    grid->padded = 1;
    grid->nchunks = grid->nblocks = 1;
    int64_t logical_items = 1;
    for (int d = grid->ndim - 1; d >= 0; d--) {
        int64_t shape = spec->shape[d], chunk = spec->chunkshape[d], block = spec->blockshape[d];
        if (shape < 0 || chunk <= 0 || block <= 0 ||
            (shape && logical_items > INT64_MAX / shape) ||
            grid->padded > (size_t)INT_MAX / (size_t)block) goto invalid;
        logical_items *= shape;
        int64_t chunks = me_grid_div_ceil(shape, chunk);
        int64_t blocks = me_grid_div_ceil(chunk, block);
        if ((chunks && grid->nchunks > INT64_MAX / chunks) ||
            grid->nblocks > INT64_MAX / blocks) goto invalid;
        grid->shape[d] = shape;
        grid->chunks[d] = chunk;
        grid->blocks[d] = block;
        grid->chunk_grid[d] = chunks;
        grid->block_grid[d] = blocks;
        grid->strides[d] = grid->padded;
        grid->padded *= (size_t)block;
        grid->nchunks *= chunks;
        grid->nblocks *= blocks;
    }
    *out = grid;
    return ME_SUCCESS;
invalid:
    free(grid);
    return grid_error(error, ME_ERR_INVALID_ARG, "invalid or overflowing grid geometry");
}

me_status me_grid_describe_block(const me_grid *grid, int64_t nchunk, int64_t nblock,
    me_grid_block_info *out, me_error *error) {
    if (error) memset(error, 0, sizeof(*error));
    if (!grid || !out || out->struct_size < ME_GRID_BLOCK_INFO_V1_SIZE ||
        nchunk < 0 || nchunk >= grid->nchunks || nblock < 0 || nblock >= grid->nblocks) {
        return grid_error(error, ME_ERR_INVALID_ARG, "invalid grid block index or descriptor");
    }
    size_t size = out->struct_size;
    memset(out, 0, size < sizeof(*out) ? size : sizeof(*out));
    out->struct_size = size;
    out->ndim = grid->ndim;
    out->padded_nitems = grid->padded;
    out->valid_nitems = 1;
    for (int d = grid->ndim - 1; d >= 0; d--) {
        int64_t chunk = nchunk % grid->chunk_grid[d];
        int64_t block = nblock % grid->block_grid[d];
        nchunk /= grid->chunk_grid[d];
        nblock /= grid->block_grid[d];
        int64_t valid = me_grid_axis_window(grid->shape[d], grid->chunks[d], grid->blocks[d],
            chunk, block, &out->origin[d]);
        out->extent[d] = valid;
        out->valid_nitems *= (size_t)valid;
    }
    return ME_SUCCESS;
}

static bool grid_reserve_scratch(size_t *total, size_t bytes, size_t alignment, size_t *offset) {
    size_t padding = (alignment - *total % alignment) % alignment;
    if (padding > SIZE_MAX - *total || bytes > SIZE_MAX - *total - padding) return false;
    *offset = *total + padding;
    *total = *offset + bytes;
    return true;
}

me_status me_grid_eval_block(const me_grid *grid, const me_kernel *kernel,
    int64_t nchunk, int64_t nblock, const me_buffer *inputs, size_t ninputs,
    me_mut_buffer output, const me_run_options *options, me_error *error) {
    me_fp_status fp;
    me_status status = me_kernel_run_begin(options, &fp, error);
    if (status != ME_SUCCESS) return status;
    me_grid_block_info info = ME_GRID_BLOCK_INFO_INITIALIZER;
    status = me_grid_describe_block(grid, nchunk, nblock, &info, error);
    if (status != ME_SUCCESS) return status;
    if (!kernel || ninputs != kernel->ninputs || (ninputs && !inputs) ||
        (kernel->info.context_ndim && kernel->info.context_ndim != grid->ndim)) {
        return grid_error(error, ME_ERR_INVALID_ARG, "grid and kernel signature disagree");
    }
    bool scalar = kernel->info.cardinality == ME_BLOCK_SCALAR;
    size_t width = kernel->info.output_itemsize;
    size_t count = scalar ? 1 : info.padded_nitems;
    if (count > SIZE_MAX / width || output.capacity < count * width ||
        !me_kernel_range(output.data, count * width, me_kernel_alignment(kernel->info.output_dtype))) {
        return grid_error(error, ME_ERR_INVALID_ARG, "invalid physical output buffer");
    }
    size_t output_bytes = count * width;
    for (size_t i = 0; i < ninputs; i++) {
        size_t input_width = kernel->widths[i];
        if (info.padded_nitems > SIZE_MAX / input_width ||
            inputs[i].capacity < info.padded_nitems * input_width ||
            !me_kernel_range(inputs[i].data, info.padded_nitems * input_width, me_kernel_alignment(kernel->dtypes[i])) ||
            me_kernel_overlap(inputs[i].data, info.padded_nitems * input_width, output.data, output_bytes)) {
            return grid_error(error, ME_ERR_INVALID_ARG, "invalid or overlapping physical input buffer");
        }
    }
    /* Physical buffers and run options have still been validated. Empty scalar
     * groups must execute, but elementwise groups have no numerical work. */
    if (!scalar && !info.valid_nitems) {
        memset(output.data, 0, output_bytes);
        return ME_SUCCESS;
    }
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = info.valid_nitems;
    if (kernel->info.context_ndim) {
        block.ndim = grid->ndim;
        block.logical_shape = grid->shape;
        block.origin = info.origin;
        block.extent = info.extent;
    }
    if (info.valid_nitems == info.padded_nitems) {
        return me_eval_block(kernel, inputs, ninputs, output, &block, options, error);
    }
    me_buffer packed[ME_MAX_VARS] = {{0}};
    size_t offsets[ME_MAX_VARS];
    size_t result_count = scalar ? 1 : info.valid_nitems;
    size_t result_bytes = result_count * width; /* Physical capacity already checked. */
    size_t total = result_bytes;
    for (size_t i = 0; i < ninputs; i++) {
        if (!grid_reserve_scratch(&total, info.valid_nitems * kernel->widths[i],
            me_kernel_alignment(kernel->dtypes[i]), &offsets[i])) {
            return grid_error(error, ME_ERR_INVALID_ARG, "padded scratch byte extent overflows");
        }
    }
    /* malloc supplies the maximum native alignment; each region preserves its
     * own dtype alignment even after an odd-width string/byte region. */
    void *scratch = malloc(total);
    if (!scratch) return grid_error(error, ME_ERR_OOM, "out of memory gathering padded block");
    for (size_t i = 0; i < ninputs; i++) {
        size_t input_width = kernel->widths[i];
        size_t bytes = info.valid_nitems * input_width;
        packed[i] = (me_buffer){(unsigned char *)scratch + offsets[i], bytes};
        me_grid_copy_runs(grid->ndim, grid->blocks, info.extent, grid->strides,
            info.valid_nitems, (void *)packed[i].data, (void *)inputs[i].data, input_width, true);
    }
    status = me_eval_block(kernel, packed, ninputs, (me_mut_buffer){scratch, result_bytes}, &block, options, error);
    if (status == ME_SUCCESS) {
        if (scalar) memcpy(output.data, scratch, width);
        else {
            memset(output.data, 0, output_bytes);
            me_grid_copy_runs(grid->ndim, grid->blocks, info.extent, grid->strides,
                info.valid_nitems, scratch, output.data, width, false);
        }
    }
    free(scratch);
    return status;
}
