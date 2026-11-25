/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_GRID_H
#define MINIEXPR_GRID_H
#include "miniexpr_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ME_GRID_MAX_NDIM 8
typedef struct me_grid me_grid;

typedef struct {
    size_t struct_size;
    int ndim;
    const int64_t *shape;
    const int32_t *chunkshape;
    const int32_t *blockshape;
} me_grid_spec;

#define ME_GRID_SPEC_INITIALIZER { sizeof(me_grid_spec), 0, NULL, NULL, NULL }

typedef struct {
    size_t struct_size;
    int ndim;
    size_t padded_nitems;
    size_t valid_nitems;
    int64_t origin[ME_GRID_MAX_NDIM];
    int64_t extent[ME_GRID_MAX_NDIM];
} me_grid_block_info;

#define ME_GRID_BLOCK_INFO_INITIALIZER { sizeof(me_grid_block_info), 0, 0, 0, {0}, {0} }

/* Provisional storage-neutral padded-grid adapter. Copies C-order geometry.
 * Rank zero denotes one scalar block. Nonzero rank requires positive physical
 * chunk/block extents, even for an empty logical domain. */
me_status me_grid_create(const me_grid_spec *spec, me_grid **out, me_error *error);
me_status me_grid_describe_block(const me_grid *grid, int64_t nchunk, int64_t nblock,
    me_grid_block_info *out, me_error *error);

/* Buffers start at one physical block, not at an array or chunk. Elementwise
 * output retains physical spacing and zero padding. Block-scalar output needs
 * one item and reduces ALL live rows in ONE invocation, never row by row.
 * Partially padded inputs are gathered into call-local compact scratch.
 * Fully padded scalar groups retain the kernel's empty-reduction behavior.
 * Output must be disjoint from the entire physical input spans. */
me_status me_grid_eval_block(const me_grid *grid, const me_kernel *kernel,
    int64_t nchunk, int64_t nblock, const me_buffer *inputs, size_t ninputs,
    me_mut_buffer output, const me_run_options *options, me_error *error);
void me_grid_free(me_grid *grid);

#ifdef __cplusplus
}
#endif
#endif
