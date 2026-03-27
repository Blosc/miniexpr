/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_INTERNAL_H
#define MINIEXPR_INTERNAL_H

#include "functions.h"
#include <stdbool.h>
#include <complex.h>
#include <stdint.h>

/* ND metadata attached to compiled expressions (used by me_eval_nd). */
typedef struct {
    int ndims;
    /* Layout: shape[ndims], chunkshape[ndims], blockshape[ndims] (all int64_t). */
    int64_t data[1];
} me_nd_info;

typedef union {
    bool b;
    int64_t i64;
    uint64_t u64;
    float f32;
    double f64;
    float _Complex c64;
    double _Complex c128;
} me_scalar;

extern char synthetic_var_addresses[ME_MAX_VARS];

int64_t dsl_i64_add_wrap(int64_t a, int64_t b);
int64_t dsl_i64_addmul_wrap(int64_t acc, int64_t a, int64_t b);

bool contains_reduction(const me_expr *n);
bool output_is_scalar(const me_expr *n);
bool reduce_strided_variable(const me_expr *expr, const void **vars_block, int n_vars,
                             const int64_t *valid_len, const int64_t *stride, int nd,
                             int64_t valid_items, void *output_block);
bool reduce_strided_predicate(const me_expr *expr, const void **vars_block, int n_vars,
                              const int64_t *valid_len, const int64_t *stride, int nd,
                              int64_t valid_items, void *output_block);
void write_scalar(void *out, me_dtype out_type, me_dtype in_type, const me_scalar *v);
void read_scalar(const void *in, me_dtype in_type, me_scalar *v);

int me_nd_compute_valid_items(const me_expr *expr, int64_t nchunk, int64_t nblock,
                              int chunk_nitems, int64_t *valid_items, int64_t *padded_items);
int me_eval_nd_classic(const me_expr *expr, const void **vars_block,
                       int n_vars, void *output_block, int block_nitems,
                       int64_t nchunk, int64_t nblock, const me_eval_params *params);

#endif
