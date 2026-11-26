/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_KERNEL_INTERNAL_H
#define MINIEXPR_KERNEL_INTERNAL_H
#include "miniexpr_kernel.h"
#include "dsl_compile_internal.h"

/* Private transition bridge for the existing array/graph scheduler. */
struct me_kernel {
    me_expr *formula;
    me_dsl_compiled_program *program;
    size_t ninputs;
    size_t nbindings;
    me_dtype dtypes[ME_MAX_VARS];
    size_t widths[ME_MAX_VARS];
    void *captures[ME_MAX_VARS];
    char *entry_point;
    me_math_dispatch formula_math;
    me_kernel_info info;
};

/* Private scheduler entry: preserves legacy AUTO policy without parsing twice. */
me_status me_kernel_prepare_menudet(const char *version, const char *source,
    const me_signature *signature, const me_prepare_options *options, int jit_mode,
    me_kernel **out, me_error *error);

unsigned dsl_menudet_block_features(const me_dsl_compiled_block *block);

static inline size_t me_kernel_alignment(me_dtype dtype) {
    if (dtype == ME_BYTES) return 1;
    if (dtype == ME_STRING) return 4;
    if (dtype == ME_COMPLEX64) return sizeof(float);
    if (dtype == ME_COMPLEX128) return sizeof(double);
    return dtype_size(dtype);
}

static inline bool me_kernel_range(const void *data, size_t bytes, size_t alignment) {
    uintptr_t address = (uintptr_t)data;
    return !bytes || (data && address % alignment == 0 && bytes <= UINTPTR_MAX - address);
}

static inline bool me_kernel_overlap(const void *a, size_t na, const void *b, size_t nb) {
    if (!na || !nb) return false;
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < na : x - y < nb;
}

/* Validate options and initialize reporting before any grid/block preflight. */
me_status me_kernel_run_begin(const me_run_options *options, me_fp_status *status,
    me_error *error);
#endif
