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
    me_kernel_info info;
};
#endif
