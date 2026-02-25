/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_JIT_CGEN_H
#define MINIEXPR_DSL_JIT_CGEN_H

#include <stdbool.h>

#include "dsl_jit_ir.h"
#include "dsl_parser.h"
#include "miniexpr.h"

typedef struct {
    const char *symbol_name;
    bool use_runtime_math_bridge;
    bool has_enable_vector_math;
    bool enable_vector_math;
    char *trace_lowering_mode;
    size_t trace_lowering_mode_cap;
    char *trace_vector_ops;
    size_t trace_vector_ops_cap;
    char *trace_lowering_reason;
    size_t trace_lowering_reason_cap;
    bool synth_reserved_non_nd;
    bool synth_reserved_nd;
    const char *synth_nd_ctx_name;
    int synth_nd_compile_ndims;
} me_dsl_jit_cgen_options;

bool me_dsl_jit_codegen_c(const me_dsl_jit_ir_program *program, me_dtype output_dtype,
                          const me_dsl_jit_cgen_options *options,
                          char **out_source, me_dsl_error *error);

#endif
