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
    bool synth_reserved_non_nd;
} me_dsl_jit_cgen_options;

bool me_dsl_jit_codegen_c(const me_dsl_jit_ir_program *program, me_dtype output_dtype,
                          const me_dsl_jit_cgen_options *options,
                          char **out_source, me_dsl_error *error);

#endif
