/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_EVAL_INTERNAL_H
#define MINIEXPR_DSL_EVAL_INTERNAL_H

#include "dsl_compile_internal.h"

#include <complex.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef ME_DSL_JIT_SYNTH_ND_CTX_V2_VERSION
#define ME_DSL_JIT_SYNTH_ND_CTX_V2_VERSION 2
#endif

#ifndef ME_DSL_ND_CTX_FLAG_SEQ
#define ME_DSL_ND_CTX_FLAG_SEQ 1
#endif

#ifndef ME_DSL_JIT_SYNTH_ND_CTX_BASE_WORDS
#define ME_DSL_JIT_SYNTH_ND_CTX_BASE_WORDS (1 + 4 * ME_DSL_MAX_NDIM)
#endif

#ifndef ME_DSL_JIT_SYNTH_ND_CTX_WORDS
#define ME_DSL_JIT_SYNTH_ND_CTX_WORDS (ME_DSL_JIT_SYNTH_ND_CTX_BASE_WORDS + 3)
#endif

bool me_eval_jit_disabled(const me_eval_params *params);

bool dsl_any_nonzero(const void *data, me_dtype dtype, int nitems);
void dsl_fill_i64(int64_t *out, int nitems, int64_t value);
void dsl_fill_iota_i64(int64_t *out, int nitems, int64_t start);
bool dsl_read_int64(const void *data, me_dtype dtype, int64_t *out);

float me_crealf(float _Complex v);
float me_cimagf(float _Complex v);
double me_creal(double _Complex v);
double me_cimag(double _Complex v);

int64_t dsl_i64_add_wrap(int64_t a, int64_t b);
int64_t dsl_i64_addmul_wrap(int64_t acc, int64_t a, int64_t b);

int dsl_eval_program(const me_dsl_compiled_program *program,
                     const void **vars_block, int n_vars,
                     void *output_block, int nitems,
                     const me_eval_params *params,
                     int ndim, const int64_t *shape,
                     int64_t **idx_buffers,
                     const int64_t *global_linear_idx_buffer,
                     const int64_t *nd_synth_ctx_buffer);

#endif
