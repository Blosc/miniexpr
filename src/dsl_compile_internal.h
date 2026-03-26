/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_COMPILE_INTERNAL_H
#define MINIEXPR_DSL_COMPILE_INTERNAL_H

#include "dsl_jit_runtime_internal.h"
#include "dsl_parser.h"

#include <stdbool.h>
#include <stddef.h>

void dsl_var_table_init(me_dsl_var_table *table);
void dsl_var_table_free(me_dsl_var_table *table);
int dsl_var_table_find(const me_dsl_var_table *table, const char *name);
int dsl_var_table_add_with_uniform(me_dsl_var_table *table, const char *name, me_dtype dtype,
                                   size_t itemsize, bool uniform);
int dsl_var_table_add(me_dsl_var_table *table, const char *name, me_dtype dtype);

extern char synthetic_var_addresses[ME_MAX_VARS];

int private_compile_ex(const char *expression, const me_variable *variables, int var_count,
                       void *output, int nitems, me_dtype dtype, int *error, me_expr **out);
bool contains_reduction(const me_expr *n);
bool output_is_scalar(const me_expr *n);
me_dsl_compiled_program *dsl_compile_program(const char *source,
                                             const me_variable *variables,
                                             int var_count,
                                             me_dtype dtype,
                                             int compile_ndims,
                                             int jit_mode,
                                             int *error_pos,
                                             bool *is_dsl,
                                             char *error_reason,
                                             size_t error_reason_cap);
me_dsl_compiled_program *dsl_compiled_program_alloc(const me_dsl_program *parsed,
                                                    const char *source,
                                                    int compile_ndims,
                                                    char *error_reason,
                                                    size_t error_reason_cap);
void dsl_compiled_expr_free(me_dsl_compiled_expr *expr);
void dsl_compiled_stmt_free(me_dsl_compiled_stmt *stmt);
void dsl_compiled_program_free(me_dsl_compiled_program *program);
bool dsl_compiled_block_push(me_dsl_compiled_block *block, me_dsl_compiled_stmt *stmt);
bool dsl_build_var_lookup(const me_dsl_var_table *table, const me_variable *funcs,
                          int n_funcs, me_variable **out_vars, int *out_count);
bool dsl_program_add_local(me_dsl_compiled_program *program, int var_index);

#endif
