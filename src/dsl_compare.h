/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

/* Internal comparison-chain lowering shared by all native DSL callers. */
#ifndef MINIEXPR_DSL_COMPARE_H
#define MINIEXPR_DSL_COMPARE_H

#include <stdbool.h>
#include "dsl_parser.h"

me_dsl_expr *dsl_expr_new(char *text, int line, int column);
me_dsl_stmt *dsl_stmt_new(me_dsl_stmt_kind kind, int line, int column);
void dsl_stmt_free(me_dsl_stmt *stmt);
void dsl_block_free(me_dsl_block *block);
bool dsl_block_push(me_dsl_block *block, me_dsl_stmt *stmt, me_dsl_error *error);
bool dsl_lower_comparisons(me_dsl_program *program, const char *source, me_dsl_error *error);

#endif
