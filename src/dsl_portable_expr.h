/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_PORTABLE_EXPR_H
#define MINIEXPR_DSL_PORTABLE_EXPR_H

#include "functions.h"

/* Internal typed 1.0 path shared by source validation and artifact import.
 * Reuses the native parser's me_expr tree; no source execution or callbacks. */
bool dsl_portable_type_expr(me_expr **expr, me_dtype output_dtype,
                             char *reason, size_t reason_cap);
bool dsl_portable_type_expr_profile(me_expr **expr, me_dtype output_dtype,
                                    me_dsl_semantic_profile profile, char *reason, size_t reason_cap);
bool dsl_portable_convert_expr(me_expr **expr, me_dtype dtype);
double dsl_portable_jit_unary_math(const void *node, double x);
double dsl_portable_jit_binary_math(const void *node, double x, double y);
bool dsl_portable_jit_predicate(const void *node, double x);
uint64_t dsl_portable_jit_int_op(const void *node, uint64_t a, uint64_t b);
me_dtype dsl_portable_cast_dtype(const me_expr *expr);
int dsl_portable_eval_expr(const me_expr *expr, const void *const *vars, int nvars,
                           const uint8_t *const *initialized,
                           int item, int nitems, const uint8_t *mask, void *out);
int dsl_portable_eval_expr_masked(const me_expr *expr, const void *const *vars, int nvars,
                                  const uint8_t *const *initialized,
                                  int nitems, const uint8_t *mask, size_t stride, void *out);

#endif
