/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_MENUDET_EXPR_H
#define MINIEXPR_DSL_MENUDET_EXPR_H

#include "functions.h"

/* Internal typed 1.0 path shared by source validation and artifact import.
 * Reuses the native parser's me_expr tree; no source execution or callbacks. */
bool dsl_menudet_type_expr(me_expr **expr, me_dtype output_dtype,
                             char *reason, size_t reason_cap);
bool dsl_menudet_type_expr_profile(me_expr **expr, me_dtype output_dtype,
                                    me_dsl_semantic_profile profile, char *reason, size_t reason_cap);
bool dsl_menudet_convert_expr(me_expr **expr, me_dtype dtype);
bool dsl_menudet_convert_range_expr(me_expr **expr);
double dsl_menudet_jit_unary_math(const void *node, double x);
double dsl_menudet_jit_binary_math(const void *node, double x, double y);
/* Resolve exact interpreter/libm equivalents, with float or double arguments and
 * result according to node->dtype (ldexp's exponent is C int). NULL retains generic semantics. Addresses use
 * the private JIT input ABI's function-pointer transport, never serialization. */
const void *dsl_menudet_resolve_unary_math(const me_expr *node);
const void *dsl_menudet_resolve_binary_math(const me_expr *node);
bool dsl_menudet_integer_power(const me_expr *node, int64_t *exponent);
const void *dsl_menudet_resolve_ternary_math(const me_expr *node);
const void *dsl_menudet_resolve_predicate(const me_expr *node);
bool dsl_menudet_jit_predicate(const void *node, double x);
uint64_t dsl_menudet_jit_int_op(const void *node, uint64_t a, uint64_t b, int *status);
uint64_t dsl_menudet_jit_checked(const void *node, uint64_t a, uint64_t b, uint64_t c, int *status);
me_dtype dsl_menudet_cast_dtype(const me_expr *expr);
int dsl_menudet_eval_expr(const me_expr *expr, const void *const *vars, int nvars,
                           const uint8_t *const *initialized,
                           int item, int nitems, const uint8_t *mask, void *out);
int dsl_menudet_eval_expr_masked(const me_expr *expr, const void *const *vars, int nvars,
                                  const uint8_t *const *initialized,
                                  int nitems, const uint8_t *mask, size_t stride, void *out);

#endif
