/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_PARSER_H
#define MINIEXPR_DSL_PARSER_H
#include "dsl_semantic_profile.h"

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    ME_DSL_STMT_ASSIGN = 0,
    ME_DSL_STMT_EXPR,
    ME_DSL_STMT_RETURN,
    ME_DSL_STMT_PRINT,
    ME_DSL_STMT_IF,
    ME_DSL_STMT_WHILE,
    ME_DSL_STMT_FOR,
    ME_DSL_STMT_BREAK,
    ME_DSL_STMT_CONTINUE
} me_dsl_stmt_kind;

typedef enum {
    ME_DSL_FP_STRICT = 0,
    ME_DSL_FP_CONTRACT = 1,
    ME_DSL_FP_FAST = 2
} me_dsl_fp_mode;

typedef enum {
    ME_DSL_COMPILER_LIBTCC = 0,
    ME_DSL_COMPILER_CC = 1
} me_dsl_compiler;

/* Explicit or environment-selected math library. AUTO is the source spelling
 * for the default system policy; the resolver retains it so an explicit
 * `me:math=auto` never consults the environment. */
typedef enum {
    ME_DSL_MATH_AUTO = 0,
    ME_DSL_MATH_SYSTEM = 1,
    ME_DSL_MATH_SLEEF = 2,
    ME_DSL_MATH_ACCELERATE = 3,
    ME_DSL_MATH_SCALAR = 4
} me_dsl_math_backend;

typedef enum {
    ME_DSL_ULP_OMIT = 0,
    ME_DSL_ULP_U10 = 1,
    ME_DSL_ULP_U35 = 2
} me_dsl_sleef_ulp;

/* Artifact-declared policy. Presence is independent of the parsed enum value,
 * so an omitted field can inherit the environment instead of silently
 * becoming a built-in default. */
typedef struct {
    bool has_fp;
    me_dsl_fp_mode fp;
    bool has_math;
    me_dsl_math_backend math;
    bool has_ulp;
    me_dsl_sleef_ulp ulp;
    bool integer_powers;
} me_dsl_declared_policy;

typedef struct me_dsl_expr {
    char *text;
    int line;
    int column;
} me_dsl_expr;

typedef struct me_dsl_stmt me_dsl_stmt;

typedef struct {
    me_dsl_stmt **stmts;
    int nstmts;
    int capacity;
} me_dsl_block;

typedef struct {
    me_dsl_expr *cond;
    me_dsl_block block;
} me_dsl_if_branch;

struct me_dsl_stmt {
    me_dsl_stmt_kind kind;
    int line;
    int column;
    union {
        struct {
            char *name;
            me_dsl_expr *value;
            bool synthetic;  /* Internal operand capture: infer type and obey lane masks. */
        } assign;
        struct {
            me_dsl_expr *expr;
        } expr_stmt;
        struct {
            me_dsl_expr *expr;
        } return_stmt;
        struct {
            me_dsl_expr *call;
        } print_stmt;
        struct {
            me_dsl_expr *cond;
            me_dsl_block then_block;
            me_dsl_if_branch *elif_branches;
            int n_elifs;
            int elif_capacity;
            me_dsl_block else_block;
            int has_else;
        } if_stmt;
        struct {
            me_dsl_expr *cond;
            me_dsl_block body;
            int condition_nstmts;  /* Lowered condition prefix, excluded from the body iteration cap. */
        } while_loop;
        struct {
            char *var;
            me_dsl_expr *limit;
            me_dsl_block body;
        } for_loop;
        struct {
            me_dsl_expr *cond;
        } flow;
    } as;
};

typedef struct {
    char *name;
    char **params;
    int nparams;
    int param_capacity;
    me_dsl_fp_mode fp_mode;
    me_dsl_compiler compiler;
    me_dsl_math_backend math;
    me_dsl_sleef_ulp ulp;
    bool fp_explicit;
    bool compiler_explicit;
    bool math_explicit;
    bool ulp_explicit;
    me_dsl_block block;
} me_dsl_program;

typedef struct {
    int line;
    int column;
    char message[128];
} me_dsl_error;

me_dsl_program *me_dsl_parse(const char *source, me_dsl_error *error);
me_dsl_program *me_dsl_parse_profile(const char *source, me_dsl_semantic_profile profile,
                                     me_dsl_error *error);
void me_dsl_program_free(me_dsl_program *program);

#endif
