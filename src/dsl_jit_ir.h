/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_JIT_IR_H
#define MINIEXPR_DSL_JIT_IR_H

#include <stdbool.h>
#include <stdint.h>

#include "dsl_parser.h"
#include "miniexpr.h"

typedef enum {
    ME_DSL_JIT_IR_STMT_ASSIGN = 0,
    ME_DSL_JIT_IR_STMT_RETURN,
    ME_DSL_JIT_IR_STMT_IF,
    ME_DSL_JIT_IR_STMT_WHILE,
    ME_DSL_JIT_IR_STMT_FOR,
    ME_DSL_JIT_IR_STMT_BREAK,
    ME_DSL_JIT_IR_STMT_CONTINUE
} me_dsl_jit_ir_stmt_kind;

typedef struct {
    char *text;
    me_dtype dtype;
} me_dsl_jit_ir_expr;

typedef struct me_dsl_jit_ir_stmt me_dsl_jit_ir_stmt;

typedef struct {
    me_dsl_jit_ir_stmt **stmts;
    int nstmts;
    int capacity;
} me_dsl_jit_ir_block;

typedef struct {
    me_dsl_jit_ir_expr cond;
    me_dsl_jit_ir_block block;
} me_dsl_jit_ir_if_branch;

struct me_dsl_jit_ir_stmt {
    me_dsl_jit_ir_stmt_kind kind;
    int line;
    int column;
    union {
        struct {
            char *name;
            me_dtype dtype;
            me_dsl_jit_ir_expr value;
        } assign;
        struct {
            me_dsl_jit_ir_expr expr;
        } return_stmt;
        struct {
            me_dsl_jit_ir_expr cond;
            me_dsl_jit_ir_block then_block;
            me_dsl_jit_ir_if_branch *elif_branches;
            int n_elifs;
            int elif_capacity;
            me_dsl_jit_ir_block else_block;
            bool has_else;
        } if_stmt;
        struct {
            me_dsl_jit_ir_expr cond;
            me_dsl_jit_ir_block body;
        } while_loop;
        struct {
            char *var;
            me_dsl_jit_ir_expr start;
            me_dsl_jit_ir_expr stop;
            me_dsl_jit_ir_expr step;
            me_dsl_jit_ir_block body;
        } for_loop;
    } as;
};

typedef struct {
    char *name;
    char **params;
    me_dtype *param_dtypes;
    int nparams;
    me_dsl_fp_mode fp_mode;
    me_dsl_jit_ir_block block;
} me_dsl_jit_ir_program;

typedef enum {
    ME_DSL_JIT_IR_RESOLVE_AUTO = 0,
    ME_DSL_JIT_IR_RESOLVE_OUTPUT
} me_dsl_jit_ir_resolve_mode;

typedef bool (*me_dsl_jit_ir_dtype_resolver)(void *ctx, const me_dsl_expr *expr,
                                             me_dsl_jit_ir_resolve_mode mode,
                                             me_dtype *out_dtype);

bool me_dsl_jit_ir_build(const me_dsl_program *program, const char **param_names,
                         const me_dtype *param_dtypes, int nparams,
                         me_dsl_jit_ir_dtype_resolver resolve_dtype, void *resolve_ctx,
                         me_dsl_jit_ir_program **out_ir, me_dsl_error *error);

void me_dsl_jit_ir_free(me_dsl_jit_ir_program *program);

uint64_t me_dsl_jit_ir_fingerprint(const me_dsl_jit_ir_program *program);

#endif
