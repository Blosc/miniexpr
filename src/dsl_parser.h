/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_PARSER_H
#define MINIEXPR_DSL_PARSER_H

#include <stddef.h>

typedef enum {
    ME_DSL_STMT_ASSIGN = 0,
    ME_DSL_STMT_EXPR,
    ME_DSL_STMT_RETURN,
    ME_DSL_STMT_PRINT,
    ME_DSL_STMT_IF,
    ME_DSL_STMT_FOR,
    ME_DSL_STMT_BREAK,
    ME_DSL_STMT_CONTINUE
} me_dsl_stmt_kind;

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
    me_dsl_block block;
} me_dsl_program;

typedef struct {
    int line;
    int column;
    char message[128];
} me_dsl_error;

me_dsl_program *me_dsl_parse(const char *source, me_dsl_error *error);
void me_dsl_program_free(me_dsl_program *program);

#endif
