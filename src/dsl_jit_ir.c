/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_ir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *name;
    me_dtype dtype;
    bool is_param;
} me_dsl_jit_symbol;

typedef struct {
    me_dsl_jit_symbol *items;
    int count;
    int capacity;
} me_dsl_jit_symbol_table;

typedef struct {
    me_dsl_jit_ir_dtype_resolver resolve_dtype;
    void *resolve_ctx;
    me_dsl_jit_symbol_table symbols;
    me_dsl_error *error;
} me_dsl_jit_build_ctx;

static void dsl_jit_set_error(me_dsl_error *error, int line, int column, const char *msg) {
    if (!error) {
        return;
    }
    error->line = line;
    error->column = column;
    if (!msg) {
        msg = "jit ir build failed";
    }
    snprintf(error->message, sizeof(error->message), "%s", msg);
}

static bool dsl_jit_dtype_supported(me_dtype dtype) {
    switch (dtype) {
    case ME_BOOL:
    case ME_INT8:
    case ME_INT16:
    case ME_INT32:
    case ME_INT64:
    case ME_UINT8:
    case ME_UINT16:
    case ME_UINT32:
    case ME_UINT64:
    case ME_FLOAT32:
    case ME_FLOAT64:
        return true;
    default:
        return false;
    }
}

static char *dsl_jit_strdup(const char *s) {
    if (!s) {
        return NULL;
    }
    size_t len = strlen(s);
    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len + 1);
    return out;
}

static void dsl_jit_ir_expr_free(me_dsl_jit_ir_expr *expr) {
    if (!expr) {
        return;
    }
    free(expr->text);
    expr->text = NULL;
    expr->dtype = ME_AUTO;
}

static bool dsl_jit_ir_expr_init(me_dsl_jit_ir_expr *out, const me_dsl_expr *expr, me_dtype dtype) {
    if (!out || !expr || !expr->text) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->text = dsl_jit_strdup(expr->text);
    if (!out->text) {
        return false;
    }
    out->dtype = dtype;
    return true;
}

static void dsl_jit_ir_block_free(me_dsl_jit_ir_block *block);

static void dsl_jit_ir_stmt_free(me_dsl_jit_ir_stmt *stmt) {
    if (!stmt) {
        return;
    }
    switch (stmt->kind) {
    case ME_DSL_JIT_IR_STMT_ASSIGN:
        free(stmt->as.assign.name);
        dsl_jit_ir_expr_free(&stmt->as.assign.value);
        break;
    case ME_DSL_JIT_IR_STMT_RETURN:
        dsl_jit_ir_expr_free(&stmt->as.return_stmt.expr);
        break;
    case ME_DSL_JIT_IR_STMT_IF:
        dsl_jit_ir_expr_free(&stmt->as.if_stmt.cond);
        dsl_jit_ir_block_free(&stmt->as.if_stmt.then_block);
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            dsl_jit_ir_expr_free(&stmt->as.if_stmt.elif_branches[i].cond);
            dsl_jit_ir_block_free(&stmt->as.if_stmt.elif_branches[i].block);
        }
        free(stmt->as.if_stmt.elif_branches);
        if (stmt->as.if_stmt.has_else) {
            dsl_jit_ir_block_free(&stmt->as.if_stmt.else_block);
        }
        break;
    case ME_DSL_JIT_IR_STMT_FOR:
        free(stmt->as.for_loop.var);
        dsl_jit_ir_expr_free(&stmt->as.for_loop.limit);
        dsl_jit_ir_block_free(&stmt->as.for_loop.body);
        break;
    case ME_DSL_JIT_IR_STMT_BREAK:
    case ME_DSL_JIT_IR_STMT_CONTINUE:
        break;
    }
    free(stmt);
}

static void dsl_jit_ir_block_free(me_dsl_jit_ir_block *block) {
    if (!block) {
        return;
    }
    for (int i = 0; i < block->nstmts; i++) {
        dsl_jit_ir_stmt_free(block->stmts[i]);
    }
    free(block->stmts);
    block->stmts = NULL;
    block->nstmts = 0;
    block->capacity = 0;
}

void me_dsl_jit_ir_free(me_dsl_jit_ir_program *program) {
    if (!program) {
        return;
    }
    free(program->name);
    if (program->params) {
        for (int i = 0; i < program->nparams; i++) {
            free(program->params[i]);
        }
    }
    free(program->params);
    free(program->param_dtypes);
    dsl_jit_ir_block_free(&program->block);
    free(program);
}

static bool dsl_jit_ir_block_push(me_dsl_jit_ir_block *block, me_dsl_jit_ir_stmt *stmt) {
    if (!block || !stmt) {
        return false;
    }
    if (block->nstmts == block->capacity) {
        int new_cap = block->capacity ? block->capacity * 2 : 8;
        me_dsl_jit_ir_stmt **next = realloc(block->stmts, (size_t)new_cap * sizeof(*next));
        if (!next) {
            return false;
        }
        block->stmts = next;
        block->capacity = new_cap;
    }
    block->stmts[block->nstmts++] = stmt;
    return true;
}

static void dsl_jit_symbols_free(me_dsl_jit_symbol_table *table) {
    if (!table) {
        return;
    }
    for (int i = 0; i < table->count; i++) {
        free(table->items[i].name);
    }
    free(table->items);
    table->items = NULL;
    table->count = 0;
    table->capacity = 0;
}

static int dsl_jit_symbols_find(const me_dsl_jit_symbol_table *table, const char *name) {
    if (!table || !name) {
        return -1;
    }
    for (int i = 0; i < table->count; i++) {
        if (strcmp(table->items[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static bool dsl_jit_symbols_add(me_dsl_jit_symbol_table *table, const char *name,
                                me_dtype dtype, bool is_param) {
    if (!table || !name) {
        return false;
    }
    if (table->count == table->capacity) {
        int new_cap = table->capacity ? table->capacity * 2 : 8;
        me_dsl_jit_symbol *next = realloc(table->items, (size_t)new_cap * sizeof(*next));
        if (!next) {
            return false;
        }
        table->items = next;
        table->capacity = new_cap;
    }
    me_dsl_jit_symbol *item = &table->items[table->count++];
    item->name = dsl_jit_strdup(name);
    if (!item->name) {
        table->count--;
        return false;
    }
    item->dtype = dtype;
    item->is_param = is_param;
    return true;
}

static bool dsl_jit_resolve_expr_dtype(me_dsl_jit_build_ctx *ctx, const me_dsl_expr *expr,
                                       int line, int column, me_dtype *out_dtype) {
    if (!ctx || !expr || !out_dtype || !ctx->resolve_dtype) {
        dsl_jit_set_error(ctx ? ctx->error : NULL, line, column,
                          "missing dtype resolver for jit ir");
        return false;
    }
    me_dtype dtype = ME_AUTO;
    if (!ctx->resolve_dtype(ctx->resolve_ctx, expr, &dtype)) {
        dsl_jit_set_error(ctx->error, line, column, "failed to resolve expression dtype for jit ir");
        return false;
    }
    if (!dsl_jit_dtype_supported(dtype)) {
        dsl_jit_set_error(ctx->error, line, column, "unsupported expression dtype for jit ir");
        return false;
    }
    *out_dtype = dtype;
    return true;
}

static bool dsl_jit_expr_has_comma(const char *text) {
    bool in_string = false;
    char quote = '\0';
    int depth = 0;
    if (!text) {
        return false;
    }
    for (const char *p = text; *p; p++) {
        char c = *p;
        if (in_string) {
            if (c == '\\' && p[1]) {
                p++;
                continue;
            }
            if (c == quote) {
                in_string = false;
            }
            continue;
        }
        if (c == '"' || c == '\'') {
            in_string = true;
            quote = c;
            continue;
        }
        if (c == '(') {
            depth++;
            continue;
        }
        if (c == ')') {
            if (depth > 0) {
                depth--;
            }
            continue;
        }
        if (c == ',' && depth == 0) {
            return true;
        }
    }
    return false;
}

static bool dsl_jit_build_block(me_dsl_jit_build_ctx *ctx, const me_dsl_block *in,
                                me_dsl_jit_ir_block *out, bool in_loop) {
    if (!ctx || !in || !out) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    for (int i = 0; i < in->nstmts; i++) {
        me_dsl_stmt *stmt = in->stmts[i];
        if (!stmt) {
            continue;
        }
        me_dsl_jit_ir_stmt *ir_stmt = calloc(1, sizeof(*ir_stmt));
        if (!ir_stmt) {
            dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        ir_stmt->line = stmt->line;
        ir_stmt->column = stmt->column;

        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN: {
            if (!stmt->as.assign.name || !stmt->as.assign.value) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "invalid assignment in dsl");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            me_dtype rhs_dtype = ME_AUTO;
            if (!dsl_jit_resolve_expr_dtype(ctx, stmt->as.assign.value, stmt->line, stmt->column,
                                            &rhs_dtype)) {
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            int sym_index = dsl_jit_symbols_find(&ctx->symbols, stmt->as.assign.name);
            if (sym_index >= 0 && ctx->symbols.items[sym_index].is_param) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "assigning to kernel input is not supported by jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (sym_index < 0) {
                if (!dsl_jit_symbols_add(&ctx->symbols, stmt->as.assign.name, rhs_dtype, false)) {
                    dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                    dsl_jit_ir_stmt_free(ir_stmt);
                    return false;
                }
            }
            else if (ctx->symbols.items[sym_index].dtype != rhs_dtype) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "assignment dtype mismatch for jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->kind = ME_DSL_JIT_IR_STMT_ASSIGN;
            ir_stmt->as.assign.name = dsl_jit_strdup(stmt->as.assign.name);
            if (!ir_stmt->as.assign.name) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->as.assign.dtype = rhs_dtype;
            if (!dsl_jit_ir_expr_init(&ir_stmt->as.assign.value, stmt->as.assign.value, rhs_dtype)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            break;
        }
        case ME_DSL_STMT_RETURN: {
            if (!stmt->as.return_stmt.expr) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "invalid return in dsl");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            me_dtype ret_dtype = ME_AUTO;
            if (!dsl_jit_resolve_expr_dtype(ctx, stmt->as.return_stmt.expr, stmt->line, stmt->column,
                                            &ret_dtype)) {
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->kind = ME_DSL_JIT_IR_STMT_RETURN;
            if (!dsl_jit_ir_expr_init(&ir_stmt->as.return_stmt.expr, stmt->as.return_stmt.expr, ret_dtype)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            break;
        }
        case ME_DSL_STMT_IF: {
            if (!stmt->as.if_stmt.cond) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "invalid if condition");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            me_dtype cond_dtype = ME_AUTO;
            if (!dsl_jit_resolve_expr_dtype(ctx, stmt->as.if_stmt.cond, stmt->line, stmt->column,
                                            &cond_dtype)) {
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->kind = ME_DSL_JIT_IR_STMT_IF;
            if (!dsl_jit_ir_expr_init(&ir_stmt->as.if_stmt.cond, stmt->as.if_stmt.cond, cond_dtype)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (!dsl_jit_build_block(ctx, &stmt->as.if_stmt.then_block, &ir_stmt->as.if_stmt.then_block,
                                     in_loop)) {
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->as.if_stmt.n_elifs = stmt->as.if_stmt.n_elifs;
            ir_stmt->as.if_stmt.elif_capacity = stmt->as.if_stmt.n_elifs;
            if (stmt->as.if_stmt.n_elifs > 0) {
                ir_stmt->as.if_stmt.elif_branches = calloc((size_t)stmt->as.if_stmt.n_elifs,
                                                           sizeof(*ir_stmt->as.if_stmt.elif_branches));
                if (!ir_stmt->as.if_stmt.elif_branches) {
                    dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                    dsl_jit_ir_stmt_free(ir_stmt);
                    return false;
                }
            }
            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                me_dsl_if_branch *in_branch = &stmt->as.if_stmt.elif_branches[j];
                me_dsl_jit_ir_if_branch *out_branch = &ir_stmt->as.if_stmt.elif_branches[j];
                if (!in_branch->cond) {
                    dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "invalid elif condition");
                    dsl_jit_ir_stmt_free(ir_stmt);
                    return false;
                }
                me_dtype elif_dtype = ME_AUTO;
                if (!dsl_jit_resolve_expr_dtype(ctx, in_branch->cond, in_branch->cond->line,
                                                in_branch->cond->column, &elif_dtype)) {
                    dsl_jit_ir_stmt_free(ir_stmt);
                    return false;
                }
                if (!dsl_jit_ir_expr_init(&out_branch->cond, in_branch->cond, elif_dtype)) {
                    dsl_jit_set_error(ctx->error, in_branch->cond->line, in_branch->cond->column,
                                      "out of memory");
                    dsl_jit_ir_stmt_free(ir_stmt);
                    return false;
                }
                if (!dsl_jit_build_block(ctx, &in_branch->block, &out_branch->block, in_loop)) {
                    dsl_jit_ir_stmt_free(ir_stmt);
                    return false;
                }
            }
            if (stmt->as.if_stmt.has_else) {
                ir_stmt->as.if_stmt.has_else = true;
                if (!dsl_jit_build_block(ctx, &stmt->as.if_stmt.else_block,
                                         &ir_stmt->as.if_stmt.else_block, in_loop)) {
                    dsl_jit_ir_stmt_free(ir_stmt);
                    return false;
                }
            }
            break;
        }
        case ME_DSL_STMT_FOR: {
            if (!stmt->as.for_loop.var || !stmt->as.for_loop.limit) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "invalid for-loop in dsl");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (dsl_jit_symbols_find(&ctx->symbols, stmt->as.for_loop.var) >= 0) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "for-loop variable must be a new temporary for jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (dsl_jit_expr_has_comma(stmt->as.for_loop.limit->text)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "range() with start/stop/step is not supported by jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            me_dtype limit_dtype = ME_AUTO;
            if (!dsl_jit_resolve_expr_dtype(ctx, stmt->as.for_loop.limit, stmt->line, stmt->column,
                                            &limit_dtype)) {
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->kind = ME_DSL_JIT_IR_STMT_FOR;
            ir_stmt->as.for_loop.var = dsl_jit_strdup(stmt->as.for_loop.var);
            if (!ir_stmt->as.for_loop.var) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (!dsl_jit_ir_expr_init(&ir_stmt->as.for_loop.limit, stmt->as.for_loop.limit, limit_dtype)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (!dsl_jit_symbols_add(&ctx->symbols, stmt->as.for_loop.var, ME_INT64, false)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (!dsl_jit_build_block(ctx, &stmt->as.for_loop.body, &ir_stmt->as.for_loop.body, true)) {
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            break;
        }
        case ME_DSL_STMT_BREAK:
            if (!in_loop) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "break is only supported inside for-loops in jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (stmt->as.flow.cond) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "conditional break is not supported by jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->kind = ME_DSL_JIT_IR_STMT_BREAK;
            break;
        case ME_DSL_STMT_EXPR:
            dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                              "expression statements are not supported by jit ir");
            dsl_jit_ir_stmt_free(ir_stmt);
            return false;
        case ME_DSL_STMT_PRINT:
            dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                              "print statements are not supported by jit ir");
            dsl_jit_ir_stmt_free(ir_stmt);
            return false;
        case ME_DSL_STMT_CONTINUE:
            if (!in_loop) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "continue is only supported inside for-loops in jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (stmt->as.flow.cond) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "conditional continue is not supported by jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            ir_stmt->kind = ME_DSL_JIT_IR_STMT_CONTINUE;
            break;
        }

        if (!dsl_jit_ir_block_push(out, ir_stmt)) {
            dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            dsl_jit_ir_stmt_free(ir_stmt);
            return false;
        }
    }
    return true;
}

bool me_dsl_jit_ir_build(const me_dsl_program *program, const char **param_names,
                         const me_dtype *param_dtypes, int nparams,
                         me_dsl_jit_ir_dtype_resolver resolve_dtype, void *resolve_ctx,
                         me_dsl_jit_ir_program **out_ir, me_dsl_error *error) {
    if (out_ir) {
        *out_ir = NULL;
    }
    if (!program || !out_ir || nparams < 0 || (nparams > 0 && (!param_names || !param_dtypes))) {
        dsl_jit_set_error(error, 0, 0, "invalid arguments for jit ir build");
        return false;
    }

    me_dsl_jit_ir_program *ir = calloc(1, sizeof(*ir));
    if (!ir) {
        dsl_jit_set_error(error, 0, 0, "out of memory");
        return false;
    }

    ir->name = dsl_jit_strdup(program->name ? program->name : "kernel");
    if (!ir->name) {
        me_dsl_jit_ir_free(ir);
        dsl_jit_set_error(error, 0, 0, "out of memory");
        return false;
    }
    ir->dialect = program->dialect;

    ir->nparams = nparams;
    if (nparams > 0) {
        ir->params = calloc((size_t)nparams, sizeof(*ir->params));
        ir->param_dtypes = calloc((size_t)nparams, sizeof(*ir->param_dtypes));
        if (!ir->params || !ir->param_dtypes) {
            me_dsl_jit_ir_free(ir);
            dsl_jit_set_error(error, 0, 0, "out of memory");
            return false;
        }
    }

    me_dsl_jit_build_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.resolve_dtype = resolve_dtype;
    ctx.resolve_ctx = resolve_ctx;
    ctx.error = error;

    for (int i = 0; i < nparams; i++) {
        if (!param_names[i] || !dsl_jit_dtype_supported(param_dtypes[i])) {
            dsl_jit_set_error(error, 0, 0, "invalid parameter metadata for jit ir");
            dsl_jit_symbols_free(&ctx.symbols);
            me_dsl_jit_ir_free(ir);
            return false;
        }
        ir->params[i] = dsl_jit_strdup(param_names[i]);
        if (!ir->params[i]) {
            dsl_jit_set_error(error, 0, 0, "out of memory");
            dsl_jit_symbols_free(&ctx.symbols);
            me_dsl_jit_ir_free(ir);
            return false;
        }
        ir->param_dtypes[i] = param_dtypes[i];
        if (dsl_jit_symbols_find(&ctx.symbols, param_names[i]) >= 0) {
            dsl_jit_set_error(error, 0, 0, "duplicate parameter in jit ir metadata");
            dsl_jit_symbols_free(&ctx.symbols);
            me_dsl_jit_ir_free(ir);
            return false;
        }
        if (!dsl_jit_symbols_add(&ctx.symbols, param_names[i], param_dtypes[i], true)) {
            dsl_jit_set_error(error, 0, 0, "out of memory");
            dsl_jit_symbols_free(&ctx.symbols);
            me_dsl_jit_ir_free(ir);
            return false;
        }
    }

    if (!dsl_jit_build_block(&ctx, &program->block, &ir->block, false)) {
        dsl_jit_symbols_free(&ctx.symbols);
        me_dsl_jit_ir_free(ir);
        return false;
    }

    dsl_jit_symbols_free(&ctx.symbols);
    *out_ir = ir;
    return true;
}

static uint64_t dsl_jit_hash_bytes(uint64_t h, const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t dsl_jit_hash_string(uint64_t h, const char *s) {
    if (!s) {
        return dsl_jit_hash_bytes(h, "", 1);
    }
    return dsl_jit_hash_bytes(h, s, strlen(s) + 1);
}

static uint64_t dsl_jit_hash_i32(uint64_t h, int v) {
    return dsl_jit_hash_bytes(h, &v, sizeof(v));
}

static uint64_t dsl_jit_hash_dtype(uint64_t h, me_dtype dtype) {
    int v = (int)dtype;
    return dsl_jit_hash_bytes(h, &v, sizeof(v));
}

static uint64_t dsl_jit_hash_dialect(uint64_t h, me_dsl_dialect dialect) {
    int v = (int)dialect;
    return dsl_jit_hash_bytes(h, &v, sizeof(v));
}

static uint64_t dsl_jit_ir_hash_expr(uint64_t h, const me_dsl_jit_ir_expr *expr) {
    h = dsl_jit_hash_string(h, expr ? expr->text : "");
    h = dsl_jit_hash_dtype(h, expr ? expr->dtype : ME_AUTO);
    return h;
}

static uint64_t dsl_jit_ir_hash_block(uint64_t h, const me_dsl_jit_ir_block *block);

static uint64_t dsl_jit_ir_hash_stmt(uint64_t h, const me_dsl_jit_ir_stmt *stmt) {
    if (!stmt) {
        return dsl_jit_hash_i32(h, -1);
    }
    h = dsl_jit_hash_i32(h, (int)stmt->kind);
    h = dsl_jit_hash_i32(h, stmt->line);
    h = dsl_jit_hash_i32(h, stmt->column);
    switch (stmt->kind) {
    case ME_DSL_JIT_IR_STMT_ASSIGN:
        h = dsl_jit_hash_string(h, stmt->as.assign.name);
        h = dsl_jit_hash_dtype(h, stmt->as.assign.dtype);
        h = dsl_jit_ir_hash_expr(h, &stmt->as.assign.value);
        break;
    case ME_DSL_JIT_IR_STMT_RETURN:
        h = dsl_jit_ir_hash_expr(h, &stmt->as.return_stmt.expr);
        break;
    case ME_DSL_JIT_IR_STMT_IF:
        h = dsl_jit_ir_hash_expr(h, &stmt->as.if_stmt.cond);
        h = dsl_jit_ir_hash_block(h, &stmt->as.if_stmt.then_block);
        h = dsl_jit_hash_i32(h, stmt->as.if_stmt.n_elifs);
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            h = dsl_jit_ir_hash_expr(h, &stmt->as.if_stmt.elif_branches[i].cond);
            h = dsl_jit_ir_hash_block(h, &stmt->as.if_stmt.elif_branches[i].block);
        }
        h = dsl_jit_hash_i32(h, stmt->as.if_stmt.has_else ? 1 : 0);
        if (stmt->as.if_stmt.has_else) {
            h = dsl_jit_ir_hash_block(h, &stmt->as.if_stmt.else_block);
        }
        break;
    case ME_DSL_JIT_IR_STMT_FOR:
        h = dsl_jit_hash_string(h, stmt->as.for_loop.var);
        h = dsl_jit_ir_hash_expr(h, &stmt->as.for_loop.limit);
        h = dsl_jit_ir_hash_block(h, &stmt->as.for_loop.body);
        break;
    case ME_DSL_JIT_IR_STMT_BREAK:
    case ME_DSL_JIT_IR_STMT_CONTINUE:
        break;
    }
    return h;
}

static uint64_t dsl_jit_ir_hash_block(uint64_t h, const me_dsl_jit_ir_block *block) {
    if (!block) {
        return dsl_jit_hash_i32(h, -1);
    }
    h = dsl_jit_hash_i32(h, block->nstmts);
    for (int i = 0; i < block->nstmts; i++) {
        h = dsl_jit_ir_hash_stmt(h, block->stmts[i]);
    }
    return h;
}

uint64_t me_dsl_jit_ir_fingerprint(const me_dsl_jit_ir_program *program) {
    uint64_t h = UINT64_C(1469598103934665603);
    if (!program) {
        return h;
    }
    h = dsl_jit_hash_string(h, program->name);
    h = dsl_jit_hash_dialect(h, program->dialect);
    h = dsl_jit_hash_i32(h, program->nparams);
    for (int i = 0; i < program->nparams; i++) {
        h = dsl_jit_hash_string(h, program->params[i]);
        h = dsl_jit_hash_dtype(h, program->param_dtypes[i]);
    }
    h = dsl_jit_ir_hash_block(h, &program->block);
    return h;
}
