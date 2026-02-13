/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_ir.h"

#include <ctype.h>
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
        dsl_jit_ir_expr_free(&stmt->as.for_loop.start);
        dsl_jit_ir_expr_free(&stmt->as.for_loop.stop);
        dsl_jit_ir_expr_free(&stmt->as.for_loop.step);
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

static char *dsl_jit_trim_copy(const char *start, const char *end) {
    if (!start || !end || end < start) {
        return NULL;
    }
    while (start < end && isspace((unsigned char)*start)) {
        start++;
    }
    while (end > start && isspace((unsigned char)end[-1])) {
        end--;
    }
    size_t len = (size_t)(end - start);
    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, start, len);
    out[len] = '\0';
    return out;
}

static bool dsl_jit_split_top_level_csv(const char *text, char ***out_parts, int *out_nparts) {
    if (!text || !out_parts || !out_nparts) {
        return false;
    }
    *out_parts = NULL;
    *out_nparts = 0;

    const char *part_start = text;
    const char *p = text;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    char quote = '\0';
    char **parts = NULL;
    int nparts = 0;

    while (*p) {
        char c = *p;
        if (quote) {
            if (c == '\\' && p[1] != '\0') {
                p += 2;
                continue;
            }
            if (c == quote) {
                quote = '\0';
            }
            p++;
            continue;
        }
        if (c == '"' || c == '\'') {
            quote = c;
            p++;
            continue;
        }
        if (c == '(') {
            paren_depth++;
            p++;
            continue;
        }
        if (c == ')') {
            if (paren_depth > 0) {
                paren_depth--;
            }
            p++;
            continue;
        }
        if (c == '[') {
            bracket_depth++;
            p++;
            continue;
        }
        if (c == ']') {
            if (bracket_depth > 0) {
                bracket_depth--;
            }
            p++;
            continue;
        }
        if (c == '{') {
            brace_depth++;
            p++;
            continue;
        }
        if (c == '}') {
            if (brace_depth > 0) {
                brace_depth--;
            }
            p++;
            continue;
        }
        if (c == ',' && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
            char *part = dsl_jit_trim_copy(part_start, p);
            if (!part || part[0] == '\0') {
                free(part);
                for (int i = 0; i < nparts; i++) {
                    free(parts[i]);
                }
                free(parts);
                return false;
            }
            char **next = realloc(parts, (size_t)(nparts + 1) * sizeof(*next));
            if (!next) {
                free(part);
                for (int i = 0; i < nparts; i++) {
                    free(parts[i]);
                }
                free(parts);
                return false;
            }
            parts = next;
            parts[nparts++] = part;
            part_start = p + 1;
        }
        p++;
    }

    if (quote || paren_depth != 0 || bracket_depth != 0 || brace_depth != 0) {
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }

    char *tail = dsl_jit_trim_copy(part_start, p);
    if (!tail || tail[0] == '\0') {
        free(tail);
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }
    char **next = realloc(parts, (size_t)(nparts + 1) * sizeof(*next));
    if (!next) {
        free(tail);
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }
    parts = next;
    parts[nparts++] = tail;

    *out_parts = parts;
    *out_nparts = nparts;
    return true;
}

static bool dsl_jit_ir_expr_init_from_text(me_dsl_jit_ir_expr *out, const char *text, me_dtype dtype) {
    if (!out || !text) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->text = dsl_jit_strdup(text);
    if (!out->text) {
        return false;
    }
    out->dtype = dtype;
    return true;
}

static bool dsl_jit_resolve_expr_dtype_from_text(me_dsl_jit_build_ctx *ctx, const char *text,
                                                 int line, int column, me_dtype *out_dtype) {
    if (!ctx || !text || !out_dtype) {
        return false;
    }
    me_dsl_expr expr;
    memset(&expr, 0, sizeof(expr));
    expr.text = (char *)text;
    expr.line = line;
    expr.column = column;
    return dsl_jit_resolve_expr_dtype(ctx, &expr, line, column, out_dtype);
}

static bool dsl_jit_ident_is_reduction(const char *ident, size_t len) {
    if (!ident || len == 0) {
        return false;
    }
    return (len == 3 && strncmp(ident, "any", 3) == 0) ||
           (len == 3 && strncmp(ident, "all", 3) == 0) ||
           (len == 3 && strncmp(ident, "sum", 3) == 0) ||
           (len == 4 && strncmp(ident, "mean", 4) == 0) ||
           (len == 3 && strncmp(ident, "min", 3) == 0) ||
           (len == 3 && strncmp(ident, "max", 3) == 0) ||
           (len == 4 && strncmp(ident, "prod", 4) == 0);
}

static bool dsl_jit_expr_has_reduction_call(const char *text) {
    bool in_string = false;
    char quote = '\0';
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
        if (isalpha((unsigned char)c) || c == '_') {
            const char *start = p;
            p++;
            while (isalnum((unsigned char)*p) || *p == '_') {
                p++;
            }
            size_t len = (size_t)(p - start);
            const char *q = p;
            while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n') {
                q++;
            }
            if (*q == '(' && dsl_jit_ident_is_reduction(start, len)) {
                return true;
            }
            p--;
        }
    }
    return false;
}

static bool dsl_jit_expr_validate_subset(me_dsl_jit_build_ctx *ctx, const me_dsl_expr *expr,
                                         int line, int column) {
    if (!ctx || !expr || !expr->text) {
        dsl_jit_set_error(ctx ? ctx->error : NULL, line, column,
                          "invalid expression in jit ir");
        return false;
    }
    if (dsl_jit_expr_has_reduction_call(expr->text)) {
        dsl_jit_set_error(ctx->error, line, column,
                          "reduction functions are not supported by jit ir");
        return false;
    }
    return true;
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
            if (!dsl_jit_expr_validate_subset(ctx, stmt->as.assign.value,
                                              stmt->as.assign.value->line,
                                              stmt->as.assign.value->column)) {
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
            if (!dsl_jit_expr_validate_subset(ctx, stmt->as.return_stmt.expr,
                                              stmt->as.return_stmt.expr->line,
                                              stmt->as.return_stmt.expr->column)) {
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
            if (!dsl_jit_expr_validate_subset(ctx, stmt->as.if_stmt.cond,
                                              stmt->as.if_stmt.cond->line,
                                              stmt->as.if_stmt.cond->column)) {
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
                if (!dsl_jit_expr_validate_subset(ctx, in_branch->cond,
                                                  in_branch->cond->line,
                                                  in_branch->cond->column)) {
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

            char **range_parts = NULL;
            int n_range_parts = 0;
            if (!dsl_jit_split_top_level_csv(stmt->as.for_loop.limit->text, &range_parts, &n_range_parts)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "invalid range() argument list for jit ir");
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (n_range_parts < 1 || n_range_parts > 3) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column,
                                  "range() expects 1 to 3 arguments for jit ir");
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }

            const char *start_text = "0";
            const char *stop_text = range_parts[0];
            const char *step_text = "1";
            if (n_range_parts == 2) {
                start_text = range_parts[0];
                stop_text = range_parts[1];
            }
            else if (n_range_parts == 3) {
                start_text = range_parts[0];
                stop_text = range_parts[1];
                step_text = range_parts[2];
            }

            me_dsl_expr tmp_expr;
            memset(&tmp_expr, 0, sizeof(tmp_expr));
            tmp_expr.line = stmt->line;
            tmp_expr.column = stmt->column;
            tmp_expr.text = (char *)start_text;
            if (!dsl_jit_expr_validate_subset(ctx, &tmp_expr, stmt->line, stmt->column)) {
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            tmp_expr.text = (char *)stop_text;
            if (!dsl_jit_expr_validate_subset(ctx, &tmp_expr, stmt->line, stmt->column)) {
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            tmp_expr.text = (char *)step_text;
            if (!dsl_jit_expr_validate_subset(ctx, &tmp_expr, stmt->line, stmt->column)) {
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }

            me_dtype start_dtype = ME_AUTO;
            me_dtype stop_dtype = ME_AUTO;
            me_dtype step_dtype = ME_AUTO;
            if (!dsl_jit_resolve_expr_dtype_from_text(ctx, start_text, stmt->line, stmt->column,
                                                      &start_dtype) ||
                !dsl_jit_resolve_expr_dtype_from_text(ctx, stop_text, stmt->line, stmt->column,
                                                      &stop_dtype) ||
                !dsl_jit_resolve_expr_dtype_from_text(ctx, step_text, stmt->line, stmt->column,
                                                      &step_dtype)) {
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }

            ir_stmt->kind = ME_DSL_JIT_IR_STMT_FOR;
            ir_stmt->as.for_loop.var = dsl_jit_strdup(stmt->as.for_loop.var);
            if (!ir_stmt->as.for_loop.var) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (!dsl_jit_ir_expr_init_from_text(&ir_stmt->as.for_loop.start, start_text, start_dtype) ||
                !dsl_jit_ir_expr_init_from_text(&ir_stmt->as.for_loop.stop, stop_text, stop_dtype) ||
                !dsl_jit_ir_expr_init_from_text(&ir_stmt->as.for_loop.step, step_text, step_dtype)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (!dsl_jit_symbols_add(&ctx->symbols, stmt->as.for_loop.var, ME_INT64, false)) {
                dsl_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            if (!dsl_jit_build_block(ctx, &stmt->as.for_loop.body, &ir_stmt->as.for_loop.body, true)) {
                for (int i = 0; i < n_range_parts; i++) {
                    free(range_parts[i]);
                }
                free(range_parts);
                dsl_jit_ir_stmt_free(ir_stmt);
                return false;
            }
            for (int i = 0; i < n_range_parts; i++) {
                free(range_parts[i]);
            }
            free(range_parts);
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
    ir->fp_mode = program->fp_mode;

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

static uint64_t dsl_jit_hash_fp_mode(uint64_t h, me_dsl_fp_mode fp_mode) {
    int v = (int)fp_mode;
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
        h = dsl_jit_ir_hash_expr(h, &stmt->as.for_loop.start);
        h = dsl_jit_ir_hash_expr(h, &stmt->as.for_loop.stop);
        h = dsl_jit_ir_hash_expr(h, &stmt->as.for_loop.step);
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
    h = dsl_jit_hash_fp_mode(h, program->fp_mode);
    h = dsl_jit_hash_i32(h, program->nparams);
    for (int i = 0; i < program->nparams; i++) {
        h = dsl_jit_hash_string(h, program->params[i]);
        h = dsl_jit_hash_dtype(h, program->param_dtypes[i]);
    }
    h = dsl_jit_ir_hash_block(h, &program->block);
    return h;
}
