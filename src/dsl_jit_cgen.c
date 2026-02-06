/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_cgen.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} me_jit_strbuf;

typedef struct {
    char *name;
    me_dtype dtype;
} me_jit_local;

typedef struct {
    me_jit_local *items;
    int count;
    int capacity;
} me_jit_locals;

typedef struct {
    me_jit_strbuf source;
    me_jit_locals locals;
    me_dtype output_dtype;
    const char *out_var_name;
    me_dsl_error *error;
} me_jit_codegen_ctx;

static void me_jit_set_error(me_dsl_error *error, int line, int column, const char *msg) {
    if (!error) {
        return;
    }
    error->line = line;
    error->column = column;
    snprintf(error->message, sizeof(error->message), "%s", msg ? msg : "jit c codegen failed");
}

static bool me_jit_dtype_supported(me_dtype dtype) {
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

static bool me_jit_dtype_is_integral(me_dtype dtype) {
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
        return true;
    default:
        return false;
    }
}

static const char *me_jit_c_type(me_dtype dtype) {
    switch (dtype) {
    case ME_BOOL: return "bool";
    case ME_INT8: return "int8_t";
    case ME_INT16: return "int16_t";
    case ME_INT32: return "int32_t";
    case ME_INT64: return "int64_t";
    case ME_UINT8: return "uint8_t";
    case ME_UINT16: return "uint16_t";
    case ME_UINT32: return "uint32_t";
    case ME_UINT64: return "uint64_t";
    case ME_FLOAT32: return "float";
    case ME_FLOAT64: return "double";
    default: return NULL;
    }
}

static bool me_jit_buf_reserve(me_jit_strbuf *buf, size_t add_len) {
    if (!buf) {
        return false;
    }
    size_t need = buf->len + add_len + 1;
    if (need <= buf->cap) {
        return true;
    }
    size_t next_cap = buf->cap ? buf->cap * 2 : 256;
    while (next_cap < need) {
        next_cap *= 2;
    }
    char *next = realloc(buf->data, next_cap);
    if (!next) {
        return false;
    }
    buf->data = next;
    buf->cap = next_cap;
    return true;
}

static bool me_jit_buf_append_n(me_jit_strbuf *buf, const char *s, size_t n) {
    if (!buf || !s) {
        return false;
    }
    if (!me_jit_buf_reserve(buf, n)) {
        return false;
    }
    memcpy(buf->data + buf->len, s, n);
    buf->len += n;
    buf->data[buf->len] = '\0';
    return true;
}

static bool me_jit_buf_append(me_jit_strbuf *buf, const char *s) {
    if (!s) {
        return false;
    }
    return me_jit_buf_append_n(buf, s, strlen(s));
}

static bool me_jit_emit_indent(me_jit_strbuf *buf, int indent) {
    for (int i = 0; i < indent; i++) {
        if (!me_jit_buf_append(buf, "    ")) {
            return false;
        }
    }
    return true;
}

static bool me_jit_emit_line(me_jit_strbuf *buf, int indent, const char *line) {
    if (!me_jit_emit_indent(buf, indent)) {
        return false;
    }
    if (!me_jit_buf_append(buf, line)) {
        return false;
    }
    if (!me_jit_buf_append(buf, "\n")) {
        return false;
    }
    return true;
}

static char *me_jit_strdup(const char *s) {
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

static void me_jit_locals_free(me_jit_locals *locals) {
    if (!locals) {
        return;
    }
    for (int i = 0; i < locals->count; i++) {
        free(locals->items[i].name);
    }
    free(locals->items);
    locals->items = NULL;
    locals->count = 0;
    locals->capacity = 0;
}

static int me_jit_locals_find(const me_jit_locals *locals, const char *name) {
    if (!locals || !name) {
        return -1;
    }
    for (int i = 0; i < locals->count; i++) {
        if (strcmp(locals->items[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static bool me_jit_locals_add(me_jit_locals *locals, const char *name, me_dtype dtype) {
    if (!locals || !name) {
        return false;
    }
    int idx = me_jit_locals_find(locals, name);
    if (idx >= 0) {
        return locals->items[idx].dtype == dtype;
    }
    if (locals->count == locals->capacity) {
        int new_cap = locals->capacity ? locals->capacity * 2 : 8;
        me_jit_local *next = realloc(locals->items, (size_t)new_cap * sizeof(*next));
        if (!next) {
            return false;
        }
        locals->items = next;
        locals->capacity = new_cap;
    }
    me_jit_local *entry = &locals->items[locals->count++];
    entry->name = me_jit_strdup(name);
    if (!entry->name) {
        locals->count--;
        return false;
    }
    entry->dtype = dtype;
    return true;
}

static bool me_jit_collect_locals_block(me_jit_locals *locals, const me_dsl_jit_ir_block *block);

static bool me_jit_collect_locals_stmt(me_jit_locals *locals, const me_dsl_jit_ir_stmt *stmt) {
    if (!locals || !stmt) {
        return false;
    }
    switch (stmt->kind) {
    case ME_DSL_JIT_IR_STMT_ASSIGN:
        if (!me_jit_locals_add(locals, stmt->as.assign.name, stmt->as.assign.dtype)) {
            return false;
        }
        break;
    case ME_DSL_JIT_IR_STMT_IF:
        if (!me_jit_collect_locals_block(locals, &stmt->as.if_stmt.then_block)) {
            return false;
        }
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            if (!me_jit_collect_locals_block(locals, &stmt->as.if_stmt.elif_branches[i].block)) {
                return false;
            }
        }
        if (stmt->as.if_stmt.has_else) {
            if (!me_jit_collect_locals_block(locals, &stmt->as.if_stmt.else_block)) {
                return false;
            }
        }
        break;
    case ME_DSL_JIT_IR_STMT_FOR:
        if (!me_jit_locals_add(locals, stmt->as.for_loop.var, ME_INT64)) {
            return false;
        }
        if (!me_jit_collect_locals_block(locals, &stmt->as.for_loop.body)) {
            return false;
        }
        break;
    case ME_DSL_JIT_IR_STMT_RETURN:
    case ME_DSL_JIT_IR_STMT_BREAK:
    case ME_DSL_JIT_IR_STMT_CONTINUE:
        break;
    }
    return true;
}

static bool me_jit_collect_locals_block(me_jit_locals *locals, const me_dsl_jit_ir_block *block) {
    if (!locals || !block) {
        return false;
    }
    for (int i = 0; i < block->nstmts; i++) {
        if (!me_jit_collect_locals_stmt(locals, block->stmts[i])) {
            return false;
        }
    }
    return true;
}

static bool me_jit_expr_contains_unsupported_tokens(const char *expr, me_dtype dtype) {
    bool in_string = false;
    char quote = '\0';
    if (!expr) {
        return true;
    }
    for (const char *p = expr; *p; p++) {
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
        if (c == '*') {
            if (p[1] == '*') {
                return true;
            }
            continue;
        }
        if (c == '%') {
            return true;
        }
        if (c == '<' && p[1] == '<') {
            if (!me_jit_dtype_is_integral(dtype)) {
                return true;
            }
            p++;
            continue;
        }
        if (c == '>' && p[1] == '>') {
            if (!me_jit_dtype_is_integral(dtype)) {
                return true;
            }
            p++;
            continue;
        }
        if (c == '&') {
            if (p[1] == '&') {
                p++;
                continue;
            }
            if (!me_jit_dtype_is_integral(dtype)) {
                return true;
            }
            continue;
        }
        if (c == '|') {
            if (p[1] == '|') {
                p++;
                continue;
            }
            if (!me_jit_dtype_is_integral(dtype)) {
                return true;
            }
            continue;
        }
        if (c == '^' || c == '~') {
            if (!me_jit_dtype_is_integral(dtype)) {
                return true;
            }
            continue;
        }
    }
    return false;
}

static bool me_jit_expr_to_c(const me_dsl_jit_ir_expr *expr, char **out_c,
                             me_dsl_error *error, int line, int column) {
    if (out_c) {
        *out_c = NULL;
    }
    if (!expr || !out_c || !expr->text) {
        me_jit_set_error(error, line, column, "invalid expression in jit c codegen");
        return false;
    }
    if (me_jit_expr_contains_unsupported_tokens(expr->text, expr->dtype)) {
        me_jit_set_error(error, line, column, "expression uses unsupported operator for jit c codegen");
        return false;
    }

    size_t len = strlen(expr->text);
    size_t cap = len * 2 + 16;
    char *out = malloc(cap);
    if (!out) {
        me_jit_set_error(error, line, column, "out of memory");
        return false;
    }
    size_t o = 0;
    bool in_string = false;
    char quote = '\0';
    const char *p = expr->text;

    while (*p) {
        char c = *p;
        if (in_string) {
            if (o + 2 >= cap) {
                cap *= 2;
                char *next = realloc(out, cap);
                if (!next) {
                    free(out);
                    me_jit_set_error(error, line, column, "out of memory");
                    return false;
                }
                out = next;
            }
            out[o++] = c;
            if (c == '\\' && p[1]) {
                out[o++] = p[1];
                p += 2;
                continue;
            }
            if (c == quote) {
                in_string = false;
            }
            p++;
            continue;
        }

        if (c == '"' || c == '\'') {
            if (o + 2 >= cap) {
                cap *= 2;
                char *next = realloc(out, cap);
                if (!next) {
                    free(out);
                    me_jit_set_error(error, line, column, "out of memory");
                    return false;
                }
                out = next;
            }
            in_string = true;
            quote = c;
            out[o++] = c;
            p++;
            continue;
        }

        if (isalpha((unsigned char)c) || c == '_') {
            const char *start = p;
            p++;
            while (isalnum((unsigned char)*p) || *p == '_') {
                p++;
            }
            size_t ident_len = (size_t)(p - start);
            const char *rep = NULL;
            if (ident_len == 3 && strncmp(start, "and", ident_len) == 0) {
                rep = "&&";
            }
            else if (ident_len == 2 && strncmp(start, "or", ident_len) == 0) {
                rep = "||";
            }
            else if (ident_len == 3 && strncmp(start, "not", ident_len) == 0) {
                rep = "!";
            }
            if (rep) {
                size_t rep_len = strlen(rep);
                if (o + rep_len + 1 >= cap) {
                    while (o + rep_len + 1 >= cap) {
                        cap *= 2;
                    }
                    char *next = realloc(out, cap);
                    if (!next) {
                        free(out);
                        me_jit_set_error(error, line, column, "out of memory");
                        return false;
                    }
                    out = next;
                }
                memcpy(out + o, rep, rep_len);
                o += rep_len;
            }
            else {
                if (o + ident_len + 1 >= cap) {
                    while (o + ident_len + 1 >= cap) {
                        cap *= 2;
                    }
                    char *next = realloc(out, cap);
                    if (!next) {
                        free(out);
                        me_jit_set_error(error, line, column, "out of memory");
                        return false;
                    }
                    out = next;
                }
                memcpy(out + o, start, ident_len);
                o += ident_len;
            }
            continue;
        }

        if (o + 2 >= cap) {
            cap *= 2;
            char *next = realloc(out, cap);
            if (!next) {
                free(out);
                me_jit_set_error(error, line, column, "out of memory");
                return false;
            }
            out = next;
        }
        out[o++] = c;
        p++;
    }

    out[o] = '\0';
    *out_c = out;
    return true;
}

static bool me_jit_emit_casted_expr_line(me_jit_codegen_ctx *ctx, int indent,
                                         const char *lhs, me_dtype lhs_dtype,
                                         const me_dsl_jit_ir_expr *rhs,
                                         int line, int column) {
    char *rhs_c = NULL;
    const char *ctype = me_jit_c_type(lhs_dtype);
    if (!ctype) {
        me_jit_set_error(ctx->error, line, column, "unsupported dtype in jit c codegen");
        return false;
    }
    if (!me_jit_expr_to_c(rhs, &rhs_c, ctx->error, line, column)) {
        free(rhs_c);
        return false;
    }

    size_t need = strlen(lhs) + strlen(ctype) * 2 + strlen(rhs_c) + 16;
    char *line_buf = malloc(need);
    if (!line_buf) {
        free(rhs_c);
        me_jit_set_error(ctx->error, line, column, "out of memory");
        return false;
    }
    snprintf(line_buf, need, "%s = (%s)(%s);", lhs, ctype, rhs_c);
    free(rhs_c);
    bool ok = me_jit_emit_line(&ctx->source, indent, line_buf);
    free(line_buf);
    if (!ok) {
        me_jit_set_error(ctx->error, line, column, "out of memory");
    }
    return ok;
}

static bool me_jit_emit_truthy_if_open(me_jit_codegen_ctx *ctx, int indent,
                                       const me_dsl_jit_ir_expr *cond,
                                       int line, int column) {
    char *cond_c = NULL;
    if (!me_jit_expr_to_c(cond, &cond_c, ctx->error, line, column)) {
        free(cond_c);
        return false;
    }
    const char *ctype = me_jit_c_type(cond->dtype);
    if (!ctype) {
        free(cond_c);
        me_jit_set_error(ctx->error, line, column, "unsupported condition dtype in jit c codegen");
        return false;
    }
    size_t need = strlen(cond_c) + strlen(ctype) + 48;
    char *line_buf = malloc(need);
    if (!line_buf) {
        free(cond_c);
        me_jit_set_error(ctx->error, line, column, "out of memory");
        return false;
    }
    snprintf(line_buf, need, "if (((%s)(%s)) != (%s)0) {", ctype, cond_c, ctype);
    free(cond_c);
    bool ok = me_jit_emit_line(&ctx->source, indent, line_buf);
    free(line_buf);
    if (!ok) {
        me_jit_set_error(ctx->error, line, column, "out of memory");
    }
    return ok;
}

static bool me_jit_emit_block(me_jit_codegen_ctx *ctx, const me_dsl_jit_ir_block *block, int indent);

static bool me_jit_emit_stmt(me_jit_codegen_ctx *ctx, const me_dsl_jit_ir_stmt *stmt, int indent) {
    if (!ctx || !stmt) {
        return false;
    }
    switch (stmt->kind) {
    case ME_DSL_JIT_IR_STMT_ASSIGN:
        return me_jit_emit_casted_expr_line(ctx, indent, stmt->as.assign.name, stmt->as.assign.dtype,
                                            &stmt->as.assign.value, stmt->line, stmt->column);
    case ME_DSL_JIT_IR_STMT_RETURN:
        if (!me_jit_emit_casted_expr_line(ctx, indent, ctx->out_var_name, ctx->output_dtype,
                                          &stmt->as.return_stmt.expr, stmt->line, stmt->column)) {
            return false;
        }
        if (!me_jit_emit_line(&ctx->source, indent, "goto __me_return_idx;")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        return true;
    case ME_DSL_JIT_IR_STMT_IF:
        if (!me_jit_emit_truthy_if_open(ctx, indent, &stmt->as.if_stmt.cond, stmt->line, stmt->column)) {
            return false;
        }
        if (!me_jit_emit_block(ctx, &stmt->as.if_stmt.then_block, indent + 1)) {
            return false;
        }
        if (!me_jit_emit_line(&ctx->source, indent, "}")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            const me_dsl_jit_ir_if_branch *branch = &stmt->as.if_stmt.elif_branches[i];
            char *cond_c = NULL;
            if (!me_jit_expr_to_c(&branch->cond, &cond_c, ctx->error, stmt->line, stmt->column)) {
                free(cond_c);
                return false;
            }
            const char *ctype = me_jit_c_type(branch->cond.dtype);
            if (!ctype) {
                free(cond_c);
                me_jit_set_error(ctx->error, stmt->line, stmt->column,
                                 "unsupported elif condition dtype in jit c codegen");
                return false;
            }
            size_t need = strlen(cond_c) + strlen(ctype) * 2 + 32;
            char *line_buf = malloc(need);
            if (!line_buf) {
                free(cond_c);
                me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                return false;
            }
            snprintf(line_buf, need, "else if (((%s)(%s)) != (%s)0) {", ctype, cond_c, ctype);
            free(cond_c);
            bool ok = me_jit_emit_line(&ctx->source, indent, line_buf);
            free(line_buf);
            if (!ok) {
                me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                return false;
            }
            if (!me_jit_emit_block(ctx, &branch->block, indent + 1)) {
                return false;
            }
            if (!me_jit_emit_line(&ctx->source, indent, "}")) {
                me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                return false;
            }
        }
        if (stmt->as.if_stmt.has_else) {
            if (!me_jit_emit_line(&ctx->source, indent, "else {")) {
                me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                return false;
            }
            if (!me_jit_emit_block(ctx, &stmt->as.if_stmt.else_block, indent + 1)) {
                return false;
            }
            if (!me_jit_emit_line(&ctx->source, indent, "}")) {
                me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
                return false;
            }
        }
        return true;
    case ME_DSL_JIT_IR_STMT_FOR: {
        char *limit_c = NULL;
        if (!me_jit_expr_to_c(&stmt->as.for_loop.limit, &limit_c, ctx->error, stmt->line, stmt->column)) {
            free(limit_c);
            return false;
        }
        size_t decl_need = strlen(stmt->as.for_loop.var) + strlen(limit_c) + 48;
        char *decl_line = malloc(decl_need);
        if (!decl_line) {
            free(limit_c);
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        snprintf(decl_line, decl_need, "int64_t __me_limit = (int64_t)(%s);", limit_c);
        free(limit_c);
        if (!me_jit_emit_line(&ctx->source, indent, decl_line)) {
            free(decl_line);
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        free(decl_line);
        if (!me_jit_emit_line(&ctx->source, indent, "if (__me_limit > 0) {")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        size_t for_need = strlen(stmt->as.for_loop.var) * 3 + 96;
        char *for_line = malloc(for_need);
        if (!for_line) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        snprintf(for_line, for_need, "for (%s = 0; %s < __me_limit; %s++) {",
                 stmt->as.for_loop.var, stmt->as.for_loop.var, stmt->as.for_loop.var);
        if (!me_jit_emit_line(&ctx->source, indent + 1, for_line)) {
            free(for_line);
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        free(for_line);
        if (!me_jit_emit_block(ctx, &stmt->as.for_loop.body, indent + 2)) {
            return false;
        }
        if (!me_jit_emit_line(&ctx->source, indent + 1, "}")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        if (!me_jit_emit_line(&ctx->source, indent, "}")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        return true;
    }
    case ME_DSL_JIT_IR_STMT_BREAK:
        if (!me_jit_emit_line(&ctx->source, indent, "break;")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        return true;
    case ME_DSL_JIT_IR_STMT_CONTINUE:
        if (!me_jit_emit_line(&ctx->source, indent, "continue;")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        return true;
    }
    me_jit_set_error(ctx->error, stmt->line, stmt->column, "unknown jit ir statement");
    return false;
}

static bool me_jit_emit_block(me_jit_codegen_ctx *ctx, const me_dsl_jit_ir_block *block, int indent) {
    if (!ctx || !block) {
        return false;
    }
    for (int i = 0; i < block->nstmts; i++) {
        if (!me_jit_emit_stmt(ctx, block->stmts[i], indent)) {
            return false;
        }
    }
    return true;
}

static bool me_jit_collect_return_dtype(const me_dsl_jit_ir_block *block,
                                        bool *has_return, me_dtype *out_dtype,
                                        me_dsl_error *error) {
    if (!block || !has_return || !out_dtype) {
        return false;
    }
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_jit_ir_stmt *stmt = block->stmts[i];
        if (!stmt) {
            continue;
        }
        switch (stmt->kind) {
        case ME_DSL_JIT_IR_STMT_RETURN:
            if (!*has_return) {
                *has_return = true;
                *out_dtype = stmt->as.return_stmt.expr.dtype;
            }
            else if (*out_dtype != stmt->as.return_stmt.expr.dtype) {
                me_jit_set_error(error, stmt->line, stmt->column,
                                 "mismatched return dtypes in jit ir");
                return false;
            }
            break;
        case ME_DSL_JIT_IR_STMT_IF:
            if (!me_jit_collect_return_dtype(&stmt->as.if_stmt.then_block, has_return, out_dtype, error)) {
                return false;
            }
            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                if (!me_jit_collect_return_dtype(&stmt->as.if_stmt.elif_branches[j].block,
                                                 has_return, out_dtype, error)) {
                    return false;
                }
            }
            if (stmt->as.if_stmt.has_else) {
                if (!me_jit_collect_return_dtype(&stmt->as.if_stmt.else_block, has_return, out_dtype, error)) {
                    return false;
                }
            }
            break;
        case ME_DSL_JIT_IR_STMT_FOR:
            if (!me_jit_collect_return_dtype(&stmt->as.for_loop.body, has_return, out_dtype, error)) {
                return false;
            }
            break;
        case ME_DSL_JIT_IR_STMT_ASSIGN:
        case ME_DSL_JIT_IR_STMT_BREAK:
        case ME_DSL_JIT_IR_STMT_CONTINUE:
            break;
        }
    }
    return true;
}

bool me_dsl_jit_codegen_c(const me_dsl_jit_ir_program *program, me_dtype output_dtype,
                          const me_dsl_jit_cgen_options *options,
                          char **out_source, me_dsl_error *error) {
    if (out_source) {
        *out_source = NULL;
    }
    if (!program || !out_source) {
        me_jit_set_error(error, 0, 0, "invalid arguments for jit c codegen");
        return false;
    }
    if (!me_jit_dtype_supported(output_dtype)) {
        me_jit_set_error(error, 0, 0, "unsupported output dtype for jit c codegen");
        return false;
    }
    for (int i = 0; i < program->nparams; i++) {
        if (!program->params[i] || !me_jit_dtype_supported(program->param_dtypes[i])) {
            me_jit_set_error(error, 0, 0, "invalid parameter metadata for jit c codegen");
            return false;
        }
    }

    bool has_return = false;
    me_dtype return_dtype = ME_AUTO;
    if (!me_jit_collect_return_dtype(&program->block, &has_return, &return_dtype, error)) {
        return false;
    }
    if (!has_return) {
        me_jit_set_error(error, 0, 0, "jit c codegen requires at least one return");
        return false;
    }
    if (return_dtype != output_dtype) {
        me_jit_set_error(error, 0, 0, "output dtype does not match jit ir return dtype");
        return false;
    }

    me_jit_codegen_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.output_dtype = output_dtype;
    ctx.out_var_name = "__me_out";
    ctx.error = error;

    if (!me_jit_collect_locals_block(&ctx.locals, &program->block)) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        return false;
    }
    for (int i = 0; i < program->nparams; i++) {
        if (me_jit_locals_find(&ctx.locals, program->params[i]) >= 0) {
            me_jit_set_error(error, 0, 0, "local name collides with parameter name in jit c codegen");
            me_jit_locals_free(&ctx.locals);
            return false;
        }
    }

    const char *symbol = "me_dsl_jit_kernel";
    if (options && options->symbol_name && options->symbol_name[0] != '\0') {
        symbol = options->symbol_name;
    }

    if (!me_jit_emit_line(&ctx.source, 0, "#include <stdbool.h>") ||
        !me_jit_emit_line(&ctx.source, 0, "#include <stddef.h>") ||
        !me_jit_emit_line(&ctx.source, 0, "#include <stdint.h>") ||
        !me_jit_emit_line(&ctx.source, 0, "")) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }

    size_t sig_need = strlen(symbol) + 96;
    char *sig = malloc(sig_need);
    if (!sig) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    snprintf(sig, sig_need, "int %s(const void **inputs, void *output, int64_t nitems) {", symbol);
    if (!me_jit_emit_line(&ctx.source, 0, sig)) {
        free(sig);
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    free(sig);

    if (!me_jit_emit_line(&ctx.source, 1, "if (!output || nitems < 0) {") ||
        !me_jit_emit_line(&ctx.source, 2, "return -1;") ||
        !me_jit_emit_line(&ctx.source, 1, "}")) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    if (program->nparams > 0) {
        if (!me_jit_emit_line(&ctx.source, 1, "if (!inputs) {") ||
            !me_jit_emit_line(&ctx.source, 2, "return -1;") ||
            !me_jit_emit_line(&ctx.source, 1, "}")) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
    }

    const char *out_ctype = me_jit_c_type(output_dtype);
    size_t out_need = strlen(out_ctype) + 32;
    char *out_decl = malloc(out_need);
    if (!out_decl) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    snprintf(out_decl, out_need, "%s *out = (%s *)output;", out_ctype, out_ctype);
    if (!me_jit_emit_line(&ctx.source, 1, out_decl)) {
        free(out_decl);
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    free(out_decl);

    for (int i = 0; i < program->nparams; i++) {
        const char *ptype = me_jit_c_type(program->param_dtypes[i]);
        size_t need = strlen(ptype) * 2 + strlen(program->params[i]) + 48;
        char *line = malloc(need);
        if (!line) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        snprintf(line, need, "const %s *in_%s = (const %s *)inputs[%d];",
                 ptype, program->params[i], ptype, i);
        bool ok = me_jit_emit_line(&ctx.source, 1, line);
        free(line);
        if (!ok) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
    }

    if (!me_jit_emit_line(&ctx.source, 1, "for (int64_t idx = 0; idx < nitems; idx++) {")) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }

    for (int i = 0; i < program->nparams; i++) {
        const char *ptype = me_jit_c_type(program->param_dtypes[i]);
        size_t need = strlen(ptype) + strlen(program->params[i]) * 2 + 32;
        char *line = malloc(need);
        if (!line) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        snprintf(line, need, "%s %s = in_%s[idx];", ptype, program->params[i], program->params[i]);
        bool ok = me_jit_emit_line(&ctx.source, 2, line);
        free(line);
        if (!ok) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
    }

    for (int i = 0; i < ctx.locals.count; i++) {
        const char *ltype = me_jit_c_type(ctx.locals.items[i].dtype);
        if (!ltype) {
            me_jit_set_error(error, 0, 0, "unsupported local dtype for jit c codegen");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        size_t need = strlen(ltype) + strlen(ctx.locals.items[i].name) + 24;
        char *line = malloc(need);
        if (!line) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        snprintf(line, need, "%s %s = (%s)0;", ltype, ctx.locals.items[i].name, ltype);
        bool ok = me_jit_emit_line(&ctx.source, 2, line);
        free(line);
        if (!ok) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
    }

    size_t ret_need = strlen(out_ctype) * 2 + 32;
    char *ret_line = malloc(ret_need);
    if (!ret_line) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    snprintf(ret_line, ret_need, "%s %s = (%s)0;", out_ctype, ctx.out_var_name, out_ctype);
    if (!me_jit_emit_line(&ctx.source, 2, ret_line)) {
        free(ret_line);
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    free(ret_line);

    if (!me_jit_emit_block(&ctx, &program->block, 2)) {
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }

    if (!me_jit_emit_line(&ctx.source, 2, "__me_return_idx:") ||
        !me_jit_emit_line(&ctx.source, 2, "out[idx] = __me_out;") ||
        !me_jit_emit_line(&ctx.source, 1, "}") ||
        !me_jit_emit_line(&ctx.source, 1, "return 0;") ||
        !me_jit_emit_line(&ctx.source, 0, "}")) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }

    me_jit_locals_free(&ctx.locals);
    *out_source = ctx.source.data;
    return true;
}
