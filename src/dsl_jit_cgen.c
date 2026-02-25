/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_cgen.h"
#include "dsl_jit_bridge_contract.h"

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
    bool use_runtime_math_bridge;
} me_jit_codegen_ctx;

typedef enum {
    ME_JIT_VEC_UNARY_NONE = 0,
    ME_JIT_VEC_UNARY_SIN,
    ME_JIT_VEC_UNARY_COS,
    ME_JIT_VEC_UNARY_EXP,
    ME_JIT_VEC_UNARY_LOG,
    ME_JIT_VEC_UNARY_EXP10,
    ME_JIT_VEC_UNARY_SINPI,
    ME_JIT_VEC_UNARY_COSPI,
    ME_JIT_VEC_UNARY_ABS,
    ME_JIT_VEC_UNARY_SQRT,
    ME_JIT_VEC_UNARY_LOG1P,
    ME_JIT_VEC_UNARY_EXP2,
    ME_JIT_VEC_UNARY_LOG2,
    ME_JIT_VEC_UNARY_EXPM1,
    ME_JIT_VEC_UNARY_LOG10,
    ME_JIT_VEC_UNARY_SINH,
    ME_JIT_VEC_UNARY_COSH,
    ME_JIT_VEC_UNARY_TANH,
    ME_JIT_VEC_UNARY_ASINH,
    ME_JIT_VEC_UNARY_ACOSH,
    ME_JIT_VEC_UNARY_ATANH
} me_jit_vec_unary_kind;

typedef struct {
    me_jit_vec_unary_kind kind;
    int param_index;
    bool has_offset;
    double offset;
} me_jit_vec_unary_plan;

typedef enum {
    ME_JIT_VEC_BINARY_NONE = 0,
    ME_JIT_VEC_BINARY_ATAN2,
    ME_JIT_VEC_BINARY_HYPOT,
    ME_JIT_VEC_BINARY_POW,
    ME_JIT_VEC_BINARY_FMAX,
    ME_JIT_VEC_BINARY_FMIN
} me_jit_vec_binary_kind;

typedef struct {
    me_jit_vec_binary_kind kind;
    int param_index_a;
    int param_index_b;
    bool arg_a_const;
    bool arg_b_const;
    double arg_a_const_value;
    double arg_b_const_value;
} me_jit_vec_binary_plan;

#define ME_JIT_BRIDGE_DECL_ENTRY(pub_sym, bridge_fn, sig_type, decl) decl,
static const char *const me_jit_runtime_bridge_extern_decls[] = {
    ME_DSL_JIT_BRIDGE_SYMBOL_CONTRACT(ME_JIT_BRIDGE_DECL_ENTRY)
};
#undef ME_JIT_BRIDGE_DECL_ENTRY
#undef ME_DSL_JIT_BRIDGE_SYMBOL_CONTRACT

static bool me_jit_emit_line(me_jit_strbuf *buf, int indent, const char *line);

static bool me_jit_emit_runtime_bridge_decls(me_jit_strbuf *buf) {
    for (size_t i = 0; i < (sizeof(me_jit_runtime_bridge_extern_decls) / sizeof(me_jit_runtime_bridge_extern_decls[0])); i++) {
        if (!me_jit_emit_line(buf, 0, me_jit_runtime_bridge_extern_decls[i])) {
            return false;
        }
    }
    if (!me_jit_emit_line(buf, 0, "")) {
        return false;
    }
    return true;
}

static void me_jit_set_error(me_dsl_error *error, int line, int column, const char *msg) {
    if (!error) {
        return;
    }
    error->line = line;
    error->column = column;
    snprintf(error->message, sizeof(error->message), "%s", msg ? msg : "jit c codegen failed");
}

static void me_jit_copy_text(char *dst, size_t dst_cap, const char *src) {
    if (!dst || dst_cap == 0) {
        return;
    }
    if (!src) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, dst_cap, "%s", src);
}

static void me_jit_set_lowering_trace(const me_dsl_jit_cgen_options *options,
                                      const char *mode,
                                      const char *ops,
                                      const char *reason) {
    if (!options) {
        return;
    }
    me_jit_copy_text(options->trace_lowering_mode, options->trace_lowering_mode_cap, mode);
    me_jit_copy_text(options->trace_vector_ops, options->trace_vector_ops_cap, ops);
    me_jit_copy_text(options->trace_lowering_reason, options->trace_lowering_reason_cap, reason);
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
    case ME_DSL_JIT_IR_STMT_WHILE:
        if (!me_jit_collect_locals_block(locals, &stmt->as.while_loop.body)) {
            return false;
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

static bool me_jit_ident_equals(const char *start, size_t ident_len, const char *name) {
    if (!start || !name) {
        return false;
    }
    size_t name_len = strlen(name);
    return ident_len == name_len && strncmp(start, name, ident_len) == 0;
}

static const char *me_jit_function_name_rewrite(const char *start, size_t ident_len,
                                                bool use_runtime_math_bridge) {
    if (me_jit_ident_equals(start, ident_len, "int")) {
        return "ME_DSL_CAST_INT";
    }
    if (me_jit_ident_equals(start, ident_len, "float")) {
        return "ME_DSL_CAST_FLOAT";
    }
    if (me_jit_ident_equals(start, ident_len, "bool")) {
        return "ME_DSL_CAST_BOOL";
    }
    if (me_jit_ident_equals(start, ident_len, "arctan2")) {
        return "atan2";
    }
    if (me_jit_ident_equals(start, ident_len, "abs")) {
#if defined(__EMSCRIPTEN__)
        return "fabs";
#else
        return use_runtime_math_bridge ? "me_jit_abs" : "fabs";
#endif
    }
#if !defined(__EMSCRIPTEN__)
    if (use_runtime_math_bridge && me_jit_ident_equals(start, ident_len, "sin")) {
        return "me_jit_sin";
    }
    if (use_runtime_math_bridge && me_jit_ident_equals(start, ident_len, "cos")) {
        return "me_jit_cos";
    }
    if (use_runtime_math_bridge && me_jit_ident_equals(start, ident_len, "exp")) {
        return "me_jit_exp";
    }
    if (use_runtime_math_bridge && me_jit_ident_equals(start, ident_len, "log")) {
        return "me_jit_log";
    }
    if (use_runtime_math_bridge && me_jit_ident_equals(start, ident_len, "sqrt")) {
        return "me_jit_sqrt";
    }
#endif
    if (me_jit_ident_equals(start, ident_len, "exp10")) {
        return "me_jit_exp10";
    }
    if (me_jit_ident_equals(start, ident_len, "sinpi")) {
        return "me_jit_sinpi";
    }
    if (me_jit_ident_equals(start, ident_len, "cospi")) {
        return "me_jit_cospi";
    }
    if (me_jit_ident_equals(start, ident_len, "logaddexp")) {
        return "me_jit_logaddexp";
    }
    if (me_jit_ident_equals(start, ident_len, "where")) {
        return "me_jit_where";
    }
    return NULL;
}

static const char *me_jit_skip_ws(const char *p) {
    if (!p) {
        return NULL;
    }
    while (*p && isspace((unsigned char)*p)) {
        p++;
    }
    return p;
}

static bool me_jit_parse_ident(const char **pp, const char **out_start, size_t *out_len) {
    if (!pp || !*pp || !out_start || !out_len) {
        return false;
    }
    const char *p = *pp;
    if (!(isalpha((unsigned char)*p) || *p == '_')) {
        return false;
    }
    const char *start = p++;
    while (isalnum((unsigned char)*p) || *p == '_') {
        p++;
    }
    *out_start = start;
    *out_len = (size_t)(p - start);
    *pp = p;
    return true;
}

static bool me_jit_parse_number(const char **pp, const char **out_start, size_t *out_len,
                                double *out_value) {
    if (!pp || !*pp || !out_start || !out_len || !out_value) {
        return false;
    }
    const char *p = *pp;
    char *end_num = NULL;
    double v = strtod(p, &end_num);
    if (!end_num || end_num == p) {
        return false;
    }
    *out_start = p;
    *out_len = (size_t)(end_num - p);
    *out_value = v;
    *pp = end_num;
    return true;
}

static bool me_jit_parse_ident_or_number(const char **pp,
                                         const char **out_start, size_t *out_len,
                                         bool *out_is_number, double *out_number) {
    if (!pp || !*pp || !out_start || !out_len || !out_is_number || !out_number) {
        return false;
    }
    const char *p = *pp;
    const char *start = NULL;
    size_t len = 0;
    if (me_jit_parse_ident(&p, &start, &len)) {
        *out_start = start;
        *out_len = len;
        *out_is_number = false;
        *out_number = 0.0;
        *pp = p;
        return true;
    }
    double number = 0.0;
    if (!me_jit_parse_number(&p, &start, &len, &number)) {
        return false;
    }
    *out_start = start;
    *out_len = len;
    *out_is_number = true;
    *out_number = number;
    *pp = p;
    return true;
}

static bool me_jit_parse_simple_unary_call(const char *expr,
                                           const char **out_fn_start, size_t *out_fn_len,
                                           const char **out_arg_start, size_t *out_arg_len,
                                           bool *out_has_offset, double *out_offset) {
    if (!expr || !out_fn_start || !out_fn_len || !out_arg_start || !out_arg_len ||
        !out_has_offset || !out_offset) {
        return false;
    }
    const char *p = me_jit_skip_ws(expr);
    const char *fn_start = NULL;
    size_t fn_len = 0;
    if (!me_jit_parse_ident(&p, &fn_start, &fn_len)) {
        return false;
    }
    p = me_jit_skip_ws(p);
    if (!p || *p != '(') {
        return false;
    }
    p++;
    const char *arg_expr = me_jit_skip_ws(p);
    const char *rparen = strrchr(arg_expr, ')');
    if (!rparen || rparen <= arg_expr) {
        return false;
    }
    const char *tail = me_jit_skip_ws(rparen + 1);
    if (!tail || *tail != '\0') {
        return false;
    }

    char *arg_copy = malloc((size_t)(rparen - arg_expr) + 1);
    if (!arg_copy) {
        return false;
    }
    memcpy(arg_copy, arg_expr, (size_t)(rparen - arg_expr));
    arg_copy[rparen - arg_expr] = '\0';

    const char *arg_start = NULL;
    size_t arg_len = 0;
    bool has_offset = false;
    double offset = 0.0;

    const char *q = me_jit_skip_ws(arg_copy);
    const char *id0_start = NULL;
    size_t id0_len = 0;
    if (me_jit_parse_ident(&q, &id0_start, &id0_len)) {
        q = me_jit_skip_ws(q);
        if (!q || *q == '\0') {
            arg_start = arg_expr + (id0_start - arg_copy);
            arg_len = id0_len;
        }
        else if (*q == '+' || *q == '-') {
            char op = *q++;
            q = me_jit_skip_ws(q);
            if (!q || *q == '\0') {
                free(arg_copy);
                return false;
            }
            char *end_num = NULL;
            double c = strtod(q, &end_num);
            if (!end_num) {
                free(arg_copy);
                return false;
            }
            end_num = (char *)me_jit_skip_ws(end_num);
            if (!end_num || *end_num != '\0') {
                free(arg_copy);
                return false;
            }
            arg_start = arg_expr + (id0_start - arg_copy);
            arg_len = id0_len;
            has_offset = true;
            offset = (op == '-') ? -c : c;
        }
        else {
            free(arg_copy);
            return false;
        }
    }
    else {
        char *end_num = NULL;
        double c = strtod(q, &end_num);
        if (!end_num) {
            free(arg_copy);
            return false;
        }
        q = me_jit_skip_ws(end_num);
        if (!q || *q != '+') {
            free(arg_copy);
            return false;
        }
        q++;
        q = me_jit_skip_ws(q);
        const char *id1_start = NULL;
        size_t id1_len = 0;
        if (!me_jit_parse_ident(&q, &id1_start, &id1_len)) {
            free(arg_copy);
            return false;
        }
        q = me_jit_skip_ws(q);
        if (!q || *q != '\0') {
            free(arg_copy);
            return false;
        }
        arg_start = arg_expr + (id1_start - arg_copy);
        arg_len = id1_len;
        has_offset = true;
        offset = c;
    }

    free(arg_copy);
    if (!arg_start || arg_len == 0) {
        return false;
    }

    *out_fn_start = fn_start;
    *out_fn_len = fn_len;
    *out_arg_start = arg_start;
    *out_arg_len = arg_len;
    *out_has_offset = has_offset;
    *out_offset = offset;
    return true;
}

static bool me_jit_parse_simple_binary_call(const char *expr,
                                            const char **out_fn_start, size_t *out_fn_len,
                                            const char **out_arg0_start, size_t *out_arg0_len,
                                            const char **out_arg1_start, size_t *out_arg1_len,
                                            bool *out_arg0_is_const, double *out_arg0_const,
                                            bool *out_arg1_is_const, double *out_arg1_const) {
    if (!expr || !out_fn_start || !out_fn_len ||
        !out_arg0_start || !out_arg0_len || !out_arg1_start || !out_arg1_len ||
        !out_arg0_is_const || !out_arg0_const || !out_arg1_is_const || !out_arg1_const) {
        return false;
    }
    const char *p = me_jit_skip_ws(expr);
    const char *fn_start = NULL;
    size_t fn_len = 0;
    if (!me_jit_parse_ident(&p, &fn_start, &fn_len)) {
        return false;
    }
    p = me_jit_skip_ws(p);
    if (!p || *p != '(') {
        return false;
    }
    p++;
    p = me_jit_skip_ws(p);
    const char *arg0_start = NULL;
    size_t arg0_len = 0;
    bool arg0_is_const = false;
    double arg0_const = 0.0;
    if (!me_jit_parse_ident_or_number(&p, &arg0_start, &arg0_len,
                                      &arg0_is_const, &arg0_const)) {
        return false;
    }
    p = me_jit_skip_ws(p);
    if (!p || *p != ',') {
        return false;
    }
    p++;
    p = me_jit_skip_ws(p);
    const char *arg1_start = NULL;
    size_t arg1_len = 0;
    bool arg1_is_const = false;
    double arg1_const = 0.0;
    if (!me_jit_parse_ident_or_number(&p, &arg1_start, &arg1_len,
                                      &arg1_is_const, &arg1_const)) {
        return false;
    }
    p = me_jit_skip_ws(p);
    if (!p || *p != ')') {
        return false;
    }
    p++;
    p = me_jit_skip_ws(p);
    if (!p || *p != '\0') {
        return false;
    }
    *out_fn_start = fn_start;
    *out_fn_len = fn_len;
    *out_arg0_start = arg0_start;
    *out_arg0_len = arg0_len;
    *out_arg1_start = arg1_start;
    *out_arg1_len = arg1_len;
    *out_arg0_is_const = arg0_is_const;
    *out_arg0_const = arg0_const;
    *out_arg1_is_const = arg1_is_const;
    *out_arg1_const = arg1_const;
    return true;
}

static bool me_jit_expr_is_single_ident(const char *expr,
                                        const char *name) {
    if (!expr || !name || name[0] == '\0') {
        return false;
    }
    const char *p = me_jit_skip_ws(expr);
    const char *ident_start = NULL;
    size_t ident_len = 0;
    if (!me_jit_parse_ident(&p, &ident_start, &ident_len)) {
        return false;
    }
    p = me_jit_skip_ws(p);
    if (!p || *p != '\0') {
        return false;
    }
    return me_jit_ident_equals(ident_start, ident_len, name);
}

static const me_dsl_jit_ir_expr *me_jit_vector_candidate_expr(const me_dsl_jit_ir_program *program,
                                                              me_dtype output_dtype) {
    if (!program) {
        return NULL;
    }
    if (output_dtype != ME_FLOAT64 && output_dtype != ME_FLOAT32) {
        return NULL;
    }
    if (program->block.nstmts == 1) {
        const me_dsl_jit_ir_stmt *stmt = program->block.stmts[0];
        if (!stmt || stmt->kind != ME_DSL_JIT_IR_STMT_RETURN) {
            return NULL;
        }
        if (stmt->as.return_stmt.expr.dtype != output_dtype ||
            !stmt->as.return_stmt.expr.text) {
            return NULL;
        }
        return &stmt->as.return_stmt.expr;
    }
    if (program->block.nstmts == 2) {
        const me_dsl_jit_ir_stmt *assign_stmt = program->block.stmts[0];
        const me_dsl_jit_ir_stmt *ret_stmt = program->block.stmts[1];
        if (!assign_stmt || !ret_stmt ||
            assign_stmt->kind != ME_DSL_JIT_IR_STMT_ASSIGN ||
            ret_stmt->kind != ME_DSL_JIT_IR_STMT_RETURN) {
            return NULL;
        }
        if (assign_stmt->as.assign.dtype != output_dtype ||
            assign_stmt->as.assign.value.dtype != output_dtype ||
            !assign_stmt->as.assign.name ||
            !assign_stmt->as.assign.value.text ||
            !ret_stmt->as.return_stmt.expr.text ||
            ret_stmt->as.return_stmt.expr.dtype != output_dtype) {
            return NULL;
        }
        if (!me_jit_expr_is_single_ident(ret_stmt->as.return_stmt.expr.text,
                                         assign_stmt->as.assign.name)) {
            return NULL;
        }
        return &assign_stmt->as.assign.value;
    }
    return NULL;
}

static bool me_jit_detect_vec_unary_plan(const me_dsl_jit_ir_program *program,
                                         me_dtype output_dtype,
                                         me_jit_vec_unary_plan *out_plan) {
    if (!program || !out_plan) {
        return false;
    }
    out_plan->kind = ME_JIT_VEC_UNARY_NONE;
    out_plan->param_index = -1;
    out_plan->has_offset = false;
    out_plan->offset = 0.0;

    const me_dsl_jit_ir_expr *expr = me_jit_vector_candidate_expr(program, output_dtype);
    if (!expr || !expr->text || expr->dtype != output_dtype) {
        return false;
    }

    const char *fn_start = NULL;
    const char *arg_start = NULL;
    size_t fn_len = 0;
    size_t arg_len = 0;
    bool has_offset = false;
    double offset = 0.0;
    if (!me_jit_parse_simple_unary_call(expr->text, &fn_start, &fn_len,
                                        &arg_start, &arg_len,
                                        &has_offset, &offset)) {
        return false;
    }

    if (me_jit_ident_equals(fn_start, fn_len, "sin")) {
        out_plan->kind = ME_JIT_VEC_UNARY_SIN;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "cos")) {
        out_plan->kind = ME_JIT_VEC_UNARY_COS;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "exp")) {
        out_plan->kind = ME_JIT_VEC_UNARY_EXP;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "log")) {
        out_plan->kind = ME_JIT_VEC_UNARY_LOG;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "sinpi")) {
        out_plan->kind = ME_JIT_VEC_UNARY_SINPI;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "cospi")) {
        out_plan->kind = ME_JIT_VEC_UNARY_COSPI;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "exp10")) {
        out_plan->kind = ME_JIT_VEC_UNARY_EXP10;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "abs")) {
        out_plan->kind = ME_JIT_VEC_UNARY_ABS;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "sqrt")) {
        out_plan->kind = ME_JIT_VEC_UNARY_SQRT;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "log1p")) {
        out_plan->kind = ME_JIT_VEC_UNARY_LOG1P;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "exp2")) {
        out_plan->kind = ME_JIT_VEC_UNARY_EXP2;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "log2")) {
        out_plan->kind = ME_JIT_VEC_UNARY_LOG2;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "expm1")) {
        out_plan->kind = ME_JIT_VEC_UNARY_EXPM1;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "log10")) {
        out_plan->kind = ME_JIT_VEC_UNARY_LOG10;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "sinh")) {
        out_plan->kind = ME_JIT_VEC_UNARY_SINH;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "cosh")) {
        out_plan->kind = ME_JIT_VEC_UNARY_COSH;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "tanh")) {
        out_plan->kind = ME_JIT_VEC_UNARY_TANH;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "asinh")) {
        out_plan->kind = ME_JIT_VEC_UNARY_ASINH;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "acosh")) {
        out_plan->kind = ME_JIT_VEC_UNARY_ACOSH;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "atanh")) {
        out_plan->kind = ME_JIT_VEC_UNARY_ATANH;
    }
    else {
        return false;
    }

    for (int i = 0; i < program->nparams; i++) {
        if (program->params[i] && program->param_dtypes[i] == output_dtype &&
            me_jit_ident_equals(arg_start, arg_len, program->params[i])) {
            out_plan->param_index = i;
            out_plan->has_offset = has_offset;
            out_plan->offset = offset;
            return true;
        }
    }
    out_plan->kind = ME_JIT_VEC_UNARY_NONE;
    out_plan->param_index = -1;
    out_plan->has_offset = false;
    out_plan->offset = 0.0;
    return false;
}

static bool me_jit_detect_vec_binary_plan(const me_dsl_jit_ir_program *program,
                                          me_dtype output_dtype,
                                          me_jit_vec_binary_plan *out_plan) {
    if (!program || !out_plan) {
        return false;
    }
    out_plan->kind = ME_JIT_VEC_BINARY_NONE;
    out_plan->param_index_a = -1;
    out_plan->param_index_b = -1;
    out_plan->arg_a_const = false;
    out_plan->arg_b_const = false;
    out_plan->arg_a_const_value = 0.0;
    out_plan->arg_b_const_value = 0.0;

    const me_dsl_jit_ir_expr *expr = me_jit_vector_candidate_expr(program, output_dtype);
    if (!expr || !expr->text || expr->dtype != output_dtype) {
        return false;
    }

    const char *fn_start = NULL;
    const char *arg0_start = NULL;
    const char *arg1_start = NULL;
    size_t fn_len = 0;
    size_t arg0_len = 0;
    size_t arg1_len = 0;
    bool arg0_is_const = false;
    bool arg1_is_const = false;
    double arg0_const = 0.0;
    double arg1_const = 0.0;
    if (!me_jit_parse_simple_binary_call(expr->text, &fn_start, &fn_len,
                                         &arg0_start, &arg0_len, &arg1_start, &arg1_len,
                                         &arg0_is_const, &arg0_const,
                                         &arg1_is_const, &arg1_const)) {
        return false;
    }

    if (me_jit_ident_equals(fn_start, fn_len, "atan2")) {
        out_plan->kind = ME_JIT_VEC_BINARY_ATAN2;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "hypot")) {
        out_plan->kind = ME_JIT_VEC_BINARY_HYPOT;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "pow")) {
        out_plan->kind = ME_JIT_VEC_BINARY_POW;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "fmax")) {
        out_plan->kind = ME_JIT_VEC_BINARY_FMAX;
    }
    else if (me_jit_ident_equals(fn_start, fn_len, "fmin")) {
        out_plan->kind = ME_JIT_VEC_BINARY_FMIN;
    }
    else {
        return false;
    }

    for (int i = 0; i < program->nparams; i++) {
        if (program->params[i] && program->param_dtypes[i] == output_dtype &&
            me_jit_ident_equals(arg0_start, arg0_len, program->params[i])) {
            out_plan->param_index_a = i;
            break;
        }
    }
    for (int i = 0; i < program->nparams; i++) {
        if (program->params[i] && program->param_dtypes[i] == output_dtype &&
            me_jit_ident_equals(arg1_start, arg1_len, program->params[i])) {
            out_plan->param_index_b = i;
            break;
        }
    }
    if (out_plan->param_index_a < 0 && arg0_is_const) {
        out_plan->arg_a_const = true;
        out_plan->arg_a_const_value = arg0_const;
    }
    if (out_plan->param_index_b < 0 && arg1_is_const) {
        out_plan->arg_b_const = true;
        out_plan->arg_b_const_value = arg1_const;
    }
    if ((out_plan->param_index_a >= 0 || out_plan->arg_a_const) &&
        (out_plan->param_index_b >= 0 || out_plan->arg_b_const) &&
        !(out_plan->arg_a_const && out_plan->arg_b_const)) {
        return true;
    }
    out_plan->kind = ME_JIT_VEC_BINARY_NONE;
    out_plan->param_index_a = -1;
    out_plan->param_index_b = -1;
    out_plan->arg_a_const = false;
    out_plan->arg_b_const = false;
    out_plan->arg_a_const_value = 0.0;
    out_plan->arg_b_const_value = 0.0;
    return false;
}

static const char *me_jit_vec_unary_symbol(me_jit_vec_unary_kind kind, me_dtype dtype) {
    switch (kind) {
    case ME_JIT_VEC_UNARY_SIN:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_sin_f64" : "me_jit_vec_sin_f32";
    case ME_JIT_VEC_UNARY_COS:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_cos_f64" : "me_jit_vec_cos_f32";
    case ME_JIT_VEC_UNARY_EXP:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_exp_f64" : "me_jit_vec_exp_f32";
    case ME_JIT_VEC_UNARY_LOG:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_log_f64" : "me_jit_vec_log_f32";
    case ME_JIT_VEC_UNARY_EXP10:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_exp10_f64" : "me_jit_vec_exp10_f32";
    case ME_JIT_VEC_UNARY_SINPI:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_sinpi_f64" : "me_jit_vec_sinpi_f32";
    case ME_JIT_VEC_UNARY_COSPI:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_cospi_f64" : "me_jit_vec_cospi_f32";
    case ME_JIT_VEC_UNARY_ABS:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_abs_f64" : "me_jit_vec_abs_f32";
    case ME_JIT_VEC_UNARY_SQRT:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_sqrt_f64" : "me_jit_vec_sqrt_f32";
    case ME_JIT_VEC_UNARY_LOG1P:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_log1p_f64" : "me_jit_vec_log1p_f32";
    case ME_JIT_VEC_UNARY_EXP2:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_exp2_f64" : "me_jit_vec_exp2_f32";
    case ME_JIT_VEC_UNARY_LOG2:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_log2_f64" : "me_jit_vec_log2_f32";
    case ME_JIT_VEC_UNARY_EXPM1:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_expm1_f64" : "me_jit_vec_expm1_f32";
    case ME_JIT_VEC_UNARY_LOG10:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_log10_f64" : "me_jit_vec_log10_f32";
    case ME_JIT_VEC_UNARY_SINH:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_sinh_f64" : "me_jit_vec_sinh_f32";
    case ME_JIT_VEC_UNARY_COSH:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_cosh_f64" : "me_jit_vec_cosh_f32";
    case ME_JIT_VEC_UNARY_TANH:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_tanh_f64" : "me_jit_vec_tanh_f32";
    case ME_JIT_VEC_UNARY_ASINH:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_asinh_f64" : "me_jit_vec_asinh_f32";
    case ME_JIT_VEC_UNARY_ACOSH:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_acosh_f64" : "me_jit_vec_acosh_f32";
    case ME_JIT_VEC_UNARY_ATANH:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_atanh_f64" : "me_jit_vec_atanh_f32";
    case ME_JIT_VEC_UNARY_NONE:
        return NULL;
    }
    return NULL;
}

static const char *me_jit_vec_binary_symbol(me_jit_vec_binary_kind kind, me_dtype dtype) {
    switch (kind) {
    case ME_JIT_VEC_BINARY_ATAN2:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_atan2_f64" : "me_jit_vec_atan2_f32";
    case ME_JIT_VEC_BINARY_HYPOT:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_hypot_f64" : "me_jit_vec_hypot_f32";
    case ME_JIT_VEC_BINARY_POW:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_pow_f64" : "me_jit_vec_pow_f32";
    case ME_JIT_VEC_BINARY_FMAX:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_fmax_f64" : "me_jit_vec_fmax_f32";
    case ME_JIT_VEC_BINARY_FMIN:
        return (dtype == ME_FLOAT64) ? "me_jit_vec_fmin_f64" : "me_jit_vec_fmin_f32";
    case ME_JIT_VEC_BINARY_NONE:
        return NULL;
    }
    return NULL;
}

static const char *me_jit_vec_unary_op_name(me_jit_vec_unary_kind kind) {
    switch (kind) {
    case ME_JIT_VEC_UNARY_SIN: return "sin";
    case ME_JIT_VEC_UNARY_COS: return "cos";
    case ME_JIT_VEC_UNARY_EXP: return "exp";
    case ME_JIT_VEC_UNARY_LOG: return "log";
    case ME_JIT_VEC_UNARY_EXP10: return "exp10";
    case ME_JIT_VEC_UNARY_SINPI: return "sinpi";
    case ME_JIT_VEC_UNARY_COSPI: return "cospi";
    case ME_JIT_VEC_UNARY_ABS: return "abs";
    case ME_JIT_VEC_UNARY_SQRT: return "sqrt";
    case ME_JIT_VEC_UNARY_LOG1P: return "log1p";
    case ME_JIT_VEC_UNARY_EXP2: return "exp2";
    case ME_JIT_VEC_UNARY_LOG2: return "log2";
    case ME_JIT_VEC_UNARY_EXPM1: return "expm1";
    case ME_JIT_VEC_UNARY_LOG10: return "log10";
    case ME_JIT_VEC_UNARY_SINH: return "sinh";
    case ME_JIT_VEC_UNARY_COSH: return "cosh";
    case ME_JIT_VEC_UNARY_TANH: return "tanh";
    case ME_JIT_VEC_UNARY_ASINH: return "asinh";
    case ME_JIT_VEC_UNARY_ACOSH: return "acosh";
    case ME_JIT_VEC_UNARY_ATANH: return "atanh";
    case ME_JIT_VEC_UNARY_NONE: return "";
    }
    return "";
}

static const char *me_jit_vec_binary_op_name(me_jit_vec_binary_kind kind) {
    switch (kind) {
    case ME_JIT_VEC_BINARY_ATAN2: return "atan2";
    case ME_JIT_VEC_BINARY_HYPOT: return "hypot";
    case ME_JIT_VEC_BINARY_POW: return "pow";
    case ME_JIT_VEC_BINARY_FMAX: return "fmax";
    case ME_JIT_VEC_BINARY_FMIN: return "fmin";
    case ME_JIT_VEC_BINARY_NONE: return "";
    }
    return "";
}

static bool me_jit_emit_vec_unary_call(me_jit_codegen_ctx *ctx,
                                       me_dtype output_dtype,
                                       const char *vec_sym,
                                       const char *param_name,
                                       bool has_offset,
                                       double offset) {
    if (!ctx || !vec_sym || !param_name) {
        return false;
    }
    if (!has_offset) {
        size_t need = strlen(vec_sym) + strlen(param_name) + 32;
        char *line = malloc(need);
        if (!line) {
            return false;
        }
        snprintf(line, need, "%s(in_%s, out, nitems);", vec_sym, param_name);
        bool ok = me_jit_emit_line(&ctx->source, 1, line);
        free(line);
        return ok;
    }

    const char *ctype = (output_dtype == ME_FLOAT32) ? "float" : "double";
    char offset_buf[96];
    snprintf(offset_buf, sizeof(offset_buf), "%.17g", offset);

    if (!me_jit_emit_line(&ctx->source, 1, "for (int64_t __me_i = 0; __me_i < nitems; __me_i++) {")) {
        return false;
    }
    size_t prep_need = strlen(ctype) * 2 + strlen(param_name) + strlen(offset_buf) + 64;
    char *prep_line = malloc(prep_need);
    if (!prep_line) {
        return false;
    }
    snprintf(prep_line, prep_need,
             "out[__me_i] = (%s)(in_%s[__me_i] + (%s)%s);",
             ctype, param_name, ctype, offset_buf);
    bool ok = me_jit_emit_line(&ctx->source, 2, prep_line);
    free(prep_line);
    if (!ok ||
        !me_jit_emit_line(&ctx->source, 1, "}")) {
        return false;
    }
    size_t call_need = strlen(vec_sym) + 64;
    char *call_line = malloc(call_need);
    if (!call_line) {
        return false;
    }
    snprintf(call_line, call_need, "%s(out, out, nitems);", vec_sym);
    ok = me_jit_emit_line(&ctx->source, 1, call_line);
    free(call_line);
    if (!ok) {
        return false;
    }
    return true;
}

static bool me_jit_emit_vec_binary_call(me_jit_codegen_ctx *ctx,
                                        me_dtype output_dtype,
                                        const char *vec_sym,
                                        const char *arg_a_param,
                                        bool arg_a_const, double arg_a_const_value,
                                        const char *arg_b_param,
                                        bool arg_b_const, double arg_b_const_value) {
    if (!ctx || !vec_sym) {
        return false;
    }
    if (arg_a_const && arg_b_const) {
        return false;
    }
    const char *ctype = (output_dtype == ME_FLOAT32) ? "float" : "double";
    if (!arg_a_const && !arg_b_const) {
        if (!arg_a_param || !arg_b_param) {
            return false;
        }
        size_t need = strlen(vec_sym) + strlen(arg_a_param) + strlen(arg_b_param) + 40;
        char *line = malloc(need);
        if (!line) {
            return false;
        }
        snprintf(line, need, "%s(in_%s, in_%s, out, nitems);",
                 vec_sym, arg_a_param, arg_b_param);
        bool ok = me_jit_emit_line(&ctx->source, 1, line);
        free(line);
        return ok;
    }

    char const_buf[96];
    double fill_value = arg_a_const ? arg_a_const_value : arg_b_const_value;
    snprintf(const_buf, sizeof(const_buf), "%.17g", fill_value);
    if (!me_jit_emit_line(&ctx->source, 1, "for (int64_t __me_i = 0; __me_i < nitems; __me_i++) {")) {
        return false;
    }
    size_t fill_need = strlen(ctype) + strlen(const_buf) + 48;
    char *fill_line = malloc(fill_need);
    if (!fill_line) {
        return false;
    }
    snprintf(fill_line, fill_need, "out[__me_i] = (%s)%s;", ctype, const_buf);
    bool ok = me_jit_emit_line(&ctx->source, 2, fill_line);
    free(fill_line);
    if (!ok ||
        !me_jit_emit_line(&ctx->source, 1, "}")) {
        return false;
    }

    if (arg_a_const) {
        if (!arg_b_param) {
            return false;
        }
        size_t need = strlen(vec_sym) + strlen(arg_b_param) + 32;
        char *line = malloc(need);
        if (!line) {
            return false;
        }
        snprintf(line, need, "%s(out, in_%s, out, nitems);", vec_sym, arg_b_param);
        ok = me_jit_emit_line(&ctx->source, 1, line);
        free(line);
        return ok;
    }

    if (!arg_a_param) {
        return false;
    }
    size_t need = strlen(vec_sym) + strlen(arg_a_param) + 32;
    char *line = malloc(need);
    if (!line) {
        return false;
    }
    snprintf(line, need, "%s(in_%s, out, out, nitems);", vec_sym, arg_a_param);
    ok = me_jit_emit_line(&ctx->source, 1, line);
    free(line);
    return ok;
}

static bool me_jit_expr_to_c(const me_dsl_jit_ir_expr *expr, char **out_c,
                             me_dsl_error *error, int line, int column,
                             bool use_runtime_math_bridge) {
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
            else {
                const char *q = p;
                while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n') {
                    q++;
                }
                if (*q == '(') {
                    rep = me_jit_function_name_rewrite(start, ident_len, use_runtime_math_bridge);
                }
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
    if (!me_jit_expr_to_c(rhs, &rhs_c, ctx->error, line, column, ctx->use_runtime_math_bridge)) {
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
    if (!me_jit_expr_to_c(cond, &cond_c, ctx->error, line, column, ctx->use_runtime_math_bridge)) {
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
            if (!me_jit_expr_to_c(&branch->cond, &cond_c, ctx->error, stmt->line, stmt->column,
                                  ctx->use_runtime_math_bridge)) {
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
    case ME_DSL_JIT_IR_STMT_WHILE: {
        char *cond_c = NULL;
        if (!me_jit_expr_to_c(&stmt->as.while_loop.cond, &cond_c, ctx->error, stmt->line, stmt->column,
                              ctx->use_runtime_math_bridge)) {
            free(cond_c);
            return false;
        }
        const char *ctype = me_jit_c_type(stmt->as.while_loop.cond.dtype);
        if (!ctype) {
            free(cond_c);
            me_jit_set_error(ctx->error, stmt->line, stmt->column,
                             "unsupported while condition dtype in jit c codegen");
            return false;
        }
        size_t need = strlen(cond_c) + strlen(ctype) * 2 + 30;
        char *line_buf = malloc(need);
        if (!line_buf) {
            free(cond_c);
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        snprintf(line_buf, need, "while (((%s)(%s)) != (%s)0) {", ctype, cond_c, ctype);
        free(cond_c);
        bool ok = me_jit_emit_line(&ctx->source, indent, line_buf);
        free(line_buf);
        if (!ok) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        if (!me_jit_emit_block(ctx, &stmt->as.while_loop.body, indent + 1)) {
            return false;
        }
        if (!me_jit_emit_line(&ctx->source, indent, "}")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        return true;
    }
    case ME_DSL_JIT_IR_STMT_FOR: {
        char *start_c = NULL;
        char *stop_c = NULL;
        char *step_c = NULL;
        if (!me_jit_expr_to_c(&stmt->as.for_loop.start, &start_c, ctx->error, stmt->line, stmt->column,
                              ctx->use_runtime_math_bridge) ||
            !me_jit_expr_to_c(&stmt->as.for_loop.stop, &stop_c, ctx->error, stmt->line, stmt->column,
                              ctx->use_runtime_math_bridge) ||
            !me_jit_expr_to_c(&stmt->as.for_loop.step, &step_c, ctx->error, stmt->line, stmt->column,
                              ctx->use_runtime_math_bridge)) {
            free(start_c);
            free(stop_c);
            free(step_c);
            return false;
        }

        size_t start_need = strlen(start_c) + 48;
        size_t stop_need = strlen(stop_c) + 46;
        size_t step_need = strlen(step_c) + 46;
        char *start_line = malloc(start_need);
        char *stop_line = malloc(stop_need);
        char *step_line = malloc(step_need);
        if (!start_line || !stop_line || !step_line) {
            free(start_c);
            free(stop_c);
            free(step_c);
            free(start_line);
            free(stop_line);
            free(step_line);
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        snprintf(start_line, start_need, "int64_t __me_start = (int64_t)(%s);", start_c);
        snprintf(stop_line, stop_need, "int64_t __me_stop = (int64_t)(%s);", stop_c);
        snprintf(step_line, step_need, "int64_t __me_step = (int64_t)(%s);", step_c);
        free(start_c);
        free(stop_c);
        free(step_c);

        if (!me_jit_emit_line(&ctx->source, indent, start_line) ||
            !me_jit_emit_line(&ctx->source, indent, stop_line) ||
            !me_jit_emit_line(&ctx->source, indent, step_line)) {
            free(start_line);
            free(stop_line);
            free(step_line);
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        free(start_line);
        free(stop_line);
        free(step_line);

        if (!me_jit_emit_line(&ctx->source, indent, "if (__me_step == 0) {")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        if (!me_jit_emit_line(&ctx->source, indent + 1, "return 1;")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        if (!me_jit_emit_line(&ctx->source, indent, "}")) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        size_t for_need = strlen(stmt->as.for_loop.var) * 4 + 160;
        char *for_line = malloc(for_need);
        if (!for_line) {
            me_jit_set_error(ctx->error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        snprintf(for_line, for_need,
                 "for (%s = __me_start; ((__me_step > 0) ? (%s < __me_stop) : (%s > __me_stop)); %s += __me_step) {",
                 stmt->as.for_loop.var, stmt->as.for_loop.var,
                 stmt->as.for_loop.var, stmt->as.for_loop.var);
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
        case ME_DSL_JIT_IR_STMT_WHILE:
            if (!me_jit_collect_return_dtype(&stmt->as.while_loop.body, has_return, out_dtype, error)) {
                return false;
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

typedef enum {
    ME_JIT_RESERVED_PARAM_NONE = 0,
    ME_JIT_RESERVED_PARAM_I,
    ME_JIT_RESERVED_PARAM_N,
    ME_JIT_RESERVED_PARAM_NDIM,
    ME_JIT_RESERVED_PARAM_GLOBAL_LINEAR_IDX,
    ME_JIT_RESERVED_PARAM_ND_CTX
} me_jit_reserved_param_kind;

static me_jit_reserved_param_kind me_jit_reserved_param_from_name(const char *name,
                                                                  const char *nd_ctx_name,
                                                                  int *out_dim) {
    if (out_dim) {
        *out_dim = -1;
    }
    if (!name) {
        return ME_JIT_RESERVED_PARAM_NONE;
    }
    if (strcmp(name, "_ndim") == 0) {
        return ME_JIT_RESERVED_PARAM_NDIM;
    }
    if (strcmp(name, "_global_linear_idx") == 0) {
        return ME_JIT_RESERVED_PARAM_GLOBAL_LINEAR_IDX;
    }
    if (nd_ctx_name && nd_ctx_name[0] != '\0' && strcmp(name, nd_ctx_name) == 0) {
        return ME_JIT_RESERVED_PARAM_ND_CTX;
    }
    if (name[0] == '_' &&
        (name[1] == 'i' || name[1] == 'n') &&
        isdigit((unsigned char)name[2]) &&
        name[3] == '\0') {
        int dim = name[2] - '0';
        if (dim < 0 || dim > 7) {
            return ME_JIT_RESERVED_PARAM_NONE;
        }
        if (out_dim) {
            *out_dim = dim;
        }
        return (name[1] == 'i') ? ME_JIT_RESERVED_PARAM_I : ME_JIT_RESERVED_PARAM_N;
    }
    return ME_JIT_RESERVED_PARAM_NONE;
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
    bool use_runtime_math_bridge = false;
    bool enable_vector_math = true;
    bool synth_reserved_non_nd = false;
    bool synth_reserved_nd = false;
    const char *synth_nd_ctx_name = "__me_nd_ctx";
    int synth_nd_compile_ndims = 0;
    if (options && options->symbol_name && options->symbol_name[0] != '\0') {
        symbol = options->symbol_name;
    }
    if (options && options->use_runtime_math_bridge) {
        use_runtime_math_bridge = true;
    }
    if (options && options->has_enable_vector_math) {
        enable_vector_math = options->enable_vector_math;
    }
    ctx.use_runtime_math_bridge = use_runtime_math_bridge;
    if (options && options->synth_reserved_non_nd) {
        synth_reserved_non_nd = true;
    }
    if (options && options->synth_reserved_nd) {
        synth_reserved_nd = true;
    }
    if (options && options->synth_nd_ctx_name && options->synth_nd_ctx_name[0] != '\0') {
        synth_nd_ctx_name = options->synth_nd_ctx_name;
    }
    if (options &&
        options->synth_nd_compile_ndims > 0 &&
        options->synth_nd_compile_ndims <= 8) {
        synth_nd_compile_ndims = options->synth_nd_compile_ndims;
    }
    if (!use_runtime_math_bridge) {
        me_jit_set_lowering_trace(options, "scalar", "", "runtime-math-bridge-disabled");
    }
    else if (!enable_vector_math) {
        me_jit_set_lowering_trace(options, "scalar", "", "vector-math-disabled");
    }
    else {
        me_jit_set_lowering_trace(options, "scalar", "", "no-vector-lowering-match");
    }
    const bool synth_nd_fixed_ndim = synth_reserved_nd && synth_nd_compile_ndims > 0;
    bool has_nd_ctx_param = false;
    if (synth_reserved_nd) {
        for (int i = 0; i < program->nparams; i++) {
            if (strcmp(program->params[i], synth_nd_ctx_name) == 0) {
                has_nd_ctx_param = true;
                break;
            }
        }
        if (!has_nd_ctx_param) {
            me_jit_set_error(error, 0, 0, "nd synth enabled but nd context parameter is missing");
            me_jit_locals_free(&ctx.locals);
            return false;
        }
    }
    bool synth_nd_has_i = false;
    bool synth_nd_has_global = false;
    if (synth_reserved_nd) {
        for (int i = 0; i < program->nparams; i++) {
            int reserved_dim = -1;
            me_jit_reserved_param_kind reserved_kind =
                me_jit_reserved_param_from_name(program->params[i], synth_nd_ctx_name, &reserved_dim);
            (void)reserved_dim;
            if (reserved_kind == ME_JIT_RESERVED_PARAM_I) {
                synth_nd_has_i = true;
            }
            else if (reserved_kind == ME_JIT_RESERVED_PARAM_GLOBAL_LINEAR_IDX) {
                synth_nd_has_global = true;
            }
        }
    }
    const bool synth_nd_global_only = synth_reserved_nd && synth_nd_has_global && !synth_nd_has_i;
    const bool synth_nd_needs_coord = synth_nd_has_i;

    if (!me_jit_emit_line(&ctx.source, 0, "typedef _Bool bool;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef signed char int8_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef short int16_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef int int32_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef long long int64_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef unsigned char uint8_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef unsigned short uint16_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef unsigned int uint32_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "typedef unsigned long long uint64_t;") ||
        !me_jit_emit_line(&ctx.source, 0, "#ifndef true") ||
        !me_jit_emit_line(&ctx.source, 0, "#define true 1") ||
        !me_jit_emit_line(&ctx.source, 0, "#endif") ||
        !me_jit_emit_line(&ctx.source, 0, "#ifndef false") ||
        !me_jit_emit_line(&ctx.source, 0, "#define false 0") ||
        !me_jit_emit_line(&ctx.source, 0, "#endif") ||
        !me_jit_emit_line(&ctx.source, 0, "#define ME_DSL_CAST_INT(x) ((int64_t)(x))") ||
        !me_jit_emit_line(&ctx.source, 0, "#define ME_DSL_CAST_FLOAT(x) ((double)(x))") ||
        !me_jit_emit_line(&ctx.source, 0, "#define ME_DSL_CAST_BOOL(x) ((x) != 0)") ||
        !me_jit_emit_line(&ctx.source, 0, "#define me_jit_i64_add_wrap(a, b) ((int64_t)((uint64_t)(a) + (uint64_t)(b)))") ||
        !me_jit_emit_line(&ctx.source, 0, "#define me_jit_i64_mul_wrap(a, b) ((int64_t)((uint64_t)(a) * (uint64_t)(b)))") ||
        !me_jit_emit_line(&ctx.source, 0, "#define me_jit_i64_addmul_wrap(acc, a, b) me_jit_i64_add_wrap((acc), me_jit_i64_mul_wrap((a), (b)))") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double acos(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double acosh(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double asin(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double asinh(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double atan(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double atan2(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double atanh(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double cbrt(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double ceil(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double copysign(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double cos(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double cosh(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double erf(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double erfc(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double exp(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double exp2(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double expm1(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double fabs(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double fdim(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double floor(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double fma(double, double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double fmax(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double fmin(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double fmod(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double hypot(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double ldexp(double, int);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double lgamma(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double log(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double log10(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double log1p(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double log2(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double nextafter(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double pow(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double remainder(double, double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double rint(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double round(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double sin(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double sinh(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double sqrt(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double tan(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double tanh(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double tgamma(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "extern double trunc(double);") ||
        !me_jit_emit_line(&ctx.source, 0, "")) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }
    if (use_runtime_math_bridge) {
        if (!me_jit_emit_runtime_bridge_decls(&ctx.source)) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
    }
    else {
        if (!me_jit_emit_line(&ctx.source, 0, "static double me_jit_exp10(double x) { return pow(10.0, x); }") ||
            !me_jit_emit_line(&ctx.source, 0, "static double me_jit_sinpi(double x) { return sin(3.14159265358979323846 * x); }") ||
            !me_jit_emit_line(&ctx.source, 0, "static double me_jit_cospi(double x) { return cos(3.14159265358979323846 * x); }") ||
            !me_jit_emit_line(&ctx.source, 0, "static double me_jit_logaddexp(double a, double b) { double hi = (a > b) ? a : b; double lo = (a > b) ? b : a; return hi + log1p(exp(lo - hi)); }") ||
            !me_jit_emit_line(&ctx.source, 0, "static double me_jit_where(double c, double x, double y) { return (c != 0.0) ? x : y; }") ||
            !me_jit_emit_line(&ctx.source, 0, "")) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
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
    bool needs_inputs = false;
    for (int i = 0; i < program->nparams; i++) {
        int reserved_dim = -1;
        me_jit_reserved_param_kind reserved_kind =
            (synth_reserved_non_nd || synth_reserved_nd)
                ? me_jit_reserved_param_from_name(program->params[i], synth_nd_ctx_name, &reserved_dim)
                                  : ME_JIT_RESERVED_PARAM_NONE;
        (void)reserved_dim;
        if (reserved_kind == ME_JIT_RESERVED_PARAM_NONE ||
            reserved_kind == ME_JIT_RESERVED_PARAM_ND_CTX) {
            needs_inputs = true;
            break;
        }
    }
    if (needs_inputs) {
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
        int reserved_dim = -1;
        me_jit_reserved_param_kind reserved_kind =
            (synth_reserved_non_nd || synth_reserved_nd)
                ? me_jit_reserved_param_from_name(program->params[i], synth_nd_ctx_name, &reserved_dim)
                                  : ME_JIT_RESERVED_PARAM_NONE;
        (void)reserved_dim;
        if (reserved_kind == ME_JIT_RESERVED_PARAM_I ||
            reserved_kind == ME_JIT_RESERVED_PARAM_N ||
            reserved_kind == ME_JIT_RESERVED_PARAM_NDIM ||
            reserved_kind == ME_JIT_RESERVED_PARAM_GLOBAL_LINEAR_IDX) {
            continue;
        }
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

    if (use_runtime_math_bridge && enable_vector_math) {
        me_jit_vec_unary_plan vec_plan;
        if (me_jit_detect_vec_unary_plan(program, output_dtype, &vec_plan)) {
            const char *vec_sym = me_jit_vec_unary_symbol(vec_plan.kind, output_dtype);
            if (!vec_sym || vec_plan.param_index < 0 || vec_plan.param_index >= program->nparams) {
                me_jit_set_error(error, 0, 0, "invalid vector bridge lowering state");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            bool ok = me_jit_emit_vec_unary_call(&ctx, output_dtype, vec_sym,
                                                 program->params[vec_plan.param_index],
                                                 vec_plan.has_offset, vec_plan.offset);
            if (!ok ||
                !me_jit_emit_line(&ctx.source, 1, "return 0;") ||
                !me_jit_emit_line(&ctx.source, 0, "}")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            me_jit_set_lowering_trace(options, "vector",
                                      me_jit_vec_unary_op_name(vec_plan.kind),
                                      "vector-lowered");
            me_jit_locals_free(&ctx.locals);
            *out_source = ctx.source.data;
            return true;
        }
        me_jit_vec_binary_plan vec_plan2;
        if (me_jit_detect_vec_binary_plan(program, output_dtype, &vec_plan2)) {
            const char *vec_sym = me_jit_vec_binary_symbol(vec_plan2.kind, output_dtype);
            const char *arg_a_param = NULL;
            const char *arg_b_param = NULL;
            if (vec_plan2.param_index_a >= 0) {
                if (vec_plan2.param_index_a >= program->nparams) {
                    me_jit_set_error(error, 0, 0, "invalid vector bridge lowering state");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
                arg_a_param = program->params[vec_plan2.param_index_a];
            }
            if (vec_plan2.param_index_b >= 0) {
                if (vec_plan2.param_index_b >= program->nparams) {
                    me_jit_set_error(error, 0, 0, "invalid vector bridge lowering state");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
                arg_b_param = program->params[vec_plan2.param_index_b];
            }
            if (!vec_sym ||
                (!arg_a_param && !vec_plan2.arg_a_const) ||
                (!arg_b_param && !vec_plan2.arg_b_const)) {
                me_jit_set_error(error, 0, 0, "invalid vector bridge lowering state");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            bool ok = me_jit_emit_vec_binary_call(&ctx, output_dtype, vec_sym,
                                                  arg_a_param,
                                                  vec_plan2.arg_a_const,
                                                  vec_plan2.arg_a_const_value,
                                                  arg_b_param,
                                                  vec_plan2.arg_b_const,
                                                  vec_plan2.arg_b_const_value);
            if (!ok ||
                !me_jit_emit_line(&ctx.source, 1, "return 0;") ||
                !me_jit_emit_line(&ctx.source, 0, "}")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            me_jit_set_lowering_trace(options, "vector",
                                      me_jit_vec_binary_op_name(vec_plan2.kind),
                                      "vector-lowered");
            me_jit_locals_free(&ctx.locals);
            *out_source = ctx.source.data;
            return true;
        }
    }

    if (synth_nd_global_only) {
        if (!me_jit_emit_line(&ctx.source, 1, "const int64_t *__me_nd_ctx = in___me_nd_ctx;")) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        if (synth_nd_fixed_ndim) {
            char line[96];
            if (snprintf(line, sizeof(line), "const int64_t __me_ndim_rt = %d;",
                         synth_nd_compile_ndims) >= (int)sizeof(line) ||
                !me_jit_emit_line(&ctx.source, 1, line)) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
        }
        else if (!me_jit_emit_line(&ctx.source, 1, "const int64_t __me_ndim_rt = __me_nd_ctx[0];")) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        if (!me_jit_emit_line(&ctx.source, 1, "int64_t __me_glin = 0;") ||
            !me_jit_emit_line(&ctx.source, 1, "int64_t __me_pos[8] = {0};") ||
            !me_jit_emit_line(&ctx.source, 1, "bool __me_seq = true;")) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        if (synth_nd_fixed_ndim) {
            char line[256];
            if (snprintf(line, sizeof(line), "const int64_t __me_ctx_tail = %d;",
                         1 + 4 * synth_nd_compile_ndims) >= (int)sizeof(line) ||
                !me_jit_emit_line(&ctx.source, 1, line) ||
                !me_jit_emit_line(&ctx.source, 1, "const int64_t __me_ctx_ver = __me_nd_ctx[__me_ctx_tail];")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            for (int d = 0; d < synth_nd_compile_ndims; d++) {
                if (snprintf(line, sizeof(line),
                             "const int64_t __me_stride_%d = __me_nd_ctx[%d];",
                             d, 1 + synth_nd_compile_ndims + d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 1, line)) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
            for (int d = 0; d < synth_nd_compile_ndims; d++) {
                char line[128];
                if (snprintf(line, sizeof(line),
                             "const int64_t __me_len_%d = __me_nd_ctx[%d];",
                             d, 1 + 3 * synth_nd_compile_ndims + d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 1, line)) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
            if (!me_jit_emit_line(&ctx.source, 1, "if (__me_ctx_ver >= 2) {") ||
                !me_jit_emit_line(&ctx.source, 2, "__me_seq = (__me_nd_ctx[__me_ctx_tail + 1] & 1) != 0;") ||
                !me_jit_emit_line(&ctx.source, 2, "__me_glin = __me_nd_ctx[__me_ctx_tail + 2];") ||
                !me_jit_emit_line(&ctx.source, 1, "}") ||
                !me_jit_emit_line(&ctx.source, 1, "else {")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            for (int d = 0; d < synth_nd_compile_ndims; d++) {
                if (snprintf(line, sizeof(line),
                             "__me_glin = me_jit_i64_addmul_wrap(__me_glin, __me_nd_ctx[%d], __me_stride_%d);",
                             1 + 2 * synth_nd_compile_ndims + d,
                             d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 2, line)) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
            for (int d = synth_nd_compile_ndims - 1; d >= 1; d--) {
                if (snprintf(line, sizeof(line),
                             "if (__me_nd_ctx[%d] != 0 || __me_len_%d != __me_nd_ctx[%d]) { __me_seq = false; }",
                             1 + 2 * synth_nd_compile_ndims + d,
                             d,
                             1 + d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 2, line)) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
            if (!me_jit_emit_line(&ctx.source, 1, "}")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
        }
        else if (!me_jit_emit_line(&ctx.source, 1, "const int64_t __me_ctx_tail = 1 + 4 * __me_ndim_rt;") ||
                 !me_jit_emit_line(&ctx.source, 1, "const int64_t __me_ctx_ver = __me_nd_ctx[__me_ctx_tail];") ||
                 !me_jit_emit_line(&ctx.source, 1, "if (__me_ctx_ver >= 2) {") ||
                 !me_jit_emit_line(&ctx.source, 2, "__me_seq = (__me_nd_ctx[__me_ctx_tail + 1] & 1) != 0;") ||
                 !me_jit_emit_line(&ctx.source, 2, "__me_glin = __me_nd_ctx[__me_ctx_tail + 2];") ||
                 !me_jit_emit_line(&ctx.source, 1, "}") ||
                 !me_jit_emit_line(&ctx.source, 1, "else {") ||
                 !me_jit_emit_line(&ctx.source, 2, "for (int64_t __me_d = 0; __me_d < __me_ndim_rt; __me_d++) {") ||
                 !me_jit_emit_line(&ctx.source, 3, "__me_glin = me_jit_i64_addmul_wrap(__me_glin, __me_nd_ctx[1 + 2 * __me_ndim_rt + __me_d], __me_nd_ctx[1 + __me_ndim_rt + __me_d]);") ||
                 !me_jit_emit_line(&ctx.source, 2, "}") ||
                 !me_jit_emit_line(&ctx.source, 2, "for (int64_t __me_d = __me_ndim_rt - 1; __me_d >= 1; __me_d--) {") ||
                 !me_jit_emit_line(&ctx.source, 3, "if (__me_nd_ctx[1 + 2 * __me_ndim_rt + __me_d] != 0 || __me_nd_ctx[1 + 3 * __me_ndim_rt + __me_d] != __me_nd_ctx[1 + __me_d]) { __me_seq = false; }") ||
                 !me_jit_emit_line(&ctx.source, 2, "}") ||
                 !me_jit_emit_line(&ctx.source, 1, "}")) {
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
    if (synth_reserved_nd) {
        if (synth_nd_global_only) {
            if (!me_jit_emit_line(&ctx.source, 2, "int64_t __me_global_linear_idx_rt = __me_seq ? me_jit_i64_add_wrap(__me_glin, idx) : __me_glin;")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
        }
        else {
            if (!me_jit_emit_line(&ctx.source, 2, "const int64_t *__me_nd_ctx = in___me_nd_ctx;")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            if (synth_nd_fixed_ndim) {
                char line[96];
                if (snprintf(line, sizeof(line), "const int64_t __me_ndim_rt = %d;",
                             synth_nd_compile_ndims) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 2, line)) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
            else if (!me_jit_emit_line(&ctx.source, 2, "const int64_t __me_ndim_rt = __me_nd_ctx[0];")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            if (synth_nd_needs_coord) {
                if (!me_jit_emit_line(&ctx.source, 2, "int64_t __me_coord[8] = {0};") ||
                    !me_jit_emit_line(&ctx.source, 2, "int64_t __me_rem = idx;")) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
                if (synth_nd_fixed_ndim) {
                    if (!me_jit_emit_line(&ctx.source, 2, "int64_t __me_len = 0;") ||
                        !me_jit_emit_line(&ctx.source, 2, "int64_t __me_q = 0;") ||
                        !me_jit_emit_line(&ctx.source, 2, "int64_t __me_digit = 0;")) {
                        me_jit_set_error(error, 0, 0, "out of memory");
                        me_jit_locals_free(&ctx.locals);
                        free(ctx.source.data);
                        return false;
                    }
                    for (int d = synth_nd_compile_ndims - 1; d >= 0; d--) {
                        char line[192];
                        if (snprintf(line, sizeof(line),
                                     "__me_len = __me_nd_ctx[%d];",
                                     1 + 3 * synth_nd_compile_ndims + d) >= (int)sizeof(line) ||
                            !me_jit_emit_line(&ctx.source, 2, line) ||
                            !me_jit_emit_line(&ctx.source, 2, "__me_q = (__me_len > 0) ? (__me_rem / __me_len) : 0;") ||
                            !me_jit_emit_line(&ctx.source, 2, "__me_digit = (__me_len > 0) ? (__me_rem - __me_q * __me_len) : 0;") ||
                            !me_jit_emit_line(&ctx.source, 2, "__me_rem = __me_q;") ||
                            snprintf(line, sizeof(line),
                                     "__me_coord[%d] = __me_nd_ctx[%d] + __me_digit;",
                                     d, 1 + 2 * synth_nd_compile_ndims + d) >= (int)sizeof(line) ||
                            !me_jit_emit_line(&ctx.source, 2, line)) {
                            me_jit_set_error(error, 0, 0, "out of memory");
                            me_jit_locals_free(&ctx.locals);
                            free(ctx.source.data);
                            return false;
                        }
                    }
                }
                else if (!me_jit_emit_line(&ctx.source, 2, "for (int64_t __me_d = __me_ndim_rt - 1; __me_d >= 0; __me_d--) {") ||
                         !me_jit_emit_line(&ctx.source, 3, "int64_t __me_len = __me_nd_ctx[1 + 3 * __me_ndim_rt + __me_d];") ||
                         !me_jit_emit_line(&ctx.source, 3, "int64_t __me_q = (__me_len > 0) ? (__me_rem / __me_len) : 0;") ||
                         !me_jit_emit_line(&ctx.source, 3, "int64_t __me_digit = (__me_len > 0) ? (__me_rem - __me_q * __me_len) : 0;") ||
                         !me_jit_emit_line(&ctx.source, 3, "__me_rem = __me_q;") ||
                         !me_jit_emit_line(&ctx.source, 3, "__me_coord[__me_d] = __me_nd_ctx[1 + 2 * __me_ndim_rt + __me_d] + __me_digit;") ||
                         !me_jit_emit_line(&ctx.source, 2, "}")) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
            if (synth_nd_has_global) {
                if (!me_jit_emit_line(&ctx.source, 2, "int64_t __me_global_linear_idx_rt = 0;")) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
                if (synth_nd_fixed_ndim) {
                    for (int d = 0; d < synth_nd_compile_ndims; d++) {
                        char line[192];
                        if (snprintf(line, sizeof(line),
                                     "__me_global_linear_idx_rt = me_jit_i64_addmul_wrap(__me_global_linear_idx_rt, __me_coord[%d], __me_nd_ctx[%d]);",
                                     d, 1 + synth_nd_compile_ndims + d) >= (int)sizeof(line) ||
                            !me_jit_emit_line(&ctx.source, 2, line)) {
                            me_jit_set_error(error, 0, 0, "out of memory");
                            me_jit_locals_free(&ctx.locals);
                            free(ctx.source.data);
                            return false;
                        }
                    }
                }
                else if (!me_jit_emit_line(&ctx.source, 2, "for (int64_t __me_d = 0; __me_d < __me_ndim_rt; __me_d++) {") ||
                         !me_jit_emit_line(&ctx.source, 3, "__me_global_linear_idx_rt = me_jit_i64_addmul_wrap(__me_global_linear_idx_rt, __me_coord[__me_d], __me_nd_ctx[1 + __me_ndim_rt + __me_d]);") ||
                         !me_jit_emit_line(&ctx.source, 2, "}")) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
        }
    }

    for (int i = 0; i < program->nparams; i++) {
        const char *ptype = me_jit_c_type(program->param_dtypes[i]);
        int reserved_dim = -1;
        me_jit_reserved_param_kind reserved_kind =
            (synth_reserved_non_nd || synth_reserved_nd)
                ? me_jit_reserved_param_from_name(program->params[i], synth_nd_ctx_name, &reserved_dim)
                                  : ME_JIT_RESERVED_PARAM_NONE;
        if (reserved_kind == ME_JIT_RESERVED_PARAM_ND_CTX) {
            continue;
        }
        /* ND synth declarations include a long ternary expression that can exceed
           the generic small buffer used for regular input loads. */
        size_t need = strlen(ptype) + strlen(program->params[i]) * 2 + 192;
        char *line = malloc(need);
        if (!line) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
        if (synth_reserved_non_nd && reserved_kind == ME_JIT_RESERVED_PARAM_I) {
            const char *value = (reserved_dim == 0) ? "idx" : "0";
            snprintf(line, need, "%s %s = (%s)%s;", ptype, program->params[i], ptype, value);
        }
        else if (synth_reserved_non_nd && reserved_kind == ME_JIT_RESERVED_PARAM_N) {
            const char *value = (reserved_dim == 0) ? "nitems" : "1";
            snprintf(line, need, "%s %s = (%s)%s;", ptype, program->params[i], ptype, value);
        }
        else if (synth_reserved_non_nd && reserved_kind == ME_JIT_RESERVED_PARAM_NDIM) {
            snprintf(line, need, "%s %s = (%s)1;", ptype, program->params[i], ptype);
        }
        else if (synth_reserved_non_nd && reserved_kind == ME_JIT_RESERVED_PARAM_GLOBAL_LINEAR_IDX) {
            snprintf(line, need, "%s %s = (%s)idx;", ptype, program->params[i], ptype);
        }
        else if (synth_reserved_nd && reserved_kind == ME_JIT_RESERVED_PARAM_I) {
            if (synth_nd_fixed_ndim) {
                if (reserved_dim >= 0 && reserved_dim < synth_nd_compile_ndims) {
                    snprintf(line, need, "%s %s = (%s)__me_coord[%d];",
                             ptype, program->params[i], ptype, reserved_dim);
                }
                else {
                    snprintf(line, need, "%s %s = (%s)0;",
                             ptype, program->params[i], ptype);
                }
            }
            else {
                snprintf(line, need, "%s %s = (%s)((%d < (int)__me_ndim_rt) ? __me_coord[%d] : 0);",
                         ptype, program->params[i], ptype, reserved_dim, reserved_dim);
            }
        }
        else if (synth_reserved_nd && reserved_kind == ME_JIT_RESERVED_PARAM_N) {
            if (synth_nd_fixed_ndim) {
                if (reserved_dim >= 0 && reserved_dim < synth_nd_compile_ndims) {
                    snprintf(line, need, "%s %s = (%s)__me_nd_ctx[%d];",
                             ptype, program->params[i], ptype, 1 + reserved_dim);
                }
                else {
                    snprintf(line, need, "%s %s = (%s)1;",
                             ptype, program->params[i], ptype);
                }
            }
            else {
                snprintf(line, need, "%s %s = (%s)((%d < (int)__me_ndim_rt) ? __me_nd_ctx[1 + %d] : 1);",
                         ptype, program->params[i], ptype, reserved_dim, reserved_dim);
            }
        }
        else if (synth_reserved_nd && reserved_kind == ME_JIT_RESERVED_PARAM_NDIM) {
            if (synth_nd_fixed_ndim) {
                snprintf(line, need, "%s %s = (%s)%d;",
                         ptype, program->params[i], ptype, synth_nd_compile_ndims);
            }
            else {
                snprintf(line, need, "%s %s = (%s)__me_ndim_rt;",
                         ptype, program->params[i], ptype);
            }
        }
        else if (synth_reserved_nd && reserved_kind == ME_JIT_RESERVED_PARAM_GLOBAL_LINEAR_IDX) {
            snprintf(line, need, "%s %s = (%s)__me_global_linear_idx_rt;",
                     ptype, program->params[i], ptype);
        }
        else {
            snprintf(line, need, "%s %s = in_%s[idx];", ptype, program->params[i], program->params[i]);
        }
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
        !me_jit_emit_line(&ctx.source, 2, "out[idx] = __me_out;")) {
        me_jit_set_error(error, 0, 0, "out of memory");
        me_jit_locals_free(&ctx.locals);
        free(ctx.source.data);
        return false;
    }

    if (synth_nd_global_only) {
        if (synth_nd_fixed_ndim) {
            if (!me_jit_emit_line(&ctx.source, 2, "if (!__me_seq && idx + 1 < nitems) {") ||
                !me_jit_emit_line(&ctx.source, 3, "bool __me_advanced = false;")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
            for (int d = synth_nd_compile_ndims - 1; d >= 0; d--) {
                char line[192];
                if (!me_jit_emit_line(&ctx.source, 3, "if (!__me_advanced) {") ||
                    snprintf(line, sizeof(line), "if (__me_len_%d > 0) {", d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 4, line) ||
                    snprintf(line, sizeof(line), "int64_t __me_next_%d = __me_pos[%d] + 1;", d, d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 5, line) ||
                    snprintf(line, sizeof(line), "if (__me_next_%d < __me_len_%d) {", d, d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 5, line) ||
                    snprintf(line, sizeof(line), "__me_pos[%d] = __me_next_%d;", d, d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 6, line) ||
                    snprintf(line, sizeof(line), "__me_glin = me_jit_i64_add_wrap(__me_glin, __me_stride_%d);", d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 6, line) ||
                    !me_jit_emit_line(&ctx.source, 6, "__me_advanced = true;") ||
                    !me_jit_emit_line(&ctx.source, 5, "}") ||
                    !me_jit_emit_line(&ctx.source, 5, "else {") ||
                    snprintf(line, sizeof(line), "__me_glin = me_jit_i64_addmul_wrap(__me_glin, -(__me_len_%d - 1), __me_stride_%d);", d, d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 6, line) ||
                    snprintf(line, sizeof(line), "__me_pos[%d] = 0;", d) >= (int)sizeof(line) ||
                    !me_jit_emit_line(&ctx.source, 6, line) ||
                    !me_jit_emit_line(&ctx.source, 5, "}") ||
                    !me_jit_emit_line(&ctx.source, 4, "}") ||
                    !me_jit_emit_line(&ctx.source, 3, "}")) {
                    me_jit_set_error(error, 0, 0, "out of memory");
                    me_jit_locals_free(&ctx.locals);
                    free(ctx.source.data);
                    return false;
                }
            }
            if (!me_jit_emit_line(&ctx.source, 2, "}")) {
                me_jit_set_error(error, 0, 0, "out of memory");
                me_jit_locals_free(&ctx.locals);
                free(ctx.source.data);
                return false;
            }
        }
        else if (!me_jit_emit_line(&ctx.source, 2, "if (!__me_seq && idx + 1 < nitems) {") ||
                 !me_jit_emit_line(&ctx.source, 3, "for (int64_t __me_d = __me_ndim_rt - 1; __me_d >= 0; __me_d--) {") ||
                 !me_jit_emit_line(&ctx.source, 4, "int64_t __me_len = __me_nd_ctx[1 + 3 * __me_ndim_rt + __me_d];") ||
                 !me_jit_emit_line(&ctx.source, 4, "if (__me_len <= 0) { continue; }") ||
                 !me_jit_emit_line(&ctx.source, 4, "int64_t __me_next = __me_pos[__me_d] + 1;") ||
                 !me_jit_emit_line(&ctx.source, 4, "if (__me_next < __me_len) {") ||
                 !me_jit_emit_line(&ctx.source, 5, "__me_pos[__me_d] = __me_next;") ||
                 !me_jit_emit_line(&ctx.source, 5, "__me_glin = me_jit_i64_add_wrap(__me_glin, __me_nd_ctx[1 + __me_ndim_rt + __me_d]);") ||
                 !me_jit_emit_line(&ctx.source, 5, "break;") ||
                 !me_jit_emit_line(&ctx.source, 4, "}") ||
                 !me_jit_emit_line(&ctx.source, 4, "__me_glin = me_jit_i64_addmul_wrap(__me_glin, -(__me_len - 1), __me_nd_ctx[1 + __me_ndim_rt + __me_d]);") ||
                 !me_jit_emit_line(&ctx.source, 4, "__me_pos[__me_d] = 0;") ||
                 !me_jit_emit_line(&ctx.source, 3, "}") ||
                 !me_jit_emit_line(&ctx.source, 2, "}")) {
            me_jit_set_error(error, 0, 0, "out of memory");
            me_jit_locals_free(&ctx.locals);
            free(ctx.source.data);
            return false;
        }
    }

    if (!me_jit_emit_line(&ctx.source, 1, "}") ||
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
