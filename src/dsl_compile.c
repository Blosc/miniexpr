/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_compile_internal.h"

#include "dsl_jit_cgen.h"
#include "functions.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#endif

#define ME_DSL_JIT_SYNTH_ND_CTX_PARAM "__me_nd_ctx"

typedef struct {
    const char *source;
    me_dtype output_dtype;
    bool output_dtype_auto;
    int loop_depth;
    bool allow_new_locals;
    int *error_pos;
    me_dsl_compiled_expr *output_expr;
    bool has_return;
    me_dtype return_dtype;
    me_dsl_compiled_program *program;
    const me_variable *funcs;
    int func_count;
    char *error_reason;
    size_t error_reason_cap;
} dsl_compile_ctx;

static bool is_function_entry(const me_variable *var) {
    if (!var) {
        return false;
    }
    return IS_FUNCTION(var->type) || IS_CLOSURE(var->type);
}

static bool is_variable_entry(const me_variable *var) {
    if (!var) {
        return false;
    }
    if (var->type == 0) {
        return true;
    }
    return TYPE_MASK(var->type) == ME_VARIABLE;
}

static bool is_valid_dtype(me_dtype dtype) {
    return dtype >= ME_AUTO && dtype <= ME_STRING;
}

static void dsl_set_error_reason(dsl_compile_ctx *ctx, const char *msg) {
    if (!ctx || !ctx->error_reason || ctx->error_reason_cap == 0) {
        return;
    }
    if (!msg || msg[0] == '\0') {
        ctx->error_reason[0] = '\0';
        return;
    }
    snprintf(ctx->error_reason, ctx->error_reason_cap, "%s", msg);
}

static int dsl_offset_from_linecol(const char *source, int line, int column) {
    if (!source || line <= 0 || column <= 0) {
        return -1;
    }
    int current_line = 1;
    int current_col = 1;
    for (int i = 0; source[i] != '\0'; i++) {
        if (current_line == line && current_col == column) {
            return i;
        }
        if (source[i] == '\n') {
            current_line++;
            current_col = 1;
        }
        else {
            current_col++;
        }
    }
    return -1;
}

static bool dsl_is_reserved_name(const char *name) {
    if (!name) {
        return false;
    }
    if (strcmp(name, "print") == 0) {
        return true;
    }
    if (strcmp(name, "int") == 0 || strcmp(name, "float") == 0 || strcmp(name, "bool") == 0) {
        return true;
    }
    if (strcmp(name, "def") == 0 || strcmp(name, "return") == 0) {
        return true;
    }
    if (strcmp(name, "_ndim") == 0) {
        return true;
    }
    if (strcmp(name, "_flat_idx") == 0) {
        return true;
    }
    if ((name[0] == '_' && (name[1] == 'i' || name[1] == 'n')) && isdigit((unsigned char)name[2])) {
        return true;
    }
    return false;
}

static double dsl_cast_int_intrinsic(double x) {
    return (double)(int64_t)x;
}

static double dsl_cast_float_intrinsic(double x) {
    return x;
}

static double dsl_cast_bool_intrinsic(double x) {
    return x != 0.0 ? 1.0 : 0.0;
}

static bool dsl_dtype_is_integer(me_dtype dtype) {
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

static bool dsl_dtype_is_float_or_complex(me_dtype dtype) {
    switch (dtype) {
    case ME_FLOAT32:
    case ME_FLOAT64:
    case ME_COMPLEX64:
    case ME_COMPLEX128:
        return true;
    default:
        return false;
    }
}

static me_dtype dsl_cast_int_target_dtype(me_dtype expr_dtype) {
    switch (expr_dtype) {
    case ME_INT8:
    case ME_INT16:
    case ME_INT32:
    case ME_INT64:
        return expr_dtype;
    default:
        return ME_INT64;
    }
}

static me_dtype dsl_cast_float_target_dtype(me_dtype expr_dtype) {
    switch (expr_dtype) {
    case ME_FLOAT32:
    case ME_FLOAT64:
        return expr_dtype;
    default:
        return ME_FLOAT64;
    }
}

static bool dsl_expr_uses_identifier(const char *expr, const char *ident) {
    if (!expr || !ident) {
        return false;
    }
    size_t ident_len = strlen(ident);
    const char *p = expr;
    while (*p) {
        if (isalpha((unsigned char)*p) || *p == '_') {
            const char *start = p;
            p++;
            while (isalnum((unsigned char)*p) || *p == '_') {
                p++;
            }
            size_t len = (size_t)(p - start);
            if (len == ident_len && strncmp(start, ident, len) == 0) {
                return true;
            }
        }
        else {
            p++;
        }
    }
    return false;
}

static bool dsl_is_ident_char_scan(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static bool dsl_is_cast_intrinsic_name_len(const char *s, size_t n) {
    return (n == 3 && strncmp(s, "int", 3) == 0) ||
        (n == 5 && strncmp(s, "float", 5) == 0) ||
        (n == 4 && strncmp(s, "bool", 4) == 0);
}

static bool dsl_validate_cast_intrinsics_usage(const char *text, int *error_offset) {
    if (error_offset) {
        *error_offset = -1;
    }
    if (!text) {
        return true;
    }

    const char *p = text;
    while (*p) {
        if (*p == '"' || *p == '\'') {
            char quote = *p++;
            while (*p) {
                if (*p == '\\' && p[1] != '\0') {
                    p += 2;
                    continue;
                }
                if (*p == quote) {
                    p++;
                    break;
                }
                p++;
            }
            continue;
        }

        if (!(isalpha((unsigned char)*p) || *p == '_')) {
            p++;
            continue;
        }

        const char *ident_start = p;
        p++;
        while (*p && dsl_is_ident_char_scan(*p)) {
            p++;
        }
        size_t ident_len = (size_t)(p - ident_start);
        if (!dsl_is_cast_intrinsic_name_len(ident_start, ident_len)) {
            continue;
        }

        const char *q = p;
        while (*q && isspace((unsigned char)*q)) {
            q++;
        }
        if (*q != '(') {
            continue;
        }

        const char *arg_start = q + 1;
        const char *r = arg_start;
        int depth = 1;
        int top_level_commas = 0;
        char quote = '\0';
        while (*r && depth > 0) {
            char c = *r;
            if (quote) {
                if (c == '\\' && r[1] != '\0') {
                    r += 2;
                    continue;
                }
                if (c == quote) {
                    quote = '\0';
                }
                r++;
                continue;
            }
            if (c == '"' || c == '\'') {
                quote = c;
                r++;
                continue;
            }
            if (c == '(') {
                depth++;
            }
            else if (c == ')') {
                depth--;
                if (depth == 0) {
                    break;
                }
            }
            else if (c == ',' && depth == 1) {
                top_level_commas++;
            }
            r++;
        }

        if (depth != 0) {
            if (error_offset) {
                *error_offset = (int)(ident_start - text) + 1;
            }
            return false;
        }

        const char *arg_end = r;
        const char *trim_start = arg_start;
        const char *trim_end = arg_end;
        while (trim_start < trim_end && isspace((unsigned char)*trim_start)) {
            trim_start++;
        }
        while (trim_end > trim_start && isspace((unsigned char)trim_end[-1])) {
            trim_end--;
        }

        if (trim_start == trim_end || top_level_commas > 0) {
            if (error_offset) {
                *error_offset = (int)(ident_start - text) + 1;
            }
            return false;
        }

        p = r + 1;
    }

    return true;
}

static bool dsl_collect_var_indices(const me_expr *expr, int **out_indices, int *out_count) {
    if (!expr || !out_indices || !out_count) {
        return false;
    }
    bool used[ME_MAX_VARS];
    memset(used, 0, sizeof(used));
    int max_idx = -1;

    const me_expr *stack[512];
    int sp = 0;
    stack[sp++] = expr;

    while (sp > 0) {
        const me_expr *node = stack[--sp];
        if (!node) {
            continue;
        }
        if (TYPE_MASK(node->type) == ME_VARIABLE) {
            const char *ptr = (const char *)node->bound;
            int idx = (int)(ptr - synthetic_var_addresses);
            if (idx >= 0 && idx < ME_MAX_VARS) {
                used[idx] = true;
                if (idx > max_idx) {
                    max_idx = idx;
                }
            }
        }
        else if (IS_FUNCTION(node->type) || IS_CLOSURE(node->type)) {
            int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                if (sp < (int)(sizeof(stack) / sizeof(stack[0]))) {
                    stack[sp++] = (const me_expr *)node->parameters[i];
                }
            }
        }
    }

    int count = 0;
    for (int i = 0; i <= max_idx; i++) {
        if (used[i]) {
            count++;
        }
    }
    if (count == 0) {
        *out_indices = NULL;
        *out_count = 0;
        return true;
    }
    int *indices = malloc((size_t)count * sizeof(*indices));
    if (!indices) {
        return false;
    }
    int pos = 0;
    for (int i = 0; i <= max_idx; i++) {
        if (used[i]) {
            indices[pos++] = i;
        }
    }
    *out_indices = indices;
    *out_count = count;
    return true;
}

static bool dsl_expr_is_uniform(const me_expr *n, const bool *uniform, int nvars) {
    if (!n) {
        return true;
    }
    if (is_reduction_node(n)) {
        return true;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
    case ME_STRING_CONSTANT:
        return true;
    case ME_VARIABLE: {
        if (!is_synthetic_address(n->bound)) {
            return false;
        }
        int idx = (int)((const char *)n->bound - synthetic_var_addresses);
        if (idx < 0 || idx >= nvars) {
            return false;
        }
        return uniform[idx];
    }
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7: {
        int arity = ARITY(n->type);
        for (int i = 0; i < arity; i++) {
            if (!dsl_expr_is_uniform((const me_expr *)n->parameters[i], uniform, nvars)) {
                return false;
            }
        }
        return true;
    }
    default:
        return true;
    }
}

static bool dsl_program_is_dsl(const me_dsl_program *program) {
    if (!program) {
        return false;
    }
    return program->name != NULL;
}

static void dsl_scan_reserved_usage_block(const me_dsl_block *block, int *uses_i_mask,
                                          int *uses_n_mask, bool *uses_ndim,
                                          bool *uses_flat_idx) {
    if (!block) {
        return;
    }
    for (int i = 0; i < block->nstmts; i++) {
        me_dsl_stmt *stmt = block->stmts[i];
        const char *expr_text = NULL;
        if (!stmt) {
            continue;
        }
        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN:
            expr_text = stmt->as.assign.value ? stmt->as.assign.value->text : NULL;
            break;
        case ME_DSL_STMT_EXPR:
            expr_text = stmt->as.expr_stmt.expr ? stmt->as.expr_stmt.expr->text : NULL;
            break;
        case ME_DSL_STMT_RETURN:
            expr_text = stmt->as.return_stmt.expr ? stmt->as.return_stmt.expr->text : NULL;
            break;
        case ME_DSL_STMT_PRINT:
            expr_text = stmt->as.print_stmt.call ? stmt->as.print_stmt.call->text : NULL;
            break;
        case ME_DSL_STMT_IF:
            expr_text = stmt->as.if_stmt.cond ? stmt->as.if_stmt.cond->text : NULL;
            break;
        case ME_DSL_STMT_WHILE:
            expr_text = stmt->as.while_loop.cond ? stmt->as.while_loop.cond->text : NULL;
            break;
        case ME_DSL_STMT_FOR:
            expr_text = stmt->as.for_loop.limit ? stmt->as.for_loop.limit->text : NULL;
            break;
        case ME_DSL_STMT_BREAK:
        case ME_DSL_STMT_CONTINUE:
            expr_text = stmt->as.flow.cond ? stmt->as.flow.cond->text : NULL;
            break;
        }
        if (expr_text) {
            if (dsl_expr_uses_identifier(expr_text, "_ndim")) {
                *uses_ndim = true;
            }
            if (dsl_expr_uses_identifier(expr_text, "_flat_idx")) {
                *uses_flat_idx = true;
            }
            for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
                char name[8];
                snprintf(name, sizeof(name), "_i%d", d);
                if (dsl_expr_uses_identifier(expr_text, name)) {
                    *uses_i_mask |= (1 << d);
                }
                snprintf(name, sizeof(name), "_n%d", d);
                if (dsl_expr_uses_identifier(expr_text, name)) {
                    *uses_n_mask |= (1 << d);
                }
            }
        }
        if (stmt->kind == ME_DSL_STMT_IF) {
            dsl_scan_reserved_usage_block(&stmt->as.if_stmt.then_block, uses_i_mask, uses_n_mask, uses_ndim,
                                          uses_flat_idx);
            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                dsl_scan_reserved_usage_block(&stmt->as.if_stmt.elif_branches[j].block,
                                              uses_i_mask, uses_n_mask, uses_ndim,
                                              uses_flat_idx);
                const char *elif_text = stmt->as.if_stmt.elif_branches[j].cond
                                            ? stmt->as.if_stmt.elif_branches[j].cond->text
                                            : NULL;
                if (elif_text) {
                    if (dsl_expr_uses_identifier(elif_text, "_ndim")) {
                        *uses_ndim = true;
                    }
                    if (dsl_expr_uses_identifier(elif_text, "_flat_idx")) {
                        *uses_flat_idx = true;
                    }
                    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
                        char name[8];
                        snprintf(name, sizeof(name), "_i%d", d);
                        if (dsl_expr_uses_identifier(elif_text, name)) {
                            *uses_i_mask |= (1 << d);
                        }
                        snprintf(name, sizeof(name), "_n%d", d);
                        if (dsl_expr_uses_identifier(elif_text, name)) {
                            *uses_n_mask |= (1 << d);
                        }
                    }
                }
            }
            if (stmt->as.if_stmt.has_else) {
                dsl_scan_reserved_usage_block(&stmt->as.if_stmt.else_block,
                                              uses_i_mask, uses_n_mask, uses_ndim,
                                              uses_flat_idx);
            }
        }
        if (stmt->kind == ME_DSL_STMT_FOR) {
            dsl_scan_reserved_usage_block(&stmt->as.for_loop.body, uses_i_mask, uses_n_mask, uses_ndim,
                                          uses_flat_idx);
        }
        if (stmt->kind == ME_DSL_STMT_WHILE) {
            dsl_scan_reserved_usage_block(&stmt->as.while_loop.body, uses_i_mask, uses_n_mask, uses_ndim,
                                          uses_flat_idx);
        }
    }
}

static const char *dsl_skip_space_inline(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) {
        p++;
    }
    return p;
}

static bool dsl_is_ident_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static char *dsl_trim_copy(const char *start, const char *end) {
    if (!start || !end || end <= start) {
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

static int dsl_utf8_encode(uint32_t cp, char *out, int cap) {
    if (cap < 1) {
        return 0;
    }
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF && cap >= 2) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF && cap >= 3) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF && cap >= 4) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

static bool dsl_parse_hex_digits(const char **p, int digits, uint32_t *out) {
    uint32_t value = 0;
    for (int i = 0; i < digits; i++) {
        char c = (*p)[i];
        uint32_t v;
        if (c >= '0' && c <= '9') {
            v = (uint32_t)(c - '0');
        }
        else if (c >= 'a' && c <= 'f') {
            v = (uint32_t)(10 + c - 'a');
        }
        else if (c >= 'A' && c <= 'F') {
            v = (uint32_t)(10 + c - 'A');
        }
        else {
            return false;
        }
        value = (value << 4) | v;
    }
    *p += digits;
    *out = value;
    return true;
}

static char *dsl_unescape_string_literal(const char *text) {
    if (!text) {
        return NULL;
    }
    const char *p = dsl_skip_space_inline(text);
    if (!p || (*p != '"' && *p != '\'')) {
        return NULL;
    }
    char quote = *p++;
    size_t cap = 64;
    size_t len = 0;
    char *out = malloc(cap);
    if (!out) {
        return NULL;
    }
    while (*p && *p != quote) {
        uint32_t cp = 0;
        if (*p == '\\') {
            p++;
            if (!*p) {
                free(out);
                return NULL;
            }
            char esc = *p++;
            switch (esc) {
            case '\\': cp = '\\'; break;
            case '"': cp = '"'; break;
            case '\'': cp = '\''; break;
            case 'n': cp = '\n'; break;
            case 't': cp = '\t'; break;
            case 'u':
                if (!dsl_parse_hex_digits(&p, 4, &cp)) {
                    free(out);
                    return NULL;
                }
                break;
            case 'U':
                if (!dsl_parse_hex_digits(&p, 8, &cp)) {
                    free(out);
                    return NULL;
                }
                break;
            default:
                free(out);
                return NULL;
            }
        }
        else {
            cp = (unsigned char)*p++;
        }

        char utf8[4];
        int wrote = dsl_utf8_encode(cp, utf8, (int)sizeof(utf8));
        if (wrote <= 0) {
            free(out);
            return NULL;
        }
        if (len + (size_t)wrote + 1 > cap) {
            size_t next_cap = cap * 2;
            while (len + (size_t)wrote + 1 > next_cap) {
                next_cap *= 2;
            }
            char *next = realloc(out, next_cap);
            if (!next) {
                free(out);
                return NULL;
            }
            out = next;
            cap = next_cap;
        }
        memcpy(out + len, utf8, (size_t)wrote);
        len += (size_t)wrote;
    }
    if (*p != quote) {
        free(out);
        return NULL;
    }
    out[len] = '\0';
    return out;
}

static int dsl_count_placeholders(const char *fmt) {
    int count = 0;
    if (!fmt) {
        return -1;
    }
    for (size_t i = 0; fmt[i] != '\0'; i++) {
        if (fmt[i] == '{') {
            if (fmt[i + 1] == '{') {
                i++;
                continue;
            }
            if (fmt[i + 1] == '}') {
                count++;
                i++;
                continue;
            }
            return -1;
        }
        if (fmt[i] == '}') {
            if (fmt[i + 1] == '}') {
                i++;
                continue;
            }
            return -1;
        }
    }
    return count;
}

static bool dsl_split_print_args(const char *text, char ***out_args, int *out_nargs) {
    if (!text || !out_args || !out_nargs) {
        return false;
    }
    *out_args = NULL;
    *out_nargs = 0;

    const char *p = dsl_skip_space_inline(text);
    const char *ident = "print";
    size_t ident_len = strlen(ident);
    if (strncmp(p, ident, ident_len) != 0 || dsl_is_ident_char(p[ident_len])) {
        return false;
    }
    p += ident_len;
    p = dsl_skip_space_inline(p);
    if (*p != '(') {
        return false;
    }
    p++;

    const char *arg_start = p;
    int depth = 0;
    bool in_string = false;
    char quote = '\0';
    char **args = NULL;
    int nargs = 0;

    bool closed = false;
    for (; *p; p++) {
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
            if (depth == 0) {
                char *arg = dsl_trim_copy(arg_start, p);
                if (!arg) {
                    goto fail;
                }
                char **next_args = realloc(args, (size_t)(nargs + 1) * sizeof(*next_args));
                if (!next_args) {
                    free(arg);
                    goto fail;
                }
                args = next_args;
                args[nargs++] = arg;
                p++;
                closed = true;
                break;
            }
            depth--;
            continue;
        }
        if (c == ',' && depth == 0) {
            char *arg = dsl_trim_copy(arg_start, p);
            if (!arg) {
                goto fail;
            }
            char **next_args = realloc(args, (size_t)(nargs + 1) * sizeof(*next_args));
            if (!next_args) {
                free(arg);
                goto fail;
            }
            args = next_args;
            args[nargs++] = arg;
            arg_start = p + 1;
        }
    }

    if (!closed) {
        goto fail;
    }
    p = dsl_skip_space_inline(p);
    if (*p != '\0') {
        goto fail;
    }

    *out_args = args;
    *out_nargs = nargs;
    return true;

fail:
    if (args) {
        for (int i = 0; i < nargs; i++) {
            free(args[i]);
        }
    }
    free(args);
    return false;
}

static bool dsl_compile_expr(dsl_compile_ctx *ctx, const me_dsl_expr *expr_node,
                             me_dtype expr_dtype, me_dsl_compiled_expr *out_expr) {
    if (!ctx || !expr_node || !out_expr) {
        return false;
    }
    memset(out_expr, 0, sizeof(*out_expr));
    int cast_error_offset = -1;
    if (!dsl_validate_cast_intrinsics_usage(expr_node->text, &cast_error_offset)) {
        dsl_set_error_reason(ctx, "invalid cast intrinsic usage: int()/float()/bool() must be called as functions");
        if (ctx->error_pos) {
            int offset = dsl_offset_from_linecol(ctx->source, expr_node->line, expr_node->column);
            if (offset >= 0 && cast_error_offset > 0) {
                *ctx->error_pos = offset + cast_error_offset - 1;
            }
            else {
                *ctx->error_pos = offset >= 0 ? offset : -1;
            }
        }
        return false;
    }
    me_variable cast_funcs[3];
    int cast_count = 0;
    cast_funcs[cast_count++] = (me_variable){
        .name = "int",
        .dtype = dsl_cast_int_target_dtype(expr_dtype),
        .address = (const void *)dsl_cast_int_intrinsic,
        .type = ME_FUNCTION1 | ME_FLAG_PURE,
        .context = NULL,
        .itemsize = 0
    };
    cast_funcs[cast_count++] = (me_variable){
        .name = "float",
        .dtype = dsl_cast_float_target_dtype(expr_dtype),
        .address = (const void *)dsl_cast_float_intrinsic,
        .type = ME_FUNCTION1 | ME_FLAG_PURE,
        .context = NULL,
        .itemsize = 0
    };
    cast_funcs[cast_count++] = (me_variable){
        .name = "bool",
        .dtype = ME_BOOL,
        .address = (const void *)dsl_cast_bool_intrinsic,
        .type = ME_FUNCTION1 | ME_FLAG_PURE,
        .context = NULL,
        .itemsize = 0
    };

    me_variable *all_funcs = NULL;
    int all_func_count = ctx->func_count + cast_count;
    if (all_func_count > 0) {
        all_funcs = calloc((size_t)all_func_count, sizeof(*all_funcs));
        if (!all_funcs) {
            return false;
        }
        for (int i = 0; i < ctx->func_count; i++) {
            all_funcs[i] = ctx->funcs[i];
        }
        for (int i = 0; i < cast_count; i++) {
            all_funcs[ctx->func_count + i] = cast_funcs[i];
        }
    }

    me_variable *lookup = NULL;
    int lookup_count = 0;
    if (!dsl_build_var_lookup(&ctx->program->vars, all_funcs, all_func_count,
                              &lookup, &lookup_count)) {
        free(all_funcs);
        return false;
    }
    free(all_funcs);
    me_expr *compiled = NULL;
    int local_error = 0;
    int rc = private_compile_ex(expr_node->text, lookup, lookup_count,
                                NULL, 0, expr_dtype, &local_error, &compiled);
    free(lookup);
    if (rc != ME_COMPILE_SUCCESS || !compiled) {
        dsl_set_error_reason(ctx, "failed to compile DSL expression");
        if (ctx->error_pos) {
            int offset = dsl_offset_from_linecol(ctx->source, expr_node->line, expr_node->column);
            if (offset >= 0 && local_error > 0) {
                *ctx->error_pos = offset + local_error - 1;
            }
            else {
                *ctx->error_pos = offset >= 0 ? offset : -1;
            }
        }
        if (compiled) {
            me_free(compiled);
        }
        return false;
    }
    int *indices = NULL;
    int count = 0;
    if (!dsl_collect_var_indices(compiled, &indices, &count)) {
        me_free(compiled);
        free(indices);
        return false;
    }
    out_expr->expr = compiled;
    out_expr->var_indices = indices;
    out_expr->n_vars = count;
    return true;
}

static bool dsl_compile_condition_expr(dsl_compile_ctx *ctx, const me_dsl_expr *expr_node,
                                       me_dsl_compiled_expr *out_expr) {
    if (!ctx || !expr_node || !expr_node->text || !out_expr) {
        return false;
    }
    int saved_error = 0;
    if (ctx->error_pos) {
        saved_error = *ctx->error_pos;
    }

    if (dsl_compile_expr(ctx, expr_node, ME_AUTO, out_expr)) {
        return true;
    }

    if (ctx->error_pos) {
        *ctx->error_pos = saved_error;
    }
    size_t expr_len = strlen(expr_node->text);
    size_t need = expr_len + 16;
    char *truthy_text = malloc(need);
    if (!truthy_text) {
        return false;
    }
    snprintf(truthy_text, need, "(%s) != \"\"", expr_node->text);
    me_dsl_expr truthy_expr = {
        .text = truthy_text,
        .line = expr_node->line,
        .column = expr_node->column
    };
    bool ok = dsl_compile_expr(ctx, &truthy_expr, ME_BOOL, out_expr);
    free(truthy_text);
    return ok;
}

static bool dsl_split_top_level_csv(const char *text, char ***out_parts, int *out_nparts) {
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
            char *part = dsl_trim_copy(part_start, p);
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

    char *tail = dsl_trim_copy(part_start, p);
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

static bool dsl_compile_range_bound_expr(dsl_compile_ctx *ctx, const char *text,
                                         int line, int column,
                                         me_dsl_compiled_expr *out_expr) {
    if (!ctx || !text || !out_expr) {
        return false;
    }
    me_dsl_expr expr = {
        .text = (char *)text,
        .line = line,
        .column = column
    };
    return dsl_compile_expr(ctx, &expr, ME_AUTO, out_expr);
}

static bool dsl_compile_for_range_args(dsl_compile_ctx *ctx, const me_dsl_stmt *stmt,
                                       me_dsl_compiled_stmt *compiled) {
    if (!ctx || !stmt || !compiled || !stmt->as.for_loop.limit || !stmt->as.for_loop.limit->text) {
        return false;
    }

    char **parts = NULL;
    int nparts = 0;
    if (!dsl_split_top_level_csv(stmt->as.for_loop.limit->text, &parts, &nparts)) {
        if (ctx->error_pos) {
            *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
        }
        return false;
    }
    if (nparts < 1 || nparts > 3) {
        if (ctx->error_pos) {
            *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
        }
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }

    const char *start_text = "0";
    const char *stop_text = parts[0];
    const char *step_text = "1";
    if (nparts == 2) {
        start_text = parts[0];
        stop_text = parts[1];
    }
    else if (nparts == 3) {
        start_text = parts[0];
        stop_text = parts[1];
        step_text = parts[2];
    }

    bool ok = dsl_compile_range_bound_expr(ctx, start_text, stmt->line, stmt->column,
                                           &compiled->as.for_loop.start) &&
              dsl_compile_range_bound_expr(ctx, stop_text, stmt->line, stmt->column,
                                           &compiled->as.for_loop.stop) &&
              dsl_compile_range_bound_expr(ctx, step_text, stmt->line, stmt->column,
                                           &compiled->as.for_loop.step);
    if (!ok) {
        dsl_compiled_expr_free(&compiled->as.for_loop.start);
        dsl_compiled_expr_free(&compiled->as.for_loop.stop);
        dsl_compiled_expr_free(&compiled->as.for_loop.step);
    }

    for (int i = 0; i < nparts; i++) {
        free(parts[i]);
    }
    free(parts);
    return ok;
}

static bool dsl_jit_ir_resolve_dtype(void *resolve_ctx, const me_dsl_expr *expr,
                                     me_dsl_jit_ir_resolve_mode mode,
                                     me_dtype *out_dtype) {
    dsl_compile_ctx *ctx = (dsl_compile_ctx *)resolve_ctx;
    if (!ctx || !expr || !out_dtype) {
        return false;
    }
    me_dsl_compiled_expr compiled_expr;
    me_dtype expr_dtype = ME_AUTO;
    if (mode == ME_DSL_JIT_IR_RESOLVE_OUTPUT) {
        expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
    }
    int saved_error = 0;
    if (ctx->error_pos) {
        saved_error = *ctx->error_pos;
    }
    if (!dsl_compile_expr(ctx, expr, expr_dtype, &compiled_expr)) {
        if (ctx->error_pos) {
            *ctx->error_pos = saved_error;
        }
        return false;
    }
    *out_dtype = me_get_dtype(compiled_expr.expr);
    dsl_compiled_expr_free(&compiled_expr);
    if (ctx->error_pos) {
        *ctx->error_pos = saved_error;
    }
    return true;
}

static bool dsl_jit_tcc_reserved_index_mix_auto_disabled(const me_dsl_compiled_program *program) {
    if (!program || program->compiler != ME_DSL_COMPILER_LIBTCC) {
        return false;
    }
    if (!program->uses_flat_idx) {
        return false;
    }
    if ((program->uses_i_mask == 0) && (program->uses_n_mask == 0) && !program->uses_ndim) {
        return false;
    }
    return true;
}

static bool dsl_compiled_block_guarantees_return(const me_dsl_compiled_block *block);

static bool dsl_compiled_stmt_guarantees_return(const me_dsl_compiled_stmt *stmt) {
    if (!stmt) {
        return false;
    }
    if (stmt->kind == ME_DSL_STMT_RETURN) {
        return true;
    }
    if (stmt->kind != ME_DSL_STMT_IF) {
        return false;
    }
    if (!stmt->as.if_stmt.has_else) {
        return false;
    }
    if (!dsl_compiled_block_guarantees_return(&stmt->as.if_stmt.then_block)) {
        return false;
    }
    for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
        if (!dsl_compiled_block_guarantees_return(&stmt->as.if_stmt.elif_branches[i].block)) {
            return false;
        }
    }
    return dsl_compiled_block_guarantees_return(&stmt->as.if_stmt.else_block);
}

static bool dsl_compiled_block_guarantees_return(const me_dsl_compiled_block *block) {
    if (!block) {
        return false;
    }
    for (int i = 0; i < block->nstmts; i++) {
        if (dsl_compiled_stmt_guarantees_return(block->stmts[i])) {
            return true;
        }
    }
    return false;
}

static void dsl_try_build_jit_ir(dsl_compile_ctx *ctx, const me_dsl_program *parsed,
                                 me_dsl_compiled_program *program,
                                 bool prepare_runtime) {
    if (!ctx || !parsed || !program) {
        return;
    }

    program->jit_ir = NULL;
    program->jit_ir_fingerprint = 0;
    program->jit_ir_error_line = 0;
    program->jit_ir_error_column = 0;
    program->jit_ir_error[0] = '\0';
    free(program->jit_param_bindings);
    program->jit_param_bindings = NULL;
    program->jit_nparams = 0;
    program->jit_kernel_fn = NULL;
#if ME_USE_LIBTCC_FALLBACK
    if (program->jit_tcc_state) {
        dsl_jit_libtcc_delete_state(program->jit_tcc_state);
        program->jit_tcc_state = NULL;
    }
#endif
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
    if (program->jit_dl_handle) {
        if (!program->jit_dl_handle_cached) {
            dlclose(program->jit_dl_handle);
        }
    }
#endif
    program->jit_dl_handle = NULL;
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    free(program->jit_c_source);
    program->jit_c_source = NULL;
    program->jit_use_runtime_math_bridge = false;
    program->jit_scalar_math_bridge_enabled = false;
    program->jit_synth_reserved_non_nd = false;
    program->jit_synth_reserved_nd = false;
    program->jit_vec_math_enabled = false;
    program->jit_branch_aware_if_lowering_enabled = false;
    program->jit_hybrid_expr_vec_math_enabled = false;
    program->jit_c_error_line = 0;
    program->jit_c_error_column = 0;
    program->jit_c_error[0] = '\0';
    program->jit_lowering_mode[0] = '\0';
    program->jit_vector_ops[0] = '\0';
    program->jit_lowering_reason[0] = '\0';

    if (!program->guaranteed_return) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "program may reach function end without return");
        dsl_tracef("jit ir skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
        return;
    }

    if (parsed->nparams < 0) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "invalid dsl parameter metadata");
        dsl_tracef("jit ir skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
        return;
    }

    bool uses_reserved_index_vars = (program->uses_i_mask != 0) ||
                                    (program->uses_n_mask != 0) ||
                                    program->uses_ndim ||
                                    program->uses_flat_idx;
    if (uses_reserved_index_vars && !dsl_jit_index_vars_enabled()) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "reserved index vars disabled by ME_DSL_JIT_INDEX_VARS");
        dsl_tracef("jit ir skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
        return;
    }
    if (uses_reserved_index_vars && dsl_jit_tcc_reserved_index_mix_auto_disabled(program)) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "tcc mixed reserved index vars auto-disabled");
        dsl_tracef("jit ir skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
        return;
    }
    if (uses_reserved_index_vars) {
        int compile_ndims = program->compile_ndims;
        if (compile_ndims <= 0) {
            program->jit_synth_reserved_non_nd = true;
        }
        else {
            program->jit_synth_reserved_nd = true;
        }
    }

    const char **param_names = NULL;
    me_dtype *param_dtypes = NULL;
    me_dsl_jit_param_binding *param_bindings = NULL;
    int jit_param_count = parsed->nparams;
    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
        if ((program->uses_i_mask & (1 << d)) && program->idx_i[d] >= 0) {
            jit_param_count++;
        }
        if ((program->uses_n_mask & (1 << d)) && program->idx_n[d] >= 0) {
            jit_param_count++;
        }
    }
    if (program->uses_ndim && program->idx_ndim >= 0) {
        jit_param_count++;
    }
    if (program->uses_flat_idx && program->idx_flat_idx >= 0) {
        jit_param_count++;
    }
    if (program->jit_synth_reserved_nd) {
        jit_param_count++;
    }
    if (jit_param_count > ME_MAX_VARS) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "jit parameter metadata exceeds ME_MAX_VARS");
        dsl_tracef("jit ir skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
        return;
    }

    if (jit_param_count > 0) {
        param_names = calloc((size_t)jit_param_count, sizeof(*param_names));
        param_dtypes = calloc((size_t)jit_param_count, sizeof(*param_dtypes));
        param_bindings = calloc((size_t)jit_param_count, sizeof(*param_bindings));
        if (!param_names || !param_dtypes || !param_bindings) {
            snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                     "out of memory building jit ir metadata");
            dsl_tracef("jit ir skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
            free(param_names);
            free(param_dtypes);
            free(param_bindings);
            return;
        }
    }

    int nparams = 0;
    for (int i = 0; i < parsed->nparams; i++) {
        int idx = dsl_var_table_find(&program->vars, parsed->params[i]);
        if (idx < 0 || idx >= program->vars.count) {
            snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                     "failed to resolve dsl parameter dtype for jit ir");
            dsl_tracef("jit ir skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
            free(param_names);
            free(param_dtypes);
            free(param_bindings);
            return;
        }
        param_names[nparams] = program->vars.names[idx];
        param_dtypes[nparams] = program->vars.dtypes[idx];
        param_bindings[nparams].var_index = idx;
        param_bindings[nparams].dim = -1;
        param_bindings[nparams].kind = ME_DSL_JIT_BIND_USER_INPUT;
        nparams++;
    }
    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
        if ((program->uses_i_mask & (1 << d)) && program->idx_i[d] >= 0) {
            int idx = program->idx_i[d];
            param_names[nparams] = program->vars.names[idx];
            param_dtypes[nparams] = program->vars.dtypes[idx];
            param_bindings[nparams].var_index = idx;
            param_bindings[nparams].dim = d;
            param_bindings[nparams].kind = ME_DSL_JIT_BIND_RESERVED_I;
            nparams++;
        }
    }
    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
        if ((program->uses_n_mask & (1 << d)) && program->idx_n[d] >= 0) {
            int idx = program->idx_n[d];
            param_names[nparams] = program->vars.names[idx];
            param_dtypes[nparams] = program->vars.dtypes[idx];
            param_bindings[nparams].var_index = idx;
            param_bindings[nparams].dim = d;
            param_bindings[nparams].kind = ME_DSL_JIT_BIND_RESERVED_N;
            nparams++;
        }
    }
    if (program->uses_ndim && program->idx_ndim >= 0) {
        int idx = program->idx_ndim;
        param_names[nparams] = program->vars.names[idx];
        param_dtypes[nparams] = program->vars.dtypes[idx];
        param_bindings[nparams].var_index = idx;
        param_bindings[nparams].dim = -1;
        param_bindings[nparams].kind = ME_DSL_JIT_BIND_RESERVED_NDIM;
        nparams++;
    }
    if (program->uses_flat_idx && program->idx_flat_idx >= 0) {
        int idx = program->idx_flat_idx;
        param_names[nparams] = program->vars.names[idx];
        param_dtypes[nparams] = program->vars.dtypes[idx];
        param_bindings[nparams].var_index = idx;
        param_bindings[nparams].dim = -1;
        param_bindings[nparams].kind = ME_DSL_JIT_BIND_RESERVED_GLOBAL_LINEAR_IDX;
        nparams++;
    }
    if (program->jit_synth_reserved_nd) {
        param_names[nparams] = ME_DSL_JIT_SYNTH_ND_CTX_PARAM;
        param_dtypes[nparams] = ME_INT64;
        param_bindings[nparams].var_index = -1;
        param_bindings[nparams].dim = -1;
        param_bindings[nparams].kind = ME_DSL_JIT_BIND_SYNTH_ND_CTX;
        nparams++;
    }

    me_dsl_error ir_error;
    memset(&ir_error, 0, sizeof(ir_error));
    me_dsl_jit_ir_program *jit_ir = NULL;
    bool ok = me_dsl_jit_ir_build(parsed, param_names, param_dtypes, nparams,
                                  dsl_jit_ir_resolve_dtype, ctx, &jit_ir, &ir_error);

    free(param_names);
    free(param_dtypes);

    if (!ok || !jit_ir) {
        program->jit_ir_error_line = ir_error.line;
        program->jit_ir_error_column = ir_error.column;
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 ir_error.message[0] ? ir_error.message : "jit ir build rejected");
        dsl_tracef("jit ir reject: fp=%s at %d:%d reason=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   program->jit_ir_error_line, program->jit_ir_error_column,
                   program->jit_ir_error);
        me_dsl_jit_ir_free(jit_ir);
        free(param_bindings);
        return;
    }

    program->jit_ir = jit_ir;
    program->jit_ir_fingerprint = me_dsl_jit_ir_fingerprint(jit_ir);
    program->jit_param_bindings = param_bindings;
    program->jit_nparams = nparams;

    me_dsl_error cg_error;
    memset(&cg_error, 0, sizeof(cg_error));
    me_dsl_jit_cgen_options cg_options;
    memset(&cg_options, 0, sizeof(cg_options));
    cg_options.symbol_name = ME_DSL_JIT_SYMBOL_NAME;
    cg_options.synth_reserved_non_nd = program->jit_synth_reserved_non_nd;
    cg_options.synth_reserved_nd = program->jit_synth_reserved_nd;
    cg_options.synth_nd_ctx_name = ME_DSL_JIT_SYNTH_ND_CTX_PARAM;
    cg_options.synth_nd_compile_ndims = program->compile_ndims;
    char lowering_mode[16] = {0};
    char vector_ops[128] = {0};
    char lowering_reason[128] = {0};
    cg_options.trace_lowering_mode = lowering_mode;
    cg_options.trace_lowering_mode_cap = sizeof(lowering_mode);
    cg_options.trace_vector_ops = vector_ops;
    cg_options.trace_vector_ops_cap = sizeof(vector_ops);
    cg_options.trace_lowering_reason = lowering_reason;
    cg_options.trace_lowering_reason_cap = sizeof(lowering_reason);
    cg_options.has_enable_scalar_math_bridge = true;
    cg_options.enable_scalar_math_bridge = dsl_jit_scalar_math_bridge_enabled();
    cg_options.has_enable_vector_math = true;
    cg_options.enable_vector_math = dsl_jit_vec_math_enabled();
    cg_options.has_enable_hybrid_vector_math = true;
    cg_options.enable_hybrid_vector_math = cg_options.enable_vector_math;
    cg_options.has_enable_hybrid_expr_vector_math = true;
    cg_options.enable_hybrid_expr_vector_math = dsl_jit_hybrid_expr_vector_math_enabled();
    cg_options.has_enable_branch_aware_if_lowering = true;
    cg_options.enable_branch_aware_if_lowering = dsl_jit_branch_aware_if_lowering_enabled();
    if (program->compiler == ME_DSL_COMPILER_LIBTCC) {
        cg_options.enable_hybrid_vector_math = false;
    }
    bool use_bridge = false;
    bool bridge_gate_enabled = dsl_jit_math_bridge_enabled();
    bool cc_bridge_forced = false;
    if (bridge_gate_enabled && program->compiler == ME_DSL_COMPILER_CC) {
        const char *scalar_env = getenv("ME_DSL_JIT_SCALAR_MATH_BRIDGE");
        const char *vec_env = getenv("ME_DSL_JIT_VEC_MATH");
        const char *expr_vec_env = getenv("ME_DSL_JIT_HYBRID_EXPR_VEC_MATH");
        cc_bridge_forced = (scalar_env && cg_options.enable_scalar_math_bridge) ||
            (vec_env && cg_options.enable_vector_math) ||
            (expr_vec_env && cg_options.enable_hybrid_expr_vector_math);
    }
    if (bridge_gate_enabled && program->compiler == ME_DSL_COMPILER_LIBTCC) {
        use_bridge = true;
    }
    else if (bridge_gate_enabled && program->compiler == ME_DSL_COMPILER_CC) {
        if (dsl_jit_cc_math_bridge_available() || cc_bridge_forced) {
            use_bridge = true;
        }
        else {
            dsl_tracef("jit codegen: runtime math bridge unavailable for cc backend");
        }
    }
    else if (!bridge_gate_enabled) {
        dsl_tracef("jit codegen: runtime math bridge disabled by ME_DSL_JIT_MATH_BRIDGE");
    }
    if (program->compiler == ME_DSL_COMPILER_LIBTCC &&
        cg_options.enable_vector_math &&
        !cg_options.enable_hybrid_vector_math) {
        dsl_tracef("jit codegen: hybrid statement vector lowering disabled for tcc backend");
    }
    cg_options.use_runtime_math_bridge = use_bridge;
    char *generated_c = NULL;
    bool cg_ok = me_dsl_jit_codegen_c(jit_ir, ctx->return_dtype, &cg_options,
                                      &generated_c, &cg_error);
    if (!cg_ok || !generated_c) {
        program->jit_c_error_line = cg_error.line;
        program->jit_c_error_column = cg_error.column;
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 cg_error.message[0] ? cg_error.message : "jit c codegen rejected");
        dsl_tracef("jit codegen reject: fp=%s at %d:%d reason=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   program->jit_c_error_line, program->jit_c_error_column,
                   program->jit_c_error);
        free(generated_c);
        free(program->jit_param_bindings);
        program->jit_param_bindings = NULL;
        program->jit_nparams = 0;
        return;
    }
    program->jit_c_source = generated_c;
    program->jit_use_runtime_math_bridge = cg_options.use_runtime_math_bridge;
    program->jit_scalar_math_bridge_enabled = cg_options.use_runtime_math_bridge &&
                                              cg_options.enable_scalar_math_bridge;
    program->jit_vec_math_enabled = cg_options.use_runtime_math_bridge && cg_options.enable_vector_math;
    program->jit_branch_aware_if_lowering_enabled = cg_options.enable_branch_aware_if_lowering;
    program->jit_hybrid_expr_vec_math_enabled = cg_options.use_runtime_math_bridge &&
                                                cg_options.enable_vector_math &&
                                                cg_options.enable_hybrid_vector_math &&
                                                cg_options.enable_hybrid_expr_vector_math;
    snprintf(program->jit_lowering_mode, sizeof(program->jit_lowering_mode), "%s", lowering_mode);
    snprintf(program->jit_vector_ops, sizeof(program->jit_vector_ops), "%s", vector_ops);
    snprintf(program->jit_lowering_reason, sizeof(program->jit_lowering_reason), "%s", lowering_reason);
    if (program->jit_use_runtime_math_bridge) {
        dsl_tracef("jit codegen: runtime math bridge enabled (scalar=%s vec=%s expr=%s if=%s)",
                   program->jit_scalar_math_bridge_enabled ? "bridge" : "libm",
                   program->jit_vec_math_enabled ? "on" : "off",
                   program->jit_hybrid_expr_vec_math_enabled ? "on" : "off",
                   program->jit_branch_aware_if_lowering_enabled ? "select" : "if");
    }
    dsl_tracef("jit codegen: lowering=%s vec_ops=%s reason=%s",
               program->jit_lowering_mode[0] ? program->jit_lowering_mode : "scalar",
               program->jit_vector_ops[0] ? program->jit_vector_ops : "-",
               program->jit_lowering_reason[0] ? program->jit_lowering_reason : "-");
    dsl_tracef("jit ir built: fp=%s compiler=%s fingerprint=%016llx",
               dsl_fp_mode_name(program->fp_mode),
               dsl_compiler_name(program->compiler),
               (unsigned long long)program->jit_ir_fingerprint);
    if (prepare_runtime) {
        dsl_try_prepare_jit_runtime(program);
    }
    else {
        dsl_tracef("jit runtime skipped: fp=%s compiler=%s reason=jit_mode=off",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler));
    }
}

static bool dsl_compile_block(dsl_compile_ctx *ctx, const me_dsl_block *block,
                              me_dsl_compiled_block *out_block);

static bool dsl_compile_block(dsl_compile_ctx *ctx, const me_dsl_block *block,
                              me_dsl_compiled_block *out_block) {
    if (!ctx || !block || !out_block) {
        return false;
    }
    memset(out_block, 0, sizeof(*out_block));
    for (int i = 0; i < block->nstmts; i++) {
        me_dsl_stmt *stmt = block->stmts[i];
        if (!stmt) {
            continue;
        }
        me_dsl_compiled_stmt *compiled = calloc(1, sizeof(*compiled));
        if (!compiled) {
            return false;
        }
        compiled->kind = stmt->kind;
        compiled->line = stmt->line;
        compiled->column = stmt->column;

        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN: {
            const char *name = stmt->as.assign.name;
            if (!name) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (dsl_is_reserved_name(name)) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            int var_index = dsl_var_table_find(&ctx->program->vars, name);
            if (var_index >= 0 && var_index < ctx->program->n_inputs) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            me_dtype assigned_dtype = ME_AUTO;
            bool rhs_compiled = false;

            if (var_index >= 0 && var_index < ctx->program->vars.count) {
                me_dtype expr_dtype = ctx->program->vars.dtypes[var_index];
                if (!dsl_compile_expr(ctx, stmt->as.assign.value, expr_dtype, &compiled->as.assign.value)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                assigned_dtype = me_get_dtype(compiled->as.assign.value.expr);
                rhs_compiled = true;
            }
            else if (!ctx->output_dtype_auto && dsl_dtype_is_integer(ctx->output_dtype)) {
                me_dsl_compiled_expr probe_expr;
                memset(&probe_expr, 0, sizeof(probe_expr));
                int saved_error = 0;
                if (ctx->error_pos) {
                    saved_error = *ctx->error_pos;
                }
                if (dsl_compile_expr(ctx, stmt->as.assign.value, ME_AUTO, &probe_expr)) {
                    me_dtype probe_dtype = me_get_dtype(probe_expr.expr);
                    if (dsl_dtype_is_float_or_complex(probe_dtype)) {
                        compiled->as.assign.value = probe_expr;
                        assigned_dtype = probe_dtype;
                        rhs_compiled = true;
                    }
                    else {
                        dsl_compiled_expr_free(&probe_expr);
                    }
                }
                else if (ctx->error_pos) {
                    *ctx->error_pos = saved_error;
                }
            }

            if (!rhs_compiled) {
                me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
                if (!dsl_compile_expr(ctx, stmt->as.assign.value, expr_dtype, &compiled->as.assign.value)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                assigned_dtype = me_get_dtype(compiled->as.assign.value.expr);
            }

            bool is_uniform = dsl_expr_is_uniform(compiled->as.assign.value.expr,
                                                  ctx->program->vars.uniform,
                                                  ctx->program->vars.count);

            if (var_index < 0) {
                if (!ctx->allow_new_locals) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                var_index = dsl_var_table_add_with_uniform(&ctx->program->vars, name,
                                                           assigned_dtype, 0, is_uniform);
                if (var_index < 0) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            else if (ctx->program->vars.dtypes[var_index] != assigned_dtype) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            else {
                ctx->program->vars.uniform[var_index] = is_uniform;
            }

            if (!dsl_program_add_local(ctx->program, var_index)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            compiled->as.assign.local_slot = ctx->program->local_slots[var_index];
            break;
        }
        case ME_DSL_STMT_EXPR: {
            me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
            if (!dsl_compile_expr(ctx, stmt->as.expr_stmt.expr, expr_dtype, &compiled->as.expr_stmt.expr)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            break;
        }
        case ME_DSL_STMT_RETURN: {
            me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
            if (!dsl_compile_expr(ctx, stmt->as.return_stmt.expr, expr_dtype, &compiled->as.return_stmt.expr)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            me_dtype return_dtype = me_get_dtype(compiled->as.return_stmt.expr.expr);
            if (!ctx->has_return) {
                ctx->has_return = true;
                ctx->return_dtype = return_dtype;
                ctx->output_expr = &compiled->as.return_stmt.expr;
            }
            else if (ctx->return_dtype != return_dtype) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            break;
        }
        case ME_DSL_STMT_PRINT: {
            const char *call = stmt->as.print_stmt.call ? stmt->as.print_stmt.call->text : NULL;
            char **args = NULL;
            int nargs = 0;
            if (!call || !dsl_split_print_args(call, &args, &nargs) || nargs < 1) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            char *format = NULL;
            int arg_count = nargs - 1;
            bool first_is_string = false;
            if (args[0][0] == '"' || args[0][0] == '\'') {
                format = dsl_unescape_string_literal(args[0]);
                if (!format) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                first_is_string = true;
            }
            if (!first_is_string) {
                size_t fmt_len = 0;
                for (int i = 0; i < nargs; i++) {
                    fmt_len += (i == 0) ? 2 : 3;
                }
                format = malloc(fmt_len + 1);
                if (!format) {
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                char *out = format;
                for (int i = 0; i < nargs; i++) {
                    if (i > 0) {
                        *out++ = ' ';
                    }
                    *out++ = '{';
                    *out++ = '}';
                }
                *out = '\0';
                arg_count = nargs;
            }
            int placeholder_count = dsl_count_placeholders(format);
            if (placeholder_count < 0) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                free(format);
                for (int i = 0; i < nargs; i++) {
                    free(args[i]);
                }
                free(args);
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (first_is_string && placeholder_count == 0 && arg_count > 0) {
                bool needs_space = format[0] != '\0' && !isspace((unsigned char)format[strlen(format) - 1]);
                size_t fmt_len = strlen(format);
                size_t extra = (needs_space ? 1 : 0) + (size_t)(arg_count * 2) + (size_t)(arg_count - 1);
                char *expanded = malloc(fmt_len + extra + 1);
                if (!expanded) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    free(format);
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                memcpy(expanded, format, fmt_len);
                char *out = expanded + fmt_len;
                if (needs_space) {
                    *out++ = ' ';
                }
                for (int i = 0; i < arg_count; i++) {
                    if (i > 0) {
                        *out++ = ' ';
                    }
                    *out++ = '{';
                    *out++ = '}';
                }
                *out = '\0';
                free(format);
                format = expanded;
                placeholder_count = arg_count;
            }
            if (placeholder_count != arg_count) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                free(format);
                for (int i = 0; i < nargs; i++) {
                    free(args[i]);
                }
                free(args);
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            if (arg_count > 0) {
                compiled->as.print_stmt.args = calloc((size_t)arg_count, sizeof(*compiled->as.print_stmt.args));
                if (!compiled->as.print_stmt.args) {
                    free(format);
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            compiled->as.print_stmt.format = format;
            compiled->as.print_stmt.nargs = arg_count;

            for (int i = 0; i < arg_count; i++) {
                int arg_index = first_is_string ? (i + 1) : i;
                me_dsl_expr temp_expr = {
                    .text = args[arg_index],
                    .line = stmt->line,
                    .column = stmt->column
                };
                me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
                if (!dsl_compile_expr(ctx, &temp_expr, expr_dtype, &compiled->as.print_stmt.args[i])) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    dsl_compiled_stmt_free(compiled);
                    for (int j = 0; j < nargs; j++) {
                        free(args[j]);
                    }
                    free(args);
                    return false;
                }
                if (!dsl_expr_is_uniform(compiled->as.print_stmt.args[i].expr,
                                         ctx->program->vars.uniform,
                                         ctx->program->vars.count)) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    dsl_compiled_stmt_free(compiled);
                    for (int j = 0; j < nargs; j++) {
                        free(args[j]);
                    }
                    free(args);
                    return false;
                }
            }
            for (int i = 0; i < nargs; i++) {
                free(args[i]);
            }
            free(args);
            break;
        }
        case ME_DSL_STMT_IF: {
            if (!dsl_compile_condition_expr(ctx, stmt->as.if_stmt.cond, &compiled->as.if_stmt.cond)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            if (!dsl_compile_block(ctx, &stmt->as.if_stmt.then_block,
                                   &compiled->as.if_stmt.then_block)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            compiled->as.if_stmt.n_elifs = stmt->as.if_stmt.n_elifs;
            compiled->as.if_stmt.elif_capacity = stmt->as.if_stmt.n_elifs;
            if (compiled->as.if_stmt.n_elifs > 0) {
                compiled->as.if_stmt.elif_branches = calloc((size_t)compiled->as.if_stmt.n_elifs,
                                                            sizeof(*compiled->as.if_stmt.elif_branches));
                if (!compiled->as.if_stmt.elif_branches) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
                me_dsl_if_branch *elif_branch = &stmt->as.if_stmt.elif_branches[i];
                me_dsl_compiled_if_branch *out_branch = &compiled->as.if_stmt.elif_branches[i];
                if (!dsl_compile_condition_expr(ctx, elif_branch->cond, &out_branch->cond)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                if (!dsl_compile_block(ctx, &elif_branch->block, &out_branch->block)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }

            if (stmt->as.if_stmt.has_else) {
                if (!dsl_compile_block(ctx, &stmt->as.if_stmt.else_block,
                                       &compiled->as.if_stmt.else_block)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                compiled->as.if_stmt.has_else = true;
            }
            break;
        }
        case ME_DSL_STMT_WHILE: {
            if (!dsl_compile_condition_expr(ctx, stmt->as.while_loop.cond, &compiled->as.while_loop.cond)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth++;
            if (!dsl_compile_block(ctx, &stmt->as.while_loop.body, &compiled->as.while_loop.body)) {
                ctx->loop_depth--;
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth--;
            break;
        }
        case ME_DSL_STMT_FOR: {
            const char *var = stmt->as.for_loop.var;
            if (!var || dsl_is_reserved_name(var)) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (!ctx->allow_new_locals) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            int var_index = dsl_var_table_find(&ctx->program->vars, var);
            if (var_index >= 0) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            var_index = dsl_var_table_add_with_uniform(&ctx->program->vars, var, ME_INT64, 0, true);
            if (var_index < 0) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (!dsl_program_add_local(ctx->program, var_index)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            compiled->as.for_loop.loop_var_slot = ctx->program->local_slots[var_index];

            if (!dsl_compile_for_range_args(ctx, stmt, compiled)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth++;
            if (!dsl_compile_block(ctx, &stmt->as.for_loop.body, &compiled->as.for_loop.body)) {
                ctx->loop_depth--;
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth--;
            break;
        }
        case ME_DSL_STMT_BREAK:
        case ME_DSL_STMT_CONTINUE: {
            if (ctx->loop_depth <= 0) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (stmt->as.flow.cond) {
                if (!dsl_compile_condition_expr(ctx, stmt->as.flow.cond, &compiled->as.flow.cond)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            else {
                memset(&compiled->as.flow.cond, 0, sizeof(compiled->as.flow.cond));
            }
            break;
        }
        }

        if (!dsl_compiled_block_push(out_block, compiled)) {
            dsl_compiled_stmt_free(compiled);
            return false;
        }
    }
    return true;
}

me_dsl_compiled_program *dsl_compile_program(const char *source,
                                             const me_variable *variables,
                                             int var_count,
                                             me_dtype dtype,
                                             int compile_ndims,
                                             int jit_mode,
                                             int *error_pos,
                                             bool *is_dsl,
                                             char *error_reason,
                                             size_t error_reason_cap) {
    me_dsl_error parse_error;
    if (error_reason && error_reason_cap > 0) {
        error_reason[0] = '\0';
    }
    if (is_dsl) {
        *is_dsl = false;
    }
    me_dsl_program *parsed = me_dsl_parse(source, &parse_error);
    if (!parsed) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap, "dsl parse error: %s",
                     parse_error.message[0] ? parse_error.message : "unknown parse error");
        }
        if (error_pos) {
            int off = dsl_offset_from_linecol(source, parse_error.line, parse_error.column);
            *error_pos = off >= 0 ? off : -1;
        }
        if (is_dsl) {
            *is_dsl = true;
        }
        return NULL;
    }
    if (!dsl_program_is_dsl(parsed)) {
        me_dsl_program_free(parsed);
        return NULL;
    }
    if (is_dsl) {
        *is_dsl = true;
    }

    me_dsl_compiled_program *program = dsl_compiled_program_alloc(parsed, source, compile_ndims,
                                                                  error_reason, error_reason_cap);
    if (!program) {
        me_dsl_program_free(parsed);
        if (error_pos) {
            *error_pos = -1;
        }
        return NULL;
    }
    if (jit_mode == ME_JIT_ON || jit_mode == ME_JIT_OFF) {
        program->jit_request_mode = (me_jit_mode)jit_mode;
    }
    else {
        program->jit_request_mode = ME_JIT_DEFAULT;
    }

    me_variable *funcs = NULL;
    int func_count = 0;
    int func_capacity = 0;
    int input_count = 0;

    for (int i = 0; i < var_count; i++) {
        const me_variable *entry = &variables[i];
        const char *name = entry->name;
        if (!name) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap, "invalid DSL input: variable name is NULL");
            }
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
        if (!is_variable_entry(entry) && !is_function_entry(entry)) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap, "invalid DSL input entry type for '%s'", name);
            }
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            return NULL;
        }
        if (is_function_entry(entry)) {
            size_t name_len = strlen(name);
            if (dsl_is_reserved_name(name) || me_is_builtin_function_name(name, name_len)) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL function name '%s' is reserved or collides with a builtin", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            for (int j = 0; j < parsed->nparams; j++) {
                if (strcmp(parsed->params[j], name) == 0) {
                    if (error_reason && error_reason_cap > 0) {
                        snprintf(error_reason, error_reason_cap,
                                 "DSL function name '%s' collides with a parameter name", name);
                    }
                    if (error_pos) {
                        *error_pos = -1;
                    }
                    free(funcs);
                    dsl_compiled_program_free(program);
                    me_dsl_program_free(parsed);
                    return NULL;
                }
            }
            if (entry->dtype == ME_AUTO || !is_valid_dtype(entry->dtype) || entry->dtype == ME_STRING) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL function '%s' has unsupported return dtype", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            if (!entry->address) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL function '%s' has NULL address", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            if (dsl_var_table_find(&program->vars, name) >= 0) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "duplicate DSL symbol '%s' in input/function table", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            for (int j = 0; j < func_count; j++) {
                if (strcmp(funcs[j].name, name) == 0) {
                    if (error_reason && error_reason_cap > 0) {
                        snprintf(error_reason, error_reason_cap,
                                 "duplicate DSL function name '%s'", name);
                    }
                    if (error_pos) {
                        *error_pos = -1;
                    }
                    free(funcs);
                    dsl_compiled_program_free(program);
                    me_dsl_program_free(parsed);
                    return NULL;
                }
            }
            if (func_count == func_capacity) {
                int new_cap = func_capacity ? func_capacity * 2 : 8;
                me_variable *next = realloc(funcs, (size_t)new_cap * sizeof(*next));
                if (!next) {
                    if (error_reason && error_reason_cap > 0) {
                        snprintf(error_reason, error_reason_cap,
                                 "out of memory while storing DSL function symbols");
                    }
                    if (error_pos) {
                        *error_pos = -1;
                    }
                    free(funcs);
                    dsl_compiled_program_free(program);
                    me_dsl_program_free(parsed);
                    return NULL;
                }
                funcs = next;
                func_capacity = new_cap;
            }
            funcs[func_count++] = *entry;
            continue;
        }
        if (dsl_is_reserved_name(name)) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "DSL input variable '%s' uses a reserved name", name);
            }
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
        for (int j = 0; j < func_count; j++) {
            if (strcmp(funcs[j].name, name) == 0) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL input variable '%s' collides with a function name", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
        me_dtype vtype = entry->dtype;
        if (vtype == ME_AUTO && dtype != ME_AUTO) {
            vtype = dtype;
        }
        size_t itemsize = 0;
        if (entry->dtype == ME_STRING) {
            itemsize = entry->itemsize;
        }
        int idx = dsl_var_table_add_with_uniform(&program->vars, name, vtype, itemsize, false);
        if (idx < 0) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "failed to register DSL input variable '%s'", name);
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            if (error_pos) {
                *error_pos = -1;
            }
            return NULL;
        }
        input_count++;
    }
    if (input_count != parsed->nparams) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap,
                     "DSL input count mismatch: function expects %d parameter(s), got %d",
                     parsed->nparams, input_count);
        }
        if (error_pos) {
            *error_pos = -1;
        }
        free(funcs);
        dsl_compiled_program_free(program);
        me_dsl_program_free(parsed);
        return NULL;
    }
    for (int i = 0; i < parsed->nparams; i++) {
        if (dsl_var_table_find(&program->vars, parsed->params[i]) < 0) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "missing DSL input for parameter '%s'", parsed->params[i]);
            }
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
    }
    program->n_inputs = input_count;

    if (dtype == ME_AUTO) {
        for (int i = 0; i < program->vars.count; i++) {
            if (program->vars.dtypes[i] == ME_AUTO) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "output dtype is ME_AUTO but variable '%s' dtype is unspecified",
                             program->vars.names[i]);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
    }

    int uses_i_mask = 0;
    int uses_n_mask = 0;
    bool uses_ndim = false;
    bool uses_flat_idx = false;
    dsl_scan_reserved_usage_block(&parsed->block, &uses_i_mask, &uses_n_mask, &uses_ndim,
                                  &uses_flat_idx);

    program->uses_i_mask = uses_i_mask;
    program->uses_n_mask = uses_n_mask;
    program->uses_ndim = uses_ndim;
    program->uses_flat_idx = uses_flat_idx;

    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
        if (uses_i_mask & (1 << d)) {
            char name[8];
            snprintf(name, sizeof(name), "_i%d", d);
            program->idx_i[d] = dsl_var_table_add(&program->vars, name, ME_INT64);
            if (program->idx_i[d] < 0) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "failed to register reserved index symbol '%s'", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
        if (uses_n_mask & (1 << d)) {
            char name[8];
            snprintf(name, sizeof(name), "_n%d", d);
            program->idx_n[d] = dsl_var_table_add_with_uniform(&program->vars, name, ME_INT64,
                                                               0, true);
            if (program->idx_n[d] < 0) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "failed to register reserved shape symbol '%s'", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
    }
    if (uses_ndim) {
        program->idx_ndim = dsl_var_table_add_with_uniform(&program->vars, "_ndim", ME_INT64,
                                                           0, true);
        if (program->idx_ndim < 0) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "failed to register reserved symbol '_ndim'");
            }
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
    }
    if (uses_flat_idx) {
        program->idx_flat_idx = dsl_var_table_add(&program->vars, "_flat_idx", ME_INT64);
        if (program->idx_flat_idx < 0) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "failed to register reserved symbol '_flat_idx'");
            }
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
    }

    dsl_compile_ctx ctx;
    ctx.source = source;
    ctx.output_dtype = dtype;
    ctx.output_dtype_auto = (dtype == ME_AUTO);
    ctx.loop_depth = 0;
    ctx.allow_new_locals = true;
    ctx.error_pos = error_pos;
    ctx.output_expr = NULL;
    ctx.has_return = false;
    ctx.return_dtype = ME_AUTO;
    ctx.program = program;
    ctx.funcs = funcs;
    ctx.func_count = func_count;
    ctx.error_reason = error_reason;
    ctx.error_reason_cap = error_reason_cap;

    if (!dsl_compile_block(&ctx, &parsed->block, &program->block)) {
        if (error_reason && error_reason_cap > 0 && error_reason[0] == '\0') {
            snprintf(error_reason, error_reason_cap, "failed to compile DSL statement block");
        }
        free(funcs);
        dsl_compiled_program_free(program);
        me_dsl_program_free(parsed);
        return NULL;
    }

    if (!ctx.has_return || !ctx.output_expr || !ctx.output_expr->expr) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap,
                     "DSL kernel must return a value via a return statement");
        }
        if (error_pos) {
            *error_pos = -1;
        }
        free(funcs);
        me_dsl_program_free(parsed);
        dsl_compiled_program_free(program);
        return NULL;
    }

    program->output_dtype = ctx.return_dtype;
    program->guaranteed_return = dsl_compiled_block_guarantees_return(&program->block);
    program->output_is_scalar = contains_reduction(ctx.output_expr->expr) &&
                                output_is_scalar(ctx.output_expr->expr);
    dsl_try_build_jit_ir(&ctx, parsed, program, jit_mode != ME_JIT_OFF);

    me_dsl_program_free(parsed);
    free(funcs);

    return program;
}
