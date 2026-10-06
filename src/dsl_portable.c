/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/

/* Draft portable-profile validation; the native parser/compiler remain authoritative. */
#include "miniexpr.h"
#include "dsl_compile_internal.h"
#include "dsl_parser.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static me_portable_status portable_error(me_portable_error *error,
                                         me_portable_status status,
                                         int line, int column, const char *message) {
    if (error) {
        error->line = line;
        error->column = column;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}

static bool portable_dtype(me_dtype dtype) {
    return dtype == ME_BOOL || dtype == ME_INT32 || dtype == ME_INT64 ||
           dtype == ME_FLOAT32 || dtype == ME_FLOAT64;
}

static bool portable_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool portable_ident_char(char c) {
    return portable_ident_start(c) || (c >= '0' && c <= '9');
}

static bool portable_ident_equal(const char *start, size_t length, const char *name) {
    return strlen(name) == length && !strncmp(start, name, length);
}

static bool portable_reserved(const char *start, size_t length) {
    if (portable_ident_equal(start, length, "_ndim") ||
        portable_ident_equal(start, length, "_flat_idx")) {
        return true;
    }
    if (length > 2 && start[0] == '_' && (start[1] == 'i' || start[1] == 'n')) {
        for (size_t i = 2; i < length; i++) {
            if (start[i] < '0' || start[i] > '9') {
                return false;
            }
        }
        return true;
    }
    return false;
}

static bool portable_name(const char *name) {
    if (!name || !portable_ident_start(*name)) {
        return false;
    }
    for (const char *p = name + 1; *p; p++) {
        if (!portable_ident_char(*p)) {
            return false;
        }
    }
    return !portable_reserved(name, strlen(name));
}

static bool portable_call(const char *start, size_t length, bool range_allowed) {
    const char *functions[] = {"sin", "cos", "int", "float", "bool"};
    for (size_t i = 0; i < sizeof(functions) / sizeof(functions[0]); i++) {
        if (portable_ident_equal(start, length, functions[i])) {
            return true;
        }
    }
    return range_allowed && portable_ident_equal(start, length, "range");
}

typedef struct {
    const char *names[ME_MAX_VARS];
    int count;
} portable_names;

static me_portable_status portable_add_name(portable_names *names, const char *name,
                                           me_portable_error *error) {
    if (!portable_name(name)) {
        return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                              "local/parameter name is outside portable profile 0.1");
    }
    for (int i = 0; i < names->count; i++) {
        if (!strcmp(names->names[i], name)) {
            return ME_PORTABLE_SUCCESS;
        }
    }
    if (names->count == ME_MAX_VARS) {
        return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                              "portable profile variable limit exceeded");
    }
    names->names[names->count++] = name;
    return ME_PORTABLE_SUCCESS;
}

static me_portable_status portable_collect_names(const me_dsl_block *block, portable_names *names,
                                                int depth, me_portable_error *error) {
    if (depth > 128) {
        return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                              "portable validation nesting limit exceeded");
    }
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_stmt *stmt = block->stmts[i];
        me_portable_status rc = ME_PORTABLE_SUCCESS;
        if (stmt->kind == ME_DSL_STMT_ASSIGN) {
            rc = portable_add_name(names, stmt->as.assign.name, error);
        } else if (stmt->kind == ME_DSL_STMT_FOR) {
            rc = portable_add_name(names, stmt->as.for_loop.var, error);
            if (rc == ME_PORTABLE_SUCCESS) {
                rc = portable_collect_names(&stmt->as.for_loop.body, names, depth + 1, error);
            }
        } else if (stmt->kind == ME_DSL_STMT_WHILE) {
            rc = portable_collect_names(&stmt->as.while_loop.body, names, depth + 1, error);
        } else if (stmt->kind == ME_DSL_STMT_IF) {
            rc = portable_collect_names(&stmt->as.if_stmt.then_block, names, depth + 1, error);
            for (int j = 0; rc == ME_PORTABLE_SUCCESS && j < stmt->as.if_stmt.n_elifs; j++) {
                rc = portable_collect_names(&stmt->as.if_stmt.elif_branches[j].block,
                                            names, depth + 1, error);
            }
            if (rc == ME_PORTABLE_SUCCESS && stmt->as.if_stmt.has_else) {
                rc = portable_collect_names(&stmt->as.if_stmt.else_block, names, depth + 1, error);
            }
        }
        if (rc != ME_PORTABLE_SUCCESS) {
            return rc;
        }
    }
    return ME_PORTABLE_SUCCESS;
}

/* Parsed expression text already has native numeric-literal normalization and
 * comparison lowering. This is a feature filter, not another expression parser. */
static me_portable_status portable_expr(const me_dsl_expr *expr, bool range_allowed,
                                       bool floating_inputs, const portable_names *names,
                                       me_portable_error *error) {
    if (!expr || !expr->text) {
        return ME_PORTABLE_SUCCESS;
    }
    const char *p = expr->text;
    while (*p) {
        if (isspace((unsigned char)*p)) {
            p++;
            continue;
        }
        if (portable_ident_start(*p)) {
            const char *start = p++;
            while (portable_ident_char(*p)) {
                p++;
            }
            size_t length = (size_t)(p - start);
            const char *next = p;
            while (isspace((unsigned char)*next)) {
                next++;
            }
            if (portable_reserved(start, length)) {
                return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, expr->line, expr->column,
                                      "index/shape symbols are outside portable profile 0.1");
            }
            if (*next == '.' || *next == '[') {
                return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, expr->line, expr->column,
                                      "attribute access and indexing are outside portable profile 0.1");
            }
            bool keyword = portable_ident_equal(start, length, "not") ||
                           portable_ident_equal(start, length, "and") ||
                           portable_ident_equal(start, length, "or");
            if (*next == '(' && !portable_call(start, length, range_allowed) && !keyword) {
                char message[256];
                snprintf(message, sizeof(message), "call '%.*s' is outside portable profile 0.1",
                         (int)(length > 128 ? 128 : length), start);
                return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED,
                                      expr->line, expr->column, message);
            }
            if (*next != '(' && !keyword) {
                bool bound = false;
                for (int i = 0; i < names->count; i++) {
                    bound = bound || portable_ident_equal(start, length, names->names[i]);
                }
                if (!bound) {
                    char message[256];
                    snprintf(message, sizeof(message), "unbound name '%.*s' in portable source",
                             (int)(length > 128 ? 128 : length), start);
                    return portable_error(error, ME_PORTABLE_ERR_SOURCE, expr->line, expr->column,
                                          message);
                }
            }
            continue;
        }
        if ((*p >= '0' && *p <= '9') || (*p == '.' && p[1] >= '0' && p[1] <= '9')) {
            char *end = NULL;
            double value = strtod(p, &end);
            if (end == p || !isfinite(value) || (!floating_inputs && value > 9007199254740992.0)) {
                return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, expr->line, expr->column,
                                      "numeric literal is outside the draft portable range");
            }
            p = end;
            continue;
        }
        if (strchr("()+-,", *p)) {
            p++;
            continue;
        }
        if (*p == '*' && p[1] != '*') {
            p++;
            continue;
        }
        if (*p == '/' && p[1] != '/' && floating_inputs) {
            p++;
            continue;
        }
        if ((*p == '<' || *p == '>') && p[1] != *p) {
            p++;
            if (*p == '=') {
                p++;
            }
            continue;
        }
        if (*p == '!' || (*p == '=' && p[1] == '=')) {
            p++;
            if (*p == '=') {
                p++;
            }
            continue;
        }
        return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, expr->line, expr->column,
                              "operator, string literal, or indexing is outside portable profile 0.1");
    }
    return ME_PORTABLE_SUCCESS;
}

static me_portable_status portable_block(const me_dsl_block *block, bool floating_inputs,
                                        const portable_names *names, int depth, me_portable_error *error) {
    if (depth > 128) {
        return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                              "portable validation nesting limit exceeded");
    }
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_stmt *stmt = block->stmts[i];
        me_portable_status rc = ME_PORTABLE_SUCCESS;
        switch (stmt->kind) {
            case ME_DSL_STMT_ASSIGN:
                rc = portable_expr(stmt->as.assign.value, false, floating_inputs, names, error);
                break;
            case ME_DSL_STMT_EXPR:
                rc = portable_expr(stmt->as.expr_stmt.expr, false, floating_inputs, names, error);
                break;
            case ME_DSL_STMT_RETURN:
                rc = portable_expr(stmt->as.return_stmt.expr, false, floating_inputs, names, error);
                break;
            case ME_DSL_STMT_PRINT:
                return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, stmt->line, stmt->column,
                                      "print is outside portable profile 0.1");
            case ME_DSL_STMT_IF:
                rc = portable_expr(stmt->as.if_stmt.cond, false, floating_inputs, names, error);
                if (rc == ME_PORTABLE_SUCCESS) {
                    rc = portable_block(&stmt->as.if_stmt.then_block, floating_inputs, names, depth + 1, error);
                }
                for (int j = 0; rc == ME_PORTABLE_SUCCESS && j < stmt->as.if_stmt.n_elifs; j++) {
                    rc = portable_expr(stmt->as.if_stmt.elif_branches[j].cond, false, floating_inputs, names, error);
                    if (rc == ME_PORTABLE_SUCCESS) {
                        rc = portable_block(&stmt->as.if_stmt.elif_branches[j].block,
                                            floating_inputs, names, depth + 1, error);
                    }
                }
                if (rc == ME_PORTABLE_SUCCESS && stmt->as.if_stmt.has_else) {
                    rc = portable_block(&stmt->as.if_stmt.else_block, floating_inputs, names, depth + 1, error);
                }
                break;
            case ME_DSL_STMT_WHILE:
                rc = portable_expr(stmt->as.while_loop.cond, false, floating_inputs, names, error);
                if (rc == ME_PORTABLE_SUCCESS) {
                    rc = portable_block(&stmt->as.while_loop.body, floating_inputs, names, depth + 1, error);
                }
                break;
            case ME_DSL_STMT_FOR:
                rc = portable_expr(stmt->as.for_loop.limit, true, floating_inputs, names, error);
                if (rc == ME_PORTABLE_SUCCESS) {
                    rc = portable_block(&stmt->as.for_loop.body, floating_inputs, names, depth + 1, error);
                }
                break;
            case ME_DSL_STMT_BREAK:
            case ME_DSL_STMT_CONTINUE:
                rc = portable_expr(stmt->as.flow.cond, false, floating_inputs, names, error);
                break;
        }
        if (rc != ME_PORTABLE_SUCCESS) {
            return rc;
        }
    }
    return ME_PORTABLE_SUCCESS;
}

me_portable_status me_validate_portable_dsl(const char *source, const char *version,
    const me_variable *inputs, int ninputs, me_dtype output_dtype,
    me_portable_error *error) {
    if (error) {
        memset(error, 0, sizeof(*error));
    }
    if (!version || strcmp(version, ME_PORTABLE_DSL_VERSION)) {
        return portable_error(error, ME_PORTABLE_ERR_VERSION, 0, 0,
                              "unsupported portable DSL version; expected 0.1");
    }
    if (!source) {
        return portable_error(error, ME_PORTABLE_ERR_SOURCE, 0, 0, "source must not be NULL");
    }
    if (ninputs < 0 || ninputs > ME_MAX_VARS || (ninputs && !inputs)) {
        return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "invalid input signature");
    }
    if (!portable_dtype(output_dtype)) {
        return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                              "output dtype is outside portable profile 0.1");
    }
    for (int i = 0; i < ninputs; i++) {
        if (!portable_name(inputs[i].name) || inputs[i].type != ME_VARIABLE ||
            inputs[i].address || inputs[i].context || inputs[i].itemsize) {
            return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0,
                                  "inputs must be plain named, explicitly typed signature variables");
        }
        if (!portable_dtype(inputs[i].dtype) || (i && inputs[i].dtype != inputs[0].dtype)) {
            return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                                  "input dtype or mixed input dtypes are outside portable profile 0.1");
        }
        for (int j = 0; j < i; j++) {
            if (!strcmp(inputs[i].name, inputs[j].name)) {
                return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "duplicate input name");
            }
        }
    }
    me_dsl_error parse_error;
    me_dsl_program *parsed = me_dsl_parse(source, &parse_error);
    if (!parsed) {
        return portable_error(error, strstr(parse_error.message, "out of memory")
                              ? ME_PORTABLE_ERR_OOM : ME_PORTABLE_ERR_SOURCE,
                              parse_error.line, parse_error.column, parse_error.message);
    }
    me_portable_status rc = ME_PORTABLE_SUCCESS;
    if (!portable_name(parsed->name)) {
        rc = portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                            "entry-point name is outside portable profile 0.1");
    } else if (parsed->nparams != ninputs) {
        rc = portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "parameter/input count mismatch");
    }
    for (int i = 0; rc == ME_PORTABLE_SUCCESS && i < parsed->nparams; i++) {
        bool found = false;
        for (int j = 0; j < ninputs; j++) {
            found = found || !strcmp(parsed->params[i], inputs[j].name);
        }
        if (!found) {
            rc = portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "unbound parameter name");
        }
    }
    if (rc == ME_PORTABLE_SUCCESS && parsed->fp_mode != ME_DSL_FP_STRICT) {
        rc = portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                            "non-strict floating-point pragmas are outside draft profile 0.1");
    }
    if (rc == ME_PORTABLE_SUCCESS) {
        portable_names names = {{0}, 0};
        for (int i = 0; rc == ME_PORTABLE_SUCCESS && i < parsed->nparams; i++) {
            rc = portable_add_name(&names, parsed->params[i], error);
        }
        if (rc == ME_PORTABLE_SUCCESS) {
            rc = portable_collect_names(&parsed->block, &names, 0, error);
        }
        bool floating = !ninputs || inputs[0].dtype == ME_FLOAT32 || inputs[0].dtype == ME_FLOAT64;
        if (rc == ME_PORTABLE_SUCCESS) {
            rc = portable_block(&parsed->block, floating, &names, 0, error);
        }
    }
    me_dsl_program_free(parsed);
    if (rc != ME_PORTABLE_SUCCESS) {
        return rc;
    }
    int position = -1;
    bool is_dsl = false;
    char reason[256] = {0};
    me_dsl_compiled_program *compiled = dsl_compile_program(source, inputs, ninputs,
        output_dtype, 0, ME_JIT_OFF, &position, &is_dsl, reason, sizeof(reason));
    if (!compiled) {
        int line = 0, column = 0;
        if (position >= 0) {
            line = column = 1;
            for (int i = 0; i < position && source[i]; i++) {
                if (source[i] == '\n') {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
            }
        }
        return portable_error(error, strstr(reason, "out of memory")
                              ? ME_PORTABLE_ERR_OOM : ME_PORTABLE_ERR_SOURCE,
                              line, column, reason[0] ? reason : "native DSL compilation failed");
    }
    dsl_compiled_program_free(compiled);
    return ME_PORTABLE_SUCCESS;
}
