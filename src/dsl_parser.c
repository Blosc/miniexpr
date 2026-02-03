/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_parser.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *source;
    const char *current;
    int line;
    int column;
    int indent_stack[32];  /* Stack of indentation levels */
    int indent_depth;      /* Current depth in indent stack */
} me_dsl_lexer;

static void dsl_set_error(me_dsl_error *error, int line, int column, const char *message) {
    if (!error) {
        return;
    }
    error->line = line;
    error->column = column;
    snprintf(error->message, sizeof(error->message), "%s", message ? message : "parse error");
}

static void lexer_init(me_dsl_lexer *lex, const char *source) {
    lex->source = source ? source : "";
    lex->current = lex->source;
    lex->line = 1;
    lex->column = 1;
    lex->indent_stack[0] = 0;  /* Base indentation is 0 */
    lex->indent_depth = 0;
}

static void lexer_advance(me_dsl_lexer *lex) {
    if (!lex->current || !*lex->current) {
        return;
    }
    if (*lex->current == '\n') {
        lex->line++;
        lex->column = 1;
    }
    else {
        lex->column++;
    }
    lex->current++;
}

/* Skip spaces and tabs only (not newlines) */
static void lexer_skip_space(me_dsl_lexer *lex) {
    while (*lex->current) {
        char c = *lex->current;
        if (c == ' ' || c == '\t' || c == '\r') {
            lexer_advance(lex);
            continue;
        }
        break;
    }
}

/* Skip comment starting with # until end of line */
static void lexer_skip_comment(me_dsl_lexer *lex) {
    if (*lex->current == '#') {
        while (*lex->current && *lex->current != '\n') {
            lexer_advance(lex);
        }
    }
}

/* Skip whitespace, newlines, and comments between statements at same level */
static void lexer_skip_separators(me_dsl_lexer *lex) {
    bool progressed = true;
    while (progressed) {
        progressed = false;
        lexer_skip_space(lex);
        lexer_skip_comment(lex);
        while (*lex->current == ';' || *lex->current == '\n') {
            lexer_advance(lex);
            progressed = true;
            lexer_skip_space(lex);
            lexer_skip_comment(lex);
        }
    }
}

/* Measure indentation at start of current line (spaces only, tabs = 4 spaces) */
static int measure_indent(me_dsl_lexer *lex) {
    const char *p = lex->current;
    int indent = 0;
    while (*p == ' ' || *p == '\t') {
        if (*p == ' ') {
            indent++;
        } else {
            indent += 4;  /* Tab = 4 spaces */
        }
        p++;
    }
    return indent;
}

/* Skip to next non-blank, non-comment line and measure its indentation */
static int peek_next_line_indent(me_dsl_lexer *lex) {
    me_dsl_lexer snapshot = *lex;

    /* Skip to end of current line */
    while (*lex->current && *lex->current != '\n') {
        lexer_advance(lex);
    }
    if (*lex->current == '\n') {
        lexer_advance(lex);
    }

    /* Skip blank lines and comment-only lines */
    while (*lex->current) {
        int indent = measure_indent(lex);
        /* Skip the whitespace */
        while (*lex->current == ' ' || *lex->current == '\t') {
            lexer_advance(lex);
        }
        /* Check if it's a blank line or comment-only line */
        if (*lex->current == '\n') {
            lexer_advance(lex);
            continue;
        }
        if (*lex->current == '#') {
            /* Skip comment line */
            while (*lex->current && *lex->current != '\n') {
                lexer_advance(lex);
            }
            if (*lex->current == '\n') {
                lexer_advance(lex);
            }
            continue;
        }
        /* Found a real line, restore and return its indent */
        *lex = snapshot;
        return indent;
    }

    /* End of file */
    *lex = snapshot;
    return 0;
}

static bool is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_';
}

static bool is_ident_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static bool lexer_read_identifier(me_dsl_lexer *lex, const char **start_out, size_t *len_out) {
    if (!is_ident_start(*lex->current)) {
        return false;
    }
    const char *start = lex->current;
    while (is_ident_char(*lex->current)) {
        lexer_advance(lex);
    }
    if (start_out) {
        *start_out = start;
    }
    if (len_out) {
        *len_out = (size_t)(lex->current - start);
    }
    return true;
}

static bool lexer_match_keyword(me_dsl_lexer *lex, const char *keyword) {
    me_dsl_lexer snapshot = *lex;
    const char *start = NULL;
    size_t len = 0;
    if (!lexer_read_identifier(lex, &start, &len)) {
        return false;
    }
    if (strlen(keyword) == len && strncmp(start, keyword, len) == 0) {
        return true;
    }
    *lex = snapshot;
    return false;
}

static char *dsl_strndup(const char *start, size_t len) {
    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, start, len);
    out[len] = '\0';
    return out;
}

static char *dsl_copy_trimmed(const char *start, const char *end) {
    while (start < end && isspace((unsigned char)*start)) {
        start++;
    }
    while (end > start && isspace((unsigned char)end[-1])) {
        end--;
    }
    return dsl_strndup(start, (size_t)(end - start));
}

static me_dsl_expr *dsl_expr_new(char *text, int line, int column) {
    me_dsl_expr *expr = malloc(sizeof(*expr));
    if (!expr) {
        free(text);
        return NULL;
    }
    expr->text = text;
    expr->line = line;
    expr->column = column;
    return expr;
}

static void dsl_expr_free(me_dsl_expr *expr) {
    if (!expr) {
        return;
    }
    free(expr->text);
    free(expr);
}

static void dsl_stmt_free(me_dsl_stmt *stmt);

static void dsl_block_free(me_dsl_block *block) {
    if (!block) {
        return;
    }
    for (int i = 0; i < block->nstmts; i++) {
        dsl_stmt_free(block->stmts[i]);
    }
    free(block->stmts);
    block->stmts = NULL;
    block->nstmts = 0;
    block->capacity = 0;
}

static bool dsl_block_push(me_dsl_block *block, me_dsl_stmt *stmt, me_dsl_error *error) {
    if (!block || !stmt) {
        return false;
    }
    if (block->nstmts == block->capacity) {
        int new_cap = block->capacity ? block->capacity * 2 : 8;
        me_dsl_stmt **next = realloc(block->stmts, (size_t)new_cap * sizeof(*next));
        if (!next) {
            dsl_set_error(error, stmt->line, stmt->column, "out of memory");
            return false;
        }
        block->stmts = next;
        block->capacity = new_cap;
    }
    block->stmts[block->nstmts++] = stmt;
    return true;
}

static me_dsl_stmt *dsl_stmt_new(me_dsl_stmt_kind kind, int line, int column) {
    me_dsl_stmt *stmt = calloc(1, sizeof(*stmt));
    if (!stmt) {
        return NULL;
    }
    stmt->kind = kind;
    stmt->line = line;
    stmt->column = column;
    return stmt;
}

static void dsl_stmt_free(me_dsl_stmt *stmt) {
    if (!stmt) {
        return;
    }
    switch (stmt->kind) {
    case ME_DSL_STMT_ASSIGN:
        free(stmt->as.assign.name);
        dsl_expr_free(stmt->as.assign.value);
        break;
    case ME_DSL_STMT_EXPR:
        dsl_expr_free(stmt->as.expr_stmt.expr);
        break;
    case ME_DSL_STMT_PRINT:
        dsl_expr_free(stmt->as.print_stmt.call);
        break;
    case ME_DSL_STMT_FOR:
        free(stmt->as.for_loop.var);
        dsl_expr_free(stmt->as.for_loop.limit);
        dsl_block_free(&stmt->as.for_loop.body);
        break;
    case ME_DSL_STMT_BREAK:
    case ME_DSL_STMT_CONTINUE:
        dsl_expr_free(stmt->as.flow.cond);
        break;
    }
    free(stmt);
}

static bool consume_char(me_dsl_lexer *lex, char c) {
    lexer_skip_space(lex);
    if (*lex->current != c) {
        return false;
    }
    lexer_advance(lex);
    return true;
}

static char *parse_expression_until_stmt_end(me_dsl_lexer *lex, me_dsl_error *error, int line, int column) {
    lexer_skip_space(lex);
    const char *start = lex->current;
    int depth = 0;

    while (*lex->current) {
        char c = *lex->current;
        if (c == '"' || c == '\'') {
            char quote = c;
            lexer_advance(lex);
            while (*lex->current) {
                if (*lex->current == '\\') {
                    lexer_advance(lex);
                    if (!*lex->current || *lex->current == '\n') {
                        dsl_set_error(error, line, column, "unterminated string literal");
                        return NULL;
                    }
                    lexer_advance(lex);
                    continue;
                }
                if (*lex->current == quote) {
                    lexer_advance(lex);
                    break;
                }
                if (*lex->current == '\n') {
                    dsl_set_error(error, line, column, "unterminated string literal");
                    return NULL;
                }
                lexer_advance(lex);
            }
            if (!*lex->current && *(lex->current - 1) != quote) {
                dsl_set_error(error, line, column, "unterminated string literal");
                return NULL;
            }
            continue;
        }
        if (c == '(') {
            depth++;
        }
        else if (c == ')') {
            if (depth == 0) {
                dsl_set_error(error, lex->line, lex->column, "unexpected ')'");
                return NULL;
            }
            depth--;
        }
        /* Stop at statement end: semicolon, newline, or comment */
        if (depth == 0 && (c == ';' || c == '\n' || c == '#')) {
            break;
        }
        lexer_advance(lex);
    }

    if (depth != 0) {
        dsl_set_error(error, line, column, "unclosed '(' in expression");
        return NULL;
    }

    char *text = dsl_copy_trimmed(start, lex->current);
    if (!text || text[0] == '\0') {
        free(text);
        dsl_set_error(error, line, column, "expected expression");
        return NULL;
    }
    return text;
}

static char *parse_expression_in_parens(me_dsl_lexer *lex, me_dsl_error *error, int line, int column) {
    lexer_skip_space(lex);
    if (*lex->current != '(') {
        dsl_set_error(error, line, column, "expected '('");
        return NULL;
    }
    lexer_advance(lex);
    const char *start = lex->current;
    int depth = 1;

    while (*lex->current) {
        char c = *lex->current;
        if (c == '"' || c == '\'') {
            char quote = c;
            lexer_advance(lex);
            while (*lex->current) {
                if (*lex->current == '\\') {
                    lexer_advance(lex);
                    if (!*lex->current || *lex->current == '\n') {
                        dsl_set_error(error, line, column, "unterminated string literal");
                        return NULL;
                    }
                    lexer_advance(lex);
                    continue;
                }
                if (*lex->current == quote) {
                    lexer_advance(lex);
                    break;
                }
                if (*lex->current == '\n') {
                    dsl_set_error(error, line, column, "unterminated string literal");
                    return NULL;
                }
                lexer_advance(lex);
            }
            if (!*lex->current && *(lex->current - 1) != quote) {
                dsl_set_error(error, line, column, "unterminated string literal");
                return NULL;
            }
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
        lexer_advance(lex);
    }

    if (*lex->current != ')') {
        dsl_set_error(error, line, column, "unclosed '(' in range");
        return NULL;
    }

    const char *end = lex->current;
    lexer_advance(lex);

    char *text = dsl_copy_trimmed(start, end);
    if (!text || text[0] == '\0') {
        free(text);
        dsl_set_error(error, line, column, "expected expression in range()");
        return NULL;
    }
    return text;
}

static bool parse_indented_block(me_dsl_lexer *lex, me_dsl_block *block, int min_indent,
                                 bool in_loop, me_dsl_error *error);

static bool parse_break_or_continue(me_dsl_lexer *lex, me_dsl_block *block,
                                    me_dsl_stmt_kind kind, int line, int column,
                                    bool in_loop, me_dsl_error *error) {
    me_dsl_stmt *stmt = dsl_stmt_new(kind, line, column);
    if (!stmt) {
        dsl_set_error(error, line, column, "out of memory");
        return false;
    }

    if (!in_loop) {
        dsl_set_error(error, line, column, "break/continue only allowed inside loops");
        dsl_stmt_free(stmt);
        return false;
    }

    lexer_skip_space(lex);
    if (lexer_match_keyword(lex, "if")) {
        dsl_set_error(error, line, column,
                      "deprecated 'break if'/'continue if' syntax; use 'if <cond>:' with break/continue");
        dsl_stmt_free(stmt);
        return false;
    }

    if (!dsl_block_push(block, stmt, error)) {
        dsl_stmt_free(stmt);
        return false;
    }
    return true;
}

static bool parse_if(me_dsl_lexer *lex, me_dsl_block *block, int line, int column,
                     bool in_loop, me_dsl_error *error) {
    if (!in_loop) {
        dsl_set_error(error, line, column, "if only allowed inside loops");
        return false;
    }

    char *cond_text = parse_expression_until_stmt_end(lex, error, line, column);
    if (!cond_text) {
        return false;
    }

    size_t len = strlen(cond_text);
    while (len > 0 && isspace((unsigned char)cond_text[len - 1])) {
        len--;
    }
    if (len == 0 || cond_text[len - 1] != ':') {
        dsl_set_error(error, line, column, "expected ':' after if condition");
        free(cond_text);
        return false;
    }
    cond_text[len - 1] = '\0';
    len = strlen(cond_text);
    while (len > 0 && isspace((unsigned char)cond_text[len - 1])) {
        cond_text[len - 1] = '\0';
        len--;
    }
    if (len == 0) {
        dsl_set_error(error, line, column, "expected condition after 'if'");
        free(cond_text);
        return false;
    }

    int body_indent = peek_next_line_indent(lex);
    int current_indent = lex->indent_stack[lex->indent_depth];
    if (body_indent <= current_indent) {
        dsl_set_error(error, lex->line, lex->column, "expected indented block after ':'");
        free(cond_text);
        return false;
    }

    me_dsl_expr *cond = dsl_expr_new(cond_text, line, column);
    if (!cond) {
        dsl_set_error(error, line, column, "out of memory");
        return false;
    }

    if (lex->indent_depth >= 31) {
        dsl_set_error(error, line, column, "too many nested blocks");
        dsl_expr_free(cond);
        return false;
    }

    lex->indent_depth++;
    lex->indent_stack[lex->indent_depth] = body_indent;

    me_dsl_block inner;
    if (!parse_indented_block(lex, &inner, body_indent, true, error)) {
        lex->indent_depth--;
        dsl_expr_free(cond);
        dsl_block_free(&inner);
        return false;
    }

    lex->indent_depth--;

    if (inner.nstmts != 1) {
        dsl_set_error(error, line, column, "if block must contain exactly one statement");
        dsl_expr_free(cond);
        dsl_block_free(&inner);
        return false;
    }
    me_dsl_stmt *stmt = inner.stmts[0];
    if (!stmt || (stmt->kind != ME_DSL_STMT_BREAK && stmt->kind != ME_DSL_STMT_CONTINUE)) {
        dsl_set_error(error, line, column, "if block must contain break or continue");
        dsl_expr_free(cond);
        dsl_block_free(&inner);
        return false;
    }
    if (stmt->as.flow.cond) {
        dsl_set_error(error, line, column, "break/continue inside if cannot have a condition");
        dsl_expr_free(cond);
        dsl_block_free(&inner);
        return false;
    }
    stmt->as.flow.cond = cond;

    free(inner.stmts);
    inner.stmts = NULL;
    inner.nstmts = 0;
    inner.capacity = 0;

    if (!dsl_block_push(block, stmt, error)) {
        dsl_stmt_free(stmt);
        return false;
    }

    return true;
}

static bool parse_for(me_dsl_lexer *lex, me_dsl_block *block, int line, int column, me_dsl_error *error) {
    lexer_skip_space(lex);
    const char *var_start = NULL;
    size_t var_len = 0;
    if (!lexer_read_identifier(lex, &var_start, &var_len)) {
        dsl_set_error(error, line, column, "expected loop variable");
        return false;
    }
    char *var = dsl_strndup(var_start, var_len);
    if (!var) {
        dsl_set_error(error, line, column, "out of memory");
        return false;
    }

    lexer_skip_space(lex);
    if (!lexer_match_keyword(lex, "in")) {
        free(var);
        dsl_set_error(error, line, column, "expected 'in' after loop variable");
        return false;
    }

    lexer_skip_space(lex);
    if (!lexer_match_keyword(lex, "range")) {
        free(var);
        dsl_set_error(error, line, column, "expected 'range' in loop");
        return false;
    }

    char *limit_text = parse_expression_in_parens(lex, error, line, column);
    if (!limit_text) {
        free(var);
        return false;
    }

    /* Expect colon for Python-style syntax */
    lexer_skip_space(lex);
    if (!consume_char(lex, ':')) {
        free(var);
        free(limit_text);
        dsl_set_error(error, line, column, "expected ':' after range()");
        return false;
    }

    /* Check for body indentation */
    int body_indent = peek_next_line_indent(lex);
    int current_indent = lex->indent_stack[lex->indent_depth];

    if (body_indent <= current_indent) {
        free(var);
        free(limit_text);
        dsl_set_error(error, lex->line, lex->column, "expected indented block after ':'");
        return false;
    }

    me_dsl_stmt *stmt = dsl_stmt_new(ME_DSL_STMT_FOR, line, column);
    if (!stmt) {
        free(var);
        free(limit_text);
        dsl_set_error(error, line, column, "out of memory");
        return false;
    }
    stmt->as.for_loop.var = var;
    stmt->as.for_loop.limit = dsl_expr_new(limit_text, line, column);
    if (!stmt->as.for_loop.limit) {
        dsl_set_error(error, line, column, "out of memory");
        dsl_stmt_free(stmt);
        return false;
    }

    /* Push new indentation level */
    if (lex->indent_depth >= 31) {
        dsl_set_error(error, line, column, "too many nested blocks");
        dsl_stmt_free(stmt);
        return false;
    }
    lex->indent_depth++;
    lex->indent_stack[lex->indent_depth] = body_indent;

    /* Parse the indented block */
    if (!parse_indented_block(lex, &stmt->as.for_loop.body, body_indent, true, error)) {
        lex->indent_depth--;
        dsl_stmt_free(stmt);
        return false;
    }

    /* Pop indentation level */
    lex->indent_depth--;

    if (!dsl_block_push(block, stmt, error)) {
        dsl_stmt_free(stmt);
        return false;
    }
    return true;
}

static bool parse_assignment_or_expr(me_dsl_lexer *lex, me_dsl_block *block, me_dsl_error *error) {
    me_dsl_lexer snapshot = *lex;
    const char *ident_start = NULL;
    size_t ident_len = 0;
    if (!lexer_read_identifier(lex, &ident_start, &ident_len)) {
        return false;
    }

    int line = snapshot.line;
    int column = snapshot.column;

    lexer_skip_space(lex);
    if (*lex->current == '=') {
        lexer_advance(lex);
        char *expr_text = parse_expression_until_stmt_end(lex, error, line, column);
        if (!expr_text) {
            return false;
        }
        me_dsl_stmt *stmt = dsl_stmt_new(ME_DSL_STMT_ASSIGN, line, column);
        if (!stmt) {
            free(expr_text);
            dsl_set_error(error, line, column, "out of memory");
            return false;
        }
        stmt->as.assign.name = dsl_strndup(ident_start, ident_len);
        stmt->as.assign.value = dsl_expr_new(expr_text, line, column);
        if (!stmt->as.assign.name || !stmt->as.assign.value) {
            dsl_set_error(error, line, column, "out of memory");
            dsl_stmt_free(stmt);
            return false;
        }
        if (!dsl_block_push(block, stmt, error)) {
            dsl_stmt_free(stmt);
            return false;
        }
        return true;
    }

    *lex = snapshot;
    return false;
}

static bool parse_expression_stmt(me_dsl_lexer *lex, me_dsl_block *block, me_dsl_error *error) {
    int line = lex->line;
    int column = lex->column;
    char *expr_text = parse_expression_until_stmt_end(lex, error, line, column);
    if (!expr_text) {
        return false;
    }
    me_dsl_stmt *stmt = dsl_stmt_new(ME_DSL_STMT_EXPR, line, column);
    if (!stmt) {
        free(expr_text);
        dsl_set_error(error, line, column, "out of memory");
        return false;
    }
    stmt->as.expr_stmt.expr = dsl_expr_new(expr_text, line, column);
    if (!stmt->as.expr_stmt.expr) {
        dsl_set_error(error, line, column, "out of memory");
        dsl_stmt_free(stmt);
        return false;
    }
    if (!dsl_block_push(block, stmt, error)) {
        dsl_stmt_free(stmt);
        return false;
    }
    return true;
}

static bool parse_print_stmt(me_dsl_lexer *lex, me_dsl_block *block, int line, int column, me_dsl_error *error) {
    char *expr_text = parse_expression_until_stmt_end(lex, error, line, column);
    if (!expr_text) {
        return false;
    }
    me_dsl_stmt *stmt = dsl_stmt_new(ME_DSL_STMT_PRINT, line, column);
    if (!stmt) {
        free(expr_text);
        dsl_set_error(error, line, column, "out of memory");
        return false;
    }
    stmt->as.print_stmt.call = dsl_expr_new(expr_text, line, column);
    if (!stmt->as.print_stmt.call) {
        dsl_set_error(error, line, column, "out of memory");
        dsl_stmt_free(stmt);
        return false;
    }
    if (!dsl_block_push(block, stmt, error)) {
        dsl_stmt_free(stmt);
        return false;
    }
    return true;
}

static bool parse_statement(me_dsl_lexer *lex, me_dsl_block *block, bool in_loop, me_dsl_error *error) {
    lexer_skip_separators(lex);
    if (*lex->current == '\0') {
        return false;
    }

    if (is_ident_start(*lex->current)) {
        me_dsl_lexer snapshot = *lex;
        const char *ident_start = NULL;
        size_t ident_len = 0;
        lexer_read_identifier(lex, &ident_start, &ident_len);

        if (ident_len == 3 && strncmp(ident_start, "for", ident_len) == 0) {
            return parse_for(lex, block, snapshot.line, snapshot.column, error);
        }
        if (ident_len == 2 && strncmp(ident_start, "if", ident_len) == 0) {
            return parse_if(lex, block, snapshot.line, snapshot.column, in_loop, error);
        }
        if (ident_len == 5 && strncmp(ident_start, "print", ident_len) == 0) {
            *lex = snapshot;
            return parse_print_stmt(lex, block, snapshot.line, snapshot.column, error);
        }
        if (ident_len == 5 && strncmp(ident_start, "break", ident_len) == 0) {
            return parse_break_or_continue(lex, block, ME_DSL_STMT_BREAK,
                                           snapshot.line, snapshot.column, in_loop, error);
        }
        if (ident_len == 8 && strncmp(ident_start, "continue", ident_len) == 0) {
            return parse_break_or_continue(lex, block, ME_DSL_STMT_CONTINUE,
                                           snapshot.line, snapshot.column, in_loop, error);
        }

        *lex = snapshot;
        if (parse_assignment_or_expr(lex, block, error)) {
            return true;
        }
    }

    return parse_expression_stmt(lex, block, error);
}

/* Skip to the start of the next line and consume leading whitespace up to expected indent */
static void skip_to_line_start(me_dsl_lexer *lex) {
    /* Skip rest of current line */
    while (*lex->current && *lex->current != '\n') {
        lexer_advance(lex);
    }
    if (*lex->current == '\n') {
        lexer_advance(lex);
    }
}

/* Get line indentation from current position (must be at start of line content or whitespace) */
static int measure_line_indent_from_start(const char *line_start) {
    const char *p = line_start;
    int indent = 0;
    while (*p == ' ' || *p == '\t') {
        if (*p == ' ') {
            indent++;
        } else {
            indent += 4;
        }
        p++;
    }
    return indent;
}

/* Skip whitespace at current position (spaces/tabs only) */
static void skip_line_whitespace(me_dsl_lexer *lex) {
    while (*lex->current == ' ' || *lex->current == '\t') {
        lexer_advance(lex);
    }
}

/* Parse an indented block (Python-style) */
static bool parse_indented_block(me_dsl_lexer *lex, me_dsl_block *block, int min_indent,
                                 bool in_loop, me_dsl_error *error) {
    memset(block, 0, sizeof(*block));

    /* Skip to the first line of the block */
    skip_to_line_start(lex);

    while (*lex->current) {
        /* Remember position at start of line */
        me_dsl_lexer line_snapshot = *lex;
        const char *line_start = lex->current;

        /* Get this line's indentation */
        int line_indent = measure_line_indent_from_start(line_start);

        /* Skip the leading whitespace */
        skip_line_whitespace(lex);

        /* Blank line? */
        if (*lex->current == '\n') {
            lexer_advance(lex);
            continue;
        }

        /* Comment-only line? */
        if (*lex->current == '#') {
            while (*lex->current && *lex->current != '\n') {
                lexer_advance(lex);
            }
            if (*lex->current == '\n') {
                lexer_advance(lex);
            }
            continue;
        }

        /* End of file? */
        if (*lex->current == '\0') {
            return true;
        }

        /* Check if indentation is enough for this block */
        if (line_indent < min_indent) {
            /* Dedent - block is done. Restore to start of this line. */
            *lex = line_snapshot;
            return true;
        }

        /* Indentation is sufficient, parse the statement */
        if (!parse_statement(lex, block, in_loop, error)) {
            return false;
        }

        /* After statement, move to next line if we haven't already.
         * But if we're already at a line start (column 1), the statement
         * parser (e.g., for-loop) already positioned us correctly. */
        if (lex->column != 1) {
            while (*lex->current && *lex->current != '\n') {
                lexer_advance(lex);
            }
            if (*lex->current == '\n') {
                lexer_advance(lex);
            }
        }
    }

    return true;
}

static bool parse_program(me_dsl_lexer *lex, me_dsl_program *program, me_dsl_error *error) {
    memset(program, 0, sizeof(*program));

    while (*lex->current) {
        lexer_skip_separators(lex);
        if (*lex->current == '\0') {
            break;
        }

        /* Skip leading whitespace for top-level */
        while (*lex->current == ' ' || *lex->current == '\t') {
            lexer_advance(lex);
        }

        /* Skip comment-only lines */
        if (*lex->current == '#') {
            while (*lex->current && *lex->current != '\n') {
                lexer_advance(lex);
            }
            continue;
        }

        /* Skip blank lines */
        if (*lex->current == '\n') {
            lexer_advance(lex);
            continue;
        }

        if (!parse_statement(lex, &program->block, false, error)) {
            return false;
        }
    }
    return true;
}

me_dsl_program *me_dsl_parse(const char *source, me_dsl_error *error) {
    if (error) {
        error->line = 0;
        error->column = 0;
        error->message[0] = '\0';
    }
    me_dsl_program *program = calloc(1, sizeof(*program));
    if (!program) {
        dsl_set_error(error, 0, 0, "out of memory");
        return NULL;
    }

    me_dsl_lexer lex;
    lexer_init(&lex, source);
    if (!parse_program(&lex, program, error)) {
        me_dsl_program_free(program);
        return NULL;
    }

    return program;
}

void me_dsl_program_free(me_dsl_program *program) {
    if (!program) {
        return;
    }
    dsl_block_free(&program->block);
    free(program);
}
