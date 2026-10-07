/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

/* Comparison chains are lowered in the native DSL front end, before either
 * interpreter compilation or JIT IR construction. No operand trees are copied:
 * each value is captured once and later links run in guarded statement blocks.
 * The small precedence parser below is only used to locate chains and their
 * enclosing evaluation contexts; chain-free expressions remain untouched. */
#include "dsl_compare.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { C_ATOM, C_GROUP, C_UNARY, C_BINARY, C_BOOL, C_COMPARE, C_CALL } chain_kind;
typedef struct chain_node {
    chain_kind kind;
    const char *start, *end;
    char op[8];
    struct chain_node *left, *right, *args, *arg_next, *allocated_next;
    bool has_chain;
} chain_node;

typedef struct {
    const char *cursor, *token, *end;
    char op[8];
    int precedence;
    bool atom, failed;
    int depth;
    chain_node *allocated;
} chain_parser;

typedef struct {
    const char *source;
    unsigned counter;
    int line, column;
    me_dsl_error *error;
    bool failed;
    int depth;
} chain_context;

static char *chain_copy(const char *start, const char *end) {
    size_t len = (size_t)(end - start);
    char *out = malloc(len + 1);
    if (out) {
        memcpy(out, start, len);
        out[len] = 0;
    }
    return out;
}

static void chain_token(chain_parser *p) {
    while (isspace((unsigned char)*p->cursor)) {
        p->cursor++;
    }
    p->token = p->cursor;
    p->precedence = -1;
    p->atom = false;
    memset(p->op, 0, sizeof(p->op));
    const char *s = p->cursor;
    if (*s == '\'' || *s == '"' || ((*s == 'b' || *s == 'B') && (s[1] == '\'' || s[1] == '"'))) {
        if (*s == 'b' || *s == 'B') {
            s++;
        }
        char quote = *s++;
        while (*s && *s != quote) {
            if (*s == '\\' && s[1]) {
                s++;
            }
            s++;
        }
        if (*s) {
            s++;
        } else {
            p->failed = true;
        }
        p->atom = true;
    } else if (isdigit((unsigned char)*s) || (*s == '.' && isdigit((unsigned char)s[1]))) {
        char *end = NULL;
        (void)strtod(s, &end);
        s = end;
        p->atom = true;
    } else if (isalpha((unsigned char)*s) || *s == '_') {
        do {
            s++;
        } while (isalnum((unsigned char)*s) || *s == '_');
        size_t len = (size_t)(s - p->token);
        if (len == 2 && !strncmp(p->token, "or", 2)) {
            strcpy(p->op, "or");
            p->precedence = 1;
        } else if (len == 3 && !strncmp(p->token, "and", 3)) {
            strcpy(p->op, "and");
            p->precedence = 2;
        } else if (len == 3 && !strncmp(p->token, "not", 3)) {
            strcpy(p->op, "not");
        } else {
            p->atom = true;
        }
    } else if (*s) {
        p->op[0] = *s++;
        if ((*p->token == '<' || *p->token == '>' || *p->token == '=' || *p->token == '!') && *s == '=') {
            p->op[1] = *s++;
            p->op[2] = 0;
            p->precedence = 4;
        } else if ((*p->token == '<' || *p->token == '>') && *s == *p->token) {
            p->op[1] = *s++;
            p->op[2] = 0;
            p->precedence = 8;
        } else if ((*p->token == '&' || *p->token == '|') && *s == *p->token) {
            p->op[1] = *s++;
            p->op[2] = 0;
            p->precedence = *p->token == '&' ? 2 : 1;
        } else if (*p->token == '*' && *s == '*') {
            strcpy(p->op, "**");
            s++;
            p->precedence = 12;
        } else if (*p->token == '/' && *s == '/') {
            strcpy(p->op, "//");
            s++;
            p->precedence = 10;
        } else {
            switch (*p->token) {
            case ',': p->precedence = 0; break;
            case '<': case '>': p->precedence = 4; break;
            case '|': p->precedence = 5; break;
            case '^': p->precedence = 6; break;
            case '&': p->precedence = 7; break;
            case '+': case '-': p->precedence = 9; break;
            case '*': case '/': case '%': p->precedence = 10; break;
            default: break;
            }
        }
    }
    p->end = p->cursor = s;
}

static chain_node *chain_node_new(chain_parser *p, chain_kind kind) {
    chain_node *node = calloc(1, sizeof(*node));
    if (!node) {
        p->failed = true;
        return NULL;
    }
    node->kind = kind;
    node->allocated_next = p->allocated;
    p->allocated = node;
    return node;
}

static chain_node *chain_parse_expr(chain_parser *p, int minimum);

static chain_node *chain_parse_primary(chain_parser *p) {
    chain_node *node = NULL;
    const char *start = p->token;
    if (!strcmp(p->op, "(")) {
        node = chain_node_new(p, C_GROUP);
        if (!node) {
            return NULL;
        }
        chain_token(p);
        node->left = chain_parse_expr(p, 0);
        if (!node->left || strcmp(p->op, ")")) {
            p->failed = true;
            return NULL;
        }
        node->has_chain = node->left->has_chain;
        node->end = p->end;
        chain_token(p);
    } else if (!strcmp(p->op, "+") || !strcmp(p->op, "-") ||
               !strcmp(p->op, "!") || !strcmp(p->op, "not")) {
        node = chain_node_new(p, C_UNARY);
        if (!node) {
            return NULL;
        }
        strcpy(node->op, p->op);
        int precedence = (!strcmp(p->op, "!") || !strcmp(p->op, "not")) ? 3 : 11;
        chain_token(p);
        node->left = chain_parse_expr(p, precedence);
        if (!node->left) {
            return NULL;
        }
        node->has_chain = node->left->has_chain;
        node->end = node->left->end;
    } else if (p->atom) {
        node = chain_node_new(p, C_ATOM);
        if (!node) {
            return NULL;
        }
        node->end = p->end;
        chain_token(p);
        if (!strcmp(p->op, "(")) {
            node->kind = C_CALL;
            chain_node **tail = &node->args;
            chain_token(p);
            while (strcmp(p->op, ")")) {
                chain_node *arg = chain_parse_expr(p, 1);
                if (!arg) {
                    return NULL;
                }
                *tail = arg;
                tail = &arg->arg_next;
                node->has_chain |= arg->has_chain;
                if (strcmp(p->op, ",")) {
                    break;
                }
                chain_token(p);
                if (!strcmp(p->op, ")")) {
                    p->failed = true;
                    return NULL;
                }
            }
            if (strcmp(p->op, ")")) {
                p->failed = true;
                return NULL;
            }
            node->end = p->end;
            chain_token(p);
        }
    } else {
        p->failed = true;
        return NULL;
    }
    node->start = start;
    return node;
}

static chain_node *chain_parse_expr(chain_parser *p, int minimum) {
    /* Bound recursive syntax nesting rather than overflowing the C stack. */
    if (++p->depth > 256) {
        p->failed = true;
        p->depth--;
        return NULL;
    }
    chain_node *left = chain_parse_primary(p);
    while (left && p->precedence >= minimum) {
        int precedence = p->precedence;
        chain_kind kind = precedence == 4 ? C_COMPARE :
                          (precedence == 1 || precedence == 2) ? C_BOOL : C_BINARY;
        chain_node *node = chain_node_new(p, kind);
        if (!node) {
            left = NULL;
            break;
        }
        strcpy(node->op, p->op);
        chain_token(p);
        node->left = left;
        node->right = chain_parse_expr(p, precedence + (precedence != 12));
        if (!node->right) {
            left = NULL;
            break;
        }
        node->start = left->start;
        node->end = node->right->end;
        node->has_chain = left->has_chain || node->right->has_chain ||
                          (kind == C_COMPARE && left->kind == C_COMPARE);
        left = node;
    }
    p->depth--;
    return left;
}

static bool chain_push(chain_context *ctx, me_dsl_block *block, me_dsl_stmt *stmt) {
    if (!stmt || !dsl_block_push(block, stmt, ctx->error)) {
        dsl_stmt_free(stmt);
        ctx->failed = true;
        return false;
    }
    return true;
}

static me_dsl_stmt *chain_statement(chain_context *ctx, me_dsl_stmt_kind kind) {
    me_dsl_stmt *stmt = dsl_stmt_new(kind, ctx->line, ctx->column);
    if (!stmt) {
        ctx->failed = true;
    }
    return stmt;
}

/* Takes ownership of value, whether or not allocation succeeds. */
static bool chain_assign(chain_context *ctx, me_dsl_block *block, const char *name, char *value) {
    me_dsl_stmt *stmt = chain_statement(ctx, ME_DSL_STMT_ASSIGN);
    if (!stmt) {
        free(value);
        return false;
    }
    stmt->as.assign.name = chain_copy(name, name + strlen(name));
    stmt->as.assign.synthetic = true;
    stmt->as.assign.value = dsl_expr_new(value, ctx->line, ctx->column);
    if (!stmt->as.assign.name || !stmt->as.assign.value || !value) {
        dsl_stmt_free(stmt);
        ctx->failed = true;
        return false;
    }
    return chain_push(ctx, block, stmt);
}

static char *chain_capture(chain_context *ctx, me_dsl_block *block, char *value) {
    char name[64];
    do {
        snprintf(name, sizeof(name), "b2_chain_%u", ctx->counter++);
    } while (strstr(ctx->source, name));
    if (!value || !chain_assign(ctx, block, name, value)) {
        ctx->failed = true;
        return NULL;
    }
    return chain_copy(name, name + strlen(name));
}

static char *chain_binary(const char *left, const char *op, const char *right) {
    if (!left || !right) {
        return NULL;
    }
    size_t size = strlen(left) + strlen(right) + strlen(op) + 8;
    char *text = malloc(size);
    if (text) {
        snprintf(text, size, "(%s %s %s)", left, op, right);
    }
    return text;
}

static char *chain_unary(const char *op, const char *value) {
    if (!value) {
        return NULL;
    }
    size_t size = strlen(op) + strlen(value) + 5;
    char *text = malloc(size);
    if (text) {
        snprintf(text, size, "%s(%s)", op, value);
    }
    return text;
}

static me_dsl_stmt *chain_guard(chain_context *ctx, const char *name, bool negate) {
    me_dsl_stmt *stmt = chain_statement(ctx, ME_DSL_STMT_IF);
    if (!stmt) {
        return NULL;
    }
    char *text = negate ? chain_unary("not ", name) : chain_copy(name, name + strlen(name));
    stmt->as.if_stmt.cond = dsl_expr_new(text, ctx->line, ctx->column);
    if (!text || !stmt->as.if_stmt.cond) {
        dsl_stmt_free(stmt);
        ctx->failed = true;
        return NULL;
    }
    return stmt;
}

static char *chain_lower_node(chain_context *ctx, chain_node *node, me_dsl_block *prelude);

static char *chain_operand(chain_context *ctx, chain_node *node, me_dsl_block *prelude) {
    return chain_capture(ctx, prelude, chain_lower_node(ctx, node, prelude));
}

static bool chain_boolean_result(const chain_node *node) {
    if (node->kind == C_GROUP) {
        return chain_boolean_result(node->left);
    }
    return node->kind == C_COMPARE || node->kind == C_BOOL ||
           (node->kind == C_UNARY && (!strcmp(node->op, "not") || !strcmp(node->op, "!")));
}

static char *chain_numeric_operand(chain_context *ctx, chain_node *node, me_dsl_block *prelude) {
    char *value = chain_operand(ctx, node, prelude);
    if (chain_boolean_result(node)) {
        /* Boolean SIMD arithmetic has logical semantics. Explicit conversion
         * makes chain arithmetic numerical, just as in scalar C/Python. */
        char *numeric = chain_unary("int", value);
        free(value);
        return numeric;
    }
    return value;
}

static char *chain_lower_compare(chain_context *ctx, chain_node *node, me_dsl_block *prelude) {
    size_t count = 0;
    for (chain_node *n = node; n && n->kind == C_COMPARE; n = n->left) {
        count++;
    }
    chain_node **links = malloc(count * sizeof(*links));
    if (!links) {
        return NULL;
    }
    chain_node *n = node;
    for (size_t i = count; i-- > 0;) {
        links[i] = n;
        n = n->left;
    }
    char *left = chain_operand(ctx, n, prelude);
    char *right = chain_operand(ctx, links[0]->right, prelude);
    char *result = chain_capture(ctx, prelude, chain_binary(left, links[0]->op, right));
    free(left);
    for (size_t i = 1; i < count && result && right && !ctx->failed; i++) {
        me_dsl_stmt *guard = chain_guard(ctx, result, false);
        if (!guard) {
            break;
        }
        char *next = chain_operand(ctx, links[i]->right, &guard->as.if_stmt.then_block);
        chain_assign(ctx, &guard->as.if_stmt.then_block, result, chain_binary(right, links[i]->op, next));
        free(right);
        right = next;
        chain_push(ctx, prelude, guard);
    }
    free(right);
    free(links);
    return result;
}

static char *chain_lower_node_impl(chain_context *ctx, chain_node *node, me_dsl_block *prelude) {
    if (!node->has_chain) {
        return chain_copy(node->start, node->end);
    }
    if (node->kind == C_COMPARE) {
        return chain_lower_compare(ctx, node, prelude);
    }
    if (node->kind == C_GROUP) {
        return chain_lower_node(ctx, node->left, prelude);
    }
    if (node->kind == C_BOOL) {
        char *left = chain_operand(ctx, node->left, prelude);
        char *result = chain_capture(ctx, prelude, chain_unary("bool", left));
        free(left);
        if (!result) {
            return NULL;
        }
        bool is_or = !strcmp(node->op, "or") || !strcmp(node->op, "||");
        me_dsl_stmt *guard = chain_guard(ctx, result, is_or);
        if (!guard) {
            free(result);
            return NULL;
        }
        char *right = chain_operand(ctx, node->right, &guard->as.if_stmt.then_block);
        chain_assign(ctx, &guard->as.if_stmt.then_block, result, chain_unary("bool", right));
        free(right);
        chain_push(ctx, prelude, guard);
        return result;
    }
    if (node->kind == C_UNARY || node->kind == C_BINARY) {
        bool arithmetic = node->op[0] == '+' || node->op[0] == '-' || node->op[0] == '*' ||
                          node->op[0] == '/' || node->op[0] == '%';
        char *left = arithmetic ? chain_numeric_operand(ctx, node->left, prelude) :
                                  chain_operand(ctx, node->left, prelude);
        char *text;
        if (node->kind == C_UNARY) {
            text = chain_unary(node->op, left);
        } else {
            char *right = arithmetic ? chain_numeric_operand(ctx, node->right, prelude) :
                                       chain_operand(ctx, node->right, prelude);
            text = chain_binary(left, node->op, right);
            free(right);
        }
        free(left);
        return text;
    }
    if (node->kind == C_CALL) {
        const char *end_name = node->start;
        while (isalnum((unsigned char)*end_name) || *end_name == '_') {
            end_name++;
        }
        char *text = chain_copy(node->start, end_name);
        if (!text) {
            return NULL;
        }
        size_t len = strlen(text);
        bool is_print = !strcmp(text, "print");
        for (chain_node *arg = node->args; arg; arg = arg->arg_next) {
            /* The print front end requires its format literal in place. */
            char *value = is_print && arg == node->args && !arg->has_chain ?
                          chain_copy(arg->start, arg->end) : chain_operand(ctx, arg, prelude);
            if (!value) {
                free(text);
                return NULL;
            }
            size_t size = len + strlen(value) + 4;
            char *next = realloc(text, size);
            if (!next) {
                free(value);
                free(text);
                return NULL;
            }
            text = next;
            snprintf(text + len, size - len, "%s%s", arg == node->args ? "(" : ", ", value);
            len = strlen(text);
            free(value);
        }
        text[len++] = ')';
        text[len] = 0;
        return text;
    }
    return NULL;
}

static char *chain_lower_node(chain_context *ctx, chain_node *node, me_dsl_block *prelude) {
    if (++ctx->depth > 256) {
        ctx->failed = true;
        ctx->depth--;
        return NULL;
    }
    char *text = chain_lower_node_impl(ctx, node, prelude);
    ctx->depth--;
    return text;
}

static bool chain_lower_expr(chain_context *ctx, me_dsl_expr *expr, me_dsl_block *prelude) {
    ctx->line = expr->line;
    ctx->column = expr->column;
    chain_parser parser = {.cursor = expr->text};
    chain_token(&parser);
    int comparisons = 0;
    while (*parser.token) {
        comparisons += parser.precedence == 4;
        chain_token(&parser);
    }
    if (comparisons < 2) {
        return true;
    }
    parser = (chain_parser){.cursor = expr->text};
    chain_token(&parser);
    chain_node *root = chain_parse_expr(&parser, 0);
    /* Leave chain-free (including unsupported) syntax to the normal compiler.
     * This pass must not broaden the expression language on its own. */
    if (parser.failed || !root || *parser.token) {
        ctx->failed = true;
        if (ctx->error) {
            ctx->error->line = expr->line;
            ctx->error->column = expr->column;
            snprintf(ctx->error->message, sizeof(ctx->error->message), "invalid or too deeply nested comparison expression");
        }
    } else if (root->has_chain) {
        char *text = chain_lower_node(ctx, root, prelude);
        if (text && !ctx->failed) {
            free(expr->text);
            expr->text = text;
        } else {
            free(text);
            ctx->failed = true;
        }
    }
    while (parser.allocated) {
        chain_node *node = parser.allocated;
        parser.allocated = node->allocated_next;
        free(node);
    }
    return !ctx->failed;
}

static bool chain_expr_has_chain(const me_dsl_expr *expr) {
    chain_parser parser = {.cursor = expr->text};
    chain_token(&parser);
    chain_node *root = chain_parse_expr(&parser, 0);
    bool found = !parser.failed && root && !*parser.token && root->has_chain;
    while (parser.allocated) {
        chain_node *node = parser.allocated;
        parser.allocated = node->allocated_next;
        free(node);
    }
    return found;
}

static bool chain_lower_block(chain_context *ctx, me_dsl_block *block);

static bool chain_lower_range(chain_context *ctx, me_dsl_expr *expr, me_dsl_block *prelude) {
    if (!chain_expr_has_chain(expr)) {
        return true;
    }
    /* The for AST stores only the argument list, not the range() call. Parse
     * it as a call so commas stay argument separators rather than one value. */
    size_t size = strlen(expr->text) + 8;
    char *call = malloc(size);
    if (!call) {
        ctx->failed = true;
        return false;
    }
    snprintf(call, size, "range(%s)", expr->text);
    free(expr->text);
    expr->text = call;
    if (!chain_lower_expr(ctx, expr, prelude)) {
        return false;
    }
    size_t len = strlen(expr->text);
    memmove(expr->text, expr->text + 6, len - 7);
    expr->text[len - 7] = 0;
    return true;
}

static bool chain_lower_stmt(chain_context *ctx, me_dsl_stmt *stmt, me_dsl_block *prelude) {
    switch (stmt->kind) {
    case ME_DSL_STMT_ASSIGN:
        return chain_lower_expr(ctx, stmt->as.assign.value, prelude);
    case ME_DSL_STMT_RETURN:
        return chain_lower_expr(ctx, stmt->as.return_stmt.expr, prelude);
    case ME_DSL_STMT_EXPR:
        return chain_lower_expr(ctx, stmt->as.expr_stmt.expr, prelude);
    case ME_DSL_STMT_PRINT:
        return chain_lower_expr(ctx, stmt->as.print_stmt.call, prelude);
    case ME_DSL_STMT_FOR:
        return chain_lower_range(ctx, stmt->as.for_loop.limit, prelude) &&
               chain_lower_block(ctx, &stmt->as.for_loop.body);
    case ME_DSL_STMT_IF: {
        /* An elif prelude must execute only after earlier branches fail. Turn
         * elif into nested else/if before lowering its condition. */
        bool lower_elifs = false;
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            lower_elifs |= chain_expr_has_chain(stmt->as.if_stmt.elif_branches[i].cond);
        }
        for (int i = lower_elifs ? stmt->as.if_stmt.n_elifs : 0; i-- > 0;) {
            me_dsl_stmt *nested = chain_statement(ctx, ME_DSL_STMT_IF);
            if (!nested) {
                return false;
            }
            nested->as.if_stmt.cond = stmt->as.if_stmt.elif_branches[i].cond;
            nested->as.if_stmt.then_block = stmt->as.if_stmt.elif_branches[i].block;
            memset(&stmt->as.if_stmt.elif_branches[i], 0, sizeof(me_dsl_if_branch));
            nested->as.if_stmt.else_block = stmt->as.if_stmt.else_block;
            nested->as.if_stmt.has_else = stmt->as.if_stmt.has_else;
            memset(&stmt->as.if_stmt.else_block, 0, sizeof(me_dsl_block));
            stmt->as.if_stmt.has_else = true;
            if (!chain_push(ctx, &stmt->as.if_stmt.else_block, nested)) {
                return false;
            }
        }
        if (lower_elifs) {
            free(stmt->as.if_stmt.elif_branches);
            stmt->as.if_stmt.elif_branches = NULL;
            stmt->as.if_stmt.n_elifs = stmt->as.if_stmt.elif_capacity = 0;
        }
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            if (!chain_lower_block(ctx, &stmt->as.if_stmt.elif_branches[i].block)) {
                return false;
            }
        }
        return chain_lower_expr(ctx, stmt->as.if_stmt.cond, prelude) &&
               chain_lower_block(ctx, &stmt->as.if_stmt.then_block) &&
               chain_lower_block(ctx, &stmt->as.if_stmt.else_block);
    }
    case ME_DSL_STMT_WHILE: {
        me_dsl_block condition = {0};
        if (!chain_lower_expr(ctx, stmt->as.while_loop.cond, &condition) ||
            !chain_lower_block(ctx, &stmt->as.while_loop.body)) {
            dsl_block_free(&condition);
            return false;
        }
        if (!condition.nstmts) {
            return true;
        }
        me_dsl_stmt *stop = chain_guard(ctx, stmt->as.while_loop.cond->text, true);
        if (!stop) {
            dsl_block_free(&condition);
            return false;
        }
        chain_push(ctx, &stop->as.if_stmt.then_block, chain_statement(ctx, ME_DSL_STMT_BREAK));
        chain_push(ctx, &condition, stop);
        stmt->as.while_loop.condition_nstmts = condition.nstmts;
        for (int i = 0; i < stmt->as.while_loop.body.nstmts; i++) {
            me_dsl_stmt *body_stmt = stmt->as.while_loop.body.stmts[i];
            stmt->as.while_loop.body.stmts[i] = NULL;
            chain_push(ctx, &condition, body_stmt);
        }
        dsl_block_free(&stmt->as.while_loop.body);
        stmt->as.while_loop.body = condition;
        free(stmt->as.while_loop.cond->text);
        const char *always = "1";
        stmt->as.while_loop.cond->text = chain_copy(always, always + 1);
        if (!stmt->as.while_loop.cond->text) {
            ctx->failed = true;
        }
        return !ctx->failed;
    }
    case ME_DSL_STMT_BREAK:
    case ME_DSL_STMT_CONTINUE:
        return true;
    }
    return false;
}

static bool chain_lower_block(chain_context *ctx, me_dsl_block *block) {
    me_dsl_block lowered = {0};
    for (int i = 0; i < block->nstmts; i++) {
        me_dsl_stmt *stmt = block->stmts[i];
        block->stmts[i] = NULL;
        if (!chain_lower_stmt(ctx, stmt, &lowered)) {
            dsl_stmt_free(stmt);
            dsl_block_free(&lowered);
            return false;
        }
        if (!chain_push(ctx, &lowered, stmt)) {
            dsl_block_free(&lowered);
            return false;
        }
    }
    dsl_block_free(block);
    *block = lowered;
    return true;
}

bool dsl_lower_comparisons(me_dsl_program *program, const char *source, me_dsl_error *error) {
    chain_context ctx = {.source = source, .error = error};
    if (!chain_lower_block(&ctx, &program->block)) {
        if (error && !error->message[0]) {
            error->line = ctx.line;
            error->column = ctx.column;
            snprintf(error->message, sizeof(error->message), "failed to lower comparison chain");
        }
        return false;
    }
    return true;
}
