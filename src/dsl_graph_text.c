/* Syntax-only restricted expression frontend, terminating in the JSON validator. */
#include "miniexpr_graph.h"
#include "yyjson.h"
#include <ctype.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    const char *text;
    size_t length, pos;
    char token[80], nodes[ME_GRAPH_MAX_NODES][2048], names[ME_MAX_VARS][64];
    int kind, ninputs, count, depth;
    const char *error;
} text_parser;
static const char *t_dtype(me_dtype d) {
    switch (d) {
        case ME_BOOL: return "bool";
        case ME_INT8: return "int8"; case ME_INT16: return "int16";
        case ME_INT32: return "int32"; case ME_INT64: return "int64";
        case ME_UINT8: return "uint8"; case ME_UINT16: return "uint16";
        case ME_UINT32: return "uint32"; case ME_UINT64: return "uint64";
        case ME_FLOAT32: return "float32"; case ME_FLOAT64: return "float64";
        default: return NULL;
    }
}
static bool t_name(const char *s) {
    if (!s || !*s || strlen(s) > 63) return false;
    for (size_t i = 0; s[i]; i++) {
        bool alpha = (s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z') || s[i] == '_';
        if (!alpha && (!i || s[i] < '0' || s[i] > '9')) return false;
    }
    return true;
}
static void t_next(text_parser *p) {
    while (p->pos < p->length && isspace((unsigned char)p->text[p->pos])) p->pos++;
    p->kind = 0; p->token[0] = 0;
    if (p->pos == p->length || p->error) return;
    size_t start = p->pos; char c = p->text[p->pos++];
    if (isalpha((unsigned char)c) || c == '_') {
        p->kind = 1;
        while (p->pos < p->length && (isalnum((unsigned char)p->text[p->pos]) || p->text[p->pos] == '_')) p->pos++;
    } else if (isdigit((unsigned char)c) || (c == '.' && p->pos < p->length && isdigit((unsigned char)p->text[p->pos]))) {
        p->kind = 2;
        while (p->pos < p->length) {
            char x = p->text[p->pos];
            if (!isdigit((unsigned char)x) && x != '.' && x != 'e' && x != 'E' &&
                !((x == '+' || x == '-') && (p->text[p->pos - 1] == 'e' || p->text[p->pos - 1] == 'E'))) break;
            p->pos++;
        }
    } else {
        p->kind = 3;
        if (p->pos < p->length) {
            char d = p->text[p->pos];
            if ((c == '*' && d == '*') || (c == '/' && d == '/') || (c == '<' && d == '<') ||
                (c == '>' && d == '>') || ((c == '<' || c == '>' || c == '=' || c == '!') && d == '=')) p->pos++;
        }
    }
    size_t n = p->pos - start;
    if (n >= sizeof(p->token)) { p->error = "token length limit exceeded"; return; }
    memcpy(p->token, p->text + start, n); p->token[n] = 0;
}
static bool t_take(text_parser *p, const char *s) {
    if (strcmp(p->token, s)) return false;
    t_next(p); return true;
}
static void t_expect(text_parser *p, const char *s) {
    if (!t_take(p, s)) p->error = "unexpected expression token";
}
static int t_node(text_parser *p, const char *body) {
    if (p->count == ME_GRAPH_MAX_NODES) { p->error = "expression node limit exceeded"; return -1; }
    int id = p->count++;
    int n = snprintf(p->nodes[id], sizeof(p->nodes[id]), "{\"id\":%d,%s}", id, body);
    if (n < 0 || (size_t)n >= sizeof(p->nodes[id])) p->error = "node encoding limit exceeded";
    return id;
}
static int t_op(text_parser *p, const char *op, int a, int b) {
    char body[256];
    if (b < 0) snprintf(body, sizeof(body), "\"op\":\"%s\",\"args\":[%d]", op, a);
    else snprintf(body, sizeof(body), "\"op\":\"%s\",\"args\":[%d,%d]", op, a, b);
    return t_node(p, body);
}
static int t_expr(text_parser *p, int minimum);
static int t_primary(text_parser *p);
static int t_literal(text_parser *p, bool negative) {
    char body[320], text[84];
    snprintf(text, sizeof(text), "%s%s", negative ? "-" : "", p->token);
    if (strchr(text, '.') || strchr(text, 'e') || strchr(text, 'E')) {
        fenv_t saved;
        if (feholdexcept(&saved)) { p->error = "cannot preserve literal parsing environment"; return -1; }
        fesetround(FE_TONEAREST);
        /* yyjson's decimal parser is locale independent. Normalize .5/1. for
         * JSON transport while preserving the exact numerical token. */
        char normalized[88];
        const char *digits = text + negative;
        snprintf(normalized, sizeof(normalized), "%s%s%s", negative ? "-" : "", *digits == '.' ? "0" : "", digits);
        size_t length = strlen(normalized);
        if (length && normalized[length - 1] == '.') strcat(normalized, "0");
        yyjson_doc *literal = yyjson_read(normalized, strlen(normalized), 0);
        yyjson_val *number = literal ? yyjson_doc_get_root(literal) : NULL;
        double value = yyjson_get_num(number);
        bool valid = yyjson_is_num(number) && isfinite(value);
        uint64_t bits; memcpy(&bits, &value, sizeof(bits)); fesetenv(&saved);
        yyjson_doc_free(literal);
        if (!valid) { p->error = "invalid or nonfinite weak real literal"; return -1; }
        snprintf(body, sizeof(body), "\"op\":\"constant\",\"dtype\":\"float64\",\"category\":\"weak\",\"encoding\":\"ieee754-hex\",\"value\":\"%016llx\"", (unsigned long long)bits);
    } else {
        const char *digits = text + negative;
        if (digits[0] == '0' && digits[1]) { p->error = "noncanonical integer literal"; return -1; }
        if (negative && !strcmp(digits, "0")) strcpy(text, "0");
        snprintf(body, sizeof(body), "\"op\":\"constant\",\"dtype\":\"int64\",\"category\":\"weak\",\"encoding\":\"decimal\",\"value\":\"%s\"", text);
    }
    t_next(p); return t_node(p, body);
}
static bool t_reduction(const char *s) {
    return !strcmp(s, "sum") || !strcmp(s, "prod") || !strcmp(s, "min") || !strcmp(s, "max") || !strcmp(s, "any") || !strcmp(s, "all");
}
static int t_call(text_parser *p, const char *name) {
    char body[1800];
    if (t_reduction(name)) {
        int map = t_expr(p, 0), where = -1;
        char axes[256] = "null", dtype[64] = "auto", initial[768] = "null";
        bool keepdims = false; unsigned keywords = 0;
        while (t_take(p, ",") && !p->error) {
            char keyword[80]; strcpy(keyword, p->token); t_next(p); t_expect(p, "=");
            unsigned bit = !strcmp(keyword, "axis") ? 1 : !strcmp(keyword, "keepdims") ? 2 : !strcmp(keyword, "dtype") ? 4 : !strcmp(keyword, "where") ? 8 : !strcmp(keyword, "initial") ? 16 : 0;
            if (!bit || (keywords & bit)) { p->error = "unknown or duplicate reduction keyword"; break; }
            keywords |= bit;
            if (bit == 1) {
                if (t_take(p, "None") || t_take(p, "null")) strcpy(axes, "null");
                else {
                    char close = 0;
                    if (!strcmp(p->token, "[") || !strcmp(p->token, "(")) { close = p->token[0] == '[' ? ']' : ')'; t_next(p); }
                    size_t offset = 1; axes[0] = '['; axes[1] = 0; int count = 0;
                    while (!p->error && !(close && p->token[0] == close && !p->token[1])) {
                        bool neg = t_take(p, "-");
                        if (p->kind != 2 || strspn(p->token, "0123456789") != strlen(p->token) || strlen(p->token) > 2 || count == ME_ARRAY_MAX_RANK) { p->error = "invalid reduction axis"; break; }
                        int axis = atoi(p->token) * (neg ? -1 : 1);
                        if (axis < -ME_ARRAY_MAX_RANK || axis >= ME_ARRAY_MAX_RANK) { p->error = "axis exceeds maximum rank"; break; }
                        offset += (size_t)snprintf(axes + offset, sizeof(axes) - offset, "%s%d", count++ ? "," : "", axis);
                        t_next(p); if (!close || !t_take(p, ",")) break;
                    }
                    strcpy(axes + offset, "]");
                    if (close) { char end[2] = {close, 0}; t_expect(p, end); }
                }
            } else if (bit == 2) {
                if (t_take(p, "True") || t_take(p, "true")) keepdims = true;
                else if (!(t_take(p, "False") || t_take(p, "false"))) p->error = "keepdims requires Boolean literal";
            } else if (bit == 4) {
                if (p->kind != 1 || strlen(p->token) >= sizeof(dtype)) p->error = "dtype requires a dtype name";
                else { strcpy(dtype, p->token); t_next(p); }
            } else if (bit == 8) where = t_expr(p, 0);
            else {
                /* Use the depth-counted parser even for rejected nonliteral
                 * initial expressions; nested reduction keywords must not
                 * bypass the expression recursion bound. */
                int before = p->count, id = t_expr(p, 0);
                yyjson_doc *doc = id >= before && p->count == before + 1 ? yyjson_read(p->nodes[id], strlen(p->nodes[id]), 0) : NULL;
                yyjson_val *v = doc ? yyjson_doc_get_root(doc) : NULL;
                const char *op = yyjson_get_str(yyjson_obj_get(v, "op"));
                if (!op || strcmp(op, "constant")) p->error = "initial requires a scalar literal";
                else {
                    const char *fields[] = {"dtype", "category", "encoding", "value"};
                    size_t used = 0; initial[used++] = '{'; initial[used] = 0;
                    for (int i = 0; i < 4; i++) {
                        char *encoded = yyjson_val_write(yyjson_obj_get(v, fields[i]), 0, NULL);
                        if (!encoded) { p->error = "literal encoding allocation failed"; break; }
                        used += (size_t)snprintf(initial + used, sizeof(initial) - used, "%s\"%s\":%s", i ? "," : "", fields[i], encoded);
                        free(encoded);
                    }
                    strcat(initial, "}");
                }
                p->count = before; yyjson_doc_free(doc);
            }
        }
        t_expect(p, ")");
        char mask[32] = "null"; if (where >= 0) snprintf(mask, sizeof(mask), "%d", where);
        snprintf(body, sizeof(body), "\"op\":\"%s\",\"args\":[%d],\"axes\":%s,\"keepdims\":%s,\"dtype\":\"%s\",\"initial\":%s,\"where\":%s", name, map, axes, keepdims ? "true" : "false", dtype, initial, mask);
    } else {
        int args[3], n = 0;
        if (strcmp(p->token, ")")) do {
            if (n == 3) { p->error = "function arity exceeds graph limit"; break; }
            args[n++] = t_expr(p, 0);
        } while (t_take(p, ","));
        t_expect(p, ")");
        bool cast = false;
        me_dtype types[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
        for (int i = 0; i < 11; i++) if (!strcmp(name, t_dtype(types[i]))) cast = true;
        size_t used = (size_t)snprintf(body, sizeof(body), "\"op\":\"%s\"", cast ? "cast" : !strcmp(name, "where") ? "select" : "function");
        if (cast || strcmp(name, "where")) used += (size_t)snprintf(body + used, sizeof(body) - used, ",\"%s\":\"%s\"", cast ? "dtype" : "name", name);
        used += (size_t)snprintf(body + used, sizeof(body) - used, ",\"args\":[");
        for (int i = 0; i < n; i++) used += (size_t)snprintf(body + used, sizeof(body) - used, "%s%d", i ? "," : "", args[i]);
        snprintf(body + used, sizeof(body) - used, "]");
    }
    return p->error ? -1 : t_node(p, body);
}
static int t_primary(text_parser *p) {
    if (t_take(p, "(")) { int id = t_expr(p, 0); t_expect(p, ")"); return id; }
    if (!strcmp(p->token, "-") || !strcmp(p->token, "+") || !strcmp(p->token, "~") || !strcmp(p->token, "not")) {
        char op[8]; strcpy(op, p->token); t_next(p);
        /* A negative numeric token is a literal, not folded arithmetic. */
        if (!strcmp(op, "-") && p->kind == 2) {
            size_t pos = p->pos; char token[80]; strcpy(token, p->token);
            int count = p->count, result = t_literal(p, true);
            if (strcmp(p->token, "**")) return result;
            p->pos = pos; p->kind = 2; strcpy(p->token, token); p->count = count;
        }
        int arg = t_expr(p, !strcmp(op, "not") ? 25 : 100);
        return t_op(p, !strcmp(op, "-") ? "neg" : !strcmp(op, "+") ? "pos" : !strcmp(op, "~") ? "invert" : "not", arg, -1);
    }
    if (p->kind == 2) return t_literal(p, false);
    if (p->kind == 1) {
        char name[80]; strcpy(name, p->token); t_next(p);
        if (t_take(p, "(")) return t_call(p, name);
        if (!strcmp(name, "True") || !strcmp(name, "False") || !strcmp(name, "true") || !strcmp(name, "false")) {
            char body[256]; snprintf(body, sizeof(body), "\"op\":\"constant\",\"dtype\":\"bool\",\"category\":\"weak\",\"encoding\":\"boolean\",\"value\":%s", name[0] == 'T' || name[0] == 't' ? "true" : "false");
            return t_node(p, body);
        }
        for (int i = 0; i < p->ninputs; i++) if (!strcmp(name, p->names[i])) return i;
        p->error = "unbound expression name"; return -1;
    }
    p->error = "expected expression operand"; return -1;
}
static int t_precedence(const char *s, const char **op) {
    const char *symbols[] = {"or", "and", "==", "!=", "<", "<=", ">", ">=", "|", "^", "&", "<<", ">>", "+", "-", "*", "/", "//", "%", "**"};
    const char *names[] = {"or", "and", "eq", "ne", "lt", "le", "gt", "ge", "bitor", "bitxor", "bitand", "lshift", "rshift", "add", "sub", "mul", "div", "floordiv", "mod", "pow"};
    const int precedence[] = {10, 20, 30, 30, 30, 30, 30, 30, 40, 50, 60, 70, 70, 80, 80, 90, 90, 90, 90, 110};
    for (int i = 0; i < 20; i++) if (!strcmp(s, symbols[i])) { *op = names[i]; return precedence[i]; }
    return -1;
}
static int t_expr(text_parser *p, int minimum) {
    if (++p->depth > ME_GRAPH_MAX_DEPTH) { p->error = "expression nesting limit exceeded"; p->depth--; return -1; }
    int left = t_primary(p); bool compared = false;
    while (!p->error) {
        const char *op = NULL; int precedence = t_precedence(p->token, &op);
        if (precedence < minimum) break;
        if (precedence == 30 && compared) { p->error = "chained comparisons require explicit Boolean operators"; break; }
        compared |= precedence == 30; t_next(p);
        int right = t_expr(p, precedence + (precedence == 110 ? 0 : 1));
        left = t_op(p, op, left, right);
    }
    p->depth--; return left;
}
me_graph_status me_graph_prepare_expression(const char *expression, size_t length,
    const me_graph_input_metadata *inputs, int ninputs, const me_graph_prepare_options *options,
    me_graph_plan **out, me_graph_error *error) {
    if (out) *out = NULL;
    if (error) { memset(error, 0, sizeof(*error)); error->node = error->stage = -1; }
    if (!out || !expression || !length || length > ME_GRAPH_MAX_BYTES || memchr(expression, 0, length) ||
        ninputs < 0 || ninputs > ME_MAX_VARS || (ninputs && !inputs)) {
        if (error) snprintf(error->native.message, sizeof(error->native.message), "invalid expression buffer or signature limit");
        return ME_GRAPH_ERR_FORMAT;
    }
    text_parser *p = calloc(1, sizeof(*p));
    if (!p) {
        if (error) snprintf(error->native.message, sizeof(error->native.message), "expression parser allocation failed");
        return ME_GRAPH_ERR_OOM;
    }
    p->text = expression; p->length = length; p->ninputs = ninputs;
    for (int i = 0; i < ninputs; i++) {
        const char *dtype = t_dtype(inputs[i].dtype);
        if (!dtype || !t_name(inputs[i].name)) { p->error = "invalid text input signature"; break; }
        strcpy(p->names[i], inputs[i].name);
        char body[256]; snprintf(body, sizeof(body), "\"op\":\"input\",\"name\":\"%s\",\"dtype\":\"%s\"", inputs[i].name, dtype); t_node(p, body);
    }
    t_next(p); int root = p->error ? -1 : t_expr(p, 0);
    if (p->kind) p->error = "unsupported or trailing expression syntax";
    me_graph_status rc = ME_GRAPH_ERR_FORMAT;
    if (p->error) {
        if (error) { error->native.column = (int)p->pos; snprintf(error->native.message, sizeof(error->native.message), "%s", p->error); }
    } else {
        size_t capacity = 512;
        for (int i = 0; i < p->count; i++) capacity += strlen(p->nodes[i]) + 1;
        char *json = malloc(capacity);
        if (!json) {
            rc = ME_GRAPH_ERR_OOM;
            if (error) snprintf(error->native.message, sizeof(error->native.message), "expression graph allocation failed");
        }
        else {
            size_t used = (size_t)snprintf(json, capacity, "{\"format\":\"%s\",\"semantics\":\"%s\",\"requires\":[\"numeric\"],\"nodes\":[", ME_GRAPH_FORMAT, ME_GRAPH_SEMANTICS);
            for (int i = 0; i < p->count; i++) used += (size_t)snprintf(json + used, capacity - used, "%s%s", i ? "," : "", p->nodes[i]);
            used += (size_t)snprintf(json + used, capacity - used, "],\"root\":%d,\"output\":{\"dtype\":\"auto\",\"casting\":\"unsafe\"}}", root);
            rc = me_graph_prepare_json(json, used, options, out, error); free(json);
        }
    }
    free(p); return rc;
}
