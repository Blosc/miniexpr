/* Syntax-only frontend sharing graph validation through an in-memory tree. */
#include "miniexpr_graph.h"
#include "dsl_graph_internal.h"
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
    char token[80], names[ME_MAX_VARS][64];
    yyjson_mut_doc *doc;
    yyjson_mut_val *nodes[ME_GRAPH_MAX_NODES];
    bool oom;
    int kind, ninputs, count, depth;
    bool reductions[ME_GRAPH_MAX_NODES];
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
static void t_add(text_parser *p, yyjson_mut_val *object, const char *key, yyjson_mut_val *value) {
    if (!yyjson_mut_obj_add_val(p->doc, object, key, value)) {
        p->oom = true; p->error = "expression allocation failed";
    }
}
static void t_string(text_parser *p, yyjson_mut_val *object, const char *key, const char *value) {
    t_add(p, object, key, yyjson_mut_strcpy(p->doc, value));
}
static void t_append(text_parser *p, yyjson_mut_val *array, yyjson_mut_val *value) {
    if (!yyjson_mut_arr_append(array, value)) {
        p->oom = true; p->error = "expression allocation failed";
    }
}
static int t_node(text_parser *p, const char *op) {
    if (p->error) return -1;
    if (p->count == ME_GRAPH_MAX_NODES) { p->error = "expression node limit exceeded"; return -1; }
    int id = p->count++;
    p->reductions[id] = false;
    p->nodes[id] = yyjson_mut_obj(p->doc);
    t_add(p, p->nodes[id], "id", yyjson_mut_uint(p->doc, (uint64_t)id));
    t_string(p, p->nodes[id], "op", op);
    return id;
}
static int t_arguments(text_parser *p, const char *op, const int *args, int count) {
    int id = t_node(p, op);
    if (id < 0) return id;
    yyjson_mut_val *array = yyjson_mut_arr(p->doc);
    t_add(p, p->nodes[id], "args", array);
    for (int i = 0; i < count; i++) t_append(p, array, yyjson_mut_uint(p->doc, (uint64_t)args[i]));
    return id;
}
static int t_op(text_parser *p, const char *op, int a, int b) {
    int args[] = {a, b};
    return t_arguments(p, op, args, b < 0 ? 1 : 2);
}
static int t_constant(text_parser *p, const char *dtype, const char *encoding, yyjson_mut_val *value) {
    int id = t_node(p, "constant");
    if (id < 0) return id;
    t_string(p, p->nodes[id], "dtype", dtype);
    t_string(p, p->nodes[id], "category", "weak");
    t_string(p, p->nodes[id], "encoding", encoding);
    t_add(p, p->nodes[id], "value", value);
    return id;
}
static int t_expr(text_parser *p, int minimum);
static int t_primary(text_parser *p);
static int t_literal(text_parser *p, bool negative) {
    char text[84];
    int id;
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
        char hex[17]; snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)bits);
        id = t_constant(p, "float64", "ieee754-hex", yyjson_mut_strcpy(p->doc, hex));
    } else {
        const char *digits = text + negative;
        if (digits[0] == '0' && digits[1]) { p->error = "noncanonical integer literal"; return -1; }
        if (negative && !strcmp(digits, "0")) strcpy(text, "0");
        id = t_constant(p, "int64", "decimal", yyjson_mut_strcpy(p->doc, text));
    }
    t_next(p); return id;
}
static bool t_reduction(const char *s) {
    return !strcmp(s, "sum") || !strcmp(s, "prod") || !strcmp(s, "min") || !strcmp(s, "max") || !strcmp(s, "any") || !strcmp(s, "all");
}
static int t_call(text_parser *p, const char *name) {
    if (t_reduction(name)) {
        int map = t_expr(p, 0), where = -1;
        yyjson_mut_val *axes = yyjson_mut_null(p->doc), *initial = yyjson_mut_null(p->doc);
        char dtype[64] = "auto";
        bool keepdims = false; unsigned keywords = 0;
        while (t_take(p, ",") && !p->error) {
            char keyword[80]; strcpy(keyword, p->token); t_next(p); t_expect(p, "=");
            unsigned bit = !strcmp(keyword, "axis") ? 1 : !strcmp(keyword, "keepdims") ? 2 : !strcmp(keyword, "dtype") ? 4 : !strcmp(keyword, "where") ? 8 : !strcmp(keyword, "initial") ? 16 : 0;
            if (!bit || (keywords & bit)) { p->error = "unknown or duplicate reduction keyword"; break; }
            keywords |= bit;
            if (bit == 1) {
                if (t_take(p, "None") || t_take(p, "null")) axes = yyjson_mut_null(p->doc);
                else {
                    char close = 0;
                    if (!strcmp(p->token, "[") || !strcmp(p->token, "(")) { close = p->token[0] == '[' ? ']' : ')'; t_next(p); }
                    axes = yyjson_mut_arr(p->doc); int count = 0;
                    while (!p->error && !(close && p->token[0] == close && !p->token[1])) {
                        bool neg = t_take(p, "-");
                        if (p->kind != 2 || strspn(p->token, "0123456789") != strlen(p->token) || strlen(p->token) > 2 || count == ME_ARRAY_MAX_RANK) { p->error = "invalid reduction axis"; break; }
                        int axis = atoi(p->token) * (neg ? -1 : 1);
                        if (axis < -ME_ARRAY_MAX_RANK || axis >= ME_ARRAY_MAX_RANK) { p->error = "axis exceeds maximum rank"; break; }
                        t_append(p, axes, yyjson_mut_sint(p->doc, axis)); count++;
                        t_next(p); if (!close || !t_take(p, ",")) break;
                    }
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
                yyjson_mut_val *v = id >= before && p->count == before + 1 ? p->nodes[id] : NULL;
                const char *op = yyjson_mut_get_str(yyjson_mut_obj_get(v, "op"));
                if (!op || strcmp(op, "constant")) p->error = "initial requires a scalar literal";
                else {
                    initial = yyjson_mut_val_mut_copy(p->doc, v);
                    if (!initial) { p->oom = true; p->error = "literal allocation failed"; }
                    yyjson_mut_obj_remove_key(initial, "id");
                    yyjson_mut_obj_remove_key(initial, "op");
                }
                p->count = before;
            }
        }
        t_expect(p, ")");
        int id = t_arguments(p, name, &map, 1);
        if (id < 0) return id;
        yyjson_mut_val *node = p->nodes[id];
        t_add(p, node, "axes", axes);
        t_add(p, node, "keepdims", yyjson_mut_bool(p->doc, keepdims));
        t_string(p, node, "dtype", dtype);
        t_add(p, node, "initial", initial);
        t_add(p, node, "where", where < 0 ? yyjson_mut_null(p->doc) : yyjson_mut_uint(p->doc, (uint64_t)where));
        p->reductions[id] = true;
        return id;
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
        int id = t_arguments(p, cast ? "cast" : !strcmp(name, "where") ? "select" : "function", args, n);
        if (id >= 0 && (cast || strcmp(name, "where"))) t_string(p, p->nodes[id], cast ? "dtype" : "name", name);
        return id;
    }
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
            return t_constant(p, "bool", "boolean", yyjson_mut_bool(p->doc, name[0] == 'T' || name[0] == 't'));
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
    p->doc = yyjson_mut_doc_new(NULL);
    if (!p->doc) {
        free(p);
        if (error) snprintf(error->native.message, sizeof(error->native.message), "expression allocation failed");
        return ME_GRAPH_ERR_OOM;
    }
    p->text = expression; p->length = length; p->ninputs = ninputs;
    for (int i = 0; i < ninputs; i++) {
        const char *dtype = t_dtype(inputs[i].dtype);
        if (!dtype || !t_name(inputs[i].name)) { p->error = "invalid text input signature"; break; }
        strcpy(p->names[i], inputs[i].name);
        int id = t_node(p, "input");
        if (id < 0) break;
        t_string(p, p->nodes[id], "name", inputs[i].name);
        t_string(p, p->nodes[id], "dtype", dtype);
    }
    t_next(p); int root = p->error ? -1 : t_expr(p, 0);
    if (p->kind) p->error = "unsupported or trailing expression syntax";
    me_graph_status rc = ME_GRAPH_ERR_FORMAT;
    if (!p->error) {
        bool staged = false;
        for (int i = 0; i < p->count; i++) if (i != root && p->reductions[i]) staged = true;
        yyjson_mut_val *graph = yyjson_mut_obj(p->doc), *nodes = yyjson_mut_arr(p->doc);
        yyjson_mut_val *requires = yyjson_mut_arr(p->doc), *output = yyjson_mut_obj(p->doc);
        t_string(p, graph, "format", ME_GRAPH_FORMAT);
        t_string(p, graph, "semantics", ME_GRAPH_SEMANTICS);
        t_append(p, requires, yyjson_mut_str(p->doc, "numeric"));
        if (staged) t_append(p, requires, yyjson_mut_str(p->doc, "staged"));
        t_add(p, graph, "requires", requires);
        for (int i = 0; i < p->count; i++) t_append(p, nodes, p->nodes[i]);
        t_add(p, graph, "nodes", nodes);
        t_add(p, graph, "root", yyjson_mut_uint(p->doc, (uint64_t)root));
        t_string(p, output, "dtype", "auto");
        t_string(p, output, "casting", "unsafe");
        t_add(p, graph, "output", output);
        yyjson_mut_doc_set_root(p->doc, graph);
        yyjson_doc *doc = p->error ? NULL : yyjson_mut_doc_imut_copy(p->doc, NULL);
        if (!doc) { p->oom = true; p->error = "expression allocation failed"; }
        else {
            rc = dsl_graph_prepare_root(yyjson_doc_get_root(doc), options, out, error);
            yyjson_doc_free(doc);
        }
    }
    if (p->error) {
        rc = p->oom ? ME_GRAPH_ERR_OOM : ME_GRAPH_ERR_FORMAT;
        if (error) { error->native.column = (int)p->pos; snprintf(error->native.message, sizeof(error->native.message), "%s", p->error); }
    }
    yyjson_mut_doc_free(p->doc);
    free(p); return rc;
}
