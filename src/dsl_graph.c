/* Bounded graph frontend. Numerical rules, traversal and accumulation belong to
 * the portable compiler and logical-array adapter, not to this decoder. The
 * temporary textual lowering is deliberately tree-only for effectful nodes. */
#include "miniexpr_graph.h"
#include "dsl_graph_internal.h"
#include "dsl_portable_types.h"
#include "dsl_portable_fp.h"
#include "yyjson.h"
#include <limits.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *data;
    size_t size, capacity;
    bool failed, oom;
} graph_buffer;
typedef struct {
    yyjson_val *value;
    const char *op;
    int args[3], nargs, uses, depth;
    bool reachable;
} graph_node;
struct me_graph_plan {
    atomic_uint references;
    me_artifact *map;
    me_artifact *conversion;
    me_artifact *initial_conversion;
    char *json;
    char *map_json;
    size_t json_size;
    int ninputs, mask;
    char *names[ME_MAX_VARS];
    me_dtype types[ME_MAX_VARS];
    int map_indices[ME_MAX_VARS];
    me_array_options reduction;
    uint64_t initial;
    me_dtype initial_dtype, inferred;
};
struct me_graph_schedule {
    me_graph_plan *plan;
    me_graph_input_metadata inputs[ME_MAX_VARS];
    int rank, output_rank;
    int64_t shape[ME_ARRAY_MAX_RANK], output_shape[ME_ARRAY_MAX_RANK];
    me_dtype dtype;
    size_t bytes, scratch, intermediate;
    me_array_options reduction;
};
static me_graph_status g_error(me_graph_error *error, int node, me_graph_status rc, const char *text) {
    if (error) {
        memset(error, 0, sizeof(*error));
        error->node = node;
        error->stage = -1;
        snprintf(error->native.message, sizeof(error->native.message), "%s", text);
    }
    return rc;
}
static me_graph_status g_native(me_graph_error *error, int node, int rc, const me_artifact_error *native) {
    me_graph_status status = rc == ME_ARTIFACT_ERR_OOM ? ME_GRAPH_ERR_OOM :
        rc == ME_ARTIFACT_ERR_UNSUPPORTED ? ME_GRAPH_ERR_UNSUPPORTED : ME_GRAPH_ERR_SIGNATURE;
    g_error(error, node, status, native->message);
    if (error) error->native = *native;
    return status;
}
static void g_add(graph_buffer *b, const char *s) {
    size_t n = strlen(s);
    if (b->failed) return;
    if (n > ME_GRAPH_MAX_BYTES - b->size) { b->failed = true; return; }
    size_t needed = b->size + n + 1;
    if (needed > b->capacity) {
        size_t cap = b->capacity ? b->capacity * 2 : 256;
        if (cap < needed) cap = needed;
        char *p = realloc(b->data, cap);
        if (!p) { b->failed = b->oom = true; return; }
        b->data = p;
        b->capacity = cap;
    }
    memcpy(b->data + b->size, s, n + 1);
    b->size += n;
}
static const char *g_str(yyjson_val *v) {
    const char *s = yyjson_get_str(v);
    return s && strlen(s) == yyjson_get_len(v) ? s : NULL;
}
static bool g_equal(yyjson_val *v, const char *s) {
    const char *p = g_str(v);
    return p && !strcmp(p, s);
}
/* Canonical object-key order, original array/node order, lossless scalar strings. */
static void g_json(graph_buffer *b, yyjson_val *v) {
    if (yyjson_is_obj(v)) {
        yyjson_val *keys[32], *values[32];
        size_t i, n, count = 0;
        yyjson_val *key, *value;
        yyjson_obj_foreach(v, i, n, key, value) {
            if (count == 32) { b->failed = true; return; }
            size_t j = count++;
            while (j && strcmp(g_str(keys[j - 1]), g_str(key)) > 0) {
                keys[j] = keys[j - 1]; values[j] = values[j - 1]; j--;
            }
            keys[j] = key; values[j] = value;
        }
        g_add(b, "{");
        for (i = 0; i < count; i++) {
            if (i) g_add(b, ",");
            g_json(b, keys[i]); g_add(b, ":"); g_json(b, values[i]);
        }
        g_add(b, "}");
    } else if (yyjson_is_arr(v)) {
        size_t i, n; yyjson_val *value;
        g_add(b, "[");
        yyjson_arr_foreach(v, i, n, value) {
            if (i) g_add(b, ",");
            g_json(b, value);
        }
        g_add(b, "]");
    } else {
        char *s = yyjson_val_write(v, 0, NULL);
        if (!s) b->failed = b->oom = true;
        else { g_add(b, s); free(s); }
    }
}
static bool g_tree(yyjson_val *v, int depth) {
    if (depth > 16) return false;
    if (yyjson_is_str(v)) return g_str(v) != NULL;
    size_t i, n; yyjson_val *child, *key;
    if (yyjson_is_obj(v)) {
        if (yyjson_obj_size(v) > 32) return false;
        yyjson_obj_foreach(v, i, n, key, child) {
            if (!g_str(key) || !g_tree(child, depth + 1)) return false;
            size_t j, m; yyjson_val *other, *unused;
            yyjson_obj_foreach(v, j, m, other, unused) {
                if (j >= i) break;
                if (!strcmp(g_str(key), g_str(other))) return false;
            }
        }
    } else if (yyjson_is_arr(v)) {
        if (yyjson_arr_size(v) > ME_GRAPH_MAX_NODES) return false;
        yyjson_arr_foreach(v, i, n, child) if (!g_tree(child, depth + 1)) return false;
    }
    return true;
}
/* Fields listed after required may be absent; no unknown fields accepted. */
static bool g_fields(yyjson_val *v, const char *const *fields, int required, int count) {
    if (!yyjson_is_obj(v)) return false;
    for (int i = 0; i < required; i++) if (!yyjson_obj_get(v, fields[i])) return false;
    size_t i, n; yyjson_val *key, *child;
    yyjson_obj_foreach(v, i, n, key, child) {
        bool known = false;
        for (int j = 0; j < count; j++) known |= !strcmp(g_str(key), fields[j]);
        if (!known) return false;
    }
    return true;
}
static bool g_name(const char *s) {
    if (!s || !*s || strlen(s) > 63) return false;
    for (size_t i = 0; s[i]; i++) {
        bool alpha = (s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z') || s[i] == '_';
        if (!alpha && (!i || s[i] < '0' || s[i] > '9')) return false;
    }
    return true;
}
static me_dtype g_dtype(const char *s) {
    const char *names[] = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"};
    const me_dtype types[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    if (s) for (int i = 0; i < 11; i++) if (!strcmp(s, names[i])) return types[i];
    return ME_AUTO;
}
static size_t g_width(me_dtype d) {
    switch (d) {
        case ME_BOOL: case ME_INT8: case ME_UINT8: return 1;
        case ME_INT16: case ME_UINT16: return 2;
        case ME_INT32: case ME_UINT32: case ME_FLOAT32: return 4;
        case ME_INT64: case ME_UINT64: case ME_FLOAT64: return 8;
        default: return 0;
    }
}
static const char *g_dtype_name(me_dtype d) {
    const char *names[] = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"};
    for (int i = 0; i < 11; i++) if (g_dtype(names[i]) == d) return names[i];
    return "auto";
}
static int g_conversion(me_dtype source, me_dtype target, const char *casting,
    me_artifact **out, me_artifact_error *error) {
    char json[1024];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\"],\"source\":\"def convert(value):\\n    return value\\n\","
        "\"entry_point\":\"convert\",\"inputs\":[{\"name\":\"value\",\"dtype\":\"%s\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"%s\"}}",
        g_dtype_name(source), g_dtype_name(target), casting);
    return me_artifact_load(json, (size_t)n, ME_JIT_OFF, out, error);
}
static int g_initial_conversion(yyjson_val *initial, me_dtype target,
    me_artifact **out, me_artifact_error *error) {
    graph_buffer json = {0};
    g_add(&json, "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\"],\"source\":\"def convert(initial):\\n    return initial\\n\","
        "\"entry_point\":\"convert\",\"inputs\":[],\"constants\":[{\"name\":\"initial\"");
    const char *fields[] = {"dtype", "category", "encoding", "value"};
    for (int i = 0; i < 4; i++) {
        g_add(&json, ",\""); g_add(&json, fields[i]); g_add(&json, "\":");
        g_json(&json, yyjson_obj_get(initial, fields[i]));
    }
    g_add(&json, "}],\"output\":{\"dtype\":\""); g_add(&json, g_dtype_name(target));
    g_add(&json, "\",\"contract\":\"elementwise\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}");
    int rc = json.failed ? ME_ARTIFACT_ERR_OOM :
        me_artifact_load(json.data, json.size, ME_JIT_OFF, out, error);
    if (json.failed && error) {
        memset(error, 0, sizeof(*error));
        snprintf(error->message, sizeof(error->message), "initial conversion allocation failed");
    }
    free(json.data);
    return rc;
}
static bool g_product(int rank, const int64_t *shape, size_t width, size_t *bytes) {
    if (rank < 0 || rank > ME_ARRAY_MAX_RANK || (rank && !shape)) return false;
    size_t count = 1;
    for (int i = 0; i < rank; i++) {
        if (shape[i] < 0 || (uint64_t)shape[i] > SIZE_MAX || (shape[i] && count > SIZE_MAX / (uint64_t)shape[i])) return false;
        count *= (size_t)shape[i];
    }
    if (!width || count > SIZE_MAX / width) return false;
    *bytes = count * width;
    return true;
}
static bool g_signed_size(size_t bytes) {
#if SIZE_MAX > INT64_MAX
    return bytes <= INT64_MAX;
#else
    (void)bytes;
    return true;
#endif
}
static int g_reduce(const char *op) {
    const char *names[] = {"", "sum", "prod", "min", "max", "any", "all"};
    for (int i = 1; i < 7; i++) if (!strcmp(op, names[i])) return i;
    return 0;
}
static const char *g_operator(const char *op, int *arity) {
    const char *names[] = {"add", "sub", "mul", "div", "floordiv", "mod", "pow", "eq", "ne", "lt", "le", "gt", "ge", "bitand", "bitor", "bitxor", "lshift", "rshift", "and", "or", "neg", "pos", "not", "invert"};
    const char *symbols[] = {"+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", "<=", ">", ">=", "&", "|", "^", "<<", ">>", "and", "or", "-", "+", "not ", "~"};
    for (int i = 0; i < 24; i++) if (!strcmp(op, names[i])) {
        *arity = i >= 20 ? 1 : 2;
        return symbols[i];
    }
    return NULL;
}
static void g_expr(graph_buffer *b, graph_node *nodes, int id) {
    graph_node *n = &nodes[id];
    if (!strcmp(n->op, "input") || !strcmp(n->op, "constant")) {
        if (!strcmp(n->op, "input")) g_add(b, g_str(yyjson_obj_get(n->value, "name")));
        else { char name[32]; snprintf(name, sizeof(name), "_me_capture_%d", id); g_add(b, name); }
        return;
    }
    int arity; const char *symbol = g_operator(n->op, &arity);
    g_add(b, "(");
    if (symbol) {
        if (arity == 1) g_add(b, symbol);
        g_expr(b, nodes, n->args[0]);
        if (arity == 2) { g_add(b, " "); g_add(b, symbol); g_add(b, " "); g_expr(b, nodes, n->args[1]); }
    } else {
        const char *name = !strcmp(n->op, "select") ? "where" : g_str(yyjson_obj_get(n->value, !strcmp(n->op, "cast") ? "dtype" : "name"));
        g_add(b, name); g_add(b, "(");
        for (int i = 0; i < n->nargs; i++) { if (i) g_add(b, ", "); g_expr(b, nodes, n->args[i]); }
        g_add(b, ")");
    }
    g_add(b, ")");
}
static char *g_copy(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}
static me_graph_status g_prepare_json(const char *json, size_t length,
    const me_graph_prepare_options *options, me_graph_plan **out, me_graph_error *error) {
    if (out) *out = NULL;
    g_error(error, -1, ME_GRAPH_SUCCESS, "");
    if (!out || !json || !length || length > ME_GRAPH_MAX_BYTES || memchr(json, 0, length))
        return g_error(error, -1, ME_GRAPH_ERR_FORMAT, "invalid graph buffer or byte limit");
    if (options && (options->struct_size != sizeof(*options) || options->version != ME_GRAPH_VERSION ||
        (options->jit != ME_JIT_OFF && options->jit != ME_JIT_ON && options->jit != ME_JIT_DEFAULT)))
        return g_error(error, -1, ME_GRAPH_ERR_FORMAT, "invalid preparation options");
    yyjson_doc *doc = yyjson_read(json, length, 0);
    if (!doc) return g_error(error, -1, ME_GRAPH_ERR_FORMAT, "invalid graph JSON");
    me_graph_status rc = ME_GRAPH_ERR_FORMAT;
    me_graph_plan *p = NULL;
    graph_buffer source = {0}, manifest = {0}, canonical = {0};
    graph_node nodes[ME_GRAPH_MAX_NODES] = {0};
    int input_nodes[ME_MAX_VARS], constant_nodes[ME_MAX_VARS], ni = 0, nc = 0;
    int current = -1, mask_node = -1;
    const char *message = "invalid graph schema, fields, keys or nesting";
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *const root_fields[] = {"format", "semantics", "requires", "nodes", "root", "output"};
    if (!g_tree(root, 0) || !g_fields(root, root_fields, 6, 6)) goto done;
    if (!g_equal(yyjson_obj_get(root, "format"), ME_GRAPH_FORMAT) || !g_equal(yyjson_obj_get(root, "semantics"), ME_GRAPH_SEMANTICS)) {
        rc = ME_GRAPH_ERR_UNSUPPORTED; message = "unsupported graph format or numerical semantics"; goto done;
    }
    yyjson_val *requires = yyjson_obj_get(root, "requires");
    if (!yyjson_is_arr(requires) || yyjson_arr_size(requires) != 1 || !g_equal(yyjson_arr_get(requires, 0), "numeric")) {
        rc = ME_GRAPH_ERR_CAPABILITY; message = "only the numeric graph capability is supported"; goto done;
    }
    const char *const output_fields[] = {"dtype", "casting"};
    yyjson_val *output = yyjson_obj_get(root, "output");
    if (!g_fields(output, output_fields, 2, 2)) goto done;
    yyjson_val *final_type = yyjson_obj_get(output, "dtype");
    const char *casting = g_str(yyjson_obj_get(output, "casting"));
    me_dtype final_dtype = g_dtype(g_str(final_type));
    if ((!g_equal(final_type, "auto") && final_dtype == ME_AUTO) || !casting ||
        (strcmp(casting, "safe") && strcmp(casting, "same_kind") && strcmp(casting, "unsafe"))) {
        rc = ME_GRAPH_ERR_UNSUPPORTED; message = "unsupported final conversion dtype or policy"; goto done;
    }
    yyjson_val *array = yyjson_obj_get(root, "nodes"), *root_id = yyjson_obj_get(root, "root");
    if (!yyjson_is_arr(array) || !yyjson_arr_size(array) || yyjson_arr_size(array) > ME_GRAPH_MAX_NODES ||
        !yyjson_is_uint(root_id) || yyjson_get_uint(root_id) >= yyjson_arr_size(array)) goto done;
    int count = (int)yyjson_arr_size(array), result = (int)yyjson_get_uint(root_id);
    for (int i = 0; i < count; i++) {
        current = i;
        graph_node *n = &nodes[i]; n->value = yyjson_arr_get(array, (size_t)i);
        yyjson_val *id = yyjson_obj_get(n->value, "id");
        n->op = g_str(yyjson_obj_get(n->value, "op"));
        if (!n->op || !yyjson_is_uint(id) || yyjson_get_uint(id) != (uint64_t)i) goto done;
        bool input = !strcmp(n->op, "input"), constant = !strcmp(n->op, "constant");
        int reduction = g_reduce(n->op), arity = -1;
        const char *const input_fields[] = {"id", "op", "name", "dtype"};
        const char *const constant_fields[] = {"id", "op", "dtype", "category", "encoding", "value"};
        const char *const op_fields[] = {"id", "op", "args"};
        const char *const function_fields[] = {"id", "op", "args", "name"};
        const char *const cast_fields[] = {"id", "op", "args", "dtype"};
        const char *const reduction_fields[] = {"id", "op", "args", "axes", "keepdims", "dtype", "initial", "where"};
        if (input || constant) {
            if (!g_fields(n->value, input ? input_fields : constant_fields, input ? 4 : 6, input ? 4 : 6)) goto done;
            me_dtype dtype = g_dtype(g_str(yyjson_obj_get(n->value, "dtype")));
            if (dtype == ME_AUTO) { rc = ME_GRAPH_ERR_SIGNATURE; message = "unsupported input/constant dtype"; goto done; }
            if (ni + nc == ME_MAX_VARS) { message = "binding limit exceeded"; goto done; }
            if (input) {
                const char *name = g_str(yyjson_obj_get(n->value, "name"));
                if (!g_name(name)) { message = "invalid binding name"; goto done; }
                for (int j = 0; j < ni; j++) if (g_equal(yyjson_obj_get(nodes[input_nodes[j]].value, "name"), name)) {
                    message = "duplicate binding name"; goto done;
                }
                input_nodes[ni++] = i;
            } else {
                yyjson_val *category = yyjson_obj_get(n->value, "category");
                if (!(g_equal(category, "typed_scalar") || (g_equal(category, "weak") &&
                    (dtype == ME_BOOL || dtype == ME_INT64 || dtype == ME_FLOAT64)))) {
                    rc = ME_GRAPH_ERR_SIGNATURE; message = "invalid scalar category or weak transport dtype"; goto done;
                }
                constant_nodes[nc++] = i;
            }
        } else {
            if (reduction) {
                if (i != result) { rc = ME_GRAPH_ERR_UNSUPPORTED; message = "intermediate reductions are not supported"; goto done; }
                if (!g_fields(n->value, reduction_fields, 8, 8)) goto done;
                arity = 1;
                yyjson_val *where = yyjson_obj_get(n->value, "where");
                if (!yyjson_is_null(where)) {
                    if (!yyjson_is_uint(where) || yyjson_get_uint(where) >= (uint64_t)i) goto done;
                    mask_node = (int)yyjson_get_uint(where);
                    if (strcmp(nodes[mask_node].op, "input") || !g_equal(yyjson_obj_get(nodes[mask_node].value, "dtype"), "bool")) {
                        rc = ME_GRAPH_ERR_UNSUPPORTED; message = "participation requires a strong Boolean input"; goto done;
                    }
                    nodes[mask_node].uses++;
                }
            } else if (g_operator(n->op, &arity)) {
                if (!g_fields(n->value, op_fields, 3, 3)) goto done;
            } else if (!strcmp(n->op, "select")) {
                arity = 3; if (!g_fields(n->value, op_fields, 3, 3)) goto done;
            } else if (!strcmp(n->op, "cast")) {
                arity = 1;
                if (!g_fields(n->value, cast_fields, 4, 4) || g_dtype(g_str(yyjson_obj_get(n->value, "dtype"))) == ME_AUTO) goto done;
            } else if (!strcmp(n->op, "function")) {
                const char *name = g_str(yyjson_obj_get(n->value, "name"));
                if (!g_fields(n->value, function_fields, 4, 4) || !g_name(name)) goto done;
                if (g_reduce(name)) { rc = ME_GRAPH_ERR_UNSUPPORTED; message = "reductions must be graph nodes"; goto done; }
            } else { rc = ME_GRAPH_ERR_UNSUPPORTED; message = "unknown graph opcode"; goto done; }
            yyjson_val *args = yyjson_obj_get(n->value, "args");
            if (!yyjson_is_arr(args) || yyjson_arr_size(args) > 3 ||
                (arity >= 0 && yyjson_arr_size(args) != (size_t)arity)) { message = "invalid operation arity"; goto done; }
            n->nargs = (int)yyjson_arr_size(args);
            for (int j = 0; j < n->nargs; j++) {
                yyjson_val *arg = yyjson_arr_get(args, (size_t)j);
                if (!yyjson_is_uint(arg) || yyjson_get_uint(arg) >= (uint64_t)i) { message = "forward, cyclic or missing node reference"; goto done; }
                n->args[j] = (int)yyjson_get_uint(arg);
                graph_node *child = &nodes[n->args[j]]; child->uses++;
                if (child->depth + 1 > n->depth) n->depth = child->depth + 1;
            }
            if (n->depth > ME_GRAPH_MAX_DEPTH) { message = "graph nesting limit exceeded"; goto done; }
        }
    }
    nodes[result].reachable = true;
    if (mask_node >= 0) nodes[mask_node].reachable = true;
    for (int i = count - 1; i >= 0; i--) if (nodes[i].reachable)
        for (int j = 0; j < nodes[i].nargs; j++) nodes[nodes[i].args[j]].reachable = true;
    for (int i = 0; i < count; i++) {
        current = i;
        if (!nodes[i].reachable) { message = "unreachable graph node"; goto done; }
        if (nodes[i].uses > 1 && strcmp(nodes[i].op, "input") && strcmp(nodes[i].op, "constant")) {
            rc = ME_GRAPH_ERR_UNSUPPORTED; message = "shared computed nodes require a participation-aware scheduler"; goto done;
        }
    }
    p = calloc(1, sizeof(*p));
    if (!p) { rc = ME_GRAPH_ERR_OOM; message = "plan allocation failed"; goto done; }
    atomic_init(&p->references, 1);
    p->mask = -1; p->ninputs = ni;
    p->reduction.version = ME_ARTIFACT_ARRAY_VERSION;
    int reduction = g_reduce(nodes[result].op), map_root = reduction ? nodes[result].args[0] : result;
    p->reduction.reduction = (me_array_reduction)reduction;
    p->reduction.naxes = reduction ? -1 : 0;
    if (reduction) {
        current = result;
        yyjson_val *v = nodes[result].value, *axes = yyjson_obj_get(v, "axes");
        yyjson_val *keepdims = yyjson_obj_get(v, "keepdims");
        if (!yyjson_is_bool(keepdims)) goto done;
        p->reduction.keepdims = yyjson_get_bool(keepdims);
        if (!yyjson_is_null(axes)) {
            if (!yyjson_is_arr(axes) || yyjson_arr_size(axes) > ME_ARRAY_MAX_RANK) goto done;
            p->reduction.naxes = (int)yyjson_arr_size(axes);
            for (int i = 0; i < p->reduction.naxes; i++) {
                yyjson_val *axis = yyjson_arr_get(axes, (size_t)i);
                if (!yyjson_is_int(axis) || yyjson_get_sint(axis) < -ME_ARRAY_MAX_RANK || yyjson_get_sint(axis) >= ME_ARRAY_MAX_RANK) goto done;
                p->reduction.axes[i] = (int)yyjson_get_sint(axis);
            }
        }
        yyjson_val *dtype = yyjson_obj_get(v, "dtype");
        p->reduction.accumulator = g_dtype(g_str(dtype));
        if (p->reduction.accumulator == ME_AUTO && !g_equal(dtype, "auto")) goto done;
        yyjson_val *initial = yyjson_obj_get(v, "initial");
        if (!yyjson_is_null(initial)) {
            const char *const initial_fields[] = {"dtype", "encoding", "value", "category"};
            if (!g_fields(initial, initial_fields, 4, 4) ||
                !(g_equal(yyjson_obj_get(initial, "category"), "typed_scalar") ||
                  g_equal(yyjson_obj_get(initial, "category"), "weak"))) goto done;
            char *s = yyjson_val_write(initial, 0, NULL);
            bool ok = s && dsl_graph_scalar(s, strlen(s), &p->initial_dtype, &p->initial);
            free(s);
            if (!ok) { message = "invalid initial scalar"; goto done; }
            if (g_equal(yyjson_obj_get(initial, "category"), "weak") && p->initial_dtype != ME_BOOL &&
                p->initial_dtype != ME_INT64 && p->initial_dtype != ME_FLOAT64) {
                message = "invalid weak initial transport dtype"; goto done;
            }
            p->reduction.initial = &p->initial;
        }
    }
    g_add(&source, "def graph_map(");
    int binding = 0;
    for (int kind = 0; kind < 2; kind++) for (int j = 0; j < (kind ? nc : ni); j++) {
        int id = kind ? constant_nodes[j] : input_nodes[j];
        if (id == mask_node && nodes[id].uses == 1) continue;
        if (binding++) g_add(&source, ", ");
        if (!kind) g_add(&source, g_str(yyjson_obj_get(nodes[id].value, "name")));
        else { char name[32]; snprintf(name, sizeof(name), "_me_capture_%d", id); g_add(&source, name); }
    }
    g_add(&source, "):\n    return "); g_expr(&source, nodes, map_root); g_add(&source, "\n");
    g_add(&manifest, "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},\"requires\":[\"numeric\"],\"entry_point\":\"graph_map\",\"source\":");
    /* Source contains only generated ASCII names/operators; JSON-escape it. */
    g_add(&manifest, "\"");
    if (!source.failed) for (size_t i = 0; i < source.size; i++) {
        char c[2] = {source.data[i], 0};
        g_add(&manifest, source.data[i] == '\n' ? "\\n" : c);
    }
    g_add(&manifest, "\",\"inputs\":[");
    int map_index = 0;
    for (int j = 0; j < ni; j++) {
        int id = input_nodes[j]; yyjson_val *v = nodes[id].value;
        p->names[j] = g_copy(g_str(yyjson_obj_get(v, "name")));
        p->types[j] = g_dtype(g_str(yyjson_obj_get(v, "dtype")));
        p->map_indices[j] = -1;
        if (!p->names[j]) { rc = ME_GRAPH_ERR_OOM; message = "binding allocation failed"; goto done; }
        if (id == mask_node) p->mask = j;
        if (id == mask_node && nodes[id].uses == 1) continue;
        p->map_indices[j] = map_index;
        if (map_index++) g_add(&manifest, ",");
        g_add(&manifest, "{\"name\":"); g_json(&manifest, yyjson_obj_get(v, "name")); g_add(&manifest, ",\"dtype\":");
        g_json(&manifest, yyjson_obj_get(v, "dtype")); g_add(&manifest, "}");
    }
    g_add(&manifest, "],\"constants\":[");
    for (int j = 0; j < nc; j++) {
        if (j) g_add(&manifest, ",");
        int id = constant_nodes[j]; char text[64];
        snprintf(text, sizeof(text), "{\"name\":\"_me_capture_%d\"", id); g_add(&manifest, text);
        const char *fields[] = {"dtype", "category", "encoding", "value"};
        for (int k = 0; k < 4; k++) {
            g_add(&manifest, ",\""); g_add(&manifest, fields[k]); g_add(&manifest, "\":");
            g_json(&manifest, yyjson_obj_get(nodes[id].value, fields[k]));
        }
        g_add(&manifest, "}");
    }
    g_add(&manifest, "],\"output\":{\"dtype\":\"auto\",\"contract\":\"elementwise\"},\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"context\":{\"ndim\":0}}");
    g_json(&canonical, root);
    if (source.failed || manifest.failed || canonical.failed) {
        rc = source.oom || manifest.oom || canonical.oom ? ME_GRAPH_ERR_OOM : ME_GRAPH_ERR_FORMAT;
        message = "lowering allocation or byte limit exceeded"; goto done;
    }
    me_artifact_error native;
    int status = dsl_graph_load_map(manifest.data, manifest.size, options ? options->jit : ME_JIT_OFF, &p->map, &native);
    if (status) { rc = g_native(error, map_root, status, &native); message = NULL; goto done; }
    p->inferred = me_artifact_inferred_dtype(p->map);
    if (!reduction && final_dtype == ME_AUTO) {
        graph_buffer exported = {0};
        char *marker = strstr(manifest.data, "\"output\":{\"dtype\":\"auto\"");
        char *type = marker + strlen("\"output\":{\"dtype\":\"");
        *type = 0;
        g_add(&exported, manifest.data); g_add(&exported, g_dtype_name(p->inferred)); g_add(&exported, type + 4);
        if (exported.failed) { free(exported.data); rc = exported.oom ? ME_GRAPH_ERR_OOM : ME_GRAPH_ERR_FORMAT; message = "map export allocation or byte limit exceeded"; goto done; }
        p->map_json = exported.data;
    }
    /* Validate accumulator/initial using a rank that covers all legal axes. */
    if (reduction) {
        int rank = 0;
        for (int i = 0; i < p->reduction.naxes; i++) {
            int a = p->reduction.axes[i]; int needed = a < 0 ? -a : a + 1;
            if (needed > rank) rank = needed;
        }
        int64_t shape[ME_ARRAY_MAX_RANK] = {0}, output_shape[ME_ARRAY_MAX_RANK];
        for (int i = 0; i < rank; i++) shape[i] = 1;
        int output_rank; me_dtype dtype;
        me_array_options inference = p->reduction;
        inference.naxes = -1; /* Axis normalization depends on runtime rank. */
        status = me_array_result_shape(p->map, rank, shape, &inference, &output_rank, output_shape, &dtype, &native);
        if (status) {
            rc = ME_GRAPH_ERR_SIGNATURE; message = "invalid accumulator, axes or initial dtype"; goto done;
        }
        if (p->reduction.initial && p->initial_dtype != dtype) {
            status = g_initial_conversion(yyjson_obj_get(nodes[result].value, "initial"), dtype,
                &p->initial_conversion, &native);
            if (status) { rc = g_native(error, result, status, &native); message = NULL; goto done; }
        }
        p->inferred = dtype;
    }
    if (options && options->require_jit && !me_artifact_has_jit(p->map)) {
        rc = ME_GRAPH_ERR_CAPABILITY; message = "required map JIT is unavailable"; goto done;
    }
    if (final_dtype != ME_AUTO && final_dtype != p->inferred) {
        if (!dsl_numpy_can_cast(p->inferred, final_dtype, casting)) {
            rc = ME_GRAPH_ERR_SIGNATURE; message = "result does not satisfy final casting policy"; goto done;
        }
        status = g_conversion(p->inferred, final_dtype, casting, &p->conversion, &native);
        if (status) { rc = g_native(error, result, status, &native); message = NULL; goto done; }
    }
    p->json = canonical.data; p->json_size = canonical.size; canonical.data = NULL;
    *out = p; p = NULL; rc = ME_GRAPH_SUCCESS;
done:
    if (rc && message) g_error(error, current, rc, message);
    me_graph_plan_free(p);
    free(source.data); free(manifest.data); free(canonical.data); yyjson_doc_free(doc);
    return rc;
}
me_graph_status me_graph_prepare_json(const char *json, size_t length,
    const me_graph_prepare_options *options, me_graph_plan **out, me_graph_error *error) {
    if (out) *out = NULL;
    /* Even malformed JSON can contain exceptional real tokens. Decoding and
     * validation must not leak parser/compiler FP flags or caller rounding. */
    fenv_t saved;
    if (feholdexcept(&saved)) return g_error(error, -1, ME_GRAPH_ERR_CAPABILITY, "cannot preserve preparation floating environment");
    if (fesetround(FE_TONEAREST)) {
        fesetenv(&saved);
        return g_error(error, -1, ME_GRAPH_ERR_CAPABILITY, "cannot establish preparation rounding");
    }
    me_graph_status rc = g_prepare_json(json, length, options, out, error);
    if (fesetenv(&saved)) {
        if (out) { me_graph_plan_free(*out); *out = NULL; }
        return g_error(error, -1, ME_GRAPH_ERR_CAPABILITY, "cannot restore preparation floating environment");
    }
    return rc;
}
static int g_find(const me_graph_plan *p, const char *name) {
    if (name) for (int i = 0; i < p->ninputs; i++) if (!strcmp(name, p->names[i])) return i;
    return -1;
}
me_graph_status me_graph_specialize(const me_graph_plan *p,
    const me_graph_input_metadata *inputs, int ninputs,
    const me_graph_specialize_options *options, me_graph_schedule **out, me_graph_error *error) {
    if (out) *out = NULL;
    g_error(error, -1, ME_GRAPH_SUCCESS, "");
    if (!out || !p || ninputs != p->ninputs || (ninputs && !inputs))
        return g_error(error, -1, ME_GRAPH_ERR_BINDING, "input metadata must exactly cover the graph");
    if (options && (options->struct_size != sizeof(*options) || options->version != ME_GRAPH_VERSION || options->tile_items > INT32_MAX))
        return g_error(error, -1, ME_GRAPH_ERR_FORMAT, "invalid specialization options");
    me_graph_schedule *s = calloc(1, sizeof(*s));
    if (!s) return g_error(error, -1, ME_GRAPH_ERR_OOM, "schedule allocation failed");
    bool found[ME_MAX_VARS] = {0};
    me_graph_status rc = ME_GRAPH_ERR_SHAPE; const char *message = "invalid or overflowing input shape";
    for (int i = 0; i < ninputs; i++) {
        int j = g_find(p, inputs[i].name); size_t bytes;
        if (j < 0 || found[j] || inputs[i].dtype != p->types[j]) {
            rc = ME_GRAPH_ERR_BINDING; message = "duplicate, unknown or mistyped metadata binding"; goto fail;
        }
        found[j] = true;
        if (!g_product(inputs[i].rank, inputs[i].shape, g_width(inputs[i].dtype), &bytes)) goto fail;
        s->inputs[j] = inputs[i]; s->inputs[j].name = p->names[j];
        if (p->map_indices[j] >= 0 && inputs[i].rank > s->rank) s->rank = inputs[i].rank;
    }
    for (int a = 0; a < s->rank; a++) s->shape[a] = 1;
    for (int j = 0; j < ninputs; j++) if (p->map_indices[j] >= 0) for (int a = 0; a < s->inputs[j].rank; a++) {
        int axis = s->rank - s->inputs[j].rank + a;
        int64_t extent = s->inputs[j].shape[a];
        if (s->shape[axis] == 1) s->shape[axis] = extent;
        else if (extent != 1 && extent != s->shape[axis]) { message = "incompatible broadcast shapes"; goto fail; }
    }
    if (p->mask >= 0) {
        const me_graph_input_metadata *mask = &s->inputs[p->mask];
        if (mask->rank > s->rank) { message = "mask cannot expand the map domain"; goto fail; }
        for (int a = 0; a < mask->rank; a++) if (mask->shape[a] != 1 && mask->shape[a] != s->shape[s->rank - mask->rank + a]) {
            message = "incompatible participation shape"; goto fail;
        }
    }
    size_t total;
    if (!g_product(s->rank, s->shape, me_artifact_output_itemsize(p->map), &total)) goto fail;
    s->reduction = p->reduction;
    s->reduction.tile_items = options && options->tile_items ? options->tile_items : 1024;
    me_artifact_error native;
    int status = me_array_result_shape(p->map, s->rank, s->shape, &s->reduction,
        &s->output_rank, s->output_shape, &s->dtype, &native);
    if (status) { message = native.message; goto fail; }
    status = dsl_array_validate_domain(p->map, s->rank, s->shape, &s->reduction, p->mask >= 0, &native);
    if (status) { message = native.message; goto fail; }
    if (!g_product(s->output_rank, s->output_shape, g_width(s->dtype), &s->bytes)) goto fail;
    if (p->conversion) {
        s->intermediate = s->bytes;
        if (!g_signed_size(s->intermediate)) { message = "conversion geometry exceeds signed stride range"; goto fail; }
        size_t budget = options && options->intermediate_budget ? options->intermediate_budget : 64 * 1024 * 1024;
        if (s->intermediate > budget) { message = "final conversion intermediate budget exceeded"; goto fail; }
        s->dtype = me_artifact_output_dtype(p->conversion);
        if (!g_product(s->output_rank, s->output_shape, g_width(s->dtype), &s->bytes)) goto fail;
    }
    size_t per_item = me_artifact_output_itemsize(p->map) + (p->mask >= 0 ? 1 : 0);
    for (int j = 0; j < p->ninputs; j++) if (p->map_indices[j] >= 0) per_item += g_width(p->types[j]);
    if (s->reduction.tile_items > SIZE_MAX / per_item) { message = "iterator scratch overflow"; goto fail; }
    s->scratch = s->reduction.tile_items * per_item;
    atomic_fetch_add_explicit((atomic_uint *)&p->references, 1, memory_order_relaxed);
    s->plan = (me_graph_plan *)p; *out = s; return ME_GRAPH_SUCCESS;
fail:
    g_error(error, -1, rc, message); free(s); return rc;
}
me_graph_status me_graph_execute(const me_graph_schedule *s,
    const me_array_view *inputs, int ninputs, void *output, size_t capacity,
    const me_graph_execute_options *options, me_graph_report *report, me_graph_error *error) {
    g_error(error, -1, ME_GRAPH_SUCCESS, "");
    if (report) memset(report, 0, sizeof(*report));
    if (!s || ninputs != s->plan->ninputs || (ninputs && !inputs))
        return g_error(error, -1, ME_GRAPH_ERR_BINDING, "invalid invocation bindings");
    if (options && (options->struct_size != sizeof(*options) || options->version != ME_GRAPH_VERSION || (options->raise_mask & ~15u)))
        return g_error(error, -1, ME_GRAPH_ERR_FORMAT, "invalid execution options");
#ifdef __EMSCRIPTEN__
    if (options && options->raise_mask)
        return g_error(error, -1, ME_GRAPH_ERR_CAPABILITY, "floating status unavailable on this target");
#endif
    const me_graph_plan *p = s->plan;
    if (capacity < s->bytes || (capacity && (!output || (uintptr_t)output > UINTPTR_MAX - capacity)) ||
        (s->bytes && (uintptr_t)output % g_width(s->dtype)))
        return g_error(error, -1, ME_GRAPH_ERR_BINDING, "invalid final output capacity or alignment");
    me_array_view views[ME_MAX_VARS], mask = {0}; bool found[ME_MAX_VARS] = {0};
    for (int i = 0; i < ninputs; i++) {
        int j = g_find(p, inputs[i].name);
        if (j < 0 || found[j] || inputs[i].dtype != p->types[j] || inputs[i].rank != s->inputs[j].rank)
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "duplicate, unknown or mismatched invocation binding");
        found[j] = true;
        if (inputs[i].capacity && (!inputs[i].base || (uintptr_t)inputs[i].base > UINTPTR_MAX - inputs[i].capacity))
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "invalid input allocation");
        if (capacity && inputs[i].capacity && (uintptr_t)output < (uintptr_t)inputs[i].base + inputs[i].capacity &&
            (uintptr_t)inputs[i].base < (uintptr_t)output + capacity)
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "output overlaps input allocation");
        for (int a = 0; a < inputs[i].rank; a++) if (inputs[i].shape[a] != s->inputs[j].shape[a])
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "input shape changed; specialize again");
        if (j == p->mask) mask = inputs[i];
        int index = p->map_indices[j];
        if (index >= 0) { views[index] = inputs[i]; views[index].name = me_artifact_input_name(p->map, index); }
    }
    me_array_options reduction = s->reduction;
    if (p->mask >= 0) reduction.where = &mask;
    me_array_report array; me_artifact_error native;
    int active_stage = 0;
    uint64_t initial = 0;
    me_artifact_fp_status initial_status = {0};
    void *temporary = p->conversion && s->intermediate ? malloc(s->intermediate) : NULL;
    if (p->conversion && s->intermediate && !temporary)
        return g_error(error, -1, ME_GRAPH_ERR_OOM, "final conversion allocation failed");
    int status = dsl_array_preflight(p->map, views, me_artifact_ninputs(p->map), s->rank, s->shape,
        &reduction, p->conversion ? temporary : output, p->conversion ? s->intermediate : capacity, &native);
    if (status) {
        free(temporary);
        return g_error(error, -1, ME_GRAPH_ERR_BINDING, native.message);
    }
    me_array_view value = {0};
    me_array_options conversion = {0};
    conversion.version = ME_ARTIFACT_ARRAY_VERSION; conversion.tile_items = reduction.tile_items;
    if (p->conversion) {
        value.name = "value"; value.dtype = p->inferred; value.base = temporary; value.capacity = s->intermediate;
        value.rank = s->output_rank; size_t stride = g_width(value.dtype);
        for (int a = value.rank - 1; a >= 0; a--) {
            if (!g_signed_size(stride)) { free(temporary); return g_error(error, -1, ME_GRAPH_ERR_BINDING, "conversion stride overflow"); }
            value.shape[a] = s->output_shape[a]; value.strides[a] = (int64_t)stride;
            stride *= (size_t)value.shape[a];
        }
        status = dsl_array_preflight(p->conversion, &value, 1, value.rank, value.shape, &conversion,
            output, capacity, &native);
        if (status) { free(temporary); return g_error(error, -1, ME_GRAPH_ERR_BINDING, native.message); }
    }
    if (p->initial_conversion) {
        me_artifact_eval_descriptor descriptor = {0};
        descriptor.struct_size = sizeof(descriptor); descriptor.version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION;
        descriptor.nitems = 1; descriptor.output_capacity = g_width(p->inferred);
        status = me_artifact_eval_status(p->initial_conversion, NULL, 0, &initial, &descriptor, 0, &initial_status, &native);
        if (status) {
            free(temporary);
            if (report) {
                report->array.fp_flags = initial_status.flags;
                report->array.fp_supported = initial_status.supported;
                report->has_jit = me_artifact_has_jit(p->map);
                report->stages = me_graph_stage_count(p);
            }
            me_graph_status failure = status == ME_ARTIFACT_ERR_UNSUPPORTED ? ME_GRAPH_ERR_CAPABILITY :
                status == ME_ARTIFACT_ERR_OOM ? ME_GRAPH_ERR_OOM : ME_GRAPH_ERR_EXECUTION;
            g_error(error, -1, failure, native.message);
            if (error) { error->stage = 0; error->native = native; }
            return failure;
        }
        reduction.initial = &initial;
    }
    status = me_artifact_eval_array(p->map, views, me_artifact_ninputs(p->map), s->rank, s->shape,
        &reduction, p->conversion ? temporary : output, p->conversion ? s->intermediate : capacity, &array, &native);
    if (!status && p->conversion) {
        active_stage = 1;
        me_array_report converted;
        status = me_artifact_eval_array(p->conversion, &value, 1, value.rank, value.shape, &conversion,
            output, capacity, &converted, &native);
        array.fp_flags |= converted.fp_flags;
        array.gathered_bytes += converted.gathered_bytes;
        array.evaluated_tiles += converted.evaluated_tiles;
        if (converted.temporary_bytes > array.temporary_bytes) array.temporary_bytes = converted.temporary_bytes;
    }
    free(temporary);
    array.fp_flags |= initial_status.flags;
    if (report) { report->array = array; report->has_jit = me_artifact_has_jit(p->map); report->stages = p->conversion ? 2 : 1; }
    if (status) {
        me_graph_status rc = status == ME_ARTIFACT_ERR_OOM ? ME_GRAPH_ERR_OOM :
            status == ME_ARTIFACT_ERR_UNSUPPORTED ? ME_GRAPH_ERR_CAPABILITY :
            status == ME_ARTIFACT_ERR_BINDING ? ME_GRAPH_ERR_BINDING : ME_GRAPH_ERR_EXECUTION;
        g_error(error, -1, rc, native.message);
        if (error) { error->stage = active_stage; error->native = native; }
        return rc;
    }
    if (options && (array.fp_flags & options->raise_mask))
        return g_error(error, -1, ME_GRAPH_ERR_EXECUTION, "active floating operation raised a requested flag");
    return ME_GRAPH_SUCCESS;
}
int me_graph_ninputs(const me_graph_plan *p) {
    return p ? p->ninputs : 0;
}
const char *me_graph_input_name(const me_graph_plan *p, int i) {
    return p && i >= 0 && i < p->ninputs ? p->names[i] : NULL;
}
me_dtype me_graph_input_dtype(const me_graph_plan *p, int i) {
    return p && i >= 0 && i < p->ninputs ? p->types[i] : ME_AUTO;
}
me_dtype me_graph_inferred_dtype(const me_graph_plan *p) {
    return p ? p->inferred : ME_AUTO;
}
bool me_graph_has_jit(const me_graph_plan *p) {
    return p && me_artifact_has_jit(p->map);
}
unsigned me_graph_capabilities(const me_graph_plan *p) {
    return p ? ME_ARTIFACT_CAP_NUMERIC : 0;
}
const char *me_graph_export_json(const me_graph_plan *p, size_t *n) {
    if (n) *n = p ? p->json_size : 0;
    return p ? p->json : NULL;
}
const char *me_graph_export_map_json(const me_graph_plan *p) {
    return p ? p->map_json : NULL;
}
const me_artifact *me_graph_map_artifact(const me_graph_plan *p) {
    return p ? p->map : NULL;
}
int me_graph_output_rank(const me_graph_schedule *s) {
    return s ? s->output_rank : -1;
}
const int64_t *me_graph_output_shape(const me_graph_schedule *s) {
    return s ? s->output_shape : NULL;
}
const int64_t *me_graph_map_shape(const me_graph_schedule *s, int *rank) {
    if (rank) *rank = s ? s->rank : -1;
    return s ? s->shape : NULL;
}
me_dtype me_graph_output_dtype(const me_graph_schedule *s) {
    return s ? s->dtype : ME_AUTO;
}
size_t me_graph_output_bytes(const me_graph_schedule *s) {
    return s ? s->bytes : 0;
}
size_t me_graph_scratch_bytes(const me_graph_schedule *s) {
    return s ? s->scratch : 0;
}
size_t me_graph_intermediate_bytes(const me_graph_schedule *s) {
    return s ? s->intermediate : 0;
}
size_t me_graph_stage_count(const me_graph_plan *p) {
    return p ? p->conversion ? 2 : 1 : 0;
}
size_t me_graph_plan_bytes(const me_graph_plan *p) {
    if (!p) return 0;
    size_t bytes = sizeof(*p) + p->json_size + 1 + (p->map_json ? strlen(p->map_json) + 1 : 0);
    for (int i = 0; i < p->ninputs; i++) bytes += strlen(p->names[i]) + 1;
    return bytes;
}
size_t me_graph_schedule_bytes(const me_graph_schedule *s) {
    return s ? sizeof(*s) : 0;
}
void me_graph_plan_free(me_graph_plan *p) {
    if (!p || atomic_fetch_sub_explicit(&p->references, 1, memory_order_acq_rel) != 1) return;
    me_artifact_free(p->map); me_artifact_free(p->conversion); me_artifact_free(p->initial_conversion);
    free(p->json); free(p->map_json);
    for (int i = 0; i < p->ninputs; i++) free(p->names[i]);
    free(p);
}
void me_graph_schedule_free(me_graph_schedule *s) {
    if (!s) return;
    me_graph_plan_free(s->plan); free(s);
}
