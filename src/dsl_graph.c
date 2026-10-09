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
    int nstages;
    me_graph_plan *stages[ME_GRAPH_MAX_STAGES];
    int sources[ME_GRAPH_MAX_STAGES][ME_MAX_VARS]; /* >=0 stage; -1-input index. */
    int last_consumer[ME_GRAPH_MAX_STAGES];
    bool portable_stage;
    bool declared_staged;
};
struct me_graph_schedule {
    me_graph_plan *plan;
    me_graph_input_metadata inputs[ME_MAX_VARS];
    int rank, output_rank;
    int64_t shape[ME_ARRAY_MAX_RANK], output_shape[ME_ARRAY_MAX_RANK];
    me_dtype dtype;
    size_t bytes, scratch, intermediate;
    me_array_options reduction;
    me_graph_schedule *stages[ME_GRAPH_MAX_STAGES];
};
static me_graph_status g_prepare_staged(yyjson_val *root, const me_graph_prepare_options *options,
    me_graph_plan **out, me_graph_error *error);
static me_graph_status g_split_graph(yyjson_val *root, graph_node *nodes, int count, int result,
    const me_graph_prepare_options *options, me_graph_plan **out, me_graph_error *error);
static me_graph_status g_specialize_staged(me_graph_schedule *schedule,
    const me_graph_specialize_options *options, me_graph_error *error);
static me_graph_status g_execute_staged(const me_graph_schedule *schedule,
    const me_array_view *inputs, int ninputs, void *output, size_t capacity,
    const me_graph_execute_options *options, me_graph_report *report, me_graph_error *error);
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
static bool g_strides(int rank, const int64_t *shape, size_t width, int64_t *strides) {
    size_t stride = width;
    for (int a = rank - 1; a >= 0; a--) {
        if (!g_signed_size(stride)) return false;
        if (strides) strides[a] = (int64_t)stride;
        size_t extent = (size_t)shape[a];
        if (extent && stride > SIZE_MAX / extent) return false;
        stride *= extent;
    }
    return true;
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
    yyjson_read_err decode_error;
    yyjson_doc *doc = yyjson_read_opts((char *)json, length, 0, NULL, &decode_error);
    if (!doc) return g_error(error, -1,
        decode_error.code == YYJSON_READ_ERROR_MEMORY_ALLOCATION ? ME_GRAPH_ERR_OOM : ME_GRAPH_ERR_FORMAT,
        "graph JSON decode failed");
    me_graph_status rc = ME_GRAPH_ERR_FORMAT;
    me_graph_plan *p = NULL;
    graph_buffer source = {0}, manifest = {0}, canonical = {0};
    graph_node nodes[ME_GRAPH_MAX_NODES] = {0};
    int input_nodes[ME_MAX_VARS], constant_nodes[ME_MAX_VARS], ni = 0, nc = 0;
    int current = -1, mask_node = -1;
    const char *message = "invalid graph schema, fields, keys or nesting";
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (g_tree(root, 0) && g_equal(yyjson_obj_get(root, "format"), ME_GRAPH_STAGED_FORMAT)) {
        rc = g_prepare_staged(root, options, out, error);
        yyjson_doc_free(doc);
        return rc;
    }
    const char *const root_fields[] = {"format", "semantics", "requires", "nodes", "root", "output"};
    if (!g_tree(root, 0) || !g_fields(root, root_fields, 6, 6)) goto done;
    if (!g_equal(yyjson_obj_get(root, "format"), ME_GRAPH_FORMAT) || !g_equal(yyjson_obj_get(root, "semantics"), ME_GRAPH_SEMANTICS)) {
        rc = ME_GRAPH_ERR_UNSUPPORTED; message = "unsupported graph format or numerical semantics"; goto done;
    }
    yyjson_val *requires = yyjson_obj_get(root, "requires");
    bool staged_capability = yyjson_is_arr(requires) && yyjson_arr_size(requires) == 2 &&
        g_equal(yyjson_arr_get(requires, 1), "staged");
    if (!yyjson_is_arr(requires) || (yyjson_arr_size(requires) != 1 && !staged_capability) || !g_equal(yyjson_arr_get(requires, 0), "numeric")) {
        rc = ME_GRAPH_ERR_CAPABILITY; message = "unsupported graph capability list; expected numeric or numeric/staged"; goto done;
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
    bool needs_stages = false;
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
                if (i != result) needs_stages = true;
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
    for (int i = count - 1; i >= 0; i--) if (nodes[i].reachable) {
        for (int j = 0; j < nodes[i].nargs; j++) nodes[nodes[i].args[j]].reachable = true;
        if (g_reduce(nodes[i].op)) {
            yyjson_val *where = yyjson_obj_get(nodes[i].value, "where");
            if (!yyjson_is_null(where)) nodes[(int)yyjson_get_uint(where)].reachable = true;
        }
    }
    for (int i = 0; i < count; i++) {
        current = i;
        if (!nodes[i].reachable) { message = "unreachable graph node"; goto done; }
        if (nodes[i].uses > 1 && strcmp(nodes[i].op, "input") && strcmp(nodes[i].op, "constant")) {
            needs_stages = true;
        }
    }
    if (needs_stages) {
        if (!staged_capability) {
            rc = ME_GRAPH_ERR_CAPABILITY; message = "intermediate reductions/shared computed nodes require the staged capability"; goto done;
        }
        rc = g_split_graph(root, nodes, count, result, options, out, error);
        message = NULL;
        goto done;
    }
    p = calloc(1, sizeof(*p));
    if (!p) { rc = ME_GRAPH_ERR_OOM; message = "plan allocation failed"; goto done; }
    atomic_init(&p->references, 1);
    p->declared_staged = staged_capability;
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
            me_artifact_status scalar_status = s ? dsl_graph_scalar(s, strlen(s), &p->initial_dtype, &p->initial) : ME_ARTIFACT_ERR_OOM;
            free(s);
            if (scalar_status) {
                rc = scalar_status == ME_ARTIFACT_ERR_OOM ? ME_GRAPH_ERR_OOM : ME_GRAPH_ERR_FORMAT;
                message = "initial scalar decode failed"; goto done;
            }
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
    if (p->nstages) {
        s->plan = (me_graph_plan *)p;
        rc = g_specialize_staged(s, options, error);
        if (rc) {
            for (int i = 0; i < p->nstages; i++) me_graph_schedule_free(s->stages[i]);
            free(s);
            return rc;
        }
        atomic_fetch_add_explicit((atomic_uint *)&p->references, 1, memory_order_relaxed);
        *out = s;
        return ME_GRAPH_SUCCESS;
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
        if (!g_signed_size(s->intermediate) || !g_strides(s->output_rank, s->output_shape, g_width(s->dtype), NULL)) {
            message = "conversion geometry exceeds signed stride range"; goto fail;
        }
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
    if (p->nstages) return g_execute_staged(s, inputs, ninputs, output, capacity, options, report, error);
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
        value.rank = s->output_rank;
        memcpy(value.shape, s->output_shape, sizeof(value.shape));
        if (!g_strides(value.rank, value.shape, g_width(value.dtype), value.strides)) {
            free(temporary); return g_error(error, -1, ME_GRAPH_ERR_BINDING, "conversion stride overflow");
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
    if (report) {
        report->array = array;
        report->has_jit = me_artifact_has_jit(p->map);
        report->stages = p->conversion ? 2 : 1;
        report->jit_stages = report->has_jit ? 1 : 0;
        report->interpreter_stages = (size_t)active_stage + 1 - report->jit_stages;
    }
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
    if (p && p->nstages) {
        for (int i = 0; i < p->nstages; i++) if (!me_graph_has_jit(p->stages[i])) return false;
        return true;
    }
    return p && me_artifact_has_jit(p->map);
}
unsigned me_graph_capabilities(const me_graph_plan *p) {
    return p ? ME_ARTIFACT_CAP_NUMERIC | (p->nstages || p->declared_staged ? ME_GRAPH_CAP_STAGED : 0) : 0;
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
    if (p && p->nstages) return (size_t)p->nstages;
    return p ? p->conversion ? 2 : 1 : 0;
}
size_t me_graph_plan_bytes(const me_graph_plan *p) {
    if (!p) return 0;
    size_t bytes = sizeof(*p) + p->json_size + 1 + (p->map_json ? strlen(p->map_json) + 1 : 0);
    for (int i = 0; i < p->ninputs; i++) bytes += strlen(p->names[i]) + 1;
    for (int i = 0; i < p->nstages; i++) bytes += me_graph_plan_bytes(p->stages[i]);
    return bytes;
}
size_t me_graph_schedule_bytes(const me_graph_schedule *s) {
    if (!s) return 0;
    size_t bytes = sizeof(*s);
    for (int i = 0; i < s->plan->nstages; i++) bytes += me_graph_schedule_bytes(s->stages[i]);
    return bytes;
}
void me_graph_plan_free(me_graph_plan *p) {
    if (!p || atomic_fetch_sub_explicit(&p->references, 1, memory_order_acq_rel) != 1) return;
    me_artifact_free(p->map); me_artifact_free(p->conversion); me_artifact_free(p->initial_conversion);
    free(p->json); free(p->map_json);
    for (int i = 0; i < p->ninputs; i++) free(p->names[i]);
    for (int i = 0; i < p->nstages; i++) me_graph_plan_free(p->stages[i]);
    free(p);
}
void me_graph_schedule_free(me_graph_schedule *s) {
    if (!s) return;
    for (int i = 0; i < s->plan->nstages; i++) me_graph_schedule_free(s->stages[i]);
    me_graph_plan_free(s->plan); free(s);
}

static me_dtype g_result_dtype(const me_graph_plan *p) {
    return p->conversion ? me_artifact_output_dtype(p->conversion) : p->inferred;
}
static me_graph_status g_split_graph(yyjson_val *root, graph_node *nodes, int count, int result,
    const me_graph_prepare_options *options, me_graph_plan **out, me_graph_error *error) {
    bool boundary[ME_GRAPH_MAX_NODES] = {0};
    bool strong_dependency[ME_GRAPH_MAX_NODES] = {0};
    int stage_ids[ME_GRAPH_MAX_NODES], roots[ME_GRAPH_MAX_STAGES], nstages = 0;
    for (int i = 0; i < count; i++) {
        stage_ids[i] = -1;
        strong_dependency[i] = !strcmp(nodes[i].op, "input") || g_reduce(nodes[i].op);
        for (int j = 0; j < nodes[i].nargs; j++) strong_dependency[i] |= strong_dependency[nodes[i].args[j]];
        if (!strcmp(nodes[i].op, "select") || !strcmp(nodes[i].op, "and") || !strcmp(nodes[i].op, "or") ||
            (g_reduce(nodes[i].op) && !yyjson_is_null(yyjson_obj_get(nodes[i].value, "where"))))
            return g_error(error, i, ME_GRAPH_ERR_UNSUPPORTED, "automatic stages reject conditional or masked participation domains; declare independent stages explicitly");
        boundary[i] = i == result || g_reduce(nodes[i].op) ||
            (nodes[i].uses > 1 && strcmp(nodes[i].op, "input") && strcmp(nodes[i].op, "constant"));
        if (boundary[i]) {
            if (i != result && !g_reduce(nodes[i].op) && !strong_dependency[i])
                return g_error(error, i, ME_GRAPH_ERR_UNSUPPORTED, "shared scalar-only computations cannot be materialized as strong stage inputs");
            if (nstages == ME_GRAPH_MAX_STAGES)
                return g_error(error, i, ME_GRAPH_ERR_FORMAT, "automatic stage limit exceeded");
            stage_ids[i] = nstages;
            roots[nstages++] = i;
        }
    }
    if (!g_equal(yyjson_obj_get(yyjson_obj_get(root, "output"), "dtype"), "auto"))
        return g_error(error, result, ME_GRAPH_ERR_UNSUPPORTED, "staged results require auto output; use explicit cast nodes");
    graph_buffer json = {0};
    g_add(&json, "{\"format\":\"" ME_GRAPH_STAGED_FORMAT "\",\"semantics\":\"" ME_GRAPH_SEMANTICS
        "\",\"requires\":[\"numeric\",\"staged\"],\"inputs\":[");
    bool first = true;
    for (int i = 0; i < count; i++) if (!strcmp(nodes[i].op, "input")) {
        if (!first) g_add(&json, ",");
        first = false;
        g_add(&json, "{\"name\":"); g_json(&json, yyjson_obj_get(nodes[i].value, "name"));
        g_add(&json, ",\"dtype\":"); g_json(&json, yyjson_obj_get(nodes[i].value, "dtype")); g_add(&json, "}");
    }
    g_add(&json, "],\"stages\":[");
    for (int stage = 0; stage < nstages; stage++) {
        int stage_root = roots[stage], mapped[ME_GRAPH_MAX_NODES];
        bool needed[ME_GRAPH_MAX_NODES] = {0};
        needed[stage_root] = true;
        for (int i = stage_root; i >= 0; i--) if (needed[i] && (i == stage_root || !boundary[i])) {
            for (int j = 0; j < nodes[i].nargs; j++) needed[nodes[i].args[j]] = true;
        }
        char text[96];
        snprintf(text, sizeof(text), "%s{\"id\":%d,\"kind\":\"graph\",\"inputs\":{", stage ? "," : "", stage);
        g_add(&json, text);
        first = true;
        for (int i = 0; i <= stage_root; i++) if (needed[i]) {
            bool intermediate = boundary[i] && i != stage_root;
            if (!intermediate && strcmp(nodes[i].op, "input")) continue;
            if (!first) g_add(&json, ",");
            first = false;
            if (intermediate) {
                snprintf(text, sizeof(text), "\"_me_stage_%d\":{\"stage\":%d}", i, stage_ids[i]);
                g_add(&json, text);
            } else {
                yyjson_val *name = yyjson_obj_get(nodes[i].value, "name");
                const char *s = g_str(name);
                if (!strncmp(s, "_me_stage_", strlen("_me_stage_"))) {
                    free(json.data);
                    return g_error(error, i, ME_GRAPH_ERR_BINDING, "input name collides with automatic stage namespace");
                }
                g_json(&json, name); g_add(&json, ":{\"input\":"); g_json(&json, name); g_add(&json, "}");
            }
        }
        g_add(&json, "},\"graph\":{\"format\":\"" ME_GRAPH_FORMAT "\",\"semantics\":\"" ME_GRAPH_SEMANTICS
            "\",\"requires\":[\"numeric\"],\"nodes\":[");
        int n = 0;
        for (int i = 0; i <= stage_root; i++) if (needed[i]) {
            mapped[i] = n++;
            if (mapped[i]) g_add(&json, ",");
            if (boundary[i] && i != stage_root) {
                snprintf(text, sizeof(text), "{\"id\":%d,\"op\":\"input\",\"name\":\"_me_stage_%d\",\"dtype\":\"auto\"}", mapped[i], i);
                g_add(&json, text);
            } else {
                yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
                yyjson_mut_val *node = doc ? yyjson_val_mut_copy(doc, nodes[i].value) : NULL;
                bool ok = node && yyjson_mut_obj_put(node, yyjson_mut_str(doc, "id"), yyjson_mut_uint(doc, (uint64_t)mapped[i]));
                if (ok && nodes[i].nargs) {
                    yyjson_mut_val *args = yyjson_mut_arr(doc);
                    for (int j = 0; ok && j < nodes[i].nargs; j++)
                        ok = yyjson_mut_arr_add_int(doc, args, mapped[nodes[i].args[j]]);
                    ok = ok && yyjson_mut_obj_put(node, yyjson_mut_str(doc, "args"), args);
                }
                char *encoded = ok ? yyjson_mut_val_write(node, 0, NULL) : NULL;
                if (!encoded) json.failed = json.oom = true;
                else g_add(&json, encoded);
                free(encoded); yyjson_mut_doc_free(doc);
            }
        }
        snprintf(text, sizeof(text), "],\"root\":%d,\"output\":", mapped[stage_root]);
        g_add(&json, text);
        g_json(&json, yyjson_obj_get(root, "output")); g_add(&json, "}}");
    }
    char end[32]; snprintf(end, sizeof(end), "],\"root\":%d}", nstages - 1); g_add(&json, end);
    me_graph_status rc = json.failed ? g_error(error, -1, json.oom ? ME_GRAPH_ERR_OOM : ME_GRAPH_ERR_FORMAT,
        "automatic stage allocation or byte limit exceeded") :
        me_graph_prepare_json(json.data, json.size, options, out, error);
    free(json.data);
    return rc;
}
static int g_source(const me_graph_plan *p, yyjson_val *binding, int stage) {
    const char *const input_field[] = {"input"}, *const stage_field[] = {"stage"};
    if (g_fields(binding, input_field, 1, 1)) {
        int index = g_find(p, g_str(yyjson_obj_get(binding, "input")));
        return index < 0 ? INT_MIN : -1 - index;
    }
    yyjson_val *id = yyjson_obj_get(binding, "stage");
    if (g_fields(binding, stage_field, 1, 1) && yyjson_is_uint(id) && yyjson_get_uint(id) < (uint64_t)stage)
        return (int)yyjson_get_uint(id);
    return INT_MIN;
}
static me_dtype g_source_dtype(const me_graph_plan *p, int source) {
    return source < 0 ? p->types[-1 - source] : g_result_dtype(p->stages[source]);
}
static me_graph_status g_prepare_portable(yyjson_val *stage, const me_graph_prepare_options *options,
    me_graph_plan **out, me_graph_error *error) {
    const char *const fields[] = {"cardinality", "context", "effects", "mask"};
    yyjson_val *contract = yyjson_obj_get(stage, "contract"), *artifact = yyjson_obj_get(stage, "artifact");
    if (!g_fields(contract, fields, 4, 4) ||
        !g_equal(yyjson_obj_get(contract, "cardinality"), "elementwise") ||
        !g_equal(yyjson_obj_get(contract, "context"), "none") ||
        !g_equal(yyjson_obj_get(contract, "effects"), "ordered-lazy") ||
        !g_equal(yyjson_obj_get(contract, "mask"), "none") ||
        !g_equal(yyjson_obj_get(artifact, "schema_version"), "1.1") ||
        !g_equal(yyjson_obj_get(yyjson_obj_get(artifact, "language"), "version"), "1.1"))
        return g_error(error, -1, ME_GRAPH_ERR_UNSUPPORTED, "trusted stages require portable 1.1 elementwise/context-free/ordered-lazy/unmasked contracts");
    me_graph_plan *p = calloc(1, sizeof(*p));
    if (!p) return g_error(error, -1, ME_GRAPH_ERR_OOM, "portable stage allocation failed");
    atomic_init(&p->references, 1);
    p->mask = -1;
    p->portable_stage = true;
    p->reduction.version = ME_ARTIFACT_ARRAY_VERSION;
    graph_buffer json = {0};
    g_json(&json, artifact);
    me_artifact_error native = {0};
    int status = json.failed ? ME_ARTIFACT_ERR_OOM :
        me_artifact_load(json.data, json.size, options ? options->jit : ME_JIT_OFF, &p->map, &native);
    free(json.data);
    me_graph_status rc = ME_GRAPH_SUCCESS;
    if (status) rc = g_native(error, -1, status, &native);
    else if (me_artifact_result_cardinality(p->map) != ME_ARTIFACT_ELEMENTWISE ||
        me_artifact_context_ndim(p->map) || me_artifact_capabilities(p->map) != ME_ARTIFACT_CAP_NUMERIC)
        rc = g_error(error, -1, ME_GRAPH_ERR_UNSUPPORTED, "trusted stage must be numeric elementwise without control-flow, reductions or ND context");
    else if (options && options->require_jit && !me_artifact_has_jit(p->map))
        rc = g_error(error, -1, ME_GRAPH_ERR_CAPABILITY, "required trusted-stage JIT unavailable");
    if (!rc) {
        p->ninputs = me_artifact_ninputs(p->map);
        p->inferred = me_artifact_output_dtype(p->map);
        for (int i = 0; i < p->ninputs; i++) {
            p->names[i] = g_copy(me_artifact_input_name(p->map, i));
            p->types[i] = me_artifact_input_dtype(p->map, i);
            p->map_indices[i] = i;
            if (!p->names[i]) rc = g_error(error, -1, ME_GRAPH_ERR_OOM, "trusted signature allocation failed");
        }
    }
    if (rc) me_graph_plan_free(p);
    else *out = p;
    return rc;
}
static me_graph_status g_prepare_staged(yyjson_val *root, const me_graph_prepare_options *options,
    me_graph_plan **out, me_graph_error *error) {
    const char *const fields[] = {"format", "semantics", "requires", "inputs", "stages", "root"};
    yyjson_val *requires = yyjson_obj_get(root, "requires"), *inputs = yyjson_obj_get(root, "inputs");
    yyjson_val *stages = yyjson_obj_get(root, "stages"), *result = yyjson_obj_get(root, "root");
    if (!g_fields(root, fields, 6, 6) || !yyjson_is_arr(inputs) || yyjson_arr_size(inputs) > ME_MAX_VARS ||
        !yyjson_is_arr(stages) || !yyjson_arr_size(stages) || yyjson_arr_size(stages) > ME_GRAPH_MAX_STAGES ||
        !yyjson_is_uint(result) || yyjson_get_uint(result) != yyjson_arr_size(stages) - 1)
        return g_error(error, -1, ME_GRAPH_ERR_FORMAT, "invalid staged schema or stage/root limit");
    if (!g_equal(yyjson_obj_get(root, "semantics"), ME_GRAPH_SEMANTICS) ||
        !yyjson_is_arr(requires) || yyjson_arr_size(requires) != 2 ||
        !g_equal(yyjson_arr_get(requires, 0), "numeric") || !g_equal(yyjson_arr_get(requires, 1), "staged"))
        return g_error(error, -1, ME_GRAPH_ERR_CAPABILITY, "staged graphs require explicit numeric/staged capabilities and supported semantics");
    me_graph_plan *p = calloc(1, sizeof(*p));
    if (!p) return g_error(error, -1, ME_GRAPH_ERR_OOM, "staged plan allocation failed");
    atomic_init(&p->references, 1);
    p->mask = -1;
    p->ninputs = (int)yyjson_arr_size(inputs);
    p->nstages = (int)yyjson_arr_size(stages);
    me_graph_status rc = ME_GRAPH_ERR_FORMAT;
    bool used[ME_MAX_VARS] = {0}, reachable[ME_GRAPH_MAX_STAGES] = {0};
    int current = -1;
    for (int i = 0; i < p->ninputs; i++) {
        const char *const input_fields[] = {"name", "dtype"};
        yyjson_val *v = yyjson_arr_get(inputs, (size_t)i);
        const char *name = g_str(yyjson_obj_get(v, "name"));
        me_dtype dtype = g_dtype(g_str(yyjson_obj_get(v, "dtype")));
        if (!g_fields(v, input_fields, 2, 2) || !g_name(name) || dtype == ME_AUTO) goto failure;
        for (int j = 0; j < i; j++) if (!strcmp(p->names[j], name)) goto failure;
        p->names[i] = g_copy(name);
        if (!p->names[i]) { rc = ME_GRAPH_ERR_OOM; goto failure; }
        p->types[i] = dtype;
    }
    for (int i = 0; i < p->nstages; i++) {
        current = i;
        p->last_consumer[i] = i;
        yyjson_val *stage = yyjson_arr_get(stages, (size_t)i), *id = yyjson_obj_get(stage, "id");
        yyjson_val *bindings = yyjson_obj_get(stage, "inputs");
        const char *const graph_fields[] = {"id", "kind", "inputs", "graph"};
        const char *const portable_fields[] = {"id", "kind", "inputs", "artifact", "contract"};
        bool portable = g_equal(yyjson_obj_get(stage, "kind"), "portable");
        if (!(portable || g_equal(yyjson_obj_get(stage, "kind"), "graph")) ||
            !g_fields(stage, portable ? portable_fields : graph_fields, portable ? 5 : 4, portable ? 5 : 4) ||
            !yyjson_is_uint(id) || yyjson_get_uint(id) != (uint64_t)i || !yyjson_is_obj(bindings) ||
            yyjson_obj_size(bindings) > ME_MAX_VARS) goto failure;
        size_t j, count; yyjson_val *key, *binding;
        yyjson_obj_foreach(bindings, j, count, key, binding) {
            if (!g_name(g_str(key)) || g_source(p, binding, i) == INT_MIN) goto failure;
        }
        if (portable) {
            rc = g_prepare_portable(stage, options, &p->stages[i], error);
        } else {
            yyjson_val *graph = yyjson_obj_get(stage, "graph");
            if (!g_equal(yyjson_obj_get(graph, "format"), ME_GRAPH_FORMAT)) goto failure;
            /* Native inferred signatures replace auto only for stage-bound inputs.
             * Explicit declarations are independently checked after preparation. */
            yyjson_mut_doc *copy = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *value = copy ? yyjson_val_mut_copy(copy, graph) : NULL;
            if (!value) { yyjson_mut_doc_free(copy); rc = ME_GRAPH_ERR_OOM; goto failure; }
            yyjson_mut_doc_set_root(copy, value);
            yyjson_mut_val *nodes = yyjson_mut_obj_get(value, "nodes"), *node;
            yyjson_mut_arr_foreach(nodes, j, count, node) {
                const char *op = yyjson_mut_get_str(yyjson_mut_obj_get(node, "op"));
                if (!op || strcmp(op, "input")) continue;
                const char *name = yyjson_mut_get_str(yyjson_mut_obj_get(node, "name"));
                int source = g_source(p, name ? yyjson_obj_get(bindings, name) : NULL, i);
                if (source == INT_MIN) { yyjson_mut_doc_free(copy); goto failure; }
                const char *dtype = yyjson_mut_get_str(yyjson_mut_obj_get(node, "dtype"));
                if (dtype && !strcmp(dtype, "auto")) {
                    yyjson_mut_val *k = yyjson_mut_str(copy, "dtype");
                    yyjson_mut_val *v = yyjson_mut_str(copy, g_dtype_name(g_source_dtype(p, source)));
                    if (!k || !v || !yyjson_mut_obj_put(node, k, v)) {
                        yyjson_mut_doc_free(copy); rc = ME_GRAPH_ERR_OOM; goto failure;
                    }
                }
            }
            size_t length = 0;
            char *json = yyjson_mut_write(copy, 0, &length);
            yyjson_mut_doc_free(copy);
            if (!json) { rc = ME_GRAPH_ERR_OOM; goto failure; }
            rc = me_graph_prepare_json(json, length, options, &p->stages[i], error);
            free(json);
        }
        if (rc) goto nested_failure;
        me_graph_plan *child = p->stages[i];
        if (child->conversion || child->nstages) {
            rc = ME_GRAPH_ERR_UNSUPPORTED;
            g_error(error, -1, rc, "staged regions must be nonnested with auto result dtype; express conversion as a cast node");
            goto nested_failure;
        }
        if (yyjson_obj_size(bindings) != (size_t)child->ninputs) { rc = ME_GRAPH_ERR_BINDING; goto failure; }
        for (int j = 0; j < child->ninputs; j++) {
            int source = g_source(p, yyjson_obj_get(bindings, child->names[j]), i);
            if (source == INT_MIN) { rc = ME_GRAPH_ERR_BINDING; goto failure; }
            if (child->types[j] != g_source_dtype(p, source)) { rc = ME_GRAPH_ERR_SIGNATURE; goto failure; }
            p->sources[i][j] = source;
            if (source >= 0) p->last_consumer[source] = i;
            else used[-1 - source] = true;
        }
    }
    reachable[p->nstages - 1] = true;
    for (int i = p->nstages - 1; i >= 0; i--) if (reachable[i]) {
        for (int j = 0; j < p->stages[i]->ninputs; j++) if (p->sources[i][j] >= 0)
            reachable[p->sources[i][j]] = true;
    }
    for (int i = 0; i < p->nstages; i++) if (!reachable[i]) goto failure;
    for (int i = 0; i < p->ninputs; i++) if (!used[i]) goto failure;
    graph_buffer canonical = {0};
    g_json(&canonical, root);
    if (canonical.failed) { free(canonical.data); rc = ME_GRAPH_ERR_OOM; goto failure; }
    p->json = canonical.data;
    p->json_size = canonical.size;
    p->inferred = g_result_dtype(p->stages[p->nstages - 1]);
    *out = p;
    return ME_GRAPH_SUCCESS;
failure:
    g_error(error, -1, rc == ME_GRAPH_SUCCESS ? ME_GRAPH_ERR_FORMAT : rc,
        "invalid staged graph, source reference, signature, reachability or allocation");
    if (!rc) rc = ME_GRAPH_ERR_FORMAT;
nested_failure:
    if (error) error->stage = current;
    me_graph_plan_free(p);
    return rc;
}
static me_graph_status g_specialize_staged(me_graph_schedule *s,
    const me_graph_specialize_options *options, me_graph_error *error) {
    const me_graph_plan *p = s->plan;
    size_t budget = options && options->intermediate_budget ? options->intermediate_budget : 64 * 1024 * 1024;
    for (int i = 0; i < p->nstages; i++) {
        me_graph_input_metadata inputs[ME_MAX_VARS] = {0};
        me_graph_plan *child = p->stages[i];
        for (int j = 0; j < child->ninputs; j++) {
            int source = p->sources[i][j];
            if (source < 0) inputs[j] = s->inputs[-1 - source];
            else {
                me_graph_schedule *producer = s->stages[source];
                inputs[j].dtype = producer->dtype;
                inputs[j].rank = producer->output_rank;
                memcpy(inputs[j].shape, producer->output_shape, sizeof(inputs[j].shape));
            }
            inputs[j].name = child->names[j];
        }
        me_graph_status rc = me_graph_specialize(child, inputs, child->ninputs, options, &s->stages[i], error);
        if (rc) {
            if (error) error->stage = i;
            return rc;
        }
        me_graph_schedule *region = s->stages[i];
        if (region->scratch > s->scratch) s->scratch = region->scratch;
        if (i < p->nstages - 1) {
            if (!g_signed_size(region->bytes) ||
                !g_strides(region->output_rank, region->output_shape, g_width(region->dtype), NULL) ||
                region->bytes > budget - s->intermediate)
                return g_error(error, -1, ME_GRAPH_ERR_SHAPE, "staged intermediate budget or stride range exceeded");
            s->intermediate += region->bytes;
        } else {
            s->dtype = region->dtype;
            s->bytes = region->bytes;
            s->rank = region->rank;
            s->output_rank = region->output_rank;
            memcpy(s->shape, region->shape, sizeof(s->shape));
            memcpy(s->output_shape, region->output_shape, sizeof(s->output_shape));
        }
    }
    return ME_GRAPH_SUCCESS;
}
static void g_stage_views(const me_graph_schedule *s, int stage, const me_array_view *external,
    const me_array_view *results, me_array_view *inputs, me_array_view *maps, me_array_options *reduction) {
    const me_graph_plan *p = s->plan, *child = p->stages[stage];
    *reduction = s->stages[stage]->reduction;
    for (int j = 0; j < child->ninputs; j++) {
        int source = p->sources[stage][j];
        inputs[j] = source < 0 ? external[-1 - source] : results[source];
        inputs[j].name = child->names[j];
        if (j == child->mask) reduction->where = &inputs[j];
        if (child->map_indices[j] >= 0) maps[child->map_indices[j]] = inputs[j];
    }
}
static me_graph_status g_execute_staged(const me_graph_schedule *s, const me_array_view *inputs,
    int ninputs, void *output, size_t capacity, const me_graph_execute_options *options,
    me_graph_report *report, me_graph_error *error) {
    const me_graph_plan *p = s->plan;
    me_array_view external[ME_MAX_VARS], results[ME_GRAPH_MAX_STAGES] = {0};
    bool seen[ME_MAX_VARS] = {0};
    if (capacity < s->bytes || (capacity && (!output || (uintptr_t)output > UINTPTR_MAX - capacity)) ||
        (s->bytes && (uintptr_t)output % g_width(s->dtype)))
        return g_error(error, -1, ME_GRAPH_ERR_BINDING, "invalid staged output capacity or alignment");
    for (int i = 0; i < ninputs; i++) {
        int j = g_find(p, inputs[i].name);
        if (j < 0 || seen[j] || inputs[i].dtype != p->types[j] || inputs[i].rank != s->inputs[j].rank)
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "invalid staged invocation binding");
        seen[j] = true;
        for (int a = 0; a < inputs[i].rank; a++) if (inputs[i].shape[a] != s->inputs[j].shape[a])
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "staged shape changed; specialize again");
        if (inputs[i].capacity && (!inputs[i].base || (uintptr_t)inputs[i].base > UINTPTR_MAX - inputs[i].capacity))
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "invalid staged input allocation");
        if (capacity && inputs[i].capacity && (uintptr_t)output < (uintptr_t)inputs[i].base + inputs[i].capacity &&
            (uintptr_t)inputs[i].base < (uintptr_t)output + capacity)
            return g_error(error, -1, ME_GRAPH_ERR_BINDING, "staged output overlaps input");
        external[j] = inputs[i];
    }
    /* Allocate every materialized dependency before executing anything. The
     * budget reports this honest reservation, not a hypothetical live peak. */
    me_graph_status rc = ME_GRAPH_SUCCESS;
    int current = -1;
    for (int i = 0; i < p->nstages; i++) {
        current = i;
        const me_graph_schedule *region = s->stages[i];
        results[i].dtype = region->dtype;
        results[i].rank = region->output_rank;
        memcpy(results[i].shape, region->output_shape, sizeof(results[i].shape));
        results[i].capacity = i == p->nstages - 1 ? capacity : region->bytes;
        results[i].base = i == p->nstages - 1 ? output : region->bytes ? malloc(region->bytes) : NULL;
        if (region->bytes && !results[i].base) { rc = ME_GRAPH_ERR_OOM; goto failure; }
        if (!g_strides(region->output_rank, results[i].shape, g_width(region->dtype), results[i].strides)) {
            rc = ME_GRAPH_ERR_BINDING; goto failure;
        }
    }
    for (int i = 0; i < p->nstages; i++) {
        current = i;
        me_array_view bindings[ME_MAX_VARS], maps[ME_MAX_VARS];
        me_array_options reduction;
        g_stage_views(s, i, external, results, bindings, maps, &reduction);
        const me_graph_schedule *region = s->stages[i];
        me_artifact_error native;
        int status = dsl_array_preflight(region->plan->map, maps, me_artifact_ninputs(region->plan->map),
            region->rank, region->shape, &reduction, (void *)results[i].base, results[i].capacity, &native);
        if (status) {
            rc = g_error(error, -1, ME_GRAPH_ERR_BINDING, native.message);
            goto cleanup;
        }
    }
    for (int i = 0; i < p->nstages; i++) {
        current = i;
        me_array_view bindings[ME_MAX_VARS], maps[ME_MAX_VARS];
        me_array_options reduction;
        g_stage_views(s, i, external, results, bindings, maps, &reduction);
        me_graph_report region = {0};
        rc = me_graph_execute(s->stages[i], bindings, p->stages[i]->ninputs, (void *)results[i].base,
            results[i].capacity, options, &region, error);
        if (report) {
            report->array.fp_flags |= region.array.fp_flags;
            report->array.fp_supported = region.array.fp_supported;
            report->array.gathered_bytes += region.array.gathered_bytes;
            report->array.evaluated_tiles += region.array.evaluated_tiles;
            report->array.zero_copy_tiles += region.array.zero_copy_tiles;
            if (region.array.temporary_bytes > report->array.temporary_bytes)
                report->array.temporary_bytes = region.array.temporary_bytes;
            report->jit_stages += region.jit_stages;
            report->interpreter_stages += region.interpreter_stages;
            report->stages = (size_t)p->nstages;
            report->has_jit = me_graph_has_jit(p);
        }
        if (rc) goto cleanup;
        for (int j = 0; j < i; j++) if (p->last_consumer[j] == i) {
            free((void *)results[j].base);
            results[j].base = NULL;
        }
    }
    goto cleanup;
failure:
    g_error(error, -1, rc, "staged intermediate allocation or geometry failed");
cleanup:
    if (rc && error) error->stage = current;
    for (int i = 0; i < p->nstages - 1; i++) free((void *)results[i].base);
    return rc;
}
const char *me_graph_stage_kind(const me_graph_plan *p, int stage) {
    if (!p || stage < 0 || (size_t)stage >= me_graph_stage_count(p)) return NULL;
    if (p->nstages) return p->stages[stage]->portable_stage ? "portable" : "graph";
    return stage ? "conversion" : p->portable_stage ? "portable" : "graph";
}
int me_graph_stage_last_consumer(const me_graph_plan *p, int stage) {
    if (!p || stage < 0 || (size_t)stage >= me_graph_stage_count(p)) return -1;
    return p->nstages ? p->last_consumer[stage] : (int)me_graph_stage_count(p) - 1;
}
int me_graph_stage_output_rank(const me_graph_schedule *s, int stage) {
    if (!s || stage < 0 || (size_t)stage >= me_graph_stage_count(s->plan)) return -1;
    return s->plan->nstages ? s->stages[stage]->output_rank : s->output_rank;
}
const int64_t *me_graph_stage_output_shape(const me_graph_schedule *s, int stage) {
    if (me_graph_stage_output_rank(s, stage) < 0) return NULL;
    return s->plan->nstages ? s->stages[stage]->output_shape : s->output_shape;
}
me_dtype me_graph_stage_output_dtype(const me_graph_schedule *s, int stage) {
    if (me_graph_stage_output_rank(s, stage) < 0) return ME_AUTO;
    return s->plan->nstages ? s->stages[stage]->dtype : stage ? s->dtype : s->plan->inferred;
}
size_t me_graph_stage_output_bytes(const me_graph_schedule *s, int stage) {
    if (me_graph_stage_output_rank(s, stage) < 0) return 0;
    return s->plan->nstages ? s->stages[stage]->bytes : stage ? s->bytes : s->intermediate ? s->intermediate : s->bytes;
}
int me_graph_schedule_stage_last_consumer(const me_graph_schedule *s, int stage) {
    return s ? me_graph_stage_last_consumer(s->plan, stage) : -1;
}
