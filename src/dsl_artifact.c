/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE)
**********************************************************************/
#include "miniexpr_artifact.h"
#include "dsl_parser.h"
#include "yyjson.h"

#include <ctype.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARTIFACT_TILE 256
#define ARTIFACT_MAX_FIELDS 128
#define ARTIFACT_MAX_DEPTH 32

typedef union {
    bool boolean;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
} artifact_scalar;

typedef struct {
    me_variable variable;
    bool constant;
    artifact_scalar value;
} artifact_binding;

struct me_artifact {
    char *source;
    char *entry_point;
    artifact_binding bindings[ME_MAX_VARS];
    int nbindings;
    int ninputs;
    me_dtype output_dtype;
    me_jit_mode jit_mode;
    me_expr *expr;
};

static me_artifact_status artifact_error(me_artifact_error *error,
                                        me_artifact_status status, const char *message) {
    if (error) {
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}

static void artifact_clear_error(me_artifact_error *error) {
    if (error) {
        memset(error, 0, sizeof(*error));
    }
}

static const char *artifact_string(yyjson_val *value) {
    const char *text = yyjson_get_str(value);
    return text && strlen(text) == yyjson_get_len(value) ? text : NULL;
}

static bool artifact_equal(yyjson_val *value, const char *expected) {
    const char *text = artifact_string(value);
    return text && !strcmp(text, expected);
}

static char *artifact_copy(const char *text) {
    size_t size = strlen(text) + 1;
    char *result = malloc(size);
    if (result) {
        memcpy(result, text, size);
    }
    return result;
}

/* yyjson preserves object members, including duplicates. Check every object,
 * including informational metadata, before using any keyed lookups. */
static bool artifact_tree(yyjson_val *value, int depth) {
    if (depth > ARTIFACT_MAX_DEPTH) {
        return false;
    }
    if (yyjson_is_str(value)) {
        return artifact_string(value) != NULL;
    }
    if (yyjson_is_obj(value)) {
        if (yyjson_obj_size(value) > ARTIFACT_MAX_FIELDS) {
            return false;
        }
        size_t index, max;
        yyjson_val *key, *child;
        yyjson_obj_foreach(value, index, max, key, child) {
            const char *name = artifact_string(key);
            if (!name || !artifact_tree(child, depth + 1)) {
                return false;
            }
            size_t other_index, other_max;
            yyjson_val *other_key, *other_child;
            yyjson_obj_foreach(value, other_index, other_max, other_key, other_child) {
                if (other_index >= index) {
                    break;
                }
                if (!strcmp(name, yyjson_get_str(other_key))) {
                    return false;
                }
            }
        }
    } else if (yyjson_is_arr(value)) {
        if (yyjson_arr_size(value) > ARTIFACT_MAX_FIELDS) {
            return false;
        }
        size_t index, max;
        yyjson_val *child;
        yyjson_arr_foreach(value, index, max, child) {
            if (!artifact_tree(child, depth + 1)) {
                return false;
            }
        }
    }
    return true;
}

/* Require all schema keys except designated optional metadata. */
static bool artifact_fields(yyjson_val *object, const char *const *fields, size_t count,
                            bool metadata) {
    if (!yyjson_is_obj(object)) {
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (!yyjson_obj_get(object, fields[i])) {
            return false;
        }
    }
    size_t index, max;
    yyjson_val *key, *value;
    yyjson_obj_foreach(object, index, max, key, value) {
        const char *name = artifact_string(key);
        bool found = metadata && !strcmp(name, "metadata") && yyjson_is_obj(value);
        for (size_t i = 0; i < count; i++) {
            found = found || !strcmp(name, fields[i]);
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

static me_dtype artifact_dtype(yyjson_val *value) {
    if (artifact_equal(value, "bool")) return ME_BOOL;
    if (artifact_equal(value, "int32")) return ME_INT32;
    if (artifact_equal(value, "int64")) return ME_INT64;
    if (artifact_equal(value, "float32")) return ME_FLOAT32;
    if (artifact_equal(value, "float64")) return ME_FLOAT64;
    return ME_AUTO;
}

static size_t artifact_itemsize(me_dtype dtype) {
    switch (dtype) {
        case ME_BOOL: return sizeof(bool);
        case ME_INT32: return sizeof(int32_t);
        case ME_INT64: return sizeof(int64_t);
        case ME_FLOAT32: return sizeof(float);
        case ME_FLOAT64: return sizeof(double);
        default: return 0;
    }
}

static bool artifact_integer(const char *text, me_dtype dtype, artifact_scalar *value) {
    if (!text || !*text) {
        return false;
    }
    bool negative = *text == '-';
    const char *digits = text + negative;
    if (!*digits || (*digits == '0' && (digits[1] || negative))) {
        return false;
    }
    uint64_t limit = dtype == ME_INT32 ? INT32_MAX : INT64_MAX;
    limit += negative;
    uint64_t magnitude = 0;
    for (const char *p = digits; *p; p++) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        uint64_t digit = (uint64_t)(*p - '0');
        if (magnitude > (limit - digit) / 10) {
            return false;
        }
        magnitude = magnitude * 10 + digit;
    }
    int64_t integer = negative ? -(int64_t)(magnitude - 1) - 1 : (int64_t)magnitude;
    if (dtype == ME_INT32) {
        value->i32 = (int32_t)integer;
    } else {
        value->i64 = integer;
    }
    return true;
}

static bool artifact_float(const char *text, me_dtype dtype, artifact_scalar *value) {
    size_t digits = dtype == ME_FLOAT32 ? 8 : 16;
    if (!text || strlen(text) != digits) {
        return false;
    }
    uint64_t bits = 0;
    for (size_t i = 0; i < digits; i++) {
        unsigned digit;
        if (text[i] >= '0' && text[i] <= '9') {
            digit = (unsigned)(text[i] - '0');
        } else if (text[i] >= 'a' && text[i] <= 'f') {
            digit = (unsigned)(text[i] - 'a' + 10);
        } else {
            return false;
        }
        bits = (bits << 4) | digit;
    }
    if (dtype == ME_FLOAT32) {
        uint32_t bits32 = (uint32_t)bits;
        memcpy(&value->f32, &bits32, sizeof(bits32));
    } else {
        memcpy(&value->f64, &bits, sizeof(bits));
    }
    return true;
}

static bool artifact_constant(yyjson_val *object, artifact_binding *binding) {
    yyjson_val *encoding = yyjson_obj_get(object, "encoding");
    yyjson_val *value = yyjson_obj_get(object, "value");
    switch (binding->variable.dtype) {
        case ME_BOOL:
            if (!artifact_equal(encoding, "boolean") || !yyjson_is_bool(value)) return false;
            binding->value.boolean = yyjson_get_bool(value);
            return true;
        case ME_INT32:
        case ME_INT64:
            return artifact_equal(encoding, "decimal") &&
                artifact_integer(artifact_string(value), binding->variable.dtype, &binding->value);
        case ME_FLOAT32:
        case ME_FLOAT64:
            return artifact_equal(encoding, "ieee754-hex") &&
                artifact_float(artifact_string(value), binding->variable.dtype, &binding->value);
        default:
            return false;
    }
}

/* Native parsing has already checked the header's pragmas. Find an explicit FP
 * declaration so we can inject strict mode only when absent, without modifying
 * the stored interchange source or overriding a retained compiler preference. */
static bool artifact_has_fp_pragma(const char *source) {
    const char *p = source;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\r') p++;
        if (*p == '#') {
            p++;
            while (*p && *p != '\n' && isspace((unsigned char)*p)) p++;
            if (!strncmp(p, "me:fp", 5)) return true;
            while (*p && *p != '\n') p++;
        } else if (*p && *p != '\n') {
            break;
        }
        if (*p == '\n') p++;
    }
    return false;
}

me_artifact_status me_artifact_load(const char *json, size_t length, me_jit_mode jit_mode,
                                  me_artifact **out, me_artifact_error *error) {
    artifact_clear_error(error);
    if (out) *out = NULL;
    if (!out || !json || !length || length > ME_ARTIFACT_MAX_BYTES || memchr(json, 0, length)) {
        return artifact_error(error, ME_ARTIFACT_ERR_FORMAT, "invalid JSON buffer or artifact size");
    }
    if (jit_mode != ME_JIT_DEFAULT && jit_mode != ME_JIT_ON && jit_mode != ME_JIT_OFF) {
        return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid host JIT mode");
    }
    if (FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53 ||
        sizeof(float) != 4 || sizeof(double) != 8) {
        return artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "IEEE-754 host floats are required");
    }
    yyjson_read_err json_error;
    yyjson_doc *doc = yyjson_read_opts((char *)json, length, 0, NULL, &json_error);
    if (!doc) {
        return artifact_error(error, json_error.code == YYJSON_READ_ERROR_MEMORY_ALLOCATION
                              ? ME_ARTIFACT_ERR_OOM : ME_ARTIFACT_ERR_FORMAT,
                              json_error.msg ? json_error.msg : "invalid JSON");
    }
    me_artifact_status status = ME_ARTIFACT_ERR_FORMAT;
    me_artifact *artifact = NULL;
    me_dsl_program *parsed = NULL;
    char *execution_source = NULL;
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *const root_fields[] = {"schema_version", "language", "requires", "source",
        "entry_point", "inputs", "constants", "output", "semantics"};
    const char *const language_fields[] = {"name", "version"};
    const char *const output_fields[] = {"dtype", "contract"};
    const char *const semantics_fields[] = {"fp"};
    const char *const input_fields[] = {"name", "dtype"};
    const char *const constant_fields[] = {"name", "dtype", "encoding", "value"};
    if (!artifact_tree(root, 0) || !artifact_fields(root, root_fields, 9, true)) {
        artifact_error(error, status, "invalid fields, duplicate keys, NUL strings, or nesting limits");
        goto cleanup;
    }
    yyjson_val *language = yyjson_obj_get(root, "language");
    yyjson_val *output = yyjson_obj_get(root, "output");
    yyjson_val *semantics = yyjson_obj_get(root, "semantics");
    yyjson_val *requires = yyjson_obj_get(root, "requires");
    yyjson_val *inputs = yyjson_obj_get(root, "inputs");
    yyjson_val *constants = yyjson_obj_get(root, "constants");
    const char *source = artifact_string(yyjson_obj_get(root, "source"));
    const char *entry = artifact_string(yyjson_obj_get(root, "entry_point"));
    if (!artifact_fields(language, language_fields, 2, false) ||
        !artifact_fields(output, output_fields, 2, false) ||
        !artifact_fields(semantics, semantics_fields, 1, false) ||
        !yyjson_is_arr(requires) || !yyjson_is_arr(inputs) || !yyjson_is_arr(constants) ||
        !source || !entry || !*entry) {
        artifact_error(error, status, "invalid manifest structure");
        goto cleanup;
    }
    if (!artifact_string(yyjson_obj_get(root, "schema_version")) ||
        !artifact_string(yyjson_obj_get(language, "name")) ||
        !artifact_string(yyjson_obj_get(language, "version")) ||
        !artifact_string(yyjson_obj_get(output, "contract")) ||
        !artifact_string(yyjson_obj_get(output, "dtype")) ||
        !artifact_string(yyjson_obj_get(semantics, "fp"))) {
        artifact_error(error, status, "versions, semantic requirements, and dtypes must be strings");
        goto cleanup;
    }
    if (!artifact_equal(yyjson_obj_get(root, "schema_version"), ME_ARTIFACT_SCHEMA_VERSION) ||
        !artifact_equal(yyjson_obj_get(language, "name"), "miniexpr") ||
        !artifact_equal(yyjson_obj_get(language, "version"), ME_PORTABLE_DSL_VERSION) ||
        !artifact_equal(yyjson_obj_get(output, "contract"), "scalar-per-element") ||
        !artifact_equal(yyjson_obj_get(semantics, "fp"), "strict")) {
        status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported version or semantic requirement");
        goto cleanup;
    }
    bool core = false;
    size_t index, max;
    yyjson_val *value;
    yyjson_arr_foreach(requires, index, max, value) {
        if (!artifact_string(value)) {
            artifact_error(error, status, "capabilities must be strings");
            goto cleanup;
        }
        if (!artifact_equal(value, "core-scalar")) {
            status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported required capability");
            goto cleanup;
        }
        if (core) {
            artifact_error(error, status, "duplicate required capability");
            goto cleanup;
        }
        core = true;
    }
    if (!core || yyjson_arr_size(inputs) + yyjson_arr_size(constants) > ME_MAX_VARS) {
        artifact_error(error, status, "missing core capability or too many bindings");
        goto cleanup;
    }
    artifact = calloc(1, sizeof(*artifact));
    if (!artifact) {
        status = artifact_error(error, ME_ARTIFACT_ERR_OOM, "out of memory");
        goto cleanup;
    }
    artifact->jit_mode = jit_mode;
    artifact->source = artifact_copy(source);
    artifact->entry_point = artifact_copy(entry);
    artifact->output_dtype = artifact_dtype(yyjson_obj_get(output, "dtype"));
    if (!artifact->source || !artifact->entry_point) {
        status = artifact_error(error, ME_ARTIFACT_ERR_OOM, "out of memory");
        goto cleanup;
    }
    if (artifact->output_dtype == ME_AUTO) {
        status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported output dtype");
        goto cleanup;
    }
    for (int kind = 0; kind < 2; kind++) {
        yyjson_val *array = kind ? constants : inputs;
        yyjson_arr_foreach(array, index, max, value) {
            if (!artifact_fields(value, kind ? constant_fields : input_fields, kind ? 4 : 2, false)) {
                artifact_error(error, status, "invalid binding fields");
                goto cleanup;
            }
            const char *name = artifact_string(yyjson_obj_get(value, "name"));
            if (!name || !*name) {
                artifact_error(error, status, "binding name must be a nonempty string");
                goto cleanup;
            }
            if (!artifact_string(yyjson_obj_get(value, "dtype"))) {
                artifact_error(error, status, "binding dtype must be a string");
                goto cleanup;
            }
            for (int j = 0; j < artifact->nbindings; j++) {
                if (!strcmp(name, artifact->bindings[j].variable.name)) {
                    status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "duplicate or overlapping binding name");
                    goto cleanup;
                }
            }
            artifact_binding *binding = &artifact->bindings[artifact->nbindings++];
            binding->variable.name = artifact_copy(name);
            binding->variable.dtype = artifact_dtype(yyjson_obj_get(value, "dtype"));
            binding->constant = kind != 0;
            if (!binding->variable.name) {
                status = artifact_error(error, ME_ARTIFACT_ERR_OOM, "out of memory");
                goto cleanup;
            }
            if (binding->variable.dtype == ME_AUTO) {
                status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported binding dtype");
                goto cleanup;
            }
            if (kind && !artifact_constant(value, binding)) {
                artifact_error(error, status, "invalid scalar encoding or out-of-range constant");
                goto cleanup;
            }
            if (!kind) artifact->ninputs++;
        }
    }
    me_dsl_error parse_error;
    parsed = me_dsl_parse(source, &parse_error);
    if (!parsed) {
        if (error) {
            error->line = parse_error.line;
            error->column = parse_error.column;
        }
        status = artifact_error(error, strstr(parse_error.message, "out of memory")
                                ? ME_ARTIFACT_ERR_OOM : ME_ARTIFACT_ERR_SOURCE, parse_error.message);
        goto cleanup;
    }
    if (strcmp(parsed->name, entry) || parsed->nparams != artifact->nbindings) {
        status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "entry point or binding coverage mismatch");
        goto cleanup;
    }
    int input_index = 0;
    for (int p = 0; p < parsed->nparams; p++) {
        int found = -1;
        for (int b = 0; b < artifact->nbindings; b++) {
            if (!strcmp(parsed->params[p], artifact->bindings[b].variable.name)) found = b;
        }
        if (found < 0 || (!artifact->bindings[found].constant && found != input_index++)) {
            status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "missing binding or inputs not in source order");
            goto cleanup;
        }
    }
    me_variable variables[ME_MAX_VARS];
    for (int b = 0; b < artifact->nbindings; b++) variables[b] = artifact->bindings[b].variable;
    me_portable_error profile_error;
    me_portable_status profile = me_validate_portable_dsl(source, ME_PORTABLE_DSL_VERSION,
        variables, artifact->nbindings, artifact->output_dtype, &profile_error);
    if (profile != ME_PORTABLE_SUCCESS) {
        if (error) {
            error->native_status = profile;
            error->line = profile_error.line;
            error->column = profile_error.column;
        }
        status = artifact_error(error, profile == ME_PORTABLE_ERR_OOM ? ME_ARTIFACT_ERR_OOM :
            profile == ME_PORTABLE_ERR_UNSUPPORTED ? ME_ARTIFACT_ERR_UNSUPPORTED : ME_ARTIFACT_ERR_SOURCE,
            profile_error.message);
        goto cleanup;
    }
    const char *compiled_source = source;
    if (!artifact_has_fp_pragma(source)) {
        const char prefix[] = "# me:fp=strict\n";
        execution_source = malloc(sizeof(prefix) + strlen(source));
        if (!execution_source) {
            status = artifact_error(error, ME_ARTIFACT_ERR_OOM, "out of memory");
            goto cleanup;
        }
        strcpy(execution_source, prefix);
        strcat(execution_source, source);
        compiled_source = execution_source;
    }
    int position = 0;
    int64_t shape[] = {ARTIFACT_TILE};
    int32_t grid[] = {ARTIFACT_TILE};
    int rc = me_compile_nd_jit(compiled_source, variables, artifact->nbindings,
        artifact->output_dtype, 1, shape, grid, grid, jit_mode, &position, &artifact->expr);
    if (rc != ME_COMPILE_SUCCESS) {
        if (error) error->native_status = rc;
        const char *reason = me_get_last_error_message();
        status = artifact_error(error, rc == ME_COMPILE_ERR_OOM ? ME_ARTIFACT_ERR_OOM : ME_ARTIFACT_ERR_SOURCE,
                                reason ? reason : "native compilation failed");
        goto cleanup;
    }
    *out = artifact;
    artifact = NULL;
    status = ME_ARTIFACT_SUCCESS;
cleanup:
    free(execution_source);
    me_dsl_program_free(parsed);
    me_artifact_free(artifact);
    yyjson_doc_free(doc);
    return status;
}

me_artifact_status me_artifact_eval(const me_artifact *artifact,
    const me_artifact_input *inputs, int ninputs, void *output, size_t nitems,
    me_artifact_error *error) {
    artifact_clear_error(error);
    if (!artifact || ninputs != artifact->ninputs || (ninputs && !inputs) ||
        (nitems && !output) || (artifact && nitems > SIZE_MAX / artifact_itemsize(artifact->output_dtype))) {
        return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid handle, count, or output buffer");
    }
    const void *data[ME_MAX_VARS];
    for (int b = 0; b < artifact->ninputs; b++) {
        int found = -1;
        for (int i = 0; i < ninputs; i++) {
            if (!inputs[i].name) {
                return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "input name must not be NULL");
            }
            if (!strcmp(inputs[i].name, artifact->bindings[b].variable.name)) {
                if (found >= 0) {
                    return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "duplicate runtime input name");
                }
                found = i;
            }
        }
        size_t width = artifact_itemsize(artifact->bindings[b].variable.dtype);
        if (found < 0 || inputs[found].dtype != artifact->bindings[b].variable.dtype ||
            inputs[found].nitems != nitems || (nitems && !inputs[found].data) || nitems > SIZE_MAX / width) {
            return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "missing input, dtype, or length mismatch");
        }
        data[b] = inputs[found].data;
    }
    if (!nitems) return ME_ARTIFACT_SUCCESS;
    int nconstants = artifact->nbindings - artifact->ninputs;
    size_t tile = nitems < ARTIFACT_TILE ? nitems : ARTIFACT_TILE;
    unsigned char *constants = nconstants ? malloc((size_t)nconstants * ARTIFACT_TILE * 8) : NULL;
    if (nconstants && !constants) {
        return artifact_error(error, ME_ARTIFACT_ERR_OOM, "out of memory broadcasting constants");
    }
    for (int c = 0; c < nconstants; c++) {
        int b = artifact->ninputs + c;
        size_t width = artifact_itemsize(artifact->bindings[b].variable.dtype);
        unsigned char *buffer = constants + (size_t)c * ARTIFACT_TILE * 8;
        const artifact_scalar *scalar = &artifact->bindings[b].value;
        const void *value = NULL;
        switch (artifact->bindings[b].variable.dtype) {
            case ME_BOOL: value = &scalar->boolean; break;
            case ME_INT32: value = &scalar->i32; break;
            case ME_INT64: value = &scalar->i64; break;
            case ME_FLOAT32: value = &scalar->f32; break;
            case ME_FLOAT64: value = &scalar->f64; break;
            default: break;
        }
        for (size_t i = 0; i < tile; i++) memcpy(buffer + i * width, value, width);
        data[b] = buffer;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.jit_mode = artifact->jit_mode;
    size_t output_width = artifact_itemsize(artifact->output_dtype);
    me_artifact_status status = ME_ARTIFACT_SUCCESS;
    for (size_t offset = 0; offset < nitems;) {
        size_t remaining = nitems - offset;
        int count = (int)(remaining < tile ? remaining : tile);
        const void *block[ME_MAX_VARS];
        for (int b = 0; b < artifact->nbindings; b++) {
            size_t width = artifact_itemsize(artifact->bindings[b].variable.dtype);
            block[b] = artifact->bindings[b].constant ? data[b] : (const unsigned char *)data[b] + offset * width;
        }
        int rc = me_eval(artifact->expr, block, artifact->nbindings,
            (unsigned char *)output + offset * output_width, count, &params);
        if (rc != ME_EVAL_SUCCESS) {
            if (error) error->native_status = rc;
            status = artifact_error(error, rc == ME_EVAL_ERR_OOM ? ME_ARTIFACT_ERR_OOM : ME_ARTIFACT_ERR_EVAL,
                                    "native evaluation failed; output contents are unspecified");
            break;
        }
        offset += (size_t)count;
    }
    free(constants);
    return status;
}

const char *me_artifact_source(const me_artifact *artifact) {
    return artifact ? artifact->source : NULL;
}
const char *me_artifact_entry_point(const me_artifact *artifact) {
    return artifact ? artifact->entry_point : NULL;
}
int me_artifact_ninputs(const me_artifact *artifact) {
    return artifact ? artifact->ninputs : 0;
}
const char *me_artifact_input_name(const me_artifact *artifact, int index) {
    return artifact && index >= 0 && index < artifact->ninputs ? artifact->bindings[index].variable.name : NULL;
}
me_dtype me_artifact_input_dtype(const me_artifact *artifact, int index) {
    return artifact && index >= 0 && index < artifact->ninputs ? artifact->bindings[index].variable.dtype : ME_AUTO;
}
me_dtype me_artifact_output_dtype(const me_artifact *artifact) {
    return artifact ? artifact->output_dtype : ME_AUTO;
}
bool me_artifact_has_jit(const me_artifact *artifact) {
    return artifact && me_expr_has_jit_kernel(artifact->expr);
}
void me_artifact_free(me_artifact *artifact) {
    if (!artifact) return;
    me_free(artifact->expr);
    for (int b = 0; b < artifact->nbindings; b++) free((void *)artifact->bindings[b].variable.name);
    free(artifact->source);
    free(artifact->entry_point);
    free(artifact);
}
