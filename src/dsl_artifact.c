/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE)
**********************************************************************/
#include "miniexpr_artifact.h"
#include "dsl_parser.h"
#include "dsl_eval_internal.h"
#include "dsl_portable_types.h"
#include "dsl_portable_fp.h"
#include "functions.h"
#include "yyjson.h"

#include <ctype.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARTIFACT_MAX_FIELDS 128
#define ARTIFACT_MAX_DEPTH 32

typedef union {
    bool boolean;
    int32_t i32;
    int64_t i64;
    int8_t i8;
    int16_t i16;
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    float f32;
    double f64;
} artifact_scalar;

typedef struct {
    me_variable variable;
    bool constant;
    bool weak;
    artifact_scalar value;
    void *string_value;
} artifact_binding;

struct me_artifact {
    char *source;
    char *entry_point;
    artifact_binding bindings[ME_MAX_VARS];
    int nbindings;
    int ninputs;
    me_dtype output_dtype;
    me_dtype inferred_dtype;
    size_t output_itemsize;
    me_dsl_compiled_program *program;
    unsigned capabilities;
    int context_ndim;
    me_artifact_cardinality cardinality;
    me_dsl_semantic_profile profile;
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
        case ME_INT8: case ME_UINT8: return 1;
        case ME_INT16: case ME_UINT16: return 2;
        case ME_UINT32: return 4;
        case ME_UINT64: return 8;
        case ME_INT32: return sizeof(int32_t);
        case ME_INT64: return sizeof(int64_t);
        case ME_FLOAT32: return sizeof(float);
        case ME_FLOAT64: return sizeof(double);
        default: return 0;
    }
}

static me_dtype artifact_dtype1(yyjson_val *value) {
    me_dtype dtype = artifact_dtype(value);
    const char *name = artifact_string(value);
    if (!name || dtype != ME_AUTO) return dtype;
    const char *names[] = {"int8", "int16", "uint8", "uint16", "uint32", "uint64", "bytes", "unicode32"};
    me_dtype types[] = {ME_INT8, ME_INT16, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_BYTES, ME_STRING};
    for (int i = 0; i < 8; i++) if (!strcmp(name, names[i])) return types[i];
    return ME_AUTO;
}

static bool artifact_width(yyjson_val *object, me_dtype dtype, size_t *width) {
    if (!is_string_dtype(dtype)) {
        *width = dtype_size(dtype);
        return *width > 0;
    }
    yyjson_val *size = yyjson_obj_get(object, "itemsize");
    if (!yyjson_is_uint(size) || !yyjson_get_uint(size) || yyjson_get_uint(size) > ME_ARTIFACT_MAX_BYTES ||
        yyjson_get_uint(size) % dtype_code_unit(dtype)) return false;
    *width = (size_t)yyjson_get_uint(size);
    return true;
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
    bool unsigned_type = dtype >= ME_UINT8 && dtype <= ME_UINT64;
    size_t width = artifact_itemsize(dtype);
    if (!width || (negative && unsigned_type)) return false;
    uint64_t limit = unsigned_type ? (width == 8 ? UINT64_MAX : (UINT64_C(1) << (width * 8)) - 1) :
        (width == 8 ? INT64_MAX : (UINT64_C(1) << (width * 8 - 1)) - 1);
    if (!unsigned_type) limit += negative;
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
    if (unsigned_type) {
        switch (dtype) {
        case ME_UINT8: value->u8 = (uint8_t)magnitude; break;
        case ME_UINT16: value->u16 = (uint16_t)magnitude; break;
        case ME_UINT32: value->u32 = (uint32_t)magnitude; break;
        default: value->u64 = magnitude; break;
        }
    }
    else {
        int64_t integer = negative ? -(int64_t)(magnitude - 1) - 1 : (int64_t)magnitude;
        switch (dtype) {
        case ME_INT8: value->i8 = (int8_t)integer; break;
        case ME_INT16: value->i16 = (int16_t)integer; break;
        case ME_INT32: value->i32 = (int32_t)integer; break;
        default: value->i64 = integer; break;
        }
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
        case ME_BYTES: case ME_STRING: {
            const char *text = artifact_string(value);
            size_t width = binding->variable.itemsize;
            bool unicode = binding->variable.dtype == ME_STRING;
            if (!artifact_equal(encoding, unicode ? "unicode32be-hex" : "bytes-hex") || !text || strlen(text) != width * 2) return false;
            binding->string_value = calloc(1, width);
            if (!binding->string_value) return false;
            for (size_t i = 0; i < width; i++) {
                unsigned byte = 0;
                for (int h = 0; h < 2; h++) {
                    char c = text[2 * i + h];
                    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
                    byte = byte * 16 + (unsigned)(c <= '9' ? c - '0' : c - 'a' + 10);
                }
                if (!unicode) ((uint8_t *)binding->string_value)[i] = (uint8_t)byte;
                else ((uint32_t *)binding->string_value)[i / 4] = (((uint32_t *)binding->string_value)[i / 4] << 8) | byte;
            }
            if (unicode) {
                for (size_t i = 0; i < width / 4; i++) {
                    uint32_t cp = ((uint32_t *)binding->string_value)[i];
                    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
                }
            }
            return true;
        }
        case ME_BOOL:
            if (!artifact_equal(encoding, "boolean") || !yyjson_is_bool(value)) return false;
            binding->value.boolean = yyjson_get_bool(value);
            return true;
        case ME_INT32:
        case ME_INT64:
        case ME_INT8: case ME_INT16:
        case ME_UINT8: case ME_UINT16: case ME_UINT32: case ME_UINT64:
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

static unsigned artifact_expr_capabilities(const me_expr *expr) {
    if (!expr) return 0;
    unsigned required = is_reduction_node(expr) ? 4 : 0;
    if (is_string_dtype(expr->dtype) || TYPE_MASK(expr->type) == ME_STRING_CONSTANT) required |= 16;
    for (int i = 0; i < ARITY(expr->type); i++) required |= artifact_expr_capabilities(expr->parameters[i]);
    return required;
}

static unsigned artifact_block_capabilities(const me_dsl_compiled_block *block) {
    unsigned required = 1;
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_compiled_stmt *stmt = block->stmts[i];
        const me_expr *expr = NULL;
        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN: expr = stmt->as.assign.value.expr; break;
        case ME_DSL_STMT_RETURN: expr = stmt->as.return_stmt.expr.expr; break;
        case ME_DSL_STMT_EXPR: expr = stmt->as.expr_stmt.expr.expr; break;
        case ME_DSL_STMT_BREAK: case ME_DSL_STMT_CONTINUE: expr = stmt->as.flow.cond.expr; required |= 2; break;
        case ME_DSL_STMT_IF:
            expr = stmt->as.if_stmt.cond.expr;
            required |= 2 | artifact_block_capabilities(&stmt->as.if_stmt.then_block);
            if (stmt->as.if_stmt.has_else) required |= artifact_block_capabilities(&stmt->as.if_stmt.else_block);
            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                required |= artifact_expr_capabilities(stmt->as.if_stmt.elif_branches[j].cond.expr);
                required |= artifact_block_capabilities(&stmt->as.if_stmt.elif_branches[j].block);
            }
            break;
        case ME_DSL_STMT_WHILE:
            expr = stmt->as.while_loop.cond.expr;
            required |= 2 | artifact_block_capabilities(&stmt->as.while_loop.body);
            break;
        case ME_DSL_STMT_FOR:
            expr = stmt->as.for_loop.start.expr;
            required |= artifact_expr_capabilities(stmt->as.for_loop.stop.expr) | artifact_expr_capabilities(stmt->as.for_loop.step.expr);
            required |= 2 | artifact_block_capabilities(&stmt->as.for_loop.body);
            break;
        default: break;
        }
        required |= artifact_expr_capabilities(expr);
    }
    return required;
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
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!artifact_tree(root, 0) || !yyjson_is_obj(root)) {
        artifact_error(error, status, "invalid fields, duplicate keys, NUL strings, or nesting limits");
        goto cleanup;
    }
    yyjson_val *schema = yyjson_obj_get(root, "schema_version");
    yyjson_val *language_version = yyjson_obj_get(yyjson_obj_get(root, "language"), "version");
    bool numpy_profile = artifact_equal(schema, "1.1") && artifact_equal(language_version, "1.1");
    if (!numpy_profile && ((artifact_string(schema) && !artifact_equal(schema, ME_ARTIFACT_SCHEMA_VERSION)) ||
        (artifact_string(language_version) && !artifact_equal(language_version, ME_PORTABLE_DSL_VERSION)))) {
        status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED,
                                "unsupported portable artifact version pair; expected 1.0/1.0 or 1.1/1.1");
        goto cleanup;
    }
    const char *const language_fields[] = {"name", "version"};
    const char *const output_fields[] = {"dtype", "contract"};
    const char *const semantics_fields[] = {"fp"};
    const char *const numpy_semantics_fields[] = {"fp", "numeric", "casting"};
    const char *const input_fields[] = {"name", "dtype"};
    const char *const constant_fields[] = {"name", "dtype", "encoding", "value"};
    const char *const numpy_constant_fields[] = {"name", "dtype", "encoding", "value", "category"};
    const char *const string_output_fields[] = {"dtype", "contract", "itemsize"};
    const char *const string_input_fields[] = {"name", "dtype", "itemsize"};
    const char *const string_constant_fields[] = {"name", "dtype", "encoding", "value", "itemsize"};
    const char *const root_fields1[] = {"schema_version", "language", "requires", "source",
        "entry_point", "inputs", "constants", "output", "semantics", "context"};
    if (!artifact_fields(root, root_fields1, 10, true)) {
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
    bool string_output = is_string_dtype(artifact_dtype1(yyjson_obj_get(output, "dtype")));
    if (!artifact_fields(language, language_fields, 2, false) ||
        !artifact_fields(output, string_output ? string_output_fields : output_fields, string_output ? 3 : 2, false) ||
        !artifact_fields(semantics, numpy_profile ? numpy_semantics_fields : semantics_fields, numpy_profile ? 3 : 1, false) ||
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
    if ((!numpy_profile && !artifact_equal(yyjson_obj_get(root, "schema_version"), ME_ARTIFACT_SCHEMA_VERSION)) ||
        !artifact_equal(yyjson_obj_get(language, "name"), "miniexpr") ||
        (!numpy_profile && !artifact_equal(yyjson_obj_get(language, "version"), ME_PORTABLE_DSL_VERSION)) ||
        !(artifact_equal(yyjson_obj_get(output, "contract"), "elementwise") ||
          artifact_equal(yyjson_obj_get(output, "contract"), "block_scalar")) ||
        !artifact_equal(yyjson_obj_get(semantics, "fp"), "strict")) {
        status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported version or semantic requirement");
        goto cleanup;
    }
    if (numpy_profile && (!artifact_equal(yyjson_obj_get(semantics, "numeric"), "numpy-2.5") ||
        !(artifact_equal(yyjson_obj_get(semantics, "casting"), "unsafe") ||
          artifact_equal(yyjson_obj_get(semantics, "casting"), "same_kind") ||
          artifact_equal(yyjson_obj_get(semantics, "casting"), "safe")))) {
        status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported 1.1 numerical or casting policy");
        goto cleanup;
    }
    bool core = false;
    unsigned capabilities = 0;
    size_t index, max;
    yyjson_val *value;
    yyjson_arr_foreach(requires, index, max, value) {
        if (!artifact_string(value)) {
            artifact_error(error, status, "capabilities must be strings");
            goto cleanup;
        }
        if (artifact_equal(value, "jit-required")) {
            status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "required JIT is unavailable for the draft interpreter profile");
            goto cleanup;
        }
        unsigned capability = artifact_equal(value, "numeric") ? 1 :
            artifact_equal(value, "control-flow") ? 2 :
            artifact_equal(value, "block-reductions") ? 4 :
            artifact_equal(value, "nd-context") ? 8 : 0;
        if (artifact_equal(value, "fixed-strings")) capability = 16;
        if (!capability) {
            status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported required capability");
            goto cleanup;
        }
        if (capabilities & capability) {
            artifact_error(error, status, "duplicate required capability");
            goto cleanup;
        }
        capabilities |= capability;
        core |= capability == 1;
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
    artifact->capabilities = capabilities;
    artifact->profile = numpy_profile ? ME_DSL_PROFILE_PORTABLE_1_1 : ME_DSL_PROFILE_PORTABLE_1_0;
    {
        const char *const context_fields[] = {"ndim"};
        yyjson_val *context = yyjson_obj_get(root, "context");
        yyjson_val *rank = yyjson_obj_get(context, "ndim");
        if (!artifact_fields(context, context_fields, 1, false) || !yyjson_is_int(rank) ||
            yyjson_get_int(rank) < 0 || yyjson_get_int(rank) > ME_DSL_MAX_NDIM) {
            status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "invalid logical context rank");
            goto cleanup;
        }
        artifact->context_ndim = (int)yyjson_get_int(rank);
        if (artifact->context_ndim && !(capabilities & 8)) {
            status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "missing nd-context capability");
            goto cleanup;
        }
        artifact->cardinality = artifact_equal(yyjson_obj_get(output, "contract"), "block_scalar") ?
            ME_ARTIFACT_BLOCK_SCALAR : ME_ARTIFACT_ELEMENTWISE;
    }
    artifact->source = artifact_copy(source);
    artifact->entry_point = artifact_copy(entry);
    artifact->output_dtype = artifact_dtype1(yyjson_obj_get(output, "dtype"));
    if (!artifact->source || !artifact->entry_point) {
        status = artifact_error(error, ME_ARTIFACT_ERR_OOM, "out of memory");
        goto cleanup;
    }
    if (artifact->output_dtype == ME_AUTO) {
        status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported output dtype");
        goto cleanup;
    }
    for (int kind = 0; kind < 2; kind++) {
        if (!artifact_width(output, artifact->output_dtype, &artifact->output_itemsize)) {
            status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid output string width");
            goto cleanup;
        }
        yyjson_val *array = kind ? constants : inputs;
        yyjson_arr_foreach(array, index, max, value) {
            me_dtype binding_dtype = artifact_dtype1(yyjson_obj_get(value, "dtype"));
            bool string_binding = is_string_dtype(binding_dtype);
            if (!artifact_fields(value, string_binding ? (kind ? string_constant_fields : string_input_fields) :
                (kind ? (numpy_profile ? numpy_constant_fields : constant_fields) : input_fields),
                (kind ? (numpy_profile && !string_binding ? 5 : 4) : 2) + string_binding, false)) {
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
            binding->variable.dtype = binding_dtype;
            binding->constant = kind != 0;
            if (kind && numpy_profile && !string_binding) {
                yyjson_val *category = yyjson_obj_get(value, "category");
                binding->weak = artifact_equal(category, "weak");
                if (!(binding->weak || artifact_equal(category, "typed_scalar")) ||
                    (binding->weak && binding_dtype != ME_INT64 && binding_dtype != ME_FLOAT64 && binding_dtype != ME_BOOL)) {
                    status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "invalid scalar category or weak scalar dtype");
                    goto cleanup;
                }
            }
            if (!binding->variable.name) {
                status = artifact_error(error, ME_ARTIFACT_ERR_OOM, "out of memory");
                goto cleanup;
            }
            if (binding->variable.dtype == ME_AUTO) {
                status = artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported binding dtype");
                goto cleanup;
            }
            if (!artifact_width(value, binding_dtype, &binding->variable.itemsize) || (string_binding && !(capabilities & 16))) {
                status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid string width or missing fixed-strings capability");
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
    parsed = me_dsl_parse_profile(source, artifact->profile, &parse_error);
    if (!parsed) {
        if (error) {
            error->line = parse_error.line;
            error->column = parse_error.column;
        }
        status = artifact_error(error, strstr(parse_error.message, "out of memory")
                                ? ME_ARTIFACT_ERR_OOM : ME_ARTIFACT_ERR_SOURCE, parse_error.message);
        goto cleanup;
    }
    if (!parsed->name || strcmp(parsed->name, entry) || parsed->nparams != artifact->nbindings) {
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
    for (int b = 0; b < artifact->nbindings; b++) {
        variables[b] = artifact->bindings[b].variable;
        if (artifact->bindings[b].constant) variables[b].type |= ME_DSL_UNIFORM_INPUT;
        if (artifact->bindings[b].weak) variables[b].type |= ME_DSL_WEAK_INPUT;
    }
    {
        char reason[256];
        int position = 0;
        bool is_dsl;
        if (numpy_profile) {
            /* Metadata-only inference: output policy cannot drive intermediates. */
            me_dsl_compiled_program *inferred = dsl_compile_program_profile(source, variables, artifact->nbindings,
                ME_AUTO, artifact->context_ndim, ME_JIT_OFF, artifact->profile,
                &position, &is_dsl, reason, sizeof(reason));
            if (!inferred) {
                status = artifact_error(error, ME_ARTIFACT_ERR_SOURCE, reason);
                goto cleanup;
            }
            bool allowed = is_string_dtype(inferred->output_dtype) ? inferred->output_dtype == artifact->output_dtype :
                dsl_numpy_can_cast(inferred->output_dtype, artifact->output_dtype, artifact_string(yyjson_obj_get(semantics, "casting")));
            artifact->inferred_dtype = inferred->output_dtype;
            dsl_compiled_program_free(inferred);
            if (!allowed) {
                status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "inferred result cannot be converted under the declared cast policy");
                goto cleanup;
            }
        }
        artifact->program = dsl_compile_program_profile(source, variables, artifact->nbindings,
            artifact->output_dtype, artifact->context_ndim, ME_JIT_OFF, artifact->profile,
            &position, &is_dsl, reason, sizeof(reason));
        if (!artifact->program) {
            status = artifact_error(error, ME_ARTIFACT_ERR_SOURCE, reason);
            goto cleanup;
        }
        if (artifact->program->uses_i_mask || artifact->program->uses_n_mask ||
            artifact->program->uses_ndim || artifact->program->uses_flat_idx) {
            unsigned symbols = (unsigned)(artifact->program->uses_i_mask | artifact->program->uses_n_mask);
            if (!artifact->context_ndim || (symbols >> artifact->context_ndim) || !(capabilities & 8)) {
                status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "reserved symbols exceed declared ND context");
                goto cleanup;
            }
        }
        if (artifact_block_capabilities(&artifact->program->block) & ~capabilities) {
            status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "missing source capability declaration");
            goto cleanup;
        }
        if (artifact->program->output_is_scalar != (artifact->cardinality == ME_ARTIFACT_BLOCK_SCALAR)) {
            status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "declared return cardinality disagrees with source");
            goto cleanup;
        }
        if (artifact->program->output_itemsize != artifact->output_itemsize || (string_output && !(capabilities & 16))) {
            status = artifact_error(error, ME_ARTIFACT_ERR_BINDING, "declared output width disagrees with source");
            goto cleanup;
        }
        *out = artifact;
        artifact = NULL;
        status = ME_ARTIFACT_SUCCESS;
        goto cleanup;
    }
cleanup:
    me_dsl_program_free(parsed);
    me_artifact_free(artifact);
    yyjson_doc_free(doc);
    return status;
}

me_artifact_status me_artifact_eval(const me_artifact *artifact,
    const me_artifact_input *inputs, int ninputs, void *output, size_t nitems,
    me_artifact_error *error) {
    artifact_clear_error(error);
    if (!artifact || ninputs != artifact->ninputs || (ninputs && !inputs)) {
        return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid handle, count, or output buffer");
    }
    /* Compatibility call adapter, not an older artifact format/profile. */
    if (artifact->context_ndim || artifact->cardinality != ME_ARTIFACT_ELEMENTWISE ||
        is_string_dtype(artifact->output_dtype)) {
        return artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "this contract requires descriptor evaluation");
    }
    if (nitems > SIZE_MAX / artifact->output_itemsize) {
        return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "output byte extent overflow");
    }
    me_artifact_buffer buffers[ME_MAX_VARS];
    for (int i = 0; i < ninputs; i++) {
        size_t width = artifact_itemsize(inputs[i].dtype);
        if (!width || inputs[i].nitems != nitems || nitems > SIZE_MAX / width) {
            return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "input dtype or length mismatch");
        }
        buffers[i] = (me_artifact_buffer){inputs[i].name, inputs[i].dtype, width,
                                         inputs[i].data, nitems * width};
    }
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems = nitems, .output_capacity = nitems * artifact->output_itemsize};
    return me_artifact_eval_ex(artifact, ninputs ? buffers : NULL, ninputs, output, &descriptor, error);
}

static bool artifact_range_valid(const void *data, size_t length, size_t alignment) {
    if (!length) return true;
    uintptr_t address = (uintptr_t)data;
    return data && address % alignment == 0 && length <= UINTPTR_MAX - address;
}

static bool artifact_ranges_overlap(const void *a, size_t a_length, const void *b, size_t b_length) {
    if (!a_length || !b_length) return false;
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < a_length : x - y < b_length;
}

me_artifact_status me_artifact_eval_status(const me_artifact *artifact,
    const me_artifact_buffer *inputs, int ninputs, void *output,
    const me_artifact_eval_descriptor *descriptor, unsigned raise_mask,
    me_artifact_fp_status *status, me_artifact_error *error) {
    artifact_clear_error(error);
    if (!status) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "floating status output is required");
    status->flags = 0;
#ifdef __EMSCRIPTEN__
    status->supported = 0;
#else
    status->supported = 1;
#endif
    if (!artifact || artifact->profile != ME_DSL_PROFILE_PORTABLE_1_1 || (raise_mask & ~15u) ||
        (raise_mask && !status->supported)) return artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported floating-status profile or policy");
    unsigned *previous = dsl_portable_status_begin(&status->flags);
    me_artifact_status result = me_artifact_eval_ex(artifact, inputs, ninputs, output, descriptor, error);
    dsl_portable_status_end(previous);
    if (result == ME_ARTIFACT_SUCCESS && (status->flags & raise_mask)) {
        return artifact_error(error, ME_ARTIFACT_ERR_EVAL, "floating exception selected by raise policy");
    }
    return result;
}

me_artifact_status me_artifact_eval_ex(const me_artifact *artifact,
    const me_artifact_buffer *inputs, int ninputs, void *output,
    const me_artifact_eval_descriptor *descriptor, me_artifact_error *error) {
    artifact_clear_error(error);
    if (!artifact || !descriptor || descriptor->struct_size < sizeof(*descriptor) ||
        ninputs != artifact->ninputs || (ninputs && !inputs)) {
        return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid extended evaluation descriptor or bindings");
    }
    if (descriptor->version != ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION) {
        return artifact_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unknown evaluation descriptor version");
    }
    size_t nitems = descriptor->nitems;
    if (artifact->program && descriptor->ndim != artifact->context_ndim) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "logical context rank mismatch");
    size_t output_width = artifact->output_itemsize;
    size_t output_count = artifact->cardinality == ME_ARTIFACT_BLOCK_SCALAR ? 1 : nitems;
    if (!output_width || output_count > SIZE_MAX / output_width ||
        descriptor->output_capacity < output_count * output_width ||
        !artifact_range_valid(output, output_count * output_width, is_string_dtype(artifact->output_dtype) ? dtype_code_unit(artifact->output_dtype) : output_width)) {
        return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "output capacity, alignment, or byte extent is invalid");
    }
    for (int i = 0; i < ninputs; i++) {
        size_t width = is_string_dtype(inputs[i].dtype) ? inputs[i].itemsize : artifact_itemsize(inputs[i].dtype);
        if (!width || width != inputs[i].itemsize || nitems > SIZE_MAX / width ||
            inputs[i].capacity < nitems * width ||
            !artifact_range_valid(inputs[i].data, nitems * width, is_string_dtype(inputs[i].dtype) ? dtype_code_unit(inputs[i].dtype) : width)) {
            return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "input itemsize, capacity, alignment, or byte extent is invalid");
        }
        if (artifact_ranges_overlap(inputs[i].data, nitems * width, output, output_count * output_width)) {
            return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "output must not overlap an input buffer");
        }
    }
    if (nitems > INT32_MAX || (descriptor->valid_mask && descriptor->valid_mask_capacity < nitems) ||
        (!descriptor->valid_mask && descriptor->valid_mask_capacity)) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid lane/mask capacity");
    if (descriptor->valid_mask && (!artifact_range_valid(descriptor->valid_mask, nitems, 1) ||
        artifact_ranges_overlap(descriptor->valid_mask, nitems, output, output_count * output_width))) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid or overlapping lane mask");
    if (descriptor->ndim) {
        size_t bytes = (size_t)descriptor->ndim * sizeof(int64_t);
        const int64_t *context[] = {descriptor->logical_shape, descriptor->block_origin, descriptor->block_extent};
        for (int i = 0; i < 3; i++) {
            if (!artifact_range_valid(context[i], bytes, sizeof(int64_t)) ||
                artifact_ranges_overlap(context[i], bytes, output, output_count * output_width)) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid or overlapping logical context");
        }
    }
    for (size_t lane = 0; lane < nitems; lane++) {
        if (descriptor->valid_mask && descriptor->valid_mask[lane] > 1) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid lane mask value");
    }
    const void *data[ME_MAX_VARS];
    void *owned[ME_MAX_VARS] = {0};
    for (int b = 0; b < artifact->ninputs; b++) {
        int found = -1;
        for (int i = 0; i < ninputs; i++) {
            if (!inputs[i].name) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "NULL input name");
            if (!strcmp(inputs[i].name, artifact->bindings[b].variable.name)) {
                if (found >= 0) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "duplicate input name");
                found = i;
            }
        }
        if (found < 0 || inputs[found].dtype != artifact->bindings[b].variable.dtype ||
            inputs[found].itemsize != artifact->bindings[b].variable.itemsize) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "input signature mismatch");
        data[b] = inputs[found].data;
        if (inputs[found].dtype == ME_STRING) {
            size_t width = inputs[found].itemsize / 4;
            for (size_t lane = 0; lane < nitems; lane++) {
                if (descriptor->valid_mask && !descriptor->valid_mask[lane]) continue;
                const uint32_t *slot = (const uint32_t *)data[b] + lane * width;
                for (size_t unit = 0; unit < width && slot[unit]; unit++) {
                    if (slot[unit] > 0x10ffff || (slot[unit] >= 0xd800 && slot[unit] <= 0xdfff)) return artifact_error(error, ME_ARTIFACT_ERR_BINDING, "invalid Unicode scalar in active input lane");
                }
            }
        }
    }
    me_artifact_status result = ME_ARTIFACT_SUCCESS;
    for (int b = artifact->ninputs; b < artifact->nbindings; b++) {
        size_t width = artifact->bindings[b].variable.itemsize;
        size_t count = nitems ? nitems : 1;
        if (count > SIZE_MAX / width || !(owned[b] = malloc(count * width))) {
            result = artifact_error(error, ME_ARTIFACT_ERR_OOM, "constant buffer allocation failed");
            break;
        }
        const void *constant = artifact->bindings[b].string_value ? artifact->bindings[b].string_value : (const void *)&artifact->bindings[b].value;
        for (size_t i = 0; i < count; i++) memcpy((char *)owned[b] + i * width, constant, width);
        data[b] = owned[b];
    }
    if (result == ME_ARTIFACT_SUCCESS) {
        me_dsl_portable_eval_descriptor native = {(int)nitems, descriptor->valid_mask, descriptor->output_capacity,
            descriptor->ndim, descriptor->logical_shape, descriptor->block_origin, descriptor->block_extent};
        int rc = dsl_eval_program_portable(artifact->program, data, artifact->nbindings, output, &native);
        if (rc) {
            if (error) error->native_status = rc;
            result = artifact_error(error, ME_ARTIFACT_ERR_EVAL, "portable interpreter evaluation failed");
        }
    }
    for (int b = 0; b < artifact->nbindings; b++) free(owned[b]);
    return result;
}

me_artifact_cardinality me_artifact_result_cardinality(const me_artifact *artifact) {
    return artifact ? artifact->cardinality : ME_ARTIFACT_CARDINALITY_INVALID;
}

size_t me_artifact_input_itemsize(const me_artifact *artifact, int index) {
    return artifact && index >= 0 && index < artifact->ninputs ?
           artifact->bindings[index].variable.itemsize : 0;
}

size_t me_artifact_output_itemsize(const me_artifact *artifact) {
    return artifact ? artifact->output_itemsize : 0;
}

const char *me_artifact_source(const me_artifact *artifact) {
    return artifact ? artifact->source : NULL;
}

unsigned int me_artifact_capabilities(const me_artifact *artifact) {
    return artifact && artifact->program ? artifact->capabilities : 0;
}

int me_artifact_context_ndim(const me_artifact *artifact) {
    return artifact ? artifact->context_ndim : -1;
}

const char *me_artifact_schema_version(const me_artifact *artifact) {
    return artifact ? (artifact->profile == ME_DSL_PROFILE_PORTABLE_1_1 ? "1.1" : ME_ARTIFACT_SCHEMA_VERSION) : NULL;
}

me_dtype me_artifact_inferred_dtype(const me_artifact *artifact) {
    return artifact && artifact->profile == ME_DSL_PROFILE_PORTABLE_1_1 ? artifact->inferred_dtype : ME_AUTO;
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
    (void)artifact;
    return false; /* Draft 1.0 is interpreter-first. */
}
void me_artifact_free(me_artifact *artifact) {
    if (!artifact) return;
    dsl_compiled_program_free(artifact->program);
    for (int b = 0; b < artifact->nbindings; b++) {
        free((void *)artifact->bindings[b].variable.name);
        free(artifact->bindings[b].string_value);
    }
    free(artifact->source);
    free(artifact->entry_point);
    free(artifact);
}
