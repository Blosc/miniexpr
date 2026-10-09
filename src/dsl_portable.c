/*********************************************************************
  Copyright (c) 2025-2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/

/* Draft portable validation; the typed native compiler is authoritative. */
#include "miniexpr.h"
#include "dsl_compile_internal.h"
#include "dsl_parser.h"
#include "dsl_eval_internal.h"
#include "functions.h"

#include <stdio.h>
#include <string.h>

static me_portable_status portable_error(me_portable_error *error,
    me_portable_status status, int line, int column, const char *message) {
    if (error) {
        error->line = line;
        error->column = column;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}

static bool portable_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool portable_name(const char *name) {
    if (!name || !portable_ident_start(*name)) return false;
    for (const char *p = name + 1; *p; p++) {
        if (!portable_ident_start(*p) && (*p < '0' || *p > '9')) return false;
    }
    if (!strcmp(name, "_ndim") || !strcmp(name, "_flat_idx")) return false;
    if (strlen(name) > 2 && name[0] == '_' && (name[1] == 'i' || name[1] == 'n')) {
        bool reserved = true;
        for (const char *p = name + 2; *p; p++) reserved &= *p >= '0' && *p <= '9';
        if (reserved) return false;
    }
    return true;
}

me_portable_status me_validate_portable_dsl(const char *source, const char *version,
    const me_variable *inputs, int ninputs, me_dtype output_dtype, me_portable_error *error) {
    return me_validate_portable_dsl_ex(source, version, inputs, ninputs, output_dtype, NULL, error);
}

static bool portable_width1(me_dtype dtype, size_t width) {
    if (dtype == ME_STRING || dtype == ME_BYTES) {
        return width > 0 && width <= 1024 * 1024 && (dtype != ME_STRING || width % 4 == 0);
    }
    bool numeric = dtype == ME_BOOL || dtype == ME_INT8 || dtype == ME_INT16 ||
        dtype == ME_INT32 || dtype == ME_INT64 || dtype == ME_UINT8 || dtype == ME_UINT16 ||
        dtype == ME_UINT32 || dtype == ME_UINT64 || dtype == ME_FLOAT32 || dtype == ME_FLOAT64;
    return numeric && (!width || width == dtype_size(dtype));
}

me_portable_status me_validate_portable_dsl_ex(const char *source, const char *version,
    const me_variable *inputs, int ninputs, me_dtype output_dtype,
    const me_portable_validation_descriptor *descriptor, me_portable_error *error) {
    if (error) memset(error, 0, sizeof(*error));
    if (!version || (strcmp(version, ME_PORTABLE_DSL_VERSION) && strcmp(version, "1.1"))) {
        return portable_error(error, ME_PORTABLE_ERR_VERSION, 0, 0,
                              "unsupported portable DSL version; expected 1.0 or 1.1");
    }
    me_portable_validation_descriptor defaults = {
        .struct_size = sizeof(defaults), .version = ME_PORTABLE_DSL_VALIDATION_DESCRIPTOR_VERSION};
    if (!descriptor) descriptor = &defaults;
    if (descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->version != ME_PORTABLE_DSL_VALIDATION_DESCRIPTOR_VERSION ||
        descriptor->ndim < 0 || descriptor->ndim > ME_DSL_MAX_NDIM ||
        descriptor->cardinality < ME_PORTABLE_CARDINALITY_INFER ||
        descriptor->cardinality > ME_PORTABLE_BLOCK_SCALAR) {
        return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "invalid validation descriptor");
    }
    if (!source) return portable_error(error, ME_PORTABLE_ERR_SOURCE, 0, 0, "source must not be NULL");
    if (ninputs < 0 || ninputs > ME_MAX_VARS || (ninputs && !inputs)) {
        return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "invalid input signature");
    }
    if (!portable_width1(output_dtype, descriptor->output_itemsize)) {
        return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                              "unsupported output dtype or invalid fixed width");
    }
    for (int i = 0; i < ninputs; i++) {
        if (!portable_name(inputs[i].name) || inputs[i].type != ME_VARIABLE ||
            inputs[i].address || inputs[i].context) {
            return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0,
                                  "inputs must be plain named signature variables, not buffers or callbacks");
        }
        if (!portable_width1(inputs[i].dtype, inputs[i].itemsize)) {
            return portable_error(error, ME_PORTABLE_ERR_UNSUPPORTED, 0, 0,
                                  "unsupported input dtype or invalid fixed width");
        }
        for (int j = 0; j < i; j++) {
            if (!strcmp(inputs[i].name, inputs[j].name)) {
                return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "duplicate input name");
            }
        }
    }
    me_dsl_error parse_error;
    me_dsl_semantic_profile profile = !strcmp(version, "1.1") ? ME_DSL_PROFILE_PORTABLE_1_1 : ME_DSL_PROFILE_PORTABLE_1_0;
    me_dsl_program *parsed = me_dsl_parse_profile(source, profile, &parse_error);
    if (!parsed) {
        return portable_error(error, strstr(parse_error.message, "out of memory")
                              ? ME_PORTABLE_ERR_OOM : ME_PORTABLE_ERR_SOURCE,
                              parse_error.line, parse_error.column, parse_error.message);
    }
    bool bound = parsed->nparams == ninputs;
    for (int p = 0; bound && p < parsed->nparams; p++) {
        bool found = false;
        for (int i = 0; i < ninputs; i++) found |= !strcmp(parsed->params[p], inputs[i].name);
        bound = found;
    }
    me_dsl_program_free(parsed);
    if (!bound) return portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0, "parameter binding coverage mismatch");
    int position = -1;
    bool is_dsl = false;
    char reason[256] = {0};
    me_dsl_compiled_program *program = dsl_compile_program_profile(source, inputs, ninputs,
        output_dtype, descriptor->ndim, ME_JIT_OFF, profile,
        &position, &is_dsl, reason, sizeof(reason));
    if (!program) {
        int line = 0, column = 0;
        if (position >= 0) {
            line = column = 1;
            for (int i = 0; i < position && source[i]; i++) {
                if (source[i] == '\n') { line++; column = 1; }
                else column++;
            }
        }
        return portable_error(error, strstr(reason, "out of memory") ? ME_PORTABLE_ERR_OOM : ME_PORTABLE_ERR_SOURCE,
                              line, column, reason[0] ? reason : "native portable compilation failed");
    }
    me_portable_status rc = ME_PORTABLE_SUCCESS;
    if (descriptor->cardinality != ME_PORTABLE_CARDINALITY_INFER &&
        program->output_is_scalar != (descriptor->cardinality == ME_PORTABLE_BLOCK_SCALAR)) {
        rc = portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0,
                            "declared return cardinality disagrees with source");
    } else if ((output_dtype == ME_STRING || output_dtype == ME_BYTES) &&
               program->output_itemsize != descriptor->output_itemsize) {
        rc = portable_error(error, ME_PORTABLE_ERR_SIGNATURE, 0, 0,
                            "declared output width disagrees with source");
    }
    dsl_compiled_program_free(program);
    return rc;
}
