/*********************************************************************
  Copyright (c) 2025-2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/

/* Draft menudet validation; the typed native compiler is authoritative. */
#include "miniexpr.h"
#include "dsl_compile_internal.h"
#include "dsl_parser.h"
#include "dsl_eval_internal.h"
#include "functions.h"

#include <stdio.h>
#include <string.h>

static me_menudet_status menudet_error(me_menudet_error *error,
    me_menudet_status status, int line, int column, const char *message) {
    if (error) {
        error->line = line;
        error->column = column;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}

static bool menudet_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool menudet_name(const char *name) {
    if (!name || !menudet_ident_start(*name)) return false;
    for (const char *p = name + 1; *p; p++) {
        if (!menudet_ident_start(*p) && (*p < '0' || *p > '9')) return false;
    }
    if (!strcmp(name, "_ndim") || !strcmp(name, "_flat_idx")) return false;
    if (strlen(name) > 2 && name[0] == '_' && (name[1] == 'i' || name[1] == 'n')) {
        bool reserved = true;
        for (const char *p = name + 2; *p; p++) reserved &= *p >= '0' && *p <= '9';
        if (reserved) return false;
    }
    return true;
}

me_menudet_status me_validate_menudet(const char *source, const char *version,
    const me_variable *inputs, int ninputs, me_dtype output_dtype, me_menudet_error *error) {
    return me_validate_menudet_ex(source, version, inputs, ninputs, output_dtype, NULL, error);
}

static bool menudet_width1(me_dtype dtype, size_t width) {
    if (dtype == ME_STRING || dtype == ME_BYTES) {
        return width > 0 && width <= 1024 * 1024 && (dtype != ME_STRING || width % 4 == 0);
    }
    bool numeric = dtype == ME_BOOL || dtype == ME_INT8 || dtype == ME_INT16 ||
        dtype == ME_INT32 || dtype == ME_INT64 || dtype == ME_UINT8 || dtype == ME_UINT16 ||
        dtype == ME_UINT32 || dtype == ME_UINT64 || dtype == ME_FLOAT32 || dtype == ME_FLOAT64 ||
        dtype == ME_COMPLEX64 || dtype == ME_COMPLEX128;
    return numeric && (!width || width == dtype_size(dtype));
}

me_menudet_status me_validate_menudet_ex(const char *source, const char *version,
    const me_variable *inputs, int ninputs, me_dtype output_dtype,
    const me_menudet_validation_descriptor *descriptor, me_menudet_error *error) {
    if (error) memset(error, 0, sizeof(*error));
    if (!version || strcmp(version, ME_MENUDET_DSL_VERSION)) {
        return menudet_error(error, ME_MENUDET_ERR_VERSION, 0, 0,
                              "unsupported menudet DSL version; expected Menudet 0.1 beta");
    }
    me_menudet_validation_descriptor defaults = {
        .struct_size = sizeof(defaults), .version = ME_MENUDET_DSL_VALIDATION_DESCRIPTOR_VERSION};
    if (!descriptor) descriptor = &defaults;
    if (descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->version != ME_MENUDET_DSL_VALIDATION_DESCRIPTOR_VERSION ||
        descriptor->ndim < 0 || descriptor->ndim > ME_DSL_MAX_NDIM ||
        descriptor->cardinality < ME_MENUDET_CARDINALITY_INFER ||
        descriptor->cardinality > ME_MENUDET_BLOCK_SCALAR) {
        return menudet_error(error, ME_MENUDET_ERR_SIGNATURE, 0, 0, "invalid validation descriptor");
    }
    if (!source) return menudet_error(error, ME_MENUDET_ERR_SOURCE, 0, 0, "source must not be NULL");
    if (ninputs < 0 || ninputs > ME_MAX_VARS || (ninputs && !inputs)) {
        return menudet_error(error, ME_MENUDET_ERR_SIGNATURE, 0, 0, "invalid input signature");
    }
    if (!menudet_width1(output_dtype, descriptor->output_itemsize)) {
        return menudet_error(error, ME_MENUDET_ERR_UNSUPPORTED, 0, 0,
                              "unsupported output dtype or invalid fixed width");
    }
    for (int i = 0; i < ninputs; i++) {
        if (!menudet_name(inputs[i].name) || inputs[i].type != ME_VARIABLE ||
            inputs[i].address || inputs[i].context) {
            return menudet_error(error, ME_MENUDET_ERR_SIGNATURE, 0, 0,
                                  "inputs must be plain named signature variables, not buffers or callbacks");
        }
        if (!menudet_width1(inputs[i].dtype, inputs[i].itemsize)) {
            return menudet_error(error, ME_MENUDET_ERR_UNSUPPORTED, 0, 0,
                                  "unsupported input dtype or invalid fixed width");
        }
        for (int j = 0; j < i; j++) {
            if (!strcmp(inputs[i].name, inputs[j].name)) {
                return menudet_error(error, ME_MENUDET_ERR_SIGNATURE, 0, 0, "duplicate input name");
            }
        }
    }
    me_dsl_error parse_error;
    /* The public beta uses the NumPy-style semantic profile. */
    me_dsl_semantic_profile profile = ME_DSL_PROFILE_MENUDET_0_1;
    me_dsl_program *parsed = me_dsl_parse_profile(source, profile, &parse_error);
    if (!parsed) {
        return menudet_error(error, strstr(parse_error.message, "out of memory")
                              ? ME_MENUDET_ERR_OOM : ME_MENUDET_ERR_SOURCE,
                              parse_error.line, parse_error.column, parse_error.message);
    }
    bool bound = parsed->nparams == ninputs;
    for (int p = 0; bound && p < parsed->nparams; p++) {
        bool found = false;
        for (int i = 0; i < ninputs; i++) found |= !strcmp(parsed->params[p], inputs[i].name);
        bound = found;
    }
    me_dsl_program_free(parsed);
    if (!bound) return menudet_error(error, ME_MENUDET_ERR_SIGNATURE, 0, 0, "parameter binding coverage mismatch");
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
        return menudet_error(error, strstr(reason, "out of memory") ? ME_MENUDET_ERR_OOM : ME_MENUDET_ERR_SOURCE,
                              line, column, reason[0] ? reason : "native menudet compilation failed");
    }
    me_menudet_status rc = ME_MENUDET_SUCCESS;
    if (descriptor->cardinality != ME_MENUDET_CARDINALITY_INFER &&
        program->output_is_scalar != (descriptor->cardinality == ME_MENUDET_BLOCK_SCALAR)) {
        rc = menudet_error(error, ME_MENUDET_ERR_SIGNATURE, 0, 0,
                            "declared return cardinality disagrees with source");
    } else if ((output_dtype == ME_STRING || output_dtype == ME_BYTES) &&
               program->output_itemsize != descriptor->output_itemsize) {
        rc = menudet_error(error, ME_MENUDET_ERR_SIGNATURE, 0, 0,
                            "declared output width disagrees with source");
    }
    dsl_compiled_program_free(program);
    return rc;
}
