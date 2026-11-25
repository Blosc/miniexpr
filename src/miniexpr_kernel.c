/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "miniexpr_kernel.h"
#include "miniexpr_kernel_internal.h"
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include "dsl_menudet_expr.h"
#include "dsl_menudet_fp.h"
#include "dsl_menudet_types.h"
#include "dsl_math_policy.h"
#include "functions-simd.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static me_status kernel_error(me_error *error, me_status status, const char *message) {
    if (error) {
        error->status = status;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}

static me_status kernel_source_error(me_error *error, const char *source, int position,
                                    const char *message) {
    if (error && position >= 0) {
        error->line = error->column = 1;
        for (int i = 0; i < position && source[i]; i++) {
            if (source[i] == '\n') {
                error->line++;
                error->column = 1;
            }
            else error->column++;
        }
    }
    return kernel_error(error, strstr(message, "out of memory") ? ME_ERR_OOM : ME_ERR_SOURCE,
                        message);
}

static size_t kernel_width(me_dtype dtype, size_t itemsize) {
    if (dtype == ME_BYTES || dtype == ME_STRING) {
        return itemsize && itemsize <= 1024 * 1024 &&
            (dtype != ME_STRING || itemsize % 4 == 0) ? itemsize : 0;
    }
    if (dtype < ME_BOOL || dtype > ME_COMPLEX128) return 0;
    size_t width = dtype_size(dtype);
    return !itemsize || itemsize == width ? width : 0;
}

static bool kernel_output_type(me_dtype dtype, size_t itemsize) {
    if (dtype == ME_AUTO) return itemsize == 0;
    if ((dtype == ME_STRING || dtype == ME_BYTES) && !itemsize) return true;
    return kernel_width(dtype, itemsize) != 0;
}

static bool kernel_identifier(const char *name) {
    if (!name || !*name) return false;
    for (size_t i = 0; name[i]; i++) {
        char c = name[i];
        if (c != '_' && !(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') &&
            !(i && c >= '0' && c <= '9')) return false;
    }
    if (!strcmp(name, "_ndim") || !strcmp(name, "_flat_idx")) return false;
    if (name[0] == '_' && (name[1] == 'i' || name[1] == 'n') && name[2]) {
        bool reserved = true;
        for (size_t i = 2; name[i]; i++) reserved &= name[i] >= '0' && name[i] <= '9';
        if (reserved) return false;
    }
    return true;
}

static bool kernel_unicode(const void *data, size_t nitems, size_t width,
                           const uint8_t *mask) {
    const uint32_t *values = data;
    for (size_t lane = 0; lane < nitems; lane++) {
        if (mask && !mask[lane]) continue;
        const uint32_t *slot = values + lane * (width / 4);
        for (size_t i = 0; i < width / 4 && slot[i]; i++) {
            if (slot[i] > 0x10ffff || (slot[i] >= 0xd800 && slot[i] <= 0xdfff)) return false;
        }
    }
    return true;
}

void me_kernel_free(me_kernel *kernel) {
    if (!kernel) return;
    me_free(kernel->formula);
    dsl_compiled_program_free(kernel->program);
    for (size_t i = 0; i < kernel->nbindings; i++) free(kernel->captures[i]);
    free(kernel->entry_point);
    free(kernel);
}

static me_status kernel_signature(const me_signature *sig, const me_prepare_options *options,
                                  me_kernel **out, me_variable *variables, me_error *error) {
    if (!sig || sig->struct_size < ME_SIGNATURE_V1_SIZE ||
        sig->ninputs > ME_MAX_VARS || sig->ncaptures > ME_MAX_VARS - sig->ninputs ||
        (sig->ninputs && !sig->inputs) || (sig->ncaptures && !sig->captures) ||
        sig->context_ndim < 0 || sig->context_ndim > ME_DSL_MAX_NDIM ||
        sig->cardinality < ME_CARDINALITY_INFER || sig->cardinality > ME_BLOCK_SCALAR ||
        sig->casting < ME_CAST_UNSAFE || sig->casting > ME_CAST_SAFE ||
        !kernel_output_type(sig->output_dtype, sig->output_itemsize)) {
        return kernel_error(error, ME_ERR_SIGNATURE, "invalid signature");
    }
    if (options && (options->struct_size < ME_PREPARE_OPTIONS_V1_SIZE ||
        options->execution < ME_EXEC_INTERPRET || options->execution > ME_EXEC_REQUIRE_JIT)) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid preparation options");
    }
    me_kernel *kernel = calloc(1, sizeof(*kernel));
    if (!kernel) return kernel_error(error, ME_ERR_OOM, "out of memory preparing kernel");
    *out = kernel;
    kernel->ninputs = sig->ninputs;
    kernel->nbindings = sig->ninputs + sig->ncaptures;
    kernel->info = (me_kernel_info)ME_KERNEL_INFO_INITIALIZER;
    for (size_t i = 0; i < kernel->nbindings; i++) {
        const me_capture *capture = i < sig->ninputs ? NULL : &sig->captures[i - sig->ninputs];
        const me_input_spec *input = capture ? &capture->spec : &sig->inputs[i];
        size_t width = kernel_width(input->dtype, input->itemsize);
        if (!kernel_identifier(input->name) || !width) {
            return kernel_error(error, ME_ERR_SIGNATURE, "invalid binding name, dtype or itemsize");
        }
        for (size_t j = 0; j < i; j++) {
            if (!strcmp(variables[j].name, input->name)) {
                return kernel_error(error, ME_ERR_SIGNATURE, "duplicate binding name");
            }
        }
        variables[i] = (me_variable){.name = input->name, .dtype = input->dtype,
            .address = &synthetic_var_addresses[i], .itemsize = width};
        kernel->dtypes[i] = input->dtype;
        kernel->widths[i] = width;
        if (capture) {
            if (!capture->value || (capture->category != ME_CAPTURE_TYPED &&
                capture->category != ME_CAPTURE_WEAK) ||
                (capture->category == ME_CAPTURE_WEAK && input->dtype != ME_BOOL &&
                 input->dtype != ME_INT64 && input->dtype != ME_FLOAT64 &&
                 input->dtype != ME_COMPLEX128)) {
                return kernel_error(error, ME_ERR_SIGNATURE, "invalid scalar capture or weak scalar dtype");
            }
            kernel->captures[i] = malloc(width);
            if (!kernel->captures[i]) return kernel_error(error, ME_ERR_OOM, "out of memory copying capture");
            memcpy(kernel->captures[i], capture->value, width);
            if (input->dtype == ME_STRING && !kernel_unicode(kernel->captures[i], 1, width, NULL)) {
                return kernel_error(error, ME_ERR_SIGNATURE, "invalid Unicode capture");
            }
            variables[i].type = ME_DSL_UNIFORM_INPUT |
                (capture->category == ME_CAPTURE_WEAK ? ME_DSL_WEAK_INPUT : 0);
        }
    }
    return ME_SUCCESS;
}

static void kernel_mark_captures(me_expr *expr, const me_signature *sig) {
    if (!expr) return;
    if (TYPE_MASK(expr->type) == ME_VARIABLE) {
        int index = dsl_synthetic_var_index(expr->bound);
        if (index >= (int)sig->ninputs && index < (int)(sig->ninputs + sig->ncaptures)) {
            expr->flags |= ME_EXPR_FLAG_UNIFORM_INPUT;
            if (sig->captures[index - sig->ninputs].category == ME_CAPTURE_WEAK) {
                expr->flags |= ME_EXPR_FLAG_WEAK_SCALAR;
            }
        }
    }
    for (int i = 0; i < ARITY(expr->type); i++) kernel_mark_captures(expr->parameters[i], sig);
}

static me_status kernel_result(me_kernel *kernel, const me_signature *sig,
                               me_dtype inferred, me_error *error) {
    static const char *casting[] = {"unsafe", "same_kind", "safe"};
    me_dtype output = kernel->info.output_dtype;
    if (inferred != output && !dsl_numpy_can_cast(inferred, output, casting[sig->casting])) {
        return kernel_error(error, ME_ERR_SIGNATURE, "result conversion violates declared cast policy");
    }
    if (sig->output_itemsize && sig->output_itemsize != kernel->info.output_itemsize) {
        return kernel_error(error, ME_ERR_SIGNATURE, "output width disagrees with source");
    }
    if (sig->cardinality != ME_CARDINALITY_INFER && sig->cardinality != kernel->info.cardinality) {
        return kernel_error(error, ME_ERR_SIGNATURE, kernel->info.cardinality == ME_BLOCK_SCALAR ?
            "declared elementwise result disagrees with block scalar source" :
            "declared block scalar result disagrees with elementwise source");
    }
    return ME_SUCCESS;
}

static me_status kernel_prepare(const char *version, const char *source,
    const me_signature *sig, const me_prepare_options *options,
    int jit_mode, me_kernel **out, me_error *error) {
    if (error) memset(error, 0, sizeof(*error));
    if (out) *out = NULL;
    if (!out || !source) return kernel_error(error, ME_ERR_INVALID_ARG, "source and output handle are required");
    bool formula = version == NULL;
    if (!formula && strcmp(version, "0.1") && strcmp(version, "0.2")) {
        return kernel_error(error, ME_ERR_UNSUPPORTED, "unsupported Menudet version");
    }
    me_kernel *kernel = NULL;
    me_variable variables[ME_MAX_VARS];
    me_status status = kernel_signature(sig, options, &kernel, variables, error);
    if (status != ME_SUCCESS) {
        me_kernel_free(kernel);
        return status;
    }
    me_dtype inferred = ME_AUTO;
    me_execution_policy execution = options ? options->execution : ME_EXEC_INTERPRET;
    char reason[256] = {0};
    int position = -1;
    if (formula) {
        if (sig->context_ndim || sig->cardinality == ME_BLOCK_SCALAR || sig->integer_powers) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "formula context and reductions are not yet supported");
            goto cleanup;
        }
        me_dsl_program parsed = {0};
        me_dsl_declared_policy policy = {
            .has_fp = !options || !options->inherit_environment, .fp = ME_DSL_FP_STRICT,
            .has_math = !options || !options->inherit_environment, .math = ME_DSL_MATH_SYSTEM};
        me_dsl_fp_mode fp;
        me_dsl_math_backend math;
        me_dsl_sleef_ulp ulp;
        if (!dsl_math_policy_resolve(&parsed, &policy, ME_DSL_COMPILER_CC,
            &fp, &math, &ulp, &kernel->formula_math, reason, sizeof(reason))) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, reason);
            goto cleanup;
        }
        dsl_menudet_fp_state saved;
        if (!dsl_menudet_fp_begin(&saved)) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "cannot establish nearest rounding");
            goto cleanup;
        }
        int rc = private_compile_profile_ex(source, variables, (int)kernel->nbindings,
            NULL, 0, ME_AUTO, ME_DSL_PROFILE_MENUDET_0_1, &position, &kernel->formula);
        if (rc != ME_COMPILE_SUCCESS) {
            status = kernel_source_error(error, source, position,
                rc == ME_COMPILE_ERR_OOM ? "out of memory parsing formula" : "invalid formula");
        }
        else if (contains_reduction(kernel->formula)) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "formula reductions are deferred in the provisional API");
        }
        else {
            kernel_mark_captures(kernel->formula, sig);
            if (!dsl_menudet_type_expr_profile(&kernel->formula, ME_AUTO,
                ME_DSL_PROFILE_MENUDET_0_1, reason, sizeof(reason))) {
                status = kernel_source_error(error, source, -1, reason);
            }
            else {
                inferred = me_get_dtype(kernel->formula);
                if (sig->output_dtype != ME_AUTO && sig->output_dtype != inferred &&
                    !dsl_menudet_convert_expr(&kernel->formula, sig->output_dtype)) {
                    status = kernel_error(error, ME_ERR_SIGNATURE, "unsupported formula output conversion");
                }
            }
        }
        if (!dsl_menudet_fp_end(&saved)) status = kernel_error(error, ME_ERR_UNSUPPORTED, "cannot restore floating environment");
        if (status != ME_SUCCESS) goto cleanup;
        if (execution == ME_EXEC_REQUIRE_JIT) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "formula JIT is not yet supported");
            goto cleanup;
        }
        kernel->info.output_dtype = me_get_dtype(kernel->formula);
        kernel->info.output_itemsize = me_get_itemsize(kernel->formula);
        kernel->info.cardinality = ME_ELEMENTWISE;
    }
    else {
        /* Direct preparation defaults are deterministic; source pragmas remain
         * explicit authored policy. Legacy import retains its own defaults. */
        me_dsl_declared_policy policy = {
            .has_fp = !options || !options->inherit_environment, .fp = ME_DSL_FP_STRICT,
            .has_math = !options || !options->inherit_environment, .math = ME_DSL_MATH_SYSTEM,
            .integer_powers = sig->integer_powers};
        bool binding_error = false;
        int jit = jit_mode == ME_JIT_DEFAULT ?
            (execution == ME_EXEC_INTERPRET ? ME_JIT_OFF : ME_JIT_ON) : jit_mode;
        kernel->program = dsl_compile_menudet_deferred(source, !strcmp(version, "0.2"),
            variables, (int)kernel->nbindings, sig->output_dtype, sig->context_ndim, jit,
            &policy, &position, &binding_error, &kernel->entry_point, reason, sizeof(reason));
        if (!kernel->program) {
            status = binding_error ? kernel_error(error, ME_ERR_SIGNATURE, reason) :
                kernel_source_error(error, source, position, reason[0] ? reason : "invalid Menudet kernel");
            goto cleanup;
        }
        inferred = kernel->program->inferred_dtype;
        kernel->info.output_dtype = kernel->program->output_dtype;
        kernel->info.output_itemsize = kernel->program->output_itemsize;
        kernel->info.cardinality = kernel->program->output_is_scalar ? ME_BLOCK_SCALAR : ME_ELEMENTWISE;
        bool nd = kernel->program->uses_i_mask || kernel->program->uses_n_mask ||
            kernel->program->uses_ndim || kernel->program->uses_flat_idx;
        kernel->info.context_ndim = nd ? sig->context_ndim : 0;
    }
    status = kernel_result(kernel, sig, inferred, error);
    if (status == ME_SUCCESS && kernel->program && kernel->program->jit_request_mode == ME_JIT_ON) {
        dsl_menudet_fp_state saved;
        if (!dsl_menudet_fp_begin(&saved)) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "cannot establish nearest rounding preparing JIT");
            goto cleanup;
        }
        /* Capture specialization follows typing: an annotated formerly weak
         * capture may now be strong. Never invoke a compiler for a rejected
         * result contract. */
        if (!kernel->program->jit_c_source) {
            int64_t values[ME_MAX_VARS] = {0};
            bool known[ME_MAX_VARS] = {false};
            bool any = false;
            for (size_t i = kernel->ninputs; i < kernel->nbindings; i++) {
                if (!kernel->program->numpy_weak_vars[i]) continue;
                if (kernel->dtypes[i] == ME_INT64) memcpy(&values[i], kernel->captures[i], sizeof(int64_t));
                else if (kernel->dtypes[i] == ME_BOOL) values[i] = *(const bool *)kernel->captures[i];
                else continue;
                known[i] = true;
                any = true;
            }
            if (any) dsl_menudet_prepare_jit_constants_source(kernel->program, values, known);
        }
        if (!kernel->program->jit_kernel_fn) dsl_try_prepare_jit_runtime(kernel->program);
        if (!dsl_menudet_fp_end(&saved)) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "cannot restore floating environment after JIT preparation");
        }
    }
    if (status == ME_SUCCESS && !formula && execution == ME_EXEC_REQUIRE_JIT &&
        !kernel->program->jit_kernel_fn) {
        status = kernel_error(error, ME_ERR_UNSUPPORTED, "JIT is required but unavailable for this kernel");
    }
cleanup:
    if (status == ME_SUCCESS) *out = kernel;
    else me_kernel_free(kernel);
    return status;
}

me_status me_prepare_formula(const char *source, const me_signature *signature,
    const me_prepare_options *options, me_kernel **out, me_error *error) {
    return kernel_prepare(NULL, source, signature, options, ME_JIT_DEFAULT, out, error);
}

me_status me_prepare_menudet(const char *version, const char *source,
    const me_signature *signature, const me_prepare_options *options,
    me_kernel **out, me_error *error) {
    if (!version) {
        if (error) memset(error, 0, sizeof(*error));
        if (out) *out = NULL;
        return kernel_error(error, ME_ERR_INVALID_ARG, "Menudet version is required");
    }
    return kernel_prepare(version, source, signature, options, ME_JIT_DEFAULT, out, error);
}

me_status me_kernel_prepare_menudet(const char *version, const char *source,
    const me_signature *signature, const me_prepare_options *options, int jit_mode,
    me_kernel **out, me_error *error) {
    return kernel_prepare(version, source, signature, options, jit_mode, out, error);
}

me_status me_describe(const me_kernel *kernel, me_kernel_info *out) {
    if (!kernel || !out || out->struct_size < ME_KERNEL_INFO_V1_SIZE) return ME_ERR_INVALID_ARG;
    size_t size = out->struct_size;
    memcpy(out, &kernel->info, size < sizeof(*out) ? size : sizeof(*out));
    out->struct_size = size;
    return ME_SUCCESS;
}

static me_status kernel_preflight(const me_kernel *kernel, const me_buffer *inputs,
    size_t ninputs, me_mut_buffer output, const me_block *block, bool *all_active, me_error *error) {
    if (!kernel || !block || block->struct_size < ME_BLOCK_V1_SIZE ||
        ninputs != kernel->ninputs || (ninputs && !inputs) || block->nitems > INT_MAX ||
        block->ndim != kernel->info.context_ndim) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid block, context rank or input count");
    }
    size_t n = block->nitems;
    size_t count = kernel->info.cardinality == ME_BLOCK_SCALAR ? 1 : n;
    size_t width = kernel->info.output_itemsize;
    if (!width || count > SIZE_MAX / width || output.capacity < count * width ||
        !me_kernel_range(output.data, count * width, me_kernel_alignment(kernel->info.output_dtype))) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid output capacity, alignment or byte extent");
    }
    size_t bytes = count * width;
    if ((!block->valid_mask && block->valid_mask_capacity) ||
        (block->valid_mask && (block->valid_mask_capacity < n ||
          !me_kernel_range(block->valid_mask, n, 1) ||
          me_kernel_overlap(block->valid_mask, n, output.data, bytes)))) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid or overlapping mask");
    }
    for (size_t i = 0; block->valid_mask && i < n; i++) {
        if (block->valid_mask[i] > 1) return kernel_error(error, ME_ERR_INVALID_ARG, "mask values must be zero or one");
        if (!block->valid_mask[i]) *all_active = false;
    }
    if (!block->ndim && (block->logical_shape || block->origin || block->extent)) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "unexpected coordinate pointers");
    }
    if (block->ndim) {
        const int64_t *context[] = {block->logical_shape, block->origin, block->extent};
        for (size_t i = 0; i < 3; i++) {
            size_t length = (size_t)block->ndim * sizeof(int64_t);
            if (!me_kernel_range(context[i], length, sizeof(int64_t)) ||
                me_kernel_overlap(context[i], length, output.data, bytes)) {
                return kernel_error(error, ME_ERR_INVALID_ARG, "invalid or overlapping coordinates");
            }
        }
    }
    for (size_t i = 0; i < ninputs; i++) {
        width = kernel->widths[i];
        if (n > SIZE_MAX / width || inputs[i].capacity < n * width ||
            !me_kernel_range(inputs[i].data, n * width, me_kernel_alignment(kernel->dtypes[i])) ||
            me_kernel_overlap(inputs[i].data, n * width, output.data, bytes)) {
            return kernel_error(error, ME_ERR_INVALID_ARG, "invalid or overlapping input buffer");
        }
        if (kernel->dtypes[i] == ME_STRING &&
            !kernel_unicode(inputs[i].data, n, width, block->valid_mask)) {
            return kernel_error(error, ME_ERR_INVALID_ARG, "invalid Unicode input");
        }
    }
    return ME_SUCCESS;
}

me_status me_kernel_run_begin(const me_run_options *options, me_fp_status *status,
    me_error *error) {
    if (error) memset(error, 0, sizeof(*error));
    if (options && (options->struct_size < ME_RUN_OPTIONS_V1_SIZE || (options->raise_mask & ~15u))) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid run options");
    }
    *status = (me_fp_status){0};
#ifndef __EMSCRIPTEN__
    status->supported = true;
#endif
    if (options && options->fp_status) *options->fp_status = *status;
    if (options && options->raise_mask && !status->supported) {
        return kernel_error(error, ME_ERR_UNSUPPORTED, "floating exception reporting is unavailable");
    }
    return ME_SUCCESS;
}

me_status me_eval_block(const me_kernel *kernel, const me_buffer *inputs,
    size_t ninputs, me_mut_buffer output, const me_block *block,
    const me_run_options *options, me_error *error) {
    me_fp_status status;
    me_status result = me_kernel_run_begin(options, &status, error);
    if (result != ME_SUCCESS) return result;
    bool all_active = true;
    result = kernel_preflight(kernel, inputs, ninputs, output, block, &all_active, error);
    if (result != ME_SUCCESS) return result;
    if (kernel->formula && block->nitems == 0) return ME_SUCCESS;
    const void *data[ME_MAX_VARS];
    for (size_t i = 0; i < kernel->nbindings; i++) {
        data[i] = i < ninputs ? inputs[i].data : kernel->captures[i];
    }
    unsigned *previous = dsl_menudet_status_begin(&status.flags);
    int rc;
    if (kernel->formula) {
        const me_math_dispatch *math = me_math_dispatch_set(&kernel->formula_math);
        me_simd_accuracy_state accuracy;
        me_simd_accuracy_push(kernel->formula_math.ulp, &accuracy);
        rc = dsl_menudet_eval_expr_masked(kernel->formula, data, (int)kernel->nbindings,
            NULL, (int)block->nitems, all_active ? NULL : block->valid_mask,
            kernel->info.output_itemsize, output.data);
        me_simd_accuracy_pop(&accuracy);
        me_math_dispatch_set(math);
    }
    else {
        me_dsl_menudet_eval_descriptor native = {
            .nitems = (int)block->nitems, .valid_mask = all_active ? NULL : block->valid_mask,
            .output_capacity = output.capacity, .ndim = block->ndim,
            .logical_shape = block->logical_shape, .block_origin = block->origin,
            .block_extent = block->extent};
        rc = dsl_eval_program_menudet(kernel->program, data, (int)kernel->nbindings, output.data, &native);
    }
    dsl_menudet_status_end(previous);
    if (options && options->fp_status) *options->fp_status = status;
    if (rc) return kernel_error(error, rc == ME_EVAL_ERR_OOM ? ME_ERR_OOM : ME_ERR_EVAL,
        rc == ME_DSL_MENUDET_ERR_NEG_INT_POW ?
        "integers to negative integer powers are not allowed" : "kernel evaluation failed");
    if (options && (status.flags & options->raise_mask)) {
        return kernel_error(error, ME_ERR_EVAL, "floating exception selected by raise policy");
    }
    return ME_SUCCESS;
}
