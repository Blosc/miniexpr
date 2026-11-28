/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "miniexpr_kernel.h"
#include "miniexpr_kernel_internal.h"
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include "dsl_jit_runtime_internal.h"
#include "dsl_menudet_expr.h"
#include "dsl_menudet_fp.h"
#include "dsl_menudet_types.h"
#include "dsl_math_policy.h"
#include "functions-simd.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dsl_partition_internal.h"

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

static unsigned kernel_expr_features(const me_expr *expr) {
    if (!expr) return 0;
    unsigned features = is_reduction_node(expr) ? ME_MENUDET_FEATURE_BLOCK_REDUCTIONS : 0;
    if (is_string_dtype(expr->dtype) || TYPE_MASK(expr->type) == ME_STRING_CONSTANT) {
        features |= ME_MENUDET_FEATURE_FIXED_STRINGS;
    }
    for (int i = 0; i < ARITY(expr->type); i++) features |= kernel_expr_features(expr->parameters[i]);
    return features;
}

unsigned dsl_menudet_block_features(const me_dsl_compiled_block *block) {
    unsigned features = 0;
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_compiled_stmt *stmt = block->stmts[i];
        const me_expr *expr = NULL;
        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN: expr = stmt->as.assign.value.expr; break;
        case ME_DSL_STMT_RETURN: expr = stmt->as.return_stmt.expr.expr; break;
        case ME_DSL_STMT_EXPR: expr = stmt->as.expr_stmt.expr.expr; break;
        case ME_DSL_STMT_BREAK: case ME_DSL_STMT_CONTINUE:
            expr = stmt->as.flow.cond.expr;
            features |= ME_MENUDET_FEATURE_CONTROL_FLOW;
            break;
        case ME_DSL_STMT_IF:
            expr = stmt->as.if_stmt.cond.expr;
            features |= ME_MENUDET_FEATURE_CONTROL_FLOW |
                dsl_menudet_block_features(&stmt->as.if_stmt.then_block);
            if (stmt->as.if_stmt.has_else) {
                features |= dsl_menudet_block_features(&stmt->as.if_stmt.else_block);
            }
            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                features |= kernel_expr_features(stmt->as.if_stmt.elif_branches[j].cond.expr);
                features |= dsl_menudet_block_features(&stmt->as.if_stmt.elif_branches[j].block);
            }
            break;
        case ME_DSL_STMT_WHILE:
            expr = stmt->as.while_loop.cond.expr;
            features |= ME_MENUDET_FEATURE_CONTROL_FLOW |
                dsl_menudet_block_features(&stmt->as.while_loop.body);
            break;
        case ME_DSL_STMT_FOR:
            expr = stmt->as.for_loop.start.expr;
            features |= kernel_expr_features(stmt->as.for_loop.stop.expr);
            features |= kernel_expr_features(stmt->as.for_loop.step.expr);
            features |= ME_MENUDET_FEATURE_CONTROL_FLOW |
                dsl_menudet_block_features(&stmt->as.for_loop.body);
            break;
        default: break;
        }
        features |= kernel_expr_features(expr);
    }
    return features;
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
        options->execution < ME_EXEC_INTERPRET || options->execution > ME_EXEC_AUTO)) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid preparation options");
    }
    if (options && options->struct_size >= ME_PREPARE_OPTIONS_V3_SIZE &&
        (options->math_backend < ME_MATH_DEFAULT || options->math_backend > ME_MATH_SCALAR ||
         options->jit_compiler < ME_COMPILER_DEFAULT || options->jit_compiler > ME_COMPILER_CC)) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid compiler or math policy");
    }
    if (options && options->struct_size >= ME_PREPARE_OPTIONS_V5_SIZE &&
        (options->fp_mode < ME_FP_DEFAULT || options->fp_mode > ME_FP_FAST ||
         options->math_accuracy < ME_ACCURACY_DEFAULT || options->math_accuracy > ME_ACCURACY_U35)) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid floating-point preparation policy");
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
            "declared elementwise cardinality disagrees with block scalar source" :
            "declared block scalar cardinality disagrees with elementwise source");
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
    unsigned forbidden = 0;
    if (options && options->struct_size >= ME_PREPARE_OPTIONS_V2_SIZE) {
        forbidden = options->menudet_forbidden_features;
    }
    const unsigned known_features = ME_MENUDET_FEATURE_CONTROL_FLOW |
        ME_MENUDET_FEATURE_BLOCK_REDUCTIONS | ME_MENUDET_FEATURE_FIXED_STRINGS |
        ME_MENUDET_FEATURE_ND_CONTEXT | ME_MENUDET_FEATURE_FLEXIBLE_MATH;
    if ((forbidden & ~known_features) || (formula && forbidden)) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid Menudet feature restrictions");
    }
    me_kernel *kernel = NULL;
    me_variable variables[ME_MAX_VARS];
    me_status status = kernel_signature(sig, options, &kernel, variables, error);
    if (status != ME_SUCCESS) {
        me_kernel_free(kernel);
        return status;
    }
    const me_jit_options *prev_jit_opts = NULL;
    bool exchanged_jit_opts = false;
    if (options && options->struct_size >= ME_PREPARE_OPTIONS_V4_SIZE && options->jit_options) {
        prev_jit_opts = dsl_jit_exchange_options(options->jit_options);
        exchanged_jit_opts = true;
    }
    me_dtype inferred = ME_AUTO;
    me_execution_policy execution = options ? options->execution : ME_EXEC_INTERPRET;
    me_math_backend math_policy = ME_MATH_DEFAULT;
    me_jit_compiler compiler_policy = ME_COMPILER_DEFAULT;
    me_fp_mode fp_policy = ME_FP_DEFAULT;
    me_math_accuracy accuracy = ME_ACCURACY_DEFAULT;
    if (options && options->struct_size >= ME_PREPARE_OPTIONS_V3_SIZE) {
        math_policy = options->math_backend;
        compiler_policy = options->jit_compiler;
    }
    if (options && options->struct_size >= ME_PREPARE_OPTIONS_V5_SIZE) {
        fp_policy = options->fp_mode;
        accuracy = options->math_accuracy;
    }
    char reason[256] = {0};
    int position = -1;
    if (formula) {
        if (fp_policy != ME_FP_DEFAULT && fp_policy != ME_FP_STRICT) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "formula preparation requires strict FP");
            goto cleanup;
        }
        if (sig->context_ndim || sig->cardinality == ME_BLOCK_SCALAR || sig->integer_powers) {
            status = kernel_error(error, ME_ERR_UNSUPPORTED, "formula context and reductions are not yet supported");
            goto cleanup;
        }
        me_dsl_program parsed = {0};
        me_dsl_declared_policy policy = {
            .has_fp = !options || !options->inherit_environment, .fp = ME_DSL_FP_STRICT,
            .has_math = !options || !options->inherit_environment, .math = ME_DSL_MATH_SYSTEM};
        if (math_policy != ME_MATH_DEFAULT) {
            policy.has_math = true;
            policy.math = (me_dsl_math_backend)(math_policy - ME_MATH_AUTO);
        }
        policy.has_ulp = accuracy != ME_ACCURACY_DEFAULT;
        policy.ulp = (me_dsl_sleef_ulp)accuracy;
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
                kernel->inferred_dtype = inferred;
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
        if (fp_policy != ME_FP_DEFAULT) {
            policy.has_fp = true;
            policy.fp = (me_dsl_fp_mode)(fp_policy - ME_FP_STRICT);
        }
        policy.has_ulp = accuracy != ME_ACCURACY_DEFAULT;
        policy.ulp = (me_dsl_sleef_ulp)accuracy;
        if (math_policy != ME_MATH_DEFAULT) {
            policy.has_math = true;
            policy.math = (me_dsl_math_backend)(math_policy - ME_MATH_AUTO);
        }
        if (compiler_policy != ME_COMPILER_DEFAULT) {
            policy.has_compiler = true;
            policy.compiler = compiler_policy == ME_COMPILER_CC ?
                ME_DSL_COMPILER_CC : ME_DSL_COMPILER_LIBTCC;
        }
        bool binding_error = false;
        int jit = jit_mode == ME_JIT_DEFAULT ?
            (execution == ME_EXEC_AUTO ? ME_JIT_AUTO :
             execution == ME_EXEC_INTERPRET ? ME_JIT_OFF : ME_JIT_ON) : jit_mode;
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
        if (nd) {
            unsigned symbols = (unsigned)(kernel->program->uses_i_mask | kernel->program->uses_n_mask);
            if (!sig->context_ndim || (symbols >> sig->context_ndim)) {
                status = kernel_error(error, ME_ERR_SIGNATURE, "reserved symbols exceed declared ND context");
                goto cleanup;
            }
        }
        kernel->info.context_ndim = nd ? sig->context_ndim : 0;
    }
    status = kernel_result(kernel, sig, inferred, error);
    if (status == ME_SUCCESS && forbidden) {
        unsigned features = dsl_menudet_block_features(&kernel->program->block);
        if (is_string_dtype(kernel->info.output_dtype)) features |= ME_MENUDET_FEATURE_FIXED_STRINGS;
        for (size_t i = 0; i < kernel->nbindings; i++) {
            if (is_string_dtype(kernel->dtypes[i])) features |= ME_MENUDET_FEATURE_FIXED_STRINGS;
        }
        if (sig->context_ndim > 0 || kernel->info.context_ndim > 0 ||
            kernel->program->uses_i_mask || kernel->program->uses_n_mask ||
            kernel->program->uses_ndim || kernel->program->uses_flat_idx) {
            features |= ME_MENUDET_FEATURE_ND_CONTEXT;
        }
        if (kernel->program->fp_mode != ME_DSL_FP_STRICT ||
            (kernel->program->math_backend != ME_DSL_MATH_SYSTEM &&
             kernel->program->math_backend != ME_DSL_MATH_AUTO) ||
            kernel->program->math_explicit || kernel->program->ulp_explicit ||
            kernel->program->sleef_ulp != ME_DSL_ULP_OMIT) {
            features |= ME_MENUDET_FEATURE_FLEXIBLE_MATH;
        }
        unsigned blocked = features & forbidden;
        if (blocked) {
            if (blocked == ME_MENUDET_FEATURE_FLEXIBLE_MATH) {
                status = kernel_error(error, ME_ERR_UNSUPPORTED,
                    "flexible math policy requires the 'flexible-math' capability");
            } else {
                status = kernel_error(error, ME_ERR_SIGNATURE,
                    "source uses a forbidden Menudet feature; missing source capability declaration");
            }
        }
    }
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
    if (exchanged_jit_opts) {
        dsl_jit_exchange_options(prev_jit_opts);
    }
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

me_status me_describe_input_count(const me_kernel *kernel, size_t *count) {
    if (!kernel || !count) {
        return ME_ERR_INVALID_ARG;
    }
    *count = kernel->ninputs;
    return ME_SUCCESS;
}

bool me_kernel_has_jit(const me_kernel *kernel) {
    return kernel && kernel->program && kernel->program->jit_kernel_fn != NULL;
}

static bool kernel_real_dtype(me_dtype dtype) {
    return dtype >= ME_BOOL && dtype <= ME_FLOAT64;
}

static bool kernel_int_fits(int64_t value, me_dtype output) {
    switch (output) {
        case ME_INT8: return value >= INT8_MIN && value <= INT8_MAX;
        case ME_INT16: return value >= INT16_MIN && value <= INT16_MAX;
        case ME_INT32: return value >= INT32_MIN && value <= INT32_MAX;
        case ME_INT64: return true;
        case ME_UINT8: return value >= 0 && value <= UINT8_MAX;
        case ME_UINT16: return value >= 0 && value <= UINT16_MAX;
        case ME_UINT32: return value >= 0 && (uint64_t)value <= UINT32_MAX;
        case ME_UINT64: return value >= 0;
        default: return false;
    }
}

static const void *kernel_weak_capture_data(const me_kernel *kernel, const me_expr *arg) {
    if (!kernel || !arg || !(arg->flags & (ME_EXPR_FLAG_WEAK_SCALAR | ME_EXPR_FLAG_WEAK_LITERAL)) ||
        TYPE_MASK(arg->type) != ME_VARIABLE) {
        return NULL;
    }
    int slot = dsl_synthetic_var_index(arg->bound);
    if (slot < (int)kernel->ninputs || (size_t)slot >= kernel->nbindings) {
        return NULL;
    }
    return kernel->captures[slot];
}

static bool kernel_partition_cast(const me_kernel *kernel, const me_expr *arg, me_dtype output) {
    if (!kernel_real_dtype(arg->dtype) || !kernel_real_dtype(output)) return false;
    if (output == arg->dtype) return true;
    bool weak = (arg->flags & (ME_EXPR_FLAG_WEAK_SCALAR | ME_EXPR_FLAG_WEAK_LITERAL)) != 0;
    const void *capture = kernel_weak_capture_data(kernel, arg);
    if (output == ME_FLOAT32 || output == ME_FLOAT64) {
        /* Strong real conversions have total IEEE results. Weak floating
         * narrowing is proven safe when the constant value is representable. */
        if (output == ME_FLOAT32 && arg->dtype == ME_FLOAT64 && weak) {
            if (TYPE_MASK(arg->type) == ME_CONSTANT) {
                return isfinite(arg->value) && isfinite((float)arg->value);
            }
            if (capture) {
                double value = *(const double *)capture;
                return isfinite(value) && isfinite((float)value);
            }
            return false;
        }
        return true;
    }
    if (weak && arg->dtype == ME_INT64) {
        if (capture) {
            return kernel_int_fits(*(const int64_t *)capture, output);
        }
        if (TYPE_MASK(arg->type) == ME_CONSTANT && isfinite(arg->value) &&
            trunc(arg->value) == arg->value && arg->value >= (double)INT64_MIN &&
            arg->value < 9223372036854775808.0) {
            return kernel_int_fits((int64_t)arg->value, output);
        }
        return false;
    }
    return dsl_numpy_can_cast(arg->dtype, output, "safe");
}

static bool kernel_in_list(const char *value, const char *const *items, size_t count) {
    if (!value) {
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (!strcmp(value, items[i])) {
            return true;
        }
    }
    return false;
}

static bool kernel_partition_operator(const char *op, int arity, bool *is_total_bool) {
    static const char *const unary[] = {"+", "-", "~", "not"};
    static const char *const binary[] = {"+", "-", "*", "/", "&", "|", "^"};
    static const char *const cmp_logic[] = {"==", "!=", "<", "<=", ">", ">=", "and", "or"};
    *is_total_bool = false;
    if (!op) {
        return false;
    }
    if (arity == 1) {
        *is_total_bool = !strcmp(op, "not");
        return kernel_in_list(op, unary, sizeof(unary) / sizeof(*unary));
    }
    if (arity == 2) {
        if (kernel_in_list(op, cmp_logic, sizeof(cmp_logic) / sizeof(*cmp_logic))) {
            *is_total_bool = true;
            return true;
        }
        return kernel_in_list(op, binary, sizeof(binary) / sizeof(*binary));
    }
    return arity == 3 && !strcmp(op, "where");
}

static bool kernel_partition_math(const char *name, int arity, me_dtype dtype, bool *is_total_bool) {
    static const char *const predicates[] = {"isfinite", "isnan", "isinf", "signbit"};
    static const char *const unary_float[] = {
        "abs", "fabs", "floating_abs", "sqrt", "cbrt", "exp", "exp2", "expm1",
        "log", "log1p", "log2", "log10", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "arcsin", "arccos", "arctan",
        "arcsinh", "arccosh", "arctanh", "floor", "ceil", "trunc", "round", "rint", "sign"
    };
    static const char *const binary_float[] = {
        "atan2", "arctan2", "hypot", "fmod", "copysign", "nextafter", "fmin", "fmax"
    };
    static const char *const binary_ordered[] = {"minimum", "maximum"};
    *is_total_bool = false;
    if (!name) {
        return false;
    }
    if (arity == 1) {
        if (dtype == ME_BOOL && kernel_in_list(name, predicates, sizeof(predicates) / sizeof(*predicates))) {
            *is_total_bool = true;
            return true;
        }
        return (dtype == ME_FLOAT32 || dtype == ME_FLOAT64) &&
               kernel_in_list(name, unary_float, sizeof(unary_float) / sizeof(*unary_float));
    }
    if (arity == 2) {
        if (kernel_in_list(name, binary_ordered, sizeof(binary_ordered) / sizeof(*binary_ordered))) {
            return true;
        }
        return (dtype == ME_FLOAT32 || dtype == ME_FLOAT64) &&
               kernel_in_list(name, binary_float, sizeof(binary_float) / sizeof(*binary_float));
    }
    return arity == 3 && (dtype == ME_FLOAT32 || dtype == ME_FLOAT64) && !strcmp(name, "fma");
}

static bool kernel_partition_expr(const me_kernel *kernel, const me_expr *expr,
                                  const bool *defined, size_t count, int depth, int *budget) {
    if (!expr || depth > 128 || --*budget < 0 || !kernel_real_dtype(expr->dtype) ||
        is_reduction_node(expr)) return false;
    int kind = TYPE_MASK(expr->type);
    if (kind == ME_VARIABLE) {
        int slot = dsl_synthetic_var_index(expr->bound);
        return slot >= 0 && (size_t)slot < count && defined[slot];
    }
    if (kind == ME_CONSTANT) return true;
    if (!IS_FUNCTION(expr->type)) return false;
    int arity = ARITY(expr->type);
    if (arity < 1 || arity > 3) return false;
    for (int i = 0; i < arity; i++) {
        if (!kernel_partition_expr(kernel, expr->parameters[i], defined, count, depth + 1, budget)) return false;
    }
    if (!expr->function) {
        return arity == 1 && kernel_partition_cast(kernel, expr->parameters[0], expr->dtype);
    }
    if (expr->dtype != ME_FLOAT32 && expr->dtype != ME_FLOAT64 &&
        (!(expr->flags & ME_EXPR_FLAG_NUMPY_1_1) || (expr->flags & ME_EXPR_FLAG_WEAK_SCALAR))) {
        return false;
    }
    const char *op = me_menudet_operator(expr);
    bool is_total_bool = false;
    bool is_where = false;
    if (op) {
        if (!kernel_partition_operator(op, arity, &is_total_bool)) {
            return false;
        }
        is_where = arity == 3 && !strcmp(op, "where");
    } else if (!kernel_partition_math(me_menudet_math_name(expr), arity, expr->dtype, &is_total_bool)) {
        return false;
    }
    if (is_total_bool) {
        return true;
    }
    for (int i = is_where ? 1 : 0; i < arity; i++) {
        if (!kernel_partition_cast(kernel, expr->parameters[i], expr->dtype)) return false;
    }
    return true;
}

static bool kernel_straight_partitionable(const me_kernel *kernel) {
    if (!kernel || kernel->info.context_ndim ||
        kernel->info.cardinality != ME_ELEMENTWISE) return false;
    bool defined[ME_MAX_VARS] = {false};
    for (size_t i = 0; i < kernel->nbindings; i++) {
        if (!kernel_real_dtype(kernel->dtypes[i])) return false;
        defined[i] = true;
    }
    int budget = 65536;
    if (kernel->formula) {
        return kernel_partition_expr(kernel, kernel->formula, defined, kernel->nbindings, 0, &budget);
    }
    const me_dsl_compiled_program *program = kernel->program;
    if (!program || program->vars.count > ME_MAX_VARS || program->fp_mode != ME_DSL_FP_STRICT) return false;
    for (int i = 0; i < program->block.nstmts; i++) {
        const me_dsl_compiled_stmt *stmt = program->block.stmts[i];
        if (stmt->kind == ME_DSL_STMT_ASSIGN) {
            int slot = program->local_var_indices[stmt->as.assign.local_slot];
            const me_expr *expr = stmt->as.assign.value.expr;
            if (slot < 0 || slot >= program->vars.count ||
                !kernel_partition_expr(kernel, expr, defined, (size_t)program->vars.count, 0, &budget) ||
                !kernel_partition_cast(kernel, expr, program->vars.dtypes[slot])) return false;
            defined[slot] = true;
        } else if (stmt->kind == ME_DSL_STMT_RETURN) {
            const me_expr *expr = stmt->as.return_stmt.expr.expr;
            return i == program->block.nstmts - 1 &&
                kernel_partition_expr(kernel, expr, defined, (size_t)program->vars.count, 0, &budget) &&
                kernel_partition_cast(kernel, expr, kernel->info.output_dtype);
        } else {
            return false;
        }
    }
    return false;
}

bool me_kernel_is_partitionable(const me_kernel *kernel) {
    if (kernel_straight_partitionable(kernel)) return true;
    if (!kernel || !kernel->program || kernel->info.context_ndim ||
        kernel->info.cardinality != ME_ELEMENTWISE ||
        (kernel->info.output_dtype != ME_FLOAT32 && kernel->info.output_dtype != ME_FLOAT64 &&
         kernel->info.output_dtype != ME_BOOL)) return false;
    const me_dsl_compiled_program *p = kernel->program;
    if (p->vars.count > ME_MAX_VARS || p->fp_mode != ME_DSL_FP_STRICT ||
        p->output_is_scalar || p->uses_i_mask || p->uses_n_mask || p->uses_ndim ||
        p->uses_flat_idx) return false;
    /* Reuse the adapter's bounded loop/control-flow proof, including checked
     * counter arithmetic and weak final-narrowing guards. No source replay. */
    int budget = 65536;
    artifact_proof proof = {.program = p, .budget = &budget};
    for (size_t i = 0; i < kernel->nbindings; i++) {
        me_dtype dtype = kernel->dtypes[i];
        if (i < kernel->ninputs) {
            if (dtype != ME_FLOAT32 && dtype != ME_FLOAT64) return false;
            proof.values[i] = (artifact_proof_value){.valid = true};
        } else if (dtype == ME_INT64) {
            int64_t value;
            memcpy(&value, kernel->captures[i], sizeof(value));
            proof.values[i] = artifact_proof_integer(value);
        } else if (dtype == ME_BOOL) {
            bool value;
            memcpy(&value, kernel->captures[i], sizeof(value));
            proof.values[i] = artifact_proof_integer(value);
        } else if (dtype == ME_FLOAT32 || dtype == ME_FLOAT64) {
            proof.values[i] = artifact_proof_floating(dtype, kernel->captures[i]);
        }
    }
    artifact_proof_flow flow = ARTIFACT_FLOW_NEXT;
    return artifact_proof_block(&proof, &p->block, 0, 0, &flow, NULL, NULL);
}

me_status me_describe_input(const me_kernel *kernel, size_t slot, me_input_spec *out) {
    if (!kernel || !out || slot >= kernel->ninputs) {
        return ME_ERR_INVALID_ARG;
    }
    *out = (me_input_spec){NULL, kernel->dtypes[slot], kernel->widths[slot]};
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

typedef struct {
    dsl_menudet_fp_state saved;
    unsigned *previous;
} kernel_fp_scope;
typedef char kernel_fp_scope_size_check[sizeof(kernel_fp_scope) <= sizeof(me_fp_scope) ? 1 : -1];

me_status me_fp_scope_begin(me_fp_scope *scope, me_fp_status *status, me_error *error) {
    if (error) {
        memset(error, 0, sizeof(*error));
    }
    if (!scope || !status) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid floating scope");
    }
    *status = (me_fp_status){0};
    kernel_fp_scope state = {0};
    if (!dsl_menudet_fp_begin(&state.saved)) {
        return kernel_error(error, ME_ERR_UNSUPPORTED, "cannot establish strict floating environment");
    }
#ifndef __EMSCRIPTEN__
    status->supported = true;
#endif
    state.previous = dsl_menudet_status_begin(&status->flags);
    memcpy(scope->opaque, &state, sizeof(state));
    return ME_SUCCESS;
}

me_status me_fp_scope_end(me_fp_scope *scope, me_error *error) {
    if (error) {
        memset(error, 0, sizeof(*error));
    }
    if (!scope) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid floating scope");
    }
    kernel_fp_scope state;
    memcpy(&state, scope->opaque, sizeof(state));
    bool restored = dsl_menudet_fp_end(&state.saved);
    dsl_menudet_status_end(state.previous);
    return restored ? ME_SUCCESS :
        kernel_error(error, ME_ERR_UNSUPPORTED, "cannot restore floating environment");
}

me_status me_kernel_run_begin(const me_run_options *options, me_fp_status *status,
    me_error *error) {
    if (error) memset(error, 0, sizeof(*error));
    if (options && (options->struct_size < ME_RUN_OPTIONS_V1_SIZE || (options->raise_mask & ~15u))) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid run options");
    }
    if (options && options->execution != ME_RUN_PREPARED && options->execution != ME_RUN_INTERPRET) {
        return kernel_error(error, ME_ERR_INVALID_ARG, "invalid run execution policy");
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
    return me_eval_block_backend(kernel, inputs, ninputs, output, block, options, NULL, error);
}

me_status me_eval_block_backend(const me_kernel *kernel, const me_buffer *inputs,
    size_t ninputs, me_mut_buffer output, const me_block *block,
    const me_run_options *options, me_execution_backend *backend, me_error *error) {
    if (backend) *backend = ME_BACKEND_NONE;
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
        if (backend) *backend = ME_BACKEND_INTERPRETER;
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
        me_execution_backend actual_backend = ME_BACKEND_NONE;
        me_dsl_menudet_eval_descriptor native = {
            .nitems = (int)block->nitems, .valid_mask = all_active ? NULL : block->valid_mask,
            .output_capacity = output.capacity, .ndim = block->ndim,
            .logical_shape = block->logical_shape, .block_origin = block->origin,
            .block_extent = block->extent,
            .force_interpreter = options && options->execution == ME_RUN_INTERPRET,
            .actual_backend = backend ? &actual_backend : NULL};
        rc = dsl_eval_program_menudet(kernel->program, data, (int)kernel->nbindings, output.data, &native);
        if (backend) *backend = actual_backend;
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

me_dtype me_kernel_inferred_dtype(const me_kernel *kernel) {
    if (!kernel) return ME_AUTO;
    if (kernel->formula) return kernel->inferred_dtype;
    return kernel->program ? kernel->program->inferred_dtype : ME_AUTO;
}

const char *me_kernel_entry_point(const me_kernel *kernel) {
    return kernel ? kernel->entry_point : NULL;
}

const char *me_kernel_execution_kind(const me_kernel *kernel) {
    if (!kernel) return NULL;
    if (kernel->formula) return "expression";
    return kernel->program ? (kernel->program->expression_only ? "expression" : "program") : NULL;
}

const char *me_kernel_math_backend(const me_kernel *kernel) {
    if (!kernel) return "unknown";
    if (kernel->formula) return dsl_math_backend_name(kernel->formula_math.backend);
    return kernel->program ? dsl_math_backend_name(kernel->program->math_backend) : "unknown";
}

const char *me_kernel_sleef_accuracy(const me_kernel *kernel) {
    if (!kernel) return "";
    if (kernel->formula) {
        return kernel->formula_math.backend == ME_DSL_MATH_SLEEF ?
            dsl_sleef_ulp_name(kernel->formula_math.ulp) : "";
    }
    if (!kernel->program || kernel->program->math_backend != ME_DSL_MATH_SLEEF) return "";
    return dsl_sleef_ulp_name(kernel->program->sleef_ulp);
}

const char *me_kernel_fp_mode(const me_kernel *kernel) {
    if (!kernel) return "unknown";
    if (kernel->formula) return "strict";
    return kernel->program ? dsl_fp_mode_name(kernel->program->fp_mode) : "unknown";
}
