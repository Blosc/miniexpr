/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_eval_internal.h"
#include "dsl_jit_cgen.h"
#include "dsl_portable_fp.h"
#include "dsl_portable_expr.h"

#include "functions.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const me_dsl_compiled_program *program;
    void **var_buffers;
    void **local_buffers;
    int nitems;
    const me_eval_params *params;
    void *output_block;
    const uint8_t *valid_mask;
    int group_nitems;
    bool scalar_output;
    bool empty_group;
    uint8_t **initialized;
} dsl_eval_ctx;

static int dsl_eval_expr_nitems(dsl_eval_ctx *ctx, const me_dsl_compiled_expr *expr,
                                void *out, int nitems) {
    if (!expr || !expr->expr) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (dsl_portable_typed_profile(ctx->program->semantic_profile)) {
        return dsl_portable_eval_expr_masked(expr->expr, (const void *const *)ctx->var_buffers,
                                             ctx->program->vars.count, (const uint8_t *const *)ctx->initialized, nitems, NULL,
                                             dtype_size(expr->expr->dtype), out);
    }
    const void *vars[ME_MAX_VARS];
    for (int i = 0; i < expr->n_vars; i++) {
        vars[i] = ctx->var_buffers[expr->var_indices[i]];
    }
    return me_eval(expr->expr, vars, expr->n_vars, out, nitems, ctx->params);
}

static int dsl_eval_expr_item(dsl_eval_ctx *ctx, const me_dsl_compiled_expr *expr,
                              int item, void *out) {
    if (!ctx || !ctx->program || !expr || !expr->expr || !out || item < 0 || item >= ctx->nitems) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (dsl_portable_typed_profile(ctx->program->semantic_profile)) {
        return dsl_portable_eval_expr(expr->expr, (const void *const *)ctx->var_buffers,
                                      ctx->program->vars.count, (const uint8_t *const *)ctx->initialized, item, ctx->nitems, NULL, out);
    }
    const void *vars[ME_MAX_VARS];
    for (int i = 0; i < expr->n_vars; i++) {
        const int var_index = expr->var_indices[i];
        if (var_index < 0 || var_index >= ctx->program->vars.count || !ctx->var_buffers[var_index]) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        size_t var_size = dsl_var_item_size(ctx->program, var_index);
        if (var_size == 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        /* Locals always own full per-lane buffers. A compile-time uniform RHS
         * does not guarantee uniform storage after masked assignments/updates:
         * lane zero may have exited while other lanes continue changing it. */
        bool is_local = ctx->program->local_slots && ctx->program->local_slots[var_index] >= 0;
        if (!is_local && ctx->program->vars.uniform && ctx->program->vars.uniform[var_index]) {
            vars[i] = (const unsigned char *)ctx->var_buffers[var_index];
        }
        else {
            vars[i] = (const unsigned char *)ctx->var_buffers[var_index] + (size_t)item * var_size;
        }
    }
    return me_eval(expr->expr, vars, expr->n_vars, out, 1, ctx->params);
}

static int dsl_eval_block_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_block *block,
                                       uint8_t *run_mask, uint8_t *break_mask,
                                       uint8_t *continue_mask, uint8_t *return_mask);
static int dsl_eval_while_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_stmt *stmt,
                                       const uint8_t *input_mask, uint8_t *return_mask);
static int dsl_eval_for_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_stmt *stmt,
                                     const uint8_t *input_mask, uint8_t *return_mask);

static bool dsl_mask_any(const uint8_t *mask, int nitems) {
    if (!mask || nitems <= 0) {
        return false;
    }
    for (int i = 0; i < nitems; i++) {
        if (mask[i]) {
            return true;
        }
    }
    return false;
}

static void dsl_mask_remove_flow(uint8_t *run_mask, const uint8_t *break_mask,
                                 const uint8_t *continue_mask,
                                 const uint8_t *return_mask, int nitems) {
    if (!run_mask || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        if ((break_mask && break_mask[i]) ||
            (continue_mask && continue_mask[i]) ||
            (return_mask && return_mask[i])) {
            run_mask[i] = 0;
        }
    }
}

static bool dsl_string_nonempty_at(const void *data, size_t item_size, int idx) {
    if (!data || idx < 0 || item_size == 0 || (item_size % sizeof(uint32_t)) != 0) {
        return false;
    }
    const char *base = (const char *)data + (size_t)idx * item_size;
    const uint32_t *s = (const uint32_t *)base;
    return s[0] != 0;
}

static bool dsl_value_nonzero_at(const void *data, me_dtype dtype, size_t item_size, int idx) {
    if (!data || idx < 0) {
        return false;
    }
    switch (dtype) {
    case ME_BOOL:
        return ((const bool *)data)[idx];
    case ME_INT8:
        return ((const int8_t *)data)[idx] != 0;
    case ME_INT16:
        return ((const int16_t *)data)[idx] != 0;
    case ME_INT32:
        return ((const int32_t *)data)[idx] != 0;
    case ME_INT64:
        return ((const int64_t *)data)[idx] != 0;
    case ME_UINT8:
        return ((const uint8_t *)data)[idx] != 0;
    case ME_UINT16:
        return ((const uint16_t *)data)[idx] != 0;
    case ME_UINT32:
        return ((const uint32_t *)data)[idx] != 0;
    case ME_UINT64:
        return ((const uint64_t *)data)[idx] != 0;
    case ME_FLOAT32:
        return ((const float *)data)[idx] != 0.0f;
    case ME_FLOAT64:
        return ((const double *)data)[idx] != 0.0;
    case ME_COMPLEX64: {
        float _Complex v = ((const float _Complex *)data)[idx];
        return (me_crealf(v) != 0.0f || me_cimagf(v) != 0.0f);
    }
    case ME_COMPLEX128: {
        double _Complex v = ((const double _Complex *)data)[idx];
        return (me_creal(v) != 0.0 || me_cimag(v) != 0.0);
    }
    case ME_STRING:
    case ME_BYTES:
        return dsl_string_nonempty_at(data, item_size, idx);
    default:
        return false;
    }
}

/* Copy the value of `expr` into `dst` for the items `mask` selects.
 *
 * `dst_item_size` is the destination slot width, which for strings need not be
 * the expression's own: the program's output width is the widest of its return
 * expressions, so a narrower branch must be widened (NUL-padded) on the way in.
 * Pass 0 when the widths are known to agree (numeric output). */
static int dsl_eval_expr_masked_copy(dsl_eval_ctx *ctx, const me_dsl_compiled_expr *expr,
                                      void *dst, const uint8_t *mask, int nitems,
                                      size_t dst_item_size, bool active_only) {
    if (!ctx || !expr || !expr->expr || !dst || nitems < 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (nitems == 0) {
        return ME_EVAL_SUCCESS;
    }

    const size_t item_size = me_get_itemsize(expr->expr);
    if (item_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (dst_item_size == 0) {
        dst_item_size = item_size;
    }
    if (dst_item_size < item_size) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (ctx->empty_group) {
        if (!dsl_compiled_expr_uniform(expr->expr, ctx->program->vars.uniform, ctx->program->vars.count)) return ME_EVAL_SUCCESS;
        return dsl_portable_eval_expr(expr->expr, (const void *const *)ctx->var_buffers,
                                      ctx->program->vars.count, (const uint8_t *const *)ctx->initialized, 0, ctx->group_nitems, ctx->valid_mask, dst);
    }
    if (dsl_portable_typed_profile(ctx->program->semantic_profile)) {
        return dsl_portable_eval_expr_masked(expr->expr, (const void *const *)ctx->var_buffers,
                                             ctx->program->vars.count, (const uint8_t *const *)ctx->initialized, nitems, mask, dst_item_size, dst);
    }
    /* me_eval writes one result for a root reduction. DSL destinations are
     * full per-lane buffers, so materialize that scalar at every copied lane.
     * Otherwise masked chain captures can read uninitialized local slots. */
    bool scalar_result = is_reduction_node(expr->expr);

    bool all_active = (mask == NULL);
    if (!all_active) {
        all_active = true;
        for (int i = 0; i < nitems; i++) {
            if (!mask[i]) {
                all_active = false;
                break;
            }
        }
    }
    if (all_active && dst_item_size == item_size) {
        int rc = dsl_eval_expr_nitems(ctx, expr, dst, nitems);
        if (rc == ME_EVAL_SUCCESS && scalar_result) {
            for (int i = 1; i < nitems; i++) {
                memcpy((unsigned char *)dst + (size_t)i * item_size, dst, item_size);
            }
        }
        return rc;
    }

    if (active_only || dsl_portable_typed_profile(ctx->program->semantic_profile)) {
        /* Masking the copy after full-vector evaluation is not short-circuit
         * evaluation: skipped callbacks and invalid arithmetic still run.
         * Chain operand captures must evaluate only the active lanes. */
        for (int i = 0; i < nitems; i++) {
            if (mask && !mask[i]) {
                continue;
            }
            unsigned char *slot = (unsigned char *)dst + (size_t)i * dst_item_size;
            int rc = dsl_eval_expr_item(ctx, expr, i, slot);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            if (dst_item_size > item_size) {
                memset(slot + item_size, 0, dst_item_size - item_size);
            }
        }
        return ME_EVAL_SUCCESS;
    }

    void *tmp = malloc((size_t)nitems * item_size);
    if (!tmp) {
        return ME_EVAL_ERR_OOM;
    }
    int rc = dsl_eval_expr_nitems(ctx, expr, tmp, nitems);
    if (rc != ME_EVAL_SUCCESS) {
        free(tmp);
        return rc;
    }

    unsigned char *dst_bytes = (unsigned char *)dst;
    const unsigned char *src_bytes = (const unsigned char *)tmp;
    for (int i = 0; i < nitems; i++) {
        if (mask && !mask[i]) {
            continue;
        }
        unsigned char *slot = dst_bytes + (size_t)i * dst_item_size;
        memcpy(slot, src_bytes + (scalar_result ? 0 : (size_t)i * item_size), item_size);
        if (dst_item_size > item_size) {
            memset(slot + item_size, 0, dst_item_size - item_size);
        }
    }

    free(tmp);
    return ME_EVAL_SUCCESS;
}

static int dsl_eval_condition_masked(dsl_eval_ctx *ctx, const me_dsl_compiled_expr *cond,
                                     const uint8_t *input_mask, uint8_t *true_mask,
                                     bool *is_reduction, bool *scalar_true) {
    if (!ctx || !cond || !cond->expr || !is_reduction || !scalar_true) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    *is_reduction = is_reduction_node(cond->expr);
    *scalar_true = false;

    me_dtype cond_dtype = me_get_dtype(cond->expr);
    size_t cond_size = dtype_size(cond_dtype);
    if (is_string_dtype(cond_dtype)) {
        cond_size = cond->expr->itemsize;
    }
    if (cond_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    int cond_nitems = *is_reduction ? 1 : ctx->nitems;
    if (cond_nitems <= 0) {
        if (true_mask && !*is_reduction && ctx->nitems > 0) {
            memset(true_mask, 0, (size_t)ctx->nitems);
        }
        return ME_EVAL_SUCCESS;
    }

    void *cond_buf = malloc((size_t)cond_nitems * cond_size);
    if (!cond_buf) {
        return ME_EVAL_ERR_OOM;
    }
    int rc;
    if (dsl_portable_typed_profile(ctx->program->semantic_profile)) {
        memset(cond_buf, 0, (size_t)cond_nitems * cond_size);
        if (*is_reduction) {
            rc = dsl_portable_eval_expr(cond->expr, (const void *const *)ctx->var_buffers,
                                        ctx->program->vars.count, (const uint8_t *const *)ctx->initialized, 0, ctx->nitems, input_mask, cond_buf);
        }
        else rc = dsl_eval_expr_masked_copy(ctx, cond, cond_buf, input_mask, ctx->nitems, 0, true);
    }
    else rc = dsl_eval_expr_nitems(ctx, cond, cond_buf, ctx->nitems);
    if (rc != ME_EVAL_SUCCESS) {
        free(cond_buf);
        return rc;
    }

    if (*is_reduction) {
        *scalar_true = dsl_any_nonzero(cond_buf, cond_dtype, cond_nitems);
    }
    else {
        if (!true_mask) {
            free(cond_buf);
            return ME_EVAL_ERR_INVALID_ARG;
        }
        for (int i = 0; i < ctx->nitems; i++) {
            bool active = !input_mask || input_mask[i];
            true_mask[i] = (uint8_t)(active &&
                                     dsl_value_nonzero_at(cond_buf, cond_dtype, cond_size, i));
        }
    }

    free(cond_buf);
    return ME_EVAL_SUCCESS;
}

static int dsl_eval_element_conditional_branch(dsl_eval_ctx *ctx,
                                               const me_dsl_compiled_expr *cond,
                                               const me_dsl_compiled_block *branch_block,
                                               uint8_t *remaining_mask,
                                               uint8_t *break_mask,
                                               uint8_t *continue_mask,
                                               uint8_t *return_mask) {
    if (!ctx || !cond || !branch_block || !remaining_mask ||
        !break_mask || !continue_mask || !return_mask) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (!dsl_mask_any(remaining_mask, ctx->nitems)) {
        return ME_EVAL_SUCCESS;
    }

    uint8_t *cond_mask = calloc((size_t)ctx->nitems, sizeof(*cond_mask));
    if (!cond_mask) {
        return ME_EVAL_ERR_OOM;
    }
    bool cond_is_reduction = false;
    bool cond_scalar_true = false;
    int rc = dsl_eval_condition_masked(ctx, cond, remaining_mask, cond_mask,
                                       &cond_is_reduction, &cond_scalar_true);
    if (rc != ME_EVAL_SUCCESS) {
        free(cond_mask);
        return rc;
    }

    if (cond_is_reduction) {
        if (cond_scalar_true) {
            uint8_t *branch_run = malloc((size_t)ctx->nitems);
            if (!branch_run) {
                free(cond_mask);
                return ME_EVAL_ERR_OOM;
            }
            memcpy(branch_run, remaining_mask, (size_t)ctx->nitems);
            memset(remaining_mask, 0, (size_t)ctx->nitems);
            rc = dsl_eval_block_element_loop(ctx, branch_block, branch_run,
                                             break_mask, continue_mask, return_mask);
            free(branch_run);
        }
        free(cond_mask);
        return rc;
    }

    if (!dsl_mask_any(cond_mask, ctx->nitems)) {
        free(cond_mask);
        return ME_EVAL_SUCCESS;
    }

    uint8_t *branch_run = malloc((size_t)ctx->nitems);
    if (!branch_run) {
        free(cond_mask);
        return ME_EVAL_ERR_OOM;
    }
    memcpy(branch_run, cond_mask, (size_t)ctx->nitems);
    for (int i = 0; i < ctx->nitems; i++) {
        if (cond_mask[i]) {
            remaining_mask[i] = 0;
        }
    }
    rc = dsl_eval_block_element_loop(ctx, branch_block, branch_run,
                                     break_mask, continue_mask, return_mask);
    free(branch_run);
    free(cond_mask);
    return rc;
}

static void dsl_format_value(char *buf, size_t cap, me_dtype dtype, const void *data) {
    if (!buf || cap == 0 || !data) {
        return;
    }
    switch (dtype) {
    case ME_BOOL:
        snprintf(buf, cap, "%s", (*(const bool *)data) ? "true" : "false");
        break;
    case ME_INT8:
        snprintf(buf, cap, "%lld", (long long)*(const int8_t *)data);
        break;
    case ME_INT16:
        snprintf(buf, cap, "%lld", (long long)*(const int16_t *)data);
        break;
    case ME_INT32:
        snprintf(buf, cap, "%lld", (long long)*(const int32_t *)data);
        break;
    case ME_INT64:
        snprintf(buf, cap, "%lld", (long long)*(const int64_t *)data);
        break;
    case ME_UINT8:
        snprintf(buf, cap, "%llu", (unsigned long long)*(const uint8_t *)data);
        break;
    case ME_UINT16:
        snprintf(buf, cap, "%llu", (unsigned long long)*(const uint16_t *)data);
        break;
    case ME_UINT32:
        snprintf(buf, cap, "%llu", (unsigned long long)*(const uint32_t *)data);
        break;
    case ME_UINT64:
        snprintf(buf, cap, "%llu", (unsigned long long)*(const uint64_t *)data);
        break;
    case ME_FLOAT32:
        snprintf(buf, cap, "%.9g", (double)*(const float *)data);
        break;
    case ME_FLOAT64:
        snprintf(buf, cap, "%.17g", *(const double *)data);
        break;
    case ME_COMPLEX64: {
        float _Complex v = *(const float _Complex *)data;
        snprintf(buf, cap, "%.9g%+.9gj", (double)me_crealf(v), (double)me_cimagf(v));
        break;
    }
    case ME_COMPLEX128: {
        double _Complex v = *(const double _Complex *)data;
        snprintf(buf, cap, "%.17g%+.17gj", me_creal(v), me_cimag(v));
        break;
    }
    default:
        snprintf(buf, cap, "<unsupported>");
        break;
    }
}

static void dsl_print_formatted(const char *fmt, char **arg_strs, int nargs) {
    int arg_idx = 0;
    for (size_t i = 0; fmt && fmt[i] != '\0'; i++) {
        if (fmt[i] == '{') {
            if (fmt[i + 1] == '{') {
                fputc('{', stdout);
                i++;
                continue;
            }
            if (fmt[i + 1] == '}') {
                if (arg_idx < nargs) {
                    fputs(arg_strs[arg_idx], stdout);
                }
                arg_idx++;
                i++;
                continue;
            }
        }
        if (fmt[i] == '}' && fmt[i + 1] == '}') {
            fputc('}', stdout);
            i++;
            continue;
        }
        fputc(fmt[i], stdout);
    }
    fputc('\n', stdout);
    fflush(stdout);
}

static int dsl_eval_block_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_block *block,
                                       uint8_t *run_mask, uint8_t *break_mask,
                                       uint8_t *continue_mask, uint8_t *return_mask) {
    if (!ctx || !block || !run_mask || !break_mask || !continue_mask || !return_mask) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    for (int i = 0; i < block->nstmts; i++) {
        dsl_mask_remove_flow(run_mask, break_mask, continue_mask, return_mask, ctx->nitems);
        if (!dsl_mask_any(run_mask, ctx->nitems)) {
            break;
        }

        me_dsl_compiled_stmt *stmt = block->stmts[i];
        if (!stmt) {
            continue;
        }

        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN: {
            int slot = stmt->as.assign.local_slot;
            if (ctx->empty_group && !dsl_compiled_expr_uniform(stmt->as.assign.value.expr, ctx->program->vars.uniform, ctx->program->vars.count)) break;
            void *out = ctx->local_buffers[slot];
            int rc = dsl_eval_expr_masked_copy(ctx, &stmt->as.assign.value, out,
                                               run_mask, ctx->nitems,
                                               dsl_var_item_size(ctx->program,
                                                                  ctx->program->local_var_indices[slot]),
                                               stmt->as.assign.active_only);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            if (ctx->initialized) {
                uint8_t *defined = ctx->initialized[ctx->program->local_var_indices[slot]];
                for (int j = 0; j < ctx->nitems; j++) if (run_mask[j]) defined[j] = 1;
            }
            break;
        }
        case ME_DSL_STMT_EXPR: {
            if (ctx->scalar_output) {
                size_t size = me_get_itemsize(stmt->as.expr_stmt.expr.expr);
                void *scratch = malloc((size_t)ctx->nitems * size);
                if (!scratch) return ME_EVAL_ERR_OOM;
                int rc = dsl_eval_expr_masked_copy(ctx, &stmt->as.expr_stmt.expr, scratch, run_mask, ctx->nitems, size, true);
                free(scratch);
                if (rc) return rc;
                break;
            }
            int rc = dsl_eval_expr_masked_copy(ctx, &stmt->as.expr_stmt.expr,
                                               ctx->output_block, run_mask, ctx->nitems,
                                               ctx->program->output_itemsize, false);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_RETURN: {
            int rc;
            if (ctx->scalar_output) {
                int lane = 0;
                while (lane < ctx->nitems && !run_mask[lane]) lane++;
                rc = dsl_portable_eval_expr(stmt->as.return_stmt.expr.expr,
                    (const void *const *)ctx->var_buffers, ctx->program->vars.count,
                    (const uint8_t *const *)ctx->initialized, lane, ctx->group_nitems,
                    ctx->empty_group ? ctx->valid_mask : run_mask, ctx->output_block);
            }
            else rc = dsl_eval_expr_masked_copy(ctx, &stmt->as.return_stmt.expr,
                                               ctx->output_block, run_mask, ctx->nitems,
                                               ctx->program->output_itemsize, false);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            for (int j = 0; j < ctx->nitems; j++) {
                if (run_mask[j]) {
                    return_mask[j] = 1;
                    run_mask[j] = 0;
                }
            }
            break;
        }
        case ME_DSL_STMT_PRINT: {
            if (!dsl_mask_any(run_mask, ctx->nitems)) {
                break;
            }
            int nargs = stmt->as.print_stmt.nargs;
            char **arg_strs = NULL;
            void **arg_bufs = NULL;
            if (nargs > 0) {
                arg_strs = calloc((size_t)nargs, sizeof(*arg_strs));
                arg_bufs = calloc((size_t)nargs, sizeof(*arg_bufs));
                if (!arg_strs || !arg_bufs) {
                    free(arg_strs);
                    free(arg_bufs);
                    return ME_EVAL_ERR_OOM;
                }
            }
            for (int j = 0; j < nargs; j++) {
                me_dsl_compiled_expr *arg = &stmt->as.print_stmt.args[j];
                me_dtype dtype = me_get_dtype(arg->expr);
                size_t size = dtype_size(dtype);
                if (size == 0) {
                    size = sizeof(double);
                }
                arg_bufs[j] = calloc(1, size);
                if (!arg_bufs[j]) {
                    for (int k = 0; k < j; k++) {
                        free(arg_bufs[k]);
                        free(arg_strs[k]);
                    }
                    free(arg_bufs);
                    free(arg_strs);
                    return ME_EVAL_ERR_OOM;
                }
                int rc = dsl_eval_expr_nitems(ctx, arg, arg_bufs[j], 1);
                if (rc != ME_EVAL_SUCCESS) {
                    for (int k = 0; k <= j; k++) {
                        free(arg_bufs[k]);
                        free(arg_strs[k]);
                    }
                    free(arg_bufs);
                    free(arg_strs);
                    return rc;
                }
                arg_strs[j] = calloc(1, 64);
                if (!arg_strs[j]) {
                    for (int k = 0; k <= j; k++) {
                        free(arg_bufs[k]);
                        free(arg_strs[k]);
                    }
                    free(arg_bufs);
                    free(arg_strs);
                    return ME_EVAL_ERR_OOM;
                }
                dsl_format_value(arg_strs[j], 64, dtype, arg_bufs[j]);
            }
            dsl_print_formatted(stmt->as.print_stmt.format, arg_strs, nargs);
            for (int j = 0; j < nargs; j++) {
                free(arg_bufs[j]);
                free(arg_strs[j]);
            }
            free(arg_bufs);
            free(arg_strs);
            break;
        }
        case ME_DSL_STMT_IF: {
            if (ctx->empty_group) break; /* No lanes participate in control-flow bodies. */
            uint8_t *remaining = malloc((size_t)ctx->nitems);
            if (!remaining) {
                return ME_EVAL_ERR_OOM;
            }
            memcpy(remaining, run_mask, (size_t)ctx->nitems);

            int rc = dsl_eval_element_conditional_branch(ctx, &stmt->as.if_stmt.cond,
                                                         &stmt->as.if_stmt.then_block,
                                                         remaining, break_mask,
                                                         continue_mask, return_mask);
            if (rc != ME_EVAL_SUCCESS) {
                free(remaining);
                return rc;
            }

            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                if (!dsl_mask_any(remaining, ctx->nitems)) {
                    break;
                }
                me_dsl_compiled_if_branch *branch = &stmt->as.if_stmt.elif_branches[j];
                rc = dsl_eval_element_conditional_branch(ctx, &branch->cond, &branch->block,
                                                         remaining, break_mask,
                                                         continue_mask, return_mask);
                if (rc != ME_EVAL_SUCCESS) {
                    free(remaining);
                    return rc;
                }
            }

            if (stmt->as.if_stmt.has_else && dsl_mask_any(remaining, ctx->nitems)) {
                uint8_t *else_run = malloc((size_t)ctx->nitems);
                if (!else_run) {
                    free(remaining);
                    return ME_EVAL_ERR_OOM;
                }
                memcpy(else_run, remaining, (size_t)ctx->nitems);
                rc = dsl_eval_block_element_loop(ctx, &stmt->as.if_stmt.else_block,
                                                 else_run, break_mask,
                                                 continue_mask, return_mask);
                free(else_run);
                if (rc != ME_EVAL_SUCCESS) {
                    free(remaining);
                    return rc;
                }
            }

            free(remaining);
            dsl_mask_remove_flow(run_mask, break_mask, continue_mask, return_mask, ctx->nitems);
            break;
        }
        case ME_DSL_STMT_FOR: {
            if (ctx->empty_group) break;
            int rc = dsl_eval_for_element_loop(ctx, stmt, run_mask, return_mask);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_WHILE: {
            if (ctx->empty_group) break;
            int rc = dsl_eval_while_element_loop(ctx, stmt, run_mask, return_mask);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_BREAK:
        case ME_DSL_STMT_CONTINUE: {
            bool cond_is_reduction = false;
            bool cond_scalar_true = false;
            uint8_t *trigger_mask = calloc((size_t)ctx->nitems, sizeof(*trigger_mask));
            if (!trigger_mask) {
                return ME_EVAL_ERR_OOM;
            }

            int rc = ME_EVAL_SUCCESS;
            if (stmt->as.flow.cond.expr) {
                rc = dsl_eval_condition_masked(ctx, &stmt->as.flow.cond, run_mask,
                                               trigger_mask, &cond_is_reduction,
                                               &cond_scalar_true);
                if (rc != ME_EVAL_SUCCESS) {
                    free(trigger_mask);
                    return rc;
                }
                if (cond_is_reduction) {
                    memset(trigger_mask, 0, (size_t)ctx->nitems);
                    if (cond_scalar_true) {
                        memcpy(trigger_mask, run_mask, (size_t)ctx->nitems);
                    }
                }
            }
            else {
                memcpy(trigger_mask, run_mask, (size_t)ctx->nitems);
            }

            for (int j = 0; j < ctx->nitems; j++) {
                if (!trigger_mask[j]) {
                    continue;
                }
                run_mask[j] = 0;
                if (stmt->kind == ME_DSL_STMT_BREAK) {
                    break_mask[j] = 1;
                }
                else {
                    continue_mask[j] = 1;
                }
            }
            free(trigger_mask);
            break;
        }
        }
    }

    return ME_EVAL_SUCCESS;
}

static int dsl_eval_while_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_stmt *stmt,
                                       const uint8_t *input_mask, uint8_t *return_mask) {
    if (!ctx || !stmt || stmt->kind != ME_DSL_STMT_WHILE) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (!stmt->as.while_loop.cond.expr) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (ctx->nitems <= 0) {
        return ME_EVAL_SUCCESS;
    }

    uint8_t *active_mask = malloc((size_t)ctx->nitems);
    if (!active_mask) {
        return ME_EVAL_ERR_OOM;
    }
    if (input_mask) {
        memcpy(active_mask, input_mask, (size_t)ctx->nitems);
    }
    else {
        memset(active_mask, 1, (size_t)ctx->nitems);
    }
    dsl_mask_remove_flow(active_mask, NULL, NULL, return_mask, ctx->nitems);
    if (!dsl_mask_any(active_mask, ctx->nitems)) {
        free(active_mask);
        return ME_EVAL_SUCCESS;
    }

    int64_t max_iters = dsl_while_max_iters();
    int64_t iter_count = 0;

    int rc = ME_EVAL_SUCCESS;
    for (;;) {
        if (stmt->as.while_loop.condition_nstmts && dsl_mask_any(active_mask, ctx->nitems)) {
            /* Lowered condition statements belong to condition evaluation,
             * not to the body iteration count (especially at the cap). */
            uint8_t *prefix_run = malloc((size_t)ctx->nitems);
            uint8_t *prefix_break = calloc((size_t)ctx->nitems, 1);
            uint8_t *prefix_continue = calloc((size_t)ctx->nitems, 1);
            if (!prefix_run || !prefix_break || !prefix_continue) {
                free(prefix_run);
                free(prefix_break);
                free(prefix_continue);
                free(active_mask);
                return ME_EVAL_ERR_OOM;
            }
            memcpy(prefix_run, active_mask, (size_t)ctx->nitems);
            me_dsl_compiled_block prefix = stmt->as.while_loop.body;
            prefix.nstmts = stmt->as.while_loop.condition_nstmts;
            rc = dsl_eval_block_element_loop(ctx, &prefix, prefix_run,
                                             prefix_break, prefix_continue, return_mask);
            for (int i = 0; i < ctx->nitems; i++) {
                if (prefix_break[i] || return_mask[i]) {
                    active_mask[i] = 0;
                }
            }
            free(prefix_run);
            free(prefix_break);
            free(prefix_continue);
            if (rc != ME_EVAL_SUCCESS) {
                free(active_mask);
                return rc;
            }
        }
        if (!dsl_mask_any(active_mask, ctx->nitems)) {
            break;
        }

        uint8_t *cond_mask = calloc((size_t)ctx->nitems, sizeof(*cond_mask));
        if (!cond_mask) {
            free(active_mask);
            return ME_EVAL_ERR_OOM;
        }
        bool cond_is_reduction = false;
        bool cond_scalar_true = false;
        rc = dsl_eval_condition_masked(ctx, &stmt->as.while_loop.cond, active_mask,
                                       cond_mask, &cond_is_reduction, &cond_scalar_true);
        if (rc != ME_EVAL_SUCCESS) {
            free(cond_mask);
            free(active_mask);
            return rc;
        }

        if (cond_is_reduction && !cond_scalar_true) {
            free(cond_mask);
            break;
        }
        if (!cond_is_reduction && !dsl_mask_any(cond_mask, ctx->nitems)) {
            free(cond_mask);
            break;
        }
        if (max_iters > 0 && iter_count >= max_iters) {
            dsl_tracef("while iteration cap hit at %d:%d after %lld iterations (limit=%lld)",
                       stmt->line, stmt->column,
                       (long long)iter_count, (long long)max_iters);
            free(cond_mask);
            free(active_mask);
            return ME_EVAL_ERR_INVALID_ARG;
        }
        iter_count++;

        uint8_t *run_mask = malloc((size_t)ctx->nitems);
        uint8_t *break_mask = calloc((size_t)ctx->nitems, sizeof(*break_mask));
        uint8_t *continue_mask = calloc((size_t)ctx->nitems, sizeof(*continue_mask));
        if (!run_mask || !break_mask || !continue_mask) {
            free(run_mask);
            free(break_mask);
            free(continue_mask);
            free(cond_mask);
            free(active_mask);
            return ME_EVAL_ERR_OOM;
        }

        if (cond_is_reduction) {
            memcpy(run_mask, active_mask, (size_t)ctx->nitems);
        }
        else {
            memcpy(run_mask, cond_mask, (size_t)ctx->nitems);
            for (int i = 0; i < ctx->nitems; i++) {
                if (!cond_mask[i]) {
                    active_mask[i] = 0;
                }
            }
        }

        me_dsl_compiled_block body = stmt->as.while_loop.body;
        if (stmt->as.while_loop.condition_nstmts) {
            body.stmts += stmt->as.while_loop.condition_nstmts;
            body.nstmts -= stmt->as.while_loop.condition_nstmts;
        }
        rc = dsl_eval_block_element_loop(ctx, &body, run_mask,
                                         break_mask, continue_mask, return_mask);
        free(run_mask);
        free(cond_mask);
        if (rc != ME_EVAL_SUCCESS) {
            free(break_mask);
            free(continue_mask);
            free(active_mask);
            return rc;
        }

        for (int i = 0; i < ctx->nitems; i++) {
            if (break_mask[i] || return_mask[i]) {
                active_mask[i] = 0;
            }
        }

        free(break_mask);
        free(continue_mask);
    }

    free(active_mask);
    return ME_EVAL_SUCCESS;
}

static int dsl_eval_for_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_stmt *stmt,
                                     const uint8_t *input_mask, uint8_t *return_mask) {
    if (!ctx || !stmt || stmt->kind != ME_DSL_STMT_FOR) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const me_dsl_compiled_expr *start_expr = &stmt->as.for_loop.start;
    const me_dsl_compiled_expr *stop_expr = &stmt->as.for_loop.stop;
    const me_dsl_compiled_expr *step_expr = &stmt->as.for_loop.step;
    if (!start_expr->expr || !stop_expr->expr || !step_expr->expr) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    if (ctx->nitems <= 0) {
        return ME_EVAL_SUCCESS;
    }

    uint8_t *active_mask = malloc((size_t)ctx->nitems);
    if (!active_mask) {
        return ME_EVAL_ERR_OOM;
    }

    if (input_mask) {
        memcpy(active_mask, input_mask, (size_t)ctx->nitems);
    }
    else {
        memset(active_mask, 1, (size_t)ctx->nitems);
    }
    dsl_mask_remove_flow(active_mask, NULL, NULL, return_mask, ctx->nitems);

    if (!dsl_mask_any(active_mask, ctx->nitems)) {
        free(active_mask);
        return ME_EVAL_SUCCESS;
    }

    int rc = ME_EVAL_SUCCESS;
    me_dtype start_dtype = me_get_dtype(start_expr->expr);
    me_dtype stop_dtype = me_get_dtype(stop_expr->expr);
    me_dtype step_dtype = me_get_dtype(step_expr->expr);
    size_t start_size = dtype_size(start_dtype);
    size_t stop_size = dtype_size(stop_dtype);
    size_t step_size = dtype_size(step_dtype);
    if (start_size == 0 || stop_size == 0 || step_size == 0) {
        free(active_mask);
        return ME_EVAL_ERR_INVALID_ARG;
    }

    void *start_buf = malloc(start_size);
    void *stop_buf = malloc(stop_size);
    void *step_buf = malloc(step_size);
    int64_t *iter_vals = malloc((size_t)ctx->nitems * sizeof(*iter_vals));
    int64_t *stop_vals = malloc((size_t)ctx->nitems * sizeof(*stop_vals));
    int64_t *step_vals = malloc((size_t)ctx->nitems * sizeof(*step_vals));
    if (!start_buf || !stop_buf || !step_buf || !iter_vals || !stop_vals || !step_vals) {
        free(start_buf);
        free(stop_buf);
        free(step_buf);
        free(iter_vals);
        free(stop_vals);
        free(step_vals);
        free(active_mask);
        return ME_EVAL_ERR_OOM;
    }

    for (int i = 0; i < ctx->nitems; i++) {
        if (!active_mask[i]) {
            continue;
        }
        int64_t start_val = 0;
        int64_t stop_val = 0;
        int64_t step_val = 0;
        rc = dsl_eval_expr_item(ctx, start_expr, i, start_buf);
        if (rc != ME_EVAL_SUCCESS) {
            goto cleanup;
        }
        rc = dsl_eval_expr_item(ctx, stop_expr, i, stop_buf);
        if (rc != ME_EVAL_SUCCESS) {
            goto cleanup;
        }
        rc = dsl_eval_expr_item(ctx, step_expr, i, step_buf);
        if (rc != ME_EVAL_SUCCESS) {
            goto cleanup;
        }
        if (!dsl_read_int64(start_buf, start_dtype, &start_val) ||
            !dsl_read_int64(stop_buf, stop_dtype, &stop_val) ||
            !dsl_read_int64(step_buf, step_dtype, &step_val)) {
            rc = ME_EVAL_ERR_INVALID_ARG;
            goto cleanup;
        }
        if (step_val == 0) {
            rc = ME_EVAL_ERR_INVALID_ARG;
            goto cleanup;
        }
        iter_vals[i] = start_val;
        stop_vals[i] = stop_val;
        step_vals[i] = step_val;
        if ((step_val > 0 && start_val >= stop_val) ||
            (step_val < 0 && start_val <= stop_val)) {
            active_mask[i] = 0;
        }
    }

    int slot = stmt->as.for_loop.loop_var_slot;
    int64_t *loop_buf = (int64_t *)ctx->local_buffers[slot];

    while (dsl_mask_any(active_mask, ctx->nitems)) {
        for (int i = 0; i < ctx->nitems; i++) {
            /* A lane that exhausted its range or broke retains its last local
             * value. Do not let other lanes' iterations overwrite it. */
            if (!active_mask[i] && dsl_portable_typed_profile(ctx->program->semantic_profile)) continue;
            loop_buf[i] = active_mask[i] ? iter_vals[i] : 0;
            if (ctx->initialized && active_mask[i]) ctx->initialized[ctx->program->local_var_indices[slot]][i] = 1;
        }

        uint8_t *run_mask = malloc((size_t)ctx->nitems);
        uint8_t *break_mask = calloc((size_t)ctx->nitems, sizeof(*break_mask));
        uint8_t *continue_mask = calloc((size_t)ctx->nitems, sizeof(*continue_mask));
        if (!run_mask || !break_mask || !continue_mask) {
            free(run_mask);
            free(break_mask);
            free(continue_mask);
            rc = ME_EVAL_ERR_OOM;
            goto cleanup;
        }
        memcpy(run_mask, active_mask, (size_t)ctx->nitems);

        rc = dsl_eval_block_element_loop(ctx, &stmt->as.for_loop.body, run_mask,
                                         break_mask, continue_mask, return_mask);
        free(run_mask);
        if (rc != ME_EVAL_SUCCESS) {
            free(break_mask);
            free(continue_mask);
            goto cleanup;
        }

        for (int i = 0; i < ctx->nitems; i++) {
            if (!active_mask[i]) {
                continue;
            }
            if (break_mask[i] || (return_mask && return_mask[i])) {
                active_mask[i] = 0;
                continue;
            }
            int64_t step_val = step_vals[i];
            if (step_val > 0) {
                if (iter_vals[i] > INT64_MAX - step_val) {
                    active_mask[i] = 0;
                    continue;
                }
            }
            else {
                if (iter_vals[i] < INT64_MIN - step_val) {
                    active_mask[i] = 0;
                    continue;
                }
            }
            iter_vals[i] += step_val;
            if ((step_val > 0 && iter_vals[i] >= stop_vals[i]) ||
                (step_val < 0 && iter_vals[i] <= stop_vals[i])) {
                active_mask[i] = 0;
            }
        }

        free(break_mask);
        free(continue_mask);
    }

    rc = ME_EVAL_SUCCESS;

cleanup:
    free(start_buf);
    free(stop_buf);
    free(step_buf);
    free(iter_vals);
    free(stop_vals);
    free(step_vals);
    free(active_mask);
    return rc;
}

static int dsl_eval_block(dsl_eval_ctx *ctx, const me_dsl_compiled_block *block,
                          bool *did_break, bool *did_continue, bool *did_return) {
    if (!ctx || !block) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (ctx->nitems <= 0) {
        if (did_break) {
            *did_break = false;
        }
        if (did_continue) {
            *did_continue = false;
        }
        if (did_return) {
            *did_return = true;
        }
        return ME_EVAL_SUCCESS;
    }

    uint8_t *run_mask = malloc((size_t)ctx->nitems);
    uint8_t *break_mask = calloc((size_t)ctx->nitems, sizeof(*break_mask));
    uint8_t *continue_mask = calloc((size_t)ctx->nitems, sizeof(*continue_mask));
    uint8_t *return_mask = calloc((size_t)ctx->nitems, sizeof(*return_mask));
    if (!run_mask || !break_mask || !continue_mask || !return_mask) {
        free(run_mask);
        free(break_mask);
        free(continue_mask);
        free(return_mask);
        return ME_EVAL_ERR_OOM;
    }
    if (ctx->valid_mask && !ctx->empty_group) memcpy(run_mask, ctx->valid_mask, (size_t)ctx->nitems);
    else memset(run_mask, 1, (size_t)ctx->nitems);

    int rc = dsl_eval_block_element_loop(ctx, block, run_mask, break_mask, continue_mask, return_mask);
    if (did_break) {
        *did_break = dsl_mask_any(break_mask, ctx->nitems);
    }
    if (did_continue) {
        *did_continue = dsl_mask_any(continue_mask, ctx->nitems);
    }
    if (did_return) {
        *did_return = true;
        for (int i = 0; i < ctx->nitems; i++) {
            if (!return_mask[i] && (!ctx->valid_mask || ctx->empty_group || ctx->valid_mask[i])) {
                *did_return = false;
                break;
            }
        }
    }

    free(run_mask);
    free(break_mask);
    free(continue_mask);
    free(return_mask);
    return rc;
}

static bool dsl_fill_reserved_from_nd_synth_ctx(int nitems,
                                                const int64_t *nd_ctx,
                                                int64_t **idx_outputs,
                                                uint32_t idx_mask,
                                                int64_t *global_output);

#if ME_USE_WASM32_JIT
/* wasm32 runtime JIT lowers int64/uint64 generated C types to 32-bit.
   Marshal 64-bit outputs through temporary 32-bit buffers and widen back. */
static bool dsl_wasm32_prepare_jit_output(const me_dsl_compiled_program *program,
                                          void *output_block, int nitems,
                                          void **jit_output, void **jit_output_tmp) {
    if (!program || !output_block || !jit_output || !jit_output_tmp || nitems < 0) {
        return false;
    }

    *jit_output = output_block;
    *jit_output_tmp = NULL;
    if (nitems == 0) {
        return true;
    }

    switch (program->output_dtype) {
    case ME_INT64: {
        int *tmp = malloc((size_t)nitems * sizeof(*tmp));
        if (!tmp) {
            return false;
        }
        *jit_output = (void *)tmp;
        *jit_output_tmp = (void *)tmp;
        return true;
    }
    case ME_UINT64: {
        unsigned int *tmp = malloc((size_t)nitems * sizeof(*tmp));
        if (!tmp) {
            return false;
        }
        *jit_output = (void *)tmp;
        *jit_output_tmp = (void *)tmp;
        return true;
    }
    default:
        return true;
    }
}

static void dsl_wasm32_finalize_jit_output(const me_dsl_compiled_program *program,
                                           void *output_block, int nitems,
                                           void *jit_output_tmp, bool jit_success) {
    if (!jit_output_tmp) {
        return;
    }

    if (jit_success && program && output_block && nitems > 0) {
        switch (program->output_dtype) {
        case ME_INT64: {
            const int *src = (const int *)jit_output_tmp;
            int64_t *dst = (int64_t *)output_block;
            for (int i = 0; i < nitems; i++) {
                dst[i] = (int64_t)src[i];
            }
            break;
        }
        case ME_UINT64: {
            const unsigned int *src = (const unsigned int *)jit_output_tmp;
            uint64_t *dst = (uint64_t *)output_block;
            for (int i = 0; i < nitems; i++) {
                dst[i] = (uint64_t)src[i];
            }
            break;
        }
        default:
            break;
        }
    }

    free(jit_output_tmp);
}
#endif

static int dsl_eval_program_impl(const me_dsl_compiled_program *program,
                     const void **vars_block, int n_vars,
                     void *output_block, int nitems,
                     const me_eval_params *params,
                     int ndim, const int64_t *shape,
                     int64_t **idx_buffers,
                     const int64_t *global_linear_idx_buffer,
                      const int64_t *nd_synth_ctx_buffer,
                      const me_dsl_portable_eval_descriptor *descriptor) {
    if (!program || !output_block || nitems < 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (n_vars != program->n_inputs) {
        return ME_EVAL_ERR_VAR_MISMATCH;
    }
    int group_nitems = nitems;
    bool empty_group = descriptor && program->output_is_scalar &&
                       (nitems == 0 || (descriptor->valid_mask && !dsl_mask_any(descriptor->valid_mask, nitems)));
    if (empty_group && nitems == 0) nitems = 1; /* Scalar dependency storage, not a valid input lane. */

    bool jit_attempted = false;
    if (program->semantic_profile == ME_DSL_PROFILE_PORTABLE_1_1 &&
        program->jit_kernel_fn && !me_eval_jit_disabled(params)) {
        if (n_vars && !vars_block) return ME_EVAL_ERR_VAR_MISMATCH;
        const void *inputs[ME_MAX_VARS + 130];
        for (int i = 0; i < n_vars; i++) {
            if (nitems && !vars_block[i]) return ME_EVAL_ERR_VAR_MISMATCH;
            inputs[i] = vars_block[i];
        }
        inputs[n_vars] = descriptor ? descriptor->valid_mask : NULL;
        inputs[n_vars+1] = (const void *)dsl_portable_float_compare;
        for (int i = 0; i < program->portable_jit_ncomparisons; i++) {
            inputs[n_vars+2+i] = program->portable_jit_comparisons[i];
        }
        fenv_t saved;
        if (!dsl_portable_fp_begin(&saved)) return ME_EVAL_ERR_INVALID_ARG;
        int rc = program->jit_kernel_fn(inputs,output_block,(int64_t)nitems);
        if (!dsl_portable_fp_end(&saved)) return ME_EVAL_ERR_INVALID_ARG;
        return rc == 0 ? ME_EVAL_SUCCESS : ME_EVAL_ERR_INVALID_ARG;
    }
    int64_t current_cap = dsl_while_max_iters();
    if (current_cap < 0) current_cap = 0;
    bool jit_cap_matches = !program->jit_ir || !program->jit_ir->has_while ||
        program->jit_ir->while_max_iters == current_cap;
    /* JIT is best-effort for backend failures, not semantic execution errors. */
    if (!dsl_portable_typed_profile(program->semantic_profile) &&
        jit_cap_matches && !me_eval_jit_disabled(params) &&
        program->jit_kernel_fn &&
        program->jit_nparams >= 0 &&
        program->jit_nparams <= ME_MAX_VARS) {
        const void *jit_inputs_stack[ME_MAX_VARS];
        void *jit_temp_buffers[ME_MAX_VARS];
        int jit_temp_count = 0;
        const void **jit_inputs = (program->jit_nparams > 0) ? jit_inputs_stack : NULL;
        bool can_run_jit_direct = true;
        if (program->jit_nparams > 0) {
            if (!program->jit_param_bindings) {
                can_run_jit_direct = false;
            }
            else {
                for (int i = 0; i < program->jit_nparams; i++) {
                    const me_dsl_jit_param_binding *binding = &program->jit_param_bindings[i];
                    if (binding->kind == ME_DSL_JIT_BIND_USER_INPUT) {
                        int idx = binding->var_index;
                        if (!vars_block || idx < 0 || idx >= n_vars || !vars_block[idx]) {
                            can_run_jit_direct = false;
                            break;
                        }
                        jit_inputs_stack[i] = vars_block[idx];
                    }
                    else if ((program->jit_synth_reserved_non_nd || program->jit_synth_reserved_nd) &&
                             (binding->kind == ME_DSL_JIT_BIND_RESERVED_I ||
                              binding->kind == ME_DSL_JIT_BIND_RESERVED_N ||
                              binding->kind == ME_DSL_JIT_BIND_RESERVED_NDIM ||
                              binding->kind == ME_DSL_JIT_BIND_RESERVED_GLOBAL_LINEAR_IDX)) {
                        jit_inputs_stack[i] = NULL;
                    }
                    else if (binding->kind == ME_DSL_JIT_BIND_SYNTH_ND_CTX &&
                             program->jit_synth_reserved_nd &&
                             nd_synth_ctx_buffer) {
                        jit_inputs_stack[i] = (const void *)nd_synth_ctx_buffer;
                    }
                    else {
                        can_run_jit_direct = false;
                        break;
                    }
#if ME_USE_WASM32_JIT
                    if (jit_inputs_stack[i] &&
                        program->jit_ir &&
                        i < program->jit_ir->nparams &&
                        program->jit_ir->param_dtypes[i] == ME_INT64) {
                        size_t count = (size_t)nitems;
                        if (binding->kind == ME_DSL_JIT_BIND_SYNTH_ND_CTX) {
                            count = (size_t)ME_DSL_JIT_SYNTH_ND_CTX_WORDS;
                        }
                        int *tmp_i32 = malloc(count * sizeof(*tmp_i32));
                        if (!tmp_i32) {
                            can_run_jit_direct = false;
                            break;
                        }
                        const int64_t *src_i64 = (const int64_t *)jit_inputs_stack[i];
                        for (size_t j = 0; j < count; j++) {
                            tmp_i32[j] = (int)src_i64[j];
                        }
                        jit_inputs_stack[i] = (const void *)tmp_i32;
                        jit_temp_buffers[jit_temp_count++] = tmp_i32;
                    }
#endif
                }
            }
        }
        if (can_run_jit_direct) {
            void *jit_output = output_block;
            void *jit_output_tmp = NULL;
#if ME_USE_WASM32_JIT
            if (!dsl_wasm32_prepare_jit_output(program, output_block, nitems,
                                               &jit_output, &jit_output_tmp)) {
                can_run_jit_direct = false;
            }
#endif
            if (can_run_jit_direct) {
                int jit_rc = program->jit_kernel_fn(program->jit_nparams > 0 ? jit_inputs : NULL,
                                                    jit_output, (int64_t)nitems);
                bool jit_success = (jit_rc == 0);
                jit_attempted = true;
#if ME_USE_WASM32_JIT
                dsl_wasm32_finalize_jit_output(program, output_block, nitems,
                                               jit_output_tmp, jit_success);
                jit_output_tmp = NULL;
#endif
                for (int i = 0; i < jit_temp_count; i++) {
                    free(jit_temp_buffers[i]);
                }
                if (jit_success) {
                    return ME_EVAL_SUCCESS;
                }
                if (jit_rc == ME_DSL_JIT_MISSING_RETURN || jit_rc == ME_DSL_JIT_LOOP_CAP) {
                    return ME_EVAL_ERR_INVALID_ARG;
                }
            }
#if ME_USE_WASM32_JIT
            dsl_wasm32_finalize_jit_output(program, output_block, nitems,
                                           jit_output_tmp, false);
#endif
        }
        if (!can_run_jit_direct) {
            for (int i = 0; i < jit_temp_count; i++) {
                free(jit_temp_buffers[i]);
            }
        }
    }

    void **var_buffers = calloc(program->vars.count ? (size_t)program->vars.count : 1, sizeof(*var_buffers));
    void **local_buffers = calloc(program->n_locals ? (size_t)program->n_locals : 1, sizeof(*local_buffers));
    if (!var_buffers || !local_buffers) {
        free(var_buffers);
        free(local_buffers);
        return ME_EVAL_ERR_OOM;
    }

    for (int i = 0; i < program->n_inputs; i++) {
        var_buffers[i] = (void *)vars_block[i];
    }

    for (int i = 0; i < program->n_locals; i++) {
        int var_index = program->local_var_indices[i];
        size_t sz = dsl_var_item_size(program, var_index);
        if (sz == 0) {
            free(var_buffers);
            free(local_buffers);
            return ME_EVAL_ERR_INVALID_ARG;
        }
        local_buffers[i] = malloc((size_t)nitems * sz);
        if (!local_buffers[i]) {
            for (int j = 0; j < i; j++) {
                if (local_buffers[j]) {
                    free(local_buffers[j]);
                }
            }
            free(var_buffers);
            free(local_buffers);
            return ME_EVAL_ERR_OOM;
        }
        var_buffers[var_index] = local_buffers[i];
    }

    void *reserved_buffers[ME_DSL_MAX_NDIM * 2 + 2];
    int reserved_count = 0;
    bool reserved_ctx_error = false;
    const bool use_nd_ctx_reserved = program->jit_synth_reserved_nd &&
                                     nd_synth_ctx_buffer &&
                                     ndim > 0;
    int64_t *synth_idx_outputs[ME_DSL_MAX_NDIM];
    memset(synth_idx_outputs, 0, sizeof(synth_idx_outputs));
    uint32_t synth_idx_mask = 0;
    int64_t *synth_global_output = NULL;
    if (program->uses_ndim && program->idx_ndim >= 0) {
        int64_t *buf = malloc((size_t)nitems * sizeof(int64_t));
        if (!buf) {
            reserved_count = -1;
        }
        else {
            dsl_fill_i64(buf, nitems, (int64_t)ndim);
            var_buffers[program->idx_ndim] = buf;
            reserved_buffers[reserved_count++] = buf;
        }
    }
    for (int d = 0; d < ME_DSL_MAX_NDIM && reserved_count >= 0; d++) {
        if ((program->uses_n_mask & (1 << d)) && program->idx_n[d] >= 0) {
            int64_t *buf = malloc((size_t)nitems * sizeof(int64_t));
            if (!buf) {
                reserved_count = -1;
                break;
            }
            int64_t val = 1;
            if (shape && d < ndim) {
                val = shape[d];
            }
            else if (d == 0) {
                val = nitems;
            }
            dsl_fill_i64(buf, nitems, val);
            var_buffers[program->idx_n[d]] = buf;
            reserved_buffers[reserved_count++] = buf;
        }
        if ((program->uses_i_mask & (1 << d)) && program->idx_i[d] >= 0) {
            if (idx_buffers && idx_buffers[d]) {
                var_buffers[program->idx_i[d]] = idx_buffers[d];
            }
            else {
                int64_t *buf = malloc((size_t)nitems * sizeof(int64_t));
                if (!buf) {
                    reserved_count = -1;
                    break;
                }
                var_buffers[program->idx_i[d]] = buf;
                reserved_buffers[reserved_count++] = buf;
                if (use_nd_ctx_reserved) {
                    synth_idx_outputs[d] = buf;
                    synth_idx_mask |= (1u << d);
                }
                else if (d == 0) {
                    dsl_fill_iota_i64(buf, nitems, 0);
                }
                else {
                    dsl_fill_i64(buf, nitems, 0);
                }
            }
        }
    }
    if (program->uses_flat_idx && program->idx_flat_idx >= 0 && reserved_count >= 0) {
        if (global_linear_idx_buffer) {
            var_buffers[program->idx_flat_idx] = (void *)global_linear_idx_buffer;
        }
        else {
            int64_t *buf = malloc((size_t)nitems * sizeof(int64_t));
            if (!buf) {
                reserved_count = -1;
            }
            else {
                var_buffers[program->idx_flat_idx] = buf;
                reserved_buffers[reserved_count++] = buf;
                if (use_nd_ctx_reserved) {
                    synth_global_output = buf;
                }
                else {
                    dsl_fill_iota_i64(buf, nitems, 0);
                }
            }
        }
    }

    if (reserved_count >= 0 &&
        use_nd_ctx_reserved &&
        (synth_idx_mask != 0 || synth_global_output != NULL)) {
        if (!dsl_fill_reserved_from_nd_synth_ctx(nitems, nd_synth_ctx_buffer,
                                                 synth_idx_outputs, synth_idx_mask,
                                                 synth_global_output)) {
            reserved_ctx_error = true;
        }
    }

    if (reserved_count < 0 || reserved_ctx_error) {
        for (int i = 0; i < reserved_count; i++) {
            free(reserved_buffers[i]);
        }
        for (int i = 0; i < program->n_locals; i++) {
            if (local_buffers[i]) {
                free(local_buffers[i]);
            }
        }
        free(var_buffers);
        free(local_buffers);
        return reserved_ctx_error ? ME_EVAL_ERR_INVALID_ARG : ME_EVAL_ERR_OOM;
    }

    if (!dsl_portable_typed_profile(program->semantic_profile) &&
        jit_cap_matches && !jit_attempted &&
        !me_eval_jit_disabled(params) &&
        program->jit_kernel_fn &&
        program->jit_nparams >= 0 &&
        program->jit_nparams <= ME_MAX_VARS) {
        const void *jit_inputs_stack[ME_MAX_VARS];
        void *jit_temp_buffers[ME_MAX_VARS];
        int jit_temp_count = 0;
        const void **jit_inputs = jit_inputs_stack;
        bool can_run_jit = true;
        if (program->jit_nparams > 0) {
            if (!program->jit_param_bindings) {
                can_run_jit = false;
            }
            else {
                for (int i = 0; i < program->jit_nparams; i++) {
                    const me_dsl_jit_param_binding *binding = &program->jit_param_bindings[i];
                    int idx = binding->var_index;
                    bool synth_reserved = false;
                    switch (binding->kind) {
                    case ME_DSL_JIT_BIND_USER_INPUT:
                        break;
                    case ME_DSL_JIT_BIND_RESERVED_NDIM:
                    case ME_DSL_JIT_BIND_RESERVED_GLOBAL_LINEAR_IDX:
                        synth_reserved = program->jit_synth_reserved_non_nd ||
                                         program->jit_synth_reserved_nd;
                        break;
                    case ME_DSL_JIT_BIND_RESERVED_I:
                    case ME_DSL_JIT_BIND_RESERVED_N:
                        if (binding->dim < 0 || binding->dim >= ME_DSL_MAX_NDIM) {
                            can_run_jit = false;
                        }
                        synth_reserved = program->jit_synth_reserved_non_nd ||
                                         program->jit_synth_reserved_nd;
                        break;
                    case ME_DSL_JIT_BIND_SYNTH_ND_CTX:
                        if (!program->jit_synth_reserved_nd || !nd_synth_ctx_buffer) {
                            can_run_jit = false;
                        }
                        break;
                    default:
                        can_run_jit = false;
                        break;
                    }
                    if (!can_run_jit) {
                        break;
                    }
                    if (synth_reserved) {
                        jit_inputs_stack[i] = NULL;
                    }
                    else if (binding->kind == ME_DSL_JIT_BIND_SYNTH_ND_CTX) {
                        jit_inputs_stack[i] = (const void *)nd_synth_ctx_buffer;
                    }
                    else {
                        if (idx < 0 || idx >= program->vars.count || !var_buffers[idx]) {
                            can_run_jit = false;
                            break;
                        }
                        jit_inputs_stack[i] = (const void *)var_buffers[idx];
                    }
#if ME_USE_WASM32_JIT
                    if (jit_inputs_stack[i] &&
                        program->jit_ir &&
                        i < program->jit_ir->nparams &&
                        program->jit_ir->param_dtypes[i] == ME_INT64) {
                        size_t count = (size_t)nitems;
                        if (binding->kind == ME_DSL_JIT_BIND_SYNTH_ND_CTX) {
                            count = (size_t)ME_DSL_JIT_SYNTH_ND_CTX_WORDS;
                        }
                        int *tmp_i32 = malloc(count * sizeof(*tmp_i32));
                        if (!tmp_i32) {
                            can_run_jit = false;
                            break;
                        }
                        const int64_t *src_i64 = (const int64_t *)jit_inputs_stack[i];
                        for (size_t j = 0; j < count; j++) {
                            tmp_i32[j] = (int)src_i64[j];
                        }
                        jit_inputs_stack[i] = (const void *)tmp_i32;
                        jit_temp_buffers[jit_temp_count++] = tmp_i32;
                    }
#endif
                }
            }
        }
        if (can_run_jit) {
            void *jit_output = output_block;
            void *jit_output_tmp = NULL;
#if ME_USE_WASM32_JIT
            if (!dsl_wasm32_prepare_jit_output(program, output_block, nitems,
                                               &jit_output, &jit_output_tmp)) {
                can_run_jit = false;
            }
#endif
            if (can_run_jit) {
                int jit_rc = program->jit_kernel_fn(program->jit_nparams > 0 ? jit_inputs : NULL,
                                                    jit_output, (int64_t)nitems);
                bool jit_success = (jit_rc == 0);
#if ME_USE_WASM32_JIT
                dsl_wasm32_finalize_jit_output(program, output_block, nitems,
                                               jit_output_tmp, jit_success);
                jit_output_tmp = NULL;
#endif
                for (int i = 0; i < jit_temp_count; i++) {
                    free(jit_temp_buffers[i]);
                }
                if (jit_success || jit_rc == ME_DSL_JIT_MISSING_RETURN || jit_rc == ME_DSL_JIT_LOOP_CAP) {
                    for (int i = 0; i < reserved_count; i++) {
                        free(reserved_buffers[i]);
                    }
                    for (int i = 0; i < program->n_locals; i++) {
                        if (local_buffers[i]) {
                            free(local_buffers[i]);
                        }
                    }
                    free(var_buffers);
                    free(local_buffers);
                    return jit_success ? ME_EVAL_SUCCESS : ME_EVAL_ERR_INVALID_ARG;
                }
            }
#if ME_USE_WASM32_JIT
            dsl_wasm32_finalize_jit_output(program, output_block, nitems,
                                           jit_output_tmp, false);
#endif
        }
        if (!can_run_jit) {
            for (int i = 0; i < jit_temp_count; i++) {
                free(jit_temp_buffers[i]);
            }
        }
    }

    dsl_eval_ctx ctx;
    ctx.program = program;
    ctx.var_buffers = var_buffers;
    ctx.local_buffers = local_buffers;
    ctx.nitems = nitems;
    ctx.params = params;
    ctx.output_block = output_block;
    ctx.valid_mask = descriptor ? descriptor->valid_mask : NULL;
    ctx.group_nitems = group_nitems;
    ctx.scalar_output = descriptor && program->output_is_scalar;
    ctx.empty_group = empty_group;
    if (empty_group) ctx.nitems = 1;
    ctx.initialized = NULL;
    uint8_t *defined_storage = NULL;
    int rc = ME_EVAL_SUCCESS;
    if (dsl_portable_typed_profile(program->semantic_profile)) {
        size_t storage_count = program->n_locals ? (size_t)program->n_locals : 1;
        size_t lane_count = nitems ? (size_t)nitems : 1;
        ctx.initialized = calloc(program->vars.count ? (size_t)program->vars.count : 1, sizeof(*ctx.initialized));
        if (storage_count > SIZE_MAX / lane_count) rc = ME_EVAL_ERR_OOM;
        else defined_storage = calloc(storage_count * lane_count, 1);
        if (!ctx.initialized || !defined_storage) rc = ME_EVAL_ERR_OOM;
        if (rc == ME_EVAL_SUCCESS) {
            for (int i = 0; i < program->n_locals; i++) ctx.initialized[program->local_var_indices[i]] = defined_storage + (size_t)i * lane_count;
        }
    }

    bool did_break = false;
    bool did_continue = false;
    bool did_return = false;
    if (rc == ME_EVAL_SUCCESS) rc = dsl_eval_block(&ctx, &program->block, &did_break, &did_continue, &did_return);
    free(defined_storage);
    free(ctx.initialized);

    for (int i = 0; i < reserved_count; i++) {
        free(reserved_buffers[i]);
    }
    for (int i = 0; i < program->n_locals; i++) {
        if (local_buffers[i]) {
            free(local_buffers[i]);
        }
    }
    free(var_buffers);
    free(local_buffers);

    if (rc == ME_EVAL_SUCCESS && !did_return) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    return rc;
}

int dsl_eval_program(const me_dsl_compiled_program *program,
                      const void **vars_block, int n_vars, void *output_block, int nitems,
                      const me_eval_params *params, int ndim, const int64_t *shape,
                      int64_t **idx_buffers, const int64_t *global_linear_idx_buffer,
                      const int64_t *nd_synth_ctx_buffer) {
    return dsl_eval_program_impl(program, vars_block, n_vars, output_block, nitems, params,
                                 ndim, shape, idx_buffers, global_linear_idx_buffer, nd_synth_ctx_buffer, NULL);
}

int dsl_eval_program_portable(const me_dsl_compiled_program *program,
                               const void **inputs, int ninputs, void *output,
                               const me_dsl_portable_eval_descriptor *descriptor) {
    if (!program || !dsl_portable_typed_profile(program->semantic_profile) || !descriptor ||
        descriptor->nitems < 0 || ninputs != program->n_inputs || (ninputs && !inputs)) return ME_EVAL_ERR_INVALID_ARG;
    size_t count = program->output_is_scalar ? 1 : (size_t)descriptor->nitems;
    size_t itemsize = program->output_itemsize;
    if (!itemsize || count > SIZE_MAX / itemsize || descriptor->output_capacity < count * itemsize ||
        (count && !output)) return ME_EVAL_ERR_INVALID_ARG;
    for (int i = 0; i < descriptor->nitems; i++) {
        if (descriptor->valid_mask && descriptor->valid_mask[i] > 1) return ME_EVAL_ERR_INVALID_ARG;
    }
    int ndim = descriptor->ndim;
    bool needs_context = program->uses_i_mask || program->uses_n_mask || program->uses_ndim || program->uses_flat_idx;
    if (ndim < 0 || ndim > ME_DSL_MAX_NDIM || (needs_context && ndim != program->compile_ndims) ||
        (ndim && (!descriptor->logical_shape || !descriptor->block_origin || !descriptor->block_extent)) ||
        (!ndim && (needs_context || descriptor->logical_shape || descriptor->block_origin || descriptor->block_extent))) return ME_EVAL_ERR_INVALID_ARG;
    int64_t *indices[ME_DSL_MAX_NDIM] = {0};
    int64_t *flat = NULL;
    int64_t strides[ME_DSL_MAX_NDIM];
    size_t block_count = 1;
    int64_t shape_count = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        int64_t size = descriptor->logical_shape[d], origin = descriptor->block_origin[d], extent = descriptor->block_extent[d];
        if (size < 0 || origin < 0 || origin > size || extent < 0 ||
            (size && shape_count > INT64_MAX / size) ||
            (extent && block_count > SIZE_MAX / (uint64_t)extent)) return ME_EVAL_ERR_INVALID_ARG;
        strides[d] = shape_count;
        shape_count *= size;
        block_count *= (size_t)extent;
    }
    if (ndim && block_count != (size_t)descriptor->nitems) return ME_EVAL_ERR_INVALID_ARG;
    if (!count) return ME_EVAL_SUCCESS;
    size_t allocated_lanes = descriptor->nitems ? (size_t)descriptor->nitems : 1;
    if (allocated_lanes > SIZE_MAX / sizeof(int64_t)) return ME_EVAL_ERR_INVALID_ARG;
    int rc = ME_EVAL_SUCCESS;
    for (int d = 0; d < ndim; d++) {
        if (program->uses_i_mask & (1 << d)) {
            indices[d] = calloc(allocated_lanes, sizeof(int64_t));
            if (!indices[d]) rc = ME_EVAL_ERR_OOM;
        }
    }
    if (program->uses_flat_idx) {
        flat = calloc(allocated_lanes, sizeof(int64_t));
        if (!flat) rc = ME_EVAL_ERR_OOM;
    }
    for (int lane = 0; rc == ME_EVAL_SUCCESS && lane < descriptor->nitems && ndim; lane++) {
        size_t offset = (size_t)lane;
        bool valid = !descriptor->valid_mask || descriptor->valid_mask[lane];
        int64_t linear = 0;
        for (int d = ndim - 1; d >= 0; d--) {
            size_t extent = (size_t)descriptor->block_extent[d];
            size_t relative = offset % extent;
            offset /= extent;
            int64_t remaining = descriptor->logical_shape[d] - descriptor->block_origin[d];
            if (relative >= (uint64_t)remaining) {
                if (valid) rc = ME_EVAL_ERR_INVALID_ARG;
                continue;
            }
            int64_t coordinate = descriptor->block_origin[d] + (int64_t)relative;
            if (indices[d]) indices[d][lane] = coordinate;
            linear += coordinate * strides[d]; /* Validated shape product fits int64. */
        }
        if (flat) flat[lane] = linear;
    }
    if (rc == ME_EVAL_SUCCESS) rc = dsl_eval_program_impl(program, inputs, ninputs, output, descriptor->nitems, NULL,
        ndim, descriptor->logical_shape, indices, flat, NULL, descriptor);
    for (int d = 0; d < ndim; d++) free(indices[d]);
    free(flat);
    return rc;
}

/* Fill missing reserved index buffers from ND synth context.
   This is used only as a fallback path when ND synth JIT is enabled but direct
   JIT execution does not run and interpreter buffers are still needed. */
static bool dsl_fill_reserved_from_nd_synth_ctx(int nitems,
                                                const int64_t *nd_ctx,
                                                int64_t **idx_outputs,
                                                uint32_t idx_mask,
                                                int64_t *global_output) {
    if (nitems < 0 || !nd_ctx) {
        return false;
    }
    int nd = (int)nd_ctx[0];
    if (nd <= 0 || nd > ME_DSL_MAX_NDIM) {
        return false;
    }
    const int64_t *shape_stride = nd_ctx + 1 + nd;
    const int64_t *base_idx = nd_ctx + 1 + 2 * nd;
    const int64_t *iter_len = nd_ctx + 1 + 3 * nd;
    const int64_t tail = 1 + 4 * nd;
    const bool has_v2_tail = (nd_ctx[tail] >= (int64_t)ME_DSL_JIT_SYNTH_ND_CTX_V2_VERSION);
    const bool seq = has_v2_tail && ((nd_ctx[tail + 1] & (int64_t)ME_DSL_ND_CTX_FLAG_SEQ) != 0);

    for (int d = nd; d < ME_DSL_MAX_NDIM; d++) {
        if ((idx_mask & (1u << d)) && idx_outputs && idx_outputs[d]) {
            dsl_fill_i64(idx_outputs[d], nitems, 0);
        }
    }

    if (seq && idx_mask == 0 && global_output) {
        int64_t glin = nd_ctx[tail + 2];
        for (int64_t it = 0; it < nitems; it++) {
            global_output[it] = glin;
            glin = dsl_i64_add_wrap(glin, 1);
        }
        return true;
    }

    int64_t indices[ME_DSL_MAX_NDIM];
    memset(indices, 0, sizeof(indices));
    for (int64_t it = 0; it < nitems; it++) {
        int64_t global_idx = 0;
        for (int d = 0; d < nd; d++) {
            int64_t coord = dsl_i64_add_wrap(base_idx[d], indices[d]);
            if ((idx_mask & (1u << d)) && idx_outputs && idx_outputs[d]) {
                idx_outputs[d][it] = coord;
            }
            if (global_output) {
                global_idx = dsl_i64_addmul_wrap(global_idx, coord, shape_stride[d]);
            }
        }
        if (global_output) {
            global_output[it] = global_idx;
        }
        for (int d = nd - 1; d >= 0; d--) {
            int64_t len = iter_len[d];
            if (len <= 0) {
                indices[d] = 0;
                continue;
            }
            indices[d]++;
            if (indices[d] < len) {
                break;
            }
            indices[d] = 0;
        }
    }
    return true;
}
