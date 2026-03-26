/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_compile_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#endif

static bool dsl_var_table_grow(me_dsl_var_table *table, int min_cap) {
    if (table->capacity >= min_cap) {
        return true;
    }
    int new_cap = table->capacity ? table->capacity * 2 : 16;
    if (new_cap < min_cap) {
        new_cap = min_cap;
    }
    char **names = malloc((size_t)new_cap * sizeof(*names));
    me_dtype *dtypes = malloc((size_t)new_cap * sizeof(*dtypes));
    size_t *itemsizes = malloc((size_t)new_cap * sizeof(*itemsizes));
    bool *uniform = malloc((size_t)new_cap * sizeof(*uniform));
    if (!names || !dtypes || !itemsizes || !uniform) {
        free(names);
        free(dtypes);
        free(itemsizes);
        free(uniform);
        return false;
    }
    if (table->count > 0) {
        memcpy(names, table->names, (size_t)table->count * sizeof(*names));
        memcpy(dtypes, table->dtypes, (size_t)table->count * sizeof(*dtypes));
        memcpy(itemsizes, table->itemsizes, (size_t)table->count * sizeof(*itemsizes));
        memcpy(uniform, table->uniform, (size_t)table->count * sizeof(*uniform));
    }
    free(table->names);
    free(table->dtypes);
    free(table->itemsizes);
    free(table->uniform);
    table->names = names;
    table->dtypes = dtypes;
    table->itemsizes = itemsizes;
    table->uniform = uniform;
    table->capacity = new_cap;
    return true;
}

static char *me_strdup(const char *s) {
#if defined(_MSC_VER)
    return _strdup(s);
#else
    return strdup(s);
#endif
}

static bool dsl_source_has_fp_pragma(const char *source) {
    if (!source) {
        return false;
    }
    const char *line = source;
    while (*line) {
        const char *line_end = line;
        while (*line_end && *line_end != '\n') {
            line_end++;
        }
        const char *p = line;
        while (p < line_end && (*p == ' ' || *p == '\t' || *p == '\r')) {
            p++;
        }
        if (p == line_end) {
            if (*line_end == '\0') {
                break;
            }
            line = line_end + 1;
            continue;
        }
        if (*p != '#') {
            break;
        }
        p++;
        while (p < line_end && isspace((unsigned char)*p)) {
            p++;
        }
        if ((size_t)(line_end - p) >= 5 && strncmp(p, "me:fp", 5) == 0) {
            return true;
        }
        if (*line_end == '\0') {
            break;
        }
        line = line_end + 1;
    }
    return false;
}

static void dsl_compiled_block_free(me_dsl_compiled_block *block);

void dsl_compiled_expr_free(me_dsl_compiled_expr *expr) {
    if (!expr) {
        return;
    }
    if (expr->expr) {
        me_free(expr->expr);
        expr->expr = NULL;
    }
    free(expr->var_indices);
    expr->var_indices = NULL;
    expr->n_vars = 0;
}

void dsl_compiled_stmt_free(me_dsl_compiled_stmt *stmt) {
    if (!stmt) {
        return;
    }
    switch (stmt->kind) {
    case ME_DSL_STMT_ASSIGN:
        dsl_compiled_expr_free(&stmt->as.assign.value);
        break;
    case ME_DSL_STMT_EXPR:
        dsl_compiled_expr_free(&stmt->as.expr_stmt.expr);
        break;
    case ME_DSL_STMT_RETURN:
        dsl_compiled_expr_free(&stmt->as.return_stmt.expr);
        break;
    case ME_DSL_STMT_PRINT:
        free(stmt->as.print_stmt.format);
        if (stmt->as.print_stmt.args) {
            for (int i = 0; i < stmt->as.print_stmt.nargs; i++) {
                dsl_compiled_expr_free(&stmt->as.print_stmt.args[i]);
            }
        }
        free(stmt->as.print_stmt.args);
        break;
    case ME_DSL_STMT_IF:
        dsl_compiled_expr_free(&stmt->as.if_stmt.cond);
        dsl_compiled_block_free(&stmt->as.if_stmt.then_block);
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            dsl_compiled_expr_free(&stmt->as.if_stmt.elif_branches[i].cond);
            dsl_compiled_block_free(&stmt->as.if_stmt.elif_branches[i].block);
        }
        free(stmt->as.if_stmt.elif_branches);
        if (stmt->as.if_stmt.has_else) {
            dsl_compiled_block_free(&stmt->as.if_stmt.else_block);
        }
        break;
    case ME_DSL_STMT_WHILE:
        dsl_compiled_expr_free(&stmt->as.while_loop.cond);
        dsl_compiled_block_free(&stmt->as.while_loop.body);
        break;
    case ME_DSL_STMT_FOR:
        dsl_compiled_expr_free(&stmt->as.for_loop.start);
        dsl_compiled_expr_free(&stmt->as.for_loop.stop);
        dsl_compiled_expr_free(&stmt->as.for_loop.step);
        dsl_compiled_block_free(&stmt->as.for_loop.body);
        break;
    case ME_DSL_STMT_BREAK:
    case ME_DSL_STMT_CONTINUE:
        dsl_compiled_expr_free(&stmt->as.flow.cond);
        break;
    }
    free(stmt);
}

static void dsl_compiled_block_free(me_dsl_compiled_block *block) {
    if (!block) {
        return;
    }
    for (int i = 0; i < block->nstmts; i++) {
        dsl_compiled_stmt_free(block->stmts[i]);
    }
    free(block->stmts);
    block->stmts = NULL;
    block->nstmts = 0;
    block->capacity = 0;
}

void dsl_var_table_init(me_dsl_var_table *table) {
    memset(table, 0, sizeof(*table));
}

void dsl_var_table_free(me_dsl_var_table *table) {
    if (!table) {
        return;
    }
    for (int i = 0; i < table->count; i++) {
        free(table->names[i]);
    }
    free(table->names);
    free(table->dtypes);
    free(table->itemsizes);
    free(table->uniform);
    table->names = NULL;
    table->dtypes = NULL;
    table->itemsizes = NULL;
    table->uniform = NULL;
    table->count = 0;
    table->capacity = 0;
}

int dsl_var_table_find(const me_dsl_var_table *table, const char *name) {
    if (!table || !name) {
        return -1;
    }
    for (int i = 0; i < table->count; i++) {
        if (strcmp(table->names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

int dsl_var_table_add_with_uniform(me_dsl_var_table *table, const char *name, me_dtype dtype,
                                   size_t itemsize, bool uniform) {
    if (!table || !name) {
        return -1;
    }
    if (table->count >= ME_MAX_VARS) {
        return -1;
    }
    if (!dsl_var_table_grow(table, table->count + 1)) {
        return -1;
    }
    table->names[table->count] = me_strdup(name);
    if (!table->names[table->count]) {
        return -1;
    }
    table->dtypes[table->count] = dtype;
    table->itemsizes[table->count] = itemsize;
    table->uniform[table->count] = uniform;
    table->count++;
    return table->count - 1;
}

int dsl_var_table_add(me_dsl_var_table *table, const char *name, me_dtype dtype) {
    return dsl_var_table_add_with_uniform(table, name, dtype, 0, false);
}

me_dsl_compiled_program *dsl_compiled_program_alloc(const me_dsl_program *parsed,
                                                    const char *source,
                                                    int compile_ndims,
                                                    char *error_reason,
                                                    size_t error_reason_cap) {
    if (!parsed) {
        return NULL;
    }

    me_dsl_compiled_program *program = calloc(1, sizeof(*program));
    if (!program) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap, "out of memory while allocating DSL program");
        }
        return NULL;
    }
    program->fp_mode = parsed->fp_mode;
    if (!dsl_source_has_fp_pragma(source)) {
        program->fp_mode = dsl_default_fp_mode_from_env();
    }
    program->compiler = parsed->compiler;
    program->compile_ndims = compile_ndims;
    dsl_var_table_init(&program->vars);
    program->idx_ndim = -1;
    program->idx_flat_idx = -1;
    for (int i = 0; i < ME_DSL_MAX_NDIM; i++) {
        program->idx_i[i] = -1;
        program->idx_n[i] = -1;
    }
    program->local_slots = malloc(ME_MAX_VARS * sizeof(*program->local_slots));
    if (!program->local_slots) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap, "out of memory while allocating DSL locals");
        }
        dsl_compiled_program_free(program);
        return NULL;
    }
    for (int i = 0; i < ME_MAX_VARS; i++) {
        program->local_slots[i] = -1;
    }
    return program;
}

void dsl_compiled_program_free(me_dsl_compiled_program *program) {
    if (!program) {
        return;
    }
#if ME_USE_LIBTCC_FALLBACK
    if (program->jit_tcc_state) {
        dsl_jit_libtcc_delete_state(program->jit_tcc_state);
        program->jit_tcc_state = NULL;
    }
#endif
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
    if (program->jit_dl_handle) {
        if (!program->jit_dl_handle_cached) {
            dlclose(program->jit_dl_handle);
        }
        program->jit_dl_handle = NULL;
    }
#endif
#if ME_USE_WASM32_JIT
    if (program->jit_kernel_fn && !program->jit_dl_handle_cached) {
        dsl_wasm_jit_free_dispatch((int)(uintptr_t)program->jit_kernel_fn);
    }
    /* jit_dl_handle holds the wasm32 JIT scratch memory. */
    free(program->jit_dl_handle);
    program->jit_dl_handle = NULL;
#endif
    program->jit_kernel_fn = NULL;
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    free(program->jit_c_source);
    free(program->jit_param_bindings);
    me_dsl_jit_ir_free(program->jit_ir);
    dsl_compiled_block_free(&program->block);
    dsl_var_table_free(&program->vars);
    free(program->local_var_indices);
    free(program->local_slots);
    free(program);
}

bool dsl_compiled_block_push(me_dsl_compiled_block *block, me_dsl_compiled_stmt *stmt) {
    if (!block || !stmt) {
        return false;
    }
    if (block->nstmts == block->capacity) {
        int new_cap = block->capacity ? block->capacity * 2 : 8;
        me_dsl_compiled_stmt **next = realloc(block->stmts, (size_t)new_cap * sizeof(*next));
        if (!next) {
            return false;
        }
        block->stmts = next;
        block->capacity = new_cap;
    }
    block->stmts[block->nstmts++] = stmt;
    return true;
}

bool dsl_build_var_lookup(const me_dsl_var_table *table, const me_variable *funcs,
                          int n_funcs, me_variable **out_vars, int *out_count) {
    if (!table || !out_vars || !out_count || n_funcs < 0) {
        return false;
    }
    int total = table->count + n_funcs;
    if (total == 0) {
        *out_vars = NULL;
        *out_count = 0;
        return true;
    }
    me_variable *vars = calloc((size_t)total, sizeof(*vars));
    if (!vars) {
        return false;
    }
    for (int i = 0; i < table->count; i++) {
        vars[i].name = table->names[i];
        vars[i].dtype = table->dtypes[i];
        vars[i].address = &synthetic_var_addresses[i];
        vars[i].type = ME_VARIABLE;
        vars[i].context = NULL;
        vars[i].itemsize = table->itemsizes ? table->itemsizes[i] : 0;
    }
    for (int i = 0; i < n_funcs; i++) {
        vars[table->count + i] = funcs[i];
    }
    *out_vars = vars;
    *out_count = total;
    return true;
}

bool dsl_program_add_local(me_dsl_compiled_program *program, int var_index) {
    if (!program || var_index < 0 || var_index >= ME_MAX_VARS) {
        return false;
    }
    if (program->local_slots[var_index] >= 0) {
        return true;
    }
    int slot = program->n_locals;
    int *local_var_indices = realloc(program->local_var_indices,
                                     (size_t)(slot + 1) * sizeof(*local_var_indices));
    if (!local_var_indices) {
        return false;
    }
    program->local_var_indices = local_var_indices;
    program->local_var_indices[slot] = var_index;
    program->local_slots[var_index] = slot;
    program->n_locals++;
    return true;
}
