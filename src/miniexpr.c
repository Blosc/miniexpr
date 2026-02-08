/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

// Loosely based on https://github.com/CodePlea/tinyexpr. License follows:
// SPDX-License-Identifier: Zlib
/*
 * TINYEXPR - Tiny recursive descent parser and evaluation engine in C
 *
 * Copyright (c) 2015-2020 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgement in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

/* COMPILE TIME OPTIONS */

/* Exponentiation associativity:
For a**b**c = (a**b)**c and -a**b = (-a)**b do nothing.
For a**b**c = a**(b**c) and -a**b = -(a**b) uncomment the next line.*/
/* #define ME_POW_FROM_RIGHT */

/* Logarithms
For log = natural log do nothing (NumPy compatible)
For log = base 10 log comment the next line. */
#define ME_NAT_LOG

#include "functions.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>

#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

#include "dsl_parser.h"
#include "dsl_jit_ir.h"
#include "dsl_jit_cgen.h"

#define ME_DSL_MAX_NDIM 8
#define ME_DSL_JIT_SYMBOL_NAME "me_dsl_jit_kernel"
#define ME_DSL_JIT_CGEN_VERSION 1
#ifndef ME_USE_LIBTCC_FALLBACK
#define ME_USE_LIBTCC_FALLBACK 0
#endif
#ifndef ME_DSL_JIT_LIBTCC_DEFAULT_PATH
#define ME_DSL_JIT_LIBTCC_DEFAULT_PATH ""
#endif

typedef int (*me_dsl_jit_kernel_fn)(const void **inputs, void *output, int64_t nitems);

/* ND metadata attached to compiled expressions (used by me_eval_nd). */
typedef struct {
    int ndims;
    /* Layout: shape[ndims], chunkshape[ndims], blockshape[ndims] (all int64_t). */
    int64_t data[1];
} me_nd_info;

static float me_crealf(float _Complex v);
static float me_cimagf(float _Complex v);
static double me_creal(double _Complex v);
static double me_cimag(double _Complex v);

static int private_compile_ex(const char* expression, const me_variable_ex* variables, int var_count,
                              void* output, int nitems, me_dtype dtype, int* error, me_expr** out);

static bool is_function_entry(const me_variable_ex *var) {
    if (!var) {
        return false;
    }
    return IS_FUNCTION(var->type) || IS_CLOSURE(var->type);
}

static bool is_variable_entry(const me_variable_ex *var) {
    if (!var) {
        return false;
    }
    if (var->type == 0) {
        return true;
    }
    return TYPE_MASK(var->type) == ME_VARIABLE;
}

typedef struct {
    me_expr *expr;
    int *var_indices;
    int n_vars;
} me_dsl_compiled_expr;

typedef struct me_dsl_compiled_stmt me_dsl_compiled_stmt;

typedef struct {
    me_dsl_compiled_stmt **stmts;
    int nstmts;
    int capacity;
} me_dsl_compiled_block;

typedef struct {
    me_dsl_compiled_expr cond;
    me_dsl_compiled_block block;
} me_dsl_compiled_if_branch;

struct me_dsl_compiled_stmt {
    me_dsl_stmt_kind kind;
    int line;
    int column;
    union {
        struct {
            int local_slot;
            me_dsl_compiled_expr value;
        } assign;
        struct {
            me_dsl_compiled_expr expr;
        } expr_stmt;
        struct {
            me_dsl_compiled_expr expr;
        } return_stmt;
        struct {
            char *format;
            me_dsl_compiled_expr *args;
            int nargs;
        } print_stmt;
        struct {
            me_dsl_compiled_expr cond;
            me_dsl_compiled_block then_block;
            me_dsl_compiled_if_branch *elif_branches;
            int n_elifs;
            int elif_capacity;
            me_dsl_compiled_block else_block;
            bool has_else;
        } if_stmt;
        struct {
            int loop_var_slot;
            me_dsl_compiled_expr limit;
            me_dsl_compiled_block body;
        } for_loop;
        struct {
            me_dsl_compiled_expr cond;
        } flow;
    } as;
};

typedef struct {
    char **names;
    me_dtype *dtypes;
    size_t *itemsizes;
    bool *uniform;
    int count;
    int capacity;
} me_dsl_var_table;

typedef struct {
    me_dsl_compiled_block block;
    me_dsl_var_table vars;
    int n_inputs;
    int n_locals;
    int *local_var_indices;
    int *local_slots;
    int idx_ndim;
    int idx_i[ME_DSL_MAX_NDIM];
    int idx_n[ME_DSL_MAX_NDIM];
    int uses_i_mask;
    int uses_n_mask;
    bool uses_ndim;
    me_dsl_dialect dialect;
    me_dsl_fp_mode fp_mode;
    bool output_is_scalar;
    me_dtype output_dtype;
    me_dsl_jit_ir_program *jit_ir;
    uint64_t jit_ir_fingerprint;
    int jit_ir_error_line;
    int jit_ir_error_column;
    char jit_ir_error[128];
    char *jit_c_source;
    int jit_c_error_line;
    int jit_c_error_column;
    char jit_c_error[128];
    int *jit_param_input_indices;
    int jit_nparams;
    me_dsl_jit_kernel_fn jit_kernel_fn;
    void *jit_dl_handle;
    void *jit_tcc_state;
    uint64_t jit_runtime_key;
    bool jit_dl_handle_cached;
} me_dsl_compiled_program;

#if !defined(_WIN32) && !defined(_WIN64)
static void dsl_jit_libtcc_delete_state(void *state);
static bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program);
static const char *dsl_jit_libtcc_error_message(void);
static bool dsl_jit_c_compiler_available(void);
#endif

static int64_t ceil_div64(int64_t a, int64_t b) {
    return (b == 0) ? 0 : (a + b - 1) / b;
}

#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

/* Type promotion table following NumPy rules */
/* Note: ME_AUTO (0) should never appear in type promotion, so we index from 1 */
static const me_dtype type_promotion_table[13][13] = {
    /* Rows: left operand, Columns: right operand */
    /* BOOL,  INT8,    INT16,   INT32,   INT64,   UINT8,   UINT16,  UINT32,  UINT64,  FLOAT32, FLOAT64, COMPLEX64, COMPLEX128 */
    {
        ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* BOOL */
    {
        ME_INT8, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* INT8 */
    {
        ME_INT16, ME_INT16, ME_INT16, ME_INT32, ME_INT64, ME_INT32, ME_INT32, ME_INT64, ME_FLOAT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* INT16 */
    {
        ME_INT32, ME_INT32, ME_INT32, ME_INT32, ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* INT32 */
    {
        ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* INT64 */
    {
        ME_UINT8, ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* UINT8 */
    {
        ME_UINT16, ME_INT32, ME_INT32, ME_INT64, ME_FLOAT64, ME_UINT16, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32,
        ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* UINT16 */
    {
        ME_UINT32, ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64, ME_UINT32, ME_UINT32, ME_UINT32, ME_UINT64, ME_FLOAT64,
        ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* UINT32 */
    {
        ME_UINT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_UINT64, ME_UINT64, ME_UINT64, ME_UINT64,
        ME_FLOAT64, ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* UINT64 */
    {
        ME_FLOAT32, ME_FLOAT32, ME_FLOAT32, ME_FLOAT64, ME_FLOAT64, ME_FLOAT32, ME_FLOAT32, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT32, ME_FLOAT64, ME_COMPLEX64, ME_COMPLEX128
    }, /* FLOAT32 */
    {
        ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64, ME_FLOAT64,
        ME_FLOAT64, ME_FLOAT64, ME_COMPLEX128, ME_COMPLEX128
    }, /* FLOAT64 */
    {
        ME_COMPLEX64, ME_COMPLEX64, ME_COMPLEX64, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX64,
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX128, ME_COMPLEX64, ME_COMPLEX128
    }, /* COMPLEX64 */
    {
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128,
        ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128, ME_COMPLEX128
    } /* COMPLEX128 */
};

/* Promote two types according to NumPy rules */
static me_dtype promote_types(me_dtype a, me_dtype b) {
    // ME_AUTO should have been resolved during compilation
    if (a == ME_AUTO || b == ME_AUTO) {
        fprintf(stderr, "FATAL: ME_AUTO in type promotion (a=%d, b=%d). This is a bug.\n", a, b);
#ifdef NDEBUG
        abort(); // Release build: terminate immediately
#else
        assert(0 && "ME_AUTO should be resolved during compilation"); // Debug: trigger debugger
#endif
    }

    if (a == ME_STRING || b == ME_STRING) {
        return ME_STRING;
    }

    // Adjust indices since table starts at ME_BOOL (index 1), not ME_AUTO (index 0)
    int a_idx = a - 1;
    int b_idx = b - 1;
    if (a_idx >= 0 && a_idx < 13 && b_idx >= 0 && b_idx < 13) {
        return type_promotion_table[a_idx][b_idx];
    }
    fprintf(stderr, "WARNING: Invalid dtype in type promotion (a=%d, b=%d). Falling back to FLOAT64.\n", a, b);
    return ME_FLOAT64; // Fallback for out-of-range types
}

static bool is_integral_or_bool(me_dtype dtype) {
    return dtype == ME_BOOL || (dtype >= ME_INT8 && dtype <= ME_UINT64);
}

static bool is_valid_dtype(me_dtype dtype) {
    return dtype >= ME_AUTO && dtype <= ME_STRING;
}

static bool is_string_operand_node(const me_expr* n) {
    if (!n) return false;
    if (TYPE_MASK(n->type) == ME_STRING_CONSTANT) return true;
    return TYPE_MASK(n->type) == ME_VARIABLE && n->dtype == ME_STRING;
}

static me_dtype promote_float_math_result(me_dtype param_type) {
    if (param_type == ME_STRING) {
        return ME_STRING;
    }
    if (param_type == ME_COMPLEX64 || param_type == ME_COMPLEX128) {
        return param_type;
    }
    if (param_type == ME_FLOAT32) {
        return ME_FLOAT32;
    }
    if (param_type == ME_FLOAT64) {
        return ME_FLOAT64;
    }
    if (is_integral_or_bool(param_type)) {
        return ME_FLOAT64;
    }
    return param_type;
}


static bool contains_reduction(const me_expr* n) {
    if (!n) return false;
    if (is_reduction_node(n)) return true;

    switch (TYPE_MASK(n->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (contains_reduction((const me_expr*)n->parameters[i])) {
                    return true;
                }
            }
            return false;
        }
    default:
        return false;
    }
}

// Synthetic addresses for ordinal matching (when user provides NULL addresses)
static char synthetic_var_addresses[ME_MAX_VARS];

static void dsl_var_table_init(me_dsl_var_table *table) {
    memset(table, 0, sizeof(*table));
}

static void dsl_var_table_free(me_dsl_var_table *table) {
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

static int dsl_var_table_find(const me_dsl_var_table *table, const char *name) {
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

static int dsl_var_table_add_with_uniform(me_dsl_var_table *table, const char *name, me_dtype dtype,
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

static int dsl_var_table_add(me_dsl_var_table *table, const char *name, me_dtype dtype) {
    return dsl_var_table_add_with_uniform(table, name, dtype, 0, false);
}

static bool dsl_is_reserved_name(const char *name) {
    if (!name) {
        return false;
    }
    if (strcmp(name, "print") == 0) {
        return true;
    }
    if (strcmp(name, "def") == 0 || strcmp(name, "return") == 0) {
        return true;
    }
    if (strcmp(name, "_ndim") == 0) {
        return true;
    }
    if ((name[0] == '_' && (name[1] == 'i' || name[1] == 'n')) && isdigit((unsigned char)name[2])) {
        return true;
    }
    return false;
}

static bool dsl_is_reserved_index_name(const char *name, int *is_index, int *dim) {
    if (!name || name[0] != '_' || !name[1] || !name[2]) {
        return false;
    }
    if (name[1] != 'i' && name[1] != 'n') {
        return false;
    }
    if (!isdigit((unsigned char)name[2])) {
        return false;
    }
    int d = name[2] - '0';
    if (d < 0 || d >= ME_DSL_MAX_NDIM || name[3] != '\0') {
        return false;
    }
    if (is_index) {
        *is_index = (name[1] == 'i');
    }
    if (dim) {
        *dim = d;
    }
    return true;
}

static bool dsl_expr_uses_identifier(const char *expr, const char *ident) {
    if (!expr || !ident) {
        return false;
    }
    size_t ident_len = strlen(ident);
    const char *p = expr;
    while (*p) {
        if (isalpha((unsigned char)*p) || *p == '_') {
            const char *start = p;
            p++;
            while (isalnum((unsigned char)*p) || *p == '_') {
                p++;
            }
            size_t len = (size_t)(p - start);
            if (len == ident_len && strncmp(start, ident, len) == 0) {
                return true;
            }
        }
        else {
            p++;
        }
    }
    return false;
}

static void dsl_compiled_block_free(me_dsl_compiled_block *block);

static void dsl_compiled_expr_free(me_dsl_compiled_expr *expr) {
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

static void dsl_compiled_stmt_free(me_dsl_compiled_stmt *stmt) {
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
    case ME_DSL_STMT_FOR:
        dsl_compiled_expr_free(&stmt->as.for_loop.limit);
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

static void dsl_compiled_program_free(me_dsl_compiled_program *program) {
    if (!program) {
        return;
    }
#if !defined(_WIN32) && !defined(_WIN64)
    if (program->jit_tcc_state) {
        dsl_jit_libtcc_delete_state(program->jit_tcc_state);
        program->jit_tcc_state = NULL;
    }
    if (program->jit_dl_handle) {
        if (!program->jit_dl_handle_cached) {
            dlclose(program->jit_dl_handle);
        }
        program->jit_dl_handle = NULL;
    }
#endif
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    free(program->jit_c_source);
    free(program->jit_param_input_indices);
    me_dsl_jit_ir_free(program->jit_ir);
    dsl_compiled_block_free(&program->block);
    dsl_var_table_free(&program->vars);
    free(program->local_var_indices);
    free(program->local_slots);
    free(program);
}

static bool dsl_compiled_block_push(me_dsl_compiled_block *block, me_dsl_compiled_stmt *stmt) {
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

static bool dsl_collect_var_indices(const me_expr *expr, int **out_indices, int *out_count) {
    if (!expr || !out_indices || !out_count) {
        return false;
    }
    bool used[ME_MAX_VARS];
    memset(used, 0, sizeof(used));
    int max_idx = -1;

    const me_expr *stack[512];
    int sp = 0;
    stack[sp++] = expr;

    while (sp > 0) {
        const me_expr *node = stack[--sp];
        if (!node) {
            continue;
        }
        if (TYPE_MASK(node->type) == ME_VARIABLE) {
            const char *ptr = (const char *)node->bound;
            int idx = (int)(ptr - synthetic_var_addresses);
            if (idx >= 0 && idx < ME_MAX_VARS) {
                used[idx] = true;
                if (idx > max_idx) {
                    max_idx = idx;
                }
            }
        }
        else if (IS_FUNCTION(node->type) || IS_CLOSURE(node->type)) {
            int arity = ARITY(node->type);
            for (int i = 0; i < arity; i++) {
                if (sp < (int)(sizeof(stack) / sizeof(stack[0]))) {
                    stack[sp++] = (const me_expr *)node->parameters[i];
                }
            }
        }
    }

    int count = 0;
    for (int i = 0; i <= max_idx; i++) {
        if (used[i]) {
            count++;
        }
    }
    if (count == 0) {
        *out_indices = NULL;
        *out_count = 0;
        return true;
    }
    int *indices = malloc((size_t)count * sizeof(*indices));
    if (!indices) {
        return false;
    }
    int pos = 0;
    for (int i = 0; i <= max_idx; i++) {
        if (used[i]) {
            indices[pos++] = i;
        }
    }
    *out_indices = indices;
    *out_count = count;
    return true;
}

static bool output_is_scalar(const me_expr* n) {
    if (!n) return true;
    if (is_reduction_node(n)) return true;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
    case ME_STRING_CONSTANT:
        return true;
    case ME_VARIABLE:
        return false;
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!output_is_scalar((const me_expr*)n->parameters[i])) {
                    return false;
                }
            }
            return true;
        }
    default:
        return true;
    }
}

static bool dsl_expr_is_uniform(const me_expr* n, const bool *uniform, int nvars) {
    if (!n) {
        return true;
    }
    if (is_reduction_node(n)) {
        return true;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
    case ME_STRING_CONSTANT:
        return true;
    case ME_VARIABLE: {
        if (!is_synthetic_address(n->bound)) {
            return false;
        }
        int idx = (int)((const char *)n->bound - synthetic_var_addresses);
        if (idx < 0 || idx >= nvars) {
            return false;
        }
        return uniform[idx];
    }
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!dsl_expr_is_uniform((const me_expr*)n->parameters[i], uniform, nvars)) {
                    return false;
                }
            }
            return true;
        }
    default:
        return true;
    }
}

typedef union {
    bool b;
    int64_t i64;
    uint64_t u64;
    float f32;
    double f64;
    float _Complex c64;
    double _Complex c128;
} me_scalar;

static bool dsl_any_nonzero(const void *data, me_dtype dtype, int nitems) {
    if (!data || nitems <= 0) {
        return false;
    }
    switch (dtype) {
    case ME_BOOL: {
        const bool *v = (const bool *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i]) return true;
        }
        return false;
    }
    case ME_INT8: {
        const int8_t *v = (const int8_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_INT16: {
        const int16_t *v = (const int16_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_INT32: {
        const int32_t *v = (const int32_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_INT64: {
        const int64_t *v = (const int64_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_UINT8: {
        const uint8_t *v = (const uint8_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_UINT16: {
        const uint16_t *v = (const uint16_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_UINT32: {
        const uint32_t *v = (const uint32_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_UINT64: {
        const uint64_t *v = (const uint64_t *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0) return true;
        }
        return false;
    }
    case ME_FLOAT32: {
        const float *v = (const float *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0.0f) return true;
        }
        return false;
    }
    case ME_FLOAT64: {
        const double *v = (const double *)data;
        for (int i = 0; i < nitems; i++) {
            if (v[i] != 0.0) return true;
        }
        return false;
    }
    case ME_COMPLEX64: {
        const float _Complex *v = (const float _Complex *)data;
        for (int i = 0; i < nitems; i++) {
            if (me_crealf(v[i]) != 0.0f || me_cimagf(v[i]) != 0.0f) return true;
        }
        return false;
    }
    case ME_COMPLEX128: {
        const double _Complex *v = (const double _Complex *)data;
        for (int i = 0; i < nitems; i++) {
            if (me_creal(v[i]) != 0.0 || me_cimag(v[i]) != 0.0) return true;
        }
        return false;
    }
    case ME_STRING:
        return false;
    default:
        return false;
    }
}

static void dsl_fill_i64(int64_t *out, int nitems, int64_t value) {
    if (!out || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        out[i] = value;
    }
}

static void dsl_fill_iota_i64(int64_t *out, int nitems, int64_t start) {
    if (!out || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        out[i] = start + (int64_t)i;
    }
}

static bool dsl_read_int64(const void *data, me_dtype dtype, int64_t *out) {
    if (!data || !out) {
        return false;
    }
    switch (dtype) {
    case ME_BOOL: *out = ((const bool *)data)[0] ? 1 : 0; return true;
    case ME_INT8: *out = ((const int8_t *)data)[0]; return true;
    case ME_INT16: *out = ((const int16_t *)data)[0]; return true;
    case ME_INT32: *out = ((const int32_t *)data)[0]; return true;
    case ME_INT64: *out = ((const int64_t *)data)[0]; return true;
    case ME_UINT8: *out = (int64_t)((const uint8_t *)data)[0]; return true;
    case ME_UINT16: *out = (int64_t)((const uint16_t *)data)[0]; return true;
    case ME_UINT32: *out = (int64_t)((const uint32_t *)data)[0]; return true;
    case ME_UINT64: *out = (int64_t)((const uint64_t *)data)[0]; return true;
    case ME_FLOAT32: *out = (int64_t)((const float *)data)[0]; return true;
    case ME_FLOAT64: *out = (int64_t)((const double *)data)[0]; return true;
    case ME_COMPLEX64: {
        float _Complex v = ((const float _Complex *)data)[0];
        *out = (int64_t)me_crealf(v);
        return true;
    }
    case ME_COMPLEX128: {
        double _Complex v = ((const double _Complex *)data)[0];
        *out = (int64_t)me_creal(v);
        return true;
    }
    case ME_STRING:
        return false;
    default:
        return false;
    }
}

static float me_crealf(float _Complex v) {
#if defined(_MSC_VER)
    return __real__ v;
#else
    return crealf(v);
#endif
}

static float me_cimagf(float _Complex v) {
#if defined(_MSC_VER)
    return __imag__ v;
#else
    return cimagf(v);
#endif
}

static double me_creal(double _Complex v) {
#if defined(_MSC_VER)
    return __real__ v;
#else
    return creal(v);
#endif
}

static double me_cimag(double _Complex v) {
#if defined(_MSC_VER)
    return __imag__ v;
#else
    return cimag(v);
#endif
}

static float _Complex me_cmplxf(float re, float im) {
#if defined(_MSC_VER)
    float _Complex v;
    __real__ v = re;
    __imag__ v = im;
    return v;
#else
    return re + im * I;
#endif
}

static double _Complex me_cmplx(double re, double im) {
#if defined(_MSC_VER)
    double _Complex v;
    __real__ v = re;
    __imag__ v = im;
    return v;
#else
    return re + im * I;
#endif
}

static void write_scalar(void* out, me_dtype out_type, me_dtype in_type, const me_scalar* v) {
    if (out_type == in_type) {
        switch (out_type) {
        case ME_BOOL: *(bool*)out = v->b; return;
        case ME_INT8: *(int8_t*)out = (int8_t)v->i64; return;
        case ME_INT16: *(int16_t*)out = (int16_t)v->i64; return;
        case ME_INT32: *(int32_t*)out = (int32_t)v->i64; return;
        case ME_INT64: *(int64_t*)out = v->i64; return;
        case ME_UINT8: *(uint8_t*)out = (uint8_t)v->u64; return;
        case ME_UINT16: *(uint16_t*)out = (uint16_t)v->u64; return;
        case ME_UINT32: *(uint32_t*)out = (uint32_t)v->u64; return;
        case ME_UINT64: *(uint64_t*)out = v->u64; return;
        case ME_FLOAT32: *(float*)out = v->f32; return;
        case ME_FLOAT64: *(double*)out = v->f64; return;
        case ME_COMPLEX64: *(float _Complex*)out = v->c64; return;
        case ME_COMPLEX128: *(double _Complex*)out = v->c128; return;
        default: return;
        }
    }

    switch (out_type) {
    case ME_BOOL:
        switch (in_type) {
        case ME_BOOL: *(bool*)out = v->b; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(bool*)out = v->i64 != 0; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(bool*)out = v->u64 != 0; break;
        case ME_FLOAT32: *(bool*)out = v->f32 != 0.0f; break;
        case ME_FLOAT64: *(bool*)out = v->f64 != 0.0; break;
        case ME_COMPLEX64: *(bool*)out = (me_crealf(v->c64) != 0.0f || me_cimagf(v->c64) != 0.0f); break;
        case ME_COMPLEX128: *(bool*)out = (me_creal(v->c128) != 0.0 || me_cimag(v->c128) != 0.0); break;
        default: *(bool*)out = false; break;
        }
        break;
    case ME_INT8:
    case ME_INT16:
    case ME_INT32:
    case ME_INT64:
        switch (in_type) {
        case ME_BOOL: *(int64_t*)out = v->b ? 1 : 0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(int64_t*)out = v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(int64_t*)out = (int64_t)v->u64; break;
        case ME_FLOAT32: *(int64_t*)out = (int64_t)v->f32; break;
        case ME_FLOAT64: *(int64_t*)out = (int64_t)v->f64; break;
        default: *(int64_t*)out = 0; break;
        }
        break;
    case ME_UINT8:
    case ME_UINT16:
    case ME_UINT32:
    case ME_UINT64:
        switch (in_type) {
        case ME_BOOL: *(uint64_t*)out = v->b ? 1 : 0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(uint64_t*)out = (uint64_t)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(uint64_t*)out = v->u64; break;
        case ME_FLOAT32: *(uint64_t*)out = (uint64_t)v->f32; break;
        case ME_FLOAT64: *(uint64_t*)out = (uint64_t)v->f64; break;
        default: *(uint64_t*)out = 0; break;
        }
        break;
    case ME_FLOAT32:
        switch (in_type) {
        case ME_BOOL: *(float*)out = v->b ? 1.0f : 0.0f; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(float*)out = (float)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(float*)out = (float)v->u64; break;
        case ME_FLOAT32: *(float*)out = v->f32; break;
        case ME_FLOAT64: *(float*)out = (float)v->f64; break;
        default: *(float*)out = 0.0f; break;
        }
        break;
    case ME_FLOAT64:
        switch (in_type) {
        case ME_BOOL: *(double*)out = v->b ? 1.0 : 0.0; break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(double*)out = (double)v->i64; break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(double*)out = (double)v->u64; break;
        case ME_FLOAT32: *(double*)out = (double)v->f32; break;
        case ME_FLOAT64: *(double*)out = v->f64; break;
        default: *(double*)out = 0.0; break;
        }
        break;
    case ME_COMPLEX64:
        switch (in_type) {
        case ME_COMPLEX64: *(float _Complex*)out = v->c64; break;
        case ME_COMPLEX128: *(float _Complex*)out = (float _Complex)v->c128; break;
        case ME_FLOAT32: *(float _Complex*)out = me_cmplxf(v->f32, 0.0f); break;
        case ME_FLOAT64: *(float _Complex*)out = me_cmplxf((float)v->f64, 0.0f); break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(float _Complex*)out = me_cmplxf((float)v->i64, 0.0f); break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(float _Complex*)out = me_cmplxf((float)v->u64, 0.0f); break;
        case ME_BOOL: *(float _Complex*)out = me_cmplxf(v->b ? 1.0f : 0.0f, 0.0f); break;
        default: *(float _Complex*)out = me_cmplxf(0.0f, 0.0f); break;
        }
        break;
    case ME_COMPLEX128:
        switch (in_type) {
        case ME_COMPLEX64: *(double _Complex*)out = (double _Complex)v->c64; break;
        case ME_COMPLEX128: *(double _Complex*)out = v->c128; break;
        case ME_FLOAT32: *(double _Complex*)out = me_cmplx((double)v->f32, 0.0); break;
        case ME_FLOAT64: *(double _Complex*)out = me_cmplx(v->f64, 0.0); break;
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: *(double _Complex*)out = me_cmplx((double)v->i64, 0.0); break;
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: *(double _Complex*)out = me_cmplx((double)v->u64, 0.0); break;
        case ME_BOOL: *(double _Complex*)out = me_cmplx(v->b ? 1.0 : 0.0, 0.0); break;
        default: *(double _Complex*)out = me_cmplx(0.0, 0.0); break;
        }
        break;
    default:
        break;
    }
}

static void read_scalar(const void* in, me_dtype in_type, me_scalar* v) {
    switch (in_type) {
    case ME_BOOL: v->b = *(const bool*)in; break;
    case ME_INT8: v->i64 = *(const int8_t*)in; break;
    case ME_INT16: v->i64 = *(const int16_t*)in; break;
    case ME_INT32: v->i64 = *(const int32_t*)in; break;
    case ME_INT64: v->i64 = *(const int64_t*)in; break;
    case ME_UINT8: v->u64 = *(const uint8_t*)in; break;
    case ME_UINT16: v->u64 = *(const uint16_t*)in; break;
    case ME_UINT32: v->u64 = *(const uint32_t*)in; break;
    case ME_UINT64: v->u64 = *(const uint64_t*)in; break;
    case ME_FLOAT32: v->f32 = *(const float*)in; break;
    case ME_FLOAT64: v->f64 = *(const double*)in; break;
    case ME_COMPLEX64: v->c64 = *(const float _Complex*)in; break;
    case ME_COMPLEX128: v->c128 = *(const double _Complex*)in; break;
    default: break;
    }
}

static bool read_as_bool(const void* base, int64_t off, me_dtype type, bool* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off]; return true;
    case ME_INT8: *out = ((const int8_t*)base)[off] != 0; return true;
    case ME_INT16: *out = ((const int16_t*)base)[off] != 0; return true;
    case ME_INT32: *out = ((const int32_t*)base)[off] != 0; return true;
    case ME_INT64: *out = ((const int64_t*)base)[off] != 0; return true;
    case ME_UINT8: *out = ((const uint8_t*)base)[off] != 0; return true;
    case ME_UINT16: *out = ((const uint16_t*)base)[off] != 0; return true;
    case ME_UINT32: *out = ((const uint32_t*)base)[off] != 0; return true;
    case ME_UINT64: *out = ((const uint64_t*)base)[off] != 0; return true;
    case ME_FLOAT32: *out = ((const float*)base)[off] != 0.0f; return true;
    case ME_FLOAT64: *out = ((const double*)base)[off] != 0.0; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_int64(const void* base, int64_t off, me_dtype type, int64_t* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1 : 0; return true;
    case ME_INT8: *out = ((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = ((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = ((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = ((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = (int64_t)((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = (int64_t)((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = (int64_t)((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = (int64_t)((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = (int64_t)((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = (int64_t)((const double*)base)[off]; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_uint64(const void* base, int64_t off, me_dtype type, uint64_t* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1 : 0; return true;
    case ME_INT8: *out = (uint64_t)((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = (uint64_t)((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = (uint64_t)((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = (uint64_t)((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = ((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = ((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = ((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = ((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = (uint64_t)((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = (uint64_t)((const double*)base)[off]; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_float(const void* base, int64_t off, me_dtype type, float* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1.0f : 0.0f; return true;
    case ME_INT8: *out = (float)((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = (float)((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = (float)((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = (float)((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = (float)((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = (float)((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = (float)((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = (float)((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = ((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = (float)((const double*)base)[off]; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool read_as_double(const void* base, int64_t off, me_dtype type, double* out) {
    switch (type) {
    case ME_BOOL: *out = ((const bool*)base)[off] ? 1.0 : 0.0; return true;
    case ME_INT8: *out = (double)((const int8_t*)base)[off]; return true;
    case ME_INT16: *out = (double)((const int16_t*)base)[off]; return true;
    case ME_INT32: *out = (double)((const int32_t*)base)[off]; return true;
    case ME_INT64: *out = (double)((const int64_t*)base)[off]; return true;
    case ME_UINT8: *out = (double)((const uint8_t*)base)[off]; return true;
    case ME_UINT16: *out = (double)((const uint16_t*)base)[off]; return true;
    case ME_UINT32: *out = (double)((const uint32_t*)base)[off]; return true;
    case ME_UINT64: *out = (double)((const uint64_t*)base)[off]; return true;
    case ME_FLOAT32: *out = (double)((const float*)base)[off]; return true;
    case ME_FLOAT64: *out = ((const double*)base)[off]; return true;
    case ME_STRING: return false;
    default:
        return false;
    }
}

static bool cmp_int64(me_cmp_kind cmp, int64_t a, int64_t b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static bool cmp_uint64(me_cmp_kind cmp, uint64_t a, uint64_t b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static bool cmp_float(me_cmp_kind cmp, float a, float b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static bool cmp_double(me_cmp_kind cmp, double a, double b) {
    switch (cmp) {
    case ME_CMP_EQ: return a == b;
    case ME_CMP_NE: return a != b;
    case ME_CMP_LT: return a < b;
    case ME_CMP_LE: return a <= b;
    case ME_CMP_GT: return a > b;
    case ME_CMP_GE: return a >= b;
    default: return false;
    }
}

static me_cmp_kind invert_cmp_kind(me_cmp_kind cmp) {
    switch (cmp) {
    case ME_CMP_LT: return ME_CMP_GT;
    case ME_CMP_LE: return ME_CMP_GE;
    case ME_CMP_GT: return ME_CMP_LT;
    case ME_CMP_GE: return ME_CMP_LE;
    case ME_CMP_EQ: return ME_CMP_EQ;
    case ME_CMP_NE: return ME_CMP_NE;
    default: return ME_CMP_NONE;
    }
}

static bool reduce_strided_variable(const me_expr* expr, const void** vars_block, int n_vars,
                                    const int64_t* valid_len, const int64_t* stride, int nd,
                                    int64_t valid_items, void* output_block) {
    if (!expr || !is_reduction_node(expr) || valid_items <= 0) {
        return false;
    }
    const me_expr* arg = (const me_expr*)expr->parameters[0];
    if (!arg || TYPE_MASK(arg->type) != ME_VARIABLE || !is_synthetic_address(arg->bound)) {
        return false;
    }
    int idx = (int)((const char*)arg->bound - synthetic_var_addresses);
    if (idx < 0 || idx >= n_vars) {
        return false;
    }

    const me_reduce_kind rkind = reduction_kind(expr->function);
    if (rkind == ME_REDUCE_NONE) {
        return false;
    }
    const bool is_mean = (rkind == ME_REDUCE_MEAN);

    const me_dtype arg_type = infer_result_type(arg);
    const me_dtype result_type = reduction_output_dtype(arg_type, expr->function);
    const me_dtype output_type = expr->dtype;

    int64_t indices[64] = {0};
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) total_iters *= valid_len[i];

    me_scalar acc;
    switch (result_type) {
    case ME_BOOL: acc.b = (rkind == ME_REDUCE_ALL); break;
    case ME_INT64: acc.i64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
    case ME_UINT64: acc.u64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
    case ME_FLOAT32: acc.f64 = (rkind == ME_REDUCE_PROD) ? 1.0 : 0.0; break;
    case ME_FLOAT64: acc.f64 = (rkind == ME_REDUCE_PROD) ? 1.0 : 0.0; break;
    case ME_COMPLEX64: acc.c64 = (rkind == ME_REDUCE_PROD) ? (float _Complex)1.0f : (float _Complex)0.0f; break;
    case ME_COMPLEX128: acc.c128 = (rkind == ME_REDUCE_PROD) ? (double _Complex)1.0 : (double _Complex)0.0; break;
    default: break;
    }

    const unsigned char* base = (const unsigned char*)vars_block[idx];
    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }

        switch (arg_type) {
        case ME_BOOL: {
            bool v = ((const bool*)base)[off];
            if (is_mean) {
                acc.f64 += v ? 1.0 : 0.0;
            }
            else if (rkind == ME_REDUCE_ANY) { if (v) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!v) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v ? 1 : 0;
            else acc.i64 += v ? 1 : 0;
            break;
        }
        case ME_INT8: {
            int8_t v = ((const int8_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int8_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int8_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT16: {
            int16_t v = ((const int16_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int16_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int16_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT32: {
            int32_t v = ((const int32_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (int32_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (int32_t)acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_INT64: {
            int64_t v = ((const int64_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.i64) acc.i64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.i64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.i64 += v;
            break;
        }
        case ME_UINT8: {
            uint8_t v = ((const uint8_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint8_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint8_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT16: {
            uint16_t v = ((const uint16_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint16_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint16_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT32: {
            uint32_t v = ((const uint32_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (uint32_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (uint32_t)acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_UINT64: {
            uint64_t v = ((const uint64_t*)base)[off];
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.u64) acc.u64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.u64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0) { acc.b = false; goto done_reduce; } }
            else acc.u64 += v;
            break;
        }
        case ME_FLOAT32: {
            float v = ((const float*)base)[off];
            if (v != v) { acc.f64 = NAN; goto done_reduce; }
            if (is_mean) { acc.f64 += (double)v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < (float)acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > (float)acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.f64 *= (double)v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0.0f) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0.0f) { acc.b = false; goto done_reduce; } }
            else acc.f64 += (double)v;
            break;
        }
        case ME_FLOAT64: {
            double v = ((const double*)base)[off];
            if (v != v) { acc.f64 = NAN; goto done_reduce; }
            if (is_mean) { acc.f64 += v; }
            else if (rkind == ME_REDUCE_MIN) { if (it == 0 || v < acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_MAX) { if (it == 0 || v > acc.f64) acc.f64 = v; }
            else if (rkind == ME_REDUCE_PROD) acc.f64 *= v;
            else if (rkind == ME_REDUCE_ANY) { if (v != 0.0) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (v == 0.0) { acc.b = false; goto done_reduce; } }
            else acc.f64 += v;
            break;
        }
        case ME_COMPLEX64: {
            float _Complex v = ((const float _Complex*)base)[off];
            bool nonzero = (me_crealf(v) != 0.0f || me_cimagf(v) != 0.0f);
            if (is_mean) { acc.c128 += (double _Complex)v; }
            else if (rkind == ME_REDUCE_ANY) { if (nonzero) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!nonzero) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.c64 *= v;
            else acc.c64 += v;
            break;
        }
        case ME_COMPLEX128: {
            double _Complex v = ((const double _Complex*)base)[off];
            bool nonzero = (me_creal(v) != 0.0 || me_cimag(v) != 0.0);
            if (is_mean) { acc.c128 += v; }
            else if (rkind == ME_REDUCE_ANY) { if (nonzero) { acc.b = true; goto done_reduce; } }
            else if (rkind == ME_REDUCE_ALL) { if (!nonzero) { acc.b = false; goto done_reduce; } }
            else if (rkind == ME_REDUCE_PROD) acc.c128 *= v;
            else acc.c128 += v;
            break;
        }
        default:
            break;
        }

        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) break;
            indices[i] = 0;
        }
    }

done_reduce:
    if (is_mean) {
        if (result_type == ME_COMPLEX128) {
            acc.c128 /= (double)valid_items;
        }
        else {
            acc.f64 /= (double)valid_items;
        }
    }
    if (result_type == ME_FLOAT32) {
        acc.f32 = (float)acc.f64;
    }
    write_scalar(output_block, output_type, result_type, &acc);
    return true;
}

static bool reduce_strided_predicate(const me_expr* expr, const void** vars_block, int n_vars,
                                     const int64_t* valid_len, const int64_t* stride, int nd,
                                     int64_t valid_items, void* output_block) {
    if (!expr || !is_reduction_node(expr) || valid_items <= 0) {
        return false;
    }
    const me_expr* arg = (const me_expr*)expr->parameters[0];
    if (!arg || !is_comparison_node(arg)) {
        return false;
    }

    me_reduce_kind rkind = reduction_kind(expr->function);
    if (!(rkind == ME_REDUCE_ANY || rkind == ME_REDUCE_ALL)) {
        /* Keep only any/all predicate reductions; sum(x == c) uses pack path. */
        return false;
    }

    const me_expr* left = (const me_expr*)arg->parameters[0];
    const me_expr* right = (const me_expr*)arg->parameters[1];
    if (!left || !right) {
        return false;
    }

    const me_expr* var_node = NULL;
    const me_expr* const_node = NULL;
    bool const_on_left = false;

    if (TYPE_MASK(left->type) == ME_VARIABLE && right->type == ME_CONSTANT) {
        var_node = left;
        const_node = right;
    }
    else if (TYPE_MASK(right->type) == ME_VARIABLE && left->type == ME_CONSTANT) {
        var_node = right;
        const_node = left;
        const_on_left = true;
    }
    else {
        return false;
    }

    if (!is_synthetic_address(var_node->bound)) {
        return false;
    }
    int idx = (int)((const char*)var_node->bound - synthetic_var_addresses);
    if (idx < 0 || idx >= n_vars) {
        return false;
    }

    me_cmp_kind cmp = comparison_kind(arg->function);
    if (cmp == ME_CMP_NONE) {
        return false;
    }
    if (const_on_left) {
        cmp = invert_cmp_kind(cmp);
        if (cmp == ME_CMP_NONE) {
            return false;
        }
    }

    const me_dtype eval_type = infer_result_type(arg);
    if (eval_type == ME_COMPLEX64 || eval_type == ME_COMPLEX128) {
        return false;
    }

    const me_dtype output_type = expr->dtype;
    const me_dtype result_type = reduction_output_dtype(ME_BOOL, expr->function);

    int64_t indices[64] = {0};
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) total_iters *= valid_len[i];

    me_scalar acc;
    acc.b = (rkind == ME_REDUCE_ALL);

    const unsigned char* base = (const unsigned char*)vars_block[idx];
    const double cval = const_node->value;

    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }

        bool pred = false;
        switch (eval_type) {
        case ME_BOOL: {
            bool v = false;
            if (!read_as_bool(base, off, var_node->input_dtype, &v)) return false;
            bool c = (cval != 0.0);
            pred = cmp_int64(cmp, v ? 1 : 0, c ? 1 : 0);
            break;
        }
        case ME_INT8:
        case ME_INT16:
        case ME_INT32:
        case ME_INT64: {
            int64_t v = 0;
            if (!read_as_int64(base, off, var_node->input_dtype, &v)) return false;
            int64_t c = (int64_t)cval;
            pred = cmp_int64(cmp, v, c);
            break;
        }
        case ME_UINT8:
        case ME_UINT16:
        case ME_UINT32:
        case ME_UINT64: {
            uint64_t v = 0;
            if (!read_as_uint64(base, off, var_node->input_dtype, &v)) return false;
            uint64_t c = (uint64_t)cval;
            pred = cmp_uint64(cmp, v, c);
            break;
        }
        case ME_FLOAT32: {
            float v = 0.0f;
            if (!read_as_float(base, off, var_node->input_dtype, &v)) return false;
            float c = (float)cval;
            pred = cmp_float(cmp, v, c);
            break;
        }
        case ME_FLOAT64: {
            double v = 0.0;
            if (!read_as_double(base, off, var_node->input_dtype, &v)) return false;
            pred = cmp_double(cmp, v, cval);
            break;
        }
        default:
            return false;
        }

        if (rkind == ME_REDUCE_ANY) {
            if (pred) { acc.b = true; goto done_pred; }
        }
        else if (rkind == ME_REDUCE_ALL) {
            if (!pred) { acc.b = false; goto done_pred; }
        }

        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) break;
            indices[i] = 0;
        }
    }

done_pred:
    write_scalar(output_block, output_type, result_type, &acc);
    return true;
}
static bool reduction_usage_is_valid(const me_expr* n) {
    if (!n) return true;
    if (is_reduction_node(n)) {
        me_expr* arg = (me_expr*)n->parameters[0];
        if (!arg) return false;
        if (contains_reduction(arg)) return false;
        me_dtype arg_type = infer_output_type(arg);
        if (n->function == (void*)min_reduce || n->function == (void*)max_reduce) {
            if (arg_type == ME_COMPLEX64 || arg_type == ME_COMPLEX128) {
                return false;
            }
        }
        return true;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity; i++) {
                if (!reduction_usage_is_valid((const me_expr*)n->parameters[i])) {
                    return false;
                }
            }
            return true;
        }
    default:
        return true;
    }
}

/* Infer computation type from expression tree (for evaluation) */
me_dtype infer_result_type(const me_expr* n) {
    if (!n) return ME_FLOAT64;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return n->dtype;

    case ME_STRING_CONSTANT:
        return ME_STRING;

    case ME_VARIABLE:
        return n->dtype;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            if ((n->flags & ME_EXPR_FLAG_EXPLICIT_DTYPE) != 0) {
                return n->dtype;
            }
            if (is_reduction_node(n)) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                return reduction_output_dtype(param_type, n->function);
            }
            // Special case: imag() and real() return real type from complex input
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
                if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper) {
                    me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    else if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    // If input is not complex, return as-is (shouldn't happen, but be safe)
                    return param_type;
                }
                if (n->function == (void*)fabs) {
                    me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    return param_type;
                }
            }

            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && is_float_math_function(n->function)) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[0]);
                return promote_float_math_result(param_type);
            }

            if (ARITY(n->type) == 2) {
                const me_expr* left = (const me_expr*)n->parameters[0];
                const me_expr* right = (const me_expr*)n->parameters[1];
                if (is_string_operand_node(left) && is_string_operand_node(right)) {
                    return ME_BOOL;
                }
            }

            // For comparisons with ME_BOOL output, we still need to infer the
            // computation type from operands (e.g., float64 for float inputs).
            // Don't return ME_BOOL early - let the operand types determine
            // the computation type.

            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_result_type((const me_expr*)n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Infer logical output type from expression tree (for compilation with ME_AUTO) */
me_dtype infer_output_type(const me_expr* n) {
    if (!n) return ME_FLOAT64;

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT:
        return n->dtype;

    case ME_STRING_CONSTANT:
        return ME_STRING;

    case ME_VARIABLE:
        return n->dtype;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        {
            if ((n->flags & ME_EXPR_FLAG_EXPLICIT_DTYPE) != 0) {
                return n->dtype;
            }
            if (is_reduction_node(n)) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                return reduction_output_dtype(param_type, n->function);
            }
            // Special case: imag() and real() return real type from complex input
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1) {
                if (n->function == (void*)imag_wrapper || n->function == (void*)real_wrapper) {
                    me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    else if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    // If input is not complex, return as-is (shouldn't happen, but be safe)
                    return param_type;
                }
                if (n->function == (void*)fabs) {
                    me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                    if (param_type == ME_COMPLEX64) {
                        return ME_FLOAT32;
                    }
                    if (param_type == ME_COMPLEX128) {
                        return ME_FLOAT64;
                    }
                    return param_type;
                }
            }

            // Special case: where(cond, x, y) -> promote(x, y), regardless of cond type.
            if (IS_FUNCTION(n->type) && ARITY(n->type) == 3 &&
                n->function == (void*)where_scalar) {
                me_dtype x_type = infer_output_type((const me_expr*)n->parameters[1]);
                me_dtype y_type = infer_output_type((const me_expr*)n->parameters[2]);
                return promote_types(x_type, y_type);
            }

            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && is_float_math_function(n->function)) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[0]);
                return promote_float_math_result(param_type);
            }

            // If this node is a comparison (dtype == ME_BOOL set during parsing),
            // the output type is ME_BOOL
            if (n->dtype == ME_BOOL) {
                return ME_BOOL;
            }

            // Otherwise, infer from operands
            const int arity = ARITY(n->type);
            me_dtype result = ME_BOOL;

            for (int i = 0; i < arity; i++) {
                me_dtype param_type = infer_output_type((const me_expr*)n->parameters[i]);
                result = promote_types(result, param_type);
            }

            return result;
        }
    }

    return ME_FLOAT64;
}

/* Apply type promotion to a binary operation node */
static me_expr* create_conversion_node(me_expr* source, me_dtype target_dtype) {
    /* Create a unary conversion node that converts source to target_dtype */
    me_expr* conv = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, source);
    if (conv) {
        conv->function = NULL; // Mark as conversion
        conv->dtype = target_dtype;
        conv->input_dtype = source->dtype;
    }
    return conv;
}

void apply_type_promotion(me_expr* node) {
    if (!node || ARITY(node->type) < 2) return;

    me_expr* left = (me_expr*)node->parameters[0];
    me_expr* right = (me_expr*)node->parameters[1];

    if (left && right) {
        me_dtype left_type = left->dtype;
        me_dtype right_type = right->dtype;
        me_dtype promoted = promote_types(left_type, right_type);

        // Store the promoted output type
        node->dtype = promoted;

        // Insert conversion nodes if needed for nested expressions with different dtype
        if (left_type != promoted && TYPE_MASK(left->type) >= ME_FUNCTION0) {
            me_expr *conv_left = create_conversion_node(left, promoted);
            if (conv_left) {
                node->parameters[0] = conv_left;
            }
        }

        if (right_type != promoted && TYPE_MASK(right->type) >= ME_FUNCTION0) {
            me_expr *conv_right = create_conversion_node(right, promoted);
            if (conv_right) {
                node->parameters[1] = conv_right;
            }
        }
    }
}

/* Check for mixed-type nested expressions (currently not supported) */
static int check_mixed_type_nested(const me_expr* node, me_dtype parent_dtype) {
    if (!node) return 0;

    switch (TYPE_MASK(node->type)) {
    case ME_CONSTANT:
    case ME_STRING_CONSTANT:
    case ME_VARIABLE:
        return 0;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        /* Skip reduction nodes - they handle their own type conversions */
        if (is_reduction_node(node)) {
            return 0;
        }

        /* Skip comparison nodes - they naturally have different output type (bool) than operands */
        if (is_comparison_node(node)) {
            return 0;
        }

        /* Only check binary operations (arity 2) for mixed-type nested expressions */
        /* Unary operations are fine */
        const int arity = ARITY(node->type);
        if (arity == 2 && IS_FUNCTION(node->type)) {
            me_expr* left = (me_expr*)node->parameters[0];
            me_expr* right = (me_expr*)node->parameters[1];

            /* If either operand is a nested expression (not constant/variable) with different dtype, flag it */
            if (left && TYPE_MASK(left->type) >= ME_FUNCTION0 &&
                left->dtype != ME_AUTO && node->dtype != ME_AUTO &&
                left->dtype != node->dtype) {
                return 1;
            }
            if (right && TYPE_MASK(right->type) >= ME_FUNCTION0 &&
                right->dtype != ME_AUTO && node->dtype != ME_AUTO &&
                right->dtype != node->dtype) {
                return 1;
            }
        }

        /* Recursively check children */
        for (int i = 0; i < arity; i++) {
            if (check_mixed_type_nested((const me_expr*)node->parameters[i], node->dtype)) {
                return 1;
            }
        }
        break;
    }

    return 0;
}

me_expr* new_expr(const int type, const me_expr* parameters[]) {
    const int arity = ARITY(type);
    const int psize = sizeof(void*) * arity;
    const int size = (sizeof(me_expr) - sizeof(void*)) + psize + (IS_CLOSURE(type) ? sizeof(void*) : 0);
    me_expr* ret = malloc(size);
    CHECK_NULL(ret);

    memset(ret, 0, size);
    if (arity && parameters) {
        memcpy(ret->parameters, parameters, psize);
    }
    ret->type = type;
    ret->bound = 0;
    ret->output = NULL;
    ret->nitems = 0;
    ret->dtype = ME_FLOAT64; // Default to double
    ret->bytecode = NULL;
    ret->ncode = 0;
    ret->dsl_program = NULL;
    return ret;
}


void me_free_parameters(me_expr* n) {
    if (!n) return;
    switch (TYPE_MASK(n->type)) {
    case ME_FUNCTION7:
    case ME_CLOSURE7:
        if (n->parameters[6] && ((me_expr*)n->parameters[6])->output &&
            ((me_expr*)n->parameters[6])->output != n->output) {
            free(((me_expr*)n->parameters[6])->output);
        }
        me_free(n->parameters[6]);
    case ME_FUNCTION6:
    case ME_CLOSURE6:
        if (n->parameters[5] && ((me_expr*)n->parameters[5])->output &&
            ((me_expr*)n->parameters[5])->output != n->output) {
            free(((me_expr*)n->parameters[5])->output);
        }
        me_free(n->parameters[5]);
    case ME_FUNCTION5:
    case ME_CLOSURE5:
        if (n->parameters[4] && ((me_expr*)n->parameters[4])->output &&
            ((me_expr*)n->parameters[4])->output != n->output) {
            free(((me_expr*)n->parameters[4])->output);
        }
        me_free(n->parameters[4]);
    case ME_FUNCTION4:
    case ME_CLOSURE4:
        if (n->parameters[3] && ((me_expr*)n->parameters[3])->output &&
            ((me_expr*)n->parameters[3])->output != n->output) {
            free(((me_expr*)n->parameters[3])->output);
        }
        me_free(n->parameters[3]);
    case ME_FUNCTION3:
    case ME_CLOSURE3:
        if (n->parameters[2] && ((me_expr*)n->parameters[2])->output &&
            ((me_expr*)n->parameters[2])->output != n->output) {
            free(((me_expr*)n->parameters[2])->output);
        }
        me_free(n->parameters[2]);
    case ME_FUNCTION2:
    case ME_CLOSURE2:
        if (n->parameters[1] && ((me_expr*)n->parameters[1])->output &&
            ((me_expr*)n->parameters[1])->output != n->output) {
            free(((me_expr*)n->parameters[1])->output);
        }
        me_free(n->parameters[1]);
    case ME_FUNCTION1:
    case ME_CLOSURE1:
        if (n->parameters[0] && ((me_expr*)n->parameters[0])->output &&
            ((me_expr*)n->parameters[0])->output != n->output) {
            free(((me_expr*)n->parameters[0])->output);
        }
        me_free(n->parameters[0]);
    }
}


void me_free(me_expr* n) {
    if (!n) return;
    me_free_parameters(n);
    if (n->bytecode) {
        free(n->bytecode);
    }
    if (n->dsl_program) {
        dsl_compiled_program_free((me_dsl_compiled_program *)n->dsl_program);
    }
    if (TYPE_MASK(n->type) == ME_STRING_CONSTANT &&
        (n->flags & ME_EXPR_FLAG_OWNS_STRING) != 0) {
        free((void*)n->bound);
    }
    free(n);
}

static int private_compile_ex(const char* expression, const me_variable_ex* variables, int var_count,
                              void* output, int nitems, me_dtype dtype, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!expression || !out || var_count < 0) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }
    if (!variables && var_count > 0) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    if (dtype != ME_AUTO && !is_valid_dtype(dtype)) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG_TYPE;
    }
    if (dtype == ME_STRING) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG_TYPE;
    }

    if (variables && var_count > 0) {
        for (int i = 0; i < var_count; i++) {
            if (!is_variable_entry(&variables[i]) && !is_function_entry(&variables[i])) {
                if (error) *error = -1;
                return ME_COMPILE_ERR_INVALID_ARG_TYPE;
            }
            if (is_function_entry(&variables[i])) {
                if (!variables[i].address) {
                    if (error) *error = -1;
                    return ME_COMPILE_ERR_INVALID_ARG;
                }
                if (variables[i].dtype == ME_STRING) {
                    if (error) *error = -1;
                    return ME_COMPILE_ERR_INVALID_ARG_TYPE;
                }
            }
            if (!is_valid_dtype(variables[i].dtype)) {
                if (error) *error = -1;
                return ME_COMPILE_ERR_INVALID_ARG_TYPE;
            }
            if (is_variable_entry(&variables[i]) && variables[i].dtype == ME_STRING) {
                if (variables[i].itemsize == 0 || (variables[i].itemsize % 4) != 0) {
                    if (error) *error = -1;
                    return ME_COMPILE_ERR_INVALID_ARG_TYPE;
                }
            }
        }
    }

    // Validate dtype usage: either all vars are ME_AUTO (use dtype), or dtype is ME_AUTO (use var dtypes)
    if (variables && var_count > 0) {
        int auto_count = 0;
        int specified_count = 0;

        for (int i = 0; i < var_count; i++) {
            if (!is_variable_entry(&variables[i])) {
                continue;
            }
            if (variables[i].dtype == ME_AUTO) {
                auto_count++;
            }
            else {
                specified_count++;
            }
        }

        // Check the two valid modes
        if (dtype == ME_AUTO) {
            // Mode 1: Output dtype is ME_AUTO, all variables must have explicit dtypes
            if (auto_count > 0) {
                fprintf(
                    stderr,
                    "Error: When output dtype is ME_AUTO, all variable dtypes must be specified (not ME_AUTO)\n");
                if (error) *error = -1;
                return ME_COMPILE_ERR_VAR_UNSPECIFIED;
            }
        }
        else {
            // Mode 2: Output dtype is specified
            // Two sub-modes: all ME_AUTO (homogeneous), or all explicit (heterogeneous with conversion)
            if (auto_count > 0 && specified_count > 0) {
                // Mixed mode not allowed
                fprintf(stderr, "Error: Variable dtypes must be all ME_AUTO or all explicitly specified\n");
                if (error) *error = -1;
                return ME_COMPILE_ERR_VAR_MIXED;
            }
        }
    }

    // Create a copy of variables with dtype filled in (if not already set)
    me_variable_ex* vars_copy = NULL;
    if (variables && var_count > 0) {
        vars_copy = malloc(var_count * sizeof(me_variable_ex));
        if (!vars_copy) {
            if (error) *error = -1;
            return ME_COMPILE_ERR_OOM;
        }
        for (int i = 0; i < var_count; i++) {
            vars_copy[i] = variables[i];
            // If dtype not set (ME_AUTO), use the provided dtype
            if (vars_copy[i].dtype == ME_AUTO && vars_copy[i].type == 0) {
                vars_copy[i].dtype = dtype;
                vars_copy[i].type = ME_VARIABLE;
            }
        }
    }

    state s;
    s.start = s.next = expression;
    s.lookup = vars_copy ? vars_copy : variables;
    s.lookup_len = var_count;
    s.itemsize = 0;
    s.str_data = NULL;
    s.str_len = 0;
    // When dtype is ME_AUTO, infer target dtype from variables to avoid type mismatch
    if (dtype != ME_AUTO) {
        s.target_dtype = dtype;
    }
    else if (variables && var_count > 0) {
        // Use the first variable's dtype as the target for constants
        // This prevents type promotion issues when mixing float32 vars with float64 constants
        s.target_dtype = ME_AUTO;
        for (int i = 0; i < var_count; i++) {
            if (!is_variable_entry(&variables[i])) {
                continue;
            }
            if (variables[i].dtype != ME_STRING) {
                s.target_dtype = variables[i].dtype;
                break;
            }
        }
    }
    else {
        s.target_dtype = ME_AUTO;
    }

    next_token(&s);
    me_expr* root = list(&s);

    if (root == NULL) {
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        if (s.str_data) {
            free((void*)s.str_data);
        }
        return ME_COMPILE_ERR_OOM;
    }

    if (!validate_string_usage(root)) {
        me_free(root);
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        if (s.str_data) {
            free((void*)s.str_data);
        }
        return ME_COMPILE_ERR_INVALID_ARG_TYPE;
    }

    if (contains_reduction(root) && !reduction_usage_is_valid(root)) {
        me_free(root);
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_REDUCTION_INVALID;
    }

    bool any_complex_vars = false;
    if (variables && var_count > 0) {
        const me_variable_ex* vars_check = vars_copy ? vars_copy : variables;
        for (int i = 0; i < var_count; i++) {
            if (!is_variable_entry(&vars_check[i])) {
                continue;
            }
            if (vars_check[i].dtype == ME_COMPLEX64 || vars_check[i].dtype == ME_COMPLEX128) {
                any_complex_vars = true;
                break;
            }
        }
    }

    if ((any_complex_vars || has_complex_input_types(root)) && has_unsupported_complex_function(root)) {
        me_free(root);
        if (error) *error = -1;
        if (vars_copy) free(vars_copy);
        return ME_COMPILE_ERR_INVALID_ARG_TYPE;
    }

#if defined(_WIN32) || defined(_WIN64)
    {
        const me_variable_ex* vars_check = vars_copy ? vars_copy : variables;
        bool complex_vars = false;
        if (vars_check) {
            for (int i = 0; i < var_count; i++) {
                if (!is_variable_entry(&vars_check[i])) {
                    continue;
                }
                if (vars_check[i].dtype == ME_COMPLEX64 || vars_check[i].dtype == ME_COMPLEX128) {
                    complex_vars = true;
                    break;
                }
            }
        }
        if (complex_vars ||
            dtype == ME_COMPLEX64 || dtype == ME_COMPLEX128 ||
            has_complex_node(root) || has_complex_input(root)) {
            fprintf(stderr, "Error: Complex expressions are not supported on Windows (no C99 complex ABI)\n");
            me_free(root);
            if (error) *error = -1;
            if (vars_copy) free(vars_copy);
            return ME_COMPILE_ERR_COMPLEX_UNSUPPORTED;
        }
    }
#endif

    if (s.type != TOK_END) {
        me_free(root);
        if (error) {
            *error = (s.next - s.start);
            if (*error == 0) *error = 1;
        }
        if (vars_copy) free(vars_copy);
        if (s.str_data) {
            free((void*)s.str_data);
        }
        return ME_COMPILE_ERR_PARSE;
    }
    else {
        optimize(root);
        root->output = output;
        root->nitems = nitems;

        // If dtype is ME_AUTO, infer from expression; otherwise use provided dtype
        if (dtype == ME_AUTO) {
            root->dtype = infer_output_type(root);
        }
        else {
            // User explicitly requested a dtype - use it (will cast if needed)
            root->dtype = dtype;
        }

        // Mixed-type nested expressions now handled via conversion nodes
        // (see apply_type_promotion which inserts conversion nodes when needed)

        if (error) *error = 0;
        if (vars_copy) free(vars_copy);
        *out = root;
        return ME_COMPILE_SUCCESS;
    }
}

typedef struct {
    const char *source;
    me_dtype output_dtype;
    bool output_dtype_auto;
    int loop_depth;
    me_dsl_dialect dialect;
    bool allow_new_locals;
    int *error_pos;
    me_dsl_compiled_expr *output_expr;
    bool has_return;
    me_dtype return_dtype;
    me_dsl_compiled_program *program;
    const me_variable_ex *funcs;
    int func_count;
} dsl_compile_ctx;

static const char *dsl_dialect_name(me_dsl_dialect dialect) {
    switch (dialect) {
    case ME_DSL_DIALECT_VECTOR:
        return "vector";
    case ME_DSL_DIALECT_ELEMENT:
        return "element";
    default:
        return "unknown";
    }
}

static const char *dsl_fp_mode_name(me_dsl_fp_mode fp_mode) {
    switch (fp_mode) {
    case ME_DSL_FP_STRICT:
        return "strict";
    case ME_DSL_FP_CONTRACT:
        return "contract";
    case ME_DSL_FP_FAST:
        return "fast";
    default:
        return "unknown";
    }
}

static const char *dsl_jit_fp_mode_cflags(me_dsl_fp_mode fp_mode) {
    switch (fp_mode) {
    case ME_DSL_FP_STRICT:
        return "-fno-fast-math -ffp-contract=off";
    case ME_DSL_FP_CONTRACT:
        return "-fno-fast-math -ffp-contract=fast";
    case ME_DSL_FP_FAST:
        return "-ffast-math";
    default:
        return "-fno-fast-math -ffp-contract=off";
    }
}

static bool dsl_trace_enabled(void) {
    const char *env = getenv("ME_DSL_TRACE");
    if (!env || env[0] == '\0') {
        return false;
    }
    return strcmp(env, "0") != 0;
}

static void dsl_tracef(const char *fmt, ...) {
    if (!dsl_trace_enabled() || !fmt) {
        return;
    }
    fprintf(stderr, "[me-dsl] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static bool dsl_element_dialect_enabled(void) {
    const char *env = getenv("ME_DSL_ELEMENT");
    if (!env || env[0] == '\0') {
        return true;
    }
    return strcmp(env, "0") != 0;
}

static int dsl_offset_from_linecol(const char *source, int line, int column) {
    if (!source || line <= 0 || column <= 0) {
        return -1;
    }
    int current_line = 1;
    int current_col = 1;
    for (int i = 0; source[i] != '\0'; i++) {
        if (current_line == line && current_col == column) {
            return i;
        }
        if (source[i] == '\n') {
            current_line++;
            current_col = 1;
        }
        else {
            current_col++;
        }
    }
    return -1;
}

static bool dsl_is_candidate(const char *source) {
    if (!source) {
        return false;
    }
    const unsigned char *p = (const unsigned char *)source;
    while (*p) {
        unsigned char c = *p;
        if (c == '\n' || c == ';' || c == '{' || c == '}') {
            return true;
        }
        if (c == '=') {
            char prev = (p == (const unsigned char *)source) ? '\0' : (char)p[-1];
            char next = (char)p[1];
            if (next != '=' && prev != '=' && prev != '!' && prev != '<' && prev != '>') {
                return true;
            }
        }
        if (isalpha(c) || c == '_') {
            const unsigned char *start = p;
            p++;
            while (isalnum(*p) || *p == '_') {
                p++;
            }
            size_t len = (size_t)(p - start);
            switch (len) {
            case 2:
                if (start[0] == 'i' && start[1] == 'f') return true;
                break;
            case 3:
                if ((start[0] == 'd' && start[1] == 'e' && start[2] == 'f') ||
                    (start[0] == 'f' && start[1] == 'o' && start[2] == 'r')) {
                    return true;
                }
                break;
            case 4:
                if ((start[0] == 'e' && start[1] == 'l' && start[2] == 's' && start[3] == 'e') ||
                    (start[0] == 'e' && start[1] == 'l' && start[2] == 'i' && start[3] == 'f')) {
                    return true;
                }
                break;
            case 5:
                if (memcmp(start, "break", 5) == 0 || memcmp(start, "print", 5) == 0) {
                    return true;
                }
                break;
            case 6:
                if (memcmp(start, "return", 6) == 0) {
                    return true;
                }
                break;
            case 8:
                if (memcmp(start, "continue", 8) == 0) {
                    return true;
                }
                break;
            default:
                break;
            }
            continue;
        }
        p++;
    }
    return false;
}

static bool dsl_program_is_dsl(const me_dsl_program *program) {
    if (!program) {
        return false;
    }
    return program->name != NULL;
}

static void dsl_scan_reserved_usage_block(const me_dsl_block *block, int *uses_i_mask,
                                          int *uses_n_mask, bool *uses_ndim) {
    if (!block) {
        return;
    }
    for (int i = 0; i < block->nstmts; i++) {
        me_dsl_stmt *stmt = block->stmts[i];
        const char *expr_text = NULL;
        if (!stmt) {
            continue;
        }
        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN:
            expr_text = stmt->as.assign.value ? stmt->as.assign.value->text : NULL;
            break;
        case ME_DSL_STMT_EXPR:
            expr_text = stmt->as.expr_stmt.expr ? stmt->as.expr_stmt.expr->text : NULL;
            break;
        case ME_DSL_STMT_RETURN:
            expr_text = stmt->as.return_stmt.expr ? stmt->as.return_stmt.expr->text : NULL;
            break;
        case ME_DSL_STMT_PRINT:
            expr_text = stmt->as.print_stmt.call ? stmt->as.print_stmt.call->text : NULL;
            break;
        case ME_DSL_STMT_IF:
            expr_text = stmt->as.if_stmt.cond ? stmt->as.if_stmt.cond->text : NULL;
            break;
        case ME_DSL_STMT_FOR:
            expr_text = stmt->as.for_loop.limit ? stmt->as.for_loop.limit->text : NULL;
            break;
        case ME_DSL_STMT_BREAK:
        case ME_DSL_STMT_CONTINUE:
            expr_text = stmt->as.flow.cond ? stmt->as.flow.cond->text : NULL;
            break;
        }
        if (expr_text) {
            if (dsl_expr_uses_identifier(expr_text, "_ndim")) {
                *uses_ndim = true;
            }
            for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
                char name[8];
                snprintf(name, sizeof(name), "_i%d", d);
                if (dsl_expr_uses_identifier(expr_text, name)) {
                    *uses_i_mask |= (1 << d);
                }
                snprintf(name, sizeof(name), "_n%d", d);
                if (dsl_expr_uses_identifier(expr_text, name)) {
                    *uses_n_mask |= (1 << d);
                }
            }
        }
        if (stmt->kind == ME_DSL_STMT_IF) {
            dsl_scan_reserved_usage_block(&stmt->as.if_stmt.then_block, uses_i_mask, uses_n_mask, uses_ndim);
            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                dsl_scan_reserved_usage_block(&stmt->as.if_stmt.elif_branches[j].block,
                                              uses_i_mask, uses_n_mask, uses_ndim);
                const char *elif_text = stmt->as.if_stmt.elif_branches[j].cond
                                            ? stmt->as.if_stmt.elif_branches[j].cond->text
                                            : NULL;
                if (elif_text) {
                    if (dsl_expr_uses_identifier(elif_text, "_ndim")) {
                        *uses_ndim = true;
                    }
                    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
                        char name[8];
                        snprintf(name, sizeof(name), "_i%d", d);
                        if (dsl_expr_uses_identifier(elif_text, name)) {
                            *uses_i_mask |= (1 << d);
                        }
                        snprintf(name, sizeof(name), "_n%d", d);
                        if (dsl_expr_uses_identifier(elif_text, name)) {
                            *uses_n_mask |= (1 << d);
                        }
                    }
                }
            }
            if (stmt->as.if_stmt.has_else) {
                dsl_scan_reserved_usage_block(&stmt->as.if_stmt.else_block,
                                              uses_i_mask, uses_n_mask, uses_ndim);
            }
        }
        if (stmt->kind == ME_DSL_STMT_FOR) {
            dsl_scan_reserved_usage_block(&stmt->as.for_loop.body, uses_i_mask, uses_n_mask, uses_ndim);
        }
    }
}

static const char *dsl_skip_space_inline(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) {
        p++;
    }
    return p;
}

static bool dsl_is_ident_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static char *dsl_trim_copy(const char *start, const char *end) {
    if (!start || !end || end <= start) {
        return NULL;
    }
    while (start < end && isspace((unsigned char)*start)) {
        start++;
    }
    while (end > start && isspace((unsigned char)end[-1])) {
        end--;
    }
    size_t len = (size_t)(end - start);
    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, start, len);
    out[len] = '\0';
    return out;
}

static int dsl_utf8_encode(uint32_t cp, char *out, int cap) {
    if (cap < 1) return 0;
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF && cap >= 2) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF && cap >= 3) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF && cap >= 4) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

static bool dsl_parse_hex_digits(const char **p, int digits, uint32_t *out) {
    uint32_t value = 0;
    for (int i = 0; i < digits; i++) {
        char c = (*p)[i];
        uint32_t v;
        if (c >= '0' && c <= '9') v = (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v = (uint32_t)(10 + c - 'a');
        else if (c >= 'A' && c <= 'F') v = (uint32_t)(10 + c - 'A');
        else return false;
        value = (value << 4) | v;
    }
    *p += digits;
    *out = value;
    return true;
}

static char *dsl_unescape_string_literal(const char *text) {
    if (!text) {
        return NULL;
    }
    const char *p = dsl_skip_space_inline(text);
    if (!p || (*p != '"' && *p != '\'')) {
        return NULL;
    }
    char quote = *p++;
    size_t cap = 64;
    size_t len = 0;
    char *out = malloc(cap);
    if (!out) {
        return NULL;
    }
    while (*p && *p != quote) {
        uint32_t cp = 0;
        if (*p == '\\') {
            p++;
            if (!*p) {
                free(out);
                return NULL;
            }
            char esc = *p++;
            switch (esc) {
            case '\\': cp = '\\'; break;
            case '"': cp = '"'; break;
            case '\'': cp = '\''; break;
            case 'n': cp = '\n'; break;
            case 't': cp = '\t'; break;
            case 'u':
                if (!dsl_parse_hex_digits(&p, 4, &cp)) {
                    free(out);
                    return NULL;
                }
                break;
            case 'U':
                if (!dsl_parse_hex_digits(&p, 8, &cp)) {
                    free(out);
                    return NULL;
                }
                break;
            default:
                free(out);
                return NULL;
            }
        }
        else {
            cp = (unsigned char)*p++;
        }

        char utf8[4];
        int wrote = dsl_utf8_encode(cp, utf8, (int)sizeof(utf8));
        if (wrote <= 0) {
            free(out);
            return NULL;
        }
        if (len + (size_t)wrote + 1 > cap) {
            size_t next_cap = cap * 2;
            while (len + (size_t)wrote + 1 > next_cap) {
                next_cap *= 2;
            }
            char *next = realloc(out, next_cap);
            if (!next) {
                free(out);
                return NULL;
            }
            out = next;
            cap = next_cap;
        }
        memcpy(out + len, utf8, (size_t)wrote);
        len += (size_t)wrote;
    }
    if (*p != quote) {
        free(out);
        return NULL;
    }
    out[len] = '\0';
    return out;
}

static int dsl_count_placeholders(const char *fmt) {
    int count = 0;
    if (!fmt) return -1;
    for (size_t i = 0; fmt[i] != '\0'; i++) {
        if (fmt[i] == '{') {
            if (fmt[i + 1] == '{') {
                i++;
                continue;
            }
            if (fmt[i + 1] == '}') {
                count++;
                i++;
                continue;
            }
            return -1;
        }
        if (fmt[i] == '}') {
            if (fmt[i + 1] == '}') {
                i++;
                continue;
            }
            return -1;
        }
    }
    return count;
}

static bool dsl_split_print_args(const char *text, char ***out_args, int *out_nargs) {
    if (!text || !out_args || !out_nargs) {
        return false;
    }
    *out_args = NULL;
    *out_nargs = 0;

    const char *p = dsl_skip_space_inline(text);
    const char *ident = "print";
    size_t ident_len = strlen(ident);
    if (strncmp(p, ident, ident_len) != 0 || dsl_is_ident_char(p[ident_len])) {
        return false;
    }
    p += ident_len;
    p = dsl_skip_space_inline(p);
    if (*p != '(') {
        return false;
    }
    p++;

    const char *arg_start = p;
    int depth = 0;
    bool in_string = false;
    char quote = '\0';
    char **args = NULL;
    int nargs = 0;

    bool closed = false;
    for (; *p; p++) {
        char c = *p;
        if (in_string) {
            if (c == '\\' && p[1]) {
                p++;
                continue;
            }
            if (c == quote) {
                in_string = false;
            }
            continue;
        }
        if (c == '"' || c == '\'') {
            in_string = true;
            quote = c;
            continue;
        }
        if (c == '(') {
            depth++;
            continue;
        }
        if (c == ')') {
            if (depth == 0) {
                char *arg = dsl_trim_copy(arg_start, p);
                if (!arg) {
                    goto fail;
                }
                char **next_args = realloc(args, (size_t)(nargs + 1) * sizeof(*next_args));
                if (!next_args) {
                    free(arg);
                    goto fail;
                }
                args = next_args;
                args[nargs++] = arg;
                p++;
                closed = true;
                break;
            }
            depth--;
            continue;
        }
        if (c == ',' && depth == 0) {
            char *arg = dsl_trim_copy(arg_start, p);
            if (!arg) {
                goto fail;
            }
            char **next_args = realloc(args, (size_t)(nargs + 1) * sizeof(*next_args));
            if (!next_args) {
                free(arg);
                goto fail;
            }
            args = next_args;
            args[nargs++] = arg;
            arg_start = p + 1;
        }
    }

    if (!closed) {
        goto fail;
    }
    p = dsl_skip_space_inline(p);
    if (*p != '\0') {
        goto fail;
    }

    *out_args = args;
    *out_nargs = nargs;
    return true;

fail:
    if (args) {
        for (int i = 0; i < nargs; i++) {
            free(args[i]);
        }
    }
    free(args);
    return false;
}

static bool dsl_build_var_lookup(const me_dsl_var_table *table, const me_variable_ex *funcs,
                                 int n_funcs, me_variable_ex **out_vars, int *out_count) {
    if (!table || !out_vars || !out_count || n_funcs < 0) {
        return false;
    }
    int total = table->count + n_funcs;
    if (total == 0) {
        *out_vars = NULL;
        *out_count = 0;
        return true;
    }
    me_variable_ex *vars = calloc((size_t)total, sizeof(*vars));
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

static bool dsl_program_add_local(me_dsl_compiled_program *program, int var_index) {
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

static bool dsl_compile_expr(dsl_compile_ctx *ctx, const me_dsl_expr *expr_node,
                             me_dtype expr_dtype, me_dsl_compiled_expr *out_expr) {
    if (!ctx || !expr_node || !out_expr) {
        return false;
    }
    memset(out_expr, 0, sizeof(*out_expr));
    me_variable_ex *lookup = NULL;
    int lookup_count = 0;
    if (!dsl_build_var_lookup(&ctx->program->vars, ctx->funcs, ctx->func_count,
                              &lookup, &lookup_count)) {
        return false;
    }
    me_expr *compiled = NULL;
    int local_error = 0;
    int rc = private_compile_ex(expr_node->text, lookup, lookup_count,
                                NULL, 0, expr_dtype, &local_error, &compiled);
    free(lookup);
    if (rc != ME_COMPILE_SUCCESS || !compiled) {
        if (ctx->error_pos) {
            int offset = dsl_offset_from_linecol(ctx->source, expr_node->line, expr_node->column);
            if (offset >= 0 && local_error > 0) {
                *ctx->error_pos = offset + local_error - 1;
            }
            else {
                *ctx->error_pos = offset >= 0 ? offset : -1;
            }
        }
        if (compiled) {
            me_free(compiled);
        }
        return false;
    }
    int *indices = NULL;
    int count = 0;
    if (!dsl_collect_var_indices(compiled, &indices, &count)) {
        me_free(compiled);
        free(indices);
        return false;
    }
    out_expr->expr = compiled;
    out_expr->var_indices = indices;
    out_expr->n_vars = count;
    return true;
}

static bool dsl_jit_ir_resolve_dtype(void *resolve_ctx, const me_dsl_expr *expr,
                                     me_dtype *out_dtype) {
    dsl_compile_ctx *ctx = (dsl_compile_ctx *)resolve_ctx;
    if (!ctx || !expr || !out_dtype) {
        return false;
    }
    me_dsl_compiled_expr compiled_expr;
    me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
    int saved_error = 0;
    if (ctx->error_pos) {
        saved_error = *ctx->error_pos;
    }
    if (!dsl_compile_expr(ctx, expr, expr_dtype, &compiled_expr)) {
        if (ctx->error_pos) {
            *ctx->error_pos = saved_error;
        }
        return false;
    }
    *out_dtype = me_get_dtype(compiled_expr.expr);
    dsl_compiled_expr_free(&compiled_expr);
    if (ctx->error_pos) {
        *ctx->error_pos = saved_error;
    }
    return true;
}

static uint64_t dsl_jit_hash_bytes(uint64_t h, const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t dsl_jit_hash_i32(uint64_t h, int v) {
    return dsl_jit_hash_bytes(h, &v, sizeof(v));
}

static uint64_t dsl_jit_hash_u64(uint64_t h, uint64_t v) {
    return dsl_jit_hash_bytes(h, &v, sizeof(v));
}

static int dsl_jit_target_tag(void) {
#if defined(__APPLE__)
    return 1;
#elif defined(__linux__)
    return 2;
#else
    return 3;
#endif
}

static bool dsl_jit_force_libtcc(void) {
#if ME_USE_LIBTCC_FALLBACK
    const char *env = getenv("ME_DSL_JIT_FORCE_LIBTCC");
    if (!env || env[0] == '\0') {
        return false;
    }
    return strcmp(env, "0") != 0;
#else
    return false;
#endif
}

static int dsl_jit_backend_tag(void) {
    return dsl_jit_force_libtcc() ? 2 : 1;
}

static uint64_t dsl_jit_runtime_cache_key(const me_dsl_compiled_program *program) {
    uint64_t h = 1469598103934665603ULL;
    if (!program) {
        return h;
    }
    h = dsl_jit_hash_u64(h, program->jit_ir_fingerprint);
    h = dsl_jit_hash_i32(h, (int)program->output_dtype);
    h = dsl_jit_hash_i32(h, (int)program->fp_mode);
    h = dsl_jit_hash_i32(h, program->jit_nparams);
    if (program->jit_ir) {
        for (int i = 0; i < program->jit_ir->nparams; i++) {
            h = dsl_jit_hash_i32(h, (int)program->jit_ir->param_dtypes[i]);
        }
    }
    h = dsl_jit_hash_i32(h, (int)sizeof(void *));
    h = dsl_jit_hash_i32(h, ME_DSL_JIT_CGEN_VERSION);
    h = dsl_jit_hash_i32(h, dsl_jit_target_tag());
    h = dsl_jit_hash_i32(h, dsl_jit_backend_tag());
    return h;
}

#if !defined(_WIN32) && !defined(_WIN64)
/* In-process negative cache for recent JIT runtime failures. */
#define ME_DSL_JIT_NEG_CACHE_SLOTS 64
#define ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET 2
#define ME_DSL_JIT_NEG_CACHE_SHORT_COOLDOWN_SEC 10
#define ME_DSL_JIT_NEG_CACHE_LONG_COOLDOWN_SEC 120
#define ME_DSL_JIT_POS_CACHE_SLOTS 64
#define ME_DSL_JIT_META_MAGIC 0x4d454a49544d4554ULL
#define ME_DSL_JIT_META_VERSION 3

typedef enum {
    ME_DSL_JIT_NEG_FAIL_CACHE_DIR = 1,
    ME_DSL_JIT_NEG_FAIL_PATH = 2,
    ME_DSL_JIT_NEG_FAIL_WRITE = 3,
    ME_DSL_JIT_NEG_FAIL_COMPILE = 4,
    ME_DSL_JIT_NEG_FAIL_LOAD = 5,
    ME_DSL_JIT_NEG_FAIL_METADATA = 6
} me_dsl_jit_neg_failure_class;

typedef struct {
    bool valid;
    uint64_t key;
    uint64_t last_failure_at;
    uint64_t retry_after_at;
    uint8_t retries_left;
    uint8_t failure_class;
} me_dsl_jit_neg_cache_entry;

static me_dsl_jit_neg_cache_entry g_dsl_jit_neg_cache[ME_DSL_JIT_NEG_CACHE_SLOTS];
static int g_dsl_jit_neg_cache_cursor = 0;

typedef struct {
    bool valid;
    uint64_t key;
    void *handle;
    me_dsl_jit_kernel_fn kernel_fn;
} me_dsl_jit_pos_cache_entry;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t cgen_version;
    uint32_t target_tag;
    uint32_t ptr_size;
    uint64_t cache_key;
    uint64_t ir_fingerprint;
    int32_t output_dtype;
    int32_t dialect;
    int32_t fp_mode;
    int32_t nparams;
    int32_t param_dtypes[ME_MAX_VARS];
    uint64_t cc_hash;
} me_dsl_jit_cache_meta;

static me_dsl_jit_pos_cache_entry g_dsl_jit_pos_cache[ME_DSL_JIT_POS_CACHE_SLOTS];

static uint64_t dsl_jit_now_seconds(void) {
    time_t now = time(NULL);
    if (now < 0) {
        return 0;
    }
    return (uint64_t)now;
}

static int dsl_jit_neg_cache_find_slot(uint64_t key) {
    for (int i = 0; i < ME_DSL_JIT_NEG_CACHE_SLOTS; i++) {
        if (g_dsl_jit_neg_cache[i].valid && g_dsl_jit_neg_cache[i].key == key) {
            return i;
        }
    }
    return -1;
}

static int dsl_jit_neg_cache_alloc_slot(void) {
    for (int i = 0; i < ME_DSL_JIT_NEG_CACHE_SLOTS; i++) {
        if (!g_dsl_jit_neg_cache[i].valid) {
            return i;
        }
    }
    int slot = g_dsl_jit_neg_cache_cursor;
    g_dsl_jit_neg_cache_cursor = (g_dsl_jit_neg_cache_cursor + 1) % ME_DSL_JIT_NEG_CACHE_SLOTS;
    return slot;
}

static bool dsl_jit_neg_cache_should_skip(uint64_t key) {
    int slot = dsl_jit_neg_cache_find_slot(key);
    if (slot < 0) {
        return false;
    }
    me_dsl_jit_neg_cache_entry *entry = &g_dsl_jit_neg_cache[slot];
    uint64_t now = dsl_jit_now_seconds();
    if (now < entry->retry_after_at) {
        return true;
    }
    if (entry->retries_left == 0) {
        entry->retries_left = ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET;
    }
    return false;
}

static void dsl_jit_neg_cache_record_failure(uint64_t key, me_dsl_jit_neg_failure_class failure_class) {
    int slot = dsl_jit_neg_cache_find_slot(key);
    if (slot < 0) {
        slot = dsl_jit_neg_cache_alloc_slot();
    }
    me_dsl_jit_neg_cache_entry *entry = &g_dsl_jit_neg_cache[slot];
    if (!entry->valid || entry->key != key) {
        memset(entry, 0, sizeof(*entry));
        entry->key = key;
        entry->valid = true;
        entry->retries_left = ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET;
    }
    if (entry->retries_left > 0) {
        entry->retries_left--;
    }
    uint64_t now = dsl_jit_now_seconds();
    uint64_t cooldown = (entry->retries_left == 0)
        ? ME_DSL_JIT_NEG_CACHE_LONG_COOLDOWN_SEC
        : ME_DSL_JIT_NEG_CACHE_SHORT_COOLDOWN_SEC;
    entry->last_failure_at = now;
    entry->retry_after_at = now + cooldown;
    entry->failure_class = (uint8_t)failure_class;
}

static void dsl_jit_neg_cache_clear(uint64_t key) {
    int slot = dsl_jit_neg_cache_find_slot(key);
    if (slot < 0) {
        return;
    }
    memset(&g_dsl_jit_neg_cache[slot], 0, sizeof(g_dsl_jit_neg_cache[slot]));
}

static int dsl_jit_pos_cache_find_slot(uint64_t key) {
    for (int i = 0; i < ME_DSL_JIT_POS_CACHE_SLOTS; i++) {
        if (g_dsl_jit_pos_cache[i].valid && g_dsl_jit_pos_cache[i].key == key) {
            return i;
        }
    }
    return -1;
}

static int dsl_jit_pos_cache_find_free_slot(void) {
    for (int i = 0; i < ME_DSL_JIT_POS_CACHE_SLOTS; i++) {
        if (!g_dsl_jit_pos_cache[i].valid) {
            return i;
        }
    }
    return -1;
}

static bool dsl_jit_pos_cache_enabled(void) {
    const char *env = getenv("ME_DSL_JIT_POS_CACHE");
    if (!env || env[0] == '\0') {
        return true;
    }
    return strcmp(env, "0") != 0;
}

static bool dsl_jit_runtime_enabled(void) {
    const char *env = getenv("ME_DSL_JIT");
    if (!env || env[0] == '\0') {
        return true;
    }
    return strcmp(env, "0") != 0;
}

static bool dsl_jit_pos_cache_bind_program(me_dsl_compiled_program *program, uint64_t key) {
    if (!program) {
        return false;
    }
    int slot = dsl_jit_pos_cache_find_slot(key);
    if (slot < 0) {
        return false;
    }
    program->jit_dl_handle = g_dsl_jit_pos_cache[slot].handle;
    program->jit_kernel_fn = g_dsl_jit_pos_cache[slot].kernel_fn;
    program->jit_runtime_key = key;
    program->jit_dl_handle_cached = true;
    return true;
}

static bool dsl_jit_pos_cache_store_program(me_dsl_compiled_program *program, uint64_t key) {
    if (!program || !program->jit_dl_handle || !program->jit_kernel_fn) {
        return false;
    }

    int slot = dsl_jit_pos_cache_find_slot(key);
    if (slot >= 0) {
        if (program->jit_dl_handle != g_dsl_jit_pos_cache[slot].handle) {
            dlclose(program->jit_dl_handle);
            program->jit_dl_handle = g_dsl_jit_pos_cache[slot].handle;
            program->jit_kernel_fn = g_dsl_jit_pos_cache[slot].kernel_fn;
        }
        program->jit_runtime_key = key;
        program->jit_dl_handle_cached = true;
        return true;
    }

    slot = dsl_jit_pos_cache_find_free_slot();
    if (slot < 0) {
        program->jit_runtime_key = key;
        program->jit_dl_handle_cached = false;
        return false;
    }

    g_dsl_jit_pos_cache[slot].valid = true;
    g_dsl_jit_pos_cache[slot].key = key;
    g_dsl_jit_pos_cache[slot].handle = program->jit_dl_handle;
    g_dsl_jit_pos_cache[slot].kernel_fn = program->jit_kernel_fn;
    program->jit_runtime_key = key;
    program->jit_dl_handle_cached = true;
    return true;
}

static uint64_t dsl_jit_hash_cstr(uint64_t h, const char *s) {
    if (!s) {
        return dsl_jit_hash_i32(h, 0);
    }
    return dsl_jit_hash_bytes(h, s, strlen(s));
}

static uint64_t dsl_jit_cc_hash(me_dsl_fp_mode fp_mode) {
    const char *cc = getenv("CC");
    const char *jit_cflags = getenv("ME_DSL_JIT_CFLAGS");
    const char *fp_cflags = dsl_jit_fp_mode_cflags(fp_mode);
    if (!cc || cc[0] == '\0') {
        cc = "cc";
    }
    if (!jit_cflags) {
        jit_cflags = "";
    }
    if (!fp_cflags) {
        fp_cflags = "";
    }
    uint64_t h = dsl_jit_hash_cstr(1469598103934665603ULL, cc);
    h = dsl_jit_hash_cstr(h, fp_cflags);
    return dsl_jit_hash_cstr(h, jit_cflags);
}

static void dsl_jit_fill_cache_meta(me_dsl_jit_cache_meta *meta,
                                    const me_dsl_compiled_program *program,
                                    uint64_t key) {
    if (!meta) {
        return;
    }
    memset(meta, 0, sizeof(*meta));
    meta->magic = ME_DSL_JIT_META_MAGIC;
    meta->version = ME_DSL_JIT_META_VERSION;
    meta->cgen_version = ME_DSL_JIT_CGEN_VERSION;
    meta->target_tag = (uint32_t)dsl_jit_target_tag();
    meta->ptr_size = (uint32_t)sizeof(void *);
    meta->cache_key = key;
    if (!program) {
        return;
    }
    meta->ir_fingerprint = program->jit_ir_fingerprint;
    meta->output_dtype = (int32_t)program->output_dtype;
    meta->dialect = (int32_t)program->dialect;
    meta->fp_mode = (int32_t)program->fp_mode;
    meta->nparams = (int32_t)program->jit_nparams;
    for (int i = 0; i < ME_MAX_VARS; i++) {
        meta->param_dtypes[i] = -1;
    }
    if (program->jit_ir && program->jit_nparams > 0) {
        int n = program->jit_nparams;
        if (n > ME_MAX_VARS) {
            n = ME_MAX_VARS;
        }
        for (int i = 0; i < n; i++) {
            meta->param_dtypes[i] = (int32_t)program->jit_ir->param_dtypes[i];
        }
    }
    meta->cc_hash = dsl_jit_cc_hash(program->fp_mode);
}

static bool dsl_jit_write_meta_file(const char *path, const me_dsl_jit_cache_meta *meta) {
    if (!path || !meta) {
        return false;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    bool ok = (fwrite(meta, 1, sizeof(*meta), f) == sizeof(*meta));
    if (fclose(f) != 0) {
        ok = false;
    }
    return ok;
}

static bool dsl_jit_read_meta_file(const char *path, me_dsl_jit_cache_meta *out_meta) {
    if (!path || !out_meta) {
        return false;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    bool ok = (fread(out_meta, 1, sizeof(*out_meta), f) == sizeof(*out_meta));
    if (fclose(f) != 0) {
        ok = false;
    }
    return ok;
}

static bool dsl_jit_meta_file_matches(const char *path, const me_dsl_jit_cache_meta *expected) {
    if (!path || !expected) {
        return false;
    }
    me_dsl_jit_cache_meta actual;
    if (!dsl_jit_read_meta_file(path, &actual)) {
        return false;
    }
    return memcmp(&actual, expected, sizeof(actual)) == 0;
}

static bool dsl_jit_ensure_dir(const char *path) {
    if (!path || !path[0]) {
        return false;
    }
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    if (mkdir(path, 0700) == 0) {
        return true;
    }
    if (errno == EEXIST) {
        return true;
    }
    return false;
}

static bool dsl_jit_get_cache_dir(char *out, size_t out_size) {
    if (!out || out_size == 0) {
        return false;
    }
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || tmpdir[0] == '\0') {
        tmpdir = "/tmp";
    }
    if (snprintf(out, out_size, "%s/miniexpr-jit", tmpdir) >= (int)out_size) {
        return false;
    }
    return dsl_jit_ensure_dir(out);
}

static bool dsl_jit_write_text_file(const char *path, const char *text) {
    if (!path || !text) {
        return false;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    size_t n = strlen(text);
    bool ok = (fwrite(text, 1, n, f) == n);
    if (fclose(f) != 0) {
        ok = false;
    }
    return ok;
}

static bool dsl_jit_extract_command_name(const char *cmd, char *out, size_t out_size) {
    if (!cmd || !out || out_size == 0) {
        return false;
    }
    const char *p = cmd;
    while (*p && isspace((unsigned char)*p)) {
        p++;
    }
    if (*p == '\0') {
        return false;
    }
    char quote = '\0';
    if (*p == '"' || *p == '\'') {
        quote = *p++;
    }
    size_t n = 0;
    while (*p) {
        if (quote) {
            if (*p == quote) {
                break;
            }
        }
        else if (isspace((unsigned char)*p)) {
            break;
        }
        if (n + 1 >= out_size) {
            return false;
        }
        out[n++] = *p++;
    }
    out[n] = '\0';
    return n > 0;
}

static bool dsl_jit_command_exists(const char *cmd) {
    char tool[512];
    if (!dsl_jit_extract_command_name(cmd, tool, sizeof(tool))) {
        return false;
    }
    if (strchr(tool, '/')) {
        return access(tool, X_OK) == 0;
    }
    const char *path = getenv("PATH");
    if (!path || path[0] == '\0') {
        return false;
    }
    char candidate[1024];
    const char *seg = path;
    while (seg && seg[0] != '\0') {
        const char *next = strchr(seg, ':');
        size_t seg_len = next ? (size_t)(next - seg) : strlen(seg);
        if (seg_len == 0) {
            seg = next ? next + 1 : NULL;
            continue;
        }
        if (seg_len + 1 + strlen(tool) + 1 < sizeof(candidate)) {
            memcpy(candidate, seg, seg_len);
            candidate[seg_len] = '/';
            strcpy(candidate + seg_len + 1, tool);
            if (access(candidate, X_OK) == 0) {
                return true;
            }
        }
        seg = next ? next + 1 : NULL;
    }
    return false;
}

static bool dsl_jit_c_compiler_available(void) {
    const char *cc = getenv("CC");
    if (!cc || cc[0] == '\0') {
        cc = "cc";
    }
    return dsl_jit_command_exists(cc);
}

#if ME_USE_LIBTCC_FALLBACK
typedef struct TCCState me_tcc_state;

typedef me_tcc_state *(*me_tcc_new_fn)(void);
typedef void (*me_tcc_delete_fn)(me_tcc_state *s);
typedef int (*me_tcc_set_output_type_fn)(me_tcc_state *s, int output_type);
typedef int (*me_tcc_compile_string_fn)(me_tcc_state *s, const char *buf);
typedef int (*me_tcc_relocate_fn)(me_tcc_state *s);
typedef void *(*me_tcc_get_symbol_fn)(me_tcc_state *s, const char *name);
typedef int (*me_tcc_set_options_fn)(me_tcc_state *s, const char *str);
typedef int (*me_tcc_add_library_fn)(me_tcc_state *s, const char *libraryname);
typedef void (*me_tcc_set_lib_path_fn)(me_tcc_state *s, const char *path);

typedef struct {
    bool attempted;
    bool available;
    void *handle;
    me_tcc_new_fn tcc_new_fn;
    me_tcc_delete_fn tcc_delete_fn;
    me_tcc_set_output_type_fn tcc_set_output_type_fn;
    me_tcc_compile_string_fn tcc_compile_string_fn;
    me_tcc_relocate_fn tcc_relocate_fn;
    me_tcc_get_symbol_fn tcc_get_symbol_fn;
    me_tcc_set_options_fn tcc_set_options_fn;
    me_tcc_add_library_fn tcc_add_library_fn;
    me_tcc_set_lib_path_fn tcc_set_lib_path_fn;
    char error[160];
} me_dsl_tcc_api;

static me_dsl_tcc_api g_dsl_tcc_api;

static const char *dsl_jit_libtcc_error_message(void) {
    if (g_dsl_tcc_api.error[0]) {
        return g_dsl_tcc_api.error;
    }
    return "libtcc fallback unavailable";
}

static bool dsl_jit_libtcc_enabled(void) {
    const char *env = getenv("ME_DSL_JIT_LIBTCC");
    if (!env || env[0] == '\0') {
        return true;
    }
    return strcmp(env, "0") != 0;
}

static bool dsl_jit_path_dirname(const char *path, char *out, size_t out_size) {
    if (!path || !out || out_size == 0) {
        return false;
    }
    const char *slash = strrchr(path, '/');
    if (!slash) {
        if (out_size < 2) {
            return false;
        }
        out[0] = '.';
        out[1] = '\0';
        return true;
    }
    if (slash == path) {
        if (out_size < 2) {
            return false;
        }
        out[0] = '/';
        out[1] = '\0';
        return true;
    }
    size_t len = (size_t)(slash - path);
    if (len + 1 > out_size) {
        return false;
    }
    memcpy(out, path, len);
    out[len] = '\0';
    return true;
}

static bool dsl_jit_libtcc_path_near_self(char *out, size_t out_size) {
    if (!out || out_size == 0) {
        return false;
    }
    Dl_info info;
    if (dladdr((void *)&dsl_jit_libtcc_enabled, &info) == 0 ||
        !info.dli_fname || info.dli_fname[0] == '\0') {
        return false;
    }
    char dir[PATH_MAX];
    if (!dsl_jit_path_dirname(info.dli_fname, dir, sizeof(dir))) {
        return false;
    }
#if defined(__APPLE__)
    const char *name = "libtcc.dylib";
#else
    const char *name = "libtcc.so";
#endif
    int n = snprintf(out, out_size, "%s/%s", dir, name);
    return n > 0 && (size_t)n < out_size;
}

static bool dsl_jit_libtcc_runtime_dir(char *out, size_t out_size) {
    if (!out || out_size == 0) {
        return false;
    }
    const char *env = getenv("ME_DSL_JIT_TCC_LIB_PATH");
    if (env && env[0] != '\0') {
        int n = snprintf(out, out_size, "%s", env);
        return n > 0 && (size_t)n < out_size;
    }
    if (!g_dsl_tcc_api.tcc_new_fn) {
        return false;
    }
    Dl_info info;
    if (dladdr((void *)g_dsl_tcc_api.tcc_new_fn, &info) == 0 ||
        !info.dli_fname || info.dli_fname[0] == '\0') {
        return false;
    }
    return dsl_jit_path_dirname(info.dli_fname, out, out_size);
}

static bool dsl_jit_libtcc_load_api(void) {
    if (g_dsl_tcc_api.attempted) {
        return g_dsl_tcc_api.available;
    }
    g_dsl_tcc_api.attempted = true;
    if (!dsl_jit_libtcc_enabled()) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "libtcc fallback disabled by environment");
        return false;
    }

    const char *env_path = getenv("ME_DSL_JIT_LIBTCC_PATH");
    const char *default_path = ME_DSL_JIT_LIBTCC_DEFAULT_PATH;
    const char *candidates[9];
    char self_candidate[PATH_MAX];
    int ncandidates = 0;
    if (env_path && env_path[0] != '\0') {
        candidates[ncandidates++] = env_path;
    }
    if (default_path && default_path[0] != '\0') {
        candidates[ncandidates++] = default_path;
    }
    if (dsl_jit_libtcc_path_near_self(self_candidate, sizeof(self_candidate))) {
        candidates[ncandidates++] = self_candidate;
    }
#if defined(__APPLE__)
    candidates[ncandidates++] = "libtcc.dylib";
    candidates[ncandidates++] = "libtcc.so";
    candidates[ncandidates++] = "libtcc.so.1";
#else
    candidates[ncandidates++] = "libtcc.so";
    candidates[ncandidates++] = "libtcc.so.1";
#endif
    candidates[ncandidates] = NULL;

    void *handle = NULL;
    for (int i = 0; i < ncandidates; i++) {
        handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            break;
        }
    }
    if (!handle) {
        const char *err = dlerror();
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error),
                 "failed to load libtcc shared library%s%s",
                 err ? ": " : "", err ? err : "");
        return false;
    }

#define ME_LOAD_TCC_SYM(field, sym_name, fn_type) \
    do { \
        g_dsl_tcc_api.field = (fn_type)dlsym(handle, sym_name); \
        if (!g_dsl_tcc_api.field) { \
            dlclose(handle); \
            snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), \
                     "libtcc missing required symbol %s", sym_name); \
            return false; \
        } \
    } while (0)

    ME_LOAD_TCC_SYM(tcc_new_fn, "tcc_new", me_tcc_new_fn);
    ME_LOAD_TCC_SYM(tcc_delete_fn, "tcc_delete", me_tcc_delete_fn);
    ME_LOAD_TCC_SYM(tcc_set_output_type_fn, "tcc_set_output_type", me_tcc_set_output_type_fn);
    ME_LOAD_TCC_SYM(tcc_compile_string_fn, "tcc_compile_string", me_tcc_compile_string_fn);
    ME_LOAD_TCC_SYM(tcc_relocate_fn, "tcc_relocate", me_tcc_relocate_fn);
    ME_LOAD_TCC_SYM(tcc_get_symbol_fn, "tcc_get_symbol", me_tcc_get_symbol_fn);
#undef ME_LOAD_TCC_SYM

    g_dsl_tcc_api.tcc_set_options_fn = (me_tcc_set_options_fn)dlsym(handle, "tcc_set_options");
    g_dsl_tcc_api.tcc_add_library_fn = (me_tcc_add_library_fn)dlsym(handle, "tcc_add_library");
    g_dsl_tcc_api.tcc_set_lib_path_fn = (me_tcc_set_lib_path_fn)dlsym(handle, "tcc_set_lib_path");
    g_dsl_tcc_api.handle = handle;
    g_dsl_tcc_api.available = true;
    g_dsl_tcc_api.error[0] = '\0';
    return true;
}

static void dsl_jit_libtcc_delete_state(void *state) {
    if (!state) {
        return;
    }
    if (!dsl_jit_libtcc_load_api() || !g_dsl_tcc_api.tcc_delete_fn) {
        return;
    }
    g_dsl_tcc_api.tcc_delete_fn((me_tcc_state *)state);
}

static bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program) {
    if (!program || !program->jit_c_source) {
        return false;
    }
    if (program->fp_mode != ME_DSL_FP_STRICT) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "libtcc fallback supports only strict fp mode");
        return false;
    }
    if (!dsl_jit_libtcc_load_api()) {
        return false;
    }
    me_tcc_state *state = g_dsl_tcc_api.tcc_new_fn();
    if (!state) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_new failed");
        return false;
    }

    char tcc_lib_dir[PATH_MAX];
    if (g_dsl_tcc_api.tcc_set_lib_path_fn &&
        dsl_jit_libtcc_runtime_dir(tcc_lib_dir, sizeof(tcc_lib_dir))) {
        g_dsl_tcc_api.tcc_set_lib_path_fn(state, tcc_lib_dir);
    }

    const char *tcc_opts = getenv("ME_DSL_JIT_TCC_OPTIONS");
    if (g_dsl_tcc_api.tcc_set_options_fn && tcc_opts && tcc_opts[0] != '\0') {
        (void)g_dsl_tcc_api.tcc_set_options_fn(state, tcc_opts);
    }
    if (g_dsl_tcc_api.tcc_set_output_type_fn(state, 1) < 0) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_set_output_type failed");
        return false;
    }
#if !defined(__APPLE__)
    if (g_dsl_tcc_api.tcc_add_library_fn) {
        (void)g_dsl_tcc_api.tcc_add_library_fn(state, "m");
    }
#endif
    if (g_dsl_tcc_api.tcc_compile_string_fn(state, program->jit_c_source) < 0) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_compile_string failed");
        return false;
    }
    if (g_dsl_tcc_api.tcc_relocate_fn(state) < 0) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_relocate failed");
        return false;
    }
    void *sym = g_dsl_tcc_api.tcc_get_symbol_fn(state, ME_DSL_JIT_SYMBOL_NAME);
    if (!sym) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_get_symbol failed");
        return false;
    }

    if (program->jit_tcc_state) {
        dsl_jit_libtcc_delete_state(program->jit_tcc_state);
    }
    program->jit_tcc_state = state;
    program->jit_kernel_fn = (me_dsl_jit_kernel_fn)sym;
    program->jit_c_error_line = 0;
    program->jit_c_error_column = 0;
    program->jit_c_error[0] = '\0';
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    return true;
}
#else
static const char *dsl_jit_libtcc_error_message(void) {
    return "libtcc fallback not built";
}

static void dsl_jit_libtcc_delete_state(void *state) {
    (void)state;
}

static bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program) {
    (void)program;
    return false;
}
#endif

static bool dsl_jit_compile_shared(const me_dsl_compiled_program *program,
                                   const char *src_path, const char *so_path) {
    if (!program || !src_path || !so_path) {
        return false;
    }
    const char *cc = getenv("CC");
    const char *jit_cflags = getenv("ME_DSL_JIT_CFLAGS");
    const char *fp_cflags = dsl_jit_fp_mode_cflags(program->fp_mode);
    if (!cc || cc[0] == '\0') {
        cc = "cc";
    }
    if (!jit_cflags) {
        jit_cflags = "";
    }
    if (!fp_cflags) {
        fp_cflags = "";
    }
    char cmd[2048];
#if defined(__APPLE__)
    int n = snprintf(cmd, sizeof(cmd),
                     "%s -std=c99 -O3 -fPIC %s %s -dynamiclib -o \"%s\" \"%s\" >/dev/null 2>&1",
                     cc, fp_cflags, jit_cflags, so_path, src_path);
#else
    int n = snprintf(cmd, sizeof(cmd),
                     "%s -std=c99 -O3 -fPIC %s %s -shared -o \"%s\" \"%s\" >/dev/null 2>&1",
                     cc, fp_cflags, jit_cflags, so_path, src_path);
#endif
    if (n <= 0 || (size_t)n >= sizeof(cmd)) {
        return false;
    }
    int rc = system(cmd);
    return rc == 0;
}

static bool dsl_jit_load_kernel(me_dsl_compiled_program *program, const char *shared_path) {
    if (!program || !shared_path) {
        return false;
    }
    void *handle = dlopen(shared_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        return false;
    }
    void *sym = dlsym(handle, ME_DSL_JIT_SYMBOL_NAME);
    if (!sym) {
        dlclose(handle);
        return false;
    }
    program->jit_dl_handle = handle;
    program->jit_kernel_fn = (me_dsl_jit_kernel_fn)sym;
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    return true;
}

static void dsl_try_prepare_jit_runtime(me_dsl_compiled_program *program) {
    if (!program || !program->jit_ir || !program->jit_c_source) {
        return;
    }
    if (program->output_is_scalar) {
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=scalar output",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (program->uses_i_mask || program->uses_n_mask || program->uses_ndim) {
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=reserved index vars used",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (program->jit_nparams != program->jit_ir->nparams) {
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=parameter metadata mismatch",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (!dsl_jit_runtime_enabled()) {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime disabled by environment");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }

    uint64_t key = dsl_jit_runtime_cache_key(program);
    if (dsl_jit_pos_cache_enabled() && dsl_jit_pos_cache_bind_program(program, key)) {
        dsl_tracef("jit runtime hit: dialect=%s fp=%s source=process-cache key=%016llx",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
        dsl_jit_neg_cache_clear(key);
        return;
    }
    if (dsl_jit_neg_cache_should_skip(key)) {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime skipped after recent failure");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s key=%016llx",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error,
                   (unsigned long long)key);
        return;
    }

    char cache_dir[1024];
    if (!dsl_jit_get_cache_dir(cache_dir, sizeof(cache_dir))) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_CACHE_DIR);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime cache directory unavailable");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }

    const char *ext = "so";
#if defined(__APPLE__)
    ext = "dylib";
#endif
    char src_path[1300];
    char so_path[1300];
    char meta_path[1300];
    if (snprintf(src_path, sizeof(src_path), "%s/kernel_%016llx.c",
                 cache_dir, (unsigned long long)key) >= (int)sizeof(src_path) ||
        snprintf(so_path, sizeof(so_path), "%s/kernel_%016llx.%s",
                 cache_dir, (unsigned long long)key, ext) >= (int)sizeof(so_path) ||
        snprintf(meta_path, sizeof(meta_path), "%s/kernel_%016llx.meta",
                 cache_dir, (unsigned long long)key) >= (int)sizeof(meta_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_PATH);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime cache path too long");
        dsl_tracef("jit runtime skip: dialect=%s reason=%s",
                   dsl_dialect_name(program->dialect), program->jit_c_error);
        return;
    }

    me_dsl_jit_cache_meta expected_meta;
    dsl_jit_fill_cache_meta(&expected_meta, program, key);

    bool so_exists = (access(so_path, F_OK) == 0);
    bool meta_matches = so_exists && dsl_jit_meta_file_matches(meta_path, &expected_meta);
    if (meta_matches) {
        if (dsl_jit_load_kernel(program, so_path)) {
            if (dsl_jit_pos_cache_enabled()) {
                (void)dsl_jit_pos_cache_store_program(program, key);
            }
            dsl_tracef("jit runtime hit: dialect=%s fp=%s source=disk-cache key=%016llx",
                       dsl_dialect_name(program->dialect),
                       dsl_fp_mode_name(program->fp_mode),
                       (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
        dsl_tracef("jit runtime cache reload failed: dialect=%s fp=%s key=%016llx",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
    }

    if (dsl_jit_force_libtcc()) {
        if (dsl_jit_compile_libtcc_in_memory(program)) {
            dsl_tracef("jit runtime built: dialect=%s fp=%s source=libtcc-forced key=%016llx",
                       dsl_dialect_name(program->dialect),
                       dsl_fp_mode_name(program->fp_mode),
                       (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime forced libtcc compilation failed");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s detail=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode),
                   program->jit_c_error,
                   dsl_jit_libtcc_error_message());
        return;
    }

    if (!dsl_jit_c_compiler_available()) {
        if (dsl_jit_compile_libtcc_in_memory(program)) {
            dsl_tracef("jit runtime built: dialect=%s fp=%s source=libtcc-in-memory key=%016llx",
                       dsl_dialect_name(program->dialect),
                       dsl_fp_mode_name(program->fp_mode),
                       (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_tracef("jit runtime fallback miss: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_jit_libtcc_error_message());
    }

    if (!dsl_jit_write_text_file(src_path, program->jit_c_source)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_WRITE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write source");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (!dsl_jit_compile_shared(program, src_path, so_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime compilation failed");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (!dsl_jit_write_meta_file(meta_path, &expected_meta)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_METADATA);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write cache metadata");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (!dsl_jit_load_kernel(program, so_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime shared object load failed");
        dsl_tracef("jit runtime skip: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (dsl_jit_pos_cache_enabled()) {
        (void)dsl_jit_pos_cache_store_program(program, key);
    }
    dsl_tracef("jit runtime built: dialect=%s fp=%s key=%016llx",
               dsl_dialect_name(program->dialect),
               dsl_fp_mode_name(program->fp_mode),
               (unsigned long long)key);
    dsl_jit_neg_cache_clear(key);
}
#else
static void dsl_try_prepare_jit_runtime(me_dsl_compiled_program *program) {
    (void)program;
}
#endif

static void dsl_try_build_jit_ir(dsl_compile_ctx *ctx, const me_dsl_program *parsed,
                                 me_dsl_compiled_program *program) {
    if (!ctx || !parsed || !program) {
        return;
    }

    program->jit_ir = NULL;
    program->jit_ir_fingerprint = 0;
    program->jit_ir_error_line = 0;
    program->jit_ir_error_column = 0;
    program->jit_ir_error[0] = '\0';
    free(program->jit_param_input_indices);
    program->jit_param_input_indices = NULL;
    program->jit_nparams = 0;
    program->jit_kernel_fn = NULL;
#if !defined(_WIN32) && !defined(_WIN64)
    if (program->jit_tcc_state) {
        dsl_jit_libtcc_delete_state(program->jit_tcc_state);
        program->jit_tcc_state = NULL;
    }
    if (program->jit_dl_handle) {
        if (!program->jit_dl_handle_cached) {
            dlclose(program->jit_dl_handle);
        }
    }
#endif
    program->jit_dl_handle = NULL;
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    free(program->jit_c_source);
    program->jit_c_source = NULL;
    program->jit_c_error_line = 0;
    program->jit_c_error_column = 0;
    program->jit_c_error[0] = '\0';

    if (parsed->nparams < 0) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "invalid dsl parameter metadata");
        dsl_tracef("jit ir skip: dialect=%s fp=%s reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
        return;
    }

    const char **param_names = NULL;
    me_dtype *param_dtypes = NULL;
    int *param_input_indices = NULL;

    if (parsed->nparams > 0) {
        param_names = calloc((size_t)parsed->nparams, sizeof(*param_names));
        param_dtypes = calloc((size_t)parsed->nparams, sizeof(*param_dtypes));
        param_input_indices = calloc((size_t)parsed->nparams, sizeof(*param_input_indices));
        if (!param_names || !param_dtypes || !param_input_indices) {
            snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                     "out of memory building jit ir metadata");
            dsl_tracef("jit ir skip: dialect=%s fp=%s reason=%s",
                       dsl_dialect_name(program->dialect),
                       dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
            free(param_names);
            free(param_dtypes);
            free(param_input_indices);
            return;
        }
        for (int i = 0; i < parsed->nparams; i++) {
            int idx = dsl_var_table_find(&program->vars, parsed->params[i]);
            if (idx < 0 || idx >= program->vars.count) {
                snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                         "failed to resolve dsl parameter dtype for jit ir");
                dsl_tracef("jit ir skip: dialect=%s fp=%s reason=%s",
                           dsl_dialect_name(program->dialect),
                           dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
                free(param_names);
                free(param_dtypes);
                free(param_input_indices);
                return;
            }
            param_names[i] = parsed->params[i];
            param_dtypes[i] = program->vars.dtypes[idx];
            param_input_indices[i] = idx;
        }
    }

    me_dsl_error ir_error;
    memset(&ir_error, 0, sizeof(ir_error));
    me_dsl_jit_ir_program *jit_ir = NULL;
    bool ok = me_dsl_jit_ir_build(parsed, param_names, param_dtypes, parsed->nparams,
                                  dsl_jit_ir_resolve_dtype, ctx, &jit_ir, &ir_error);

    free(param_names);
    free(param_dtypes);

    if (!ok || !jit_ir) {
        program->jit_ir_error_line = ir_error.line;
        program->jit_ir_error_column = ir_error.column;
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 ir_error.message[0] ? ir_error.message : "jit ir build rejected");
        dsl_tracef("jit ir reject: dialect=%s fp=%s at %d:%d reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode),
                   program->jit_ir_error_line, program->jit_ir_error_column,
                   program->jit_ir_error);
        me_dsl_jit_ir_free(jit_ir);
        free(param_input_indices);
        return;
    }

    program->jit_ir = jit_ir;
    program->jit_ir_fingerprint = me_dsl_jit_ir_fingerprint(jit_ir);
    program->jit_param_input_indices = param_input_indices;
    program->jit_nparams = parsed->nparams;

    me_dsl_error cg_error;
    memset(&cg_error, 0, sizeof(cg_error));
    me_dsl_jit_cgen_options cg_options;
    cg_options.symbol_name = ME_DSL_JIT_SYMBOL_NAME;
    char *generated_c = NULL;
    bool cg_ok = me_dsl_jit_codegen_c(jit_ir, ctx->return_dtype, &cg_options,
                                      &generated_c, &cg_error);
    if (!cg_ok || !generated_c) {
        program->jit_c_error_line = cg_error.line;
        program->jit_c_error_column = cg_error.column;
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 cg_error.message[0] ? cg_error.message : "jit c codegen rejected");
        dsl_tracef("jit codegen reject: dialect=%s fp=%s at %d:%d reason=%s",
                   dsl_dialect_name(program->dialect),
                   dsl_fp_mode_name(program->fp_mode),
                   program->jit_c_error_line, program->jit_c_error_column,
                   program->jit_c_error);
        free(generated_c);
        free(program->jit_param_input_indices);
        program->jit_param_input_indices = NULL;
        program->jit_nparams = 0;
        return;
    }
    program->jit_c_source = generated_c;
    dsl_tracef("jit ir built: dialect=%s fp=%s fingerprint=%016llx",
               dsl_dialect_name(program->dialect),
               dsl_fp_mode_name(program->fp_mode),
               (unsigned long long)program->jit_ir_fingerprint);
    dsl_try_prepare_jit_runtime(program);
}

static bool dsl_block_guarantees_return(const me_dsl_block *block);

static bool dsl_stmt_guarantees_return(const me_dsl_stmt *stmt) {
    if (!stmt) {
        return false;
    }
    switch (stmt->kind) {
    case ME_DSL_STMT_RETURN:
        return true;
    case ME_DSL_STMT_IF:
        if (!stmt->as.if_stmt.has_else) {
            return false;
        }
        if (!dsl_block_guarantees_return(&stmt->as.if_stmt.then_block)) {
            return false;
        }
        for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
            if (!dsl_block_guarantees_return(&stmt->as.if_stmt.elif_branches[i].block)) {
                return false;
            }
        }
        if (!dsl_block_guarantees_return(&stmt->as.if_stmt.else_block)) {
            return false;
        }
        return true;
    case ME_DSL_STMT_FOR:
    case ME_DSL_STMT_ASSIGN:
    case ME_DSL_STMT_EXPR:
    case ME_DSL_STMT_PRINT:
    case ME_DSL_STMT_BREAK:
    case ME_DSL_STMT_CONTINUE:
        return false;
    }
    return false;
}

static bool dsl_block_guarantees_return(const me_dsl_block *block) {
    if (!block) {
        return false;
    }
    for (int i = 0; i < block->nstmts; i++) {
        if (dsl_stmt_guarantees_return(block->stmts[i])) {
            return true;
        }
    }
    return false;
}

static void dsl_block_first_linecol(const me_dsl_block *block, int *line, int *column) {
    if (line) {
        *line = 0;
    }
    if (column) {
        *column = 0;
    }
    if (!block || block->nstmts <= 0 || !block->stmts || !block->stmts[0]) {
        return;
    }
    if (line) {
        *line = block->stmts[0]->line;
    }
    if (column) {
        *column = block->stmts[0]->column;
    }
}

static bool dsl_compile_block(dsl_compile_ctx *ctx, const me_dsl_block *block,
                              me_dsl_compiled_block *out_block) {
    if (!ctx || !block || !out_block) {
        return false;
    }
    memset(out_block, 0, sizeof(*out_block));
    for (int i = 0; i < block->nstmts; i++) {
        me_dsl_stmt *stmt = block->stmts[i];
        if (!stmt) {
            continue;
        }
        me_dsl_compiled_stmt *compiled = calloc(1, sizeof(*compiled));
        if (!compiled) {
            return false;
        }
        compiled->kind = stmt->kind;
        compiled->line = stmt->line;
        compiled->column = stmt->column;

        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN: {
            const char *name = stmt->as.assign.name;
            if (!name) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (dsl_is_reserved_name(name)) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            int var_index = dsl_var_table_find(&ctx->program->vars, name);
            if (var_index >= 0 && var_index < ctx->program->n_inputs) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
            if (!dsl_compile_expr(ctx, stmt->as.assign.value, expr_dtype, &compiled->as.assign.value)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            me_dtype assigned_dtype = me_get_dtype(compiled->as.assign.value.expr);
            bool is_uniform = dsl_expr_is_uniform(compiled->as.assign.value.expr,
                                                  ctx->program->vars.uniform,
                                                  ctx->program->vars.count);

            if (var_index < 0) {
                if (!ctx->allow_new_locals) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                var_index = dsl_var_table_add_with_uniform(&ctx->program->vars, name,
                                                           assigned_dtype, 0, is_uniform);
                if (var_index < 0) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            else if (ctx->program->vars.dtypes[var_index] != assigned_dtype) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            else {
                ctx->program->vars.uniform[var_index] = is_uniform;
            }

            if (!dsl_program_add_local(ctx->program, var_index)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            compiled->as.assign.local_slot = ctx->program->local_slots[var_index];
            break;
        }
        case ME_DSL_STMT_EXPR: {
            me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
            if (!dsl_compile_expr(ctx, stmt->as.expr_stmt.expr, expr_dtype, &compiled->as.expr_stmt.expr)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            break;
        }
        case ME_DSL_STMT_RETURN: {
            if (ctx->dialect == ME_DSL_DIALECT_ELEMENT && ctx->loop_depth > 0) {
                dsl_tracef("compile reject: dialect=%s does not support return inside loops at %d:%d",
                           dsl_dialect_name(ctx->dialect), stmt->line, stmt->column);
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
            if (!dsl_compile_expr(ctx, stmt->as.return_stmt.expr, expr_dtype, &compiled->as.return_stmt.expr)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            me_dtype return_dtype = me_get_dtype(compiled->as.return_stmt.expr.expr);
            if (!ctx->has_return) {
                ctx->has_return = true;
                ctx->return_dtype = return_dtype;
                ctx->output_expr = &compiled->as.return_stmt.expr;
            }
            else if (ctx->return_dtype != return_dtype) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            break;
        }
        case ME_DSL_STMT_PRINT: {
            const char *call = stmt->as.print_stmt.call ? stmt->as.print_stmt.call->text : NULL;
            char **args = NULL;
            int nargs = 0;
            if (!call || !dsl_split_print_args(call, &args, &nargs) || nargs < 1) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            char *format = NULL;
            int arg_count = nargs - 1;
            bool first_is_string = false;
            if (args[0][0] == '"' || args[0][0] == '\'') {
                format = dsl_unescape_string_literal(args[0]);
                if (!format) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                first_is_string = true;
            }
            if (!first_is_string) {
                size_t fmt_len = 0;
                for (int i = 0; i < nargs; i++) {
                    fmt_len += (i == 0) ? 2 : 3;
                }
                format = malloc(fmt_len + 1);
                if (!format) {
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                char *out = format;
                for (int i = 0; i < nargs; i++) {
                    if (i > 0) {
                        *out++ = ' ';
                    }
                    *out++ = '{';
                    *out++ = '}';
                }
                *out = '\0';
                arg_count = nargs;
            }
            int placeholder_count = dsl_count_placeholders(format);
            if (placeholder_count < 0) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                free(format);
                for (int i = 0; i < nargs; i++) {
                    free(args[i]);
                }
                free(args);
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (first_is_string && placeholder_count == 0 && arg_count > 0) {
                bool needs_space = format[0] != '\0' && !isspace((unsigned char)format[strlen(format) - 1]);
                size_t fmt_len = strlen(format);
                size_t extra = (needs_space ? 1 : 0) + (size_t)(arg_count * 2) + (size_t)(arg_count - 1);
                char *expanded = malloc(fmt_len + extra + 1);
                if (!expanded) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    free(format);
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                memcpy(expanded, format, fmt_len);
                char *out = expanded + fmt_len;
                if (needs_space) {
                    *out++ = ' ';
                }
                for (int i = 0; i < arg_count; i++) {
                    if (i > 0) {
                        *out++ = ' ';
                    }
                    *out++ = '{';
                    *out++ = '}';
                }
                *out = '\0';
                free(format);
                format = expanded;
                placeholder_count = arg_count;
            }
            if (placeholder_count != arg_count) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                free(format);
                for (int i = 0; i < nargs; i++) {
                    free(args[i]);
                }
                free(args);
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            if (arg_count > 0) {
                compiled->as.print_stmt.args = calloc((size_t)arg_count, sizeof(*compiled->as.print_stmt.args));
                if (!compiled->as.print_stmt.args) {
                    free(format);
                    for (int i = 0; i < nargs; i++) {
                        free(args[i]);
                    }
                    free(args);
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            compiled->as.print_stmt.format = format;
            compiled->as.print_stmt.nargs = arg_count;

            for (int i = 0; i < arg_count; i++) {
                int arg_index = first_is_string ? (i + 1) : i;
                me_dsl_expr temp_expr = {
                    .text = args[arg_index],
                    .line = stmt->line,
                    .column = stmt->column
                };
                me_dtype expr_dtype = ctx->output_dtype_auto ? ME_AUTO : ctx->output_dtype;
                if (!dsl_compile_expr(ctx, &temp_expr, expr_dtype, &compiled->as.print_stmt.args[i])) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    dsl_compiled_stmt_free(compiled);
                    for (int j = 0; j < nargs; j++) {
                        free(args[j]);
                    }
                    free(args);
                    return false;
                }
                if (!dsl_expr_is_uniform(compiled->as.print_stmt.args[i].expr,
                                         ctx->program->vars.uniform,
                                         ctx->program->vars.count)) {
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                    }
                    dsl_compiled_stmt_free(compiled);
                    for (int j = 0; j < nargs; j++) {
                        free(args[j]);
                    }
                    free(args);
                    return false;
                }
            }
            for (int i = 0; i < nargs; i++) {
                free(args[i]);
            }
            free(args);
            break;
        }
        case ME_DSL_STMT_IF: {
            /* Element dialect allows per-item loop control conditions inside loops. */
            bool require_uniform_cond = (ctx->dialect == ME_DSL_DIALECT_VECTOR ||
                                         ctx->loop_depth <= 0);
            if (!dsl_compile_expr(ctx, stmt->as.if_stmt.cond, ME_AUTO, &compiled->as.if_stmt.cond)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (require_uniform_cond &&
                !dsl_expr_is_uniform(compiled->as.if_stmt.cond.expr,
                                     ctx->program->vars.uniform,
                                     ctx->program->vars.count)) {
                dsl_tracef("compile reject: dialect=%s requires uniform loop condition at %d:%d; "
                           "use '# me:dialect=element' for per-item loop conditions",
                           dsl_dialect_name(ctx->dialect),
                           stmt->as.if_stmt.cond->line, stmt->as.if_stmt.cond->column);
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source,
                                                             stmt->as.if_stmt.cond->line,
                                                             stmt->as.if_stmt.cond->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            bool prev_allow_new = ctx->allow_new_locals;
            ctx->allow_new_locals = false;

            if (!dsl_compile_block(ctx, &stmt->as.if_stmt.then_block,
                                   &compiled->as.if_stmt.then_block)) {
                ctx->allow_new_locals = prev_allow_new;
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            compiled->as.if_stmt.n_elifs = stmt->as.if_stmt.n_elifs;
            compiled->as.if_stmt.elif_capacity = stmt->as.if_stmt.n_elifs;
            if (compiled->as.if_stmt.n_elifs > 0) {
                compiled->as.if_stmt.elif_branches = calloc((size_t)compiled->as.if_stmt.n_elifs,
                                                            sizeof(*compiled->as.if_stmt.elif_branches));
                if (!compiled->as.if_stmt.elif_branches) {
                    ctx->allow_new_locals = prev_allow_new;
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
                me_dsl_if_branch *elif_branch = &stmt->as.if_stmt.elif_branches[i];
                me_dsl_compiled_if_branch *out_branch = &compiled->as.if_stmt.elif_branches[i];
                if (!dsl_compile_expr(ctx, elif_branch->cond, ME_AUTO, &out_branch->cond)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                if (require_uniform_cond &&
                    !dsl_expr_is_uniform(out_branch->cond.expr,
                                         ctx->program->vars.uniform,
                                         ctx->program->vars.count)) {
                    dsl_tracef("compile reject: dialect=%s requires uniform loop condition at %d:%d; "
                               "use '# me:dialect=element' for per-item loop conditions",
                               dsl_dialect_name(ctx->dialect),
                               elif_branch->cond->line, elif_branch->cond->column);
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source,
                                                                 elif_branch->cond->line,
                                                                 elif_branch->cond->column);
                    }
                    ctx->allow_new_locals = prev_allow_new;
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                if (!dsl_compile_block(ctx, &elif_branch->block, &out_branch->block)) {
                    ctx->allow_new_locals = prev_allow_new;
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }

            if (stmt->as.if_stmt.has_else) {
                if (!dsl_compile_block(ctx, &stmt->as.if_stmt.else_block,
                                       &compiled->as.if_stmt.else_block)) {
                    ctx->allow_new_locals = prev_allow_new;
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                compiled->as.if_stmt.has_else = true;
            }
            ctx->allow_new_locals = prev_allow_new;
            break;
        }
        case ME_DSL_STMT_FOR: {
            const char *var = stmt->as.for_loop.var;
            if (!var || dsl_is_reserved_name(var)) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (!ctx->allow_new_locals) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (strchr(stmt->as.for_loop.limit->text, ',')) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            int var_index = dsl_var_table_find(&ctx->program->vars, var);
            if (var_index >= 0) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            var_index = dsl_var_table_add_with_uniform(&ctx->program->vars, var, ME_INT64, 0, true);
            if (var_index < 0) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (!dsl_program_add_local(ctx->program, var_index)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            compiled->as.for_loop.loop_var_slot = ctx->program->local_slots[var_index];

            if (!dsl_compile_expr(ctx, stmt->as.for_loop.limit, ME_AUTO, &compiled->as.for_loop.limit)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth++;
            if (!dsl_compile_block(ctx, &stmt->as.for_loop.body, &compiled->as.for_loop.body)) {
                ctx->loop_depth--;
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth--;
            break;
        }
        case ME_DSL_STMT_BREAK:
        case ME_DSL_STMT_CONTINUE: {
            if (ctx->loop_depth <= 0) {
                if (ctx->error_pos) {
                    *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
                }
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            if (stmt->as.flow.cond) {
                if (!dsl_compile_expr(ctx, stmt->as.flow.cond, ME_AUTO, &compiled->as.flow.cond)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                if (ctx->dialect == ME_DSL_DIALECT_VECTOR &&
                    !dsl_expr_is_uniform(compiled->as.flow.cond.expr,
                                         ctx->program->vars.uniform,
                                         ctx->program->vars.count)) {
                    dsl_tracef("compile reject: dialect=%s requires uniform break/continue condition at %d:%d",
                               dsl_dialect_name(ctx->dialect),
                               stmt->as.flow.cond->line, stmt->as.flow.cond->column);
                    if (ctx->error_pos) {
                        *ctx->error_pos = dsl_offset_from_linecol(ctx->source,
                                                                 stmt->as.flow.cond->line,
                                                                 stmt->as.flow.cond->column);
                    }
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            else {
                memset(&compiled->as.flow.cond, 0, sizeof(compiled->as.flow.cond));
            }
            break;
        }
        }

        if (!dsl_compiled_block_push(out_block, compiled)) {
            dsl_compiled_stmt_free(compiled);
            return false;
        }
    }
    return true;
}

static me_dsl_compiled_program *dsl_compile_program(const char *source,
                                                    const me_variable_ex *variables,
                                                    int var_count,
                                                    me_dtype dtype,
                                                    int *error_pos,
                                                    bool *is_dsl) {
    me_dsl_error parse_error;
    if (is_dsl) {
        *is_dsl = false;
    }
    me_dsl_program *parsed = me_dsl_parse(source, &parse_error);
    if (!parsed) {
        if (error_pos) {
            int off = dsl_offset_from_linecol(source, parse_error.line, parse_error.column);
            *error_pos = off >= 0 ? off : -1;
        }
        if (is_dsl) {
            *is_dsl = true;
        }
        return NULL;
    }
    if (!dsl_program_is_dsl(parsed)) {
        me_dsl_program_free(parsed);
        return NULL;
    }
    if (is_dsl) {
        *is_dsl = true;
    }
    if (parsed->dialect == ME_DSL_DIALECT_ELEMENT && !dsl_element_dialect_enabled()) {
        if (error_pos) {
            *error_pos = -1;
        }
        dsl_tracef("compile reject: dialect=%s disabled by ME_DSL_ELEMENT=0",
                   dsl_dialect_name(parsed->dialect));
        me_dsl_program_free(parsed);
        return NULL;
    }
    if (!dsl_block_guarantees_return(&parsed->block)) {
        if (error_pos) {
            int line = 0;
            int column = 0;
            dsl_block_first_linecol(&parsed->block, &line, &column);
            int off = dsl_offset_from_linecol(source, line, column);
            *error_pos = off >= 0 ? off : -1;
        }
        me_dsl_program_free(parsed);
        return NULL;
    }

    me_dsl_compiled_program *program = calloc(1, sizeof(*program));
    if (!program) {
        me_dsl_program_free(parsed);
        if (error_pos) {
            *error_pos = -1;
        }
        return NULL;
    }
    program->dialect = parsed->dialect;
    program->fp_mode = parsed->fp_mode;
    dsl_var_table_init(&program->vars);
    program->idx_ndim = -1;
    for (int i = 0; i < ME_DSL_MAX_NDIM; i++) {
        program->idx_i[i] = -1;
        program->idx_n[i] = -1;
    }
    program->local_slots = malloc(ME_MAX_VARS * sizeof(*program->local_slots));
    if (!program->local_slots) {
        dsl_compiled_program_free(program);
        me_dsl_program_free(parsed);
        if (error_pos) {
            *error_pos = -1;
        }
        return NULL;
    }
    for (int i = 0; i < ME_MAX_VARS; i++) {
        program->local_slots[i] = -1;
    }

    me_variable_ex *funcs = NULL;
    int func_count = 0;
    int func_capacity = 0;
    int input_count = 0;

    for (int i = 0; i < var_count; i++) {
        const me_variable_ex *entry = &variables[i];
        const char *name = entry->name;
        if (!name) {
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
        if (!is_variable_entry(entry) && !is_function_entry(entry)) {
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            return NULL;
        }
        if (is_function_entry(entry)) {
            size_t name_len = strlen(name);
            if (dsl_is_reserved_name(name) || me_is_builtin_function_name(name, name_len)) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            for (int j = 0; j < parsed->nparams; j++) {
                if (strcmp(parsed->params[j], name) == 0) {
                    if (error_pos) {
                        *error_pos = -1;
                    }
                    free(funcs);
                    dsl_compiled_program_free(program);
                    me_dsl_program_free(parsed);
                    return NULL;
                }
            }
            if (entry->dtype == ME_AUTO || !is_valid_dtype(entry->dtype) || entry->dtype == ME_STRING) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            if (!entry->address) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            if (dsl_var_table_find(&program->vars, name) >= 0) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            for (int j = 0; j < func_count; j++) {
                if (strcmp(funcs[j].name, name) == 0) {
                    if (error_pos) {
                        *error_pos = -1;
                    }
                    free(funcs);
                    dsl_compiled_program_free(program);
                    me_dsl_program_free(parsed);
                    return NULL;
                }
            }
            if (func_count == func_capacity) {
                int new_cap = func_capacity ? func_capacity * 2 : 8;
                me_variable_ex *next = realloc(funcs, (size_t)new_cap * sizeof(*next));
                if (!next) {
                    if (error_pos) {
                        *error_pos = -1;
                    }
                    free(funcs);
                    dsl_compiled_program_free(program);
                    me_dsl_program_free(parsed);
                    return NULL;
                }
                funcs = next;
                func_capacity = new_cap;
            }
            funcs[func_count++] = *entry;
            continue;
        }
        if (dsl_is_reserved_name(name)) {
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
        for (int j = 0; j < func_count; j++) {
            if (strcmp(funcs[j].name, name) == 0) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
        me_dtype vtype = entry->dtype;
        if (vtype == ME_AUTO && dtype != ME_AUTO) {
            vtype = dtype;
        }
        size_t itemsize = 0;
        if (entry->dtype == ME_STRING) {
            itemsize = entry->itemsize;
        }
        int idx = dsl_var_table_add_with_uniform(&program->vars, name, vtype, itemsize, false);
        if (idx < 0) {
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            if (error_pos) {
                *error_pos = -1;
            }
            return NULL;
        }
        input_count++;
    }
    if (input_count != parsed->nparams) {
        if (error_pos) {
            *error_pos = -1;
        }
        free(funcs);
        dsl_compiled_program_free(program);
        me_dsl_program_free(parsed);
        return NULL;
    }
    for (int i = 0; i < parsed->nparams; i++) {
        if (dsl_var_table_find(&program->vars, parsed->params[i]) < 0) {
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
    }
    program->n_inputs = input_count;

    if (dtype == ME_AUTO) {
        for (int i = 0; i < program->vars.count; i++) {
            if (program->vars.dtypes[i] == ME_AUTO) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
    }

    int uses_i_mask = 0;
    int uses_n_mask = 0;
    bool uses_ndim = false;
    dsl_scan_reserved_usage_block(&parsed->block, &uses_i_mask, &uses_n_mask, &uses_ndim);

    program->uses_i_mask = uses_i_mask;
    program->uses_n_mask = uses_n_mask;
    program->uses_ndim = uses_ndim;

    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
        if (uses_i_mask & (1 << d)) {
            char name[8];
            snprintf(name, sizeof(name), "_i%d", d);
            program->idx_i[d] = dsl_var_table_add(&program->vars, name, ME_INT64);
            if (program->idx_i[d] < 0) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
        if (uses_n_mask & (1 << d)) {
            char name[8];
            snprintf(name, sizeof(name), "_n%d", d);
            program->idx_n[d] = dsl_var_table_add_with_uniform(&program->vars, name, ME_INT64,
                                                               0, true);
            if (program->idx_n[d] < 0) {
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
        }
    }
    if (uses_ndim) {
        program->idx_ndim = dsl_var_table_add_with_uniform(&program->vars, "_ndim", ME_INT64,
                                                           0, true);
        if (program->idx_ndim < 0) {
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
    }

    dsl_compile_ctx ctx;
    ctx.source = source;
    ctx.output_dtype = dtype;
    ctx.output_dtype_auto = (dtype == ME_AUTO);
    ctx.loop_depth = 0;
    ctx.dialect = parsed->dialect;
    ctx.allow_new_locals = true;
    ctx.error_pos = error_pos;
    ctx.output_expr = NULL;
    ctx.has_return = false;
    ctx.return_dtype = ME_AUTO;
    ctx.program = program;
    ctx.funcs = funcs;
    ctx.func_count = func_count;

    if (!dsl_compile_block(&ctx, &parsed->block, &program->block)) {
        free(funcs);
        dsl_compiled_program_free(program);
        me_dsl_program_free(parsed);
        return NULL;
    }

    if (!ctx.has_return || !ctx.output_expr || !ctx.output_expr->expr) {
        if (error_pos) {
            *error_pos = -1;
        }
        free(funcs);
        me_dsl_program_free(parsed);
        dsl_compiled_program_free(program);
        return NULL;
    }

    program->output_dtype = ctx.return_dtype;
    program->output_is_scalar = contains_reduction(ctx.output_expr->expr) &&
                                output_is_scalar(ctx.output_expr->expr);
    dsl_try_build_jit_ir(&ctx, parsed, program);

    me_dsl_program_free(parsed);
    free(funcs);

    return program;
}

// Check if a pointer is a synthetic address
int is_synthetic_address(const void* ptr) {
    const char* p = (const char*)ptr;
    return (p >= synthetic_var_addresses && p < synthetic_var_addresses + ME_MAX_VARS);
}

int me_compile_ex(const char* expression, const me_variable_ex* variables,
                  int var_count, me_dtype dtype, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!expression || !out) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    if (dsl_is_candidate(expression)) {
        me_variable_ex *vars_dsl = NULL;
        if (variables && var_count > 0) {
            vars_dsl = malloc((size_t)var_count * sizeof(*vars_dsl));
            if (!vars_dsl) {
                if (error) *error = -1;
                return ME_COMPILE_ERR_OOM;
            }
            for (int i = 0; i < var_count; i++) {
                vars_dsl[i] = variables[i];
                if (is_function_entry(&vars_dsl[i])) {
                    continue;
                }
                vars_dsl[i].address = &synthetic_var_addresses[i];
                if (vars_dsl[i].type == 0) {
                    vars_dsl[i].type = ME_VARIABLE;
                }
            }
        }

        bool is_dsl = false;
        int dsl_error = -1;
        me_dsl_compiled_program *program = dsl_compile_program(
            expression, vars_dsl ? vars_dsl : variables, var_count, dtype, &dsl_error, &is_dsl);
        free(vars_dsl);

        if (program) {
            me_expr *expr = new_expr(ME_CONSTANT, NULL);
            if (!expr) {
                dsl_compiled_program_free(program);
                if (error) *error = -1;
                return ME_COMPILE_ERR_OOM;
            }
            expr->dsl_program = program;
            expr->dtype = program->output_dtype;
            if (error) *error = 0;
            *out = expr;
            return ME_COMPILE_SUCCESS;
        }
        if (is_dsl) {
            if (error) *error = dsl_error;
            return ME_COMPILE_ERR_PARSE;
        }
    }

    // For chunked evaluation, we compile without specific output/nitems
    // If variables have NULL addresses, assign synthetic unique addresses for ordinal matching
    me_variable_ex* vars_copy = NULL;
    int needs_synthetic = 0;

    if (variables && var_count > 0) {
        // Check if any variables have NULL addresses
        for (int i = 0; i < var_count; i++) {
            if (variables[i].address == NULL && is_variable_entry(&variables[i])) {
                needs_synthetic = 1;
                break;
            }
        }

        if (needs_synthetic) {
            // Create copy with synthetic addresses
            vars_copy = malloc(var_count * sizeof(me_variable_ex));
            if (!vars_copy) {
                if (error) *error = -1;
                return ME_COMPILE_ERR_OOM;
            }

            for (int i = 0; i < var_count; i++) {
                vars_copy[i] = variables[i];
                if (vars_copy[i].address == NULL && is_variable_entry(&vars_copy[i])) {
                    // Use address in synthetic array (each index is unique)
                    vars_copy[i].address = &synthetic_var_addresses[i];
                }
            }

            int status = private_compile_ex(expression, vars_copy, var_count, NULL, 0, dtype, error, out);
            free(vars_copy);
            return status;
        }
    }

    // No NULL addresses, use variables as-is
    return private_compile_ex(expression, variables, var_count, NULL, 0, dtype, error, out);
}

int me_compile(const char* expression, const me_variable* variables,
               int var_count, me_dtype dtype, int* error, me_expr** out) {
    if (!variables || var_count <= 0) {
        return me_compile_ex(expression, NULL, var_count, dtype, error, out);
    }

    me_variable_ex* vars_ex = malloc((size_t)var_count * sizeof(*vars_ex));
    if (!vars_ex) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_OOM;
    }
    for (int i = 0; i < var_count; i++) {
        vars_ex[i].name = variables[i].name;
        vars_ex[i].dtype = variables[i].dtype;
        vars_ex[i].address = variables[i].address;
        vars_ex[i].type = variables[i].type;
        vars_ex[i].context = variables[i].context;
        vars_ex[i].itemsize = 0;
    }

    int rc = me_compile_ex(expression, vars_ex, var_count, dtype, error, out);
    free(vars_ex);
    return rc;
}

int me_compile_nd_ex(const char* expression, const me_variable_ex* variables,
                     int var_count, me_dtype dtype, int ndims,
                     const int64_t* shape, const int32_t* chunkshape,
                     const int32_t* blockshape, int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!expression || !out || ndims <= 0 || !shape || !chunkshape || !blockshape) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    for (int i = 0; i < ndims; i++) {
        if (chunkshape[i] <= 0 || blockshape[i] <= 0) {
            if (error) *error = -1;
            return ME_COMPILE_ERR_INVALID_ARG;
        }
    }

    me_expr* expr = NULL;
    int rc = me_compile_ex(expression, variables, var_count, dtype, error, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        return rc;
    }

    const size_t extra_items = (size_t)(3 * ndims - 1);
    const size_t info_size = sizeof(me_nd_info) + extra_items * sizeof(int64_t);
    me_nd_info* info = malloc(info_size);
    if (!info) {
        me_free(expr);
        if (error) *error = -1;
        return ME_COMPILE_ERR_OOM;
    }

    info->ndims = ndims;
    int64_t* ptr = info->data;
    for (int i = 0; i < ndims; i++) {
        ptr[i] = shape[i];
    }
    ptr += ndims;
    for (int i = 0; i < ndims; i++) {
        ptr[i] = (int64_t)chunkshape[i];
    }
    ptr += ndims;
    for (int i = 0; i < ndims; i++) {
        ptr[i] = (int64_t)blockshape[i];
    }

    expr->bytecode = info;
    *out = expr;
    return rc;
}

int me_compile_nd(const char* expression, const me_variable* variables,
                  int var_count, me_dtype dtype, int ndims,
                  const int64_t* shape, const int32_t* chunkshape,
                  const int32_t* blockshape, int* error, me_expr** out) {
    if (!variables || var_count <= 0) {
        return me_compile_nd_ex(expression, NULL, var_count, dtype, ndims,
                                shape, chunkshape, blockshape, error, out);
    }

    me_variable_ex* vars_ex = malloc((size_t)var_count * sizeof(*vars_ex));
    if (!vars_ex) {
        if (error) *error = -1;
        return ME_COMPILE_ERR_OOM;
    }
    for (int i = 0; i < var_count; i++) {
        vars_ex[i].name = variables[i].name;
        vars_ex[i].dtype = variables[i].dtype;
        vars_ex[i].address = variables[i].address;
        vars_ex[i].type = variables[i].type;
        vars_ex[i].context = variables[i].context;
        vars_ex[i].itemsize = 0;
    }

    int rc = me_compile_nd_ex(expression, vars_ex, var_count, dtype, ndims,
                              shape, chunkshape, blockshape, error, out);
    free(vars_ex);
    return rc;
}

static void pn(const me_expr* n, int depth) {
    int i, arity;
    printf("%*s", depth, "");

    if (!n) {
        printf("NULL\n");
        return;
    }

    switch (TYPE_MASK(n->type)) {
    case ME_CONSTANT: printf("%f\n", n->value);
        break;
    case ME_STRING_CONSTANT: printf("<string>\n");
        break;
    case ME_VARIABLE: printf("bound %p\n", n->bound);
        break;

    case ME_FUNCTION0:
    case ME_FUNCTION1:
    case ME_FUNCTION2:
    case ME_FUNCTION3:
    case ME_FUNCTION4:
    case ME_FUNCTION5:
    case ME_FUNCTION6:
    case ME_FUNCTION7:
    case ME_CLOSURE0:
    case ME_CLOSURE1:
    case ME_CLOSURE2:
    case ME_CLOSURE3:
    case ME_CLOSURE4:
    case ME_CLOSURE5:
    case ME_CLOSURE6:
    case ME_CLOSURE7:
        arity = ARITY(n->type);
        printf("f%d", arity);
        for (i = 0; i < arity; i++) {
            printf(" %p", n->parameters[i]);
        }
        printf("\n");
        for (i = 0; i < arity; i++) {
            pn(n->parameters[i], depth + 1);
        }
        break;
    }
}

void me_print(const me_expr* n) {
    pn(n, 0);
}

me_dtype me_get_dtype(const me_expr* expr) {
    return expr ? expr->dtype : ME_AUTO;
}

const char* me_version(void) {
    return ME_VERSION_STRING;
}

static int compute_valid_items(const me_expr* expr, int64_t nchunk, int64_t nblock,
                               int chunk_nitems, int64_t* valid_items, int64_t* padded_items) {
    if (!expr || !valid_items || !padded_items) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const me_nd_info* info = (const me_nd_info*)expr->bytecode;
    if (!info || info->ndims <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const int nd = info->ndims;
    if (nd > 64) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    const int64_t* shape = info->data;
    const int64_t* chunkshape = shape + nd;
    const int64_t* blockshape = chunkshape + nd;

    int64_t total_chunks = 1;
    int64_t total_blocks = 1;
    int64_t padded = 1;

    for (int i = 0; i < nd; i++) {
        if (chunkshape[i] <= 0 || blockshape[i] <= 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        const int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        const int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        if (nchunks_d <= 0 || nblocks_d <= 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        if (total_chunks > LLONG_MAX / nchunks_d || total_blocks > LLONG_MAX / nblocks_d) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        total_chunks *= nchunks_d;
        total_blocks *= nblocks_d;
        if (padded > LLONG_MAX / blockshape[i]) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        padded *= blockshape[i];
    }

    if (nchunk < 0 || nchunk >= total_chunks || nblock < 0 || nblock >= total_blocks) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (chunk_nitems > 0 && (int64_t)chunk_nitems < padded) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t chunk_idx[64];
    int64_t block_idx[64];

    int64_t tmp = nchunk;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
        tmp /= nchunks_d;
    }

    tmp = nblock;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
        tmp /= nblocks_d;
    }

    int64_t valid = 1;
    for (int i = 0; i < nd; i++) {
        const int64_t chunk_start = chunk_idx[i] * chunkshape[i];
        if (shape[i] <= chunk_start) {
            valid = 0;
            break;
        }
        int64_t chunk_len = shape[i] - chunk_start;
        if (chunk_len > chunkshape[i]) {
            chunk_len = chunkshape[i];
        }

        const int64_t block_start = block_idx[i] * blockshape[i];
        if (block_start >= chunk_len) {
            valid = 0;
            break;
        }
        const int64_t remain = chunk_len - block_start;
        const int64_t len = (remain < blockshape[i]) ? remain : blockshape[i];
        if (valid > LLONG_MAX / len) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
        valid *= len;
    }

    if (chunk_nitems > 0 && valid > (int64_t)chunk_nitems) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    *valid_items = valid;
    *padded_items = padded;
    return ME_EVAL_SUCCESS;
}

typedef struct {
    const me_dsl_compiled_program *program;
    void **var_buffers;
    void **local_buffers;
    int nitems;
    const me_eval_params *params;
    void *output_block;
} dsl_eval_ctx;

static int dsl_eval_expr_nitems(dsl_eval_ctx *ctx, const me_dsl_compiled_expr *expr,
                                void *out, int nitems) {
    if (!expr || !expr->expr) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    const void *vars[ME_MAX_VARS];
    for (int i = 0; i < expr->n_vars; i++) {
        vars[i] = ctx->var_buffers[expr->var_indices[i]];
    }
    return me_eval(expr->expr, vars, expr->n_vars, out, nitems, ctx->params);
}

static int dsl_eval_block_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_block *block,
                                       uint8_t *run_mask, uint8_t *break_mask,
                                       uint8_t *continue_mask, bool *did_return);
static int dsl_eval_for_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_stmt *stmt,
                                     const uint8_t *input_mask, bool *did_return);

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
                                 const uint8_t *continue_mask, int nitems) {
    if (!run_mask || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        if ((break_mask && break_mask[i]) || (continue_mask && continue_mask[i])) {
            run_mask[i] = 0;
        }
    }
}

static bool dsl_value_nonzero_at(const void *data, me_dtype dtype, int idx) {
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
        return false;
    default:
        return false;
    }
}

static int dsl_eval_expr_masked_copy(dsl_eval_ctx *ctx, const me_dsl_compiled_expr *expr,
                                     void *dst, const uint8_t *mask, int nitems) {
    if (!ctx || !expr || !expr->expr || !dst || nitems < 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (nitems == 0) {
        return ME_EVAL_SUCCESS;
    }

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
    if (all_active) {
        return dsl_eval_expr_nitems(ctx, expr, dst, nitems);
    }

    me_dtype dtype = me_get_dtype(expr->expr);
    size_t item_size = dtype_size(dtype);
    if (item_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
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
        if (!mask[i]) {
            continue;
        }
        memcpy(dst_bytes + (size_t)i * item_size,
               src_bytes + (size_t)i * item_size, item_size);
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
    if (cond_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    int cond_nitems = *is_reduction ? 1 : ctx->nitems;
    if (cond_nitems <= 0) {
        if (true_mask && !*is_reduction) {
            memset(true_mask, 0, (size_t)ctx->nitems);
        }
        return ME_EVAL_SUCCESS;
    }

    void *cond_buf = malloc((size_t)cond_nitems * cond_size);
    if (!cond_buf) {
        return ME_EVAL_ERR_OOM;
    }
    int rc = dsl_eval_expr_nitems(ctx, cond, cond_buf, ctx->nitems);
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
            true_mask[i] = (uint8_t)(active && dsl_value_nonzero_at(cond_buf, cond_dtype, i));
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
                                               bool *did_return) {
    if (!ctx || !cond || !branch_block || !remaining_mask || !break_mask || !continue_mask) {
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
                                             break_mask, continue_mask, did_return);
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
                                     break_mask, continue_mask, did_return);
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
                                       uint8_t *continue_mask, bool *did_return) {
    if (!ctx || !block || !run_mask || !break_mask || !continue_mask) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    for (int i = 0; i < block->nstmts; i++) {
        if (did_return && *did_return) {
            break;
        }
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
            void *out = ctx->local_buffers[slot];
            int rc = dsl_eval_expr_masked_copy(ctx, &stmt->as.assign.value, out,
                                               run_mask, ctx->nitems);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_EXPR: {
            int rc = dsl_eval_expr_masked_copy(ctx, &stmt->as.expr_stmt.expr,
                                               ctx->output_block, run_mask, ctx->nitems);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_RETURN: {
            int rc = dsl_eval_expr_masked_copy(ctx, &stmt->as.return_stmt.expr,
                                               ctx->output_block, run_mask, ctx->nitems);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            if (did_return) {
                *did_return = true;
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
            uint8_t *remaining = malloc((size_t)ctx->nitems);
            if (!remaining) {
                return ME_EVAL_ERR_OOM;
            }
            memcpy(remaining, run_mask, (size_t)ctx->nitems);

            int rc = dsl_eval_element_conditional_branch(ctx, &stmt->as.if_stmt.cond,
                                                         &stmt->as.if_stmt.then_block,
                                                         remaining, break_mask,
                                                         continue_mask, did_return);
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
                                                         continue_mask, did_return);
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
                                                 continue_mask, did_return);
                free(else_run);
                if (rc != ME_EVAL_SUCCESS) {
                    free(remaining);
                    return rc;
                }
            }

            free(remaining);
            dsl_mask_remove_flow(run_mask, break_mask, continue_mask, ctx->nitems);
            break;
        }
        case ME_DSL_STMT_FOR: {
            int rc = dsl_eval_for_element_loop(ctx, stmt, run_mask, did_return);
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

static int dsl_eval_for_element_loop(dsl_eval_ctx *ctx, const me_dsl_compiled_stmt *stmt,
                                     const uint8_t *input_mask, bool *did_return) {
    if (!ctx || !stmt || stmt->kind != ME_DSL_STMT_FOR) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const me_dsl_compiled_expr *limit = &stmt->as.for_loop.limit;
    me_dtype limit_dtype = me_get_dtype(limit->expr);
    size_t limit_size = dtype_size(limit_dtype);
    void *limit_buf = malloc(limit_size ? limit_size : sizeof(int64_t));
    if (!limit_buf) {
        return ME_EVAL_ERR_OOM;
    }
    int rc = dsl_eval_expr_nitems(ctx, limit, limit_buf, 1);
    if (rc != ME_EVAL_SUCCESS) {
        free(limit_buf);
        return rc;
    }
    int64_t limit_val = 0;
    if (!dsl_read_int64(limit_buf, limit_dtype, &limit_val)) {
        free(limit_buf);
        return ME_EVAL_ERR_INVALID_ARG;
    }
    free(limit_buf);
    if (limit_val <= 0 || ctx->nitems <= 0) {
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

    if (!dsl_mask_any(active_mask, ctx->nitems)) {
        free(active_mask);
        return ME_EVAL_SUCCESS;
    }

    int slot = stmt->as.for_loop.loop_var_slot;
    int64_t *loop_buf = (int64_t *)ctx->local_buffers[slot];

    for (int64_t iter = 0; iter < limit_val; iter++) {
        if (!dsl_mask_any(active_mask, ctx->nitems)) {
            break;
        }

        dsl_fill_i64(loop_buf, ctx->nitems, iter);

        uint8_t *run_mask = malloc((size_t)ctx->nitems);
        uint8_t *break_mask = calloc((size_t)ctx->nitems, sizeof(*break_mask));
        uint8_t *continue_mask = calloc((size_t)ctx->nitems, sizeof(*continue_mask));
        if (!run_mask || !break_mask || !continue_mask) {
            free(run_mask);
            free(break_mask);
            free(continue_mask);
            free(active_mask);
            return ME_EVAL_ERR_OOM;
        }
        memcpy(run_mask, active_mask, (size_t)ctx->nitems);

        rc = dsl_eval_block_element_loop(ctx, &stmt->as.for_loop.body, run_mask,
                                         break_mask, continue_mask, did_return);
        free(run_mask);
        if (rc != ME_EVAL_SUCCESS) {
            free(break_mask);
            free(continue_mask);
            free(active_mask);
            return rc;
        }
        if (did_return && *did_return) {
            free(break_mask);
            free(continue_mask);
            free(active_mask);
            return ME_EVAL_SUCCESS;
        }

        for (int i = 0; i < ctx->nitems; i++) {
            if (break_mask[i]) {
                active_mask[i] = 0;
            }
        }

        free(break_mask);
        free(continue_mask);
    }

    free(active_mask);
    return ME_EVAL_SUCCESS;
}

static int dsl_eval_block(dsl_eval_ctx *ctx, const me_dsl_compiled_block *block,
                          bool *did_break, bool *did_continue, bool *did_return) {
    if (!block) {
        return ME_EVAL_SUCCESS;
    }
    for (int i = 0; i < block->nstmts; i++) {
        if (did_return && *did_return) {
            break;
        }
        if (did_break && *did_break) {
            break;
        }
        if (did_continue && *did_continue) {
            break;
        }
        me_dsl_compiled_stmt *stmt = block->stmts[i];
        if (!stmt) {
            continue;
        }
        switch (stmt->kind) {
        case ME_DSL_STMT_ASSIGN: {
            int slot = stmt->as.assign.local_slot;
            void *out = ctx->local_buffers[slot];
            int rc = dsl_eval_expr_nitems(ctx, &stmt->as.assign.value, out, ctx->nitems);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_EXPR: {
            int rc = dsl_eval_expr_nitems(ctx, &stmt->as.expr_stmt.expr, ctx->output_block, ctx->nitems);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_RETURN: {
            int rc = dsl_eval_expr_nitems(ctx, &stmt->as.return_stmt.expr,
                                          ctx->output_block, ctx->nitems);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            if (did_return) {
                *did_return = true;
            }
            break;
        }
        case ME_DSL_STMT_PRINT: {
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
            for (int i = 0; i < nargs; i++) {
                me_dsl_compiled_expr *arg = &stmt->as.print_stmt.args[i];
                me_dtype dtype = me_get_dtype(arg->expr);
                size_t size = dtype_size(dtype);
                if (size == 0) {
                    size = sizeof(double);
                }
                arg_bufs[i] = calloc(1, size);
                if (!arg_bufs[i]) {
                    for (int j = 0; j < i; j++) {
                        free(arg_bufs[j]);
                        free(arg_strs[j]);
                    }
                    free(arg_bufs);
                    free(arg_strs);
                    return ME_EVAL_ERR_OOM;
                }
                int rc = dsl_eval_expr_nitems(ctx, arg, arg_bufs[i], 1);
                if (rc != ME_EVAL_SUCCESS) {
                    for (int j = 0; j <= i; j++) {
                        free(arg_bufs[j]);
                        free(arg_strs[j]);
                    }
                    free(arg_bufs);
                    free(arg_strs);
                    return rc;
                }
                arg_strs[i] = calloc(1, 64);
                if (!arg_strs[i]) {
                    for (int j = 0; j <= i; j++) {
                        free(arg_bufs[j]);
                        free(arg_strs[j]);
                    }
                    free(arg_bufs);
                    free(arg_strs);
                    return ME_EVAL_ERR_OOM;
                }
                dsl_format_value(arg_strs[i], 64, dtype, arg_bufs[i]);
            }
            dsl_print_formatted(stmt->as.print_stmt.format, arg_strs, nargs);
            for (int i = 0; i < nargs; i++) {
                free(arg_bufs[i]);
                free(arg_strs[i]);
            }
            free(arg_bufs);
            free(arg_strs);
            break;
        }
        case ME_DSL_STMT_IF: {
            bool matched = false;
            me_dtype cond_dtype = me_get_dtype(stmt->as.if_stmt.cond.expr);
            size_t cond_size = dtype_size(cond_dtype);
            bool cond_is_reduction = is_reduction_node(stmt->as.if_stmt.cond.expr);
            int cond_nitems = cond_is_reduction ? 1 : ctx->nitems;
            void *cond_buf = malloc((size_t)cond_nitems * cond_size);
            if (!cond_buf) {
                return ME_EVAL_ERR_OOM;
            }
            int rc = dsl_eval_expr_nitems(ctx, &stmt->as.if_stmt.cond, cond_buf, ctx->nitems);
            if (rc != ME_EVAL_SUCCESS) {
                free(cond_buf);
                return rc;
            }
            matched = dsl_any_nonzero(cond_buf, cond_dtype, cond_nitems);
            free(cond_buf);
            if (matched) {
                rc = dsl_eval_block(ctx, &stmt->as.if_stmt.then_block,
                                    did_break, did_continue, did_return);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }
                break;
            }
            for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
                me_dsl_compiled_if_branch *branch = &stmt->as.if_stmt.elif_branches[i];
                cond_dtype = me_get_dtype(branch->cond.expr);
                cond_size = dtype_size(cond_dtype);
                cond_is_reduction = is_reduction_node(branch->cond.expr);
                cond_nitems = cond_is_reduction ? 1 : ctx->nitems;
                cond_buf = malloc((size_t)cond_nitems * cond_size);
                if (!cond_buf) {
                    return ME_EVAL_ERR_OOM;
                }
                rc = dsl_eval_expr_nitems(ctx, &branch->cond, cond_buf, ctx->nitems);
                if (rc != ME_EVAL_SUCCESS) {
                    free(cond_buf);
                    return rc;
                }
                matched = dsl_any_nonzero(cond_buf, cond_dtype, cond_nitems);
                free(cond_buf);
                if (matched) {
                    rc = dsl_eval_block(ctx, &branch->block,
                                        did_break, did_continue, did_return);
                    if (rc != ME_EVAL_SUCCESS) {
                        return rc;
                    }
                    break;
                }
            }
            if (!matched && stmt->as.if_stmt.has_else) {
                rc = dsl_eval_block(ctx, &stmt->as.if_stmt.else_block,
                                    did_break, did_continue, did_return);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }
            }
            break;
        }
        case ME_DSL_STMT_FOR: {
            if (ctx->program && ctx->program->dialect == ME_DSL_DIALECT_ELEMENT) {
                int rc = dsl_eval_for_element_loop(ctx, stmt, NULL, did_return);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }
                break;
            }
            me_dsl_compiled_expr *limit = &stmt->as.for_loop.limit;
            me_dtype limit_dtype = me_get_dtype(limit->expr);
            size_t limit_size = dtype_size(limit_dtype);
            void *limit_buf = malloc(limit_size ? limit_size : sizeof(int64_t));
            if (!limit_buf) {
                return ME_EVAL_ERR_OOM;
            }
            int rc = dsl_eval_expr_nitems(ctx, limit, limit_buf, 1);
            if (rc != ME_EVAL_SUCCESS) {
                free(limit_buf);
                return rc;
            }
            int64_t limit_val = 0;
            if (!dsl_read_int64(limit_buf, limit_dtype, &limit_val)) {
                free(limit_buf);
                return ME_EVAL_ERR_INVALID_ARG;
            }
            free(limit_buf);
            if (limit_val <= 0) {
                break;
            }
            int slot = stmt->as.for_loop.loop_var_slot;
            int64_t *loop_buf = (int64_t *)ctx->local_buffers[slot];
            for (int64_t iter = 0; iter < limit_val; iter++) {
                dsl_fill_i64(loop_buf, ctx->nitems, iter);
                bool inner_break = false;
                bool inner_continue = false;
                bool inner_return = false;
                rc = dsl_eval_block(ctx, &stmt->as.for_loop.body,
                                    &inner_break, &inner_continue, &inner_return);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }
                if (inner_return) {
                    if (did_return) {
                        *did_return = true;
                    }
                    return ME_EVAL_SUCCESS;
                }
                if (inner_break) {
                    break;
                }
            }
            break;
        }
        case ME_DSL_STMT_BREAK:
        case ME_DSL_STMT_CONTINUE: {
            bool trigger = true;
            if (stmt->as.flow.cond.expr) {
                me_dtype cond_dtype = me_get_dtype(stmt->as.flow.cond.expr);
                size_t cond_size = dtype_size(cond_dtype);
                bool cond_is_reduction = is_reduction_node(stmt->as.flow.cond.expr);
                int cond_nitems = cond_is_reduction ? 1 : ctx->nitems;
                void *cond_buf = malloc((size_t)cond_nitems * cond_size);
                if (!cond_buf) {
                    return ME_EVAL_ERR_OOM;
                }
                int rc = dsl_eval_expr_nitems(ctx, &stmt->as.flow.cond, cond_buf, ctx->nitems);
                if (rc != ME_EVAL_SUCCESS) {
                    free(cond_buf);
                    return rc;
                }
                trigger = dsl_any_nonzero(cond_buf, cond_dtype, cond_nitems);
                free(cond_buf);
            }
            if (trigger) {
                if (stmt->kind == ME_DSL_STMT_BREAK && did_break) {
                    *did_break = true;
                }
                if (stmt->kind == ME_DSL_STMT_CONTINUE && did_continue) {
                    *did_continue = true;
                }
            }
            break;
        }
        }
    }
    return ME_EVAL_SUCCESS;
}

static int dsl_eval_program(const me_dsl_compiled_program *program,
                            const void **vars_block, int n_vars,
                            void *output_block, int nitems,
                            const me_eval_params *params,
                            int ndim, const int64_t *shape,
                            int64_t **idx_buffers) {
    if (!program || !output_block || nitems < 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (n_vars != program->n_inputs) {
        return ME_EVAL_ERR_VAR_MISMATCH;
    }

    /* JIT is best-effort: if kernel call fails, execution falls back to interpreter. */
    if (program->jit_kernel_fn && program->jit_nparams >= 0 && program->jit_nparams <= ME_MAX_VARS) {
        const void *jit_inputs_stack[ME_MAX_VARS];
        const void **jit_inputs = vars_block;
        bool can_run_jit = true;
        if (program->jit_nparams > 0) {
            if (!vars_block || !program->jit_param_input_indices) {
                can_run_jit = false;
            }
            else {
                for (int i = 0; i < program->jit_nparams; i++) {
                    int idx = program->jit_param_input_indices[i];
                    if (idx < 0 || idx >= n_vars) {
                        can_run_jit = false;
                        break;
                    }
                    jit_inputs_stack[i] = vars_block[idx];
                }
                jit_inputs = jit_inputs_stack;
            }
        }
        if (can_run_jit) {
            int jit_rc = program->jit_kernel_fn(jit_inputs, output_block, (int64_t)nitems);
            if (jit_rc == 0) {
                return ME_EVAL_SUCCESS;
            }
        }
    }

    void **var_buffers = calloc((size_t)program->vars.count, sizeof(*var_buffers));
    void **local_buffers = calloc((size_t)program->n_locals, sizeof(*local_buffers));
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
        size_t sz = dtype_size(program->vars.dtypes[var_index]);
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

    void *reserved_buffers[ME_DSL_MAX_NDIM * 2 + 1];
    int reserved_count = 0;
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
                if (d == 0) {
                    dsl_fill_iota_i64(buf, nitems, 0);
                }
                else {
                    dsl_fill_i64(buf, nitems, 0);
                }
                var_buffers[program->idx_i[d]] = buf;
                reserved_buffers[reserved_count++] = buf;
            }
        }
    }

    if (reserved_count < 0) {
        for (int i = 0; i < program->n_locals; i++) {
            if (local_buffers[i]) {
                free(local_buffers[i]);
            }
        }
        free(var_buffers);
        free(local_buffers);
        return ME_EVAL_ERR_OOM;
    }

    dsl_eval_ctx ctx;
    ctx.program = program;
    ctx.var_buffers = var_buffers;
    ctx.local_buffers = local_buffers;
    ctx.nitems = nitems;
    ctx.params = params;
    ctx.output_block = output_block;

    bool did_break = false;
    bool did_continue = false;
    bool did_return = false;
    int rc = dsl_eval_block(&ctx, &program->block, &did_break, &did_continue, &did_return);

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

int me_eval_dsl_program(const me_expr *expr, const void **vars_block,
                        int n_vars, void *output_block, int block_nitems,
                        const me_eval_params *params) {
    if (!expr || !expr->dsl_program) {
        return ME_EVAL_ERR_NULL_EXPR;
    }
    if (!output_block || block_nitems < 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    const me_dsl_compiled_program *program = (const me_dsl_compiled_program *)expr->dsl_program;
    return dsl_eval_program(program, vars_block, n_vars, output_block, block_nitems, params,
                            1, NULL, NULL);
}

static int me_eval_dsl_nd(const me_expr *expr, const void **vars_block,
                          int n_vars, void *output_block, int block_nitems,
                          int64_t nchunk, int64_t nblock,
                          const me_eval_params *params) {
    if (!expr || !expr->dsl_program) {
        return ME_EVAL_ERR_NULL_EXPR;
    }
    const me_dsl_compiled_program *program = (const me_dsl_compiled_program *)expr->dsl_program;
    if (!output_block || block_nitems <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t valid_items = 0;
    int64_t padded_items = 0;
    int rc = compute_valid_items(expr, nchunk, nblock, block_nitems, &valid_items, &padded_items);
    if (rc != ME_EVAL_SUCCESS) {
        return rc;
    }
    if (valid_items > INT_MAX) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const me_nd_info *info = (const me_nd_info *)expr->bytecode;
    if (!info || info->ndims <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    const int nd = info->ndims;
    const int64_t *shape = info->data;
    const int64_t *chunkshape = shape + nd;
    const int64_t *blockshape = chunkshape + nd;

    const size_t item_size = dtype_size(me_get_dtype(expr));
    if (item_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    if (valid_items == 0) {
        if (program->output_is_scalar) {
            memset(output_block, 0, item_size);
        }
        else {
            memset(output_block, 0, (size_t)padded_items * item_size);
        }
        return ME_EVAL_SUCCESS;
    }

    int64_t chunk_idx[64];
    int64_t block_idx[64];
    if (nd > 64) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t tmp = nchunk;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
        tmp /= nchunks_d;
    }

    tmp = nblock;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
        tmp /= nblocks_d;
    }

    int64_t base_idx[64];
    for (int i = 0; i < nd; i++) {
        base_idx[i] = chunk_idx[i] * chunkshape[i] + block_idx[i] * blockshape[i];
    }

    int64_t *idx_buffers[ME_DSL_MAX_NDIM];
    for (int i = 0; i < ME_DSL_MAX_NDIM; i++) {
        idx_buffers[i] = NULL;
    }

    if (valid_items == padded_items) {
        if (program->uses_i_mask) {
            for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
                if (program->uses_i_mask & (1 << d)) {
                    idx_buffers[d] = malloc((size_t)valid_items * sizeof(int64_t));
                    if (!idx_buffers[d]) {
                        for (int j = 0; j < ME_DSL_MAX_NDIM; j++) free(idx_buffers[j]);
                        return ME_EVAL_ERR_OOM;
                    }
                }
            }
            int64_t indices[64] = {0};
            int64_t total_iters = padded_items;
            for (int64_t it = 0; it < total_iters; it++) {
                for (int d = 0; d < ME_DSL_MAX_NDIM && d < nd; d++) {
                    if (idx_buffers[d]) {
                        idx_buffers[d][it] = base_idx[d] + indices[d];
                    }
                }
                for (int i = nd - 1; i >= 0; i--) {
                    indices[i]++;
                    if (indices[i] < blockshape[i]) break;
                    indices[i] = 0;
                }
            }
        }

        rc = dsl_eval_program(program, vars_block, n_vars, output_block,
                              (int)valid_items, params, nd, shape, idx_buffers);
        for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
            free(idx_buffers[d]);
        }
        return rc;
    }

    int64_t valid_len[64];
    for (int i = 0; i < nd; i++) {
        const int64_t chunk_start = chunk_idx[i] * chunkshape[i];
        int64_t chunk_len = shape[i] - chunk_start;
        if (chunk_len > chunkshape[i]) {
            chunk_len = chunkshape[i];
        }
        const int64_t block_start = block_idx[i] * blockshape[i];
        if (block_start >= chunk_len) {
            valid_len[i] = 0;
        }
        else {
            int64_t len = chunk_len - block_start;
            if (len > blockshape[i]) {
                len = blockshape[i];
            }
            valid_len[i] = len;
        }
    }

    int64_t stride[64];
    stride[nd - 1] = 1;
    for (int i = nd - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * blockshape[i + 1];
    }

    size_t var_sizes[ME_MAX_VARS];
    for (int v = 0; v < n_vars; v++) {
        var_sizes[v] = dtype_size(program->vars.dtypes[v]);
        if (var_sizes[v] == 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
    }

    void *packed_vars[ME_MAX_VARS];
    for (int v = 0; v < n_vars; v++) {
        packed_vars[v] = malloc((size_t)valid_items * var_sizes[v]);
        if (!packed_vars[v]) {
            for (int u = 0; u < v; u++) free(packed_vars[u]);
            return ME_EVAL_ERR_OOM;
        }
    }

    void *packed_out = NULL;
    if (!program->output_is_scalar) {
        packed_out = malloc((size_t)valid_items * item_size);
        if (!packed_out) {
            for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
            return ME_EVAL_ERR_OOM;
        }
    }

    for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
        if (program->uses_i_mask & (1 << d)) {
            idx_buffers[d] = malloc((size_t)valid_items * sizeof(int64_t));
            if (!idx_buffers[d]) {
                for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
                free(packed_out);
                for (int j = 0; j < ME_DSL_MAX_NDIM; j++) free(idx_buffers[j]);
                return ME_EVAL_ERR_OOM;
            }
        }
    }

    int64_t indices[64] = {0};
    int64_t write_idx = 0;
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) total_iters *= valid_len[i];
    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }
        for (int v = 0; v < n_vars; v++) {
            const unsigned char *src = (const unsigned char *)vars_block[v] + (size_t)off * var_sizes[v];
            memcpy((unsigned char *)packed_vars[v] + (size_t)write_idx * var_sizes[v], src, var_sizes[v]);
        }
        for (int d = 0; d < ME_DSL_MAX_NDIM && d < nd; d++) {
            if (idx_buffers[d]) {
                idx_buffers[d][write_idx] = base_idx[d] + indices[d];
            }
        }
        write_idx++;
        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) break;
            indices[i] = 0;
        }
    }

    void *dsl_out = program->output_is_scalar ? malloc((size_t)valid_items * item_size) : packed_out;
    if (!dsl_out) {
        for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
        free(packed_out);
        for (int j = 0; j < ME_DSL_MAX_NDIM; j++) free(idx_buffers[j]);
        return ME_EVAL_ERR_OOM;
    }

    rc = dsl_eval_program(program, (const void **)packed_vars, n_vars, dsl_out,
                          (int)valid_items, params, nd, shape, idx_buffers);
    if (rc != ME_EVAL_SUCCESS) {
        for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
        if (program->output_is_scalar) free(dsl_out);
        free(packed_out);
        for (int j = 0; j < ME_DSL_MAX_NDIM; j++) free(idx_buffers[j]);
        return rc;
    }

    if (program->output_is_scalar) {
        memcpy(output_block, dsl_out, item_size);
        memset((unsigned char *)output_block + item_size, 0, (size_t)(padded_items - 1) * item_size);
        free(dsl_out);
    }
    else {
        memset(output_block, 0, (size_t)padded_items * item_size);
        memset(indices, 0, sizeof(indices));
        write_idx = 0;
        for (int64_t it = 0; it < total_iters; it++) {
            int64_t off = 0;
            for (int i = 0; i < nd; i++) {
                off += indices[i] * stride[i];
            }
            unsigned char *dst = (unsigned char *)output_block + (size_t)off * item_size;
            memcpy(dst, (unsigned char *)packed_out + (size_t)write_idx * item_size, item_size);
            write_idx++;
            for (int i = nd - 1; i >= 0; i--) {
                indices[i]++;
                if (indices[i] < valid_len[i]) break;
                indices[i] = 0;
            }
        }
    }

    for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
    if (packed_out) free(packed_out);
    for (int j = 0; j < ME_DSL_MAX_NDIM; j++) free(idx_buffers[j]);

    return ME_EVAL_SUCCESS;
}

static int collect_var_sizes(const me_expr* expr, size_t* var_sizes, int n_vars) {
    if (!expr || !var_sizes || n_vars <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    for (int i = 0; i < n_vars; i++) {
        var_sizes[i] = 0;
    }

    /* DFS to collect sizes from variable nodes (synthetic address index). */
    const me_expr* stack[256];
    int top = 0;
    stack[top++] = expr;
    while (top) {
        const me_expr* n = stack[--top];
        if (!n) continue;
        if (TYPE_MASK(n->type) == ME_VARIABLE && is_synthetic_address(n->bound)) {
            int idx = (int)((const char*)n->bound - synthetic_var_addresses);
            if (idx >= 0 && idx < n_vars && var_sizes[idx] == 0) {
                var_sizes[idx] = dtype_size(n->input_dtype);
            }
        }
        else if (IS_FUNCTION(n->type) || IS_CLOSURE(n->type)) {
            const int arity = ARITY(n->type);
            for (int i = 0; i < arity && top < 256; i++) {
                stack[top++] = (const me_expr*)n->parameters[i];
            }
        }
    }

    for (int i = 0; i < n_vars; i++) {
        if (var_sizes[i] == 0) {
            return ME_EVAL_ERR_INVALID_ARG;
        }
    }
    return ME_EVAL_SUCCESS;
}

int me_eval_nd(const me_expr* expr, const void** vars_block,
               int n_vars, void* output_block, int block_nitems,
               int64_t nchunk, int64_t nblock, const me_eval_params* params) {
    if (!expr) {
        return ME_EVAL_ERR_NULL_EXPR;
    }
    if (expr->dtype == ME_STRING) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (expr->dsl_program) {
        return me_eval_dsl_nd(expr, vars_block, n_vars, output_block, block_nitems, nchunk, nblock, params);
    }
    if (!output_block || block_nitems <= 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t valid_items = 0;
    int64_t padded_items = 0;
    int rc = compute_valid_items(expr, nchunk, nblock, block_nitems, &valid_items, &padded_items);
    if (rc != ME_EVAL_SUCCESS) {
        return rc;
    }
    if (valid_items > INT_MAX) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const size_t item_size = dtype_size(me_get_dtype(expr));
    if (item_size == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    const bool is_reduction_output = contains_reduction(expr) && output_is_scalar(expr);

    /* Fast path: no padding needed (valid == padded), single call. */
    if (valid_items == padded_items) {
        if (valid_items == 0) {
            /* Scalar outputs only write the first item. */
            if (is_reduction_output) {
                memset(output_block, 0, item_size);
            }
            else {
                memset(output_block, 0, (size_t)padded_items * item_size);
            }
            return ME_EVAL_SUCCESS;
        }
        return me_eval(expr, vars_block, n_vars, output_block, (int)valid_items, params);
    }

    const me_nd_info* info = (const me_nd_info*)expr->bytecode;
    const int nd = info->ndims;
    const int64_t* shape = info->data;
    const int64_t* chunkshape = shape + nd;
    const int64_t* blockshape = chunkshape + nd;

    size_t var_sizes[ME_MAX_VARS];
    rc = collect_var_sizes(expr, var_sizes, n_vars);
    if (rc != ME_EVAL_SUCCESS) {
        return rc;
    }

    /* Compute per-dim lengths for this chunk/block. */
    int64_t chunk_idx[64];
    int64_t block_idx[64];
    int64_t valid_len[64];
    if (nd > 64) {
        return ME_EVAL_ERR_INVALID_ARG;
    }

    int64_t tmp = nchunk;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
        tmp /= nchunks_d;
    }

    tmp = nblock;
    for (int i = nd - 1; i >= 0; i--) {
        const int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
        tmp /= nblocks_d;
    }

    for (int i = 0; i < nd; i++) {
        const int64_t chunk_start = chunk_idx[i] * chunkshape[i];
        int64_t chunk_len = shape[i] - chunk_start;
        if (chunk_len > chunkshape[i]) {
            chunk_len = chunkshape[i];
        }
        const int64_t block_start = block_idx[i] * blockshape[i];
        if (block_start >= chunk_len) {
            valid_len[i] = 0;
        }
        else {
            int64_t len = chunk_len - block_start;
            if (len > blockshape[i]) {
                len = blockshape[i];
            }
            valid_len[i] = len;
        }
    }

    /* Strides inside padded block (C-order). */
    int64_t stride[64];
    stride[nd - 1] = 1;
    for (int i = nd - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * blockshape[i + 1];
    }

    /* Pack → single eval → scatter */
    if (valid_items == 0) {
        if (is_reduction_output) {
            if (is_reduction_node(expr) && reduction_kind(expr->function) == ME_REDUCE_MEAN) {
                const me_expr* arg = (const me_expr*)expr->parameters[0];
                me_dtype arg_type = arg ? infer_result_type(arg) : ME_FLOAT64;
                me_dtype result_type = reduction_output_dtype(arg_type, expr->function);
                me_scalar acc;
                if (result_type == ME_COMPLEX128) {
                    acc.c128 = me_cmplx(NAN, NAN);
                }
                else {
                    acc.f64 = NAN;
                }
                write_scalar(output_block, expr->dtype, result_type, &acc);
            }
            else {
                memset(output_block, 0, item_size);
            }
        }
        else {
            memset(output_block, 0, (size_t)padded_items * item_size);
        }
        return ME_EVAL_SUCCESS;
    }

    bool allow_repeat_reduce = false;
    me_reduce_kind rkind = ME_REDUCE_NONE;
    if (is_reduction_output && is_reduction_node(expr)) {
        rkind = reduction_kind(expr->function);
        if (rkind == ME_REDUCE_ANY || rkind == ME_REDUCE_ALL) {
            allow_repeat_reduce = true;
        }
        else if (rkind == ME_REDUCE_SUM) {
            const me_expr* arg = (const me_expr*)expr->parameters[0];
            if (arg && TYPE_MASK(arg->type) == ME_VARIABLE) {
                allow_repeat_reduce = true;
            }
        }
    }

    /* Decide whether repeat-eval is applicable, and precompute run layout. */
    int split_dim = -2;
    int64_t run_len = 0;
    int64_t total_runs = 0;
    bool repeat_eval_selected = false;
    if (!is_reduction_output || allow_repeat_reduce) {
        split_dim = nd - 2;
        run_len = valid_len[nd - 1];
        bool can_extend = (valid_len[nd - 1] == blockshape[nd - 1]);
        for (int i = nd - 2; i >= 0; i--) {
            if (can_extend && valid_len[i] == blockshape[i]) {
                if (run_len > LLONG_MAX / blockshape[i]) {
                    split_dim = -2;
                    break;
                }
                run_len *= blockshape[i];
                split_dim = i - 1;
            }
            else {
                break;
            }
        }

        if (split_dim >= -1 && run_len > 0 && run_len <= INT_MAX) {
            total_runs = 1;
            bool overflow = false;
            if (split_dim >= 0) {
                for (int i = 0; i <= split_dim; i++) {
                    if (total_runs > LLONG_MAX / valid_len[i]) {
                        overflow = true;
                        break;
                    }
                    total_runs *= valid_len[i];
                }
            }
            if (!overflow) {
                if (!is_reduction_output) {
                    repeat_eval_selected = true;
                }
                else if (rkind == ME_REDUCE_SUM) {
                    repeat_eval_selected = (total_runs <= 16);
                }
                else {
                    repeat_eval_selected = allow_repeat_reduce;
                }
            }
        }
    }

    /* Reduction fast paths (skip when repeat-eval is selected). */
    if (is_reduction_output && !repeat_eval_selected) {
        if (reduce_strided_predicate(expr, vars_block, n_vars, valid_len, stride, nd,
                                     valid_items, output_block)) {
            return ME_EVAL_SUCCESS;
        }
        if (reduce_strided_variable(expr, vars_block, n_vars, valid_len, stride, nd,
                                    valid_items, output_block)) {
            return ME_EVAL_SUCCESS;
        }
    }

    /* Repeat me_eval on contiguous valid runs instead of packing. */
    if (repeat_eval_selected) {
        const void* run_ptrs[ME_MAX_VARS];
        if (is_reduction_output) {
            me_scalar acc;
            bool acc_init = (rkind != ME_REDUCE_MIN && rkind != ME_REDUCE_MAX);
            const me_dtype output_type = expr->dtype;
            switch (output_type) {
            case ME_BOOL: acc.b = (rkind == ME_REDUCE_ALL); break;
            case ME_INT8:
            case ME_INT16:
            case ME_INT32:
            case ME_INT64: acc.i64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
            case ME_UINT8:
            case ME_UINT16:
            case ME_UINT32:
            case ME_UINT64: acc.u64 = (rkind == ME_REDUCE_PROD) ? 1 : 0; break;
            case ME_FLOAT32:
            case ME_FLOAT64: acc.f64 = (rkind == ME_REDUCE_PROD) ? 1.0 : 0.0; break;
            case ME_COMPLEX64: acc.c64 = (rkind == ME_REDUCE_PROD) ? (float _Complex)1.0f : (float _Complex)0.0f; break;
            case ME_COMPLEX128: acc.c128 = (rkind == ME_REDUCE_PROD) ? (double _Complex)1.0 : (double _Complex)0.0; break;
            default: acc_init = false; break;
            }

            int64_t indices[64] = {0};
            bool done = false;
            for (int64_t run = 0; run < total_runs && !done; run++) {
                int64_t off = 0;
                if (split_dim >= 0) {
                    for (int i = 0; i <= split_dim; i++) {
                        off += indices[i] * stride[i];
                    }
                }
                for (int v = 0; v < n_vars; v++) {
                    run_ptrs[v] = (const unsigned char*)vars_block[v] + (size_t)off * var_sizes[v];
                }
                me_scalar run_out;
                rc = me_eval(expr, run_ptrs, n_vars, &run_out, (int)run_len, params);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }

                me_scalar run_val;
                read_scalar(&run_out, output_type, &run_val);

                if (!acc_init) {
                    if (output_type == ME_FLOAT32) {
                        acc.f64 = (double)run_val.f32;
                    }
                    else if (output_type == ME_FLOAT64) {
                        acc.f64 = run_val.f64;
                    }
                    else {
                        acc = run_val;
                    }
                    acc_init = true;
                }
                else {
                    switch (rkind) {
                    case ME_REDUCE_SUM:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: acc.i64 += run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: acc.u64 += run_val.u64; break;
                        case ME_FLOAT32: acc.f64 += (double)run_val.f32; break;
                        case ME_FLOAT64: acc.f64 += run_val.f64; break;
                        case ME_COMPLEX64: acc.c64 += run_val.c64; break;
                        case ME_COMPLEX128: acc.c128 += run_val.c128; break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_PROD:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: acc.i64 *= run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: acc.u64 *= run_val.u64; break;
                        case ME_FLOAT32: acc.f64 *= (double)run_val.f32; break;
                        case ME_FLOAT64: acc.f64 *= run_val.f64; break;
                        case ME_COMPLEX64: acc.c64 *= run_val.c64; break;
                        case ME_COMPLEX128: acc.c128 *= run_val.c128; break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_MIN:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: if (run_val.i64 < acc.i64) acc.i64 = run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: if (run_val.u64 < acc.u64) acc.u64 = run_val.u64; break;
                        case ME_FLOAT32:
                            if (run_val.f32 != run_val.f32) { acc.f64 = NAN; done = true; }
                            else if (run_val.f32 < (float)acc.f64) acc.f64 = (double)run_val.f32;
                            break;
                        case ME_FLOAT64:
                            if (run_val.f64 != run_val.f64) { acc.f64 = NAN; done = true; }
                            else if (run_val.f64 < acc.f64) acc.f64 = run_val.f64;
                            break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_MAX:
                        switch (output_type) {
                        case ME_INT8:
                        case ME_INT16:
                        case ME_INT32:
                        case ME_INT64: if (run_val.i64 > acc.i64) acc.i64 = run_val.i64; break;
                        case ME_UINT8:
                        case ME_UINT16:
                        case ME_UINT32:
                        case ME_UINT64: if (run_val.u64 > acc.u64) acc.u64 = run_val.u64; break;
                        case ME_FLOAT32:
                            if (run_val.f32 != run_val.f32) { acc.f64 = NAN; done = true; }
                            else if (run_val.f32 > (float)acc.f64) acc.f64 = (double)run_val.f32;
                            break;
                        case ME_FLOAT64:
                            if (run_val.f64 != run_val.f64) { acc.f64 = NAN; done = true; }
                            else if (run_val.f64 > acc.f64) acc.f64 = run_val.f64;
                            break;
                        default: break;
                        }
                        break;
                    case ME_REDUCE_ANY:
                        if (output_type == ME_BOOL) {
                            acc.b = acc.b || run_val.b;
                            if (acc.b) done = true;
                        }
                        break;
                    case ME_REDUCE_ALL:
                        if (output_type == ME_BOOL) {
                            acc.b = acc.b && run_val.b;
                            if (!acc.b) done = true;
                        }
                        break;
                    default:
                        break;
                    }
                }

                if (split_dim >= 0) {
                    for (int i = split_dim; i >= 0; i--) {
                        indices[i]++;
                        if (indices[i] < valid_len[i]) break;
                        indices[i] = 0;
                    }
                }
            }

            if (output_type == ME_FLOAT32) {
                acc.f32 = (float)acc.f64;
            }
            write_scalar(output_block, output_type, output_type, &acc);
            return ME_EVAL_SUCCESS;
        }
        else {
            memset(output_block, 0, (size_t)padded_items * item_size);
            int64_t indices[64] = {0};
            for (int64_t run = 0; run < total_runs; run++) {
                int64_t off = 0;
                if (split_dim >= 0) {
                    for (int i = 0; i <= split_dim; i++) {
                        off += indices[i] * stride[i];
                    }
                }
                for (int v = 0; v < n_vars; v++) {
                    run_ptrs[v] = (const unsigned char*)vars_block[v] + (size_t)off * var_sizes[v];
                }
                void* out_ptr = (unsigned char*)output_block + (size_t)off * item_size;
                rc = me_eval(expr, run_ptrs, n_vars, out_ptr, (int)run_len, params);
                if (rc != ME_EVAL_SUCCESS) {
                    return rc;
                }
                if (split_dim >= 0) {
                    for (int i = split_dim; i >= 0; i--) {
                        indices[i]++;
                        if (indices[i] < valid_len[i]) break;
                        indices[i] = 0;
                    }
                }
            }
            return ME_EVAL_SUCCESS;
        }
    }

    void* packed_vars[ME_MAX_VARS];
    for (int v = 0; v < n_vars; v++) {
        packed_vars[v] = malloc((size_t)valid_items * var_sizes[v]);
        if (!packed_vars[v]) {
            for (int u = 0; u < v; u++) free(packed_vars[u]);
            return ME_EVAL_ERR_OOM;
        }
    }
    void* packed_out = NULL;
    if (!is_reduction_output) {
        packed_out = malloc((size_t)valid_items * item_size);
        if (!packed_out) {
            for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
            return ME_EVAL_ERR_OOM;
        }
    }

    /* Pack valid elements */
    int64_t indices[64] = {0};
    int64_t write_idx = 0;
    int64_t total_iters = 1;
    for (int i = 0; i < nd; i++) total_iters *= valid_len[i];
    for (int64_t it = 0; it < total_iters; it++) {
        int64_t off = 0;
        for (int i = 0; i < nd; i++) {
            off += indices[i] * stride[i];
        }
        for (int v = 0; v < n_vars; v++) {
            const unsigned char* src = (const unsigned char*)vars_block[v] + (size_t)off * var_sizes[v];
            memcpy((unsigned char*)packed_vars[v] + (size_t)write_idx * var_sizes[v], src, var_sizes[v]);
        }
        write_idx++;
        for (int i = nd - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < valid_len[i]) break;
            indices[i] = 0;
        }
    }

    if (is_reduction_output) {
        rc = me_eval(expr, (const void**)packed_vars, n_vars, output_block, (int)valid_items, params);
        if (rc != ME_EVAL_SUCCESS) {
            for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
            return rc;
        }
    }
    else {
        rc = me_eval(expr, (const void**)packed_vars, n_vars, packed_out, (int)valid_items, params);
        if (rc != ME_EVAL_SUCCESS) {
            for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
            free(packed_out);
            return rc;
        }

        /* Scatter back and zero padding */
        memset(output_block, 0, (size_t)padded_items * item_size);
        indices[0] = 0;
        for (int i = 1; i < nd; i++) indices[i] = 0;
        write_idx = 0;
        for (int64_t it = 0; it < total_iters; it++) {
            int64_t off = 0;
            for (int i = 0; i < nd; i++) {
                off += indices[i] * stride[i];
            }
            unsigned char* dst = (unsigned char*)output_block + (size_t)off * item_size;
            memcpy(dst, (unsigned char*)packed_out + (size_t)write_idx * item_size, item_size);
            write_idx++;
            for (int i = nd - 1; i >= 0; i--) {
                indices[i]++;
                if (indices[i] < valid_len[i]) break;
                indices[i] = 0;
            }
        }
    }

    for (int v = 0; v < n_vars; v++) free(packed_vars[v]);
    free(packed_out);

    return ME_EVAL_SUCCESS;
}

int me_nd_valid_nitems(const me_expr* expr, int64_t nchunk, int64_t nblock, int64_t* valid_nitems) {
    if (!valid_nitems) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    int64_t padded = 0;
    int rc = compute_valid_items(expr, nchunk, nblock, -1, valid_nitems, &padded);
    return rc;
}
