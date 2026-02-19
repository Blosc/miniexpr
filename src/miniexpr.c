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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "functions.h"
#include <stdlib.h>
#include "functions-simd.h"
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

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
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
#include "dsl_jit_test.h"

#define ME_DSL_MAX_NDIM 8
#define ME_DSL_JIT_SYMBOL_NAME "me_dsl_jit_kernel"
#define ME_DSL_JIT_CGEN_VERSION 5
#ifndef ME_USE_LIBTCC_FALLBACK
#define ME_USE_LIBTCC_FALLBACK 0
#endif
#ifndef ME_USE_WASM32_JIT
#define ME_USE_WASM32_JIT 0
#endif
#ifndef ME_WASM32_SIDE_MODULE
#define ME_WASM32_SIDE_MODULE 0
#endif
#ifndef ME_DSL_TRACE_DEFAULT
#define ME_DSL_TRACE_DEFAULT 0
#endif
#ifndef ME_DSL_WHILE_MAX_ITERS_DEFAULT
#define ME_DSL_WHILE_MAX_ITERS_DEFAULT 10000000LL
#endif
#ifndef ME_DSL_JIT_LIBTCC_DEFAULT_PATH
#define ME_DSL_JIT_LIBTCC_DEFAULT_PATH ""
#endif
#define ME_LAST_ERROR_MSG_CAP 256

#if defined(_MSC_VER)
#define ME_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define ME_THREAD_LOCAL _Thread_local
#elif defined(__GNUC__)
#define ME_THREAD_LOCAL __thread
#else
#define ME_THREAD_LOCAL
#endif

#if ME_USE_WASM32_JIT
/* wasm32 kernels use int for nitems (TCC wasm32 int64_t limitation). */
typedef int (*me_dsl_jit_kernel_fn)(const void **inputs, void *output, int nitems);
#else
typedef int (*me_dsl_jit_kernel_fn)(const void **inputs, void *output, int64_t nitems);
#endif

#if ME_USE_WASM32_JIT
#if !ME_WASM32_SIDE_MODULE
/* Forward declarations for main-module EM_JS helpers defined later in the file. */
int me_wasm_jit_instantiate(const unsigned char *wasm_bytes, int wasm_len,
                            int bridge_lookup_fn_idx);
void me_wasm_jit_free_fn(int idx);
#endif

static me_wasm_jit_instantiate_helper g_me_wasm_jit_instantiate_helper = NULL;
static me_wasm_jit_free_helper g_me_wasm_jit_free_helper = NULL;

#if ME_WASM32_SIDE_MODULE
static bool me_wasm_jit_helpers_available(void) {
    return g_me_wasm_jit_instantiate_helper != NULL &&
           g_me_wasm_jit_free_helper != NULL;
}
#endif

static int me_wasm_jit_instantiate_dispatch(const unsigned char *wasm_bytes, int wasm_len,
                                            int bridge_lookup_fn_idx) {
#if ME_WASM32_SIDE_MODULE
    if (!g_me_wasm_jit_instantiate_helper) {
        return 0;
    }
    return g_me_wasm_jit_instantiate_helper(wasm_bytes, wasm_len, bridge_lookup_fn_idx);
#else
    return me_wasm_jit_instantiate(wasm_bytes, wasm_len, bridge_lookup_fn_idx);
#endif
}

static void me_wasm_jit_free_dispatch(int idx) {
#if ME_WASM32_SIDE_MODULE
    if (g_me_wasm_jit_free_helper) {
        g_me_wasm_jit_free_helper(idx);
    }
#else
    me_wasm_jit_free_fn(idx);
#endif
}
#endif

/* ND metadata attached to compiled expressions (used by me_eval_nd). */
typedef struct {
    int ndims;
    /* Layout: shape[ndims], chunkshape[ndims], blockshape[ndims] (all int64_t). */
    int64_t data[1];
} me_nd_info;

static ME_THREAD_LOCAL char g_me_last_error_msg[ME_LAST_ERROR_MSG_CAP];

static void me_clear_last_error_message(void) {
    g_me_last_error_msg[0] = '\0';
}

static void me_set_last_error_message(const char *msg) {
    if (!msg || msg[0] == '\0') {
        me_clear_last_error_message();
        return;
    }
    snprintf(g_me_last_error_msg, sizeof(g_me_last_error_msg), "%s", msg);
}

static void me_set_last_error_messagef(const char *fmt, ...) {
    if (!fmt || fmt[0] == '\0') {
        me_clear_last_error_message();
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_me_last_error_msg, sizeof(g_me_last_error_msg), fmt, args);
    va_end(args);
}

static float me_crealf(float _Complex v);
static float me_cimagf(float _Complex v);
static double me_creal(double _Complex v);
static double me_cimag(double _Complex v);

static int private_compile_ex(const char* expression, const me_variable* variables, int var_count,
                              void* output, int nitems, me_dtype dtype, int* error, me_expr** out);

static bool is_function_entry(const me_variable *var) {
    if (!var) {
        return false;
    }
    return IS_FUNCTION(var->type) || IS_CLOSURE(var->type);
}

static bool is_variable_entry(const me_variable *var) {
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
            me_dsl_compiled_expr start;
            me_dsl_compiled_expr stop;
            me_dsl_compiled_expr step;
            me_dsl_compiled_block body;
        } for_loop;
        struct {
            me_dsl_compiled_expr cond;
            me_dsl_compiled_block body;
        } while_loop;
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
    me_dsl_fp_mode fp_mode;
    me_dsl_compiler compiler;
    bool guaranteed_return;
    bool output_is_scalar;
    me_dtype output_dtype;
    me_dsl_jit_ir_program *jit_ir;
    uint64_t jit_ir_fingerprint;
    int jit_ir_error_line;
    int jit_ir_error_column;
    char jit_ir_error[128];
    char *jit_c_source;
    bool jit_use_runtime_math_bridge;
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

static void dsl_jit_libtcc_delete_state(void *state);
static bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program);
static const char *dsl_jit_libtcc_error_message(void);
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
static bool dsl_jit_c_compiler_available(void);
#endif
static bool dsl_jit_cc_math_bridge_available(void);

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
    if (strcmp(name, "int") == 0 || strcmp(name, "float") == 0 || strcmp(name, "bool") == 0) {
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

static double dsl_cast_int_intrinsic(double x) {
    return (double)(int64_t)x;
}

static double dsl_cast_float_intrinsic(double x) {
    return x;
}

static double dsl_cast_bool_intrinsic(double x) {
    return x != 0.0 ? 1.0 : 0.0;
}

static me_dtype dsl_cast_int_target_dtype(me_dtype expr_dtype) {
    switch (expr_dtype) {
    case ME_INT8:
    case ME_INT16:
    case ME_INT32:
    case ME_INT64:
        return expr_dtype;
    default:
        return ME_INT64;
    }
}

static me_dtype dsl_cast_float_target_dtype(me_dtype expr_dtype) {
    switch (expr_dtype) {
    case ME_FLOAT32:
    case ME_FLOAT64:
        return expr_dtype;
    default:
        return ME_FLOAT64;
    }
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

static bool dsl_is_ident_char_scan(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static bool dsl_is_cast_intrinsic_name_len(const char *s, size_t n) {
    return (n == 3 && strncmp(s, "int", 3) == 0) ||
        (n == 5 && strncmp(s, "float", 5) == 0) ||
        (n == 4 && strncmp(s, "bool", 4) == 0);
}

static bool dsl_validate_cast_intrinsics_usage(const char *text, int *error_offset) {
    if (error_offset) {
        *error_offset = -1;
    }
    if (!text) {
        return true;
    }

    const char *p = text;
    while (*p) {
        if (*p == '"' || *p == '\'') {
            char quote = *p++;
            while (*p) {
                if (*p == '\\' && p[1] != '\0') {
                    p += 2;
                    continue;
                }
                if (*p == quote) {
                    p++;
                    break;
                }
                p++;
            }
            continue;
        }

        if (!(isalpha((unsigned char)*p) || *p == '_')) {
            p++;
            continue;
        }

        const char *ident_start = p;
        p++;
        while (*p && dsl_is_ident_char_scan(*p)) {
            p++;
        }
        size_t ident_len = (size_t)(p - ident_start);
        if (!dsl_is_cast_intrinsic_name_len(ident_start, ident_len)) {
            continue;
        }

        const char *q = p;
        while (*q && isspace((unsigned char)*q)) {
            q++;
        }
        if (*q != '(') {
            continue;
        }

        const char *arg_start = q + 1;
        const char *r = arg_start;
        int depth = 1;
        int top_level_commas = 0;
        char quote = '\0';
        while (*r && depth > 0) {
            char c = *r;
            if (quote) {
                if (c == '\\' && r[1] != '\0') {
                    r += 2;
                    continue;
                }
                if (c == quote) {
                    quote = '\0';
                }
                r++;
                continue;
            }
            if (c == '"' || c == '\'') {
                quote = c;
                r++;
                continue;
            }
            if (c == '(') {
                depth++;
            }
            else if (c == ')') {
                depth--;
                if (depth == 0) {
                    break;
                }
            }
            else if (c == ',' && depth == 1) {
                top_level_commas++;
            }
            r++;
        }

        if (depth != 0) {
            if (error_offset) {
                *error_offset = (int)(ident_start - text) + 1;
            }
            return false;
        }

        const char *arg_end = r;
        const char *trim_start = arg_start;
        const char *trim_end = arg_end;
        while (trim_start < trim_end && isspace((unsigned char)*trim_start)) {
            trim_start++;
        }
        while (trim_end > trim_start && isspace((unsigned char)trim_end[-1])) {
            trim_end--;
        }

        if (trim_start == trim_end || top_level_commas > 0) {
            if (error_offset) {
                *error_offset = (int)(ident_start - text) + 1;
            }
            return false;
        }

        p = r + 1;
    }

    return true;
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

static void dsl_compiled_program_free(me_dsl_compiled_program *program) {
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
        me_wasm_jit_free_dispatch((int)(uintptr_t)program->jit_kernel_fn);
    }
    /* jit_dl_handle holds the wasm32 JIT scratch memory. */
    free(program->jit_dl_handle);
    program->jit_dl_handle = NULL;
#endif
    program->jit_kernel_fn = NULL;
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

static int private_compile_ex(const char* expression, const me_variable* variables, int var_count,
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
    me_variable* vars_copy = NULL;
    if (variables && var_count > 0) {
        vars_copy = malloc(var_count * sizeof(me_variable));
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
        const me_variable* vars_check = vars_copy ? vars_copy : variables;
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
        const me_variable* vars_check = vars_copy ? vars_copy : variables;
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
    bool allow_new_locals;
    int *error_pos;
    me_dsl_compiled_expr *output_expr;
    bool has_return;
    me_dtype return_dtype;
    me_dsl_compiled_program *program;
    const me_variable *funcs;
    int func_count;
    char *error_reason;
    size_t error_reason_cap;
} dsl_compile_ctx;

static void dsl_set_error_reason(dsl_compile_ctx *ctx, const char *msg) {
    if (!ctx || !ctx->error_reason || ctx->error_reason_cap == 0) {
        return;
    }
    if (!msg || msg[0] == '\0') {
        ctx->error_reason[0] = '\0';
        return;
    }
    snprintf(ctx->error_reason, ctx->error_reason_cap, "%s", msg);
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

static const char *dsl_compiler_name(me_dsl_compiler compiler) {
    switch (compiler) {
    case ME_DSL_COMPILER_LIBTCC:
        return "tcc";
    case ME_DSL_COMPILER_CC:
        return "cc";
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
        return ME_DSL_TRACE_DEFAULT != 0;
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

static int64_t dsl_while_max_iters(void) {
    const char *env = getenv("ME_DSL_WHILE_MAX_ITERS");
    if (!env || env[0] == '\0') {
        return (int64_t)ME_DSL_WHILE_MAX_ITERS_DEFAULT;
    }
    errno = 0;
    char *end = NULL;
    long long v = strtoll(env, &end, 10);
    if (errno != 0 || end == env) {
        return (int64_t)ME_DSL_WHILE_MAX_ITERS_DEFAULT;
    }
    while (*end && isspace((unsigned char)*end)) {
        end++;
    }
    if (*end != '\0') {
        return (int64_t)ME_DSL_WHILE_MAX_ITERS_DEFAULT;
    }
    return (int64_t)v;
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

static bool me_eval_jit_disabled(const me_eval_params *params) {
    return params && params->jit_mode == ME_JIT_OFF;
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
                if (memcmp(start, "break", 5) == 0 ||
                    memcmp(start, "print", 5) == 0 ||
                    memcmp(start, "while", 5) == 0) {
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

static char *dsl_wrap_expression_as_function(const char *expression,
                                             const me_variable *variables,
                                             int var_count) {
    if (!expression) {
        return NULL;
    }
    size_t expr_len = strlen(expression);
    size_t params_len = 0;
    int n_params = 0;
    if (variables && var_count > 0) {
        for (int i = 0; i < var_count; i++) {
            if (!is_variable_entry(&variables[i])) {
                continue;
            }
            if (!variables[i].name) {
                return NULL;
            }
            params_len += strlen(variables[i].name);
            n_params++;
        }
    }
    if (n_params > 0) {
        params_len += (size_t)(n_params - 1) * 2;  // ", "
    }

    const char prefix[] = "def __me_auto(";
    const char middle[] = "):\n    return ";
    size_t total = sizeof(prefix) - 1 + params_len + sizeof(middle) - 1 + expr_len + 1;
    char *dsl = malloc(total);
    if (!dsl) {
        return NULL;
    }

    char *p = dsl;
    memcpy(p, prefix, sizeof(prefix) - 1);
    p += sizeof(prefix) - 1;
    if (variables && var_count > 0) {
        bool first = true;
        for (int i = 0; i < var_count; i++) {
            if (!is_variable_entry(&variables[i])) {
                continue;
            }
            const char *name = variables[i].name;
            if (!first) {
                *p++ = ',';
                *p++ = ' ';
            }
            size_t nlen = strlen(name);
            memcpy(p, name, nlen);
            p += nlen;
            first = false;
        }
    }
    memcpy(p, middle, sizeof(middle) - 1);
    p += sizeof(middle) - 1;
    memcpy(p, expression, expr_len);
    p += expr_len;
    *p = '\0';
    return dsl;
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
        case ME_DSL_STMT_WHILE:
            expr_text = stmt->as.while_loop.cond ? stmt->as.while_loop.cond->text : NULL;
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
        if (stmt->kind == ME_DSL_STMT_WHILE) {
            dsl_scan_reserved_usage_block(&stmt->as.while_loop.body, uses_i_mask, uses_n_mask, uses_ndim);
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

static bool dsl_build_var_lookup(const me_dsl_var_table *table, const me_variable *funcs,
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
    int cast_error_offset = -1;
    if (!dsl_validate_cast_intrinsics_usage(expr_node->text, &cast_error_offset)) {
        dsl_set_error_reason(ctx, "invalid cast intrinsic usage: int()/float()/bool() must be called as functions");
        if (ctx->error_pos) {
            int offset = dsl_offset_from_linecol(ctx->source, expr_node->line, expr_node->column);
            if (offset >= 0 && cast_error_offset > 0) {
                *ctx->error_pos = offset + cast_error_offset - 1;
            }
            else {
                *ctx->error_pos = offset >= 0 ? offset : -1;
            }
        }
        return false;
    }
    me_variable cast_funcs[3];
    int cast_count = 0;
    cast_funcs[cast_count++] = (me_variable){
        .name = "int",
        .dtype = dsl_cast_int_target_dtype(expr_dtype),
        .address = (const void *)dsl_cast_int_intrinsic,
        .type = ME_FUNCTION1 | ME_FLAG_PURE,
        .context = NULL,
        .itemsize = 0
    };
    cast_funcs[cast_count++] = (me_variable){
        .name = "float",
        .dtype = dsl_cast_float_target_dtype(expr_dtype),
        .address = (const void *)dsl_cast_float_intrinsic,
        .type = ME_FUNCTION1 | ME_FLAG_PURE,
        .context = NULL,
        .itemsize = 0
    };
    cast_funcs[cast_count++] = (me_variable){
        .name = "bool",
        .dtype = ME_BOOL,
        .address = (const void *)dsl_cast_bool_intrinsic,
        .type = ME_FUNCTION1 | ME_FLAG_PURE,
        .context = NULL,
        .itemsize = 0
    };

    me_variable *all_funcs = NULL;
    int all_func_count = ctx->func_count + cast_count;
    if (all_func_count > 0) {
        all_funcs = calloc((size_t)all_func_count, sizeof(*all_funcs));
        if (!all_funcs) {
            return false;
        }
        for (int i = 0; i < ctx->func_count; i++) {
            all_funcs[i] = ctx->funcs[i];
        }
        for (int i = 0; i < cast_count; i++) {
            all_funcs[ctx->func_count + i] = cast_funcs[i];
        }
    }

    me_variable *lookup = NULL;
    int lookup_count = 0;
    if (!dsl_build_var_lookup(&ctx->program->vars, all_funcs, all_func_count,
                              &lookup, &lookup_count)) {
        free(all_funcs);
        return false;
    }
    free(all_funcs);
    me_expr *compiled = NULL;
    int local_error = 0;
    int rc = private_compile_ex(expr_node->text, lookup, lookup_count,
                                NULL, 0, expr_dtype, &local_error, &compiled);
    free(lookup);
    if (rc != ME_COMPILE_SUCCESS || !compiled) {
        dsl_set_error_reason(ctx, "failed to compile DSL expression");
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

static bool dsl_compile_condition_expr(dsl_compile_ctx *ctx, const me_dsl_expr *expr_node,
                                       me_dsl_compiled_expr *out_expr) {
    if (!ctx || !expr_node || !expr_node->text || !out_expr) {
        return false;
    }
    int saved_error = 0;
    if (ctx->error_pos) {
        saved_error = *ctx->error_pos;
    }

    if (dsl_compile_expr(ctx, expr_node, ME_AUTO, out_expr)) {
        return true;
    }

    if (ctx->error_pos) {
        *ctx->error_pos = saved_error;
    }
    size_t expr_len = strlen(expr_node->text);
    size_t need = expr_len + 16;
    char *truthy_text = malloc(need);
    if (!truthy_text) {
        return false;
    }
    snprintf(truthy_text, need, "(%s) != \"\"", expr_node->text);
    me_dsl_expr truthy_expr = {
        .text = truthy_text,
        .line = expr_node->line,
        .column = expr_node->column
    };
    bool ok = dsl_compile_expr(ctx, &truthy_expr, ME_BOOL, out_expr);
    free(truthy_text);
    return ok;
}

static bool dsl_split_top_level_csv(const char *text, char ***out_parts, int *out_nparts) {
    if (!text || !out_parts || !out_nparts) {
        return false;
    }
    *out_parts = NULL;
    *out_nparts = 0;

    const char *part_start = text;
    const char *p = text;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    char quote = '\0';
    char **parts = NULL;
    int nparts = 0;

    while (*p) {
        char c = *p;
        if (quote) {
            if (c == '\\' && p[1] != '\0') {
                p += 2;
                continue;
            }
            if (c == quote) {
                quote = '\0';
            }
            p++;
            continue;
        }
        if (c == '"' || c == '\'') {
            quote = c;
            p++;
            continue;
        }
        if (c == '(') {
            paren_depth++;
            p++;
            continue;
        }
        if (c == ')') {
            if (paren_depth > 0) {
                paren_depth--;
            }
            p++;
            continue;
        }
        if (c == '[') {
            bracket_depth++;
            p++;
            continue;
        }
        if (c == ']') {
            if (bracket_depth > 0) {
                bracket_depth--;
            }
            p++;
            continue;
        }
        if (c == '{') {
            brace_depth++;
            p++;
            continue;
        }
        if (c == '}') {
            if (brace_depth > 0) {
                brace_depth--;
            }
            p++;
            continue;
        }
        if (c == ',' && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
            char *part = dsl_trim_copy(part_start, p);
            if (!part || part[0] == '\0') {
                free(part);
                for (int i = 0; i < nparts; i++) {
                    free(parts[i]);
                }
                free(parts);
                return false;
            }
            char **next = realloc(parts, (size_t)(nparts + 1) * sizeof(*next));
            if (!next) {
                free(part);
                for (int i = 0; i < nparts; i++) {
                    free(parts[i]);
                }
                free(parts);
                return false;
            }
            parts = next;
            parts[nparts++] = part;
            part_start = p + 1;
        }
        p++;
    }

    if (quote || paren_depth != 0 || bracket_depth != 0 || brace_depth != 0) {
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }

    char *tail = dsl_trim_copy(part_start, p);
    if (!tail || tail[0] == '\0') {
        free(tail);
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }
    char **next = realloc(parts, (size_t)(nparts + 1) * sizeof(*next));
    if (!next) {
        free(tail);
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }
    parts = next;
    parts[nparts++] = tail;

    *out_parts = parts;
    *out_nparts = nparts;
    return true;
}

static bool dsl_compile_range_bound_expr(dsl_compile_ctx *ctx, const char *text,
                                         int line, int column,
                                         me_dsl_compiled_expr *out_expr) {
    if (!ctx || !text || !out_expr) {
        return false;
    }
    me_dsl_expr expr = {
        .text = (char *)text,
        .line = line,
        .column = column
    };
    return dsl_compile_expr(ctx, &expr, ME_AUTO, out_expr);
}

static bool dsl_compile_for_range_args(dsl_compile_ctx *ctx, const me_dsl_stmt *stmt,
                                       me_dsl_compiled_stmt *compiled) {
    if (!ctx || !stmt || !compiled || !stmt->as.for_loop.limit || !stmt->as.for_loop.limit->text) {
        return false;
    }

    char **parts = NULL;
    int nparts = 0;
    if (!dsl_split_top_level_csv(stmt->as.for_loop.limit->text, &parts, &nparts)) {
        if (ctx->error_pos) {
            *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
        }
        return false;
    }
    if (nparts < 1 || nparts > 3) {
        if (ctx->error_pos) {
            *ctx->error_pos = dsl_offset_from_linecol(ctx->source, stmt->line, stmt->column);
        }
        for (int i = 0; i < nparts; i++) {
            free(parts[i]);
        }
        free(parts);
        return false;
    }

    const char *start_text = "0";
    const char *stop_text = parts[0];
    const char *step_text = "1";
    if (nparts == 2) {
        start_text = parts[0];
        stop_text = parts[1];
    }
    else if (nparts == 3) {
        start_text = parts[0];
        stop_text = parts[1];
        step_text = parts[2];
    }

    bool ok = dsl_compile_range_bound_expr(ctx, start_text, stmt->line, stmt->column,
                                           &compiled->as.for_loop.start) &&
              dsl_compile_range_bound_expr(ctx, stop_text, stmt->line, stmt->column,
                                           &compiled->as.for_loop.stop) &&
              dsl_compile_range_bound_expr(ctx, step_text, stmt->line, stmt->column,
                                           &compiled->as.for_loop.step);
    if (!ok) {
        dsl_compiled_expr_free(&compiled->as.for_loop.start);
        dsl_compiled_expr_free(&compiled->as.for_loop.stop);
        dsl_compiled_expr_free(&compiled->as.for_loop.step);
    }

    for (int i = 0; i < nparts; i++) {
        free(parts[i]);
    }
    free(parts);
    return ok;
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

static int dsl_jit_backend_tag(const me_dsl_compiled_program *program) {
    if (!program) {
        return 1;
    }
    if (program->compiler == ME_DSL_COMPILER_CC) {
        return program->jit_use_runtime_math_bridge ? 3 : 2;
    }
    return 1;
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
    h = dsl_jit_hash_i32(h, dsl_jit_backend_tag(program));
    return h;
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
/* In-process negative cache for recent JIT runtime failures. */
#define ME_DSL_JIT_NEG_CACHE_SLOTS 64
#define ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET 2
#define ME_DSL_JIT_NEG_CACHE_SHORT_COOLDOWN_SEC 10
#define ME_DSL_JIT_NEG_CACHE_LONG_COOLDOWN_SEC 120
#define ME_DSL_JIT_POS_CACHE_SLOTS 64
#define ME_DSL_JIT_META_MAGIC 0x4d454a49544d4554ULL
#define ME_DSL_JIT_META_VERSION 5

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
    int32_t fp_mode;
    int32_t compiler;
    int32_t nparams;
    int32_t param_dtypes[ME_MAX_VARS];
    uint64_t toolchain_hash;
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

static void dsl_jit_pos_cache_evict(uint64_t key) {
    int slot = dsl_jit_pos_cache_find_slot(key);
    if (slot < 0) {
        return;
    }
    if (g_dsl_jit_pos_cache[slot].handle) {
        dlclose(g_dsl_jit_pos_cache[slot].handle);
    }
    memset(&g_dsl_jit_pos_cache[slot], 0, sizeof(g_dsl_jit_pos_cache[slot]));
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

static uint64_t dsl_jit_toolchain_hash(const me_dsl_compiled_program *program) {
    if (!program) {
        return 1469598103934665603ULL;
    }
    if (program->compiler == ME_DSL_COMPILER_LIBTCC) {
        const char *tcc_opts = getenv("ME_DSL_JIT_TCC_OPTIONS");
        const char *tcc_lib_path = getenv("ME_DSL_JIT_TCC_LIB_PATH");
        uint64_t h = dsl_jit_hash_cstr(1469598103934665603ULL, "tcc");
        h = dsl_jit_hash_cstr(h, tcc_opts ? tcc_opts : "");
        return dsl_jit_hash_cstr(h, tcc_lib_path ? tcc_lib_path : "");
    }
    const char *cc = getenv("CC");
    const char *cflags = getenv("CFLAGS");
    const char *fp_cflags = dsl_jit_fp_mode_cflags(program->fp_mode);
    if (!cc || cc[0] == '\0') {
        cc = "cc";
    }
    if (!cflags) {
        cflags = "";
    }
    if (!fp_cflags) {
        fp_cflags = "";
    }
    uint64_t h = dsl_jit_hash_cstr(1469598103934665603ULL, cc);
    h = dsl_jit_hash_cstr(h, fp_cflags);
    return dsl_jit_hash_cstr(h, cflags);
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
    meta->fp_mode = (int32_t)program->fp_mode;
    meta->compiler = (int32_t)program->compiler;
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
    meta->toolchain_hash = dsl_jit_toolchain_hash(program);
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
        /* Avoid cross-user permission conflicts when TMPDIR is not set. */
        if (snprintf(out, out_size, "/tmp/miniexpr-jit-%lu", (unsigned long)getuid()) >= (int)out_size) {
            return false;
        }
        return dsl_jit_ensure_dir(out);
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

static bool dsl_jit_copy_file(const char *src, const char *dst) {
    if (!src || !dst) {
        return false;
    }
    FILE *fin = fopen(src, "rb");
    if (!fin) {
        return false;
    }
    FILE *fout = fopen(dst, "wb");
    if (!fout) {
        fclose(fin);
        return false;
    }
    unsigned char buf[4096];
    bool ok = true;
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), fin)) > 0) {
        if (fwrite(buf, 1, n, fout) != n) {
            ok = false;
            break;
        }
    }
    if (ferror(fin)) {
        ok = false;
    }
    if (fclose(fin) != 0) {
        ok = false;
    }
    if (fclose(fout) != 0) {
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

#endif

#if ME_USE_LIBTCC_FALLBACK
typedef struct TCCState me_tcc_state;

typedef me_tcc_state *(*me_tcc_new_fn)(void);
typedef void (*me_tcc_delete_fn)(me_tcc_state *s);
typedef int (*me_tcc_set_output_type_fn)(me_tcc_state *s, int output_type);
typedef int (*me_tcc_compile_string_fn)(me_tcc_state *s, const char *buf);
typedef int (*me_tcc_relocate_fn)(me_tcc_state *s);
typedef void *(*me_tcc_get_symbol_fn)(me_tcc_state *s, const char *name);
typedef int (*me_tcc_set_options_fn)(me_tcc_state *s, const char *str);
typedef int (*me_tcc_add_library_path_fn)(me_tcc_state *s, const char *path);
typedef int (*me_tcc_add_library_fn)(me_tcc_state *s, const char *libraryname);
typedef int (*me_tcc_add_symbol_fn)(me_tcc_state *s, const char *name, const void *val);
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
    me_tcc_add_library_path_fn tcc_add_library_path_fn;
    me_tcc_add_library_fn tcc_add_library_fn;
    me_tcc_add_symbol_fn tcc_add_symbol_fn;
    me_tcc_set_lib_path_fn tcc_set_lib_path_fn;
    char error[160];
} me_dsl_tcc_api;

static me_dsl_tcc_api g_dsl_tcc_api;

static const char *dsl_jit_libtcc_error_message(void) {
    if (g_dsl_tcc_api.error[0]) {
        return g_dsl_tcc_api.error;
    }
    return "tcc backend unavailable";
}

static bool dsl_jit_module_path_from_symbol(const void *symbol, char *out, size_t out_size) {
    if (!symbol || !out || out_size == 0) {
        return false;
    }
#if defined(_WIN32) || defined(_WIN64)
    HMODULE module = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)symbol, &module)) {
        return false;
    }
    DWORD n = GetModuleFileNameA(module, out, (DWORD)out_size);
    if (n == 0 || n >= (DWORD)out_size) {
        return false;
    }
    out[n] = '\0';
    return true;
#else
    Dl_info info;
    if (dladdr(symbol, &info) == 0 || !info.dli_fname || info.dli_fname[0] == '\0') {
        return false;
    }
    int n = snprintf(out, out_size, "%s", info.dli_fname);
    return n > 0 && (size_t)n < out_size;
#endif
}

static void *dsl_jit_dynlib_open(const char *path) {
#if defined(_WIN32) || defined(_WIN64)
    return (void *)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *dsl_jit_dynlib_symbol(void *handle, const char *name) {
#if defined(_WIN32) || defined(_WIN64)
    return (void *)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

static void dsl_jit_dynlib_close(void *handle) {
    if (!handle) {
        return;
    }
#if defined(_WIN32) || defined(_WIN64)
    (void)FreeLibrary((HMODULE)handle);
#else
    dlclose(handle);
#endif
}

static const char *dsl_jit_dynlib_last_error(void) {
#if defined(_WIN32) || defined(_WIN64)
    static char err[160];
    DWORD code = GetLastError();
    if (code == 0) {
        err[0] = '\0';
        return err;
    }
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             NULL, code, 0, err, (DWORD)sizeof(err), NULL);
    if (n == 0) {
        snprintf(err, sizeof(err), "Win32 error %lu", (unsigned long)code);
        return err;
    }
    while (n > 0 && (err[n - 1] == '\r' || err[n - 1] == '\n' ||
                     err[n - 1] == ' ' || err[n - 1] == '\t')) {
        err[n - 1] = '\0';
        n--;
    }
    return err;
#else
    const char *err = dlerror();
    return err ? err : "";
#endif
}

static bool dsl_jit_path_dirname(const char *path, char *out, size_t out_size) {
    if (!path || !out || out_size == 0) {
        return false;
    }
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    if (!slash || (backslash && backslash > slash)) {
        slash = backslash;
    }
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
    char module_path[PATH_MAX];
    if (!dsl_jit_module_path_from_symbol((const void *)&dsl_jit_libtcc_path_near_self,
                                         module_path, sizeof(module_path))) {
        return false;
    }
    char dir[PATH_MAX];
    if (!dsl_jit_path_dirname(module_path, dir, sizeof(dir))) {
        return false;
    }
#if defined(_WIN32) || defined(_WIN64)
    const char *name = "tcc.dll";
#elif defined(__APPLE__)
    const char *name = "libtcc.dylib";
#else
    const char *name = "libtcc.so";
#endif
    char sep = '/';
    if (strchr(dir, '\\')) {
        sep = '\\';
    }
    int n = snprintf(out, out_size, "%s%c%s", dir, sep, name);
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
    char module_path[PATH_MAX];
    if (!dsl_jit_module_path_from_symbol((const void *)g_dsl_tcc_api.tcc_new_fn,
                                         module_path, sizeof(module_path))) {
        return false;
    }
    return dsl_jit_path_dirname(module_path, out, out_size);
}

static void dsl_jit_libtcc_add_library_path_if_exists(me_tcc_state *state, const char *path) {
    if (!state || !g_dsl_tcc_api.tcc_add_library_path_fn || !path || path[0] == '\0') {
        return;
    }
#if defined(_WIN32) || defined(_WIN64)
    DWORD attrs = GetFileAttributesA(path);
    if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        (void)g_dsl_tcc_api.tcc_add_library_path_fn(state, path);
    }
#else
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        (void)g_dsl_tcc_api.tcc_add_library_path_fn(state, path);
    }
#endif
}

static void dsl_jit_libtcc_add_multiarch_paths(me_tcc_state *state) {
#if defined(__linux__)
    if (!state || !g_dsl_tcc_api.tcc_add_library_path_fn) {
        return;
    }
    const char *paths[] = {
#if defined(__x86_64__) || defined(__amd64__)
        "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
#elif defined(__aarch64__)
        "/usr/lib/aarch64-linux-gnu", "/lib/aarch64-linux-gnu",
#elif defined(__arm__)
        "/usr/lib/arm-linux-gnueabihf", "/lib/arm-linux-gnueabihf",
        "/usr/lib/arm-linux-gnueabi", "/lib/arm-linux-gnueabi",
#elif defined(__riscv) && (__riscv_xlen == 64)
        "/usr/lib/riscv64-linux-gnu", "/lib/riscv64-linux-gnu",
#elif defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
        "/usr/lib/powerpc64le-linux-gnu", "/lib/powerpc64le-linux-gnu",
#elif defined(__s390x__)
        "/usr/lib/s390x-linux-gnu", "/lib/s390x-linux-gnu",
#elif defined(__i386__)
        "/usr/lib/i386-linux-gnu", "/lib/i386-linux-gnu",
#endif
        "/usr/lib64", "/lib64",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        dsl_jit_libtcc_add_library_path_if_exists(state, paths[i]);
    }
#else
    (void)state;
#endif
}

static double dsl_jit_bridge_apply_unary_f64(void (*fn)(const double *, double *, int), double x) {
    double in_buf[1];
    double out_buf[1];
    in_buf[0] = x;
    out_buf[0] = 0.0;
    fn(in_buf, out_buf, 1);
    return out_buf[0];
}

static void dsl_jit_bridge_apply_unary_vector_f64(void (*fn)(const double *, double *, int),
                                                   const double *in, double *out, int64_t nitems) {
    if (!fn || !in || !out || nitems <= 0) {
        return;
    }
    int64_t remaining = nitems;
    const double *pin = in;
    double *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > INT_MAX) ? INT_MAX : (int)remaining;
        fn(pin, pout, chunk);
        pin += chunk;
        pout += chunk;
        remaining -= chunk;
    }
}

static void dsl_jit_bridge_apply_unary_vector_f32(void (*fn)(const float *, float *, int),
                                                   const float *in, float *out, int64_t nitems) {
    if (!fn || !in || !out || nitems <= 0) {
        return;
    }
    int64_t remaining = nitems;
    const float *pin = in;
    float *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > INT_MAX) ? INT_MAX : (int)remaining;
        fn(pin, pout, chunk);
        pin += chunk;
        pout += chunk;
        remaining -= chunk;
    }
}

static void dsl_jit_bridge_apply_binary_vector_f64(void (*fn)(const double *, const double *, double *, int),
                                                    const double *a, const double *b,
                                                    double *out, int64_t nitems) {
    if (!fn || !a || !b || !out || nitems <= 0) {
        return;
    }
    int64_t remaining = nitems;
    const double *pa = a;
    const double *pb = b;
    double *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > INT_MAX) ? INT_MAX : (int)remaining;
        fn(pa, pb, pout, chunk);
        pa += chunk;
        pb += chunk;
        pout += chunk;
        remaining -= chunk;
    }
}

static void dsl_jit_bridge_apply_binary_vector_f32(void (*fn)(const float *, const float *, float *, int),
                                                    const float *a, const float *b,
                                                    float *out, int64_t nitems) {
    if (!fn || !a || !b || !out || nitems <= 0) {
        return;
    }
    int64_t remaining = nitems;
    const float *pa = a;
    const float *pb = b;
    float *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > INT_MAX) ? INT_MAX : (int)remaining;
        fn(pa, pb, pout, chunk);
        pa += chunk;
        pb += chunk;
        pout += chunk;
        remaining -= chunk;
    }
}

static double dsl_jit_bridge_exp10(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_exp10_dispatch, x);
}

static double dsl_jit_bridge_sinpi(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_sinpi_dispatch, x);
}

static double dsl_jit_bridge_cospi(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_cospi_dispatch, x);
}

static double dsl_jit_bridge_logaddexp(double a, double b) {
    if (a == b) {
        return a + log1p(1.0);
    }
    double hi = (a > b) ? a : b;
    double lo = (a > b) ? b : a;
    return hi + log1p(exp(lo - hi));
}

static double dsl_jit_bridge_where(double c, double x, double y) {
    return (c != 0.0) ? x : y;
}

static void dsl_jit_bridge_vec_exp10_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_exp10_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sin_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_sin_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_cos_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_cos_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_exp_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_exp_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_log_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sinpi_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_sinpi_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_cospi_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_cospi_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_exp10_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_exp10_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sin_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_sin_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_cos_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_cos_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_exp_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_exp_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_log_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sinpi_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_sinpi_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_cospi_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_cospi_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_abs_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_abs_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sqrt_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_sqrt_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log1p_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_log1p_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_exp2_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_exp2_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log2_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_log2_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_abs_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_abs_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sqrt_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_sqrt_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log1p_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_log1p_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_exp2_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_exp2_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log2_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_log2_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_atan2_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f64(vec_atan2_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_hypot_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f64(vec_hypot_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_atan2_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f32(vec_atan2_f32_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_hypot_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f32(vec_hypot_f32_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_pow_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f64(vec_pow_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_pow_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f32(vec_pow_f32_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_expm1_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_expm1_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log10_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_log10_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sinh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_sinh_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_cosh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_cosh_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_tanh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_tanh_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_asinh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_asinh_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_acosh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_acosh_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_atanh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f64(vec_atanh_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_expm1_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_expm1_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_log10_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_log10_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_sinh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_sinh_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_cosh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_cosh_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_tanh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_tanh_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_asinh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_asinh_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_acosh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_acosh_f32_dispatch, in, out, nitems);
}

static void dsl_jit_bridge_vec_atanh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_unary_vector_f32(vec_atanh_f32_dispatch, in, out, nitems);
}

double me_jit_exp10(double x) {
    return dsl_jit_bridge_exp10(x);
}

double me_jit_sinpi(double x) {
    return dsl_jit_bridge_sinpi(x);
}

double me_jit_cospi(double x) {
    return dsl_jit_bridge_cospi(x);
}

double me_jit_logaddexp(double a, double b) {
    return dsl_jit_bridge_logaddexp(a, b);
}

double me_jit_where(double c, double x, double y) {
    return dsl_jit_bridge_where(c, x, y);
}

void me_jit_vec_sin_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_sin_f64(in, out, nitems);
}

void me_jit_vec_cos_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_cos_f64(in, out, nitems);
}

void me_jit_vec_exp_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_exp_f64(in, out, nitems);
}

void me_jit_vec_log_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_log_f64(in, out, nitems);
}

void me_jit_vec_exp10_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_exp10_f64(in, out, nitems);
}

void me_jit_vec_sinpi_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_sinpi_f64(in, out, nitems);
}

void me_jit_vec_cospi_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_cospi_f64(in, out, nitems);
}

void me_jit_vec_atan2_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_atan2_f64(a, b, out, nitems);
}

void me_jit_vec_hypot_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_hypot_f64(a, b, out, nitems);
}

void me_jit_vec_pow_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_pow_f64(a, b, out, nitems);
}

void me_jit_vec_sin_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_sin_f32(in, out, nitems);
}

void me_jit_vec_cos_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_cos_f32(in, out, nitems);
}

void me_jit_vec_exp_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_exp_f32(in, out, nitems);
}

void me_jit_vec_log_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_log_f32(in, out, nitems);
}

void me_jit_vec_exp10_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_exp10_f32(in, out, nitems);
}

void me_jit_vec_sinpi_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_sinpi_f32(in, out, nitems);
}

void me_jit_vec_cospi_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_cospi_f32(in, out, nitems);
}

void me_jit_vec_atan2_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_atan2_f32(a, b, out, nitems);
}

void me_jit_vec_hypot_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_hypot_f32(a, b, out, nitems);
}

void me_jit_vec_pow_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_pow_f32(a, b, out, nitems);
}

void me_jit_vec_expm1_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_expm1_f64(in, out, nitems);
}

void me_jit_vec_log10_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_log10_f64(in, out, nitems);
}

void me_jit_vec_sinh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_sinh_f64(in, out, nitems);
}

void me_jit_vec_cosh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_cosh_f64(in, out, nitems);
}

void me_jit_vec_tanh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_tanh_f64(in, out, nitems);
}

void me_jit_vec_asinh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_asinh_f64(in, out, nitems);
}

void me_jit_vec_acosh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_acosh_f64(in, out, nitems);
}

void me_jit_vec_atanh_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_atanh_f64(in, out, nitems);
}

void me_jit_vec_expm1_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_expm1_f32(in, out, nitems);
}

void me_jit_vec_log10_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_log10_f32(in, out, nitems);
}

void me_jit_vec_sinh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_sinh_f32(in, out, nitems);
}

void me_jit_vec_cosh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_cosh_f32(in, out, nitems);
}

void me_jit_vec_tanh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_tanh_f32(in, out, nitems);
}

void me_jit_vec_asinh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_asinh_f32(in, out, nitems);
}

void me_jit_vec_acosh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_acosh_f32(in, out, nitems);
}

void me_jit_vec_atanh_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_atanh_f32(in, out, nitems);
}

void me_jit_vec_abs_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_abs_f64(in, out, nitems);
}

void me_jit_vec_sqrt_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_sqrt_f64(in, out, nitems);
}

void me_jit_vec_log1p_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_log1p_f64(in, out, nitems);
}

void me_jit_vec_exp2_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_exp2_f64(in, out, nitems);
}

void me_jit_vec_log2_f64(const double *in, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_log2_f64(in, out, nitems);
}

void me_jit_vec_abs_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_abs_f32(in, out, nitems);
}

void me_jit_vec_sqrt_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_sqrt_f32(in, out, nitems);
}

void me_jit_vec_log1p_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_log1p_f32(in, out, nitems);
}

void me_jit_vec_exp2_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_exp2_f32(in, out, nitems);
}

void me_jit_vec_log2_f32(const float *in, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_log2_f32(in, out, nitems);
}

static bool dsl_jit_libtcc_register_math_bridge(me_tcc_state *state) {
    if (!state) {
        return false;
    }
    if (!g_dsl_tcc_api.tcc_add_symbol_fn) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc backend missing required symbol tcc_add_symbol for math bridge");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_exp10", (const void *)&dsl_jit_bridge_exp10) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_exp10");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_sinpi", (const void *)&dsl_jit_bridge_sinpi) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_sinpi");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_cospi", (const void *)&dsl_jit_bridge_cospi) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_cospi");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_logaddexp",
                                        (const void *)&dsl_jit_bridge_logaddexp) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_logaddexp");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_where", (const void *)&dsl_jit_bridge_where) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_where");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sin_f64",
                                        (const void *)&dsl_jit_bridge_vec_sin_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sin_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_cos_f64",
                                        (const void *)&dsl_jit_bridge_vec_cos_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_cos_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_exp_f64",
                                        (const void *)&dsl_jit_bridge_vec_exp_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_exp_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log_f64",
                                        (const void *)&dsl_jit_bridge_vec_log_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_exp10_f64",
                                        (const void *)&dsl_jit_bridge_vec_exp10_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_exp10_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sinpi_f64",
                                        (const void *)&dsl_jit_bridge_vec_sinpi_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sinpi_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_cospi_f64",
                                        (const void *)&dsl_jit_bridge_vec_cospi_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_cospi_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_atan2_f64",
                                        (const void *)&dsl_jit_bridge_vec_atan2_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_atan2_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_hypot_f64",
                                        (const void *)&dsl_jit_bridge_vec_hypot_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_hypot_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_pow_f64",
                                        (const void *)&dsl_jit_bridge_vec_pow_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_pow_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_expm1_f64",
                                        (const void *)&dsl_jit_bridge_vec_expm1_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_expm1_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log10_f64",
                                        (const void *)&dsl_jit_bridge_vec_log10_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log10_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sinh_f64",
                                        (const void *)&dsl_jit_bridge_vec_sinh_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sinh_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_cosh_f64",
                                        (const void *)&dsl_jit_bridge_vec_cosh_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_cosh_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_tanh_f64",
                                        (const void *)&dsl_jit_bridge_vec_tanh_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_tanh_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_asinh_f64",
                                        (const void *)&dsl_jit_bridge_vec_asinh_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_asinh_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_acosh_f64",
                                        (const void *)&dsl_jit_bridge_vec_acosh_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_acosh_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_atanh_f64",
                                        (const void *)&dsl_jit_bridge_vec_atanh_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_atanh_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sin_f32",
                                        (const void *)&dsl_jit_bridge_vec_sin_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sin_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_cos_f32",
                                        (const void *)&dsl_jit_bridge_vec_cos_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_cos_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_exp_f32",
                                        (const void *)&dsl_jit_bridge_vec_exp_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_exp_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log_f32",
                                        (const void *)&dsl_jit_bridge_vec_log_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_exp10_f32",
                                        (const void *)&dsl_jit_bridge_vec_exp10_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_exp10_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sinpi_f32",
                                        (const void *)&dsl_jit_bridge_vec_sinpi_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sinpi_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_cospi_f32",
                                        (const void *)&dsl_jit_bridge_vec_cospi_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_cospi_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_atan2_f32",
                                        (const void *)&dsl_jit_bridge_vec_atan2_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_atan2_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_hypot_f32",
                                        (const void *)&dsl_jit_bridge_vec_hypot_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_hypot_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_pow_f32",
                                        (const void *)&dsl_jit_bridge_vec_pow_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_pow_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_expm1_f32",
                                        (const void *)&dsl_jit_bridge_vec_expm1_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_expm1_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log10_f32",
                                        (const void *)&dsl_jit_bridge_vec_log10_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log10_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sinh_f32",
                                        (const void *)&dsl_jit_bridge_vec_sinh_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sinh_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_cosh_f32",
                                        (const void *)&dsl_jit_bridge_vec_cosh_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_cosh_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_tanh_f32",
                                        (const void *)&dsl_jit_bridge_vec_tanh_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_tanh_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_asinh_f32",
                                        (const void *)&dsl_jit_bridge_vec_asinh_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_asinh_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_acosh_f32",
                                        (const void *)&dsl_jit_bridge_vec_acosh_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_acosh_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_atanh_f32",
                                        (const void *)&dsl_jit_bridge_vec_atanh_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_atanh_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_abs_f64",
                                        (const void *)&dsl_jit_bridge_vec_abs_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_abs_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sqrt_f64",
                                        (const void *)&dsl_jit_bridge_vec_sqrt_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sqrt_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log1p_f64",
                                        (const void *)&dsl_jit_bridge_vec_log1p_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log1p_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_exp2_f64",
                                        (const void *)&dsl_jit_bridge_vec_exp2_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_exp2_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log2_f64",
                                        (const void *)&dsl_jit_bridge_vec_log2_f64) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log2_f64");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_abs_f32",
                                        (const void *)&dsl_jit_bridge_vec_abs_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_abs_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_sqrt_f32",
                                        (const void *)&dsl_jit_bridge_vec_sqrt_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_sqrt_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log1p_f32",
                                        (const void *)&dsl_jit_bridge_vec_log1p_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log1p_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_exp2_f32",
                                        (const void *)&dsl_jit_bridge_vec_exp2_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_exp2_f32");
        return false;
    }
    if (g_dsl_tcc_api.tcc_add_symbol_fn(state, "me_jit_vec_log2_f32",
                                        (const void *)&dsl_jit_bridge_vec_log2_f32) < 0) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_add_symbol failed for me_jit_vec_log2_f32");
        return false;
    }
    return true;
}

static bool dsl_jit_libtcc_load_api(void) {
    if (g_dsl_tcc_api.attempted) {
        return g_dsl_tcc_api.available;
    }
    g_dsl_tcc_api.attempted = true;

    const char *env_path = getenv("ME_DSL_JIT_LIBTCC_PATH");
    const char *default_path = ME_DSL_JIT_LIBTCC_DEFAULT_PATH;
    const char *candidates[12];
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
#if defined(_WIN32) || defined(_WIN64)
    candidates[ncandidates++] = "tcc.dll";
    candidates[ncandidates++] = "libtcc.dll";
#elif defined(__APPLE__)
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
        handle = dsl_jit_dynlib_open(candidates[i]);
        if (handle) {
            break;
        }
    }
    if (!handle) {
        const char *err = dsl_jit_dynlib_last_error();
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error),
                 "failed to load libtcc shared library%s%s",
                 (err && err[0]) ? ": " : "", (err && err[0]) ? err : "");
        return false;
    }

#define ME_LOAD_TCC_SYM(field, sym_name, fn_type) \
    do { \
        g_dsl_tcc_api.field = (fn_type)dsl_jit_dynlib_symbol(handle, sym_name); \
        if (!g_dsl_tcc_api.field) { \
            dsl_jit_dynlib_close(handle); \
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

    g_dsl_tcc_api.tcc_set_options_fn = (me_tcc_set_options_fn)dsl_jit_dynlib_symbol(handle, "tcc_set_options");
    g_dsl_tcc_api.tcc_add_library_path_fn = (me_tcc_add_library_path_fn)dsl_jit_dynlib_symbol(handle, "tcc_add_library_path");
    g_dsl_tcc_api.tcc_add_library_fn = (me_tcc_add_library_fn)dsl_jit_dynlib_symbol(handle, "tcc_add_library");
    g_dsl_tcc_api.tcc_add_symbol_fn = (me_tcc_add_symbol_fn)dsl_jit_dynlib_symbol(handle, "tcc_add_symbol");
    g_dsl_tcc_api.tcc_set_lib_path_fn = (me_tcc_set_lib_path_fn)dsl_jit_dynlib_symbol(handle, "tcc_set_lib_path");
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
                 "tcc backend supports only strict fp mode");
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
    dsl_jit_libtcc_add_multiarch_paths(state);
#if !defined(__APPLE__) && !defined(_WIN32) && !defined(_WIN64)
    if (g_dsl_tcc_api.tcc_add_library_fn) {
        (void)g_dsl_tcc_api.tcc_add_library_fn(state, "m");
    }
#endif
    if (program->jit_use_runtime_math_bridge &&
        !dsl_jit_libtcc_register_math_bridge(state)) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        return false;
    }
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
    return "tcc backend not built";
}

static void dsl_jit_libtcc_delete_state(void *state) {
    (void)state;
}

static bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program) {
    (void)program;
    return false;
}
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
static bool dsl_jit_cc_math_bridge_available(void) {
#if defined(_WIN32) || defined(_WIN64)
    return false;
#else
    if (dlsym(RTLD_DEFAULT, "me_jit_exp10") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_where") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_exp_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_pow_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_abs_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_sqrt_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_log1p_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_exp2_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_log2_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_expm1_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_log10_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_sinh_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_cosh_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_tanh_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_asinh_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_acosh_f64") == NULL) {
        return false;
    }
    if (dlsym(RTLD_DEFAULT, "me_jit_vec_atanh_f64") == NULL) {
        return false;
    }
    return true;
#endif
}

static void dsl_jit_cc_add_library_flag_if_exists(char *flags, size_t flags_size, const char *path) {
    if (!flags || flags_size == 0 || !path || path[0] == '\0') {
        return;
    }
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        return;
    }
    size_t len = strlen(flags);
    if (len >= flags_size - 1) {
        return;
    }
    int n = snprintf(flags + len, flags_size - len, " -L%s", path);
    if (n <= 0 || (size_t)n >= (flags_size - len)) {
        flags[flags_size - 1] = '\0';
    }
}

static void dsl_jit_cc_add_multiarch_library_flags(char *flags, size_t flags_size) {
    if (!flags || flags_size == 0) {
        return;
    }
    flags[0] = '\0';
#if defined(__linux__)
    /* Match libtcc fallback dirs so cc JIT can resolve -lm on multiarch layouts. */
    const char *paths[] = {
#if defined(__x86_64__) || defined(__amd64__)
        "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
#elif defined(__aarch64__)
        "/usr/lib/aarch64-linux-gnu", "/lib/aarch64-linux-gnu",
#elif defined(__arm__)
        "/usr/lib/arm-linux-gnueabihf", "/lib/arm-linux-gnueabihf",
        "/usr/lib/arm-linux-gnueabi", "/lib/arm-linux-gnueabi",
#elif defined(__riscv) && (__riscv_xlen == 64)
        "/usr/lib/riscv64-linux-gnu", "/lib/riscv64-linux-gnu",
#elif defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
        "/usr/lib/powerpc64le-linux-gnu", "/lib/powerpc64le-linux-gnu",
#elif defined(__s390x__)
        "/usr/lib/s390x-linux-gnu", "/lib/s390x-linux-gnu",
#elif defined(__i386__)
        "/usr/lib/i386-linux-gnu", "/lib/i386-linux-gnu",
#endif
        "/usr/lib64", "/lib64",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        dsl_jit_cc_add_library_flag_if_exists(flags, flags_size, paths[i]);
    }
#else
    (void)flags_size;
#endif
}

static bool dsl_jit_compile_shared(const me_dsl_compiled_program *program,
                                   const char *src_path, const char *so_path) {
    if (!program || !src_path || !so_path) {
        return false;
    }
    const char *cc = getenv("CC");
    const char *cflags = getenv("CFLAGS");
    const char *fp_cflags = dsl_jit_fp_mode_cflags(program->fp_mode);
    if (!cc || cc[0] == '\0') {
        cc = "cc";
    }
    if (!cflags) {
        cflags = "";
    }
    if (!fp_cflags) {
        fp_cflags = "";
    }
    const char *debug_cc = getenv("ME_DSL_JIT_DEBUG_CC");
    bool show_cc_output = (debug_cc && debug_cc[0] != '\0' && strcmp(debug_cc, "0") != 0);
    const char *bridge_ldflags = "";
    char multiarch_ldflags[512];
    const char *math_ldflags = "";
    dsl_jit_cc_add_multiarch_library_flags(multiarch_ldflags, sizeof(multiarch_ldflags));
#if defined(__APPLE__)
    if (program->jit_use_runtime_math_bridge) {
        bridge_ldflags = " -Wl,-undefined,dynamic_lookup";
    }
#elif !defined(_WIN32) && !defined(_WIN64)
    math_ldflags = " -lm";
#endif
    char cmd[2048];
#if defined(__APPLE__)
    int n = snprintf(cmd, sizeof(cmd),
                     "%s -std=c99 -O3 -fPIC %s %s -dynamiclib -o \"%s\" \"%s\"%s%s%s%s",
                     cc, fp_cflags, cflags, so_path, src_path, bridge_ldflags,
                     multiarch_ldflags, math_ldflags,
                     show_cc_output ? "" : " >/dev/null 2>&1");
#else
    int n = snprintf(cmd, sizeof(cmd),
                     "%s -std=c99 -O3 -fPIC %s %s -shared -o \"%s\" \"%s\"%s%s%s%s",
                     cc, fp_cflags, cflags, so_path, src_path, bridge_ldflags,
                     multiarch_ldflags, math_ldflags,
                     show_cc_output ? "" : " >/dev/null 2>&1");
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
        dsl_tracef("jit runtime skip: fp=%s reason=scalar output",
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (program->uses_i_mask || program->uses_n_mask || program->uses_ndim) {
        dsl_tracef("jit runtime skip: fp=%s reason=reserved index vars used",
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (program->jit_nparams != program->jit_ir->nparams) {
        dsl_tracef("jit runtime skip: fp=%s reason=parameter metadata mismatch",
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (!dsl_jit_runtime_enabled()) {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime disabled by environment");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }

    uint64_t key = dsl_jit_runtime_cache_key(program);
    if (dsl_jit_pos_cache_enabled() && dsl_jit_pos_cache_bind_program(program, key)) {
        dsl_tracef("jit runtime hit: fp=%s source=process-cache key=%016llx",
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
        dsl_jit_neg_cache_clear(key);
        return;
    }
    if (dsl_jit_neg_cache_should_skip(key)) {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime skipped after recent failure");
        dsl_tracef("jit runtime skip: fp=%s reason=%s key=%016llx",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error,
                   (unsigned long long)key);
        return;
    }

    char cache_dir[1024];
    if (!dsl_jit_get_cache_dir(cache_dir, sizeof(cache_dir))) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_CACHE_DIR);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime cache directory unavailable");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
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
        dsl_tracef("jit runtime skip: reason=%s", program->jit_c_error);
        return;
    }

    me_dsl_jit_cache_meta expected_meta;
    dsl_jit_fill_cache_meta(&expected_meta, program, key);

    bool so_exists = (access(so_path, F_OK) == 0);
    bool meta_matches = so_exists && dsl_jit_meta_file_matches(meta_path, &expected_meta);
    if (so_exists && !meta_matches) {
        /* Evict stale positive-cache entry so the old dlopen handle is
           closed before we overwrite the .so file on disk. */
        dsl_jit_pos_cache_evict(key);
    }
    if (meta_matches) {
        if (dsl_jit_load_kernel(program, so_path)) {
            if (dsl_jit_pos_cache_enabled()) {
                (void)dsl_jit_pos_cache_store_program(program, key);
            }
            dsl_tracef("jit runtime hit: fp=%s source=disk-cache key=%016llx",
                       dsl_fp_mode_name(program->fp_mode),
                       (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
        dsl_tracef("jit runtime cache reload failed: fp=%s key=%016llx",
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
    }

    if (program->compiler == ME_DSL_COMPILER_LIBTCC) {
        if (dsl_jit_compile_libtcc_in_memory(program)) {
            dsl_tracef("jit runtime built: fp=%s compiler=%s key=%016llx",
                       dsl_fp_mode_name(program->fp_mode),
                       dsl_compiler_name(program->compiler),
                       (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime tcc compilation failed");
        dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s detail=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler),
                   program->jit_c_error,
                   dsl_jit_libtcc_error_message());
        return;
    }

    if (!dsl_jit_write_text_file(src_path, program->jit_c_source)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_WRITE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write source");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    const char *jit_stub_path = getenv("ME_DSL_JIT_TEST_STUB_SO");
    if (jit_stub_path && jit_stub_path[0] != '\0') {
        const char *cflags = getenv("CFLAGS");
        if (cflags && strstr(cflags, ME_DSL_JIT_TEST_NEG_CACHE_FLAG)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime compilation failed");
            dsl_tracef("jit runtime skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        if (!dsl_jit_copy_file(jit_stub_path, so_path)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime stub copy failed");
            dsl_tracef("jit runtime skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        if (!dsl_jit_write_meta_file(meta_path, &expected_meta)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_METADATA);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime failed to write cache metadata");
            dsl_tracef("jit runtime skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        if (!dsl_jit_load_kernel(program, so_path)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime shared object load failed");
            dsl_tracef("jit runtime skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        if (dsl_jit_pos_cache_enabled()) {
            (void)dsl_jit_pos_cache_store_program(program, key);
        }
        dsl_tracef("jit runtime stubbed: fp=%s key=%016llx",
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
        dsl_jit_neg_cache_clear(key);
        return;
    }
    if (!dsl_jit_c_compiler_available()) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime c compiler unavailable");
        dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler),
                   program->jit_c_error);
        return;
    }
    if (!dsl_jit_compile_shared(program, src_path, so_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime compilation failed");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (!dsl_jit_write_meta_file(meta_path, &expected_meta)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_METADATA);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write cache metadata");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (!dsl_jit_load_kernel(program, so_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime shared object load failed");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (dsl_jit_pos_cache_enabled()) {
        (void)dsl_jit_pos_cache_store_program(program, key);
    }
    dsl_tracef("jit runtime built: fp=%s key=%016llx",
               dsl_fp_mode_name(program->fp_mode),
               (unsigned long long)key);
    dsl_jit_neg_cache_clear(key);
}
#else
static bool dsl_jit_cc_math_bridge_available(void) {
    return false;
}

static bool dsl_jit_runtime_enabled(void) {
    const char *env = getenv("ME_DSL_JIT");
    if (!env || env[0] == '\0') {
        return true;
    }
    return strcmp(env, "0") != 0;
}

#if ME_USE_WASM32_JIT
#include "libtcc.h"

#if !ME_WASM32_SIDE_MODULE
#include <emscripten.h>

/* JS helper: patch a TCC-emitted wasm module so it imports the host's linear
   memory instead of defining its own, then instantiate and return a
   function-table index for the exported kernel.
   The kernel uses int (i32) for nitems, so the wasm signature is
   (i32, i32, i32) -> i32. */
EM_JS(int, me_wasm_jit_instantiate,
      (const unsigned char *wasm_bytes, int wasm_len, int bridge_lookup_fn_idx), {
    var src = HEAPU8.subarray(wasm_bytes, wasm_bytes + wasm_len);
    var enc = new TextEncoder();
    var dec = new TextDecoder();
    /* --- LEB128 helpers ------------------------------------------------- */
    function readULEB(buf, pos) {
        var r = 0, s = 0, b;
        do { b = buf[pos++]; r |= (b & 0x7f) << s; s += 7; } while (b & 0x80);
        return [r, pos];
    }
    function encULEB(v) {
        var a = [];
        do { var b = v & 0x7f; v >>>= 7; if (v) b |= 0x80; a.push(b); } while (v);
        return a;
    }
    function encStr(s) {
        var b = enc.encode(s);
        return encULEB(b.length).concat(Array.from(b));
    }
    function readName(buf, pos) {
        var t = readULEB(buf, pos);
        var n = t[0];
        pos = t[1];
        var s = dec.decode(buf.subarray(pos, pos + n));
        return [s, pos + n];
    }
    function skipLimits(buf, pos) {
        var t = readULEB(buf, pos);
        var flags = t[0];
        pos = t[1];
        t = readULEB(buf, pos);
        pos = t[1];
        if (flags & 0x01) {
            t = readULEB(buf, pos);
            pos = t[1];
        }
        return pos;
    }
    function encMemoryImport() {
        var imp = [];
        imp = imp.concat(encStr("env"), encStr("memory"));
        imp.push(0x02, 0x00); /* memory, limits-flag: no-max */
        imp = imp.concat(encULEB(256));
        return imp;
    }
    function buildImportSecWithMemory() {
        var body = encULEB(1);
        body = body.concat(encMemoryImport());
        var sec = [0x02];
        sec = sec.concat(encULEB(body.length));
        return sec.concat(body);
    }
    function patchImportSec(secData) {
        var pos = 0;
        var t = readULEB(secData, pos);
        var nimports = t[0];
        pos = t[1];
        var entries = [];
        var hasEnvMemory = false;
        for (var i = 0; i < nimports; i++) {
            var start = pos;
            var moduleName = "";
            var fieldName = "";
            t = readName(secData, pos);
            moduleName = t[0];
            pos = t[1];
            t = readName(secData, pos);
            fieldName = t[0];
            pos = t[1];
            var kind = secData[pos++];
            if (kind === 0x00) {
                t = readULEB(secData, pos);
                pos = t[1];
            }
            else if (kind === 0x01) {
                pos++; /* elem type */
                pos = skipLimits(secData, pos);
            }
            else if (kind === 0x02) {
                pos = skipLimits(secData, pos);
                if (moduleName === "env" && fieldName === "memory") {
                    hasEnvMemory = true;
                }
            }
            else if (kind === 0x03) {
                pos += 2; /* valtype + mutability */
            }
            else {
                throw new Error("unsupported wasm import kind " + kind);
            }
            entries.push(Array.from(secData.subarray(start, pos)));
        }
        if (!hasEnvMemory) {
            entries.push(encMemoryImport());
        }
        var body = encULEB(entries.length);
        for (var ei = 0; ei < entries.length; ei++) {
            body = body.concat(entries[ei]);
        }
        var sec = [0x02];
        sec = sec.concat(encULEB(body.length));
        return sec.concat(body);
    }
    function buildEnvImports() {
        var bridgeLookup = null;
        var bridgeCache = Object.create(null);
        if (bridge_lookup_fn_idx) {
            bridgeLookup = wasmTable.get(bridge_lookup_fn_idx);
        }
        function lookupBridge(name) {
            if (!bridgeLookup) {
                return null;
            }
            if (Object.prototype.hasOwnProperty.call(bridgeCache, name)) {
                return bridgeCache[name];
            }
            var sp = stackSave();
            try {
                var nbytes = lengthBytesUTF8(name) + 1;
                var namePtr = stackAlloc(nbytes);
                stringToUTF8(name, namePtr, nbytes);
                var fnIdx = bridgeLookup(namePtr) | 0;
                bridgeCache[name] = fnIdx ? wasmTable.get(fnIdx) : null;
            } finally {
                stackRestore(sp);
            }
            return bridgeCache[name];
        }
        function bindBridge(name, fallback) {
            var fn = lookupBridge(name);
            return fn ? fn : fallback;
        }
        function fdim(x, y) { return x > y ? (x - y) : 0.0; }
        function copysign(x, y) {
            if (y === 0) {
                return (1 / y === -Infinity) ? -Math.abs(x) : Math.abs(x);
            }
            return y < 0 ? -Math.abs(x) : Math.abs(x);
        }
        function ldexp(x, e) { return x * Math.pow(2.0, e); }
        function rint(x) {
            if (!isFinite(x)) {
                return x;
            }
            var n = Math.round(x);
            if (Math.abs(x - n) === 0.5) {
                n = 2 * Math.round(x / 2);
            }
            return n;
        }
        function remainder(x, y) {
            if (!isFinite(x) || !isFinite(y) || y === 0.0) {
                return NaN;
            }
            return x - y * Math.round(x / y);
        }
        function erfApprox(x) {
            var sign = x < 0 ? -1.0 : 1.0;
            x = Math.abs(x);
            var a1 = 0.254829592;
            var a2 = -0.284496736;
            var a3 = 1.421413741;
            var a4 = -1.453152027;
            var a5 = 1.061405429;
            var p = 0.3275911;
            var t = 1.0 / (1.0 + p * x);
            var y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
            return sign * y;
        }
        function erfcApprox(x) { return 1.0 - erfApprox(x); }
        function tgammaApprox(z) {
            var p = [
                676.5203681218851, -1259.1392167224028, 771.32342877765313,
                -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                9.9843695780195716e-6, 1.5056327351493116e-7
            ];
            if (z < 0.5) {
                return Math.PI / (Math.sin(Math.PI * z) * tgammaApprox(1.0 - z));
            }
            z -= 1.0;
            var x = 0.99999999999980993;
            for (var i = 0; i < p.length; i++) {
                x += p[i] / (z + i + 1.0);
            }
            var t = z + p.length - 0.5;
            return Math.sqrt(2.0 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
        }
        function lgammaApprox(x) {
            var g = tgammaApprox(x);
            return Math.log(Math.abs(g));
        }
        function nextafterApprox(x, y) {
            if (isNaN(x) || isNaN(y)) {
                return NaN;
            }
            if (x === y) {
                return y;
            }
            if (x === 0.0) {
                return y > 0.0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
            }
            var buf = new ArrayBuffer(8);
            var dv = new DataView(buf);
            dv.setFloat64(0, x, true);
            var bits = dv.getBigUint64(0, true);
            if ((y > x) === (x > 0.0)) {
                bits += 1n;
            }
            else {
                bits -= 1n;
            }
            dv.setBigUint64(0, bits, true);
            return dv.getFloat64(0, true);
        }
        function meJitExp10(x) { return Math.pow(10.0, x); }
        function meJitSinpi(x) { return Math.sin(Math.PI * x); }
        function meJitCospi(x) { return Math.cos(Math.PI * x); }
        var mathExp2 = Math.exp2 ? Math.exp2 : function(x) { return Math.pow(2.0, x); };
        function meJitLogaddexp(a, b) {
            var hi = a > b ? a : b;
            var lo = a > b ? b : a;
            return hi + Math.log1p(Math.exp(lo - hi));
        }
        function meJitWhere(c, x, y) { return c !== 0.0 ? x : y; }
        function vecUnaryF64(inPtr, outPtr, n, fn) {
            var ii = inPtr >> 3;
            var oo = outPtr >> 3;
            for (var i = 0; i < n; i++) {
                HEAPF64[oo + i] = fn(HEAPF64[ii + i]);
            }
        }
        function vecBinaryF64(aPtr, bPtr, outPtr, n, fn) {
            var aa = aPtr >> 3;
            var bb = bPtr >> 3;
            var oo = outPtr >> 3;
            for (var i = 0; i < n; i++) {
                HEAPF64[oo + i] = fn(HEAPF64[aa + i], HEAPF64[bb + i]);
            }
        }
        function vecUnaryF32(inPtr, outPtr, n, fn) {
            var ii = inPtr >> 2;
            var oo = outPtr >> 2;
            for (var i = 0; i < n; i++) {
                HEAPF32[oo + i] = fn(HEAPF32[ii + i]);
            }
        }
        function vecBinaryF32(aPtr, bPtr, outPtr, n, fn) {
            var aa = aPtr >> 2;
            var bb = bPtr >> 2;
            var oo = outPtr >> 2;
            for (var i = 0; i < n; i++) {
                HEAPF32[oo + i] = fn(HEAPF32[aa + i], HEAPF32[bb + i]);
            }
        }
        var env = {
            memory: wasmMemory,
            acos: Math.acos, acosh: Math.acosh, asin: Math.asin, asinh: Math.asinh,
            atan: Math.atan, atan2: Math.atan2, atanh: Math.atanh, cbrt: Math.cbrt,
            ceil: Math.ceil, copysign: copysign, cos: Math.cos, cosh: Math.cosh,
            erf: erfApprox, erfc: erfcApprox, exp: Math.exp, exp2: mathExp2,
            expm1: Math.expm1, fabs: Math.abs, fdim: fdim, floor: Math.floor,
            fma: function(a, b, c) { return a * b + c; }, fmax: Math.max, fmin: Math.min,
            fmod: function(a, b) { return a % b; }, hypot: Math.hypot, ldexp: ldexp,
            lgamma: lgammaApprox, log: Math.log, log10: Math.log10, log1p: Math.log1p,
            log2: Math.log2, nextafter: nextafterApprox, pow: Math.pow, remainder: remainder,
            rint: rint, round: Math.round, sin: Math.sin, sinh: Math.sinh, sqrt: Math.sqrt,
            tan: Math.tan, tanh: Math.tanh, tgamma: tgammaApprox, trunc: Math.trunc,
            me_jit_exp10: meJitExp10, me_jit_sinpi: meJitSinpi, me_jit_cospi: meJitCospi,
            me_jit_logaddexp: meJitLogaddexp, me_jit_where: meJitWhere
        };
        env.me_wasm32_cast_int = function(x) {
            return x < 0 ? Math.ceil(x) : Math.floor(x);
        };
        env.me_wasm32_cast_float = function(x) {
            return x;
        };
        env.me_wasm32_cast_bool = function(x) {
            return x !== 0 ? 1 : 0;
        };
        /* Prefer host wasm bridge symbols; keep JS fallbacks for robustness. */
        env.me_jit_exp10 = bindBridge("me_jit_exp10", env.me_jit_exp10);
        env.me_jit_sinpi = bindBridge("me_jit_sinpi", env.me_jit_sinpi);
        env.me_jit_cospi = bindBridge("me_jit_cospi", env.me_jit_cospi);
        env.me_jit_logaddexp = bindBridge("me_jit_logaddexp", env.me_jit_logaddexp);
        env.me_jit_where = bindBridge("me_jit_where", env.me_jit_where);
        env.me_jit_vec_sin_f64 = bindBridge("me_jit_vec_sin_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.sin); });
        env.me_jit_vec_cos_f64 = bindBridge("me_jit_vec_cos_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.cos); });
        env.me_jit_vec_exp_f64 = bindBridge("me_jit_vec_exp_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.exp); });
        env.me_jit_vec_log_f64 = bindBridge("me_jit_vec_log_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log); });
        env.me_jit_vec_exp10_f64 = bindBridge("me_jit_vec_exp10_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, meJitExp10); });
        env.me_jit_vec_sinpi_f64 = bindBridge("me_jit_vec_sinpi_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, meJitSinpi); });
        env.me_jit_vec_cospi_f64 = bindBridge("me_jit_vec_cospi_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, meJitCospi); });
        env.me_jit_vec_atan2_f64 = bindBridge("me_jit_vec_atan2_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.atan2); });
        env.me_jit_vec_hypot_f64 = bindBridge("me_jit_vec_hypot_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.hypot); });
        env.me_jit_vec_pow_f64 = bindBridge("me_jit_vec_pow_f64", function(aPtr, bPtr, outPtr, n) { vecBinaryF64(aPtr, bPtr, outPtr, n, Math.pow); });
        env.me_jit_vec_expm1_f64 = bindBridge("me_jit_vec_expm1_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.expm1); });
        env.me_jit_vec_log10_f64 = bindBridge("me_jit_vec_log10_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log10); });
        env.me_jit_vec_sinh_f64 = bindBridge("me_jit_vec_sinh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.sinh); });
        env.me_jit_vec_cosh_f64 = bindBridge("me_jit_vec_cosh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.cosh); });
        env.me_jit_vec_tanh_f64 = bindBridge("me_jit_vec_tanh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.tanh); });
        env.me_jit_vec_asinh_f64 = bindBridge("me_jit_vec_asinh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.asinh); });
        env.me_jit_vec_acosh_f64 = bindBridge("me_jit_vec_acosh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.acosh); });
        env.me_jit_vec_atanh_f64 = bindBridge("me_jit_vec_atanh_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.atanh); });
        env.me_jit_vec_abs_f64 = bindBridge("me_jit_vec_abs_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.abs); });
        env.me_jit_vec_sqrt_f64 = bindBridge("me_jit_vec_sqrt_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.sqrt); });
        env.me_jit_vec_log1p_f64 = bindBridge("me_jit_vec_log1p_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log1p); });
        env.me_jit_vec_exp2_f64 = bindBridge("me_jit_vec_exp2_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, mathExp2); });
        env.me_jit_vec_log2_f64 = bindBridge("me_jit_vec_log2_f64", function(inPtr, outPtr, n) { vecUnaryF64(inPtr, outPtr, n, Math.log2); });
        env.me_jit_vec_sin_f32 = bindBridge("me_jit_vec_sin_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.sin); });
        env.me_jit_vec_cos_f32 = bindBridge("me_jit_vec_cos_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.cos); });
        env.me_jit_vec_exp_f32 = bindBridge("me_jit_vec_exp_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.exp); });
        env.me_jit_vec_log_f32 = bindBridge("me_jit_vec_log_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log); });
        env.me_jit_vec_exp10_f32 = bindBridge("me_jit_vec_exp10_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, meJitExp10); });
        env.me_jit_vec_sinpi_f32 = bindBridge("me_jit_vec_sinpi_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, meJitSinpi); });
        env.me_jit_vec_cospi_f32 = bindBridge("me_jit_vec_cospi_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, meJitCospi); });
        env.me_jit_vec_atan2_f32 = bindBridge("me_jit_vec_atan2_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.atan2); });
        env.me_jit_vec_hypot_f32 = bindBridge("me_jit_vec_hypot_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.hypot); });
        env.me_jit_vec_pow_f32 = bindBridge("me_jit_vec_pow_f32", function(aPtr, bPtr, outPtr, n) { vecBinaryF32(aPtr, bPtr, outPtr, n, Math.pow); });
        env.me_jit_vec_expm1_f32 = bindBridge("me_jit_vec_expm1_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.expm1); });
        env.me_jit_vec_log10_f32 = bindBridge("me_jit_vec_log10_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log10); });
        env.me_jit_vec_sinh_f32 = bindBridge("me_jit_vec_sinh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.sinh); });
        env.me_jit_vec_cosh_f32 = bindBridge("me_jit_vec_cosh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.cosh); });
        env.me_jit_vec_tanh_f32 = bindBridge("me_jit_vec_tanh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.tanh); });
        env.me_jit_vec_asinh_f32 = bindBridge("me_jit_vec_asinh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.asinh); });
        env.me_jit_vec_acosh_f32 = bindBridge("me_jit_vec_acosh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.acosh); });
        env.me_jit_vec_atanh_f32 = bindBridge("me_jit_vec_atanh_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.atanh); });
        env.me_jit_vec_abs_f32 = bindBridge("me_jit_vec_abs_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.abs); });
        env.me_jit_vec_sqrt_f32 = bindBridge("me_jit_vec_sqrt_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.sqrt); });
        env.me_jit_vec_log1p_f32 = bindBridge("me_jit_vec_log1p_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log1p); });
        env.me_jit_vec_exp2_f32 = bindBridge("me_jit_vec_exp2_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, mathExp2); });
        env.me_jit_vec_log2_f32 = bindBridge("me_jit_vec_log2_f32", function(inPtr, outPtr, n) { vecUnaryF32(inPtr, outPtr, n, Math.log2); });
        return env;
    }
    /* --- parse sections ------------------------------------------------- */
    var pos = 8, sections = [];
    while (pos < src.length) {
        var id = src[pos++];
        var tmp = readULEB(src, pos), len = tmp[0]; pos = tmp[1];
        sections.push({ id: id, data: src.subarray(pos, pos + len) });
        pos += len;
    }
    /* --- reassemble with patched memory -------------------------------- */
    var out = [0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00];
    var impDone = false;
    for (var i = 0; i < sections.length; i++) {
        var s = sections[i];
        if (s.id === 5) continue; /* drop memory section */
        if (s.id === 2) {
            out = out.concat(patchImportSec(s.data));
            impDone = true;
            continue;
        }
        if (!impDone && s.id > 2) {
            out = out.concat(buildImportSecWithMemory());
            impDone = true;
        }
        if (s.id === 7) { /* strip memory export from export section */
            var ep = 0, et = readULEB(s.data, ep), ne = et[0]; ep = et[1];
            var exps = [];
            for (var e = 0; e < ne; e++) {
                var nt = readULEB(s.data, ep), nl = nt[0]; ep = nt[1];
                var nm = dec.decode(s.data.subarray(ep, ep + nl)); ep += nl;
                var kd = s.data[ep++];
                var xt = readULEB(s.data, ep), xi = xt[0]; ep = xt[1];
                if (nm === "memory" && kd === 0x02) continue;
                exps.push({ n: nm, k: kd, i: xi });
            }
            var eb = encULEB(exps.length);
            for (var e = 0; e < exps.length; e++) {
                eb = eb.concat(encStr(exps[e].n));
                eb.push(exps[e].k);
                eb = eb.concat(encULEB(exps[e].i));
            }
            out.push(0x07);
            out = out.concat(encULEB(eb.length));
            out = out.concat(eb);
            continue;
        }
        out.push(s.id);
        out = out.concat(encULEB(s.data.length));
        out = out.concat(Array.from(s.data));
    }
    if (!impDone) {
        out = out.concat(buildImportSecWithMemory());
    }
    /* --- instantiate with shared memory -------------------------------- */
    var patched = new Uint8Array(out);
    try {
        var mod = new WebAssembly.Module(patched);
        var inst = new WebAssembly.Instance(mod, { env: buildEnvImports() });
    } catch (e) {
        err("[me-wasm-jit] " + e.message);
        return 0;
    }
    var fn = inst.exports["me_dsl_jit_kernel"];
    if (!fn) { err("[me-wasm-jit] missing export"); return 0; }
    return addFunction(fn, "iiii");
});

/* Free a function-table slot previously allocated by me_wasm_jit_instantiate. */
EM_JS(void, me_wasm_jit_free_fn, (int idx), {
    if (idx) removeFunction(idx);
});
#endif

static void dsl_wasm_tcc_error_handler(void *opaque, const char *msg) {
    (void)opaque;
    dsl_tracef("jit tcc error: %s", msg);
}

/* Replace "int64_t nitems" with "int nitems" in the generated C source to
   avoid TCC wasm32 int64_t comparison bugs.  Returns a malloc'd copy. */
static char *dsl_wasm32_patch_source(const char *src) {
    /* On wasm32 we patch the generated C source to work around TCC
       wasm32 backend limitations:
       1. Replace all int64_t declarations/casts with int (64-bit
          comparisons trigger wasm32 backend bugs in gen_opl)
       2. Split "if (!output || nitems < 0)" into two separate ifs
          (TCC wasm32 can't mix comparison types in ||) */
    typedef struct { const char *old; const char *rep; } Repl;
    Repl repls[] = {
        { "#define ME_DSL_CAST_INT(x) ((int64_t)(x))",
          "extern int me_wasm32_cast_int(double);\n"
          "#define ME_DSL_CAST_INT(x) (me_wasm32_cast_int((double)(x)))" },
        { "#define ME_DSL_CAST_FLOAT(x) ((double)(x))",
          "extern double me_wasm32_cast_float(double);\n"
          "#define ME_DSL_CAST_FLOAT(x) (me_wasm32_cast_float((double)(x)))" },
        { "#define ME_DSL_CAST_BOOL(x) ((x) != 0)",
          "extern int me_wasm32_cast_bool(double);\n"
          "#define ME_DSL_CAST_BOOL(x) (me_wasm32_cast_bool((double)(x)))" },
        { "int64_t ",       "int "           },
        { "(int64_t)",      "(int)"          },
        { "if (!output || nitems < 0) {\n"
          "        return -1;\n"
          "    }",
          "if (!output) {\n"
          "        return -1;\n"
          "    }\n"
          "    if (nitems < 0) {\n"
          "        return -1;\n"
          "    }" },
    };
    size_t nrepls = sizeof(repls) / sizeof(repls[0]);
    size_t src_len = strlen(src);
    size_t alloc = src_len + 2048;
    char *patched = (char *)malloc(alloc);
    if (!patched) return NULL;
    const char *p = src;
    char *d = patched;
    while (*p) {
        bool matched = false;
        for (size_t ri = 0; ri < nrepls; ri++) {
            size_t olen = strlen(repls[ri].old);
            if (strncmp(p, repls[ri].old, olen) == 0) {
                size_t rlen = strlen(repls[ri].rep);
                if ((size_t)(d - patched) + rlen + 1 > alloc) break;
                memcpy(d, repls[ri].rep, rlen);
                d += rlen;
                p += olen;
                matched = true;
                break;
            }
        }
        if (!matched) *d++ = *p++;
    }
    *d = '\0';
    return patched;
}

static bool dsl_wasm32_source_calls_symbol(const char *src, const char *name) {
    if (!src || !name || name[0] == '\0') {
        return false;
    }
    size_t name_len = strlen(name);
    const char *p = src;
    while (*p) {
        if (strncmp(p, "extern ", 7) == 0 ||
            strncmp(p, "static ", 7) == 0) {
            while (*p && *p != '\n') {
                p++;
            }
            if (*p) {
                p++;
            }
            continue;
        }
        if (strncmp(p, name, name_len) == 0 && p[name_len] == '(') {
            if (p == src || !((p[-1] >= 'a' && p[-1] <= 'z') ||
                              (p[-1] >= 'A' && p[-1] <= 'Z') ||
                              (p[-1] >= '0' && p[-1] <= '9') ||
                               p[-1] == '_')) {
                return true;
            }
        }
        p++;
    }
    return false;
}

typedef struct {
    const char *name;
    const void *addr;
} dsl_wasm32_symbol_binding;

#define ME_WASM32_BRIDGE_SYM(fn) { #fn, (const void *)&fn }
#define ME_WASM32_BRIDGE_SYM_VEC(fn) { #fn, (const void *)&fn }

static const dsl_wasm32_symbol_binding dsl_wasm32_symbol_bindings[] = {
    ME_WASM32_BRIDGE_SYM(acos), ME_WASM32_BRIDGE_SYM(acosh), ME_WASM32_BRIDGE_SYM(asin),
    ME_WASM32_BRIDGE_SYM(asinh), ME_WASM32_BRIDGE_SYM(atan), ME_WASM32_BRIDGE_SYM(atan2),
    ME_WASM32_BRIDGE_SYM(atanh), ME_WASM32_BRIDGE_SYM(cbrt), ME_WASM32_BRIDGE_SYM(ceil),
    ME_WASM32_BRIDGE_SYM(copysign), ME_WASM32_BRIDGE_SYM(cos), ME_WASM32_BRIDGE_SYM(cosh),
    ME_WASM32_BRIDGE_SYM(erf), ME_WASM32_BRIDGE_SYM(erfc), ME_WASM32_BRIDGE_SYM(exp),
    ME_WASM32_BRIDGE_SYM(exp2), ME_WASM32_BRIDGE_SYM(expm1), ME_WASM32_BRIDGE_SYM(fabs),
    ME_WASM32_BRIDGE_SYM(fdim), ME_WASM32_BRIDGE_SYM(floor), ME_WASM32_BRIDGE_SYM(fma),
    ME_WASM32_BRIDGE_SYM(fmax), ME_WASM32_BRIDGE_SYM(fmin), ME_WASM32_BRIDGE_SYM(fmod),
    ME_WASM32_BRIDGE_SYM(hypot), ME_WASM32_BRIDGE_SYM(ldexp), ME_WASM32_BRIDGE_SYM(lgamma),
    ME_WASM32_BRIDGE_SYM(log), ME_WASM32_BRIDGE_SYM(log10), ME_WASM32_BRIDGE_SYM(log1p),
    ME_WASM32_BRIDGE_SYM(log2), ME_WASM32_BRIDGE_SYM(nextafter), ME_WASM32_BRIDGE_SYM(pow),
    ME_WASM32_BRIDGE_SYM(remainder), ME_WASM32_BRIDGE_SYM(rint), ME_WASM32_BRIDGE_SYM(round),
    ME_WASM32_BRIDGE_SYM(sin), ME_WASM32_BRIDGE_SYM(sinh), ME_WASM32_BRIDGE_SYM(sqrt),
    ME_WASM32_BRIDGE_SYM(tan), ME_WASM32_BRIDGE_SYM(tanh), ME_WASM32_BRIDGE_SYM(tgamma),
    ME_WASM32_BRIDGE_SYM(trunc),
    ME_WASM32_BRIDGE_SYM(me_jit_exp10), ME_WASM32_BRIDGE_SYM(me_jit_sinpi),
    ME_WASM32_BRIDGE_SYM(me_jit_cospi), ME_WASM32_BRIDGE_SYM(me_jit_logaddexp),
    ME_WASM32_BRIDGE_SYM(me_jit_where),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sin_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_cos_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_exp_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_exp10_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sinpi_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_cospi_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_atan2_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_hypot_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_pow_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_expm1_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log10_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sinh_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_cosh_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_tanh_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_asinh_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_acosh_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_atanh_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_abs_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sqrt_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log1p_f64), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_exp2_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log2_f64),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sin_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_cos_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_exp_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_exp10_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sinpi_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_cospi_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_atan2_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_hypot_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_pow_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_expm1_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log10_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sinh_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_cosh_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_tanh_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_asinh_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_acosh_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_atanh_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_abs_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_sqrt_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log1p_f32), ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_exp2_f32),
    ME_WASM32_BRIDGE_SYM_VEC(me_jit_vec_log2_f32)
};

static int dsl_wasm32_lookup_bridge_symbol(const char *name) {
    if (!name || name[0] == '\0') {
        return 0;
    }
    for (size_t i = 0; i < sizeof(dsl_wasm32_symbol_bindings) / sizeof(dsl_wasm32_symbol_bindings[0]); i++) {
        if (strcmp(name, dsl_wasm32_symbol_bindings[i].name) != 0) {
            continue;
        }
        return (int)(uintptr_t)dsl_wasm32_symbol_bindings[i].addr;
    }
    return 0;
}

static bool dsl_wasm32_register_required_symbols(TCCState *state, const char *src) {
    if (!state || !src) {
        return false;
    }
    for (size_t i = 0; i < sizeof(dsl_wasm32_symbol_bindings) / sizeof(dsl_wasm32_symbol_bindings[0]); i++) {
        if (!dsl_wasm32_source_calls_symbol(src, dsl_wasm32_symbol_bindings[i].name)) {
            continue;
        }
        if (tcc_add_symbol(state, dsl_wasm32_symbol_bindings[i].name, dsl_wasm32_symbol_bindings[i].addr) < 0) {
            dsl_tracef("jit runtime skip: tcc_add_symbol failed for '%s'", dsl_wasm32_symbol_bindings[i].name);
        }
    }
    return true;
}

#undef ME_WASM32_BRIDGE_SYM
#undef ME_WASM32_BRIDGE_SYM_VEC

static bool dsl_jit_compile_wasm32(me_dsl_compiled_program *program) {
    if (!program || !program->jit_c_source) {
        return false;
    }
#if ME_WASM32_SIDE_MODULE
    if (!me_wasm_jit_helpers_available()) {
        dsl_tracef("jit runtime skip: side-module wasm32 helpers are not registered");
        return false;
    }
#endif
    /* Patch int64_t → int for nitems (wasm32 backend limitation). */
    char *patched_src = dsl_wasm32_patch_source(program->jit_c_source);
    if (!patched_src) return false;
    dsl_tracef("jit wasm32: source patched (%zu bytes)", strlen(patched_src));

    TCCState *state = tcc_new();
    if (!state) {
        free(patched_src);
        dsl_tracef("jit runtime skip: tcc_new failed");
        return false;
    }
    tcc_set_error_func(state, NULL, dsl_wasm_tcc_error_handler);
    tcc_set_options(state, "-nostdlib -nostdinc");

    if (tcc_set_output_type(state, TCC_OUTPUT_EXE) < 0) {
        tcc_delete(state);
        free(patched_src);
        dsl_tracef("jit runtime skip: tcc_set_output_type failed");
        return false;
    }

    /* Relocate data/stack to a safe region that doesn't collide with
       the host module's linear memory layout. */
    void *jit_scratch = malloc(256 * 1024);
    if (!jit_scratch) { tcc_delete(state); free(patched_src); return false; }
    unsigned int jit_base = ((unsigned int)(uintptr_t)jit_scratch + 0xFFFFu) & ~0xFFFFu;
    tcc_set_wasm_data_base(state, jit_base);

    if (!dsl_wasm32_register_required_symbols(state, patched_src)) {
        tcc_delete(state);
        free(patched_src);
        free(jit_scratch);
        return false;
    }

    if (tcc_compile_string(state, patched_src) < 0) {
        dsl_tracef("jit runtime skip: tcc_compile_string failed");
        tcc_delete(state);
        free(patched_src);
        free(jit_scratch);
        return false;
    }
    free(patched_src);

    const char *wasm_path = "/tmp/me_jit_kernel.wasm";
    if (tcc_output_file(state, wasm_path) < 0) {
        tcc_delete(state);
        free(jit_scratch);
        dsl_tracef("jit runtime skip: tcc_output_file failed");
        return false;
    }
    tcc_delete(state);

    /* Read the .wasm bytes from Emscripten's virtual filesystem. */
    FILE *fp = fopen(wasm_path, "rb");
    if (!fp) {
        free(jit_scratch);
        dsl_tracef("jit runtime skip: cannot read wasm file");
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long wasm_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (wasm_len <= 0 || wasm_len > 1024 * 1024) {
        fclose(fp);
        free(jit_scratch);
        dsl_tracef("jit runtime skip: wasm file size %ld unexpected", wasm_len);
        return false;
    }
    unsigned char *wasm_bytes = (unsigned char *)malloc((size_t)wasm_len);
    if (!wasm_bytes) {
        fclose(fp);
        free(jit_scratch);
        return false;
    }
    if ((long)fread(wasm_bytes, 1, (size_t)wasm_len, fp) != wasm_len) {
        free(wasm_bytes);
        fclose(fp);
        free(jit_scratch);
        return false;
    }
    fclose(fp);
    remove(wasm_path);

    /* Instantiate the wasm module and get a callable function pointer. */
    int fn_idx = me_wasm_jit_instantiate_dispatch(wasm_bytes, (int)wasm_len,
                                                  (int)(uintptr_t)&dsl_wasm32_lookup_bridge_symbol);
    free(wasm_bytes);
    if (fn_idx == 0) {
        free(jit_scratch);
        dsl_tracef("jit runtime skip: wasm instantiation failed");
        return false;
    }

    program->jit_kernel_fn = (me_dsl_jit_kernel_fn)(uintptr_t)fn_idx;
    /* Keep scratch memory alive — the JIT module's data/stack lives there. */
    program->jit_dl_handle = jit_scratch;
    program->jit_c_error_line = 0;
    program->jit_c_error_column = 0;
    program->jit_c_error[0] = '\0';
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    return true;
}
#endif /* ME_USE_WASM32_JIT */

static void dsl_try_prepare_jit_runtime(me_dsl_compiled_program *program) {
#if ME_USE_WASM32_JIT
    if (!program || !program->jit_c_source || program->jit_kernel_fn) {
        return;
    }
    if (!dsl_jit_runtime_enabled()) {
        return;
    }
    if (dsl_jit_compile_wasm32(program)) {
        dsl_tracef("jit runtime built: fp=%s compiler=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler));
        return;
    }
    if (program->jit_c_error[0] == '\0') {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime wasm32 compilation failed");
    }
    dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s",
               dsl_fp_mode_name(program->fp_mode),
               dsl_compiler_name(program->compiler),
               program->jit_c_error);
#elif ME_USE_LIBTCC_FALLBACK
    if (!program || !program->jit_c_source || program->jit_kernel_fn) {
        return;
    }
    if (!dsl_jit_runtime_enabled()) {
        return;
    }
    if (program->compiler != ME_DSL_COMPILER_LIBTCC) {
        return;
    }
    if (dsl_jit_compile_libtcc_in_memory(program)) {
        dsl_tracef("jit runtime built: fp=%s compiler=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler));
        return;
    }
    snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
             "jit runtime tcc compilation failed");
    dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s detail=%s",
               dsl_fp_mode_name(program->fp_mode),
               dsl_compiler_name(program->compiler),
               program->jit_c_error,
               dsl_jit_libtcc_error_message());
#else
    (void)program;
#endif
}
#endif

static bool dsl_compiled_block_guarantees_return(const me_dsl_compiled_block *block);

static bool dsl_compiled_stmt_guarantees_return(const me_dsl_compiled_stmt *stmt) {
    if (!stmt) {
        return false;
    }
    if (stmt->kind == ME_DSL_STMT_RETURN) {
        return true;
    }
    if (stmt->kind != ME_DSL_STMT_IF) {
        return false;
    }
    if (!stmt->as.if_stmt.has_else) {
        return false;
    }
    if (!dsl_compiled_block_guarantees_return(&stmt->as.if_stmt.then_block)) {
        return false;
    }
    for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
        if (!dsl_compiled_block_guarantees_return(&stmt->as.if_stmt.elif_branches[i].block)) {
            return false;
        }
    }
    return dsl_compiled_block_guarantees_return(&stmt->as.if_stmt.else_block);
}

static bool dsl_compiled_block_guarantees_return(const me_dsl_compiled_block *block) {
    if (!block) {
        return false;
    }
    for (int i = 0; i < block->nstmts; i++) {
        if (dsl_compiled_stmt_guarantees_return(block->stmts[i])) {
            return true;
        }
    }
    return false;
}

static void dsl_try_build_jit_ir(dsl_compile_ctx *ctx, const me_dsl_program *parsed,
                                 me_dsl_compiled_program *program,
                                 bool prepare_runtime) {
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
    }
#endif
    program->jit_dl_handle = NULL;
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    free(program->jit_c_source);
    program->jit_c_source = NULL;
    program->jit_use_runtime_math_bridge = false;
    program->jit_c_error_line = 0;
    program->jit_c_error_column = 0;
    program->jit_c_error[0] = '\0';

    if (!program->guaranteed_return) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "program may reach function end without return");
        dsl_tracef("jit ir skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_ir_error);
        return;
    }

    if (parsed->nparams < 0) {
        snprintf(program->jit_ir_error, sizeof(program->jit_ir_error), "%s",
                 "invalid dsl parameter metadata");
        dsl_tracef("jit ir skip: fp=%s reason=%s",
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
            dsl_tracef("jit ir skip: fp=%s reason=%s",
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
                dsl_tracef("jit ir skip: fp=%s reason=%s",
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
        dsl_tracef("jit ir reject: fp=%s at %d:%d reason=%s",
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
    memset(&cg_options, 0, sizeof(cg_options));
    cg_options.symbol_name = ME_DSL_JIT_SYMBOL_NAME;
    bool use_bridge = false;
    if (program->compiler == ME_DSL_COMPILER_LIBTCC) {
        use_bridge = true;
    }
    else if (program->compiler == ME_DSL_COMPILER_CC) {
        if (dsl_jit_cc_math_bridge_available()) {
            use_bridge = true;
        }
        else {
            dsl_tracef("jit codegen: runtime math bridge unavailable for cc backend");
        }
    }
    cg_options.use_runtime_math_bridge = use_bridge;
    char *generated_c = NULL;
    bool cg_ok = me_dsl_jit_codegen_c(jit_ir, ctx->return_dtype, &cg_options,
                                      &generated_c, &cg_error);
    if (!cg_ok || !generated_c) {
        program->jit_c_error_line = cg_error.line;
        program->jit_c_error_column = cg_error.column;
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 cg_error.message[0] ? cg_error.message : "jit c codegen rejected");
        dsl_tracef("jit codegen reject: fp=%s at %d:%d reason=%s",
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
    program->jit_use_runtime_math_bridge = cg_options.use_runtime_math_bridge;
    if (program->jit_use_runtime_math_bridge) {
        dsl_tracef("jit codegen: runtime math bridge enabled");
    }
    dsl_tracef("jit ir built: fp=%s compiler=%s fingerprint=%016llx",
               dsl_fp_mode_name(program->fp_mode),
               dsl_compiler_name(program->compiler),
               (unsigned long long)program->jit_ir_fingerprint);
    if (prepare_runtime) {
        dsl_try_prepare_jit_runtime(program);
    }
    else {
        dsl_tracef("jit runtime skipped: fp=%s compiler=%s reason=jit_mode=off",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler));
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
            if (!dsl_compile_condition_expr(ctx, stmt->as.if_stmt.cond, &compiled->as.if_stmt.cond)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            if (!dsl_compile_block(ctx, &stmt->as.if_stmt.then_block,
                                   &compiled->as.if_stmt.then_block)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }

            compiled->as.if_stmt.n_elifs = stmt->as.if_stmt.n_elifs;
            compiled->as.if_stmt.elif_capacity = stmt->as.if_stmt.n_elifs;
            if (compiled->as.if_stmt.n_elifs > 0) {
                compiled->as.if_stmt.elif_branches = calloc((size_t)compiled->as.if_stmt.n_elifs,
                                                            sizeof(*compiled->as.if_stmt.elif_branches));
                if (!compiled->as.if_stmt.elif_branches) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }
            for (int i = 0; i < stmt->as.if_stmt.n_elifs; i++) {
                me_dsl_if_branch *elif_branch = &stmt->as.if_stmt.elif_branches[i];
                me_dsl_compiled_if_branch *out_branch = &compiled->as.if_stmt.elif_branches[i];
                if (!dsl_compile_condition_expr(ctx, elif_branch->cond, &out_branch->cond)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                if (!dsl_compile_block(ctx, &elif_branch->block, &out_branch->block)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
            }

            if (stmt->as.if_stmt.has_else) {
                if (!dsl_compile_block(ctx, &stmt->as.if_stmt.else_block,
                                       &compiled->as.if_stmt.else_block)) {
                    dsl_compiled_stmt_free(compiled);
                    return false;
                }
                compiled->as.if_stmt.has_else = true;
            }
            break;
        }
        case ME_DSL_STMT_WHILE: {
            if (!dsl_compile_condition_expr(ctx, stmt->as.while_loop.cond, &compiled->as.while_loop.cond)) {
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth++;
            if (!dsl_compile_block(ctx, &stmt->as.while_loop.body, &compiled->as.while_loop.body)) {
                ctx->loop_depth--;
                dsl_compiled_stmt_free(compiled);
                return false;
            }
            ctx->loop_depth--;
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

            if (!dsl_compile_for_range_args(ctx, stmt, compiled)) {
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
                if (!dsl_compile_condition_expr(ctx, stmt->as.flow.cond, &compiled->as.flow.cond)) {
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
                                                    const me_variable *variables,
                                                    int var_count,
                                                    me_dtype dtype,
                                                    int jit_mode,
                                                    int *error_pos,
                                                    bool *is_dsl,
                                                    char *error_reason,
                                                    size_t error_reason_cap) {
    me_dsl_error parse_error;
    if (error_reason && error_reason_cap > 0) {
        error_reason[0] = '\0';
    }
    if (is_dsl) {
        *is_dsl = false;
    }
    me_dsl_program *parsed = me_dsl_parse(source, &parse_error);
    if (!parsed) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap, "dsl parse error: %s",
                     parse_error.message[0] ? parse_error.message : "unknown parse error");
        }
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

    me_dsl_compiled_program *program = calloc(1, sizeof(*program));
    if (!program) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap, "out of memory while allocating DSL program");
        }
        me_dsl_program_free(parsed);
        if (error_pos) {
            *error_pos = -1;
        }
        return NULL;
    }
    program->fp_mode = parsed->fp_mode;
    program->compiler = parsed->compiler;
    dsl_var_table_init(&program->vars);
    program->idx_ndim = -1;
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
        me_dsl_program_free(parsed);
        if (error_pos) {
            *error_pos = -1;
        }
        return NULL;
    }
    for (int i = 0; i < ME_MAX_VARS; i++) {
        program->local_slots[i] = -1;
    }

    me_variable *funcs = NULL;
    int func_count = 0;
    int func_capacity = 0;
    int input_count = 0;

    for (int i = 0; i < var_count; i++) {
        const me_variable *entry = &variables[i];
        const char *name = entry->name;
        if (!name) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap, "invalid DSL input: variable name is NULL");
            }
            if (error_pos) {
                *error_pos = -1;
            }
            free(funcs);
            dsl_compiled_program_free(program);
            me_dsl_program_free(parsed);
            return NULL;
        }
        if (!is_variable_entry(entry) && !is_function_entry(entry)) {
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap, "invalid DSL input entry type for '%s'", name);
            }
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
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL function name '%s' is reserved or collides with a builtin", name);
                }
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
                    if (error_reason && error_reason_cap > 0) {
                        snprintf(error_reason, error_reason_cap,
                                 "DSL function name '%s' collides with a parameter name", name);
                    }
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
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL function '%s' has unsupported return dtype", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            if (!entry->address) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL function '%s' has NULL address", name);
                }
                if (error_pos) {
                    *error_pos = -1;
                }
                free(funcs);
                dsl_compiled_program_free(program);
                me_dsl_program_free(parsed);
                return NULL;
            }
            if (dsl_var_table_find(&program->vars, name) >= 0) {
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "duplicate DSL symbol '%s' in input/function table", name);
                }
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
                    if (error_reason && error_reason_cap > 0) {
                        snprintf(error_reason, error_reason_cap,
                                 "duplicate DSL function name '%s'", name);
                    }
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
                me_variable *next = realloc(funcs, (size_t)new_cap * sizeof(*next));
                if (!next) {
                    if (error_reason && error_reason_cap > 0) {
                        snprintf(error_reason, error_reason_cap,
                                 "out of memory while storing DSL function symbols");
                    }
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
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "DSL input variable '%s' uses a reserved name", name);
            }
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
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "DSL input variable '%s' collides with a function name", name);
                }
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
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "failed to register DSL input variable '%s'", name);
            }
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
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap,
                     "DSL input count mismatch: function expects %d parameter(s), got %d",
                     parsed->nparams, input_count);
        }
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
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "missing DSL input for parameter '%s'", parsed->params[i]);
            }
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
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "output dtype is ME_AUTO but variable '%s' dtype is unspecified",
                             program->vars.names[i]);
                }
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
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "failed to register reserved index symbol '%s'", name);
                }
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
                if (error_reason && error_reason_cap > 0) {
                    snprintf(error_reason, error_reason_cap,
                             "failed to register reserved shape symbol '%s'", name);
                }
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
            if (error_reason && error_reason_cap > 0) {
                snprintf(error_reason, error_reason_cap,
                         "failed to register reserved symbol '_ndim'");
            }
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
    ctx.allow_new_locals = true;
    ctx.error_pos = error_pos;
    ctx.output_expr = NULL;
    ctx.has_return = false;
    ctx.return_dtype = ME_AUTO;
    ctx.program = program;
    ctx.funcs = funcs;
    ctx.func_count = func_count;
    ctx.error_reason = error_reason;
    ctx.error_reason_cap = error_reason_cap;

    if (!dsl_compile_block(&ctx, &parsed->block, &program->block)) {
        if (error_reason && error_reason_cap > 0 && error_reason[0] == '\0') {
            snprintf(error_reason, error_reason_cap, "failed to compile DSL statement block");
        }
        free(funcs);
        dsl_compiled_program_free(program);
        me_dsl_program_free(parsed);
        return NULL;
    }

    if (!ctx.has_return || !ctx.output_expr || !ctx.output_expr->expr) {
        if (error_reason && error_reason_cap > 0) {
            snprintf(error_reason, error_reason_cap,
                     "DSL kernel must return a value via a return statement");
        }
        if (error_pos) {
            *error_pos = -1;
        }
        free(funcs);
        me_dsl_program_free(parsed);
        dsl_compiled_program_free(program);
        return NULL;
    }

    program->output_dtype = ctx.return_dtype;
    program->guaranteed_return = dsl_compiled_block_guarantees_return(&program->block);
    program->output_is_scalar = contains_reduction(ctx.output_expr->expr) &&
                                output_is_scalar(ctx.output_expr->expr);
    dsl_try_build_jit_ir(&ctx, parsed, program, jit_mode != ME_JIT_OFF);

    me_dsl_program_free(parsed);
    free(funcs);

    return program;
}

// Check if a pointer is a synthetic address
int is_synthetic_address(const void* ptr) {
    const char* p = (const char*)ptr;
    return (p >= synthetic_var_addresses && p < synthetic_var_addresses + ME_MAX_VARS);
}

static int compile_with_jit(const char* expression, const me_variable* variables,
                                       int var_count, me_dtype dtype, int jit_mode,
                                       int* error, me_expr** out) {
    me_clear_last_error_message();
    if (out) *out = NULL;
    if (!expression || !out) {
        me_set_last_error_message("invalid compile arguments: expression and output pointer are required");
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    if (dsl_is_candidate(expression)) {
        me_variable *vars_dsl = NULL;
        if (variables && var_count > 0) {
            vars_dsl = malloc((size_t)var_count * sizeof(*vars_dsl));
            if (!vars_dsl) {
                me_set_last_error_message("out of memory while preparing DSL variable table");
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
        char dsl_reason[ME_LAST_ERROR_MSG_CAP] = {0};
        me_dsl_compiled_program *program = dsl_compile_program(
            expression, vars_dsl ? vars_dsl : variables, var_count, dtype, jit_mode,
            &dsl_error, &is_dsl, dsl_reason, sizeof(dsl_reason));
        free(vars_dsl);

        if (program) {
            me_expr *expr = new_expr(ME_CONSTANT, NULL);
            if (!expr) {
                dsl_compiled_program_free(program);
                me_set_last_error_message("out of memory while creating compiled DSL expression");
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
            if (dsl_reason[0] != '\0') {
                me_set_last_error_message(dsl_reason);
            }
            else if (dsl_error >= 0) {
                me_set_last_error_messagef("dsl parse/compile failed near offset %d", dsl_error);
            }
            else {
                me_set_last_error_message("dsl parse/compile failed");
            }
            if (error) *error = dsl_error;
            return ME_COMPILE_ERR_PARSE;
        }
    }

    /* Optional: auto-lift plain expressions into a synthetic DSL kernel for JIT.
     * Keep this opt-in to preserve legacy parser semantics by default.
     */
    if (jit_mode == ME_JIT_ON) {
        me_variable *vars_dsl = NULL;
        if (variables && var_count > 0) {
            vars_dsl = malloc((size_t)var_count * sizeof(*vars_dsl));
            if (!vars_dsl) {
                me_set_last_error_message("out of memory while preparing JIT DSL wrapper inputs");
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

        char *dsl_wrapped = dsl_wrap_expression_as_function(
            expression, vars_dsl ? vars_dsl : variables, var_count);
        if (dsl_wrapped) {
            bool is_dsl = false;
            int dsl_error = -1;
            char dsl_reason[ME_LAST_ERROR_MSG_CAP] = {0};
            me_dsl_compiled_program *program = dsl_compile_program(
                dsl_wrapped, vars_dsl ? vars_dsl : variables, var_count, dtype, jit_mode,
                &dsl_error, &is_dsl, dsl_reason, sizeof(dsl_reason));
            free(dsl_wrapped);
            free(vars_dsl);
            if (program) {
                me_expr *expr = new_expr(ME_CONSTANT, NULL);
                if (!expr) {
                    dsl_compiled_program_free(program);
                    me_set_last_error_message("out of memory while creating JIT DSL wrapped expression");
                    if (error) *error = -1;
                    return ME_COMPILE_ERR_OOM;
                }
                expr->dsl_program = program;
                expr->dtype = program->output_dtype;
                if (error) *error = 0;
                *out = expr;
                return ME_COMPILE_SUCCESS;
            }
            if (is_dsl && dsl_reason[0] != '\0') {
                me_set_last_error_message(dsl_reason);
            }
        }
        else {
            free(vars_dsl);
        }
    }

    // For chunked evaluation, we compile without specific output/nitems
    // If variables have NULL addresses, assign synthetic unique addresses for ordinal matching
    me_variable* vars_copy = NULL;
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
            vars_copy = malloc(var_count * sizeof(me_variable));
            if (!vars_copy) {
                me_set_last_error_message("out of memory while creating synthetic variable addresses");
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
            if (status != ME_COMPILE_SUCCESS) {
                if (error && *error > 0) {
                    me_set_last_error_messagef("expression parse failed near offset %d", *error);
                }
                else {
                    me_set_last_error_message("expression compilation failed");
                }
            }
            return status;
        }
    }

    // No NULL addresses, use variables as-is
    int status = private_compile_ex(expression, variables, var_count, NULL, 0, dtype, error, out);
    if (status != ME_COMPILE_SUCCESS) {
        if (error && *error > 0) {
            me_set_last_error_messagef("expression parse failed near offset %d", *error);
        }
        else {
            me_set_last_error_message("expression compilation failed");
        }
    }
    return status;
}

int me_compile(const char* expression, const me_variable* variables,
               int var_count, me_dtype dtype, int* error, me_expr** out) {
    return compile_with_jit(expression, variables, var_count, dtype, ME_JIT_DEFAULT, error, out);
}

static int compile_nd_with_jit(const char* expression, const me_variable* variables,
                                          int var_count, me_dtype dtype, int ndims,
                                          const int64_t* shape, const int32_t* chunkshape,
                                          const int32_t* blockshape, int jit_mode,
                                          int* error, me_expr** out) {
    if (out) *out = NULL;
    if (!expression || !out || ndims <= 0 || !shape || !chunkshape || !blockshape) {
        me_set_last_error_message("invalid nd compile arguments: expression, shapes and output pointer are required");
        if (error) *error = -1;
        return ME_COMPILE_ERR_INVALID_ARG;
    }

    for (int i = 0; i < ndims; i++) {
        if (chunkshape[i] <= 0 || blockshape[i] <= 0) {
            me_set_last_error_messagef("invalid nd shape metadata at dim %d: chunkshape and blockshape must be > 0", i);
            if (error) *error = -1;
            return ME_COMPILE_ERR_INVALID_ARG;
        }
    }

    me_expr* expr = NULL;
    int rc = compile_with_jit(expression, variables, var_count, dtype, jit_mode, error, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        return rc;
    }

    const size_t extra_items = (size_t)(3 * ndims - 1);
    const size_t info_size = sizeof(me_nd_info) + extra_items * sizeof(int64_t);
    me_nd_info* info = malloc(info_size);
    if (!info) {
        me_free(expr);
        me_set_last_error_message("out of memory while allocating nd metadata");
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

int me_compile_nd_jit(const char* expression, const me_variable* variables,
                      int var_count, me_dtype dtype, int ndims,
                      const int64_t* shape, const int32_t* chunkshape,
                      const int32_t* blockshape, int jit_mode,
                      int* error, me_expr** out) {
    return compile_nd_with_jit(expression, variables, var_count, dtype, ndims,
                                          shape, chunkshape, blockshape, jit_mode, error, out);
}

int me_compile_nd(const char* expression, const me_variable* variables,
                  int var_count, me_dtype dtype, int ndims,
                  const int64_t* shape, const int32_t* chunkshape,
                  const int32_t* blockshape, int* error, me_expr** out) {
    return me_compile_nd_jit(expression, variables, var_count, dtype, ndims,
                             shape, chunkshape, blockshape, ME_JIT_DEFAULT, error, out);
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

const char *me_get_last_error_message(void) {
    if (g_me_last_error_msg[0] == '\0') {
        return NULL;
    }
    return g_me_last_error_msg;
}

bool me_expr_has_jit_kernel(const me_expr *expr) {
    if (!expr || !expr->dsl_program) {
        return false;
    }
    const me_dsl_compiled_program *program = (const me_dsl_compiled_program *)expr->dsl_program;
    return program->jit_kernel_fn != NULL;
}

void me_register_wasm_jit_helpers(me_wasm_jit_instantiate_helper instantiate_helper,
                                  me_wasm_jit_free_helper free_helper) {
#if ME_USE_WASM32_JIT
    g_me_wasm_jit_instantiate_helper = instantiate_helper;
    g_me_wasm_jit_free_helper = free_helper;
#else
    (void)instantiate_helper;
    (void)free_helper;
#endif
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
        return dsl_string_nonempty_at(data, item_size, idx);
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
    if (cond_dtype == ME_STRING) {
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
            int rc = dsl_eval_for_element_loop(ctx, stmt, run_mask, return_mask);
            if (rc != ME_EVAL_SUCCESS) {
                return rc;
            }
            break;
        }
        case ME_DSL_STMT_WHILE: {
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

        rc = dsl_eval_block_element_loop(ctx, &stmt->as.while_loop.body, run_mask,
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

    int64_t start_val = 0;
    int64_t stop_val = 0;
    int64_t step_val = 0;
    int rc = ME_EVAL_SUCCESS;

    {
        me_dtype dtype = me_get_dtype(start_expr->expr);
        size_t size = dtype_size(dtype);
        void *buf = malloc(size ? size : sizeof(int64_t));
        if (!buf) {
            return ME_EVAL_ERR_OOM;
        }
        rc = dsl_eval_expr_nitems(ctx, start_expr, buf, 1);
        if (rc == ME_EVAL_SUCCESS && !dsl_read_int64(buf, dtype, &start_val)) {
            rc = ME_EVAL_ERR_INVALID_ARG;
        }
        free(buf);
        if (rc != ME_EVAL_SUCCESS) {
            return rc;
        }
    }
    {
        me_dtype dtype = me_get_dtype(stop_expr->expr);
        size_t size = dtype_size(dtype);
        void *buf = malloc(size ? size : sizeof(int64_t));
        if (!buf) {
            return ME_EVAL_ERR_OOM;
        }
        rc = dsl_eval_expr_nitems(ctx, stop_expr, buf, 1);
        if (rc == ME_EVAL_SUCCESS && !dsl_read_int64(buf, dtype, &stop_val)) {
            rc = ME_EVAL_ERR_INVALID_ARG;
        }
        free(buf);
        if (rc != ME_EVAL_SUCCESS) {
            return rc;
        }
    }
    {
        me_dtype dtype = me_get_dtype(step_expr->expr);
        size_t size = dtype_size(dtype);
        void *buf = malloc(size ? size : sizeof(int64_t));
        if (!buf) {
            return ME_EVAL_ERR_OOM;
        }
        rc = dsl_eval_expr_nitems(ctx, step_expr, buf, 1);
        if (rc == ME_EVAL_SUCCESS && !dsl_read_int64(buf, dtype, &step_val)) {
            rc = ME_EVAL_ERR_INVALID_ARG;
        }
        free(buf);
        if (rc != ME_EVAL_SUCCESS) {
            return rc;
        }
    }

    if (step_val == 0) {
        return ME_EVAL_ERR_INVALID_ARG;
    }
    if (ctx->nitems <= 0) {
        return ME_EVAL_SUCCESS;
    }
    if ((step_val > 0 && start_val >= stop_val) ||
        (step_val < 0 && start_val <= stop_val)) {
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

    int slot = stmt->as.for_loop.loop_var_slot;
    int64_t *loop_buf = (int64_t *)ctx->local_buffers[slot];

    for (int64_t iter = start_val;
         (step_val > 0) ? (iter < stop_val) : (iter > stop_val);) {
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
                                         break_mask, continue_mask, return_mask);
        free(run_mask);
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

        if (step_val > 0) {
            if (iter > INT64_MAX - step_val) {
                break;
            }
        }
        else {
            if (iter < INT64_MIN - step_val) {
                break;
            }
        }
        iter += step_val;
    }

    free(active_mask);
    return ME_EVAL_SUCCESS;
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
    memset(run_mask, 1, (size_t)ctx->nitems);

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
            if (!return_mask[i]) {
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
    if (!me_eval_jit_disabled(params) &&
        program->jit_kernel_fn &&
        program->jit_nparams >= 0 &&
        program->jit_nparams <= ME_MAX_VARS) {
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
