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
#include "miniexpr_internal.h"
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
#include "dsl_config.h"
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include "dsl_jit_ir.h"
#include "dsl_jit_cgen.h"
#include "dsl_jit_test.h"
#include "dsl_jit_runtime_internal.h"

#define ME_DSL_JIT_SYNTH_ND_CTX_PARAM "__me_nd_ctx"
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

float me_crealf(float _Complex v);
float me_cimagf(float _Complex v);
double me_creal(double _Complex v);
double me_cimag(double _Complex v);

int private_compile_ex(const char* expression, const me_variable* variables, int var_count,
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

static int64_t ceil_div64(int64_t a, int64_t b) {
    return (b == 0) ? 0 : (a + b - 1) / b;
}

int64_t dsl_i64_add_wrap(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a + (uint64_t)b);
}

static int64_t dsl_i64_mul_wrap(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a * (uint64_t)b);
}

int64_t dsl_i64_addmul_wrap(int64_t acc, int64_t a, int64_t b) {
    return dsl_i64_add_wrap(acc, dsl_i64_mul_wrap(a, b));
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


bool contains_reduction(const me_expr* n) {
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
char synthetic_var_addresses[ME_MAX_VARS];

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
    if (strcmp(name, "_flat_idx") == 0) {
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

static bool dsl_dtype_is_integer(me_dtype dtype) {
    switch (dtype) {
    case ME_BOOL:
    case ME_INT8:
    case ME_INT16:
    case ME_INT32:
    case ME_INT64:
    case ME_UINT8:
    case ME_UINT16:
    case ME_UINT32:
    case ME_UINT64:
        return true;
    default:
        return false;
    }
}

static bool dsl_dtype_is_float_or_complex(me_dtype dtype) {
    switch (dtype) {
    case ME_FLOAT32:
    case ME_FLOAT64:
    case ME_COMPLEX64:
    case ME_COMPLEX128:
        return true;
    default:
        return false;
    }
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

bool output_is_scalar(const me_expr* n) {
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

bool dsl_any_nonzero(const void *data, me_dtype dtype, int nitems) {
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

void dsl_fill_i64(int64_t *out, int nitems, int64_t value) {
    if (!out || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        out[i] = value;
    }
}

void dsl_fill_iota_i64(int64_t *out, int nitems, int64_t start) {
    if (!out || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        out[i] = start + (int64_t)i;
    }
}

bool dsl_read_int64(const void *data, me_dtype dtype, int64_t *out) {
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

float me_crealf(float _Complex v) {
#if defined(_MSC_VER)
    return __real__ v;
#else
    return crealf(v);
#endif
}

float me_cimagf(float _Complex v) {
#if defined(_MSC_VER)
    return __imag__ v;
#else
    return cimagf(v);
#endif
}

double me_creal(double _Complex v) {
#if defined(_MSC_VER)
    return __real__ v;
#else
    return creal(v);
#endif
}

double me_cimag(double _Complex v) {
#if defined(_MSC_VER)
    return __imag__ v;
#else
    return cimag(v);
#endif
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

            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && (n->flags & ME_EXPR_FLAG_FLOAT_MATH)) {
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

            if (IS_FUNCTION(n->type) && ARITY(n->type) == 1 && (n->flags & ME_EXPR_FLAG_FLOAT_MATH)) {
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

int private_compile_ex(const char* expression, const me_variable* variables, int var_count,
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

        // If dtype is ME_AUTO, infer from expression; otherwise honor explicit dtype.
        if (dtype == ME_AUTO) {
            root->dtype = infer_output_type(root);
        }
        else {
            me_dtype inferred_dtype = infer_output_type(root);
            if (inferred_dtype != dtype && TYPE_MASK(root->type) == ME_VARIABLE) {
                me_expr *converted = create_conversion_node(root, dtype);
                if (!converted) {
                    me_free(root);
                    if (error) *error = -1;
                    if (vars_copy) free(vars_copy);
                    if (s.str_data) {
                        free((void *)s.str_data);
                    }
                    return ME_COMPILE_ERR_OOM;
                }
                converted->flags |= ME_EXPR_FLAG_EXPLICIT_DTYPE;
                converted->output = output;
                converted->nitems = nitems;
                root = converted;
            }
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

bool me_eval_jit_disabled(const me_eval_params *params) {
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
    int max_chunk = dsl_jit_bridge_chunk_items();
    int64_t remaining = nitems;
    const double *pin = in;
    double *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > (int64_t)max_chunk) ? max_chunk : (int)remaining;
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
    int max_chunk = dsl_jit_bridge_chunk_items();
    int64_t remaining = nitems;
    const float *pin = in;
    float *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > (int64_t)max_chunk) ? max_chunk : (int)remaining;
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
    int max_chunk = dsl_jit_bridge_chunk_items();
    int64_t remaining = nitems;
    const double *pa = a;
    const double *pb = b;
    double *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > (int64_t)max_chunk) ? max_chunk : (int)remaining;
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
    int max_chunk = dsl_jit_bridge_chunk_items();
    int64_t remaining = nitems;
    const float *pa = a;
    const float *pb = b;
    float *pout = out;
    while (remaining > 0) {
        int chunk = (remaining > (int64_t)max_chunk) ? max_chunk : (int)remaining;
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

static double dsl_jit_bridge_abs(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_abs_dispatch, x);
}

static double dsl_jit_bridge_sin(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_sin_dispatch, x);
}

static double dsl_jit_bridge_cos(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_cos_dispatch, x);
}

static double dsl_jit_bridge_exp(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_exp_dispatch, x);
}

static double dsl_jit_bridge_log(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_log_dispatch, x);
}

static double dsl_jit_bridge_sqrt(double x) {
    return dsl_jit_bridge_apply_unary_f64(vec_sqrt_dispatch, x);
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

static void dsl_jit_bridge_vec_fmax_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f64(vec_fmax_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_fmax_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f32(vec_fmax_f32_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_fmin_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f64(vec_fmin_dispatch, a, b, out, nitems);
}

static void dsl_jit_bridge_vec_fmin_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_apply_binary_vector_f32(vec_fmin_f32_dispatch, a, b, out, nitems);
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

double me_jit_abs(double x) {
    return dsl_jit_bridge_abs(x);
}

double me_jit_sin(double x) {
    return dsl_jit_bridge_sin(x);
}

double me_jit_cos(double x) {
    return dsl_jit_bridge_cos(x);
}

double me_jit_exp(double x) {
    return dsl_jit_bridge_exp(x);
}

double me_jit_log(double x) {
    return dsl_jit_bridge_log(x);
}

double me_jit_sqrt(double x) {
    return dsl_jit_bridge_sqrt(x);
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

void me_jit_vec_fmax_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_fmax_f64(a, b, out, nitems);
}

void me_jit_vec_fmin_f64(const double *a, const double *b, double *out, int64_t nitems) {
    dsl_jit_bridge_vec_fmin_f64(a, b, out, nitems);
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

void me_jit_vec_fmax_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_fmax_f32(a, b, out, nitems);
}

void me_jit_vec_fmin_f32(const float *a, const float *b, float *out, int64_t nitems) {
    dsl_jit_bridge_vec_fmin_f32(a, b, out, nitems);
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
// Check if a pointer is a synthetic address
int is_synthetic_address(const void* ptr) {
    const char* p = (const char*)ptr;
    return (p >= synthetic_var_addresses && p < synthetic_var_addresses + ME_MAX_VARS);
}

static int compile_with_jit(const char* expression, const me_variable* variables,
                                       int var_count, me_dtype dtype, int compile_ndims,
                                       int jit_mode,
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
            expression, vars_dsl ? vars_dsl : variables, var_count, dtype, compile_ndims, jit_mode,
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
                dsl_wrapped, vars_dsl ? vars_dsl : variables, var_count, dtype, compile_ndims, jit_mode,
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
    return compile_with_jit(expression, variables, var_count, dtype,
                            0, ME_JIT_DEFAULT, error, out);
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
    int rc = compile_with_jit(expression, variables, var_count, dtype,
                              ndims, jit_mode, error, &expr);
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
    dsl_register_wasm_jit_helpers(instantiate_helper, free_helper);
#else
    (void)instantiate_helper;
    (void)free_helper;
#endif
}

const char* me_version(void) {
    return ME_VERSION_STRING;
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
                            1, NULL, NULL, NULL, NULL);
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
        return me_eval_nd_dsl(expr, vars_block, n_vars, output_block, block_nitems, nchunk, nblock, params);
    }
    return me_eval_nd_classic(expr, vars_block, n_vars, output_block, block_nitems, nchunk, nblock, params);
}
