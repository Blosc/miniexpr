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

#ifndef MINIEXPR_H
#define MINIEXPR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {


#endif

/* Version information */
#define ME_VERSION_MAJOR 0
#define ME_VERSION_MINOR 1
#define ME_VERSION_PATCH 1
#define ME_VERSION_STRING "0.1.1.dev"

/* Internal eval block size (elements). Compile-time fixed. */
#ifndef ME_EVAL_BLOCK_NITEMS
#define ME_EVAL_BLOCK_NITEMS 4096
#endif

/* Maximum number of variables supported in a single expression. */
#ifndef ME_MAX_VARS
#define ME_MAX_VARS 128
#endif

/* Enable internal eval blocking for large chunks (1 = on, 0 = off). */
#ifndef ME_EVAL_ENABLE_BLOCKING
#define ME_EVAL_ENABLE_BLOCKING 1
#endif


/* Data type enumeration - Full C99 support */
typedef enum {
    /* Automatic type inference */
    ME_AUTO,

    /* Boolean */
    ME_BOOL,

    /* Signed integers */
    ME_INT8,
    ME_INT16,
    ME_INT32,
    ME_INT64,

    /* Unsigned integers */
    ME_UINT8,
    ME_UINT16,
    ME_UINT32,
    ME_UINT64,

    /* Floating point */
    ME_FLOAT32,
    ME_FLOAT64,

    /* Complex (C99) */
    ME_COMPLEX64, /* float complex */
    ME_COMPLEX128, /* double complex */

    /* Fixed-size UCS4 strings (NUL-terminated, no embedded NULs) */
    ME_STRING
} me_dtype;

/* Opaque type for compiled expressions */
typedef struct me_expr me_expr;


enum {
    ME_VARIABLE = 0,

    ME_FUNCTION0 = 8, ME_FUNCTION1, ME_FUNCTION2, ME_FUNCTION3,
    ME_FUNCTION4, ME_FUNCTION5, ME_FUNCTION6, ME_FUNCTION7,

    ME_CLOSURE0 = 16, ME_CLOSURE1, ME_CLOSURE2, ME_CLOSURE3,
    ME_CLOSURE4, ME_CLOSURE5, ME_CLOSURE6, ME_CLOSURE7,

    ME_FLAG_PURE = 32
};

/* Variable definition with per-element byte size (required for ME_STRING).
 * For numeric types, itemsize can be 0 to use the default dtype size.
 * For function/closure entries (ME_FUNCTION* / ME_CLOSURE*), dtype is the return type.
 */
typedef struct me_variable {
    const char *name;
    me_dtype dtype; // Data type of this variable or function return (ME_AUTO = use output dtype)
    const void *address; // Pointer to data (NULL for me_compile)
    int type; // ME_VARIABLE for user variables (0 = auto-set to ME_VARIABLE)
    void *context; // For closures/functions (NULL for normal variables)
    size_t itemsize; // Bytes per element (required for ME_STRING; 0 = default dtype size)
} me_variable;

/* Note: When initializing variables, only name/dtype/address/itemsize are typically needed.
 * Unspecified fields default to 0/NULL, which is correct for normal use:
 *   {"varname"}                          → defaults all fields
 *   {"varname", ME_FLOAT64}              → for me_compile with mixed types
 *   {"varname", ME_FLOAT64, var_array}   → for me_compile with address
 *   {"varname", ME_STRING, var_array, 0, NULL, item_size} → for strings
 * Advanced users can specify type for closures/functions if needed.
 */


/* Compile expression for chunked evaluation.
 * This function is optimized for use with me_eval(),
 * where variable and output pointers are provided later during evaluation.
 *
 * Parameters:
 *   expression: The expression string to compile
 *   variables: Array of variable definitions. Only the 'name' field is required.
 *              Variables will be matched by position (ordinal order) during me_eval().
 *   var_count: Number of variables
 *   dtype: Data type handling:
 *          - ME_AUTO: All variables must specify their dtypes, output is inferred
 *          - Specific type: Either all variables are ME_AUTO (homogeneous, all use this type),
 *            OR all variables have explicit dtypes (heterogeneous, result cast to this type)
 *   error: Optional pointer to receive error position (0 on success, >0 on parse error)
 *   out: Output pointer to receive the compiled expression
 *
 * Returns: ME_COMPILE_SUCCESS (0) on success, or a negative ME_COMPILE_ERR_* code on failure
 *
 * Example 1 (simple - all same type):
 *   me_variable vars[] = {{"x"}, {"y"}};  // Both ME_AUTO
 *   me_expr *expr = NULL;
 *   if (me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) { return; }
 *
 * Example 2 (mixed types with ME_AUTO):
 *   me_variable vars[] = {{"x", ME_INT32}, {"y", ME_FLOAT64}};
 *   me_expr *expr = NULL;
 *   if (me_compile("x + y", vars, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { return; }
 *
 * Example 3 (mixed types with explicit output):
 *   me_variable vars[] = {{"x", ME_INT32}, {"y", ME_FLOAT64}};
 *   me_expr *expr = NULL;
 *   if (me_compile("x + y", vars, 2, ME_FLOAT32, &err, &expr) != ME_COMPILE_SUCCESS) { return; }
 *   // Variables keep their types, result is cast to FLOAT32
 *
 *   // Later, provide data in same order as variable definitions
 *   const void *data[] = {x_array, y_array};  // x first, y second
 *   if (me_eval(expr, data, 2, output, nitems) != ME_EVAL_SUCCESS) { return; }
 */
int me_compile(const char *expression, const me_variable *variables,
               int var_count, me_dtype dtype, int *error, me_expr **out);

/* Compile expression with multidimensional metadata (b2nd-aware).
 * Additional parameters describe the logical array shape, chunkshape and
 * blockshape (all C-order). Padding is implied by these shapes.
 *
 * Parameters:
 *   ndims: Number of dimensions.
 *   shape: Logical array shape (length ndims).
 *   chunkshape: Chunk shape (length ndims).
 *   blockshape: Block shape inside a chunk (length ndims).
 *
 * Returns the same status codes as me_compile().
 */
int me_compile_nd(const char *expression, const me_variable *variables,
                  int var_count, me_dtype dtype, int ndims,
                  const int64_t *shape, const int32_t *chunkshape,
                  const int32_t *blockshape, int *error, me_expr **out);

/* Compile expression with multidimensional metadata and a JIT-mode hint.
 * jit_mode values:
 *   0 -> default policy
 *   1 -> prefer JIT
 *   2 -> disable JIT preparation at compile time
 */
int me_compile_nd_jit(const char *expression, const me_variable *variables,
                      int var_count, me_dtype dtype, int ndims,
                      const int64_t *shape, const int32_t *chunkshape,
                      const int32_t *blockshape, int jit_mode,
                      int *error, me_expr **out);

/* Returns a thread-local, human-readable diagnostic message for the most
 * recent failure in this thread (primarily compile/setup failures).
 * Returns NULL when no detailed diagnostic is available.
 */
const char *me_get_last_error_message(void);

/* Status codes for me_compile(). */
typedef enum {
    ME_COMPILE_SUCCESS = 0,
    ME_COMPILE_ERR_OOM = -1,
    ME_COMPILE_ERR_PARSE = -2,
    ME_COMPILE_ERR_INVALID_ARG = -3,
    ME_COMPILE_ERR_COMPLEX_UNSUPPORTED = -4,
    ME_COMPILE_ERR_REDUCTION_INVALID = -5,
    ME_COMPILE_ERR_VAR_MIXED = -6,
    ME_COMPILE_ERR_VAR_UNSPECIFIED = -7,
    ME_COMPILE_ERR_INVALID_ARG_TYPE = -8,
    ME_COMPILE_ERR_MIXED_TYPE_NESTED = -9  /* Nested expressions with mixed types not supported */
} me_compile_status;

/* Status codes for me_eval(). */
typedef enum {
    ME_EVAL_SUCCESS = 0,
    ME_EVAL_ERR_OOM = -1,
    ME_EVAL_ERR_NULL_EXPR = -2,
    ME_EVAL_ERR_TOO_MANY_VARS = -3,
    ME_EVAL_ERR_VAR_MISMATCH = -4,
    ME_EVAL_ERR_INVALID_ARG = -5
} me_eval_status;

/* SIMD precision options for transcendentals. */
typedef enum {
    ME_SIMD_ULP_DEFAULT = 0,
    ME_SIMD_ULP_1 = 1,
    ME_SIMD_ULP_3_5 = 2
} me_simd_ulp_mode;

/* JIT policy for a single evaluation call. */
typedef enum {
    ME_JIT_DEFAULT = 0,  /* auto policy (environment/default behavior) */
    ME_JIT_ON = 1,       /* prefer JIT when available */
    ME_JIT_OFF = 2       /* disable JIT for this call */
} me_jit_mode;

#ifndef ME_SIMD_ULP_DEFAULT_MODE
#define ME_SIMD_ULP_DEFAULT_MODE ME_SIMD_ULP_3_5
#endif

/* Optional per-call evaluation parameters. */
typedef struct {
    bool disable_simd;
    me_simd_ulp_mode simd_ulp_mode;
    me_jit_mode jit_mode;
} me_eval_params;

#define ME_EVAL_PARAMS_DEFAULTS ((me_eval_params){false, ME_SIMD_ULP_DEFAULT, ME_JIT_DEFAULT})

/* Evaluates compiled expression with variable and output pointers.
 * This function can be safely called from multiple threads simultaneously on the
 * same compiled expression. It creates a temporary clone of the expression tree
 * for each call, eliminating race conditions at the cost of some memory allocation.
 *
 * Parameters:
 *   expr: Compiled expression (from me_compile)
 *   vars_block: Array of pointers to variable data blocks (same order as in me_compile)
 *   n_vars: Number of variables (must match the number used in me_compile)
 *   output_block: Pointer to output buffer for this block
 *   block_nitems: Number of elements in this block. This is an element count
 *                 (not bytes) and must correspond to the input arrays' element
 *                 count; the output buffer must be sized for this many output
 *                 elements (using the output dtype size).
 *   params: Optional SIMD evaluation settings (NULL for defaults).
 *
 * Returns:
 *   ME_EVAL_SUCCESS (0) on success, or a negative ME_EVAL_ERR_* code on failure.
 *
 * Use this function for both serial and parallel evaluation. It is thread-safe
 * and can be used from multiple threads to process different chunks simultaneously.
 */
int me_eval(const me_expr *expr, const void **vars_block,
            int n_vars, void *output_block, int block_nitems,
            const me_eval_params *params);

/* Evaluate a padded b2nd block.
 * Only the valid (unpadded) elements are computed; padded output is zeroed.
 * nchunk and nblock are C-order indices for the chunk within the array
 * and the block within that chunk.
 * vars_block points to block buffers (not base arrays).
 */
int me_eval_nd(const me_expr *expr, const void **vars_block,
               int n_vars, void *output_block, int block_nitems,
               int64_t nchunk, int64_t nblock, const me_eval_params *params);

/* Query number of valid (unpadded) elements for a given chunk/block. */
int me_nd_valid_nitems(const me_expr *expr, int64_t nchunk, int64_t nblock, int64_t *valid_nitems);

/* Prints the expression tree for debugging purposes. */
void me_print(const me_expr *n);

/* Frees the expression. */
/* This is safe to call on NULL pointers. */
void me_free(me_expr *n);

/* Get the result data type of a compiled expression.
 * Returns the dtype that will be used for the output of me_eval().
 */
me_dtype me_get_dtype(const me_expr *expr);

/* Returns true when a DSL expression has a callable JIT runtime kernel bound. */
bool me_expr_has_jit_kernel(const me_expr *expr);

/* Host-registered wasm32 JIT helpers (for side-module runtimes like Pyodide).
 * instantiate must return a non-zero function-table index on success, or 0 on
 * failure. free receives that index and should release it (if non-zero).
 * When helpers are not registered, wasm32 JIT runtime compilation is skipped.
 */
typedef int (*me_wasm_jit_instantiate_helper)(const unsigned char *wasm_bytes,
                                              int wasm_len,
                                              int bridge_lookup_fn_idx);
typedef void (*me_wasm_jit_free_helper)(int fn_idx);
void me_register_wasm_jit_helpers(me_wasm_jit_instantiate_helper instantiate_helper,
                                  me_wasm_jit_free_helper free_helper);

/* Get the library version string (e.g., "1.0.0"). */
const char *me_version(void);


#ifdef __cplusplus
}
#endif

#endif /*MINIEXPR_H*/
