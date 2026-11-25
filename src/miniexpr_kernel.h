/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_KERNEL_H
#define MINIEXPR_KERNEL_H

#include "miniexpr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Provisional 1.0 API: qualify downstream callers before freezing this ABI.
 * No JSON dependency. Existing miniexpr.h entry points remain unchanged. */
typedef struct me_kernel me_kernel;

typedef enum {
    ME_SUCCESS = 0,
    ME_ERR_INVALID_ARG = -1,
    ME_ERR_UNSUPPORTED = -2,
    ME_ERR_SOURCE = -3,
    ME_ERR_SIGNATURE = -4,
    ME_ERR_EVAL = -5,
    ME_ERR_OOM = -6
} me_status;

typedef struct {
    me_status status;
    int line;
    int column;
    char message[256];
} me_error;

typedef struct {
    const char *name;
    me_dtype dtype;
    size_t itemsize; /* Numeric: zero or dtype width; strings: exact byte width. */
} me_input_spec;

typedef enum {
    ME_CAPTURE_TYPED = 0,
    ME_CAPTURE_WEAK = 1
} me_capture_category;

typedef struct {
    me_input_spec spec;
    const void *value; /* One native value, copied at preparation. */
    me_capture_category category;
} me_capture;

typedef enum {
    ME_CARDINALITY_INFER = 0,
    ME_ELEMENTWISE = 1,
    ME_BLOCK_SCALAR = 2
} me_cardinality;

typedef enum {
    ME_CAST_UNSAFE = 0,
    ME_CAST_SAME_KIND = 1,
    ME_CAST_SAFE = 2
} me_casting;

typedef struct {
    size_t struct_size;
    const me_input_spec *inputs; /* Runtime buffer slot order. */
    size_t ninputs;
    const me_capture *captures; /* Not runtime slots; source parameters by name. */
    size_t ncaptures;
    me_dtype output_dtype; /* ME_AUTO: infer; Menudet return annotation may constrain. */
    size_t output_itemsize; /* Zero: infer; otherwise assert the resulting width. */
    me_cardinality cardinality;
    int context_ndim;
    me_casting casting;
} me_signature;

#define ME_SIGNATURE_INITIALIZER { sizeof(me_signature), NULL, 0, NULL, 0, \
    ME_AUTO, 0, ME_CARDINALITY_INFER, 0, ME_CAST_UNSAFE }

typedef enum {
    ME_EXEC_INTERPRET = 0,
    ME_EXEC_PREFER_JIT = 1,
    ME_EXEC_REQUIRE_JIT = 2
} me_execution_policy;

typedef struct {
    size_t struct_size;
    me_execution_policy execution;
} me_prepare_options;

#define ME_PREPARE_OPTIONS_INITIALIZER { sizeof(me_prepare_options), ME_EXEC_INTERPRET }

typedef struct {
    size_t struct_size;
    me_dtype output_dtype;
    size_t output_itemsize;
    me_cardinality cardinality;
    int context_ndim;
} me_kernel_info;

#define ME_KERNEL_INFO_INITIALIZER { sizeof(me_kernel_info), ME_AUTO, 0, \
    ME_CARDINALITY_INFER, 0 }

typedef struct {
    const void *data;
    size_t capacity; /* Accessible bytes from data. */
} me_buffer;

typedef struct {
    void *data;
    size_t capacity;
} me_mut_buffer;

typedef struct {
    size_t struct_size;
    size_t nitems;
    const uint8_t *valid_mask; /* Optional 0/1 byte per lane; no compaction. */
    size_t valid_mask_capacity;
    int ndim;
    const int64_t *logical_shape;
    const int64_t *origin;
    const int64_t *extent;
} me_block;

#define ME_BLOCK_INITIALIZER { sizeof(me_block), 0, NULL, 0, 0, NULL, NULL, NULL }

typedef struct {
    unsigned flags;
    bool supported;
} me_fp_status;

typedef struct {
    size_t struct_size;
    unsigned raise_mask;
    me_fp_status *fp_status; /* Optional invocation-local report. */
} me_run_options;

#define ME_RUN_OPTIONS_INITIALIZER { sizeof(me_run_options), 0, NULL }

/* Source is NUL-terminated. Inputs must have concrete dtypes. Captures and all
 * retained metadata are copied. On failure *out is NULL; error may be NULL.
 * Formulas currently admit elementwise expressions only: reductions are
 * deliberately unsupported until their 1.0 contract is settled. Formula JIT
 * preference never changes syntax or typing; require-JIT reports unsupported.
 * Menudet accepts "0.1" and the provisional annotated "0.2" language.
 * Input annotations require exact dtypes; return annotations convert last. */
me_status me_prepare_formula(const char *source, const me_signature *signature,
    const me_prepare_options *options, me_kernel **out, me_error *error);
me_status me_prepare_menudet(const char *version, const char *source,
    const me_signature *signature, const me_prepare_options *options,
    me_kernel **out, me_error *error);

/* O(1), allocation-free; reports final output, not intermediate arithmetic.
 * Initialize out with ME_KERNEL_INFO_INITIALIZER. */
me_status me_describe(const me_kernel *kernel, me_kernel_info *out);

/* Native aligned buffers, disjoint output, synchronous execution. Input slots
 * are in signature order, including unused inputs. Masked outputs retain lane
 * spacing; inactive output values are unspecified. A scalar block always needs
 * one output item, even for nitems=0; empty elementwise calls need none.
 * Independent calls may share a kernel, never invocation buffers or status.
 * FP state is restored; failure does not replay execution. Output on failure
 * is unspecified. No pointer supplied here survives the call. */
me_status me_eval_block(const me_kernel *kernel, const me_buffer *inputs,
    size_t ninputs, me_mut_buffer output, const me_block *block,
    const me_run_options *options, me_error *error);

void me_kernel_free(me_kernel *kernel); /* NULL-safe; not concurrent with use. */

#ifdef __cplusplus
}
#endif
#endif
