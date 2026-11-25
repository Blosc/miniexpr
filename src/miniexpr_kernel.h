/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_KERNEL_H
#define MINIEXPR_KERNEL_H

#include "miniexpr_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1.0 embedding contract. No JSON dependency. Existing miniexpr.h entry points
 * remain compatibility interfaces with their original source-language routing. */
#define ME_KERNEL_API_VERSION 1
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
    bool integer_powers; /* Authored multiply-chain policy for integer exponents. */
    uint8_t reserved[3]; /* Zero-initialize; explicit storage, never ABI padding. */
} me_signature;

#define ME_SIGNATURE_INITIALIZER { sizeof(me_signature), NULL, 0, NULL, 0, \
    ME_AUTO, 0, ME_CARDINALITY_INFER, 0, ME_CAST_UNSAFE, false, {0} }
/* Fixed 1.x baseline prefixes, including explicit reserved storage. Append-only future
 * fields must default when absent; never replace these minima with sizeof.
 * New fields must follow the entire previously released layout, not reuse
 * reserved storage. Each released revision must avoid unnamed tail padding.
 * Unknown input suffixes are ignored and output suffixes are left untouched. */
#define ME_SIGNATURE_V1_SIZE \
    (offsetof(me_signature, reserved) + sizeof(((me_signature *)0)->reserved))

typedef enum {
    ME_EXEC_INTERPRET = 0,
    ME_EXEC_PREFER_JIT = 1,
    ME_EXEC_REQUIRE_JIT = 2
} me_execution_policy;

typedef struct {
    size_t struct_size;
    me_execution_policy execution;
    bool inherit_environment; /* False: deterministic strict/system defaults. */
    uint8_t reserved[3]; /* Zero-initialize; new fields must follow this storage. */
} me_prepare_options;

#define ME_PREPARE_OPTIONS_INITIALIZER { sizeof(me_prepare_options), ME_EXEC_INTERPRET, false, {0} }
#define ME_PREPARE_OPTIONS_V1_SIZE \
    (offsetof(me_prepare_options, reserved) + sizeof(((me_prepare_options *)0)->reserved))

typedef struct {
    size_t struct_size;
    me_dtype output_dtype;
    size_t output_itemsize;
    me_cardinality cardinality;
    int context_ndim;
} me_kernel_info;

#define ME_KERNEL_INFO_INITIALIZER { sizeof(me_kernel_info), ME_AUTO, 0, \
    ME_CARDINALITY_INFER, 0 }
#define ME_KERNEL_INFO_V1_SIZE (offsetof(me_kernel_info, context_ndim) + sizeof(int))

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
#define ME_BLOCK_V1_SIZE (offsetof(me_block, extent) + sizeof(const int64_t *))

typedef struct {
    unsigned flags;
    bool supported;
} me_fp_status;

typedef enum {
    ME_RUN_PREPARED = 0,
    ME_RUN_INTERPRET = 1
} me_run_execution;

typedef struct {
    size_t struct_size;
    unsigned raise_mask;
    me_fp_status *fp_status; /* Optional invocation-local report. */
    me_run_execution execution; /* Invocation override; never mutates the kernel. */
    uint8_t reserved[4];
} me_run_options;

#define ME_RUN_OPTIONS_INITIALIZER { sizeof(me_run_options), 0, NULL, ME_RUN_PREPARED, {0} }
#define ME_RUN_OPTIONS_V1_SIZE (offsetof(me_run_options, fp_status) + sizeof(me_fp_status *))
#define ME_RUN_OPTIONS_EXECUTION_SIZE (offsetof(me_run_options, execution) + sizeof(me_run_execution))

/* Source is NUL-terminated. Inputs must have concrete dtypes. Captures and all
 * retained metadata are copied. On failure *out is NULL; error may be NULL.
 * Formulas currently admit elementwise expressions only: reductions are
 * deliberately unsupported until their 1.0 contract is settled. Formula JIT
 * preference never changes syntax or typing; require-JIT reports unsupported.
 * Menudet accepts "0.1" and the annotated "0.2" language.
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
 * is unspecified. No pointer supplied here survives the call.
 * ME_RUN_INTERPRET bypasses a prepared JIT, even after REQUIRE_JIT preparation.
 * Absent execution fields default to ME_RUN_PREPARED. No compilation or replay
 * occurs during evaluation; the interpreter implements the same semantics. */
me_status me_eval_block(const me_kernel *kernel, const me_buffer *inputs,
    size_t ninputs, me_mut_buffer output, const me_block *block,
    const me_run_options *options, me_error *error);

void me_kernel_free(me_kernel *kernel); /* NULL-safe; not concurrent with use. */

#ifdef __cplusplus
}
#endif
#endif
