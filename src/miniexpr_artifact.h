/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE)
**********************************************************************/
#ifndef MINIEXPR_ARTIFACT_H
#define MINIEXPR_ARTIFACT_H

#include "miniexpr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ME_ARTIFACT_SCHEMA_VERSION "1.0"
#define ME_ARTIFACT_NUMPY_SCHEMA_VERSION "1.1"
#define ME_ARTIFACT_MAX_BYTES (1024 * 1024)
#define ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION 1
#define ME_ARTIFACT_DRAFT_SCHEMA_VERSION "1.0"

enum {
    ME_ARTIFACT_CAP_NUMERIC = 1,
    ME_ARTIFACT_CAP_CONTROL_FLOW = 2,
    ME_ARTIFACT_CAP_BLOCK_REDUCTIONS = 4,
    ME_ARTIFACT_CAP_ND_CONTEXT = 8,
    ME_ARTIFACT_CAP_FIXED_STRINGS = 16
};

typedef struct me_artifact me_artifact;

typedef enum {
    ME_ARTIFACT_SUCCESS = 0,
    ME_ARTIFACT_ERR_FORMAT = -1,
    ME_ARTIFACT_ERR_UNSUPPORTED = -2,
    ME_ARTIFACT_ERR_SOURCE = -3,
    ME_ARTIFACT_ERR_BINDING = -4,
    ME_ARTIFACT_ERR_EVAL = -5,
    ME_ARTIFACT_ERR_OOM = -6
} me_artifact_status;

typedef struct {
    int native_status; /* Native validation/compile/eval status where applicable. */
    int line;          /* Source location when available; otherwise 0. */
    int column;
    char message[256];
} me_artifact_error;

typedef struct {
    const char *name;
    me_dtype dtype;
    const void *data; /* Contiguous host-native, naturally aligned logical values. */
    size_t nitems;
} me_artifact_input;

typedef enum {
    ME_ARTIFACT_CARDINALITY_INVALID = -1,
    ME_ARTIFACT_ELEMENTWISE = 0,
    ME_ARTIFACT_BLOCK_SCALAR = 1
} me_artifact_cardinality;

/* Extended ABI, independent of schema/language version. Capacities are bytes;
 * itemsize is immutable signature metadata, including fixed-string widths. */
typedef struct {
    const char *name;
    me_dtype dtype;
    size_t itemsize;
    const void *data;
    size_t capacity;
} me_artifact_buffer;

typedef struct {
    size_t struct_size;
    unsigned int version;
    size_t nitems;
    size_t output_capacity;
    const uint8_t *valid_mask;
    size_t valid_mask_capacity;
    int ndim;
    const int64_t *logical_shape;
    const int64_t *block_origin;
    const int64_t *block_extent;
} me_artifact_eval_descriptor;

/* Decode, validate, and compile without executing. The handle owns a copy of
 * source, names, and constants; JSON storage can be released on return. jit_mode
 * is a host preference, not an artifact requirement. No sandbox is provided.
 * Schema/language pairs 1.0/1.0 and opt-in 1.1/1.1 are accepted.
 * On any failure *out is NULL. error may be NULL. */
me_artifact_status me_artifact_load(const char *json, size_t length, me_jit_mode jit_mode,
                                  me_artifact **out, me_artifact_error *error);

/* Bind inputs by name, not caller order. All inputs and the output have nitems
 * elements. Host owns buffers and must allocate their declared lengths/dtypes;
 * output must not overlap inputs; adapters handle endian/stride conversions.
 * For nitems == 0 data/output may be
 * NULL; bindings are still validated, but kernel execution is skipped.
 * This numeric elementwise adapter requires rank zero; other contracts use eval_ex.
 * Evaluation does not mutate the handle; independent buffers may be used from
 * multiple threads. Output after a failure is unspecified. */
me_artifact_status me_artifact_eval(const me_artifact *artifact,
    const me_artifact_input *inputs, int ninputs, void *output, size_t nitems,
    me_artifact_error *error);

/* Checked capacities/alignment/overlap before execution. Draft 1.0 validates
 * signatures/capabilities on load and uses the typed interpreter, never full-DSL JIT. */
me_artifact_status me_artifact_eval_ex(const me_artifact *artifact,
    const me_artifact_buffer *inputs, int ninputs, void *output,
    const me_artifact_eval_descriptor *descriptor, me_artifact_error *error);

#define ME_ARTIFACT_FP_STATUS_VERSION 1
#define ME_ARTIFACT_ARRAY_VERSION 1
#define ME_ARRAY_MAX_RANK 16
/* Bounds describe the entire accessible allocation, not only the logical data
 * pointer. Offset/strides are bytes; byte_order: 0 native, 1 little, 2 big.
 * Caller owns all buffers; output is disjoint, aligned C-order storage. */
typedef struct {
    const char *name;
    me_dtype dtype;
    const void *base;
    size_t capacity;
    size_t offset;
    int rank;
    int64_t shape[ME_ARRAY_MAX_RANK];
    int64_t strides[ME_ARRAY_MAX_RANK];
    unsigned byte_order;
} me_array_view;
typedef enum { ME_ARRAY_NONE, ME_ARRAY_SUM, ME_ARRAY_PROD, ME_ARRAY_MIN,
               ME_ARRAY_MAX, ME_ARRAY_ANY, ME_ARRAY_ALL } me_array_reduction;
typedef struct {
    unsigned version;
    me_array_reduction reduction;
    int naxes; /* -1 selects every axis; 0 selects no axes. */
    int axes[ME_ARRAY_MAX_RANK]; /* Negative axes normalized natively. */
    bool keepdims;
    me_dtype accumulator; /* AUTO uses int64/uint64 widening for sum/prod. */
    const void *initial; /* One host-native accumulator value; NULL = default. */
    size_t tile_items; /* 0 = 1024. Storage chunking does not affect grouping. */
    const me_array_view *where; /* Optional broadcast Boolean participating mask. */
} me_array_options;
typedef struct {
    size_t temporary_bytes; /* Peak iterator scratch, excluding output/DSL internals. */
    size_t gathered_bytes; /* Cumulative copied input/mask bytes. */
    size_t zero_copy_tiles;
    size_t evaluated_tiles;
    unsigned fp_flags;
    unsigned fp_supported;
} me_array_report;
me_artifact_status me_array_result_shape(const me_artifact *artifact, int rank,
    const int64_t *shape, const me_array_options *options, int *output_rank,
    int64_t *output_shape, me_dtype *output_dtype, me_artifact_error *error);
me_artifact_status me_artifact_eval_array(const me_artifact *artifact,
    const me_array_view *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *options, void *output, size_t output_capacity,
    me_array_report *report, me_artifact_error *error);
/* Metadata-only borrowed views. No mutation/view-alias guarantees for results.
 * Reshape requires C contiguous input. Slice uses normalized start/count/step. */
me_artifact_status me_array_reshape(const me_array_view *input, int rank,
    const int64_t *shape, me_array_view *output, me_artifact_error *error);
me_artifact_status me_array_transpose(const me_array_view *input, const int *axes,
    me_array_view *output, me_artifact_error *error);
me_artifact_status me_array_slice(const me_array_view *input, int axis, int64_t start,
    int64_t count, int64_t step, me_array_view *output, me_artifact_error *error);
enum { ME_FP_INVALID = 1u, ME_FP_DIVIDE = 2u, ME_FP_OVERFLOW = 4u, ME_FP_UNDERFLOW = 8u };
typedef struct {
    unsigned flags; /* OR of active evaluated operations; cleared per call. */
    unsigned supported; /* 1 if host IEEE flags are observable, 0 on WASM. */
} me_artifact_fp_status;
/* Per-call policy, never mutable handle/global state. raise_mask selects flags
 * that turn a completed evaluation into ME_ARTIFACT_ERR_EVAL. Outputs on failure
 * are unspecified. Collection restores caller flags/rounding and is thread-safe.
 * On WASM reporting is unavailable: ignore computes values; raise rejects before
 * execution. Cross-block callers aggregate flags explicitly by OR. */
me_artifact_status me_artifact_eval_status(const me_artifact *artifact,
    const me_artifact_buffer *inputs, int ninputs, void *output,
    const me_artifact_eval_descriptor *descriptor, unsigned raise_mask,
    me_artifact_fp_status *status, me_artifact_error *error);
me_artifact_cardinality me_artifact_result_cardinality(const me_artifact *artifact);
size_t me_artifact_input_itemsize(const me_artifact *artifact, int index);
size_t me_artifact_output_itemsize(const me_artifact *artifact);
unsigned int me_artifact_capabilities(const me_artifact *artifact);
int me_artifact_context_ndim(const me_artifact *artifact);
const char *me_artifact_schema_version(const me_artifact *artifact);

/* Borrowed strings remain valid until free; invalid query indices return NULL
 * or ME_AUTO. Invalid handles return NULL/0/ME_AUTO as appropriate. */
const char *me_artifact_source(const me_artifact *artifact);
const char *me_artifact_entry_point(const me_artifact *artifact);
int me_artifact_ninputs(const me_artifact *artifact);
const char *me_artifact_input_name(const me_artifact *artifact, int index);
me_dtype me_artifact_input_dtype(const me_artifact *artifact, int index);
me_dtype me_artifact_output_dtype(const me_artifact *artifact);
/* Metadata-only result dtype before final output conversion (1.1 artifacts).
 * Returns ME_AUTO for legacy artifacts and NULL handles. */
me_dtype me_artifact_inferred_dtype(const me_artifact *artifact);
bool me_artifact_has_jit(const me_artifact *artifact);
void me_artifact_free(me_artifact *artifact); /* NULL-safe. */

#ifdef __cplusplus
}
#endif
#endif
