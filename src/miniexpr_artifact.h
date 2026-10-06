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

#define ME_ARTIFACT_SCHEMA_VERSION "0.1"
#define ME_ARTIFACT_MAX_BYTES (1024 * 1024)

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

/* Decode, validate, and compile without executing. The handle owns a copy of
 * source, names, and constants; JSON storage can be released on return. jit_mode
 * is a host preference, not an artifact requirement. No sandbox is provided.
 * Only schema/language 0.1 is accepted; see doc/dsl-spec/artifact-0.1.md.
 * On any failure *out is NULL. error may be NULL. */
me_artifact_status me_artifact_load(const char *json, size_t length, me_jit_mode jit_mode,
                                  me_artifact **out, me_artifact_error *error);

/* Bind inputs by name, not caller order. All inputs and the output have nitems
 * elements. Host owns buffers and must allocate their declared lengths/dtypes;
 * output must not overlap inputs; adapters handle endian/stride conversions.
 * For nitems == 0 data/output may be
 * NULL; bindings are still validated, but kernel execution is skipped.
 * Constants broadcast using bounded per-call workspace, not array-sized copies.
 * Evaluation does not mutate the handle; independent buffers may be used from
 * multiple threads. Output after a failure is unspecified. */
me_artifact_status me_artifact_eval(const me_artifact *artifact,
    const me_artifact_input *inputs, int ninputs, void *output, size_t nitems,
    me_artifact_error *error);

/* Borrowed strings remain valid until free; invalid query indices return NULL
 * or ME_AUTO. Invalid handles return NULL/0/ME_AUTO as appropriate. */
const char *me_artifact_source(const me_artifact *artifact);
const char *me_artifact_entry_point(const me_artifact *artifact);
int me_artifact_ninputs(const me_artifact *artifact);
const char *me_artifact_input_name(const me_artifact *artifact, int index);
me_dtype me_artifact_input_dtype(const me_artifact *artifact, int index);
me_dtype me_artifact_output_dtype(const me_artifact *artifact);
bool me_artifact_has_jit(const me_artifact *artifact);
void me_artifact_free(me_artifact *artifact); /* NULL-safe. */

#ifdef __cplusplus
}
#endif
#endif
