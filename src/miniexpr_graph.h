/* Opt-in numerical graphs; independent of portable artifact schema versions. */
#ifndef MINIEXPR_GRAPH_H
#define MINIEXPR_GRAPH_H
#include "miniexpr_artifact.h"
#ifdef __cplusplus
extern "C" {
#endif
#define ME_GRAPH_VERSION 1
#define ME_GRAPH_FORMAT "menudet-graph-1"
#define ME_GRAPH_SEMANTICS "menudet-numpy-1.1"
#define ME_GRAPH_MAX_BYTES (1024 * 1024)
#define ME_GRAPH_MAX_NODES 256
#define ME_GRAPH_MAX_DEPTH 64
typedef struct me_graph_plan me_graph_plan;
typedef struct me_graph_schedule me_graph_schedule;
typedef enum {
    ME_GRAPH_SUCCESS = 0, ME_GRAPH_ERR_FORMAT = -1,
    ME_GRAPH_ERR_UNSUPPORTED = -2, ME_GRAPH_ERR_SIGNATURE = -3,
    ME_GRAPH_ERR_SHAPE = -4, ME_GRAPH_ERR_BINDING = -5,
    ME_GRAPH_ERR_EXECUTION = -6, ME_GRAPH_ERR_OOM = -7,
    ME_GRAPH_ERR_CAPABILITY = -8
} me_graph_status;
typedef struct {
    int node; /* -1 when not associated with a node. */
    int stage;
    me_artifact_error native;
} me_graph_error;
typedef struct {
    size_t struct_size;
    unsigned version;
    me_jit_mode jit;
    bool require_jit;
    bool disable_optimization; /* Currently all graph rewrites are disabled. */
} me_graph_prepare_options;
/* Pointer-free, strong inputs. Names/dtypes must exactly cover the plan. */
typedef struct {
    const char *name;
    me_dtype dtype;
    int rank;
    int64_t shape[ME_ARRAY_MAX_RANK];
} me_graph_input_metadata;
typedef struct {
    size_t struct_size;
    unsigned version;
    size_t tile_items; /* 0 = 1024; bounded by INT32_MAX. */
    size_t intermediate_budget; /* 0 = 64 MiB for a requested final conversion. */
} me_graph_specialize_options;
typedef struct {
    size_t struct_size;
    unsigned version;
    unsigned raise_mask;
} me_graph_execute_options;
typedef struct {
    me_array_report array;
    bool has_jit; /* Actual map route, not requested preference. */
    size_t stages;
} me_graph_report;
/* NULL options select interpretation/default geometry. Errors may be NULL.
 * Preparation copies graph/captures, reads no arrays and evaluates no constants.
 * Failure sets *out=NULL. There is no Python or full-DSL fallback. */
me_graph_status me_graph_prepare_json(const char *json, size_t length,
    const me_graph_prepare_options *options, me_graph_plan **out, me_graph_error *error);
/* Restricted text frontend. Signatures supply input names/types (shapes ignored).
 * Captures can be expressed in JSON; text literals are weak scalars. */
me_graph_status me_graph_prepare_expression(const char *expression, size_t length,
    const me_graph_input_metadata *inputs, int ninputs,
    const me_graph_prepare_options *options, me_graph_plan **out, me_graph_error *error);
me_graph_status me_graph_specialize(const me_graph_plan *plan,
    const me_graph_input_metadata *inputs, int ninputs,
    const me_graph_specialize_options *options, me_graph_schedule **out, me_graph_error *error);
me_graph_status me_graph_execute(const me_graph_schedule *schedule,
    const me_array_view *inputs, int ninputs, void *output, size_t capacity,
    const me_graph_execute_options *options, me_graph_report *report, me_graph_error *error);
int me_graph_ninputs(const me_graph_plan *plan);
const char *me_graph_input_name(const me_graph_plan *plan, int index);
me_dtype me_graph_input_dtype(const me_graph_plan *plan, int index);
me_dtype me_graph_inferred_dtype(const me_graph_plan *plan);
bool me_graph_has_jit(const me_graph_plan *plan);
unsigned me_graph_capabilities(const me_graph_plan *plan);
/* Borrowed canonical declarative JSON; valid until plan free. Import is prepare.
 * No machine resources, input arrays, pointers or cached results are exported. */
const char *me_graph_export_json(const me_graph_plan *plan, size_t *length);
const char *me_graph_export_map_json(const me_graph_plan *plan); /* NULL for reductions/final conversions. */
const me_artifact *me_graph_map_artifact(const me_graph_plan *plan); /* Borrowed; deployment adapter. */
int me_graph_output_rank(const me_graph_schedule *schedule);
const int64_t *me_graph_output_shape(const me_graph_schedule *schedule);
const int64_t *me_graph_map_shape(const me_graph_schedule *schedule, int *rank);
me_dtype me_graph_output_dtype(const me_graph_schedule *schedule);
size_t me_graph_output_bytes(const me_graph_schedule *schedule);
size_t me_graph_scratch_bytes(const me_graph_schedule *schedule); /* Iterator only, not compiler/interpreter. */
size_t me_graph_intermediate_bytes(const me_graph_schedule *schedule);
size_t me_graph_stage_count(const me_graph_plan *plan);
size_t me_graph_plan_bytes(const me_graph_plan *plan); /* Owned graph metadata, excluding compiler/artifact resources. */
size_t me_graph_schedule_bytes(const me_graph_schedule *schedule);
/* Schedules retain plans. Immutable shared handles support independent calls;
 * freeing a handle concurrently with its use is unsupported. */
void me_graph_plan_free(me_graph_plan *plan);
void me_graph_schedule_free(me_graph_schedule *schedule);
#ifdef __cplusplus
}
#endif
#endif
