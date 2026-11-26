#ifndef MINIEXPR_GRAPH_INTERNAL_H
#define MINIEXPR_GRAPH_INTERNAL_H
#include "miniexpr_artifact.h"
#include "dsl_graph_plan.h"
#include "yyjson.h"

me_graph_status dsl_graph_prepare_nodes(graph_node *nodes, int count, int result, bool staged,
    yyjson_val *interchange, const me_graph_prepare_options *options, me_graph_plan **out,
    me_graph_error *error);
me_artifact_status dsl_array_output_validate(const me_array_output *output,
    int rank, const int64_t *shape, me_dtype dtype, me_artifact_error *error);
me_artifact_status dsl_array_run_into(const me_artifact *artifact,
    const me_array_view *inputs, const me_array_source *sources, int ninputs,
    int rank, const int64_t *shape, const me_array_options *options,
    const me_array_output *output, me_array_report *report, me_artifact_error *error,
    bool execute, bool identity_map);
/* Private graph lowering boundary. Only this entry permits inferred output;
 * public artifact schemas and validation remain unchanged. */
me_artifact_status dsl_graph_load_map(const char *json, size_t length,
    me_jit_mode jit, me_artifact **out, me_artifact_error *error);
me_artifact_status dsl_graph_load_map_root(yyjson_val *root,
    me_jit_mode jit, me_artifact **out, me_artifact_error *error);
me_artifact_status dsl_graph_scalar(yyjson_val *root, me_dtype *dtype, void *value);
me_artifact_status dsl_array_preflight(const me_artifact *artifact,
    const me_array_view *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *options, void *output, size_t capacity, me_artifact_error *error);
/* Only a validated graph whose map root is a plain input may use this entry.
 * Maps can use a direct copy; reductions can bypass evaluation of identity tiles. */
me_artifact_status dsl_array_eval_direct_input(const me_artifact *artifact,
    const me_array_view *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *options, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error);
me_artifact_status dsl_array_validate_domain(const me_artifact *artifact, int rank,
    const int64_t *shape, const me_array_options *options, bool participating,
    me_artifact_error *error);
me_artifact_status dsl_array_sources_preflight(const me_artifact *artifact,
    const me_array_source *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *options, void *output, size_t capacity, me_artifact_error *error);
me_artifact_status dsl_array_eval_direct_sources(const me_artifact *artifact,
    const me_array_source *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *options, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error);
#endif
