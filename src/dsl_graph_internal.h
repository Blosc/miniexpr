#ifndef MINIEXPR_GRAPH_INTERNAL_H
#define MINIEXPR_GRAPH_INTERNAL_H
#include "miniexpr_artifact.h"
/* Private graph lowering boundary. Only this entry permits inferred output;
 * public artifact schemas and validation remain unchanged. */
me_artifact_status dsl_graph_load_map(const char *json, size_t length,
    me_jit_mode jit, me_artifact **out, me_artifact_error *error);
me_artifact_status dsl_graph_scalar(const char *json, size_t length, me_dtype *dtype, void *value);
me_artifact_status dsl_array_preflight(const me_artifact *artifact,
    const me_array_view *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *options, void *output, size_t capacity, me_artifact_error *error);
/* Only a validated graph whose map root is a plain input may use this entry. */
me_artifact_status dsl_array_eval_direct_input(const me_artifact *artifact,
    const me_array_view *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *options, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error);
me_artifact_status dsl_array_validate_domain(const me_artifact *artifact, int rank,
    const int64_t *shape, const me_array_options *options, bool participating,
    me_artifact_error *error);
#endif
