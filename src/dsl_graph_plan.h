#ifndef MINIEXPR_GRAPH_PLAN_H
#define MINIEXPR_GRAPH_PLAN_H
#include "miniexpr_graph.h"

typedef struct {
    me_array_options options;
    me_dtype initial_dtype;
    uint64_t initial;
    bool has_initial, initial_weak;
    const char *invalid;
} graph_reduction;

/* Private frontend descriptor. Strings are borrowed only during prepare. */
typedef struct {
    const char *op, *name, *power_literal;
    me_dtype dtype;
    uint64_t scalar;
    int args[3], nargs, uses, depth, mask;
    bool reachable, weak, scalar_valid;
    graph_reduction reduction;
} graph_node;

/* Prepare frontend-validated nodes, without constructing interchange exports.
 * `nodes` is temporary workspace; retained names and values are copied. */
me_graph_status dsl_graph_prepare_typed(graph_node *nodes, int count, int result, bool staged,
    const me_graph_prepare_options *options, me_graph_plan **out, me_graph_error *error);
#endif
