/* Test-only failure injection for graph-owned metadata buffers. Portable compiler
 * and yyjson allocator failure qualification remains a separate requirement. */
#undef malloc
#undef calloc
#undef realloc
#undef free
#include "miniexpr_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int remaining = -1;
static bool fail(void) {
    if (remaining < 0) return false;
    if (!remaining) return true;
    remaining--;
    return false;
}
void *graph_test_malloc(size_t n) { return fail() ? NULL : malloc(n); }
void *graph_test_calloc(size_t n, size_t width) { return fail() ? NULL : calloc(n, width); }
void *graph_test_realloc(void *p, size_t n) { return fail() ? NULL : realloc(p, n); }
void graph_test_free(void *p) { free(p); }
int main(void) {
    const char *json = "{\"format\":\"menudet-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\"],"
        "\"nodes\":[{\"id\":0,\"op\":\"input\",\"name\":\"x\",\"dtype\":\"float32\"}],\"root\":0,"
        "\"output\":{\"dtype\":\"auto\",\"casting\":\"unsafe\"}}";
    me_graph_error error;
    int failures = 0;
    for (int i = 0; i < 100; i++) {
        remaining = i; me_graph_plan *p = (void *)1;
        int rc = me_graph_prepare_json(json, strlen(json), NULL, &p, &error);
        remaining = -1;
        if (!rc) {
            me_graph_input_metadata input = {"x", ME_FLOAT32, 1, {3}};
            me_graph_schedule *s = (void *)1; remaining = 0;
            rc = me_graph_specialize(p, &input, 1, NULL, &s, &error); remaining = -1;
            if (rc != ME_GRAPH_ERR_OOM || s) return 1;
            if (me_graph_specialize(p, &input, 1, NULL, &s, &error)) return 1;
            me_graph_schedule_free(s); me_graph_plan_free(p); break;
        }
        if (rc != ME_GRAPH_ERR_OOM || p) { fprintf(stderr, "allocation checkpoint %d: %d %s\n", i, rc, error.native.message); return 1; }
        failures++;
    }
    printf("graph metadata allocation failures checked=%d; recovery passed\n", failures);
    return !failures;
}
