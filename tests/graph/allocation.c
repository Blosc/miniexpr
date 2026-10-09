/* Test-only failure injection for graph metadata/materializations and yyjson
 * decode/copy/export. Compiler allocator qualification is separate. */
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
static int check(const char *json) {
    me_graph_error error;
    int failures = 0;
    bool prepared = false;
    for (int i = 0; i < 1000; i++) {
        remaining = i; me_graph_plan *p = (void *)1;
        int rc = me_graph_prepare_json(json, strlen(json), NULL, &p, &error);
        remaining = -1;
        if (!rc) {
            me_graph_input_metadata input = {"x", ME_FLOAT32, 1, {3}};
            me_graph_schedule *s = NULL;
            bool specialized = false;
            for (int j = 0; j < 100; j++) {
                remaining = j; s = (void *)1;
                rc = me_graph_specialize(p, &input, 1, NULL, &s, &error); remaining = -1;
                if (!rc) { specialized = true; break; }
                if (rc != ME_GRAPH_ERR_OOM || s) return 1;
                failures++;
            }
            if (!specialized) return 1;
            float x[] = {1, 2, 3}; double out[3] = {99, 99, 99};
            me_array_view v = {0};
            v.name = "x"; v.dtype = ME_FLOAT32; v.rank = 1; v.shape[0] = 3;
            v.strides[0] = sizeof(float); v.base = x; v.capacity = sizeof(x);
            bool executed = false;
            for (int j = 0; j < 100; j++) {
                remaining = j;
                rc = me_graph_execute(s, &v, 1, out, sizeof(out), NULL, NULL, &error); remaining = -1;
                if (!rc) { executed = true; break; }
                if (rc != ME_GRAPH_ERR_OOM || out[0] != 99) return 1;
                failures++;
            }
            if (!executed) return 1;
            me_graph_schedule_free(s); me_graph_plan_free(p); prepared = true; break;
        }
        if (rc != ME_GRAPH_ERR_OOM || p) { fprintf(stderr, "allocation checkpoint %d: %d %s\n", i, rc, error.native.message); return 1; }
        failures++;
    }
    printf("graph metadata allocation failures checked=%d; recovery passed\n", failures);
    return !prepared || !failures;
}
int main(void) {
    const char *map = "{\"format\":\"menudet-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\"],"
        "\"nodes\":[{\"id\":0,\"op\":\"input\",\"name\":\"x\",\"dtype\":\"float32\"}],\"root\":0,"
        "\"output\":{\"dtype\":\"auto\",\"casting\":\"unsafe\"}}";
    const char *staged = "{\"format\":\"menudet-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\",\"staged\"],"
        "\"nodes\":[{\"id\":0,\"op\":\"input\",\"name\":\"x\",\"dtype\":\"float32\"},"
        "{\"id\":1,\"op\":\"sum\",\"args\":[0],\"axes\":null,\"keepdims\":false,\"dtype\":\"auto\",\"initial\":null,\"where\":null},"
        "{\"id\":2,\"op\":\"sub\",\"args\":[0,1]}],\"root\":2,\"output\":{\"dtype\":\"auto\",\"casting\":\"unsafe\"}}";
    const char *conversion = "{\"format\":\"menudet-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\"],"
        "\"nodes\":[{\"id\":0,\"op\":\"input\",\"name\":\"x\",\"dtype\":\"float32\"}],\"root\":0,"
        "\"output\":{\"dtype\":\"float64\",\"casting\":\"safe\"}}";
    const char *initial = "{\"format\":\"menudet-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\"],"
        "\"nodes\":[{\"id\":0,\"op\":\"input\",\"name\":\"x\",\"dtype\":\"float32\"},"
        "{\"id\":1,\"op\":\"sum\",\"args\":[0],\"axes\":null,\"keepdims\":false,\"dtype\":\"auto\","
        "\"initial\":{\"dtype\":\"int64\",\"category\":\"weak\",\"encoding\":\"decimal\",\"value\":\"2\"},\"where\":null}],"
        "\"root\":1,\"output\":{\"dtype\":\"auto\",\"casting\":\"unsafe\"}}";
    return check(map) || check(staged) || check(conversion) || check(initial);
}
