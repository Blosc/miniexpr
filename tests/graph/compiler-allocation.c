/* Instrument every malloc/calloc/realloc in miniexpr, graph, artifact, iterator
 * and yyjson translation units. System and dynamically loaded JIT allocators are
 * deliberately outside this boundary. Failures persist for the whole operation. */
#undef malloc
#undef calloc
#undef realloc
#undef free
#include "miniexpr_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static long fail_at = -1, calls;
static bool fail(void) {
    return fail_at >= 0 ? calls++ >= fail_at : (calls++, false);
}
void *graph_test_malloc(size_t n) { return fail() ? NULL : malloc(n); }
void *graph_test_calloc(size_t n, size_t width) { return fail() ? NULL : calloc(n, width); }
void *graph_test_realloc(void *p, size_t n) { return fail() ? NULL : realloc(p, n); }
void graph_test_free(void *p) { free(p); }
static me_graph_error error;
static int check_result(me_graph_plan *p, const float *expected) {
    me_graph_input_metadata metadata[] = {{"x", ME_FLOAT32, 1, {3}}, {"y", ME_FLOAT32, 1, {3}}};
    me_graph_schedule *s = NULL;
    if (me_graph_specialize(p, metadata, 2, NULL, &s, &error)) return 1;
    float x[] = {1, 4, 9}, y[] = {1, 2, 3}, output[3] = {0};
    me_array_view inputs[2] = {0};
    for (int i = 0; i < 2; i++) {
        inputs[i].name = metadata[i].name; inputs[i].dtype = ME_FLOAT32;
        inputs[i].rank = 1; inputs[i].shape[0] = 3; inputs[i].strides[0] = sizeof(float);
        inputs[i].base = i ? y : x; inputs[i].capacity = sizeof(x);
    }
    int rc = me_graph_execute(s, inputs, 2, output, sizeof(output), NULL, NULL, &error);
    me_graph_schedule_free(s);
    return rc || memcmp(output, expected, sizeof(output));
}
static int check(const char *text, const float *expected, me_jit_mode jit) {
    me_graph_input_metadata metadata[] = {{"x", ME_FLOAT32, 0, {0}}, {"y", ME_FLOAT32, 0, {0}}};
    me_graph_prepare_options options = {sizeof(options), ME_GRAPH_VERSION, jit, false, true};
    me_graph_plan *p = NULL;
    calls = 0;
    int rc = me_graph_prepare_expression(text, strlen(text), metadata, 2, &options, &p, &error);
    long checkpoints = calls;
    if (rc || !p || check_result(p, expected)) return 1;
    me_graph_plan_free(p);
    int rejected = 0, recovered = 0;
    for (long i = 0; i < checkpoints; i++) {
        p = (void *)1; calls = 0; fail_at = i;
        rc = me_graph_prepare_expression(text, strlen(text), metadata, 2, &options, &p, &error);
        fail_at = -1;
        if (rc) {
            if (p || !error.native.message[0]) {
                fprintf(stderr, "%s checkpoint %ld: invalid failure ownership/diagnostic rc=%d plan=%p message=%s\n", text, i, rc, (void *)p, error.native.message);
                return 1;
            }
            rejected++;
        } else {
            if (!p || check_result(p, expected)) {
                fprintf(stderr, "%s checkpoint %ld: invalid successful fallback\n", text, i);
                return 1;
            }
            recovered++;
            me_graph_plan_free(p);
        }
        /* Each failure must be recoverable, not poison a later preparation. */
        p = NULL;
        rc = me_graph_prepare_expression(text, strlen(text), metadata, 2, &options, &p, &error);
        if (rc || !p || check_result(p, expected)) return 1;
        me_graph_plan_free(p);
    }
    printf("compiler graph jit=%d checkpoints=%ld rejected=%d optional-recovery=%d\n",
        jit, checkpoints, rejected, recovered);
    return !checkpoints || !rejected;
}
int main(void) {
    const float affine[] = {3, 10, 21}, staged[] = {7, 8, 9};
    return check("x * 2 + y", affine, ME_JIT_OFF) ||
        check("sqrt(x) + sum(y)", staged, ME_JIT_OFF) ||
        check("x * 2 + y", affine, ME_JIT_ON);
}
