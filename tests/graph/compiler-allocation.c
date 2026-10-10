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
/* A simple scalar return must execute even when every compiler-owned allocation
 * fails. Explicit masks/empty groups/locals/elementwise returns retain their
 * bookkeeping, and must still recover after an injected allocation failure. */
static int scalar_dispatch(const char *body, const char *output_dtype,
    const char *contract, int count, const uint8_t *mask, bool direct, double expected) {
    char json[2048];
    snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\",\"block-reductions\"],\"source\":\"def k(x):\\n    %s\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"float64\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"%s\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        body, output_dtype, contract);
    me_artifact *a = NULL; me_artifact_error native;
    if (me_artifact_load(json, strlen(json), ME_JIT_OFF, &a, &native)) return 1;
    double x[] = {1, 2, 3}, output[3] = {0};
    me_artifact_buffer buffer = {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)};
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = (size_t)count,
        .output_capacity = sizeof(output), .valid_mask = mask,
        .valid_mask_capacity = mask ? (size_t)count : 0};
    calls = 0; fail_at = 0;
    int rc = me_artifact_eval_ex(a, &buffer, 1, output, &descriptor, &native);
    fail_at = -1;
    if ((direct && (rc || calls)) || (!direct && (!rc || !calls))) {
        fprintf(stderr, "scalar dispatch %s direct=%d rc=%d allocations=%ld\n", body, direct, rc, calls);
        me_artifact_free(a); return 1;
    }
    memset(output, 0, sizeof(output));
    rc = me_artifact_eval_ex(a, &buffer, 1, output, &descriptor, &native);
    double actual = output[0];
    if (!strcmp(output_dtype, "float32")) { float value; memcpy(&value, output, sizeof(value)); actual = value; }
    if (rc || actual != expected) { me_artifact_free(a); return 1; }
    /* The shortcut must not evade descriptor/buffer validation. */
    int invalid = 0;
    for (int test = 0; test < 5; test++) {
        me_artifact_buffer bad = buffer;
        me_artifact_eval_descriptor invalid_descriptor = descriptor;
        uint8_t invalid_mask[] = {2, 1, 1};
        if (test == 0) invalid_descriptor.output_capacity = 0;
        else if (test == 1) bad.name = "missing";
        else if (test == 2) bad.dtype = ME_INT64;
        else if (test == 3 && count) bad.capacity = (size_t)count * sizeof(double) - 1;
        else if (test == 4 && count) {
            invalid_descriptor.valid_mask = invalid_mask;
            invalid_descriptor.valid_mask_capacity = sizeof(invalid_mask);
        } else continue;
        calls = 0; fail_at = 0;
        rc = me_artifact_eval_ex(a, &bad, 1, output, &invalid_descriptor, &native);
        fail_at = -1;
        invalid |= rc != ME_ARTIFACT_ERR_BINDING || calls != 0;
    }
    me_artifact_free(a);
    return invalid;
}
static int scalar_dispatches(void) {
    uint8_t full[] = {1, 1, 1}, partial[] = {0, 1, 1}, none[] = {0, 0, 0};
    return scalar_dispatch("return sum(x)", "float64", "block_scalar", 3, NULL, true, 6) ||
        scalar_dispatch("return sum(x * 2)", "float64", "block_scalar", 3, NULL, true, 12) ||
        scalar_dispatch("return sum(x) + sum(x)", "float64", "block_scalar", 3, NULL, true, 12) ||
        scalar_dispatch("return sum(x) / 3", "float64", "block_scalar", 3, NULL, true, 2) ||
        scalar_dispatch("return sum(x)", "float32", "block_scalar", 3, NULL, true, 6) ||
        scalar_dispatch("s = sum(x)\\n    return s", "float64", "block_scalar", 3, NULL, false, 6) ||
        scalar_dispatch("return x * 2", "float64", "elementwise", 3, NULL, false, 2) ||
        scalar_dispatch("return sum(x)", "float64", "block_scalar", 3, full, false, 6) ||
        scalar_dispatch("return sum(x)", "float64", "block_scalar", 3, partial, false, 5) ||
        scalar_dispatch("return sum(x)", "float64", "block_scalar", 3, none, false, 0) ||
        scalar_dispatch("return sum(x)", "float64", "block_scalar", 0, NULL, false, 0);
}
static int weak_conversion_result(me_graph_plan *p) {
    me_graph_input_metadata input = {"x", ME_INT32, 1, {3}};
    me_graph_schedule *s = NULL;
    if (me_graph_specialize(p, &input, 1, NULL, &s, &error)) return 1;
    int32_t x[] = {-11, 4, 19}, output[3], expected[] = {3, 4, 5};
    me_array_view v = {.name = "x", .dtype = ME_INT32, .base = x,
        .capacity = sizeof(x), .rank = 1, .shape = {3}, .strides = {sizeof(int32_t)}};
    int rc = me_graph_execute(s, &v, 1, output, sizeof(output), NULL, NULL, &error);
    me_graph_schedule_free(s);
    return rc || memcmp(output, expected, sizeof(output));
}
static int weak_conversion_allocations(void) {
    me_graph_input_metadata input = {"x", ME_INT32, 1, {3}};
    me_graph_prepare_options options = {sizeof(options), ME_GRAPH_VERSION, ME_JIT_ON, false, true};
    me_graph_plan *p = NULL;
    calls = 0;
    int rc = me_graph_prepare_expression("x % 7", 5, &input, 1, &options, &p, &error);
    long checkpoints = calls;
    if (rc || !p || weak_conversion_result(p)) return 1;
    me_graph_plan_free(p);
    for (long i = 0; i < checkpoints; i++) {
        p = (void *)1; calls = 0; fail_at = i;
        rc = me_graph_prepare_expression("x % 7", 5, &input, 1, &options, &p, &error);
        fail_at = -1;
        if (rc ? p != NULL || !error.native.message[0] : !p || weak_conversion_result(p)) return 1;
        me_graph_plan_free(p);
        p = NULL;
        rc = me_graph_prepare_expression("x % 7", 5, &input, 1, &options, &p, &error);
        if (rc || !p || weak_conversion_result(p)) return 1;
        me_graph_plan_free(p);
    }
    printf("weak conversion compiler checkpoints=%ld\n", checkpoints);
    return !checkpoints;
}
int main(void) {
    const float affine[] = {3, 10, 21}, staged[] = {7, 8, 9};
    return scalar_dispatches() || weak_conversion_allocations() || check("x * 2 + y", affine, ME_JIT_OFF) ||
        check("sqrt(x) + sum(y)", staged, ME_JIT_OFF) ||
        check("x * 2 + y", affine, ME_JIT_ON);
}
