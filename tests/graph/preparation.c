#include "miniexpr_graph.h"
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, error.native.message); exit(1); } } while (0)
static me_graph_error error;
static me_graph_plan *prepare(const char *source, me_graph_input_metadata *inputs, int n, bool jit) {
    me_graph_plan *p = NULL;
    me_graph_prepare_options options = {sizeof(options), ME_GRAPH_VERSION, jit ? ME_JIT_ON : ME_JIT_OFF, false, true};
    CHECK(!me_graph_prepare_expression(source, strlen(source), inputs, n, &options, &p, &error));
    CHECK(p != NULL);
    return p;
}
static me_array_view view(const char *name, me_dtype dtype, void *base, size_t capacity, int rank, int64_t a, int64_t b) {
    me_array_view v = {0};
    v.name = name; v.dtype = dtype; v.base = base; v.capacity = capacity; v.rank = rank;
    v.shape[0] = a; v.shape[1] = b;
    size_t width = dtype == ME_BOOL ? 1 : dtype == ME_FLOAT32 ? 4 : 8;
    v.strides[1] = (int64_t)width; v.strides[0] = (int64_t)(rank == 2 ? b * width : width);
    return v;
}
static void maps(bool jit) {
    me_graph_input_metadata inputs[] = {{"x", ME_FLOAT32, 2, {2, 1}}, {"y", ME_FLOAT32, 1, {3}}};
    float x[] = {1, 2}, y[] = {3, 4, 5}, output[6] = {0};
    me_graph_plan *p = prepare("x * 2 + y", inputs, 2, jit), *copy = NULL;
    CHECK(me_graph_inferred_dtype(p) == ME_FLOAT32);
    size_t length; const char *json = me_graph_export_json(p, &length);
    CHECK(!me_graph_prepare_json(json, length, NULL, &copy, &error));
    CHECK(!strcmp(json, me_graph_export_json(copy, NULL)));
    me_graph_plan_free(copy);
    me_graph_schedule *s = NULL;
    CHECK(!me_graph_specialize(p, inputs, 2, NULL, &s, &error));
    CHECK(me_graph_output_rank(s) == 2 && me_graph_output_shape(s)[1] == 3);
    CHECK(me_graph_output_bytes(s) == sizeof(output));
    me_graph_plan_free(p); /* The schedule owns retention, not input pointers. */
    me_array_view bindings[] = {view("y", ME_FLOAT32, y, sizeof(y), 1, 3, 0), view("x", ME_FLOAT32, x, sizeof(x), 2, 2, 1)};
    me_graph_report report;
    CHECK(!me_graph_execute(s, bindings, 2, output, sizeof(output), NULL, &report, &error));
    CHECK(output[0] == 5 && output[5] == 9);
    x[0] = 10;
    CHECK(!me_graph_execute(s, bindings, 2, output, sizeof(output), NULL, &report, &error));
    CHECK(output[0] == 23); /* No result cache. */
    output[0] = 99; bindings[0].shape[0] = 2;
    CHECK(me_graph_execute(s, bindings, 2, output, sizeof(output), NULL, &report, &error) == ME_GRAPH_ERR_BINDING);
    CHECK(output[0] == 99); bindings[0].shape[0] = 3;
    CHECK(me_graph_execute(s, bindings, 2, output, 1, NULL, NULL, &error) == ME_GRAPH_ERR_BINDING);
    CHECK(output[0] == 99);
    CHECK(me_graph_execute(s, bindings, 2, y, sizeof(y), NULL, NULL, &error) == ME_GRAPH_ERR_BINDING);
    me_graph_schedule_free(s);
}
static void lazy(bool jit) {
    me_graph_input_metadata inputs[] = {{"x", ME_FLOAT64, 1, {3}}, {"y", ME_FLOAT64, 0, {0}}};
    double x[] = {0, 2, 4}, y = 8, output[3];
    me_graph_plan *p = prepare("where(x != 0, y / x, y)", inputs, 2, jit);
    me_graph_schedule *s = NULL; CHECK(!me_graph_specialize(p, inputs, 2, NULL, &s, &error));
    me_array_view bindings[] = {view("x", ME_FLOAT64, x, sizeof(x), 1, 3, 0), view("y", ME_FLOAT64, &y, sizeof(y), 0, 0, 0)};
    me_graph_report report;
    CHECK(!me_graph_execute(s, bindings, 2, output, sizeof(output), NULL, &report, &error));
    CHECK(output[0] == 8 && output[1] == 4 && output[2] == 2 && report.array.fp_flags == 0);
    me_graph_schedule_free(s); me_graph_plan_free(p);
}
static void reductions(bool jit) {
    me_graph_input_metadata inputs[] = {{"x", ME_FLOAT64, 2, {2, 3}}, {"mask", ME_BOOL, 1, {3}}};
    double x[] = {1, -1, 4, 9, -2, 16}, output[2]; bool mask[] = {true, false, true};
    me_graph_plan *p = prepare("sum(sqrt(x), axis=-1, keepdims=True, where=mask)", inputs, 2, jit);
    me_graph_schedule *s = NULL;
    me_graph_specialize_options options = {sizeof(options), ME_GRAPH_VERSION, 1, 0};
    CHECK(!me_graph_specialize(p, inputs, 2, &options, &s, &error));
    CHECK(me_graph_output_rank(s) == 2 && me_graph_output_shape(s)[1] == 1);
    me_array_view bindings[] = {view("x", ME_FLOAT64, x, sizeof(x), 2, 2, 3), view("mask", ME_BOOL, mask, sizeof(mask), 1, 3, 0)};
    me_graph_report report;
    CHECK(!me_graph_execute(s, bindings, 2, output, sizeof(output), NULL, &report, &error));
    CHECK(output[0] == 3 && output[1] == 7 && report.array.fp_flags == 0);
    options.tile_items = 4; me_graph_schedule *other = NULL;
    CHECK(!me_graph_specialize(p, inputs, 2, &options, &other, &error));
    double second[2];
    CHECK(!me_graph_execute(other, bindings, 2, second, sizeof(second), NULL, &report, &error));
    CHECK(!memcmp(output, second, sizeof(output)));
    me_graph_schedule_free(s); me_graph_schedule_free(other); me_graph_plan_free(p);
    const char *ops[] = {"sum(x)", "prod(x)", "min(x)", "max(x)", "any(x)", "all(x)"};
    inputs[0].dtype = ME_INT64; inputs[0].rank = 1; inputs[0].shape[0] = 3;
    int64_t integers[] = {2, 3, 4}, expected[] = {9, 24, 2, 4, 1, 1};
    for (int i = 0; i < 6; i++) {
        p = prepare(ops[i], inputs, 1, jit);
        CHECK(!me_graph_specialize(p, inputs, 1, NULL, &s, &error));
        uint64_t result = 0; bindings[0] = view("x", ME_INT64, integers, sizeof(integers), 1, 3, 0);
        CHECK(!me_graph_execute(s, bindings, 1, &result, sizeof(result), NULL, &report, &error));
        CHECK((int64_t)result == expected[i]);
        me_graph_schedule_free(s); me_graph_plan_free(p);
    }
}
static void validation(void) {
    me_graph_input_metadata input = {"x", ME_FLOAT64, 1, {0}};
    const char *bad[] = {"x.thing", "where(x > 0, sum(x), x)", "sum(x, bogus=1)", "where(x, x)", "x < 1 < 2", "x + missing", "x + 9223372036854775808"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        me_graph_plan *p = (void *)1;
        CHECK(me_graph_prepare_expression(bad[i], strlen(bad[i]), &input, 1, NULL, &p, &error) != 0);
        CHECK(p == NULL);
    }
    me_graph_plan *p = prepare("x + 1", &input, 1, false); me_graph_schedule *s = NULL;
    CHECK(!me_graph_specialize(p, &input, 1, NULL, &s, &error));
    me_array_view v = view("x", ME_FLOAT64, NULL, 0, 1, 0, 0);
    CHECK(!me_graph_execute(s, &v, 1, NULL, 0, NULL, NULL, &error));
    me_graph_schedule_free(s); me_graph_plan_free(p);
    int flags = fetestexcept(FE_ALL_EXCEPT); int rounding = fegetround();
    p = prepare("x + 1e-300", &input, 1, false);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == flags && fegetround() == rounding);
    me_graph_plan_free(p);
    fenv_t saved; fegetenv(&saved);
#ifndef __EMSCRIPTEN__
    feraiseexcept(FE_DIVBYZERO); fesetround(FE_DOWNWARD);
#endif
    flags = fetestexcept(FE_ALL_EXCEPT); rounding = fegetround();
    p = prepare("1 / 0", NULL, 0, false);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == flags && fegetround() == rounding);
    me_graph_plan_free(p);
    const char *malformed = "{\"value\":1e999}";
    CHECK(me_graph_prepare_json(malformed, strlen(malformed), NULL, &p, &error) != 0);
    CHECK(p == NULL && fetestexcept(FE_ALL_EXCEPT) == flags && fegetround() == rounding);
    fesetenv(&saved);
    me_graph_plan_free(NULL); me_graph_schedule_free(NULL);
}
static void limits(void) {
    char json[20000];
    for (int depth = ME_GRAPH_MAX_DEPTH; depth <= ME_GRAPH_MAX_DEPTH + 1; depth++) {
        size_t n = (size_t)snprintf(json, sizeof(json), "{\"format\":\"menudet-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\"],\"nodes\":[{\"id\":0,\"op\":\"input\",\"name\":\"x\",\"dtype\":\"float64\"}");
        for (int i = 1; i <= depth; i++) n += (size_t)snprintf(json + n, sizeof(json) - n, ",{\"id\":%d,\"op\":\"neg\",\"args\":[%d]}", i, i - 1);
        n += (size_t)snprintf(json + n, sizeof(json) - n, "],\"root\":%d,\"output\":{\"dtype\":\"auto\",\"casting\":\"unsafe\"}}", depth);
        me_graph_plan *p = NULL;
        int rc = me_graph_prepare_json(json, n, NULL, &p, &error);
        CHECK(depth == ME_GRAPH_MAX_DEPTH ? rc == 0 : rc == ME_GRAPH_ERR_FORMAT);
        me_graph_plan_free(p);
    }
    me_graph_input_metadata input = {"x", ME_FLOAT64, 2, {INT64_MAX, 2}};
    me_graph_plan *p = prepare("x", &input, 1, false); me_graph_schedule *s = NULL;
    CHECK(me_graph_specialize(p, &input, 1, NULL, &s, &error) == ME_GRAPH_ERR_SHAPE && s == NULL);
    input.rank = 1; input.shape[0] = 3;
    me_graph_specialize_options options = {sizeof(options), ME_GRAPH_VERSION, (size_t)INT32_MAX + 1, 0};
    CHECK(me_graph_specialize(p, &input, 1, &options, &s, &error) == ME_GRAPH_ERR_FORMAT && s == NULL);
    me_graph_plan_free(p);
}
static void staged(bool jit) {
    me_graph_input_metadata input = {"x", ME_FLOAT64, 2, {2, 3}};
    me_graph_plan *p = prepare("x - sum(x, axis=0)", &input, 1, jit), *copy = NULL;
    CHECK(me_graph_stage_count(p) == 2);
    CHECK(me_graph_capabilities(p) & ME_GRAPH_CAP_STAGED);
    CHECK(me_graph_stage_last_consumer(p, 0) == 1);
    size_t length;
    const char *json = me_graph_export_json(p, &length);
    CHECK(!me_graph_prepare_json(json, length, NULL, &copy, &error));
    me_graph_plan_free(copy);
    me_graph_schedule *s = NULL;
    me_graph_specialize_options options = {sizeof(options), ME_GRAPH_VERSION, 1, 23};
    CHECK(me_graph_specialize(p, &input, 1, &options, &s, &error) == ME_GRAPH_ERR_SHAPE);
    CHECK(s == NULL);
    options.intermediate_budget = 24;
    CHECK(!me_graph_specialize(p, &input, 1, &options, &s, &error));
    CHECK(me_graph_intermediate_bytes(s) == 24);
    CHECK(me_graph_stage_output_rank(s, 0) == 1 && me_graph_stage_output_shape(s, 0)[0] == 3);
    double x[] = {1, 2, 3, 4, 5, 6}, output[6], expected[] = {-4, -5, -6, -1, -2, -3};
    me_array_view v = view("x", ME_FLOAT64, x, sizeof(x), 2, 2, 3);
    me_graph_report report;
    CHECK(!me_graph_execute(s, &v, 1, output, sizeof(output), NULL, &report, &error));
    CHECK(!memcmp(output, expected, sizeof(output)) && report.stages == 2);
    output[0] = 99; v.capacity = 1;
    CHECK(me_graph_execute(s, &v, 1, output, sizeof(output), NULL, &report, &error) == ME_GRAPH_ERR_BINDING);
    CHECK(output[0] == 99 && error.stage == 0);
    v.capacity = sizeof(x);
    x[0] = 10;
    CHECK(!me_graph_execute(s, &v, 1, output, sizeof(output), NULL, &report, &error));
    CHECK(output[0] == -4 && output[3] == -10);
    me_graph_schedule_free(s); me_graph_plan_free(p);
}
static void trusted(bool jit) {
    me_graph_input_metadata input = {"x", ME_FLOAT32, 1, {3}};
    me_graph_plan *map = prepare("x * 2", &input, 1, jit), *p = NULL;
    const char *artifact = me_graph_export_map_json(map);
    CHECK(artifact != NULL);
    size_t capacity = strlen(artifact) + 2048;
    char *json = malloc(capacity);
    CHECK(json != NULL);
    int size = snprintf(json, capacity,
        "{\"format\":\"menudet-staged-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\",\"staged\"],"
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"float32\"}],\"root\":1,\"stages\":["
        "{\"id\":0,\"kind\":\"portable\",\"inputs\":{\"x\":{\"input\":\"x\"}},\"artifact\":%s,"
        "\"contract\":{\"cardinality\":\"elementwise\",\"context\":\"none\",\"effects\":\"ordered-lazy\",\"mask\":\"none\"}},"
        "{\"id\":1,\"kind\":\"graph\",\"inputs\":{\"y\":{\"stage\":0}},\"graph\":{"
        "\"format\":\"menudet-graph-1\",\"semantics\":\"menudet-numpy-1.1\",\"requires\":[\"numeric\"],"
        "\"nodes\":[{\"id\":0,\"op\":\"input\",\"name\":\"y\",\"dtype\":\"auto\"},"
        "{\"id\":1,\"op\":\"sum\",\"args\":[0],\"axes\":null,\"keepdims\":false,\"dtype\":\"auto\",\"initial\":null,\"where\":null}],"
        "\"root\":1,\"output\":{\"dtype\":\"auto\",\"casting\":\"unsafe\"}}}]}", artifact);
    CHECK(size > 0 && (size_t)size < capacity);
    me_graph_prepare_options options = {sizeof(options), ME_GRAPH_VERSION, jit ? ME_JIT_ON : ME_JIT_OFF, false, true};
    CHECK(!me_graph_prepare_json(json, (size_t)size, &options, &p, &error));
    free(json); me_graph_plan_free(map);
    CHECK(!strcmp(me_graph_stage_kind(p, 0), "portable"));
    me_graph_schedule *s = NULL;
    CHECK(!me_graph_specialize(p, &input, 1, NULL, &s, &error));
    float x[] = {1, 2, 3}, output = 0;
    me_array_view v = view("x", ME_FLOAT32, x, sizeof(x), 1, 3, 0);
    CHECK(!me_graph_execute(s, &v, 1, &output, sizeof(output), NULL, NULL, &error));
    CHECK(output == 12);
    me_graph_schedule_free(s); me_graph_plan_free(p);
}
int main(void) {
    maps(false); lazy(false); reductions(false); validation(); limits();
    maps(true); lazy(true); reductions(true);
    staged(false); staged(true);
    trusted(false); trusted(true);
    puts("native graph preparation/execution passed"); return 0;
}
