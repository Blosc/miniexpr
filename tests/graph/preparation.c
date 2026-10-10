#include "miniexpr_graph.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, error.native.message); exit(1); } } while (0)
static me_graph_error error;
static me_graph_plan *prepare(const char *source, me_graph_input_metadata *inputs, int n, bool jit) {
    fprintf(stderr, "  prepare jit=%d: %s\n", jit, source); fflush(stderr);
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
        fprintf(stderr, "  graph depth=%d\n", depth); fflush(stderr);
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
/* Compare the optimized graph route against the unchanged artifact-array
 * reducer, including exact serial rounding and floating status. */
static void direct_reductions(void) {
    const char *names[] = {"sum", "prod", "min", "max", "any", "all"};
    me_dtype types[] = {ME_FLOAT32, ME_FLOAT64, ME_INT32, ME_BOOL, ME_INT8, ME_INT16,
        ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64};
    float f[] = {1, -2, 3, 0, 5, -6};
    double d[] = {1, -2, 3, 0, 5, -6};
    int32_t integers[] = {1, -2, 3, 0, 5, -6};
    bool booleans[] = {true, false, true, false, true, true};
    int8_t i8[] = {1, -2, 3, 0, 5, -6}; int16_t i16[] = {1, -2, 3, 0, 5, -6};
    int64_t i64[] = {INT64_MAX, 1, INT64_MIN, -1, 0, 3};
    uint8_t u8[] = {1, 2, 3, 0, 5, 6}; uint16_t u16[] = {1, 2, 3, 0, 5, 6};
    uint32_t u32[] = {1, 2, 3, 0, 5, 6}; uint64_t u64[] = {UINT64_MAX, 1, 0, 1, 2, 3};
    void *data[] = {f, d, integers, booleans, i8, i16, i64, u8, u16, u32, u64};
    size_t widths[] = {4, 8, 4, 1, 1, 2, 8, 1, 2, 4, 8};
    for (int type = 0; type < 11; type++) for (int op = 0; op < 6; op++) {
        char source[32]; snprintf(source, sizeof(source), "%s(x)", names[op]);
        me_graph_input_metadata input = {"x", types[type], 1, {6}};
        me_graph_plan *p = prepare(source, &input, 1, false);
        size_t width = widths[type];
        me_array_view v = {0}; v.name = "x"; v.dtype = types[type];
        v.rank = 1; v.shape[0] = 6; v.strides[0] = (int64_t)width;
        v.base = data[type]; v.capacity = 6 * width;
        for (size_t tile = 1; tile <= 9; tile += 4) {
            me_graph_specialize_options options = {sizeof(options), ME_GRAPH_VERSION, tile, 0};
            me_graph_schedule *s = NULL;
            CHECK(!me_graph_specialize(p, &input, 1, &options, &s, &error));
            me_array_options reduction = {0}; reduction.version = ME_ARTIFACT_ARRAY_VERSION;
            reduction.reduction = (me_array_reduction)(op + 1); reduction.naxes = -1;
            reduction.tile_items = tile;
            uint64_t actual = 0, expected = 0;
            me_array_report reference; me_graph_report report; me_artifact_error native;
            CHECK(!me_artifact_eval_array(me_graph_map_artifact(p), &v, 1, 1, input.shape,
                &reduction, &expected, sizeof(expected), &reference, &native));
            CHECK(!me_graph_execute(s, &v, 1, &actual, sizeof(actual), NULL, &report, &error));
            CHECK(!memcmp(&actual, &expected, me_graph_output_bytes(s)));
            CHECK(report.array.fp_flags == reference.fp_flags);
            CHECK(report.array.evaluated_tiles == 0 && report.array.temporary_bytes == 0);
            CHECK(!report.has_jit && report.jit_stages == 0);
            me_graph_schedule_free(s);
        }
        me_graph_plan_free(p);
    }
    /* Cancellation-sensitive order, overflow, NaNs and signed zero. */
    double special[][6] = {{1e16, 1, -1e16, 1, -0.0, 0},
        {DBL_MAX, DBL_MAX, -DBL_MAX, 1, 2, 3}, {INFINITY, -INFINITY, 1, 2, 3, 4},
        {NAN, 1, 2, 3, 4, 5}, {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0}};
    for (size_t i = 0; i < sizeof(special) / sizeof(*special); i++) {
        me_graph_input_metadata input = {"x", ME_FLOAT64, 1, {6}};
        me_graph_plan *p = prepare("sum(x)", &input, 1, false);
        me_graph_schedule *s = NULL;
        CHECK(!me_graph_specialize(p, &input, 1, NULL, &s, &error));
        me_array_view v = view("x", ME_FLOAT64, special[i], sizeof(special[i]), 1, 6, 0);
        me_array_options o = {0}; o.version = ME_ARTIFACT_ARRAY_VERSION;
        o.reduction = ME_ARRAY_SUM; o.naxes = -1; o.tile_items = 1;
        double expected, actual; me_array_report reference; me_graph_report report; me_artifact_error native;
        CHECK(!me_artifact_eval_array(me_graph_map_artifact(p), &v, 1, 1, input.shape,
            &o, &expected, sizeof(expected), &reference, &native));
        CHECK(!me_graph_execute(s, &v, 1, &actual, sizeof(actual), NULL, &report, &error));
        CHECK(!memcmp(&actual, &expected, sizeof(actual)) && report.array.fp_flags == reference.fp_flags);
        me_graph_schedule_free(s); me_graph_plan_free(p);
    }
}
static void dsl_direct_sum(void) {
    const char *json =
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\",\"block-reductions\"],\"source\":\"def k(x):\\n    return sum(x)\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"float64\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"block_scalar\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}";
    me_artifact *a = NULL; me_artifact_error native;
    CHECK(!me_artifact_load(json, strlen(json), ME_JIT_OFF, &a, &native));
    double x[] = {1e16, 1, -1e16, 3}, result;
    me_artifact_buffer buffer = {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)};
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 4, .output_capacity = sizeof(result)};
    me_artifact_fp_status status;
    CHECK(!me_artifact_eval_status(a, &buffer, 1, &result, &descriptor, 0, &status, &native));
    CHECK(result == 3 && status.flags == 0);
    uint8_t mask[] = {0, 0, 1, 1};
    x[0] = INFINITY; x[1] = -INFINITY; x[2] = 2;
    descriptor.valid_mask = mask; descriptor.valid_mask_capacity = sizeof(mask);
    CHECK(!me_artifact_eval_status(a, &buffer, 1, &result, &descriptor, 0, &status, &native));
    CHECK(result == 5 && status.flags == 0);
    descriptor.valid_mask = NULL; descriptor.valid_mask_capacity = 0;
    CHECK(!me_artifact_eval_status(a, &buffer, 1, &result, &descriptor, 0, &status, &native));
    CHECK(isnan(result));
#ifndef __EMSCRIPTEN__
    CHECK(status.flags & 1);
#endif
    me_artifact_free(a);
}
static void dsl_direct_reductions(void) {
    const char *names[] = {"float32", "float64", "int64", "uint64", "bool"};
    me_dtype types[] = {ME_FLOAT32, ME_FLOAT64, ME_INT64, ME_UINT64, ME_BOOL};
    size_t widths[] = {4, 8, 8, 8, sizeof(bool)};
    const char *ops[] = {"prod", "min", "max", "any", "all"};
    for (int type = 0; type < 5; type++) for (int op = 0; op < 5; op++) {
        char json[2048];
        const char *output = op >= 3 ? "bool" : op == 0 && type == 4 ? "int64" : names[type];
        snprintf(json, sizeof(json),
            "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
            "\"requires\":[\"numeric\",\"block-reductions\"],\"source\":\"def k(x):\\n    return %s(x)\\n\","
            "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"}],\"constants\":[],"
            "\"output\":{\"dtype\":\"%s\",\"contract\":\"block_scalar\"},\"context\":{\"ndim\":0},"
            "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
            ops[op], names[type], output);
        me_artifact *a = NULL; me_artifact_error native;
        CHECK(!me_artifact_load(json, strlen(json), ME_JIT_OFF, &a, &native));
        for (int edge = 0; edge < 3; edge++) {
            float f[] = {1, -0.0f, 2, 3, 0};
            double d[] = {1, -0.0, 2, 3, 0};
            int64_t s[] = {-1, 2, 3, 4, 0};
            uint64_t u[] = {1, 2, 3, 4, 0};
            bool b[] = {true, true, false, true, false};
            if (edge == 1) { f[1] = NAN; d[1] = NAN; s[0] = INT64_MIN; s[1] = -1; u[0] = UINT64_MAX; }
            if (edge == 2) { f[0] = INFINITY; d[0] = INFINITY; }
            void *data[] = {f, d, s, u, b};
            me_artifact_buffer buffer = {"x", types[type], widths[type], data[type], widths[type] * 5};
            me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 4, .output_capacity = 8};
            uint64_t direct = 0, generic = 0; me_artifact_fp_status status, reference;
            int rc = me_artifact_eval_status(a, &buffer, 1, &direct, &descriptor, 0, &status, &native);
            uint8_t mask[] = {1, 1, 1, 1, 0};
            descriptor.nitems = 5; descriptor.valid_mask = mask; descriptor.valid_mask_capacity = sizeof(mask);
            int expected_rc = me_artifact_eval_status(a, &buffer, 1, &generic, &descriptor, 0, &reference, &native);
            CHECK(rc == expected_rc);
            if (!rc) { CHECK(direct == generic); CHECK(status.flags == reference.flags); }
        }
        me_artifact_free(a);
    }
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
#define RUN_CASE(call) do { fprintf(stderr, "graph preparation: %s\n", #call); fflush(stderr); call; } while (0)
    RUN_CASE(maps(false)); RUN_CASE(lazy(false)); RUN_CASE(reductions(false));
    RUN_CASE(validation()); RUN_CASE(limits());
    RUN_CASE(maps(true)); RUN_CASE(lazy(true)); RUN_CASE(reductions(true));
    RUN_CASE(staged(false)); RUN_CASE(staged(true));
    RUN_CASE(direct_reductions());
    RUN_CASE(dsl_direct_sum());
    RUN_CASE(dsl_direct_reductions());
    RUN_CASE(trusted(false)); RUN_CASE(trusted(true));
#undef RUN_CASE
    puts("native graph preparation/execution passed"); return 0;
}
