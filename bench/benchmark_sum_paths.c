/* Compare warm sum(x) through DSL block interpretation/JIT and the native
 * graph logical-array reducer. No Python, compression, or preparation timed.
 * Usage: benchmark_sum_paths [items=1048576] [samples=9] [tile_items=1024]
 *                            [dtype=float64] [DSL operand=direct|computed]
 *                            [reduction=sum|prod|min|max|any|all] [off|jit]
 * computed uses block_sum(x + 0), including the operand addition. Optional jit
 * requests DSL compilation with the configured backend and reports fallback;
 * the graph reducer stays interpreted/native for a stable baseline. */
#include "miniexpr_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

static double now(void) {
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter); QueryPerformanceFrequency(&frequency);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}
typedef struct {
    me_artifact *artifact;
    me_graph_schedule *schedule;
    me_artifact_buffer buffer;
    me_artifact_eval_descriptor descriptor;
    me_array_view view;
    double expected;
    me_dtype result_dtype;
} workload;

static void evaluate(workload *w, int graph) {
    uint64_t result = 0;
    int rc;
    if (graph) {
        me_graph_error error;
        rc = me_graph_execute(w->schedule, &w->view, 1, &result, sizeof(result), NULL, NULL, &error);
        if (rc) { fprintf(stderr, "%s\n", error.native.message); exit(1); }
    } else {
        me_artifact_error error;
        rc = me_artifact_eval_ex(w->artifact, &w->buffer, 1, &result, &w->descriptor, &error);
        if (rc) { fprintf(stderr, "%s\n", error.message); exit(1); }
    }
    double actual;
#define LOAD(type) do { type value; memcpy(&value, &result, sizeof(value)); actual = (double)value; } while (0)
    switch (w->result_dtype) {
    case ME_BOOL: LOAD(bool); break;
    case ME_INT8: LOAD(int8_t); break;
    case ME_INT16: LOAD(int16_t); break;
    case ME_INT32: LOAD(int32_t); break;
    case ME_INT64: LOAD(int64_t); break;
    case ME_UINT8: LOAD(uint8_t); break;
    case ME_UINT16: LOAD(uint16_t); break;
    case ME_UINT32: LOAD(uint32_t); break;
    case ME_UINT64: LOAD(uint64_t); break;
    case ME_FLOAT32: LOAD(float); break;
    case ME_FLOAT64: LOAD(double); break;
    default: exit(2);
    }
#undef LOAD
    if (actual != w->expected) {
        fprintf(stderr, "path=%d reduction mismatch: %.17g != %.17g\n", graph, actual, w->expected);
        exit(1);
    }
}
static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
int main(int argc, char **argv) {
    size_t count = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 1048576;
    int samples = argc > 2 ? atoi(argv[2]) : 9;
    size_t tile = argc > 3 ? (size_t)strtoull(argv[3], NULL, 10) : 1024;
    const char *name = argc > 4 ? argv[4] : "float64";
    const char *names[] = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"};
    me_dtype types[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    size_t widths[] = {1, 1, 2, 4, 8, 1, 2, 4, 8, 4, 8};
    int type = -1;
    for (int i = 0; i < 11; i++) if (!strcmp(name, names[i])) type = i;
    if (type < 0) return 2;
    me_dtype dtype = types[type]; size_t width = widths[type];
    bool floating = dtype == ME_FLOAT32 || dtype == ME_FLOAT64;
    bool unsigned_input = type >= 5 && type <= 8;
    const char *output_type = floating ? name : unsigned_input ? "uint64" : "int64";
    bool computed = argc > 5 && !strcmp(argv[5], "computed");
    const char *op = argc > 6 ? argv[6] : "sum";
    bool request_jit = argc > 7 && !strcmp(argv[7], "jit");
    const char *ops[] = {"sum", "prod", "min", "max", "any", "all"};
    int reduction = -1;
    for (int i = 0; i < 6; i++) if (!strcmp(op, ops[i])) reduction = i;
    if (reduction < 0 || (computed && dtype == ME_BOOL && reduction != 0 && reduction != 1)) return 2;
    if (reduction == 2 || reduction == 3) output_type = name;
    if (reduction == 4 || reduction == 5) output_type = "bool";
    if (!count || count > 100000000 || samples < 3 || samples > 99 || !tile || tile > INT32_MAX) return 2;
    workload w = {0};
    w.result_dtype = reduction >= 4 ? ME_BOOL : reduction >= 2 ? dtype :
        floating ? dtype : unsigned_input ? ME_UINT64 : ME_INT64;
    w.expected = reduction == 1 || reduction == 5;
    unsigned char *x = malloc(count * width);
    if (!x) return 2;
    /* Binary-exact values keep summation-order differences out of this timing
     * comparison. Both paths must produce the independently accumulated sum. */
    for (size_t i = 0; i < count; i++) {
        double value = floating ? (double)((int)(i % 257) - 128) / 16.0 :
            dtype == ME_BOOL ? (i % 3 == 0) : unsigned_input ? (double)(i % 17) : (double)((int)(i % 17) - 8);
        if (reduction == 1) value = unsigned_input || dtype == ME_BOOL ? 1 : i % 2 ? -1 : 1;
#define STORE(type) do { type v = (type)value; memcpy(x + i * width, &v, sizeof(v)); } while (0)
        switch (dtype) {
        case ME_BOOL: STORE(bool); break;
        case ME_INT8: STORE(int8_t); break;
        case ME_INT16: STORE(int16_t); break;
        case ME_INT32: STORE(int32_t); break;
        case ME_INT64: STORE(int64_t); break;
        case ME_UINT8: STORE(uint8_t); break;
        case ME_UINT16: STORE(uint16_t); break;
        case ME_UINT32: STORE(uint32_t); break;
        case ME_UINT64: STORE(uint64_t); break;
        case ME_FLOAT32: STORE(float); break;
        case ME_FLOAT64: STORE(double); break;
        default: return 2;
        }
#undef STORE
        if (reduction == 0) w.expected += value;
        else if (reduction == 1) w.expected *= value;
        else if (reduction == 2) { if (!i || value < w.expected) w.expected = value; }
        else if (reduction == 3) { if (!i || value > w.expected) w.expected = value; }
        else if (reduction == 4) w.expected = w.expected != 0 || value != 0;
        else w.expected = w.expected != 0 && value != 0;
    }
    char json[2048];
    snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\",\"block-reductions\"],\"source\":\"def k(x):\\n    return block_%s(%s)\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"block_scalar\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}",
        op, computed ? "x + 0" : "x", name, output_type);
    me_artifact_error artifact_error;
    if (me_artifact_load(json, strlen(json), request_jit ? ME_JIT_ON : ME_JIT_OFF, &w.artifact, &artifact_error)) {
        fprintf(stderr, "%s\n", artifact_error.message); return 1;
    }
    me_graph_input_metadata input = {"x", dtype, 1, {(int64_t)count}};
    me_graph_prepare_options prepare = {sizeof(prepare), ME_GRAPH_VERSION, ME_JIT_OFF, false, true};
    me_graph_specialize_options specialize = {sizeof(specialize), ME_GRAPH_VERSION, tile, 0};
    me_graph_plan *plan = NULL;
    me_graph_error error;
    char expression[32]; snprintf(expression, sizeof(expression), "%s(x)", op);
    if (me_graph_prepare_expression(expression, strlen(expression), &input, 1, &prepare, &plan, &error) ||
        me_graph_specialize(plan, &input, 1, &specialize, &w.schedule, &error)) {
        fprintf(stderr, "%s\n", error.native.message); return 1;
    }
    if (me_graph_has_jit(plan) || (!request_jit && me_artifact_has_jit(w.artifact))) return 1;
    bool compiled = me_artifact_has_jit(w.artifact);
    w.buffer = (me_artifact_buffer){"x", dtype, width, x, count * width};
    w.descriptor = (me_artifact_eval_descriptor){.struct_size = sizeof(w.descriptor),
        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = count, .output_capacity = sizeof(double)};
    w.view = (me_array_view){.name = "x", .dtype = dtype, .base = x,
        .capacity = count * width, .rank = 1, .shape = {(int64_t)count}, .strides = {(int64_t)width}};
    int batch[2]; double timings[2][99];
    for (int path = 0; path < 2; path++) {
        evaluate(&w, path);
        double start = now(); evaluate(&w, path);
        double elapsed = now() - start;
        batch[path] = elapsed > 0 ? (int)(0.005 / elapsed) : 1;
        if (batch[path] < 1) batch[path] = 1;
        if (batch[path] > 10000) batch[path] = 10000;
    }
    for (int sample = 0; sample < samples; sample++) {
        /* Alternate order to avoid always favoring one route's cache state. */
        for (int position = 0; position < 2; position++) {
            int path = (sample + position) % 2;
            double start = now();
            for (int i = 0; i < batch[path]; i++) evaluate(&w, path);
            timings[path][sample] = 1000 * (now() - start) / batch[path];
        }
    }
    for (int path = 0; path < 2; path++) qsort(timings[path], samples, sizeof(double), compare);
    printf("items=%zu %s input_MiB=%.2f tile=%zu samples=%d %s=%.17g graph_JIT=off DSL_JIT=%s DSL=%s\n",
        count, name, count * width / 1048576.0, tile, samples, op, w.expected,
        compiled ? "compiled" : request_jit ? "fallback" : "off", computed ? "computed" : "direct");
    printf("%-24s %12s %12s\n", "path", "best_ms", "median_ms");
    for (int path = 0; path < 2; path++) {
        printf("%-24s %12.4f %12.4f\n", path ? "native graph reducer" : compiled ? "DSL block JIT" : "DSL block interpreter",
            timings[path][0], timings[path][samples / 2]);
    }
    printf("graph / DSL median time: %.2fx\n", timings[1][samples / 2] / timings[0][samples / 2]);
    me_graph_schedule_free(w.schedule); me_graph_plan_free(plan);
    me_artifact_free(w.artifact); free(x);
    return 0;
}
