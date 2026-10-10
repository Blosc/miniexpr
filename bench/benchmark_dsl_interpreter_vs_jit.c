/*
 * Portable 1.1 interpreter vs JIT vs graphs: warm per-evaluation benchmark.
 *
 * Every case is a valid portable 1.1 artifact that the interpreter executes,
 * and represent families originally outside the portable JIT. Math calls,
 * integer arithmetic, locals and simple branches now lower; the remaining
 * families still fall back. A final JIT-eligible control kernel shows
 * that the harness does detect real compiled execution when a backend can.
 *
 * Reported per backend: best (warm) wall time of one full-array evaluation.
 * Cold load/compile time is deliberately excluded.
 *
 * Usage:
 *   ./benchmark_dsl_interpreter_vs_jit [nitems] [repeats] [graph_backend=tcc]
 * Graph backend: interpreter, tcc, gcc-16 or clang. Graph preparation and
 * specialization are excluded. NO_COLOR disables color; FORCE_COLOR enables it
 * even when stdout is redirected. Otherwise highlight winners only on a TTY.
 */

#include "miniexpr_graph.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MAX_INPUTS 2
#define N_BACKENDS 4
#define N_COLUMNS (N_BACKENDS + 1)

typedef struct {
    const char *label;
    const char *compiler; /* ME_DSL_JIT_COMPILER value; NULL leaves it unset */
    const char *cc;       /* CC value for the "cc" backend; NULL leaves it unset */
    me_jit_mode mode;
} backend_def;

static const backend_def backends[N_BACKENDS] = {
    {"interpreter", NULL, NULL, ME_JIT_OFF},
    {"tcc", "tcc", NULL, ME_JIT_ON},
    {"gcc-16", "cc", "gcc-16", ME_JIT_ON},
    {"clang", "cc", "clang", ME_JIT_ON},
};

typedef struct {
    const char *label;
    const char *source;
    const char *inputs_json;
    const char *requires_json;
    const char *out_dtype;
    const char *contract;
    int context_ndim;
    me_dtype in_dtype;
    const char *in_names[MAX_INPUTS];
    int n_inputs;
    int scalar;
    const char *graph_expression; /* Equivalent numeric graph; NULL if unavailable. */
} bench_case;

/*
 * One representative per family of originally interpreter-only kernels.
 * Keep sources minimal: they are reduced reproducers, not workloads.
 */
static const bench_case cases[] = {
    {"A1 math-call: sin(x)+cos(x)",
     "def k(x):\n    return sin(x) + cos(x)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0, "sin(x) + cos(x)"},
    {"A2 binary-math: hypot(x, y)",
     "def k(x, y):\n    return hypot(x, y)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"},{\"name\":\"y\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x", "y"}, 2, 0, "hypot(x, y)"},
    {"A3 predicate: isfinite(x)",
     "def k(x):\n    return isfinite(x)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "bool", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0, "isfinite(x)"},
    {"B1 integer-add: int32 x+y",
     "def k(x, y):\n    return x + y\n",
     "[{\"name\":\"x\",\"dtype\":\"int32\"},{\"name\":\"y\",\"dtype\":\"int32\"}]", "[\"numeric\"]",
     "int32", "elementwise", 0, ME_INT32, {"x", "y"}, 2, 0, "x + y"},
    {"B2 integer-mod: x % 7",
     "def k(x):\n    return x % 7\n",
     "[{\"name\":\"x\",\"dtype\":\"int32\"}]", "[\"numeric\"]",
     "int32", "elementwise", 0, ME_INT32, {"x"}, 1, 0, "x % 7"},
    {"B3 integer-shift: x << 2",
     "def k(x):\n    return x << 2\n",
     "[{\"name\":\"x\",\"dtype\":\"int32\"}]", "[\"numeric\"]",
     "int32", "elementwise", 0, ME_INT32, {"x"}, 1, 0, "x << 2"},
    {"B4 narrow-cast: int16(x)*2",
     "def k(x):\n    return int16(x) * 2\n",
     "[{\"name\":\"x\",\"dtype\":\"int8\"}]", "[\"numeric\"]",
     "int16", "elementwise", 0, ME_INT8, {"x"}, 1, 0, "int16(x) * 2"},
    {"C1 float-floordiv: x // 3.0",
     "def k(x):\n    return x // 3.0\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0, "x // 3.0"},
    {"C2 float-rem: x % 1.5",
     "def k(x):\n    return x % 1.5\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0, "x % 1.5"},
    {"C3 float-pow: x ** 2.5",
     "def k(x):\n    return x ** 2.5\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0, "x ** 2.5"},
    {"D local-temp: y=x*2; y+1",
     "def k(x):\n    y = x * 2.0\n    return y + 1.0\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0, "(x * 2.0) + 1.0"},
    {"E control-flow: if x>0 else -x",
     "def k(x):\n    if x > 0.0:\n        return x\n    return -x\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\",\"control-flow\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0, "where(x > 0.0, x, -x)"},
    {"F reduction: sum(x)",
     "def k(x):\n    return sum(x)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\",\"block-reductions\"]",
     "float64", "block_scalar", 0, ME_FLOAT64, {"x"}, 1, 1, "sum(x)"},
    {"G nd-context: x + _i0",
     "def k(x):\n    return x + _i0\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\",\"nd-context\"]",
     "float64", "elementwise", 1, ME_FLOAT64, {"x"}, 1, 0, NULL},
    {"control eligible: where(x!=0,y/x,y)",
     "def k(x, y):\n    return where(x != 0, y / x, y)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"},{\"name\":\"y\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x", "y"}, 2, 0, "where(x != 0, y / x, y)"},
};

static size_t dtype_bytes(me_dtype dtype) {
    switch (dtype) {
    case ME_BOOL: return 1;
    case ME_INT8: case ME_UINT8: return 1;
    case ME_INT16: case ME_UINT16: return 2;
    case ME_INT32: case ME_UINT32: case ME_FLOAT32: return 4;
    default: return 8;
    }
}

static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    return 1e9 * (double)ticks.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e9 * ts.tv_sec + ts.tv_nsec;
#endif
}

static void bench_setenv(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static void apply_backend(const backend_def *backend) {
    bench_setenv("ME_DSL_JIT_COMPILER", backend->compiler ? backend->compiler : "");
    bench_setenv("CC", backend->cc ? backend->cc : "");
    /* The portable JIT fails closed under user toolchain overrides. */
    bench_setenv("CFLAGS", "");
    bench_setenv("ME_DSL_JIT_TCC_OPTIONS", "");
}

static void json_escape(const char *src, char *dst, size_t cap) {
    size_t out = 0;
    for (const char *p = src; *p && out + 1 < cap; p++) {
        const char *esc = NULL;
        switch (*p) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        default: break;
        }
        if (esc) {
            size_t len = strlen(esc);
            if (out + len + 1 >= cap) break;
            memcpy(dst + out, esc, len);
            out += len;
        }
        else {
            dst[out++] = *p;
        }
    }
    dst[out] = '\0';
}

static bool build_json(const bench_case *c, char *out, size_t cap) {
    char source[1024];
    json_escape(c->source, source, sizeof(source));
    int n = snprintf(out, cap,
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":%s,\"source\":\"%s\",\"entry_point\":\"k\",\"inputs\":%s,"
        "\"constants\":[],\"output\":{\"dtype\":\"%s\",\"contract\":\"%s\"},"
        "\"context\":{\"ndim\":%d},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},"
        "\"metadata\":{}}",
        c->requires_json, source, c->inputs_json, c->out_dtype, c->contract, c->context_ndim);
    return n > 0 && (size_t)n < cap;
}

static void fill_input(me_dtype dtype, void *buffer, size_t nitems) {
    for (size_t i = 0; i < nitems; i++) {
        double value = (i % 5 == 0) ? 0.0 : (double)(i % 1000) * 0.001 + 0.5;
        int64_t integer = (int64_t)(i % 1000) - 500;
        switch (dtype) {
        case ME_FLOAT32: ((float *)buffer)[i] = (float)value; break;
        case ME_BOOL: ((uint8_t *)buffer)[i] = (uint8_t)(i % 3 == 0); break;
        case ME_INT8: ((int8_t *)buffer)[i] = (int8_t)integer; break;
        case ME_INT16: ((int16_t *)buffer)[i] = (int16_t)integer; break;
        case ME_INT32: ((int32_t *)buffer)[i] = (int32_t)integer; break;
        case ME_INT64: ((int64_t *)buffer)[i] = integer; break;
        case ME_UINT8: ((uint8_t *)buffer)[i] = (uint8_t)integer; break;
        case ME_UINT16: ((uint16_t *)buffer)[i] = (uint16_t)integer; break;
        case ME_UINT32: ((uint32_t *)buffer)[i] = (uint32_t)integer; break;
        case ME_UINT64: ((uint64_t *)buffer)[i] = (uint64_t)integer; break;
        default: ((double *)buffer)[i] = value; break;
        }
    }
}

static int eval_case(const me_artifact *artifact, const bench_case *c,
                     void *const in_data[MAX_INPUTS], void *out, size_t nitems,
                     me_artifact_error *error) {
    size_t width = dtype_bytes(c->in_dtype);
    me_artifact_buffer buffers[MAX_INPUTS];
    for (int i = 0; i < c->n_inputs; i++) {
        buffers[i] = (me_artifact_buffer){c->in_names[i], c->in_dtype, width,
                                         in_data[i], nitems * width};
    }
    size_t out_width = me_artifact_output_itemsize(artifact);
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems = nitems, .output_capacity = (c->scalar ? 1 : nitems) * out_width};
    int64_t shape[1] = {(int64_t)nitems};
    int64_t origin[1] = {0};
    int64_t extent[1] = {(int64_t)nitems};
    if (c->context_ndim) {
        descriptor.ndim = c->context_ndim;
        descriptor.logical_shape = shape;
        descriptor.block_origin = origin;
        descriptor.block_extent = extent;
    }
    return (int)me_artifact_eval_ex(artifact, buffers, c->n_inputs, out, &descriptor, error);
}

static bool use_color(void) {
    if (getenv("NO_COLOR")) return false;
    if (getenv("FORCE_COLOR")) return true;
#ifdef _WIN32
    if (!_isatty(_fileno(stdout))) return false;
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    return GetConsoleMode(handle, &mode) && SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
    const char *term = getenv("TERM");
    return isatty(STDOUT_FILENO) && (!term || strcmp(term, "dumb"));
#endif
}

static void print_row(const char *label, char cell[N_COLUMNS][20],
    const double times[N_COLUMNS], bool color) {
    double fastest = HUGE_VAL;
    for (int b = 0; b < N_COLUMNS; b++) if (times[b] >= 0 && times[b] < fastest) fastest = times[b];
    printf("%-38s", label);
    for (int b = 0; b < N_COLUMNS; b++) {
        bool winner = color && times[b] >= 0 && times[b] == fastest;
        printf(" %s%16s%s", winner ? "\033[1;32m" : "", cell[b], winner ? "\033[0m" : "");
    }
    printf("\n");
}

static double bench_graph(const bench_case *c, const backend_def *backend,
    void *const in_data[MAX_INPUTS], void *out, size_t capacity, size_t nitems,
    int repeats, const void *reference, size_t ref_bytes, me_dtype ref_dtype,
    char cell[20], bool *value_mismatch) {
    if (!c->graph_expression) { snprintf(cell, 20, "n/a"); return -1; }
    apply_backend(backend);
    me_graph_input_metadata metadata[MAX_INPUTS] = {0};
    me_array_view inputs[MAX_INPUTS] = {0};
    size_t width = dtype_bytes(c->in_dtype);
    for (int i = 0; i < c->n_inputs; i++) {
        metadata[i] = (me_graph_input_metadata){c->in_names[i], c->in_dtype, 1, {(int64_t)nitems}};
        inputs[i] = (me_array_view){.name = c->in_names[i], .dtype = c->in_dtype,
            .base = in_data[i], .capacity = nitems * width, .rank = 1,
            .shape = {(int64_t)nitems}, .strides = {(int64_t)width}};
    }
    me_graph_prepare_options prepare = {sizeof(prepare), ME_GRAPH_VERSION, backend->mode, false, true};
    me_graph_plan *plan = NULL;
    me_graph_schedule *schedule = NULL;
    me_graph_error error = {0};
    int rc = me_graph_prepare_expression(c->graph_expression, strlen(c->graph_expression),
        metadata, c->n_inputs, &prepare, &plan, &error);
    if (!rc) rc = me_graph_specialize(plan, metadata, c->n_inputs, NULL, &schedule, &error);
    double best = -1;
    if (rc) {
        snprintf(cell, 20, rc == ME_GRAPH_ERR_UNSUPPORTED || rc == ME_GRAPH_ERR_CAPABILITY ? "n/a" : "prep!");
        fprintf(stderr, "%s graph: %s\n", c->label, error.native.message);
        goto cleanup;
    }
    me_graph_report report;
    rc = me_graph_execute(schedule, inputs, c->n_inputs, out, capacity, NULL, &report, &error);
    if (rc) { snprintf(cell, 20, "eval!"); goto cleanup; }
    bool compiled = report.has_jit;
    bool direct_reduction = c->scalar && report.array.evaluated_tiles == 0;
    int batch = 1;
    for (int r = 0; r < repeats; r++) {
        double start = now_ns();
        for (int k = 0; k < batch && !rc; k++) {
            rc = me_graph_execute(schedule, inputs, c->n_inputs, out, capacity, NULL, NULL, &error);
        }
        double elapsed = (now_ns() - start) / batch;
        if (rc) { snprintf(cell, 20, "eval!"); best = -1; goto cleanup; }
        if (r == 0 && elapsed > 0 && elapsed < 2e6) {
            double target = 2e6 / elapsed + 1;
            batch = target > 100000 ? 100000 : (int)target;
        }
        if (r == 0 || elapsed < best) best = elapsed;
    }
    size_t bytes = me_graph_output_bytes(schedule);
    bool match = ref_bytes && ref_dtype == me_graph_output_dtype(schedule) &&
        ref_bytes == bytes && memcmp(reference, out, bytes) == 0;
    if (ref_bytes && !match) *value_mismatch = true;
    snprintf(cell, 20, "%8.3f %s%s", best / 1e6,
        compiled ? "JIT" : direct_reduction ? "R" : backend->mode == ME_JIT_OFF ? "I" : "fb",
        ref_bytes ? match ? "" : "*" : "?");
    if (!match) best = -1; /* Unverified/mismatching cells cannot win a row. */
cleanup:
    me_graph_schedule_free(schedule);
    me_graph_plan_free(plan);
    return best;
}

int main(int argc, char **argv) {
    size_t nitems = 1u << 16;
    int repeats = 7;
    if (argc > 1) nitems = (size_t)strtoull(argv[1], NULL, 10);
    if (argc > 2) repeats = atoi(argv[2]);
    if (nitems < 1) nitems = 1;
    if (repeats < 1) repeats = 1;
    const backend_def *graph_backend = &backends[1];
    if (argc > 3) {
        graph_backend = NULL;
        for (int b = 0; b < N_BACKENDS; b++) if (!strcmp(argv[3], backends[b].label)) graph_backend = &backends[b];
        if (!graph_backend) { fprintf(stderr, "Graph backend must be interpreter, tcc, gcc-16 or clang\n"); return 2; }
    }
    if (nitems > INT32_MAX || nitems > (SIZE_MAX - 64) / 8) {
        fprintf(stderr, "nitems exceeds native block or allocation limits\n"); return 2;
    }
    bool color = use_color();
    char graph_label[32]; snprintf(graph_label, sizeof(graph_label), "graph (%s)", graph_backend->label);

    printf("MiniExpr portable 1.1: interpreter vs JIT vs graph (warm per-evaluation time)\n");
    printf("nitems=%zu repeats=%d  (cold load/compile excluded; best of warm runs; ms per full-array call)\n\n",
           nitems, repeats);
    printf("graph: equivalent expressions, %s preference, 1024-item tiles; preparation/specialization excluded\n\n", graph_backend->label);
    printf("%-38s %16s %16s %16s %16s %16s\n", "case (family / example)",
           backends[0].label, backends[1].label, backends[2].label, backends[3].label, graph_label);

    bool backend_compiled[N_BACKENDS] = {false, false, false, false};
    bool value_mismatch = false;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const bench_case *c = &cases[ci];
        char json[4096];
        if (!build_json(c, json, sizeof(json))) {
            printf("%-38s %16s %16s %16s %16s %16s\n", c->label, "BAD-JSON", "", "", "", "");
            continue;
        }

        size_t width = dtype_bytes(c->in_dtype);
        void *in_data[MAX_INPUTS] = {NULL, NULL};
        for (int i = 0; i < c->n_inputs; i++) {
            in_data[i] = malloc(nitems * width);
            if (in_data[i]) fill_input(c->in_dtype, in_data[i], nitems);
        }
        size_t out_cap = nitems * 8 + 64;
        void *out = malloc(out_cap);
        void *reference = malloc(out_cap);
        size_t ref_bytes = 0;
        me_dtype ref_dtype = ME_AUTO;

        char cell[N_COLUMNS][20];
        double times[N_COLUMNS];
        for (int b = 0; b < N_COLUMNS; b++) times[b] = -1;
        bool input_ok = out && reference;
        for (int i = 0; i < c->n_inputs; i++) input_ok = input_ok && in_data[i];
        for (int b = 0; b < N_BACKENDS; b++) {
            if (!input_ok) {
                snprintf(cell[b], sizeof(cell[b]), "OOM");
                continue;
            }
            apply_backend(&backends[b]);
            me_artifact *artifact = NULL;
            me_artifact_error error = {0};
            if (me_artifact_load(json, strlen(json), backends[b].mode, &artifact, &error) != ME_ARTIFACT_SUCCESS) {
                snprintf(cell[b], sizeof(cell[b]), "load!");
                continue;
            }
            bool compiled = me_artifact_has_jit(artifact);
            if (b != 0 && compiled) backend_compiled[b] = true;

            if (eval_case(artifact, c, in_data, out, nitems, &error) != ME_ARTIFACT_SUCCESS) {
                snprintf(cell[b], sizeof(cell[b]), "eval!");
            }
            else {
                /* One warmup, then the best warm run; cold compile already happened.
                 * A per-sample batch is auto-sized to ~2 ms so the fast JIT cells
                 * are not dominated by timer granularity. */
                (void)eval_case(artifact, c, in_data, out, nitems, &error);
                double best = 0.0;
                int batch = 1;
                for (int r = 0; r < repeats; r++) {
                    double start = now_ns();
                    int rc = ME_ARTIFACT_SUCCESS;
                    for (int k = 0; k < batch && rc == ME_ARTIFACT_SUCCESS; k++) {
                        rc = eval_case(artifact, c, in_data, out, nitems, &error);
                    }
                    double elapsed = (now_ns() - start) / batch;
                    if (rc != ME_ARTIFACT_SUCCESS) { best = -1.0; break; }
                    if (r == 0 && elapsed > 0 && elapsed < 2e6) {
                        double target = 2e6 / elapsed + 1;
                        batch = target > 100000 ? 100000 : (int)target;
                    }
                    if (r == 0 || elapsed < best) best = elapsed;
                }
                size_t out_bytes = (c->scalar ? 1 : nitems) * me_artifact_output_itemsize(artifact);
                if (best < 0.0) {
                    snprintf(cell[b], sizeof(cell[b]), "eval!");
                }
                else if (b == 0) {
                    memcpy(reference, out, out_bytes);
                    ref_bytes = out_bytes;
                    ref_dtype = me_artifact_output_dtype(artifact);
                    times[b] = best;
                    snprintf(cell[b], sizeof(cell[b]), "%8.3f %s", best / 1e6, "I");
                }
                else {
                    bool match = ref_bytes && ref_dtype == me_artifact_output_dtype(artifact) &&
                        ref_bytes == out_bytes && memcmp(reference, out, out_bytes) == 0;
                    if (ref_bytes && !match) value_mismatch = true;
                    if (match) times[b] = best;
                    snprintf(cell[b], sizeof(cell[b]), "%8.3f %s%s", best / 1e6,
                              compiled ? "JIT" : "fb", ref_bytes ? match ? "" : "*" : "?");
                }
            }
            me_artifact_free(artifact);
        }

        if (input_ok) {
            times[N_BACKENDS] = bench_graph(c, graph_backend, in_data, out, out_cap,
                nitems, repeats, reference, ref_bytes, ref_dtype, cell[N_BACKENDS], &value_mismatch);
        } else snprintf(cell[N_BACKENDS], sizeof(cell[N_BACKENDS]), "OOM");
        print_row(c->label, cell, times, color);

        for (int i = 0; i < c->n_inputs; i++) free(in_data[i]);
        free(out);
        free(reference);
    }

    printf("\nlegend: I=interpreter  JIT=compiled kernel  fb=JIT declined, interpreter fallback");
    printf("\n        *=warm result differs bitwise from the interpreter reference\n");
    printf("        ?=no interpreter reference  n/a=no equivalent graph/context support\n");
    printf("        R=direct native graph reduction (identity map skipped)\n");
    if (color) printf("        green=fastest matching cell per row (using unrounded timings)\n");
    printf("compiled control kernel per backend:");
    for (int b = 1; b < N_BACKENDS; b++) {
        printf("  %s=%s", backends[b].label, backend_compiled[b] ? "yes" : "no");
    }
    printf("\nvalue checks: %s\n", value_mismatch ? "MISMATCH" : "all match interpreter");
    return value_mismatch ? 1 : 0;
}
