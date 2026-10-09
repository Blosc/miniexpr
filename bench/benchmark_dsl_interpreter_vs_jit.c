/*
 * Portable 1.1 interpreter vs JIT lowering: warm per-evaluation benchmark.
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
 *   ./benchmark_dsl_interpreter_vs_jit [nitems] [repeats]
 */

#include "miniexpr_artifact.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define MAX_INPUTS 2
#define N_BACKENDS 4

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
} bench_case;

/*
 * One representative per family of originally interpreter-only kernels.
 * Keep sources minimal: they are reduced reproducers, not workloads.
 */
static const bench_case cases[] = {
    {"A math-call: sin(x)+cos(x)",
     "def k(x):\n    return sin(x) + cos(x)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0},
    {"B integer-dtype: int32 x+y",
     "def k(x, y):\n    return x + y\n",
     "[{\"name\":\"x\",\"dtype\":\"int32\"},{\"name\":\"y\",\"dtype\":\"int32\"}]", "[\"numeric\"]",
     "int32", "elementwise", 0, ME_INT32, {"x", "y"}, 2, 0},
    {"C float-operator: x // 3.0",
     "def k(x):\n    return x // 3.0\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0},
    {"D local-temp: y=x*2; y+1",
     "def k(x):\n    y = x * 2.0\n    return y + 1.0\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0},
    {"E control-flow: if x>0 else -x",
     "def k(x):\n    if x > 0.0:\n        return x\n    return -x\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\",\"control-flow\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x"}, 1, 0},
    {"F reduction: sum(x)",
     "def k(x):\n    return sum(x)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\",\"block-reductions\"]",
     "float64", "block_scalar", 0, ME_FLOAT64, {"x"}, 1, 1},
    {"G nd-context: x + _i0",
     "def k(x):\n    return x + _i0\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"}]", "[\"numeric\",\"nd-context\"]",
     "float64", "elementwise", 1, ME_FLOAT64, {"x"}, 1, 0},
    {"control eligible: where(x!=0,y/x,y)",
     "def k(x, y):\n    return where(x != 0, y / x, y)\n",
     "[{\"name\":\"x\",\"dtype\":\"float64\"},{\"name\":\"y\",\"dtype\":\"float64\"}]", "[\"numeric\"]",
     "float64", "elementwise", 0, ME_FLOAT64, {"x", "y"}, 2, 0},
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
        switch (dtype) {
        case ME_FLOAT32: ((float *)buffer)[i] = (float)value; break;
        case ME_INT32: ((int32_t *)buffer)[i] = (int32_t)(i % 1000); break;
        case ME_INT64: ((int64_t *)buffer)[i] = (int64_t)(i % 1000); break;
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

int main(int argc, char **argv) {
    size_t nitems = 1u << 16;
    int repeats = 7;
    if (argc > 1) nitems = (size_t)strtoull(argv[1], NULL, 10);
    if (argc > 2) repeats = atoi(argv[2]);
    if (nitems < 1) nitems = 1;
    if (repeats < 1) repeats = 1;

    printf("MiniExpr portable 1.1: interpreter vs JIT (warm per-evaluation time)\n");
    printf("nitems=%zu repeats=%d  (cold load/compile excluded; best of warm runs; ms per full-array call)\n\n",
           nitems, repeats);
    printf("%-38s %16s %16s %16s %16s\n", "case (family / example)",
           backends[0].label, backends[1].label, backends[2].label, backends[3].label);

    bool backend_compiled[N_BACKENDS] = {false, false, false, false};
    bool value_mismatch = false;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const bench_case *c = &cases[ci];
        char json[4096];
        if (!build_json(c, json, sizeof(json))) {
            printf("%-38s %16s %16s %16s %16s\n", c->label, "BAD-JSON", "", "", "");
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

        char cell[N_BACKENDS][20];
        for (int b = 0; b < N_BACKENDS; b++) {
            bool input_ok = out && reference;
            for (int i = 0; i < c->n_inputs; i++) input_ok = input_ok && in_data[i];
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
                    if (r == 0 && elapsed < 2e6) {
                        batch = (int)(2e6 / elapsed) + 1;
                        if (batch > 100000) batch = 100000;
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
                    snprintf(cell[b], sizeof(cell[b]), "%8.3f %s", best / 1e6, "I");
                }
                else {
                    bool match = ref_bytes == out_bytes && memcmp(reference, out, out_bytes) == 0;
                    if (!match) value_mismatch = true;
                    snprintf(cell[b], sizeof(cell[b]), "%8.3f %s%s", best / 1e6,
                             compiled ? "JIT" : "fb", match ? "" : "*");
                }
            }
            me_artifact_free(artifact);
        }

        printf("%-38s %16s %16s %16s %16s\n", c->label,
               cell[0], cell[1], cell[2], cell[3]);

        for (int i = 0; i < c->n_inputs; i++) free(in_data[i]);
        free(out);
        free(reference);
    }

    printf("\nlegend: I=interpreter  JIT=compiled kernel  fb=JIT declined, interpreter fallback");
    printf("\n        *=warm result differs bitwise from the interpreter reference\n");
    printf("compiled control kernel per backend:");
    for (int b = 1; b < N_BACKENDS; b++) {
        printf("  %s=%s", backends[b].label, backend_compiled[b] ? "yes" : "no");
    }
    printf("\nvalue checks: %s\n", value_mismatch ? "MISMATCH" : "all match interpreter");
    return value_mismatch ? 1 : 0;
}
