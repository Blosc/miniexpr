#include "miniexpr_artifact.h"
#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__EMSCRIPTEN__)
#include <time.h>
#endif
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
EM_JS(double, portable_bench_now_ms, (void), { return performance.now(); });
#if ME_WASM32_SIDE_MODULE
EM_JS(int, portable_wasm32_instantiate,
      (const unsigned char *wasm_bytes, int wasm_len, int bridge_lookup_fn_idx), {
    if (typeof _meJitInstantiate !== "function") return 0;
    var runtime = {
        HEAPF64: HEAPF64, HEAPF32: HEAPF32, wasmMemory: wasmMemory,
        wasmTable: wasmTable, addFunction: addFunction,
        meJitEmitWarnings: !!globalThis.__meJitEmitWarnings, err: err
    };
    if (bridge_lookup_fn_idx) Object.assign(runtime, {
        stackSave: stackSave, stackAlloc: stackAlloc, stackRestore: stackRestore,
        lengthBytesUTF8: lengthBytesUTF8, stringToUTF8: stringToUTF8
    });
    var src = HEAPU8.subarray(wasm_bytes, wasm_bytes + wasm_len);
    return _meJitInstantiate(runtime, src, bridge_lookup_fn_idx,
                             runtime.meJitEmitWarnings) | 0;
});
EM_JS(void, portable_wasm32_free, (int idx), {
    if (typeof _meJitFreeFn === "function")
        _meJitFreeFn({ removeFunction: removeFunction }, idx);
    else if (idx) removeFunction(idx);
});
#endif
#else
static double portable_bench_now_ms(void) {
    struct timespec now;
    assert(clock_gettime(CLOCK_MONOTONIC,&now) == 0);
    return (double)now.tv_sec*1000.0 + (double)now.tv_nsec/1000000.0;
}
#endif

static me_artifact *load_kernel(const char *source, const char *dtype, me_jit_mode mode) {
    char json[2048];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\"],\"source\":\"%s\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"},{\"name\":\"y\",\"dtype\":\"%s\"}],"
        "\"constants\":[],\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":0},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}",
        source, dtype, dtype, dtype);
    assert(n > 0 && (size_t)n < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error = {0};
    if (me_artifact_load(json, (size_t)n, mode, &artifact, &error) != 0) {
        fprintf(stderr, "artifact load failed: %s\n", error.message);
        abort();
    }
    return artifact;
}

static void check_i64(void) {
    const char *source = "def k(x, y):\\n    return x + y\\n";
    int64_t x[] = {INT64_MIN, INT64_MAX, INT64_C(9007199254740993), -1};
    int64_t y[] = {-1, 1, 7, INT64_MAX};
    int64_t expected[] = {INT64_MAX, INT64_MIN, INT64_C(9007199254741000), INT64_MAX - 1};
    int64_t output[4] = {0};
    me_artifact *artifact = load_kernel(source, "int64", ME_JIT_ON);
    assert(me_artifact_has_jit(artifact));
    me_artifact_buffer inputs[] = {
        {"x", ME_INT64, sizeof(int64_t), x, sizeof(x)},
        {"y", ME_INT64, sizeof(int64_t), y, sizeof(y)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 4,
        .output_capacity = sizeof(output),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(artifact, inputs, 2, output, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    assert(memcmp(output, expected, sizeof(output)) == 0);
    me_artifact_free(artifact);
}

static void check_u64(void) {
    const char *source = "def k(x, y):\\n    return x + y\\n";
    uint64_t x[] = {UINT64_MAX, UINT64_C(9007199254740993), UINT64_C(1) << 63, 1};
    uint64_t y[] = {1, 9, 7, UINT64_MAX};
    uint64_t expected[] = {0, UINT64_C(9007199254741002), (UINT64_C(1) << 63) + 7, 0};
    uint64_t output[4] = {0};
    me_artifact *artifact = load_kernel(source, "uint64", ME_JIT_ON);
    assert(me_artifact_has_jit(artifact));
    me_artifact_buffer inputs[] = {
        {"x", ME_UINT64, sizeof(uint64_t), x, sizeof(x)},
        {"y", ME_UINT64, sizeof(uint64_t), y, sizeof(y)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 4,
        .output_capacity = sizeof(output),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(artifact, inputs, 2, output, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    assert(memcmp(output, expected, sizeof(output)) == 0);
    me_artifact_free(artifact);
}

static void check_float_mask(void) {
    const char *source = "def k(x, y):\\n    return x * 2.0 + y\\n";
    double x[] = {1, 100, -2, 4};
    double y[] = {3, 100, 1, -1};
    double output[4] = {0};
    uint8_t mask[] = {1, 0, 1, 1};
    me_artifact *artifact = load_kernel(source, "float64", ME_JIT_ON);
    assert(me_artifact_has_jit(artifact));
    me_artifact_buffer inputs[] = {
        {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 4,
        .output_capacity = sizeof(output), .valid_mask = mask,
        .valid_mask_capacity = sizeof(mask),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(artifact, inputs, 2, output, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    assert(output[0] == 5 && output[2] == -3 && output[3] == 7);
    me_artifact_free(artifact);
}

static void check_unsupported_falls_back(void) {
    const char *source = "def k(x, y):\\n    return x + sin(y)\\n";
    double x[] = {1, 2, 3, 4}, y[] = {0, 1, 2, 3}, output[4] = {0};
    me_artifact *artifact = load_kernel(source, "float64", ME_JIT_ON);
#if defined(__EMSCRIPTEN__)
    assert(!me_artifact_has_jit(artifact));
#endif
    me_artifact_buffer inputs[] = {
        {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 4,
        .output_capacity = sizeof(output),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(artifact, inputs, 2, output, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    assert(output[0] == 1.0 && output[1] > 2.8 && output[1] < 2.9);
    me_artifact_free(artifact);
}

static void benchmark_case(const char *name, const char *source, int n, int repeats,
                           const double *x, const double *y, double *out) {
    size_t bytes = (size_t)n * sizeof(*out);
    me_artifact_buffer inputs[] = {
        {"x", ME_FLOAT64, sizeof(double), x, bytes},
        {"y", ME_FLOAT64, sizeof(double), y, bytes},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = (size_t)n,
        .output_capacity = bytes,
    };
    me_artifact_error error = {0};
    double start = portable_bench_now_ms();
    me_artifact *interp = load_kernel(source, "float64", ME_JIT_OFF);
    double interp_load_ms = portable_bench_now_ms() - start;
    start = portable_bench_now_ms();
    me_artifact *jit = load_kernel(source, "float64", ME_JIT_ON);
    double compile_and_instantiate_ms = portable_bench_now_ms() - start;
    assert(!me_artifact_has_jit(interp));
    if (!me_artifact_has_jit(jit)) {
        printf("WASM_BENCH case=%s route=interpreter (not wasm-eligible)\n", name);
        me_artifact_free(interp);
        me_artifact_free(jit);
        return;
    }

    start = portable_bench_now_ms();
    assert(me_artifact_eval_ex(interp, inputs, 2, out, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    double interp_first_ms = portable_bench_now_ms() - start;
    double *expected = malloc(bytes);
    assert(expected);
    memcpy(expected, out, bytes);
    start = portable_bench_now_ms();
    assert(me_artifact_eval_ex(jit, inputs, 2, out, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    double jit_first_ms = portable_bench_now_ms() - start;
    assert(memcmp(out, expected, bytes) == 0);

    start = portable_bench_now_ms();
    for (int i = 0; i < repeats; i++)
        assert(me_artifact_eval_ex(interp, inputs, 2, out, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    double interp_warm_ms = (portable_bench_now_ms() - start) / repeats;
    start = portable_bench_now_ms();
    for (int i = 0; i < repeats; i++)
        assert(me_artifact_eval_ex(jit, inputs, 2, out, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    double jit_warm_ms = (portable_bench_now_ms() - start) / repeats;
    double speedup = interp_warm_ms / jit_warm_ms;
    double break_even = speedup > 1.0 ?
        compile_and_instantiate_ms / (interp_warm_ms - jit_warm_ms) : -1.0;
     printf("PORTABLE_BENCH case=%s n=%d repeats=%d interpreter_load_ms=%.3f "
           "tcc_compile_plus_module_instantiate_ms=%.3f interpreter_first_ms=%.3f "
           "jit_first_ms=%.3f interpreter_warm_ms=%.3f jit_warm_ms=%.3f "
           "warm_speedup=%.2fx reuse_break_even_calls=%.1f\n",
           name, n, repeats, interp_load_ms, compile_and_instantiate_ms, interp_first_ms,
           jit_first_ms, interp_warm_ms, jit_warm_ms, speedup, break_even);
    free(expected);
    me_artifact_free(interp);
    me_artifact_free(jit);
}

static void benchmark_reuse(void) {
    enum { N = 1048576, REPEATS = 24 };
    static const struct {
        const char *name;
        const char *source;
    } cases[] = {
        {"affine", "def k(x, y):\\n    return x * 1.25 + y * 0.75\\n"},
        {"arithmetic-heavy", "def k(x, y):\\n    return x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y\\n"},
        {"arithmetic-heavy-cse", "def k(x, y):\\n    p = x*y\\n    return p + p + p + p + p + p + p + p\\n"},
        {"bounded-loop", "def k(x, y):\\n    v = x\\n    for i in range(16):\\n        v = v + y\\n    return v\\n"},
        {"branch", "def k(x, y):\\n    if x > 0.0:\\n        return x + y\\n    else:\\n        return x - y\\n"},
    };
    double *x = malloc((size_t)N * sizeof(*x));
    double *y = malloc((size_t)N * sizeof(*y));
    double *out = malloc((size_t)N * sizeof(*out));
    assert(x && y && out);
    for (int i = 0; i < N; i++) {
        x[i] = (i & 1) ? i * 0.125 : -i * 0.125;
        y[i] = 3.0 - i * 0.0625;
    }
    enum { LARGE_N = 10 * N, LARGE_REPEATS = 6 };
    double *large_x = malloc((size_t)LARGE_N * sizeof(*large_x));
    double *large_y = malloc((size_t)LARGE_N * sizeof(*large_y));
    double *large_out = malloc((size_t)LARGE_N * sizeof(*large_out));
    assert(large_x && large_y && large_out);
    for (int i = 0; i < LARGE_N; i++) {
        large_x[i] = (i & 1) ? i * 0.125 : -i * 0.125;
        large_y[i] = 3.0 - i * 0.0625;
    }
    benchmark_case("affine-10x", cases[0].source, LARGE_N, LARGE_REPEATS,
                   large_x, large_y, large_out);
    free(large_x); free(large_y); free(large_out);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        benchmark_case(cases[i].name, cases[i].source, N, REPEATS, x, y, out);
    free(x); free(y); free(out);
}

int main(void) {
#if defined(__EMSCRIPTEN__)
#if ME_WASM32_SIDE_MODULE
    me_register_wasm_jit_helpers(portable_wasm32_instantiate, portable_wasm32_free);
#endif
    check_i64();
    check_u64();
    check_float_mask();
    check_unsupported_falls_back();
    benchmark_reuse();
#if ME_WASM32_SIDE_MODULE
    me_register_wasm_jit_helpers(NULL, NULL);
#endif
    puts("PASS: portable wasm32 TCC JIT exact-i64 arithmetic and masking");
#else
    check_i64();
    check_u64();
    check_float_mask();
    check_unsupported_falls_back();
    benchmark_reuse();
    puts("PASS: portable TCC JIT exact arithmetic and masking");
#endif
    return 0;
}
