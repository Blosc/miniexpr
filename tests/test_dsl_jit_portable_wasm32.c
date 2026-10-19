/* Differential first-slice qualification for portable Menudet kernels.
 *
 * The same source is prepared with JIT disabled and enabled, then both outputs
 * are compared. Under Emscripten, eligible cases must report an actual wasm JIT
 * kernel; unsupported cases must fall back without changing values. The file
 * also compiles natively, where it doubles as the JIT benchmark driver. */
#include "miniexpr_artifact.h"
#include "dsl_jit_test.h"
#undef NDEBUG
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__EMSCRIPTEN__)
#include <time.h>
#else
#include <emscripten.h>
#endif

static void compare_outputs(me_dtype type, const void *a, const void *b, size_t n);

#if defined(__EMSCRIPTEN__)
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
static int portable_wasm32_reject(const unsigned char *wasm_bytes, int wasm_len,
                                  int bridge_lookup_fn_idx) {
    (void)wasm_bytes;
    (void)wasm_len;
    (void)bridge_lookup_fn_idx;
    return 0;
}
#endif
#else
static double portable_bench_now_ms(void) {
    struct timespec now;
    assert(timespec_get(&now, TIME_UTC) == TIME_UTC);
    return (double)now.tv_sec*1000.0 + (double)now.tv_nsec/1000000.0;
}
#endif

typedef enum {
    ROUTE_JIT,
    ROUTE_INTERPRETER
} route_expectation;

static size_t type_width(me_dtype type) {
    switch (type) {
        case ME_BOOL: case ME_INT8: case ME_UINT8: return 1;
        case ME_INT16: case ME_UINT16: return 2;
        case ME_INT32: case ME_UINT32: case ME_FLOAT32: return 4;
        case ME_INT64: case ME_UINT64: case ME_FLOAT64: return 8;
        default: return 0;
    }
}

static me_artifact *load_elementwise(const char *source, const char *dtype,
                                     const char *out_dtype, me_jit_mode mode) {
    char json[8192];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\"],\"source\":\"%s\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"},{\"name\":\"y\",\"dtype\":\"%s\"}],"
        "\"constants\":[],\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":0},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}",
        source, dtype, dtype, out_dtype);
    assert(n > 0 && (size_t)n < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error = {0};
    if (me_artifact_load(json, (size_t)n, mode, &artifact, &error) != 0) {
        fprintf(stderr, "artifact load failed for %s: %s\n", source, error.message);
        abort();
    }
    return artifact;
}

static me_artifact *load_reduction(const char *source, const char *dtype,
                                   const char *out_dtype, me_jit_mode mode) {
    char json[8192];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"block-reductions\"],\"source\":\"%s\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"}],"
        "\"constants\":[],\"output\":{\"dtype\":\"%s\",\"contract\":\"block_scalar\"},"
        "\"context\":{\"ndim\":0},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}",
        source, dtype, out_dtype);
    assert(n > 0 && (size_t)n < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error = {0};
    if (me_artifact_load(json, (size_t)n, mode, &artifact, &error) != 0) {
        fprintf(stderr, "reduction load failed: %s\n", error.message);
        abort();
    }
    return artifact;
}

static me_artifact *load_nd(const char *source, const char *dtype,
                            const char *out_dtype, me_jit_mode mode) {
    char json[8192];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"nd-context\"],\"source\":\"%s\","
        "\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"},{\"name\":\"y\",\"dtype\":\"%s\"}],"
        "\"constants\":[],\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":2},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}",
        source, dtype, dtype, out_dtype);
    assert(n > 0 && (size_t)n < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error = {0};
    if (me_artifact_load(json, (size_t)n, mode, &artifact, &error) != 0) {
        fprintf(stderr, "ND load failed: %s\n", error.message);
        abort();
    }
    return artifact;
}

static me_artifact *load_mixed_four(me_jit_mode mode) {
    const char json[] =
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\"],\"source\":\"def k(a1, a2, a3, a4):\\n    return a1 + a2 - a3 * a4\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"a1\",\"dtype\":\"int8\"},"
        "{\"name\":\"a2\",\"dtype\":\"int16\"},{\"name\":\"a3\",\"dtype\":\"int8\"},"
        "{\"name\":\"a4\",\"dtype\":\"int16\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"int16\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":0},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}";
    me_artifact *artifact = NULL;
    me_artifact_error error = {0};
    if (me_artifact_load(json, sizeof(json) - 1, mode, &artifact, &error) != 0) {
        fprintf(stderr, "mixed load failed: %s\n", error.message);
        abort();
    }
    return artifact;
}

static void check_mixed_four(void) {
    int8_t a1[] = {0, 1, 2, 3, INT8_MIN, INT8_MAX, -1, 17};
    int16_t a2[] = {10, 11, 12, 13, INT16_MAX, INT16_MIN, -30000, 23456};
    int8_t a3[] = {0, 1, 2, 3, INT8_MAX, INT8_MIN, -1, 101};
    int16_t a4[] = {10, 11, 12, 13, INT16_MIN, INT16_MAX, -32767, 30000};
    int16_t actual[8] = {0}, expected[8] = {0};
    me_artifact_buffer inputs[] = {
        {"a1", ME_INT8, sizeof(int8_t), a1, sizeof(a1)},
        {"a2", ME_INT16, sizeof(int16_t), a2, sizeof(a2)},
        {"a3", ME_INT8, sizeof(int8_t), a3, sizeof(a3)},
        {"a4", ME_INT16, sizeof(int16_t), a4, sizeof(a4)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 8,
        .output_capacity = sizeof(actual),
    };
    me_artifact_error error = {0};
    me_artifact *jit = load_mixed_four(ME_JIT_ON);
    me_artifact *interp = load_mixed_four(ME_JIT_OFF);
#if defined(__EMSCRIPTEN__)
    assert(me_artifact_has_jit(jit));
#endif
    assert(me_artifact_eval_ex(jit, inputs, 4, actual, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    descriptor.output_capacity = sizeof(expected);
    assert(me_artifact_eval_ex(interp, inputs, 4, expected, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    compare_outputs(ME_INT16, actual, expected, 8);
    me_artifact_free(jit);
    me_artifact_free(interp);
}

static bool same_value(me_dtype type, const unsigned char *a, const unsigned char *b) {
    if (type == ME_FLOAT32) {
        float x, y;
        memcpy(&x, a, sizeof(x));
        memcpy(&y, b, sizeof(y));
        if (isnan(x) && isnan(y)) return true;
        return memcmp(a, b, sizeof(x)) == 0;
    }
    if (type == ME_FLOAT64) {
        double x, y;
        memcpy(&x, a, sizeof(x));
        memcpy(&y, b, sizeof(y));
        if (isnan(x) && isnan(y)) return true;
        return memcmp(a, b, sizeof(x)) == 0;
    }
    return memcmp(a, b, type_width(type)) == 0;
}

static void compare_outputs(me_dtype type, const void *a, const void *b, size_t n) {
    size_t width = type_width(type);
    for (size_t i = 0; i < n; i++) {
        if (!same_value(type, (const unsigned char *)a + i*width,
                        (const unsigned char *)b + i*width)) {
            fprintf(stderr, "output mismatch at lane %zu\n", i);
            assert(0);
        }
    }
}

/* Inactive lanes are not written. Verify both sentinels survive and compare
 * only participating lanes. */
static void compare_masked_outputs(me_dtype type, const void *a, const void *b,
                                   size_t n, const uint8_t *mask) {
    if (!mask) {
        compare_outputs(type, a, b, n);
        return;
    }
    size_t width = type_width(type);
    for (size_t i = 0; i < n; i++) {
        const unsigned char *pa = (const unsigned char *)a + i*width;
        const unsigned char *pb = (const unsigned char *)b + i*width;
        if (!mask[i]) {
            for (size_t k = 0; k < width; k++) assert(pa[k] == 0xA5 && pb[k] == 0x5A);
            continue;
        }
        if (!same_value(type, pa, pb)) {
            fprintf(stderr, "masked output mismatch at lane %zu\n", i);
            assert(0);
        }
    }
}

static me_artifact_status eval_elementwise(const me_artifact *artifact,
        const void *x, const void *y, size_t n, size_t in_width, size_t out_width,
        void *output, const uint8_t *mask, size_t mask_capacity,
        int ndim, const int64_t *shape) {
    me_artifact_buffer inputs[2] = {
        {"x", me_artifact_input_dtype(artifact,0), in_width, x, n*in_width},
        {"y", me_artifact_input_dtype(artifact,1), in_width, y, n*in_width},
    };
    int64_t origin[2] = {0,0}, extent[2] = {0,0};
    if (shape) {
        extent[0] = shape[0];
        extent[1] = shape[1];
    }
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = n,
        .output_capacity = n*out_width,
        .valid_mask = mask, .valid_mask_capacity = mask_capacity,
        .ndim = ndim,
    };
    if (ndim) {
        descriptor.logical_shape = shape;
        descriptor.block_origin = origin;
        descriptor.block_extent = extent;
    }
    me_artifact_error error = {0};
    return me_artifact_eval_ex(artifact, inputs, 2, output, &descriptor, &error);
}

static void check_route(const char *name, bool has_jit, route_expectation expected) {
    printf("WASM_CASE name=%s route=%s\n", name, has_jit ? "jit" : "interpreter");
    fflush(stdout);
#if defined(__EMSCRIPTEN__)
    if (expected == ROUTE_JIT) assert(has_jit);
    if (expected == ROUTE_INTERPRETER) assert(!has_jit);
#else
    (void)expected;
#endif
}

static void check_equiv(const char *name, const char *source, const char *dtype,
                        const char *out_dtype, const void *x, const void *y,
                        size_t n, const uint8_t *mask, size_t mask_capacity,
                        route_expectation expected) {
    me_artifact *interp = load_elementwise(source, dtype, out_dtype, ME_JIT_OFF);
    me_artifact *jit = load_elementwise(source, dtype, out_dtype, ME_JIT_ON);
    assert(!me_artifact_has_jit(interp));
    check_route(name, me_artifact_has_jit(jit), expected);
    size_t in_width = type_width(me_artifact_input_dtype(jit,0));
    me_dtype out_type = me_artifact_output_dtype(jit);
    size_t out_width = type_width(out_type);
    if (n) {
        assert(me_artifact_input_itemsize(jit,0) == in_width);
        assert(me_artifact_input_itemsize(jit,1) == in_width);
        assert(me_artifact_output_itemsize(jit) == out_width);
    }
    unsigned char *expected_out = malloc(n ? n*out_width : 1);
    unsigned char *actual_out = malloc(n ? n*out_width : 1);
    assert(expected_out && actual_out);
    memset(expected_out, 0xA5, n ? n*out_width : 1);
    memset(actual_out, 0x5A, n ? n*out_width : 1);
    me_artifact_status a = eval_elementwise(interp, x, y, n, in_width, out_width,
                                            expected_out, mask, mask_capacity, 0, NULL);
    me_artifact_status b = eval_elementwise(jit, x, y, n, in_width, out_width,
                                            actual_out, mask, mask_capacity, 0, NULL);
    assert(a == b);
    assert(a == ME_ARTIFACT_SUCCESS);
    compare_masked_outputs(out_type, expected_out, actual_out, n, mask);
    free(expected_out);
    free(actual_out);
    me_artifact_free(interp);
    me_artifact_free(jit);
}

static void check_reduction_fallback(void) {
    const char *source = "def k(x):\\n    return block_sum(x)\\n";
    double x[] = {1, 2, 3, 4};
    double expected = 10.0, actual = 0;
    me_artifact *interp = load_reduction(source, "float64", "float64", ME_JIT_OFF);
    me_artifact *jit = load_reduction(source, "float64", "float64", ME_JIT_ON);
    check_route("reduction-fallback", me_artifact_has_jit(jit), ROUTE_INTERPRETER);
    me_artifact_buffer inputs[] = {{"x", ME_FLOAT64, sizeof(double), x, sizeof(x)}};
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 4,
        .output_capacity = sizeof(double),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(interp, inputs, 1, &expected, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    assert(me_artifact_eval_ex(jit, inputs, 1, &actual, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    assert(expected == actual);
    me_artifact_free(interp);
    me_artifact_free(jit);
}

static void check_nd_fallback(void) {
    const char *source = "def k(x, y):\\n    return x + _flat_idx + y\\n";
    double x[] = {1, 2, 3, 4}, y[] = {1, 1, 1, 1}, out_a[4] = {0}, out_b[4] = {0};
    int64_t shape[] = {2, 2};
    me_artifact *interp = load_nd(source, "float64", "float64", ME_JIT_OFF);
    me_artifact *jit = load_nd(source, "float64", "float64", ME_JIT_ON);
    check_route("nd-fallback", me_artifact_has_jit(jit), ROUTE_INTERPRETER);
    assert(eval_elementwise(interp, x, y, 4, 8, 8, out_a, NULL, 0, 2, shape) == ME_ARTIFACT_SUCCESS);
    assert(eval_elementwise(jit, x, y, 4, 8, 8, out_b, NULL, 0, 2, shape) == ME_ARTIFACT_SUCCESS);
    compare_outputs(ME_FLOAT64, out_a, out_b, 4);
    me_artifact_free(interp);
    me_artifact_free(jit);
}

static void check_integer_boundaries(void) {
    const int64_t ix[] = {INT64_MIN, INT64_MAX, INT64_C(9007199254740993), -1, 0,
                          INT64_C(1) << 62, -3, 7, INT32_MIN, INT32_MAX,
                          INT64_C(9007199254740991), INT64_C(4294967296)};
    const int64_t iy[] = {-1, 1, 7, INT64_MAX, 0, 3, 5, -9, -1, 1, 1, -1};
    size_t n = sizeof(ix)/sizeof(ix[0]);
    check_equiv("i64-add", "def k(x, y):\\n    return x + y\\n", "int64", "int64",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-sub", "def k(x, y):\\n    return x - y\\n", "int64", "int64",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-mul", "def k(x, y):\\n    return x * y\\n", "int64", "int64",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-neg", "def k(x, y):\\n    return -x\\n", "int64", "int64",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-and", "def k(x, y):\\n    return x & y\\n", "int64", "int64",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-or", "def k(x, y):\\n    return x | y\\n", "int64", "int64",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-xor", "def k(x, y):\\n    return x ^ y\\n", "int64", "int64",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-shift-left-literal", "def k(x, y):\\n    return x << 1\\n", "int64",
                "int64", ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-shift-right-literal", "def k(x, y):\\n    return x >> 1\\n", "int64",
                "int64", ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-lt", "def k(x, y):\\n    return x < y\\n", "int64", "bool",
                ix, iy, n, NULL, 0, ROUTE_JIT);
    check_equiv("i64-eq", "def k(x, y):\\n    return x == y\\n", "int64", "bool",
                ix, iy, n, NULL, 0, ROUTE_JIT);

    const uint64_t ux[] = {UINT64_MAX, UINT64_C(9007199254740993), UINT64_C(1) << 63,
                           1, 0, UINT64_C(1) << 32, 3, 11,
                           UINT64_C(9007199254740991), UINT64_C(1) << 53,
                           UINT32_MAX, UINT64_C(1) << 32};
    const uint64_t uy[] = {1, 9, 7, UINT64_MAX, 0, UINT64_C(1) << 33, 5, 13,
                           1, 1, 1, UINT32_MAX};
    n = sizeof(ux)/sizeof(ux[0]);
    check_equiv("u64-add", "def k(x, y):\\n    return x + y\\n", "uint64", "uint64",
                ux, uy, n, NULL, 0, ROUTE_JIT);
    check_equiv("u64-sub", "def k(x, y):\\n    return x - y\\n", "uint64", "uint64",
                ux, uy, n, NULL, 0, ROUTE_JIT);
    check_equiv("u64-mul", "def k(x, y):\\n    return x * y\\n", "uint64", "uint64",
                ux, uy, n, NULL, 0, ROUTE_JIT);
    check_equiv("u64-neg", "def k(x, y):\\n    return -x\\n", "uint64", "uint64",
                ux, uy, n, NULL, 0, ROUTE_JIT);
}

static void check_float_values(void) {
    const double dx[] = {0.0, -0.0, 1.0, -1.0, INFINITY, -INFINITY, NAN,
                         -NAN, 1.0e-308, 5.0e-324, 1.0e308, -1.0e308};
    const double dy[] = {0.0, 0.0, 2.5, -2.5, 1.0, 1.0, 1.0,
                         1.0, 1.0e-308, 5.0e-324, 0.5, 0.5};
    size_t n = sizeof(dx)/sizeof(dx[0]);
    check_equiv("f64-add-specials", "def k(x, y):\\n    return x + y\\n", "float64",
                "float64", dx, dy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f64-sub-specials", "def k(x, y):\\n    return x - y\\n", "float64",
                "float64", dx, dy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f64-mul-specials", "def k(x, y):\\n    return x * y\\n", "float64",
                "float64", dx, dy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f64-div-specials", "def k(x, y):\\n    return x / y\\n", "float64",
                "float64", dx, dy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f64-gt", "def k(x, y):\\n    return x > y\\n", "float64", "bool",
                dx, dy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f64-ne", "def k(x, y):\\n    return x != y\\n", "float64", "bool",
                dx, dy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f64-affine", "def k(x, y):\\n    return x * 1.25 + y * 0.75\\n",
                "float64", "float64", dx, dy, n, NULL, 0, ROUTE_JIT);

    const float fx[] = {0.0f, -0.0f, 1.0f, -1.0f, INFINITY, -INFINITY, NAN,
                        -NAN, 1.0e-38f, 1.0e-45f, 3.0e38f, -3.0e38f};
    const float fy[] = {0.0f, 0.0f, 2.5f, -2.5f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0e-38f, 1.0e-45f, 0.5f, 0.5f};
    check_equiv("f32-add-specials", "def k(x, y):\\n    return x + y\\n", "float32",
                "float32", fx, fy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f32-mul-specials", "def k(x, y):\\n    return x * y\\n", "float32",
                "float32", fx, fy, n, NULL, 0, ROUTE_JIT);
    check_equiv("f32-gt", "def k(x, y):\\n    return x > y\\n", "float32", "bool",
                fx, fy, n, NULL, 0, ROUTE_JIT);
}

static void check_control_flow(void) {
    double x[] = {-3.0, -1.0, 0.0, 1.0, 2.5, -0.0, NAN, INFINITY};
    double y[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    size_t n = sizeof(x)/sizeof(x[0]);
    check_equiv("branch-if",
                "def k(x, y):\\n    if x > 0.0:\\n        return x + y\\n"
                "    else:\\n        return x - y\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
    check_equiv("branch-elif",
                "def k(x, y):\\n    if x > 1.0:\\n        return x + y\\n"
                "    elif x < 0.0:\\n        return x - y\\n    return y\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
    check_equiv("where", "def k(x, y):\\n    return where(x > 0.0, x + y, x - y)\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
    check_equiv("where-nested",
                "def k(x, y):\\n    return where(x != 0.0, where(y > 0.0, x / y, -x), y)\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
    check_equiv("where-deep",
                "def k(x, y):\\n    return where(x > 0.0, where(y > 0.0, "
                "where(x > y, x + y, x - y), where(y < 0.0, x * y, y)), x)\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
    check_equiv("bounded-loop",
                "def k(x, y):\\n    v = x\\n    for i in range(16):\\n"
                "        v = v + y\\n    return v\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
    check_equiv("while-loop",
                "def k(x, y):\\n    v = x\\n    i = 0\\n"
                "    while i < 8:\\n        v = v * 0.5 + y\\n        i = i + 1\\n"
                "    return v\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
}

static void check_masks(void) {
    double x[] = {1, 100, -2, 4, 0, NAN};
    double y[] = {3, 100, 1, -1, 0, 1};
    size_t n = sizeof(x)/sizeof(x[0]);
    uint8_t mixed[] = {1, 0, 1, 1, 0, 1};
    uint8_t all_active[] = {1, 1, 1, 1, 1, 1};
    uint8_t all_inactive[] = {0, 0, 0, 0, 0, 0};
    check_equiv("mask-mixed", "def k(x, y):\\n    return x * 2.0 + y\\n", "float64",
                "float64", x, y, n, mixed, sizeof(mixed), ROUTE_JIT);
    check_equiv("mask-all-active", "def k(x, y):\\n    return x * 2.0 + y\\n",
                "float64", "float64", x, y, n, all_active, sizeof(all_active),
                ROUTE_JIT);
    check_equiv("mask-all-inactive", "def k(x, y):\\n    return x * 2.0 + y\\n",
                "float64", "float64", x, y, n, all_inactive, sizeof(all_inactive),
                ROUTE_JIT);
    /* Inactive lanes must not evaluate the division, even for a zero divisor. */
    check_equiv("mask-lazy-division", "def k(x, y):\\n    return y / x\\n", "float64",
                "float64", x, y, n, mixed, sizeof(mixed), ROUTE_JIT);
}

static void check_sizes(void) {
    static const size_t sizes[] = {0, 1, 2, 3, 17, 1023, 1024, 1025, 4096};
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        size_t n = sizes[s];
        double *x = n ? malloc(n*sizeof(double)) : NULL;
        double *y = n ? malloc(n*sizeof(double)) : NULL;
        assert(n == 0 || (x && y));
        for (size_t i = 0; i < n; i++) {
            x[i] = (double)i * 0.25 - 3.0;
            y[i] = (double)(n - i) * 0.5 + 1.0;
        }
        char name[64];
        snprintf(name, sizeof(name), "size-%zu", n);
        check_equiv(name, "def k(x, y):\\n    return x * 1.25 + y * 0.75\\n",
                    "float64", "float64", x, y, n, NULL, 0, ROUTE_JIT);
        free(x);
        free(y);
    }
}

static void check_fallbacks(void) {
    double x[] = {1, 2, 3, 4}, y[] = {0, 1, 2, 3};
    size_t n = 4;
    check_equiv("math-fallback", "def k(x, y):\\n    return x + sin(y)\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_INTERPRETER);
    check_equiv("pow-fallback", "def k(x, y):\\n    return x ** y\\n",
                "float64", "float64", x, y, n, NULL, 0, ROUTE_INTERPRETER);
    check_equiv("floordiv-inline", "def k(x, y):\\n    return x // y\\n",
                "int64", "int64", (const int64_t[]){7,-7,7,0}, (const int64_t[]){3,-3,0,5},
                n, NULL, 0, ROUTE_JIT);
    check_equiv("predicate-fallback", "def k(x, y):\\n    return isnan(x)\\n",
                "float64", "bool", x, y, n, NULL, 0, ROUTE_INTERPRETER);
    check_equiv("cast-fallback", "def k(x, y):\\n    return int(x)\\n",
                "float64", "int64", x, y, n, NULL, 0, ROUTE_INTERPRETER);
    check_equiv("mixed-compare-fallback", "def k(x, y):\\n    return x < -1\\n",
                "uint64", "bool",
                (const uint64_t[]){0, 1, UINT64_MAX, UINT64_C(1) << 63},
                (const uint64_t[]){1, 1, 1, 1}, n, NULL, 0, ROUTE_INTERPRETER);
    check_reduction_fallback();
    check_nd_fallback();
}

static void check_binding_limits(void) {
    double x[] = {1, 2, 3, 4}, y[] = {1, 1, 1, 1}, out[4] = {0};
    me_artifact *artifact = load_elementwise(
        "def k(x, y):\\n    return x + y\\n", "float64", "float64", ME_JIT_ON);
    me_artifact_buffer inputs[] = {
        {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    me_artifact_error error = {0};
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1,
        .nitems = (size_t)INT32_MAX + 1, .output_capacity = sizeof(out),
    };
    me_artifact_status status = me_artifact_eval_ex(artifact, inputs, 2, out,
                                                   &descriptor, &error);
    assert(status != ME_ARTIFACT_SUCCESS);
    /* The contiguous zero-copy adapter requires natural alignment; an offset
     * pointer is rejected rather than passed to typed generated loads. */
    union {
        uint64_t alignment;
        unsigned char bytes[sizeof(x) + 1];
    } unaligned_storage;
    me_artifact_buffer unaligned[] = {
        {"x", ME_FLOAT64, sizeof(double), unaligned_storage.bytes + 1, sizeof(x)},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    descriptor.nitems = 4;
    descriptor.output_capacity = sizeof(out);
    status = me_artifact_eval_ex(artifact, unaligned, 2, out, &descriptor, &error);
    assert(status != ME_ARTIFACT_SUCCESS);
    union {
        uint64_t alignment;
        unsigned char bytes[sizeof(out) + 1];
    } unaligned_output;
    status = me_artifact_eval_ex(artifact, inputs, 2, unaligned_output.bytes + 1,
                                 &descriptor, &error);
    assert(status != ME_ARTIFACT_SUCCESS);
    /* Undersized output, input and mask capacities are rejected before entry. */
    descriptor.output_capacity = sizeof(out) - 1;
    status = me_artifact_eval_ex(artifact, inputs, 2, out, &descriptor, &error);
    assert(status != ME_ARTIFACT_SUCCESS);
    descriptor.output_capacity = sizeof(out);
    me_artifact_buffer short_input[] = {
        {"x", ME_FLOAT64, sizeof(double), x, sizeof(x) - 1},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    status = me_artifact_eval_ex(artifact, short_input, 2, out, &descriptor, &error);
    assert(status != ME_ARTIFACT_SUCCESS);
    uint8_t short_mask[] = {1, 1, 1};
    descriptor.valid_mask = short_mask;
    descriptor.valid_mask_capacity = sizeof(short_mask);
    status = me_artifact_eval_ex(artifact, inputs, 2, out, &descriptor, &error);
    assert(status != ME_ARTIFACT_SUCCESS);
    descriptor.valid_mask = NULL;
    descriptor.valid_mask_capacity = 0;
    /* Overlapping output and input storage is invalid. */
    status = me_artifact_eval_ex(artifact, inputs, 2, x, &descriptor, &error);
    assert(status != ME_ARTIFACT_SUCCESS);
    me_artifact_free(artifact);
}

static void check_cache_cycles(void) {
    /* More unique kernels than process-cache slots exercises eviction and
     * function-table/scratch cleanup. */
    const int unique = ME_DSL_JIT_WASM_POS_CACHE_SLOTS + 6;
    for (int i = 0; i < unique; i++) {
        char source[256];
        snprintf(source, sizeof(source),
                 "def k(x, y):\\n    return x + %.1f\\n", (double)(i + 1));
        double x[] = {1.0}, y[] = {2.0}, out = 0;
        me_artifact *artifact = load_elementwise(source, "float64", "float64", ME_JIT_ON);
#if defined(__EMSCRIPTEN__)
        assert(me_artifact_has_jit(artifact));
#endif
        me_artifact_buffer inputs[] = {
            {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
            {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
        };
        me_artifact_eval_descriptor descriptor = {
            .struct_size = sizeof(descriptor), .version = 1, .nitems = 1,
            .output_capacity = sizeof(out),
        };
        me_artifact_error error = {0};
        assert(me_artifact_eval_ex(artifact, inputs, 2, &out, &descriptor, &error) ==
               ME_ARTIFACT_SUCCESS);
        assert(out == 1.0 + (double)(i + 1));
        me_artifact_free(artifact);
    }
#if defined(__EMSCRIPTEN__)
    /* Eviction must not invalidate a kernel that is still owned by a live
     * artifact. Fill every cache slot with live programs, then execute the
     * oldest one after additional programs have fallen back to private slots. */
    me_artifact *active[ME_DSL_JIT_WASM_POS_CACHE_SLOTS + 6] = {0};
    for (int i = 0; i < unique; i++) {
        char source[256];
        snprintf(source, sizeof(source),
                 "def k(x, y):\\n    return x + %.1f\\n", (double)(i + 101));
        active[i] = load_elementwise(source, "float64", "float64", ME_JIT_ON);
        assert(me_artifact_has_jit(active[i]));
        double x[] = {1.0}, y[] = {2.0}, out = 0;
        me_artifact_buffer inputs[] = {
            {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
            {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
        };
        me_artifact_eval_descriptor descriptor = {
            .struct_size = sizeof(descriptor), .version = 1, .nitems = 1,
            .output_capacity = sizeof(out),
        };
        me_artifact_error error = {0};
        assert(me_artifact_eval_ex(active[i], inputs, 2, &out, &descriptor, &error) ==
               ME_ARTIFACT_SUCCESS);
        assert(out == 1.0 + (double)(i + 101));
    }
    {
        double x[] = {1.0}, y[] = {2.0}, out = 0;
        me_artifact_buffer inputs[] = {
            {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
            {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
        };
        me_artifact_eval_descriptor descriptor = {
            .struct_size = sizeof(descriptor), .version = 1, .nitems = 1,
            .output_capacity = sizeof(out),
        };
        me_artifact_error error = {0};
        assert(me_artifact_eval_ex(active[0], inputs, 2, &out, &descriptor, &error) ==
               ME_ARTIFACT_SUCCESS);
        assert(out == 102.0);
    }
    for (int i = 0; i < unique; i++) me_artifact_free(active[i]);
#endif
    /* Reusing one source must keep returning an executeable kernel. */
    for (int i = 0; i < 16; i++) {
        double x[] = {2.0}, y[] = {3.0}, out = 0;
        me_artifact *artifact = load_elementwise(
            "def k(x, y):\\n    return x * y + 1.0\\n", "float64", "float64", ME_JIT_ON);
#if defined(__EMSCRIPTEN__)
        assert(me_artifact_has_jit(artifact));
#endif
        me_artifact_buffer inputs[] = {
            {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
            {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
        };
        me_artifact_eval_descriptor descriptor = {
            .struct_size = sizeof(descriptor), .version = 1, .nitems = 1,
            .output_capacity = sizeof(out),
        };
        me_artifact_error error = {0};
        assert(me_artifact_eval_ex(artifact, inputs, 2, &out, &descriptor, &error) ==
               ME_ARTIFACT_SUCCESS);
        assert(out == 7.0);
        me_artifact_free(artifact);
    }
}

static void check_forced_invalid(void) {
#if defined(__EMSCRIPTEN__) && !ME_WASM32_SIDE_MODULE
    setenv("ME_WASM_JIT_TEST_FORCE_INVALID", "1", 1);
    double x[] = {4.0}, y[] = {5.0}, out = 0;
    me_artifact *artifact = load_elementwise(
        "def k(x, y):\\n    return x + y + 12345.5\\n", "float64", "float64", ME_JIT_ON);
    check_route("forced-invalid-fallback", me_artifact_has_jit(artifact),
                ROUTE_INTERPRETER);
    me_artifact_buffer inputs[] = {
        {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 1,
        .output_capacity = sizeof(out),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(artifact, inputs, 2, &out, &descriptor, &error) ==
           ME_ARTIFACT_SUCCESS);
    assert(out == 12354.5);
    unsetenv("ME_WASM_JIT_TEST_FORCE_INVALID");
    me_artifact_free(artifact);
#endif
}

static void check_side_helper_failure(void) {
#if defined(__EMSCRIPTEN__) && ME_WASM32_SIDE_MODULE
    me_register_wasm_jit_helpers(portable_wasm32_reject, portable_wasm32_free);
    double x[] = {4.0}, y[] = {5.0}, out = 0;
    me_artifact *artifact = load_elementwise(
        "def k(x, y):\\n    return x + y + 23456.5\\n", "float64", "float64", ME_JIT_ON);
    check_route("side-helper-failure-fallback", me_artifact_has_jit(artifact),
                ROUTE_INTERPRETER);
    me_artifact_buffer inputs[] = {
        {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 1,
        .output_capacity = sizeof(out),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(artifact, inputs, 2, &out, &descriptor, &error) ==
           ME_ARTIFACT_SUCCESS);
    assert(out == 23465.5);
    me_artifact_free(artifact);
    me_register_wasm_jit_helpers(portable_wasm32_instantiate, portable_wasm32_free);
#endif
}

static void check_helper_unregistration(void) {
#if defined(__EMSCRIPTEN__) && ME_WASM32_SIDE_MODULE
    dsl_jit_wasm_pos_cache_reset_for_tests();
    me_register_wasm_jit_helpers(NULL, NULL);
    double x[] = {6.0}, y[] = {7.0}, out = 0;
    me_artifact *artifact = load_elementwise(
        "def k(x, y):\\n    return x + y + 54321.0\\n", "float64", "float64", ME_JIT_ON);
    check_route("unregistered-helper-fallback", me_artifact_has_jit(artifact),
                ROUTE_INTERPRETER);
    me_artifact_buffer inputs[] = {
        {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)},
        {"y", ME_FLOAT64, sizeof(double), y, sizeof(y)},
    };
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1, .nitems = 1,
        .output_capacity = sizeof(out),
    };
    me_artifact_error error = {0};
    assert(me_artifact_eval_ex(artifact, inputs, 2, &out, &descriptor, &error) ==
           ME_ARTIFACT_SUCCESS);
    assert(out == 54334.0);
    me_artifact_free(artifact);
    me_register_wasm_jit_helpers(portable_wasm32_instantiate, portable_wasm32_free);
#endif
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
    me_artifact *interp = load_elementwise(source, "float64", "float64", ME_JIT_OFF);
    double interp_load_ms = portable_bench_now_ms() - start;
#if defined(__EMSCRIPTEN__)
    /* Correctness cases above may have populated the process cache with the
     * same source. Reset it so this timing really includes TCC and module
     * instantiation rather than a positive-cache bind. */
    dsl_jit_wasm_pos_cache_reset_for_tests();
#endif
    start = portable_bench_now_ms();
    me_artifact *jit = load_elementwise(source, "float64", "float64", ME_JIT_ON);
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
#endif
    check_integer_boundaries();
    check_float_values();
    check_control_flow();
    check_masks();
    check_sizes();
    check_fallbacks();
    check_binding_limits();
    check_mixed_four();
    check_cache_cycles();
    check_forced_invalid();
    check_side_helper_failure();
    benchmark_reuse();
    check_helper_unregistration();
#if defined(__EMSCRIPTEN__)
    dsl_jit_wasm_pos_cache_reset_for_tests();
    puts("PASS: portable wasm32 TCC JIT differential qualification");
#else
    puts("PASS: portable TCC JIT differential qualification");
#endif
    return 0;
}
