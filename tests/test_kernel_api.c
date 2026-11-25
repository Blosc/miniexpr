/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#undef NDEBUG
#include <assert.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "miniexpr_kernel.h"
#include "miniexpr_grid.h"
#include "miniexpr_kernel_internal.h"
#include "functions-simd.h"
#ifndef _WIN32
#ifndef __EMSCRIPTEN__
#include <pthread.h>
#define TEST_THREADS 1
#endif
#else
#include <windows.h>
#endif
#ifdef ME_TEST_HAS_ARTIFACT
#include "miniexpr_artifact.h"
#endif

static me_kernel *prepare(const char *version, const char *source, me_signature *sig) {
    me_kernel *kernel = NULL;
    me_error error;
    me_status rc = version ? me_prepare_menudet(version, source, sig, NULL, &kernel, &error) :
        me_prepare_formula(source, sig, NULL, &kernel, &error);
    if (rc) fprintf(stderr, "prepare failed (%d): %s\n", rc, error.message);
    assert(rc == ME_SUCCESS && kernel);
    return kernel;
}

static me_kernel_info describe(me_kernel *kernel) {
    me_kernel_info info = ME_KERNEL_INFO_INITIALIZER;
    assert(me_describe(kernel, &info) == ME_SUCCESS);
    return info;
}

static void formula(void) {
    /* Declaration order differs from use order; one input is unused. */
    me_input_spec inputs[] = {{"unused", ME_INT16, 0}, {"b", ME_FLOAT32, 0}, {"a", ME_FLOAT32, 0}};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = inputs;
    sig.ninputs = 3;
    me_kernel *kernel = prepare(NULL, "a + 2 * b", &sig);
    me_kernel_info info = describe(kernel);
    assert(info.output_dtype == ME_FLOAT32 && info.output_itemsize == sizeof(float));
    assert(info.cardinality == ME_ELEMENTWISE && info.context_ndim == 0);
    int16_t unused[] = {0, 0, 0};
    float a[] = {1, 2, 3}, b[] = {4, 5, 6}, out[] = {-1, -1, -1};
    me_buffer buffers[] = {{unused, sizeof(unused)}, {b, sizeof(b)}, {a, sizeof(a)}};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 3;
    me_mut_buffer output = {out, sizeof(out)};
    assert(me_eval_block(kernel, buffers, 3, output, &block, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 9 && out[1] == 12 && out[2] == 15);
    uint8_t mask[] = {1, 0, 1};
    block.valid_mask = mask;
    block.valid_mask_capacity = sizeof(mask);
    out[1] = -1;
    assert(me_eval_block(kernel, buffers, 3, output, &block, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 9 && out[1] == -1 && out[2] == 15);
    mask[1] = 2;
    assert(me_eval_block(kernel, buffers, 3, output, &block, NULL, NULL) == ME_ERR_INVALID_ARG);
    mask[1] = 0;
    buffers[2].capacity--;
    assert(me_eval_block(kernel, buffers, 3, output, &block, NULL, NULL) == ME_ERR_INVALID_ARG);
    buffers[2].capacity++;
    output.data = a;
    assert(me_eval_block(kernel, buffers, 3, output, &block, NULL, NULL) == ME_ERR_INVALID_ARG);
    output.data = out;
    output.capacity--;
    assert(me_eval_block(kernel, buffers, 3, output, &block, NULL, NULL) == ME_ERR_INVALID_ARG);
    output.capacity++;
    block.nitems = 0;
    block.valid_mask = NULL;
    block.valid_mask_capacity = 0;
    me_buffer empty[3] = {{0}};
    assert(me_eval_block(kernel, empty, 3, (me_mut_buffer){NULL, 0}, &block, NULL, NULL) == ME_SUCCESS);
    me_kernel_free(kernel);
}

static void annotations(void) {
    const char *source =
        "def select_reading(word: uint16, reading):\n"
        "    if (word >> 15) != 0:\n"
        "        return reading\n"
        "    return reading * 0\n";
    me_input_spec inputs[] = {{"reading", ME_FLOAT32, 0}, {"word", ME_UINT16, 0}};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = inputs;
    sig.ninputs = 2;
    me_kernel *kernel = prepare("0.2", source, &sig);
    assert(describe(kernel).output_dtype == ME_FLOAT32);
    uint16_t words[] = {0, 32768, 65535};
    float readings[] = {10, 20, 30}, out[3] = {0};
    me_buffer buffers[] = {{readings, sizeof(readings)}, {words, sizeof(words)}};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 3;
    assert(me_eval_block(kernel, buffers, 2, (me_mut_buffer){out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 0 && out[1] == 20 && out[2] == 30);
    me_kernel_free(kernel);
    inputs[0].dtype = ME_FLOAT64;
    kernel = prepare("0.2", source, &sig);
    assert(describe(kernel).output_dtype == ME_FLOAT64);
    me_kernel_free(kernel);
    inputs[1].dtype = ME_UINT32;
    me_error error;
    kernel = (me_kernel *)(uintptr_t)1;
    assert(me_prepare_menudet("0.2", source, &sig, NULL, &kernel, &error) == ME_ERR_SIGNATURE);
    assert(!kernel && strstr(error.message, "word"));
    inputs[1].dtype = ME_UINT16;
    assert(me_prepare_menudet("0.1", source, &sig, NULL, &kernel, &error) == ME_ERR_SOURCE);
    assert(!kernel && error.line == 1);
    assert(me_prepare_menudet("0.3", source, &sig, NULL, &kernel, &error) == ME_ERR_UNSUPPORTED);
    assert(me_prepare_formula(source, &sig, NULL, &kernel, &error) == ME_ERR_SOURCE);

    sig.ninputs = 1;
    kernel = prepare("0.2", "def k(reading) -> float64:\n    return reading + reading\n", &sig);
    assert(describe(kernel).output_dtype == ME_FLOAT64);
    me_kernel_free(kernel);
    sig.output_dtype = ME_FLOAT32;
    assert(me_prepare_menudet("0.2", "def k(reading) -> float64:\n    return reading\n",
        &sig, NULL, &kernel, &error) == ME_ERR_SIGNATURE && !kernel);
    sig.output_dtype = ME_BOOL;
    sig.casting = ME_CAST_SAFE;
    assert(me_prepare_menudet("0.1", "def k(reading):\n    return reading\n",
        &sig, NULL, &kernel, &error) == ME_ERR_SIGNATURE && !kernel);

    /* A return annotation does not widen intermediate float32 arithmetic. */
    inputs[0].dtype = ME_FLOAT32;
    sig.output_dtype = ME_AUTO;
    sig.casting = ME_CAST_UNSAFE;
    kernel = prepare("0.2", "def k(reading) -> float64:\n    return reading + reading\n", &sig);
    float large = FLT_MAX;
    double converted = 0;
    me_buffer in = {&large, sizeof(large)};
    block.nitems = 1;
    assert(me_eval_block(kernel, &in, 1, (me_mut_buffer){&converted, sizeof(converted)}, &block, NULL, NULL) == ME_SUCCESS);
    assert(isinf(converted));
    me_kernel_free(kernel);

    static const struct { const char *name; me_dtype dtype; } types[] = {
        {"bool", ME_BOOL}, {"int8", ME_INT8}, {"int16", ME_INT16},
        {"int32", ME_INT32}, {"int64", ME_INT64}, {"uint8", ME_UINT8},
        {"uint16", ME_UINT16}, {"uint32", ME_UINT32}, {"uint64", ME_UINT64},
        {"float32", ME_FLOAT32}, {"float64", ME_FLOAT64},
        {"complex64", ME_COMPLEX64}, {"complex128", ME_COMPLEX128}
    };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        char annotated[128];
        snprintf(annotated, sizeof(annotated), "def k(reading: %s) -> %s:\n    return reading\n",
            types[i].name, types[i].name);
        inputs[0].dtype = types[i].dtype;
        kernel = prepare("0.2", annotated, &sig);
        assert(describe(kernel).output_dtype == types[i].dtype);
        me_kernel_free(kernel);
        me_variable v = {.name = "reading", .dtype = types[i].dtype};
        me_menudet_error validation;
        assert(me_validate_menudet(annotated, "0.2", &v, 1, types[i].dtype, &validation) == ME_MENUDET_SUCCESS);
        assert(me_validate_menudet(annotated, "0.1", &v, 1, types[i].dtype, &validation) == ME_MENUDET_ERR_SOURCE);
    }
    const char *bad_annotations[] = {
        "def k(reading: np.float32):\n    return reading\n",
        "def k(reading: object):\n    return reading\n",
        "def k(reading: bytes):\n    return reading\n",
        "def k(reading) -> auto:\n    return reading\n"
    };
    for (size_t i = 0; i < sizeof(bad_annotations) / sizeof(bad_annotations[0]); i++) {
        assert(me_prepare_menudet("0.2", bad_annotations[i], &sig, NULL, &kernel, &error) == ME_ERR_SOURCE);
        assert(!kernel);
    }
}

static void fixed_bytes(void) {
    me_input_spec input = {"x", ME_BYTES, 4};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    sig.output_dtype = ME_BYTES; /* Width is inferred even with an explicit family. */
    for (int mode = 0; mode < 2; mode++) {
        me_kernel *kernel = prepare(mode ? "0.2" : NULL,
            mode ? "def k(x):\n    return upper(x)\n" : "upper(x)", &sig);
        assert(describe(kernel).output_itemsize == 4);
        const char data[8] = {'a', 'b', 0, 0, 'x', 'y', 'z', 0};
        char out[8] = {0};
        me_buffer buffer = {data, sizeof(data)};
        me_block block = ME_BLOCK_INITIALIZER;
        block.nitems = 2;
        assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
        assert(!memcmp(out, "AB\0\0XYZ", sizeof(out)));
        me_kernel_free(kernel);
    }
}

static void captures_and_ramp(void) {
    /* Signature and caller-owned source/captures need not outlive prepare. */
    double start = 1.25, step = 0.5;
    me_capture captures[] = {
        {{"step", ME_FLOAT64, 0}, &step, ME_CAPTURE_TYPED},
        {{"start", ME_FLOAT64, 0}, &start, ME_CAPTURE_TYPED}
    };
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.captures = captures;
    sig.ncaptures = 2;
    sig.context_ndim = 1;
    sig.output_dtype = ME_FLOAT32;
    char source[] = "def ramp(start, step):\n    return start + _flat_idx * step\n";
    me_kernel *kernel = prepare("0.1", source, &sig);
    memset(source, 0, sizeof(source));
    start = step = -999;
    assert(describe(kernel).context_ndim == 1);
    float out[3] = {0};
    int64_t shape[] = {100}, origin[] = {10}, extent[] = {3};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 3;
    block.ndim = 1;
    block.logical_shape = shape;
    block.origin = origin;
    block.extent = extent;
    assert(me_eval_block(kernel, NULL, 0, (me_mut_buffer){out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 6.25f && out[1] == 6.75f && out[2] == 7.25f);
#ifdef ME_TEST_HAS_ARTIFACT
    /* Golden v1 import and direct preparation use the same numerical path. */
    const char *json =
        "{\"schema_version\":\"0.1\",\"language\":{\"name\":\"menudet\",\"version\":\"0.1\"},"
        "\"requires\":[\"numeric\",\"nd-context\"],"
        "\"source\":\"def ramp():\\n    return 1.25 + _flat_idx * 0.5\\n\","
        "\"entry_point\":\"ramp\",\"inputs\":[],\"constants\":[],"
        "\"output\":{\"dtype\":\"float32\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":1},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}";
    me_artifact *artifact = NULL;
    assert(me_artifact_load(json, strlen(json), ME_JIT_OFF, &artifact, NULL) == ME_ARTIFACT_SUCCESS);
    float reference[3] = {0};
    me_artifact_eval_descriptor d = {
        .struct_size = sizeof(d), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems = 3, .output_capacity = sizeof(reference), .ndim = 1,
        .logical_shape = shape, .block_origin = origin, .block_extent = extent};
    assert(me_artifact_eval_ex(artifact, NULL, 0, reference, &d, NULL) == ME_ARTIFACT_SUCCESS);
    assert(!memcmp(out, reference, sizeof(out)));
    me_artifact_free(artifact);
#endif
    me_kernel_free(kernel);

    me_input_spec input = {"x", ME_FLOAT32, 0};
    sig = (me_signature)ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    double gain = 2;
    me_capture capture = {{"gain", ME_FLOAT64, 0}, &gain, ME_CAPTURE_WEAK};
    sig.captures = &capture;
    sig.ncaptures = 1;
    kernel = prepare("0.1", "def k(x, gain):\n    return x * gain\n", &sig);
    assert(describe(kernel).output_dtype == ME_FLOAT32);
    me_kernel_free(kernel);
    kernel = prepare(NULL, "x * gain", &sig);
    assert(describe(kernel).output_dtype == ME_FLOAT32);
    me_kernel_free(kernel);
    kernel = prepare("0.2", "def k(x, gain: float64):\n    return x * gain\n", &sig);
    assert(describe(kernel).output_dtype == ME_FLOAT64);
    me_kernel_free(kernel);
    capture.category = ME_CAPTURE_TYPED;
    kernel = prepare("0.1", "def k(x, gain):\n    return x * gain\n", &sig);
    assert(describe(kernel).output_dtype == ME_FLOAT64);
    me_kernel_free(kernel);
}

static void scalar_and_errors(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_kernel *kernel = prepare("0.1", "def total(x):\n    return block_sum(x)\n", &sig);
    assert(describe(kernel).cardinality == ME_BLOCK_SCALAR);
    me_block block = ME_BLOCK_INITIALIZER;
    me_buffer buffer = {NULL, 0};
    double out = -1;
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){&out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
    assert(out == 0);
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){NULL, 0}, &block, NULL, NULL) == ME_ERR_INVALID_ARG);
    me_kernel_free(kernel);
    me_error error;
    assert(me_prepare_formula("sum(x)", &sig, NULL, &kernel, &error) != ME_SUCCESS && !kernel);
    assert(me_prepare_menudet("0.1", "def k(x):\n    print(x)\n    return x\n",
        &sig, NULL, &kernel, &error) == ME_ERR_SOURCE && !kernel);
    assert(me_prepare_menudet(NULL, "", &sig, NULL, &kernel, &error) == ME_ERR_INVALID_ARG && !kernel);
    sig.struct_size = 0;
    assert(me_prepare_formula("x", &sig, NULL, &kernel, &error) == ME_ERR_SIGNATURE && !kernel);
    me_kernel_free(NULL);

    sig = (me_signature)ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    kernel = prepare(NULL, "1 / x", &sig);
    double zero = 0;
    buffer = (me_buffer){&zero, sizeof(zero)};
    block.nitems = 1;
    me_fp_status fp;
    me_run_options run = ME_RUN_OPTIONS_INITIALIZER;
    run.fp_status = &fp;
    run.raise_mask = ME_FP_DIVIDE;
#ifndef __EMSCRIPTEN__
    feclearexcept(FE_ALL_EXCEPT);
    feraiseexcept(FE_INVALID);
    int flags = fetestexcept(FE_ALL_EXCEPT);
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){&out, sizeof(out)}, &block, &run, &error) == ME_ERR_EVAL);
    assert(fp.supported && (fp.flags & ME_FP_DIVIDE));
    assert(fetestexcept(FE_ALL_EXCEPT) == flags);
    feclearexcept(FE_ALL_EXCEPT);
#else
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){&out, sizeof(out)}, &block, &run, &error) == ME_ERR_UNSUPPORTED);
#endif
    me_kernel_free(kernel);
}

#ifdef __EMSCRIPTEN__
static int forbidden_jit(const void **inputs, void *output, int count) {
#else
static int forbidden_jit(const void **inputs, void *output, int64_t count) {
#endif
    (void)inputs;
    (void)output;
    (void)count;
    assert(!"invocation-local interpreter selection reached JIT");
    return -1;
}

static void execution_options(bool require_jit, bool tcc) {
    me_input_spec input = {"x", ME_FLOAT32, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    options.execution = ME_EXEC_PREFER_JIT;
    me_kernel *kernel = NULL;
    me_error error;
    char source[512];
    snprintf(source, sizeof(source), "# me:compiler=%s\ndef k(x: float32) -> float64:\n    return 2 * x\n",
        tcc ? "tcc" : "cc");
    if (require_jit) options.execution = ME_EXEC_REQUIRE_JIT;
#ifndef __EMSCRIPTEN__
    fenv_t caller_environment;
    assert(fegetenv(&caller_environment) == 0);
    assert(fesetround(FE_UPWARD) == 0);
    feclearexcept(FE_ALL_EXCEPT);
    feraiseexcept(FE_INVALID);
    int caller_flags = fetestexcept(FE_ALL_EXCEPT);
#endif
    me_status rc = me_prepare_menudet("0.2", source, &sig, &options, &kernel, &error);
#ifndef __EMSCRIPTEN__
    assert(fegetround() == FE_UPWARD && fetestexcept(FE_ALL_EXCEPT) == caller_flags);
    assert(fesetenv(&caller_environment) == 0);
#endif
    if (rc) fprintf(stderr, "execution policy failed: %s\n", error.message);
    assert(rc == ME_SUCCESS);
    float x[] = {2, 4, 8};
    double out[3] = {0};
    me_buffer buffer = {x, sizeof(x)};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 3;
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 4 && out[1] == 8 && out[2] == 16);
    me_run_options run = ME_RUN_OPTIONS_INITIALIZER;
    run.execution = ME_RUN_INTERPRET;
    me_dsl_jit_kernel_fn saved_jit = kernel->program->jit_kernel_fn;
    kernel->program->jit_kernel_fn = forbidden_jit;
    memset(out, 0, sizeof(out));
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){out, sizeof(out)},
        &block, &run, NULL) == ME_SUCCESS);
    assert(out[0] == 4 && out[1] == 8 && out[2] == 16);
    kernel->program->jit_kernel_fn = saved_jit;
    uint8_t mask[] = {1, 1, 1};
    block.valid_mask = mask;
    block.valid_mask_capacity = sizeof(mask);
    memset(out, 0, sizeof(out));
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 4 && out[1] == 8 && out[2] == 16);
    me_kernel_free(kernel);

    int64_t n = 3;
    bool enabled = true;
    me_capture captures[] = {
        {{"n", ME_INT64, 0}, &n, ME_CAPTURE_WEAK},
        {{"enabled", ME_BOOL, 0}, &enabled, ME_CAPTURE_WEAK}
    };
    sig.captures = captures;
    sig.ncaptures = 2;
    snprintf(source, sizeof(source), "# me:compiler=%s\ndef k(x, n, enabled):\n"
        "    value = x\n"
        "    for i in range(n):\n"
        "        if enabled:\n"
        "            value = value + 1\n"
        "    return value\n", tcc ? "tcc" : "cc");
    for (int version = 0; version < 2; version++) {
        assert(me_prepare_menudet(version ? "0.2" : "0.1", source, &sig,
            &options, &kernel, &error) == ME_SUCCESS);
        float result[3];
        assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){result, sizeof(result)},
            &block, NULL, NULL) == ME_SUCCESS);
        assert(result[0] == 5 && result[1] == 7 && result[2] == 11);
        me_kernel_free(kernel);
    }
    options.execution = ME_EXEC_REQUIRE_JIT;
    sig.captures = NULL;
    sig.ncaptures = 0;
    assert(me_prepare_formula("2 * x", &sig, &options, &kernel, &error) == ME_ERR_UNSUPPORTED && !kernel);
    sig.context_ndim = 1;
    snprintf(source, sizeof(source), "# me:compiler=%s\ndef k(x):\n    return x + _flat_idx\n",
        tcc ? "tcc" : "cc");
    /* This checks interpreter coordinate construction, not certification of
     * additional ND JIT lowerings (which wasm32 may legitimately decline). */
    options.execution = ME_EXEC_PREFER_JIT;
    assert(me_prepare_menudet("0.2", source, &sig, &options, &kernel, &error) == ME_SUCCESS);
    int64_t shape = 3, origin = 0, extent = 3;
    block.ndim = 1;
    block.logical_shape = &shape;
    block.origin = &origin;
    block.extent = &extent;
    mask[1] = 0;
    saved_jit = kernel->program->jit_kernel_fn;
    kernel->program->jit_kernel_fn = forbidden_jit;
    double indexed[3] = {0};
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){indexed, sizeof(indexed)},
        &block, &run, NULL) == ME_SUCCESS);
    assert(indexed[0] == 2 && indexed[2] == 10);
    kernel->program->jit_kernel_fn = saved_jit;
    me_kernel_free(kernel);
}

static void rejected_jit_contracts(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    sig.output_dtype = ME_BOOL;
    sig.casting = ME_CAST_SAFE;
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    options.execution = ME_EXEC_REQUIRE_JIT;
    me_kernel *kernel = NULL;
    me_error error;
    const char *source = "# me:compiler=cc\ndef k(x):\n    return x + 1\n";
    assert(me_prepare_menudet("0.1", source, &sig, &options, &kernel, &error) == ME_ERR_SIGNATURE);
    assert(!kernel && error.status == ME_ERR_SIGNATURE);
    sig.casting = ME_CAST_UNSAFE;
    sig.output_dtype = ME_AUTO;
    sig.cardinality = ME_BLOCK_SCALAR;
    assert(me_prepare_menudet("0.1", source, &sig, &options, &kernel, &error) == ME_ERR_SIGNATURE);
    assert(!kernel);
#ifdef ME_TEST_HAS_ARTIFACT
    /* The fixed v1 reader must reject the same contracts before backend/cache
     * activity; the compiler-invocation probe exercises both entry routes. */
    for (int contract = 0; contract < 3; contract++) {
        char json[1024];
        snprintf(json, sizeof(json),
            "{\"schema_version\":\"0.1\",\"language\":{\"name\":\"menudet\",\"version\":\"0.1\"},"
            "\"requires\":[\"numeric\"],\"source\":\"# me:compiler=cc\\ndef reject_json(x):\\n    return x + 1\\n\","
            "\"entry_point\":\"reject_json\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"float64\"}],"
            "\"constants\":[],\"output\":{\"dtype\":\"%s\",\"contract\":\"%s\"},\"context\":{\"ndim\":0},"
            "\"semantics\":{\"numeric\":\"numpy-2.5\",\"casting\":\"%s\"}}",
            contract ? "float64" : "bool", contract == 1 ? "block_scalar" : "elementwise",
            contract ? "unsafe" : "safe");
        me_artifact *artifact = NULL;
        if (contract == 2) {
            assert(me_artifact_load(json, strlen(json), ME_JIT_OFF, &artifact, NULL) == ME_ARTIFACT_SUCCESS);
            assert(artifact && !me_artifact_has_jit(artifact));
            me_artifact_free(artifact);
        }
        else {
            assert(me_artifact_load(json, strlen(json), ME_JIT_ON, &artifact, NULL) == ME_ARTIFACT_ERR_BINDING);
            assert(!artifact);
        }
    }
#endif
}

static void signature_errors(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_kernel *kernel = NULL;
    me_error error;
    const char *sources[] = {
        "def k(x, y):\n    return x\n",
        "def k(y):\n    return y\n",
        "def k(x):\n    return x + _i0\n"
    };
    for (size_t i = 0; i < sizeof(sources) / sizeof(sources[0]); i++) {
        assert(me_prepare_menudet("0.1", sources[i], &sig, NULL, &kernel, &error) == ME_ERR_SIGNATURE);
        assert(!kernel && error.status == ME_ERR_SIGNATURE);
    }
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    options.struct_size = ME_PREPARE_OPTIONS_V1_SIZE - 1;
    assert(me_prepare_formula("x", &sig, &options, &kernel, &error) == ME_ERR_INVALID_ARG);
    options.struct_size = sizeof(options);
    options.execution = (me_execution_policy)99;
    assert(me_prepare_formula("x", &sig, &options, &kernel, &error) == ME_ERR_INVALID_ARG);
}

static void set_math_environment(const char *backend) {
#ifdef _WIN32
    assert(_putenv_s("ME_SIMD_MATH_BACKEND", backend) == 0);
#else
    /* Set libc's environment too: Node process.env is not Emscripten's ENV. */
    assert(setenv("ME_SIMD_MATH_BACKEND", backend, 1) == 0);
#endif
}

static void formula_environment(void) {
    /* Defaults ignore an invalid backend; opt-in validates
     * it during preparation rather than silently consulting it at evaluation. */
    set_math_environment("invalid");
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    me_kernel *kernel = NULL;
    me_error error;
    assert(me_prepare_formula("1", &sig, &options, &kernel, &error) == ME_SUCCESS);
    me_kernel_free(kernel);
    options.inherit_environment = true;
    assert(me_prepare_formula("1", &sig, &options, &kernel, &error) == ME_ERR_UNSUPPORTED);
    assert(!kernel && strstr(error.message, "ME_SIMD_MATH_BACKEND"));
    set_math_environment("scalar");
    me_input_spec input = {"x", ME_FLOAT64, 0};
    sig.inputs = &input;
    sig.ninputs = 1;
    assert(me_prepare_formula("sin(x)", &sig, &options, &kernel, &error) == ME_SUCCESS);
    set_math_environment("invalid");
    double x = 0.25, out = 0;
    me_buffer buffer = {&x, sizeof(x)};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 1;
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){&out, sizeof(out)}, &block, NULL, &error) == ME_SUCCESS);
    assert(fabs(out - sin(x)) <= DBL_EPSILON);
    me_kernel_free(kernel);
}

static void descriptor_prefixes(void) {
    assert(ME_SIGNATURE_V1_SIZE == sizeof(me_signature));
    assert(ME_PREPARE_OPTIONS_V1_SIZE == sizeof(me_prepare_options));
    assert(ME_KERNEL_INFO_V1_SIZE == sizeof(me_kernel_info));
    assert(ME_BLOCK_V1_SIZE == sizeof(me_block));
    assert(ME_RUN_OPTIONS_V1_SIZE == sizeof(me_run_options));
    assert(ME_GRID_SPEC_V1_SIZE == sizeof(me_grid_spec));
    assert(ME_GRID_BLOCK_INFO_V1_SIZE == sizeof(me_grid_block_info));
    struct { me_signature value; uint64_t suffix; } sig = {ME_SIGNATURE_INITIALIZER, UINT64_MAX};
    struct { me_prepare_options value; uint64_t suffix; } options = {ME_PREPARE_OPTIONS_INITIALIZER, UINT64_MAX};
    sig.value.struct_size = sizeof(sig);
    options.value.struct_size = sizeof(options);
    for (size_t i = 0; i < sizeof(sig.value.reserved); i++) assert(sig.value.reserved[i] == 0);
    for (size_t i = 0; i < sizeof(options.value.reserved); i++) assert(options.value.reserved[i] == 0);
    me_kernel *kernel = NULL;
    assert(me_prepare_formula("1", &sig.value, &options.value, &kernel, NULL) == ME_SUCCESS);
    struct { me_kernel_info value; uint64_t suffix; } info = {ME_KERNEL_INFO_INITIALIZER, UINT64_MAX};
    info.value.struct_size = sizeof(info);
    assert(me_describe(kernel, &info.value) == ME_SUCCESS);
    assert(info.value.struct_size == sizeof(info) && info.suffix == UINT64_MAX);
    me_block block = ME_BLOCK_INITIALIZER;
    struct { me_run_options value; uint64_t suffix; } run = {ME_RUN_OPTIONS_INITIALIZER, UINT64_MAX};
    for (size_t i = 0; i < sizeof(run.value.reserved); i++) assert(run.value.reserved[i] == 0);
    block.struct_size = ME_BLOCK_V1_SIZE;
    run.value.struct_size = sizeof(run);
    assert(me_eval_block(kernel, NULL, 0, (me_mut_buffer){0}, &block, &run.value, NULL) == ME_SUCCESS);
    run.value.execution = (me_run_execution)12345;
    assert(me_eval_block(kernel, NULL, 0, (me_mut_buffer){0}, &block, &run.value, NULL) == ME_ERR_INVALID_ARG);
    run.value.execution = ME_RUN_PREPARED;
    run.value.struct_size = ME_RUN_OPTIONS_V1_SIZE - 1;
    assert(me_eval_block(kernel, NULL, 0, (me_mut_buffer){0}, &block, &run.value, NULL) == ME_ERR_INVALID_ARG);
    run.value.struct_size = ME_RUN_OPTIONS_V1_SIZE;
    block.struct_size--;
    assert(me_eval_block(kernel, NULL, 0, (me_mut_buffer){0}, &block, &run.value, NULL) == ME_ERR_INVALID_ARG);
    me_kernel_free(kernel);
    sig.value.struct_size = ME_SIGNATURE_V1_SIZE;
    options.value.struct_size = ME_PREPARE_OPTIONS_V1_SIZE;
    assert(me_prepare_formula("1", &sig.value, &options.value, &kernel, NULL) == ME_SUCCESS);
    me_kernel_free(kernel);
}

/* Flat future layouts, not nested v1 structs: a nested prefix would itself
 * hide a trailing-padding bug by forcing the new flag past sizeof(v1). */
typedef struct {
    size_t struct_size;
    const me_input_spec *inputs;
    size_t ninputs;
    const me_capture *captures;
    size_t ncaptures;
    me_dtype output_dtype;
    size_t output_itemsize;
    me_cardinality cardinality;
    int context_ndim;
    me_casting casting;
    bool integer_powers;
    uint8_t reserved[3];
    uint8_t new_flag;
} signature_next;

typedef struct {
    size_t struct_size;
    me_execution_policy execution;
    bool inherit_environment;
    uint8_t reserved[3];
    uint8_t new_flag;
} prepare_options_next;

typedef struct {
    size_t struct_size;
    unsigned raise_mask;
    me_fp_status *fp_status;
    me_run_execution execution;
    uint8_t reserved[4];
    uint8_t new_flag;
} run_options_next;

static uint8_t optional_flag(const void *descriptor, size_t offset) {
    size_t size;
    uint8_t flag = 0;
    memcpy(&size, descriptor, sizeof(size));
    if (size >= offset + sizeof(flag)) memcpy(&flag, (const unsigned char *)descriptor + offset, sizeof(flag));
    return flag;
}

static void receive_v1_descriptors(const void *signature, const void *options, const void *run_options) {
    assert(offsetof(signature_next, new_flag) == ME_SIGNATURE_V1_SIZE);
    assert(offsetof(prepare_options_next, new_flag) == ME_PREPARE_OPTIONS_V1_SIZE);
    assert(offsetof(run_options_next, new_flag) == ME_RUN_OPTIONS_V1_SIZE);
    assert(optional_flag(signature, offsetof(signature_next, new_flag)) == 0);
    assert(optional_flag(options, offsetof(prepare_options_next, new_flag)) == 0);
    assert(optional_flag(run_options, offsetof(run_options_next, new_flag)) == 0);
    /* Exercise the real preparation and evaluation APIs with the independently
     * compiled v1 bytes as well as simulated future absent-field defaults. */
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    me_prepare_options prepare = ME_PREPARE_OPTIONS_INITIALIZER;
    me_run_options run = ME_RUN_OPTIONS_INITIALIZER;
    size_t sig_size, options_size, run_size;
    memcpy(&sig_size, signature, sizeof(sig_size));
    memcpy(&options_size, options, sizeof(options_size));
    memcpy(&run_size, run_options, sizeof(run_size));
    assert(sig_size == ME_SIGNATURE_V1_SIZE && options_size == ME_PREPARE_OPTIONS_V1_SIZE &&
        run_size == ME_RUN_OPTIONS_V1_SIZE);
    memcpy(&sig, signature, sig_size);
    memcpy(&prepare, options, options_size);
    memcpy(&run, run_options, run_size);
    me_kernel *kernel = NULL;
    assert(me_prepare_formula("1", &sig, &prepare, &kernel, NULL) == ME_SUCCESS);
    me_block block = ME_BLOCK_INITIALIZER;
    assert(me_eval_block(kernel, NULL, 0, (me_mut_buffer){0}, &block, &run, NULL) == ME_SUCCESS);
    me_kernel_free(kernel);
    signature_next future_sig = {0};
    prepare_options_next future_options = {0};
    run_options_next future_run = {0};
    future_sig.struct_size = sizeof(future_sig);
    future_options.struct_size = sizeof(future_options);
    future_run.struct_size = sizeof(future_run);
    future_sig.new_flag = future_options.new_flag = future_run.new_flag = 1;
    assert(optional_flag(&future_sig, offsetof(signature_next, new_flag)) == 1);
    assert(optional_flag(&future_options, offsetof(prepare_options_next, new_flag)) == 1);
    assert(optional_flag(&future_run, offsetof(run_options_next, new_flag)) == 1);
}

void me_test_kernel_v1_caller(void (*receive)(const void *, const void *, const void *));

static void accuracy_probe(const double *input, double *output, int nitems) {
    /* Deterministic dispatch probe: inspect active accuracy without relying on
     * an architecture-specific input where SLEEF u10 and u35 differ in bits. */
    me_simd_accuracy_state active;
    me_simd_accuracy_push(ME_DSL_ULP_OMIT, &active);
    assert(active.override_u35 == 1);
    me_simd_accuracy_pop(&active);
    for (int i = 0; i < nitems; i++) output[i] = input[i];
}

static void check_formula_accuracy(const me_kernel *kernel) {
    me_simd_accuracy_state outer, after;
    me_simd_accuracy_push(ME_DSL_ULP_U10, &outer);
    double x = 0.25, out;
    me_buffer input = {&x, sizeof(x)};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 1;
    assert(me_eval_block(kernel, &input, 1, (me_mut_buffer){&out, sizeof(out)},
        &block, NULL, NULL) == ME_SUCCESS);
    assert(out == x + 1 / x);
#ifndef __EMSCRIPTEN__
    me_run_options run = ME_RUN_OPTIONS_INITIALIZER;
    run.raise_mask = ME_FP_DIVIDE;
    x = 0;
    assert(me_eval_block(kernel, &input, 1, (me_mut_buffer){&out, sizeof(out)},
        &block, &run, NULL) == ME_ERR_EVAL);
#endif
    me_simd_accuracy_push(ME_DSL_ULP_OMIT, &after);
    assert(after.override_u35 == 0);
    me_simd_accuracy_pop(&after);
    me_simd_accuracy_pop(&outer);
}

#ifdef TEST_THREADS
static void *accuracy_worker(void *arg) {
    check_formula_accuracy(arg);
    return NULL;
}
#endif

static void formula_accuracy(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_kernel *kernel = prepare(NULL, "sin(x) + 1 / x", &sig);
    /* Test-only private dispatch injection works even in SLEEF-disabled builds;
     * production policy resolution already stores the prepared ULP in this table. */
    kernel->formula_math.ulp = ME_DSL_ULP_U35;
    kernel->formula_math.accelerated = true;
    kernel->formula_math.unary_f64[ME_MATHOP_SIN] = accuracy_probe;
    check_formula_accuracy(kernel);
#ifdef TEST_THREADS
    pthread_t ids[4];
    for (int i = 0; i < 4; i++) assert(!pthread_create(&ids[i], NULL, accuracy_worker, kernel));
    for (int i = 0; i < 4; i++) assert(!pthread_join(ids[i], NULL));
#endif
    me_kernel_free(kernel);
}

#ifdef ME_TEST_HAS_ARTIFACT
static void direct_scheduler_adapter(void) {
    me_input_spec input = {"x", ME_FLOAT32, 0};
    double gain = 2;
    me_capture capture = {{"gain", ME_FLOAT64, 0}, &gain, ME_CAPTURE_WEAK};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = sig.ncaptures = 1;
    sig.captures = &capture;
    me_artifact *artifact = NULL;
    me_artifact_error error;
    const char *source = "def k(x: float32, gain) -> float64:\n    return x * gain\n";
    assert(me_artifact_prepare_menudet("0.2", source, &sig, ME_JIT_OFF,
        NULL, &artifact, &error) == ME_ARTIFACT_SUCCESS);
    gain = 999;
    assert(me_artifact_input_dtype(artifact, 0) == ME_FLOAT32);
    assert(me_artifact_output_dtype(artifact) == ME_FLOAT64);
    assert(me_artifact_inferred_dtype(artifact) == ME_FLOAT32);
    assert(me_artifact_capabilities(artifact) == ME_ARTIFACT_CAP_NUMERIC);
    assert(!strcmp(me_artifact_source(artifact), source));
    float x[] = {1, 2, 3};
    double output[3];
    me_artifact_buffer buffer = {"x", ME_FLOAT32, sizeof(float), x, sizeof(x)};
    me_artifact_eval_descriptor d = {
        .struct_size = sizeof(d), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems = 3, .output_capacity = sizeof(output)};
    assert(me_artifact_eval_ex(artifact, &buffer, 1, output, &d, &error) == ME_ARTIFACT_SUCCESS);
    assert(output[0] == 2 && output[1] == 4 && output[2] == 6);
    me_artifact_free(artifact);
    assert(me_artifact_prepare_menudet("0.1", source, &sig, ME_JIT_OFF,
        NULL, &artifact, &error) == ME_ARTIFACT_ERR_SOURCE && !artifact);
    input.dtype = ME_FLOAT64;
    assert(me_artifact_prepare_menudet("0.2", source, &sig, ME_JIT_OFF,
        NULL, &artifact, &error) == ME_ARTIFACT_ERR_BINDING && !artifact);
    char prefix[] = {'x', 'y'};
    capture = (me_capture){{"prefix", ME_BYTES, sizeof(prefix)}, prefix, ME_CAPTURE_TYPED};
    sig = (me_signature)ME_SIGNATURE_INITIALIZER;
    sig.captures = &capture;
    sig.ncaptures = 1;
    assert(me_artifact_prepare_menudet("0.2", "def k(prefix):\n    return prefix\n",
        &sig, ME_JIT_OFF, NULL, &artifact, &error) == ME_ARTIFACT_SUCCESS);
    prefix[0] = 'z';
    char strings[4];
    d.nitems = 2;
    d.output_capacity = sizeof(strings);
    assert(me_artifact_eval_ex(artifact, NULL, 0, strings, &d, &error) == ME_ARTIFACT_SUCCESS);
    assert(!memcmp(strings, "xyxy", sizeof(strings)));
    assert(me_artifact_capabilities(artifact) == (ME_ARTIFACT_CAP_NUMERIC | ME_ARTIFACT_CAP_FIXED_STRINGS));
    me_artifact_free(artifact);

    sig = (me_signature)ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    sig.context_ndim = 1;
    static const struct { const char *source; unsigned capabilities; } cases[] = {
        {"def k(x):\n    return x\n", ME_ARTIFACT_CAP_NUMERIC},
        {"def k(x):\n    return x + _flat_idx\n", ME_ARTIFACT_CAP_NUMERIC | ME_ARTIFACT_CAP_ND_CONTEXT},
        {"def k(x):\n    return block_sum(x)\n", ME_ARTIFACT_CAP_NUMERIC | ME_ARTIFACT_CAP_BLOCK_REDUCTIONS},
        {"def k(x):\n    if x > 0:\n        return x\n    return -x\n",
            ME_ARTIFACT_CAP_NUMERIC | ME_ARTIFACT_CAP_CONTROL_FLOW},
        {"# me:math=system\ndef k(x):\n    return x\n",
            ME_ARTIFACT_CAP_NUMERIC | ME_ARTIFACT_CAP_FLEXIBLE_MATH}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        assert(me_artifact_prepare_menudet("0.2", cases[i].source, &sig, ME_JIT_OFF,
            NULL, &artifact, &error) == ME_ARTIFACT_SUCCESS);
        assert(me_artifact_capabilities(artifact) == cases[i].capabilities);
        me_artifact_free(artifact);
    }
}
#endif
#ifdef TEST_THREADS
static void *worker(void *arg) {
    const me_kernel *kernel = arg;
    double a[] = {2, 4, 8}, out[3];
    me_buffer input = {a, sizeof(a)};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 3;
    for (int i = 0; i < 30; i++) {
        assert(me_eval_block(kernel, &input, 1, (me_mut_buffer){out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
        assert(out[0] == 4 && out[1] == 8 && out[2] == 16);
    }
    return NULL;
}

static void threads(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    for (int mode = 0; mode < 2; mode++) {
        me_kernel *kernel = prepare(mode ? "0.2" : NULL,
            mode ? "def k(x: float64):\n    return 2 * x\n" : "2 * x", &sig);
        pthread_t ids[4];
        for (int i = 0; i < 4; i++) assert(!pthread_create(&ids[i], NULL, worker, kernel));
        for (int i = 0; i < 4; i++) assert(!pthread_join(ids[i], NULL));
        me_kernel_free(kernel);
    }
}
#endif

#ifdef _WIN32
typedef struct {
    void *parent;
    me_kernel *kernel;
    size_t lanes;
} fiber_call;

static VOID WINAPI eval_fiber(void *arg) {
    fiber_call *call = arg;
    double x[128] = {0}, y[128];
    me_buffer input = {x, sizeof(x)};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = call->lanes;
    for (;;) {
        assert(me_eval_block(call->kernel, &input, 1, (me_mut_buffer){y, sizeof(y)},
            &block, NULL, NULL) == ME_SUCCESS);
        assert(y[block.nitems - 1] == 2);
        SwitchToFiber(call->parent);
    }
}

static void fiber_scratch(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_kernel *kernel = prepare("0.2", "def k(x):\n    a = x + 1\n    return a + a\n", &sig);
    void *parent = ConvertThreadToFiber(NULL);
    assert(parent);
    fiber_call small = {parent, kernel, 2}, large = {parent, kernel, 128};
    void *a = CreateFiber(0, eval_fiber, &small);
    void *b = CreateFiber(0, eval_fiber, &large);
    assert(a && b);
    SwitchToFiber(a);
    SwitchToFiber(b);
    DeleteFiber(a);
    SwitchToFiber(b);
    DeleteFiber(b);
    assert(ConvertFiberToThread());
    me_kernel_free(kernel);
}
#endif

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--formula-environment")) {
        formula_environment();
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--reject-jit-contracts")) {
        rejected_jit_contracts();
        return 0;
    }
    formula();
    annotations();
    captures_and_ramp();
    scalar_and_errors();
    fixed_bytes();
    bool tcc = argc > 1 && !strcmp(argv[1], "--require-tcc");
    execution_options(tcc || (argc > 1 && !strcmp(argv[1], "--require-jit")), tcc);
    signature_errors();
    descriptor_prefixes();
    me_test_kernel_v1_caller(receive_v1_descriptors);
    formula_accuracy();
    rejected_jit_contracts();
#ifdef ME_TEST_HAS_ARTIFACT
    direct_scheduler_adapter();
#endif
#ifdef TEST_THREADS
    threads();
#endif
#ifdef _WIN32
    fiber_scratch();
#endif
    puts("Core kernel API passed");
    return 0;
}
