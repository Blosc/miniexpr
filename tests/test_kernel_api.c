/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#undef NDEBUG
#include <assert.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "miniexpr_kernel.h"
#ifndef _WIN32
#ifndef __EMSCRIPTEN__
#include <pthread.h>
#define TEST_THREADS 1
#endif
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
    assert(me_prepare_menudet("0.2", source, &sig, NULL, &kernel, &error) != ME_SUCCESS);
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
        &sig, NULL, &kernel, &error) != ME_SUCCESS && !kernel);
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

static void execution_options(bool require_jit) {
    me_input_spec input = {"x", ME_FLOAT32, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    options.execution = ME_EXEC_PREFER_JIT;
    me_kernel *kernel = NULL;
    me_error error;
    const char *source = "# me:compiler=cc\ndef k(x: float32) -> float64:\n    return 2 * x\n";
    if (require_jit) options.execution = ME_EXEC_REQUIRE_JIT;
    me_status rc = me_prepare_menudet("0.2", source, &sig, &options, &kernel, &error);
    if (rc) fprintf(stderr, "execution policy failed: %s\n", error.message);
    assert(rc == ME_SUCCESS);
    float x[] = {2, 4, 8};
    double out[3] = {0};
    me_buffer buffer = {x, sizeof(x)};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 3;
    assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){out, sizeof(out)}, &block, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 4 && out[1] == 8 && out[2] == 16);
    me_kernel_free(kernel);
    options.execution = ME_EXEC_REQUIRE_JIT;
    assert(me_prepare_formula("2 * x", &sig, &options, &kernel, &error) == ME_ERR_UNSUPPORTED && !kernel);
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
    me_artifact_free(artifact);
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

int main(int argc, char **argv) {
    formula();
    annotations();
    captures_and_ramp();
    scalar_and_errors();
    fixed_bytes();
    execution_options(argc > 1 && !strcmp(argv[1], "--require-jit"));
#ifdef ME_TEST_HAS_ARTIFACT
    direct_scheduler_adapter();
#endif
#ifdef TEST_THREADS
    threads();
#endif
    puts("Core kernel API passed");
    return 0;
}
