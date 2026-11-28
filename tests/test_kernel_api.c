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
#include "miniexpr_reduce.h"
#include "miniexpr_kernel_internal.h"
#include "dsl_math_policy.h"
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
    size_t count = SIZE_MAX;
    assert(me_describe_input_count(kernel, &count) == ME_SUCCESS);
    assert(count == sig->ninputs);
    for (size_t i = 0; i < count; i++) {
        me_input_spec input;
        assert(me_describe_input(kernel, i, &input) == ME_SUCCESS);
        assert(!input.name && input.dtype == sig->inputs[i].dtype && input.itemsize);
        assert(!sig->inputs[i].itemsize || input.itemsize == sig->inputs[i].itemsize);
    }
    me_input_spec invalid;
    assert(me_describe_input(kernel, count, &invalid) == ME_ERR_INVALID_ARG);
    return kernel;
}

static me_kernel_info describe(me_kernel *kernel) {
    me_kernel_info info = ME_KERNEL_INFO_INITIALIZER;
    assert(me_describe(kernel, &info) == ME_SUCCESS);
    return info;
}

static void partition_proof(void) {
    assert(!me_kernel_is_partitionable(NULL));
    me_input_spec input = {"x", ME_INT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_kernel *kernel = prepare(NULL, "x * 2", &sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    kernel = prepare("0.2", "def k(x):\n    value = x * 2\n    return value + x\n", &sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    kernel = prepare("0.2", "def k(x):\n    if x > 0:\n        return x\n    return -x\n", &sig);
    assert(!me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    input.dtype = ME_UINT8;
    int64_t capture = 300;
    me_capture scalar = {{"c", ME_INT64, 0}, &capture, ME_CAPTURE_WEAK};
    sig.captures = &scalar;
    sig.ncaptures = 1;
    kernel = prepare("0.2", "def k(x, c):\n    return x + c\n", &sig);
    assert(!me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    input.dtype = ME_INT64;
    capture = INT64_MAX;
    scalar.spec.dtype = ME_INT64;
    kernel = prepare("0.2", "def k(x, c):\n    return x + (c + c)\n", &sig);
    assert(!me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    capture = INT64_MIN;
    kernel = prepare("0.2", "def k(x, c):\n    return x + (-c)\n", &sig);
    assert(!me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    scalar.category = ME_CAPTURE_TYPED;
    capture = INT64_MAX;
    kernel = prepare("0.2", "def k(x, c):\n    return x + (c + c)\n", &sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    sig.captures = NULL;
    sig.ncaptures = 0;
    input.dtype = ME_FLOAT64;
    kernel = prepare(NULL, "x * 2", &sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    sig.output_dtype = ME_INT64;
    kernel = prepare(NULL, "x", &sig);
    assert(!me_kernel_is_partitionable(kernel));
    assert(me_kernel_inferred_dtype(kernel) == ME_FLOAT64);
    assert(describe(kernel).output_dtype == ME_INT64);
    me_kernel_free(kernel);
    sig.output_dtype = ME_AUTO;
    input.dtype = ME_FLOAT32;
    kernel = prepare(NULL, "x * 0.5", &sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    input.dtype = ME_FLOAT64;
    kernel = prepare(NULL, "sin(x)", &sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
    const char *partitionable[] = {
        "sin(x) + cos(x)",
        "sqrt(x * x + 1.0)",
        "x / (x + 1.0)",
        "where(x > 0.5, x + 1.0, x - 1.0)",
        "fmod(x, x + 0.1)",
        "isfinite(x) and not isnan(x)"
    };
    for (size_t i = 0; i < sizeof(partitionable) / sizeof(*partitionable); i++) {
        kernel = prepare(NULL, partitionable[i], &sig);
        assert(me_kernel_is_partitionable(kernel));
        me_kernel_free(kernel);
    }
    const char *non_partitionable[] = {"x // 2.0", "x % 2.0", "x ** 2.0", "tgamma(x)"};
    for (size_t i = 0; i < sizeof(non_partitionable) / sizeof(*non_partitionable); i++) {
        kernel = prepare(NULL, non_partitionable[i], &sig);
        assert(!me_kernel_is_partitionable(kernel));
        me_kernel_free(kernel);
    }
    input.dtype = ME_INT64;
    const char *int_partitionable[] = {
        "sin(x)", "x / 2", "(x & 7) ^ 3", "where(x > 0, x + 1, -x)", "minimum(x, 42)"
    };
    for (size_t i = 0; i < sizeof(int_partitionable) / sizeof(*int_partitionable); i++) {
        kernel = prepare(NULL, int_partitionable[i], &sig);
        assert(me_kernel_is_partitionable(kernel));
        me_kernel_free(kernel);
    }
    const char *int_non_partitionable[] = {"x // 2", "x % 2", "x ** 2", "x << 1", "x >> 1"};
    for (size_t i = 0; i < sizeof(int_non_partitionable) / sizeof(*int_non_partitionable); i++) {
        kernel = prepare(NULL, int_non_partitionable[i], &sig);
        assert(!me_kernel_is_partitionable(kernel));
        me_kernel_free(kernel);
    }

    /* Weak captures: representable float64->float32 and int64->int16 are partitionable;
     * out-of-range int64->int16 is rejected. */
    me_input_spec pair_inputs[] = {{"x", ME_FLOAT32, 0}, {"y", ME_FLOAT32, 0}};
    double weak_f64 = 0.5;
    me_capture weak_f64_cap = {{"c", ME_FLOAT64, 0}, &weak_f64, ME_CAPTURE_WEAK};
    me_signature cap_sig = ME_SIGNATURE_INITIALIZER;
    cap_sig.inputs = pair_inputs;
    cap_sig.ninputs = 2;
    cap_sig.captures = &weak_f64_cap;
    cap_sig.ncaptures = 1;
    kernel = prepare("0.2", "def k(x, y, c):\n    return x * c + y\n", &cap_sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);

    pair_inputs[0].dtype = ME_INT16;
    pair_inputs[1].dtype = ME_INT16;
    int64_t weak_i64 = 1;
    me_capture weak_i64_cap = {{"c", ME_INT64, 0}, &weak_i64, ME_CAPTURE_WEAK};
    cap_sig.captures = &weak_i64_cap;
    kernel = prepare("0.2", "def k(x, y, c):\n    return x + (y - c)\n", &cap_sig);
    assert(me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);

    weak_i64 = 40000;
    kernel = prepare("0.2", "def k(x, y, c):\n    return x + (y - c)\n", &cap_sig);
    assert(!me_kernel_is_partitionable(kernel));
    me_kernel_free(kernel);
}

static void loop_partition_proof(void) {
    me_input_spec input = {"x", ME_FLOAT32, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_kernel *kernel = prepare("0.2",
        "def k(x) -> float32:\n"
        "    acc = 0.0\n"
        "    for i in range(4):\n"
        "        if i % 2 == 0:\n"
        "            acc = acc + x + i\n"
        "        else:\n"
        "            acc = acc - x\n"
        "    return acc\n", &sig);
    assert(me_kernel_is_partitionable(kernel));
    assert(me_kernel_inferred_dtype(kernel) == ME_FLOAT64);
    assert(describe(kernel).output_dtype == ME_FLOAT32);
    me_kernel_free(kernel);
    kernel = prepare("0.2",
        "def k(x):\n"
        "    acc = x\n"
        "    for i in range(1000000):\n"
        "        acc = acc + i\n"
        "    return acc\n", &sig);
    assert(!me_kernel_is_partitionable(kernel)); /* Proof work remains bounded. */
    me_kernel_free(kernel);
}

static void preparation_fp_policy(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_prepare_options opts = ME_PREPARE_OPTIONS_INITIALIZER;
    me_kernel *kernel = NULL;
    me_error error;
    const char *modes[] = {"strict", "contract", "fast"};
    opts.jit_compiler = ME_COMPILER_CC;
    for (int mode = ME_FP_STRICT; mode <= ME_FP_FAST; mode++) {
        opts.fp_mode = (me_fp_mode)mode;
        assert(me_prepare_menudet("0.2", "def k(x):\n    return sin(x)\n",
            &sig, &opts, &kernel, &error) == ME_SUCCESS);
        assert(!strcmp(me_kernel_fp_mode(kernel), modes[mode - ME_FP_STRICT]));
        me_kernel_free(kernel);
    }
    /* Authored policy retains precedence over the runtime preference. */
    assert(me_prepare_menudet("0.2", "# me:fp=strict\ndef k(x):\n    return sin(x)\n",
        &sig, &opts, &kernel, &error) == ME_SUCCESS);
    assert(!strcmp(me_kernel_fp_mode(kernel), "strict"));
    me_kernel_free(kernel);
    opts.struct_size = ME_PREPARE_OPTIONS_V4_SIZE;
    opts.fp_mode = (me_fp_mode)999;
    opts.math_accuracy = (me_math_accuracy)999;
    assert(me_prepare_menudet("0.2", "def k(x):\n    return sin(x)\n",
        &sig, &opts, &kernel, &error) == ME_SUCCESS);
    assert(!strcmp(me_kernel_fp_mode(kernel), "strict"));
    me_kernel_free(kernel);
    opts.struct_size = ME_PREPARE_OPTIONS_V5_SIZE;
    assert(me_prepare_menudet("0.2", "def k(x):\n    return sin(x)\n",
        &sig, &opts, &kernel, &error) == ME_ERR_INVALID_ARG);
    assert(!kernel);
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
    me_execution_backend backend = ME_BACKEND_JIT;
    assert(!me_kernel_has_jit(kernel) && !me_kernel_has_jit(NULL));
#ifndef __EMSCRIPTEN__
    feclearexcept(FE_ALL_EXCEPT);
    feraiseexcept(FE_INVALID);
    int flags = fetestexcept(FE_ALL_EXCEPT);
    assert(me_eval_block_backend(kernel, &buffer, 1, (me_mut_buffer){&out, sizeof(out)},
        &block, &run, &backend, &error) == ME_ERR_EVAL);
    assert(backend == ME_BACKEND_INTERPRETER);
    assert(fp.supported && (fp.flags & ME_FP_DIVIDE));
    assert(fetestexcept(FE_ALL_EXCEPT) == flags);
    feclearexcept(FE_ALL_EXCEPT);
#else
    assert(me_eval_block_backend(kernel, &buffer, 1, (me_mut_buffer){&out, sizeof(out)},
        &block, &run, &backend, &error) == ME_ERR_UNSUPPORTED);
    assert(backend == ME_BACKEND_NONE);
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
    snprintf(source, sizeof(source), "def k(x: float32) -> float64:\n    return 2 * x\n");
    options.jit_compiler = tcc ? ME_COMPILER_TCC : ME_COMPILER_CC;
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
    assert(kernel->program->compiler == (tcc ? ME_DSL_COMPILER_LIBTCC : ME_DSL_COMPILER_CC));
    float x[] = {2, 4, 8};
    double out[3] = {0};
    me_buffer buffer = {x, sizeof(x)};
    me_block block = ME_BLOCK_INITIALIZER;
    block.nitems = 3;
    me_execution_backend backend = ME_BACKEND_NONE;
    bool prepared_jit = me_kernel_has_jit(kernel);
    assert(!require_jit || prepared_jit);
    assert(me_eval_block_backend(kernel, &buffer, 1, (me_mut_buffer){out, sizeof(out)},
        &block, NULL, &backend, NULL) == ME_SUCCESS);
    assert(backend == (prepared_jit ? ME_BACKEND_JIT : ME_BACKEND_INTERPRETER));
    assert(out[0] == 4 && out[1] == 8 && out[2] == 16);
    me_run_options run = ME_RUN_OPTIONS_INITIALIZER;
    run.execution = ME_RUN_INTERPRET;
    me_dsl_jit_kernel_fn saved_jit = kernel->program->jit_kernel_fn;
    kernel->program->jit_kernel_fn = forbidden_jit;
    memset(out, 0, sizeof(out));
    assert(me_eval_block_backend(kernel, &buffer, 1, (me_mut_buffer){out, sizeof(out)},
        &block, &run, &backend, NULL) == ME_SUCCESS);
    assert(backend == ME_BACKEND_INTERPRETER);
    assert(out[0] == 4 && out[1] == 8 && out[2] == 16);
    kernel->program->jit_kernel_fn = saved_jit;
    block.nitems = 0;
    assert(me_eval_block_backend(kernel, &buffer, 1, (me_mut_buffer){NULL, 0},
        &block, NULL, &backend, NULL) == ME_SUCCESS);
    assert(backend == ME_BACKEND_NONE);
    block.nitems = 3;
    backend = ME_BACKEND_JIT;
    assert(me_eval_block_backend(kernel, &buffer, 1, (me_mut_buffer){out, 0},
        &block, NULL, &backend, NULL) == ME_ERR_INVALID_ARG);
    assert(backend == ME_BACKEND_NONE);
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
    snprintf(source, sizeof(source), "# me:compiler=%s\ndef k(x):\n"
        "    if block_sum(x) > 0:\n"
        "        return block_sum(x)\n"
        "    return block_sum(x)\n", tcc ? "tcc" : "cc");
    options.execution = require_jit ? ME_EXEC_REQUIRE_JIT : ME_EXEC_PREFER_JIT;
    assert(me_prepare_menudet("0.2", source, &sig, &options, &kernel, &error) == ME_SUCCESS);
    assert(!require_jit || me_kernel_has_jit(kernel));
    block.nitems = 0;
    block.valid_mask = NULL;
    block.valid_mask_capacity = 0;
    float empty_result = -1;
    assert(me_eval_block_backend(kernel, &buffer, 1,
        (me_mut_buffer){&empty_result, sizeof(empty_result)}, &block, NULL,
        &backend, &error) == ME_SUCCESS);
    assert(empty_result == 0 && backend == ME_BACKEND_INTERPRETER);
    me_kernel_free(kernel);
    block.nitems = 3;
    block.valid_mask = mask;
    block.valid_mask_capacity = sizeof(mask);
    options.execution = ME_EXEC_REQUIRE_JIT;
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
    sig.cardinality = ME_ELEMENTWISE;
    options.menudet_forbidden_features = ME_MENUDET_FEATURE_CONTROL_FLOW;
    source = "# me:compiler=cc\ndef reject_flow(x):\n    if x > 0:\n        return x\n    return -x\n";
    assert(me_prepare_menudet("0.1", source, &sig, &options, &kernel, &error) == ME_ERR_SIGNATURE);
    assert(!kernel && strstr(error.message, "forbidden Menudet feature"));
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

static void menudet_feature_options(void) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    me_prepare_options options = ME_PREPARE_OPTIONS_INITIALIZER;
    me_kernel *kernel = NULL;
    const char *reduction = "def k(x):\n    return block_sum(x)\n";
    options.menudet_forbidden_features = ME_MENUDET_FEATURE_BLOCK_REDUCTIONS;
    assert(me_prepare_menudet("0.1", reduction, &sig, &options, &kernel, NULL) == ME_ERR_SIGNATURE);
    assert(!kernel);
    options.struct_size = ME_PREPARE_OPTIONS_V1_SIZE;
    assert(me_prepare_menudet("0.1", reduction, &sig, &options, &kernel, NULL) == ME_SUCCESS);
    me_kernel_free(kernel);
    options.struct_size = sizeof(options);
    options.menudet_forbidden_features = ME_MENUDET_FEATURE_FIXED_STRINGS;
    input = (me_input_spec){"x", ME_BYTES, 4};
    assert(me_prepare_menudet("0.1", "def k(x):\n    return x\n", &sig,
        &options, &kernel, NULL) == ME_ERR_SIGNATURE);
    assert(!kernel);
    input = (me_input_spec){"x", ME_FLOAT64, 0};
    options.menudet_forbidden_features = ME_MENUDET_FEATURE_ND_CONTEXT;
    sig.context_ndim = 1;
    assert(me_prepare_menudet("0.1", "def k(x):\n    return x\n", &sig,
        &options, &kernel, NULL) == ME_ERR_SIGNATURE);
    assert(!kernel);
    sig.context_ndim = 0;
    options.menudet_forbidden_features = ME_MENUDET_FEATURE_FLEXIBLE_MATH;
    assert(me_prepare_menudet("0.1", "# me:compiler=cc\n# me:fp=fast\ndef k(x):\n    return x\n",
        &sig, &options, &kernel, NULL) == ME_ERR_UNSUPPORTED);
    assert(!kernel);
    options.menudet_forbidden_features = UINT32_MAX;
    assert(me_prepare_menudet("0.1", reduction, &sig, &options, &kernel, NULL) == ME_ERR_INVALID_ARG);
    assert(!kernel);

    options = (me_prepare_options)ME_PREPARE_OPTIONS_INITIALIZER;
    options.execution = ME_EXEC_AUTO;
    me_jit_options jit_opts = {0};
    jit_opts.math_backend = "system";
    options.jit_options = &jit_opts;
    assert(me_prepare_menudet("0.1", "def entry_fn(x):\n    return x + 1\n", &sig,
        &options, &kernel, NULL) == ME_SUCCESS);
    assert(kernel != NULL);
    assert(me_kernel_inferred_dtype(kernel) == ME_FLOAT64);
    assert(!strcmp(me_kernel_entry_point(kernel), "entry_fn"));
    assert(!strcmp(me_kernel_execution_kind(kernel), "expression"));
    assert(!strcmp(me_kernel_math_backend(kernel), "system"));
    assert(!strcmp(me_kernel_sleef_accuracy(kernel), ""));
    assert(!strcmp(me_kernel_fp_mode(kernel), "strict"));
    me_kernel_free(kernel);
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
    options.execution = ME_EXEC_INTERPRET;
    options.math_backend = (me_math_backend)99;
    assert(me_prepare_formula("x", &sig, &options, &kernel, &error) == ME_ERR_INVALID_ARG);
    options.math_backend = ME_MATH_DEFAULT;
    options.jit_compiler = (me_jit_compiler)99;
    assert(me_prepare_formula("x", &sig, &options, &kernel, &error) == ME_ERR_INVALID_ARG);
    /* Old prefixes must neither validate nor read future fields. */
    options.math_backend = (me_math_backend)99;
    options.struct_size = ME_PREPARE_OPTIONS_V2_SIZE;
    assert(me_prepare_formula("x", &sig, &options, &kernel, &error) == ME_SUCCESS);
    me_kernel_free(kernel);
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
    for (int backend = ME_MATH_AUTO; backend <= ME_MATH_SCALAR; backend++) {
        options.math_backend = (me_math_backend)backend;
        me_dsl_math_backend expected = (me_dsl_math_backend)(backend - ME_MATH_AUTO);
        me_status status = me_prepare_formula("sin(x)", &sig, &options, &kernel, &error);
        if (!dsl_math_backend_available(expected)) {
            assert(status == ME_ERR_UNSUPPORTED && !kernel);
            continue;
        }
        assert(status == ME_SUCCESS && kernel->formula_math.backend == expected);
        assert(me_eval_block(kernel, &buffer, 1, (me_mut_buffer){&out, sizeof(out)},
            &block, NULL, &error) == ME_SUCCESS);
        assert(isfinite(out));
        me_kernel_free(kernel);
    }
    options.math_backend = ME_MATH_SYSTEM;
    options.jit_compiler = ME_COMPILER_CC;
    assert(me_prepare_menudet("0.2", "def k(x):\n    return sin(x)\n",
        &sig, &options, &kernel, &error) == ME_SUCCESS);
    assert(kernel->program->compiler == ME_DSL_COMPILER_CC);
    assert(kernel->program->math_backend == ME_DSL_MATH_SYSTEM);
    me_kernel_free(kernel);
    /* Authored source remains higher precedence than host default policy. */
    options.math_backend = ME_MATH_SLEEF;
    assert(me_prepare_menudet("0.2", "# me:compiler=tcc\n# me:math=system\n"
        "def k(x):\n    return sin(x)\n", &sig, &options, &kernel, &error) == ME_SUCCESS);
    assert(kernel->program->compiler == ME_DSL_COMPILER_LIBTCC);
    assert(kernel->program->math_backend == ME_DSL_MATH_SYSTEM);
    me_kernel_free(kernel);
}

static void descriptor_prefixes(void) {
    assert(ME_SIGNATURE_V1_SIZE == sizeof(me_signature));
    assert(ME_PREPARE_OPTIONS_V1_SIZE == offsetof(me_prepare_options, menudet_forbidden_features));
    assert(ME_PREPARE_OPTIONS_V2_SIZE == offsetof(me_prepare_options, math_backend));
    assert(ME_PREPARE_OPTIONS_V3_SIZE == offsetof(me_prepare_options, jit_options));
    assert(ME_PREPARE_OPTIONS_V4_SIZE == offsetof(me_prepare_options, fp_mode));
    assert(ME_PREPARE_OPTIONS_V5_SIZE == sizeof(me_prepare_options));
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
    fiber_call calls[] = {{parent, kernel, 2}, {parent, kernel, 128}};
    void *a = CreateFiber(0, eval_fiber, &calls[0]);
    void *b = CreateFiber(0, eval_fiber, &calls[1]);
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

static void public_serial_reducers(void) {
    assert(me_reduce_accumulator_dtype(ME_BOOL, ME_REDUCTION_SUM, ME_AUTO) == ME_INT64);
    assert(me_reduce_accumulator_dtype(ME_INT8, ME_REDUCTION_PROD, ME_AUTO) == ME_INT64);
    assert(me_reduce_accumulator_dtype(ME_UINT16, ME_REDUCTION_SUM, ME_AUTO) == ME_UINT64);
    assert(me_reduce_accumulator_dtype(ME_FLOAT32, ME_REDUCTION_SUM, ME_AUTO) == ME_FLOAT32);
    assert(me_reduce_accumulator_dtype(ME_COMPLEX64, ME_REDUCTION_PROD, ME_AUTO) == ME_COMPLEX64);
    assert(me_reduce_accumulator_dtype(ME_COMPLEX128, ME_REDUCTION_SUM, ME_FLOAT64) == ME_FLOAT64);
    assert(me_reduce_accumulator_dtype(ME_FLOAT32, ME_REDUCTION_SUM, ME_INT64) == ME_AUTO);
    assert(me_reduce_accumulator_dtype(ME_INT64, ME_REDUCTION_ANY, ME_AUTO) == ME_BOOL);
    assert(me_reduce_accumulator_dtype(ME_INT64, ME_REDUCTION_ALL, ME_BOOL) == ME_AUTO);
    assert(me_reduce_accumulator_dtype(ME_STRING, ME_REDUCTION_SUM, ME_AUTO) == ME_AUTO);
    int16_t narrow[] = {-32768, 32767, 2};
    uint64_t result = 42;
    assert(me_sum_span(ME_INT16, narrow, 3, 0, false, &result) && result == 1);
    int64_t signed_values[] = {INT64_MAX, 1};
    assert(me_sum_span(ME_INT64, signed_values, 2, 0, false, &result));
    assert(result == (UINT64_C(1) << 63));
    result = 42;
    assert(!me_sum_span(ME_INT64, signed_values, 2, 0, true, &result) && result == 42);
    uint64_t unsigned_values[] = {UINT64_MAX, 2};
    assert(me_prod_span(ME_UINT64, unsigned_values, 2, 1, false, &result) && result == UINT64_MAX - 1);
    result = 42;
    assert(!me_prod_span(ME_UINT64, unsigned_values, 2, 1, true, &result) && result == 42);
    float floats[] = {16777216.0f, 1, -16777216.0f, 3};
    me_fp_scope scope;
    me_fp_status fp;
    assert(me_fp_scope_begin(&scope, &fp, NULL) == ME_SUCCESS);
    assert(me_sum_span(ME_FLOAT32, floats, 2, 0, false, &result));
    assert(me_sum_span(ME_FLOAT32, floats + 2, 2, result, false, &result));
    float total;
    memcpy(&total, &result, sizeof(total));
    assert(total == 3);
    assert(me_fp_scope_end(&scope, NULL) == ME_SUCCESS);
    result = 42;
    assert(!me_sum_span(ME_COMPLEX128, floats, 0, 0, false, &result) && result == 42);
    assert(me_sum_span(ME_INT64, NULL, 0, 17, false, &result) && result == 17);

    /* Non-short-circuiting truth reductions and endian-safe scalar stores. */
    bool truth = false;
    uint8_t bools[] = {0, 0, 1};
    assert(me_truth_span(ME_BOOL, bools, 3, false, false, &truth) && truth);
    assert(me_truth_span(ME_BOOL, bools, 3, true, true, &truth) && !truth);
    assert(me_truth_span(ME_INT16, NULL, 0, true, true, &truth) && truth);
    int16_t i16_vals[] = {0, -5};
    assert(me_truth_span(ME_INT16, i16_vals, 2, false, false, &truth) && truth);
    uint64_t u64_vals[] = {1, 2};
    assert(me_truth_span(ME_UINT64, u64_vals, 2, true, true, &truth) && truth);
    double qnan_vals[] = {0.0, NAN};
    assert(me_fp_scope_begin(&scope, &fp, NULL) == ME_SUCCESS);
    assert(me_truth_span(ME_FLOAT64, qnan_vals, 2, false, false, &truth) && truth);
    assert(me_fp_scope_end(&scope, NULL) == ME_SUCCESS);
    assert(fp.flags == 0);
#ifndef __EMSCRIPTEN__
    uint32_t snan32_bits = UINT32_C(0x7f800001);
    float snan32;
    memcpy(&snan32, &snan32_bits, sizeof(snan32));
    float late_snan32[] = {1.0f, snan32};
    assert(me_fp_scope_begin(&scope, &fp, NULL) == ME_SUCCESS);
    assert(me_truth_span(ME_FLOAT32, late_snan32, 2, false, false, &truth) && truth);
    assert(me_fp_scope_end(&scope, NULL) == ME_SUCCESS);
    assert(fp.flags & ME_FP_INVALID);
    double zero_snan64[2] = {0.0, 0.0};
    uint64_t snan64_bits = UINT64_C(0x7ff0000000000001);
    memcpy(&zero_snan64[1], &snan64_bits, sizeof(snan64_bits));
    assert(me_fp_scope_begin(&scope, &fp, NULL) == ME_SUCCESS);
    assert(me_truth_span(ME_FLOAT64, zero_snan64, 2, true, true, &truth) && !truth);
    assert(me_fp_scope_end(&scope, NULL) == ME_SUCCESS);
    assert(fp.flags & ME_FP_INVALID);
#endif
    assert(!me_truth_span(ME_COMPLEX64, floats, 0, false, false, &truth));
    assert(!me_truth_span(ME_STRING, NULL, 0, false, false, &truth));

    int8_t s8 = 0;
    uint16_t u16 = 0;
    int64_t s64 = 0;
    float f32_out = 0;
    uint8_t b_out = 9;
    assert(me_reduce_store_scalar(ME_INT8, (uint64_t)(int64_t)-5, &s8) && s8 == -5);
    assert(me_reduce_store_scalar(ME_UINT16, 40000u, &u16) && u16 == 40000u);
    assert(me_reduce_store_scalar(ME_INT64, (uint64_t)(int64_t)-1234567890123LL, &s64) && s64 == -1234567890123LL);
    assert(me_reduce_store_scalar(ME_BOOL, 1u, &b_out) && b_out == 1);
    float f32_src = 3.5f;
    uint64_t f32_store = 0;
    memcpy(&f32_store, &f32_src, sizeof(f32_src));
    assert(me_reduce_store_scalar(ME_FLOAT32, f32_store, &f32_out) && f32_out == 3.5f);
    assert(!me_reduce_store_scalar(ME_COMPLEX64, 0, &s64));
}

int main(int argc, char **argv) {
    partition_proof();
    loop_partition_proof();
    preparation_fp_policy();
    public_serial_reducers();
    /* Host reducers use precisely the kernel strict-FP scope. */
    me_fp_scope scope;
    me_fp_status fp;
    me_error scope_error;
    assert(me_fp_scope_begin(NULL, &fp, &scope_error) == ME_ERR_INVALID_ARG);
#ifndef __EMSCRIPTEN__
    int original_rounding = fegetround();
    assert(fesetround(FE_DOWNWARD) == 0);
    assert(me_fp_scope_begin(&scope, &fp, &scope_error) == ME_SUCCESS);
    assert(fp.supported && fegetround() == FE_TONEAREST);
    feraiseexcept(FE_INVALID);
    assert(me_fp_scope_end(&scope, &scope_error) == ME_SUCCESS);
    assert(fp.flags & ME_FP_INVALID);
    assert(fegetround() == FE_DOWNWARD);
    assert(fesetround(original_rounding) == 0);
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t original_fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(original_fpcr) :: "memory");
    uint64_t flush_fpcr = original_fpcr | (UINT64_C(1) << 24);
    __asm__ __volatile__("msr fpcr, %0" :: "r"(flush_fpcr) : "memory");
    me_status flush_status = me_fp_scope_begin(&scope, &fp, &scope_error);
    __asm__ __volatile__("msr fpcr, %0" :: "r"(original_fpcr) : "memory");
    assert(flush_status == ME_ERR_UNSUPPORTED);
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    uint32_t original_mxcsr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(original_mxcsr) :: "memory");
    uint32_t flush_mxcsr = original_mxcsr | 0x8040u;
    __asm__ __volatile__("ldmxcsr %0" :: "m"(flush_mxcsr) : "memory");
    me_status flush_status = me_fp_scope_begin(&scope, &fp, &scope_error);
    __asm__ __volatile__("ldmxcsr %0" :: "m"(original_mxcsr) : "memory");
    assert(flush_status == ME_ERR_UNSUPPORTED);
#endif
#else
    assert(me_fp_scope_begin(&scope, &fp, &scope_error) == ME_SUCCESS);
    assert(!fp.supported);
    assert(me_fp_scope_end(&scope, &scope_error) == ME_SUCCESS);
#endif
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
    menudet_feature_options();
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
