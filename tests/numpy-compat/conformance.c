/* M2 native conformance: the corpus executes without Python/NumPy. */
#include "vector_io.h"
#include <fenv.h>
#include <math.h>
#include <stdio.h>

#define VECTOR_LIMIT 4096

static const char *category(int status) {
    switch (status) {
    case 0: return "success";
    case -1: return "invalid_artifact";
    case -2: return "unsupported_requirement";
    case -3: return "invalid_source";
    case -4: return "binding_error";
    case -5: return "evaluation_error";
    case -6: return "out_of_memory";
    default: return "native_error";
    }
}

static const char *type_name(me_dtype type) {
    switch (type) {
    case ME_BOOL: return "bool";
    case ME_INT8: return "int8";
    case ME_INT16: return "int16";
    case ME_INT32: return "int32";
    case ME_INT64: return "int64";
    case ME_UINT8: return "uint8";
    case ME_UINT16: return "uint16";
    case ME_UINT32: return "uint32";
    case ME_UINT64: return "uint64";
    case ME_FLOAT32: return "float32";
    case ME_FLOAT64: return "float64";
    default: return "unsupported";
    }
}

static int shape(yyjson_val *value, int64_t *dims, int *rank, size_t *count) {
    if (!yyjson_is_arr(value) || yyjson_arr_size(value) > 8) return 0;
    *rank = (int)yyjson_arr_size(value);
    *count = 1;
    for (int i = 0; i < *rank; i++) {
        yyjson_val *dim = yyjson_arr_get(value, (size_t)i);
        if (!yyjson_is_uint(dim) || yyjson_get_uint(dim) > VECTOR_LIMIT) return 0;
        dims[i] = (int64_t)yyjson_get_uint(dim);
        if (dims[i] && *count > VECTOR_LIMIT / (size_t)dims[i]) return 0;
        *count *= (size_t)dims[i];
    }
    return 1;
}

static int same_shape(yyjson_val *expected, const int64_t *dims, int rank) {
    int other_rank;
    int64_t other_dims[8];
    size_t count;
    if (!shape(expected, other_dims, &other_rank, &count) || rank != other_rank) return 0;
    for (int i = 0; i < rank; i++) if (dims[i] != other_dims[i]) return 0;
    return 1;
}

/* Reconstruct the physical recipe, then normalize to the contiguous native API.
 * Values are canonical logical C-order bytes. This is a copying host adapter,
 * not a claim that the runtime accepts foreign endian or negative strides. */
static void *input_buffer(yyjson_val *item, size_t *count, int64_t *dims, int *rank,
                          size_t *normalization_bytes) {
    me_dtype type = dtype(text(item, "dtype"));
    size_t width = vector_width(type);
    if (!width || !text(item, "encoding") || strcmp(text(item, "encoding"), "raw-be-hex") ||
        !shape(yyjson_obj_get(item, "shape"), dims, rank, count)) return NULL;
    unsigned char *logical = calloc(*count ? *count : 1, width);
    if (!logical || !decode(text(item, "hex"), logical, width, *count)) {
        free(logical);
        return NULL;
    }
    yyjson_val *layout = yyjson_obj_get(item, "layout");
    const char *order = text(layout, "order"), *byteorder = text(layout, "byteorder");
    yyjson_val *step_val = yyjson_obj_get(layout, "step"), *reverse = yyjson_obj_get(layout, "reverse_axis");
    if (!order || (strcmp(order, "C") && strcmp(order, "F")) || !byteorder ||
        (strcmp(byteorder, "native") && strcmp(byteorder, "little") && strcmp(byteorder, "big")) ||
        !yyjson_is_uint(step_val) || yyjson_get_uint(step_val) < 1 || yyjson_get_uint(step_val) > 4 ||
        (!yyjson_is_null(reverse) && (!yyjson_is_uint(reverse) || yyjson_get_uint(reverse) >= (size_t)*rank))) {
        free(logical);
        return NULL;
    }
    size_t step = (size_t)yyjson_get_uint(step_val), strides[8], storage_count = 1;
    int reversed = yyjson_is_null(reverse) ? -1 : (int)yyjson_get_uint(reverse);
    int foreign_little = !strcmp(byteorder, "native") ? little_endian() : !strcmp(byteorder, "little");
    for (int k = 0; k < *rank; k++) {
        int i = !strcmp(order, "F") ? k : *rank - 1 - k;
        strides[i] = storage_count;
        storage_count *= (size_t)dims[i] * (i == *rank - 1 ? step : 1);
    }
    unsigned char *physical = calloc(storage_count ? storage_count : 1, width);
    if (!physical) {
        free(logical);
        return NULL;
    }
    for (size_t lane = 0; lane < *count; lane++) {
        size_t rem = lane, offset = 0;
        for (int i = *rank - 1; i >= 0; i--) {
            size_t coordinate = rem % (size_t)dims[i];
            rem /= (size_t)dims[i];
            if (i == reversed) coordinate = (size_t)dims[i] - 1 - coordinate;
            offset += coordinate * strides[i] * (i == *rank - 1 ? step : 1);
        }
        for (size_t byte = 0; byte < width; byte++) {
            size_t source = foreign_little == little_endian() ? byte : width - 1 - byte;
            physical[offset * width + byte] = logical[lane * width + source];
        }
    }
    for (size_t lane = 0; lane < *count; lane++) {
        size_t rem = lane, offset = 0;
        for (int i = *rank - 1; i >= 0; i--) {
            size_t coordinate = rem % (size_t)dims[i];
            rem /= (size_t)dims[i];
            if (i == reversed) coordinate = (size_t)dims[i] - 1 - coordinate;
            offset += coordinate * strides[i] * (i == *rank - 1 ? step : 1);
        }
        for (size_t byte = 0; byte < width; byte++) {
            size_t source = foreign_little == little_endian() ? byte : width - 1 - byte;
            logical[lane * width + byte] = physical[offset * width + source];
        }
    }
    *normalization_bytes += *count * width;
    free(physical);
    return logical;
}

static uint64_t ordered_bits(const unsigned char *value, me_dtype type) {
    uint64_t raw = 0;
    uint64_t sign = type == ME_FLOAT32 ? UINT64_C(1) << 31 : UINT64_C(1) << 63;
    uint64_t mask = type == ME_FLOAT32 ? UINT32_MAX : UINT64_MAX;
    if (type == ME_FLOAT32) {
        uint32_t small;
        memcpy(&small, value, sizeof(small));
        raw = small;
    }
    else memcpy(&raw, value, sizeof(raw));
    return raw & sign ? ~raw & mask : raw | sign;
}

static int valid_policy(yyjson_val *policy) {
    const char *kind = text(policy, "kind"), *nan = text(policy, "nan"), *zero = text(policy, "signed_zero");
    if (!kind || (strcmp(kind, "bitwise") && strcmp(kind, "exact") && strcmp(kind, "ulp") && strcmp(kind, "tolerance")) ||
        !nan || (strcmp(nan, "bits") && strcmp(nan, "equal")) ||
        !zero || (strcmp(zero, "exact") && strcmp(zero, "ignore"))) return 0;
    yyjson_val *atol = yyjson_obj_get(policy, "atol"), *rtol = yyjson_obj_get(policy, "rtol");
    yyjson_val *ulp = yyjson_obj_get(policy, "max_ulp");
    return yyjson_is_num(atol) && yyjson_is_num(rtol) && yyjson_is_uint(ulp) &&
        isfinite(yyjson_get_num(atol)) && isfinite(yyjson_get_num(rtol)) &&
        yyjson_get_num(atol) >= 0 && yyjson_get_num(rtol) >= 0;
}

/* -1 invalid reference, 0 mismatch, 1 match; dtype and shape are mandatory. */
static int compare_output(yyjson_val *expected, const void *actual, me_dtype type,
                           const int64_t *dims, int rank, size_t count, yyjson_val *policy) {
    if (!text(expected, "dtype") || !text(expected, "encoding") ||
        strcmp(text(expected, "encoding"), "raw-be-hex")) return -1;
    if (dtype(text(expected, "dtype")) != type || !same_shape(yyjson_obj_get(expected, "shape"), dims, rank)) return 0;
    size_t width = vector_width(type);
    unsigned char *reference = malloc((count ? count : 1) * width);
    if (!reference || !decode(text(expected, "hex"), reference, width, count)) {
        free(reference);
        return -1;
    }
    int match = 1;
    const char *kind = text(policy, "kind");
    for (size_t i = 0; i < count; i++) {
        const unsigned char *a = (const unsigned char *)actual + i * width, *e = reference + i * width;
        int equal = !memcmp(a, e, width);
        if (type == ME_FLOAT32 || type == ME_FLOAT64) {
            float af, ef;
            double ad, ed;
            if (type == ME_FLOAT32) {
                memcpy(&af, a, 4);
                memcpy(&ef, e, 4);
                ad = af;
                ed = ef;
            }
            else {
                memcpy(&ad, a, 8);
                memcpy(&ed, e, 8);
            }
            if (isnan(ad) || isnan(ed)) equal = isnan(ad) && isnan(ed) &&
                (!strcmp(text(policy, "nan"), "equal") || equal);
            else if (ad == 0 && ed == 0) equal = !strcmp(text(policy, "signed_zero"), "ignore") ||
                (!!signbit(ad) == !!signbit(ed));
            else if (isinf(ad) || isinf(ed) || !strcmp(kind, "exact")) equal = ad == ed;
            else if (strcmp(kind, "bitwise")) {
                uint64_t ab = ordered_bits(a, type), eb = ordered_bits(e, type);
                uint64_t distance = ab > eb ? ab - eb : eb - ab;
                equal = distance <= yyjson_get_uint(yyjson_obj_get(policy, "max_ulp"));
                if (!strcmp(kind, "tolerance")) equal |= fabs(ad - ed) <=
                    yyjson_get_num(yyjson_obj_get(policy, "atol")) +
                    yyjson_get_num(yyjson_obj_get(policy, "rtol")) * fabs(ed);
            }
        }
        if (!equal) { match = 0; break; }
    }
    free(reference);
    return match;
}

static int available_capability(const char *cap) {
    return !strcmp(cap, "numeric") || !strcmp(cap, "control-flow") || !strcmp(cap, "block-reductions") ||
        !strcmp(cap, "nd-context") || !strcmp(cap, "fixed-strings") || !strcmp(cap, "layout-copying");
}

static int requirements(yyjson_val *test) {
    yyjson_val *values = yyjson_obj_get(test, "requires");
    if (!yyjson_is_arr(values)) return -1;
    int supported = 1;
    for (size_t i = 0; i < yyjson_arr_size(values); i++) {
        const char *cap = yyjson_get_str(yyjson_arr_get(values, i));
        if (!cap) return -1;
        for (const char *p = cap; *p; p++) {
            if (!((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '-' || *p == ':' || *p == '.' || *p == '_')) return -1;
        }
        if (!available_capability(cap)) supported = 0;
    }
    return supported;
}

static int run_v2_case(yyjson_val *test, me_jit_mode mode, bool observe) {
    const char *id = text(test, "id"), *json = text(test, "artifact");
    yyjson_val *policy = yyjson_obj_get(test, "comparison");
    if (!id || !*id || !json || !text(test, "semantic_revision") ||
        (strcmp(text(test, "semantic_revision"), "menudet-draft-1.0-checked") &&
         strcmp(text(test, "semantic_revision"), "menudet-numpy-1.1")) || !valid_policy(policy)) return 1;
    for (const char *p = id; *p; p++) if (!((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '-')) return 1;
    int required = requirements(test);
    if (required < 0) return 1;
    if (!required) {
        printf("{\"id\":\"%s\",\"outcome\":\"skipped\",\"backend\":\"none\","
               "\"skip_reason\":\"unsupported required capability\",\"skipped_capabilities\":[", id);
        yyjson_val *caps = yyjson_obj_get(test, "requires");
        int printed = 0;
        for (size_t i = 0; i < yyjson_arr_size(caps); i++) {
            const char *cap = yyjson_get_str(yyjson_arr_get(caps, i));
            if (!available_capability(cap)) {
                printf("%s\"%s\"", printed ? "," : "", cap);
                printed = 1;
            }
        }
        printf("]}\n");
        return 0;
    }
    me_artifact *artifact = NULL;
    me_artifact_error error = {0};
    int rc = me_artifact_load(json, strlen(json), mode, &artifact, &error);
    int failed = 0, reference_match = 0, regression = 0, recovery_ok = 1, environment_ok = 1;
    me_artifact_buffer inputs[16] = {{0}};
    void *buffers[16] = {0}, *output = NULL, *repeat_output = NULL;
    int64_t dims[8] = {0}, origin[8] = {0};
    int rank = 0, output_rank = 0;
    size_t count = 0, output_count = 0, width = 0, normalized = 0;
    me_dtype output_type = ME_AUTO;
    char hex[65537] = "";
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION};
    if (!rc) {
        yyjson_val *items = yyjson_obj_get(test, "inputs");
        size_t ninputs = yyjson_arr_size(items);
        if (!yyjson_is_arr(items) || !ninputs || ninputs > 16 || (int)ninputs != me_artifact_ninputs(artifact) ||
            me_artifact_has_jit(artifact)) { failed = 1; goto cleanup; }
        for (size_t i = 0; i < ninputs; i++) {
            yyjson_val *item = yyjson_arr_get(items, i);
            int64_t other_dims[8] = {0};
            int other_rank;
            size_t other_count;
            buffers[i] = input_buffer(item, &other_count, other_dims, &other_rank, &normalized);
            if (!buffers[i] || (i && (other_count != count || !same_shape(yyjson_obj_get(item, "shape"), dims, rank)))) {
                failed = 1; goto cleanup;
            }
            if (!i) { count = other_count; rank = other_rank; memcpy(dims, other_dims, sizeof(dims)); }
            inputs[i].name = text(item, "name");
            inputs[i].dtype = dtype(text(item, "dtype"));
            inputs[i].itemsize = vector_width(inputs[i].dtype);
            inputs[i].capacity = other_count * inputs[i].itemsize;
            inputs[i].data = buffers[i];
        }
        output_type = me_artifact_output_dtype(artifact);
        width = vector_width(output_type);
        output_count = me_artifact_result_cardinality(artifact) == ME_ARTIFACT_BLOCK_SCALAR ? 1 : count;
        output_rank = me_artifact_result_cardinality(artifact) == ME_ARTIFACT_BLOCK_SCALAR ? 0 : rank;
        if (!width || output_count > VECTOR_LIMIT) { failed = 1; goto cleanup; }
        output = calloc(output_count ? output_count : 1, width);
        repeat_output = calloc(output_count ? output_count : 1, width);
        if (!output || !repeat_output) { failed = 1; goto cleanup; }
        descriptor.nitems = count;
        descriptor.output_capacity = output_count * width;
        descriptor.ndim = me_artifact_context_ndim(artifact);
        if (descriptor.ndim && descriptor.ndim != rank) { failed = 1; goto cleanup; }
        if (descriptor.ndim) {
            descriptor.logical_shape = dims;
            descriptor.block_origin = origin;
            descriptor.block_extent = dims;
        }
        fenv_t saved;
        int caller_round = FE_TONEAREST;
        if (fegetenv(&saved)) { failed = 1; goto cleanup; }
#ifndef __EMSCRIPTEN__
        caller_round = FE_DOWNWARD;
        if (fesetround(caller_round) || feraiseexcept(FE_DIVBYZERO)) {
            failed = 1; goto cleanup;
        }
#endif
        int flags = fetestexcept(FE_ALL_EXCEPT);
        yyjson_val *fp_expected = yyjson_obj_get(test, "fp_expected");
        if (fp_expected) {
            me_artifact_fp_status fp_status;
            rc = me_artifact_eval_status(artifact, inputs, (int)ninputs, output, &descriptor, 0, &fp_status, &error);
            if (fp_status.supported && fp_status.flags != yyjson_get_uint(fp_expected)) failed = 1;
            if (!rc && yyjson_get_uint(fp_expected)) {
                me_artifact_error raised_error;
                me_artifact_fp_status raised_status;
                int raised = me_artifact_eval_status(artifact, inputs, (int)ninputs, repeat_output, &descriptor,
                    (unsigned)yyjson_get_uint(fp_expected), &raised_status, &raised_error);
                if (fp_status.supported ? raised != ME_ARTIFACT_ERR_EVAL || raised_status.flags != fp_status.flags :
                                          raised != ME_ARTIFACT_ERR_UNSUPPORTED) failed = 1;
            }
        }
        else rc = me_artifact_eval_ex(artifact, inputs, (int)ninputs, output, &descriptor, &error);
        environment_ok &= fegetround() == caller_round && fetestexcept(FE_ALL_EXCEPT) == flags;
        for (int repeat = 0; repeat < 3; repeat++) {
            int again = me_artifact_eval_ex(artifact, inputs, (int)ninputs, repeat_output, &descriptor, &error);
            environment_ok &= fegetround() == caller_round && fetestexcept(FE_ALL_EXCEPT) == flags;
            if (again != rc || (!rc && (error.native_status || error.message[0] ||
                memcmp(output, repeat_output, output_count * width)))) failed = 1;
        }
        if (fesetenv(&saved)) failed = 1;
        yyjson_val *recovery = yyjson_obj_get(test, "recovery");
        if (recovery) {
            yyjson_val *recovery_inputs = yyjson_obj_get(recovery, "inputs");
            if (yyjson_arr_size(recovery_inputs) != ninputs) { failed = 1; goto cleanup; }
            for (size_t i = 0; i < ninputs; i++) {
                int64_t other_dims[8];
                int other_rank;
                size_t other_count;
                void *next = input_buffer(yyjson_arr_get(recovery_inputs, i), &other_count, other_dims, &other_rank, &normalized);
                if (!next || other_count != count) { free(next); failed = 1; goto cleanup; }
                free(buffers[i]);
                buffers[i] = next;
                inputs[i].data = next;
            }
            me_artifact_error recovery_error;
            memset(&recovery_error, 0x55, sizeof(recovery_error));
            recovery_ok = me_artifact_eval_ex(artifact, inputs, (int)ninputs, repeat_output, &descriptor, &recovery_error) == 0 &&
                recovery_error.native_status == 0 && !recovery_error.message[0] &&
                compare_output(yyjson_obj_get(recovery, "expected"), repeat_output, output_type, dims,
                               output_rank, output_count, policy) == 1;
        }
        if (!rc) encode(output, width, output_count, hex);
    }
    yyjson_val *expected = yyjson_obj_get(test, "expected"), *diagnostic = yyjson_obj_get(expected, "diagnostic");
    if (diagnostic) {
        const char *expected_category = text(diagnostic, "category");
        if (!expected_category) { failed = 1; goto cleanup; }
        reference_match = !strcmp(expected_category, category(rc));
    }
    else if (!rc) {
        reference_match = compare_output(expected, output, output_type, dims, output_rank, output_count, policy);
        if (reference_match < 0) { failed = 1; goto cleanup; }
    }
    yyjson_val *baseline = yyjson_obj_get(test, "baseline");
    if (baseline) {
        if (!yyjson_is_int(yyjson_obj_get(baseline, "status"))) { failed = 1; goto cleanup; }
        regression = yyjson_get_int(yyjson_obj_get(baseline, "status")) != rc;
        if (!rc && !regression) {
            int match = compare_output(yyjson_obj_get(baseline, "output"), output, output_type, dims,
                                       output_rank, output_count, policy);
            if (match < 0) { failed = 1; goto cleanup; }
            regression = !match;
        }
    }
    yyjson_val *divergence = yyjson_obj_get(test, "divergence");
    bool reviewed = divergence && text(divergence, "id") && text(divergence, "reason");
    const char *outcome = regression ? "regression" : reference_match ? "matching" :
        baseline && reviewed ? "known_divergence" : "mismatch";
    printf("{\"id\":\"%s\",\"status\":%d,\"category\":\"%s\",\"native_status\":%d,"
           "\"backend\":\"%s\",\"requested_backend\":\"%s\",\"jit_eligible\":false,"
           "\"jit_skip_reason\":\"portable draft has no eligible JIT route\",\"reference_match\":%s,"
           "\"baseline_regression\":%s,\"outcome\":\"%s\",\"normalization_bytes\":%zu,"
           "\"environment_restored\":%s,\"recovery_ok\":%s,\"mismatches\":[%s]",
           id, rc, category(rc), error.native_status, artifact ? "interpreter" : "none", mode == ME_JIT_ON ? "jit" : "interpreter",
           reference_match ? "true" : "false", regression ? "true" : "false", outcome, normalized,
           environment_ok ? "true" : "false", recovery_ok ? "true" : "false",
           reference_match ? "" : rc || diagnostic ? "\"diagnostic\"" : "\"dtype-shape-or-values\"");
    if (!rc) {
        printf(",\"output\":{\"dtype\":\"%s\",\"shape\":[", type_name(output_type));
        for (int i = 0; i < output_rank; i++) printf("%s%lld", i ? "," : "", (long long)dims[i]);
        printf("],\"encoding\":\"raw-be-hex\",\"hex\":\"%s\"}", hex);
    }
    printf("}\n");
    if (!rc && me_artifact_inferred_dtype(artifact) != ME_AUTO) {
        yyjson_val *inferred = yyjson_obj_get(test, "inferred_dtype");
        if (inferred && dtype(yyjson_get_str(inferred)) != me_artifact_inferred_dtype(artifact)) failed = 1;
    }
    failed |= !environment_ok || !recovery_ok;
    if (!observe) failed |= regression || (!reference_match && !(baseline && reviewed));
cleanup:
    if (failed) fprintf(stderr, "invalid vector or unexpected conformance failure: %s\n", id);
    for (int i = 0; i < 16; i++) free(buffers[i]);
    free(output);
    free(repeat_output);
    me_artifact_free(artifact);
    return failed;
}

int numpy_compat_run_v2(yyjson_val *root, me_jit_mode mode, bool observe) {
    yyjson_val *cases = yyjson_obj_get(root, "cases");
    if (!text(root, "generator_revision") || !text(yyjson_obj_get(root, "reference"), "numpy") ||
        !yyjson_is_obj(yyjson_obj_get(root, "provenance")) || !yyjson_is_arr(cases) || !yyjson_arr_size(cases)) return 1;
    int failed = 0;
    for (size_t i = 0; i < yyjson_arr_size(cases); i++) {
        yyjson_val *test = yyjson_arr_get(cases, i);
        const char *id = text(test, "id");
        for (size_t j = 0; id && j < i; j++) {
            const char *previous = text(yyjson_arr_get(cases, j), "id");
            if (previous && !strcmp(id, previous)) return 1;
        }
        failed |= run_v2_case(test, mode, observe);
    }
    return failed;
}
