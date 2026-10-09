/* Standalone, reviewed vector runner. No Python runtime or source preprocessing.
 * Fixed limits intentionally cover only checkpoint-1 contiguous numeric vectors. */
#include "miniexpr_artifact.h"
#include "yyjson.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *text(yyjson_val *object, const char *key) {
    return yyjson_get_str(yyjson_obj_get(object, key));
}

static me_dtype dtype(const char *name) {
    static const char *names[] = {"bool", "int8", "int16", "int32", "int64",
                                  "uint8", "uint16", "uint32", "uint64", "float32", "float64"};
    static const me_dtype types[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
                                    ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    for (size_t i = 0; name && i < sizeof(types) / sizeof(types[0]); i++) {
        if (!strcmp(name, names[i])) return types[i];
    }
    return ME_AUTO;
}

static int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int little_endian(void) {
    const uint16_t one = 1;
    return *(const unsigned char *)&one;
}

static int decode(const char *hex, unsigned char *out, size_t width, size_t count) {
    if (!hex || strlen(hex) != width * count * 2) return 0;
    for (size_t i = 0; i < width * count; i++) {
        int high = nibble(hex[2 * i]), low = nibble(hex[2 * i + 1]);
        if (high < 0 || low < 0) return 0;
        size_t offset = little_endian() ? (i / width) * width + width - 1 - i % width : i;
        out[offset] = (unsigned char)(high * 16 + low);
    }
    return 1;
}

static void encode(const unsigned char *data, size_t width, size_t count, char *out) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < width * count; i++) {
        size_t offset = little_endian() ? (i / width) * width + width - 1 - i % width : i;
        out[2 * i] = digits[data[offset] >> 4];
        out[2 * i + 1] = digits[data[offset] & 15];
    }
    out[2 * width * count] = 0;
}

static int run_case(yyjson_val *test, me_jit_mode mode) {
    const char *id = text(test, "id"), *json = text(test, "artifact");
    if (!id || !json || !text(test, "semantic_revision") ||
        strcmp(text(test, "semantic_revision"), "menudet-draft-1.0-checked")) return 1;
    for (const char *p = id; *p; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '-')) return 1;
    }
    me_artifact *artifact = NULL;
    me_artifact_error error = {0};
    clock_t start = clock();
    int rc = me_artifact_load(json, strlen(json), mode, &artifact, &error);
    double compile_ns = (double)(clock() - start) * 1e9 / CLOCKS_PER_SEC;
    double evaluation_ns = 0;
    int failed = 0;
    char hex[65537] = "";
    me_artifact_input inputs[16] = {{0}};
    void *buffers[16] = {0};
    void *output = NULL;
    if (rc == ME_ARTIFACT_SUCCESS) {
        yyjson_val *items = yyjson_obj_get(test, "inputs");
        size_t ninputs = yyjson_arr_size(items), count = 0;
        if (ninputs == 0 || ninputs > 16 || (int)ninputs != me_artifact_ninputs(artifact) ||
            me_artifact_has_jit(artifact)) {
            failed = 1;
            goto cleanup;
        }
        for (size_t i = 0; i < ninputs; i++) {
            yyjson_val *item = yyjson_arr_get(items, i);
            yyjson_val *shape = yyjson_obj_get(item, "shape");
            size_t length = 1;
            if (!yyjson_is_arr(shape) || yyjson_arr_size(shape) > 8 ||
                !text(item, "encoding") || strcmp(text(item, "encoding"), "raw-be-hex")) {
                failed = 1;
                goto cleanup;
            }
            for (size_t j = 0; j < yyjson_arr_size(shape); j++) {
                yyjson_val *dim = yyjson_arr_get(shape, j);
                if (!yyjson_is_uint(dim) || yyjson_get_uint(dim) > 4096 ||
                    (yyjson_get_uint(dim) && length > 4096 / yyjson_get_uint(dim))) {
                    failed = 1;
                    goto cleanup;
                }
                length *= (size_t)yyjson_get_uint(dim);
            }
            size_t width = me_artifact_input_itemsize(artifact, (int)i);
            if ((i && length != count) || !width || width > 8) {
                failed = 1;
                goto cleanup;
            }
            count = length;
            buffers[i] = calloc(length ? length : 1, width);
            if (!buffers[i] || !decode(text(item, "hex"), buffers[i], width, length)) {
                failed = 1;
                goto cleanup;
            }
            inputs[i].name = text(item, "name");
            inputs[i].dtype = dtype(text(item, "dtype"));
            inputs[i].data = buffers[i];
            inputs[i].nitems = length;
        }
        size_t width = me_artifact_output_itemsize(artifact);
        if (!width || width > 8) {
            failed = 1;
            goto cleanup;
        }
        output = calloc(count ? count : 1, width);
        if (!output) {
            failed = 1;
            goto cleanup;
        }
        start = clock();
        rc = me_artifact_eval(artifact, inputs, (int)ninputs, output, count, &error);
        /* Failure must not poison an owned handle; repeat with independent output. */
        for (int repeat = 0; repeat < 100; repeat++) {
            int again = me_artifact_eval(artifact, inputs, (int)ninputs, output, count, &error);
            if (again != rc) failed = 1;
        }
        evaluation_ns = (double)(clock() - start) * 1e9 / CLOCKS_PER_SEC / 101;
        if (!rc) encode(output, width, count, hex);
    }
    printf("{\"id\":\"%s\",\"status\":%d,\"native_status\":%d,\"backend\":\"interpreter\",\"hex\":\"%s\","
           "\"compile_cpu_ns\":%.0f,\"evaluation_cpu_ns\":%.0f}\n",
           id, rc, error.native_status, hex, compile_ns, evaluation_ns);
    yyjson_val *baseline = yyjson_obj_get(test, "baseline");
    if (baseline) {
        if (!yyjson_is_int(yyjson_obj_get(baseline, "status")) ||
            yyjson_get_int(yyjson_obj_get(baseline, "status")) != rc ||
            !text(baseline, "hex") || strcmp(text(baseline, "hex"), hex)) failed = 1;
    }
cleanup:
    if (failed) fprintf(stderr, "invalid vector or changed baseline: %s\n", id);
    for (int i = 0; i < 16; i++) free(buffers[i]);
    free(output);
    me_artifact_free(artifact);
    return failed;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s vectors.json [off|on]\n", argv[0]);
        return 2;
    }
    me_jit_mode mode = ME_JIT_OFF;
    if (argc == 3) {
        if (!strcmp(argv[2], "on")) mode = ME_JIT_ON;
        else if (strcmp(argv[2], "off")) return 2;
    }
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 2;
    char *data = malloc(1024 * 1024 + 1);
    if (!data) {
        fclose(file);
        return 2;
    }
    size_t size = fread(data, 1, 1024 * 1024 + 1, file);
    int failed = ferror(file) || size > 1024 * 1024;
    fclose(file);
    yyjson_doc *doc = failed ? NULL : yyjson_read(data, size, 0);
    free(data);
    if (!doc) return 2;
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *version = text(root, "schema_version");
    yyjson_val *cases = yyjson_obj_get(root, "cases");
    if (!version || strcmp(version, "menudet-numpy-vectors-1") ||
        !yyjson_is_arr(cases) || !yyjson_arr_size(cases)) failed = 1;
    else {
        for (size_t i = 0; i < yyjson_arr_size(cases); i++) {
            failed |= run_case(yyjson_arr_get(cases, i), mode);
        }
    }
    yyjson_doc_free(doc);
    return failed ? 1 : 0;
}
