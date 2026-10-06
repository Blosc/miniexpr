/* Artifact structure, typed decoding, ownership, bindings, and runtime failures. */
#undef NDEBUG
#include "miniexpr_artifact.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_fixture(void) {
    FILE *file = fopen(MINIEXPR_ARTIFACT_FIXTURE, "rb");
    assert(file);
    char *json = calloc(ME_ARTIFACT_MAX_BYTES + 1, 1);
    assert(json);
    size_t size = fread(json, 1, ME_ARTIFACT_MAX_BYTES, file);
    assert(size && !ferror(file) && fgetc(file) == EOF);
    fclose(file);
    return json;
}

static char *replace(const char *json, const char *old, const char *replacement) {
    const char *position = strstr(json, old);
    assert(position);
    size_t prefix = (size_t)(position - json);
    char *result = malloc(strlen(json) + strlen(replacement) + 1);
    assert(result);
    memcpy(result, json, prefix);
    strcpy(result + prefix, replacement);
    strcpy(result + prefix + strlen(replacement), position + strlen(old));
    return result;
}

static void expect(const char *json, me_artifact_status status) {
    me_artifact *artifact = (me_artifact *)(uintptr_t)1;
    me_artifact_error error;
    int rc = me_artifact_load(json, strlen(json), ME_JIT_OFF, &artifact, &error);
    if (rc != status) fprintf(stderr, "expected %d, got %d: %s\n%s\n", status, rc, error.message, json);
    assert(rc == status);
    assert(artifact == NULL && error.message[0]);
    assert(me_artifact_load(json, strlen(json), ME_JIT_OFF, &artifact, NULL) == status);
    assert(artifact == NULL);
}

static me_artifact *load(const char *json) {
    me_artifact *artifact = NULL;
    me_artifact_error error;
    int rc = me_artifact_load(json, strlen(json), ME_JIT_OFF, &artifact, &error);
    if (rc) fprintf(stderr, "load failed %d: %s\n", rc, error.message);
    assert(rc == ME_ARTIFACT_SUCCESS && artifact);
    assert(!error.native_status && !error.line && !error.column && !error.message[0]);
    return artifact;
}

static void reject(const char *json, const char *old, const char *replacement, me_artifact_status status) {
    char *changed = replace(json, old, replacement);
    expect(changed, status);
    free(changed);
}

static char *scalar_manifest(const char *dtype, const char *encoding, const char *value) {
    const char *format = "{\"schema_version\":\"0.1\",\"language\":{\"name\":\"miniexpr\","
        "\"version\":\"0.1\"},\"requires\":[\"core-scalar\"],\"source\":\"def k(c):\\n    return c\\n\","
        "\"entry_point\":\"k\",\"inputs\":[],\"constants\":[{\"name\":\"c\",\"dtype\":\"%s\","
        "\"encoding\":\"%s\",\"value\":%s}],\"output\":{\"dtype\":\"%s\","
        "\"contract\":\"scalar-per-element\"},\"semantics\":{\"fp\":\"strict\"}}";
    char *json = malloc(2048);
    assert(json);
    assert(snprintf(json, 2048, format, dtype, encoding, value, dtype) < 2048);
    return json;
}

static void scalars(void) {
    const char *types[] = {"bool", "int32", "int64", "float32", "float64"};
    const char *encodings[] = {"boolean", "decimal", "decimal", "ieee754-hex", "ieee754-hex"};
    const char *values[] = {"true", "\"-2147483648\"", "\"-9223372036854775808\"",
                            "\"80000000\"", "\"8000000000000000\""};
    for (int t = 0; t < 5; t++) {
        char *json = scalar_manifest(types[t], encodings[t], values[t]);
        me_artifact *artifact = load(json);
        free(json);
        union { bool b[600]; int32_t i32[600]; int64_t i64[600]; float f32[600]; double f64[600]; } output;
        me_artifact_error error;
        assert(me_artifact_eval(artifact, NULL, 0, &output, 600, &error) == ME_ARTIFACT_SUCCESS);
        for (int i = 0; i < 600; i++) {
            if (t == 0) assert(output.b[i]);
            if (t == 1) assert(output.i32[i] == INT32_MIN);
            if (t == 2) assert(output.i64[i] == INT64_MIN);
            if (t == 3) assert(output.f32[i] == 0 && signbit(output.f32[i]));
            if (t == 4) assert(output.f64[i] == 0 && signbit(output.f64[i]));
        }
        assert(me_artifact_eval(artifact, NULL, 0, NULL, 0, NULL) == ME_ARTIFACT_SUCCESS);
        me_artifact_free(artifact);
    }
    const char *bits[] = {"7ff0000000000000", "fff0000000000000", "7ff8000000000042"};
    for (int i = 0; i < 3; i++) {
        char value[32];
        snprintf(value, sizeof(value), "\"%s\"", bits[i]);
        char *json = scalar_manifest("float64", "ieee754-hex", value);
        me_artifact *artifact = load(json);
        double output;
        assert(me_artifact_eval(artifact, NULL, 0, &output, 1, NULL) == ME_ARTIFACT_SUCCESS);
        if (i < 2) assert(isinf(output) && (signbit(output) != 0) == (i == 1));
        else assert(isnan(output));
        me_artifact_free(artifact);
        free(json);
    }
    const char *invalid_ints[] = {"2147483648", "-2147483649", "9223372036854775808",
        "-9223372036854775809", "+1", "01", "-0", "", "1e0", "1.0", " 1", "99999999999999999999999999999"};
    for (size_t i = 0; i < sizeof(invalid_ints) / sizeof(invalid_ints[0]); i++) {
        char value[128];
        snprintf(value, sizeof(value), "\"%s\"", invalid_ints[i]);
        char *json = scalar_manifest(i < 2 ? "int32" : "int64", "decimal", value);
        expect(json, ME_ARTIFACT_ERR_FORMAT);
        free(json);
    }
    char *json = scalar_manifest("int64", "decimal", "42");
    expect(json, ME_ARTIFACT_ERR_FORMAT);
    free(json);
    json = scalar_manifest("bool", "boolean", "\"true\"");
    expect(json, ME_ARTIFACT_ERR_FORMAT);
    free(json);
    json = scalar_manifest("float32", "ieee754-hex", "\"0x80000000\"");
    expect(json, ME_ARTIFACT_ERR_FORMAT);
    free(json);
    json = scalar_manifest("float64", "ieee754-hex", "\"7FF0000000000000\"");
    expect(json, ME_ARTIFACT_ERR_FORMAT);
    free(json);
    json = scalar_manifest("int64", "decimal", "\"9223372036854775807\"");
    me_artifact *artifact = load(json);
    int64_t integer;
    assert(me_artifact_eval(artifact, NULL, 0, &integer, 1, NULL) == ME_ARTIFACT_SUCCESS);
    assert(integer == INT64_MAX);
    me_artifact_free(artifact);
    free(json);
    json = scalar_manifest("float32", "ieee754-hex", "\"3f800000\"");
    artifact = load(json);
    float number;
    assert(me_artifact_eval(artifact, NULL, 0, &number, 1, NULL) == ME_ARTIFACT_SUCCESS);
    assert(number == 1.0f);
    me_artifact_free(artifact);
    free(json);
}

static void runtime_order(void) {
    const char *json = "{\"schema_version\":\"0.1\",\"language\":{\"name\":\"miniexpr\","
        "\"version\":\"0.1\"},\"requires\":[\"core-scalar\"],"
        "\"source\":\"# me:fp=strict\\ndef k(x, y):\\n    return x - y\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"float64\"},"
        "{\"name\":\"y\",\"dtype\":\"float64\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"scalar-per-element\"},"
        "\"semantics\":{\"fp\":\"strict\"}}";
    me_artifact *artifact = load(json);
    double x = 3, y = 1, output = 0;
    me_artifact_input inputs[] = {{"y", ME_FLOAT64, &y, 1}, {"x", ME_FLOAT64, &x, 1}};
    assert(me_artifact_eval(artifact, inputs, 2, &output, 1, NULL) == ME_ARTIFACT_SUCCESS);
    assert(output == 2);
    inputs[0].name = "x";
    assert(me_artifact_eval(artifact, inputs, 2, &output, 1, NULL) == ME_ARTIFACT_ERR_BINDING);
    inputs[0].name = "z";
    assert(me_artifact_eval(artifact, inputs, 2, &output, 1, NULL) == ME_ARTIFACT_ERR_BINDING);
    inputs[0].name = NULL;
    assert(me_artifact_eval(artifact, inputs, 2, &output, 1, NULL) == ME_ARTIFACT_ERR_BINDING);
    assert(me_artifact_input_name(artifact, -1) == NULL);
    assert(me_artifact_input_name(artifact, 2) == NULL);
    assert(me_artifact_input_dtype(artifact, 2) == ME_AUTO);
    me_artifact_free(artifact);
    char *changed = replace(json, "{\"name\":\"x\",\"dtype\":\"float64\"},"
        "{\"name\":\"y\",\"dtype\":\"float64\"}", "{\"name\":\"y\",\"dtype\":\"float64\"},"
        "{\"name\":\"x\",\"dtype\":\"float64\"}");
    expect(changed, ME_ARTIFACT_ERR_BINDING);
    free(changed);
    changed = replace(json, "return x - y", "while 1:\\n        pass\\n    return x");
    artifact = load(changed); /* Would reach the loop cap if loading executed it. */
    me_artifact_free(artifact);
    free(changed);
    char *empty = replace(json, "# me:fp=strict\\ndef k(x, y):\\n    return x - y\\n",
                          "def k():\\n    return 1.0\\n");
    changed = replace(empty, "{\"name\":\"x\",\"dtype\":\"float64\"},"
        "{\"name\":\"y\",\"dtype\":\"float64\"}", "");
    free(empty);
    artifact = load(changed);
    double constant_output[4];
    assert(me_artifact_eval(artifact, NULL, 0, constant_output, 4, NULL) == ME_ARTIFACT_SUCCESS);
    for (int i = 0; i < 4; i++) assert(constant_output[i] == 1);
    me_artifact_free(artifact);
    free(changed);
}

int main(void) {
    char *json = read_fixture();
    me_artifact *artifact = load(json);
    assert(me_artifact_ninputs(artifact) == 1);
    assert(!strcmp(me_artifact_entry_point(artifact), "affine"));
    assert(!strcmp(me_artifact_input_name(artifact, 0), "x"));
    assert(me_artifact_input_dtype(artifact, 0) == ME_FLOAT64);
    assert(me_artifact_output_dtype(artifact) == ME_FLOAT64);
    assert(strstr(me_artifact_source(artifact), "# me:compiler=tcc"));
    assert(!strstr(me_artifact_source(artifact), "me:fp"));
    assert(!me_artifact_has_jit(artifact));
    double x[600], output[600];
    for (int i = 0; i < 600; i++) x[i] = i;
    me_artifact_input inputs[] = {{"x", ME_FLOAT64, x, 600}};
    me_artifact_error error;
    assert(me_artifact_eval(artifact, inputs, 1, output, 600, &error) == ME_ARTIFACT_SUCCESS);
    assert(!error.message[0]);
    for (int i = 0; i < 600; i++) assert(output[i] == 2 * x[i] - 1);
    assert(me_artifact_eval(artifact, inputs, 1, output, 599, NULL) == ME_ARTIFACT_ERR_BINDING);
    inputs[0].dtype = ME_FLOAT32;
    assert(me_artifact_eval(artifact, inputs, 1, output, 600, NULL) == ME_ARTIFACT_ERR_BINDING);
    inputs[0].dtype = ME_FLOAT64;
    inputs[0].name = "unknown";
    assert(me_artifact_eval(artifact, inputs, 1, output, 600, NULL) == ME_ARTIFACT_ERR_BINDING);
    inputs[0].name = "x";
    assert(me_artifact_eval(artifact, inputs, 0, output, 600, NULL) == ME_ARTIFACT_ERR_BINDING);
    assert(me_artifact_eval(artifact, inputs, 1, NULL, 600, NULL) == ME_ARTIFACT_ERR_BINDING);
    inputs[0].nitems = 0;
    inputs[0].data = NULL;
    assert(me_artifact_eval(artifact, inputs, 1, NULL, 0, NULL) == ME_ARTIFACT_SUCCESS);
    me_artifact_free(artifact);

    reject(json, "\"schema_version\": \"0.1\"", "\"schema_version\": \"99\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"schema_version\": \"0.1\"", "\"schema_version\": 1", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"dtype\": \"float64\"", "\"dtype\": null", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"core-scalar\"", "null", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"name\": \"miniexpr\"", "\"name\": \"python\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"version\": \"0.1\"", "\"version\": \"99\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"core-scalar\"", "\"future-feature\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"core-scalar\"", "\"core-scalar\", \"core-scalar\"", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"core-scalar\"", "", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"fp\": \"strict\"", "\"fp\": \"fast\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"fp\": \"strict\"", "\"fp\": \"strict\", \"future\": true", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "scalar-per-element", "reduction", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"entry_point\": \"affine\"", "\"entry_point\": \"other\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "\"name\": \"bias\"", "\"name\": \"x\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "\"name\": \"bias\"", "\"name\": \"unused\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "\"name\": \"scale\"", "\"name\": \"bias\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "return x * scale + bias", "return sum(x)", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "return x * scale + bias", "return missing", ME_ARTIFACT_ERR_SOURCE);
    reject(json, "# me:compiler=tcc", "# me:fp=fast\\n# me:compiler=tcc", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"schema_version\":", "\"unknown\": 1, \"schema_version\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"schema_version\":", "\"schema_version\": \"0.1\", \"schema_version\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"schema_version\":", "\"schema_versi\\u006fn\": \"0.1\", \"schema_version\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"fp\":", "\"fp\": \"strict\", \"fp\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"description\":", "\"description\": null, \"description\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"dtype\": \"float64\"", "\"dtype\": \"int64\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"dtype\": \"float64\"", "\"dtype\": \"complex128\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    char *changed = replace(json, "2*x - 1", "snowman \\u2603");
    artifact = load(changed);
    me_artifact_free(artifact);
    free(changed);
    char nested[256];
    memset(nested, '[', 40);
    nested[40] = '0';
    memset(nested + 41, ']', 40);
    nested[81] = '\0';
    changed = replace(json, "\"Hand-authored affine kernel: 2*x - 1\"", nested);
    expect(changed, ME_ARTIFACT_ERR_FORMAT);
    free(changed);
    artifact = (me_artifact *)(uintptr_t)1;
    assert(me_artifact_load(NULL, 1, ME_JIT_OFF, &artifact, NULL) == ME_ARTIFACT_ERR_FORMAT);
    assert(artifact == NULL);
    assert(me_artifact_load(json, strlen(json), ME_JIT_OFF, NULL, NULL) == ME_ARTIFACT_ERR_FORMAT);
    assert(me_artifact_load(json, 0, ME_JIT_OFF, &artifact, NULL) == ME_ARTIFACT_ERR_FORMAT);
    assert(me_artifact_load(json, ME_ARTIFACT_MAX_BYTES + 1, ME_JIT_OFF, &artifact, NULL) == ME_ARTIFACT_ERR_FORMAT);
    assert(me_artifact_load(json, strlen(json), (me_jit_mode)99, &artifact, NULL) == ME_ARTIFACT_ERR_BINDING);
    assert(me_artifact_load(json, strlen(json) + 1, ME_JIT_OFF, &artifact, NULL) == ME_ARTIFACT_ERR_FORMAT);
    reject(json, "2*x - 1", "bad \\u0000", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "2*x - 1", "bad \xff", ME_ARTIFACT_ERR_FORMAT);
    expect("{}", ME_ARTIFACT_ERR_FORMAT);
    expect("[]", ME_ARTIFACT_ERR_FORMAT);
    expect("null", ME_ARTIFACT_ERR_FORMAT);
    changed = replace(json, "{", "{/* comment */");
    expect(changed, ME_ARTIFACT_ERR_FORMAT);
    free(changed);
    changed = malloc(strlen(json) + 8);
    assert(changed);
    snprintf(changed, strlen(json) + 8, "%s {}", json);
    expect(changed, ME_ARTIFACT_ERR_FORMAT);
    free(changed);
    changed = replace(json, "return x * scale + bias", "if x > 0:\\n        return x");
    artifact = load(changed);
    free(changed);
    double negative = -1;
    me_artifact_input missing_return[] = {{"x", ME_FLOAT64, &negative, 1}};
    assert(me_artifact_eval(artifact, missing_return, 1, output, 1, &error) == ME_ARTIFACT_ERR_EVAL);
    assert(error.native_status == ME_EVAL_ERR_INVALID_ARG && error.message[0]);
    missing_return[0].nitems = 0;
    missing_return[0].data = NULL;
    assert(me_artifact_eval(artifact, missing_return, 1, NULL, 0, NULL) == ME_ARTIFACT_SUCCESS);
    me_artifact_free(artifact);
    free(json);
    scalars();
    runtime_order();
    me_artifact_free(NULL);
    assert(me_artifact_source(NULL) == NULL && me_artifact_entry_point(NULL) == NULL);
    assert(me_artifact_ninputs(NULL) == 0 && !me_artifact_has_jit(NULL));
    assert(me_artifact_input_name(NULL, -1) == NULL && me_artifact_input_dtype(NULL, 0) == ME_AUTO);
    assert(me_artifact_output_dtype(NULL) == ME_AUTO);
    return 0;
}
