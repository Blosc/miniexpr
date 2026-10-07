/* Artifact structure, typed decoding, ownership, bindings, and runtime failures. */
#undef NDEBUG
#include "miniexpr_artifact.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void schema1_fixture(void);
static void schema1_strings_nd_fixture(void);

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
    const char *format = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\","
        "\"version\":\"1.0\"},\"requires\":[\"numeric\"],\"source\":\"def k(c):\\n    return c\\n\","
        "\"entry_point\":\"k\",\"inputs\":[],\"constants\":[{\"name\":\"c\",\"dtype\":\"%s\","
        "\"encoding\":\"%s\",\"value\":%s}],\"output\":{\"dtype\":\"%s\","
        "\"contract\":\"elementwise\"},\"semantics\":{\"fp\":\"strict\"},\"context\":{\"ndim\":0}}";
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
    const char *json = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\","
        "\"version\":\"1.0\"},\"requires\":[\"numeric\",\"control-flow\"],"
        "\"source\":\"# me:fp=strict\\ndef k(x, y):\\n    return x - y\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"float64\"},"
        "{\"name\":\"y\",\"dtype\":\"float64\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"elementwise\"},"
        "\"semantics\":{\"fp\":\"strict\"},\"context\":{\"ndim\":0}}";
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
    schema1_fixture();
    schema1_strings_nd_fixture();
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
    assert(me_artifact_result_cardinality(artifact) == ME_ARTIFACT_ELEMENTWISE);
    assert(me_artifact_result_cardinality(NULL) == ME_ARTIFACT_CARDINALITY_INVALID);
    assert(me_artifact_input_itemsize(artifact, 0) == sizeof(double));
    assert(me_artifact_input_itemsize(artifact, 1) == 0 && me_artifact_output_itemsize(NULL) == 0);
    assert(me_artifact_output_itemsize(artifact) == sizeof(double));
    me_artifact_buffer extended_input = {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)};
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems = 600, .output_capacity = sizeof(output)
    };
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, output, &descriptor, &error) == ME_ARTIFACT_SUCCESS);
    output[0] = 777;
    descriptor.output_capacity--;
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, output, &descriptor, &error) == ME_ARTIFACT_ERR_BINDING && output[0] == 777);
    descriptor.output_capacity++;
    extended_input.capacity--;
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, output, &descriptor, NULL) == ME_ARTIFACT_ERR_BINDING);
    extended_input.capacity++;
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, x, &descriptor, NULL) == ME_ARTIFACT_ERR_BINDING && x[0] == 0);
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, (char *)output + 1, &descriptor, NULL) == ME_ARTIFACT_ERR_BINDING);
    descriptor.version++;
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, output, &descriptor, NULL) == ME_ARTIFACT_ERR_UNSUPPORTED);
    descriptor.version--;
    descriptor.ndim = 1;
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, output, &descriptor, NULL) == ME_ARTIFACT_ERR_BINDING);
    descriptor.ndim = 0;
    descriptor.nitems = SIZE_MAX;
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, output, &descriptor, NULL) == ME_ARTIFACT_ERR_BINDING);
    descriptor.nitems = 0;
    descriptor.output_capacity = 0;
    extended_input.data = NULL;
    extended_input.capacity = 0;
    assert(me_artifact_eval_ex(artifact, &extended_input, 1, NULL, &descriptor, NULL) == ME_ARTIFACT_SUCCESS);
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

    reject(json, "\"schema_version\": \"1.0\"", "\"schema_version\": \"99\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"schema_version\": \"1.0\"", "\"schema_version\": \"0.1\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"version\": \"1.0\"", "\"version\": \"0.1\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"schema_version\": \"1.0\"", "\"schema_version\": 1", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"dtype\": \"float64\"", "\"dtype\": null", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"numeric\"", "null", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"name\": \"miniexpr\"", "\"name\": \"python\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"version\": \"1.0\"", "\"version\": \"99\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"numeric\"", "\"future-feature\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"numeric\"", "\"numeric\", \"numeric\"", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"numeric\", ", "", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"fp\": \"strict\"", "\"fp\": \"fast\"", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"fp\": \"strict\"", "\"fp\": \"strict\", \"future\": true", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "elementwise", "reduction", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"entry_point\": \"affine\"", "\"entry_point\": \"other\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "\"name\": \"bias\"", "\"name\": \"x\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "\"name\": \"bias\"", "\"name\": \"unused\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "\"name\": \"scale\"", "\"name\": \"bias\"", ME_ARTIFACT_ERR_BINDING);
    reject(json, "return x * scale + bias", "return sum(x)", ME_ARTIFACT_ERR_BINDING);
    reject(json, "return x * scale + bias", "return missing", ME_ARTIFACT_ERR_SOURCE);
    reject(json, "# me:compiler=tcc", "# me:fp=fast\\n# me:compiler=tcc", ME_ARTIFACT_ERR_SOURCE);
    reject(json, "\"schema_version\":", "\"unknown\": 1, \"schema_version\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"schema_version\":", "\"schema_version\": \"0.1\", \"schema_version\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"schema_version\":", "\"schema_versi\\u006fn\": \"0.1\", \"schema_version\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"fp\":", "\"fp\": \"strict\", \"fp\":", ME_ARTIFACT_ERR_FORMAT);
    reject(json, "\"description\":", "\"description\": null, \"description\":", ME_ARTIFACT_ERR_FORMAT);
    char *mixed = replace(json, "\"dtype\": \"float64\"", "\"dtype\": \"int64\"");
    me_artifact *mixed_artifact = load(mixed);
    int64_t mixed_value = 3;
    double mixed_output = 0;
    me_artifact_input mixed_input = {"x", ME_INT64, &mixed_value, 1};
    assert(me_artifact_eval(mixed_artifact, &mixed_input, 1, &mixed_output, 1, NULL) == ME_ARTIFACT_SUCCESS);
    assert(mixed_output == 5);
    me_artifact_free(mixed_artifact);
    free(mixed);
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

static void schema1_fixture(void) {
    const char *json = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"block-reductions\"],\"source\":\"def k(x,c):\\n    a = sum(x)\\n    return a + c\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"int64\"}],"
        "\"constants\":[{\"name\":\"c\",\"dtype\":\"int64\",\"encoding\":\"decimal\",\"value\":\"1\"}],"
        "\"output\":{\"dtype\":\"int64\",\"contract\":\"block_scalar\"},\"semantics\":{\"fp\":\"strict\"},"
        "\"context\":{\"ndim\":0},\"metadata\":{\"status\":\"implementation-draft\"}}";
    me_artifact *artifact = load(json);
    assert(!me_artifact_has_jit(artifact) && me_artifact_result_cardinality(artifact) == ME_ARTIFACT_BLOCK_SCALAR);
    assert(!strcmp(me_artifact_schema_version(artifact), "1.0"));
    assert(me_artifact_context_ndim(artifact) == 0);
    assert(me_artifact_capabilities(artifact) == (ME_ARTIFACT_CAP_NUMERIC | ME_ARTIFACT_CAP_BLOCK_REDUCTIONS));
    int64_t values[] = {INT64_MAX, 2, 3};
    struct { int64_t value; int64_t sentinel; } output = {0, 1234};
    me_artifact_buffer input = {"x", ME_INT64, 8, values, sizeof(values)};
    uint8_t mask[] = {0, 1, 1};
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor), .version = 1,
        .nitems = 3, .output_capacity = 8, .valid_mask = mask, .valid_mask_capacity = sizeof(mask)};
    assert(me_artifact_eval_ex(artifact, &input, 1, &output.value, &descriptor, NULL) == ME_ARTIFACT_SUCCESS);
    assert(output.value == 6 && output.sentinel == 1234);
    mask[0] = 1;
    assert(me_artifact_eval_ex(artifact, &input, 1, &output.value, &descriptor, NULL) == ME_ARTIFACT_ERR_EVAL);
    mask[0] = 0;
    descriptor.nitems = 0;
    descriptor.valid_mask = NULL;
    descriptor.valid_mask_capacity = 0;
    input.data = NULL;
    input.capacity = 0;
    assert(me_artifact_eval_ex(artifact, &input, 1, &output.value, &descriptor, NULL) == ME_ARTIFACT_SUCCESS && output.value == 1);
    assert(me_artifact_eval(artifact, NULL, 0, &output.value, 0, NULL) != ME_ARTIFACT_SUCCESS);
    me_artifact_free(artifact);
    reject(json, "block_scalar", "elementwise", ME_ARTIFACT_ERR_BINDING);
    reject(json, ",\"block-reductions\"", "", ME_ARTIFACT_ERR_BINDING);
    reject(json, "numeric", "unimplemented", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "numeric", "jit-required", ME_ARTIFACT_ERR_UNSUPPORTED);
    reject(json, "\"ndim\":0", "\"ndim\":1", ME_ARTIFACT_ERR_BINDING);
    char *unsigned_json = replace(json, "a = sum(x)\\n    return a + c", "return x + c");
    char *unsigned_types = replace(unsigned_json, "\"name\":\"x\",\"dtype\":\"int64\"", "\"name\":\"x\",\"dtype\":\"uint64\"");
    free(unsigned_json);
    unsigned_json = replace(unsigned_types, "\"name\":\"c\",\"dtype\":\"int64\"", "\"name\":\"c\",\"dtype\":\"uint64\"");
    free(unsigned_types);
    unsigned_types = replace(unsigned_json, "\"value\":\"1\"", "\"value\":\"18446744073709551615\"");
    free(unsigned_json);
    unsigned_json = replace(unsigned_types, "\"dtype\":\"int64\",\"contract\":\"block_scalar\"", "\"dtype\":\"uint64\",\"contract\":\"elementwise\"");
    free(unsigned_types);
    artifact = load(unsigned_json);
    uint64_t zero = 0, exact = 0;
    input = (me_artifact_buffer){"x", ME_UINT64, 8, &zero, 8};
    descriptor.nitems = 1;
    assert(me_artifact_eval_ex(artifact, &input, 1, &exact, &descriptor, NULL) == ME_ARTIFACT_SUCCESS && exact == UINT64_MAX);
    reject(unsigned_json, "18446744073709551615", "18446744073709551616", ME_ARTIFACT_ERR_FORMAT);
    me_artifact_free(artifact);
    free(unsigned_json);
}

static void schema1_strings_nd_fixture(void) {
    const char *format = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"fixed-strings\"],\"source\":\"def k(x):\\n    return %s\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"bytes\",\"itemsize\":4}],\"constants\":[],"
        "\"output\":{\"dtype\":\"bytes\",\"itemsize\":%zu,\"contract\":\"elementwise\"},"
        "\"semantics\":{\"fp\":\"strict\"},\"context\":{\"ndim\":0}}";
    struct { const char *expr; size_t width; const char *expected; } cases[] = {
        {"upper(x)", 4, " AB "}, {"lower(x)", 4, " ab "}, {"strip(x)", 4, "Ab"},
        {"lstrip(x)", 4, "Ab "}, {"rstrip(x)", 4, " Ab"},
        {"upper(strip(x))", 4, "AB"}, {"substr(x, 1, 2)", 2, "Ab"},
        {"replace(x, 'A', 'Q')", 4, " Qb "}, {"removeprefix(x, ' ')", 4, "Ab "},
        {"removesuffix(x, ' ')", 4, " Ab"}, {"split_part(x, 'b', 0)", 4, " A"},
        {"x + '!'", 5, " Ab !"}, {"where(x != '', upper(x), lower(x))", 4, " AB "},
        {"where(x != '', upper(x), '')", 4, " AB "}
    };
    char json[4096], input_bytes[8] = {' ', 'A', 'b', ' ', 'x', 0, 'z', 'z'};
    me_artifact_buffer input = {"x", ME_BYTES, 4, input_bytes, sizeof(input_bytes)};
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor), .version = 1, .nitems = 2};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        snprintf(json, sizeof(json), format, cases[i].expr, cases[i].width);
        me_artifact *artifact = load(json);
        char output[16];
        memset(output, 0x7f, sizeof(output));
        descriptor.output_capacity = cases[i].width * 2;
        me_artifact_error detail;
        me_artifact_status status = me_artifact_eval_ex(artifact, &input, 1, output, &descriptor, &detail);
        if (status) fprintf(stderr, "string case %s: %d native=%d %s\n", cases[i].expr, status, detail.native_status, detail.message);
        assert(status == ME_ARTIFACT_SUCCESS);
        assert(!memcmp(output, cases[i].expected, strlen(cases[i].expected)));
        for (size_t p = strlen(cases[i].expected); p < cases[i].width; p++) assert(output[p] == 0);
        assert(output[descriptor.output_capacity] == 0x7f);
        me_artifact_free(artifact);
    }
    const char *nd_json = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"nd-context\"],\"source\":\"def k():\\n    return _flat_idx + _i0 + _i1 + _n0 + _n1 + _ndim\\n\","
        "\"entry_point\":\"k\",\"inputs\":[],\"constants\":[],\"output\":{\"dtype\":\"int64\",\"contract\":\"elementwise\"},"
        "\"semantics\":{\"fp\":\"strict\"},\"context\":{\"ndim\":2}}";
    me_artifact *artifact = load(nd_json);
    int64_t shape[] = {4, 5}, origin[] = {2, 3}, extent[] = {2, 3}, output[6] = {-1,-1,-1,-1,-1,-1};
    uint8_t mask[] = {1, 1, 0, 1, 1, 0};
    descriptor = (me_artifact_eval_descriptor){.struct_size = sizeof(descriptor), .version = 1,
        .nitems = 6, .output_capacity = sizeof(output), .valid_mask = mask, .valid_mask_capacity = 6,
        .ndim = 2, .logical_shape = shape, .block_origin = origin, .block_extent = extent};
    assert(me_artifact_eval_ex(artifact, NULL, 0, output, &descriptor, NULL) == ME_ARTIFACT_SUCCESS);
    assert(output[0] == 29 && output[1] == 31 && output[2] == -1 && output[3] == 35 && output[4] == 37 && output[5] == -1);
    mask[2] = 1;
    assert(me_artifact_eval_ex(artifact, NULL, 0, output, &descriptor, NULL) != ME_ARTIFACT_SUCCESS);
    mask[2] = 0;
    descriptor.ndim = 0;
    assert(me_artifact_eval_ex(artifact, NULL, 0, output, &descriptor, NULL) == ME_ARTIFACT_ERR_BINDING);
    reject(nd_json, "_i1", "_i2", ME_ARTIFACT_ERR_SOURCE);
    me_artifact_free(artifact);
    const char *unicode_json = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"fixed-strings\"],\"source\":\"def k(x,c):\\n    return upper(x) + c\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"unicode32\",\"itemsize\":16}],"
        "\"constants\":[{\"name\":\"c\",\"dtype\":\"unicode32\",\"itemsize\":8,\"encoding\":\"unicode32be-hex\",\"value\":\"0000002100000000\"}],"
        "\"output\":{\"dtype\":\"unicode32\",\"itemsize\":24,\"contract\":\"elementwise\"},"
        "\"semantics\":{\"fp\":\"strict\"},\"context\":{\"ndim\":0}}";
    artifact = load(unicode_json);
    uint32_t unicode_input[] = {0xdf, ' ', 'a', 0}, unicode_output[6];
    input = (me_artifact_buffer){"x", ME_STRING, sizeof(unicode_input), unicode_input, sizeof(unicode_input)};
    descriptor = (me_artifact_eval_descriptor){.struct_size = sizeof(descriptor), .version = 1, .nitems = 1, .output_capacity = sizeof(unicode_output)};
    assert(me_artifact_input_itemsize(artifact, 0) == 16 && me_artifact_output_itemsize(artifact) == 24);
    assert(me_artifact_eval_ex(artifact, &input, 1, unicode_output, &descriptor, NULL) == ME_ARTIFACT_SUCCESS);
    uint32_t expected_unicode[] = {'S','S',' ','A','!',0};
    assert(!memcmp(unicode_output, expected_unicode, sizeof(unicode_output)));
    unicode_input[0] = 0xd800;
    assert(me_artifact_eval_ex(artifact, &input, 1, unicode_output, &descriptor, NULL) == ME_ARTIFACT_ERR_BINDING);
    uint8_t invalid_lane = 0;
    descriptor.valid_mask = &invalid_lane;
    descriptor.valid_mask_capacity = 1;
    assert(me_artifact_eval_ex(artifact, &input, 1, unicode_output, &descriptor, NULL) == ME_ARTIFACT_SUCCESS);
    reject(unicode_json, "0000002100000000", "0000d80000000000", ME_ARTIFACT_ERR_FORMAT);
    reject(unicode_json, "\"itemsize\":24", "\"itemsize\":20", ME_ARTIFACT_ERR_BINDING);
    reject(unicode_json, ",\"fixed-strings\"", "", ME_ARTIFACT_ERR_BINDING);
    me_artifact_free(artifact);
    const char *indexed_json = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"fixed-strings\"],\"source\":\"def k(x,i,n,old,new):\\n    return replace(substr(x,i,n),old,new)\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"bytes\",\"itemsize\":4},"
        "{\"name\":\"i\",\"dtype\":\"int64\"},{\"name\":\"n\",\"dtype\":\"uint64\"}],"
        "\"constants\":[{\"name\":\"old\",\"dtype\":\"bytes\",\"itemsize\":1,\"encoding\":\"bytes-hex\",\"value\":\"41\"},"
        "{\"name\":\"new\",\"dtype\":\"bytes\",\"itemsize\":2,\"encoding\":\"bytes-hex\",\"value\":\"5858\"}],"
        "\"output\":{\"dtype\":\"bytes\",\"itemsize\":8,\"contract\":\"elementwise\"},"
        "\"semantics\":{\"fp\":\"strict\"},\"context\":{\"ndim\":0}}";
    artifact = load(indexed_json);
    char subjects[] = {'A','b','c','d','A','x','y','z'}, indexed_output[16];
    int64_t starts[] = {-2, 0};
    uint64_t lengths[] = {UINT64_MAX, 2};
    me_artifact_buffer indexed_inputs[] = {{"x", ME_BYTES, 4, subjects, sizeof(subjects)},
        {"i", ME_INT64, 8, starts, sizeof(starts)}, {"n", ME_UINT64, 8, lengths, sizeof(lengths)}};
    descriptor = (me_artifact_eval_descriptor){.struct_size = sizeof(descriptor), .version = 1,
        .nitems = 2, .output_capacity = sizeof(indexed_output)};
    assert(me_artifact_eval_ex(artifact, indexed_inputs, 3, indexed_output, &descriptor, NULL) == ME_ARTIFACT_SUCCESS);
    assert(!memcmp(indexed_output, "cd\0\0\0\0\0\0XXx\0\0\0\0\0", 16));
    me_artifact_free(artifact);
    char *empty_needle = replace(indexed_json, "\"value\":\"41\"", "\"value\":\"00\"");
    artifact = load(empty_needle);
    assert(me_artifact_eval_ex(artifact, indexed_inputs, 3, indexed_output, &descriptor, NULL) == ME_ARTIFACT_ERR_EVAL);
    me_artifact_free(artifact);
    free(empty_needle);
}
