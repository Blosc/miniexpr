#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "miniexpr_artifact.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)

typedef struct {
    const char *name;
    const char *body;
    const char *input;
    const char *output;
    bool eligible;
} parity_case;

/* JSON-escaped bodies deliberately exercise the production artifact loader.
 * Eligibility expectations advance with each qualified implementation phase. */
static const parity_case cases[] = {
    {"arithmetic", "return x * 2 + y", "float64", "float64", true},
    {"conditional return", "if x > 0:\\n        return x\\n    return -x", "float64", "float64", true},
    {"range", "s = 0.0\\n    for i in range(10):\\n        s = s + x\\n    return s", "float64", "float64", true},
    {"while", "s = 0.0\\n    i = 0\\n    while i < 10:\\n        s = s + x\\n        i = i + 1\\n    return s", "float64", "float64", true},
    {"break", "s = 0.0\\n    for i in range(10):\\n        if s > y:\\n            break\\n        s = s + x\\n    return s", "float64", "float64", true},
    {"continue", "s = 0.0\\n    for i in range(10):\\n        if i < 3:\\n            continue\\n        s = s + x\\n    return s", "float64", "float64", true},
    {"loop return", "for i in range(10):\\n        if x > 0:\\n            return x\\n    return y", "float64", "float64", true},
    {"nested loops", "s = 0.0\\n    for i in range(3):\\n        for j in range(4):\\n            s = s + x\\n    return s", "float64", "float64", true},
    {"negative range", "s = 0.0\\n    for i in range(5, -2, -2):\\n        s = s + i + x\\n    return s", "float64", "float64", true},
    {"empty range", "s = x\\n    for i in range(0):\\n        s = y\\n    return s", "float64", "float64", true},
    {"nested exit targets", "s = 0.0\\n    for i in range(3):\\n        for j in range(4):\\n            if j == 1:\\n                continue\\n            if j == 3:\\n                break\\n            s = s + x\\n    return s", "float64", "float64", true},
    {"while continue", "s = 0.0\\n    i = 0\\n    while i < 5:\\n        i = i + 1\\n        if i < 3:\\n            continue\\n        s = s + x\\n    return s", "float64", "float64", true},
    {"while break", "s = 0.0\\n    i = 0\\n    while i < 5:\\n        if i == 3:\\n            break\\n        s = s + x\\n        i = i + 1\\n    return s", "float64", "float64", true},
    {"range advancement edge", "s = x\\n    for i in range(9223372036854775806, 9223372036854775807, 2):\\n        s = s + y\\n    return s", "float64", "float64", true},
    {"mandelbrot", "zr = 0.0\\n    zi = 0.0\\n    escape_iter = 64.0\\n    for i in range(64):\\n        if zr * zr + zi * zi > 4.0:\\n            escape_iter = i + 0.0\\n            break\\n        zr_new = zr * zr - zi * zi + x\\n        zi = 2.0 * zr * zi + y\\n        zr = zr_new\\n    return escape_iter", "float64", "float64", true},
    {"checked cast", "return int(x)", "float64", "int64", true},
    {"integer abs", "return abs(x)", "int32", "int32", true},
    {"integer sign", "return sign(x)", "int32", "int32", true},
    {"integer square", "return square(x)", "int32", "int32", true},
    {"integer floor", "return floor(x)", "int32", "int32", true},
    {"integer round", "return round(x)", "int32", "int32", true},
    {"integer ceil", "return ceil(x)", "int32", "int32", true},
    {"integer trunc", "return trunc(x)", "int32", "int32", true},
    {"integer real", "return real(x)", "int32", "int32", true},
    {"integer imag", "return imag(x)", "int32", "int32", true},
    {"integer conj", "return conj(x)", "int32", "int32", true},
    {"factorial", "return fac(x)", "int32", "int32", true},
    {"ncr", "return ncr(x, y)", "int32", "int32", true},
    {"npr", "return npr(x, y)", "int32", "int32", true},
    {"named power", "return pow(x, y)", "int32", "int32", true},
    {"ldexp", "return ldexp(x, 2)", "float64", "float64", true},
    {"fma", "return fma(x, y, 1.0)", "float64", "float64", true},
    {"floating math", "return sin(x) + cos(y)", "float64", "float64", true},
};

static me_artifact *load_extra(const parity_case *test, me_jit_mode mode,
    const char *second_input, const char *constants) {
    char json[4096];
    char inputs[256];
    snprintf(inputs, sizeof(inputs), "{\"name\":\"x\",\"dtype\":\"%s\"}%s%s",
        test->input, second_input ? "," : "", second_input ? second_input : "");
    int length = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\",\"control-flow\"],\"source\":\"def k(x, y):\\n    %s\\n\","
        "\"entry_point\":\"k\",\"inputs\":[%s],"
        "\"constants\":[%s],\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        test->body, inputs, constants ? constants : "", test->output);
    CHECK(length > 0 && (size_t)length < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error;
    int rc = me_artifact_load(json, (size_t)length, mode, &artifact, &error);
    if (rc) fprintf(stderr, "%s: %s\n", test->name, error.message);
    CHECK(!rc && artifact);
    return artifact;
}

static me_artifact *load(const parity_case *test, me_jit_mode mode) {
    char second[128];
    snprintf(second, sizeof(second), "{\"name\":\"y\",\"dtype\":\"%s\"}", test->input);
    return load_extra(test, mode, second, NULL);
}

static void matrix(void) {
    me_artifact *oracle = load(&cases[0], ME_JIT_ON);
    bool available = me_artifact_has_jit(oracle);
    me_artifact_free(oracle);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const parity_case *test = &cases[i];
        me_artifact *jit = load(test, ME_JIT_ON);
        me_artifact *reference = load(test, ME_JIT_OFF);
        CHECK(me_artifact_has_jit(jit) == (available && test->eligible));
        CHECK(!me_artifact_has_jit(reference));
        double x[] = {-2, 0, 3}, y[] = {4, 2, 1};
        int32_t ix[] = {-2, 0, 3}, iy[] = {4, 2, 1};
        if (!strcmp(test->name,"factorial") || !strcmp(test->name,"ncr") || !strcmp(test->name,"npr")) {
            ix[0] = 0; ix[1] = 3; ix[2] = 5;
            iy[0] = 0; iy[1] = 1; iy[2] = 2;
        }
        bool integer = !strcmp(test->input, "int32");
        me_artifact_buffer buffers[] = {
            {"x", integer ? ME_INT32 : ME_FLOAT64, integer ? 4 : 8,
                integer ? (void *)ix : (void *)x, integer ? sizeof(ix) : sizeof(x)},
            {"y", integer ? ME_INT32 : ME_FLOAT64, integer ? 4 : 8,
                integer ? (void *)iy : (void *)y, integer ? sizeof(iy) : sizeof(y)}
        };
        uint64_t result[3] = {0}, expected[3] = {0};
        me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
            .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 3,
            .output_capacity = sizeof(result)};
        me_artifact_fp_status status, reference_status;
        me_artifact_error error;
        me_artifact_error reference_error;
        int rc = me_artifact_eval_status(jit, buffers, 2, result, &descriptor, 0, &status, &error);
        int reference_rc = me_artifact_eval_status(reference, buffers, 2, expected, &descriptor, 0,
            &reference_status, &reference_error);
        CHECK(rc == reference_rc);
        if (!rc) {
            CHECK(!memcmp(result, expected, sizeof(result)));
            CHECK(status.flags == reference_status.flags);
        } else CHECK(error.native_status == reference_error.native_status);
        fprintf(stderr, "%-20s route=%s\n", test->name,
            me_artifact_has_jit(jit) ? "jit" : "interpreter");
        me_artifact_free(jit);
        me_artifact_free(reference);
    }
}

static void exact_comparisons(void) {
    const char *operators[] = {"==", "!=", "<", "<=", ">", ">="};
    int64_t x[] = {INT64_MIN, -1, 0, INT64_C(9007199254740992), INT64_MAX};
    int64_t y[] = {INT64_MIN, 0, -1, INT64_C(9007199254740993), INT64_MAX};
    uint64_t uy[] = {0, UINT64_MAX, 0, UINT64_C(9007199254740993), UINT64_MAX};
    for (int mixed = 0; mixed < 2; mixed++) for (int op = 0; op < 6; op++) {
        char body[128];
        snprintf(body, sizeof(body), "return x %s y", operators[op]);
        parity_case test = {"exact comparison", body, "int64", "bool", true};
        const char *second = mixed ? "{\"name\":\"y\",\"dtype\":\"uint64\"}" :
            "{\"name\":\"y\",\"dtype\":\"int64\"}";
        me_artifact *jit = load_extra(&test, ME_JIT_ON, second, NULL);
        me_artifact *reference = load_extra(&test, ME_JIT_OFF, second, NULL);
        parity_case oracle_case = {"oracle", "return x + x", "int64", "int64", true};
        me_artifact *oracle = load(&oracle_case, ME_JIT_ON);
        CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
        me_artifact_buffer buffers[] = {{"x", ME_INT64, 8, x, sizeof(x)},
            {"y", mixed ? ME_UINT64 : ME_INT64, 8, mixed ? (void *)uy : (void *)y, sizeof(y)}};
        bool output[5] = {0}, expected[5] = {0};
        me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
            .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 5,
            .output_capacity = sizeof(output)};
        me_artifact_error error;
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        CHECK(!me_artifact_eval_ex(reference, buffers, 2, expected, &descriptor, &error));
        CHECK(!memcmp(output, expected, sizeof(output)));
        if (!mixed && op == 2) CHECK(output[3]);
        me_artifact_free(jit);
        me_artifact_free(reference);
        me_artifact_free(oracle);
    }
}

static void weak_overflow(void) {
    const char *constant = "{\"name\":\"y\",\"dtype\":\"int64\",\"category\":\"weak\","
        "\"encoding\":\"decimal\",\"value\":\"9223372036854775807\"}";
    parity_case test = {"weak intermediate", "if x == 0:\\n        return x\\n    return x + (y + 1)",
        "int64", "int64", false};
    me_artifact *jit = load_extra(&test, ME_JIT_ON, NULL, constant);
    me_artifact *reference = load_extra(&test, ME_JIT_OFF, NULL, constant);
    /* This computed weak arithmetic now uses the checked invocation bridge. */
    me_artifact *oracle = load(&cases[0], ME_JIT_ON);
    CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
    me_artifact_free(oracle);
    int64_t x[] = {0, 0}, output[2];
    uint8_t mask[] = {1, 0};
    me_artifact_buffer buffer = {"x", ME_INT64, 8, x, sizeof(x)};
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2,
        .output_capacity = sizeof(output)};
    me_artifact_error error, expected_error;
    CHECK(!me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error));
    x[1] = 1;
    int rc = me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error);
    int expected_rc = me_artifact_eval_ex(reference, &buffer, 1, output, &descriptor, &expected_error);
    CHECK(rc && rc == expected_rc && error.native_status == expected_error.native_status);
    descriptor.valid_mask = mask;
    descriptor.valid_mask_capacity = sizeof(mask);
    CHECK(!me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error));
    x[1] = 0;
    descriptor.valid_mask = NULL;
    descriptor.valid_mask_capacity = 0;
    CHECK(!me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error));
    me_artifact_free(jit);
    me_artifact_free(reference);
}

static void environment(const char *name, const char *value) {
#ifdef _WIN32
    CHECK(!_putenv_s(name, value ? value : ""));
#else
    if (value) CHECK(!setenv(name, value, 1));
    else CHECK(!unsetenv(name));
#endif
}

static void loop_errors(void) {
    const char *old_cap = getenv("ME_DSL_WHILE_MAX_ITERS");
    char *saved_cap = old_cap ? strdup(old_cap) : NULL;
    environment("ME_DSL_WHILE_MAX_ITERS", "3");
    parity_case tests[] = {
        {"while cap", "i = 0\\n    while i < x:\\n        i = i + 1\\n    return i", "int64", "int64", true},
        {"zero range step", "s = 0\\n    for i in range(0, x, y):\\n        s = s + i\\n    return s", "int64", "int64", true}
    };
    for (int t = 0; t < 2; t++) {
        me_artifact *jit = load(&tests[t], ME_JIT_ON);
        me_artifact *reference = load(&tests[t], ME_JIT_OFF);
        me_artifact *oracle = load(&cases[0], ME_JIT_ON);
        CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
        int64_t x[] = {0, 3}, y[] = {1, 1}, output[2], expected[2];
        me_artifact_buffer buffers[] = {{"x", ME_INT64, 8, x, sizeof(x)},
            {"y", ME_INT64, 8, y, sizeof(y)}};
        uint8_t mask[] = {1, 0};
        me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
            .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2,
            .output_capacity = sizeof(output)};
        me_artifact_error error, reference_error;
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        CHECK(!me_artifact_eval_ex(reference, buffers, 2, expected, &descriptor, &reference_error));
        CHECK(!memcmp(output, expected, sizeof(output)));
        if (t == 0) x[1] = 4;
        else y[1] = 0;
        int rc = me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error);
        int reference_rc = me_artifact_eval_ex(reference, buffers, 2, expected, &descriptor, &reference_error);
        CHECK(rc && rc == reference_rc && error.native_status == reference_error.native_status);
        descriptor.valid_mask = mask;
        descriptor.valid_mask_capacity = sizeof(mask);
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        x[1] = 2;
        y[1] = 1;
        descriptor.valid_mask = NULL;
        descriptor.valid_mask_capacity = 0;
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        /* The runtime cap is read per invocation, not baked into cached code. */
        if (t == 0) {
            environment("ME_DSL_WHILE_MAX_ITERS", "1");
            CHECK(me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
            environment("ME_DSL_WHILE_MAX_ITERS", "3");
        }
        me_artifact_free(jit);
        me_artifact_free(reference);
        me_artifact_free(oracle);
    }
    environment("ME_DSL_WHILE_MAX_ITERS", saved_cap);
    free(saved_cap);
}

static void checked_cast_errors(void) {
    parity_case test = {"checked cast boundaries", "if y > 0:\\n        return int(x)\\n    return 0",
        "float64", "int64", true};
    me_artifact *jit = load(&test, ME_JIT_ON);
    me_artifact *reference = load(&test, ME_JIT_OFF);
    double edges[] = {INFINITY, -INFINITY, NAN, 0x1p63, -0x1p63, -1.75, 0x1p53};
    for (size_t edge = 0; edge < sizeof(edges)/sizeof(edges[0]); edge++) {
        double x[] = {2.75, edges[edge]}, y[] = {1, 1};
        int64_t output[2], expected[2];
        uint8_t mask[] = {1, 0};
        me_artifact_buffer buffers[] = {{"x", ME_FLOAT64, 8, x, sizeof(x)},
            {"y", ME_FLOAT64, 8, y, sizeof(y)}};
        me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
            .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2,
            .output_capacity = sizeof(output)};
        me_artifact_error error, reference_error;
        int rc = me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error);
        int reference_rc = me_artifact_eval_ex(reference, buffers, 2, expected, &descriptor, &reference_error);
        CHECK(rc == reference_rc);
        if (!rc) CHECK(!memcmp(output,expected,sizeof(output)));
        else CHECK(error.native_status == reference_error.native_status);
        descriptor.valid_mask = mask;
        descriptor.valid_mask_capacity = sizeof(mask);
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        CHECK(output[0] == 2);
        descriptor.valid_mask = NULL;
        descriptor.valid_mask_capacity = 0;
        y[1] = 0;
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        CHECK(output[1] == 0);
    }
    me_artifact_free(jit);
    me_artifact_free(reference);
}

int main(void) {
    matrix();
    exact_comparisons();
    weak_overflow();
    loop_errors();
    checked_cast_errors();
    return 0;
}
