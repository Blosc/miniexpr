#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "miniexpr_artifact.h"
#include <math.h>
#include <float.h>
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

/* PR system-compiler gates retain every recipe and known backend-sensitive
 * regression, but omit repeated dtype combinations. TCC and full qualification
 * execute the entire matrix. */
static bool compact = false;

/* JSON-escaped bodies deliberately exercise the production artifact loader.
 * Eligibility expectations advance with each qualified implementation phase. */
static const parity_case cases[] = {
    {"arithmetic", "return x * 2 + y", "float64", "float64", true},
    {"conditional return", "if x > 0:\\n        return x\\n    return -x", "float64", "float64", true},
    {"live assignment join", "if x < 0:\\n        return x\\n    else:\\n        z = x + y\\n    return z", "float64", "float64", true},
    {"checked float range", "s = 0.0\\n    for i in range(x):\\n        s = s + i\\n    return s", "float64", "float64", true},
    {"checked unsigned range", "s = 0\\n    for i in range(x):\\n        s = s + i\\n    return s", "uint64", "int64", true},
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
    {"sequenced elif", "if int(x) > 0:\\n        return x\\n    elif int(y) > 0:\\n        return y\\n    return 0.0", "float64", "float64", true},
    {"lazy where casts", "return where(x > 0, int(y), 0)", "float64", "int64", true},
    {"lazy logical casts", "return x > 0 and int(y) > 0", "float64", "bool", true},
};

static me_artifact *load_contract(const parity_case *test, me_jit_mode mode,
    const char *second_input, const char *constants, const char *contract) {
    char json[4096];
    char inputs[256];
    snprintf(inputs, sizeof(inputs), "{\"name\":\"x\",\"dtype\":\"%s\"}%s%s",
        test->input, second_input ? "," : "", second_input ? second_input : "");
    int length = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"block-reductions\"],\"source\":\"def k(%s):\\n    %s\\n\","
        "\"entry_point\":\"k\",\"inputs\":[%s],"
        "\"constants\":[%s],\"output\":{\"dtype\":\"%s\",\"contract\":\"%s\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        constants && strstr(constants,"\"name\":\"z\"") ? "x, y, z" : "x, y",
        test->body, inputs, constants ? constants : "", test->output, contract);
    CHECK(length > 0 && (size_t)length < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error;
    int rc = me_artifact_load(json, (size_t)length, mode, &artifact, &error);
    if (rc) fprintf(stderr, "%s: %s\n", test->name, error.message);
    CHECK(!rc && artifact);
    return artifact;
}

static me_artifact *load_extra(const parity_case *test, me_jit_mode mode,
    const char *second_input, const char *constants) {
    return load_contract(test,mode,second_input,constants,"elementwise");
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
        uint64_t ux[] = {0,1,3}, uy[] = {4,2,1};
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
        if (!strcmp(test->input,"uint64")) {
            buffers[0] = (me_artifact_buffer){"x",ME_UINT64,8,ux,sizeof(ux)};
            buffers[1] = (me_artifact_buffer){"y",ME_UINT64,8,uy,sizeof(uy)};
        }
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
        for (int lane = 0; lane < 5; lane++) {
            if (output[lane] != expected[lane]) fprintf(stderr,
                "comparison mixed=%d op=%s lane=%d actual=%d expected=%d\n",
                mixed,operators[op],lane,output[lane],expected[lane]);
        }
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

static void checked_cast_target(int input64, size_t target) {
    const char *outputs[] = {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"};
    const int widths[] = {8, 8, 16, 16, 32, 32, 64, 64};
    char body[128];
    snprintf(body, sizeof(body), "if y > 0:\\n        return %s(x)\\n    return 0", outputs[target]);
    parity_case test = {"checked cast boundaries", body,
        input64 ? "float64" : "float32", outputs[target], true};
    me_artifact *jit = load(&test, ME_JIT_ON);
    me_artifact *reference = load(&test, ME_JIT_OFF);
    me_artifact *oracle = load(&cases[0], ME_JIT_ON);
    CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
    me_artifact_free(oracle);
    double bound = ldexp(1.0, widths[target] - (target % 2 == 0));
    double edges[] = {INFINITY, -INFINITY, NAN, bound, -bound, -1.75, -0.75,
        0.0, -0.0, 0x1p53, nextafter(bound, 0), nextafter(-bound, -INFINITY)};
    size_t output_size = (size_t)widths[target] / 8;
    for (size_t edge = 0; edge < sizeof(edges)/sizeof(edges[0]); edge++) {
        double x[] = {2.75, edges[edge]}, y[] = {1, 1};
        float xf[] = {2.75f, (float)edges[edge]}, yf[] = {1, 1};
        uint64_t output[2] = {0}, expected[2] = {0};
        uint8_t mask[] = {1, 0};
        me_artifact_buffer buffers[] = {
            {"x", input64 ? ME_FLOAT64 : ME_FLOAT32, input64 ? 8 : 4,
                input64 ? (void *)x : (void *)xf, input64 ? sizeof(x) : sizeof(xf)},
            {"y", input64 ? ME_FLOAT64 : ME_FLOAT32, input64 ? 8 : 4,
                input64 ? (void *)y : (void *)yf, input64 ? sizeof(y) : sizeof(yf)}};
        me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
            .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2,
            .output_capacity = sizeof(output)};
        me_artifact_error error, reference_error;
        int rc = me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error);
        int reference_rc = me_artifact_eval_ex(reference, buffers, 2, expected, &descriptor, &reference_error);
        CHECK(rc == reference_rc);
        if (!rc) CHECK(!memcmp(output,expected,2 * output_size));
        else CHECK(error.native_status == reference_error.native_status);
        descriptor.valid_mask = mask;
        descriptor.valid_mask_capacity = sizeof(mask);
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        CHECK(!me_artifact_eval_ex(reference, buffers, 2, expected, &descriptor, &reference_error));
        CHECK(!memcmp(output,expected,output_size));
        descriptor.valid_mask = NULL;
        descriptor.valid_mask_capacity = 0;
        y[1] = 0;
        yf[1] = 0;
        CHECK(!me_artifact_eval_ex(jit, buffers, 2, output, &descriptor, &error));
        CHECK(!me_artifact_eval_ex(reference, buffers, 2, expected, &descriptor, &reference_error));
        CHECK(!memcmp(output,expected,2 * output_size));
    }
    me_artifact_free(jit);
    me_artifact_free(reference);
}

static void checked_cast_errors(void) {
    for (int input64 = 0; input64 < 2; input64++) {
        for (size_t target = 0; target < 8; target++) {
            checked_cast_target(input64, target);
        }
    }
}

static void computed_weak_capture_case(int64_t capture) {
    char constant[192];
    snprintf(constant,sizeof(constant),"{\"name\":\"y\",\"dtype\":\"int64\",\"category\":\"weak\","
        "\"encoding\":\"decimal\",\"value\":\"%lld\"}",(long long)capture);
    parity_case test = {"computed weak narrowing", "if x == 0:\\n        return x\\n    return x + (y + 1)",
        "int8", "int8", true};
    me_artifact *jit = load_extra(&test, ME_JIT_ON, NULL, constant);
    me_artifact *reference = load_extra(&test, ME_JIT_OFF, NULL, constant);
    me_artifact *oracle = load(&cases[0], ME_JIT_ON);
    CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
    int8_t x[] = {0, 0}, output[2], expected[2];
    me_artifact_buffer buffer = {"x", ME_INT8, 1, x, sizeof(x)};
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2,
        .output_capacity = sizeof(output)};
    me_artifact_error error, reference_error;
    CHECK(!me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error));
    x[1] = 1;
    int rc = me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error);
    int reference_rc = me_artifact_eval_ex(reference, &buffer, 1, expected, &descriptor, &reference_error);
    CHECK(rc == reference_rc);
    if (rc) CHECK(error.native_status == reference_error.native_status);
    else CHECK(!memcmp(output,expected,sizeof(output)));
    uint8_t mask[] = {1, 0};
    descriptor.valid_mask = mask;
    descriptor.valid_mask_capacity = sizeof(mask);
    CHECK(!me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error));
    me_artifact_free(jit);
    me_artifact_free(reference);
    me_artifact_free(oracle);
}

static void computed_weak_conversion(void) {
    int64_t captures[] = {-130,-129,-1,0,126,127,INT64_MAX};
    for (size_t capture = 0; capture < sizeof(captures)/sizeof(captures[0]); capture++) {
        computed_weak_capture_case(captures[capture]);
    }
}

static void uniform_capture_storage(void) {
    /* More than one lane must read the same scalar allocation in both engines.
     * Include a uniform local: its storage must remain per-lane after assignment. */
    const char *bodies[] = {"return x + (y - 1)",
        "a = y\\n    if x > 0:\\n        a = a + x\\n    return a"};
    for (int weak = 0; weak < 2; weak++) {
        char constant[192];
        snprintf(constant, sizeof(constant), "{\"name\":\"y\",\"dtype\":\"int64\","
            "\"category\":\"%s\",\"encoding\":\"decimal\",\"value\":\"1\"}",
            weak ? "weak" : "typed_scalar");
        for (size_t body = 0; body < sizeof(bodies) / sizeof(bodies[0]); body++) {
            parity_case test = {"uniform scalar storage", bodies[body], "float64", "float64", true};
            me_artifact *jit = load_extra(&test, ME_JIT_ON, NULL, constant);
            me_artifact *reference = load_extra(&test, ME_JIT_OFF, NULL, constant);
            double x[257], output[257], expected[257];
            uint8_t mask[257];
            for (int i = 0; i < 257; i++) {
                x[i] = i - 128;
                mask[i] = i % 3 != 0;
            }
            me_artifact_buffer buffer = {"x", ME_FLOAT64, sizeof(double), x, sizeof(x)};
            me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 257,
                .output_capacity = sizeof(output)};
            me_artifact_error error;
            for (int masked = 0; masked < 2; masked++) {
                memset(output, 0, sizeof(output));
                memset(expected, 0, sizeof(expected));
                descriptor.valid_mask = masked ? mask : NULL;
                descriptor.valid_mask_capacity = masked ? sizeof(mask) : 0;
                CHECK(!me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error));
                CHECK(!me_artifact_eval_ex(reference, &buffer, 1, expected, &descriptor, &error));
                CHECK(!memcmp(output, expected, sizeof(output)));
                for (int i = 0; i < 257; i++) {
                    if (!masked || mask[i]) CHECK(output[i] == (body ? (x[i] > 0 ? x[i] + 1 : 1) : x[i]));
                }
            }
            descriptor.nitems = 0;
            descriptor.valid_mask = NULL;
            descriptor.valid_mask_capacity = 0;
            CHECK(!me_artifact_eval_ex(jit, &buffer, 1, output, &descriptor, &error));
            CHECK(!me_artifact_eval_ex(reference, &buffer, 1, expected, &descriptor, &error));
            me_artifact_free(jit);
            me_artifact_free(reference);
        }
    }
}

static void scalar_reductions(void) {
    const char *bodies[] = {"return block_sum(x)", "return block_prod(x)",
        "return block_sum(x * 2)", "return block_prod(x + y)",
        "return block_sum(where(x >= 0, x, 0))", "return block_sum(sqrt(x))",
        "return block_sum(x * 2) / 2", "return sqrt(block_sum(x))",
        "s = block_sum(x)\\n    return s + 1",
        "a = 2.0\\n    s = block_sum(x * a)\\n    t = s / a\\n    return t",
        "return block_mean(x)", "return block_mean(x * 2) / 2",
        "s = block_mean(x)\\n    return s + 1",
        "return block_min(x)", "return block_max(x)",
        "s = block_min(x * 2)\\n    return s / 2",
        "return block_any(x)", "return block_all(x)",
        "return block_any(sqrt(x))", "return block_all(sqrt(x))",
        "return block_sum(x) / block_sum(x * 2)",
        "a = block_sum(x)\\n    b = block_prod(x)\\n    return a + b",
        "s = block_sum(x)\\n    if s > 0:\\n        return block_mean(x)\\n    return s",
        "s = block_sum(x)\\n    if s < 0:\\n        return s\\n    else:\\n        t = block_prod(x)\\n    return t",
        "s = block_sum(x)\\n    if s > 0:\\n        return block_max(x)\\n    elif s < 0:\\n        return block_min(x)\\n    return s",
        "s = block_sum(x)\\n    for i in range(3):\\n        s = s + 1\\n    return s",
        "s = block_mean(x)\\n    for i in range(3, 0, -1):\\n        if i == 2:\\n            continue\\n        s = s * 2\\n    return s",
        "s = block_sum(x)\\n    for i in range(4):\\n        if i == 2:\\n            break\\n        s = s + i\\n    return s",
        "s = block_mean(x)\\n    j = 0\\n    while j < 3:\\n        j = j + 1\\n        if j == 2:\\n            continue\\n        s = s + 1\\n    return s",
        "s = block_sum(x)\\n    for i in range(3):\\n        for j in range(2):\\n            s = s + 1\\n        if i == 1:\\n            return s\\n    return s",
        "s = block_sum(x)\\n    for i in range(0, 3, 0):\\n        s = s + 1\\n    return s",
        "v = x * 2\\n    return block_sum(v)",
        "v = x * 2\\n    w = v + x\\n    return block_sum(v) + block_mean(w)",
        "v = sqrt(x)\\n    return block_sum(x)",
        "v = x * 2\\n    v = v + 1\\n    return block_max(v)",
        "v = int64(x)\\n    return block_mean(v)",
        "v = x * 2\\n    s = block_sum(v)\\n    for i in range(3):\\n        s = s + 1\\n    return s"};
    for (int input64 = 0; input64 < 2; input64++) for (int output64 = 0; output64 < 2; output64++) {
        if (compact && input64 != output64) continue;
        const char *input_name = input64 ? "float64" : "float32";
        const char *output_name = output64 ? "float64" : "float32";
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",input_name);
        for (size_t b = 0; b < sizeof(bodies)/sizeof(bodies[0]); b++) {
            parity_case test = {"scalar reduction",bodies[b],input_name,output_name,true};
            me_artifact *jit = load_contract(&test,ME_JIT_ON,second,NULL,"block_scalar");
            me_artifact *reference = load_contract(&test,ME_JIT_OFF,second,NULL,"block_scalar");
            me_artifact *oracle = load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
            for (int edge = 0; edge < 6; edge++) {
                float f[] = {1e16f,1,-1e16f,3};
                double d[] = {1e16,1,-1e16,3};
                if (edge == 1) { f[1] = NAN; d[1] = NAN; }
                if (edge == 2) { f[0] = INFINITY; d[0] = INFINITY; f[2] = -INFINITY; d[2] = -INFINITY; }
                if (edge == 3) { f[0] = -0.0f; d[0] = -0.0; f[2] = 0; d[2] = 0; }
                if (edge == 4) { f[0] = FLT_MAX; d[0] = DBL_MAX; f[2] = 2; d[2] = 2; }
                if (edge == 5) { f[0] = f[2] = FLT_MIN; d[0] = d[2] = DBL_MIN; }
                void *data = input64 ? (void *)d : (void *)f;
                size_t capacity = input64 ? sizeof(d) : sizeof(f);
                me_dtype dtype = input64 ? ME_FLOAT64 : ME_FLOAT32;
                me_artifact_buffer buffers[] = {{"x",dtype,input64 ? 8 : 4,data,capacity},
                    {"y",dtype,input64 ? 8 : 4,data,capacity}};
                uint8_t partial[] = {1,0,1,0}, none[] = {0,0,0,0};
                for (int mask = 0; mask < 4; mask++) {
                    uint64_t actual = 0, expected = 0;
                    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = mask == 3 ? 0 : 4,
                        .output_capacity = sizeof(actual), .valid_mask = mask == 1 ? partial : mask == 2 ? none : NULL,
                        .valid_mask_capacity = mask == 1 || mask == 2 ? 4 : 0};
                    if (mask == 3) for (int j = 0; j < 2; j++) { buffers[j].data = NULL; buffers[j].capacity = 0; }
                    me_artifact_fp_status status, reference_status;
                    me_artifact_error error, reference_error;
                    int rc = me_artifact_eval_status(jit,buffers,2,&actual,&descriptor,0,&status,&error);
                    int reference_rc = me_artifact_eval_status(reference,buffers,2,&expected,&descriptor,0,
                        &reference_status,&reference_error);
                    CHECK(rc == reference_rc);
                    if (rc) {
                        CHECK(error.native_status == reference_error.native_status);
                        CHECK(status.flags == reference_status.flags);
                        continue;
                    }
                    bool equal = actual == expected;
                    if (output64) {
                        double a, e;
                        memcpy(&a,&actual,8); memcpy(&e,&expected,8);
                        equal |= isnan(a) && isnan(e);
                    } else {
                        float a, e;
                        uint32_t ab = (uint32_t)actual, eb = (uint32_t)expected;
                        memcpy(&a,&ab,4); memcpy(&e,&eb,4);
                        equal |= isnan(a) && isnan(e);
                    }
                    if (!equal || status.flags != reference_status.flags) {
                        fprintf(stderr,"%s %s->%s edge=%d mask=%d bits=%llx/%llx flags=%u/%u\n",
                            bodies[b],input_name,output_name,edge,mask,(unsigned long long)actual,
                            (unsigned long long)expected,status.flags,reference_status.flags);
                    }
                    CHECK(equal && status.flags == reference_status.flags);
                }
            }
            me_artifact_free(jit);
            me_artifact_free(reference);
            me_artifact_free(oracle);
        }
    }
}

static void integer_scalar_reductions(void) {
    const char *bodies[] = {"return block_sum(x)", "return block_prod(x)",
        "return block_sum(where(x != 0, x, x))", "return block_prod(where(x != 0, x, x))",
        "return block_mean(x)", "return block_mean(where(x != 0, x, x))",
        "m = block_mean(x)\\n    return m / 2",
        "return where(block_any(x), block_mean(x), 0.0)"};
    for (int unsigned_input = 0; unsigned_input < 2; unsigned_input++) {
        const char *dtype = unsigned_input ? "uint64" : "int64";
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",dtype);
        for (size_t body = 0; body < sizeof(bodies)/sizeof(bodies[0]); body++) {
            bool mean = body >= 4;
            parity_case test = {"integer scalar reduction",bodies[body],dtype,mean ? "float64" : dtype,true};
            me_artifact *jit = load_contract(&test,ME_JIT_ON,second,NULL,"block_scalar");
            me_artifact *reference = load_contract(&test,ME_JIT_OFF,second,NULL,"block_scalar");
            me_artifact *oracle = load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
            for (int edge = 0; edge < 4; edge++) {
                uint64_t data[] = {edge == 0 ? 1 : unsigned_input ? UINT64_MAX : INT64_MAX,
                    2};
                uint64_t actual = 0, expected = 0;
                me_dtype native_dtype = unsigned_input ? ME_UINT64 : ME_INT64;
                me_artifact_buffer buffers[] = {{"x",native_dtype,8,data,sizeof(data)},
                    {"y",native_dtype,8,data,sizeof(data)}};
                uint8_t mask[] = {1,0};
                me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                    .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = edge == 3 ? 0 : 2,
                    .output_capacity = sizeof(actual), .valid_mask = edge == 2 ? mask : NULL,
                    .valid_mask_capacity = edge == 2 ? sizeof(mask) : 0};
                me_artifact_error error, reference_error;
                me_artifact_fp_status status, reference_status;
                int rc = me_artifact_eval_status(jit,buffers,2,&actual,&descriptor,0,&status,&error);
                int reference_rc = me_artifact_eval_status(reference,buffers,2,&expected,&descriptor,0,
                    &reference_status,&reference_error);
                CHECK(rc == reference_rc && status.flags == reference_status.flags);
                if (rc) CHECK(error.native_status == reference_error.native_status);
                else if (mean) {
                    double a, e;
                    memcpy(&a,&actual,8); memcpy(&e,&expected,8);
                    CHECK(actual == expected || (isnan(a) && isnan(e)));
                } else CHECK(actual == expected);
            }
            me_artifact_free(jit);
            me_artifact_free(reference);
            me_artifact_free(oracle);
        }
    }
}

/* Exhaustive numeric dtype/builtin/extrema/truth qualification belongs here,
 * not in Python's repeated production-binding matrix. */
static void numeric_dtype_matrix(void) {
    const char *names[] = {"bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"};
    me_dtype dtypes[] = {ME_BOOL,ME_INT8,ME_UINT8,ME_INT16,ME_UINT16,ME_INT32,ME_UINT32,ME_INT64,ME_UINT64,ME_FLOAT32,ME_FLOAT64};
    size_t sizes[] = {1,1,1,2,2,4,4,8,8,4,8};
    const char *builtins[] = {"abs","sign","square","floor","ceil","trunc","round","real","imag","conj"};
    const char *reductions[] = {"min","max","any","all","sum","prod","mean"};
    const char *arithmetic[] = {"x + y","x - y","x * y","-x","where(x > y, x + y, x * y)","x + 1"};
    const char *floating_ops[] = {"sin(x) + cos(x)","tan(x)","exp(x)","log(x)","sqrt(x)","floor(x)","ceil(x)",
        "where(x > 0, log(x), y)","x // y","x == y","x != y","x < y","x <= y","x > y","x >= y",
        "x + (1.0 / 0.0)","where(x > 0, 1.0 / 0.0, x)"};
    for (int dtype = 0; dtype < 11; dtype++) {
        if (compact && dtype != 7 && dtype != 8 && dtype != 9 && dtype != 10) continue;
        bool floating = dtype >= 9;
        bool unsigned_input = dtype > 0 && dtype < 9 && dtype % 2 == 0;
        int bits = (int)sizes[dtype] * 8;
        uint64_t maximum = unsigned_input ? UINT64_MAX >> (64-bits) : UINT64_MAX >> (65-bits);
        uint64_t values[] = {dtype == 0 || unsigned_input ? 0 : UINT64_C(1) << (bits-1),0,1,maximum};
        if (floating) {
            values[0] = dtype == 9 ? UINT64_C(0x80000000) : UINT64_C(0x8000000000000000);
            values[2] = dtype == 9 ? UINT64_C(0x7f800001) : UINT64_C(0x7ff0000000000001);
            values[3] = dtype == 9 ? UINT64_C(0x3f800000) : UINT64_C(0x3ff0000000000000);
        } else if (dtype == 0) values[2] = values[3] = 1;
        uint64_t data_storage[4] = {0};
        unsigned char *data = (unsigned char *)data_storage;
        for (int lane = 0; lane < 4; lane++) memcpy(data+lane*sizes[dtype],&values[lane],sizes[dtype]);
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",names[dtype]);
        for (int recipe = 0; recipe < 47; recipe++) {
            bool builtin = recipe < 10;
            bool mean = recipe >= 45;
            bool scalar = (recipe >= 10 && recipe < 22) || mean;
            bool arithmetic_recipe = recipe >= 22 && !mean;
            bool comparison = recipe >= 37 && recipe <= 42;
            if (builtin && (floating || dtype == 0)) continue;
            if (arithmetic_recipe && (dtype == 0 || (recipe >= 28 && !floating))) continue;
            int reduction = mean ? 6 : builtin ? 0 : (recipe-10)/2;
            bool mapped = mean ? recipe == 46 : !builtin && (recipe-10)%2;
            char body[192];
            if (builtin) snprintf(body,sizeof(body),"return %s(x)",builtins[recipe]);
            else if (arithmetic_recipe) snprintf(body,sizeof(body),"return %s",recipe < 28 ? arithmetic[recipe-22] : floating_ops[recipe-28]);
            else snprintf(body,sizeof(body),"return block_%s(%s)",reductions[reduction],
                mapped ? "where(x != 0, x, x)" : "x");
            const char *output = mean ? (floating ? names[dtype] : "float64") : comparison ? "bool" : builtin || arithmetic_recipe || reduction < 2 ? names[dtype] :
                reduction < 4 ? "bool" : floating ? names[dtype] : unsigned_input ? "uint64" : "int64";
            size_t output_size = mean ? (floating ? sizes[dtype] : 8) : comparison ? 1 : builtin || arithmetic_recipe || reduction < 2 ? sizes[dtype] : reduction < 4 ? 1 : floating ? sizes[dtype] : 8;
            parity_case test = {"numeric dtype matrix",body,names[dtype],output,true};
            me_artifact *jit = load_contract(&test,ME_JIT_ON,second,NULL,scalar ? "block_scalar" : "elementwise");
            me_artifact *reference = load_contract(&test,ME_JIT_OFF,second,NULL,scalar ? "block_scalar" : "elementwise");
            if (floating) {
                values[2] = dtype == 9 ? UINT64_C(0x7f800001) : UINT64_C(0x7ff0000000000001);
                memcpy(data+2*sizes[dtype],&values[2],sizes[dtype]);
            }
            me_artifact *oracle = load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
            uint8_t partial[] = {1,0,1,0}, none[] = {0,0,0,0}, nan_only[] = {0,0,1,0};
            for (int mask = 0; mask < (floating ? 9 : 4); mask++) {
                int participation = mask % 4;
                if (floating && mask == 4) {
                    values[2] |= dtype == 9 ? UINT64_C(0x00400000) : UINT64_C(0x0008000000000000);
                    memcpy(data+2*sizes[dtype],&values[2],sizes[dtype]);
                }
                uint64_t actual_storage[4] = {0}, expected_storage[4] = {0};
                unsigned char *actual = (unsigned char *)actual_storage, *expected = (unsigned char *)expected_storage;
                me_artifact_buffer buffers[] = {{"x",dtypes[dtype],sizes[dtype],data,sizeof(data_storage)},
                    {"y",dtypes[dtype],sizes[dtype],data,sizeof(data_storage)}};
                me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                    .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = participation == 3 ? 0 : 4,
                    .output_capacity = sizeof(actual_storage), .valid_mask = mask == 8 ? nan_only : participation == 1 ? partial : participation == 2 ? none : NULL,
                    .valid_mask_capacity = mask == 8 || participation == 1 || participation == 2 ? 4 : 0};
                me_artifact_error error, reference_error;
                me_artifact_fp_status status, reference_status;
                int rc = me_artifact_eval_status(jit,buffers,2,actual,&descriptor,0,&status,&error);
                int reference_rc = me_artifact_eval_status(reference,buffers,2,expected,&descriptor,0,&reference_status,&reference_error);
                if (rc != reference_rc || status.flags != reference_status.flags) {
                    fprintf(stderr,"numeric matrix dtype=%s body=%s mask=%d rc=%d/%d fp=%u/%u\n",
                        names[dtype],body,mask,rc,reference_rc,status.flags,reference_status.flags);
                }
                CHECK(rc == reference_rc && status.flags == reference_status.flags);
                if (rc) CHECK(error.native_status == reference_error.native_status);
                else if (mean || (floating && !comparison && (arithmetic_recipe || reduction < 2 || reduction >= 4))) {
                    for (int lane = 0; lane < (scalar ? 1 : 4); lane++) {
                        double a, e;
                        if (dtype == 9) {
                            float af, ef;
                            memcpy(&af,actual+lane*4,4); memcpy(&ef,expected+lane*4,4); a=af; e=ef;
                        } else { memcpy(&a,actual+lane*8,8); memcpy(&e,expected+lane*8,8); }
                        CHECK((isnan(a) && isnan(e)) || !memcmp(actual+lane*output_size,expected+lane*output_size,output_size));
                    }
                } else CHECK(!memcmp(actual,expected,scalar ? output_size : sizeof(actual_storage)));
            }
            me_artifact_free(jit); me_artifact_free(reference); me_artifact_free(oracle);
        }
    }
}

static void range_bound_matrix(void) {
    const char *names[] = {"float32","float64","uint32","uint64"};
    me_dtype dtypes[] = {ME_FLOAT32,ME_FLOAT64,ME_UINT32,ME_UINT64};
    const char *bounds[] = {"x", "x, 5, 1", "0, 5, x", "5, 0, x"};
    for (int dtype = 0; dtype < 4; dtype++) {
        if (compact && (dtype == 0 || dtype == 2)) continue;
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",names[dtype]);
        for (int bound = 0; bound < 4; bound++) {
            char body[256];
            snprintf(body,sizeof(body),"s = 0\\n    for i in range(%s):\\n        s = s + i\\n    return s",bounds[bound]);
            parity_case test = {"checked range matrix",body,names[dtype],"int64",true};
            me_artifact *jit = load(&cases[0],ME_JIT_ON);
            bool available = me_artifact_has_jit(jit);
            me_artifact_free(jit);
            jit = load_extra(&test,ME_JIT_ON,second,NULL);
            me_artifact *reference = load_extra(&test,ME_JIT_OFF,second,NULL);
            CHECK(me_artifact_has_jit(jit) == available);
            double edges[] = {3.9,-2.9,0.9,NAN,INFINITY,-INFINITY,0x1p63,3};
            for (int edge = 0; edge < 8; edge++) {
                uint64_t data_storage = 0;
                unsigned char *data = (unsigned char *)&data_storage;
                size_t size = dtype == 0 || dtype == 2 ? 4 : 8;
                if (dtype == 0) { float value = (float)edges[edge]; memcpy(data,&value,4); }
                else if (dtype == 1) memcpy(data,&edges[edge],8);
                else if (dtype == 2) { uint32_t value = edge == 6 ? UINT32_MAX : 3; memcpy(data,&value,4); }
                else { uint64_t value = edge == 6 ? UINT64_MAX : 3; memcpy(data,&value,8); }
                /* Avoid impractically long, valid uint32 ranges; high-bit errors
                 * are meaningful only for uint64. */
                if (dtype == 2 && edge == 6) continue;
                for (int masked = 0; masked < 2; masked++) {
                    uint8_t mask = 0;
                    int64_t actual = 0, expected = 0;
                    me_artifact_buffer buffers[] = {{"x",dtypes[dtype],size,data,size},{"y",dtypes[dtype],size,data,size}};
                    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 1,
                        .output_capacity = sizeof(actual), .valid_mask = masked ? &mask : NULL,
                        .valid_mask_capacity = masked ? 1 : 0};
                    me_artifact_error error, reference_error;
                    me_artifact_fp_status status, reference_status;
                    int rc = me_artifact_eval_status(jit,buffers,2,&actual,&descriptor,0,&status,&error);
                    int reference_rc = me_artifact_eval_status(reference,buffers,2,&expected,&descriptor,0,&reference_status,&reference_error);
                    CHECK(rc == reference_rc && status.flags == reference_status.flags);
                    if (rc) CHECK(error.native_status == reference_error.native_status);
                    else CHECK(actual == expected);
                }
            }
            me_artifact_free(jit); me_artifact_free(reference);
        }
    }
}

static void weak_capture_matrix(void) {
    int64_t pairs[][2] = {{INT64_MAX,1},{INT64_MIN,-1},{INT64_MAX,2},{INT64_MIN,0},
        {0,INT64_MIN},{INT64_MAX,-1},{INT64_MIN,1},{7,-3},{0,0}};
    const char *expressions[] = {"y + z","y - z","y * z","-y"};
    for (int expression = 0; expression < 4; expression++) {
        for (int pair = 0; pair < 9; pair++) {
            char constants[384], body[192];
            snprintf(constants,sizeof(constants),
                "{\"name\":\"y\",\"dtype\":\"int64\",\"category\":\"weak\",\"encoding\":\"decimal\",\"value\":\"%lld\"},"
                "{\"name\":\"z\",\"dtype\":\"int64\",\"category\":\"weak\",\"encoding\":\"decimal\",\"value\":\"%lld\"}",
                (long long)pairs[pair][0],(long long)pairs[pair][1]);
            snprintf(body,sizeof(body),"if x == 0:\\n        return x\\n    return x + (%s)",expressions[expression]);
            parity_case test = {"weak capture matrix",body,"int64","int64",true};
            me_artifact *jit = load_extra(&test,ME_JIT_ON,NULL,constants);
            me_artifact *reference = load_extra(&test,ME_JIT_OFF,NULL,constants);
            me_artifact *oracle = load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
            int64_t x[] = {0,1};
            me_artifact_buffer buffer = {"x",ME_INT64,8,x,sizeof(x)};
            for (int masked = 0; masked < 3; masked++) {
                uint8_t mask[] = {1,0};
                int64_t actual[2] = {0}, expected[2] = {0};
                if (masked == 2) x[1] = 0;
                me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                    .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2,
                    .output_capacity = sizeof(actual), .valid_mask = masked == 1 ? mask : NULL,
                    .valid_mask_capacity = masked == 1 ? 2 : 0};
                me_artifact_error error, reference_error;
                me_artifact_fp_status status, reference_status;
                int rc = me_artifact_eval_status(jit,&buffer,1,actual,&descriptor,0,&status,&error);
                int reference_rc = me_artifact_eval_status(reference,&buffer,1,expected,&descriptor,0,&reference_status,&reference_error);
                CHECK(rc == reference_rc && status.flags == reference_status.flags);
                if (rc) CHECK(error.native_status == reference_error.native_status);
                else CHECK(!memcmp(actual,expected,sizeof(actual)));
            }
            me_artifact_free(jit); me_artifact_free(reference); me_artifact_free(oracle);
        }
    }
}

static void scalar_checked_order(void) {
    const char *bodies[] = {"return int64(1e100) + block_sum(sqrt(x))",
        "return block_sum(sqrt(x)) + int64(1e100)",
        "return block_sum(int64(x * 1e100)) + block_sum(sqrt(x))",
        "return block_sum(sqrt(x)) + block_sum(int64(x * 1e100))"};
    for (int first = 0; first < 4; first++) {
        parity_case test = {"scalar checked order",bodies[first],"float64","float64",true};
        const char *second = "{\"name\":\"y\",\"dtype\":\"float64\"}";
        me_artifact *jit = load_contract(&test,ME_JIT_ON,second,NULL,"block_scalar");
        me_artifact *reference = load_contract(&test,ME_JIT_OFF,second,NULL,"block_scalar");
        me_artifact *oracle = load(&cases[0],ME_JIT_ON);
        CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
        double x[] = {-1,4}, actual, expected;
        me_artifact_buffer buffers[] = {{"x",ME_FLOAT64,8,x,sizeof(x)},
            {"y",ME_FLOAT64,8,x,sizeof(x)}};
        me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
            .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2,
            .output_capacity = sizeof(actual)};
        me_artifact_error error, reference_error;
        me_artifact_fp_status status, reference_status;
        int rc = me_artifact_eval_status(jit,buffers,2,&actual,&descriptor,0,&status,&error);
        int reference_rc = me_artifact_eval_status(reference,buffers,2,&expected,&descriptor,0,
            &reference_status,&reference_error);
        CHECK(rc && rc == reference_rc && error.native_status == reference_error.native_status);
        CHECK(status.flags == reference_status.flags);
        if (status.supported) CHECK((status.flags & 1) == (unsigned)(first % 2));
        me_artifact_free(jit);
        me_artifact_free(reference);
        me_artifact_free(oracle);
    }
}

static void scalar_array_fallbacks(void) {
    const char *bodies[] = {
        "s = block_sum(x)\\n    if s > 0:\\n        v = x\\n    else:\\n        v = x * 2\\n    return block_sum(v)",
        "v = x\\n    for i in range(3):\\n        v = v * 2\\n    return block_sum(v)"};
    for (size_t i = 0; i < sizeof(bodies)/sizeof(bodies[0]); i++) {
        parity_case test = {"branch/loop array fallback",bodies[i],"float64","float64",false};
        me_artifact *artifact = load_contract(&test,ME_JIT_ON,
            "{\"name\":\"y\",\"dtype\":\"float64\"}",NULL,"block_scalar");
        CHECK(!me_artifact_has_jit(artifact));
        me_artifact_free(artifact);
    }
}

static void scalar_loop_cap(void) {
    parity_case test = {"scalar loop invocation cap",
        "s = block_mean(x)\\n    j = 0\\n    while j < 5:\\n        j = j + 1\\n        s = s + 1\\n    return s",
        "int64","float64",true};
    me_artifact *jit = load_contract(&test,ME_JIT_ON,
        "{\"name\":\"y\",\"dtype\":\"int64\"}",NULL,"block_scalar");
    me_artifact *reference = load_contract(&test,ME_JIT_OFF,
        "{\"name\":\"y\",\"dtype\":\"int64\"}",NULL,"block_scalar");
    me_artifact *oracle = load(&cases[0],ME_JIT_ON);
    CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
    me_artifact_free(oracle);
    const char *previous = getenv("ME_DSL_WHILE_MAX_ITERS");
    char *saved = previous ? strdup(previous) : NULL;
    int64_t data[] = {1,3};
    me_artifact_buffer buffers[] = {{"x",ME_INT64,8,data,sizeof(data)},
        {"y",ME_INT64,8,data,sizeof(data)}};
    me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
        .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION, .nitems = 2, .output_capacity = 8};
    for (int cap = 0; cap < 2; cap++) {
#ifdef _WIN32
        _putenv_s("ME_DSL_WHILE_MAX_ITERS",cap ? "0" : "2");
#else
        setenv("ME_DSL_WHILE_MAX_ITERS",cap ? "0" : "2",1);
#endif
        double actual = 0, expected = 0;
        me_artifact_error error, reference_error;
        me_artifact_fp_status status, reference_status;
        int rc = me_artifact_eval_status(jit,buffers,2,&actual,&descriptor,0,&status,&error);
        int reference_rc = me_artifact_eval_status(reference,buffers,2,&expected,&descriptor,0,&reference_status,&reference_error);
        CHECK(rc == reference_rc && status.flags == reference_status.flags);
        if (!cap) CHECK(rc && error.native_status == reference_error.native_status);
        else CHECK(!rc && actual == 7 && expected == 7);
    }
#ifdef _WIN32
    _putenv_s("ME_DSL_WHILE_MAX_ITERS",saved ? saved : "");
#else
    if (saved) setenv("ME_DSL_WHILE_MAX_ITERS",saved,1);
    else unsetenv("ME_DSL_WHILE_MAX_ITERS");
#endif
    free(saved);
    me_artifact_free(jit); me_artifact_free(reference);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--compact")) {
            compact = true;
            continue;
        }
        me_artifact *oracle = load(&cases[0], ME_JIT_ON);
        if (!strcmp(argv[i],"--require-jit")) CHECK(me_artifact_has_jit(oracle));
        else if (!strcmp(argv[i],"--require-interpreter")) CHECK(!me_artifact_has_jit(oracle));
        else CHECK(false);
        me_artifact_free(oracle);
    }
    matrix();
    exact_comparisons();
    weak_overflow();
    loop_errors();
    checked_cast_errors();
    computed_weak_conversion();
    uniform_capture_storage();
    scalar_reductions();
    scalar_checked_order();
    scalar_array_fallbacks();
    scalar_loop_cap();
    integer_scalar_reductions();
    numeric_dtype_matrix();
    range_bound_matrix();
    weak_capture_matrix();
    return 0;
}
