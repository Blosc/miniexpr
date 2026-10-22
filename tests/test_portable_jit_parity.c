#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "miniexpr_artifact.h"
#include "dsl_compile_internal.h"
#include "dsl_jit_test.h"
#include "dsl_portable_expr.h"
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
static void integer_extrema_lowering(void) {
    const char *names[]={"bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64"};
    me_dtype dtypes[]={ME_BOOL,ME_INT8,ME_UINT8,ME_INT16,ME_UINT16,ME_INT32,ME_UINT32,ME_INT64,ME_UINT64};
    size_t widths[]={1,1,1,2,2,4,4,8,8};
    const char *recipes[]={"minimum(x,y)","maximum(x,y)","fmin(x,y)","fmax(x,y)","fmod(x,y)"};
    for (int dtype=0; dtype<9; dtype++) {
        if (compact && dtype!=7 && dtype!=8) continue;
        int bits=(int)widths[dtype]*8;
        bool unsign=dtype && !(dtype%2);
        uint64_t high=unsign ? UINT64_MAX >> (64-bits) : UINT64_MAX >> (65-bits);
        uint64_t values[]={dtype && !unsign ? UINT64_C(1)<<(bits-1) : 0,0,1,dtype ? high : 1};
        uint64_t x[4]={0},y[4]={0};
        for (int i=0; i<4; i++) {
            memcpy((char *)x+i*widths[dtype],values+i,widths[dtype]);
            memcpy((char *)y+i*widths[dtype],values+3-i,widths[dtype]);
        }
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",names[dtype]);
        for (int recipe=0; recipe<5; recipe++) {
            if (!dtype && recipe==4) continue;
            if (recipe==4) {
                uint64_t negative_one=UINT64_MAX;
                memcpy(y,&negative_one,widths[dtype]); /* signed minimum / -1 must not execute UB */
            }
            char body[128]; snprintf(body,sizeof(body),"return %s",recipes[recipe]);
            parity_case test={"integral extrema lowering",body,names[dtype],names[dtype],true};
            me_artifact *jit=load_contract(&test,ME_JIT_ON,second,NULL,"elementwise");
            me_artifact *reference=load_contract(&test,ME_JIT_OFF,second,NULL,"elementwise");
            me_artifact *oracle=load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit)==me_artifact_has_jit(oracle));
            if (me_artifact_has_jit(jit)) {
                me_dsl_compiled_program *program=dsl_artifact_program_for_tests(jit);
                CHECK(program && program->jit_c_source && !program->portable_jit_nchecked);
            }
            uint8_t mask[]={1,0,1,0};
            me_artifact_buffer inputs[]={{"x",dtypes[dtype],widths[dtype],x,4*widths[dtype]},
                {"y",dtypes[dtype],widths[dtype],y,4*widths[dtype]}};
            for (int masked=0; masked<2; masked++) {
                uint64_t actual[4]={0},expected[4]={0};
                me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
                    .nitems=4,.output_capacity=4*widths[dtype],.valid_mask=masked ? mask : NULL,
                    .valid_mask_capacity=masked ? 4 : 0};
                me_artifact_error error;
                me_artifact_fp_status a,b;
                CHECK(!me_artifact_eval_status(reference,inputs,2,expected,&descriptor,0,&a,&error));
                CHECK(!me_artifact_eval_status(jit,inputs,2,actual,&descriptor,0,&b,&error));
                CHECK(a.flags==b.flags && a.supported==b.supported);
                CHECK(!memcmp(actual,expected,4*widths[dtype]));
            }
            me_artifact_free(jit); me_artifact_free(reference); me_artifact_free(oracle);
        }
    }
}

static bool batch_lane_equal(const void *actual, const void *expected, size_t width) {
    if (!memcmp(actual, expected, width)) return true;
    if (width == sizeof(float)) {
        float a, e;
        memcpy(&a, actual, sizeof(a));
        memcpy(&e, expected, sizeof(e));
        return isnan(a) && isnan(e);
    }
    if (width == sizeof(double)) {
        double a, e;
        memcpy(&a, actual, sizeof(a));
        memcpy(&e, expected, sizeof(e));
        return isnan(a) && isnan(e);
    }
    return false;
}

static void batch_interpreter_parity(void) {
    const char *recipes[]={"x + y","x - y","x * y","x / y","-x",
        "(x + y) * (x - y)","x * 2 + y","sin(x) + cos(y)","sqrt(x*x + y*y)",
        "((x+y)*(x-y)+x*y)/(x+y+1)"};
    enum { lanes=513 };
    const uint64_t f64[]={0,UINT64_C(0x8000000000000000),UINT64_C(0x3ff0000000000000),
        UINT64_C(0xbff0000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
        UINT64_C(0x7ff800000000002a),UINT64_C(0x7ff0000000000001),1,UINT64_C(0x7fefffffffffffff)};
    const uint64_t f32[]={0,UINT64_C(0x80000000),UINT64_C(0x3f800000),UINT64_C(0xbf800000),
        UINT64_C(0x7f800000),UINT64_C(0xff800000),UINT64_C(0x7fc0002a),UINT64_C(0x7f800001),1,UINT64_C(0x7f7fffff)};
    for (int precision=0; precision<2; precision++) {
        const char *name=precision ? "float64" : "float32";
        me_dtype dtype=precision ? ME_FLOAT64 : ME_FLOAT32;
        size_t width=precision ? 8 : 4;
        const uint64_t *bits=precision ? f64 : f32;
        uint64_t x[lanes],y[lanes],actual[lanes],expected[lanes];
        for (int i=0; i<lanes; i++) {
            memcpy((char *)x+(size_t)i*width,bits+i%10,width);
            memcpy((char *)y+(size_t)i*width,bits+(i*3+2)%10,width);
        }
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",name);
        for (size_t recipe=0; recipe<sizeof(recipes)/sizeof(*recipes); recipe++) {
            char body[128]; snprintf(body,sizeof(body),"return %s",recipes[recipe]);
            parity_case test={"batch interpreter parity",body,name,name,false};
            me_artifact *artifact=load_contract(&test,ME_JIT_OFF,second,NULL,"elementwise");
            CHECK(!me_artifact_has_jit(artifact));
            uint8_t mask[lanes];
            for (int i=0; i<lanes; i++) mask[i]=1;
            for (int masked=0; masked<3; masked++) {
                if (masked==2) for (int i=0; i<lanes; i++) mask[i]=(i%3)!=0;
                memset(actual,0xaa,sizeof(actual)); memset(expected,0xaa,sizeof(expected));
                unsigned flags=0;
                me_artifact_error error;
                for (int i=0; i<lanes; i++) {
                    me_artifact_buffer inputs[]={
                        {"x",dtype,width,(char *)x+(size_t)i*width,width},
                        {"y",dtype,width,(char *)y+(size_t)i*width,width}};
                    me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
                        .nitems=1,.output_capacity=width,.valid_mask=masked ? mask+i : NULL,.valid_mask_capacity=masked ? 1 : 0};
                    me_artifact_fp_status status;
                    CHECK(!me_artifact_eval_status(artifact,inputs,2,(char *)expected+(size_t)i*width,&descriptor,0,&status,&error));
                    flags |= status.flags;
                }
                me_artifact_buffer inputs[]={{"x",dtype,width,x,lanes*width},{"y",dtype,width,y,lanes*width}};
                me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
                    .nitems=lanes,.output_capacity=lanes*width,.valid_mask=masked ? mask : NULL,.valid_mask_capacity=masked ? lanes : 0};
                me_artifact_fp_status status;
                size_t before=dsl_portable_batch_calls_for_tests();
                CHECK(!me_artifact_eval_status(artifact,inputs,2,actual,&descriptor,0,&status,&error));
                CHECK(dsl_portable_batch_calls_for_tests()==before+(masked==2 ? 0 : 1));
                CHECK(status.flags==flags);
                bool lanes_equal = true;
                for (int lane = 0; lane < lanes; lane++) {
                    if (!batch_lane_equal(actual + lane, expected + lane, width)) {
                        lanes_equal = false;
                        break;
                    }
                }
                CHECK(lanes_equal);
            }
            me_artifact_free(artifact);
        }
    }
}

static void batch_complex_parity(void) {
    const char *recipes[]={"x + y","x - y","y - x","x * y","y * x","-x",
        "(x + y) * (x - y)","x * 2 + y","x * (-y) + y","-y",
        "((x+y)*(x-y)+x*y)*(x+y*x)-(-x+y)","x / y"};
    enum { lanes=513 };
    const uint64_t f64[]={0,UINT64_C(0x8000000000000000),UINT64_C(0x3ff0000000000000),
        UINT64_C(0xbff0000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
        UINT64_C(0x7ff800000000002a),UINT64_C(0x7ff0000000000001),1,UINT64_C(0x7fefffffffffffff)};
    const uint64_t f32[]={0,UINT64_C(0x80000000),UINT64_C(0x3f800000),UINT64_C(0xbf800000),
        UINT64_C(0x7f800000),UINT64_C(0xff800000),UINT64_C(0x7fc0002a),UINT64_C(0x7f800001),1,UINT64_C(0x7f7fffff)};
    for (int precision=0; precision<2; precision++) {
        const char *name=precision ? "complex128" : "complex64";
        me_dtype dtype=precision ? ME_COMPLEX128 : ME_COMPLEX64;
        size_t component=precision ? 8 : 4, width=2*component;
        const uint64_t *bits=precision ? f64 : f32;
        uint64_t x[2*lanes],y[2*lanes],actual[2*lanes],expected[2*lanes];
        uint64_t random=UINT64_C(0x123456789abcdef0);
        for (int i=0; i<2*lanes; i++) {
            memcpy((char *)x+(size_t)i*component,bits+i%10,component);
            memcpy((char *)y+(size_t)i*component,bits+(i*3+2)%10,component);
            /* Retain the edge cases above, then mix finite exponents/signs
             * to catch rounding/contraction differences beyond the specials. */
            if (i>=100) {
                uint64_t exponent=precision ? UINT64_C(0x7ff0000000000000) : UINT64_C(0x7f800000);
                for (int input=0; input<2; input++) {
                    random=random*UINT64_C(6364136223846793005)+1;
                    uint64_t value=precision ? random : random>>32;
                    if ((value & exponent)==exponent) value^=precision ? UINT64_C(0x0010000000000000) : UINT64_C(0x00800000);
                    memcpy((char *)(input ? y : x)+(size_t)i*component,&value,component);
                }
            }
        }
        char second[128],constant[256];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",name);
        snprintf(constant,sizeof(constant),
            "{\"name\":\"y\",\"dtype\":\"%s\",\"category\":\"typed_scalar\",\"encoding\":\"ieee754-hex\",\"value\":\"%s\"}",
            name,precision ? "40000000000000004008000000000000" : "4000000040400000");
        for (int uniform=0; uniform<2; uniform++) {
            for (size_t recipe=0; recipe<sizeof(recipes)/sizeof(*recipes); recipe++) {
                char body[128]; snprintf(body,sizeof(body),"return %s",recipes[recipe]);
                parity_case test={"batch complex parity",body,name,name,false};
                me_artifact *artifact=load_contract(&test,ME_JIT_OFF,
                    uniform ? NULL : second,uniform ? constant : NULL,"elementwise");
                uint8_t mask[lanes];
                for (int masked=0; masked<3; masked++) {
                    for (int i=0; i<lanes; i++) mask[i]=masked!=2 || i%3!=0;
                    memset(actual,0xaa,sizeof(actual)); memset(expected,0xaa,sizeof(expected));
                    unsigned flags=0;
                    me_artifact_error error;
                    for (int i=0; i<lanes; i++) {
                        me_artifact_buffer inputs[]={
                            {"x",dtype,width,(char *)x+(size_t)i*width,width},
                            {"y",dtype,width,(char *)y+(size_t)i*width,width}};
                        me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
                            .nitems=1,.output_capacity=width,.valid_mask=masked ? mask+i : NULL,
                            .valid_mask_capacity=masked ? 1 : 0};
                        me_artifact_fp_status status;
                        CHECK(!me_artifact_eval_status(artifact,inputs,uniform ? 1 : 2,
                            (char *)expected+(size_t)i*width,&descriptor,0,&status,&error));
                        flags |= status.flags;
                    }
                    me_artifact_buffer inputs[]={{"x",dtype,width,x,lanes*width},{"y",dtype,width,y,lanes*width}};
                    me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
                        .nitems=lanes,.output_capacity=lanes*width,.valid_mask=masked ? mask : NULL,
                        .valid_mask_capacity=masked ? lanes : 0};
                    me_artifact_fp_status status;
                    size_t before=dsl_portable_batch_calls_for_tests();
                    CHECK(!me_artifact_eval_status(artifact,inputs,uniform ? 1 : 2,actual,&descriptor,0,&status,&error));
                    bool batched=masked!=2 && !strchr(recipes[recipe],'/');
                    CHECK(dsl_portable_batch_calls_for_tests()==before+(batched ? 1 : 0));
                    CHECK(status.flags==flags);
                    for (int c=0; c<2*lanes; c++) {
                        if (!batch_lane_equal((char *)actual+(size_t)c*component,
                            (char *)expected+(size_t)c*component,component)) {
                            fprintf(stderr,"complex parity: %s %s uniform=%d masked=%d component=%d\n",
                                name,recipes[recipe],uniform,masked,c);
                        }
                        CHECK(batch_lane_equal((char *)actual+(size_t)c*component,
                            (char *)expected+(size_t)c*component,component));
                    }
                }
                me_artifact_free(artifact);
            }
        }
    }
}

static void batch_complex_cast_parity(void) {
    const char *names[]={"bool","int8","uint8","int16","uint16","int32","uint32",
        "int64","uint64","float32","float64","complex64","complex128"};
    me_dtype dtypes[]={ME_BOOL,ME_INT8,ME_UINT8,ME_INT16,ME_UINT16,ME_INT32,ME_UINT32,
        ME_INT64,ME_UINT64,ME_FLOAT32,ME_FLOAT64,ME_COMPLEX64,ME_COMPLEX128};
    size_t widths[]={1,1,1,2,2,4,4,8,8,4,8,8,16};
    const uint64_t integers[]={0,1,UINT64_MAX,UINT64_C(1)<<63,INT64_MAX,16777217,UINT64_C(9007199254740993)};
    const uint64_t f32[]={0,1,UINT64_C(0x7f800001),UINT64_C(0x7fc0002a),UINT64_C(0x80000000),UINT64_C(0x7f7fffff),UINT64_C(0xff800000)};
    const uint64_t f64[]={0,1,UINT64_C(0x7ff0000000000001),UINT64_C(0x7ff800000000002a),UINT64_C(0x8000000000000000),UINT64_C(0x7fefffffffffffff),UINT64_C(0xfff0000000000000)};
    enum { lanes=259 };
    for (int dtype=0; dtype<13; dtype++) for (int wide=0; wide<2; wide++) {
        const char *target=wide ? "complex128" : "complex64";
        size_t component=wide ? 8 : 4, result_width=2*component, width=widths[dtype];
        uint64_t x[2*lanes],actual[2*lanes],expected[2*lanes];
        const uint64_t *bits=dtype==9 || dtype==11 ? f32 : dtype==10 || dtype==12 ? f64 : integers;
        size_t input_component=dtype==11 || dtype==12 ? width/2 : width;
        for (size_t i=0; i<lanes*width/input_component; i++) {
            uint64_t value=dtype==0 ? i%2 : bits[i%7];
            memcpy((char *)x+i*input_component,&value,input_component);
        }
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",names[dtype]);
        parity_case test={"batch complex conversion parity","return x",names[dtype],target,false};
        me_artifact *artifact=load_contract(&test,ME_JIT_OFF,second,NULL,"elementwise");
        me_artifact_error error;
        unsigned flags=0;
        for (int i=0; i<lanes; i++) {
            me_artifact_buffer inputs[]={{"x",dtypes[dtype],width,(char *)x+(size_t)i*width,width},
                {"y",dtypes[dtype],width,(char *)x+(size_t)i*width,width}};
            me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
                .nitems=1,.output_capacity=result_width};
            me_artifact_fp_status status;
            CHECK(!me_artifact_eval_status(artifact,inputs,2,(char *)expected+(size_t)i*result_width,
                &descriptor,0,&status,&error));
            flags |= status.flags;
        }
        me_artifact_buffer inputs[]={{"x",dtypes[dtype],width,x,lanes*width},{"y",dtypes[dtype],width,x,lanes*width}};
        me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
            .nitems=lanes,.output_capacity=lanes*result_width};
        me_artifact_fp_status status;
        size_t before=dsl_portable_batch_calls_for_tests();
        CHECK(!me_artifact_eval_status(artifact,inputs,2,actual,&descriptor,0,&status,&error));
        bool same=(dtype==11 && !wide) || (dtype==12 && wide);
        CHECK(dsl_portable_batch_calls_for_tests()==before+(same ? 0 : 1));
        if (status.flags!=flags) fprintf(stderr,"complex cast %s -> %s flags=%u expected=%u\n",
            names[dtype],target,status.flags,flags);
        CHECK(status.flags==flags);
        for (int c=0; c<2*lanes; c++) {
            CHECK(batch_lane_equal((char *)actual+(size_t)c*component,
                (char *)expected+(size_t)c*component,component));
        }
        me_artifact_free(artifact);
    }
}

static void batch_complex_aliases(void) {
    parity_case test={"complex batch alias fallback","return x + y","complex128","complex128",false};
    me_artifact *artifact=load_contract(&test,ME_JIT_OFF,"{\"name\":\"y\",\"dtype\":\"complex128\"}",NULL,"elementwise");
    me_dsl_compiled_program *program=dsl_artifact_program_for_tests(artifact);
    const me_expr *root=program->block.stmts[0]->as.return_stmt.expr.expr;
    double x[514][2],y[513][2],expected[514][2];
    for (int i=0; i<514; i++) {
        x[i][0]=expected[i][0]=i; x[i][1]=expected[i][1]=-i;
        if (i<513) { y[i][0]=0.5; y[i][1]=-0.25; }
    }
    const void *vars[]={x,y};
    for (int i=0; i<513; i++) for (int c=0; c<2; c++) expected[i+1][c]=expected[i][c]+y[i][c];
    size_t before=dsl_portable_batch_calls_for_tests();
    CHECK(!dsl_portable_eval_expr_masked(root,vars,2,NULL,513,NULL,16,x+1));
    CHECK(dsl_portable_batch_calls_for_tests()==before);
    CHECK(!memcmp(x,expected,sizeof(x)));
    for (int i=0; i<513; i++) for (int c=0; c<2; c++) expected[i][c]=x[i][c]+y[i][c];
    CHECK(!dsl_portable_eval_expr_masked(root,vars,2,NULL,513,NULL,16,x));
    CHECK(dsl_portable_batch_calls_for_tests()==before+1);
    CHECK(!memcmp(x,expected,513*16));
    me_artifact_free(artifact);
}

static void batch_integer_parity(void) {
    const char *names[]={"int8","uint8","int16","uint16","int32","uint32","int64","uint64"};
    me_dtype dtypes[]={ME_INT8,ME_UINT8,ME_INT16,ME_UINT16,ME_INT32,ME_UINT32,ME_INT64,ME_UINT64};
    size_t widths[]={1,1,2,2,4,4,8,8};
    const char *recipes[]={"x + y","x - y","x * y","-x","x * 2 + y","x*x+y*y+x*y",
        "x*x+y*y+x*y+x*y+x*y+x*y"};
    enum { lanes=259 };
    for (int dtype=0; dtype<8; dtype++) {
        size_t width=widths[dtype]; unsigned bits=(unsigned)width*8;
        uint64_t values[]={0,1,UINT64_MAX,UINT64_C(1)<<(bits-1),UINT64_MAX>>(65-bits)};
        uint64_t x[lanes],y[lanes],actual[lanes],expected[lanes];
        for (int i=0; i<lanes; i++) {
            memcpy((char *)x+(size_t)i*width,values+i%5,width);
            memcpy((char *)y+(size_t)i*width,values+(i*3+1)%5,width);
        }
        char second[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",names[dtype]);
        for (size_t recipe=0; recipe<sizeof(recipes)/sizeof(*recipes); recipe++) {
            char body[128]; snprintf(body,sizeof(body),"return %s",recipes[recipe]);
            parity_case test={"batch modular integer parity",body,names[dtype],names[dtype],false};
            me_artifact *artifact=load_contract(&test,ME_JIT_OFF,second,NULL,"elementwise");
            me_artifact_error error;
            for (int i=0; i<lanes; i++) {
                me_artifact_buffer inputs[]={{"x",dtypes[dtype],width,(char *)x+(size_t)i*width,width},
                    {"y",dtypes[dtype],width,(char *)y+(size_t)i*width,width}};
                me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,
                    .nitems=1,.output_capacity=width};
                CHECK(!me_artifact_eval_ex(artifact,inputs,2,(char *)expected+(size_t)i*width,&descriptor,&error));
            }
            me_artifact_buffer inputs[]={{"x",dtypes[dtype],width,x,lanes*width},{"y",dtypes[dtype],width,y,lanes*width}};
            me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,.nitems=lanes,.output_capacity=lanes*width};
            size_t before=dsl_portable_batch_calls_for_tests();
            CHECK(!me_artifact_eval_ex(artifact,inputs,2,actual,&descriptor,&error));
            CHECK(dsl_portable_batch_calls_for_tests()==before+1);
            CHECK(!memcmp(actual,expected,lanes*width));
            me_artifact_free(artifact);
        }
    }
}

static void batch_cast_parity(void) {
    const char *names[]={"int64","uint64","float32","float64"};
    me_dtype dtypes[]={ME_INT64,ME_UINT64,ME_FLOAT32,ME_FLOAT64};
    size_t widths[]={8,8,4,8};
    const uint64_t integers[]={0,1,UINT64_MAX,UINT64_C(1)<<63,INT64_MAX,16777217,UINT64_C(9007199254740993)};
    const uint64_t f32[]={0,1,UINT64_C(0x7f800001),UINT64_C(0x7fc0002a),UINT64_C(0x80000000),UINT64_C(0x7f7fffff),UINT64_C(0xff800000)};
    const uint64_t f64[]={0,1,UINT64_C(0x7ff0000000000001),UINT64_C(0x7ff800000000002a),UINT64_C(0x8000000000000000),UINT64_C(0x7fefffffffffffff),UINT64_C(0xfff0000000000000)};
    enum { lanes=129 };
    for (int dtype=0; dtype<4; dtype++) for (int wide=0; wide<2; wide++) {
        const char *target=wide ? "float64" : "float32";
        size_t width=widths[dtype], result_width=wide ? 8 : 4;
        uint64_t x[lanes],y[lanes],actual[lanes],expected[lanes];
        const uint64_t *values=dtype==2 ? f32 : dtype==3 ? f64 : integers;
        for (int i=0; i<lanes; i++) {
            memcpy((char *)x+(size_t)i*width,values+i%7,width);
            memcpy((char *)y+(size_t)i*width,values+(i*3+2)%7,width);
        }
        char second[128],body[128];
        snprintf(second,sizeof(second),"{\"name\":\"y\",\"dtype\":\"%s\"}",names[dtype]);
        snprintf(body,sizeof(body),"return %s(x) + %s(y)",target,target);
        parity_case test={"batch conversion parity",body,names[dtype],target,false};
        me_artifact *artifact=load_contract(&test,ME_JIT_OFF,second,NULL,"elementwise");
        me_artifact_error error; unsigned flags=0;
        for (int i=0; i<lanes; i++) {
            me_artifact_buffer inputs[]={{"x",dtypes[dtype],width,(char *)x+(size_t)i*width,width},
                {"y",dtypes[dtype],width,(char *)y+(size_t)i*width,width}};
            me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,.nitems=1,.output_capacity=result_width};
            me_artifact_fp_status status;
            CHECK(!me_artifact_eval_status(artifact,inputs,2,(char *)expected+(size_t)i*result_width,&descriptor,0,&status,&error));
            flags |= status.flags;
        }
        me_artifact_buffer inputs[]={{"x",dtypes[dtype],width,x,lanes*width},{"y",dtypes[dtype],width,y,lanes*width}};
        me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),.version=1,.nitems=lanes,.output_capacity=lanes*result_width};
        me_artifact_fp_status status;
        size_t before=dsl_portable_batch_calls_for_tests();
        CHECK(!me_artifact_eval_status(artifact,inputs,2,actual,&descriptor,0,&status,&error));
        CHECK(dsl_portable_batch_calls_for_tests()==before+1);
        CHECK(status.flags==flags && !memcmp(actual,expected,lanes*result_width));
        me_artifact_free(artifact);
    }
}

static void batch_checked_and_alias_fallbacks(void) {
    parity_case test={"batch alias fallback","return x + y","float64","float64",false};
    me_artifact *artifact=load_contract(&test,ME_JIT_OFF,"{\"name\":\"y\",\"dtype\":\"float64\"}",NULL,"elementwise");
    me_dsl_compiled_program *program=dsl_artifact_program_for_tests(artifact);
    const me_expr *root=program->block.stmts[0]->as.return_stmt.expr.expr;
    double x[513],y[513],expected[513],output[513];
    for (int i=0; i<513; i++) { x[i]=expected[i]=i+1; y[i]=0.5; output[i]=-999; }
    const void *vars[]={x,y};
    for (int i=0; i<512; i++) expected[i+1]=expected[i]+y[i];
    size_t before=dsl_portable_batch_calls_for_tests();
    CHECK(!dsl_portable_eval_expr_masked(root,vars,2,NULL,512,NULL,8,x+1));
    CHECK(dsl_portable_batch_calls_for_tests()==before);
    CHECK(!memcmp(x,expected,sizeof(x)));
    /* Exact same-lane in-place stores are safe and still take the batch route. */
    for (int i=0; i<513; i++) expected[i]=x[i]+y[i];
    CHECK(!dsl_portable_eval_expr_masked(root,vars,2,NULL,513,NULL,8,x));
    CHECK(dsl_portable_batch_calls_for_tests()==before+1);
    CHECK(!memcmp(x,expected,sizeof(x)));
    uint8_t defined[513]; memset(defined,1,sizeof(defined)); defined[3]=0;
    const uint8_t *initialized[]={defined,NULL};
    before=dsl_portable_batch_calls_for_tests();
    CHECK(dsl_portable_eval_expr_masked(root,vars,2,initialized,513,NULL,8,output)==ME_EVAL_ERR_INVALID_ARG);
    CHECK(dsl_portable_batch_calls_for_tests()==before);
    for (int i=0; i<513; i++) CHECK(output[i]==(i<3 ? x[i]+y[i] : -999));
    me_artifact_free(artifact);
    test.body="return float64(int64(x)) + y";
    artifact=load_contract(&test,ME_JIT_OFF,"{\"name\":\"y\",\"dtype\":\"float64\"}",NULL,"elementwise");
    program=dsl_artifact_program_for_tests(artifact);
    root=program->block.stmts[0]->as.return_stmt.expr.expr;
    for (int i=0; i<513; i++) { x[i]=1; output[i]=-999; }
    x[64]=1e100;
    CHECK(dsl_portable_eval_expr_masked(root,vars,2,NULL,513,NULL,8,output)==ME_EVAL_ERR_INVALID_ARG);
    CHECK(dsl_portable_batch_calls_for_tests()==before);
    for (int i=0; i<513; i++) CHECK(output[i]==(i<64 ? 1.5 : -999));
    me_artifact_free(artifact);
}

static void numeric_dtype_matrix(void) {
    const char *names[] = {"bool","int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"};
    me_dtype dtypes[] = {ME_BOOL,ME_INT8,ME_UINT8,ME_INT16,ME_UINT16,ME_INT32,ME_UINT32,ME_INT64,ME_UINT64,ME_FLOAT32,ME_FLOAT64};
    size_t sizes[] = {1,1,1,2,2,4,4,8,8,4,8};
    const char *builtins[] = {"abs","sign","square","floor","ceil","trunc","round","real","imag","conj"};
    const char *reductions[] = {"min","max","any","all","sum","prod","mean"};
    const char *arithmetic[] = {"x + y","x - y","x * y","-x","where(x > y, x + y, x * y)","x + 1"};
    /* Keep exhaustive integer operator/width parity in the native matrix;
     * Python integration tests only need representative scheduler signatures. */
    const char *integer_ops[] = {"x - 1","x * 3","~x","x + (y - 1)","x ^ (y | 1)"};
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
        for (int recipe = 0; recipe < 47 + (int)(sizeof(integer_ops)/sizeof(integer_ops[0])); recipe++) {
            bool builtin = recipe < 10;
            bool integer_recipe = recipe >= 47;
            bool mean = recipe >= 45 && !integer_recipe;
            bool scalar = (recipe >= 10 && recipe < 22) || mean;
            bool arithmetic_recipe = recipe >= 22 && !mean;
            bool comparison = recipe >= 37 && recipe <= 42;
            if (builtin && (floating || dtype == 0)) continue;
            if (integer_recipe && floating) continue;
            if (arithmetic_recipe && (dtype == 0 || (recipe >= 28 && !floating && !integer_recipe))) continue;
            int reduction = mean ? 6 : builtin ? 0 : (recipe-10)/2;
            bool mapped = mean ? recipe == 46 : !builtin && (recipe-10)%2;
            char body[192];
            if (builtin) snprintf(body,sizeof(body),"return %s(x)",builtins[recipe]);
            else if (arithmetic_recipe) snprintf(body,sizeof(body),"return %s",integer_recipe ? integer_ops[recipe-47] : recipe < 28 ? arithmetic[recipe-22] : floating_ops[recipe-28]);
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
            if (me_artifact_has_jit(jit) && dtype > 0 && dtype < 9 &&
                (recipe == 22 || recipe == 23 || recipe == 24 || recipe == 27 || recipe == 47 || recipe == 48)) {
                me_dsl_compiled_program *program = dsl_artifact_program_for_tests(jit);
                CHECK(program && program->jit_c_source);
                /* Wrapping boundary/mask parity below also guards the direct
                 * unsigned store; assert that the TCC fast lowering is used. */
                char store[128];
                snprintf(store,sizeof(store),"((%s%s_t *)output)[i] = (%s%s_t)",
                    unsigned_input ? "" : "u",names[dtype],
                    unsigned_input ? "" : "u",names[dtype]);
                CHECK(strstr(program->jit_c_source,store));
                CHECK(!strstr(program->jit_c_source,"((volatile "));
                CHECK(!strstr(program->jit_c_source,"memcpy(&s"));
            }
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

static void captured_integer_trees(void) {
    const char *names[] = {"int8","uint8","int16","uint16","int32","uint32","int64","uint64"};
    me_dtype types[] = {ME_INT8,ME_UINT8,ME_INT16,ME_UINT16,ME_INT32,ME_UINT32,ME_INT64,ME_UINT64};
    int64_t captures[] = {-129,-1,0,1,127,128,32768,INT64_MAX};
    for (int dtype=0; dtype<8; dtype++) {
        if (compact && dtype != 2 && dtype != 7) continue;
        size_t width=dtype_size(types[dtype]);
        uint64_t input[8]={0}, actual[8], expected[8];
        for (int i=0; i<8; i++) {
            uint64_t bits=i%2 ? UINT64_MAX-(uint64_t)i : (uint64_t)i;
            switch (width) {
                case 1: ((uint8_t *)input)[i]=(uint8_t)bits; break;
                case 2: ((uint16_t *)input)[i]=(uint16_t)bits; break;
                case 4: ((uint32_t *)input)[i]=(uint32_t)bits; break;
                case 8: input[i]=bits; break;
            }
        }
        for (int capture=0; capture<8; capture++) {
            char constant[256];
            snprintf(constant,sizeof(constant),"{\"name\":\"y\",\"dtype\":\"int64\",\"category\":\"weak\","
                "\"encoding\":\"decimal\",\"value\":\"%lld\"}",(long long)captures[capture]);
            parity_case test={"captured modular integer tree","return x + (x - y)",names[dtype],names[dtype],true};
            me_artifact *jit=load_extra(&test,ME_JIT_ON,NULL,constant);
            me_artifact *oracle=load_extra(&test,ME_JIT_OFF,NULL,constant);
            for (int selection=0; selection<4; selection++) {
                uint8_t mask[8];
                for (int i=0; i<8; i++) mask[i]=selection == 1 ? 0 : i%2;
                memset(actual,0xa5,sizeof(actual)); memset(expected,0xa5,sizeof(expected));
                me_artifact_buffer buffer={"x",types[dtype],width,input,8*width};
                me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),
                    .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,.nitems=selection == 3 ? 0 : 8,
                    .output_capacity=8*width,.valid_mask=selection ? mask : NULL,.valid_mask_capacity=selection ? 8 : 0};
                me_artifact_error error, reference_error;
                me_artifact_fp_status status, reference_status;
                int rc=me_artifact_eval_status(jit,&buffer,1,actual,&descriptor,0,&status,&error);
                int reference_rc=me_artifact_eval_status(oracle,&buffer,1,expected,&descriptor,0,&reference_status,&reference_error);
                CHECK(rc == reference_rc && status.flags == reference_status.flags);
                if (rc) CHECK(error.native_status == reference_error.native_status);
                else CHECK(!memcmp(actual,expected,sizeof(actual)));
            }
            if (captures[capture] == 1 && me_artifact_has_jit(jit)) {
                me_dsl_compiled_program *program=dsl_artifact_program_for_tests(jit);
                CHECK(strstr(program->jit_c_source,"*)output)[i] =") != NULL);
                /* No per-node signed conversion temporaries in the fused tree. */
                CHECK(strstr(program->jit_c_source,"pj_expr_") == NULL);
            }
            me_artifact_free(jit); me_artifact_free(oracle);
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

/* Qualification stays native: every resolved libm operation/precision, exceptional
 * bit patterns, empty/masked calls and per-lane IEEE status. The special-policy
 * cases also assert that the resolver leaves them on the generic bridge. */
static void resolved_libm_parity(void) {
    const char *unary[] = {"acos","acosh","asin","asinh","atan","atanh","cbrt",
        "cos","cosh","erf","erfc","exp","exp2","expm1","lgamma","log","log10",
        "log1p","log2","sin","sinh","sqrt","tan","tanh","tgamma","abs","ceil","floor","trunc",
        "square","sign","real","imag","conj","round","rint","exp10","sinpi","cospi"};
    const char *binary[] = {"atan2","copysign","fdim","hypot","nextafter","fmod","pow",
        "minimum","maximum","fmin","fmax","logaddexp"};
    const char *special[] = {"remainder(x, y)","fma(x, y, 1.0)","ldexp(x, 2)"};
    const char *predicates[] = {"isfinite","isinf","isnan","signbit"};
    uint64_t bits64[] = {0,UINT64_C(0x8000000000000000),UINT64_C(0x3ff0000000000000),
        UINT64_C(0xbff0000000000000),UINT64_C(0x3fe0000000000000),UINT64_C(0xbfe0000000000000),
        UINT64_C(0x0010000000000000),1,UINT64_C(0x8000000000000001),
        UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
        UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001),
        UINT64_C(0x7fefffffffffffff),UINT64_C(0xffefffffffffffff),UINT64_C(0x4004000000000000)};
    uint32_t bits32[] = {0,UINT32_C(0x80000000),UINT32_C(0x3f800000),UINT32_C(0xbf800000),
        UINT32_C(0x3f000000),UINT32_C(0xbf000000),UINT32_C(0x00800000),1,UINT32_C(0x80000001),
        UINT32_C(0x7f800000),UINT32_C(0xff800000),UINT32_C(0x7fc00001),UINT32_C(0x7f800001),
        UINT32_C(0x7f7fffff),UINT32_C(0xff7fffff),UINT32_C(0x40200000)};
    int nunary = (int)(sizeof(unary)/sizeof(unary[0]));
    int nbinary = (int)(sizeof(binary)/sizeof(binary[0]));
    int nspecial = (int)(sizeof(special)/sizeof(special[0]));
    int npredicates = (int)(sizeof(predicates)/sizeof(predicates[0]));
    for (int narrow = 0; narrow < 2; narrow++) {
        me_dtype dtype = narrow ? ME_FLOAT32 : ME_FLOAT64;
        size_t width = narrow ? 4 : 8;
        const char *name = narrow ? "float32" : "float64";
        uint64_t xs[16] = {0}, ys[16] = {0};
        unsigned char *x = (unsigned char *)xs, *y = (unsigned char *)ys;
        for (int lane = 0; lane < 16; lane++) {
            memcpy(x+lane*width,narrow ? (void *)&bits32[lane] : (void *)&bits64[lane],width);
            int other = (lane+3)%16;
            memcpy(y+lane*width,narrow ? (void *)&bits32[other] : (void *)&bits64[other],width);
        }
        for (int op = 0; op < nunary+nbinary+nspecial+npredicates; op++) {
            char body[160];
            bool predicate = op >= nunary+nbinary+nspecial;
            if (op < nunary) snprintf(body,sizeof(body),"return %s(x)",unary[op]);
            else if (op < nunary+nbinary) snprintf(body,sizeof(body),"return %s(x, y)",binary[op-nunary]);
            else if (!predicate) snprintf(body,sizeof(body),"return %s",special[op-nunary-nbinary]);
            else snprintf(body,sizeof(body),"return %s(x)",predicates[op-nunary-nbinary-nspecial]);
            parity_case test = {"resolved libm",body,name,predicate ? "bool" : name,true};
            me_artifact *jit = load(&test,ME_JIT_ON), *reference = load(&test,ME_JIT_OFF);
            me_artifact *oracle = load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
            me_artifact_free(oracle);
            if (me_artifact_has_jit(jit)) {
                me_dsl_compiled_program *program = dsl_artifact_program_for_tests(jit);
                bool direct = (program->portable_jit_nmath && program->portable_jit_math_fn[0]) ||
                    (program->portable_jit_nmath2 && program->portable_jit_math2_fn[0]) ||
                    (program->portable_jit_nmath3 && program->portable_jit_math3_fn[0]) ||
                    (program->portable_jit_npreds && program->portable_jit_pred_fn[0]);
                CHECK(direct == (op != nunary+nbinary));
            }
            for (int selection = 0; selection < 20; selection++) {
                uint64_t actual_storage[16], expected_storage[16];
                memset(actual_storage,0xa5,sizeof(actual_storage));
                memset(expected_storage,0xa5,sizeof(expected_storage));
                uint8_t mask[16] = {0};
                for (int lane = 0; lane < 16; lane++) mask[lane] = selection == 0 ||
                    (selection == 2 && lane%2) || (selection >= 4 && lane == selection-4);
                me_artifact_buffer buffers[] = {{"x",dtype,width,x,16*width},{"y",dtype,width,y,16*width}};
                me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                    .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,.nitems = selection == 3 ? 0 : 16,
                    .output_capacity = sizeof(actual_storage),.valid_mask = selection ? mask : NULL,
                    .valid_mask_capacity = selection ? 16 : 0};
                me_artifact_error error, reference_error;
                me_artifact_fp_status status, reference_status;
                int rc = me_artifact_eval_status(jit,buffers,2,actual_storage,&descriptor,0,&status,&error);
                int expected_rc = me_artifact_eval_status(reference,buffers,2,expected_storage,&descriptor,0,
                    &reference_status,&reference_error);
                if (rc != expected_rc || status.flags != reference_status.flags) fprintf(stderr,
                    "resolved libm %s %s selection=%d rc=%d/%d flags=%u/%u\n",
                    name,body,selection,rc,expected_rc,status.flags,reference_status.flags);
                CHECK(rc == expected_rc && status.flags == reference_status.flags);
                CHECK(!rc);
                for (int lane = 0; lane < 16; lane++) {
                    size_t output_width = predicate ? 1 : width;
                    const unsigned char *a = (unsigned char *)actual_storage+lane*output_width;
                    const unsigned char *b = (unsigned char *)expected_storage+lane*output_width;
                    if (!memcmp(a,b,output_width)) continue;
                    CHECK(!predicate);
                    /* NaN payloads are not part of the portable contract. */
                    double av, bv;
                    if (narrow) {
                        float af, bf;
                        memcpy(&af,a,4); memcpy(&bf,b,4); av=af; bv=bf;
                    } else { memcpy(&av,a,8); memcpy(&bv,b,8); }
                    if (!(isnan(av) && isnan(bv))) fprintf(stderr,
                        "resolved libm value %s %s selection=%d lane=%d\n",name,body,selection,lane);
                    CHECK(isnan(av) && isnan(bv));
                }
            }
            me_artifact_free(jit);
            me_artifact_free(reference);
        }
    }
}

static void integer_tree_parity(void) {
    const char *names[] = {"int8","uint8","int16","uint16","int32","uint32","int64","uint64","bool"};
    me_dtype types[] = {ME_INT8,ME_UINT8,ME_INT16,ME_UINT16,ME_INT32,ME_UINT32,ME_INT64,ME_UINT64,ME_BOOL};
    const char *expressions[] = {"x + y","(x + y) * (x - y) + x","((x + y) * 2 - y) * 3 + x * 2 - y",
        "x & y","x | y","x ^ y","~x","-x","x << y","x >> y","x // y","x % y",
        "abs(x)","sign(x)","square(x)","real(x)","imag(x)","conj(x)","floor(x)","ceil(x)","trunc(x)","round(x)","x ** y"};
    const char *boolean[] = {"(x & y) ^ (~x | y)","(x + y) * (~y)"};
    for (int dtype=0; dtype<9; dtype++) {
        if (compact && dtype != 0 && dtype != 6 && dtype != 7 && dtype != 8) continue;
        size_t width=dtype_size(types[dtype]);
        int bits=(int)(width*8);
        uint64_t maximum=UINT64_MAX >> (64-bits), minimum=UINT64_C(1) << (bits-1);
        uint64_t left[] = {0,1,maximum,minimum,maximum>>1,2,7,minimum+1};
        uint64_t right[] = {0,1,maximum,(uint64_t)bits-1,(uint64_t)bits,(uint64_t)bits+1,2,3};
        void *x=malloc(8*width), *y=malloc(8*width), *actual=malloc(8*width), *expected=malloc(8*width);
        CHECK(x && y && actual && expected);
        for (int lane=0; lane<8; lane++) {
            if (types[dtype] == ME_BOOL) { ((bool *)x)[lane]=lane%2; ((bool *)y)[lane]=(lane/2)%2; }
            else switch (width) {
                case 1: ((uint8_t *)x)[lane]=(uint8_t)left[lane]; ((uint8_t *)y)[lane]=(uint8_t)right[lane]; break;
                case 2: ((uint16_t *)x)[lane]=(uint16_t)left[lane]; ((uint16_t *)y)[lane]=(uint16_t)right[lane]; break;
                case 4: ((uint32_t *)x)[lane]=(uint32_t)left[lane]; ((uint32_t *)y)[lane]=(uint32_t)right[lane]; break;
                case 8: ((uint64_t *)x)[lane]=left[lane]; ((uint64_t *)y)[lane]=right[lane]; break;
            }
        }
        int count=dtype == 8 ? 2 : (int)(sizeof(expressions)/sizeof(expressions[0]));
        for (int expression=0; expression<count; expression++) {
            char body[256];
            snprintf(body,sizeof(body),"return %s",dtype == 8 ? boolean[expression] : expressions[expression]);
            parity_case test={"integer tree",body,names[dtype],names[dtype],true};
            me_artifact *jit=load(&test,ME_JIT_ON), *reference=load(&test,ME_JIT_OFF);
            me_artifact *oracle=load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
            me_artifact_free(oracle);
            for (int selection=0; selection<12; selection++) {
                uint8_t mask[8];
                for (int lane=0; lane<8; lane++) mask[lane]=selection == 0 ||
                    (selection == 2 && lane%2) || (selection >= 4 && lane == selection-4);
                memset(actual,0xa5,8*width); memset(expected,0xa5,8*width);
                me_artifact_buffer buffers[]={{"x",types[dtype],width,x,8*width},{"y",types[dtype],width,y,8*width}};
                me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),
                    .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,.nitems=selection == 3 ? 0 : 8,
                    .output_capacity=8*width,.valid_mask=selection ? mask : NULL,.valid_mask_capacity=selection ? 8 : 0};
                me_artifact_error error, reference_error;
                me_artifact_fp_status status, reference_status;
                int rc=me_artifact_eval_status(jit,buffers,2,actual,&descriptor,0,&status,&error);
                int expected_rc=me_artifact_eval_status(reference,buffers,2,expected,&descriptor,0,&reference_status,&reference_error);
                if (rc != expected_rc || status.flags != reference_status.flags || (!rc && memcmp(actual,expected,8*width)))
                    fprintf(stderr,"integer lowering %s %s selection=%d rc=%d/%d flags=%u/%u\n",
                        names[dtype],body,selection,rc,expected_rc,status.flags,reference_status.flags);
                CHECK(rc == expected_rc && status.flags == reference_status.flags);
                if (rc) CHECK(error.native_status == reference_error.native_status);
                else CHECK(!memcmp(actual,expected,8*width));
            }
            me_artifact_free(jit); me_artifact_free(reference);
        }
        free(x); free(y); free(actual); free(expected);
    }
}

static void runtime_literal_parity(void) {
    const char *bodies[] = {
        "return x + (y - 1)", "return x + 2", "return 2 - x",
        "return x * 2", "return (x + y) * 2", "return x / 2",
        "return x + 0.25", "return x * 0", "return x * 1",
        "return x * -1", "return x / 1", "return x + 0", "return x - 0",
        "return x + (1.0 / 0.0)", "return x + (0.0 / 0.0)",
        "return x + (1e30 * 1e30)", "return x + (1e-30 * 1e-30)",
        "return x + (16777217 * 1.0)", "return x + 9007199254740993",
        "return where(x > 0, y - 1, 0.0 / 0.0)",
        "z = 1e30 * 1e30\\n    return x + z",
        "return x + (1e308 * 1e308)", "return x + (1e-300 * 1e-300)"
    };
    uint64_t bits64[] = {UINT64_C(0x7ff0000000000001), UINT64_C(0x7ff8000000000000),
        UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
        0, UINT64_C(0x8000000000000000), 1, UINT64_C(0x7fefffffffffffff)};
    uint32_t bits32[] = {UINT32_C(0x7f800001), UINT32_C(0x7fc00000),
        UINT32_C(0x7f800000), UINT32_C(0xff800000),
        0, UINT32_C(0x80000000), 1, UINT32_C(0x7f7fffff)};
    for (int narrow = 0; narrow < 2; narrow++) {
        size_t itemsize = narrow ? 4 : 8;
        uint64_t x_storage[16] = {0}, y_storage[16] = {0};
        unsigned char *x = (unsigned char *)x_storage, *y = (unsigned char *)y_storage;
        for (int lane = 0; lane < 16; lane++) {
            if (narrow) {
                memcpy(x+lane*itemsize,&bits32[lane%8],itemsize);
                float value = lane+1;
                memcpy(y+lane*itemsize,&value,itemsize);
            } else {
                memcpy(x+lane*itemsize,&bits64[lane%8],itemsize);
                double value = lane+1;
                memcpy(y+lane*itemsize,&value,itemsize);
            }
        }
        for (size_t body = 0; body < sizeof(bodies)/sizeof(bodies[0]); body++) {
            if (narrow && body >= 21) continue;
            const char *dtype = narrow ? "float32" : "float64";
            parity_case test = {"runtime literal parity",bodies[body],dtype,dtype,true};
            me_artifact *jit = load(&test,ME_JIT_ON);
            me_artifact *reference = load(&test,ME_JIT_OFF);
            me_artifact *oracle = load(&cases[0],ME_JIT_ON);
            CHECK(me_artifact_has_jit(jit) == me_artifact_has_jit(oracle));
            me_artifact_free(oracle);
            if (me_artifact_has_jit(jit)) {
                me_dsl_compiled_program *program = dsl_artifact_program_for_tests(jit);
                CHECK(program && program->jit_c_source);
                /* Also test the optimization itself: semantic parity alone
                 * would not catch restoring the per-lane volatile barrier. */
                CHECK((strstr(program->jit_c_source,"((volatile ") != NULL) == (body >= 7));
            }
            me_artifact_buffer buffers[] = {{"x",narrow ? ME_FLOAT32 : ME_FLOAT64,itemsize,x,16*itemsize},
                {"y",narrow ? ME_FLOAT32 : ME_FLOAT64,itemsize,y,16*itemsize}};
            /* Full reads exercise SIMD; singleton, empty and masked reads audit
             * per-lane exceptions and nonparticipating constant-only subtrees. */
            for (int selection = 0; selection < 20; selection++) {
                uint64_t actual_storage[16], expected_storage[16];
                unsigned char *actual = (unsigned char *)actual_storage;
                unsigned char *expected = (unsigned char *)expected_storage;
                memset(actual_storage,0xa5,sizeof(actual_storage));
                memset(expected_storage,0xa5,sizeof(expected_storage));
                uint8_t mask[16] = {0};
                for (int lane = 0; lane < 16; lane++) {
                    mask[lane] = selection == 0 || (selection == 2 && lane%2) ||
                        (selection >= 4 && lane == selection-4);
                }
                me_artifact_eval_descriptor descriptor = {.struct_size = sizeof(descriptor),
                    .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
                    .nitems = selection == 3 ? 0 : 16, .output_capacity = sizeof(actual_storage),
                    .valid_mask = selection == 0 ? NULL : mask,
                    .valid_mask_capacity = sizeof(mask)};
                me_artifact_fp_status status, reference_status;
                me_artifact_error error, reference_error;
                int rc = me_artifact_eval_status(jit,buffers,2,actual,&descriptor,0,&status,&error);
                int reference_rc = me_artifact_eval_status(reference,buffers,2,expected,&descriptor,0,
                    &reference_status,&reference_error);
                bool equal = true;
                if (!rc && !reference_rc) {
                    for (int lane = 0; lane < 16; lane++) {
                        if (!memcmp(actual+lane*itemsize,expected+lane*itemsize,itemsize)) continue;
                        /* The portable contract leaves NaN payload/sign
                         * unspecified; all other bits (including zero sign and
                         * untouched masked output) must still match exactly. */
                        double a, e;
                        if (narrow) {
                            float af, ef;
                            memcpy(&af,actual+lane*itemsize,itemsize);
                            memcpy(&ef,expected+lane*itemsize,itemsize);
                            a = af;
                            e = ef;
                        } else {
                            memcpy(&a,actual+lane*itemsize,itemsize);
                            memcpy(&e,expected+lane*itemsize,itemsize);
                        }
                        equal &= isnan(a) && isnan(e);
                    }
                }
                if (rc != reference_rc || status.flags != reference_status.flags ||
                    (!rc && !equal)) {
                    fprintf(stderr,"literal parity dtype=%s body=%s selection=%d rc=%d/%d flags=%u/%u\n",
                        dtype,bodies[body],selection,rc,reference_rc,status.flags,reference_status.flags);
                }
                CHECK(rc == reference_rc);
                CHECK(status.flags == reference_status.flags);
                CHECK(status.supported == reference_status.supported);
                if (!rc) CHECK(equal);
                else CHECK(error.native_status == reference_error.native_status);
                if (selection == 0) {
                    const unsigned raises[] = {ME_FP_INVALID,ME_FP_DIVIDE,ME_FP_OVERFLOW,ME_FP_UNDERFLOW};
                    for (size_t flag = 0; flag < sizeof(raises)/sizeof(raises[0]); flag++) {
                        rc = me_artifact_eval_status(jit,buffers,2,actual,&descriptor,raises[flag],&status,&error);
                        reference_rc = me_artifact_eval_status(reference,buffers,2,expected,&descriptor,
                            raises[flag],&reference_status,&reference_error);
                        CHECK(rc == reference_rc && status.flags == reference_status.flags);
                        if (rc) CHECK(error.native_status == reference_error.native_status);
                    }
                }
            }
            me_artifact_free(jit);
            me_artifact_free(reference);
        }
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

static void arithmetic_bridge_regressions(void) {
    parity_case basic = {"arithmetic bridge admission","return x + (y - 1)",
        "float64","float64",true};
    me_artifact *enabled = load(&basic,ME_JIT_ON);
    if (!me_artifact_has_jit(enabled)) {
        me_artifact_free(enabled);
        return;
    }
    me_dsl_compiled_program *enabled_program = dsl_artifact_program_for_tests(enabled);
    if (enabled_program && enabled_program->compiler != ME_DSL_COMPILER_LIBTCC) {
        me_artifact_free(enabled);
        return;
    }
    CHECK(enabled_program && enabled_program->portable_jit_has_arith_bridge);
    CHECK(enabled_program->portable_jit_arith_nnodes == 5);
    CHECK(enabled_program->portable_jit_arith_ntemps == 1);
    uint64_t enabled_key = dsl_jit_runtime_cache_key(enabled_program);

    parity_case conversion = {"arithmetic bridge conversion rejection","return float64(x)",
        "float64","float64",true};
    me_artifact *converted = load(&conversion,ME_JIT_ON);
    me_dsl_compiled_program *converted_program = dsl_artifact_program_for_tests(converted);
    CHECK(converted_program && !converted_program->portable_jit_has_arith_bridge);
    me_artifact_free(converted);

    const char *old_toggle = getenv("ME_DSL_JIT_ARITH_BRIDGE");
    char *saved_toggle = old_toggle ? strdup(old_toggle) : NULL;
#ifdef _WIN32
    _putenv_s("ME_DSL_JIT_ARITH_BRIDGE","0");
#else
    setenv("ME_DSL_JIT_ARITH_BRIDGE","0",1);
#endif
    me_artifact *disabled = load(&basic,ME_JIT_ON);
    me_dsl_compiled_program *disabled_program = dsl_artifact_program_for_tests(disabled);
    CHECK(disabled_program && !disabled_program->portable_jit_has_arith_bridge);
    uint64_t disabled_key = dsl_jit_runtime_cache_key(disabled_program);
    CHECK(enabled_key && disabled_key && enabled_key != disabled_key);
#ifdef _WIN32
    _putenv_s("ME_DSL_JIT_ARITH_BRIDGE",saved_toggle ? saved_toggle : "");
#else
    if (saved_toggle) setenv("ME_DSL_JIT_ARITH_BRIDGE",saved_toggle,1);
    else unsetenv("ME_DSL_JIT_ARITH_BRIDGE");
#endif
    free(saved_toggle);
    me_artifact_free(disabled);

    parity_case four_temps = {"arithmetic bridge four temporaries",
        "return (x + y) * (x - y) + (x * y + x / y)","float64","float64",true};
    me_artifact *four = load(&four_temps,ME_JIT_ON);
    me_dsl_compiled_program *four_program = dsl_artifact_program_for_tests(four);
    CHECK(four_program && four_program->portable_jit_has_arith_bridge &&
        four_program->portable_jit_arith_ntemps == 4);
    me_artifact_free(four);
    parity_case over_temps = {"arithmetic bridge temporary rejection",
        "return ((x + y) * (x - y) + (x * y + x / y)) * (((-x) + y) * ((-x) - y) + ((-x) * y + (-x) / y))",
        "float64","float64",true};
    me_artifact *over = load(&over_temps,ME_JIT_ON);
    me_dsl_compiled_program *over_program = dsl_artifact_program_for_tests(over);
    CHECK(over_program && !over_program->portable_jit_has_arith_bridge);
    me_artifact_free(over);

    /* Uniform signed arithmetic must use unsigned carriers, including overflow. */
    parity_case uniform = {"uniform modular arithmetic","return x + (z + z)",
        "int64","int64",true};
    const char *second = "{\"name\":\"y\",\"dtype\":\"int64\"}";
    const char *constant = "{\"name\":\"z\",\"dtype\":\"int64\",\"category\":\"typed_scalar\","
        "\"encoding\":\"decimal\",\"value\":\"9223372036854775807\"}";
    me_artifact *uniform_jit = load_contract(&uniform,ME_JIT_ON,second,constant,"elementwise");
    me_artifact *uniform_ref = load_contract(&uniform,ME_JIT_OFF,second,constant,"elementwise");
    me_dsl_compiled_program *uniform_program = dsl_artifact_program_for_tests(uniform_jit);
    CHECK(uniform_program && uniform_program->portable_jit_has_arith_bridge);
    CHECK(strstr(uniform_program->jit_c_source,
        "pj_int64_t((uint64_t)(pj_input_2[0]) + (uint64_t)(pj_input_2[0]))"));
    int64_t ix[] = {0,1,INT64_MIN,INT64_MAX}, iy[] = {0,0,0,0};
    int64_t actual_i[4], expected_i[4];
    me_artifact_buffer ibuffers[] = {{"x",ME_INT64,8,ix,sizeof(ix)},
        {"y",ME_INT64,8,iy,sizeof(iy)}};
    me_artifact_eval_descriptor idesc = {.struct_size=sizeof(idesc),.version=1,
        .nitems=4,.output_capacity=sizeof(actual_i)};
    me_artifact_error error, ref_error;
    CHECK(!me_artifact_eval_ex(uniform_jit,ibuffers,2,actual_i,&idesc,&error));
    CHECK(!me_artifact_eval_ex(uniform_ref,ibuffers,2,expected_i,&idesc,&ref_error));
    CHECK(!memcmp(actual_i,expected_i,sizeof(actual_i)) && actual_i[0] == -2);
    me_artifact_free(uniform_jit); me_artifact_free(uniform_ref);

    /* Audit both missing and spurious exceptions around strip boundaries. */
    parity_case division = {"arithmetic bridge tail status","return x / y",
        "float64","float64",true};
    me_artifact *division_jit = load(&division,ME_JIT_ON);
    me_artifact *division_ref = load(&division,ME_JIT_OFF);
    me_dsl_compiled_program *division_program = dsl_artifact_program_for_tests(division_jit);
    CHECK(division_program && division_program->portable_jit_has_arith_bridge);
    const int lengths[] = {1,3,4,5,511,512,513};
    double x[514], y[514], actual[513], expected[513];
    for (int i=0; i<514; i++) { x[i]=1.0; y[i]=1.0; }
    for (size_t li=0; li<sizeof(lengths)/sizeof(lengths[0]); li++) {
        int n=lengths[li];
        me_artifact_buffer buffers[] = {{"x",ME_FLOAT64,8,x,(size_t)(n+1)*8},
            {"y",ME_FLOAT64,8,y,(size_t)(n+1)*8}};
        me_artifact_eval_descriptor desc = {.struct_size=sizeof(desc),.version=1,
            .nitems=n,.output_capacity=(size_t)n*8};
        me_artifact_fp_status status, reference_status;
        y[n]=0.0; /* Exceptional padding must not participate. */
        CHECK(!me_artifact_eval_status(division_jit,buffers,2,actual,&desc,0,&status,&error));
        CHECK(status.flags == 0);
        y[n-1]=0.0; /* The final active lane must participate. */
        CHECK(!me_artifact_eval_status(division_jit,buffers,2,actual,&desc,0,&status,&error));
        CHECK(!me_artifact_eval_status(division_ref,buffers,2,expected,&desc,0,
            &reference_status,&ref_error));
        CHECK(status.flags == reference_status.flags && (status.flags & ME_FP_DIVIDE));
        y[n-1]=1.0; y[n]=1.0;
        /* Exact-size allocations let ASan detect a physical tail over-read. */
        double *exact_x=malloc((size_t)n*8), *exact_y=malloc((size_t)n*8);
        CHECK(exact_x && exact_y);
        for (int i=0; i<n; i++) { exact_x[i]=1.0; exact_y[i]=1.0; }
        me_artifact_buffer exact_buffers[]={{"x",ME_FLOAT64,8,exact_x,(size_t)n*8},
            {"y",ME_FLOAT64,8,exact_y,(size_t)n*8}};
        CHECK(!me_artifact_eval_status(division_jit,exact_buffers,2,actual,&desc,0,
            &status,&error) && status.flags == 0);
        free(exact_x); free(exact_y);
    }
    me_artifact_free(division_jit); me_artifact_free(division_ref);

    /* Copy and unary negation preserve signaling NaNs without FE_INVALID. */
    const char *snan_bodies[] = {"return x","return -x"};
    for (int operation=0; operation<2; operation++) {
        parity_case snan = {"arithmetic bridge signaling NaN",snan_bodies[operation],
            "float64","float64",true};
        me_artifact *jit=load(&snan,ME_JIT_ON), *reference=load(&snan,ME_JIT_OFF);
        uint64_t input_bits[513], actual_bits[513], expected_bits[513];
        for (int i=0; i<513; i++) input_bits[i]=UINT64_C(0x7ff0000000000001);
        me_artifact_buffer buffers[]={{"x",ME_FLOAT64,8,input_bits,sizeof(input_bits)},
            {"y",ME_FLOAT64,8,input_bits,sizeof(input_bits)}};
        me_artifact_eval_descriptor desc={.struct_size=sizeof(desc),.version=1,
            .nitems=513,.output_capacity=sizeof(actual_bits)};
        me_artifact_fp_status status, reference_status;
        CHECK(!me_artifact_eval_status(jit,buffers,2,actual_bits,&desc,0,&status,&error));
        CHECK(!me_artifact_eval_status(reference,buffers,2,expected_bits,&desc,0,
            &reference_status,&ref_error));
        CHECK(status.flags == reference_status.flags && !(status.flags & ME_FP_INVALID));
        CHECK(!memcmp(actual_bits,expected_bits,sizeof(actual_bits)));
        me_artifact_free(jit); me_artifact_free(reference);
    }

    const char *fill_json = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\","
        "\"version\":\"1.0\"},\"requires\":[\"numeric\"],\"source\":\"def k(c):\\n    return c\\n\","
        "\"entry_point\":\"k\",\"inputs\":[],\"constants\":[{\"name\":\"c\",\"dtype\":\"float64\","
        "\"encoding\":\"ieee754-hex\",\"value\":\"7ff0000000000001\",\"category\":\"typed_scalar\"}],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"elementwise\"},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"context\":{\"ndim\":0}}";
    me_artifact *fill_jit=NULL, *fill_ref=NULL;
    CHECK(!me_artifact_load(fill_json,strlen(fill_json),ME_JIT_ON,&fill_jit,&error));
    CHECK(!me_artifact_load(fill_json,strlen(fill_json),ME_JIT_OFF,&fill_ref,&ref_error));
    uint64_t fill_actual[513], fill_expected[513];
    me_artifact_eval_descriptor fill_desc={.struct_size=sizeof(fill_desc),.version=1,
        .nitems=513,.output_capacity=sizeof(fill_actual)};
    me_artifact_fp_status fill_status, fill_reference_status;
    CHECK(!me_artifact_eval_status(fill_jit,NULL,0,fill_actual,&fill_desc,0,&fill_status,&error));
    CHECK(!me_artifact_eval_status(fill_ref,NULL,0,fill_expected,&fill_desc,0,
        &fill_reference_status,&ref_error));
    CHECK(fill_status.flags == fill_reference_status.flags && !(fill_status.flags & ME_FP_INVALID));
    CHECK(!memcmp(fill_actual,fill_expected,sizeof(fill_actual)));
    me_artifact_free(fill_jit); me_artifact_free(fill_ref);
    me_artifact_free(enabled);
}

static void flat_index_1d_lowering(void) {
    const char *json =
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"block-reductions\",\"nd-context\"],"
        "\"source\":\"def ramp():\\n    return 1.5 + _flat_idx * 0.25\\n\","
        "\"entry_point\":\"ramp\",\"inputs\":[],\"constants\":[],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":1},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}";
    me_artifact *artifact = NULL;
    me_artifact_error error;
    CHECK(!me_artifact_load(json, strlen(json), ME_JIT_ON, &artifact, &error));
    CHECK(artifact);
    if (!me_artifact_has_jit(artifact)) {
        me_artifact_free(artifact);
        return;
    }
    me_dsl_compiled_program *program = dsl_artifact_program_for_tests(artifact);
    CHECK(program && program->portable_jit_has_ramp_bridge);
    CHECK(strstr(program->jit_c_source, "pj_t->ramp_f64"));
    CHECK(!strstr(program->jit_c_source, "jit_off / jit_ext"));
    me_artifact *reference = NULL;
    CHECK(!me_artifact_load(json, strlen(json), ME_JIT_OFF, &reference, &error));
    double actual[7], expected[7];
    const int64_t shape[] = {20};
    const int64_t origins[] = {0, 5, 16};
    const int64_t extents[] = {7, 6, 4};
    const int counts[] = {7, 6, 4};
    for (int block=0; block<3; block++) {
        me_artifact_eval_descriptor descriptor = {
            .struct_size=sizeof(descriptor), .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
            .nitems=(size_t)counts[block], .output_capacity=(size_t)counts[block]*sizeof(double),
            .ndim=1, .logical_shape=shape, .block_origin=&origins[block],
            .block_extent=&extents[block]
        };
        CHECK(!me_artifact_eval_ex(artifact, NULL, 0, actual, &descriptor, &error));
        CHECK(!me_artifact_eval_ex(reference, NULL, 0, expected, &descriptor, &error));
        CHECK(!memcmp(actual, expected, (size_t)counts[block]*sizeof(double)));
        for (int i=0; i<counts[block]; i++) {
            CHECK(actual[i] == 1.5 + (double)(origins[block]+i) * 0.25);
        }
    }
    me_artifact_free(reference);
    me_artifact_free(artifact);

    /* A float32 output cast must not narrow float64 ramp intermediates. */
    const char *mixed_json =
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"nd-context\"],"
        "\"source\":\"def ramp():\\n    return 0.0 + _flat_idx * 0.7777777777777778\\n\","
        "\"entry_point\":\"ramp\",\"inputs\":[],\"constants\":[],"
        "\"output\":{\"dtype\":\"float32\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":1},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}";
    artifact = NULL;
    reference = NULL;
    CHECK(!me_artifact_load(mixed_json, strlen(mixed_json), ME_JIT_ON, &artifact, &error));
    CHECK(!me_artifact_load(mixed_json, strlen(mixed_json), ME_JIT_OFF, &reference, &error));
    float mixed_actual[10], mixed_expected[10];
    const int64_t mixed_shape = 10, mixed_origin = 0, mixed_extent = 10;
    me_artifact_eval_descriptor mixed_desc = {
        .struct_size=sizeof(mixed_desc), .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems=10, .output_capacity=sizeof(mixed_actual), .ndim=1,
        .logical_shape=&mixed_shape, .block_origin=&mixed_origin, .block_extent=&mixed_extent
    };
    CHECK(!me_artifact_eval_ex(artifact, NULL, 0, mixed_actual, &mixed_desc, &error));
    CHECK(!me_artifact_eval_ex(reference, NULL, 0, mixed_expected, &mixed_desc, &error));
    CHECK(!memcmp(mixed_actual, mixed_expected, sizeof(mixed_actual)));
    me_artifact_free(reference);
    me_artifact_free(artifact);

    char integer_json[2048];
    const int64_t start = INT64_MAX-3, step = 3;
    int json_size = snprintf(integer_json, sizeof(integer_json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"block-reductions\",\"nd-context\"],"
        "\"source\":\"def ramp(start, step):\\n    return start + _flat_idx * step\\n\","
        "\"entry_point\":\"ramp\",\"inputs\":[],\"constants\":["
        "{\"name\":\"start\",\"dtype\":\"int64\",\"category\":\"typed_scalar\",\"encoding\":\"decimal\",\"value\":\"%lld\"},"
        "{\"name\":\"step\",\"dtype\":\"int64\",\"category\":\"typed_scalar\",\"encoding\":\"decimal\",\"value\":\"%lld\"}],"
        "\"output\":{\"dtype\":\"int64\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":1},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        (long long)start, (long long)step);
    CHECK(json_size > 0 && (size_t)json_size < sizeof(integer_json));
    artifact = NULL;
    CHECK(!me_artifact_load(integer_json, (size_t)json_size, ME_JIT_ON, &artifact, &error));
    CHECK(artifact && me_artifact_has_jit(artifact));
    program = dsl_artifact_program_for_tests(artifact);
    CHECK(program && program->portable_jit_has_ramp_bridge);
    CHECK(strstr(program->jit_c_source, "pj_t->ramp_i64"));
    reference = NULL;
    CHECK(!me_artifact_load(integer_json, (size_t)json_size, ME_JIT_OFF, &reference, &error));
    int64_t iactual[4], iexpected[4];
    const int64_t iorigin = 5, iextent = 4, ishape = 20;
    me_artifact_eval_descriptor idesc = {
        .struct_size=sizeof(idesc), .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems=4, .output_capacity=sizeof(iactual), .ndim=1,
        .logical_shape=&ishape, .block_origin=&iorigin, .block_extent=&iextent
    };
    CHECK(!me_artifact_eval_ex(artifact, NULL, 0, iactual, &idesc, &error));
    CHECK(!me_artifact_eval_ex(reference, NULL, 0, iexpected, &idesc, &error));
    CHECK(!memcmp(iactual, iexpected, sizeof(iactual)));
    for (int i=0; i<4; i++) {
        uint64_t bits = (uint64_t)start + (uint64_t)(iorigin+i)*(uint64_t)step;
        int64_t expected;
        memcpy(&expected, &bits, sizeof(expected));
        CHECK(iactual[i] == expected);
    }
    me_artifact_free(reference);
    me_artifact_free(artifact);

    /* 2D float64 ramp */
    const char *nd_json =
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"block-reductions\",\"nd-context\"],"
        "\"source\":\"def ramp():\\n    return 1.5 + _flat_idx * 0.25\\n\","
        "\"entry_point\":\"ramp\",\"inputs\":[],\"constants\":[],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":2},\"semantics\":{\"fp\":\"strict\","
        "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}";
    artifact = NULL;
    CHECK(!me_artifact_load(nd_json, strlen(nd_json), ME_JIT_ON, &artifact, &error));
    CHECK(artifact && me_artifact_has_jit(artifact));
    program = dsl_artifact_program_for_tests(artifact);
    CHECK(program && program->portable_jit_has_ramp_bridge);
    CHECK(strstr(program->jit_c_source, "pj_t->ramp_f64"));
    reference = NULL;
    CHECK(!me_artifact_load(nd_json, strlen(nd_json), ME_JIT_OFF, &reference, &error));
    double nd_actual[12], nd_expected[12];
    const int64_t nd_shape[] = {4, 3};
    const int64_t nd_origins[] = {1, 0};
    const int64_t nd_extents[] = {2, 3};
    me_artifact_eval_descriptor nd_desc = {
        .struct_size=sizeof(nd_desc), .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems=6, .output_capacity=sizeof(nd_actual), .ndim=2,
        .logical_shape=nd_shape, .block_origin=nd_origins, .block_extent=nd_extents
    };
    CHECK(!me_artifact_eval_ex(artifact, NULL, 0, nd_actual, &nd_desc, &error));
    CHECK(!me_artifact_eval_ex(reference, NULL, 0, nd_expected, &nd_desc, &error));
    CHECK(!memcmp(nd_actual, nd_expected, 6 * sizeof(double)));
    for (int i=0; i<6; i++) {
        CHECK(nd_actual[i] == 1.5 + (double)(3 + i) * 0.25);
    }

    /* 2D non-contiguous tile */
    const int64_t tile_origins[] = {1, 1};
    const int64_t tile_extents[] = {2, 2};
    double tile_actual[4], tile_expected[4];
    me_artifact_eval_descriptor tile_desc = {
        .struct_size=sizeof(tile_desc), .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems=4, .output_capacity=sizeof(tile_actual), .ndim=2,
        .logical_shape=nd_shape, .block_origin=tile_origins, .block_extent=tile_extents
    };
    CHECK(!me_artifact_eval_ex(artifact, NULL, 0, tile_actual, &tile_desc, &error));
    CHECK(!me_artifact_eval_ex(reference, NULL, 0, tile_expected, &tile_desc, &error));
    CHECK(!memcmp(tile_actual, tile_expected, 4 * sizeof(double)));
    CHECK(tile_actual[0] == 1.5 + 4 * 0.25);
    CHECK(tile_actual[1] == 1.5 + 5 * 0.25);
    CHECK(tile_actual[2] == 1.5 + 7 * 0.25);
    CHECK(tile_actual[3] == 1.5 + 8 * 0.25);
    me_artifact_free(reference);
    me_artifact_free(artifact);

    /* Non-contiguous higher-rank tiles must use row-wise ramp synthesis too. */
    for (int rank = 3; rank <= 4; rank++) {
        char json[2048];
        const char *rank_field = strstr(nd_json, "\"ndim\":2");
        CHECK(rank_field != NULL);
        int prefix = (int)(rank_field - nd_json);
        snprintf(json, sizeof(json), "%.*s\"ndim\":%d%s", prefix, nd_json, rank,
                 rank_field + strlen("\"ndim\":2"));
        artifact = NULL;
        reference = NULL;
        CHECK(!me_artifact_load(json, strlen(json), ME_JIT_ON, &artifact, &error));
        CHECK(!me_artifact_load(json, strlen(json), ME_JIT_OFF, &reference, &error));
        program = dsl_artifact_program_for_tests(artifact);
        CHECK(program && program->portable_jit_has_ramp_bridge);
        const int64_t shape[] = {5, 6, 7, 8};
        const int64_t origin[] = {1, 2, 3, 4};
        const int64_t extent[] = {2, 2, 2, 2};
        double actual[16], expected[16];
        me_artifact_eval_descriptor desc = {
            .struct_size=sizeof(desc), .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
            .nitems=(size_t)1 << rank, .output_capacity=sizeof(actual), .ndim=rank,
            .logical_shape=shape, .block_origin=origin, .block_extent=extent
        };
        CHECK(!me_artifact_eval_ex(artifact, NULL, 0, actual, &desc, &error));
        CHECK(!me_artifact_eval_ex(reference, NULL, 0, expected, &desc, &error));
        CHECK(!memcmp(actual, expected, desc.nitems * sizeof(double)));
        me_artifact_free(reference);
        me_artifact_free(artifact);
    }
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
    runtime_literal_parity();
    resolved_libm_parity();
    integer_tree_parity();
    arithmetic_bridge_regressions();
    flat_index_1d_lowering();
    scalar_array_fallbacks();
    scalar_loop_cap();
    integer_scalar_reductions();
    numeric_dtype_matrix();
    integer_extrema_lowering();
    batch_interpreter_parity();
    batch_complex_parity();
    batch_complex_cast_parity();
    batch_complex_aliases();
    batch_integer_parity();
    batch_checked_and_alias_fallbacks();
    batch_cast_parity();
    range_bound_matrix();
    weak_capture_matrix();
    captured_integer_trees();
    return 0;
}
