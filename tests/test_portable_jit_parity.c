#include "miniexpr_artifact.h"
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
    {"range", "s = 0.0\\n    for i in range(10):\\n        s = s + x\\n    return s", "float64", "float64", false},
    {"while", "s = 0.0\\n    i = 0\\n    while i < 10:\\n        s = s + x\\n        i = i + 1\\n    return s", "float64", "float64", false},
    {"break", "s = 0.0\\n    for i in range(10):\\n        if s > y:\\n            break\\n        s = s + x\\n    return s", "float64", "float64", false},
    {"continue", "s = 0.0\\n    for i in range(10):\\n        if i < 3:\\n            continue\\n        s = s + x\\n    return s", "float64", "float64", false},
    {"loop return", "for i in range(10):\\n        if x > 0:\\n            return x\\n    return y", "float64", "float64", false},
    {"nested loops", "s = 0.0\\n    for i in range(3):\\n        for j in range(4):\\n            s = s + x\\n    return s", "float64", "float64", false},
    {"checked cast", "return int(x)", "float64", "int64", false},
    {"integer abs", "return abs(x)", "int32", "int32", false},
    {"integer sign", "return sign(x)", "int32", "int32", false},
    {"integer square", "return square(x)", "int32", "int32", false},
    {"integer floor", "return floor(x)", "int32", "int32", false},
    {"integer round", "return round(x)", "int32", "int32", false},
    {"integer real", "return real(x)", "int32", "int32", false},
    {"integer imag", "return imag(x)", "int32", "int32", false},
    {"integer conj", "return conj(x)", "int32", "int32", false},
    {"ldexp", "return ldexp(x, 2)", "float64", "float64", false},
    {"fma", "return fma(x, y, 1.0)", "float64", "float64", false},
    {"floating math", "return sin(x) + cos(y)", "float64", "float64", true},
};

static me_artifact *load(const parity_case *test, me_jit_mode mode) {
    char json[4096];
    int length = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\",\"control-flow\"],\"source\":\"def k(x, y):\\n    %s\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"},{\"name\":\"y\",\"dtype\":\"%s\"}],"
        "\"constants\":[],\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        test->body, test->input, test->input, test->output);
    CHECK(length > 0 && (size_t)length < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error;
    int rc = me_artifact_load(json, (size_t)length, mode, &artifact, &error);
    if (rc) fprintf(stderr, "%s: %s\n", test->name, error.message);
    CHECK(!rc && artifact);
    return artifact;
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
        CHECK(!me_artifact_eval_status(jit, buffers, 2, result, &descriptor, 0, &status, &error));
        CHECK(!me_artifact_eval_status(reference, buffers, 2, expected, &descriptor, 0,
            &reference_status, &error));
        CHECK(!memcmp(result, expected, sizeof(result)));
        CHECK(status.flags == reference_status.flags);
        fprintf(stderr, "%-20s route=%s\n", test->name,
            me_artifact_has_jit(jit) ? "jit" : "interpreter");
        me_artifact_free(jit);
        me_artifact_free(reference);
    }
}

int main(void) {
    matrix();
    return 0;
}
