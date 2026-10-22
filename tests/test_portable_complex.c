/* Interpreter-first complex contract for Menudet 1.0, including Windows. */
#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "miniexpr.h"
#include "miniexpr_artifact.h"
#include "complex_compat.h"

#define ITEMS 4
#define CLOSE(a, b) (((a) == (b)) || cabs((a) - (b)) <= 1e-12 + 1e-12 * cabs(b))

static int failures;

static void check_close(const char *label, double complex actual, double complex expected) {
    if (!CLOSE(actual, expected)) {
        fprintf(stderr, "%s: actual=(%.17g,%.17g) expected=(%.17g,%.17g)\n",
                label, creal(actual), cimag(actual), creal(expected), cimag(expected));
        failures++;
    }
}

static me_artifact *load(const char *source, const char *inputs, const char *constants,
                         const char *out_dtype, const char *cardinality, int ndim,
                         me_jit_mode jit) {
    static char escaped[8192];
    size_t out = 0;
    for (const char *p = source; *p && out + 2 < sizeof(escaped); p++) {
        if (*p == '\n') { escaped[out++] = '\\'; escaped[out++] = 'n'; }
        else if (*p == '\"') { escaped[out++] = '\\'; escaped[out++] = '\"'; }
        else if (*p == '\\') { escaped[out++] = '\\'; escaped[out++] = '\\'; }
        else escaped[out++] = *p;
    }
    escaped[out] = '\0';
    static char json[16384];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"block-reductions\",\"nd-context\"],"
        "\"source\":\"%s\",\"entry_point\":\"k\",\"inputs\":[%s],\"constants\":%s,"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"%s\"},"
        "\"context\":{\"ndim\":%d},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        escaped, inputs, constants, out_dtype, cardinality, ndim);
    assert(n > 0 && (size_t)n < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error;
    int rc = me_artifact_load(json, (size_t)n, jit, &artifact, &error);
    if (rc) {
        fprintf(stderr, "load failed (%d): %s\nsource=%s\n", rc, error.message, source);
        failures++;
        return NULL;
    }
    return artifact;
}


static int eval_with(me_artifact *artifact, const me_artifact_buffer *input, int ninputs,
                     void *output, size_t capacity, size_t count) {
    int64_t shape = (int64_t)count, origin = 0, extent = (int64_t)count;
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems = count, .output_capacity = capacity, .ndim = 1,
        .logical_shape = &shape, .block_origin = &origin, .block_extent = &extent
    };
    me_artifact_error error;
    int rc = me_artifact_eval_ex(artifact, input, ninputs, output, &descriptor, &error);
    if (rc) fprintf(stderr, "eval failed (%d): %s\n", rc, error.message);
    return rc;
}
static void eval_elementwise(me_artifact *artifact, const double complex *z, size_t count,
                             void *output) {
    me_artifact_buffer input = {"z", ME_COMPLEX128, sizeof(double complex), z, count * sizeof(double complex)};
    if (eval_with(artifact, &input, 1, output, count * sizeof(double complex), count)) failures++;
}

static void check_rejected(const char *source, const char *out_dtype) {
    char escaped[2048];
    size_t used = 0;
    for (const char *p = source; *p && used + 2 < sizeof(escaped); p++) {
        if (*p == '\n') { escaped[used++] = '\\'; escaped[used++] = 'n'; }
        else escaped[used++] = *p;
    }
    escaped[used] = '\0';
    char json[4096];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"block-reductions\",\"nd-context\"],"
        "\"source\":\"%s\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"z\",\"dtype\":\"complex128\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},"
        "\"context\":{\"ndim\":1},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        escaped, out_dtype);
    assert(n > 0 && (size_t)n < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error;
    if (!me_artifact_load(json, (size_t)n, ME_JIT_OFF, &artifact, &error)) {
        fprintf(stderr, "expected rejection: %s\n", source);
        failures++;
        me_artifact_free(artifact);
    }
}

static void test_validation(void) {
    me_portable_error error;
    me_variable z = {.name = "z", .dtype = ME_COMPLEX128, .type = ME_VARIABLE};
    assert(me_validate_portable_dsl("def k(z):\n    return z * z + 1\n", "1.0", &z, 1,
                                    ME_COMPLEX128, &error) == ME_PORTABLE_SUCCESS);
    assert(me_validate_portable_dsl("def k(z):\n    return abs(z)\n", "1.0", &z, 1,
                                    ME_FLOAT64, &error) == ME_PORTABLE_SUCCESS);
    assert(me_validate_portable_dsl("def k(z):\n    return conj(z) + real(z) + imag(z)\n",
                                    "1.0", &z, 1, ME_COMPLEX128, &error) == ME_PORTABLE_SUCCESS);
    assert(me_validate_portable_dsl("def k(z):\n    return z < z\n", "1.0", &z, 1,
                                    ME_BOOL, &error) == ME_PORTABLE_SUCCESS);
    assert(me_validate_portable_dsl("def k(z):\n    return block_sum(z)\n", "1.0", &z, 1,
                                    ME_COMPLEX128, &error) == ME_PORTABLE_SUCCESS);
    const char *invalid[] = {
        "def k(z):\n    return z // 2\n",
        "def k(z):\n    return z % 2\n",
        "def k(z):\n    return z & 1\n",
        "def k(z):\n    return z << 1\n",
        "def k(z):\n    return ceil(z)\n",
        "def k(z):\n    return float(z)\n",
        "def k(z):\n    return signbit(z)\n",
        "def k(z):\n    return logaddexp(z, z)\n"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        assert(me_validate_portable_dsl(invalid[i], "1.0", &z, 1, ME_COMPLEX128,
                                        &error) == ME_PORTABLE_ERR_SOURCE);
        assert(error.message[0]);
    }
}

static void test_arithmetic_and_math(void) {
    const double complex z[ITEMS] = {1.0 + 2.0 * I, 3.0 - 4.0 * I,
                                     -2.0 + 0.5 * I, 0.0 + 0.0 * I};
    const char *inputs = "{\"name\":\"z\",\"dtype\":\"complex128\"}";
    struct { const char *source; } cases[] = {
        {"def k(z):\n    return z\n"},
        {"def k(z):\n    return -z\n"},
        {"def k(z):\n    return z * z + 1\n"},
        {"def k(z):\n    return (z + 1) / (z - 1)\n"},
        {"def k(z):\n    return conj(z)\n"},
        {"def k(z):\n    return z ** 2\n"},
        {"def k(z):\n    return sqrt(z)\n"},
        {"def k(z):\n    return exp(z)\n"},
        {"def k(z):\n    return log(z)\n"},
        {"def k(z):\n    return sin(z) * sin(z) + cos(z) * cos(z)\n"}
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        for (int jit = 0; jit < 2; jit++) {
            me_artifact *artifact = load(cases[c].source, inputs, "[]", "complex128",
                                         "elementwise", 1, jit ? ME_JIT_ON : ME_JIT_OFF);
            if (!artifact) continue;
            if (jit) assert(!me_artifact_has_jit(artifact));
            double complex out[ITEMS] = {0};
            eval_elementwise(artifact, z, ITEMS, out);
            double complex expected[ITEMS];
            for (int i = 0; i < ITEMS; i++) {
                double complex a = z[i];
                switch (c) {
                    case 0: expected[i] = a; break;
                    case 1: expected[i] = -a; break;
                    case 2: expected[i] = a * a + 1.0; break;
                    case 3: expected[i] = (a + 1.0) / (a - 1.0); break;
                    case 4: expected[i] = conj(a); break;
                    case 5: expected[i] = a * a; break;
                    case 6: expected[i] = csqrt(a); break;
                    case 7: expected[i] = cexp(a); break;
                    case 8: expected[i] = clog(a); break;
                    default: expected[i] = 1.0; break;
                }
            }
            for (int i = 0; i < ITEMS; i++) check_close(cases[c].source, out[i], expected[i]);
            me_artifact_free(artifact);
        }
    }
}

static void test_real_results_and_control_flow(void) {
    const double complex z[ITEMS] = {1.0 + 2.0 * I, 3.0 - 4.0 * I,
                                     -2.0 + 0.5 * I, 0.0 + 0.0 * I};
    const char *inputs = "{\"name\":\"z\",\"dtype\":\"complex128\"}";

    me_artifact *artifact = load("def k(z):\n    return abs(z)\n", inputs, "[]", "float64",
                                 "elementwise", 1, ME_JIT_OFF);
    if (artifact) {
        double out[ITEMS] = {0};
        me_artifact_buffer input = {"z", ME_COMPLEX128, sizeof(double complex), z, ITEMS * sizeof(double complex)};
        assert(!eval_with(artifact, &input, 1, out, sizeof(out), ITEMS));
        for (int i = 0; i < ITEMS; i++) assert(fabs(out[i] - cabs(z[i])) <= 1e-12);
        me_artifact_free(artifact);
    }

    artifact = load("def k(z):\n    v = z\n    for i in range(2):\n        v = v * z\n"
                    "    if imag(v) < 0:\n        return v + 1\n    return v\n",
                    inputs, "[]", "complex128", "elementwise", 1, ME_JIT_OFF);
    if (artifact) {
        double complex out[ITEMS] = {0};
        eval_elementwise(artifact, z, ITEMS, out);
        for (int i = 0; i < ITEMS; i++) {
            double complex v = z[i] * z[i] * z[i];
            check_close("loop", out[i], cimag(v) < 0 ? v + 1.0 : v);
        }
        me_artifact_free(artifact);
    }
}

static void test_comparisons_and_predicates(void) {
    const double complex z[ITEMS] = {1.0 + 2.0 * I, 3.0 - 4.0 * I,
                                     -2.0 + 0.5 * I, 0.0 + 0.0 * I};
    const char *inputs = "{\"name\":\"z\",\"dtype\":\"complex128\"}";
    me_artifact *artifact = load("def k(z):\n    return z == z\n", inputs, "[]", "bool",
                                 "elementwise", 1, ME_JIT_OFF);
    if (artifact) {
        bool out[ITEMS] = {0};
        me_artifact_buffer input = {"z", ME_COMPLEX128, sizeof(double complex), z, ITEMS * sizeof(double complex)};
        assert(!eval_with(artifact, &input, 1, out, sizeof(out), ITEMS));
        for (int i = 0; i < ITEMS; i++) assert(out[i]);
        me_artifact_free(artifact);
    }
    artifact = load("def k(z):\n    return where(z, 1.0, 2.0)\n", inputs, "[]", "float64",
                    "elementwise", 1, ME_JIT_OFF);
    if (artifact) {
        double out[ITEMS] = {0};
        me_artifact_buffer input = {"z", ME_COMPLEX128, sizeof(double complex), z, ITEMS * sizeof(double complex)};
        assert(!eval_with(artifact, &input, 1, out, sizeof(out), ITEMS));
        for (int i = 0; i < ITEMS; i++) assert(out[i] == (z[i] != 0.0 ? 1.0 : 2.0));
        me_artifact_free(artifact);
    }
}

static void test_constants(void) {
    /* complex128 2+3j: real 4000000000000000, imag 4008000000000000 */
    const char *constants =
        "[{\"name\":\"c\",\"dtype\":\"complex128\",\"category\":\"typed_scalar\","
        "\"encoding\":\"ieee754-hex\",\"value\":\"40000000000000004008000000000000\"}]";
    me_artifact *artifact = load("def k(z, c):\n    return z + c\n",
                                 "{\"name\":\"z\",\"dtype\":\"complex128\"}", constants,
                                 "complex128", "elementwise", 1, ME_JIT_OFF);
    if (!artifact) return;
    const double complex z[ITEMS] = {1.0 + 2.0 * I, -1.0, 0.0, 4.0 - 1.0 * I};
    double complex out[ITEMS] = {0};
    eval_elementwise(artifact, z, ITEMS, out);
    for (int i = 0; i < ITEMS; i++) check_close("constant", out[i], z[i] + (2.0 + 3.0 * I));
    me_artifact_free(artifact);
}

static void test_reductions(void) {
    const double complex z[ITEMS] = {1.0 + 2.0 * I, 3.0 - 4.0 * I,
                                     -2.0 + 0.5 * I, 0.0 + 0.0 * I};
    const char *inputs = "{\"name\":\"z\",\"dtype\":\"complex128\"}";
    struct { const char *source; } cases[] = {
        {"def k(z):\n    return block_sum(z)\n"},
        {"def k(z):\n    return block_mean(z)\n"},
        {"def k(z):\n    return block_min(z)\n"},
        {"def k(z):\n    return block_max(z)\n"}
    };
    double complex expected[] = {
        2.0 - 1.5 * I, 0.5 - 0.375 * I, -2.0 + 0.5 * I, 3.0 - 4.0 * I
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        me_artifact *artifact = load(cases[c].source, inputs, "[]", "complex128",
                                     "block_scalar", 1, ME_JIT_OFF);
        if (!artifact) continue;
        double complex out = 0;
        me_artifact_buffer input = {"z", ME_COMPLEX128, sizeof(double complex), z, ITEMS * sizeof(double complex)};
        assert(!eval_with(artifact, &input, 1, &out, sizeof(out), ITEMS));
        check_close(cases[c].source, out, expected[c]);
        me_artifact_free(artifact);
    }
    me_artifact *artifact = load("def k(z):\n    return block_any(z)\n", inputs, "[]", "bool",
                                 "block_scalar", 1, ME_JIT_OFF);
    if (artifact) {
        bool out = false;
        me_artifact_buffer input = {"z", ME_COMPLEX128, sizeof(double complex), z, ITEMS * sizeof(double complex)};
        assert(!eval_with(artifact, &input, 1, &out, sizeof(out), ITEMS));
        assert(out);
        me_artifact_free(artifact);
    }
}

static void test_rejections(void) {
    check_rejected("def k(z):\n    return z // 2\n", "complex128");
    check_rejected("def k(z):\n    return ceil(z)\n", "complex128");
    check_rejected("def k(z):\n    return float(z)\n", "float64");
}

int main(void) {
    test_validation();
    test_arithmetic_and_math();
    test_real_results_and_control_flow();
    test_comparisons_and_predicates();
    test_constants();
    test_reductions();
    test_rejections();
    if (failures) {
        fprintf(stderr, "%d complex portable checks failed\n", failures);
        return 1;
    }
    printf("portable complex checks passed\n");
    return 0;
}
