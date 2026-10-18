/* Parser, precedence, artifact, dispatch and numerical checks for flexible math. */
#undef NDEBUG
#include "miniexpr.h"
#include "miniexpr_artifact.h"
#include "functions-simd.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static me_portable_status validate(const char *source, char *reason_out,
                                   size_t reason_cap) {
    me_variable input = {"x", ME_FLOAT64, 0};
    me_portable_error error = {0};
    me_portable_validation_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = 1,
        .output_itemsize = sizeof(double), .ndim = 0,
        .cardinality = ME_PORTABLE_ELEMENTWISE};
    me_portable_status rc = me_validate_portable_dsl_ex(source, "1.0", &input, 1, ME_FLOAT64,
                                                        &descriptor, &error);
    if (reason_out && reason_cap) snprintf(reason_out, reason_cap, "%s", error.message);
    return rc;
}

static char *manifest(const char *requires, const char *semantics, const char *source) {
    const char *format =
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[%s],\"source\":\"%s\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"float64\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"elementwise\"},"
        "\"semantics\":{%s},\"context\":{\"ndim\":0}}";
    size_t size = strlen(format) + strlen(requires) + strlen(semantics) + strlen(source) + 64;
    char *json = malloc(size);
    assert(json);
    assert(snprintf(json, size, format, requires, source, semantics) < (int)size);
    return json;
}

static me_artifact *load_json(const char *json, me_artifact_error *error_out) {
    me_artifact *artifact = NULL;
    me_artifact_error error;
    me_artifact_status rc = me_artifact_load(json, strlen(json), ME_JIT_OFF, &artifact, &error);
    if (error_out) *error_out = error;
    if (rc != ME_ARTIFACT_SUCCESS) return NULL;
    return artifact;
}

static void expect_load_status(const char *json, me_artifact_status expected) {
    size_t length = strlen(json);
    char *copy = malloc(length + 1);
    assert(copy);
    memcpy(copy, json, length + 1);
    me_artifact *artifact = NULL;
    me_artifact_error error;
    me_artifact_status rc = me_artifact_load(copy, strlen(copy), ME_JIT_OFF, &artifact, &error);
    if (rc != expected || (rc != ME_ARTIFACT_SUCCESS && (artifact || !error.message[0]))) {
        fprintf(stderr, "expected artifact status %d, got %d: %s\n%s\n",
                expected, rc, error.message, copy);
        assert(0);
    }
    if (artifact) me_artifact_free(artifact);
    free(copy);
}

static void parser_and_validation(void) {
    const char *header = "# me:math=%s\n# me:compiler=cc\ndef k(x):\n    return sin(x)\n";
    char source[256];
    const char *valid[] = {"auto", "system"};
    for (size_t i = 0; i < sizeof(valid)/sizeof(valid[0]); i++) {
        snprintf(source, sizeof(source), header, valid[i]);
        assert(validate(source, NULL, 0) == ME_PORTABLE_SUCCESS);
    }

    snprintf(source, sizeof(source), header, "meta");
    assert(validate(source, NULL, 0) != ME_PORTABLE_SUCCESS);
    snprintf(source, sizeof(source),
             "# me:math=sleef\n# me:ulp=u40\n# me:compiler=cc\ndef k(x):\n    return sin(x)\n");
    assert(validate(source, NULL, 0) != ME_PORTABLE_SUCCESS);
    snprintf(source, sizeof(source),
             "# me:math=sleef\n# me:math=sleef\ndef k(x):\n    return sin(x)\n");
    assert(validate(source, NULL, 0) != ME_PORTABLE_SUCCESS);
    snprintf(source, sizeof(source),
             "# me:ulp=u35\n# me:compiler=cc\ndef k(x):\n    return sin(x)\n");
    assert(validate(source, NULL, 0) != ME_PORTABLE_SUCCESS);
    snprintf(source, sizeof(source),
             "# me:fp=fast\ndef k(x):\n    return sin(x)\n");
    assert(validate(source, NULL, 0) != ME_PORTABLE_SUCCESS);
    snprintf(source, sizeof(source),
             "# me:fp=fast\n# me:compiler=cc\ndef k(x):\n    return sin(x)\n");
    assert(validate(source, NULL, 0) == ME_PORTABLE_SUCCESS);
}

static void environment_precedence(void) {
    const char *plain = "def k(x):\n    return sin(x)\n";
    set_env("ME_SIMD_MATH_BACKEND", "bogus");
    assert(validate(plain, NULL, 0) != ME_PORTABLE_SUCCESS);
    set_env("ME_DSL_FP_MODE", "bogus");
    assert(validate(plain, NULL, 0) != ME_PORTABLE_SUCCESS);
    set_env("ME_DSL_FP_MODE", NULL);

    set_env("ME_SIMD_MATH_BACKEND", "sleef");
    const char *system = "# me:math=system\ndef k(x):\n    return sin(x)\n";
    assert(validate(system, NULL, 0) == ME_PORTABLE_SUCCESS);
    const char *auto_source = "# me:math=auto\ndef k(x):\n    return sin(x)\n";
    assert(validate(auto_source, NULL, 0) == ME_PORTABLE_SUCCESS);
    set_env("ME_SIMD_MATH_BACKEND", NULL);
}

static void artifact_policy(void) {
    const char *source = "def k(x):\\n    return sin(x) + cos(x)\\n";
    char *strict = manifest("\"numeric\"", "\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", source);
    me_artifact_error error;
    me_artifact *artifact = load_json(strict, &error);
    if (!artifact) fprintf(stderr, "strict artifact: %s\n%s\n", error.message, strict);
    assert(artifact);
    assert(!strcmp(me_artifact_math_backend(artifact), "auto"));
    me_artifact_free(artifact);
    free(strict);

    char *fast_no_cap = manifest("\"numeric\"", "\"fp\":\"fast\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", source);
    expect_load_status(fast_no_cap, ME_ARTIFACT_ERR_UNSUPPORTED);
    free(fast_no_cap);

    char *fast_tcc = manifest("\"numeric\",\"flexible-math\"",
        "\"fp\":\"fast\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", source);
    expect_load_status(fast_tcc, ME_ARTIFACT_ERR_SOURCE);
    free(fast_tcc);

    char *fast_cc = manifest("\"numeric\",\"flexible-math\"",
        "\"fp\":\"fast\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"",
        "# me:compiler=cc\\ndef k(x):\\n    return sin(x) + cos(x)\\n");
    artifact = load_json(fast_cc, &error);
    if (!artifact) fprintf(stderr, "fast cc load: %s\n", error.message);
    assert(artifact);
    assert(!strcmp(me_artifact_fp_mode(artifact), "fast"));
    me_artifact_free(artifact);
    free(fast_cc);

    char *math = manifest("\"numeric\",\"flexible-math\"",
        "\"math\":\"system\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", source);
    artifact = load_json(math, &error);
    assert(artifact);
    assert(!strcmp(me_artifact_math_backend(artifact), "system"));
    me_artifact_free(artifact);
    free(math);

    char *conflict = manifest("\"numeric\",\"flexible-math\"",
        "\"math\":\"system\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"",
        "# me:math=sleef\\ndef k(x):\\n    return sin(x)\\n");
    expect_load_status(conflict, ME_ARTIFACT_ERR_SOURCE);
    free(conflict);

    char *ulp_without_sleef = manifest("\"numeric\",\"flexible-math\"",
        "\"ulp\":\"u35\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", source);
    expect_load_status(ulp_without_sleef, ME_ARTIFACT_ERR_SOURCE);
    free(ulp_without_sleef);
}

static void numerical_backend(const char *math, const char *ulp) {
    char semantics[160];
    if (ulp) {
        snprintf(semantics, sizeof(semantics),
                 "\"math\":\"%s\",\"ulp\":\"%s\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"",
                 math, ulp);
    }
    else {
        snprintf(semantics, sizeof(semantics),
                 "\"math\":\"%s\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", math);
    }
    const char *source = "def k(x):\\n    return sin(x) + cos(x) + exp(x * 0.1)\\n";
    char *json = manifest("\"numeric\",\"flexible-math\"", semantics, source);
    me_artifact_error error;
    me_artifact *artifact = load_json(json, &error);
    free(json);
    if (!artifact) {
        /* Unavailable explicit backend is a preparation error, never a fallback. */
        assert(strstr(error.message, "unavailable") || strstr(error.message, "requires"));
        return;
    }
    const size_t counts[] = {1, 3, 17, 257};
    for (size_t c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
        size_t n = counts[c];
        double *x = malloc(n * sizeof(double));
        double *out = malloc(n * sizeof(double));
        assert(x && out);
        for (size_t i = 0; i < n; i++) x[i] = -2.0 + 4.0 * (double)i / (double)(n ? n : 1);
        me_artifact_input input = {"x", ME_FLOAT64, x, n};
        me_artifact_status rc = me_artifact_eval(artifact, &input, 1, out, n, &error);
        if (rc != ME_ARTIFACT_SUCCESS) fprintf(stderr, "%s eval: %s\n", math, error.message);
        assert(rc == ME_ARTIFACT_SUCCESS);
        for (size_t i = 0; i < n; i++) {
            double reference = sin(x[i]) + cos(x[i]) + exp(x[i] * 0.1);
            double scale = 1.0 + fabs(reference);
            assert(fabs(out[i] - reference) <= 1e-9 * scale);
        }
        free(x);
        free(out);
    }
    if (!strcmp(math, "accelerate")) {
        /* Accelerate has documented system fallbacks; they must be observable. */
        assert(me_artifact_math_covers(artifact, "sin") == 1);
        assert(me_artifact_math_covers(artifact, "erf") == 0);
        assert(me_artifact_math_covers(artifact, "not_a_function") == -1);
    }
    me_artifact_free(artifact);
}

static void jit_eligibility(void) {
    const char *source = "def k(x):\\n    return sin(x) + cos(x)\\n";
    char *system = manifest("\"numeric\",\"flexible-math\"",
        "\"math\":\"system\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", source);
    me_artifact *artifact = NULL;
    me_artifact_error error;
    me_artifact_status rc = me_artifact_load(system, strlen(system), ME_JIT_ON, &artifact, &error);
    if (rc == ME_ARTIFACT_SUCCESS) {
        /* The generated JIT is optional; an unavailable host compiler is not a
         * policy failure. When present, system math keeps JIT eligibility. */
        (void)me_artifact_has_jit(artifact);
        me_artifact_free(artifact);
    }
    free(system);

    const char *backends[] = {"sleef", "accelerate"};
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); i++) {
        char semantics[160];
        snprintf(semantics, sizeof(semantics),
                 "\"math\":\"%s\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", backends[i]);
        char *json = manifest("\"numeric\",\"flexible-math\"", semantics, source);
        artifact = NULL;
        rc = me_artifact_load(json, strlen(json), ME_JIT_ON, &artifact, &error);
        if (rc == ME_ARTIFACT_SUCCESS) {
            assert(!me_artifact_has_jit(artifact));
            me_artifact_free(artifact);
        }
        else {
            /* Explicit unavailable backends fail at preparation. */
            assert(strstr(error.message, "unavailable") || strstr(error.message, "requires"));
        }
        free(json);
    }
}

static double sleef_max_error(const char *ulp) {
    char semantics[160];
    snprintf(semantics, sizeof(semantics),
             "\"math\":\"sleef\",\"ulp\":\"%s\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", ulp);
    char *json = manifest("\"numeric\",\"flexible-math\"", semantics,
                          "def k(x):\\n    return sin(x) + cos(x)\\n");
    me_artifact_error error;
    me_artifact *artifact = load_json(json, &error);
    free(json);
    if (!artifact) return -1.0;
    assert(!strcmp(me_artifact_sleef_accuracy(artifact), ulp));
    const size_t n = 4096;
    double x[4096], out[4096];
    for (size_t i = 0; i < n; i++) x[i] = -20.0 + 40.0 * (double)i / (double)(n - 1);
    me_artifact_input input = {"x", ME_FLOAT64, x, n};
    me_artifact_status rc = me_artifact_eval(artifact, &input, 1, out, n, &error);
    assert(rc == ME_ARTIFACT_SUCCESS);
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double reference = sin(x[i]) + cos(x[i]);
        double difference = fabs(out[i] - reference);
        if (difference > worst) worst = difference;
    }
    me_artifact_free(artifact);
    return worst;
}

static double *sleef_outputs(const char *ulp, const double *x, size_t n) {
    char semantics[160];
    snprintf(semantics, sizeof(semantics),
             "\"math\":\"sleef\",\"ulp\":\"%s\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", ulp);
    char *json = manifest("\"numeric\",\"flexible-math\"", semantics,
                          "def k(x):\\n    return sin(x) + cos(x)\\n");
    me_artifact_error error;
    me_artifact *artifact = load_json(json, &error);
    free(json);
    if (!artifact) return NULL;
    double *out = malloc(n * sizeof(double));
    assert(out);
    me_artifact_input input = {"x", ME_FLOAT64, x, n};
    me_artifact_status rc = me_artifact_eval(artifact, &input, 1, out, n, &error);
    me_artifact_free(artifact);
    assert(rc == ME_ARTIFACT_SUCCESS);
    return out;
}

static void sleef_tail_library_identity(void) {
    const size_t n = 257;
    double x[257], *u10, *u35;
    for (size_t i = 0; i < n; i++) x[i] = -12.0 + 24.0 * (double)i / (double)(n - 1);
    u10 = sleef_outputs("u10", x, n);
    u35 = sleef_outputs("u35", x, n);
    if (u10 && u35) {
        double worst = 0.0;
        for (size_t i = 0; i < n; i++) {
            double difference = fabs(u10[i] - u35[i]);
            if (difference > worst) worst = difference;
        }
        /* The final partial tile must not silently fall back to system math. */
        assert(worst > 0.0);
    }
    free(u10);
    free(u35);
}

static void compiler_environment_policy(void) {
    const char *modes[] = {"fast", "contract"};
    char source[160];
    set_env("ME_DSL_JIT_COMPILER", "cc");
    for (size_t i = 0; i < 2; i++) {
        snprintf(source, sizeof(source), "# me:fp=%s\ndef k(x):\n    return x * 2 + 1\n", modes[i]);
        assert(validate(source, NULL, 0) == ME_PORTABLE_SUCCESS);
    }
    assert(validate("# me:compiler=tcc\n# me:fp=fast\ndef k(x):\n    return x\n", NULL, 0)
           != ME_PORTABLE_SUCCESS);
    set_env("ME_DSL_JIT_COMPILER", "tcc");
    assert(validate("# me:compiler=cc\n# me:fp=fast\ndef k(x):\n    return x\n", NULL, 0)
           == ME_PORTABLE_SUCCESS);
    set_env("ME_DSL_JIT_COMPILER", NULL);
}

static void source_policy_capability(void) {
    const char *sources[] = {
        "# me:math=system\\ndef k(x):\\n    return sin(x)\\n",
        "# me:math=sleef\\n# me:ulp=u35\\ndef k(x):\\n    return sin(x)\\n",
        "# me:compiler=cc\\n# me:fp=fast\\ndef k(x):\\n    return sin(x)\\n"
    };
    for (size_t i = 0; i < sizeof(sources)/sizeof(sources[0]); i++) {
        char *json = manifest("\"numeric\"", "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", sources[i]);
        expect_load_status(json, ME_ARTIFACT_ERR_UNSUPPORTED);
        free(json);
    }
    char *json = manifest("\"numeric\"", "\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"",
                          "# me:fp=strict\\ndef k(x):\\n    return sin(x)\\n");
    expect_load_status(json, ME_ARTIFACT_SUCCESS);
    free(json);
}

static void power_backend_identity(me_dsl_math_backend backend, me_dsl_sleef_ulp ulp) {
    me_math_dispatch dispatch;
    if (!me_math_dispatch_init(&dispatch, backend, ulp)) return;
    const char *name = backend == ME_DSL_MATH_SLEEF ? "sleef" : "accelerate";
    char semantics[192];
    snprintf(semantics, sizeof(semantics),
             "\"math\":\"%s\",%s\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"", name,
             backend != ME_DSL_MATH_SLEEF ? "" : ulp == ME_DSL_ULP_U10 ? "\"ulp\":\"u10\"," : "\"ulp\":\"u35\",");
    const char *expressions[] = {"x ** 1.3", "pow(x, 1.3)"};
    const size_t counts[] = {1, 3, 257};
    me_simd_accuracy_state previous;
    me_simd_accuracy_push(ulp, &previous);
    for (int precision = 0; precision < 2; precision++) {
        for (size_t expression = 0; expression < 2; expression++) {
            char source[128];
            snprintf(source, sizeof(source), "def k(x):\\n    return %s\\n", expressions[expression]);
            char *json = manifest("\"numeric\",\"flexible-math\"", semantics, source);
            if (precision == 0) {
                for (char *p = json; (p = strstr(p, "float64")) != NULL; p += 7) memcpy(p, "float32", 7);
            }
            me_artifact_error error;
            me_artifact *artifact = load_json(json, &error);
            free(json);
            assert(artifact);
            assert(me_artifact_math_covers(artifact, "pow") == 1);
            for (size_t c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
                size_t count = counts[c];
                float xf[257], of[257];
                double xd[257], od[257];
                for (size_t i = 0; i < count; i++) {
                    xd[i] = 0.1 + 9.9 * (double)(i + 1) / 257.0;
                    xf[i] = (float)xd[i];
                }
                me_artifact_input input = {"x", precision ? ME_FLOAT64 : ME_FLOAT32,
                                          precision ? (const void *)xd : (const void *)xf, count};
                assert(me_artifact_eval(artifact, &input, 1, precision ? (void *)od : (void *)of,
                                        count, &error) == ME_ARTIFACT_SUCCESS);
                for (size_t i = 0; i < count; i++) {
                    if (precision) {
                        double x[4] = {xd[i], xd[i], xd[i], xd[i]};
                        double y[4] = {1.3, 1.3, 1.3, 1.3}, expected[4];
                        dispatch.binary_f64[ME_MATHOP_POW](x, y, expected, 4);
                        assert(memcmp(&od[i], &expected[0], sizeof(double)) == 0);
                    }
                    else {
                        float x[8], y[8], expected[8];
                        for (int j = 0; j < 8; j++) { x[j] = xf[i]; y[j] = 1.3f; }
                        dispatch.binary_f32[ME_MATHOP_POW](x, y, expected, 8);
                        assert(memcmp(&of[i], &expected[0], sizeof(float)) == 0);
                    }
                }
            }
            me_artifact_free(artifact);
        }
    }
    me_simd_accuracy_pop(&previous);
}

static void numerical_dispatch(void) {
    numerical_backend("system", NULL);
    numerical_backend("auto", NULL);
    numerical_backend("sleef", "u10");
    numerical_backend("sleef", "u35");
    numerical_backend("accelerate", NULL);

    double u10 = sleef_max_error("u10");
    double u35 = sleef_max_error("u35");
    if (u10 >= 0.0 && u35 >= 0.0) {
        /* Kernel accuracy must actually reach the selected SLEEF variant. */
        assert(u10 > 0.0 || u35 > 0.0);
        assert(u10 <= u35 * 2.0 + 1e-15);
        sleef_tail_library_identity();
    }
}

int main(void) {
    parser_and_validation();
    environment_precedence();
    artifact_policy();
    jit_eligibility();
    numerical_dispatch();
    compiler_environment_policy();
    source_policy_capability();
    power_backend_identity(ME_DSL_MATH_SLEEF, ME_DSL_ULP_U10);
    power_backend_identity(ME_DSL_MATH_SLEEF, ME_DSL_ULP_U35);
    power_backend_identity(ME_DSL_MATH_ACCELERATE, ME_DSL_ULP_OMIT);
    printf("flexible math policy tests passed\n");
    return 0;
}
