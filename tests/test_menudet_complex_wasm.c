/* Focused wasm32/node interpreter parity for the complex Menudet contract. */
#undef NDEBUG
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "miniexpr.h"
#include "miniexpr_artifact.h"

static me_artifact *load(const char *body, const char *out_dtype, const char *cardinality) {
    char json[4096];
    int n = snprintf(json, sizeof(json),
        "{\"schema_version\":\"0.1\",\"language\":{\"name\":\"menudet\",\"version\":\"0.1\"},"
        "\"requires\":[\"numeric\",\"control-flow\",\"block-reductions\",\"nd-context\"],"
        "\"source\":\"def k(z):\\n%s\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"z\",\"dtype\":\"complex128\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"%s\"},"
        "\"context\":{\"ndim\":1},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"}}",
        body, out_dtype, cardinality);
    assert(n > 0 && (size_t)n < sizeof(json));
    me_artifact *artifact = NULL;
    me_artifact_error error;
    int rc = me_artifact_load(json, (size_t)n, ME_JIT_OFF, &artifact, &error);
    if (rc) fprintf(stderr, "load (%d): %s\n", rc, error.message);
    assert(!rc && artifact);
    assert(!me_artifact_has_jit(artifact));  /* complex stays interpreted */
    return artifact;
}

static int eval(me_artifact *artifact, const double complex *z, size_t count, void *out,
                size_t capacity) {
    me_artifact_buffer input = {"z", ME_COMPLEX128, 16, z, count * 16};
    int64_t shape = (int64_t)count, origin = 0, extent = (int64_t)count;
    me_artifact_eval_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,
        .nitems = count, .output_capacity = capacity, .ndim = 1,
        .logical_shape = &shape, .block_origin = &origin, .block_extent = &extent};
    me_artifact_error error;
    int rc = me_artifact_eval_ex(artifact, &input, 1, out, &descriptor, &error);
    if (rc) fprintf(stderr, "eval (%d): %s\n", rc, error.message);
    return rc;
}

int main(void) {
    const double complex z[4] = {1 + 2 * I, 3 - 4 * I, -2 + 0.5 * I, 0};
    double complex out[4] = {0};

    me_artifact *artifact = load("    v = z\\n    return v * z\\n", "complex128", "elementwise");
    assert(!eval(artifact, z, 4, out, sizeof(out)));
    for (int i = 0; i < 4; i++) assert(cabs(out[i] - z[i] * z[i]) < 1e-12);
    me_artifact_free(artifact);

    artifact = load("    return real(z)\\n", "float64", "elementwise");
    double real_out[4] = {0};
    assert(!eval(artifact, z, 4, real_out, sizeof(real_out)));
    for (int i = 0; i < 4; i++) assert(real_out[i] == creal(z[i]));
    me_artifact_free(artifact);

    artifact = load("    return conj(z)\\n", "complex128", "elementwise");
    assert(!eval(artifact, z, 4, out, sizeof(out)));
    for (int i = 0; i < 4; i++) {
        assert(creal(out[i]) == creal(z[i]));
        assert(cimag(out[i]) == -cimag(z[i]));
    }
    me_artifact_free(artifact);

    artifact = load("    v = z\\n    for i in range(2):\\n        v = v * z\\n"
                    "    if imag(v) < 0:\\n        return v + 1\\n    return v\\n",
                    "complex128", "elementwise");
    assert(!eval(artifact, z, 4, out, sizeof(out)));
    for (int i = 0; i < 4; i++) {
        double complex expected = z[i] * z[i] * z[i];
        if (cimag(expected) < 0) expected += 1;
        assert(cabs(out[i] - expected) < 1e-12);
    }
    me_artifact_free(artifact);

    artifact = load("    return abs(z) + real(z) - imag(z)\\n", "float64", "elementwise");
    assert(!eval(artifact, z, 4, real_out, sizeof(real_out)));
    for (int i = 0; i < 4; i++) {
        assert(fabs(real_out[i] - (cabs(z[i]) + creal(z[i]) - cimag(z[i]))) < 1e-12);
    }
    me_artifact_free(artifact);

    artifact = load("    return block_sum(z)\\n", "complex128", "block_scalar");
    double complex sum = 0;
    assert(!eval(artifact, z, 4, &sum, sizeof(sum)));
    assert(cabs(sum - (z[0] + z[1] + z[2] + z[3])) < 1e-12);
    me_artifact_free(artifact);

    printf("portable complex wasm checks passed\n");
    return 0;
}
