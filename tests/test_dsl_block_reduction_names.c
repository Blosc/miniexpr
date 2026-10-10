#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"

static me_dsl_compiled_program *compile(const char *source, me_dsl_semantic_profile profile) {
    me_variable x = {.name = "x", .dtype = ME_FLOAT64, .type = ME_VARIABLE};
    int error;
    bool is_dsl;
    char reason[256];
    return dsl_compile_program_profile(source, &x, 1, ME_AUTO, 0, ME_JIT_OFF,
                                      profile, &error, &is_dsl, reason, sizeof(reason));
}

int main(void) {
    const char *names[] = {"sum", "prod", "min", "max", "any", "all", "mean"};
    const double expected[] = {6, 6, 1, 3};
    const double values[] = {1, 2, 3};
    const void *inputs[] = {values};
    const me_dsl_semantic_profile profiles[] = {
        ME_DSL_PROFILE_PORTABLE_1_0, ME_DSL_PROFILE_PORTABLE_1_1};
    for (size_t j = 0; j < 2; j++) {
        for (size_t i = 0; i < 7; i++) {
            char source[128];
            snprintf(source, sizeof(source), "def k(x):\n    return %s(x)\n", names[i]);
            assert(!compile(source, profiles[j]));
            me_dsl_compiled_program *legacy = compile(source, ME_DSL_PROFILE_FULL);
            assert(legacy);
            dsl_compiled_program_free(legacy);
            snprintf(source, sizeof(source), "def k(x):\n    return block_%s(x)\n", names[i]);
            assert(!compile(source, ME_DSL_PROFILE_FULL));
            me_dsl_compiled_program *p = compile(source, profiles[j]);
            assert(p && p->output_is_scalar);
            me_dsl_portable_eval_descriptor descriptor = {
                .nitems = 3, .output_capacity = sizeof(double)};
            union { double value; bool truth; } output = {0};
            assert(dsl_eval_program_portable(p, inputs, 1, &output, &descriptor) == 0);
            if (i < 4) assert(output.value == expected[i]);
            else if (i < 6) assert(output.truth);
            else {
                assert(output.value == 2);
                descriptor.nitems = 0;
                assert(dsl_eval_program_portable(p, inputs, 1, &output, &descriptor) == 0);
                assert(isnan(output.value));
            }
            if (i == 0) {
                descriptor.nitems = 2;
                assert(dsl_eval_program_portable(p, inputs, 1, &output, &descriptor) == 0);
                assert(output.value == 3); /* Only the supplied block participates. */
            }
            dsl_compiled_program_free(p);
        }
        assert(!compile("def k(x):\n    return mean(x)\n", profiles[j]));
        assert(!compile("def k(x):\n    return block_sum(x, axis=0)\n", profiles[j]));
        assert(!compile("def k(x):\n    return block_sum(x, 0)\n", profiles[j]));
        assert(!compile("def k(x):\n    return block_sum(block_sum(x))\n", profiles[j]));
    }
    puts("Block reduction names and scope passed");
    return 0;
}
