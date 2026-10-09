/* Builtin identity must survive Windows CRT aliases and Release linker folding. */
#undef NDEBUG
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static me_dsl_compiled_program *compile(const char *source, me_variable *vars, int nvars) {
    char reason[256];
    bool is_dsl;
    int error;
    me_dsl_compiled_program *program = dsl_compile_program_profile(source, vars, nvars, ME_AUTO, 0,
        ME_JIT_OFF, ME_DSL_PROFILE_PORTABLE_1_1, &error, &is_dsl, reason, sizeof(reason));
    if (!program) fprintf(stderr, "compile failure: %s\n%s", reason, source);
    assert(program && is_dsl);
    return program;
}

int main(void) {
    me_variable vars[] = {{.name = "x", .dtype = ME_BOOL, .type = ME_VARIABLE},
                          {.name = "y", .dtype = ME_BOOL, .type = ME_VARIABLE}};
    me_dsl_compiled_program *program = compile("def k(x):\n    return real(x)\n", vars, 1);
    assert(program->output_dtype == ME_BOOL);
    dsl_compiled_program_free(program);
    program = compile("def k(x):\n    return conj(x)\n", vars, 1);
    assert(program->output_dtype == ME_INT8);
    dsl_compiled_program_free(program);

    /* NumPy remainder has integral loops even where C remainder would widen
     * to a floating loop. Its sign follows the divisor, not the dividend. */
    vars[0].dtype = vars[1].dtype = ME_INT8;
    program = compile("def k(x, y):\n    return remainder(x, y)\n", vars, 2);
    assert(program->output_dtype == ME_INT8);
    int8_t x[] = {-7, 7}, y[] = {3, -3}, result[2];
    const void *inputs[] = {x, y};
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.jit_mode = ME_JIT_OFF;
    assert(dsl_eval_program(program, inputs, 2, result, 2, &params, 0, NULL, NULL, NULL, NULL) == 0);
    assert(result[0] == 2 && result[1] == -2);
    dsl_compiled_program_free(program);
    return 0;
}
