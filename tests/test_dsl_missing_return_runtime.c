/* Semantic JIT failures must not re-execute the kernel in the interpreter. */
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include "../src/dsl_eval_internal.h"
#include "../src/dsl_jit_cgen.h"

static int kernel_status;
static int kernel_calls;

static int failing_kernel(const void **inputs, void *output, int64_t nitems) {
    (void)inputs;
    (void)output;
    (void)nitems;
    kernel_calls++;
    return kernel_status;
}

static void check_runtime_status(bool reserved_input) {
    const char *source = reserved_input
        ? "def kernel(x):\n    y = x + _i0\n    return y\n"
        : "def kernel(x):\n    y = x + 1.0\n    return y\n";
    me_variable variables[] = {{"x", ME_FLOAT64, NULL, ME_VARIABLE, NULL, 0}};
    int error = 0;
    bool is_dsl = false;
    char reason[256] = {0};
    me_dsl_compiled_program *program = dsl_compile_program(source, variables, 1,
        ME_FLOAT64, 1, ME_JIT_OFF, &error, &is_dsl, reason, sizeof(reason));
    assert(program && is_dsl);
    program->jit_kernel_fn = failing_kernel;
    if (reserved_input) {
        /* Force the buffered JIT path: reserved input is not synthesized in JIT. */
        program->jit_nparams = 1;
        program->jit_param_bindings = calloc(1, sizeof(*program->jit_param_bindings));
        assert(program->jit_param_bindings);
        program->jit_param_bindings[0].kind = ME_DSL_JIT_BIND_RESERVED_I;
        program->jit_param_bindings[0].var_index = program->idx_i[0];
    } else {
        program->jit_nparams = 0;
    }
    double x[] = {1.0, 2.0, 3.0}, output[] = {-99.0, -99.0, -99.0};
    const void *inputs[] = {x};
    int64_t shape[] = {3};
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.jit_mode = ME_JIT_ON;

    kernel_status = ME_DSL_JIT_MISSING_RETURN;
    kernel_calls = 0;
    int rc = dsl_eval_program(program, inputs, 1, output, 3, &params,
                             1, shape, NULL, NULL, NULL);
    assert(rc == ME_EVAL_ERR_INVALID_ARG);
    assert(kernel_calls == 1);
    /* The interpreter would successfully overwrite these sentinels. */
    assert(output[0] == -99.0 && output[1] == -99.0 && output[2] == -99.0);

    kernel_status = ME_DSL_JIT_LOOP_CAP;
    kernel_calls = 0;
    rc = dsl_eval_program(program, inputs, 1, output, 3, &params,
                          1, shape, NULL, NULL, NULL);
    assert(rc == ME_EVAL_ERR_INVALID_ARG);
    assert(kernel_calls == 1);
    assert(output[0] == -99.0 && output[1] == -99.0 && output[2] == -99.0);

    kernel_status = 1;  /* Existing retryable backend status still falls back. */
    kernel_calls = 0;
    rc = dsl_eval_program(program, inputs, 1, output, 3, &params,
                          1, shape, NULL, NULL, NULL);
    assert(rc == ME_EVAL_SUCCESS);
    assert(kernel_calls == 1);
    for (int i = 0; i < 3; i++) {
        assert(output[i] == x[i] + (reserved_input ? i : 1.0));
    }
    program->jit_kernel_fn = NULL;
    dsl_compiled_program_free(program);
}

int main(void) {
    check_runtime_status(false);
    check_runtime_status(true);
    return 0;
}
