/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#undef NDEBUG
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include "functions.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static me_dsl_compiled_program *compile(const char *source, me_variable *vars, int nvars) {
    char reason[256];
    bool is_dsl;
    int error;
    me_dsl_compiled_program *program = dsl_compile_program_profile(
        source, vars, nvars, ME_AUTO, 0, ME_JIT_OFF, ME_DSL_PROFILE_MENUDET_0_1,
        &error, &is_dsl, reason, sizeof(reason));
    if (!program) fprintf(stderr, "compile failure: %s\n%s", reason, source);
    assert(program && is_dsl);
    assert(program->semantic_profile == ME_DSL_PROFILE_MENUDET_0_1);
    return program;
}

static int eval(me_dsl_compiled_program *program, const void **inputs, int nvars,
                void *output, int nitems) {
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.jit_mode = ME_JIT_OFF;
    return dsl_eval_program(program, inputs, nvars, output, nitems, &params,
                            0, NULL, NULL, NULL, NULL);
}

/* A weak Python literal stored in a local and then reassigned with a strong
 * float32 value must keep the strong float32 storage width, as NumPy 2.x does. */
static void weak_float_local_keeps_float32(void) {
    me_variable x = {.name = "x", .dtype = ME_FLOAT32, .type = ME_VARIABLE};
    float input[] = {1.0f, 2.0f};
    float output[2];
    const void *inputs[] = {input};

    me_dsl_compiled_program *program = compile(
        "def k(x):\n    z = 0.0\n    z = z + x\n    return z\n", &x, 1);
    assert(program->output_dtype == ME_FLOAT32);
    assert(eval(program, inputs, 1, output, 2) == 0);
    assert(output[0] == 1.0f && output[1] == 2.0f);
    dsl_compiled_program_free(program);

    program = compile(
        "def k(x):\n    z = 0.0\n    for i in range(3):\n        z = z + x\n    return z\n", &x, 1);
    assert(program->output_dtype == ME_FLOAT32);
    assert(eval(program, inputs, 1, output, 2) == 0);
    assert(output[0] == 3.0f && output[1] == 6.0f);
    dsl_compiled_program_free(program);

    float branch_input[] = {-1.0f, 2.0f};
    float branch_output[2];
    const void *branch_inputs[] = {branch_input};
    program = compile(
        "def k(x):\n    z = 0.0\n    if x > 0:\n        z = x\n    return z\n", &x, 1);
    assert(program->output_dtype == ME_FLOAT32);
    assert(eval(program, branch_inputs, 1, branch_output, 2) == 0);
    assert(branch_output[0] == 0.0f && branch_output[1] == 2.0f);
    dsl_compiled_program_free(program);
}

/* Weak integer literals adopt strong integer widths and float32 storage. */
static void weak_integer_local_keeps_strong_width(void) {
    me_variable x = {.name = "x", .dtype = ME_INT8, .type = ME_VARIABLE};
    int8_t input8[] = {1, 2};
    int8_t output8[2];
    const void *inputs[] = {input8};

    me_dsl_compiled_program *program = compile(
        "def k(x):\n    z = 0\n    z = z + x\n    return z\n", &x, 1);
    assert(program->output_dtype == ME_INT8);
    assert(eval(program, inputs, 1, output8, 2) == 0);
    assert(output8[0] == 1 && output8[1] == 2);
    dsl_compiled_program_free(program);

    x.dtype = ME_FLOAT32;
    float input32[] = {1.0f, 2.0f};
    float output32[2];
    inputs[0] = input32;
    program = compile("def k(x):\n    z = 0\n    z = z + x\n    return z\n", &x, 1);
    assert(program->output_dtype == ME_FLOAT32);
    assert(eval(program, inputs, 1, output32, 2) == 0);
    assert(output32[0] == 1.0f && output32[1] == 2.0f);
    dsl_compiled_program_free(program);
}

/* Weak float + strong integer keeps NumPy's float64 promotion. */
static void weak_float_over_integer_uses_float64(void) {
    me_variable x = {.name = "x", .dtype = ME_INT32, .type = ME_VARIABLE};
    int32_t input[] = {1, 2};
    double output[2];
    const void *inputs[] = {input};

    me_dsl_compiled_program *program = compile(
        "def k(x):\n    z = 0.0\n    z = z + x\n    return z\n", &x, 1);
    assert(program->output_dtype == ME_FLOAT64);
    assert(eval(program, inputs, 1, output, 2) == 0);
    assert(output[0] == 1.0 && output[1] == 2.0);
    dsl_compiled_program_free(program);
}

/* Narrowing a weak literal into a strong integer local must stay checked. */
static void out_of_range_weak_literal_is_rejected(void) {
    me_variable x = {.name = "x", .dtype = ME_INT8, .type = ME_VARIABLE};
    int8_t input[] = {1, 2};
    int8_t output[2];
    const void *inputs[] = {input};
    char reason[256];
    bool is_dsl;
    int error;

    me_dsl_compiled_program *program = dsl_compile_program_profile(
        "def k(x):\n    z = 1000\n    z = z + x\n    return z\n",
        &x, 1, ME_AUTO, 0, ME_JIT_OFF, ME_DSL_PROFILE_MENUDET_0_1,
        &error, &is_dsl, reason, sizeof(reason));
    assert(is_dsl);
    if (program) {
        assert(eval(program, inputs, 1, output, 2) != 0);
        dsl_compiled_program_free(program);
    }
    else {
        assert(reason[0] != '\0');
    }
}

/* A local initialized with a weak literal before a loop and reassigned a
 * strong float32 inside the loop must be strong float32 throughout the loop,
 * contextualizing integer literals like 4 and 2 to float32 without float64
 * promotion or weak-scalar conversion wrappers. */
static void weak_float_loop_locals_stay_strong_float32(void) {
    me_variable vars[] = {
        {.name = "cr", .dtype = ME_FLOAT32, .type = ME_VARIABLE},
        {.name = "ci", .dtype = ME_FLOAT32, .type = ME_VARIABLE},
    };
    char reason[256];
    bool is_dsl;
    int error;
    const char *literal_src =
        "def mandelbrot(cr, ci):\n"
        "    zr = 0.0\n"
        "    zi = 0.0\n"
        "    escape_iter = float(200)\n"
        "    for i in range(200):\n"
        "        if zr * zr + zi * zi > 4:\n"
        "            escape_iter = float(i)\n"
        "            break\n"
        "        zr_new = zr * zr - zi * zi + cr\n"
        "        zi = 2 * zr * zi + ci\n"
        "        zr = zr_new\n"
        "    return escape_iter\n";
    const char *typed_src =
        "def mandelbrot(cr, ci):\n"
        "    zr = float32(0.0)\n"
        "    zi = float32(0.0)\n"
        "    escape_iter = float(200)\n"
        "    for i in range(200):\n"
        "        if zr * zr + zi * zi > float32(4.0):\n"
        "            escape_iter = float(i)\n"
        "            break\n"
        "        zr_new = zr * zr - zi * zi + cr\n"
        "        zi = float32(2.0) * zr * zi + ci\n"
        "        zr = zr_new\n"
        "    return escape_iter\n";

    me_dsl_compiled_program *lit_prog = dsl_compile_program_profile(
        literal_src, vars, 2, ME_FLOAT32, 0, ME_JIT_OFF, ME_DSL_PROFILE_MENUDET_0_1,
        &error, &is_dsl, reason, sizeof(reason));
    assert(lit_prog && is_dsl);
    int zr_idx = dsl_var_table_find(&lit_prog->vars, "zr");
    int zi_idx = dsl_var_table_find(&lit_prog->vars, "zi");
    int zr_new_idx = dsl_var_table_find(&lit_prog->vars, "zr_new");
    assert(zr_idx >= 0 && zi_idx >= 0 && zr_new_idx >= 0);
    assert(lit_prog->vars.dtypes[zr_idx] == ME_FLOAT32 && !lit_prog->numpy_weak_vars[zr_idx]);
    assert(lit_prog->vars.dtypes[zi_idx] == ME_FLOAT32 && !lit_prog->numpy_weak_vars[zi_idx]);
    assert(lit_prog->vars.dtypes[zr_new_idx] == ME_FLOAT32 && !lit_prog->numpy_weak_vars[zr_new_idx]);

    const me_dsl_compiled_stmt *for_stmt = lit_prog->block.stmts[3];
    assert(for_stmt->kind == ME_DSL_STMT_FOR);
    const me_dsl_compiled_stmt *if_stmt = for_stmt->as.for_loop.body.stmts[0];
    assert(if_stmt->kind == ME_DSL_STMT_IF);
    assert(if_stmt->as.if_stmt.cond.expr->input_dtype == ME_FLOAT32);
    const me_dsl_compiled_stmt *zi_stmt = for_stmt->as.for_loop.body.stmts[2];
    assert(zi_stmt->kind == ME_DSL_STMT_ASSIGN);
    assert(zi_stmt->as.assign.value.expr->dtype == ME_FLOAT32);
    assert(((const me_expr *)zi_stmt->as.assign.value.expr->parameters[0])->dtype == ME_FLOAT32);

    me_dsl_compiled_program *typed_prog = dsl_compile_program_profile(
        typed_src, vars, 2, ME_FLOAT32, 0, ME_JIT_OFF, ME_DSL_PROFILE_MENUDET_0_1,
        &error, &is_dsl, reason, sizeof(reason));
    assert(typed_prog && is_dsl);

    float cr[] = {-2.0f + 873.0f * (2.6f / 1199.0f), -2.0f + 872.0f * (2.6f / 1199.0f)};
    float ci[] = {-1.1f + 64.0f * (2.2f / 799.0f), -1.1f + 68.0f * (2.2f / 799.0f)};
    float lit_out[2] = {0}, typed_out[2] = {0};
    const void *inputs[] = {cr, ci};
    assert(eval(lit_prog, inputs, 2, lit_out, 2) == 0);
    assert(eval(typed_prog, inputs, 2, typed_out, 2) == 0);
    assert(lit_out[0] == typed_out[0] && lit_out[1] == typed_out[1]);

    dsl_compiled_program_free(lit_prog);
    dsl_compiled_program_free(typed_prog);
}

int main(void) {
    weak_float_local_keeps_float32();
    weak_integer_local_keeps_strong_width();
    weak_float_over_integer_uses_float64();
    out_of_range_weak_literal_is_rejected();
    weak_float_loop_locals_stay_strong_float32();
    puts("menudet weak local joins passed");
    return 0;
}
