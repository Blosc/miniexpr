/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include "functions.h"
#include "dsl_portable_expr.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <stdio.h>
#include <string.h>
#if (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
#include <pthread.h>
#endif

static void masked_iteration_fixture(void);
static void concurrent_handle_fixture(void);
static void pi_math_fixture(void);
static void operation_domain_fixture(void);
static void independent_anchor_fixture(void);

static me_dsl_compiled_program *compile(const char *source, me_variable *vars, int nvars, me_dtype out) {
    char reason[256];
    bool is_dsl;
    int error;
    me_dsl_compiled_program *p = dsl_compile_program_profile(source, vars, nvars, out, 0,
        ME_JIT_ON, ME_DSL_PROFILE_PORTABLE_1_0, &error, &is_dsl, reason, sizeof(reason));
    if (!p) fprintf(stderr, "compile failure: %s at %d\n%s\n", reason, error, source);
    assert(p && is_dsl);
    assert(p->semantic_profile == ME_DSL_PROFILE_PORTABLE_1_0);
    assert(p->jit_request_mode == ME_JIT_OFF && !p->jit_ir && !p->jit_kernel_fn);
    return p;
}

static int eval(me_dsl_compiled_program *p, const void **inputs, int nvars, void *out, int nitems) {
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.jit_mode = ME_JIT_ON;
    return dsl_eval_program(p, inputs, nvars, out, nitems, &params, 0, NULL, NULL, NULL, NULL);
}

static void rejected(const char *source, me_variable *vars, int nvars, const char *message) {
    char reason[256];
    int error;
    bool is_dsl;
    me_dsl_compiled_program *p = dsl_compile_program_profile(source, vars, nvars, ME_AUTO, 0,
        ME_JIT_DEFAULT, ME_DSL_PROFILE_PORTABLE_1_0, &error, &is_dsl, reason, sizeof(reason));
    assert(!p && is_dsl && strstr(reason, message));
}

int main(void) {
    me_variable x = {.name = "x", .dtype = ME_FLOAT32, .type = ME_VARIABLE};
    float f[] = {16777216.0f, 2.0f};
    const void *inputs[] = {f};
    double out[2];
    me_dsl_compiled_program *p = compile("def kernel(x):\n    return (x + 1.0) - x\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 0.0 && out[1] == 1.0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    a = x + 1.0\n    return a - x\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 0.0 && out[1] == 1.0);
    dsl_compiled_program_free(p);

    x.dtype = ME_INT64;
    int64_t ints[] = {INT64_C(9007199254740993), INT64_MAX};
    int64_t result[2];
    inputs[0] = ints;
    p = compile("def kernel(x):\n    return x - 9007199254740993\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 0 && result[1] == INT64_C(9214364837600034814));
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x + 1\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return where(x < 9223372036854775807, x + 1, x)\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == ints[0] + 1 && result[1] == INT64_MAX);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    if x < 9223372036854775807:\n        return x + 1\n    return x\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[1] == INT64_MAX);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return -9223372036854775808\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == INT64_MIN);
    dsl_compiled_program_free(p);
    rejected("def kernel(x):\n    return 9223372036854775808\n", &x, 1, "literal");

    x.dtype = ME_INT8;
    int8_t narrow[] = {126, 127};
    inputs[0] = narrow;
    p = compile("def kernel(x):\n    return x + 1\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) != 0); /* Output widening must not hide int8 overflow. */
    dsl_compiled_program_free(p);
    rejected("def kernel(x):\n    return x + 128\n", &x, 1, "literal");
    p = compile("def kernel(x):\n    return x * 2\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);
    narrow[0] = -7;
    narrow[1] = 7;
    p = compile("def kernel(x):\n    return x >> 1\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == -4 && result[1] == 3);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x << 8\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return ~x\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 6 && result[1] == -8);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x ** 2\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 49 && result[1] == 49);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x ** 3\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);

    x.dtype = ME_BOOL;
    bool truth[] = {true, false};
    inputs[0] = truth;
    p = compile("def kernel(x):\n    return x / 2\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 0.5 && out[1] == 0.0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x / (1 + 1)\n", &x, 1, ME_BOOL);
    bool bool_out[2];
    assert(eval(p, inputs, 1, bool_out, 2) == 0 && bool_out[0] && !bool_out[1]);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return int(x + x)\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 2 && result[1] == 0);
    dsl_compiled_program_free(p);

    x.dtype = ME_FLOAT64;
    double fractions[] = {1.9, -1.9};
    inputs[0] = fractions;
    p = compile("def kernel(x):\n    return int(x / 2.0)\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 0 && result[1] == 0);
    fractions[0] = 0x1p63;
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == INT64_C(4611686018427387904));
    fractions[0] = INFINITY;
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);

    x.dtype = ME_INT64;
    int64_t signs[] = {-7, 7};
    inputs[0] = signs;
    p = compile("def kernel(x):\n    return x // 3\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == -3 && result[1] == 2);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    a = x\n    a //= 3\n    return a\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == -3 && result[1] == 2);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x % 3\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 2 && result[1] == 1);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x // 0\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x < 0 or x // 0 > 1\n", &x, 1, ME_BOOL);
    assert(eval(p, inputs, 1, bool_out, 1) == 0 && bool_out[0]);
    assert(eval(p, inputs, 1, bool_out, 2) != 0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    a = x\n    for i in range(3):\n        a = a + 1\n    return a\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == -4 && result[1] == 10);
    dsl_compiled_program_free(p);
    int64_t limits[] = {2, 4};
    inputs[0] = limits;
    p = compile("def kernel(x):\n    for i in range(x):\n        y = i\n    return i\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 1 && result[1] == 3);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    for i in range(5):\n        if i == x:\n            break\n    return i\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 2 && result[1] == 4);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    for i in range(x):\n        y = i\n    return i\n", &x, 1, ME_INT64);
    limits[0] = 0;
    assert(eval(p, inputs, 1, result, 2) != 0); /* Empty ranges never define the iterator. */
    dsl_compiled_program_free(p);
    inputs[0] = signs;
    p = compile("def kernel(x):\n    return abs(x)\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) == 0 && result[0] == 7 && result[1] == 7);
    signs[0] = INT64_MIN;
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);

    x.dtype = ME_FLOAT32;
    float math_values[] = {1.0f, -1.0f};
    inputs[0] = math_values;
    p = compile("def kernel(x):\n    return sin(x)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == (double)sinf(1.0f) && out[1] == (double)sinf(-1.0f));
    dsl_compiled_program_free(p);
    math_values[0] = 2.0f;
    p = compile("def kernel(x):\n    return log(x)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == (double)logf(2.0f) && isnan(out[1]));
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return hypot(x, 2.0)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == (double)hypotf(2.0f, 2.0f));
    dsl_compiled_program_free(p);

    me_variable pair[] = {{.name = "x", .dtype = ME_INT64, .type = ME_VARIABLE},
                          {.name = "y", .dtype = ME_UINT64, .type = ME_VARIABLE}};
    rejected("def kernel(x, y):\n    return x + y\n", pair, 2, "promotion");
    p = compile("def kernel(x, y):\n    return x < y\n", pair, 2, ME_BOOL);
    int64_t signed_values[] = {-1, INT64_MAX};
    uint64_t unsigned_values[] = {0, UINT64_MAX};
    const void *mixed_inputs[] = {signed_values, unsigned_values};
    assert(eval(p, mixed_inputs, 2, bool_out, 2) == 0 && bool_out[0] && bool_out[1]);
    dsl_compiled_program_free(p);

    rejected("def kernel(x):\n    print(x)\n    return x\n", &x, 1, "print");
    x.dtype = ME_FLOAT32;
    float cancellation[] = {16777216.0f, 1.0f, -16777216.0f};
    inputs[0] = cancellation;
    p = compile("def kernel(x):\n    return x + sum(x)\n", &x, 1, ME_FLOAT64);
    double three_out[3];
    assert(eval(p, inputs, 1, three_out, 3) == 0 && three_out[0] == cancellation[0] && three_out[1] == 1.0 && three_out[2] == cancellation[2]);
    dsl_compiled_program_free(p);
    x.dtype = ME_INT64;
    rejected("def kernel(x):\n    s = sum(x)\n    if x < 0:\n        return s + 1\n    return s\n", &x, 1, "ambiguous block-scalar");
    rejected("def kernel(x):\n    s = sum(x)\n    for i in range(x):\n        return s\n    return s + 1\n", &x, 1, "ambiguous block-scalar");
    rejected("def kernel(x):\n    s = sum(x)\n    for i in range(3):\n        if x < 0:\n            break\n        return s + 1\n    return s\n", &x, 1, "ambiguous block-scalar");
    rejected("def kernel(x):\n    s = sum(x)\n    for i in range(3):\n        if x < 0:\n            continue\n        return s + 1\n    return s\n", &x, 1, "ambiguous block-scalar");
    p = compile("def kernel(x):\n    s = sum(x)\n    if all(x > 0):\n        return s + 1\n    return s\n", &x, 1, ME_INT64);
    int64_t coherent[] = {-1, 2};
    inputs[0] = coherent;
    me_dsl_portable_eval_descriptor scalar_desc = {2, NULL, sizeof(int64_t), 0, NULL, NULL, NULL};
    assert(dsl_eval_program_portable(p, inputs, 1, result, &scalar_desc) == 0 && result[0] == 1);
    coherent[0] = 1;
    assert(dsl_eval_program_portable(p, inputs, 1, result, &scalar_desc) == 0 && result[0] == 4);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    s = sum(x)\n    for i in range(3):\n        if s > 0:\n            return s + i\n    return s\n", &x, 1, ME_INT64);
    assert(dsl_eval_program_portable(p, inputs, 1, result, &scalar_desc) == 0 && result[0] == 3);
    dsl_compiled_program_free(p);
    int64_t masked[] = {INT64_MIN, 3, 4};
    inputs[0] = masked;
    p = compile("def kernel(x):\n    if x > 0:\n        if all(x < 4):\n            return 1\n        return 2\n    return 0\n", &x, 1, ME_INT64);
    int64_t three_result[3];
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 0 && three_result[1] == 2 && three_result[2] == 2);
    masked[2] = 2;
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[1] == 1 && three_result[2] == 1);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    if x > 0:\n        if all(-x < 0):\n            return 1\n    return 0\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 0 && three_result[1] == 1 && three_result[2] == 1);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    a = x\n    for i in range(3):\n        if a < 0:\n            return a\n        if all(a > 1):\n            break\n        a = a + 1\n    return a\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == INT64_MIN && three_result[1] == 3 && three_result[2] == 2);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    a = x\n    for i in range(3):\n        if i == 1:\n            continue\n        a = a + 1\n    return a\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == INT64_MIN + 2 && three_result[1] == 5 && three_result[2] == 4);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return sum(x)\n", &x, 1, ME_INT64);
    assert(p->output_is_scalar);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == INT64_MIN + 5);
    masked[0] = INT64_MAX;
    assert(eval(p, inputs, 1, three_result, 3) != 0);
    dsl_compiled_program_free(p);
    rejected("def kernel(x):\n    if x > 0:\n        return sum(x)\n    return 0\n", &x, 1, "reductions");

    /* Empty-group identities are exercised directly until the extended
     * block-scalar descriptor replaces the old nitems-output ABI. */
    const char *reductions[] = {"sum", "prod", "any", "all", "mean", "min", "max"};
    for (size_t i = 0; i < sizeof(reductions) / sizeof(reductions[0]); i++) {
        char source[128];
        snprintf(source, sizeof(source), "def kernel(x):\n    return %s(x)\n", reductions[i]);
        p = compile(source, &x, 1, ME_AUTO);
        me_expr *expr = p->block.stmts[0]->as.return_stmt.expr.expr;
        union { int64_t integer; bool truth; double real; } empty_out = {.integer = -1};
        void *destination = i < 2 || i >= 5 ? (void *)&empty_out.integer :
                            i < 4 ? (void *)&empty_out.truth : (void *)&empty_out.real;
        int rc = dsl_portable_eval_expr(expr, inputs, 1, NULL, 0, 0, NULL, destination);
        if (i >= 5) assert(rc != 0);
        else {
            assert(rc == 0);
            if (i < 2) assert(empty_out.integer == (i == 0 ? 0 : 1));
            else if (i < 4) assert(empty_out.truth == (i == 3));
            else assert(isnan(empty_out.real));
        }
        dsl_compiled_program_free(p);
    }
    int64_t join_inputs[] = {-7, 7};
    inputs[0] = join_inputs;
    p = compile("def kernel(x):\n    a = x\n    a = a / 2\n    return a\n", &x, 1, ME_AUTO);
    assert(p->output_dtype == ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == -3.5 && out[1] == 3.5);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    a = x\n    for i in range(3):\n        a = a / 2\n    return a\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == -0.875 && out[1] == 0.875);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    if x < 0:\n        return x / 2\n    return x\n", &x, 1, ME_AUTO);
    assert(p->output_dtype == ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == -3.5 && out[1] == 7.0);
    dsl_compiled_program_free(p);
    rejected("def kernel(x):\n    return x + _flat_idx\n", &x, 1, "ND context");
    rejected("def kernel(x):\n    if x > 0:\n        return x\n    return sum(x)\n", &x, 1, "cardinality");

    x.dtype = ME_UINT64;
    uint64_t wide[] = {UINT64_C(9007199254740993), UINT64_MAX};
    uint64_t unsigned_out[2];
    inputs[0] = wide;
    p = compile("def kernel(x):\n    return x - 9007199254740993\n", &x, 1, ME_UINT64);
    assert(eval(p, inputs, 1, unsigned_out, 2) == 0 && unsigned_out[0] == 0 &&
           unsigned_out[1] == UINT64_C(18437736874454810622));
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x == 18446744073709551615\n", &x, 1, ME_BOOL);
    assert(eval(p, inputs, 1, bool_out, 2) == 0 && !bool_out[0] && bool_out[1]);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x + 1\n", &x, 1, ME_UINT64);
    assert(eval(p, inputs, 1, unsigned_out, 2) != 0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x\n", &x, 1, ME_INT64);
    assert(eval(p, inputs, 1, result, 2) != 0);
    dsl_compiled_program_free(p);

    x.dtype = ME_FLOAT64;
    double remainder_inputs[] = {-7.0, 7.0};
    inputs[0] = remainder_inputs;
    p = compile("def kernel(x):\n    return fmod(x, 3.0)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == -1.0 && out[1] == 1.0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x % 3.0\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 2.0 && out[1] == 1.0);
    dsl_compiled_program_free(p);
    remainder_inputs[0] = NAN;
    remainder_inputs[1] = 0.0;
    p = compile("def kernel(x):\n    return bool(x)\n", &x, 1, ME_BOOL);
    assert(eval(p, inputs, 1, bool_out, 2) == 0 && bool_out[0] && !bool_out[1]);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x == x\n", &x, 1, ME_BOOL);
    assert(eval(p, inputs, 1, bool_out, 2) == 0 && !bool_out[0] && bool_out[1]);
    dsl_compiled_program_free(p);
    remainder_inputs[0] = 2.5;
    remainder_inputs[1] = -2.5;
    p = compile("def kernel(x):\n    return rint(x)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 2.0 && out[1] == -2.0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return round(x)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 3.0 && out[1] == -3.0);
    dsl_compiled_program_free(p);
    x.name = "log";
    p = compile("def kernel(log):\n    return log\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 2.5 && out[1] == -2.5);
    dsl_compiled_program_free(p);
    fenv_t caller_environment;
    assert(fegetenv(&caller_environment) == 0);
    /* wasm has fixed nearest rounding and no hardware exception flags. Keep
     * arithmetic/literal/restoration checks there; native hosts also exercise
     * restoration from a deliberately nondefault caller environment. */
#if defined(__EMSCRIPTEN__)
    int caller_rounding = FE_TONEAREST;
#else
    int caller_rounding = FE_UPWARD;
#endif
    assert(fesetround(caller_rounding) == 0);
#if defined(FE_INVALID)
    assert(feraiseexcept(FE_INVALID) == 0);
#endif
    int caller_flags = fetestexcept(FE_ALL_EXCEPT);
    x.name = "x";
    x.dtype = ME_FLOAT32;
    float rounding_inputs[] = {16777216.0f, 0.0f};
    inputs[0] = rounding_inputs;
    p = compile("def kernel(x):\n    return (x + 1.0) - x\n", &x, 1, ME_FLOAT64);
    assert(fegetround() == caller_rounding && fetestexcept(FE_ALL_EXCEPT) == caller_flags);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[0] == 0.0 && out[1] == 1.0);
    assert(fegetround() == caller_rounding && fetestexcept(FE_ALL_EXCEPT) == caller_flags);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x + 0.1\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 2) == 0 && out[1] == (double)0x1.99999ap-4f);
    assert(fegetround() == caller_rounding && fetestexcept(FE_ALL_EXCEPT) == caller_flags);
    dsl_compiled_program_free(p);
    assert(fesetenv(&caller_environment) == 0);
    x.dtype = ME_INT64;
    int64_t combinatoric_values[] = {20, 21};
    inputs[0] = combinatoric_values;
    p = compile("def kernel(x):\n    return fac(x)\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, result, 1) == 0 && result[0] == INT64_C(2432902008176640000));
    assert(eval(p, inputs, 1, result, 2) != 0);
    combinatoric_values[0] = -1;
    assert(eval(p, inputs, 1, result, 1) != 0);
    dsl_compiled_program_free(p);
    combinatoric_values[0] = 5;
    p = compile("def kernel(x):\n    return npr(x, 2)\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, result, 1) == 0 && result[0] == 20);
    dsl_compiled_program_free(p);
    x.dtype = ME_UINT64;
    uint64_t binomial_input[] = {67};
    inputs[0] = binomial_input;
    p = compile("def kernel(x):\n    return ncr(x, 33)\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, unsigned_out, 1) == 0 && unsigned_out[0] == UINT64_C(14226520737620288370));
    dsl_compiled_program_free(p);
    x.dtype = ME_FLOAT32;
    float fused_input[] = {0x1.000002p0f};
    inputs[0] = fused_input;
    p = compile("def kernel(x):\n    return fma(x, 2.0 - x, -1.0)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 1) == 0 && out[0] == (double)-0x1p-46f);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return ldexp(x, 3)\n", &x, 1, ME_FLOAT64);
    assert(eval(p, inputs, 1, out, 1) == 0 && out[0] == (double)0x1.000002p3f);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return logaddexp(x, x)\n", &x, 1, ME_FLOAT64);
    fused_input[0] = -INFINITY;
    assert(eval(p, inputs, 1, out, 1) == 0 && isinf(out[0]) && out[0] < 0.0);
    fused_input[0] = INFINITY;
    assert(eval(p, inputs, 1, out, 1) == 0 && isinf(out[0]) && out[0] > 0.0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return pi + e\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, out, 1) == 0 && out[0] == 0x1.921fb54442d18p1 + 0x1.5bf0a8b145769p1);
    dsl_compiled_program_free(p);
    x.dtype = ME_INT64;
    int64_t lazy_reduce_input[] = {INT64_MIN, 2, 3};
    inputs[0] = lazy_reduce_input;
    p = compile("def kernel(x):\n    return where(x > 0, sum(-x), 0)\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 0 && three_result[1] == -5 && three_result[2] == -5);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return x > 0 and all(-x < 0)\n", &x, 1, ME_BOOL);
    bool three_truth[3];
    assert(eval(p, inputs, 1, three_truth, 3) == 0 && !three_truth[0] && three_truth[1] && three_truth[2]);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return where(x > 2, sum(x), sum(where(x > 0, x, 0)))\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 2 && three_result[1] == 2 && three_result[2] == 3);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return sum(-x)\n", &x, 1, ME_AUTO);
    struct { int64_t value; int64_t sentinel; } scalar_result = {0, INT64_C(123456789)};
    uint8_t valid_mask[] = {0, 1, 1};
    me_dsl_portable_eval_descriptor descriptor = {
        .nitems = 3, .valid_mask = valid_mask, .output_capacity = sizeof(scalar_result.value)};
    assert(dsl_eval_program_portable(p, inputs, 1, &scalar_result.value, &descriptor) == 0 && scalar_result.value == -5);
    assert(scalar_result.sentinel == INT64_C(123456789));
    descriptor.output_capacity = 7;
    assert(dsl_eval_program_portable(p, inputs, 1, &scalar_result.value, &descriptor) != 0);
    descriptor.output_capacity = 8;
    valid_mask[1] = 0;
    valid_mask[2] = 0;
    assert(dsl_eval_program_portable(p, inputs, 1, &scalar_result.value, &descriptor) == 0 && scalar_result.value == 0);
    descriptor.nitems = 0;
    descriptor.valid_mask = NULL;
    const void *empty_inputs[] = {NULL};
    assert(dsl_eval_program_portable(p, empty_inputs, 1, &scalar_result.value, &descriptor) == 0 && scalar_result.value == 0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    a = sum(x)\n    return a + 1\n", &x, 1, ME_AUTO);
    assert(p->output_is_scalar);
    assert(dsl_eval_program_portable(p, empty_inputs, 1, &scalar_result.value, &descriptor) == 0 && scalar_result.value == 1);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return mean(x)\n", &x, 1, ME_AUTO);
    assert(dsl_eval_program_portable(p, empty_inputs, 1, out, &descriptor) == 0 && isnan(out[0]));
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    return min(x)\n", &x, 1, ME_AUTO);
    assert(dsl_eval_program_portable(p, empty_inputs, 1, &scalar_result.value, &descriptor) != 0);
    dsl_compiled_program_free(p);
    x.dtype = ME_INT64;
    int64_t definition_inputs[] = {-1, 2, 3};
    inputs[0] = definition_inputs;
    p = compile("def kernel(x):\n    if x > 0:\n        a = x\n    return a\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, three_result, 3) != 0);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    if x > 0:\n        a = x\n    return where(x > 0, a, 0)\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 0 && three_result[1] == 2 && three_result[2] == 3);
    dsl_compiled_program_free(p);
    p = compile("def kernel(x):\n    if x > 0:\n        a = x\n    return sum(a)\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, three_result, 3) != 0);
    valid_mask[0] = 0;
    valid_mask[1] = 1;
    valid_mask[2] = 1;
    descriptor = (me_dsl_portable_eval_descriptor){
        .nitems = 3, .valid_mask = valid_mask, .output_capacity = sizeof(scalar_result.value)};
    assert(dsl_eval_program_portable(p, inputs, 1, &scalar_result.value, &descriptor) == 0 && scalar_result.value == 5);
    dsl_compiled_program_free(p);
    x.dtype = ME_UINT64;
    uint64_t range_inputs[] = {UINT64_MAX};
    inputs[0] = range_inputs;
    p = compile("def kernel(x):\n    a = 0\n    for i in range(x):\n        a += 1\n    return a\n", &x, 1, ME_AUTO);
    assert(eval(p, inputs, 1, result, 1) != 0);
    dsl_compiled_program_free(p);
    x.dtype = ME_FLOAT64;
    double range_float_inputs[] = {NAN, INFINITY, 0x1p63};
    inputs[0] = range_float_inputs;
    p = compile("def kernel(x):\n    a = 0\n    for i in range(x):\n        a += 1\n    return a\n", &x, 1, ME_AUTO);
    for (int i = 0; i < 3; i++) {
        inputs[0] = &range_float_inputs[i];
        assert(eval(p, inputs, 1, result, 1) != 0);
    }
    dsl_compiled_program_free(p);
    masked_iteration_fixture();
    concurrent_handle_fixture();
    pi_math_fixture();
    operation_domain_fixture();
    independent_anchor_fixture();
    x.dtype = ME_INT64;
    inputs[0] = definition_inputs;
    p = compile("def k(x):\n    a = sum(x)\n    if x > 0:\n        a = 0\n    return a\n", &x, 1, ME_AUTO);
    assert(!p->output_is_scalar);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 4 && three_result[1] == 0 && three_result[2] == 0);
    dsl_compiled_program_free(p);
    p = compile("def k(x):\n    a = sum(x)\n    for i in range(x):\n        a += 1\n    return a\n", &x, 1, ME_AUTO);
    assert(!p->output_is_scalar);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 4 && three_result[1] == 6 && three_result[2] == 7);
    dsl_compiled_program_free(p);
    p = compile("def k(x):\n    a = sum(x)\n    for i in range(3):\n        if x < i:\n            break\n        a += 1\n    return a\n", &x, 1, ME_AUTO);
    assert(!p->output_is_scalar);
    assert(eval(p, inputs, 1, three_result, 3) == 0 && three_result[0] == 4 && three_result[1] == 7 && three_result[2] == 7);
    dsl_compiled_program_free(p);
    puts("portable typed interpreter tests passed");
    return 0;
}

static void masked_iteration_fixture(void) {
    me_variable vars[] = {{.name = "cx", .dtype = ME_FLOAT64}, {.name = "cy", .dtype = ME_FLOAT64}};
    const char *source = "def k(cx, cy):\n"
        "    zr = 0.0\n    zi = 0.0\n    count = 0\n"
        "    for i in range(20):\n"
        "        if zr*zr + zi*zi > 4.0:\n            break\n"
        "        next_r = zr*zr - zi*zi + cx\n"
        "        zi = 2.0*zr*zi + cy\n        zr = next_r\n        count += 1\n"
        "    return count\n";
    me_dsl_compiled_program *program = compile(source, vars, 2, ME_AUTO);
    double cx[] = {-2.0, -0.75, 0.0, 1.0, NAN, INFINITY};
    double cy[] = {0.0, 0.1, 0.0, 1.0, NAN, INFINITY};
    const void *inputs[] = {cx, cy};
    int64_t output[] = {-1, -1, -1, -1, -1, -1};
    uint8_t valid[] = {1, 1, 1, 1, 0, 0};
    me_dsl_portable_eval_descriptor descriptor = {
        .nitems = 6, .valid_mask = valid, .output_capacity = sizeof(output)};
    assert(!program->output_is_scalar);
    assert(dsl_eval_program_portable(program, inputs, 2, output, &descriptor) == 0);
    for (int j = 0; j < 4; j++) {
        double zr = 0.0, zi = 0.0;
        int64_t count = 0;
        for (int i = 0; i < 20; i++) {
            if (zr*zr + zi*zi > 4.0) break;
            double next = zr*zr - zi*zi + cx[j];
            zi = 2.0*zr*zi + cy[j];
            zr = next;
            count++;
        }
        assert(output[j] == count);
    }
    assert(output[4] == -1 && output[5] == -1);
    for (int j = 0; j < 4; j += 2) {
        const void *partition[] = {cx + j, cy + j};
        int64_t partial[2];
        descriptor = (me_dsl_portable_eval_descriptor){.nitems = 2, .output_capacity = sizeof(partial)};
        assert(dsl_eval_program_portable(program, partition, 2, partial, &descriptor) == 0);
        assert(partial[0] == output[j] && partial[1] == output[j + 1]);
    }
    dsl_compiled_program_free(program);
}

#if (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
typedef struct {
    me_dsl_compiled_program *program;
    int rounding;
    int64_t values[3];
    int64_t expected;
} concurrent_case;

static void *concurrent_evaluate(void *arg) {
    concurrent_case *test = arg;
    assert(fesetround(test->rounding) == 0);
    assert(feraiseexcept(FE_INVALID) == 0);
    int flags = fetestexcept(FE_ALL_EXCEPT);
    const void *inputs[] = {test->values};
    for (int i = 0; i < 50; i++) {
        int64_t output = 0;
        me_dsl_portable_eval_descriptor descriptor = {.nitems = 3, .output_capacity = sizeof(output)};
        assert(dsl_eval_program_portable(test->program, inputs, 1, &output, &descriptor) == 0);
        assert(output == test->expected);
        assert(fegetround() == test->rounding && fetestexcept(FE_ALL_EXCEPT) == flags);
        int64_t invalid[] = {INT64_MAX, 1, 0};
        const void *invalid_inputs[] = {invalid};
        assert(dsl_eval_program_portable(test->program, invalid_inputs, 1, &output, &descriptor) != 0);
        assert(fegetround() == test->rounding && fetestexcept(FE_ALL_EXCEPT) == flags);
    }
    return NULL;
}
#endif

static void concurrent_handle_fixture(void) {
#if (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
    me_variable x = {.name = "x", .dtype = ME_INT64};
    me_dsl_compiled_program *program = compile("def k(x):\n    a = sum(x)\n    return a + 1\n", &x, 1, ME_AUTO);
    concurrent_case cases[] = {{program, FE_DOWNWARD, {1, 2, 3}, 7}, {program, FE_UPWARD, {4, 5, 6}, 16}};
    pthread_t threads[2];
    for (int i = 0; i < 2; i++) assert(pthread_create(&threads[i], NULL, concurrent_evaluate, &cases[i]) == 0);
    for (int i = 0; i < 2; i++) assert(pthread_join(threads[i], NULL) == 0);
    dsl_compiled_program_free(program);
#endif
}

static void pi_math_fixture(void) {
    me_variable x = {.name = "x", .dtype = ME_FLOAT64};
    double values[] = {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0x1p50, 0x1p40 + 0.5};
    double expected_sine[] = {0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 0.0, 1.0};
    double expected_cosine[] = {1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0};
    const void *inputs[] = {values};
    double output[8];
    me_dsl_compiled_program *program = compile("def k(x):\n    return sinpi(x)\n", &x, 1, ME_AUTO);
    assert(eval(program, inputs, 1, output, 8) == 0);
    for (int i = 0; i < 8; i++) assert(output[i] == expected_sine[i] && !!signbit(output[i]) == !!signbit(expected_sine[i]));
    dsl_compiled_program_free(program);
    program = compile("def k(x):\n    return cospi(x)\n", &x, 1, ME_AUTO);
    assert(eval(program, inputs, 1, output, 8) == 0);
    for (int i = 0; i < 8; i++) assert(output[i] == expected_cosine[i]);
    values[0] = nextafter(0.5, 0.0);
    assert(eval(program, inputs, 1, output, 1) == 0 && output[0] > 0.0);
    assert(fabs(output[0] - sin(0x1.921fb54442d18p1 * (0.5 - values[0]))) <= 0x1p-100);
    values[0] = INFINITY;
    assert(eval(program, inputs, 1, output, 1) == 0 && isnan(output[0]));
    dsl_compiled_program_free(program);
    x.dtype = ME_FLOAT32;
    float floats[] = {0x1p23f, 0x1p20f + 0.5f, -0.5f};
    inputs[0] = floats;
    program = compile("def k(x):\n    return sinpi(x)\n", &x, 1, ME_FLOAT64);
    assert(eval(program, inputs, 1, output, 3) == 0 && output[0] == 0.0 && output[1] == 1.0 && output[2] == -1.0);
    dsl_compiled_program_free(program);
}

/* Finite host-libm/type/domain matrix, not a claim of cross-platform accuracy.
 * These fixtures certify interpreter result precision and exceptional domains;
 * independent platform ULP certification remains a release matrix gate. */
static void operation_domain_fixture(void) {
    struct { const char *name; double (*reference)(double); double argument; } cases[] = {
        {"sin", sin, 0.5}, {"cos", cos, 0.5}, {"tan", tan, 0.5},
        {"asin", asin, 0.5}, {"acos", acos, 0.5}, {"atan", atan, 0.5},
        {"sinh", sinh, 0.5}, {"cosh", cosh, 0.5}, {"tanh", tanh, 0.5},
        {"asinh", asinh, 0.5}, {"acosh", acosh, 1.5}, {"atanh", atanh, 0.5},
        {"exp", exp, 0.5}, {"exp2", exp2, 0.5}, {"expm1", expm1, 0.5},
        {"log", log, 0.5}, {"log2", log2, 0.5}, {"log10", log10, 0.5},
        {"log1p", log1p, 0.5}, {"sqrt", sqrt, 0.5}, {"cbrt", cbrt, -0.5},
        {"erf", erf, 0.5}, {"erfc", erfc, 0.5}, {"abs", fabs, -0.5},
        {"ceil", ceil, -0.5}, {"floor", floor, -0.5}
    };
    for (int precision = 0; precision < 2; precision++) {
        me_dtype dtype = precision ? ME_FLOAT64 : ME_FLOAT32;
        me_variable variable = {.name = "x", .dtype = dtype, .type = ME_VARIABLE};
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            char source[128];
            snprintf(source, sizeof(source), "def k(x):\n    return %s(x)\n", cases[i].name);
            me_dsl_compiled_program *program = compile(source, &variable, 1, ME_AUTO);
            assert(program->output_dtype == dtype);
            double input64 = cases[i].argument, output64 = 0;
            float input32 = (float)input64, output32 = 0;
            const void *inputs[] = {precision ? (void *)&input64 : (void *)&input32};
            assert(eval(program, inputs, 1, precision ? (void *)&output64 : (void *)&output32, 1) == 0);
            double expected = cases[i].reference(input64);
            double actual = precision ? output64 : output32;
            double tolerance = 8 * (precision ? DBL_EPSILON : FLT_EPSILON) * fmax(fabs(expected), DBL_MIN);
            assert(fabs(actual - expected) <= tolerance);
            dsl_compiled_program_free(program);
        }
        const char *domains[] = {"sqrt", "log", "acos", "asin", "acosh", "atanh"};
        for (size_t i = 0; i < sizeof(domains) / sizeof(domains[0]); i++) {
            char source[128];
            snprintf(source, sizeof(source), "def k(x):\n    return %s(x)\n", domains[i]);
            me_dsl_compiled_program *program = compile(source, &variable, 1, ME_AUTO);
            double input64 = -2, output64;
            float input32 = -2, output32;
            const void *inputs[] = {precision ? (void *)&input64 : (void *)&input32};
            assert(eval(program, inputs, 1, precision ? (void *)&output64 : (void *)&output32, 1) == 0);
            assert(isnan(precision ? output64 : output32));
            dsl_compiled_program_free(program);
        }
    }
}

/* Analytic/exact constants, not references obtained from the implementation's
 * own libm call. Bounds here are regression criteria, not platform guarantees. */
static void independent_anchor_fixture(void) {
    struct { const char *name; double x, expected; } unary[] = {
        {"sin", 0, 0}, {"cos", 0, 1}, {"tan", 0, 0},
        {"asin", 0, 0}, {"acos", 1, 0}, {"atan", 0, 0},
        {"sinh", 0, 0}, {"cosh", 0, 1}, {"tanh", 0, 0},
        {"asinh", 0, 0}, {"acosh", 1, 0}, {"atanh", 0, 0},
        {"exp", 0, 1}, {"exp2", 3, 8}, {"exp10", 2, 100}, {"expm1", 0, 0},
        {"log", 1, 0}, {"ln", 1, 0}, {"log2", 8, 3}, {"log10", 100, 2}, {"log1p", 0, 0},
        {"sqrt", 4, 2}, {"cbrt", 8, 2}, {"erf", 0, 0}, {"erfc", 0, 1},
        {"lgamma", 1, 0}, {"tgamma", 5, 24}, {"sinpi", 0.5, 1}, {"cospi", 1, -1},
        {"abs", -2, 2}, {"square", -2, 4}, {"sign", -2, -1},
        {"ceil", -2.5, -2}, {"floor", -2.5, -3}, {"trunc", -2.5, -2},
        {"rint", 2.5, 2}, {"round", 2.5, 3}, {"conj", -2, -2}, {"real", -2, -2}, {"imag", -2, 0},
        {"rint", -0.5, -0.0}, {"round", -0.25, -0.0}, {"sqrt", -0.0, -0.0}, {"sign", -0.0, -0.0},
        {"acos", 1, 0}, {"acosh", 1, 0}, {"asin", 0, 0},
        {"asinh", 0, 0}, {"atan", 0, 0}, {"atanh", 0, 0}
    };
    struct { const char *name; double x, y, expected; } binary[] = {
        {"atan2", 0, 1, 0}, {"atan2", 0, -1, 0x1.921fb54442d18p+1},
        {"atan2", 0, 1, 0}, {"copysign", 1, -1, -1}, {"fdim", 5, 3, 2},
        {"fmin", 5, 3, 3}, {"fmax", 5, 3, 5}, {"hypot", 3, 4, 5},
        {"fmod", 7, 2, 1}, {"remainder", 7, 2, -1}, {"ldexp", 1.5, 2, 6},
        {"logaddexp", 0, 0, 0x1.62e42fefa39efp-1}, {"pow", 2, 3, 8}, {"power", 2, 3, 8}
    };
    struct { const char *name; double x, y, expected; } specials[] = {
        {"fmin", NAN, 3, 3}, {"fmax", 3, NAN, 3}, {"fmin", NAN, NAN, NAN},
        {"fmax", NAN, NAN, NAN}, {"fmin", 0.0, -0.0, -0.0}, {"fmax", 0.0, -0.0, 0.0},
        {"fmax", -0.0, -0.0, -0.0}, {"copysign", 0, -1, -0.0},
        {"fdim", 1, 2, 0}, {"fdim", NAN, 2, NAN}, {"hypot", INFINITY, NAN, INFINITY},
        {"fmod", 7, 0, NAN}, {"remainder", 7, 0, NAN}, {"nextafter", 0.0, -0.0, -0.0},
        {"nextafter", NAN, 1, NAN}, {"pow", NAN, 0, 1}, {"power", 1, NAN, 1},
        {"logaddexp", INFINITY, INFINITY, INFINITY},
        {"logaddexp", -INFINITY, -INFINITY, -INFINITY}, {"logaddexp", NAN, 1, NAN}
    };
    for (int precision = 0; precision < 2; precision++) {
        me_dtype dtype = precision ? ME_FLOAT64 : ME_FLOAT32;
        me_variable vars[] = {{.name = "x", .dtype = dtype}, {.name = "y", .dtype = dtype},
                              {.name = "z", .dtype = dtype}};
        for (size_t i = 0; i < sizeof(unary) / sizeof(unary[0]); i++) {
            char source[128];
            snprintf(source, sizeof(source), "def k(x):\n    return %s(x)\n", unary[i].name);
            me_dsl_compiled_program *program = compile(source, vars, 1, ME_AUTO);
            if (program->output_dtype != dtype) {
                fprintf(stderr, "%s: expected dtype %d, got %d\n", source, (int)dtype,
                        (int)program->output_dtype);
            }
            assert(program->output_dtype == dtype);
            double x64 = unary[i].x, out64 = 0;
            float x32 = (float)x64, out32 = 0;
            const void *inputs[] = {precision ? (void *)&x64 : (void *)&x32};
            assert(eval(program, inputs, 1, precision ? (void *)&out64 : (void *)&out32, 1) == 0);
            double actual = precision ? out64 : out32;
            double bound = 8 * (precision ? DBL_EPSILON : FLT_EPSILON) * fabs(unary[i].expected);
            assert(fabs(actual - unary[i].expected) <= bound);
            if (unary[i].expected == 0) assert(!!signbit(actual) == !!signbit(unary[i].expected));
            dsl_compiled_program_free(program);
        }
        for (size_t i = 0; i < sizeof(binary) / sizeof(binary[0]); i++) {
            char source[128];
            snprintf(source, sizeof(source), "def k(x, y):\n    return %s(x, y)\n", binary[i].name);
            vars[1].dtype = !strcmp(binary[i].name, "ldexp") ? ME_INT64 : dtype;
            me_dsl_compiled_program *program = compile(source, vars, 2, ME_AUTO);
            assert(program->output_dtype == dtype);
            double x64 = binary[i].x, y64 = binary[i].y, out64 = 0;
            float x32 = (float)x64, y32 = (float)y64, out32 = 0;
            int64_t exponent = 2;
            const void *inputs[] = {precision ? (void *)&x64 : (void *)&x32,
                vars[1].dtype == ME_INT64 ? (void *)&exponent : precision ? (void *)&y64 : (void *)&y32};
            assert(eval(program, inputs, 2, precision ? (void *)&out64 : (void *)&out32, 1) == 0);
            double actual = precision ? out64 : out32;
            double bound = 8 * (precision ? DBL_EPSILON : FLT_EPSILON) * fabs(binary[i].expected);
            assert(fabs(actual - binary[i].expected) <= bound);
            dsl_compiled_program_free(program);
        }
        vars[1].dtype = dtype;
        for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
            char source[128];
            snprintf(source, sizeof(source), "def k(x, y):\n    return %s(x, y)\n", specials[i].name);
            me_dsl_compiled_program *program = compile(source, vars, 2, ME_AUTO);
            double x64 = specials[i].x, y64 = specials[i].y, out64 = 0;
            float x32 = (float)x64, y32 = (float)y64, out32 = 0;
            const void *inputs[] = {precision ? (void *)&x64 : (void *)&x32,
                                   precision ? (void *)&y64 : (void *)&y32};
            assert(eval(program, inputs, 2, precision ? (void *)&out64 : (void *)&out32, 1) == 0);
            double actual = precision ? out64 : out32;
            if (isnan(specials[i].expected)) assert(isnan(actual));
            else {
                assert(actual == specials[i].expected);
                if (actual == 0 || isinf(actual)) assert(!!signbit(actual) == !!signbit(specials[i].expected));
            }
            dsl_compiled_program_free(program);
        }
        vars[1].dtype = dtype;
        me_dsl_compiled_program *program = compile("def k(x, y):\n    return nextafter(x, y)\n", vars, 2, ME_AUTO);
        double x64 = 1, y64 = 2, out64;
        float x32 = 1, y32 = 2, out32;
        const void *pair[] = {precision ? (void *)&x64 : (void *)&x32, precision ? (void *)&y64 : (void *)&y32};
        assert(eval(program, pair, 2, precision ? (void *)&out64 : (void *)&out32, 1) == 0);
        assert(precision ? out64 == 0x1.0000000000001p+0 : out32 == 0x1.000002p+0f);
        dsl_compiled_program_free(program);
        program = compile("def k(x, y, z):\n    return fma(x, y, z)\n", vars, 3, ME_AUTO);
        x64 = 0x1.0000000000001p+0; y64 = 0x1.ffffffffffffep-1;
        x32 = 0x1.000002p+0f; y32 = 0x1.fffffcp-1f;
        double z64 = -1;
        float z32 = -1;
        const void *triple[] = {precision ? (void *)&x64 : (void *)&x32,
            precision ? (void *)&y64 : (void *)&y32, precision ? (void *)&z64 : (void *)&z32};
        assert(eval(program, triple, 3, precision ? (void *)&out64 : (void *)&out32, 1) == 0);
        assert(precision ? out64 == -0x1p-104 : out32 == -0x1p-46f);
        dsl_compiled_program_free(program);
    }
    const char *constants[] = {"pi", "e"};
    const double exact[] = {0x1.921fb54442d18p+1, 0x1.5bf0a8b145769p+1};
    for (int i = 0; i < 2; i++) {
        char source[64];
        snprintf(source, sizeof(source), "def k():\n    return %s()\n", constants[i]);
        me_dsl_compiled_program *program = compile(source, NULL, 0, ME_FLOAT64);
        double out;
        assert(eval(program, NULL, 0, &out, 1) == 0 && out == exact[i]);
        dsl_compiled_program_free(program);
    }
    me_variable integer = {.name = "x", .dtype = ME_INT8};
    me_dsl_compiled_program *program = compile("def k(x):\n    return ~x\n", &integer, 1, ME_AUTO);
    int8_t x = 127, out;
    const void *input[] = {&x};
    assert(eval(program, input, 1, &out, 1) == 0 && out == -128);
    dsl_compiled_program_free(program);
    program = compile("def k(x):\n    return fac(x)\n", &integer, 1, ME_AUTO);
    x = 5;
    assert(eval(program, input, 1, &out, 1) == 0 && out == 120);
    x = 6;
    assert(eval(program, input, 1, &out, 1) == ME_EVAL_ERR_INVALID_ARG);
    x = -1;
    assert(eval(program, input, 1, &out, 1) == ME_EVAL_ERR_INVALID_ARG);
    dsl_compiled_program_free(program);
    struct { me_dtype dtype; unsigned limit; uint64_t expected; } factorials[] = {
        {ME_INT8, 5, 120}, {ME_UINT8, 5, 120}, {ME_INT16, 7, 5040}, {ME_UINT16, 8, 40320},
        {ME_INT32, 12, 479001600}, {ME_UINT32, 12, 479001600},
        {ME_INT64, 20, UINT64_C(2432902008176640000)}, {ME_UINT64, 20, UINT64_C(2432902008176640000)}
    };
    for (size_t i = 0; i < sizeof(factorials) / sizeof(factorials[0]); i++) {
        integer.dtype = factorials[i].dtype;
        program = compile("def k(x):\n    return fac(x)\n", &integer, 1, ME_AUTO);
        assert(program->output_dtype == integer.dtype);
        /* Width-specific native fields, not low bytes of uint64 on little-endian. */
        union { uint8_t u8; uint16_t u16; uint32_t u32; uint64_t u64; } value, result;
        size_t width = dtype_size(integer.dtype);
        for (unsigned delta = 0; delta < 2; delta++) {
            unsigned n = factorials[i].limit + delta;
            if (width == 1) value.u8 = (uint8_t)n;
            else if (width == 2) value.u16 = (uint16_t)n;
            else if (width == 4) value.u32 = n;
            else value.u64 = n;
            const void *inputs[] = {&value};
            int rc = eval(program, inputs, 1, &result, 1);
            if (delta) assert(rc == ME_EVAL_ERR_INVALID_ARG);
            else {
                assert(rc == 0);
                uint64_t actual = width == 1 ? result.u8 : width == 2 ? result.u16 :
                    width == 4 ? result.u32 : result.u64;
                assert(actual == factorials[i].expected);
            }
        }
        dsl_compiled_program_free(program);
    }
    integer.dtype = ME_UINT64;
    program = compile("def k(x):\n    return ~x\n", &integer, 1, ME_AUTO);
    uint64_t unsigned_input[] = {UINT64_MAX, 0}, unsigned_output[2];
    input[0] = unsigned_input;
    assert(eval(program, input, 1, unsigned_output, 2) == 0);
    assert(unsigned_output[0] == 0 && unsigned_output[1] == UINT64_MAX);
    dsl_compiled_program_free(program);
    integer.dtype = ME_BOOL;
    program = compile("def k(x):\n    return ~x\n", &integer, 1, ME_AUTO);
    bool boolean_input[] = {true, false}, boolean_output[2];
    input[0] = boolean_input;
    assert(eval(program, input, 1, boolean_output, 2) == 0);
    assert(!boolean_output[0] && boolean_output[1]);
    dsl_compiled_program_free(program);
}
