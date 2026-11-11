/* Regression for #27: binary real math must not run in integer arithmetic. */
#include "../src/miniexpr.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 257
static int failures;

static void store(void *buf, me_dtype dtype, int i, int value) {
    switch (dtype) {
    case ME_BOOL: ((bool *)buf)[i] = value != 0; break;
    case ME_INT8: ((int8_t *)buf)[i] = (int8_t)value; break;
    case ME_UINT8: ((uint8_t *)buf)[i] = (uint8_t)value; break;
    case ME_INT16: ((int16_t *)buf)[i] = (int16_t)value; break;
    case ME_UINT16: ((uint16_t *)buf)[i] = (uint16_t)value; break;
    case ME_INT32: ((int32_t *)buf)[i] = value; break;
    case ME_UINT32: ((uint32_t *)buf)[i] = (uint32_t)value; break;
    case ME_INT64: ((int64_t *)buf)[i] = value; break;
    case ME_UINT64: ((uint64_t *)buf)[i] = (uint64_t)value; break;
    case ME_FLOAT32: ((float *)buf)[i] = (float)value; break;
    case ME_FLOAT64: ((double *)buf)[i] = value; break;
    default: break;
    }
}

static double reference(int fn, double a, double b, bool single) {
    switch (fn) {
    case 0: return single ? hypotf((float)a, (float)b) : hypot(a, b);
    case 1: return single ? atan2f((float)a, (float)b) : atan2(a, b);
    case 2: return single ? (float)(fmax(a, b) + log1p(exp(-fabs(a - b)))) :
                           fmax(a, b) + log1p(exp(-fabs(a - b)));
    case 3: return single ? nextafterf((float)a, (float)b) : nextafter(a, b);
    default: return copysign(a, b);
    }
}

static void check(me_dtype dtype, me_dtype result_dtype, int fn, int form, bool scalar) {
    static const char *names[] = {"hypot", "atan2", "logaddexp", "nextafter", "copysign"};
    double a[N] = {0}, b[N] = {0}, out[N] = {0};
    char call[80], source[100];
    snprintf(call, sizeof(call), "%s(a, b)", names[fn]);
    snprintf(source, sizeof(source), form == 1 ? "%s > 2" : form == 2 ? "sum(%s)" :
             form == 3 ? "(%s + 0.25) * 2" : "%s", call);
    double expected[N], sum = 0;
    for (int i = 0; i < N; i++) {
        int av = dtype == ME_BOOL ? i % 2 : i % 4 + 1;
        int bv = dtype == ME_BOOL ? 1 : 4;
        if (dtype >= ME_INT8 && dtype <= ME_INT64 && i % 2) bv = -bv;
        store(a, dtype, i, av);
        store(b, dtype, i, bv);
        expected[i] = reference(fn, av, bv, result_dtype == ME_FLOAT32);
        sum += expected[i];
    }
    me_variable vars[] = {{"a", dtype}, {"b", dtype}};
    const void *ptrs[] = {a, b};
    me_expr *expr = NULL;
    int err = 0;
    me_dtype output_dtype = form == 1 ? ME_BOOL : result_dtype;
    if (me_compile(source, vars, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS ||
        me_get_dtype(expr) != output_dtype) {
        printf("FAIL inference: %s input=%d\n", source, dtype);
        failures++;
        me_free(expr);
        return;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.disable_simd = scalar;
    params.jit_mode = ME_JIT_OFF;
    if (me_eval(expr, ptrs, 2, out, N, &params) != ME_EVAL_SUCCESS) {
        printf("FAIL evaluation: %s input=%d\n", source, dtype);
        failures++;
    } else {
        for (int i = 0; i < (form == 2 ? 1 : N); i++) {
            double actual = output_dtype == ME_BOOL ? ((bool *)out)[i] :
                            output_dtype == ME_FLOAT32 ? ((float *)out)[i] : out[i];
            double want = form == 1 ? expected[i] > 2 : form == 2 ? sum :
                          form == 3 ? (expected[i] + 0.25) * 2 : expected[i];
            double tolerance = output_dtype == ME_FLOAT32 ? 2e-6 : 2e-13;
            if (fn == 3 && form != 2) tolerance = 0;
            if (!isfinite(actual) || fabs(actual - want) > tolerance * fmax(1, fabs(want))) {
                printf("FAIL value: %s input=%d lane=%d got=%.17g expected=%.17g\n",
                       source, dtype, i, actual, want);
                failures++;
                break;
            }
        }
    }
    me_free(expr);

    /* Explicit float64 output must not truncate integer-input functions either. */
    if (form == 0 && result_dtype == ME_FLOAT64) {
        expr = NULL;
        if (me_compile(source, vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS ||
            me_eval(expr, ptrs, 2, out, N, &params) != ME_EVAL_SUCCESS ||
            fabs(out[0] - expected[0]) > 2e-13) {
            printf("FAIL explicit output: %s input=%d\n", source, dtype);
            failures++;
        }
        me_free(expr);
    }
}

int main(void) {
    const me_dtype types[] = {ME_BOOL, ME_INT8, ME_UINT8, ME_INT16, ME_UINT16,
                             ME_INT32, ME_UINT32, ME_INT64, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    for (unsigned t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        me_dtype dt = types[t];
        me_dtype result = dt == ME_BOOL || dt == ME_INT8 || dt == ME_UINT8 ||
                          dt == ME_INT16 || dt == ME_UINT16 || dt == ME_FLOAT32 ?
                          ME_FLOAT32 : ME_FLOAT64;
        for (int fn = 0; fn < 5; fn++) {
            for (int form = 0; form < (result == ME_FLOAT64 ? 4 : 3); form++) {
                for (int scalar = 0; scalar < 2; scalar++) {
                    check(dt, result, fn, form, scalar != 0);
                }
            }
        }
    }
    printf("Binary floating math: %d failures\n", failures);
    return failures != 0;
}
