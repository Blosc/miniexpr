/* Regression for #26: integral / computes in float64 throughout the tree. */
#include "../src/miniexpr.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

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

static void check(me_dtype dtype, int form, me_dtype requested, bool scalar, bool zeros) {
    const char *sources[] = {"a / b", "a / b + 0.25", "(a / b) * b",
                            "a / b > 0.25", "sum(a / b)", "a / 2", "2 / b",
                            "(a + b) / b", "a / (b + b)"};
    double a[N] = {0}, b[N] = {0}, out[N] = {0}, expected[N], sum = 0;
    for (int i = 0; i < N; i++) {
        int av = dtype == ME_BOOL ? i % 2 : i % 4 + 1;
        int bv = dtype == ME_BOOL ? 1 : 4;
        if (dtype >= ME_INT8 && dtype <= ME_INT64 && i % 2) av = -av;
        if (zeros && i % 3 == 0) bv = 0;
        if (zeros && i % 6 == 0) av = 0;
        store(a, dtype, i, av);
        store(b, dtype, i, bv);
        double value = (double)av / bv;
        switch (form) {
        case 1: value += 0.25; break;
        case 2: value *= bv; break;
        case 3: value = value > 0.25; break;
        case 5: value = (double)av / 2; break;
        case 6: value = 2.0 / bv; break;
        case 7: value = (double)(av + bv) / bv; break;
        case 8: value = (double)av / (bv + bv); break;
        default: break;
        }
        expected[i] = value;
        sum += value;
    }
    me_dtype output = form == 3 ? ME_BOOL : requested == ME_AUTO ?
                      (dtype == ME_FLOAT32 ? ME_FLOAT32 : ME_FLOAT64) : requested;
    me_variable vars[] = {{"a", dtype}, {"b", dtype}};
    const void *ptrs[] = {a, b};
    int nvars = form == 5 || form == 6 ? 1 : 2;
    if (form == 6) {
        vars[0] = vars[1];
        ptrs[0] = ptrs[1];
    }
    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(sources[form], vars, nvars, form == 3 ? ME_BOOL : requested, &err, &expr) != ME_COMPILE_SUCCESS ||
        me_get_dtype(expr) != output) {
        printf("FAIL compile/dtype: %s input=%d requested=%d\n", sources[form], dtype, requested);
        failures++;
        me_free(expr);
        return;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.disable_simd = scalar;
    params.jit_mode = ME_JIT_OFF;
    /* Repeated evaluation also catches stale promoted bindings/buffers. */
    for (int repeat = 0; repeat < 2; repeat++) {
        int rc = me_eval(expr, ptrs, nvars, out, N, &params);
        if (rc != ME_EVAL_SUCCESS) {
            printf("FAIL eval: %s input=%d rc=%d\n", sources[form], dtype, rc);
            failures++;
            break;
        }
        for (int i = 0; i < (form == 4 ? 1 : N); i++) {
            double actual = output == ME_BOOL ? ((bool *)out)[i] :
                            output == ME_FLOAT32 ? ((float *)out)[i] : out[i];
            double want = form == 4 ? sum : expected[i];
            double tolerance = output == ME_FLOAT32 ? 2e-6 : 2e-13;
            bool equal = isnan(want) ? isnan(actual) : isinf(want) ? actual == want :
                         isfinite(actual) && fabs(actual - want) <= tolerance * fmax(1, fabs(want));
            if (!equal) {
                printf("FAIL value: %s input=%d lane=%d got=%.17g expected=%.17g\n",
                       sources[form], dtype, i, actual, want);
                failures++;
                break;
            }
        }
    }
    me_free(expr);
}

static void check_exact_integer_subtrees(void) {
    /* Converting leaves before subtraction would lose this unit difference. */
    int64_t a[] = {INT64_C(9007199254740993)}, b[] = {INT64_C(9007199254740992)};
    const void *ptrs[] = {a, b};
    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_INT64}};
    const char *sources[] = {"(a - b) / 2", "2 / (a - b)", "((a - b) / 2) > 0.25"};
    for (int i = 0; i < 3; i++) {
        me_expr *expr = NULL;
        int err = 0;
        double out = 0;
        if (me_compile(sources[i], vars, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS ||
            me_eval(expr, ptrs, 2, &out, 1, NULL) != ME_EVAL_SUCCESS ||
            (i == 2 ? !*(bool *)&out : out != (i == 0 ? 0.5 : 2.0))) {
            printf("FAIL exact integer subtree: %s\n", sources[i]);
            failures++;
        }
        me_free(expr);
    }
}

int main(void) {
    const me_dtype types[] = {ME_BOOL, ME_INT8, ME_UINT8, ME_INT16, ME_UINT16,
                             ME_INT32, ME_UINT32, ME_INT64, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    const me_dtype outputs[] = {ME_AUTO, ME_FLOAT32, ME_FLOAT64};
    for (unsigned t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        for (int form = 0; form < 9; form++) {
            /* Boolean arithmetic inside a numerator/denominator has separate
             * classic-expression rules; the issue concerns division itself. */
            if (types[t] == ME_BOOL && form >= 7) continue;
            for (unsigned o = 0; o < sizeof(outputs) / sizeof(outputs[0]); o++) {
                for (int scalar = 0; scalar < 2; scalar++) {
                    check(types[t], form, outputs[o], scalar != 0, false);
                    if (form < 4) check(types[t], form, outputs[o], scalar != 0, true);
                }
            }
        }
    }
    check_exact_integer_subtrees();
    printf("True division: %d failures\n", failures);
    return failures != 0;
}
