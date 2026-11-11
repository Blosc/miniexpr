/*
 * and/or/not with a numeric (non-bool) operand.  The bool evaluator gave up
 * on such an operand after allocating 1-byte operand buffers, and the
 * numeric evaluation that followed wrote int64/float64 values into them:
 * heap corruption for `not a + b`, `(a + b) and c`, `a and b and c`.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 20000

static int failures = 0;
static int64_t a[N], b[N], c[N];

typedef bool (*ref_fn)(int);

/* Uses the first nvars of a, b, c: me_eval() binds every compiled variable. */
static void check(const char *src, int nvars, ref_fn ref) {
    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_INT64}, {"c", ME_INT64}};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile(src, vars, nvars, ME_BOOL, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s'\n", src);
        failures++;
        return;
    }
    static bool out[N];
    const void *ptrs[] = {a, b, c};
    ME_EVAL_CHECK(e, ptrs, nvars, out, N);
    for (int i = 0; i < N; i++) {
        if (out[i] != ref(i)) {
            printf("  FAIL: '%s' [%d] = %d, expected %d\n", src, i, out[i], ref(i));
            failures++;
            break;
        }
    }
    me_free(e);
}

static bool r_and3(int i) { return a[i] && b[i] && c[i]; }
static bool r_or_and(int i) { return a[i] || (b[i] && c[i]); }
static bool r_not_sum(int i) { return a[i] + b[i] == 0; }
static bool r_sum_and(int i) { return a[i] + b[i] != 0 && c[i] != 0; }
static bool r_nested(int i) { return a[i] < b[i] && c[i] > 0 && a[i] + b[i] != 0; }
static bool r_not_prod_or(int i) { return a[i] * b[i] == 0 || c[i] < 2; }

/* A fractional or NaN operand is true (Python's truth), also when the
 * result is computed in a numeric dtype. */
static void check_float_truth(void) {
    static const double x[] = {-0.5, 0.5, 0.0, 0.25, NAN, -0.0};
    static const double y[] = {0.25, 0.0, 0.0, -0.75, 0.0, 0.0};
    static const struct {
        const char *src;
        me_dtype out;
    } cases[] = {
        {"x or y", ME_FLOAT64}, {"x and y", ME_FLOAT64}, {"(x or y) + 0", ME_BOOL},
        {"(x and y) * 2", ME_FLOAT64}, {"not x", ME_FLOAT64}, {"(not x) + 0", ME_INT8},
    };
    me_variable vars[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        const bool unary = strchr(cases[k].src, 'y') == NULL;
        int err = 0;
        me_expr *e = NULL;
        if (me_compile(cases[k].src, vars, unary ? 1 : 2, cases[k].out, &err, &e) != ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile '%s'\n", cases[k].src);
            failures++;
            continue;
        }
        double out[6] = {0};
        const void *ptrs[] = {x, y};
        ME_EVAL_CHECK(e, ptrs, unary ? 1 : 2, out, 6);
        for (int i = 0; i < 6; i++) {
            const bool tx = x[i] != 0, ty = y[i] != 0;
            double want;
            switch (k) {
            case 0: want = tx || ty; break;
            case 1: want = tx && ty; break;
            case 2: want = tx || ty; break;
            case 3: want = (tx && ty) * 2; break;
            case 4: want = !tx; break;
            default: want = !tx; break;
            }
            double got;
            if (cases[k].out == ME_BOOL) got = ((const bool *)out)[i];
            else if (cases[k].out == ME_INT8) got = ((const int8_t *)out)[i];
            else got = out[i];
            if (got != want) {
                printf("  FAIL: '%s' [%d] x=%g y=%g: got %g, expected %g\n", cases[k].src, i, x[i], y[i], got, want);
                failures++;
                break;
            }
        }
        me_free(e);
    }
}

int main(void) {
    printf("Test: logical operators with numeric operands\n");
    for (int i = 0; i < N; i++) {
        a[i] = i % 5 - 2;
        b[i] = i % 7 - 3;
        c[i] = i % 3 - 1;
    }
    check("a and b and c", 3, r_and3);
    check("a or b and c", 3, r_or_and);
    check("not a + b", 2, r_not_sum);
    check("(a + b) and c", 3, r_sum_and);
    check("(a < b and c > 0) and (a + b)", 3, r_nested);
    check("not (a * b) or (c < 2)", 3, r_not_prod_or);
    check_float_truth();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All logical-operand tests passed\n");
    return 0;
}
