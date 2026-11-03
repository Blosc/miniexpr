/*
 * Comparisons inside larger expressions compare in the operands' dtype
 * (with float32, against the literal rounded to float32, as NumPy does).
 * The generic binary path converts both operands to double, which cannot
 * tell 64-bit integers apart beyond 2**53: (a == b) + 0 and
 * where(a < b, x, y) were wrong for neighbouring large int64/uint64 values.
 * (A comparison at the root already used typed loops.)
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdint.h>
#include <stdio.h>

#define N 257

static int failures = 0;

static void check_i64(const char *src, me_dtype dt, const void *a, const void *b,
                      int64_t (*ref)(int)) {
    me_variable vars[] = {{"a", dt}, {"b", dt}};
    int err = 0;
    me_expr *e = NULL;
    /* Output in the operand dtype; the values are small either way. */
    if (me_compile(src, vars, 2, dt, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s'\n", src);
        failures++;
        return;
    }
    int64_t out[N];
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(e, ptrs, 2, out, N);
    for (int i = 0; i < N; i++) {
        if (out[i] != ref(i)) {
            printf("  FAIL: '%s' (dtype %d) [%d] = %lld, expected %lld\n", src, dt, i, (long long)out[i],
                   (long long)ref(i));
            failures++;
            break;
        }
    }
    me_free(e);
}

static int64_t sa[N], sb[N];
static uint64_t ua[N], ub[N];

static int64_t r_s_eq(int i) { return sa[i] == sb[i]; }
static int64_t r_s_lt(int i) { return sa[i] < sb[i] ? 10 : 20; }
static int64_t r_s_ge(int i) { return (sa[i] >= sb[i]) + 2 * (sa[i] != sb[i]); }
static int64_t r_u_eq(int i) { return ua[i] == ub[i]; }
static int64_t r_u_le(int i) { return (ua[i] <= ub[i]) * 3; }

/* float32 compares with the literal rounded to float32, as NumPy does:
 * where(a > 0.1, a, b) with a == float32(0.1) selects b. */
static void check_f32_literal(void) {
    float a[N], b[N], out[N];
    for (int i = 0; i < N; i++) {
        a[i] = (i % 2) ? 0.1f : 0.2f;
        b[i] = -1.0f;
    }
    me_variable vars[] = {{"a", ME_FLOAT32}, {"b", ME_FLOAT32}};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile("where(a > 0.1, a, b)", vars, 2, ME_FLOAT32, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile where(a > 0.1, a, b)\n");
        failures++;
        return;
    }
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(e, ptrs, 2, out, N);
    for (int i = 0; i < N; i++) {
        float want = a[i] > 0.1f ? a[i] : b[i];
        if (out[i] != want) {
            printf("  FAIL: where(a > 0.1, a, b) [%d] = %g, expected %g\n", i, out[i], want);
            failures++;
            break;
        }
    }
    me_free(e);
}

int main(void) {
    printf("Test: nested comparisons of 64-bit integers are exact\n");
    for (int i = 0; i < N; i++) {
        /* Neighbouring values that are equal once rounded to double. */
        sa[i] = ((int64_t)1 << 60) + (i % 3);
        sb[i] = ((int64_t)1 << 60) + (i % 2);
        if (i % 4 == 0) {
            sa[i] = -sa[i];
            sb[i] = -sb[i];
        }
        ua[i] = UINT64_MAX - (uint64_t)(i % 3);
        ub[i] = UINT64_MAX - (uint64_t)(i % 2);
    }
    check_i64("(a == b) + 0", ME_INT64, sa, sb, r_s_eq);
    check_i64("where(a < b, 10, 20)", ME_INT64, sa, sb, r_s_lt);
    check_i64("where(a >= b, 1, 0) + where(a != b, 2, 0)", ME_INT64, sa, sb, r_s_ge);
    check_i64("(a == b) + 0", ME_UINT64, ua, ub, r_u_eq);
    check_i64("(a <= b) * 3", ME_UINT64, ua, ub, r_u_le);
    check_f32_literal();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All 64-bit comparison tests passed\n");
    return 0;
}
