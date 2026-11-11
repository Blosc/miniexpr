/*
 * Bitwise operators, shifts, abs(), powers, square() and floor()/ceil()/
 * trunc() of integers compute in the integer dtype.  They went through
 * double: 64-bit values beyond 2**53 lost their low bits (int64 a | 2,
 * a << 1, ~a, abs(a), floor(a); uint64 a & b), powers that overflow were
 * converted out of range (undefined; NumPy wraps), and shift
 * counts of 64 or more used the hardware's count modulo 64.  Shifts follow
 * NumPy: a count outside [0, bits) gives 0 for << and the sign fill for >>.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 64

static int failures = 0;

static int64_t sa[N], sb[N];
static uint64_t ua[N], ub[N];
static int8_t ca[N], cb[N];

/* `b` is the second variable; abs() has a `b` in its name, so test for the operand. */
static void check(const char *src, me_dtype dt, const void *a, const void *b, int64_t (*ref)(int)) {
    me_variable vars[] = {{"a", dt}, {"b", dt}};
    const int nvars = (strstr(src, " b") || strstr(src, "(b")) ? 2 : 1;
    int err = 0;
    me_expr *e = NULL;
    if (me_compile(src, vars, nvars, dt, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s'\n", src);
        failures++;
        return;
    }
    union { int64_t i64[N]; uint64_t u64[N]; int8_t i8[N]; } out;
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(e, ptrs, nvars, &out, N);
    for (int i = 0; i < N; i++) {
        int64_t got = dt == ME_INT8 ? out.i8[i] : out.i64[i];
        if (got != ref(i)) {
            printf("  FAIL: '%s' (dtype %d) [%d] = %lld, expected %lld\n", src, dt, i, (long long)got,
                   (long long)ref(i));
            failures++;
            break;
        }
    }
    me_free(e);
}

static int64_t shl(int64_t a, int64_t b, int bits) {
    if (b < 0 || b >= bits) return 0;
    return (int64_t)((uint64_t)a << b);
}
static int64_t shr(int64_t a, int64_t b, int bits) {
    if (b < 0 || b >= bits) return a < 0 ? -1 : 0;
    return a >> b;
}

static int64_t r_or(int i) { return sa[i] | sb[i]; }
static int64_t r_xor(int i) { return sa[i] ^ sb[i]; }
static int64_t r_and_const(int i) { return sa[i] & 6; }
static int64_t r_not(int i) { return ~sa[i]; }
static int64_t r_abs(int i) { return sa[i] < 0 ? (int64_t)(0 - (uint64_t)sa[i]) : sa[i]; }
static int64_t r_shl(int i) { return shl(sa[i], sb[i] % 70, 64); }
static int64_t r_shr(int i) { return shr(sa[i], sb[i] % 70, 64); }
static int64_t r_shl1(int i) { return shl(sa[i], 1, 64); }
static int64_t r_u_and(int i) { return (int64_t)(ua[i] & ub[i]); }
static int64_t r_u_shl(int i) { return (int64_t)((ub[i] % 70) >= 64 ? 0 : ua[i] << (ub[i] % 70)); }
static int64_t r_c_shl(int i) { return (int8_t)shl(ca[i], cb[i], 8); }
static int64_t r_c_shr(int i) { return shr(ca[i], cb[i], 8); }
/* NumPy's integer power and square wrap at the operand width. */
static int64_t ipow(int64_t base, int64_t e) {
    uint64_t r = 1;
    while (e-- > 0) r *= (uint64_t)base;
    return (int64_t)r;
}
static int64_t r_pow(int i) { return ipow(sa[i], sb[i]); }
static int64_t r_square(int i) { return (int64_t)((uint64_t)sa[i] * (uint64_t)sa[i]); }
static int64_t r_identity(int i) { return sa[i]; }
static int64_t r_u_pow4(int i) {
    uint64_t r = ua[i] * ua[i];
    return (int64_t)(r * r);
}

int main(void) {
    printf("Test: integer bitwise operators, shifts and abs() are exact\n");
    for (int i = 0; i < N; i++) {
        sa[i] = ((int64_t)1 << 60) + 1 + 2 * i;
        if (i % 3 == 0) sa[i] = -sa[i];
        sb[i] = i;
        ua[i] = UINT64_MAX - (uint64_t)i;
        ub[i] = UINT64_MAX - 2 * (uint64_t)i;
        ca[i] = (int8_t)(i * 7 - 100);
        cb[i] = (int8_t)(i % 11 - 1);
    }
    sa[N - 1] = INT64_MIN;
    check("a | b", ME_INT64, sa, sb, r_or);
    check("a ^ b", ME_INT64, sa, sb, r_xor);
    check("a & 6", ME_INT64, sa, sb, r_and_const);
    check("~a", ME_INT64, sa, sb, r_not);
    check("abs(a)", ME_INT64, sa, sb, r_abs);
    /* Counts from 0 to 69 (b % 70 below), past the width. */
    for (int i = 0; i < N; i++) sb[i] = i + 6 * (i % 2);
    for (int i = 0; i < N; i++) sb[i] %= 70;
    check("a << b", ME_INT64, sa, sb, r_shl);
    check("a >> b", ME_INT64, sa, sb, r_shr);
    check("a << 1", ME_INT64, sa, sb, r_shl1);
    check("a & b", ME_UINT64, ua, ub, r_u_and);
    for (int i = 0; i < N; i++) ub[i] = (uint64_t)(i + 6 * (i % 2)) % 70;
    check("a << b", ME_UINT64, ua, ub, r_u_shl);
    check("a << b", ME_INT8, ca, cb, r_c_shl);
    check("a >> b", ME_INT8, ca, cb, r_c_shr);
    /* Powers, square() and the rounding functions of 64-bit integers. */
    for (int i = 0; i < N; i++) sb[i] = i % 45;
    check("a ** b", ME_INT64, sa, sb, r_pow);
    check("square(a)", ME_INT64, sa, sb, r_square);
    check("floor(a)", ME_INT64, sa, sb, r_identity);
    check("ceil(a)", ME_INT64, sa, sb, r_identity);
    check("trunc(a)", ME_INT64, sa, sb, r_identity);
    check("a ** 4", ME_UINT64, ua, ub, r_u_pow4);
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All integer bitwise tests passed\n");
    return 0;
}
