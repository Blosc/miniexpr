/*
 * Left shifts stay on the interpreter: signed C left shifts can be undefined
 * for negative operands or overflow, even with a valid count. Right shifts
 * compile only for literal counts below the width and below 32; other counts
 * stay on the interpreter to preserve NumPy's out-of-range sign fill.
 *
 * Named test_dsl_jit_* so that ctest runs it with the JIT enabled.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 4

static int failures = 0;

static me_expr *compile_kernel(const char *src, const me_variable *vars, int jit) {
    int64_t shape[1] = {N};
    int32_t chunks[1] = {N}, blocks[1] = {N};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile_nd_jit(src, vars, 2, ME_INT64, 1, shape, chunks, blocks, jit ? ME_JIT_ON : ME_JIT_OFF, &err,
                          &e) != ME_COMPILE_SUCCESS) {
        return NULL;
    }
    return e;
}

/* Whether this build JIT-compiles a trivial integer kernel at all. */
static bool jit_available(void) {
    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_INT64}};
    me_expr *e = compile_kernel("def k(a, b):\n    return a + b\n", vars, 1);
    const bool ok = e && me_expr_has_jit_kernel(e);
    me_free(e);
    return ok;
}

typedef struct {
    const char *src;
    me_dtype a_dtype;
    /* NumPy's values; NULL: only check that the JIT and the interpreter agree. */
    const int64_t *expected;
    /* With a JIT available: whether the kernel must be compiled by it. */
    bool jit_compiles;
    const int64_t *a_values;
} shift_case;

int main(void) {
    printf("Test: DSL integer shifts with the JIT on and off\n");
    const bool jit = jit_available();
    if (!jit) printf("  (no JIT in this build: interpreter only)\n");
    /* a = {5, -5, 5, -5}, b = {64, 64, 70, -1}; NumPy: 5 << 64 == 0, -5 >> 64 == -1, -5 >> -1 == -1. */
    static const int64_t zeros[N] = {0, 0, 0, 0}, shr[N] = {0, -1, 0, -1}, shl3[N] = {104, 24, 110, -41};
    static const int64_t edges[N] = {INT64_MAX, INT64_MIN, -1, INT64_C(1) << 62};
    static const int64_t shifted[N] = {-2, 0, -2, INT64_MIN};
    static const int64_t shr3[N] = {64, 63, 70, -2};
    static const shift_case cases[] = {
        {"def k(a, b):\n    return a << b\n", ME_INT64, zeros, false},
        {"def k(a, b):\n    return a >> b\n", ME_INT64, shr, false},
        {"def k(a, b):\n    return (a << 3) + b\n", ME_INT64, shl3, false},
        {"def k(a, b):\n    return a << 3 * b\n", ME_INT64, zeros, false},
        {"def k(a, b):\n    return a << 1\n", ME_INT64, shifted, false, edges},
        {"def k(a, b):\n    return (a >> 3) + b\n", ME_INT64, shr3, true},
        /* An int32 operand in an int64 expression: C would shift it as an int.
         * (Its NumPy value, an int32 shift, needs the mixed-dtype typing of
         * conversion nodes; here both engines compute in int64.) */
        {"def k(a, b):\n    return (a << 40) + b\n", ME_INT32, NULL, false},
        {"def k(a, b):\n    return (a >> 40) + b\n", ME_INT32, NULL, false},
    };
    const int64_t a64[N] = {5, -5, 5, -5}, b64[N] = {64, 64, 70, -1};
    const int32_t a32[N] = {5, -5, 5, -5};
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        const shift_case *c = &cases[k];
        me_variable vars[] = {{"a", c->a_dtype}, {"b", ME_INT64}};
        const void *ptrs[] = {c->a_dtype == ME_INT32 ? (const void *)a32 : (const void *)(c->a_values ? c->a_values : a64), b64};
        int64_t results[2][N];
        for (int mode = 0; mode < 2; mode++) {
            me_expr *e = compile_kernel(c->src, vars, mode);
            if (!e) {
                printf("  FAIL: compile kernel %zu (JIT %s)\n", k, mode ? "on" : "off");
                failures++;
                continue;
            }
            if (mode && jit && me_expr_has_jit_kernel(e) != c->jit_compiles) {
                printf("  FAIL: kernel %zu %s JIT-compiled\n", k, c->jit_compiles ? "was not" : "was");
                failures++;
            }
            int64_t *out = results[mode];
            memset(out, 0, sizeof(results[mode]));
            if (me_eval_nd(e, ptrs, 2, out, N, 0, 0, NULL) != ME_EVAL_SUCCESS) {
                printf("  FAIL: eval kernel %zu (JIT %s)\n", k, mode ? "on" : "off");
                failures++;
            }
            else if (c->expected && memcmp(out, c->expected, sizeof(results[mode])) != 0) {
                printf("  FAIL: kernel %zu (JIT %s) = %lld %lld %lld %lld\n", k, mode ? "on" : "off",
                       (long long)out[0], (long long)out[1], (long long)out[2], (long long)out[3]);
                failures++;
            }
            me_free(e);
        }
        if (memcmp(results[0], results[1], sizeof(results[0])) != 0) {
            printf("  FAIL: kernel %zu differs between the JIT and the interpreter\n", k);
            failures++;
        }
    }
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All DSL integer shift tests passed\n");
    return 0;
}
