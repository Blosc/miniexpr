/* Constant bases must use the same wrapping integer power as array bases,
 * including when a DSL local reuses its exponent buffer for the result. */
#include "../src/miniexpr.h"
#include <stdint.h>
#include <stdio.h>

#define N 17
static int failures;

#define CHECK_POWER(SUFFIX, TYPE, DTYPE) \
static void check_##SUFFIX(void) { \
    const char *expressions[] = {"7 ** a", \
        "def k(a):\n    t = a\n    t = 7 ** t\n    return t\n"}; \
    TYPE input[N], output[N + 1]; \
    for (int i = 0; i < N; i++) input[i] = (TYPE)(i == N - 1 ? 63 : i); \
    const me_variable vars[] = {{"a", DTYPE}}; \
    const void *ptrs[] = {input}; \
    for (int k = 0; k < 2; k++) { \
        me_expr *expr = NULL; \
        int error = 0; \
        if (me_compile(expressions[k], vars, 1, DTYPE, &error, &expr) != ME_COMPILE_SUCCESS) { \
            printf("FAIL compile %s case %d: %d\n", #SUFFIX, k, error); \
            failures++; \
            continue; \
        } \
        for (int i = 0; i <= N; i++) output[i] = (TYPE)90; \
        me_eval_params params = {0}; \
        params.jit_mode = ME_JIT_OFF; \
        if (me_eval(expr, ptrs, 1, output, N, &params) != ME_EVAL_SUCCESS) { \
            printf("FAIL eval %s case %d\n", #SUFFIX, k); \
            failures++; \
        } else { \
            for (int i = 0; i < N; i++) { \
                uint64_t expected = 1; \
                for (int j = 0; j < (int)input[i]; j++) expected *= UINT64_C(7); \
                if (output[i] != (TYPE)expected) { \
                    printf("FAIL %s case %d exponent %d\n", #SUFFIX, k, (int)input[i]); \
                    failures++; \
                    break; \
                } \
            } \
            if (output[N] != (TYPE)90) { \
                printf("FAIL canary %s case %d\n", #SUFFIX, k); \
                failures++; \
            } \
        } \
        me_free(expr); \
    } \
}

CHECK_POWER(i8, int8_t, ME_INT8)
CHECK_POWER(i16, int16_t, ME_INT16)
CHECK_POWER(i32, int32_t, ME_INT32)
CHECK_POWER(i64, int64_t, ME_INT64)
CHECK_POWER(u8, uint8_t, ME_UINT8)
CHECK_POWER(u16, uint16_t, ME_UINT16)
CHECK_POWER(u32, uint32_t, ME_UINT32)
CHECK_POWER(u64, uint64_t, ME_UINT64)

int main(void) {
    check_i8(); check_i16(); check_i32(); check_i64();
    check_u8(); check_u16(); check_u32(); check_u64();
    if (failures) return 1;
    puts("All constant-base integer power tests passed");
    return 0;
}
