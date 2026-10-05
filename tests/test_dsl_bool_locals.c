/* Numeric intermediates must retain their types in Boolean-output kernels. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "../src/miniexpr.h"

int main(void) {
    const char *source =
        "def k(x):\n"
        "    left = 0\n"
        "    middle = x\n"
        "    result = left < middle\n"
        "    if result:\n"
        "        right = 12 / x\n"
        "        result = middle < right\n"
        "    return result\n";
    int64_t x[] = {-2, -1, 0, 1, 2, 3, 4};
    const void *inputs[] = {x};
    me_variable variables[] = {{"x", ME_INT64}};
    int64_t shape[] = {7};
    int32_t chunks[] = {7}, blocks[] = {7};
    for (int mode = ME_JIT_OFF; mode <= ME_JIT_ON; mode++) {
        me_expr *expr = NULL;
        int error = 0;
        int rc = me_compile_nd_jit(source, variables, 1, ME_BOOL, 1,
                                   shape, chunks, blocks, mode, &error, &expr);
        if (rc != ME_COMPILE_SUCCESS || !expr) {
            printf("Boolean kernel compilation failed: %d at %d\n", rc, error);
            return 1;
        }
        bool out[7] = {false};
        rc = me_eval_nd(expr, inputs, 1, out, 7, 1, 0, NULL);
        me_free(expr);
        if (rc != ME_EVAL_SUCCESS) {
            printf("Boolean kernel evaluation failed: %d\n", rc);
            return 1;
        }
        for (int i = 0; i < 7; i++) {
            bool expected = x[i] > 0 && x[i] < 12 / x[i];
            if (out[i] != expected) {
                printf("Boolean kernel mismatch at %d in mode %d\n", i, mode);
                return 1;
            }
        }
    }
    return 0;
}
