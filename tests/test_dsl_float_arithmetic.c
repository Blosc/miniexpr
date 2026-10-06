/* Pure floating arithmetic rounds literals and intermediates at the native width. */
#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

static int check_arithmetic(int operation, me_dtype dtype, int count, int mode) {
    const char *expressions[] = {
        "(x + 0.1) - x", "x - 0.1", "(x * 0.1) - x",
        "(float(x) + 0.1) - float(x)", "float(x + 0.1) - float(x)",
        "float(float(x) + 0.1) - float(float(x))"
    };
    const float samples[] = {0.1f, -1.0f, 0.0f, 1.0f, 16777216.0f};
    float input[257];
    for (int i = 0; i < count; i++) input[i] = samples[i % 5];
    char source[256];
    snprintf(source, sizeof(source), "# me:fp=strict\ndef k(x):\n    return %s\n", expressions[operation]);
    const void *inputs[] = {input};
    me_variable variables[] = {{"x", ME_FLOAT32}};
    int64_t shape[] = {count};
    int32_t grid[] = {count};
    me_expr *expr = NULL;
    int error = 0;
    int rc = me_compile_nd_jit(source, variables, 1, dtype, 1, shape, grid, grid, mode, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS || !expr) {
        printf("Floating arithmetic compile failed: %d at %d\n", rc, error);
        return 1;
    }
    void *output = malloc((size_t)count * (dtype == ME_FLOAT32 ? sizeof(float) : sizeof(double)));
    if (!output) {
        me_free(expr);
        return 1;
    }
    for (int repetition = 0; repetition < 2; repetition++) {
        rc = me_eval(expr, inputs, 1, output, count, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            free(output);
            me_free(expr);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            double expected, actual;
            if (dtype == ME_FLOAT32) {
                /* Volatile prevents a host compiler fusing the reference's
                 * multiply/add/subtract or retaining a wider intermediate. */
                volatile float intermediate = operation == 0 || operation >= 3
                    ? input[i] + 0.1f : input[i] * 0.1f;
                expected = operation == 1 ? input[i] - 0.1f : intermediate - input[i];
                actual = ((float *)output)[i];
            }
            else {
                volatile double intermediate = operation == 0 || operation >= 3
                    ? (double)input[i] + 0.1 : (double)input[i] * 0.1;
                expected = operation == 1 ? (double)input[i] - 0.1 : intermediate - (double)input[i];
                actual = ((double *)output)[i];
            }
            if (actual != expected) {
                printf("Floating arithmetic mismatch: op=%d dtype=%d count=%d mode=%d element=%d: %.17g != %.17g\n",
                       operation, dtype, count, mode, i, actual, expected);
                free(output);
                me_free(expr);
                return 1;
            }
        }
    }
    free(output);
    me_free(expr);
    return 0;
}

int main(void) {
    const me_dtype types[] = {ME_FLOAT32, ME_FLOAT64};
    const int counts[] = {1, 5, 257};
    for (int operation = 0; operation < 6; operation++) {
        for (int type = 0; type < 2; type++) {
            for (int count = 0; count < 3; count++) {
                for (int mode = ME_JIT_OFF; mode <= ME_JIT_ON; mode++) {
                    if (check_arithmetic(operation, types[type], counts[count], mode)) return 1;
                }
            }
        }
    }
    return 0;
}
