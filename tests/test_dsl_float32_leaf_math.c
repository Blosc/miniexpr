/* Strict leaf math must round before widening/comparison at every block size. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

static int check_math(const char *body, int branch, int count, int mode) {
    char source[256];
    snprintf(source, sizeof(source), "# me:fp=strict\ndef k(x):\n%s", body);
    float input[257];
    double *output = malloc((size_t)count * sizeof(*output));
    if (!output) return 1;
    for (int i = 0; i < count; i++) {
        input[i] = branch ? (i % 2 ? 0.0f : 0.0001f) : (i % 3 == 0 ? -0.0f : (i % 3 == 1 ? 1.0f : 0.0f));
    }
    const void *inputs[] = {input};
    me_variable variables[] = {{"x", ME_FLOAT32}};
    int64_t shape[] = {count};
    int32_t grid[] = {count};
    me_expr *expr = NULL;
    int error = 0;
    int rc = me_compile_nd_jit(source, variables, 1, ME_FLOAT64, 1,
                               shape, grid, grid, mode, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS || !expr) {
        printf("Float32 math compile failed: %d at %d\n", rc, error);
        free(output);
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
            double expected = branch ? 1.0 : (input[i] == 1.0f ? 0.8414709568023681640625 : (double)input[i]);
            if (output[i] != expected || (expected == 0.0 && signbit(output[i]) != signbit(expected))) {
                printf("Float32 math mismatch: count=%d mode=%d element=%d: %.17g != %.17g\n",
                       count, mode, i, output[i], expected);
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
    const char *bodies[] = {
        "    return sin(x)\n",
        "    value = sin(x)\n    return value\n",
        "    if cos(x) == 1:\n        return 1.0\n    return 0.0\n",
        "    if 1 <= cos(x):\n        return 1.0\n    return 0.0\n"
    };
    const int counts[] = {1, 2, 257};
    for (int body = 0; body < 4; body++) {
        for (int count = 0; count < 3; count++) {
            for (int mode = ME_JIT_OFF; mode <= ME_JIT_ON; mode++) {
                if (check_math(bodies[body], body >= 2, counts[count], mode)) return 1;
            }
        }
    }
    return 0;
}
