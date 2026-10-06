/* A nested conversion must honor the enclosing evaluator's buffer width. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

static int check_nested_conversion(const char *expression, const double *expected,
                                   me_dtype input_dtype, me_dtype output_dtype, int count, int mode) {
    char source[256];
    snprintf(source, sizeof(source), "def k(x):\n    return %s\n", expression);
    const double values[] = {-1.75, -0.25, 0.25, 1.75, 4.75};
    float input32[257];
    double input64[257];
    for (int i = 0; i < count; i++) {
        input32[i] = (float)values[i % 5];
        input64[i] = values[i % 5];
    }
    const void *inputs[] = {input_dtype == ME_FLOAT32 ? (const void *)input32 : (const void *)input64};
    me_variable variables[] = {{"x", input_dtype}};
    int64_t shape[] = {count};
    int32_t chunks[] = {count}, blocks[] = {count};
    me_expr *expr = NULL;
    int error = 0;
    int rc = me_compile_nd_jit(source, variables, 1, output_dtype, 1,
                               shape, chunks, blocks, mode, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS || !expr) {
        printf("Nested conversion compile failed: %d at %d\n", rc, error);
        return 1;
    }
    size_t width = output_dtype == ME_FLOAT32 ? sizeof(float) : sizeof(double);
    void *output = malloc((size_t)count * width);
    if (!output) {
        me_free(expr);
        return 1;
    }
    /* Reusing the expression must also work when evaluation clones scratch state. */
    for (int repetition = 0; repetition < 2; repetition++) {
        rc = me_eval(expr, inputs, 1, output, count, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            printf("Nested conversion evaluation failed: %d\n", rc);
            free(output);
            me_free(expr);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            double actual = output_dtype == ME_FLOAT32 ? ((float *)output)[i] : ((double *)output)[i];
            if (actual != expected[i % 5]) {
                printf("Nested conversion mismatch: input=%d output=%d count=%d mode=%d element=%d: %g != %g\n",
                       input_dtype, output_dtype, count, mode, i, actual, expected[i % 5]);
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
    const char *expressions[] = {
        "float(int(x) / 2)", "int(x + 0.25)", "bool(x + 0.25)",
        "bool(int(x + 0.25))", "int(x + 0.25) / 2"
    };
    const double expected[][5] = {
        {-0.5, 0, 0, 0.5, 2}, {-1, 0, 0, 2, 5}, {1, 0, 1, 1, 1},
        {1, 0, 0, 1, 1}, {-0.5, 0, 0, 1, 2.5}
    };
    const me_dtype types[] = {ME_FLOAT32, ME_FLOAT64};
    const int counts[] = {1, 5, 257};
    for (int input = 0; input < 2; input++) {
        for (int output = 0; output < 2; output++) {
            for (int count = 0; count < 3; count++) {
                for (int mode = ME_JIT_OFF; mode <= ME_JIT_ON; mode++) {
                    for (int expression = 0; expression < 5; expression++) {
                        if (check_nested_conversion(expressions[expression], expected[expression],
                                                     types[input], types[output], counts[count], mode)) {
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}
