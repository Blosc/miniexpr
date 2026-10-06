/* Scalar chain operands must read the active lane of full-width local buffers. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

static void put_value(void *data, me_dtype dtype, int item, int value) {
    switch (dtype) {
    case ME_INT32: ((int32_t *)data)[item] = value; break;
    case ME_INT64: ((int64_t *)data)[item] = value; break;
    case ME_FLOAT32: ((float *)data)[item] = (float)value; break;
    case ME_FLOAT64: ((double *)data)[item] = (double)value; break;
    default: break;
    }
}

static double get_value(const void *data, me_dtype dtype, int item) {
    switch (dtype) {
    case ME_INT32: return ((const int32_t *)data)[item];
    case ME_INT64: return (double)((const int64_t *)data)[item];
    case ME_FLOAT32: return ((const float *)data)[item];
    case ME_FLOAT64: return ((const double *)data)[item];
    default: return -99;
    }
}

static int check_locals(me_dtype dtype, int count, int rotation, int branch, int mode) {
    const char *plain =
        "def k(x):\n    n = 0\n    while 0 <= n < x:\n        n = n + 1\n    return n\n";
    const char *masked =
        "def k(x):\n    n = 0\n    if x == 0:\n        n = 1\n"
        "    while 0 <= n < x:\n        n = n + 1\n    return n\n";
    const int samples[] = {0, 2, 3, -1};
    size_t width = dtype == ME_INT32 || dtype == ME_FLOAT32 ? 4 : 8;
    void *input = malloc((size_t)count * width);
    void *output = malloc((size_t)count * width);
    if (!input || !output) {
        free(input);
        free(output);
        return 1;
    }
    for (int i = 0; i < count; i++) put_value(input, dtype, i, samples[(i + rotation) % 4]);
    const void *inputs[] = {input};
    me_variable variables[] = {{"x", dtype}};
    int64_t shape[] = {count};
    int32_t grid[] = {count};
    me_expr *expr = NULL;
    int error = 0;
    int rc = me_compile_nd_jit(branch ? masked : plain, variables, 1, dtype, 1,
                               shape, grid, grid, mode, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS || !expr) {
        printf("Masked local compile failed: %d at %d\n", rc, error);
        free(input);
        free(output);
        return 1;
    }
    for (int repetition = 0; repetition < 2; repetition++) {
        rc = me_eval(expr, inputs, 1, output, count, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            printf("Masked local evaluation failed: dtype=%d count=%d rotation=%d branch=%d mode=%d: %d\n",
                   dtype, count, rotation, branch, mode, rc);
            free(input);
            free(output);
            me_free(expr);
            return 1;
        }
        for (int i = 0; i < count; i++) {
            int value = samples[(i + rotation) % 4];
            int expected = value > 0 ? value : (branch && value == 0 ? 1 : 0);
            if (get_value(output, dtype, i) != expected) {
                printf("Masked local mismatch at element %d: %.17g != %d\n",
                       i, get_value(output, dtype, i), expected);
                free(input);
                free(output);
                me_free(expr);
                return 1;
            }
        }
    }
    free(input);
    free(output);
    me_free(expr);
    return 0;
}

static int check_reduction_local(void) {
    const char *source = "def k(x):\n    total = sum(x)\n    return 0 < x < total\n";
    double input[] = {0, 1, 2};
    bool output[] = {false, false, false};
    const void *inputs[] = {input};
    me_variable variables[] = {{"x", ME_FLOAT64}};
    int64_t shape[] = {3};
    int32_t grid[] = {3};
    me_expr *expr = NULL;
    int error = 0;
    int rc = me_compile_nd_jit(source, variables, 1, ME_BOOL, 1,
                               shape, grid, grid, ME_JIT_OFF, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS || !expr) return 1;
    rc = me_eval(expr, inputs, 1, output, 3, NULL);
    me_free(expr);
    if (rc != ME_EVAL_SUCCESS || output[0] || !output[1] || !output[2]) {
        printf("Reduction local lost its broadcast values in masked chain evaluation\n");
        return 1;
    }
    return 0;
}

int main(void) {
    if (check_reduction_local()) return 1;
    const me_dtype types[] = {ME_INT32, ME_INT64, ME_FLOAT32, ME_FLOAT64};
    const int counts[] = {1, 3, 257};
    for (int type = 0; type < 4; type++) {
        for (int count = 0; count < 3; count++) {
            for (int rotation = 0; rotation < 4; rotation++) {
                for (int branch = 0; branch < 2; branch++) {
                    for (int mode = ME_JIT_OFF; mode <= ME_JIT_ON; mode++) {
                        if (check_locals(types[type], counts[count], rotation, branch, mode)) return 1;
                    }
                }
            }
        }
    }
    return 0;
}
