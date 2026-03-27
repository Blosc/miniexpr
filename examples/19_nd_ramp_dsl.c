/*
 * Build a 5x10 monotonic array with the DSL ND kernel:
 *   value(i, j) = _i0 * _n1 + _i1
 *
 * This example evaluates by chunks/blocks (including padded edge blocks)
 * and scatters valid items into a dense 5x10 output.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/miniexpr.h"

static int64_t ceil_div64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static int64_t linear_index2(int64_t i0, int64_t i1, int64_t dim1) {
    return i0 * dim1 + i1;
}

int main(void) {
    const char *src =
        "def kernel():\n"
        "    return _i0 * _n1 + _i1\n";

    const int64_t shape[2] = {5, 10};
    const int32_t chunkshape[2] = {4, 6};
    const int32_t blockshape[2] = {3, 4};
    const int padded_block_items = blockshape[0] * blockshape[1];

    me_expr *expr = NULL;
    int err = 0;
    if (me_compile_nd(src, NULL, 0, ME_INT64, 2,
                      shape, chunkshape, blockshape, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("compile failed at %d\n", err);
        return 1;
    }

    int64_t out[5 * 10];
    int64_t block_out[3 * 4];
    memset(out, 0, sizeof(out));

    const int64_t nchunks0 = ceil_div64(shape[0], chunkshape[0]);
    const int64_t nchunks1 = ceil_div64(shape[1], chunkshape[1]);
    const int64_t nblocks0 = ceil_div64(chunkshape[0], blockshape[0]);
    const int64_t nblocks1 = ceil_div64(chunkshape[1], blockshape[1]);

    for (int64_t c0 = 0; c0 < nchunks0; c0++) {
        for (int64_t c1 = 0; c1 < nchunks1; c1++) {
            const int64_t nchunk = linear_index2(c0, c1, nchunks1);

            for (int64_t b0 = 0; b0 < nblocks0; b0++) {
                for (int64_t b1 = 0; b1 < nblocks1; b1++) {
                    const int64_t nblock = linear_index2(b0, b1, nblocks1);
                    int rc = me_eval_nd(expr, NULL, 0, block_out, padded_block_items, nchunk, nblock, NULL);
                    if (rc != ME_EVAL_SUCCESS) {
                        printf("eval failed rc=%d at chunk=(%lld,%lld) block=(%lld,%lld)\n",
                               rc, (long long)c0, (long long)c1, (long long)b0, (long long)b1);
                        me_free(expr);
                        return 1;
                    }

                    for (int64_t i = 0; i < blockshape[0]; i++) {
                        for (int64_t j = 0; j < blockshape[1]; j++) {
                            const int64_t gi = c0 * chunkshape[0] + b0 * blockshape[0] + i;
                            const int64_t gj = c1 * chunkshape[1] + b1 * blockshape[1] + j;
                            if (gi < shape[0] && gj < shape[1]) {
                                out[gi * shape[1] + gj] = block_out[i * blockshape[1] + j];
                            }
                        }
                    }
                }
            }
        }
    }

    me_free(expr);

    for (int64_t i = 0; i < shape[0]; i++) {
        for (int64_t j = 0; j < shape[1]; j++) {
            printf("%4lld", (long long)out[i * shape[1] + j]);
        }
        printf("\n");
    }

    return 0;
}
