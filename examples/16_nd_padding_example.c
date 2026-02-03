/*
 * Demonstrates me_compile_nd() + me_eval_nd() handling padded chunks/blocks.
 *
 * Shape:      (5, 4)
 * Chunkshape: (3, 3)
 * Blockshape: (2, 2)
 *
 * - Interior chunk/block: no padding (valid = 4)
 * - Edge chunk/block: padding zeros (valid = 2, padded to 4)
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../src/miniexpr.h"

static int64_t linear_chunk(int64_t c0, int64_t c1, int64_t dim1) {
    return c0 * dim1 + c1;
}

static int64_t linear_block(int64_t b0, int64_t b1, int64_t dim1) {
    return b0 * dim1 + b1;
}

static void print_block(const char *label, const double *buf, int64_t nitems) {
    printf("%s: [", label);
    for (int64_t i = 0; i < nitems; i++) {
        if (i) printf(", ");
        printf("%.0f", buf[i]);
    }
    printf("]\n");
}

int main(void) {
    const int64_t shape[2] = {5, 4};
    const int32_t chunkshape[2] = {3, 3};
    const int32_t blockshape[2] = {2, 2};

    const int64_t nchunks_dim0 = (shape[0] + chunkshape[0] - 1) / chunkshape[0];
    const int64_t nchunks_dim1 = (shape[1] + chunkshape[1] - 1) / chunkshape[1];
    const int64_t nblocks_dim0 = (chunkshape[0] + blockshape[0] - 1) / blockshape[0];
    const int64_t nblocks_dim1 = (chunkshape[1] + blockshape[1] - 1) / blockshape[1];

    me_variable vars[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile_nd("x + y", vars, 2, ME_FLOAT64, 2,
                      shape, chunkshape, blockshape, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("Compile failed at %d\n", err);
        return 1;
    }

    const int padded_block_items = blockshape[0] * blockshape[1]; /* 4 */

    /* Interior chunk (0,0), block (0,0): full valid */
    double x_interior[4] = {1, 2, 3, 4};
    double y_interior[4] = {10, 20, 30, 40};
    double out_interior[4] = {-1, -1, -1, -1};
    const void *ptrs_interior[] = {x_interior, y_interior};

    int64_t nchunk00 = linear_chunk(0, 0, nchunks_dim1);
    int64_t nblock00 = linear_block(0, 0, nblocks_dim1);
    int64_t valid = -1;
    me_nd_valid_nitems(expr, nchunk00, nblock00, &valid);
    me_eval_nd(expr, ptrs_interior, 2, out_interior, padded_block_items, nchunk00, nblock00, NULL);

    printf("Interior (chunk 0,0 block 0,0) valid=%lld\n", (long long)valid);
    print_block("output", out_interior, padded_block_items);

    /* Edge chunk (1,1), block (0,0): padding on second dimension */
    double x_edge[4] = {5, 6, 7, 8};  /* only first 2 are valid */
    double y_edge[4] = {50, 60, 70, 80};
    double out_edge[4] = {-1, -1, -1, -1};
    const void *ptrs_edge[] = {x_edge, y_edge};

    int64_t nchunk11 = linear_chunk(1, 1, nchunks_dim1);
    int64_t nblock10 = linear_block(0, 0, nblocks_dim1);
    me_nd_valid_nitems(expr, nchunk11, nblock10, &valid);
    me_eval_nd(expr, ptrs_edge, 2, out_edge, padded_block_items, nchunk11, nblock10, NULL);

    printf("\nEdge (chunk 1,1 block 0,0) valid=%lld (expect 2)\n", (long long)valid);
    print_block("output", out_edge, padded_block_items);

    me_free(expr);
    return 0;
}
