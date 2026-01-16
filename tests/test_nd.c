#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"
#include "../src/minctest.h"

static int64_t linear_chunk_idx(int64_t c0, int64_t c1, int64_t c2,
                                int64_t dim1, int64_t dim2) {
    return (c0 * dim1 + c1) * dim2 + c2;
}

static int64_t linear_block_idx(int64_t b0, int64_t b1, int64_t b2,
                                int64_t dim1, int64_t dim2) {
    return (b0 * dim1 + b1) * dim2 + b2;
}

static int test_1d_basic(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {5};
    int32_t chunkshape[1] = {4};
    int32_t blockshape[1] = {2};
    me_variable vars[] = {{"x"}};

    int rc = me_compile_nd("x", vars, 1, ME_FLOAT64, 1,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd 1D: %d (err=%d)\n", rc, err);
        return 1;
    }

    const void* ptrs0[] = {NULL};
    double block0[2] = {1.0, 2.0};
    double out0[2] = {-1.0, -1.0};
    ptrs0[0] = block0;
    rc = me_eval_nd(expr, ptrs0, 1, out0, 2, 0, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out0[0] != 1.0 || out0[1] != 2.0) {
        printf("FAILED me_eval_nd 1D full block (rc=%d, out=[%g,%g])\n", rc, out0[0], out0[1]);
        status = 1;
        goto cleanup;
    }

    const void* ptrs1[] = {NULL};
    double block1[2] = {3.0, 999.0};
    double out1[2] = {-1.0, -1.0};
    ptrs1[0] = block1;
    rc = me_eval_nd(expr, ptrs1, 1, out1, 2, 1, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out1[0] != 3.0 || out1[1] != 0.0) {
        printf("FAILED me_eval_nd 1D padded block (rc=%d, out=[%g,%g])\n", rc, out1[0], out1[1]);
        status = 1;
        goto cleanup;
    }

    int64_t valid = -1;
    rc = me_nd_valid_nitems(expr, 1, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 1) {
        printf("FAILED me_nd_valid_nitems 1D (rc=%d, valid=%lld)\n", rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, ptrs1, 1, out1, 2, 1, 2, NULL);
    if (rc != ME_EVAL_ERR_INVALID_ARG) {
        printf("FAILED me_eval_nd 1D invalid block (rc=%d)\n", rc);
        status = 1;
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_2d_padding(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[2] = {3, 5};
    int32_t chunkshape[2] = {2, 4};
    int32_t blockshape[2] = {1, 3};
    me_variable vars[] = {{"x", ME_FLOAT32}, {"y", ME_INT32}};

    int rc = me_compile_nd("x + y", vars, 2, ME_FLOAT64, 2,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd 2D: %d (err=%d)\n", rc, err);
        return 1;
    }

    double out[3] = {-1.0, -1.0, -1.0};
    float xblock[3] = {10.0f, 20.0f, 30.0f};
    int32_t yblock[3] = {1, 2, 3};
    const void* ptrs[] = {xblock, yblock};
    int64_t valid = -1;

    rc = me_nd_valid_nitems(expr, 3, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 1) {
        printf("FAILED me_nd_valid_nitems 2D (rc=%d, valid=%lld)\n", rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, ptrs, 2, out, 3, 3, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out[0] != 11.0 || out[1] != 0.0 || out[2] != 0.0) {
        printf("FAILED me_eval_nd 2D padded (rc=%d, out=[%g,%g,%g])\n", rc, out[0], out[1], out[2]);
        status = 1;
        goto cleanup;
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_3d_partial(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[3] = {3, 4, 5};
    int32_t chunkshape[3] = {2, 3, 4};
    int32_t blockshape[3] = {2, 2, 2};
    me_variable vars[] = {{"a"}};

    int rc = me_compile_nd("a * 2", vars, 1, ME_FLOAT64, 3,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd 3D: %d (err=%d)\n", rc, err);
        return 1;
    }

    double out[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    double in[8] = {1, 2, 3, 4, 5, 6, 7, 8}; /* only first 2 valid */
    const void* ptrs[] = {in};
    int64_t valid = -1;

    rc = me_nd_valid_nitems(expr, 5, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 2) {
        printf("FAILED me_nd_valid_nitems 3D (rc=%d, valid=%lld)\n", rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, ptrs, 1, out, 8, 5, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out[0] != 2.0 || out[2] != 6.0) {
        printf("FAILED me_eval_nd 3D valid part (rc=%d, out0=%g, out2=%g)\n", rc, out[0], out[2]);
        status = 1;
        goto cleanup;
    }
    for (int i = 0; i < 8; i++) {
        if (i == 0 || i == 2) continue;
        if (out[i] != 0.0) {
            printf("FAILED me_eval_nd 3D padding at idx %d (val=%g)\n", i, out[i]);
            status = 1;
            goto cleanup;
        }
    }

    rc = me_eval_nd(expr, ptrs, 1, out, 4, 5, 0, NULL);
    if (rc != ME_EVAL_ERR_INVALID_ARG) {
        printf("FAILED me_eval_nd 3D insufficient buffer (rc=%d)\n", rc);
        status = 1;
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_big_stress(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    const int64_t shape[3] = {20000, 20000, 20000};
    const int32_t chunkshape[3] = {250, 250, 250};
    const int32_t blockshape[3] = {32, 64, 64};
    const int64_t nchunks_dim0 = (shape[0] + chunkshape[0] - 1) / chunkshape[0];
    const int64_t nchunks_dim1 = (shape[1] + chunkshape[1] - 1) / chunkshape[1];
    const int64_t nchunks_dim2 = (shape[2] + chunkshape[2] - 1) / chunkshape[2];
    const int64_t nblocks_dim0 = (chunkshape[0] + blockshape[0] - 1) / blockshape[0];
    const int64_t nblocks_dim1 = (chunkshape[1] + blockshape[1] - 1) / blockshape[1];
    const int64_t nblocks_dim2 = (chunkshape[2] + blockshape[2] - 1) / blockshape[2];
    const int padded_items = blockshape[0] * blockshape[1] * blockshape[2];

    me_variable vars[] = {{"a"}};
    int rc = me_compile_nd("a", vars, 1, ME_FLOAT64, 3,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd big: %d (err=%d)\n", rc, err);
        return 1;
    }

    double* in = malloc((size_t)padded_items * sizeof(double));
    double* out = malloc((size_t)padded_items * sizeof(double));
    if (!in || !out) {
        printf("FAILED alloc big buffers\n");
        status = 1;
        goto cleanup;
    }
    for (int i = 0; i < padded_items; i++) {
        in[i] = (double)(i + 1);
        out[i] = -1.0;
    }
    const void* ptrs[] = {in};

    int64_t nchunk_lin = linear_chunk_idx(10, 20, 30, nchunks_dim1, nchunks_dim2);
    int64_t nblock_lin = linear_block_idx(1, 2, 1, nblocks_dim1, nblocks_dim2);
    int64_t valid = -1;
    rc = me_nd_valid_nitems(expr, nchunk_lin, nblock_lin, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != padded_items) {
        printf("FAILED big valid count full block (rc=%d, valid=%lld)\n", rc, (long long)valid);
        status = 1;
        goto cleanup;
    }
    rc = me_eval_nd(expr, ptrs, 1, out, padded_items, nchunk_lin, nblock_lin, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED big eval full block rc=%d\n", rc);
        status = 1;
        goto cleanup;
    }
    for (int i = 0; i < padded_items; i++) {
        if (out[i] != in[i]) {
            printf("FAILED big eval full mismatch at %d (got=%g exp=%g)\n", i, out[i], in[i]);
            status = 1;
            goto cleanup;
        }
    }

    for (int i = 0; i < padded_items; i++) {
        out[i] = -1.0;
    }
    nchunk_lin = linear_chunk_idx(15, 10, 5, nchunks_dim1, nchunks_dim2);
    nblock_lin = linear_block_idx(nblocks_dim0 - 1, nblocks_dim1 - 1, nblocks_dim2 - 1,
                                  nblocks_dim1, nblocks_dim2);
    rc = me_nd_valid_nitems(expr, nchunk_lin, nblock_lin, &valid);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED big valid count edge rc=%d\n", rc);
        status = 1;
        goto cleanup;
    }
    const int64_t expected_valid = (chunkshape[0] - (blockshape[0] * (nblocks_dim0 - 1))) *
                                   (chunkshape[1] - (blockshape[1] * (nblocks_dim1 - 1))) *
                                   (chunkshape[2] - (blockshape[2] * (nblocks_dim2 - 1)));
    if (valid != expected_valid) {
        printf("FAILED big valid count edge value (got=%lld exp=%lld)\n", (long long)valid, (long long)expected_valid);
        status = 1;
        goto cleanup;
    }
    rc = me_eval_nd(expr, ptrs, 1, out, padded_items, nchunk_lin, nblock_lin, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED big eval edge rc=%d\n", rc);
        status = 1;
        goto cleanup;
    }
    int64_t nonzero_edge = 0;
    for (int i = 0; i < padded_items; i++) {
        if (out[i] != 0.0) {
            nonzero_edge++;
        }
    }
    if (nonzero_edge != expected_valid) {
        printf("FAILED big edge nonzero count (got=%lld exp=%lld)\n", (long long)nonzero_edge, (long long)expected_valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, ptrs, 1, out, padded_items, nchunk_lin,
                    nblocks_dim0 * nblocks_dim1 * nblocks_dim2, NULL);
    if (rc != ME_EVAL_ERR_INVALID_ARG) {
        printf("FAILED big invalid block rc=%d\n", rc);
        status = 1;
    }

cleanup:
    free(in);
    free(out);
    me_free(expr);
    return status;
}

int main(void) {
    int failed = 0;

    printf("Testing ND Evaluation\n");
    printf("=====================\n\n");

    printf("Test 1: 1D basic and padding\n");
    failed |= test_1d_basic();
    printf("Result: %s\n\n", failed ? "FAIL" : "PASS");

    printf("Test 2: 2D padding and mixed dtype\n");
    int t2 = test_2d_padding();
    failed |= t2;
    printf("Result: %s\n\n", t2 ? "FAIL" : "PASS");

    printf("Test 3: 3D partial block with padding\n");
    int t3 = test_3d_partial();
    failed |= t3;
    printf("Result: %s\n\n", t3 ? "FAIL" : "PASS");

    printf("Test 4: Large 3D stress (no real allocation beyond block)\n");
    int t4 = test_big_stress();
    failed |= t4;
    printf("Result: %s\n\n", t4 ? "FAIL" : "PASS");

    printf("=====================\n");
    printf("Summary: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
