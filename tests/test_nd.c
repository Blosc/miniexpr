#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static int test_nd_cast_intrinsics_padding(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[2] = {3, 5};
    int32_t chunkshape[2] = {2, 4};
    int32_t blockshape[2] = {2, 3};

    int rc = me_compile_nd(
        "def kernel():\n"
        "    return int(_i0 * _n1 + _i1)\n",
        NULL, 0, ME_INT64, 2, shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd cast intrinsic: %d (err=%d)\n", rc, err);
        return 1;
    }

    int64_t out[6] = {-1, -1, -1, -1, -1, -1};
    int64_t valid = -1;
    rc = me_nd_valid_nitems(expr, 1, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 2) {
        printf("FAILED me_nd_valid_nitems cast intrinsic (rc=%d, valid=%lld)\n", rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, NULL, 0, out, 6, 1, 0, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED me_eval_nd cast intrinsic (rc=%d)\n", rc);
        status = 1;
        goto cleanup;
    }

    int64_t expected[6] = {4, 0, 0, 9, 0, 0};
    for (int i = 0; i < 6; i++) {
        if (out[i] != expected[i]) {
            printf("FAILED cast intrinsic mismatch at %d (got=%lld exp=%lld)\n",
                   i, (long long)out[i], (long long)expected[i]);
            status = 1;
            goto cleanup;
        }
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_cast_intrinsics_with_input_padding(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[2] = {3, 5};
    int32_t chunkshape[2] = {2, 4};
    int32_t blockshape[2] = {2, 3};
    me_variable vars[] = {{"x", ME_FLOAT32}};

    int rc = me_compile_nd(
        "def kernel(x):\n"
        "    return int(_i0 * _n1 + _i1)\n",
        vars, 1, ME_INT64, 2, shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd cast intrinsic with input: %d (err=%d)\n", rc, err);
        return 1;
    }

    int64_t out[6] = {-1, -1, -1, -1, -1, -1};
    int64_t valid = -1;
    rc = me_nd_valid_nitems(expr, 1, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 2) {
        printf("FAILED me_nd_valid_nitems cast intrinsic with input (rc=%d, valid=%lld)\n",
               rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    float xblock[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const void* inputs[] = {xblock};
    rc = me_eval_nd(expr, inputs, 1, out, 6, 1, 0, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED me_eval_nd cast intrinsic with input (rc=%d)\n", rc);
        status = 1;
        goto cleanup;
    }

    int64_t expected[6] = {4, 0, 0, 9, 0, 0};
    for (int i = 0; i < 6; i++) {
        if (out[i] != expected[i]) {
            printf("FAILED cast intrinsic with input mismatch at %d (got=%lld exp=%lld)\n",
                   i, (long long)out[i], (long long)expected[i]);
            status = 1;
            goto cleanup;
        }
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_float_index_cast_padding(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[2] = {3, 5};
    int32_t chunkshape[2] = {2, 4};
    int32_t blockshape[2] = {2, 3};

    int rc = me_compile_nd(
        "def kernel():\n"
        "    return float(_i0) * _n1 + _i1\n",
        NULL, 0, ME_FLOAT32, 2, shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd float index cast: %d (err=%d)\n", rc, err);
        return 1;
    }

    float out[6] = {-1, -1, -1, -1, -1, -1};
    int64_t valid = -1;
    rc = me_nd_valid_nitems(expr, 1, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 2) {
        printf("FAILED me_nd_valid_nitems float index cast (rc=%d, valid=%lld)\n", rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, NULL, 0, out, 6, 1, 0, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED me_eval_nd float index cast (rc=%d)\n", rc);
        status = 1;
        goto cleanup;
    }

    float expected[6] = {4.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f};
    for (int i = 0; i < 6; i++) {
        if (fabsf(out[i] - expected[i]) > 1e-6f) {
            printf("FAILED float index cast mismatch at %d (got=%g exp=%g)\n", i, out[i], expected[i]);
            status = 1;
            goto cleanup;
        }
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_int_constant_cast_padding(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {5};
    int32_t chunkshape[1] = {4};
    int32_t blockshape[1] = {3};

    int rc = me_compile_nd(
        "def kernel():\n"
        "    return int(1.9)\n",
        NULL, 0, ME_INT64, 1, shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd int(1.9): %d (err=%d)\n", rc, err);
        return 1;
    }

    int64_t out[3] = {-1, -1, -1};
    int64_t valid = -1;
    rc = me_nd_valid_nitems(expr, 0, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 3) {
        printf("FAILED me_nd_valid_nitems int(1.9) full block (rc=%d, valid=%lld)\n",
               rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, NULL, 0, out, 3, 0, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out[0] != 1 || out[1] != 1 || out[2] != 1) {
        printf("FAILED me_eval_nd int(1.9) full block (rc=%d, out=[%lld,%lld,%lld])\n",
               rc, (long long)out[0], (long long)out[1], (long long)out[2]);
        status = 1;
        goto cleanup;
    }

    out[0] = -1;
    out[1] = -1;
    out[2] = -1;
    rc = me_nd_valid_nitems(expr, 1, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 1) {
        printf("FAILED me_nd_valid_nitems int(1.9) padded block (rc=%d, valid=%lld)\n",
               rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, NULL, 0, out, 3, 1, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out[0] != 1 || out[1] != 0 || out[2] != 0) {
        printf("FAILED me_eval_nd int(1.9) padded block (rc=%d, out=[%lld,%lld,%lld])\n",
               rc, (long long)out[0], (long long)out[1], (long long)out[2]);
        status = 1;
        goto cleanup;
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_bool_cast_numeric_padding(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {5};
    int32_t chunkshape[1] = {4};
    int32_t blockshape[1] = {3};
    me_variable vars[] = {{"x", ME_FLOAT64}};

    int rc = me_compile_nd(
        "def kernel(x):\n"
        "    return bool(x)\n",
        vars, 1, ME_BOOL, 1, shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd bool(x): %d (err=%d)\n", rc, err);
        return 1;
    }

    const void* ptrs[] = {NULL};
    double in[3] = {0.0, -2.0, 3.5};
    bool out[3] = {true, false, false};
    int64_t valid = -1;
    ptrs[0] = in;

    rc = me_nd_valid_nitems(expr, 0, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 3) {
        printf("FAILED me_nd_valid_nitems bool(x) full block (rc=%d, valid=%lld)\n",
               rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, ptrs, 1, out, 3, 0, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out[0] || !out[1] || !out[2]) {
        printf("FAILED me_eval_nd bool(x) full block (rc=%d, out=[%d,%d,%d])\n",
               rc, (int)out[0], (int)out[1], (int)out[2]);
        status = 1;
        goto cleanup;
    }

    in[0] = 7.0;
    in[1] = 123.0;
    in[2] = 123.0;
    out[0] = false;
    out[1] = true;
    out[2] = true;
    rc = me_nd_valid_nitems(expr, 1, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 1) {
        printf("FAILED me_nd_valid_nitems bool(x) padded block (rc=%d, valid=%lld)\n",
               rc, (long long)valid);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, ptrs, 1, out, 3, 1, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || !out[0] || out[1] || out[2]) {
        printf("FAILED me_eval_nd bool(x) padded block (rc=%d, out=[%d,%d,%d])\n",
               rc, (int)out[0], (int)out[1], (int)out[2]);
        status = 1;
        goto cleanup;
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_int32_ramp_kernel_sum(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[2] = {1000, 1000};
    int32_t chunkshape[2] = {257, 251};
    int32_t blockshape[2] = {129, 127};

    int rc = me_compile_nd(
        "def kernel():\n"
        "    return _i0 * _n1 + _i1\n",
        NULL, 0, ME_INT32, 2, shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd int32 ramp: %d (err=%d)\n", rc, err);
        return 1;
    }

    const int64_t padded_items = (int64_t)blockshape[0] * blockshape[1];
    int32_t *out = (int32_t *)malloc((size_t)padded_items * sizeof(int32_t));
    if (!out) {
        printf("FAILED alloc output for int32 ramp test\n");
        me_free(expr);
        return 1;
    }

    const int64_t nchunks_dim0 = (shape[0] + chunkshape[0] - 1) / chunkshape[0];
    const int64_t nchunks_dim1 = (shape[1] + chunkshape[1] - 1) / chunkshape[1];
    const int64_t nblocks_dim0 = (chunkshape[0] + blockshape[0] - 1) / blockshape[0];
    const int64_t nblocks_dim1 = (chunkshape[1] + blockshape[1] - 1) / blockshape[1];

    int64_t sum = 0;
    for (int64_t c0 = 0; c0 < nchunks_dim0; c0++) {
        for (int64_t c1 = 0; c1 < nchunks_dim1; c1++) {
            const int64_t nchunk = c0 * nchunks_dim1 + c1;
            for (int64_t b0 = 0; b0 < nblocks_dim0; b0++) {
                for (int64_t b1 = 0; b1 < nblocks_dim1; b1++) {
                    const int64_t nblock = b0 * nblocks_dim1 + b1;
                    memset(out, 0, (size_t)padded_items * sizeof(int32_t));
                    rc = me_eval_nd(expr, NULL, 0, out, padded_items, nchunk, nblock, NULL);
                    if (rc != ME_EVAL_SUCCESS) {
                        printf("FAILED me_eval_nd int32 ramp (rc=%d, chunk=%lld, block=%lld)\n",
                               rc, (long long)nchunk, (long long)nblock);
                        status = 1;
                        goto cleanup;
                    }
                    for (int64_t i = 0; i < padded_items; i++) {
                        sum += out[i];
                    }
                }
            }
        }
    }

    int64_t nitems = shape[0] * shape[1];
    int64_t expected_sum = nitems * (nitems - 1) / 2;
    if (sum != expected_sum) {
        printf("FAILED int32 ramp sum mismatch (got=%lld expected=%lld)\n",
               (long long)sum, (long long)expected_sum);
        status = 1;
    }

cleanup:
    free(out);
    me_free(expr);
    return status;
}

static int test_nd_unary_int32_float_math(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {10};
    int32_t chunkshape[1] = {6};
    int32_t blockshape[1] = {4};
    me_variable vars[] = {{"x", ME_INT32}};

    int rc = me_compile_nd("arccos(x)", vars, 1, ME_FLOAT64, 1,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd unary int32: %d (err=%d)\n", rc, err);
        return 1;
    }

    const double expected = acos(0.0);
    const void* ptrs[] = {NULL};
    double out[4] = {-1.0, -1.0, -1.0, -1.0};
    int32_t in[4] = {0, 0, 0, 0};
    ptrs[0] = in;

    int64_t valid0 = -1;
    rc = me_nd_valid_nitems(expr, 0, 0, &valid0);
    if (rc != ME_EVAL_SUCCESS || valid0 != 4) {
        printf("FAILED me_nd_valid_nitems unary full (rc=%d, valid=%lld)\n", rc, (long long)valid0);
        status = 1;
        goto cleanup;
    }

    rc = me_eval_nd(expr, ptrs, 1, out, 4, 0, 0, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED me_eval_nd unary full (rc=%d)\n", rc);
        status = 1;
        goto cleanup;
    }
    for (int i = 0; i < 4; i++) {
        if (fabs(out[i] - expected) > 1e-12) {
            printf("FAILED unary full mismatch at %d: got %.15f expected %.15f\n",
                   i, out[i], expected);
            status = 1;
            goto cleanup;
        }
    }

    int64_t valid1 = -1;
    rc = me_nd_valid_nitems(expr, 0, 1, &valid1);
    if (rc != ME_EVAL_SUCCESS || valid1 != 2) {
        printf("FAILED me_nd_valid_nitems unary padded (rc=%d, valid=%lld)\n", rc, (long long)valid1);
        status = 1;
        goto cleanup;
    }

    in[0] = 0;
    in[1] = 0;
    in[2] = 12345;
    in[3] = 12345;
    out[0] = out[1] = out[2] = out[3] = -1.0;

    rc = me_eval_nd(expr, ptrs, 1, out, 4, 0, 1, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAILED me_eval_nd unary padded (rc=%d)\n", rc);
        status = 1;
        goto cleanup;
    }
    for (int i = 0; i < (int)valid1; i++) {
        if (fabs(out[i] - expected) > 1e-12) {
            printf("FAILED unary padded mismatch at %d: got %.15f expected %.15f\n",
                   i, out[i], expected);
            status = 1;
            goto cleanup;
        }
    }
    for (int i = (int)valid1; i < 4; i++) {
        if (out[i] != 0.0) {
            printf("FAILED unary padded tail at %d: got %.15f expected 0.0\n", i, out[i]);
            status = 1;
            goto cleanup;
        }
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_unary_int32_negative_blocks(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {10};
    int32_t chunkshape[1] = {3};
    int32_t blockshape[1] = {3};
    me_variable vars[] = {{"x", ME_INT32}};

    int rc = me_compile_nd("0 - x", vars, 1, ME_INT32, 1,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd unary int32 negative: %d (err=%d)\n", rc, err);
        return 1;
    }

    const void* ptrs[] = {NULL};
    int32_t out[3] = {0, 0, 0};
    int32_t in[3] = {0, 0, 0};

    for (int64_t nchunk = 0; nchunk < 4; nchunk++) {
        int64_t valid = -1;
        rc = me_nd_valid_nitems(expr, nchunk, 0, &valid);
        if (rc != ME_EVAL_SUCCESS) {
            printf("FAILED me_nd_valid_nitems negative (rc=%d, chunk=%lld)\n",
                   rc, (long long)nchunk);
            status = 1;
            goto cleanup;
        }

        int32_t base = (int32_t)(nchunk * 3 + 1);
        for (int i = 0; i < 3; i++) {
            if (i < valid) {
                in[i] = base + i;
            } else {
                in[i] = 12345;
            }
            out[i] = 777777;
        }

        ptrs[0] = in;
        rc = me_eval_nd(expr, ptrs, 1, out, 3, nchunk, 0, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            printf("FAILED me_eval_nd negative (rc=%d, chunk=%lld)\n",
                   rc, (long long)nchunk);
            status = 1;
            goto cleanup;
        }

        for (int i = 0; i < valid; i++) {
            if (out[i] != -in[i]) {
                printf("FAILED unary negative mismatch chunk=%lld idx=%d got=%d expected=%d\n",
                       (long long)nchunk, i, out[i], -in[i]);
                status = 1;
                goto cleanup;
            }
        }
        for (int i = (int)valid; i < 3; i++) {
            if (out[i] != 0) {
                printf("FAILED unary negative padding chunk=%lld idx=%d got=%d expected=0\n",
                       (long long)nchunk, i, out[i]);
                status = 1;
                goto cleanup;
            }
        }
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_unary_int32_to_float64_padding(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {10};
    int32_t chunkshape[1] = {3};
    int32_t blockshape[1] = {3};
    me_variable vars[] = {{"x", ME_INT32}};

    int rc = me_compile_nd("arccos(x)", vars, 1, ME_FLOAT64, 1,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd unary int32->float64: %d (err=%d)\n", rc, err);
        return 1;
    }

    const double expected = acos(0.0);
    const int64_t nchunks = (shape[0] + chunkshape[0] - 1) / chunkshape[0];
    const void* ptrs[] = {NULL};
    int32_t in[3] = {0, 0, 0};
    double out[3] = {-1.0, -1.0, -1.0};

    for (int64_t nchunk = 0; nchunk < nchunks; nchunk++) {
        int64_t valid = -1;
        rc = me_nd_valid_nitems(expr, nchunk, 0, &valid);
        if (rc != ME_EVAL_SUCCESS) {
            printf("FAILED me_nd_valid_nitems int32->float64 (rc=%d, chunk=%lld)\n",
                   rc, (long long)nchunk);
            status = 1;
            goto cleanup;
        }

        int64_t expected_valid = shape[0] - nchunk * chunkshape[0];
        if (expected_valid > blockshape[0]) expected_valid = blockshape[0];
        if (expected_valid < 0) expected_valid = 0;
        if (valid != expected_valid) {
            printf("FAILED valid count int32->float64 (chunk=%lld got=%lld exp=%lld)\n",
                   (long long)nchunk, (long long)valid, (long long)expected_valid);
            status = 1;
            goto cleanup;
        }

        for (int i = 0; i < 3; i++) {
            in[i] = (i < valid) ? 0 : 12345;
            out[i] = -1.0;
        }
        ptrs[0] = in;
        rc = me_eval_nd(expr, ptrs, 1, out, 3, nchunk, 0, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            printf("FAILED me_eval_nd int32->float64 (rc=%d, chunk=%lld)\n",
                   rc, (long long)nchunk);
            status = 1;
            goto cleanup;
        }
        for (int i = 0; i < valid; i++) {
            if (fabs(out[i] - expected) > 1e-12) {
                printf("FAILED int32->float64 mismatch chunk=%lld idx=%d got=%.15f exp=%.15f\n",
                       (long long)nchunk, i, out[i], expected);
                status = 1;
                goto cleanup;
            }
        }
        for (int i = (int)valid; i < 3; i++) {
            if (out[i] != 0.0) {
                printf("FAILED int32->float64 padding chunk=%lld idx=%d got=%.15f exp=0.0\n",
                       (long long)nchunk, i, out[i]);
                status = 1;
                goto cleanup;
            }
        }
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

static int test_nd_reductions(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {3};
    int32_t chunkshape[1] = {3};
    int32_t blockshape[1] = {2};
    const int padded = 2;

    me_variable vars[] = {{"x", ME_FLOAT64}};
    if (me_compile_nd("sum(x)", vars, 1, ME_FLOAT64, 1,
                      shape, chunkshape, blockshape, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd reductions: %d\n", err);
        return 1;
    }

    double out_buf[2] = {-1.0, 123.0};
    double block0[2] = {1.0, 2.0};
    const void* ptrs0[] = {block0};
    int64_t valid = 0;
    me_nd_valid_nitems(expr, 0, 0, &valid);
    if (valid != 2) {
        printf("FAILED reductions valid block0 (got=%lld)\n", (long long)valid);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr, ptrs0, 1, out_buf, padded, 0, 0, NULL) != ME_EVAL_SUCCESS || out_buf[0] != 3.0) {
        printf("FAILED reductions sum block0 (out=%g)\n", out_buf[0]);
        status = 1;
        goto cleanup;
    }
    if (out_buf[1] != 123.0) {
        printf("FAILED reductions: scalar output overwrote tail (tail=%g)\n", out_buf[1]);
        status = 1;
        goto cleanup;
    }

    double block1[2] = {3.0, 0.0}; /* last is padding */
    const void* ptrs1[] = {block1};
    out_buf[0] = -1.0;
    out_buf[1] = 123.0;
    me_nd_valid_nitems(expr, 0, 1, &valid);
    if (valid != 1) {
        printf("FAILED reductions valid block1 (got=%lld)\n", (long long)valid);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr, ptrs1, 1, out_buf, padded, 0, 1, NULL) != ME_EVAL_SUCCESS || out_buf[0] != 3.0) {
        printf("FAILED reductions sum block1 (out=%g)\n", out_buf[0]);
        status = 1;
        goto cleanup;
    }
    if (out_buf[1] != 123.0) {
        printf("FAILED reductions: scalar output overwrote tail (tail=%g)\n", out_buf[1]);
        status = 1;
        goto cleanup;
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_reductions_prod(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    int64_t shape[1] = {4};
    int32_t chunkshape[1] = {3};
    int32_t blockshape[1] = {2};
    const int padded = 2;

    me_variable vars[] = {{"x", ME_FLOAT64}};
    if (me_compile_nd("prod(x)", vars, 1, ME_FLOAT64, 1,
                      shape, chunkshape, blockshape, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd prod: %d\n", err);
        return 1;
    }

    double out_buf[2] = {-1.0, -1.0};
    double block0[2] = {2.0, 3.0}; /* valid=2 */
    const void* ptrs0[] = {block0};
    int64_t valid = 0;
    me_nd_valid_nitems(expr, 0, 0, &valid);
    if (valid != 2) {
        printf("FAILED prod valid block0 (got=%lld)\n", (long long)valid);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr, ptrs0, 1, out_buf, padded, 0, 0, NULL) != ME_EVAL_SUCCESS || out_buf[0] != 6.0) {
        printf("FAILED prod block0 (out=%g)\n", out_buf[0]);
        status = 1;
        goto cleanup;
    }

    double block1[2] = {4.0, 0.0}; /* valid=1 */
    const void* ptrs1[] = {block1};
    out_buf[0] = -1.0;
    me_nd_valid_nitems(expr, 0, 1, &valid);
    if (valid != 1) {
        printf("FAILED prod valid block1 (got=%lld)\n", (long long)valid);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr, ptrs1, 1, out_buf, padded, 0, 1, NULL) != ME_EVAL_SUCCESS || out_buf[0] != 4.0) {
        printf("FAILED prod block1 (out=%g)\n", out_buf[0]);
        status = 1;
        goto cleanup;
    }

cleanup:
    me_free(expr);
    return status;
}

static int test_nd_predicate_reductions(void) {
    int status = 0;
    int err = 0;
    me_expr* expr_sum = NULL;
    me_expr* expr_sum_left = NULL;
    me_expr* expr_any = NULL;
    me_expr* expr_all = NULL;
    int64_t shape[1] = {3};
    int32_t chunkshape[1] = {3};
    int32_t blockshape[1] = {2};
    const int padded = 2;

    me_variable vars[] = {{"x", ME_INT32}};

    if (me_compile_nd("sum(x > 1)", vars, 1, ME_INT64, 1,
                      shape, chunkshape, blockshape, &err, &expr_sum) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd pred sum: %d\n", err);
        return 1;
    }
    if (me_compile_nd("sum(1 < x)", vars, 1, ME_INT64, 1,
                      shape, chunkshape, blockshape, &err, &expr_sum_left) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd pred sum left: %d\n", err);
        status = 1;
        goto cleanup;
    }
    if (me_compile_nd("any(x == 2)", vars, 1, ME_BOOL, 1,
                      shape, chunkshape, blockshape, &err, &expr_any) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd pred any: %d\n", err);
        status = 1;
        goto cleanup;
    }
    if (me_compile_nd("all(x == 2)", vars, 1, ME_BOOL, 1,
                      shape, chunkshape, blockshape, &err, &expr_all) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd pred all: %d\n", err);
        status = 1;
        goto cleanup;
    }

    int32_t block0[2] = {0, 2};
    int32_t block1[2] = {3, 0}; /* last is padding */
    const void* ptrs0[] = {block0};
    const void* ptrs1[] = {block1};
    int64_t valid = 0;

    int64_t out_i64[2] = {-1, -1};
    bool out_b[2] = {false, true};

    me_nd_valid_nitems(expr_sum, 0, 0, &valid);
    if (valid != 2) {
        printf("FAILED pred valid block0 (got=%lld)\n", (long long)valid);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr_sum, ptrs0, 1, out_i64, padded, 0, 0, NULL) != ME_EVAL_SUCCESS || out_i64[0] != 1) {
        printf("FAILED pred sum block0 (out=%lld)\n", (long long)out_i64[0]);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr_sum_left, ptrs0, 1, out_i64, padded, 0, 0, NULL) != ME_EVAL_SUCCESS || out_i64[0] != 1) {
        printf("FAILED pred sum left block0 (out=%lld)\n", (long long)out_i64[0]);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr_any, ptrs0, 1, out_b, padded, 0, 0, NULL) != ME_EVAL_SUCCESS || !out_b[0]) {
        printf("FAILED pred any block0 (out=%d)\n", (int)out_b[0]);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr_all, ptrs0, 1, out_b, padded, 0, 0, NULL) != ME_EVAL_SUCCESS || out_b[0]) {
        printf("FAILED pred all block0 (out=%d)\n", (int)out_b[0]);
        status = 1;
        goto cleanup;
    }

    me_nd_valid_nitems(expr_sum, 0, 1, &valid);
    if (valid != 1) {
        printf("FAILED pred valid block1 (got=%lld)\n", (long long)valid);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr_sum, ptrs1, 1, out_i64, padded, 0, 1, NULL) != ME_EVAL_SUCCESS || out_i64[0] != 1) {
        printf("FAILED pred sum block1 (out=%lld)\n", (long long)out_i64[0]);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr_any, ptrs1, 1, out_b, padded, 0, 1, NULL) != ME_EVAL_SUCCESS || out_b[0]) {
        printf("FAILED pred any block1 (out=%d)\n", (int)out_b[0]);
        status = 1;
        goto cleanup;
    }
    if (me_eval_nd(expr_all, ptrs1, 1, out_b, padded, 0, 1, NULL) != ME_EVAL_SUCCESS || out_b[0]) {
        printf("FAILED pred all block1 (out=%d)\n", (int)out_b[0]);
        status = 1;
        goto cleanup;
    }

cleanup:
    me_free(expr_sum);
    me_free(expr_sum_left);
    me_free(expr_any);
    me_free(expr_all);
    return status;
}

static int test_big_stress(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    const int64_t shape[3] = {20000, 20000, 20000};
    const int32_t chunkshape[3] = {250, 250, 250};
    const int32_t blockshape[3] = {32, 64, 64};
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

static int test_nd_mixed_reductions(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    const int64_t shape[3] = {20000, 20000, 20000};
    const int32_t chunkshape[3] = {250, 250, 250};
    const int32_t blockshape[3] = {32, 64, 64};

    const int padded_items = blockshape[0] * blockshape[1] * blockshape[2];
    me_variable vars[] = {{"x", ME_FLOAT64}};

    if (me_compile_nd("prod(sin(x)**2 + cos(x)**2) + sum(sin(x)**2 + cos(x)**2)",
                      vars, 1, ME_FLOAT64, 3,
                      shape, chunkshape, blockshape, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd mixed reductions: %d\n", err);
        return 1;
    }

    double* in = malloc((size_t)padded_items * sizeof(double));
    double* out_buf = malloc((size_t)padded_items * sizeof(double));
    if (!in || !out_buf) {
        printf("FAILED alloc for mixed reductions\n");
        me_free(expr);
        free(in);
        free(out_buf);
        return 1;
    }

    /* Use monotonically increasing x in [0,1]; sin^2+cos^2 stays 1.
       Explicitly zero padded tail to mimic b2nd padding. */
    for (int i = 0; i < padded_items; i++) {
        in[i] = (double)i / (double)padded_items;
        out_buf[i] = -1.0;
    }
    /* Zero padded rows/cols in the block buffer. */
    for (int z = blockshape[2]; z < blockshape[2]; z++) (void)z; /* placeholder to keep structure */
    /* In 3D block, padding only on highest offsets; simpler: zero any coordinates that fall outside chunkshape tail. */
    int64_t nblocks_dim0 = (chunkshape[0] + blockshape[0] - 1) / blockshape[0];
    int64_t nblocks_dim1 = (chunkshape[1] + blockshape[1] - 1) / blockshape[1];
    int64_t nblocks_dim2 = (chunkshape[2] + blockshape[2] - 1) / blockshape[2];
    /* For interior block (nblock=0), no padding; for edge block we zero tail after computing valid dims. */
    int64_t valid0 = (chunkshape[0] < blockshape[0]) ? chunkshape[0] : blockshape[0];
    int64_t valid1 = (chunkshape[1] < blockshape[1]) ? chunkshape[1] : blockshape[1];
    int64_t valid2 = (chunkshape[2] < blockshape[2]) ? chunkshape[2] : blockshape[2];
    for (int z = valid2; z < blockshape[2]; z++) {
        for (int y = 0; y < blockshape[1]; y++) {
            for (int x = 0; x < blockshape[0]; x++) {
                int idx = (x * blockshape[1] + y) * blockshape[2] + z;
                in[idx] = 0.0;
            }
        }
    }
    for (int y = valid1; y < blockshape[1]; y++) {
        for (int x = 0; x < blockshape[0]; x++) {
            for (int z = 0; z < blockshape[2]; z++) {
                int idx = (x * blockshape[1] + y) * blockshape[2] + z;
                in[idx] = 0.0;
            }
        }
    }
    for (int x = valid0; x < blockshape[0]; x++) {
        for (int y = 0; y < blockshape[1]; y++) {
            for (int z = 0; z < blockshape[2]; z++) {
                int idx = (x * blockshape[1] + y) * blockshape[2] + z;
                in[idx] = 0.0;
            }
        }
    }
    const void* ptrs[] = {in};

    /* Interior full block (no padding) */
    int64_t nchunk = 0;
    int64_t nblock = 0;
    const double expected_full = 1.0 + (double)padded_items;
    int rc = me_eval_nd(expr, ptrs, 1, out_buf, padded_items, nchunk, nblock, NULL);
    if (rc != ME_EVAL_SUCCESS || out_buf[0] != expected_full) {
        printf("FAILED mixed reductions full block (rc=%d, out=%g, exp=%g)\n", rc, out_buf[0], expected_full);
        status = 1;
        goto cleanup;
    }

    /* Edge chunk/block to exercise padding */
    int64_t nchunks_dim0 = (shape[0] + chunkshape[0] - 1) / chunkshape[0];
    int64_t nchunks_dim1 = (shape[1] + chunkshape[1] - 1) / chunkshape[1];
    int64_t nchunks_dim2 = (shape[2] + chunkshape[2] - 1) / chunkshape[2];
    nchunk = (nchunks_dim0 - 1) * nchunks_dim1 * nchunks_dim2 + (nchunks_dim1 - 1) * nchunks_dim2 + (nchunks_dim2 - 1);
    nblock = (nblocks_dim0 - 1) * nblocks_dim1 * nblocks_dim2 +
             (nblocks_dim1 - 1) * nblocks_dim2 +
             (nblocks_dim2 - 1);

    int64_t valid = 0;
    me_nd_valid_nitems(expr, nchunk, nblock, &valid);
    rc = me_eval_nd(expr, ptrs, 1, out_buf, padded_items, nchunk, nblock, NULL);
    if (rc != ME_EVAL_SUCCESS || out_buf[0] != 1.0 + (double)valid) {
        printf("FAILED mixed reductions edge block (rc=%d, valid=%lld, out=%g exp=%g)\n",
               rc, (long long)valid, out_buf[0], 1.0 + (double)valid);
        status = 1;
    }

cleanup:
    free(in);
    free(out_buf);
    me_free(expr);
    return status;
}

static int test_nd_all_padded_reductions(void) {
    int status = 0;
    int err = 0;
    me_expr* expr = NULL;
    const int64_t shape[3] = {310, 305, 299};
    const int32_t chunkshape[3] = {200, 180, 170}; /* padding in chunks */
    const int32_t blockshape[3] = {90, 90, 90};    /* padding in blocks */

    const int64_t padded_items = (int64_t)blockshape[0] * blockshape[1] * blockshape[2]; /* 729000 */
    me_variable vars[] = {{"x", ME_FLOAT64}};

    if (me_compile_nd("prod(x) + sum(x) + min(x)", vars, 1, ME_FLOAT64, 3,
                      shape, chunkshape, blockshape, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd all-padded reductions: %d\n", err);
        return 1;
    }

    double* in = malloc((size_t)padded_items * sizeof(double));
    double* out_buf = malloc((size_t)padded_items * sizeof(double));
    if (!in || !out_buf) {
        printf("FAILED alloc in all-padded reductions\n");
        free(in); free(out_buf); me_free(expr);
        return 1;
    }

    for (int64_t i = 0; i < padded_items; i++) {
        in[i] = 1.0;      /* valid values all 1 -> prod=1, sum=N, min=1 */
        out_buf[i] = -1.0;
    }
    const void* ptrs[] = {in};

    /* Walk every chunk and block to ensure padding handling everywhere. */
    const int64_t nchunks_dim0 = (shape[0] + chunkshape[0] - 1) / chunkshape[0];
    const int64_t nchunks_dim1 = (shape[1] + chunkshape[1] - 1) / chunkshape[1];
    const int64_t nchunks_dim2 = (shape[2] + chunkshape[2] - 1) / chunkshape[2];
    const int64_t nblocks_dim0 = (chunkshape[0] + blockshape[0] - 1) / blockshape[0];
    const int64_t nblocks_dim1 = (chunkshape[1] + blockshape[1] - 1) / blockshape[1];
    const int64_t nblocks_dim2 = (chunkshape[2] + blockshape[2] - 1) / blockshape[2];

    for (int64_t c0 = 0; c0 < nchunks_dim0; c0++) {
        for (int64_t c1 = 0; c1 < nchunks_dim1; c1++) {
            for (int64_t c2 = 0; c2 < nchunks_dim2; c2++) {
                const int64_t nchunk = c0 * nchunks_dim1 * nchunks_dim2 +
                                       c1 * nchunks_dim2 +
                                       c2;
                for (int64_t b0 = 0; b0 < nblocks_dim0; b0++) {
                    for (int64_t b1 = 0; b1 < nblocks_dim1; b1++) {
                        for (int64_t b2 = 0; b2 < nblocks_dim2; b2++) {
                            const int64_t nblock = b0 * nblocks_dim1 * nblocks_dim2 +
                                                   b1 * nblocks_dim2 +
                                                   b2;
                            int64_t valid = 0;
                            me_nd_valid_nitems(expr, nchunk, nblock, &valid);
                            memset(out_buf, 0, (size_t)padded_items * sizeof(double));
                            int rc = me_eval_nd(expr, ptrs, 1, out_buf, padded_items, nchunk, nblock, NULL);
                            const double expected = 1.0 /* prod */ + (double)valid /* sum */ + 1.0 /* min */;
                            if (rc != ME_EVAL_SUCCESS) {
                                printf("FAILED all-padded reductions rc=%d (chunk=%lld block=%lld)\n",
                                       rc, (long long)nchunk, (long long)nblock);
                                status = 1;
                                goto cleanup;
                            }
                            int64_t nz = 0;
                            for (int64_t i = 0; i < padded_items; i++) {
                                double v = out_buf[i];
                                if (v != 0.0) {
                                    nz++;
                                    if (v != expected) {
                                        printf("FAILED all-padded reductions val=%g exp=%g chunk=%lld block=%lld idx=%lld\n",
                                               v, expected, (long long)nchunk, (long long)nblock, (long long)i);
                                        status = 1;
                                        goto cleanup;
                                    }
                                }
                            }
                            if (nz != valid) {
                                printf("FAILED all-padded reductions nz=%lld valid=%lld chunk=%lld block=%lld\n",
                                       (long long)nz, (long long)valid, (long long)nchunk, (long long)nblock);
                                status = 1;
                                goto cleanup;
                            }
                        }
                    }
                }
            }
        }
    }

cleanup:
    free(in);
    free(out_buf);
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

    printf("Test 4: ND reductions with padding\n");
    int t4 = test_nd_reductions();
    failed |= t4;
    printf("Result: %s\n\n", t4 ? "FAIL" : "PASS");

    printf("Test 5: ND reductions (prod) with padding\n");
    int t5 = test_nd_reductions_prod();
    failed |= t5;
    printf("Result: %s\n\n", t5 ? "FAIL" : "PASS");

    printf("Test 6: Large 3D stress (no real allocation beyond block)\n");
    int t6 = test_big_stress();
    failed |= t6;
    printf("Result: %s\n\n", t6 ? "FAIL" : "PASS");

    printf("Test 7: Mixed reductions (sum + prod) with padding\n");
    int t7 = test_nd_mixed_reductions();
    failed |= t7;
    printf("Result: %s\n\n", t7 ? "FAIL" : "PASS");

    printf("Test 8: All-padded reductions (prod+sum+min) on edge chunk/block\n");
    int t8 = test_nd_all_padded_reductions();
    failed |= t8;
    printf("Result: %s\n\n", t8 ? "FAIL" : "PASS");

    printf("Test 9: Predicate reductions (sum/any/all) with padding\n");
    int t9 = test_nd_predicate_reductions();
    failed |= t9;
    printf("Result: %s\n\n", t9 ? "FAIL" : "PASS");

    printf("Test 10: Unary int32 float math with padding\n");
    int t10 = test_nd_unary_int32_float_math();
    failed |= t10;
    printf("Result: %s\n\n", t10 ? "FAIL" : "PASS");

    printf("Test 11: Unary int32 negative with padding\n");
    int t11 = test_nd_unary_int32_negative_blocks();
    failed |= t11;
    printf("Result: %s\n\n", t11 ? "FAIL" : "PASS");

    printf("Test 12: Unary int32->float64 with padding\n");
    int t12 = test_nd_unary_int32_to_float64_padding();
    failed |= t12;
    printf("Result: %s\n\n", t12 ? "FAIL" : "PASS");

    printf("Test 13: DSL cast intrinsics with ND padding\n");
    int t13 = test_nd_cast_intrinsics_padding();
    failed |= t13;
    printf("Result: %s\n\n", t13 ? "FAIL" : "PASS");

    printf("Test 14: DSL float(_i0) cast with ND padding\n");
    int t14 = test_nd_float_index_cast_padding();
    failed |= t14;
    printf("Result: %s\n\n", t14 ? "FAIL" : "PASS");

    printf("Test 15: DSL int(1.9) cast with ND padding\n");
    int t15 = test_nd_int_constant_cast_padding();
    failed |= t15;
    printf("Result: %s\n\n", t15 ? "FAIL" : "PASS");

    printf("Test 16: DSL cast intrinsics with explicit input + ND padding\n");
    int t16 = test_nd_cast_intrinsics_with_input_padding();
    failed |= t16;
    printf("Result: %s\n\n", t16 ? "FAIL" : "PASS");

    printf("Test 17: DSL bool(x) cast with ND padding\n");
    int t17 = test_nd_bool_cast_numeric_padding();
    failed |= t17;
    printf("Result: %s\n\n", t17 ? "FAIL" : "PASS");

    printf("Test 18: DSL int32 ramp kernel sum regression\n");
    int t18 = test_nd_int32_ramp_kernel_sum();
    failed |= t18;
    printf("Result: %s\n\n", t18 ? "FAIL" : "PASS");

    printf("=====================\n");
    printf("Summary: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
