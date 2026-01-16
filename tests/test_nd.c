#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"
#include "../src/minctest.h"

int main(void) {
    /* 1D basic */
    int64_t shape[1] = {5};
    int32_t chunkshape[1] = {4};
    int32_t blockshape[1] = {2};
    me_variable vars[] = {{"x"}};
    int err = 0;
    me_expr* expr = NULL;

    int rc = me_compile_nd("x", vars, 1, ME_FLOAT64, 1,
                           shape, chunkshape, blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd: %d (err=%d)\n", rc, err);
        return 1;
    }

    const void* ptrs0[] = {NULL};
    double block0[2] = {1.0, 2.0};
    double out0[2] = {-1.0, -1.0};
    ptrs0[0] = block0;
    rc = me_eval_nd(expr, ptrs0, 1, out0, 2, 0, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out0[0] != 1.0 || out0[1] != 2.0) {
        printf("FAILED me_eval_nd full block (rc=%d, out=[%g,%g])\n", rc, out0[0], out0[1]);
        me_free(expr);
        return 1;
    }

    const void* ptrs1[] = {NULL};
    double block1[2] = {3.0, 999.0};
    double out1[2] = {-1.0, -1.0};
    ptrs1[0] = block1;
    rc = me_eval_nd(expr, ptrs1, 1, out1, 2, 1, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out1[0] != 3.0 || out1[1] != 0.0) {
        printf("FAILED me_eval_nd padded block (rc=%d, out=[%g,%g])\n", rc, out1[0], out1[1]);
        me_free(expr);
        return 1;
    }

    int64_t valid = -1;
    rc = me_nd_valid_nitems(expr, 1, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 1) {
        printf("FAILED me_nd_valid_nitems (rc=%d, valid=%lld)\n", rc, (long long)valid);
        me_free(expr);
        return 1;
    }

    rc = me_eval_nd(expr, ptrs1, 1, out1, 2, 1, 2, NULL);
    if (rc != ME_EVAL_ERR_INVALID_ARG) {
        printf("FAILED me_eval_nd invalid block check (rc=%d)\n", rc);
        me_free(expr);
        return 1;
    }

    me_free(expr);

    /* 2D mixed dtype with padding */
    int64_t shape2[2] = {3, 5};
    int32_t chunkshape2[2] = {2, 4};
    int32_t blockshape2[2] = {1, 3};
    me_variable vars2[] = {{"x", ME_FLOAT32}, {"y", ME_INT32}};
    rc = me_compile_nd("x + y", vars2, 2, ME_FLOAT64, 2,
                       shape2, chunkshape2, blockshape2, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd 2D: %d (err=%d)\n", rc, err);
        return 1;
    }

    /* Chunk index (1,1) -> nchunk = 3 (C-order), block index (0,0) -> nblock = 0 */
    double out2[3] = {-1.0, -1.0, -1.0};
    float xblock[3] = {10.0f, 20.0f, 30.0f};
    int32_t yblock[3] = {1, 2, 3};
    const void* ptrs2[] = {xblock, yblock};

    rc = me_nd_valid_nitems(expr, 3, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 1) {
        printf("FAILED me_nd_valid_nitems 2D (rc=%d, valid=%lld)\n", rc, (long long)valid);
        me_free(expr);
        return 1;
    }

    rc = me_eval_nd(expr, ptrs2, 2, out2, 3, 3, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out2[0] != 11.0 || out2[1] != 0.0 || out2[2] != 0.0) {
        printf("FAILED me_eval_nd 2D padded (rc=%d, out=[%g,%g,%g])\n", rc, out2[0], out2[1], out2[2]);
        me_free(expr);
        return 1;
    }

    me_free(expr);

    /* 3D partial block with padding */
    int64_t shape3[3] = {3, 4, 5};
    int32_t chunkshape3[3] = {2, 3, 4};
    int32_t blockshape3[3] = {2, 2, 2};
    me_variable vars3[] = {{"a"}};
    rc = me_compile_nd("a * 2", vars3, 1, ME_FLOAT64, 3,
                       shape3, chunkshape3, blockshape3, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("FAILED me_compile_nd 3D: %d (err=%d)\n", rc, err);
        return 1;
    }

    /* Chunk index (1,0,1) -> nchunk = 5; Block (0,0,0) -> nblock = 0 */
    double out3[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    double in3[8] = {1, 2, 3, 4, 5, 6, 7, 8}; /* only first 2 are valid */
    const void* ptrs3[] = {in3};
    rc = me_nd_valid_nitems(expr, 5, 0, &valid);
    if (rc != ME_EVAL_SUCCESS || valid != 2) {
        printf("FAILED me_nd_valid_nitems 3D (rc=%d, valid=%lld)\n", rc, (long long)valid);
        me_free(expr);
        return 1;
    }
    rc = me_eval_nd(expr, ptrs3, 1, out3, 8, 5, 0, NULL);
    if (rc != ME_EVAL_SUCCESS || out3[0] != 2.0 || out3[1] != 4.0) {
        printf("FAILED me_eval_nd 3D valid part (rc=%d, out0=%g, out1=%g)\n", rc, out3[0], out3[1]);
        me_free(expr);
        return 1;
    }
    for (int i = 2; i < 8; i++) {
        if (out3[i] != 0.0) {
            printf("FAILED me_eval_nd 3D padding at idx %d (val=%g)\n", i, out3[i]);
            me_free(expr);
            return 1;
        }
    }

    /* Error when chunk_nitems smaller than padded size */
    rc = me_eval_nd(expr, ptrs3, 1, out3, 4, 5, 0, NULL);
    if (rc != ME_EVAL_ERR_INVALID_ARG) {
        printf("FAILED me_eval_nd 3D insufficient buffer (rc=%d)\n", rc);
        me_free(expr);
        return 1;
    }

    me_free(expr);
    printf("me_eval_nd basic tests passed\n");
    return 0;
}
