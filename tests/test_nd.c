#include <stdio.h>
#include <stdlib.h>
#include "../src/miniexpr.h"
#include "../src/minctest.h"

int main(void) {
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

    rc = me_eval_nd(expr, ptrs1, 1, out1, 2, 1, 2, NULL);
    if (rc != ME_EVAL_ERR_INVALID_ARG) {
        printf("FAILED me_eval_nd invalid block check (rc=%d)\n", rc);
        me_free(expr);
        return 1;
    }

    me_free(expr);
    printf("me_eval_nd basic tests passed\n");
    return 0;
}
