/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "miniexpr_grid.h"

static me_kernel *prepare(bool scalar) {
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    sig.context_ndim = scalar ? 0 : 2;
    const char *source = scalar ?
        "def k(x):\n    total = block_sum(x)\n    return total * total\n" :
        "def k(x):\n    return x + _flat_idx\n";
    me_kernel *kernel = NULL;
    me_error error;
    me_status rc = me_prepare_menudet("0.2", source, &sig, NULL, &kernel, &error);
    if (rc) fprintf(stderr, "%s\n", error.message);
    assert(rc == ME_SUCCESS);
    return kernel;
}

int main(void) {
    int64_t shape[] = {3, 5};
    int32_t chunks[] = {2, 4}, blocks[] = {2, 3};
    me_grid_spec spec = ME_GRID_SPEC_INITIALIZER;
    spec.ndim = 2;
    spec.shape = shape;
    spec.chunkshape = chunks;
    spec.blockshape = blocks;
    me_grid *grid = NULL;
    assert(me_grid_create(&spec, &grid, NULL) == ME_SUCCESS);
    me_grid_block_info info = ME_GRID_BLOCK_INFO_INITIALIZER;
    assert(me_grid_describe_block(grid, 0, 1, &info, NULL) == ME_SUCCESS);
    assert(info.padded_nitems == 6 && info.valid_nitems == 2);
    assert(info.origin[0] == 0 && info.origin[1] == 3);
    assert(info.extent[0] == 2 && info.extent[1] == 1);
    double x[] = {10, 999, 999, 20, 999, 999}, out[6];
    me_buffer input = {x, sizeof(x)};
    me_mut_buffer output = {out, sizeof(out)};
    me_kernel *kernel = prepare(false);
    assert(me_grid_eval_block(grid, kernel, 0, 0, &input, 1, output, NULL, NULL) == ME_SUCCESS);
    const double offsets[] = {0, 1, 2, 5, 6, 7};
    for (int i = 0; i < 6; i++) assert(out[i] == x[i] + offsets[i]);
    assert(me_grid_eval_block(grid, kernel, 0, 1, &input, 1, output, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 13 && out[3] == 28);
    assert(out[1] == 0 && out[2] == 0 && out[4] == 0 && out[5] == 0);
    assert(me_grid_eval_block(grid, kernel, 3, 0, &input, 1, output, NULL, NULL) == ME_SUCCESS);
    assert(out[0] == 24);
    for (int i = 1; i < 6; i++) assert(out[i] == 0);
    assert(me_grid_eval_block(grid, kernel, 3, 1, &input, 1, output, NULL, NULL) == ME_SUCCESS);
    for (int i = 0; i < 6; i++) assert(out[i] == 0);
    output.data = x;
    assert(me_grid_eval_block(grid, kernel, 0, 1, &input, 1, output, NULL, NULL) == ME_ERR_INVALID_ARG);
    output.data = out;
    input.capacity--;
    assert(me_grid_eval_block(grid, kernel, 0, 1, &input, 1, output, NULL, NULL) == ME_ERR_INVALID_ARG);
    input.capacity++;
    assert(me_grid_describe_block(grid, -1, 0, &info, NULL) == ME_ERR_INVALID_ARG);
    assert(me_grid_describe_block(grid, 4, 0, &info, NULL) == ME_ERR_INVALID_ARG);
    assert(me_grid_describe_block(grid, 0, 2, &info, NULL) == ME_ERR_INVALID_ARG);
    me_kernel_free(kernel);
    kernel = prepare(true);
    double total = -1;
    output = (me_mut_buffer){&total, sizeof(total)};
    /* One group spanning both physical rows: (10 + 20)^2, NOT 10^2 + 20^2. */
    assert(me_grid_eval_block(grid, kernel, 0, 1, &input, 1, output, NULL, NULL) == ME_SUCCESS);
    assert(total == 900);
    assert(me_grid_eval_block(grid, kernel, 3, 1, &input, 1, output, NULL, NULL) == ME_SUCCESS);
    assert(total == 0);
    me_kernel_free(kernel);
    me_grid_free(grid);
    shape[0] = INT64_MAX;
    shape[1] = 2;
    assert(me_grid_create(&spec, &grid, NULL) == ME_ERR_INVALID_ARG && !grid);
    shape[0] = 0;
    assert(me_grid_create(&spec, &grid, NULL) == ME_SUCCESS);
    assert(me_grid_describe_block(grid, 0, 0, &info, NULL) == ME_ERR_INVALID_ARG);
    me_grid_free(grid);
    spec = (me_grid_spec)ME_GRID_SPEC_INITIALIZER;
    assert(me_grid_create(&spec, &grid, NULL) == ME_SUCCESS);
    assert(me_grid_describe_block(grid, 0, 0, &info, NULL) == ME_SUCCESS);
    assert(info.ndim == 0 && info.valid_nitems == 1 && info.padded_nitems == 1);
    me_grid_free(grid);
    me_grid_free(NULL);
    puts("Padded grid API passed");
    return 0;
}
