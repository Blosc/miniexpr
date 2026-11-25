/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "miniexpr_grid.h"
#ifdef MINIEXPR_H
#error "Stable grid headers must not expose the callback-bearing compatibility API"
#endif

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

static void mixed_width_runs(void) {
    int32_t dimensions[] = {4, 4, 4};
    char tags[64 * 3];
    double x[64], output[64];
    int16_t y[64];
    for (int i = 0; i < 64; i++) {
        memcpy(tags + 3 * i, "tag", 3);
        x[i] = i;
        y[i] = 2;
    }
    me_input_spec inputs[] = {{"tag", ME_BYTES, 3}, {"x", ME_FLOAT64, 0}, {"y", ME_INT16, 0}};
    me_buffer buffers[] = {{tags, sizeof(tags)}, {x, sizeof(x)}, {y, sizeof(y)}};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = inputs;
    sig.ninputs = 3;
    sig.context_ndim = 3;
    me_kernel *kernel = NULL;
    assert(me_prepare_menudet("0.2", "def k(tag, x, y):\n    return x + y + _flat_idx\n",
        &sig, NULL, &kernel, NULL) == ME_SUCCESS);
    for (int edge = 0; edge < 4; edge++) {
        int64_t shape[] = {4, 4, 4};
        if (edge < 3) shape[edge] = 3;
        else shape[0] = shape[1] = shape[2] = 3;
        me_grid_spec spec = ME_GRID_SPEC_INITIALIZER;
        spec.ndim = 3;
        spec.shape = shape;
        spec.chunkshape = spec.blockshape = dimensions;
        me_grid *grid = NULL;
        assert(me_grid_create(&spec, &grid, NULL) == ME_SUCCESS);
        assert(me_grid_eval_block(grid, kernel, 0, 0, buffers, 3,
            (me_mut_buffer){output, sizeof(output)}, NULL, NULL) == ME_SUCCESS);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int physical = i * 16 + j * 4 + k;
                    double expected = i < shape[0] && j < shape[1] && k < shape[2] ?
                        x[physical] + 2 + (i * shape[1] + j) * shape[2] + k : 0;
                    assert(output[physical] == expected);
                }
            }
        }
        me_grid_free(grid);
    }
    me_kernel_free(kernel);
}

int main(void) {
    mixed_width_runs();
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
    me_fp_status fp = {15, false};
    me_run_options run = ME_RUN_OPTIONS_INITIALIZER;
    run.fp_status = &fp;
    assert(me_grid_eval_block(grid, kernel, -1, 0, &input, 1, output, &run, NULL) == ME_ERR_INVALID_ARG);
    assert(fp.flags == 0);
#ifndef __EMSCRIPTEN__
    assert(fp.supported);
#else
    assert(!fp.supported);
#endif
    run.struct_size = ME_RUN_OPTIONS_V1_SIZE - 1;
    assert(me_grid_eval_block(grid, kernel, 3, 1, &input, 1, output, &run, NULL) == ME_ERR_INVALID_ARG);
    run.struct_size = sizeof(run);
    run.raise_mask = 16;
    assert(me_grid_eval_block(grid, kernel, 3, 1, &input, 1, output, &run, NULL) == ME_ERR_INVALID_ARG);
    run.raise_mask = 0;
    run.execution = (me_run_execution)99;
    assert(me_grid_eval_block(grid, kernel, 3, 1, &input, 1, output, &run, NULL) == ME_ERR_INVALID_ARG);
    run.execution = ME_RUN_INTERPRET;
    assert(me_grid_eval_block(grid, kernel, 0, 1, &input, 1, output, &run, NULL) == ME_SUCCESS);
    assert(out[0] == 13 && out[3] == 28);
    assert(out[1] == 0 && out[2] == 0 && out[4] == 0 && out[5] == 0);
    run.execution = ME_RUN_PREPARED;
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
    assert(me_grid_eval_block(grid, kernel, 3, 1, &input, 1, output, NULL, NULL) == ME_ERR_INVALID_ARG);
    assert(me_grid_eval_block(grid, kernel, 0, 1, &input, 1, output, NULL, NULL) == ME_ERR_INVALID_ARG);
    input.capacity++;
    assert(me_grid_describe_block(grid, -1, 0, &info, NULL) == ME_ERR_INVALID_ARG);
    assert(me_grid_describe_block(grid, 4, 0, &info, NULL) == ME_ERR_INVALID_ARG);
    assert(me_grid_describe_block(grid, 0, 2, &info, NULL) == ME_ERR_INVALID_ARG);
    struct { me_grid_block_info value; uint64_t suffix; } extended = {ME_GRID_BLOCK_INFO_INITIALIZER, UINT64_MAX};
    extended.value.struct_size = sizeof(extended);
    assert(me_grid_describe_block(grid, 0, 0, &extended.value, NULL) == ME_SUCCESS);
    assert(extended.value.struct_size == sizeof(extended) && extended.suffix == UINT64_MAX);
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
    spec.struct_size = ME_GRID_SPEC_V1_SIZE;
    assert(me_grid_create(&spec, &grid, NULL) == ME_SUCCESS);
    assert(me_grid_describe_block(grid, 0, 0, &info, NULL) == ME_SUCCESS);
    assert(info.ndim == 0 && info.valid_nitems == 1 && info.padded_nitems == 1);
    me_grid_free(grid);
    me_grid_free(NULL);
    puts("Padded grid API passed");
    return 0;
}
