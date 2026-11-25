/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "miniexpr_grid.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Preparation is excluded. Compare identical builds before/after grid changes;
 * these are adapter+interpreter timings, not isolated memcpy bandwidth. */
static int measure(const char *name, int rows, int columns, int partial, int iterations) {
    int64_t shape[] = {rows - partial, columns - partial};
    int32_t chunks[] = {rows, columns}, blocks[] = {rows, columns};
    me_grid_spec spec = ME_GRID_SPEC_INITIALIZER;
    spec.ndim = 2;
    spec.shape = shape;
    spec.chunkshape = chunks;
    spec.blockshape = blocks;
    me_grid *grid = NULL;
    me_kernel *kernel = NULL;
    me_input_spec inputs[] = {{"x", ME_FLOAT64, 0}, {"y", ME_FLOAT64, 0}};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = inputs;
    sig.ninputs = 2;
    me_error error;
    if (me_grid_create(&spec, &grid, &error) != ME_SUCCESS) return 1;
    if (me_prepare_formula("x + y", &sig, NULL, &kernel, &error) != ME_SUCCESS) {
        me_grid_free(grid);
        return 1;
    }
    size_t items = (size_t)rows * (size_t)columns;
    double *storage = malloc(3 * items * sizeof(double));
    if (!storage) {
        me_kernel_free(kernel);
        me_grid_free(grid);
        return 1;
    }
    for (size_t i = 0; i < 2 * items; i++) storage[i] = 1;
    me_buffer buffers[] = {{storage, items * sizeof(double)}, {storage + items, items * sizeof(double)}};
    me_mut_buffer output = {storage + 2 * items, items * sizeof(double)};
    int failed = 0;
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        if (me_grid_eval_block(grid, kernel, 0, 0, buffers, 2, output, NULL, &error) != ME_SUCCESS) {
            failed = 1;
            break;
        }
    }
    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    double *result = output.data;
    if (result[0] != 2 || result[items - 1] != (partial ? 0 : 2)) failed = 1;
    printf("%-16s %8d %12.3f\n", name, iterations, seconds * 1e6 / iterations);
    free(storage);
    me_kernel_free(kernel);
    me_grid_free(grid);
    return failed;
}

int main(void) {
    puts("case             iterations      us/call");
    int failed = measure("full-64x64", 64, 64, 0, 10000);
    failed |= measure("edge-64x64", 64, 64, 1, 10000);
    failed |= measure("edge-8x8", 8, 8, 1, 10000);
    return failed;
}
