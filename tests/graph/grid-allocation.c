/* Fault-inject the adapter and numerical evaluator, including cleanup after a
 * packed-buffer allocation succeeds but a later evaluation allocation fails. */
#undef NDEBUG
#include "miniexpr_grid.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static long fail_at = -1, calls, live;
/* libc strdup allocations are outside the injected allocator boundary. Track
 * pointers, not raw free counts, so their frees do not look like negative leaks. */
static void *allocations[4096];

static void remember(void *p) {
    if (!p) return;
    for (size_t i = 0; i < sizeof(allocations) / sizeof(allocations[0]); i++) {
        if (!allocations[i]) {
            allocations[i] = p;
            live++;
            return;
        }
    }
    assert(!"allocation tracking capacity exceeded");
}

static void forget(void *p) {
    if (!p) return;
    for (size_t i = 0; i < sizeof(allocations) / sizeof(allocations[0]); i++) {
        if (allocations[i] == p) {
            allocations[i] = NULL;
            live--;
            return;
        }
    }
}

static bool fail(void) {
    return fail_at >= 0 ? calls++ >= fail_at : (calls++, false);
}

void *graph_test_malloc(size_t n) {
    void *p = fail() ? NULL : malloc(n);
    remember(p);
    return p;
}

void *graph_test_calloc(size_t n, size_t width) {
    void *p = fail() ? NULL : calloc(n, width);
    remember(p);
    return p;
}

void *graph_test_realloc(void *old, size_t n) {
    if (fail()) return NULL;
    if (!n) {
        forget(old);
        free(old);
        return NULL;
    }
    void *p = realloc(old, n);
    if (p) {
        forget(old);
        remember(p);
    }
    return p;
}

void graph_test_free(void *p) {
    forget(p);
    free(p);
}

int main(void) {
    int64_t shape[] = {3, 3};
    int32_t dimensions[] = {4, 4};
    me_grid_spec spec = ME_GRID_SPEC_INITIALIZER;
    spec.ndim = 2;
    spec.shape = shape;
    spec.chunkshape = spec.blockshape = dimensions;
    me_grid *grid = NULL;
    assert(me_grid_create(&spec, &grid, NULL) == ME_SUCCESS);
    me_input_spec input = {"x", ME_FLOAT64, 0};
    me_signature sig = ME_SIGNATURE_INITIALIZER;
    sig.inputs = &input;
    sig.ninputs = 1;
    sig.context_ndim = 2;
    me_kernel *kernel = NULL;
    assert(me_prepare_menudet("0.2", "def k(x):\n    return x + _flat_idx\n",
        &sig, NULL, &kernel, NULL) == ME_SUCCESS);
    double x[16] = {0}, output[16];
    me_buffer buffer = {x, sizeof(x)};
    me_mut_buffer result = {output, sizeof(output)};
    me_run_options run = ME_RUN_OPTIONS_INITIALIZER;
    me_fp_status fp;
    run.fp_status = &fp;
    long baseline = live;
    calls = 0;
    assert(me_grid_eval_block(grid, kernel, 0, 0, &buffer, 1, result, &run, NULL) == ME_SUCCESS);
    long checkpoints = calls;
    assert(checkpoints > 0 && live == baseline);
    for (long i = 0; i < checkpoints; i++) {
        calls = 0;
        fail_at = i;
        fp.flags = 15;
        me_status rc = me_grid_eval_block(grid, kernel, 0, 0, &buffer, 1, result, &run, NULL);
        fail_at = -1;
        assert(rc == ME_ERR_OOM && fp.flags == 0 && live == baseline);
        assert(me_grid_eval_block(grid, kernel, 0, 0, &buffer, 1, result, &run, NULL) == ME_SUCCESS);
        assert(output[0] == 0 && output[10] == 8 && output[15] == 0);
        assert(live == baseline);
    }
    me_kernel_free(kernel);
    me_grid_free(grid);
    if (live) fprintf(stderr, "Unbalanced instrumented allocations: %ld\n", live);
    assert(live == 0);
    puts("Grid allocation failure recovery passed");
    return 0;
}
