/*
 * ND benchmark with padding scenarios for mixed-type evaluation.
 *
 * Expression: (a + b) * c
 *   a: float64
 *   b: float32
 *   c: int16
 *   output: float64
 *
 * Scenarios:
 *   1) Chunk padding only    (chunkshape not dividing shape; blockshape divides chunk)
 *   2) Block padding only    (shape divides chunkshape; blockshape does not divide chunk)
 *   3) Chunk + block padding (neither divides cleanly)
 *
 * The benchmark sweeps array sizes (analogous span to chunk sizes in
 * benchmark_mixed_types.c): 1 KB to 1 GB.
 *
 * For each size and scenario, it visits every chunk and block using me_eval_nd,
 * summing the valid elements processed to report throughput.
 *
 * Note: Input data are synthetic per-block buffers reused across calls; the
 * goal is to measure expression/eval overhead with padding logic, not I/O.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "miniexpr.h"

typedef struct {
    const char *name;
    int32_t chunkshape[3];
    int32_t blockshape[3];
    bool align_shape_to_chunk; /* true => shape multiple of chunkshape (no chunk padding) */
} scenario_t;

static const scenario_t SCENARIOS[] = {
    {.name = "none", .chunkshape = {64, 64, 64}, .blockshape = {32, 32, 32}, .align_shape_to_chunk = true},
    {.name = "chunk-pad", .chunkshape = {64, 64, 48}, .blockshape = {16, 16, 16}, .align_shape_to_chunk = false},
    {.name = "block-pad", .chunkshape = {64, 64, 64}, .blockshape = {24, 24, 24}, .align_shape_to_chunk = true},
    {.name = "chunk+block", .chunkshape = {50, 60, 70}, .blockshape = {18, 20, 24}, .align_shape_to_chunk = false},
};

static int64_t ceil_div64(int64_t a, int64_t b) {
    return (b == 0) ? 0 : (a + b - 1) / b;
}

static void shape_near_cube(int64_t target_items, int64_t shape[3]) {
    double base = cbrt((double)target_items);
    int64_t s0 = (int64_t)ceil(base);
    int64_t s1 = s0;
    int64_t s2 = ceil_div64(target_items, s0 * s1);
    shape[0] = s0;
    shape[1] = s1;
    shape[2] = s2;
}

static void shape_from_chunks(int64_t target_items, const int32_t chunkshape[3], int64_t shape[3]) {
    const int64_t chunk_nitems = (int64_t)chunkshape[0] * chunkshape[1] * chunkshape[2];
    int64_t chunks_needed = ceil_div64(target_items, chunk_nitems);
    if (chunks_needed < 1) {
        chunks_needed = 1;
    }
    double base = cbrt((double)chunks_needed);
    int64_t c0 = (int64_t)ceil(base);
    int64_t c1 = c0;
    int64_t c2 = ceil_div64(chunks_needed, c0 * c1);
    shape[0] = (int64_t)chunkshape[0] * c0;
    shape[1] = (int64_t)chunkshape[1] * c1;
    shape[2] = (int64_t)chunkshape[2] * c2;
}

static double run_benchmark(const scenario_t *sc, int64_t target_items) {
    int64_t shape[3];
    if (sc->align_shape_to_chunk) {
        shape_from_chunks(target_items, sc->chunkshape, shape);
    }
    else {
        shape_near_cube(target_items, shape);
    }

    const int64_t block_items = (int64_t)sc->blockshape[0] * sc->blockshape[1] * sc->blockshape[2];
    const int64_t chunk_items = (int64_t)sc->chunkshape[0] * sc->chunkshape[1] * sc->chunkshape[2];

    me_variable vars[] = {
        {"a", ME_FLOAT64},
        {"b", ME_FLOAT32},
        {"c", ME_INT16}
    };
    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile_nd("(a + b) * c", vars, 3, ME_AUTO, 3,
                           shape, sc->chunkshape, sc->blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        fprintf(stderr, "Compile failed (%s): rc=%d err=%d\n", sc->name, rc, err);
        return 0.0;
    }

    double *buf_a = malloc((size_t)block_items * sizeof(double));
    float *buf_b = malloc((size_t)block_items * sizeof(float));
    int16_t *buf_c = malloc((size_t)block_items * sizeof(int16_t));
    double *buf_out = malloc((size_t)block_items * sizeof(double));
    if (!buf_a || !buf_b || !buf_c || !buf_out) {
        fprintf(stderr, "Alloc failed (%s)\n", sc->name);
        free(buf_a); free(buf_b); free(buf_c); free(buf_out);
        me_free(expr);
        return 0.0;
    }
    for (int64_t i = 0; i < block_items; i++) {
        buf_a[i] = (double)(i % 97) * 0.01;
        buf_b[i] = (float)((i + 31) % 113) * 0.02f;
        buf_c[i] = (int16_t)((i % 21) - 10);
    }
    const void *ptrs[] = {buf_a, buf_b, buf_c};

    int64_t nchunks_dim0 = ceil_div64(shape[0], sc->chunkshape[0]);
    int64_t nchunks_dim1 = ceil_div64(shape[1], sc->chunkshape[1]);
    int64_t nchunks_dim2 = ceil_div64(shape[2], sc->chunkshape[2]);
    int64_t total_chunks = nchunks_dim0 * nchunks_dim1 * nchunks_dim2;

    int64_t nblocks_dim0 = ceil_div64(sc->chunkshape[0], sc->blockshape[0]);
    int64_t nblocks_dim1 = ceil_div64(sc->chunkshape[1], sc->blockshape[1]);
    int64_t nblocks_dim2 = ceil_div64(sc->chunkshape[2], sc->blockshape[2]);
    int64_t blocks_per_chunk = nblocks_dim0 * nblocks_dim1 * nblocks_dim2;
    double throughput = 0.0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int64_t total_valid = 0;
    for (int64_t nchunk = 0; nchunk < total_chunks; nchunk++) {
        for (int64_t nblock = 0; nblock < blocks_per_chunk; nblock++) {
            int64_t valid = 0;
            rc = me_nd_valid_nitems(expr, nchunk, nblock, &valid);
            if (rc != ME_EVAL_SUCCESS) {
                fprintf(stderr, "valid_nitems failed (%s) chunk=%lld block=%lld rc=%d\n",
                        sc->name, (long long)nchunk, (long long)nblock, rc);
                goto cleanup;
            }
            total_valid += valid;
            rc = me_eval_nd(expr, ptrs, 3, buf_out, (int)block_items, nchunk, nblock, NULL);
            if (rc != ME_EVAL_SUCCESS) {
                fprintf(stderr, "eval_nd failed (%s) chunk=%lld block=%lld rc=%d\n",
                        sc->name, (long long)nchunk, (long long)nblock, rc);
                goto cleanup;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    throughput = (elapsed > 0.0) ? (total_valid / elapsed) / 1e6 : 0.0;

cleanup:
    free(buf_a);
    free(buf_b);
    free(buf_c);
    free(buf_out);
    me_free(expr);
    (void)chunk_items; // silence unused in case of early goto
    return throughput;
}

int main(void) {
    const size_t sizes_kb[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576
    };
    const int nsizes = (int)(sizeof(sizes_kb) / sizeof(sizes_kb[0]));

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  ND Mixed-Type Padding Benchmark\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Expression: (a + b) * c  |  a=f64, b=f32, c=i16  | output=f64\n");
    printf("Array sizes: 1 KB → 1 GB (output bytes), 4 padding scenarios\n\n");
    printf("Scenarios (chunkshape -> blockshape):\n");
    printf("  chunk+block : (%d,%d,%d) -> (%d,%d,%d)\n",
           SCENARIOS[3].chunkshape[0], SCENARIOS[3].chunkshape[1], SCENARIOS[3].chunkshape[2],
           SCENARIOS[3].blockshape[0], SCENARIOS[3].blockshape[1], SCENARIOS[3].blockshape[2]);
    printf("  chunk-pad   : (%d,%d,%d) -> (%d,%d,%d)\n",
           SCENARIOS[1].chunkshape[0], SCENARIOS[1].chunkshape[1], SCENARIOS[1].chunkshape[2],
           SCENARIOS[1].blockshape[0], SCENARIOS[1].blockshape[1], SCENARIOS[1].blockshape[2]);
    printf("  block-pad   : (%d,%d,%d) -> (%d,%d,%d)\n",
           SCENARIOS[2].chunkshape[0], SCENARIOS[2].chunkshape[1], SCENARIOS[2].chunkshape[2],
           SCENARIOS[2].blockshape[0], SCENARIOS[2].blockshape[1], SCENARIOS[2].blockshape[2]);
    printf("  none        : (%d,%d,%d) -> (%d,%d,%d)\n\n",
           SCENARIOS[0].chunkshape[0], SCENARIOS[0].chunkshape[1], SCENARIOS[0].chunkshape[2],
           SCENARIOS[0].blockshape[0], SCENARIOS[0].blockshape[1], SCENARIOS[0].blockshape[2]);
    printf("Throughput columns below are in GB/s (22 bytes per row)\n\n");
    printf("%10s  %12s  %12s  %12s  %12s\n", "ArrayKB",
           "chunk+block", "chunk-pad", "block-pad", "none");
    printf("-----------------------------------------------------------------------\n");

    for (int i = 0; i < nsizes; i++) {
        size_t kb = sizes_kb[i];
        int64_t items = (int64_t)(kb * 1024) / (int64_t)sizeof(double);
        if (items < 1) items = 1;

        double gbps[4] = {0.0, 0.0, 0.0, 0.0};
        const size_t nscen = sizeof(SCENARIOS) / sizeof(SCENARIOS[0]);
        for (size_t s = 0; s < nscen; s++) {
            double thr = run_benchmark(&SCENARIOS[s], items); /* Melems/s */
            gbps[s] = thr * 22.0 / 1000.0; /* 22 bytes per element -> GB/s */
        }
        printf("%10zu  %12.2f  %12.2f  %12.2f  %12.2f\n",
               kb, gbps[3], gbps[1], gbps[2], gbps[0]);
        fflush(stdout);
    }

    printf("═══════════════════════════════════════════════════════════════════\n");
    return 0;
}
