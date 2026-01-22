/*
 * Benchmark predicate reductions (sum/any/all on comparisons).
 *
 * Compares me_eval_nd (with padding-aware fast path) against a manual pack
 * + me_eval path to approximate the legacy behavior.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "miniexpr.h"

typedef struct {
    const char *name;
    const char *expr;
    me_dtype out_dtype;
} bench_case_t;

static const bench_case_t CASES[] = {
    {"sum_eq", "sum(x == 1)", ME_INT64},
    {"sum_gt", "sum(x > 1)", ME_INT64},
    {"sum_lt_left", "sum(1 < x)", ME_INT64},
    {"sum_plain", "sum(x)", ME_INT64},
    {"any_eq", "any(x == 1)", ME_BOOL},
    {"all_eq", "all(x == 1)", ME_BOOL}
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

static void compute_valid_len(const int64_t shape[3], const int32_t chunkshape[3],
                              const int32_t blockshape[3], int64_t nchunk, int64_t nblock,
                              int64_t valid_len[3]) {
    int64_t chunk_idx[3];
    int64_t block_idx[3];

    int64_t tmp = nchunk;
    for (int i = 2; i >= 0; i--) {
        int64_t nchunks_d = ceil_div64(shape[i], chunkshape[i]);
        chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
        tmp /= nchunks_d;
    }

    tmp = nblock;
    for (int i = 2; i >= 0; i--) {
        int64_t nblocks_d = ceil_div64(chunkshape[i], blockshape[i]);
        block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
        tmp /= nblocks_d;
    }

    for (int i = 0; i < 3; i++) {
        int64_t chunk_start = chunk_idx[i] * chunkshape[i];
        int64_t chunk_len = shape[i] - chunk_start;
        if (chunk_len > chunkshape[i]) chunk_len = chunkshape[i];
        int64_t block_start = block_idx[i] * blockshape[i];
        if (block_start >= chunk_len) {
            valid_len[i] = 0;
        } else {
            int64_t len = chunk_len - block_start;
            if (len > blockshape[i]) len = blockshape[i];
            valid_len[i] = len;
        }
    }
}

static void compute_stride(const int32_t blockshape[3], int64_t stride[3]) {
    stride[2] = 1;
    stride[1] = stride[2] * blockshape[2];
    stride[0] = stride[1] * blockshape[1];
}

static double seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double run_me_eval_nd(me_expr *expr_nd, const void *ptrs[], int64_t total_chunks,
                             int64_t blocks_per_chunk, int64_t block_items,
                             me_dtype out_dtype, int64_t *total_valid_out) {
    double t0 = seconds_now();
    int64_t total_valid = 0;
    volatile int64_t sink_i64 = 0;
    volatile bool sink_b = false;

    for (int64_t nchunk = 0; nchunk < total_chunks; nchunk++) {
        for (int64_t nblock = 0; nblock < blocks_per_chunk; nblock++) {
            int64_t valid = 0;
            int rc = me_nd_valid_nitems(expr_nd, nchunk, nblock, &valid);
            if (rc != ME_EVAL_SUCCESS) {
                fprintf(stderr, "valid_nitems failed rc=%d\n", rc);
                return 0.0;
            }
            total_valid += valid;
            if (out_dtype == ME_BOOL) {
                bool *out = calloc((size_t)block_items, sizeof(bool));
                if (!out) return 0.0;
                rc = me_eval_nd(expr_nd, ptrs, 1, out, (int)block_items, nchunk, nblock, NULL);
                if (rc != ME_EVAL_SUCCESS) {
                    fprintf(stderr, "eval_nd failed rc=%d\n", rc);
                    free(out);
                    return 0.0;
                }
                sink_b = sink_b || out[0];
                free(out);
            } else {
                int64_t *out = calloc((size_t)block_items, sizeof(int64_t));
                if (!out) return 0.0;
                rc = me_eval_nd(expr_nd, ptrs, 1, out, (int)block_items, nchunk, nblock, NULL);
                if (rc != ME_EVAL_SUCCESS) {
                    fprintf(stderr, "eval_nd failed rc=%d\n", rc);
                    free(out);
                    return 0.0;
                }
                sink_i64 += out[0];
                free(out);
            }
        }
    }

    double t1 = seconds_now();
    if (total_valid_out) *total_valid_out = total_valid;
    (void)sink_i64;
    (void)sink_b;
    return t1 - t0;
}

static double run_pack_eval(me_expr *expr_flat, const int32_t *block,
                            const int64_t shape[3], const int32_t chunkshape[3],
                            const int32_t blockshape[3], int64_t total_chunks,
                            int64_t blocks_per_chunk, int64_t block_items,
                            me_dtype out_dtype) {
    double t0 = seconds_now();
    int64_t valid_len[3];
    int64_t stride[3];
    compute_stride(blockshape, stride);

    volatile int64_t sink_i64 = 0;
    volatile bool sink_b = false;

    for (int64_t nchunk = 0; nchunk < total_chunks; nchunk++) {
        for (int64_t nblock = 0; nblock < blocks_per_chunk; nblock++) {
            compute_valid_len(shape, chunkshape, blockshape, nchunk, nblock, valid_len);
            int64_t total_iters = valid_len[0] * valid_len[1] * valid_len[2];
            if (total_iters <= 0) continue;

            int32_t *packed = malloc((size_t)block_items * sizeof(int32_t));
            if (!packed) return 0.0;

            int64_t indices[3] = {0, 0, 0};
            int64_t write_idx = 0;
            for (int64_t it = 0; it < total_iters; it++) {
                int64_t off = indices[0] * stride[0] + indices[1] * stride[1] + indices[2] * stride[2];
                packed[write_idx++] = block[off];
                for (int i = 2; i >= 0; i--) {
                    indices[i]++;
                    if (indices[i] < valid_len[i]) break;
                    indices[i] = 0;
                }
            }

            const void *ptrs[] = {packed};
            if (out_dtype == ME_BOOL) {
                bool out = false;
                int rc = me_eval(expr_flat, ptrs, 1, &out, (int)total_iters, NULL);
                if (rc != ME_EVAL_SUCCESS) {
                    fprintf(stderr, "me_eval failed rc=%d\n", rc);
                    return 0.0;
                }
                sink_b = sink_b || out;
            } else {
                int64_t out = 0;
                int rc = me_eval(expr_flat, ptrs, 1, &out, (int)total_iters, NULL);
                if (rc != ME_EVAL_SUCCESS) {
                    fprintf(stderr, "me_eval failed rc=%d\n", rc);
                    return 0.0;
                }
                sink_i64 += out;
            }
            free(packed);
        }
    }

    double t1 = seconds_now();
    (void)sink_i64;
    (void)sink_b;
    return t1 - t0;
}

int main(void) {
    const size_t sizes_kb[] = {1024, 4096, 16384};
    const int nsizes = (int)(sizeof(sizes_kb) / sizeof(sizes_kb[0]));

    const int32_t chunkshape[3] = {64, 64, 64};
    const int32_t blockshape[3] = {24, 24, 24};

    printf("Predicate Reduction ND Benchmark\n");
    printf("Exprs: sum/any/all on comparisons to scalar\n");
    printf("chunkshape=(%d,%d,%d) blockshape=(%d,%d,%d)\n\n",
           chunkshape[0], chunkshape[1], chunkshape[2],
           blockshape[0], blockshape[1], blockshape[2]);

    for (int s = 0; s < nsizes; s++) {
        size_t kb = sizes_kb[s];
        int64_t target_items = (int64_t)(kb * 1024) / (int64_t)sizeof(int32_t);
        int64_t shape[3];
        shape_near_cube(target_items, shape);
        if (shape[0] % chunkshape[0] == 0 && shape[1] % chunkshape[1] == 0 && shape[2] % chunkshape[2] == 0) {
            shape[0] += 1; /* ensure chunk padding */
        }

        int64_t block_items = (int64_t)blockshape[0] * blockshape[1] * blockshape[2];

        int32_t *block = malloc((size_t)block_items * sizeof(int32_t));
        if (!block) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        for (int64_t i = 0; i < block_items; i++) {
            block[i] = (int32_t)(i % 5);
        }
        const void *ptrs[] = {block};

        int64_t nchunks_dim0 = ceil_div64(shape[0], chunkshape[0]);
        int64_t nchunks_dim1 = ceil_div64(shape[1], chunkshape[1]);
        int64_t nchunks_dim2 = ceil_div64(shape[2], chunkshape[2]);
        int64_t total_chunks = nchunks_dim0 * nchunks_dim1 * nchunks_dim2;

        int64_t nblocks_dim0 = ceil_div64(chunkshape[0], blockshape[0]);
        int64_t nblocks_dim1 = ceil_div64(chunkshape[1], blockshape[1]);
        int64_t nblocks_dim2 = ceil_div64(chunkshape[2], blockshape[2]);
        int64_t blocks_per_chunk = nblocks_dim0 * nblocks_dim1 * nblocks_dim2;

        printf("\nSize: %zu KB (shape=%lld,%lld,%lld)\n", kb,
               (long long)shape[0], (long long)shape[1], (long long)shape[2]);
        printf("%-12s  %-10s  %-12s  %-10s\n", "case", "nd(ms)", "pack_nd(ms)", "speedup");

        for (size_t c = 0; c < sizeof(CASES) / sizeof(CASES[0]); c++) {
            const bench_case_t *bc = &CASES[c];

            me_variable vars_nd[] = {{"x", ME_INT32}};
            me_expr *expr_nd = NULL;
            int err = 0;
            if (me_compile_nd(bc->expr, vars_nd, 1, bc->out_dtype, 3,
                              shape, chunkshape, blockshape, &err, &expr_nd) != ME_COMPILE_SUCCESS) {
                fprintf(stderr, "compile_nd failed for %s (err=%d)\n", bc->name, err);
                free(block);
                return 1;
            }

            me_variable vars_flat[] = {{"x", ME_INT32}};
            me_expr *expr_flat = NULL;
            if (me_compile(bc->expr, vars_flat, 1, bc->out_dtype, &err, &expr_flat) != ME_COMPILE_SUCCESS) {
                fprintf(stderr, "compile failed for %s (err=%d)\n", bc->name, err);
                me_free(expr_nd);
                free(block);
                return 1;
            }

            double t_nd = run_me_eval_nd(expr_nd, ptrs, total_chunks, blocks_per_chunk,
                                         block_items, bc->out_dtype, NULL);
            double t_pack = run_pack_eval(expr_flat, block, shape, chunkshape, blockshape,
                                          total_chunks, blocks_per_chunk, block_items, bc->out_dtype);

            double speedup = (t_pack > 0.0) ? (t_pack / t_nd) : 0.0;
            printf("%-12s  %10.2f  %10.2f  %10.2fx\n",
                   bc->name, t_nd * 1e3, t_pack * 1e3, speedup);

            me_free(expr_nd);
            me_free(expr_flat);
        }

        free(block);
    }

    return 0;
}
