/*
 * ND benchmark with padding scenarios for mixed-type DSL evaluation (multi-threaded).
 *
 * DSL:
 *   def kernel(a, b, c):
 *     sum = 0;
 *     for i in range(4):
 *       tmp = (a + b) * c + i
 *       if any(tmp < -1e9):
 *         continue
 *       if any(tmp > 1e12):
 *         continue
 *       sum = sum + tmp
 *     return sum
 *   a: float64
 *   b: float32
 *   c: int16
 *   output: float64
 *
 * Scenarios:
 *   1) Chunk padding only    (chunkshape not dividing shape; blockshape divides chunk)
 *   2) Block padding only    (shape divides chunkshape; blockshape does not divide chunk)
 *   3) Chunk + block padding (neither divides cleanly)
 *   4) None                 (shape multiple of chunkshape; blockshape divides chunkshape)
 *
 * For each scenario, run on a 1 GB logical array with fixed chunk/block shapes,
 * and report throughput for threads 1..12. Also reports a pure C baseline
 * (no padding). ND runs copy blocks from full arrays to avoid cache-only reuse.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "miniexpr.h"

#define MAX_THREADS 12
#define TOTAL_SIZE_MB 512
#define BYTES_PER_ELEMENT 22.0

typedef struct {
    const char *name;
    int32_t chunkshape[3];
    int32_t blockshape[3];
    bool align_shape_to_chunk;
} scenario_t;

typedef struct {
    scenario_t sc;
    me_expr *expr;
    int64_t shape[3];
    int64_t total_items;
    int64_t total_blocks;
    int64_t blocks_per_chunk;
    int64_t nchunks_dim[3];
    int64_t nblocks_dim[3];
    int64_t strides[3];
    int block_items;
    double *full_a;
    float *full_b;
    int16_t *full_c;
    double *full_out;
} bench_scenario_t;

typedef struct {
    const me_expr *expr;
    const double *full_a;
    const float *full_b;
    const int16_t *full_c;
    double *full_out;
    double *buf_a;
    float *buf_b;
    int16_t *buf_c;
    double *buf_out;
    int64_t shape[3];
    int64_t strides[3];
    int64_t nchunks_dim[3];
    int64_t nblocks_dim[3];
    int32_t chunkshape[3];
    int32_t blockshape[3];
    int block_items;
    int64_t start_block;
    int64_t end_block;
    int64_t blocks_per_chunk;
    int *error;
} thread_args_nd_t;

typedef struct {
    const double *a;
    const float *b;
    const int16_t *c;
    double *out;
    double *buf_a;
    float *buf_b;
    int16_t *buf_c;
    double *buf_out;
    int64_t shape[3];
    int64_t strides[3];
    int64_t nchunks_dim[3];
    int64_t nblocks_dim[3];
    int32_t chunkshape[3];
    int32_t blockshape[3];
    int64_t start_idx;
    int64_t count;
    int64_t blocks_per_chunk;
    int block_items;
} thread_args_c_t;

static const scenario_t SCENARIOS[] = {
    {.name = "chunk+block", .chunkshape = {250, 242, 234}, .blockshape = {16, 16, 16}, .align_shape_to_chunk = false},
    {.name = "chunk-pad", .chunkshape = {256, 256, 192}, .blockshape = {16, 16, 16}, .align_shape_to_chunk = false},
    {.name = "block-pad", .chunkshape = {250, 250, 250}, .blockshape = {16, 16, 16}, .align_shape_to_chunk = true},
    {.name = "none", .chunkshape = {256, 256, 256}, .blockshape = {16, 16, 16}, .align_shape_to_chunk = true}
};

static const char *DSL_SOURCE =
    "def kernel(a, b, c):\n"
    "    sum = 0\n"
    "    for i in range(4):\n"
    "        tmp = (a + b) * c + i\n"
    "        if any(tmp < -1e9):\n"
    "            continue\n"
    "        if any(tmp > 1e12):\n"
    "            continue\n"
    "        sum = sum + tmp\n"
    "    return sum\n";

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

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

static void *nd_worker(void *arg) {
    thread_args_nd_t *args = (thread_args_nd_t *)arg;
    const void *vars_block[3] = {args->buf_a, args->buf_b, args->buf_c};
    int64_t bstride0 = (int64_t)args->blockshape[1] * args->blockshape[2];
    int64_t bstride1 = args->blockshape[2];

    for (int64_t idx = args->start_block; idx < args->end_block; idx++) {
        if (*args->error != 0) {
            break;
        }
        int64_t nchunk = idx / args->blocks_per_chunk;
        int64_t nblock = idx - nchunk * args->blocks_per_chunk;

        int64_t chunk_idx[3];
        int64_t block_idx[3];
        int64_t start[3];
        int64_t valid[3];

        int64_t tmp = nchunk;
        for (int i = 2; i >= 0; i--) {
            int64_t nchunks_d = args->nchunks_dim[i];
            chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
            tmp /= nchunks_d;
        }
        tmp = nblock;
        for (int i = 2; i >= 0; i--) {
            int64_t nblocks_d = args->nblocks_dim[i];
            block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
            tmp /= nblocks_d;
        }

        for (int i = 0; i < 3; i++) {
            int64_t chunk_start = chunk_idx[i] * args->chunkshape[i];
            int64_t chunk_len = args->shape[i] - chunk_start;
            if (chunk_len > args->chunkshape[i]) {
                chunk_len = args->chunkshape[i];
            }
            int64_t block_start = block_idx[i] * args->blockshape[i];
            if (block_start >= chunk_len) {
                valid[i] = 0;
            }
            else {
                int64_t len = chunk_len - block_start;
                if (len > args->blockshape[i]) {
                    len = args->blockshape[i];
                }
                valid[i] = len;
            }
            start[i] = chunk_start + block_start;
        }

        if (valid[0] > 0 && valid[1] > 0 && valid[2] > 0) {
            for (int64_t i0 = 0; i0 < valid[0]; i0++) {
                for (int64_t i1 = 0; i1 < valid[1]; i1++) {
                    int64_t src_off = (start[0] + i0) * args->strides[0]
                                      + (start[1] + i1) * args->strides[1]
                                      + start[2];
                    int64_t dst_off = i0 * bstride0 + i1 * bstride1;
                    memcpy(args->buf_a + dst_off, args->full_a + src_off,
                           (size_t)valid[2] * sizeof(double));
                    memcpy(args->buf_b + dst_off, args->full_b + src_off,
                           (size_t)valid[2] * sizeof(float));
                    memcpy(args->buf_c + dst_off, args->full_c + src_off,
                           (size_t)valid[2] * sizeof(int16_t));
                }
            }
        }

        int rc = me_eval_nd(args->expr, vars_block, 3,
                            args->buf_out, args->block_items,
                            nchunk, nblock, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            *args->error = rc;
            break;
        }

        if (valid[0] > 0 && valid[1] > 0 && valid[2] > 0) {
            for (int64_t i0 = 0; i0 < valid[0]; i0++) {
                for (int64_t i1 = 0; i1 < valid[1]; i1++) {
                    int64_t dst_off = (start[0] + i0) * args->strides[0]
                                      + (start[1] + i1) * args->strides[1]
                                      + start[2];
                    int64_t src_off = i0 * bstride0 + i1 * bstride1;
                    memcpy(args->full_out + dst_off, args->buf_out + src_off,
                           (size_t)valid[2] * sizeof(double));
                }
            }
        }
    }
    return NULL;
}


static void *c_worker(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    int64_t start = args->start_idx;
    int64_t end = start + args->count;
    for (int64_t i = start; i < end; i++) {
        double sum = 0.0;
        for (int iter = 0; iter < 4; iter++) {
            double tmp = (args->a[i] + (double)args->b[i]) * (double)args->c[i] + (double)iter;
            if (tmp < -1e9) {
                continue;
            }
            if (tmp > 1e12) {
                continue;
            }
            sum += tmp;
        }
        args->out[i] = sum;
    }
    return NULL;
}

static void *c_pack_worker(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    int64_t bstride0 = (int64_t)args->blockshape[1] * args->blockshape[2];
    int64_t bstride1 = args->blockshape[2];

    for (int64_t idx = args->start_idx; idx < args->start_idx + args->count; idx++) {
        int64_t nchunk = idx / args->blocks_per_chunk;
        int64_t nblock = idx - nchunk * args->blocks_per_chunk;

        int64_t chunk_idx[3];
        int64_t block_idx[3];
        int64_t start[3];
        int64_t valid[3];

        int64_t tmp = nchunk;
        for (int i = 2; i >= 0; i--) {
            int64_t nchunks_d = args->nchunks_dim[i];
            chunk_idx[i] = (nchunks_d == 0) ? 0 : (tmp % nchunks_d);
            tmp /= nchunks_d;
        }
        tmp = nblock;
        for (int i = 2; i >= 0; i--) {
            int64_t nblocks_d = args->nblocks_dim[i];
            block_idx[i] = (nblocks_d == 0) ? 0 : (tmp % nblocks_d);
            tmp /= nblocks_d;
        }

        for (int i = 0; i < 3; i++) {
            int64_t chunk_start = chunk_idx[i] * args->chunkshape[i];
            int64_t chunk_len = args->shape[i] - chunk_start;
            if (chunk_len > args->chunkshape[i]) {
                chunk_len = args->chunkshape[i];
            }
            int64_t block_start = block_idx[i] * args->blockshape[i];
            if (block_start >= chunk_len) {
                valid[i] = 0;
            }
            else {
                int64_t len = chunk_len - block_start;
                if (len > args->blockshape[i]) {
                    len = args->blockshape[i];
                }
                valid[i] = len;
            }
            start[i] = chunk_start + block_start;
        }

        if (valid[0] > 0 && valid[1] > 0 && valid[2] > 0) {
            for (int64_t i0 = 0; i0 < valid[0]; i0++) {
                for (int64_t i1 = 0; i1 < valid[1]; i1++) {
                    int64_t src_off = (start[0] + i0) * args->strides[0]
                                      + (start[1] + i1) * args->strides[1]
                                      + start[2];
                    int64_t dst_off = i0 * bstride0 + i1 * bstride1;
                    memcpy(args->buf_a + dst_off, args->a + src_off,
                           (size_t)valid[2] * sizeof(double));
                    memcpy(args->buf_b + dst_off, args->b + src_off,
                           (size_t)valid[2] * sizeof(float));
                    memcpy(args->buf_c + dst_off, args->c + src_off,
                           (size_t)valid[2] * sizeof(int16_t));
                }
            }
        }

        memset(args->buf_out, 0, (size_t)args->block_items * sizeof(double));
        if (valid[0] > 0 && valid[1] > 0 && valid[2] > 0) {
            for (int64_t i0 = 0; i0 < valid[0]; i0++) {
                for (int64_t i1 = 0; i1 < valid[1]; i1++) {
                    int64_t dst_off = i0 * bstride0 + i1 * bstride1;
                    int64_t src_off = (start[0] + i0) * args->strides[0]
                                      + (start[1] + i1) * args->strides[1]
                                      + start[2];
                    for (int64_t i2 = 0; i2 < valid[2]; i2++) {
                        int64_t off = dst_off + i2;
                        double sum = 0.0;
                        for (int iter = 0; iter < 4; iter++) {
                            double tmp = (args->buf_a[off] + (double)args->buf_b[off])
                                         * (double)args->buf_c[off] + (double)iter;
                            if (tmp < -1e9) {
                                continue;
                            }
                            if (tmp > 1e12) {
                                continue;
                            }
                            sum += tmp;
                        }
                        args->buf_out[off] = sum;
                    }
                    memcpy(args->out + src_off, args->buf_out + dst_off,
                           (size_t)valid[2] * sizeof(double));
                }
            }
        }
    }
    return NULL;
}

static double run_threads_nd(const bench_scenario_t *bs, int num_threads) {
    pthread_t threads[MAX_THREADS];
    thread_args_nd_t thread_args[MAX_THREADS];
    double *buf_a[MAX_THREADS];
    float *buf_b[MAX_THREADS];
    int16_t *buf_c[MAX_THREADS];
    double *buf_out[MAX_THREADS];
    int64_t total_blocks = bs->total_blocks;
    int64_t base = total_blocks / num_threads;
    int64_t rem = total_blocks % num_threads;
    int err = 0;

    for (int t = 0; t < num_threads; t++) {
        buf_a[t] = malloc((size_t)bs->block_items * sizeof(double));
        buf_b[t] = malloc((size_t)bs->block_items * sizeof(float));
        buf_c[t] = malloc((size_t)bs->block_items * sizeof(int16_t));
        buf_out[t] = malloc((size_t)bs->block_items * sizeof(double));
        if (!buf_a[t] || !buf_b[t] || !buf_c[t] || !buf_out[t]) {
            err = ME_EVAL_ERR_OOM;
            for (int u = 0; u <= t; u++) {
                free(buf_a[u]);
                free(buf_b[u]);
                free(buf_c[u]);
                free(buf_out[u]);
            }
            return 0.0;
        }
        int64_t count = base + (t < rem ? 1 : 0);
        int64_t start = (int64_t)t * base + (t < rem ? t : rem);
        thread_args[t].expr = bs->expr;
        thread_args[t].full_a = bs->full_a;
        thread_args[t].full_b = bs->full_b;
        thread_args[t].full_c = bs->full_c;
        thread_args[t].full_out = bs->full_out;
        thread_args[t].buf_a = buf_a[t];
        thread_args[t].buf_b = buf_b[t];
        thread_args[t].buf_c = buf_c[t];
        thread_args[t].buf_out = buf_out[t];
        for (int i = 0; i < 3; i++) {
            thread_args[t].shape[i] = bs->shape[i];
            thread_args[t].chunkshape[i] = bs->sc.chunkshape[i];
            thread_args[t].blockshape[i] = bs->sc.blockshape[i];
            thread_args[t].strides[i] = bs->strides[i];
            thread_args[t].nchunks_dim[i] = bs->nchunks_dim[i];
            thread_args[t].nblocks_dim[i] = bs->nblocks_dim[i];
        }
        thread_args[t].block_items = bs->block_items;
        thread_args[t].start_block = start;
        thread_args[t].end_block = start + count;
        thread_args[t].blocks_per_chunk = bs->blocks_per_chunk;
        thread_args[t].error = &err;
        pthread_create(&threads[t], NULL, nd_worker, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    for (int t = 0; t < num_threads; t++) {
        free(buf_a[t]);
        free(buf_b[t]);
        free(buf_c[t]);
        free(buf_out[t]);
    }

    if (err != 0) {
        return 0.0;
    }
    return 1.0;
}

static double run_benchmark_nd(const bench_scenario_t *bs, int num_threads) {
    double t0 = get_time();
    double ok = run_threads_nd(bs, num_threads);
    double t1 = get_time();
    if (ok == 0.0) {
        return 0.0;
    }
    return t1 - t0;
}

static double run_benchmark_c(const double *a, const float *b, const int16_t *c,
                              double *out, int64_t total_items, int num_threads) {
    pthread_t threads[MAX_THREADS];
    thread_args_c_t thread_args[MAX_THREADS];
    int64_t base = total_items / num_threads;
    int64_t rem = total_items % num_threads;

    double t0 = get_time();
    for (int t = 0; t < num_threads; t++) {
        int64_t count = base + (t < rem ? 1 : 0);
        int64_t start = (int64_t)t * base + (t < rem ? t : rem);
        thread_args[t].a = a;
        thread_args[t].b = b;
        thread_args[t].c = c;
        thread_args[t].out = out;
        thread_args[t].start_idx = start;
        thread_args[t].count = count;
        pthread_create(&threads[t], NULL, c_worker, &thread_args[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    double t1 = get_time();
    return t1 - t0;
}

static double run_benchmark_c_pack(const bench_scenario_t *bs, int num_threads) {
    pthread_t threads[MAX_THREADS];
    thread_args_c_t thread_args[MAX_THREADS];
    double *buf_a[MAX_THREADS];
    float *buf_b[MAX_THREADS];
    int16_t *buf_c[MAX_THREADS];
    double *buf_out[MAX_THREADS];
    int64_t total_blocks = bs->total_blocks;
    int64_t base = total_blocks / num_threads;
    int64_t rem = total_blocks % num_threads;

    double t0 = get_time();
    for (int t = 0; t < num_threads; t++) {
        buf_a[t] = malloc((size_t)bs->block_items * sizeof(double));
        buf_b[t] = malloc((size_t)bs->block_items * sizeof(float));
        buf_c[t] = malloc((size_t)bs->block_items * sizeof(int16_t));
        buf_out[t] = malloc((size_t)bs->block_items * sizeof(double));
        if (!buf_a[t] || !buf_b[t] || !buf_c[t] || !buf_out[t]) {
            for (int u = 0; u <= t; u++) {
                free(buf_a[u]);
                free(buf_b[u]);
                free(buf_c[u]);
                free(buf_out[u]);
            }
            return 0.0;
        }
        int64_t count = base + (t < rem ? 1 : 0);
        int64_t start = (int64_t)t * base + (t < rem ? t : rem);
        thread_args[t].a = bs->full_a;
        thread_args[t].b = bs->full_b;
        thread_args[t].c = bs->full_c;
        thread_args[t].out = bs->full_out;
        thread_args[t].buf_a = buf_a[t];
        thread_args[t].buf_b = buf_b[t];
        thread_args[t].buf_c = buf_c[t];
        thread_args[t].buf_out = buf_out[t];
        for (int i = 0; i < 3; i++) {
            thread_args[t].shape[i] = bs->shape[i];
            thread_args[t].chunkshape[i] = bs->sc.chunkshape[i];
            thread_args[t].blockshape[i] = bs->sc.blockshape[i];
            thread_args[t].strides[i] = bs->strides[i];
            thread_args[t].nchunks_dim[i] = bs->nchunks_dim[i];
            thread_args[t].nblocks_dim[i] = bs->nblocks_dim[i];
        }
        thread_args[t].start_idx = start;
        thread_args[t].count = count;
        thread_args[t].blocks_per_chunk = bs->blocks_per_chunk;
        thread_args[t].block_items = bs->block_items;
        pthread_create(&threads[t], NULL, c_pack_worker, &thread_args[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    double t1 = get_time();

    for (int t = 0; t < num_threads; t++) {
        free(buf_a[t]);
        free(buf_b[t]);
        free(buf_c[t]);
        free(buf_out[t]);
    }
    return t1 - t0;
}

static int setup_scenario(bench_scenario_t *bs, const scenario_t *sc, int64_t target_items) {
    bs->sc = *sc;
    if (strcmp(sc->name, "none") == 0) {
        int64_t plane = (int64_t)sc->chunkshape[1] * sc->chunkshape[2];
        int64_t s0 = ceil_div64(target_items, plane);
        if (sc->align_shape_to_chunk) {
            s0 = ceil_div64(s0, sc->chunkshape[0]) * sc->chunkshape[0];
        }
        bs->shape[0] = s0;
        bs->shape[1] = sc->chunkshape[1];
        bs->shape[2] = sc->chunkshape[2];
    }
    else if (sc->align_shape_to_chunk) {
        shape_from_chunks(target_items, sc->chunkshape, bs->shape);
    }
    else {
        shape_near_cube(target_items, bs->shape);
    }

    int64_t total_items = bs->shape[0] * bs->shape[1] * bs->shape[2];
    if (total_items <= 0) {
        return -1;
    }
    bs->total_items = total_items;

    me_variable vars[] = {
        {"a", ME_FLOAT64},
        {"b", ME_FLOAT32},
        {"c", ME_INT16}
    };
    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile_nd(DSL_SOURCE, vars, 3, ME_AUTO, 3,
                           bs->shape, sc->chunkshape, sc->blockshape, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        fprintf(stderr, "Compile failed (%s): rc=%d err=%d\n", sc->name, rc, err);
        return -1;
    }
    bs->expr = expr;

    bs->nchunks_dim[0] = ceil_div64(bs->shape[0], sc->chunkshape[0]);
    bs->nchunks_dim[1] = ceil_div64(bs->shape[1], sc->chunkshape[1]);
    bs->nchunks_dim[2] = ceil_div64(bs->shape[2], sc->chunkshape[2]);
    int64_t total_chunks = bs->nchunks_dim[0] * bs->nchunks_dim[1] * bs->nchunks_dim[2];

    bs->nblocks_dim[0] = ceil_div64(sc->chunkshape[0], sc->blockshape[0]);
    bs->nblocks_dim[1] = ceil_div64(sc->chunkshape[1], sc->blockshape[1]);
    bs->nblocks_dim[2] = ceil_div64(sc->chunkshape[2], sc->blockshape[2]);
    bs->blocks_per_chunk = bs->nblocks_dim[0] * bs->nblocks_dim[1] * bs->nblocks_dim[2];
    bs->total_blocks = total_chunks * bs->blocks_per_chunk;

    bs->block_items = (int)(sc->blockshape[0] * sc->blockshape[1] * sc->blockshape[2]);

    bs->strides[2] = 1;
    bs->strides[1] = bs->shape[2];
    bs->strides[0] = bs->shape[1] * bs->shape[2];
    return 0;
}

static void cleanup_scenario(bench_scenario_t *bs) {
    if (bs->expr) {
        me_free(bs->expr);
    }
}

int main(void) {
    const int64_t target_items = (int64_t)(TOTAL_SIZE_MB * 1024ULL * 1024ULL) / (int64_t)sizeof(double);
    const size_t nscen = sizeof(SCENARIOS) / sizeof(SCENARIOS[0]);
    bench_scenario_t scenarios[4];
    int64_t max_items = target_items;

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  ND Mixed-Type DSL Padding Benchmark (Threads)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("DSL: def kernel(a,b,c): sum=0; for i in range(4): tmp=(a+b)*c+i; if any(tmp<-1e9): continue; ");
    printf("if any(tmp>1e12): continue; sum=sum+tmp; return sum\n");
    printf("Types: a=f64, b=f32, c=i16  | output=f64\n");
    printf("Target size: %d MB output (~%lld elements)\n", TOTAL_SIZE_MB, (long long)target_items);
    printf("Threads: 1..%d\n", MAX_THREADS);
    printf("Scenarios (chunkshape -> blockshape):\n");
    for (size_t i = 0; i < nscen; i++) {
        const scenario_t *sc = &SCENARIOS[i];
        printf("  %-11s: (%d,%d,%d) -> (%d,%d,%d)\n",
               sc->name,
               sc->chunkshape[0], sc->chunkshape[1], sc->chunkshape[2],
               sc->blockshape[0], sc->blockshape[1], sc->blockshape[2]);
    }
    printf("Throughput columns below are in GB/s (22 bytes per row)\n\n");
    printf("%7s  %12s  %12s  %12s  %12s  %12s  %12s\n", "Threads",
           "chunk+block", "chunk-pad", "block-pad", "none-eval_nd", "c-pack", "c-no-pad");
    printf("-------------------------------------------------------------------------------------------\n");

    for (size_t i = 0; i < nscen; i++) {
        if (setup_scenario(&scenarios[i], &SCENARIOS[i], target_items) != 0) {
            for (size_t j = 0; j < i; j++) cleanup_scenario(&scenarios[j]);
            return 1;
        }
        if (scenarios[i].total_items > max_items) {
            max_items = scenarios[i].total_items;
        }
    }

    double *a = malloc((size_t)max_items * sizeof(double));
    float *b = malloc((size_t)max_items * sizeof(float));
    int16_t *c = malloc((size_t)max_items * sizeof(int16_t));
    double *out = malloc((size_t)max_items * sizeof(double));
    if (!a || !b || !c || !out) {
        fprintf(stderr, "Failed to allocate C baseline arrays\n");
        free(a); free(b); free(c); free(out);
        for (size_t i = 0; i < nscen; i++) cleanup_scenario(&scenarios[i]);
        return 1;
    }

    for (int64_t i = 0; i < max_items; i++) {
        a[i] = (double)(i % 1000) / 100.0;
        b[i] = (float)((i + 333) % 1000) / 100.0f;
        c[i] = (int16_t)((i % 100) - 50);
    }

    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        double gbps[4] = {0.0, 0.0, 0.0, 0.0};
        for (size_t s = 0; s < nscen; s++) {
            scenarios[s].full_a = a;
            scenarios[s].full_b = b;
            scenarios[s].full_c = c;
            scenarios[s].full_out = out;
            double elapsed = run_benchmark_nd(&scenarios[s], num_threads);
            if (elapsed <= 0.0) {
                gbps[s] = 0.0;
            }
            else {
                double elems_per_sec = (double)scenarios[s].total_items / elapsed;
                gbps[s] = (elems_per_sec * BYTES_PER_ELEMENT) / 1e9;
            }
        }
        double c_elapsed = run_benchmark_c(a, b, c, out, target_items, num_threads);
        double c_gbps = (c_elapsed > 0.0)
                        ? (((double)target_items / c_elapsed) * BYTES_PER_ELEMENT) / 1e9
                        : 0.0;
        double c_pack_elapsed = run_benchmark_c_pack(&scenarios[3], num_threads);
        double c_pack_gbps = (c_pack_elapsed > 0.0)
                             ? (((double)scenarios[3].total_items / c_pack_elapsed) * BYTES_PER_ELEMENT) / 1e9
                             : 0.0;

        printf("%7d  %12.2f  %12.2f  %12.2f  %12.2f  %12.2f  %12.2f\n",
               num_threads, gbps[0], gbps[1], gbps[2], gbps[3], c_pack_gbps, c_gbps);
        fflush(stdout);
    }

    free(a);
    free(b);
    free(c);
    free(out);
    for (size_t i = 0; i < nscen; i++) cleanup_scenario(&scenarios[i]);

    printf("═══════════════════════════════════════════════════════════════════\n");
    return 0;
}
