/* Benchmark: Multi-threaded sum reductions */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>
#include "miniexpr.h"
#include "minctest.h"



#define MAX_THREADS 8

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef struct {
    const me_expr *expr;
    const void *data;
    size_t elem_size;
    size_t output_stride;
    int start_idx;
    int count;
    void *output;
} thread_args_t;

typedef struct {
    const void *data;
    int start_idx;
    int count;
    void *output;
} thread_args_c_t;

static size_t dtype_size_local(me_dtype dtype) {
    switch (dtype) {
        case ME_BOOL: return sizeof(bool);
        case ME_INT8: return sizeof(int8_t);
        case ME_INT16: return sizeof(int16_t);
        case ME_INT32: return sizeof(int32_t);
        case ME_INT64: return sizeof(int64_t);
        case ME_UINT8: return sizeof(uint8_t);
        case ME_UINT16: return sizeof(uint16_t);
        case ME_UINT32: return sizeof(uint32_t);
        case ME_UINT64: return sizeof(uint64_t);
        case ME_FLOAT32: return sizeof(float);
        case ME_FLOAT64: return sizeof(double);
        case ME_COMPLEX64: return sizeof(float _Complex);
        case ME_COMPLEX128: return sizeof(double _Complex);
        default: return 0;
    }
}

static void *sum_worker(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    const void *vars_chunk[1] = {
        (const unsigned char *)args->data + (size_t)args->start_idx * args->elem_size
    };

    ME_EVAL_CHECK(args->expr, vars_chunk, 1, args->output, args->count);
    return NULL;
}

static void *sum_worker_c_int32(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    const int32_t *data = (const int32_t *)args->data + args->start_idx;
    int64_t acc = 0;
    for (int i = 0; i < args->count; i++) {
        acc += data[i];
    }
    ((int64_t *)args->output)[0] = acc;
    return NULL;
}

static void *sum_worker_c_float32(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    const float *data = (const float *)args->data + args->start_idx;
    float acc = 0.0f;
    for (int i = 0; i < args->count; i++) {
        acc += data[i];
    }
    ((float *)args->output)[0] = acc;
    return NULL;
}

static void *prod_worker_c_int32(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    const int32_t *data = (const int32_t *)args->data + args->start_idx;
    int64_t acc = 1;
    for (int i = 0; i < args->count; i++) {
        acc *= data[i];
    }
    ((int64_t *)args->output)[0] = acc;
    return NULL;
}

static void *prod_worker_c_float32(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    const float *data = (const float *)args->data + args->start_idx;
    float acc = 1.0f;
    for (int i = 0; i < args->count; i++) {
        acc *= data[i];
    }
    ((float *)args->output)[0] = acc;
    return NULL;
}

static int run_threads(const me_expr *expr, const void *data, size_t elem_size,
                       size_t output_stride, int total_elems, int num_threads,
                       void *partials) {
    pthread_t threads[MAX_THREADS];
    thread_args_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].expr = expr;
        thread_args[t].data = data;
        thread_args[t].elem_size = elem_size;
        thread_args[t].output_stride = output_stride;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        thread_args[t].output = (unsigned char *)partials + (size_t)t * output_stride;
        offset += count;

        pthread_create(&threads[t], NULL, sum_worker, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    return 0;
}

static int run_threads_c_int32(const int32_t *data, int total_elems, int num_threads,
                               int64_t *partials, void *(*worker)(void *)) {
    pthread_t threads[MAX_THREADS];
    thread_args_c_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].data = data;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        thread_args[t].output = &partials[t];
        offset += count;

        pthread_create(&threads[t], NULL, worker, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    return 0;
}

static int run_threads_c_float32(const float *data, int total_elems, int num_threads,
                                 float *partials, void *(*worker)(void *)) {
    pthread_t threads[MAX_THREADS];
    thread_args_c_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].data = data;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        thread_args[t].output = &partials[t];
        offset += count;

        pthread_create(&threads[t], NULL, worker, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    return 0;
}

static double run_benchmark(const me_expr *expr, const void *data, size_t elem_size,
                            size_t output_stride, int total_elems, int num_threads,
                            int iterations, void *partials) {
    run_threads(expr, data, elem_size, output_stride, total_elems, num_threads, partials);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads(expr, data, elem_size, output_stride, total_elems, num_threads, partials);
    }
    return (get_time() - start) / iterations;
}

static double run_benchmark_c_int32(const int32_t *data, int total_elems, int num_threads,
                                    int iterations, int64_t *partials, bool is_prod) {
    void *(*worker)(void *) = is_prod ? prod_worker_c_int32 : sum_worker_c_int32;
    run_threads_c_int32(data, total_elems, num_threads, partials, worker);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads_c_int32(data, total_elems, num_threads, partials, worker);
    }
    return (get_time() - start) / iterations;
}

static double run_benchmark_c_float32(const float *data, int total_elems, int num_threads,
                                      int iterations, float *partials, bool is_prod) {
    void *(*worker)(void *) = is_prod ? prod_worker_c_float32 : sum_worker_c_float32;
    run_threads_c_float32(data, total_elems, num_threads, partials, worker);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads_c_float32(data, total_elems, num_threads, partials, worker);
    }
    return (get_time() - start) / iterations;
}

int main(int argc, char **argv) {
    printf("===================================================\n");
    printf("MiniExpr Reduction Benchmark (Multi-threaded)\n");
    printf("===================================================\n");

    const char *op = "sum";
    if (argc > 1) {
        op = argv[1];
    }
    if (strcmp(op, "sum") != 0 && strcmp(op, "prod") != 0) {
        printf("Usage: %s [sum|prod]\n", argv[0]);
        return 1;
    }
    bool is_prod = strcmp(op, "prod") == 0;

    const size_t total_elems = 16ULL * 1024ULL * 1024ULL;
    const int iterations = 4;

    if (total_elems > (size_t)INT_MAX) {
        printf("ERROR: Dataset too large for int-sized nitems\n");
        return 1;
    }

    printf("Total elements per run: %zu\n", total_elems);
    printf("Iterations: %d\n", iterations);

    int32_t *data_int32 = malloc(total_elems * sizeof(int32_t));
    float *data_float32 = malloc(total_elems * sizeof(float));
    if (!data_int32 || !data_float32) {
        printf("Allocation failed for data arrays\n");
        free(data_int32);
        free(data_float32);
        return 1;
    }

    for (size_t i = 0; i < total_elems; i++) {
        data_int32[i] = (int32_t)(i % 97);
        data_float32[i] = (float)(i % 97) * 0.25f;
    }

    me_variable vars_int32[] = {{"x", ME_INT32, data_int32}};
    me_variable vars_float32[] = {{"x", ME_FLOAT32, data_float32}};

    int err = 0;
    me_expr *expr_int32 = NULL;
    me_expr *expr_float32 = NULL;
    char expr_buf[16];
    snprintf(expr_buf, sizeof(expr_buf), "%s(x)", op);
    int rc_expr = me_compile(expr_buf, vars_int32, 1, ME_AUTO, &err, &expr_int32);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s(x) for int32 (err=%d)\n", op, err);
        free(data_int32);
        free(data_float32);
        return 1;
    }
    rc_expr = me_compile(expr_buf, vars_float32, 1, ME_AUTO, &err, &expr_float32);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s(x) for float32 (err=%d)\n", op, err);
        me_free(expr_int32);
        free(data_int32);
        free(data_float32);
        return 1;
    }

    me_dtype out_int32 = me_get_dtype(expr_int32);
    me_dtype out_float32 = me_get_dtype(expr_float32);
    size_t stride_int32 = dtype_size_local(out_int32);
    size_t stride_float32 = dtype_size_local(out_float32);
    if (stride_int32 != sizeof(int64_t) || stride_float32 != sizeof(float)) {
        printf("Unexpected output dtype for sums: int32=%d float32=%d\n",
               out_int32, out_float32);
        me_free(expr_int32);
        me_free(expr_float32);
        free(data_int32);
        free(data_float32);
        return 1;
    }

    int64_t *partials_int_me = malloc(MAX_THREADS * sizeof(int64_t));
    float *partials_float_me = malloc(MAX_THREADS * sizeof(float));
    int64_t *partials_int_c = malloc(MAX_THREADS * sizeof(int64_t));
    float *partials_float_c = malloc(MAX_THREADS * sizeof(float));
    if (!partials_int_me || !partials_float_me || !partials_int_c || !partials_float_c) {
        printf("Allocation failed for partials\n");
        free(partials_int_me);
        free(partials_float_me);
        free(partials_int_c);
        free(partials_float_c);
        me_free(expr_int32);
        me_free(expr_float32);
        free(data_int32);
        free(data_float32);
        return 1;
    }

    double data_gb = (double)(total_elems * sizeof(int32_t)) / 1e9;

    printf("\n========================================\n");
    printf("Summary (%s, GB/s)\n", op);
    printf("========================================\n");
    printf("Threads Int32 ME Int32 C  F32 ME   F32 C\n");

    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        double me_int = run_benchmark(expr_int32, data_int32, sizeof(int32_t),
                                      stride_int32, (int)total_elems,
                                      num_threads, iterations, partials_int_me);
        double c_int = run_benchmark_c_int32(data_int32, (int)total_elems,
                                             num_threads, iterations, partials_int_c,
                                             is_prod);
        double me_f32 = run_benchmark(expr_float32, data_float32, sizeof(float),
                                      stride_float32, (int)total_elems,
                                      num_threads, iterations, partials_float_me);
        double c_f32 = run_benchmark_c_float32(data_float32, (int)total_elems,
                                               num_threads, iterations, partials_float_c,
                                               is_prod);

        printf("%6d  %8.2f %7.2f %7.2f %7.2f\n",
               num_threads,
               data_gb / me_int,
               data_gb / c_int,
               data_gb / me_f32,
               data_gb / c_f32);
    }

    printf("========================================\n");
    printf("Benchmark complete!\n");
    printf("========================================\n");

    free(partials_int_me);
    free(partials_float_me);
    free(partials_int_c);
    free(partials_float_c);
    me_free(expr_int32);
    me_free(expr_float32);
    free(data_int32);
    free(data_float32);

    return 0;
}
