/* Benchmark: Multi-threaded sum reductions */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/time.h>
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

static void benchmark_sum_threads_int32(size_t total_elems, int iterations) {
    printf("\n=== sum(int32) multi-threaded ===\n");

    int32_t *data = malloc(total_elems * sizeof(int32_t));
    if (!data) {
        printf("Allocation failed for int32 data\n");
        return;
    }

    for (size_t i = 0; i < total_elems; i++) {
        data[i] = (int32_t)(i % 97);
    }

    me_variable vars[] = {{"x", ME_INT32, data}};
    int err = 0;
    me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
    if (!expr) {
        printf("Failed to compile sum(x) for int32 (err=%d)\n", err);
        free(data);
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    size_t output_stride = dtype_size_local(output_dtype);
    if (output_stride != sizeof(int64_t)) {
        printf("Unexpected output dtype for int32 sum: %d\n", output_dtype);
        me_free(expr);
        free(data);
        return;
    }

    int64_t *partials = malloc(MAX_THREADS * sizeof(int64_t));
    if (!partials) {
        printf("Allocation failed for int32 partials\n");
        me_free(expr);
        free(data);
        return;
    }

    double data_gb = (double)(total_elems * sizeof(int32_t)) / 1e9;

    printf("\nSummary (Int32)\n");
    printf("Threads  Avg time (s)  Throughput (GB/s)\n");
    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        double elapsed = run_benchmark(expr, data, sizeof(int32_t),
                                       output_stride, (int)total_elems,
                                       num_threads, iterations, partials);
        double throughput = data_gb / elapsed;

        int64_t total = 0;
        for (int t = 0; t < num_threads; t++) {
            total += partials[t];
        }

        printf("%7d  %12.4f  %17.2f\n", num_threads, elapsed, throughput);
        printf("  Sum: %lld\n", (long long)total);
    }

    free(partials);
    me_free(expr);
    free(data);
}

static void benchmark_sum_threads_float32(size_t total_elems, int iterations) {
    printf("\n=== sum(float32) multi-threaded ===\n");

    float *data = malloc(total_elems * sizeof(float));
    if (!data) {
        printf("Allocation failed for float32 data\n");
        return;
    }

    for (size_t i = 0; i < total_elems; i++) {
        data[i] = (float)(i % 97) * 0.25f;
    }

    me_variable vars[] = {{"x", ME_FLOAT32, data}};
    int err = 0;
    me_expr *expr = me_compile("sum(x)", vars, 1, ME_AUTO, &err);
    if (!expr) {
        printf("Failed to compile sum(x) for float32 (err=%d)\n", err);
        free(data);
        return;
    }

    me_dtype output_dtype = me_get_dtype(expr);
    size_t output_stride = dtype_size_local(output_dtype);
    if (output_stride != sizeof(float)) {
        printf("Unexpected output dtype for float32 sum: %d\n", output_dtype);
        me_free(expr);
        free(data);
        return;
    }

    float *partials = malloc(MAX_THREADS * sizeof(float));
    if (!partials) {
        printf("Allocation failed for float32 partials\n");
        me_free(expr);
        free(data);
        return;
    }

    double data_gb = (double)(total_elems * sizeof(float)) / 1e9;

    printf("\nSummary (Float32)\n");
    printf("Threads  Avg time (s)  Throughput (GB/s)\n");
    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        double elapsed = run_benchmark(expr, data, sizeof(float),
                                       output_stride, (int)total_elems,
                                       num_threads, iterations, partials);
        double throughput = data_gb / elapsed;

        float total = 0.0f;
        for (int t = 0; t < num_threads; t++) {
            total += partials[t];
        }

        printf("%7d  %12.4f  %17.2f\n", num_threads, elapsed, throughput);
        printf("  Sum: %.6f\n", total);
    }

    free(partials);
    me_free(expr);
    free(data);
}

int main(void) {
    printf("===================================================\n");
    printf("MiniExpr Reduction Benchmark (Multi-threaded)\n");
    printf("===================================================\n");

    const size_t total_bytes = 1024ULL * 1024ULL * 1024ULL;
    const int iterations = 4;

    size_t elems_int32 = total_bytes / sizeof(int32_t);
    size_t elems_float32 = total_bytes / sizeof(float);

    if (elems_int32 > (size_t)INT_MAX || elems_float32 > (size_t)INT_MAX) {
        printf("ERROR: Dataset too large for int-sized nitems\n");
        return 1;
    }

    printf("Total working set: %.2f GB\n", (double)total_bytes / 1e9);
    printf("Iterations: %d\n", iterations);

    benchmark_sum_threads_int32(elems_int32, iterations);
    benchmark_sum_threads_float32(elems_float32, iterations);

    printf("\n===================================================\n");
    printf("Benchmark complete\n");
    printf("===================================================\n");

    return 0;
}
