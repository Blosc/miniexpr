/* Benchmark: Eval block size tuning with multi-threaded evaluation */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>
#include <sys/time.h>
#include "../src/miniexpr.h"

#define MAX_THREADS 8
#define GIB_BYTES (1024ULL * 1024ULL * 1024ULL)

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef struct {
    const me_expr *expr;
    const double *a_data;
    const double *b_data;
    const double *c_data;
    double *output;
    int start_idx;
    int count;
} thread_args_t;

static void *eval_worker(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    const void *vars_chunk[3] = {
        args->a_data + args->start_idx,
        args->b_data + args->start_idx,
        args->c_data + args->start_idx
    };

    me_eval(args->expr, vars_chunk, 3,
            args->output + args->start_idx, args->count);
    return NULL;
}

static void run_threads(const me_expr *expr, const double *a, const double *b, const double *c,
                        double *out, int total_elems, int num_threads) {
    pthread_t threads[MAX_THREADS];
    thread_args_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].expr = expr;
        thread_args[t].a_data = a;
        thread_args[t].b_data = b;
        thread_args[t].c_data = c;
        thread_args[t].output = out;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        offset += count;

        pthread_create(&threads[t], NULL, eval_worker, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

static double run_benchmark(const me_expr *expr, const double *a, const double *b, const double *c,
                            double *out, int total_elems, int num_threads, int iterations) {
    // Warm-up
    run_threads(expr, a, b, c, out, total_elems, num_threads);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads(expr, a, b, c, out, total_elems, num_threads);
    }
    return (get_time() - start) / iterations;
}

static void benchmark_block_sizes(size_t total_elems) {
    printf("\nExpression: (a + b) * c\n");
    printf("Total elements: %zu\n", total_elems);
    printf("Arrays: 3 inputs + 1 output (double)\n");

    double *a = malloc(total_elems * sizeof(double));
    double *b = malloc(total_elems * sizeof(double));
    double *c = malloc(total_elems * sizeof(double));
    double *out = malloc(total_elems * sizeof(double));

    if (!a || !b || !c || !out) {
        printf("ERROR: Memory allocation failed\n");
        free(a);
        free(b);
        free(c);
        free(out);
        return;
    }

    int total_elems_i = (int)total_elems;
    for (size_t i = 0; i < total_elems; i++) {
        a[i] = (double)i * 0.1;
        b[i] = (double)(total_elems - i) * 0.05;
        c[i] = (double)(i % 1024) * 0.001;
    }

    me_variable vars[] = {{"a"}, {"b"}, {"c"}};
    int err = 0;
    me_expr *expr = me_compile("(a + b) * c", vars, 3, ME_FLOAT64, &err);
    if (!expr) {
        printf("ERROR: Failed to compile expression (err=%d)\n", err);
        free(a);
        free(b);
        free(c);
        free(out);
        return;
    }

    const double data_gb = (double)(total_elems * sizeof(double) * 4ULL) / 1e9;

    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        printf("\nThreads: %d\n", num_threads);
        printf("  Block size (fixed): %d elements\n", ME_EVAL_BLOCK_NITEMS);
        double elapsed = run_benchmark(expr, a, b, c, out, total_elems_i, num_threads, 5);
        double throughput = data_gb / elapsed;
        printf("  Avg time (s): %.4f\n", elapsed);
        printf("  Throughput (GB/s): %.2f\n", throughput);
    }

    me_free(expr);
    free(a);
    free(b);
    free(c);
    free(out);
}

int main(void) {
    printf("===================================================\n");
    printf("MiniExpr Block Size Benchmark (Multi-threaded)\n");
    printf("===================================================\n");

    const size_t total_var_bytes = GIB_BYTES;
    const size_t total_elems = total_var_bytes / (3ULL * sizeof(double));

    if (total_elems > (size_t)INT_MAX) {
        printf("ERROR: Dataset too large for int-sized nitems\n");
        return 1;
    }

    printf("Total variable working set: %.2f GB\n",
           (double)total_var_bytes / 1e9);

    benchmark_block_sizes(total_elems);

    printf("\n===================================================\n");
    printf("Benchmark complete\n");
    printf("===================================================\n");

    return 0;
}
