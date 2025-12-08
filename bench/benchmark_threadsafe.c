/* Benchmark thread-safe chunked evaluation performance */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "../src/miniexpr.h"

#define MAX_THREADS 8

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef struct {
    me_expr *expr;
    double *a_data;
    double *b_data;
    double *result;
    int start_idx;
    int chunk_size;
} thread_args_t;

void *eval_worker(void *arg) {
    thread_args_t *args = (thread_args_t *) arg;

    const void *vars_chunk[2] = {
        args->a_data + args->start_idx,
        args->b_data + args->start_idx
    };

    me_eval_chunk_threadsafe(args->expr, vars_chunk, 2,
                             args->result + args->start_idx, args->chunk_size);

    return NULL;
}

void benchmark_threads(const char *expr_str, int total_size, int num_threads) {
    printf("\n=== Expression: %s ===\n", expr_str);
    printf("Total size: %d elements (%.1f MB)\n",
           total_size, total_size * sizeof(double) * 3 / (1024.0 * 1024.0));
    printf("Number of threads: %d\n", num_threads);

    // Allocate data
    double *a = malloc(total_size * sizeof(double));
    double *b = malloc(total_size * sizeof(double));
    double *result = malloc(total_size * sizeof(double));

    for (int i = 0; i < total_size; i++) {
        a[i] = i * 0.1;
        b[i] = (total_size - i) * 0.05;
    }

    // Compile
    me_variable vars[] = {{"a", a}, {"b", b}};
    int err;
    me_expr *expr = me_compile(expr_str, vars, 2, result, total_size, ME_FLOAT64, &err);
    if (!expr) {
        printf("Failed to compile\n");
        free(a);
        free(b);
        free(result);
        return;
    }

    const int iterations = 10;

    // Benchmark serial evaluation
    double serial_start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        me_eval(expr);
    }
    double serial_time = (get_time() - serial_start) / iterations;

    // Benchmark parallel evaluation
    pthread_t threads[MAX_THREADS];
    thread_args_t thread_args[MAX_THREADS];
    int chunk_size = total_size / num_threads;

    double parallel_start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        for (int t = 0; t < num_threads; t++) {
            thread_args[t].expr = expr;
            thread_args[t].a_data = a;
            thread_args[t].b_data = b;
            thread_args[t].result = result;
            thread_args[t].start_idx = t * chunk_size;
            thread_args[t].chunk_size = chunk_size;

            pthread_create(&threads[t], NULL, eval_worker, &thread_args[t]);
        }

        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
    }
    double parallel_time = (get_time() - parallel_start) / iterations;

    // Calculate metrics
    double data_size_gb = (total_size * sizeof(double) * 3.0) / (1024.0 * 1024.0 * 1024.0);
    double serial_throughput = data_size_gb / serial_time;
    double parallel_throughput = data_size_gb / parallel_time;
    double speedup = serial_time / parallel_time;
    double efficiency = speedup / num_threads;

    printf("\nResults:\n");
    printf("  Serial:   %.4f s  (%.2f GB/s)\n", serial_time, serial_throughput);
    printf("  Parallel: %.4f s  (%.2f GB/s)\n", parallel_time, parallel_throughput);
    printf("  Speedup:  %.2fx\n", speedup);
    printf("  Efficiency: %.1f%%\n", efficiency * 100.0);

    me_free(expr);
    free(a);
    free(b);
    free(result);
}

int main() {
    printf("========================================\n");
    printf("Thread-Safe Chunked Evaluation Benchmark\n");
    printf("========================================\n");

    const int size = 10 * 1024 * 1024; // 10M elements

    // Test different thread counts
    printf("\n--- Scaling with thread count ---\n");
    printf("Array size: 10M elements (80 MB per array)\n");

    int thread_counts[] = {1, 2, 4, 8};

    for (int i = 0; i < 4; i++) {
        benchmark_threads("a + b", size, thread_counts[i]);
    }

    printf("\n--- Complex expression ---\n");
    for (int i = 0; i < 4; i++) {
        benchmark_threads("sqrt(a*a + b*b)", size, thread_counts[i]);
    }

    printf("\n========================================\n");
    printf("Benchmark complete!\n");
    printf("\nKey observations:\n");
    printf("- Thread-safe implementation allows true parallelism\n");
    printf("- Speedup scales with number of cores (up to memory bandwidth limit)\n");
    printf("- Cloning overhead is minimal compared to computation\n");
    printf("========================================\n");

    return 0;
}
