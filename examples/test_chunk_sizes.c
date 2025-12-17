/*
 * Test different chunk sizes to find optimal for cache performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "../src/miniexpr.h"

#define TOTAL_SIZE 44739242
#define NUM_THREADS 4
#define NUM_RUNS 3  // Average over multiple runs

typedef struct {
    me_expr *expr;
    const double *a;
    const double *b;
    double *result;
    int start;
    int end;
    int chunk_size;
} thread_data_t;

void *eval_thread_chunked(void *arg) {
    thread_data_t *data = (thread_data_t *) arg;
    const void *var_ptrs[2];

    // Process in chunks
    for (int pos = data->start; pos < data->end; pos += data->chunk_size) {
        int count = (pos + data->chunk_size <= data->end) ? data->chunk_size : (data->end - pos);

        var_ptrs[0] = &data->a[pos];
        var_ptrs[1] = &data->b[pos];

        me_eval(data->expr, var_ptrs, 2,
                &data->result[pos], count);
    }

    return NULL;
}

double benchmark_chunk_size(int chunk_size, double *a, double *b,
                            double *result, me_expr *expr) {
    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        struct timeval start_tv, end_tv;
        gettimeofday(&start_tv, NULL);

        pthread_t threads[NUM_THREADS];
        thread_data_t thread_data[NUM_THREADS];

        int chunk_per_thread = TOTAL_SIZE / NUM_THREADS;

        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].expr = expr;
            thread_data[i].a = a;
            thread_data[i].b = b;
            thread_data[i].result = result;
            thread_data[i].start = i * chunk_per_thread;
            thread_data[i].end = (i == NUM_THREADS - 1) ? TOTAL_SIZE : (i + 1) * chunk_per_thread;
            thread_data[i].chunk_size = chunk_size;

            pthread_create(&threads[i], NULL, eval_thread_chunked,
                           &thread_data[i]);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }

        gettimeofday(&end_tv, NULL);
        double elapsed = (end_tv.tv_sec - start_tv.tv_sec) +
                         (end_tv.tv_usec - start_tv.tv_usec) / 1e6;
        total_time += elapsed;
    }

    return total_time / NUM_RUNS;
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║       CHUNK SIZE OPTIMIZATION BENCHMARK                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    printf("Configuration:\n");
    printf("  Total elements: %d (~44.7M)\n", TOTAL_SIZE);
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Runs per test: %d\n", NUM_RUNS);
    printf("  Expression: sqrt(a*a + b*b)\n\n");

    // Allocate arrays
    double *a = malloc(TOTAL_SIZE * sizeof(double));
    double *b = malloc(TOTAL_SIZE * sizeof(double));
    double *result = malloc(TOTAL_SIZE * sizeof(double));

    if (!a || !b || !result) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    // Initialize data
    for (int i = 0; i < TOTAL_SIZE; i++) {
        a[i] = i * 0.1;
        b[i] = i * 0.2;
    }

    // Compile expression
    me_variable vars[] = {{"a"}, {"b"}};
    int error;
    me_expr *expr = me_compile("sqrt(a*a + b*b)", vars, 2,
                               ME_FLOAT64, &error);
    if (!expr) {
        printf("ERROR: Failed to compile\n");
        return 1;
    }

    // Test different chunk sizes
    int chunk_sizes[] = {
        4096, // 4K   = 96 KB
        8192, // 8K   = 192 KB
        16384, // 16K  = 384 KB
        32768, // 32K  = 768 KB
        65536, // 64K  = 1.5 MB
        131072, // 128K = 3 MB
        262144, // 256K = 6 MB
        524288, // 512K = 12 MB
        1048576, // 1M   = 24 MB
        2097152 // 2M   = 48 MB
    };

    int num_tests = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    printf("Testing chunk sizes...\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Chunk Size    Memory      Time(s)   Throughput    Bandwidth\n");
    printf("═══════════════════════════════════════════════════════════\n");

    double best_time = 1e9;
    int best_chunk = 0;

    for (int i = 0; i < num_tests; i++) {
        int chunk = chunk_sizes[i];
        double elapsed = benchmark_chunk_size(chunk, a, b, result, expr);

        double melems_per_sec = (TOTAL_SIZE / 1e6) / elapsed;
        double bandwidth_gb = (TOTAL_SIZE * 3 * sizeof(double) / 1e9) / elapsed;

        char size_str[20];
        if (chunk >= 1048576) {
            sprintf(size_str, "%dM", chunk / 1048576);
        } else {
            sprintf(size_str, "%dK", chunk / 1024);
        }

        char mem_str[20];
        double mem_mb = (chunk * 24.0) / (1024 * 1024);
        if (mem_mb >= 1.0) {
            sprintf(mem_str, "%.1f MB", mem_mb);
        } else {
            sprintf(mem_str, "%.0f KB", mem_mb * 1024);
        }

        printf("%-13s %-11s %7.4f   %6.1f M/s   %6.2f GB/s",
               size_str, mem_str, elapsed, melems_per_sec, bandwidth_gb);

        if (elapsed < best_time) {
            best_time = elapsed;
            best_chunk = chunk;
            printf("  ⭐ BEST");
        }
        printf("\n");
    }

    printf("═══════════════════════════════════════════════════════════\n");
    printf("\n✅ OPTIMAL CHUNK SIZE: ");
    if (best_chunk >= 1048576) {
        printf("%dM elements", best_chunk / 1048576);
    } else {
        printf("%dK elements", best_chunk / 1024);
    }
    printf(" (%.1f MB per chunk)\n", (best_chunk * 24.0) / (1024 * 1024));
    printf("   Best time: %.4f seconds\n", best_time);
    printf("   Throughput: %.1f Melems/sec\n", (TOTAL_SIZE / 1e6) / best_time);
    printf("   Bandwidth: %.2f GB/s\n\n",
           (TOTAL_SIZE * 3 * sizeof(double) / 1e9) / best_time);

    // Cleanup
    me_free(expr);
    free(a);
    free(b);
    free(result);

    return 0;
}
