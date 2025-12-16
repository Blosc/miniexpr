/*
 * Benchmark to find optimal chunk size for thread-safe evaluation
 * Tests various chunk sizes from 1 KB to 130 MB with 4 threads
 *
 * Usage: ./benchmark_chunksize
 *
 * Output: CSV-style results showing performance for each chunk size
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include "miniexpr.h"

#define NUM_THREADS 4
#define TOTAL_SIZE_MB 1024  // 1 GB total dataset

typedef struct {
    me_expr *expr;
    const void **inputs;
    int num_inputs;
    void *output;
    size_t start_idx;
    size_t total_elements;
    size_t chunk_elements;
    pthread_mutex_t *work_mutex;
    size_t *next_chunk_idx;
} thread_data_t;

static void *eval_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;

    while (1) {
        // Get next chunk to process
        size_t my_chunk_idx;
        pthread_mutex_lock(data->work_mutex);
        my_chunk_idx = *data->next_chunk_idx;
        if (my_chunk_idx >= data->total_elements) {
            pthread_mutex_unlock(data->work_mutex);
            break;
        }
        *data->next_chunk_idx += data->chunk_elements;
        pthread_mutex_unlock(data->work_mutex);

        // Calculate actual chunk size (last chunk may be smaller)
        size_t chunk_size = data->chunk_elements;
        if (my_chunk_idx + chunk_size > data->total_elements) {
            chunk_size = data->total_elements - my_chunk_idx;
        }

        // Adjust input pointers to chunk position
        const void *adjusted_inputs[10];
        for (int i = 0; i < data->num_inputs; i++) {
            adjusted_inputs[i] = (const double *)data->inputs[i] + my_chunk_idx;
        }

        double *output = (double *)data->output + my_chunk_idx;

        me_eval_chunk_threadsafe(data->expr, adjusted_inputs, data->num_inputs,
                                output, chunk_size);
    }

    return NULL;
}

static double benchmark_chunksize(size_t chunk_bytes) {
    const size_t total_elements = (TOTAL_SIZE_MB * 1024 * 1024) / sizeof(double);
    const size_t chunk_elements = chunk_bytes / sizeof(double);

    if (chunk_elements == 0) return 0.0;

    // Allocate arrays
    double *a = malloc(total_elements * sizeof(double));
    double *b = malloc(total_elements * sizeof(double));
    double *result = malloc(total_elements * sizeof(double));

    if (!a || !b || !result) {
        free(a); free(b); free(result);
        return 0.0;
    }

    // Initialize data
    for (size_t i = 0; i < total_elements; i++) {
        a[i] = (double)(i % 1000) / 100.0;
        b[i] = (double)((i + 500) % 1000) / 100.0;
    }

    // Compile expression
    me_variable vars[] = {{"a"}, {"b"}};
    int error;
    me_expr *expr = me_compile_chunk("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &error);

    if (!expr) {
        free(a); free(b); free(result);
        return 0.0;
    }

    const void *inputs[] = {a, b};

    // Benchmark - use work queue pattern to avoid thread creation overhead
    pthread_mutex_t work_mutex = PTHREAD_MUTEX_INITIALIZER;
    size_t next_chunk_idx = 0;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create threads once - they'll pull chunks from work queue
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].expr = expr;
        thread_data[i].inputs = inputs;
        thread_data[i].num_inputs = 2;
        thread_data[i].output = result;
        thread_data[i].total_elements = total_elements;
        thread_data[i].chunk_elements = chunk_elements;
        thread_data[i].work_mutex = &work_mutex;
        thread_data[i].next_chunk_idx = &next_chunk_idx;
        
        pthread_create(&threads[i], NULL, eval_thread, &thread_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_mutex_destroy(&work_mutex);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double throughput = (total_elements / elapsed) / 1e6;  // Melems/sec

    me_free(expr);
    free(a);
    free(b);
    free(result);

    return throughput;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Chunk Size Optimization Benchmark\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Configuration:\n");
    printf("  - Expression: sqrt(a*a + b*b)\n");
    printf("  - Threads: %d\n", NUM_THREADS);
    printf("  - Total dataset: %d MB (%.1f M elements)\n",
           TOTAL_SIZE_MB, (TOTAL_SIZE_MB * 1024.0 * 1024.0) / sizeof(double) / 1e6);
    printf("  - Data type: float64\n");
    printf("  - Testing 15 chunk sizes from 1 KB to 128 MB\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("Chunk (KB)  Throughput (Melems/s)  Bandwidth (GB/s)  GFLOP/s\n");
    printf("---------------------------------------------------------------\n");

    double best_throughput = 0.0;
    size_t best_chunk_kb = 0;

    // Test 15 representative chunk sizes with exponential-like distribution
    size_t test_sizes_kb[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    };
    int num_sizes = sizeof(test_sizes_kb) / sizeof(test_sizes_kb[0]);

    for (int i = 0; i < num_sizes; i++) {
        size_t chunk_kb = test_sizes_kb[i];
        size_t chunk_bytes = chunk_kb * 1024;
        double throughput = benchmark_chunksize(chunk_bytes);

        if (throughput == 0.0) {
            fprintf(stderr, "Benchmark failed for chunk size %zu KB\n", chunk_kb);
            continue;
        }

        double bandwidth = (throughput * 3 * sizeof(double)) / 1000.0;  // MB/s to GB/s, 2 inputs + 1 output
        double gflops = throughput * 4.0 / 1000.0;  // 4 FLOP per element (2 mul, 1 add, 1 sqrt≈1)

        printf("%10zu  %21.2f  %16.2f  %8.2f\n",
               chunk_kb, throughput, bandwidth, gflops);
        fflush(stdout);

        if (throughput > best_throughput) {
            best_throughput = throughput;
            best_chunk_kb = chunk_kb;
        }
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Best Performance:\n");
    printf("  Chunk Size: %zu KB (%.2f MB)\n", best_chunk_kb, best_chunk_kb / 1024.0);
    printf("  Throughput: %.2f Melems/sec\n", best_throughput);
    printf("  Bandwidth:  %.2f GB/s\n", (best_throughput * 3 * sizeof(double)) / 1000.0);
    printf("  GFLOP/s:    %.2f\n", best_throughput * 4.0 / 1000.0);
    printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}
