/*
 * Benchmark with single persistent thread pool across all tests
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
#include <stdbool.h>
#include "miniexpr.h"

#define NUM_THREADS 4
#define TOTAL_SIZE_MB 1024  // 1 GB total dataset

typedef struct {
    // Work parameters (reset for each test)
    me_expr *expr;
    const void **inputs;
    int num_inputs;
    void *output;
    size_t total_elements;
    size_t chunk_elements;
    
    // Shared work queue
    pthread_mutex_t work_mutex;
    pthread_cond_t work_available;
    pthread_cond_t all_done;
    size_t next_chunk_idx;
    size_t completed_elements;
    bool work_ready;
    bool should_exit;
} thread_pool_t;

static void *worker_thread(void *arg) {
    thread_pool_t *pool = (thread_pool_t *)arg;
    
    while (1) {
        pthread_mutex_lock(&pool->work_mutex);
        
        // Wait for work or exit signal
        while (!pool->work_ready && !pool->should_exit) {
            pthread_cond_wait(&pool->work_available, &pool->work_mutex);
        }
        
        if (pool->should_exit) {
            pthread_mutex_unlock(&pool->work_mutex);
            break;
        }
        
        // Process chunks until all work is done
        while (pool->next_chunk_idx < pool->total_elements) {
            // Get next chunk
            size_t my_chunk_idx = pool->next_chunk_idx;
            size_t chunk_size = pool->chunk_elements;
            if (my_chunk_idx + chunk_size > pool->total_elements) {
                chunk_size = pool->total_elements - my_chunk_idx;
            }
            
            pool->next_chunk_idx += chunk_size;
            pthread_mutex_unlock(&pool->work_mutex);
            
            // Do the work (outside mutex)
            const void *adjusted_inputs[10];
            for (int i = 0; i < pool->num_inputs; i++) {
                adjusted_inputs[i] = (const double *)pool->inputs[i] + my_chunk_idx;
            }
            double *output = (double *)pool->output + my_chunk_idx;
            
            me_eval_chunk_threadsafe(pool->expr, adjusted_inputs, pool->num_inputs,
                                    output, chunk_size);
            
            // Update completion status
            pthread_mutex_lock(&pool->work_mutex);
            pool->completed_elements += chunk_size;
            if (pool->completed_elements >= pool->total_elements) {
                pool->work_ready = false;
                pthread_cond_signal(&pool->all_done);
            }
        }
        
        pthread_mutex_unlock(&pool->work_mutex);
    }
    
    return NULL;
}

static thread_pool_t* create_thread_pool(int num_threads, pthread_t **threads_out) {
    thread_pool_t *pool = malloc(sizeof(thread_pool_t));
    if (!pool) return NULL;
    
    pool->expr = NULL;
    pool->inputs = NULL;
    pool->num_inputs = 0;
    pool->output = NULL;
    pool->total_elements = 0;
    pool->chunk_elements = 0;
    pool->next_chunk_idx = 0;
    pool->completed_elements = 0;
    pool->work_ready = false;
    pool->should_exit = false;
    
    pthread_mutex_init(&pool->work_mutex, NULL);
    pthread_cond_init(&pool->work_available, NULL);
    pthread_cond_init(&pool->all_done, NULL);
    
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    if (!threads) {
        free(pool);
        return NULL;
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, pool);
    }
    
    *threads_out = threads;
    return pool;
}

static void destroy_thread_pool(thread_pool_t *pool, pthread_t *threads, int num_threads) {
    pthread_mutex_lock(&pool->work_mutex);
    pool->should_exit = true;
    pthread_cond_broadcast(&pool->work_available);
    pthread_mutex_unlock(&pool->work_mutex);
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_mutex_destroy(&pool->work_mutex);
    pthread_cond_destroy(&pool->work_available);
    pthread_cond_destroy(&pool->all_done);
    
    free(threads);
    free(pool);
}

static double benchmark_chunksize(thread_pool_t *pool, size_t chunk_bytes,
                                   double *a, double *b, double *result,
                                   size_t total_elements) {
    const size_t chunk_elements = chunk_bytes / sizeof(double);
    if (chunk_elements == 0) return 0.0;

    // Compile expression
    me_variable vars[] = {{"a"}, {"b"}};
    int error;
    me_expr *expr = me_compile_chunk("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &error);
    if (!expr) return 0.0;

    const void *inputs[] = {a, b};

    // Setup work for thread pool
    pthread_mutex_lock(&pool->work_mutex);
    pool->expr = expr;
    pool->inputs = inputs;
    pool->num_inputs = 2;
    pool->output = result;
    pool->total_elements = total_elements;
    pool->chunk_elements = chunk_elements;
    pool->next_chunk_idx = 0;
    pool->completed_elements = 0;
    pool->work_ready = true;
    pthread_mutex_unlock(&pool->work_mutex);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Signal threads that work is available
    pthread_cond_broadcast(&pool->work_available);
    
    // Wait for all work to be completed
    pthread_mutex_lock(&pool->work_mutex);
    while (pool->completed_elements < pool->total_elements) {
        pthread_cond_wait(&pool->all_done, &pool->work_mutex);
    }
    pthread_mutex_unlock(&pool->work_mutex);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double throughput = (total_elements / elapsed) / 1e6;  // Melems/sec

    me_free(expr);

    return throughput;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Chunk Size Optimization Benchmark (Single Persistent Pool)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("Configuration:\n");
    printf("  - Expression: sqrt(a*a + b*b)\n");
    printf("  - Threads: %d (single pool reused for all tests)\n", NUM_THREADS);
    printf("  - Total dataset: %d MB (%.1f M elements)\n",
           TOTAL_SIZE_MB, (TOTAL_SIZE_MB * 1024.0 * 1024.0) / sizeof(double) / 1e6);
    printf("  - Data type: float64\n");
    printf("  - Testing 18 chunk sizes from 1 KB to 128 MB\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    const size_t total_elements = (TOTAL_SIZE_MB * 1024 * 1024) / sizeof(double);

    // Allocate arrays once
    double *a = malloc(total_elements * sizeof(double));
    double *b = malloc(total_elements * sizeof(double));
    double *result = malloc(total_elements * sizeof(double));

    if (!a || !b || !result) {
        fprintf(stderr, "Failed to allocate arrays\n");
        free(a); free(b); free(result);
        return 1;
    }

    // Initialize data once
    for (size_t i = 0; i < total_elements; i++) {
        a[i] = (double)(i % 1000) / 100.0;
        b[i] = (double)((i + 500) % 1000) / 100.0;
    }

    // Create thread pool once
    pthread_t *threads;
    thread_pool_t *pool = create_thread_pool(NUM_THREADS, &threads);
    if (!pool) {
        fprintf(stderr, "Failed to create thread pool\n");
        free(a); free(b); free(result);
        return 1;
    }

    printf("Chunk (KB)  Throughput (Melems/s)  Bandwidth (GB/s)  GFLOP/s\n");
    printf("---------------------------------------------------------------\n");

    double best_throughput = 0.0;
    size_t best_chunk_kb = 0;

    // Test representative chunk sizes
    size_t test_sizes_kb[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    };
    int num_sizes = sizeof(test_sizes_kb) / sizeof(test_sizes_kb[0]);

    for (int i = 0; i < num_sizes; i++) {
        size_t chunk_kb = test_sizes_kb[i];
        size_t chunk_bytes = chunk_kb * 1024;
        double throughput = benchmark_chunksize(pool, chunk_bytes, a, b, result, total_elements);

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

    // Cleanup
    destroy_thread_pool(pool, threads, NUM_THREADS);
    free(a);
    free(b);
    free(result);

    return 0;
}
