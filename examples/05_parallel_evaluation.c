/*
 * Example 5: Parallel Evaluation with Multiple Threads
 *
 * Demonstrates thread-safe parallel evaluation using ME_EVAL_CHECK().
 * Multiple threads can safely evaluate the same compiled expression on
 * different data chunks simultaneously.
 *
 * NOTE: We use gettimeofday() instead of clock() for timing because clock()
 * measures CPU time (sum of all thread times), not wall-clock time. Using
 * clock() would inflate the measured time by the number of threads!
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include "../src/miniexpr.h"
#include "minctest.h"


#define TOTAL_SIZE 44739242  // ~44M elements = ~1GB working set
#define NUM_THREADS 4
#define CHUNK_SIZE 32768     // 32K elements = 768 KB (optimal for cache)
#define FLOPS_PER_ELEM 4     // sqrt(a*a + b*b): 2 muls + 1 add + 1 sqrt (convention)
// Note: Actual hardware cost ~23 FLOPs (sqrt ≈ 20 FLOPs in reality)

typedef struct {
    me_expr* expr;
    const double* a;
    const double* b;
    double* result;
    int start;
    int end;
    int thread_id;
} thread_data_t;

void* worker_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    printf("  Thread %d: Processing elements %d to %d\n",
           data->thread_id, data->start, data->end - 1);

    // Process assigned range in cache-friendly chunks
    for (int pos = data->start; pos < data->end; pos += CHUNK_SIZE) {
        int count = (pos + CHUNK_SIZE <= data->end) ? CHUNK_SIZE : (data->end - pos);

        const void* var_ptrs[] = {
            &data->a[pos],
            &data->b[pos]
        };

        // Evaluate this chunk (thread-safe!)
        ME_EVAL_CHECK(data->expr, var_ptrs, 2,
                      &data->result[pos], count);
    }

    return NULL;
}

int main() {
    printf("=== Parallel Evaluation Example ===\n");
    printf("Expression: sqrt(a*a + b*b)\n");
    printf("Total elements: %d (~%.1f M)\n", TOTAL_SIZE, TOTAL_SIZE / 1e6);
    printf("Working set: %.2f GB (3 arrays × 8 bytes)\n",
           TOTAL_SIZE * 3 * sizeof(double) / 1e9);
    printf("Threads: %d\n", NUM_THREADS);
    printf("Chunk size: %d elements (%.0f KB, cache-optimized)\n",
           CHUNK_SIZE, (CHUNK_SIZE * 24.0) / 1024);
    printf("FLOPs per element: %d (convention) / ~23 (actual hardware cost)\n\n", FLOPS_PER_ELEM);

    // Allocate arrays
    double* a = malloc(TOTAL_SIZE * sizeof(double));
    double* b = malloc(TOTAL_SIZE * sizeof(double));
    double* result = malloc(TOTAL_SIZE * sizeof(double));

    if (!a || !b || !result) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    // Initialize data
    for (int i = 0; i < TOTAL_SIZE; i++) {
        a[i] = i * 0.001;
        b[i] = i * 0.002;
    }

    // Compile expression once
    me_variable vars[] = {{"a"}, {"b"}};
    int error;
    me_expr* expr = NULL;
    int rc_expr = me_compile("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &error, &expr);

    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("ERROR: Failed to compile at position %d\n", error);
        free(a);
        free(b);
        free(result);
        return 1;
    }

    // Prepare thread data
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    int elements_per_thread = TOTAL_SIZE / NUM_THREADS;

    printf("Starting parallel evaluation...\n");

    // Use gettimeofday for wall-clock time (not clock() which measures CPU time!)
    struct timeval start_tv, end_tv;
    gettimeofday(&start_tv, NULL);

    // Launch threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].expr = expr;
        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].result = result;
        thread_data[i].start = i * elements_per_thread;
        thread_data[i].end = (i == NUM_THREADS - 1) ? TOTAL_SIZE : (i + 1) * elements_per_thread;
        thread_data[i].thread_id = i + 1;

        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end_tv, NULL);
    double elapsed = (end_tv.tv_sec - start_tv.tv_sec) +
        (end_tv.tv_usec - start_tv.tv_usec) / 1e6;

    // Calculate throughput metrics
    double melems_per_sec = (TOTAL_SIZE / 1e6) / elapsed;
    double gflops = (TOTAL_SIZE * (double)FLOPS_PER_ELEM / 1e9) / elapsed;
    double bandwidth_gb = (TOTAL_SIZE * 3 * sizeof(double) / 1e9) / elapsed;

    // Verify results
    printf("\nVerifying results (first 5 elements):\n");
    printf("       a        b     sqrt(a²+b²)\n");
    printf("  ------   ------   ------------\n");
    for (int i = 0; i < 5; i++) {
        printf("  %6.3f   %6.3f   %12.3f\n", a[i], b[i], result[i]);
    }

    printf("\n✅ Parallel evaluation complete!\n");
    printf("   Processed %d elements in %.3f seconds\n", TOTAL_SIZE, elapsed);
    printf("   Throughput: %.2f Melems/sec\n", melems_per_sec);
    printf("   Performance: %.2f GFLOP/s\n", gflops);
    printf("   Memory bandwidth: %.2f GB/s\n", bandwidth_gb);

    // Cleanup
    me_free(expr);
    free(a);
    free(b);
    free(result);

    return 0;
}

/*
 * PERFORMANCE NOTES:
 *
 * Q: Why doesn't parallel give 4× speedup?
 * A: This expression sqrt(a*a + b*b) is still relatively MEMORY-BOUND.
 *
 * Memory bandwidth is shared across all cores:
 *   - Single thread: ~14-20 GB/s (limited by L3 cache → RAM)
 *   - 4 threads:     ~20-30 GB/s (memory controller bottleneck)
 *   - Speedup:       ~1.5-2× (not 4×)
 *
 * Arithmetic Intensity:
 *   - Bytes: 24 bytes/element (2 reads + 1 write)
 *   - FLOPs: 4 conventional (but sqrt is really ~20 FLOPs in hardware)
 *   - Ratio: 4 FLOPs / 24 bytes = 0.17 FLOP/byte (low)
 *
 * FLOP Counting Convention vs Reality:
 *   Convention: sqrt counts as 1 FLOP (for benchmark comparison)
 *   Reality:    sqrt takes ~15-20 cycles vs ~3-5 for mul/add
 *               → sqrt ≈ 20 FLOPs worth of computation
 *
 *   Using realistic count: 23 FLOPs / 24 bytes = 0.96 FLOP/byte
 *   This is better but still memory-bound on modern CPUs.
 *
 * For highly compute-intensive expressions (e.g., sin, cos, exp),
 * you would see speedup closer to 3-4× because the bottleneck
 * shifts from memory to CPU computation.
 *
 * Try: sqrt(a*a + b*b) + sin(a) + cos(b) + exp(a/10)
 * This would give ~100+ actual FLOPs per element → near-linear scaling!
 */
