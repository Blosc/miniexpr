/*
 * Example 4: Processing Large Datasets in Chunks
 *
 * Demonstrates efficient processing of large arrays by breaking them
 * into smaller chunks. This reduces memory usage and improves cache
 * efficiency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/miniexpr.h"

#define TOTAL_SIZE 44739242   // ~44M elements = ~1GB working set
#define CHUNK_SIZE 32768     // 32K elements = 768 KB (optimal for cache)
#define FLOPS_PER_ELEM 4     // sqrt(a*a + b*b): 2 muls + 1 add + 1 sqrt (convention)
// Note: Actual hardware cost ~23 FLOPs (sqrt ≈ 20 FLOPs in reality)

int main() {
    printf("=== Large Dataset Processing Example ===\n");
    printf("Expression: sqrt(a*a + b*b)\n");
    printf("Total elements: %d (~%.1f M)\n", TOTAL_SIZE, TOTAL_SIZE / 1e6);
    printf("Working set: %.2f GB (3 arrays × 8 bytes)\n",
           TOTAL_SIZE * 3 * sizeof(double) / 1e9);
    printf("Chunk size: %d elements (%.0f KB, cache-optimized)\n",
           CHUNK_SIZE, (CHUNK_SIZE * 24.0) / 1024);
    printf("FLOPs per element: %d (convention) / ~23 (actual hardware cost)\n\n", FLOPS_PER_ELEM);

    // Allocate large arrays
    double *a = malloc(TOTAL_SIZE * sizeof(double));
    double *b = malloc(TOTAL_SIZE * sizeof(double));
    double *result = malloc(TOTAL_SIZE * sizeof(double));

    if (!a || !b || !result) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    // Initialize data
    printf("Initializing data...\n");
    for (int i = 0; i < TOTAL_SIZE; i++) {
        a[i] = i * 0.001;
        b[i] = i * 0.002;
    }

    // Compile expression once
    me_variable vars[] = {{"a"}, {"b"}};
    int error;
    me_expr *expr = me_compile("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &error);

    if (!expr) {
        printf("ERROR: Failed to compile at position %d\n", error);
        free(a);
        free(b);
        free(result);
        return 1;
    }

    // Process in chunks
    printf("Processing in chunks...\n");
    clock_t start = clock();

    int num_chunks = (TOTAL_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * CHUNK_SIZE;
        int current_size = CHUNK_SIZE;

        // Last chunk might be smaller
        if (offset + current_size > TOTAL_SIZE) {
            current_size = TOTAL_SIZE - offset;
        }

        // Pointers to current chunk
        const void *var_ptrs[] = {&a[offset], &b[offset]};

        // Evaluate this chunk
        me_eval(expr, var_ptrs, 2, &result[offset], current_size);
    }

    clock_t end = clock();
    double elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Calculate throughput metrics
    double melems_per_sec = (TOTAL_SIZE / 1e6) / elapsed;
    double gflops = (TOTAL_SIZE * (double) FLOPS_PER_ELEM / 1e9) / elapsed;
    double bandwidth_gb = (TOTAL_SIZE * 3 * sizeof(double) / 1e9) / elapsed;

    // Verify some results
    printf("\nSample results (first 5 elements):\n");
    printf("       a        b     sqrt(a²+b²)\n");
    printf("  ------   ------   ------------\n");
    for (int i = 0; i < 5; i++) {
        printf("  %6.3f   %6.3f   %12.3f\n", a[i], b[i], result[i]);
    }

    printf("\n✅ Processed %d elements in %.3f seconds\n", TOTAL_SIZE, elapsed);
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
