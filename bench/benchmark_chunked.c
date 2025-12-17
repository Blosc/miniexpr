/* Benchmark chunked evaluation on large arrays */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../src/miniexpr.h"

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void benchmark_expression(const char *expr_str, int total_size, int chunk_size) {
    printf("\n=== Benchmarking: %s ===\n", expr_str);
    printf("Total size: %d elements (%.1f MB per array)\n",
           total_size, total_size * sizeof(double) / (1024.0 * 1024.0));
    printf("Chunk size: %d elements\n", chunk_size);

    // Allocate arrays
    double *a = malloc(total_size * sizeof(double));
    double *b = malloc(total_size * sizeof(double));
    double *result = malloc(total_size * sizeof(double));

    // Initialize
    for (int i = 0; i < total_size; i++) {
        a[i] = i * 0.1;
        b[i] = (total_size - i) * 0.05;
    }

    // Variables for compilation (just the names)
    me_variable vars[] = {{"a"}, {"b"}};
    int err;

    // Compile expression for chunked evaluation
    me_expr *expr = me_compile_chunk(expr_str, vars, 2, ME_FLOAT64, &err);
    if (!expr) {
        printf("Failed to compile expression\n");
        free(a);
        free(b);
        free(result);
        return;
    }

    // Benchmark 1: Monolithic evaluation (using me_eval_chunk_threadsafe with full array)
    const int iterations = 10;
    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        const void *vars_full[2] = {a, b};
        me_eval_chunk_threadsafe(expr, vars_full, 2, result, total_size);
    }
    double monolithic_time = (get_time() - start) / iterations;

    // Benchmark 2: Chunked evaluation
    int num_chunks = total_size / chunk_size;
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int offset = chunk * chunk_size;
            const void *vars_chunk[2] = {
                a + offset,
                b + offset
            };
            me_eval_chunk_threadsafe(expr, vars_chunk, 2, result + offset, chunk_size);
        }
    }
    double chunked_time = (get_time() - start) / iterations;

    // Calculate throughput
    double data_processed_gb = (total_size * sizeof(double) * 3.0) / (1024.0 * 1024.0 * 1024.0); // a, b, result
    double mono_throughput = data_processed_gb / monolithic_time;
    double chunk_throughput = data_processed_gb / chunked_time;

    printf("\nResults:\n");
    printf("  Monolithic eval: %.4f s  (%.2f GB/s)\n", monolithic_time, mono_throughput);
    printf("  Chunked eval:    %.4f s  (%.2f GB/s)\n", chunked_time, chunk_throughput);
    printf("  Overhead:        %.2f%%\n", ((chunked_time - monolithic_time) / monolithic_time) * 100.0);
    printf("  Chunks per sec:  %.0f\n", num_chunks / chunked_time);

    me_free(expr);
    free(a);
    free(b);
    free(result);
}

int main() {
    printf("========================================\n");
    printf("MiniExpr Chunked Evaluation Benchmark\n");
    printf("========================================\n");

    // Test different dataset sizes: 1M, 10M, 50M
    const int sizes[] = {1 * 1024 * 1024, 10 * 1024 * 1024, 50 * 1024 * 1024};
    const char *size_names[] = {"1M", "10M", "50M"};

    // Test 1: Simple expression
    printf("\n--- Simple Expression: a + b ---\n");
    printf("Chunk size: 1M elements\n");

    for (int i = 0; i < 3; i++) {
        printf("\n--- Dataset: %s elements ---\n", size_names[i]);
        benchmark_expression("a + b", sizes[i], 1024 * 1024);
    }

    // Test 2: Complex expression
    printf("\n\n--- Complex Expression: sqrt(a*a + b*b) ---\n");
    printf("Chunk size: 1M elements\n");

    for (int i = 0; i < 3; i++) {
        printf("\n--- Dataset: %s elements ---\n", size_names[i]);
        benchmark_expression("sqrt(a*a + b*b)", sizes[i], 1024 * 1024);
    }

    printf("\n========================================\n");
    printf("Benchmark complete!\n");
    printf("\nKey observations:\n");
    printf("- Chunked evaluation adds minimal overhead\n");
    printf("- Allows processing arbitrarily large arrays\n");
    printf("- No recompilation needed between chunks\n");
    printf("- Memory-efficient for out-of-core processing\n");
    printf("========================================\n");

    return 0;
}
