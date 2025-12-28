/* Benchmark: Eval block size tuning for cache locality
 *
 * Build with different block sizes to compare:
 *   make CFLAGS="-O2 -DNDEBUG -DME_EVAL_BLOCK_NITEMS=4096" build/benchmark_blocksize
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include "../src/miniexpr.h"

#define GIB_BYTES (1024ULL * 1024ULL * 1024ULL)

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static double run_benchmark(const me_expr *expr, const void **var_ptrs,
                            double *out, int total_elems, int iterations) {
    // Warm-up
    me_eval(expr, var_ptrs, 3, out, total_elems);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        me_eval(expr, var_ptrs, 3, out, total_elems);
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

    const void *var_ptrs[] = {a, b, c};
    const double data_gb = (double)(total_elems * sizeof(double) * 4ULL) / 1e9;

    printf("\nInternal block size: %d elements (compile-time)\n", ME_EVAL_BLOCK_NITEMS);
    printf("Results (fixed block size):\n");
    double elapsed = run_benchmark(expr, var_ptrs, out, total_elems_i, 5);
    double throughput = data_gb / elapsed;
    printf("  Avg time:   %.4f s\n", elapsed);
    printf("  Throughput: %.2f GB/s\n", throughput);

    me_free(expr);
    free(a);
    free(b);
    free(c);
    free(out);
}

int main(void) {
    printf("=============================================\n");
    printf("MiniExpr Eval Block Size Benchmark\n");
    printf("=============================================\n");

    const size_t total_var_bytes = GIB_BYTES;
    const size_t total_elems = total_var_bytes / (3ULL * sizeof(double));

    if (total_elems > (size_t)INT_MAX) {
        printf("ERROR: Dataset too large for int-sized nitems\n");
        return 1;
    }

    printf("Total variable working set: %.2f GB\n",
           (double)total_var_bytes / 1e9);

    benchmark_block_sizes(total_elems);

    printf("\n=============================================\n");
    printf("Benchmark complete\n");
    printf("=============================================\n");

    return 0;
}
