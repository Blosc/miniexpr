/* Test thread-safe chunked evaluation */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include "../src/miniexpr.h"

#define NUM_THREADS 4
#define CHUNK_SIZE 10000
#define TOTAL_SIZE (NUM_THREADS * CHUNK_SIZE)

typedef struct {
    me_expr *expr;
    double *a_data;
    double *b_data;
    double *result;
    int chunk_id;
    int chunk_size;
} thread_data_t;

void *worker_thread(void *arg) {
    thread_data_t *data = (thread_data_t *) arg;
    int offset = data->chunk_id * data->chunk_size;

    const void *vars_chunk[2] = {
        data->a_data + offset,
        data->b_data + offset
    };

    me_eval_chunk_threadsafe(data->expr, vars_chunk, 2,
                             data->result + offset, data->chunk_size);

    return NULL;
}

int test_parallel_evaluation(void) {
    printf("\n=== Testing Thread-safe Evaluation ===\n");

    // Allocate data
    double *a = malloc(TOTAL_SIZE * sizeof(double));
    double *b = malloc(TOTAL_SIZE * sizeof(double));
    double *result_parallel = malloc(TOTAL_SIZE * sizeof(double));
    double *result_serial = malloc(TOTAL_SIZE * sizeof(double));

    // Initialize
    for (int i = 0; i < TOTAL_SIZE; i++) {
        a[i] = i * 0.1;
        b[i] = (TOTAL_SIZE - i) * 0.05;
    }

    // Variables for compilation (just the names)
    me_variable vars[] = {{"a"}, {"b"}};
    int err;
    me_expr *expr = me_compile_chunk("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &err);
    if (!expr) {
        printf("  ❌ Compilation failed\n");
        return 1;
    }

    // Serial evaluation for reference
    const void *vars_serial[2] = {a, b};
    me_eval_chunk_threadsafe(expr, vars_serial, 2, result_serial, TOTAL_SIZE);

    // Parallel evaluation
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].expr = expr;
        thread_data[i].a_data = a;
        thread_data[i].b_data = b;
        thread_data[i].result = result_parallel;
        thread_data[i].chunk_id = i;
        thread_data[i].chunk_size = CHUNK_SIZE;

        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }

    // Wait for threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Verify results
    int passed = 1;
    int mismatches = 0;
    for (int i = 0; i < TOTAL_SIZE && mismatches < 5; i++) {
        double expected = sqrt(a[i] * a[i] + b[i] * b[i]);
        if (fabs(result_parallel[i] - result_serial[i]) > 1e-10 ||
            fabs(result_parallel[i] - expected) > 1e-10) {
            passed = 0;
            printf("  Mismatch at [%d]: parallel=%.6f, serial=%.6f, expected=%.6f\n",
                   i, result_parallel[i], result_serial[i], expected);
            mismatches++;
        }
    }

    if (passed) {
        printf("  ✅ PASSED: %d elements computed correctly across %d threads\n",
               TOTAL_SIZE, NUM_THREADS);
    } else {
        printf("  ❌ FAILED: Results don't match (found %d+ mismatches)\n", mismatches);
    }

    // Cleanup
    me_free(expr);
    free(a);
    free(b);
    free(result_parallel);
    free(result_serial);

    return passed ? 0 : 1;
}

int main() {
    printf("=== Thread-Safe Evaluation Test ===\n");
    printf("Testing with %d threads, %d elements per chunk\n", NUM_THREADS, CHUNK_SIZE);

    int result = test_parallel_evaluation();

    if (result == 0) {
        printf("\n✅ Thread-safe evaluation works correctly!\n");
    } else {
        printf("\n❌ Thread-safe evaluation failed!\n");
    }

    return result;
}
