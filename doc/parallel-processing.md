# Parallel Processing with Thread-Safe Evaluation Tutorial

This tutorial demonstrates how to use `me_eval()` to process data in parallel across multiple threads, maximizing performance on multi-core systems.

## Why Parallel Processing?

Modern CPUs have multiple cores. By distributing work across threads, you can:
- **Reduce processing time** by utilizing all available cores
- **Increase throughput** for large-scale data processing
- **Maximize hardware efficiency** without complex parallelization logic

## Basic Parallel Example with Pthreads

Let's compute a complex mathematical expression across multiple threads:

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "miniexpr.h"

#define TOTAL_SIZE 1000000
#define NUM_THREADS 4

typedef struct {
    me_expr *expr;          // Compiled expression (shared)
    double *x;              // Input array x (shared)
    double *y;              // Input array y (shared)
    double *result;         // Output array (shared)
    int start_idx;          // Start index for this thread
    int end_idx;            // End index for this thread
    int thread_id;          // Thread identifier
} thread_data_t;

void* worker_thread(void *arg) {
    thread_data_t *data = (thread_data_t*)arg;

    int chunk_size = data->end_idx - data->start_idx;

    // Prepare pointers to this thread's chunk
    const void *vars_block[] = {
        &data->x[data->start_idx],
        &data->y[data->start_idx]
    };

    void *output_chunk = &data->result[data->start_idx];

    printf("Thread %d: Processing elements %d to %d (%d elements)\n",
           data->thread_id, data->start_idx, data->end_idx - 1, chunk_size);

    // Thread-safe evaluation
    if (me_eval(data->expr, vars_block, 2, output_chunk, chunk_size, NULL) != ME_EVAL_SUCCESS) {
        printf("Thread %d: me_eval failed\n", data->thread_id);
        return NULL;
    }

    printf("Thread %d: Completed\n", data->thread_id);

    return NULL;
}

int main() {
    // Allocate data arrays
    double *x = malloc(TOTAL_SIZE * sizeof(double));
    double *y = malloc(TOTAL_SIZE * sizeof(double));
    double *result = malloc(TOTAL_SIZE * sizeof(double));

    if (!x || !y || !result) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize with sample data
    printf("Initializing %d elements...\n", TOTAL_SIZE);
    for (int i = 0; i < TOTAL_SIZE; i++) {
        x[i] = (double)i / 1000.0;
        y[i] = (double)i / 2000.0;
    }

    // Compile the expression once for chunked evaluation
    // Expression: sin(x) * cos(y) + sqrt(x*x + y*y)
    // Define variables (minimal form - names only, will use ME_FLOAT64)
    me_variable vars[] = {{"x"}, {"y"}};

    int error;
    me_expr *expr = NULL;
    if (me_compile("sin(x) * cos(y) + sqrt(x*x + y*y)",
                               vars, 2, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }

    printf("Expression compiled successfully\n");
    printf("Starting parallel processing with %d threads...\n\n", NUM_THREADS);

    // Create threads
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];

    int chunk_size = TOTAL_SIZE / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].expr = expr;
        thread_data[i].x = x;
        thread_data[i].y = y;
        thread_data[i].result = result;
        thread_data[i].start_idx = i * chunk_size;
        thread_data[i].end_idx = (i == NUM_THREADS - 1) ?
                                  TOTAL_SIZE : (i + 1) * chunk_size;
        thread_data[i].thread_id = i;

        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("\nAll threads completed!\n\n");

    // Verify results
    printf("Sample results:\n");
    for (int i = 0; i < 5; i++) {
        printf("x=%.3f, y=%.3f -> result=%.6f\n",
               x[i], y[i], result[i]);
    }

    // Clean up
    me_free(expr);
    free(x);
    free(y);
    free(result);

    return 0;
}
```

### Compilation

```bash
gcc -o parallel parallel.c src/miniexpr.c src/functions.c -lm -lpthread
./parallel
```

### Expected Output

```
Initializing 1000000 elements...
Expression compiled successfully
Starting parallel processing with 4 threads...

Thread 0: Processing elements 0 to 249999 (250000 elements)
Thread 1: Processing elements 250000 to 499999 (250000 elements)
Thread 2: Processing elements 500000 to 749999 (250000 elements)
Thread 3: Processing elements 750000 to 999999 (250000 elements)
Thread 0: Completed
Thread 2: Completed
Thread 1: Completed
Thread 3: Completed

All threads completed!

Sample results:
x=0.000, y=0.000 -> result=0.000000
x=0.001, y=0.001 -> result=0.002414
x=0.002, y=0.001 -> result=0.003828
x=0.003, y=0.002 -> result=0.005243
x=0.004, y=0.002 -> result=0.006657
```

## Advanced: Work Queue Pattern

For irregular workloads, use a work queue where threads dynamically grab chunks:

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "miniexpr.h"

#define TOTAL_SIZE 1000000
#define CHUNK_SIZE 10000
#define NUM_THREADS 8

typedef struct {
    me_expr *expr;
    double *input;
    double *output;
    int next_chunk;
    int total_chunks;
    pthread_mutex_t mutex;
} work_queue_t;

void* worker_dynamic(void *arg) {
    work_queue_t *queue = (work_queue_t*)arg;
    int chunks_processed = 0;

    while (1) {
        // Get next chunk (thread-safe)
        pthread_mutex_lock(&queue->mutex);
        int chunk_id = queue->next_chunk++;
        pthread_mutex_unlock(&queue->mutex);

        if (chunk_id >= queue->total_chunks) {
            break;  // No more work
        }

        // Calculate chunk boundaries
        int start = chunk_id * CHUNK_SIZE;
        int size = CHUNK_SIZE;
        if (start + size > TOTAL_SIZE) {
            size = TOTAL_SIZE - start;
        }

        // Process this chunk
        const void *vars_block[] = {&queue->input[start]};
        void *output_chunk = &queue->output[start];

        if (me_eval(queue->expr, vars_block, 1, output_chunk, size, NULL) != ME_EVAL_SUCCESS) {
            printf("Thread: me_eval failed\n");
            return NULL;
        }

        chunks_processed++;
    }

    printf("Thread completed %d chunks\n", chunks_processed);
    return NULL;
}

int main() {
    double *input = malloc(TOTAL_SIZE * sizeof(double));
    double *output = malloc(TOTAL_SIZE * sizeof(double));

    // Initialize data
    for (int i = 0; i < TOTAL_SIZE; i++) {
        input[i] = (double)i;
    }

    // Compile expression for chunked evaluation: x**2 + 2*x + 1
    // Define variable (just the name - everything else optional)
    me_variable vars[] = {{"x"}};

    int error;
    me_expr *expr = NULL;
    if (me_compile("x*x + 2*x + 1", vars, 1, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }

    // Set up work queue
    work_queue_t queue;
    queue.expr = expr;
    queue.input = input;
    queue.output = output;
    queue.next_chunk = 0;
    queue.total_chunks = (TOTAL_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE;
    pthread_mutex_init(&queue.mutex, NULL);

    printf("Processing %d elements in %d chunks with %d threads\n",
           TOTAL_SIZE, queue.total_chunks, NUM_THREADS);

    // Create worker threads
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, worker_dynamic, &queue);
    }

    // Wait for completion
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Processing complete!\n");

    // Clean up
    pthread_mutex_destroy(&queue.mutex);
    me_free(expr);
    free(input);
    free(output);

    return 0;
}
```

## OpenMP Example (Simpler Alternative)

If you have OpenMP support, parallel processing becomes even easier:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "miniexpr.h"

#define TOTAL_SIZE 10000000
#define CHUNK_SIZE 50000

int main() {
    double *a = malloc(TOTAL_SIZE * sizeof(double));
    double *b = malloc(TOTAL_SIZE * sizeof(double));
    double *c = malloc(TOTAL_SIZE * sizeof(double));

    // Initialize
    for (int i = 0; i < TOTAL_SIZE; i++) {
        a[i] = i * 0.1;
        b[i] = i * 0.2;
    }

    // Compile expression for chunked evaluation
    // Define variables (just the names - everything else optional)
    me_variable vars[] = {{"a"}, {"b"}};

    int error;
    me_expr *expr = NULL;
    if (me_compile("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }

    printf("Processing %d elements with OpenMP...\n", TOTAL_SIZE);

    // Parallel processing with OpenMP
    #pragma omp parallel for schedule(dynamic, 1)
    for (int chunk = 0; chunk < TOTAL_SIZE; chunk += CHUNK_SIZE) {
        int size = (chunk + CHUNK_SIZE > TOTAL_SIZE) ?
                   (TOTAL_SIZE - chunk) : CHUNK_SIZE;

        const void *vars_block[] = {&a[chunk], &b[chunk]};

        if (me_eval(expr, vars_block, 2, &c[chunk], size, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
        #pragma omp critical
        {
            printf("Thread %d processed chunk at %d\n",
                   omp_get_thread_num(), chunk);
        }
    }

    printf("Done!\n");

    me_free(expr);
    free(a);
    free(b);
    free(c);

    return 0;
}
```

### Compilation with OpenMP

```bash
gcc -o parallel_omp parallel_omp.c src/miniexpr.c src/functions.c -lm -fopenmp
./parallel_omp
```

## Performance Tips

1. **Choose appropriate chunk size** - Too small creates overhead, too large reduces parallelism
2. **Balance workload** - Divide work evenly across threads
3. **Minimize synchronization** - Each thread should work independently
4. **Reuse compiled expressions** - Compile once, use across all threads
5. **Consider data locality** - Align chunks with cache line boundaries when possible

## Thread Safety Notes

- `me_eval()` is safe for concurrent calls
- Each call creates a temporary expression clone internally
- The original compiled expression remains unchanged
- This enables safe parallel processing without data races

## When to Use Parallel Processing

Parallel processing is beneficial when:
- ✅ Dataset is large (> 100K elements typically)
- ✅ Expression is complex (many operations)
- ✅ Multiple CPU cores are available
- ✅ Processing time matters

Avoid parallelization when:
- ❌ Dataset is small (overhead exceeds benefits)
- ❌ Expression is trivial (memory bandwidth limited)
- ❌ Single-threaded performance is sufficient
