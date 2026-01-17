# Processing Large Datasets with Chunks Tutorial

This tutorial demonstrates how to use `me_eval()` to process large datasets efficiently by breaking them into smaller chunks.

## Why Use Chunks?

When working with very large arrays (millions or billions of elements), you may want to:
1. **Reduce memory usage** - Process data in smaller pieces
2. **Enable streaming** - Read from disk, process, and write back without loading everything into memory
3. **Improve cache efficiency** - Smaller chunks fit better in CPU cache

## Basic Chunk Processing

Let's process a large dataset of temperature conversions (Celsius to Fahrenheit):

```c
#include <stdio.h>
#include <stdlib.h>
#include "miniexpr.h"

#define TOTAL_SIZE 1000000  // 1 million elements
#define CHUNK_SIZE 10000    // Process 10k at a time

int main() {
    // Allocate a large array of temperatures in Celsius
    double *celsius = malloc(TOTAL_SIZE * sizeof(double));
    double *fahrenheit = malloc(TOTAL_SIZE * sizeof(double));

    if (!celsius || !fahrenheit) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize with sample data
    for (int i = 0; i < TOTAL_SIZE; i++) {
        celsius[i] = -50.0 + (i * 0.0001); // Range from -50°C to 50°C
    }

    // Define variables (minimal form - name only, dtypes default to ME_AUTO)
    me_variable vars[] = {{"c"}};

    // Compile the expression once for chunked evaluation
    // All variables will use ME_FLOAT64 since output dtype is specified
    int error;
    me_expr *expr = NULL;
    if (me_compile("c * 9/5 + 32", vars, 1, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }

    // Process in chunks
    printf("Processing %d elements in chunks of %d...\n",
           TOTAL_SIZE, CHUNK_SIZE);

    int num_chunks = (TOTAL_SIZE + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * CHUNK_SIZE;
        int current_chunk_size = CHUNK_SIZE;

        // Last chunk might be smaller
        if (offset + current_chunk_size > TOTAL_SIZE) {
            current_chunk_size = TOTAL_SIZE - offset;
        }

        // Pointers to current chunk
        const void *vars_block[] = {&celsius[offset]};
        void *output_chunk = &fahrenheit[offset];

        // Evaluate this chunk
        if (me_eval(expr, vars_block, 1, output_chunk, current_chunk_size, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
        if ((chunk + 1) % 10 == 0) {
            printf("Processed chunk %d/%d (%.1f%%)\n",
                   chunk + 1, num_chunks,
                   100.0 * (chunk + 1) / num_chunks);
        }
    }

    printf("Done!\n\n");

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.2f°C = %.2f°F\n", celsius[i], fahrenheit[i]);
    }

    // Clean up
    me_free(expr);
    free(celsius);
    free(fahrenheit);

    return 0;
}
```

### Expected Output

```
Processing 1000000 elements in chunks of 10000...
Processed chunk 10/100 (10.0%)
Processed chunk 20/100 (20.0%)
Processed chunk 30/100 (30.0%)
Processed chunk 40/100 (40.0%)
Processed chunk 50/100 (50.0%)
Processed chunk 60/100 (60.0%)
Processed chunk 70/100 (70.0%)
Processed chunk 80/100 (80.0%)
Processed chunk 90/100 (90.0%)
Processed chunk 100/100 (100.0%)
Done!

Sample results:
-50.00°C = -58.00°F
-49.99°C = -57.99°F
-49.99°C = -57.99°F
-49.99°C = -57.98°F
-49.99°C = -57.98°F
```

## Streaming Data from Files

Here's an example that reads data from a file, processes it in chunks, and writes results:

```c
#include <stdio.h>
#include <stdlib.h>
#include "miniexpr.h"

#define CHUNK_SIZE 1000

int main() {
    FILE *input = fopen("input.dat", "rb");
    FILE *output = fopen("output.dat", "wb");

    if (!input || !output) {
        printf("Failed to open files\n");
        return 1;
    }

    // Allocate chunk buffers
    double *x_chunk = malloc(CHUNK_SIZE * sizeof(double));
    double *y_chunk = malloc(CHUNK_SIZE * sizeof(double));
    double *result_chunk = malloc(CHUNK_SIZE * sizeof(double));

    // Define variables (just the names - everything else optional)
    me_variable vars[] = {{"x"}, {"y"}};

    int error;
    me_expr *expr = NULL;
    if (me_compile("sqrt(x*x + y*y)", vars, 2, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }

    // Process file in chunks
    size_t total_processed = 0;
    size_t elements_read;

    while (1) {
        // Read chunk from file
        elements_read = fread(x_chunk, sizeof(double), CHUNK_SIZE, input);
        if (elements_read == 0) break;

        fread(y_chunk, sizeof(double), elements_read, input);

        // Process this chunk
        const void *vars_block[] = {x_chunk, y_chunk};
        if (me_eval(expr, vars_block, 2, result_chunk, elements_read, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
        // Write results
        fwrite(result_chunk, sizeof(double), elements_read, output);

        total_processed += elements_read;
        printf("Processed %zu elements\r", total_processed);
        fflush(stdout);
    }

    printf("\nTotal processed: %zu elements\n", total_processed);

    me_free(expr);

cleanup:
    free(x_chunk);
    free(y_chunk);
    free(result_chunk);
    fclose(input);
    fclose(output);

    return 0;
}
```

## Advanced: Multiple Variables with Different Chunks

Processing related but separate arrays:

```c
#include <stdio.h>
#include <stdlib.h>
#include "miniexpr.h"

int main() {
    const int TOTAL = 100000;
    const int CHUNK = 5000;

    // Separate arrays for different metrics
    float *temperature = malloc(TOTAL * sizeof(float));
    float *pressure = malloc(TOTAL * sizeof(float));
    float *volume = malloc(TOTAL * sizeof(float));
    float *moles = malloc(TOTAL * sizeof(float));

    // Initialize with sample data
    for (int i = 0; i < TOTAL; i++) {
        pressure[i] = 101325.0f + i * 10.0f;  // Pascals
        volume[i] = 0.001f + i * 0.000001f;   // m³
        moles[i] = 1.0f;                       // mol
    }

    // Ideal gas law: T = (P * V) / (n * R)
    // R = 8.314 J/(mol·K)
    // Define variables (just the names - everything else optional)
    me_variable vars[] = {{"P"}, {"V"}, {"n"}};

    int error;
    me_expr *expr = NULL;
    if (me_compile("(P * V) / (n * 8.314)", vars, 3, ME_FLOAT32, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }

    // Process in chunks
    for (int offset = 0; offset < TOTAL; offset += CHUNK) {
        int size = (offset + CHUNK > TOTAL) ? (TOTAL - offset) : CHUNK;

        const void *vars_block[] = {
            &pressure[offset],
            &volume[offset],
            &moles[offset]
        };

        if (me_eval(expr, vars_block, 3, &temperature[offset], size, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
    }

    printf("Computed temperatures for %d samples\n", TOTAL);
    printf("First result: T = %.2f K (%.2f°C)\n",
           temperature[0], temperature[0] - 273.15);

    me_free(expr);
    free(temperature);
    free(pressure);
    free(volume);
    free(moles);

    return 0;
}
```

## Multidimensional chunks with padding (b2nd-style)

When working with b2nd arrays (chunk/ block grids with edge padding), use the `_nd` APIs:

- `me_compile_nd(expr, vars, nvars, dtype, ndims, shape, chunkshape, blockshape, ...)`
- `me_eval_nd(expr, vars_block, nvars, out_block, block_nitems, nchunk, nblock, params)`
- `me_nd_valid_nitems(expr, nchunk, nblock, &valid)`

Key points:
1. `shape`, `chunkshape`, `blockshape` are C-order arrays (length = `ndims`).
2. `nchunk` is the zero-based chunk index over the whole array (C-order); `nblock` is the block index inside that chunk (also C-order).
3. Callers pass *padded* block buffers (size = `prod(blockshape)` elements). `me_eval_nd` computes only the valid elements and zero-fills the padded tail in the output.
4. For expressions whose overall result is a scalar (reductions like `sum(x)` or `sum(x) + 1`), `output_block` only needs space for one item; `me_eval_nd` writes a single element and does not zero any tail.
5. For best performance with padding, `me_eval_nd` packs valid elements, evaluates once, and scatters back; fully valid blocks still take a single fast path.

Minimal 2D example (padding on edges):

```c
int64_t shape[2]      = {5, 4};
int32_t chunkshape[2] = {3, 3};
int32_t blockshape[2] = {2, 2};
me_variable vars[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};
me_expr *expr = NULL;
int err;
me_compile_nd("x + y", vars, 2, ME_FLOAT64, 2,
              shape, chunkshape, blockshape, &err, &expr);

/* Block buffers are always padded to prod(blockshape) */
double x_block[4], y_block[4], out_block[4];
const void *ptrs[] = {x_block, y_block};
int64_t nchunk = 1; /* chunk (1,1) in C-order for this shape */
int64_t nblock = 0; /* first block inside that chunk */

int64_t valid = 0;
me_nd_valid_nitems(expr, nchunk, nblock, &valid); /* tells how many outputs are real */
me_eval_nd(expr, ptrs, 2, out_block, 4, nchunk, nblock, NULL);
/* out_block[valid..] is zeroed */
```

See `examples/11_nd_padding_example.c` for a fuller walkthrough, and `bench/benchmark_nd_padding` to gauge performance with different padding patterns.

## Key Points

1. **Compile once** - Create the expression once, then reuse it for all chunks
2. **Manage chunk boundaries** - Handle the last chunk which might be smaller
3. **Use const void* arrays** - Pass pointers to chunk starts via `vars_block`
4. **Update pointers** - For each chunk, point to the correct offset in your arrays
5. **Thread-safe** - `me_eval()` is safe for parallel processing from multiple threads

## Benefits of Chunk Processing

- **Memory efficient**: Process datasets larger than RAM
- **Cache friendly**: Better CPU cache utilization with smaller chunks
- **Progress tracking**: Easy to report progress during long operations
- **Flexible**: Can pause, resume, or skip chunks as needed
