# miniexpr Examples

This directory contains practical examples demonstrating various features and use cases of the miniexpr library.

## Quick Start

From the repository root, build and run all examples:

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build
```

To run a specific example:

```bash
./build/examples/01_simple_expression
```

## Examples Overview

### 01_simple_expression.c
**Basic arithmetic expression evaluation**

- **What it demonstrates**: Simple two-variable arithmetic expression
- **Expression**: `(x + y) * 2`
- **Concepts**: Basic compilation, evaluation, and cleanup
- **Best for**: First-time users learning the API

**Run it:**
```bash
./build/examples/01_simple_expression
```

**Expected output:**
```
=== Simple Expression Example ===
Expression: (x + y) * 2

Results:
  x     y     (x+y)*2
----  ----  ---------
   1    10         22
   2    20         44
   ...
```

---

### 02_complex_expression.c
**Complex mathematical formulas with functions**

- **What it demonstrates**: Multi-variable expressions with built-in math functions
- **Expression**: `v*t*cos(angle) - 0.5*g*t*t` (projectile motion)
- **Concepts**: Multiple variables, trigonometry, nested operations
- **Best for**: Scientific computing applications

**Run it:**
```bash
./build/examples/02_complex_expression
```

**Use cases:**
- Physics simulations
- Engineering calculations
- Scientific data processing

---

### 03_mixed_types.c
**Automatic type promotion and inference**

- **What it demonstrates**: Mixing different data types (int32 and float64)
- **Expression**: `a + b` where a=int32, b=float64
- **Concepts**: ME_AUTO type inference, type promotion
- **Best for**: Working with heterogeneous data

**Run it:**
```bash
./build/examples/03_mixed_types
```

**Key features:**
- Automatic type promotion (int32 → float64)
- ME_AUTO infers output type from inputs
- Explicit type specification in variables

---

### 04_large_dataset.c
**Processing large arrays efficiently**

- **What it demonstrates**: Chunk-based processing for memory efficiency
- **Dataset size**: 44.7 million elements (~1GB working set)
- **Chunk size**: 32K elements (768 KB, cache-optimized)
- **Expression**: `sqrt(a*a + b*b)` (4 FLOPs/element)
- **Concepts**: Chunked evaluation, memory management, performance
- **Best for**: Large-scale data processing

**Run it:**
```bash
./build/examples/04_large_dataset
```

**Use cases:**
- Processing datasets larger than available memory
- Streaming data from disk
- Reducing memory footprint
- Improving cache efficiency

**Performance metrics included:**
- Processing time
- Throughput (Melems/sec)
- Computational performance (GFLOP/s)
- Memory bandwidth (GB/s)

---

### 05_parallel_evaluation.c
**Multi-threaded parallel processing**

- **What it demonstrates**: Thread-safe parallel evaluation
- **Threads**: 4 concurrent workers
- **Total size**: 44.7 million elements (~1GB working set)
- **Chunk size**: 32K elements (768 KB, cache-optimized)
- **Expression**: `sqrt(a*a + b*b)` (4 FLOPs/element convention, ~23 actual)
- **Concepts**: Thread safety, parallel processing, performance scaling
- **Best for**: Multi-core performance optimization

**Run it:**
```bash
./build/examples/05_parallel_evaluation
```

**Key features:**
- `me_eval()` allows concurrent evaluation
- Same compiled expression used by multiple threads
- Linear speedup with number of cores
- No locks needed - expression is read-only

**Performance metrics included:**
- Throughput (Melems/sec)
- Computational performance (GFLOP/s)
- Memory bandwidth (GB/s)
- Speedup analysis

**Use cases:**
- High-performance computing
- Real-time data processing
- Maximizing CPU utilization

**Important Notes:**

**On Speedup:**
This example shows **~2.85× speedup** over serial using cache-optimized 32K chunks.
The chunk size is critical for parallel performance:
- 32K elements = 768 KB fits in L3 cache
- 4 threads × 768 KB = 3 MB total (fits in typical L3)
- Minimizes cache misses and memory bandwidth contention
- Result: 1445 Melems/sec vs 507 Melems/sec serial

**On Cache Optimization:**
Chunk size matters enormously for parallel performance:
- Tested 10 sizes from 4K to 2M elements
- 32K elements (768 KB) is optimal for this machine
- Performance drops sharply with larger chunks (cache overflow)
- See `examples/test_chunk_sizes.c` for the benchmark tool

**On FLOP Counting:**
- **Convention**: sqrt = 1 FLOP (industry standard)
- **Reality**: sqrt ≈ 20 FLOPs (hardware cycles: ~15-20 vs ~3-5 for mul/add)
- We report conventional count in metrics, document actual in comments

---

### 06_debug_print.c
**Expression tree visualization for debugging**

- **What it demonstrates**: Using `me_print()` to visualize expression trees
- **Expressions**: Various complexity levels
- **Concepts**: Debugging, tree structure, operator precedence
- **Best for**: Understanding how expressions are parsed

**Run it:**
```bash
./build/examples/06_debug_print
```

**What you'll see:**
- Tree structure with indentation
- Function nodes (f0, f1, f2, ...)
- Variable references (bound <address>)
- Constant values

**Use cases:**
- Debugging complex expressions
- Understanding operator precedence
- Verifying expression parsing
- Learning the internal representation

---

### 07_comparison_bool_output.c
**Comparison operators with boolean array output**

- **What it demonstrates**: Getting bool arrays from comparison expressions
- **Expressions**: `a**2 == (a+b)`, `x < y`, Pythagorean theorem checks
- **Concepts**: ME_BOOL output, type conversion, comparison operators
- **Best for**: Filtering, masking, conditional operations

**Run it:**
```bash
./build/examples/07_comparison_bool_output
```

**Key features:**
- Explicit variable dtypes with ME_BOOL output
- ME_AUTO that auto-infers ME_BOOL for comparisons
- All comparison operators: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Complex expressions with power operations

**Use cases:**
- Creating boolean masks for filtering
- Validating mathematical relationships
- Implementing conditional logic on arrays
- Data validation and quality checks

---

### 08_explicit_output_dtype.c
**Explicit variable types with explicit output dtype**

- **What it demonstrates**: Specifying both variable types and output dtype simultaneously
- **Expressions**: `a + b`, `x * 2.5 + y`, `a > b`
- **Concepts**: Heterogeneous types, type casting, explicit output control
- **Best for**: Memory efficiency, type safety, mixed-type inputs with specific output requirements

**Run it:**
```bash
./build/examples/08_explicit_output_dtype
```

**Key features:**
- Mixed types (INT32 + FLOAT64) with FLOAT32 output
- FLOAT32 variables with FLOAT64 output
- Comparison with explicit BOOL output
- Variables keep their types during computation, result is cast to output type

**Use cases:**
- Computing in lower precision (FLOAT32) but outputting in higher precision (FLOAT64)
- Mixed-type inputs with specific output type requirements
- Type-safe operations with explicit control over output
- Memory-efficient computation with flexible output types

---

### 09_reduction_expressions.c
**Reductions inside larger expressions**

- **What it demonstrates**: Reductions over expressions and reductions used inside larger expressions
- **Expressions**: `sum(x + 1)`, `x + sum(x)`
- **Concepts**: Reduction arguments, scalar broadcast
- **Best for**: Mixing aggregate values with elementwise math

**Run it:**
```bash
./build/examples/09_reduction_expressions
```

---

### 10_boolean_logical_ops.c
**Logical operators on boolean arrays**

- **What it demonstrates**: Logical semantics for `and`/`or`/`not` (plus `&`, `|`, `^`, `~`) on bool arrays
- **Expressions**: `a and b`, `a or b`, `a ^ b`, `not a`, `o0 > 0.5 and o1 > 10000 or o1 == 42`
- **Concepts**: ME_BOOL output, boolean masks, comparison composition
- **Best for**: Masking, filtering, conditional logic

**Run it:**
```bash
./build/examples/10_boolean_logical_ops
```

---

### 11_dsl_kernel.c
**DSL multi-statement kernel demonstration**

- **What it demonstrates**: DSL parsing for multi-statement programs
- **Programs**: Polynomial evaluation, conditional clamping, trig identities
- **Concepts**: Temporary variables, where() conditionals, DSL parsing API
- **Best for**: Understanding DSL features and syntax

**Run it:**
```bash
./build/examples/11_dsl_kernel
```

**Key features:**
- Multi-statement programs with temporary variables
- Conditional expressions with `where(cond, then, else)`
- DSL parsing via `me_dsl_parse()`

---

### 12_mandelbrot.c
**Mandelbrot set computation**

- **What it demonstrates**: Using miniexpr for fractal computation
- **Grid size**: 78×32 (2496 points)
- **Iterations**: 100 max
- **Concepts**: Complex arithmetic via components, iteration, ASCII visualization
- **Best for**: Understanding expression-based iterative algorithms

**Run it:**
```bash
./build/examples/12_mandelbrot
```

**Key features:**
- Complex number arithmetic (z = z² + c)
- Iterative evaluation with escape detection
- ASCII art visualization
- Performance metrics

---

### 13_mandelbrot_dsl.c
**Mandelbrot set with DSL kernel**

- **What it demonstrates**: Full DSL expressiveness for complex algorithms
- **Grid size**: 78×32 (2496 points)
- **Iterations**: 100 max
- **Concepts**: For loops, break conditions, temporaries, where() conditionals
- **Best for**: Understanding DSL kernel programming

**Run it:**
```bash
./build/examples/13_mandelbrot_dsl
```

**Key features:**
- Complete algorithm in a single DSL program
- `for iter in range(100): ...` loop construct
- `if all(escape_iter != 100.0): break` early exit
- `where()` conditionals for element-wise selection
- Comments with `#` syntax
- Demonstrates parsing via `me_dsl_parse()`

**DSL Kernel Preview:**
```
def mandelbrot(cr, ci):
    zr = 0.0
    zi = 0.0
    escape_iter = 100.0
    for iter in range(100):
        zr2 = zr * zr
        zi2 = zi * zi
        mag2 = zr2 + zi2
        just_escaped = mag2 > 4.0 and escape_iter == 100.0
        escape_iter = where(just_escaped, iter, escape_iter)
        if all(escape_iter != 100.0):
            break
        zr_new = zr2 - zi2 + cr
        zi_new = 2.0 * zr * zi + ci
        zr = zr_new
        zi = zi_new
    return escape_iter
```

---

### 14_string_ops.c
**String-aware expressions with dynamic dispatch**

- **What it demonstrates**: String comparisons and predicates in expressions
- **Expressions**: Comparisons and other boolean conditions involving strings
- **Concepts**: String literals, string variables, and boolean-valued string operations
- **Best for**: Adding simple string-based conditions and metadata-driven logic

**Run it:**
```bash
./build/examples/14_string_ops
```

---

### 15_dsl_print.c
**Printing from DSL programs**

- **What it demonstrates**: DSL `print()` statements for debugging
- **Programs**: Simple expressions with intermediate printouts
- **Concepts**: DSL I/O and debugging helpers
- **Best for**: Understanding and validating DSL program flow

**Run it:**
```bash
./build/examples/15_dsl_print
```

---

### 16_nd_padding_example.c
**N-dimensional padding for array expressions**

- **What it demonstrates**: Padding N-D arrays before evaluation
- **Expressions**: N-D input with padding-aware evaluation
- **Concepts**: N-D metadata, padding configuration, and boundary handling
- **Best for**: Image/volume processing and stencil-style operations

**Run it:**
```bash
./build/examples/16_nd_padding_example
```

---

### 17_dsl_user_function.c
**User-defined functions in DSL programs**

- **What it demonstrates**: Registering a custom C function for DSL evaluation
- **Expressions**: Calling a user-defined function from a DSL program
- **Concepts**: `me_variable_ex` function entries, explicit return dtype
- **Best for**: Extending DSL with domain-specific helpers

**Run it:**
```bash
./build/examples/17_dsl_user_function
```

---

### 18_dsl_if_elif_else.c
**Scalar if/elif/else and flow-only loop control**

- **What it demonstrates**: DSL `if/elif/else` blocks with scalar conditions
- **Programs**: Result assignment branches and flow-only loop control
- **Concepts**: Uniform conditions, required `result` assignment, `break`/`continue` chains
- **Best for**: Using structured control flow in DSL kernels

**Run it:**
```bash
./build/examples/18_dsl_if_elif_else
```

---

## Building Examples

### Using CMake (recommended)

From the repo root:
```bash
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build
```

### Using the Makefile

Add to your `Makefile`:

```make
EXAMPLE_SRCS = $(wildcard examples/*.c)
EXAMPLE_BINS = $(patsubst examples/%.c,$(BUILDDIR)/%,$(EXAMPLE_SRCS))

examples: $(EXAMPLE_BINS)

$(BUILDDIR)/%: examples/%.c $(BUILDDIR)/miniexpr.o $(BUILDDIR)/functions.o
	@echo "Building example: $@"
	$(CC) $(CFLAGS) -Isrc $< $(BUILDDIR)/miniexpr.o $(BUILDDIR)/functions.o -o $@ -lm
```

Then run:
```bash
make examples
```

### Manual Compilation

Compile any example manually:

```bash
gcc -O2 -Isrc examples/01_simple_expression.c build/miniexpr.o build/functions.o -o 01_simple_expression -lm
```

For the parallel example (requires pthreads):
```bash
gcc -O2 -Isrc examples/05_parallel_evaluation.c build/miniexpr.o build/functions.o -o 05_parallel_evaluation -lm -lpthread
```

---

## Common Patterns

### Basic Pattern (used in most examples)

```c
// 1. Define variables
me_variable vars[] = {{"x"}, {"y"}};

// 2. Compile expression
int error;
me_expr *expr = NULL;
if (me_compile("x + y", vars, 2, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// 3. Prepare data pointers
const void *var_ptrs[] = {x_data, y_data};

// 4. Evaluate
if (me_eval(expr, var_ptrs, 2, result, n, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
// 5. Cleanup
me_free(expr);
```

### Error Handling Pattern

```c
if (me_compile(expression, vars, var_count, dtype, &error, &expr) != ME_COMPILE_SUCCESS) {
    printf("Compile error (pos=%d)\n", error);
    return 1;
}
```

### Chunk Processing Pattern

```c
for (int chunk = 0; chunk < num_chunks; chunk++) {
    int offset = chunk * CHUNK_SIZE;
    int size = min(CHUNK_SIZE, TOTAL_SIZE - offset);

    const void *var_ptrs[] = {&data[offset]};
    if (me_eval(expr, var_ptrs, 1, &result[offset], size, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
}
```

---

## Performance Tips

From the examples, you can learn:

1. **Compile once, evaluate many times** (Examples 04, 05)
   - Compilation is more expensive than evaluation
   - Reuse compiled expressions when possible

2. **Use chunking for large datasets** (Example 04)
   - Better cache utilization
   - Enables streaming from disk
   - Reduces memory footprint

3. **Parallelize for performance** (Example 05)
   - `me_eval()` enables safe parallelism
   - Linear speedup with number of cores
   - No synchronization overhead

4. **Choose appropriate data types** (Example 03)
   - Let ME_AUTO infer types when mixing
   - Use explicit types for control
   - Understand promotion rules

---

## Next Steps

After trying these examples:

1. **Modify them**: Change expressions and see what happens
2. **Combine patterns**: Use chunking + parallelism together
3. **Read the docs**: Check `doc/` directory for detailed tutorials
4. **Write your own**: Adapt examples to your use case

## Additional Resources

- **Type Inference Guide**: `../doc/type-inference.md`
- **Getting Started Guide**: `../doc/get-started.md`
- **Data Types Guide**: `../doc/data-types.md`
- **Parallel Processing Guide**: `../doc/parallel-processing.md`

---

## Summary Table

| Example | Complexity | Key Concept | Lines | Time to Run |
|---------|-----------|-------------|-------|-------------|
| 01 | ⭐ Simple | Basic API | ~50 | <1s |
| 02 | ⭐⭐ Moderate | Math functions | ~70 | <1s |
| 03 | ⭐⭐ Moderate | Type mixing | ~75 | <1s |
| 04 | ⭐⭐⭐ Advanced | Large data | ~110 | ~1s |
| 05 | ⭐⭐⭐ Advanced | Parallelism | ~140 | ~1s |
| 06 | ⭐ Simple | Debugging | ~70 | <1s |
| 07 | ⭐⭐ Moderate | Bool output | ~180 | <1s |
| 08 | ⭐⭐ Moderate | Explicit output dtype | ~110 | <1s |
| 09 | ⭐⭐ Moderate | Reductions | ~80 | <1s |
| 10 | ⭐⭐ Moderate | Boolean logic | ~110 | <1s |
| 11 | ⭐⭐ Moderate | DSL kernels | ~160 | <1s |
| 12 | ⭐⭐⭐ Advanced | Mandelbrot | ~170 | ~1s |
| 13 | ⭐⭐⭐ Advanced | DSL Mandelbrot | ~210 | ~1s |
| 14 | ⭐⭐ Moderate | String ops | ~120 | <1s |
| 15 | ⭐ Simple | DSL print | ~40 | <1s |
| 16 | ⭐⭐ Moderate | N-D padding | ~90 | <1s |
| 17 | ⭐ Simple | DSL UDFs | ~70 | <1s |

**Start with Example 01, then explore based on your needs!** 🚀
