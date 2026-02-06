# Benchmarks

This directory contains microbenchmarks for different aspects of miniexpr.

## Quick Start

Build all benchmarks:

```bash
make bench
```

Run a specific benchmark (examples below). CMake builds place binaries under `build/bench/`.

## Benchmarks

### benchmark_reductions.c
Reductions (sum/prod/min/max/any/all) vs pure C.

```bash
./build/bench/benchmark_reductions sum float32
./build/bench/benchmark_reductions sum float32 multi
```

### benchmark_reductions_threads.c
Multi-threaded reductions vs pure C.

```bash
./build/bench/benchmark_reductions_threads sum float32
./build/bench/benchmark_reductions_threads sum float32 multi
```

### benchmark_blocksize.c
Effect of evaluation block size on performance.

```bash
./build/bench/benchmark_blocksize
```

### benchmark_blocksize_threads.c
Multi-threaded block size impact.

```bash
./build/bench/benchmark_blocksize_threads
```

### benchmark_chunked.c
Chunked evaluation throughput.

```bash
./build/bench/benchmark_chunked
```

### benchmark_chunksize.c
Sensitivity to chunk size selection.

```bash
./build/bench/benchmark_chunksize
```

### benchmark_comparisons.c
Comparison operator throughput.

```bash
./build/bench/benchmark_comparisons
```

### benchmark_logical_bool.c
Logical operator throughput for boolean arrays.

```bash
./build/bench/benchmark_logical_bool
```

### benchmark_sincos.c
sin/cos expression throughput across block sizes.

```bash
./build/bench/benchmark_sincos
```

Note: This benchmark uses `sin(a) ** 2 + cos(a) ** 2`. Performance should be similar to
`sin(a) * sin(a) + cos(a) * cos(a)` if sin/cos are not evaluated twice.

### benchmark_sincos_threads.c
Multi-threaded sin/cos expression throughput.

```bash
./build/bench/benchmark_sincos_threads
```

### benchmark_memory_efficiency.c
Working-set size vs throughput and memory behavior.

```bash
./build/bench/benchmark_memory_efficiency
```

### benchmark_mixed_types.c
Mixed-type expressions and type promotion overhead.

```bash
./build/bench/benchmark_mixed_types
```

### benchmark_nd_padding_threads.c
Multi-threaded ND padding scenarios vs pure C (1 GB logical array).

```bash
./build/bench/benchmark_nd_padding_threads
```

### benchmark_threadsafe.c
Thread-safe evaluation overhead vs single-threaded.

```bash
./build/bench/benchmark_threadsafe
```

### benchmark_dsl_jit_mandelbrot.c
DSL Mandelbrot-style benchmark comparing:
- JIT kernel: no `any()` (JIT-friendly loop body).
- Interpreter kernel: uses `any()` early exit.

```bash
./build/bench/benchmark_dsl_jit_mandelbrot
./build/bench/benchmark_dsl_jit_mandelbrot 1024x512 6
./build/bench/benchmark_dsl_jit_mandelbrot 1024x512 6 24
# Alternate form:
./build/bench/benchmark_dsl_jit_mandelbrot 1024 512 6 24
```

### benchmark_mandelbrot_numba.py
Optional Python/Numba baseline for the JIT-style (no `any()`) kernel.
Requires `numpy` and `numba`.

```bash
python bench/benchmark_mandelbrot_numba.py
python bench/benchmark_mandelbrot_numba.py 1024x512 6
python bench/benchmark_mandelbrot_numba.py 1024x512 6 24
# Alternate form:
python bench/benchmark_mandelbrot_numba.py 1024 512 6 24
```
