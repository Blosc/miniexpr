# Benchmarks

This directory contains microbenchmarks for different aspects of miniexpr.

## Quick Start

Build all benchmarks:

```bash
make bench
```

Run a specific benchmark (examples below).

## Benchmarks

### benchmark_reductions.c
Reductions (sum/prod/min/max/any/all) vs pure C.

```bash
./build/benchmark_reductions sum float32
./build/benchmark_reductions sum float32 multi
```

### benchmark_reductions_threads.c
Multi-threaded reductions vs pure C.

```bash
./build/benchmark_reductions_threads sum float32
./build/benchmark_reductions_threads sum float32 multi
```

### benchmark_blocksize.c
Effect of evaluation block size on performance.

```bash
./build/benchmark_blocksize
```

### benchmark_blocksize_threads.c
Multi-threaded block size impact.

```bash
./build/benchmark_blocksize_threads
```

### benchmark_chunked.c
Chunked evaluation throughput.

```bash
./build/benchmark_chunked
```

### benchmark_chunksize.c
Sensitivity to chunk size selection.

```bash
./build/benchmark_chunksize
```

### benchmark_comparisons.c
Comparison operator throughput.

```bash
./build/benchmark_comparisons
```

### benchmark_memory_efficiency.c
Working-set size vs throughput and memory behavior.

```bash
./build/benchmark_memory_efficiency
```

### benchmark_mixed_types.c
Mixed-type expressions and type promotion overhead.

```bash
./build/benchmark_mixed_types
```

### benchmark_threadsafe.c
Thread-safe evaluation overhead vs single-threaded.

```bash
./build/benchmark_threadsafe
```
