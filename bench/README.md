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
- Notebook-equivalent Mandelbrot escape-iteration kernel.
- Element-wise loop form (`if ...: break`) in interpreted, JIT cold, and JIT warm modes.
- Compiler backends: `tcc` and `cc`.

Notes:
- `jit-warm` and `interp` rows report the **best single run** over `repeats`.
- `jit-cold` runs once and includes first compile overhead separately in `compile_ms`.
- Use `ME_BENCH_FP_MODE=strict|contract|fast` to emit `# me:fp=...` in benchmark kernels
  (default: `strict`).

```bash
./build/bench/benchmark_dsl_jit_mandelbrot
./build/bench/benchmark_dsl_jit_mandelbrot 1024x512 6
./build/bench/benchmark_dsl_jit_mandelbrot 1024x512 6 24
ME_BENCH_FP_MODE=fast ./build/bench/benchmark_dsl_jit_mandelbrot 1024x512 6 24
# Alternate form:
./build/bench/benchmark_dsl_jit_mandelbrot 1024 512 6 24
```

### benchmark_dsl_jit_math_kernels.c
DSL JIT baseline for representative math kernels:
- `sin`, `exp`, `log`, `pow`, `hypot`, `atan2`, `sinpi`, `cospi`
- Per-kernel metrics:
  - JIT cold compile latency
  - JIT warm throughput
  - interpreter throughput
  - max-abs numerical diff (JIT warm vs interpreter)

Notes:
- Uses `# me:fp=strict`.
- Uses `# me:compiler=tcc` by default (when `ME_BENCH_COMPILER` is unset).
- Set `ME_BENCH_COMPILER=cc` to benchmark the `cc` backend explicitly.

```bash
./build/bench/benchmark_dsl_jit_math_kernels
./build/bench/benchmark_dsl_jit_math_kernels 262144 6
ME_BENCH_COMPILER=tcc ./build/bench/benchmark_dsl_jit_math_kernels 262144 6
ME_BENCH_COMPILER=cc ./build/bench/benchmark_dsl_jit_math_kernels 262144 6
```

### benchmark_dsl_jit_index_vars.c
DSL reserved-index-vars A/B benchmark for:
- `_i0`, `_i1`, `_n0`, `_n1`, `_ndim`, `_global_linear_idx`
- Compares interpreter vs JIT buffer-path vs JIT synthesis-path:
  - `interp`: `ME_DSL_JIT=0`
  - `jit-buffer`: `ME_DSL_JIT=1`, `ME_DSL_JIT_INDEX_VARS=1`, `ME_DSL_JIT_INDEX_VARS_SYNTH=0`
  - `jit-synth`: `ME_DSL_JIT=1`, `ME_DSL_JIT_INDEX_VARS=1`, `ME_DSL_JIT_INDEX_VARS_SYNTH=1`
  - `jit-gateoff`: `ME_DSL_JIT=1`, `ME_DSL_JIT_INDEX_VARS=0` (control)

Notes:
- Uses `# me:fp=strict`.
- Uses `# me:compiler=tcc` by default when `ME_BENCH_COMPILER` is unset.
- Set `ME_BENCH_COMPILER=cc` to force `cc` backend.

```bash
./build/bench/benchmark_dsl_jit_index_vars
./build/bench/benchmark_dsl_jit_index_vars 1048576 9
ME_BENCH_COMPILER=tcc ./build/bench/benchmark_dsl_jit_index_vars 1048576 9
ME_BENCH_COMPILER=cc ./build/bench/benchmark_dsl_jit_index_vars 1048576 9
```

### benchmark_mandelbrot_numba.py
Optional Python/Numba baseline matching notebook-style escape-iteration output
with regular early escape (`if zr*zr + zi*zi > 4.0: break`).
Requires `numpy` and `numba`.

```bash
python bench/benchmark_mandelbrot_numba.py
python bench/benchmark_mandelbrot_numba.py 1024x512 6
python bench/benchmark_mandelbrot_numba.py 1024x512 6 24
# Alternate form:
python bench/benchmark_mandelbrot_numba.py 1024 512 6 24
```
