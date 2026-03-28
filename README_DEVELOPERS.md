# Developer Notes

This document collects developer-facing build and runtime details that are not part of the regular user-facing interface.

## Versioning

miniexpr follows semantic versioning. The library version is available at compile time via `ME_VERSION_*` macros and at runtime via `me_version()`.

## API Notes

The primary user-facing API reference lives in [README.md](README.md) and [src/miniexpr.h](src/miniexpr.h). This section keeps only the more detailed semantic notes.

### `dtype` Parameter Rules

The `dtype` parameter has two mutually exclusive modes:

1. **Uniform Type**: Set `dtype` to a specific type such as `ME_FLOAT64`.
   All variables must be `ME_AUTO`.
2. **Mixed Types**: Set `dtype` to `ME_AUTO`.
   All variables must have explicit types and the result type is inferred.

Mixing modes causes compilation to fail.

`me_compile_nd_jit(...)` adds a compile-time JIT policy hint:

- `jit_mode = 0` (`ME_JIT_DEFAULT`): default behavior.
- `jit_mode = 1` (`ME_JIT_ON`): prefer runtime JIT preparation.
- `jit_mode = 2` (`ME_JIT_OFF`): skip runtime JIT preparation at compile time.

### ND/Padded Block Notes

- `shape`, `chunkshape`, and `blockshape` are C-order arrays.
- `nchunk` is the chunk index over the full array and `nblock` is the block index inside that chunk.
- `me_eval_nd` evaluates only valid elements and zero-fills the padded tail in non-scalar outputs.
- `me_nd_valid_nitems` reports how many outputs are real for a given `(nchunk, nblock)`.

## Data Types

miniexpr supports these `me_dtype` values:

- `ME_BOOL`
- `ME_INT8`, `ME_INT16`, `ME_INT32`, `ME_INT64`
- `ME_UINT8`, `ME_UINT16`, `ME_UINT32`, `ME_UINT64`
- `ME_FLOAT32`, `ME_FLOAT64`
- `ME_COMPLEX64`, `ME_COMPLEX128`

## Reductions

miniexpr provides scalar reductions over a single variable or constant:

- `sum(x)`, `prod(x)`
- `min(x)`, `max(x)`
- `any(x)`, `all(x)`

Rules:

- The argument may be any expression that does not itself contain reductions.
- Reductions may appear inside larger expressions and their scalar result is broadcast.

Result types:

- `sum`/`prod`: integer inputs promote to 64-bit; floats keep their type.
- `min`/`max`: same dtype as the input.
- `any`/`all`: `bool` output for any input type.
- `sum`/`prod` on `float32` inputs accumulate in `float64` and cast back to `float32`.

Floating-point NaNs:

- `min`/`max` propagate NaNs if any element is NaN.

## Building

The main [README.md](README.md) keeps the simplest supported build path. This section collects build variants and developer-oriented options.

### CMake Options

- `-DMINIEXPR_BUILD_SHARED=ON|OFF`
- `-DMINIEXPR_BUILD_STATIC=ON|OFF`
- `-DMINIEXPR_BUILD_TESTS=ON|OFF`
- `-DMINIEXPR_BUILD_EXAMPLES=ON|OFF`
- `-DMINIEXPR_BUILD_BENCH=ON|OFF`
- `-DMINIEXPR_USE_SLEEF=ON|OFF`
- `-DMINIEXPR_USE_ACCELERATE=ON|OFF` (macOS only)
- `-DMINIEXPR_ENABLE_TCC_JIT=ON|OFF`
- `-DMINIEXPR_BUILD_BUNDLED_LIBTCC=ON|OFF` (build bundled libtcc from minicc when TCC JIT is enabled)
- `-DMINIEXPR_DSL_TRACE_DEFAULT=ON|OFF` (emit DSL trace logs by default when `ME_DSL_TRACE` is unset)
- `-DMINIEXPR_WASM32_SIDE_MODULE=ON|OFF` (Emscripten-only side-module helper mode for wasm32 JIT; defaults to `ON` under Emscripten and `OFF` elsewhere)

### Build Notes

- On Windows, TCC JIT is enabled by default (`MINIEXPR_ENABLE_TCC_JIT=ON`).
- On Emscripten, setting `MINIEXPR_ENABLE_TCC_JIT=ON` enables wasm32 JIT support automatically.
- Setting `MINIEXPR_ENABLE_TCC_JIT=OFF` disables TCC-based JIT backends; on Linux/macOS, the separate `# me:compiler=cc` runtime path may still be available.
- `MINIEXPR_USE_SLEEF=ON` fetches SLEEF and enables SIMD math acceleration; set it to `OFF` to build without SLEEF.
- `MINIEXPR_USE_ACCELERATE=ON` enables the macOS Accelerate/vForce backend; in `auto` mode on macOS, SLEEF is preferred when available and Accelerate remains available as a fallback backend.
- When `ME_SIMD_MATH_BACKEND=accelerate` is active, the `ME_SIMD_ULP_1` / `ME_SIMD_ULP_3_5` distinction does not select different kernels. Those accuracy modes remain meaningful for the SLEEF backend.

### Alternative Build Invocations

Windows (clang-cl):

```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -T ClangCL
cmake --build .
```

Makefile fallback:

```bash
make lib
make examples
```

Tip: run a subset of tests with `ctest --test-dir build -R <pattern>`.

## DSL Runtime JIT Internals

The public/runtime-stable DSL JIT controls remain documented in [README.md](README.md). This section covers internal, test-oriented, or tuning-oriented knobs that are still useful during development, debugging, and benchmarking.

### Internal/Test-Only Environment Variables

- `ME_SIMD_MATH_BACKEND=auto|sleef|accelerate|scalar`: Force the SIMD math backend selection used by `src/functions-simd.c` for benchmarking and debugging. Default: `auto` (prefers SLEEF when available, otherwise falls back to Accelerate on macOS when enabled, then the existing scalar fallback).
- The SIMD math benchmarks print backend-aware columns. For `accelerate` and `scalar`, do not interpret the `ME_SIMD_ULP_1` / `ME_SIMD_ULP_3_5` labels as distinct math implementations.
- `ME_DSL_WHILE_MAX_ITERS=<n>`: Override the runtime safety cap for DSL `while` loops.
- `ME_DSL_JIT_COMPILER=tcc|cc`: Force the DSL JIT compiler backend at compile time. When unset, the parsed DSL compiler selection is used.
- `ME_DSL_JIT_MATH_BRIDGE=0|1`: Enable or disable runtime math-bridge lowering globally. Default: `1`.
- `ME_DSL_JIT_SCALAR_MATH_BRIDGE=0|1`: Enable scalar math-bridge lowering for the `cc` backend. Default: `0`.
- `ME_DSL_JIT_VEC_MATH=0|1`: Enable vector math lowering where available. Default: `1`.
- `ME_DSL_JIT_HYBRID_EXPR_VEC_MATH=0|1`: Enable vector lowering for hybrid-expression paths. Default: `0`.
- `ME_DSL_JIT_BRANCH_AWARE_IF=0|1`: Enable branch-aware `if` lowering optimizations. Default: `1`.
- `ME_DSL_JIT_VEC_CHUNK_ITEMS=<n>`: Maximum item count per vector-math bridge chunk. Default: `1048576`.
- `ME_DSL_JIT_DEBUG_CC=0|1`: When enabled, print `cc` backend compiler output during runtime JIT compilation.
- `ME_DSL_JIT_TEST_STUB_SO=/path/to/stub.so`: Test hook used by the runtime-cache test suite.

### Notes

- These knobs are meaningful for development and benchmarking, but they are not treated as regular user-facing interface.
- Defaults are derived from the current implementation in `src/dsl_config.h`, `src/dsl_jit_backend_cc.c`, and `src/dsl_jit_runtime_host.c`.

### Build-Time Notes

- On Linux/macOS, libtcc support is built by default and required for DSL JIT's default `# me:compiler=tcc` mode.
- CMake uses local MiniCC sources at `../minicc` and builds `libtcc` as a separate shared library.
- The staged `libtcc` path is embedded as a default runtime lookup candidate.

## Contributing

After cloning the repository, install the Git hooks:

```bash
./scripts/install-hooks.sh
```

This sets up automatic checks for code quality such as trailing-whitespace detection.
