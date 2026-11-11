# miniexpr

A small, efficient C library for parsing and evaluating mathematical expressions and portable Menudet kernels over arrays.

## Overview

miniexpr is designed to be embedded directly into larger projects (such as [C-Blosc2](https://github.com/Blosc/c-blosc2)), not distributed as a standalone library.

miniexpr has two evaluation modes:

1. **Expressions**: Simple one-line mathematical formulas (the original TinyExpr heritage), evaluated element-wise:
   ```c
   me_compile("sqrt(x*x + y*y)", vars, 2, ME_FLOAT64, &err, &expr);
   ```

2. **Menudet Kernels**: Multi-statement Python-like computational programs with `def`, temporaries, conditionals, loops, and reductions:
   ```python
   def kernel(x, y):
       t = sin(x) ** 2
       if y > 0:
           return t + cos(y) ** 2
       return t
   ```

### Key Capabilities

- **Mathematical operations & functions**: Standard arithmetic, trigonometry, hyperbolic, exponential, and special functions.
- **Rich type system**: Signed/unsigned integers (`int8` through `uint64`), floating-point (`float32`, `float64`), C99 complex numbers (`complex64`, `complex128`), strings (`ME_STRING` / UCS-4), and bytes (`ME_BYTES`).
- **Chunked & N-dimensional evaluation**: Efficient vectorized evaluation over chunked array blocks, with both fixed-width and Arrow-compatible variable-length output (`me_eval_varlen`).
- **Runtime JIT compilation**: Optional JIT compilation via bundled in-memory TinyCC or system C compilers for near-native execution speed.
- **Thread-safe & re-entrant**: Compiled expressions and kernels are evaluated concurrently across threads on separate data buffers with no internal locks.

**Windows**: Building requires clang-cl; `complex64` and `complex128` are supported under portable interpretation. JIT complex kernels remain unsupported across all platforms.

---

## Quick Example

```c
#include "miniexpr.h"
#include <stdio.h>

int main(void) {
    // 1. Define variables
    me_variable vars[] = {{"x"}, {"y"}};
    int err;

    // 2. Compile expression
    me_expr *expr = NULL;
    if (me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        fprintf(stderr, "Compilation error: %s\n", me_get_last_error_message());
        return -1;
    }

    // 3. Prepare data buffers
    double x_data[] = {1.0, 2.0, 3.0};
    double y_data[] = {4.0, 5.0, 6.0};
    double result[3];
    const void *ptrs[] = {x_data, y_data};

    // 4. Evaluate (thread-safe)
    if (me_eval(expr, ptrs, 2, result, 3, NULL) != ME_EVAL_SUCCESS) {
        fprintf(stderr, "Evaluation failed\n");
    }

    // 5. Clean up
    me_free(expr);
    return 0;
}
```

---

## API Reference

### Compilation

```c
int me_compile(const char *expression, const me_variable *variables,
               int var_count, me_dtype dtype, int *error, me_expr **out);

int me_compile_nd(const char *expression, const me_variable *variables,
                  int var_count, me_dtype dtype, int ndims,
                  const int64_t *shape, const int32_t *chunkshape,
                  const int32_t *blockshape, int *error, me_expr **out);

int me_compile_nd_jit(const char *expression, const me_variable *variables,
                      int var_count, me_dtype dtype, int ndims,
                      const int64_t *shape, const int32_t *chunkshape,
                      const int32_t *blockshape, int jit_mode,
                      int *error, me_expr **out);

int me_compile_nd_jit_options(const char *expression, const me_variable *variables,
                              int var_count, me_dtype dtype, int ndims,
                              const int64_t *shape, const int32_t *chunkshape,
                              const int32_t *blockshape, int jit_mode,
                              const me_jit_options *options, int *error, me_expr **out);
```

- Compiles an expression or Menudet kernel for evaluation. Variable and output memory pointers are bound during evaluation, not compilation.
- `jit_mode` controls compile-time runtime JIT preparation:
  - `ME_JIT_DEFAULT` (`0`): Default policy.
  - `ME_JIT_ON` (`1`): Prefer runtime JIT preparation.
  - `ME_JIT_OFF` (`2`): Skip runtime JIT preparation.
  - `ME_JIT_AUTO` (`3`): Interpret simple single-expression paths and prefer JIT for multi-statement programs.
- `me_compile_nd_jit_options()` accepts per-call compilation settings (`compiler`, `cflags`, `cache_dir`, `trace`, `math_backend`) without altering environment variables.

### Evaluation

```c
int me_eval(const me_expr *expr, const void **vars_block,
            int n_vars, void *output_block, int block_nitems,
            const me_eval_params *params);

int me_eval_nd(const me_expr *expr, const void **vars_block,
               int n_vars, void *output_block, int block_nitems,
               int64_t nchunk, int64_t nblock, const me_eval_params *params);

int me_nd_valid_nitems(const me_expr *expr, int64_t nchunk, int64_t nblock, int64_t *valid_nitems);

size_t me_varlen_data_bound(const me_expr *expr, int block_nitems);

int me_eval_varlen(const me_expr *expr, const void **vars_block, int n_vars,
                   int block_nitems, int64_t *offsets,
                   void *data, size_t data_capacity, size_t *data_used,
                   const me_eval_params *params);
```

- Evaluates compiled expressions over chunk buffers. Safe for concurrent evaluation across threads.
- For string (`ME_STRING`) or bytes (`ME_BYTES`) outputs, `me_eval_varlen()` writes directly into an Arrow-compatible layout (`int64_t` offsets array + compact data blob), avoiding fixed-width worst-case padding. Size the buffer using `me_varlen_data_bound()`.
- Use `ME_EVAL_PARAMS_DEFAULTS` to customize evaluation behavior (e.g. `params.disable_simd = true` or `params.jit_mode = ME_JIT_OFF`).

### Diagnostics and Cleanup

```c
const char *me_get_last_error_message(void);
void me_free(me_expr *n);
```

- `me_get_last_error_message()`: Returns a thread-local, human-readable message for the most recent compile or setup failure.
- `me_free()`: Safely frees expression resources (safe to call on `NULL`).

---

## Menudet Kernels

miniexpr executes computational kernels written in **Menudet**, a versioned little language for portable computation on arrays and tables:

- **Language features**: Temporary variables, `if`/`elif`/`else` conditionals, `for`/`while` loops with `break`/`continue`/`pass`, chained comparisons (`0 < x <= 10`), Python numeric literals (`0x...`, `0b...`, `1_000_000`), and block reductions (`block_sum`, `block_min`, etc.).
- **Frozen 0.1 Beta**: Menudet 0.1 is feature-frozen with strictly defined arithmetic (exact two's complement wrapping, defined shift behavior, and division-by-zero protection matching NumPy 2.x).
- **Validation**: Call `me_validate_menudet(source, ME_MENUDET_DSL_VERSION, ...)` to preflight and validate kernels before compilation.

See **[doc/menudet-intro.md](doc/menudet-intro.md)** for a complete overview of Menudet portability, arithmetic edge-case guarantees, and examples.

---

## Runtime JIT Compilation

Menudet kernels can be compiled to native code on the fly:

- **Bundled TinyCC (Default)**: Compiles completely in memory with no disk cache. On Linux, it uses anonymous `memfd` RW/RX memory mappings with no external toolchain dependencies.
- **System CC Backend**: Uses an external compiler (GCC/Clang/MSVC) to generate optimized shared libraries with persistent disk caching.
- **Source Pragmas**: Control compilation per-kernel via header pragmas:
  ```python
  # me:compiler=tcc|cc
  # me:fp=strict|contract|fast
  # me:math=auto|system|sleef|accelerate
  # me:ulp=u10|u35
  def kernel(x):
      return sqrt(x) + 1.0
  ```
- **Environment Overrides**: `ME_DSL_JIT=0` (force interpreter), `ME_DSL_TRACE=1` (diagnostic tracing), `ME_SIMD_MATH_BACKEND=...` (select math acceleration library).

For complete JIT configuration details and tuning knobs, see [doc/dsl-usage.md](doc/dsl-usage.md) and [README_DEVELOPERS.md](README_DEVELOPERS.md).

---

## Documentation

- **Getting Started & Tutorials**:
  - **[doc/get-started.md](doc/get-started.md)**: Step-by-step introduction and walkthrough.
  - **[examples/README.md](examples/README.md)**: Descriptions of complete, runnable C examples.
- **Menudet Language**:
  - **[doc/menudet-intro.md](doc/menudet-intro.md)**: **Recommended overview** of Menudet, portability dimensions, and arithmetic guarantees.
  - **[doc/dsl-usage.md](doc/dsl-usage.md)**: Practical programming guide, JIT controls, and UDF integration.
  - **[doc/dsl-syntax.md](doc/dsl-syntax.md)**: Canonical language grammar, statements, and pragmas reference.
  - **[doc/dsl-spec/0.1.md](doc/dsl-spec/0.1.md)**: Formal Menudet 0.1 language specification.
  - **[doc/dsl-spec/artifact-0.1.md](doc/dsl-spec/artifact-0.1.md)**: Computational artifact schema and requirements.
- **Types, Strings & Arrays**:
  - **[doc/data-types.md](doc/data-types.md)**: Data types reference and Windows notes.
  - **[doc/type-inference.md](doc/type-inference.md)**: Type inference and NumPy promotion rules.
  - **[doc/strings.md](doc/strings.md)**: Fixed-width strings (`ME_STRING`), bytes (`ME_BYTES`), and Arrow varlen output.
  - **[doc/chunk-processing.md](doc/chunk-processing.md)**: Chunked and blocked memory layout evaluation.
  - **[doc/parallel-processing.md](doc/parallel-processing.md)**: Thread-safe multi-threaded evaluation patterns.
- **NumPy Conformance**:
  - **[doc/numpy-functions-0.1.md](doc/numpy-functions-0.1.md)**: Numerical function catalog and status rules.
  - **[doc/numpy-arithmetic-0.1.md](doc/numpy-arithmetic-0.1.md)**: Arithmetic, promotion, and casting reference.
  - **[doc/numpy-compat-checkpoint.md](doc/numpy-compat-checkpoint.md)**: Conformance corpus and verification.
- **Native Graphs & Execution**:
  - **[doc/native-graphs-1.md](doc/native-graphs-1.md)**: Materialized graph execution contract.
  - **[doc/native-execution.md](doc/native-execution.md)**: Host executors and partitioning engine.
- **Developer & Contributor Notes**:
  - **[README_DEVELOPERS.md](README_DEVELOPERS.md)**: Build options, internal tuning knobs, and contributor guidelines.

---

## Building

Build with CMake and Ninja:

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build
```

See [README_DEVELOPERS.md](README_DEVELOPERS.md) for build variants, CMake flags, SLEEF SIMD options, and platform-specific invocations.

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

Third-party components:
- **tinyexpr**: Zlib license (see [LICENSE-TINYEXPR](LICENSE-TINYEXPR)).
- **SLEEF**: Boost Software License 1.0 (see [LICENSE-SLEEF](LICENSE-SLEEF)).
- **libtcc (TinyCC)**: LGPL-2.1-or-later (see [LICENSE-LIBTCC](LICENSE-LIBTCC)).

Copyright (c) 2025-2026, The Blosc Development Team.
