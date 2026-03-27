# miniexpr

A small, efficient C library for parsing and evaluating mathematical expressions with support for vectorized operations across multiple data types. Derived from [tinyexpr](https://github.com/codeplea/tinyexpr).

## Overview

miniexpr is designed to be embedded directly into larger projects, not distributed as a standalone library. It provides fast expression evaluation with support for:

- Standard mathematical operations and functions
- Multiple numeric data types (integers, floats, complex numbers)
- Vectorized evaluation for processing arrays efficiently
- Thread-safe operations for parallel processing

**Note**: This is a beta project.
**Windows note**: Complex types are not supported on Windows because the C99 complex ABI is not stable across MSVC/clang-cl. `me_compile()` will reject expressions that involve complex variables or outputs.

## Usage

To use miniexpr in your project, simply include the source files (`miniexpr.c` and `miniexpr.h`) directly in your build system.

For dtype rules, reduction semantics, and versioning details, see [README_DEVELOPERS.md](README_DEVELOPERS.md), [doc/get-started.md](doc/get-started.md), and [src/miniexpr.h](src/miniexpr.h).

## API Reference

miniexpr provides a simple, focused API with just two main functions plus cleanup.

### `me_compile()`, `me_compile_nd()`, and `me_compile_nd_jit()`

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
```

Compiles an expression for evaluation. Variable and output pointers are provided during evaluation rather than compilation.

Variables are matched by position in the arrays. Unspecified fields default to NULL/0.

`me_compile_nd_jit(...)` adds a compile-time JIT policy hint:

- `jit_mode = 0` (`ME_JIT_DEFAULT`): default behavior.
- `jit_mode = 1` (`ME_JIT_ON`): prefer runtime JIT preparation.
- `jit_mode = 2` (`ME_JIT_OFF`): skip runtime JIT preparation at compile time.

### `me_eval()` and `me_eval_nd()`

```c
int me_eval(const me_expr *expr, const void **vars_block,
            int n_vars, void *output_block, int block_nitems,
            const me_eval_params *params);

int me_eval_nd(const me_expr *expr, const void **vars_block,
               int n_vars, void *output_block, int block_nitems,
               int64_t nchunk, int64_t nblock, const me_eval_params *params);

int me_nd_valid_nitems(const me_expr *expr, int64_t nchunk, int64_t nblock, int64_t *valid_nitems);
```

Evaluates the compiled expression with new variable and output pointers. This allows processing arrays in chunks without recompilation and is thread-safe for parallel evaluation.

Use `ME_EVAL_PARAMS_DEFAULTS` to start from defaults and override only what you need:

```c
me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
params.disable_simd = true;
if (me_eval(expr, var_ptrs, 2, result, 3, &params) != ME_EVAL_SUCCESS) { /* handle error */ }
```

JIT policy can also be controlled per-evaluation call:

```c
me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
params.jit_mode = ME_JIT_OFF;
if (me_eval(expr, var_ptrs, 2, result, 3, &params) != ME_EVAL_SUCCESS) { /* handle error */ }
```

### `me_free()`

```c
void me_free(me_expr *n);
```

Frees the compiled expression. Safe to call on `NULL`.

### Quick Example

```c
#include "miniexpr.h"

// Define variables
me_variable vars[] = {{"x"}, {"y"}};
int err;

// Compile expression
me_expr *expr = NULL;
if (me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Prepare data
double x_data[] = {1.0, 2.0, 3.0};
double y_data[] = {4.0, 5.0, 6.0};
double result[3];

const void *var_ptrs[] = {x_data, y_data};

// Evaluate (thread-safe)
if (me_eval(expr, var_ptrs, 2, result, 3, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
// Clean up
me_free(expr);
```

## Examples

The `examples/` directory contains complete, runnable examples demonstrating various features:

- **01_simple_expression.c** - Basic arithmetic expressions (beginner)
- **02_complex_expression.c** - Complex formulas with trigonometry
- **03_mixed_types.c** - Type promotion and ME_AUTO inference
- **04_large_dataset.c** - Processing 1M elements in chunks
- **05_parallel_evaluation.c** - Multi-threaded parallel processing
- **06_debug_print.c** - Expression tree visualization with me_print()

Build and run:
```bash
cmake -S . -B build -G Ninja
cmake --build build -j
./build/examples/01_simple_expression
```

See [examples/README.md](examples/README.md) for detailed documentation of each example.

## Building

CMake + Ninja (recommended):
```bash
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build
```
For build variants, CMake options, SLEEF notes, platform-specific invocations, and contributor workflow, see [README_DEVELOPERS.md](README_DEVELOPERS.md).

## Documentation

- **[examples/README.md](examples/README.md)** - Complete examples with explanations
- **[doc/get-started.md](doc/get-started.md)** - Getting started guide
- **[doc/data-types.md](doc/data-types.md)** - Data types guide
- **[doc/type-inference.md](doc/type-inference.md)** - Type inference rules
- **[doc/parallel-processing.md](doc/parallel-processing.md)** - Parallel processing patterns
- **[doc/dsl-usage.md](doc/dsl-usage.md)** - DSL kernel programming guide
- **[doc/strings.md](doc/strings.md)** - UCS4 string support and string operators

## DSL Kernels

miniexpr includes a DSL (Domain-Specific Language) for writing multi-statement computational kernels. The DSL extends single-expression evaluation with:

- **Temporary variables**: Intermediate results for complex computations
- **Conditionals**: `where(cond, then, else)` for element-wise selection
- **Loops**: `for var in range(limit)` iteration
- **Control flow**: `break` and `continue` statements
- **Index access**: Built-in `_i0`–`_i7` (position), `_n0`–`_n7` (shape), and `_flat_idx` variables
- **Function-style kernels**: `def name(args): ... return expr`

### DSL Example

```c
const char *dsl_source =
    "def kernel(x):\n"
    "    t1 = 1.0 * x - 2.0\n"
    "    t2 = t1 * x + 3.0\n"
    "    return t2 * x - 1.0";

me_dsl_error error;
me_dsl_program *prog = me_dsl_parse(dsl_source, &error);
if (!prog) {
    printf("Parse error: %s\n", error.message);
}
// Compile and evaluate each statement's expression...
me_dsl_program_free(prog);
```

See [doc/dsl-usage.md](doc/dsl-usage.md) for the complete DSL reference and [examples/11_dsl_kernel.c](examples/11_dsl_kernel.c) for a working example.

### DSL Runtime JIT Controls

On Linux/macOS, DSL kernels may use runtime JIT compilation when eligible. The following environment variables control this path:

- `ME_DSL_JIT=0`: Disable runtime JIT and always use interpreter fallback.
- `ME_DSL_JIT_POS_CACHE=0`: Disable process-local positive cache reuse for loaded JIT kernels.
- `ME_DSL_JIT_INDEX_VARS=0`: Disable runtime JIT for DSL kernels that use reserved index vars (`_i*`, `_n*`, `_ndim`, `_flat_idx`).
- `ME_DSL_TRACE=1`: Print DSL compile/JIT trace lines to stderr. When unset, the default comes from `MINIEXPR_DSL_TRACE_DEFAULT`.
- `ME_DSL_FP_MODE=strict|contract|fast|relaxed`: Select default floating-point mode for DSL JIT kernels. `strict` is the default; `relaxed` is an alias of `fast`.
- `ME_DSL_JIT_TCC_OPTIONS="..."`: Extra options passed to `tcc_set_options()` for the `libtcc` backend.
- `CC=...`: Compiler executable used by the `# me:compiler=cc` runtime JIT backend. Defaults to `cc`.
- `CFLAGS="..."`: Standard C compiler flags honored by the `cc` backend runtime JIT path.
- `TMPDIR=...`: Root directory for runtime JIT cache artifacts. When unset, miniexpr uses a per-user directory under `/tmp`.

Per-call policy overrides are available via API:
- `me_eval_params.jit_mode = ME_JIT_ON|ME_JIT_OFF|ME_JIT_DEFAULT`
- `me_compile_nd_jit(..., jit_mode, ...)` to control compile-time runtime-JIT preparation.

When `jit_mode=ME_JIT_OFF` is used in `me_compile_nd_jit(...)`, trace logs include:
- `jit runtime skipped: ... reason=jit_mode=off`
and no runtime JIT backend step (`tcc`/`cc` compile/load) is performed for that compiled expression path.

Backend selection is done in DSL source via pragma:

- `# me:compiler=tcc` (default when omitted)
- `# me:compiler=cc`
- On Linux, the libtcc JIT automatically appends common multiarch library directories
  (for example `/usr/lib/x86_64-linux-gnu`) to help `-lm` resolve without extra flags.

Developer-oriented/internal DSL JIT knobs and build-time notes are documented in [README_DEVELOPERS.md](README_DEVELOPERS.md).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

Third-party licensing:

- tinyexpr portions are under Zlib terms (see [LICENSE-TINYEXPR](LICENSE-TINYEXPR)).
- SLEEF portions are under Boost Software License 1.0 (see [LICENSE-SLEEF](LICENSE-SLEEF)).
- Runtime `libtcc` (TinyCC) is under LGPL-2.1-or-later (see [LICENSE-LIBTCC](LICENSE-LIBTCC)).

For binary installs built with libtcc support, the corresponding TinyCC source and
license are installed under:

- `${CMAKE_INSTALL_DATADIR}/miniexpr/third_party/tinycc/source`
- `${CMAKE_INSTALL_DATADIR}/miniexpr/third_party/tinycc/COPYING`

Copyright (c) 2025-2026, The Blosc Development Team

## Acknowledgments

Based on [tinyexpr](https://github.com/codeplea/tinyexpr) by Lewis Van Winkle. See [LICENSE-TINYEXPR](LICENSE-TINYEXPR) for the original license.
