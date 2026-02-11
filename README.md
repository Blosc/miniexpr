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

## Versioning

miniexpr follows semantic versioning. The library version is available at compile time via `ME_VERSION_*` macros and at runtime via `me_version()`.

## API Functions

miniexpr provides a simple, focused API with just two main functions:

### `me_compile()` and `me_compile_nd()`
```c
int me_compile(const char *expression, const me_variable *variables,
               int var_count, me_dtype dtype, int *error, me_expr **out);

int me_compile_nd(const char *expression, const me_variable *variables,
                  int var_count, me_dtype dtype, int ndims,
                  const int64_t *shape, const int32_t *chunkshape,
                  const int32_t *blockshape, int *error, me_expr **out);
```
Compiles an expression for evaluation. Variable and output pointers are provided during evaluation rather than compilation.

**Simple Usage**: Just provide variable names - everything else is optional:

```c
me_variable vars[] = {{"x"}, {"y"}};  // Just the names!
me_expr *expr = NULL;
if (me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Later, provide data in the same order as vars array
const void *data[] = {x_array, y_array};  // x first, y second
if (me_eval(expr, data, 2, output, nitems, NULL) != ME_EVAL_SUCCESS) { /* handle error */ }
```

For mixed types (use `ME_AUTO` for output dtype to infer from variables):
```c
me_variable vars[] = {{"temp", ME_FLOAT64}, {"count", ME_INT32}};
me_expr *expr = NULL;
if (me_compile("temp * count", vars, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Result type will be inferred (ME_FLOAT64 in this case)
```

Variables are matched by position (order) in the arrays. Unspecified fields default to NULL/0.

#### `dtype` Parameter Rules

The `dtype` parameter has two mutually exclusive modes:

1. **Uniform Type** (Simple): Set `dtype` to a specific type (e.g., `ME_FLOAT64`)
   - All variables must be `ME_AUTO`
   - All data uses the specified type

2. **Mixed Types** (Advanced): Set `dtype` to `ME_AUTO`
   - All variables must have explicit types
   - Result type is inferred from type promotion rules
   - Check `expr->dtype` for the inferred type

Mixing modes (some vars with types, some `ME_AUTO`) will cause compilation to fail.

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
Evaluates the compiled expression with new variable and output pointers. This allows processing arrays in chunks without recompilation, and is thread-safe for parallel evaluation across multiple threads.

**Parameters:**
- `expr`: Compiled expression (from `me_compile`)
- `vars_block`: Array of pointers to variable block buffers (same order as in `me_compile`)
- `n_vars`: Number of variables (must match the number used in `me_compile`)
- `output_block`: Pointer to output buffer for this block
- `block_nitems`: Number of elements in this block (padded size for `me_eval_nd`)
- `params`: Optional SIMD evaluation settings (`NULL` for defaults)
- Return value: `ME_EVAL_SUCCESS` (0) on success, or a negative `ME_EVAL_ERR_*` code on failure

Use `ME_EVAL_PARAMS_DEFAULTS` to start from defaults and override only what you need:
```c
me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
params.disable_simd = true;
if (me_eval(expr, var_ptrs, 2, result, 3, &params) != ME_EVAL_SUCCESS) { /* handle error */ }
```

**ND/padded blocks (b2nd-style):**
- `shape`, `chunkshape`, `blockshape` are C-order arrays (`ndims` length).
- `nchunk` is the chunk index over the full array (C-order); `nblock` is the block index inside that chunk (C-order).
- Callers pass padded blocks of size `prod(blockshape)`. `me_eval_nd` evaluates only valid elements and zero-fills the padded tail in the output. Use `me_nd_valid_nitems` to know how many outputs are real for a given `(nchunk, nblock)`.
- For expressions whose overall result is a scalar (e.g., `sum(x)` or `sum(x) + 1`), `output_block` only needs space for one item. In this case `me_eval_nd` writes a single element and does not zero any tail.

See `examples/11_nd_padding_example.c` and `doc/chunk-processing.md` for a walkthrough, and `bench/benchmark_nd_padding` to gauge performance with different padding patterns.

### `me_free()`
```c
void me_free(me_expr *n);
```
Frees the compiled expression. Safe to call on `NULL` pointers.

## Data Types

miniexpr supports various data types through the `me_dtype` enumeration:
- Booleans: `ME_BOOL`
- Signed integers: `ME_INT8`, `ME_INT16`, `ME_INT32`, `ME_INT64`
- Unsigned integers: `ME_UINT8`, `ME_UINT16`, `ME_UINT32`, `ME_UINT64`
- Floating point: `ME_FLOAT32`, `ME_FLOAT64`
- Complex numbers: `ME_COMPLEX64`, `ME_COMPLEX128`

## Reductions

miniexpr provides scalar reductions over a single variable or constant:
- `sum(x)`, `prod(x)` (sum/product)
- `min(x)`, `max(x)`
- `any(x)`, `all(x)` (truthiness over nonzero values)

**Rules:**
- The argument may be any expression that does not itself contain reductions (e.g., `sum(x + 1)` is valid, `sum(sum(x))` is not).
- Reductions may appear inside larger expressions; their scalar result is broadcast (e.g., `x + sum(x)` is valid).

**Result types:**
- `sum`/`prod`: integer inputs promote to 64-bit (`int64`/`uint64`); floats keep their type.
- `min`/`max`: same dtype as the input.
- `any`/`all`: `bool` output for any input type (nonzero is true).
- Note: `sum`/`prod` on `float32` inputs accumulate in `float64` and cast back to `float32`.

**Floating-point NaNs:**
- `min`/`max` propagate NaNs if any element is NaN.

## Usage

To use miniexpr in your project, simply include the source files (`miniexpr.c` and `miniexpr.h`) directly in your build system.

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
Tip: run a subset of tests with `ctest --test-dir build -R <pattern>`.

Options:
- `-DMINIEXPR_BUILD_TESTS=ON|OFF`
- `-DMINIEXPR_BUILD_EXAMPLES=ON|OFF`
- `-DMINIEXPR_BUILD_BENCH=ON|OFF`
- `-DMINIEXPR_USE_SLEEF=ON|OFF`
- `-DMINIEXPR_USE_LIBTCC_FALLBACK=ON|OFF`
- `-DMINIEXPR_DSL_TRACE_DEFAULT=ON|OFF` (emit DSL trace logs by default when `ME_DSL_TRACE` is unset)
- `-DMINIEXPR_ENABLE_WINDOWS_LIBTCC_JIT=ON|OFF` (Windows-only, experimental)

### SLEEF SIMD acceleration

miniexpr can use SLEEF to accelerate transcendentals and other math kernels via SIMD. The CMake option `-DMINIEXPR_USE_SLEEF=ON` (default) fetches SLEEF and enables the SIMD paths; set it to `OFF` to build without SLEEF. At runtime you can force scalar evaluation by setting `me_eval_params.disable_simd = true`, which disables SIMD regardless of whether SLEEF was compiled in.

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
- **Index access**: Built-in `_i0`–`_i7` (position) and `_n0`–`_n7` (shape) variables
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
- `CFLAGS="..."`: Standard C compiler flags honored by the `cc` backend runtime JIT path.

Backend selection is done in DSL source via pragma:

- `# me:compiler=tcc` (default when omitted)
- `# me:compiler=cc`
- For `# me:compiler=cc`, runtime uses `CC` (default: `cc`) as the compiler executable.

Operational `libtcc` runtime location/options remain configurable:

- `ME_DSL_JIT_LIBTCC_PATH=/path/to/libtcc.{so,dylib}`: Override runtime library path used for `libtcc` loading.
- `ME_DSL_JIT_TCC_LIB_PATH=/path/to/tcc/libdir`: Override `tcc_set_lib_path()` directory (mainly for custom/legacy `libtcc` layouts).
- `ME_DSL_JIT_TCC_OPTIONS="..."`: Extra options passed to `tcc_set_options()`.

Build-time note:

- On Linux/macOS, libtcc support is built by default and required for DSL JIT's default
  `# me:compiler=tcc` mode.
- CMake uses local MiniCC sources at `../minicc` and builds `libtcc` as a separate shared library.
  `libtcc` is staged in the same directory as `libminiexpr`, and miniexpr embeds that staged
  `libtcc` path as a default runtime lookup candidate.

## Contributing

After cloning the repository, install the Git hooks:

```bash
./scripts/install-hooks.sh
```

This sets up automatic checks for code quality (e.g., trailing whitespace).

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
