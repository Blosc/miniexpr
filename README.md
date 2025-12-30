# miniexpr

A small, efficient C library for parsing and evaluating mathematical expressions with support for vectorized operations across multiple data types. Derived from [tinyexpr](https://github.com/codeplea/tinyexpr).

## Overview

miniexpr is designed to be embedded directly into larger projects, not distributed as a standalone library. It provides fast expression evaluation with support for:

- Standard mathematical operations and functions
- Multiple numeric data types (integers, floats, complex numbers)
- Vectorized evaluation for processing arrays efficiently
- Thread-safe operations for parallel processing

**Note**: This is a pre-alpha project.
**Windows note**: Complex types are not supported on Windows because the C99 complex ABI is not stable across MSVC/clang-cl. `me_compile()` will reject expressions that involve complex variables or outputs.

## API Functions

miniexpr provides a simple, focused API with just two main functions:

### `me_compile()`
```c
me_expr *me_compile(const char *expression, const me_variable *variables,
                    int var_count, me_dtype dtype, int *error);
```
Compiles an expression for evaluation. Variable and output pointers are provided during evaluation rather than compilation.

**Simple Usage**: Just provide variable names - everything else is optional:

```c
me_variable vars[] = {{"x"}, {"y"}};  // Just the names!
me_expr *expr = me_compile("x + y", vars, 2, ME_FLOAT64, &err);

// Later, provide data in the same order as vars array
const void *data[] = {x_array, y_array};  // x first, y second
if (me_eval(expr, data, 2, output, nitems) != ME_EVAL_SUCCESS) { /* handle error */ }
```

For mixed types (use `ME_AUTO` for output dtype to infer from variables):
```c
me_variable vars[] = {{"temp", ME_FLOAT64}, {"count", ME_INT32}};
me_expr *expr = me_compile("temp * count", vars, 2, ME_AUTO, &err);
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

### `me_eval()`
```c
int me_eval(const me_expr *expr, const void **vars_chunk,
            int n_vars, void *output_chunk, int chunk_nitems);
```
Evaluates the compiled expression with new variable and output pointers. This allows processing arrays in chunks without recompilation, and is thread-safe for parallel evaluation across multiple threads.

**Parameters:**
- `expr`: Compiled expression (from `me_compile`)
- `vars_chunk`: Array of pointers to variable data chunks (same order as in `me_compile`)
- `n_vars`: Number of variables (must match the number used in `me_compile`)
- `output_chunk`: Pointer to output buffer for this chunk
- `chunk_nitems`: Number of elements in this chunk
- Return value: `ME_EVAL_SUCCESS` (0) on success, or a negative `ME_EVAL_ERR_*` code on failure

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

## Usage

To use miniexpr in your project, simply include the source files (`miniexpr.c` and `miniexpr.h`) directly in your build system.

### Quick Example

```c
#include "miniexpr.h"

// Define variables
me_variable vars[] = {{"x"}, {"y"}};
int err;

// Compile expression
me_expr *expr = me_compile("x + y", vars, 2, ME_FLOAT64, &err);

// Prepare data
double x_data[] = {1.0, 2.0, 3.0};
double y_data[] = {4.0, 5.0, 6.0};
double result[3];

const void *var_ptrs[] = {x_data, y_data};

// Evaluate (thread-safe)
if (me_eval(expr, var_ptrs, 2, result, 3) != ME_EVAL_SUCCESS) { /* handle error */ }
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
make examples
./build/01_simple_expression
```

See [examples/README.md](examples/README.md) for detailed documentation of each example.

## Documentation

- **[examples/README.md](examples/README.md)** - Complete examples with explanations
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Complete API reference
- **[doc/get-started.md](doc/get-started.md)** - Getting started guide
- **[doc/data-types.md](doc/data-types.md)** - Data types guide
- **[doc/parallel-processing.md](doc/parallel-processing.md)** - Parallel processing patterns

## Contributing

After cloning the repository, install the Git hooks:

```bash
./scripts/install-hooks.sh
```

This sets up automatic checks for code quality (e.g., trailing whitespace).

See [CODE_QUALITY.md](CODE_QUALITY.md) for more details on code quality tools.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

Copyright (c) 2025-2025, The Blosc Development Team

## Acknowledgments

Based on [tinyexpr](https://github.com/codeplea/tinyexpr) by Lewis Van Winkle. See [LICENSE-TINYEXPR](LICENSE-TINYEXPR) for the original license.
