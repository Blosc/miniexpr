# miniexpr

A small, efficient C library for parsing and evaluating mathematical expressions with support for vectorized operations across multiple data types. Derived from [tinyexpr](https://github.com/codeplea/tinyexpr).

## Overview

miniexpr is designed to be embedded directly into larger projects, not distributed as a standalone library. It provides fast expression evaluation with support for:

- Standard mathematical operations and functions
- Multiple numeric data types (integers, floats, complex numbers)
- Vectorized evaluation for processing arrays efficiently
- Thread-safe operations for parallel processing

## Main Functions

### `me_compile()`
```c
me_expr *me_compile(const char *expression, const me_variable *variables,
                    int var_count, void *output, int nitems,
                    me_dtype dtype, int *error);
```
Parses an expression string and creates a compiled expression tree. Variables are bound at compile time. Returns `NULL` on error.

### `me_compile_chunk()`
```c
me_expr *me_compile_chunk(const char *expression, const me_variable *variables,
                          int var_count, me_dtype dtype, int *error);
```
Compiles an expression for chunked evaluation. This variant is optimized for use with `me_eval_chunk()` and `me_eval_chunk_threadsafe()`, where variable and output pointers are provided during evaluation rather than compilation.

**Simple Usage**: Just provide variable names - everything else is optional:

```c
me_variable vars[] = {{"x"}, {"y"}};  // Just the names!
me_expr *expr = me_compile_chunk("x + y", vars, 2, ME_FLOAT64, &err);

// Later, provide data in the same order as vars array
const void *data[] = {x_array, y_array};  // x first, y second
me_eval_chunk(expr, data, 2, output, nitems);
```

For mixed types:
```c
me_variable vars[] = {{"temp", ME_FLOAT64}, {"count", ME_INT32}};
```

Variables are matched by position (order) in the arrays. Unspecified fields (address, type, context) default to NULL/0.

Returns `NULL` on error.

### `me_eval()`
```c
void me_eval(const me_expr *n);
```
Evaluates the compiled expression on vectors. Results are written to the output buffer specified during compilation.

### `me_eval_fused()`
```c
void me_eval_fused(const me_expr *n);
```
Evaluates using fused bytecode for faster execution on complex expressions.

### `me_eval_chunk()`
```c
void me_eval_chunk(const me_expr *expr, const void **vars_chunk, int n_vars,
                   void *output_chunk, int chunk_nitems);
```
Evaluates a compiled expression with new variable and output pointers, allowing processing of large arrays in chunks without recompilation. **Not thread-safe**.

### `me_eval_chunk_threadsafe()`
```c
void me_eval_chunk_threadsafe(const me_expr *expr, const void **vars_chunk,
                               int n_vars, void *output_chunk, int chunk_nitems);
```
Thread-safe version of `me_eval_chunk()` for parallel evaluation across multiple threads.

### `me_free()`
```c
void me_free(me_expr *n);
```
Frees the compiled expression. Safe to call on `NULL` pointers.

### `me_print()`
```c
void me_print(const me_expr *n);
```
Prints debugging information about the syntax tree.

## Data Types

miniexpr supports various data types through the `me_dtype` enumeration:
- Booleans: `ME_BOOL`
- Signed integers: `ME_INT8`, `ME_INT16`, `ME_INT32`, `ME_INT64`
- Unsigned integers: `ME_UINT8`, `ME_UINT16`, `ME_UINT32`, `ME_UINT64`
- Floating point: `ME_FLOAT32`, `ME_FLOAT64`
- Complex numbers: `ME_COMPLEX64`, `ME_COMPLEX128`

## Usage

To use miniexpr in your project, simply include the source files (`miniexpr.c` and `miniexpr.h`) directly in your build system.

For examples and detailed usage, see the [Getting Started Guide](doc/get-started.md).

## Contributing

After cloning the repository, install the Git hooks:

```bash
./scripts/install-hooks.sh
```

This sets up automatic checks for code quality (e.g., trailing whitespace).

See [CODE_QUALITY.md](CODE_QUALITY.md) for more details on code quality tools.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

Copyright (c) 2021-2025, The Blosc Development Team

## Acknowledgments

Based on [tinyexpr](https://github.com/codeplea/tinyexpr) by Lewis Van Winkle. See [LICENSE-TINYEXPR](LICENSE-TINYEXPR) for the original license.
