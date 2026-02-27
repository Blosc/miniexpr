# DSL Kernel Programming Guide

This guide focuses on practical usage of miniexpr DSL kernels.
Canonical syntax, accepted statements, and exact error behavior are documented in `doc/dsl-syntax.md`.

## Overview

Use this guide for:
- End-to-end API usage
- Runtime JIT configuration
- Common kernel-writing patterns
- UDF integration examples

For language syntax details, always refer to `doc/dsl-syntax.md`.

## Quick Start

Minimal compile + eval flow:

```c
const char *src =
    "def kernel(x):\n"
    "    temp = sin(x) ** 2\n"
    "    return temp + cos(x) ** 2\n";

double x[] = {0.0, 0.5, 1.0};
double out[3] = {0.0, 0.0, 0.0};
int err = 0;
me_expr *expr = NULL;
me_variable vars[] = {
    {"x", ME_FLOAT64, NULL, 0, NULL, 0},
};

if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
    const void *inputs[] = {x};
    (void)me_eval(expr, inputs, 1, out, 3, NULL);
    me_free(expr);
}
```

Notes:
- Signature parameter names in the DSL must match input variable names passed to `me_compile()`.
- The order of `vars[]` defines input pointer order for `me_eval()`.
- Full syntax constraints are in `doc/dsl-syntax.md`.

## Runtime JIT Controls

Use source pragmas or API policy to control JIT behavior.

### Source pragmas

```python
# me:fp=strict
# me:compiler=tcc
def kernel(x):
    return x
```

Supported values:
- `# me:fp=strict|contract|fast`
- `# me:compiler=tcc|cc`

Only these pragmas are supported. See `doc/dsl-syntax.md` for exact pragma parsing rules.

### API policy override

```c
int me_compile_nd_jit(const char *expression, const me_variable *variables,
                      int var_count, int type, int32_t storage_dtype,
                      const int32_t *shape, int32_t ndim,
                      const int32_t *blockshape, int jit_mode,
                      te_expr **n);
```

`jit_mode` values:
- `ME_JIT_DEFAULT` (`0`)
- `ME_JIT_ON` (`1`)
- `ME_JIT_OFF` (`2`)

At evaluation time, `me_eval_params.jit_mode` provides the same per-call override.

### Diagnostics

- Set `ME_DSL_TRACE=1` to print compile/JIT trace details to stderr.
- With `# me:compiler=cc`, runtime uses `CC` (default `cc`) and also honors `CFLAGS`.

## Practical Patterns

These patterns are usage-oriented. For strict language rules, see `doc/dsl-syntax.md`.

### Temporary variables and inference

Temporaries are inferred from the right-hand side expression.

```python
def kernel(x):
    temp = sin(x) ** 2
    return temp + cos(x) ** 2
```

If a later assignment gives the same local an incompatible dtype, compilation fails.

### Element-wise selection

Use `where(cond, a, b)` for element-wise branching:

```python
def kernel(x):
    return where(x < 0, 0, where(x > 1, 1, x))
```

### Index-aware kernels

Reserved symbols such as `_i0`, `_n0`, `_ndim`, and `_flat_idx` let you write position-aware kernels.

```python
def kernel(x):
    return x * _i0 / _n0
```

### Loops and control flow

`for` and `while` are supported. `while` loops are protected by an iteration cap (`ME_DSL_WHILE_MAX_ITERS`).

```python
def kernel(n):
    acc = 0
    for i in range(n):
        acc += i
    return acc
```

```python
def kernel(x):
    i = 0
    acc = 0
    while i < x:
        acc += i
        i += 1
    return acc
```

Control-flow semantics are element-wise for non-reduction conditions; reduction conditions (`any`, `all`) are global to active elements.

## User-defined Functions (UDFs)

You can register C functions/closures in `me_variable` and call them from DSL.

Rules:
- Use `ME_FUNCTION*` or `ME_CLOSURE*` in `type`.
- Function return dtype must be explicit (not `ME_AUTO`).
- Function names must not collide with reserved identifiers or built-ins.

Example:

```c
static double clamp01(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

me_variable vars[] = {
    {"x", ME_FLOAT64, x, ME_VARIABLE, NULL, 0},
    {"clamp01", ME_FLOAT64, clamp01, ME_FUNCTION1 | ME_FLAG_PURE, NULL, 0},
};

me_expr *expr = NULL;
int err = 0;
me_compile("def kernel(x):\n    return clamp01(x)\n", vars, 2, ME_FLOAT64, &err, &expr);
```

## More Examples

```python
def poly(a, b, c, d, x):
    t1 = a * x + b
    t2 = t1 * x + c
    return t2 * x + d
```

```python
def softmax(x):
    max_val = max(x)
    shifted = x - max_val
    exp_vals = exp(shifted)
    return exp_vals / sum(exp_vals)
```

```python
def mandelbrot(cr, ci):
    zr = 0.0
    zi = 0.0
    for iter in range(100):
        zr_new = zr * zr - zi * zi + cr
        zi = 2 * zr * zi + ci
        zr = zr_new
        if any(zr * zr + zi * zi > 4.0):
            break
    return iter
```

## Parser API

```c
#include "dsl_parser.h"

const char *source = "def kernel(x):\\n    return x + 1";
me_dsl_error error;
me_dsl_program *prog = me_dsl_parse(source, &error);

if (!prog) {
    fprintf(stderr, "DSL error at %d:%d - %s\n",
            error.line, error.column, error.message);
}

me_dsl_program_free(prog);
```

## Performance Tips

1. Minimize temporaries when possible.
2. Prefer `where()` over branch-heavy logic for SIMD-friendly code paths.
3. Use reductions in stages instead of deeply nested reduction expressions.
4. Keep kernels simple and composable for easier JIT optimization.

## Limitations

For the canonical list, see `doc/dsl-syntax.md`.

Current high-level limits:
- No recursion
- `for` loops use `range(...)`
- Max 8 ND index dimensions (`_i0` to `_i7`)
