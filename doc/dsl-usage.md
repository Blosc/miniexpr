# DSL Kernel Programming Guide

This guide explains how to use the miniexpr Domain-Specific Language (DSL) for writing multi-statement expression kernels.

## Overview

The DSL extends single expressions to full programs with:
- **Temporary variables** - Store intermediate results
- **Multi-statement programs** - Combine multiple computations
- **Element-wise conditionals** - Using `where(cond, true_val, false_val)`
- **Conditional blocks** - `if/elif/else` with scalar conditions
- **Loop constructs** - `for` loops with `break` and `continue`
- **Index access** - Reference element positions via `_i0`, `_i1`, etc.

## Basic Syntax

### Kernel Definition

A DSL kernel is a single function definition with an explicit input signature.
The body uses Python-style indentation, and `return` defines the output:

```
def kernel(x):
    temp = sin(x) ** 2
    return temp + cos(x) ** 2
```

Signature arguments must match the variables you pass to `me_compile()` by name
(order does not matter). (Callback functions registered via `me_variable_ex` are
not listed in the signature.)

Note: The order of the variables array still defines the pointer order passed to
`me_eval()`. The signature is declarative and does not reorder inputs.

A kernel must return on all control-flow paths. Missing returns are compile errors.

### Temporary Variables

Use any identifier to store intermediate values:

```
def kernel(x):
    squared = x * x
    cubed = squared * x
    return squared + cubed
```

Variables are element-wise arrays matching the input dimensions.

### Element-wise Conditionals

The `where()` function provides element-wise selection:

```
where(condition, value_if_true, value_if_false)
```

Example - clamping values:
```
def kernel(x):
    clamped = where(x < 0, 0, where(x > 1, 1, x))
    return clamped
```

### Index Variables

Access element positions using special variables:

| Variable | Description |
|----------|-------------|
| `_i0`, `_i1`, ..., `_i7` | Current index in each dimension |
| `_n0`, `_n1`, ..., `_n7` | Array shape in each dimension |
| `_ndim` | Number of dimensions |

Example - position-dependent computation:
```
def kernel(x):
    normalized = x * _i0 / _n0
    return normalized
```

### Loops

Basic loop syntax (Python-style indentation):
```
def kernel(n):
    accumulator = 0
    for i in range(n):
        accumulator = accumulator + i
    return accumulator
```

With early exit:
```
def kernel(converged, value):
    for i in range(100):
        if any(converged > 0.99):
            break
        value = iterate(value)
    return value
```

With conditional continue:
```
def kernel(mask, acc):
    for i in range(n):
        if any(mask == 0):
            continue
        acc = acc + compute(i)
    return acc
```

### Conditionals

General `if/elif/else` blocks are supported. Conditions must be scalar/uniform.
Use reductions like `any()`/`all()` to turn an element-wise predicate into a scalar.
Conditions that are uniform across the block (e.g., loop variables like `i`,
`_n*`, `_ndim`, or reductions) are accepted directly, so `if i == 2:` is valid.

Rules:
- No new locals are allowed inside branches.
- Use `return` to produce output; all `return` expressions must infer the same dtype.
- Early returns are allowed, but ensure every control-flow path eventually returns.
  Loops never guarantee a return; add a return after any loop.

Example:
```
def kernel(mask, x):
    if any(mask == 0):
        return 0
    elif _n0 > 10:
        return sum(x)
    else:
        return mean(x)
```

Flow-only loop control:
```
def kernel(mask):
    for i in range(10):
        if any(mask == 0):
            break
        elif i == 3:
            continue
    return i
```

## Available Functions

### Arithmetic Operators
- `+`, `-`, `*`, `/`, `%` (modulo)
- `**` (power)
- Unary `-` (negation)

### Comparison Operators
- `==`, `!=`, `<`, `<=`, `>`, `>=`

### Logical Operators
- `&&` or `and`
- `||` or `or`
- `!` or `not`

Logical operator precedence matches Python: `not` binds tighter than `and`, which binds tighter than `or`
(comparisons bind tighter than all three).

### String Operators
- String literals: `"..."` or `'...'`
- Escapes: `\\`, `\"`, `\'`, `\n`, `\t`, `\uXXXX`, `\UXXXXXXXX`
- Comparisons: `==`, `!=` (string-to-string only)
- Predicates: `startswith(a, b)`, `endswith(a, b)`, `contains(a, b)` (string-to-string)

String variables must be provided with dtype `ME_STRING` and a fixed `itemsize`
via `me_variable_ex` (itemsize is bytes per element and must be a multiple of 4).
String expressions always yield boolean output.

Example (element-wise string match with a scalar control):
```
def kernel(tag, n):
    mask = 0
    for i in range(n):
        if any(startswith(tag, "pre")):
            break
        mask = contains(tag, "\u03B1")
    return mask
```

### Mathematical Functions

**Trigonometric:**
- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`, `atan2`
- `sinpi`, `cospi` (sin/cos of π·x)

**Hyperbolic:**
- `sinh`, `cosh`, `tanh`
- `asinh`, `acosh`, `atanh`

**Exponential/Logarithmic:**
- `exp`, `exp2`, `exp10`, `expm1`
- `log`, `log2`, `log10`, `log1p`

**Power/Root:**
- `sqrt`, `cbrt`, `pow`, `hypot`

**Rounding:**
- `floor`, `ceil`, `round`, `trunc`, `rint`

**Other:**
- `abs`, `copysign`, `fmax`, `fmin`, `fmod`
- `erf`, `erfc`, `tgamma`, `lgamma`
- `fma` (fused multiply-add)

### Reductions

Reduce arrays to scalars:
- `sum(x)` - Sum of elements
- `mean(x)` - Mean of elements (returns float64; NaN for empty input)
- `prod(x)` - Product of elements
- `min(x)`, `max(x)` - Minimum/maximum
- `any(x)`, `all(x)` - Logical reductions

Reductions can appear in expressions:
```
centered = x - sum(x) / _n0
```

### Debug Print

Use `print()` for scalar debugging output:

```
print("value = {}", x)
print("min={} max={}", min(a), max(a))
print("sum =", sum(a))
print(sum(a))
```

- `{}` placeholders are replaced left-to-right by arguments.
- Use `{{` and `}}` for literal braces.
- If the first argument is not a string literal, `print` inserts `{}` for each argument, separated by spaces.
- If the first argument is a string literal with no `{}`, placeholders are appended for the remaining arguments (with a single space separator).
- Arguments must be scalar or uniform across the block (use reductions like `min/max/any/all`).

### User-defined Functions

You can register custom C functions or closures and call them from DSL by
including them in the `me_variable_ex` list passed to `me_compile_ex()`.

Rules:
- Function/closure entries must use `ME_FUNCTION*` or `ME_CLOSURE*` in `type`.
- The return dtype must be explicit (not `ME_AUTO`).
- Names cannot shadow built-ins (`sin`, `sum`, etc.) or reserved identifiers
  (`def`, `return`, `print`, `_i0`, `_n0`, `_ndim`).
- String return types are not supported.

Example (pure function):
```
static double clamp01(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

me_variable_ex vars[] = {
    {"x", ME_FLOAT64, x, ME_VARIABLE, NULL, 0},
    {"clamp01", ME_FLOAT64, clamp01, ME_FUNCTION1 | ME_FLAG_PURE, NULL, 0},
};

me_expr *expr = NULL;
int err = 0;
me_compile_ex("def kernel(x):\n    return clamp01(x)\n", vars, 2, ME_FLOAT64, &err, &expr);
```

Example (closure with context):
```
static double scale(void *ctx, double x) {
    return (*(double *)ctx) * x;
}

double factor = 2.0;
me_variable_ex vars[] = {
    {"x", ME_FLOAT64, x, ME_VARIABLE, NULL, 0},
    {"scale", ME_FLOAT64, scale, ME_CLOSURE1 | ME_FLAG_PURE, &factor, 0},
};
```

## Examples

### Example 1: Polynomial Evaluation

Horner's method for `ax³ + bx² + cx + d`:
```
def poly(a, b, c, d, x):
    t1 = a * x + b
    t2 = t1 * x + c
    return t2 * x + d
```

### Example 2: Softmax Normalization

```
def softmax(x):
    max_val = max(x)
    shifted = x - max_val
    exp_vals = exp(shifted)
    return exp_vals / sum(exp_vals)
```

### Example 3: Distance from Center

```
def distance():
    cx = _i0 - _n0 / 2
    cy = _i1 - _n1 / 2
    return sqrt(cx * cx + cy * cy)
```

### Example 4: Mandelbrot Iteration

```
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

### Example 5: Conditional Thresholding

```
def threshold(x, y, threshold):
    magnitude = sqrt(x * x + y * y)
    return where(magnitude > threshold, magnitude, 0.0)
```

## API Usage

### Parsing DSL Programs

```c
#include "dsl_parser.h"

const char *source = "def kernel(x):\\n    temp = sin(x)**2\\n    return temp + cos(x)**2";
me_dsl_error error;
me_dsl_program *prog = me_dsl_parse(source, &error);

if (!prog) {
    printf("Parse error at line %d, col %d: %s\n",
           error.line, error.column, error.message);
    return;
}

// Use the program...

me_dsl_program_free(prog);
```

### Program Structure

The parsed program contains a block of statements:

```c
typedef struct {
    char *name;
    char **params;
    int nparams;
    me_dsl_block block;  // Contains array of statements
} me_dsl_program;

typedef struct {
    me_dsl_stmt **stmts;
    int nstmts;
} me_dsl_block;
```

Each statement has a kind and associated data:

```c
typedef enum {
    ME_DSL_STMT_ASSIGN,   // variable = expression
    ME_DSL_STMT_EXPR,     // bare expression
    ME_DSL_STMT_RETURN,   // return expression
    ME_DSL_STMT_PRINT,    // print(...)
    ME_DSL_STMT_IF,       // if/elif/else
    ME_DSL_STMT_FOR,      // for loop
    ME_DSL_STMT_BREAK,    // break
    ME_DSL_STMT_CONTINUE  // continue
} me_dsl_stmt_kind;
```

## Performance Tips

1. **Minimize temporaries** - Each temporary uses memory for the full block.

2. **Use `where()` instead of branching** - Element-wise conditionals are SIMD-friendly.

3. **Leverage sincos optimization** - When using both `sin(x)` and `cos(x)` on the same input, miniexpr automatically uses a combined sincos computation.

4. **Process in blocks** - The default 4096-element block size is optimized for cache efficiency.

5. **Avoid nested reductions** - `sum(sum(x))` is not supported; compute in stages if needed.

## Error Handling

The parser reports errors with line and column information:

```c
me_dsl_error error;
me_dsl_program *prog = me_dsl_parse(source, &error);

if (!prog) {
    fprintf(stderr, "DSL error at %d:%d - %s\n",
            error.line, error.column, error.message);
}
```

Common errors:
- Syntax errors (missing operators, unmatched parentheses)
- Unknown function names
- Invalid loop syntax

## Limitations

- **No string output** - String expressions only produce boolean results
- **No recursion** - Functions cannot call themselves
- **Fixed loop limits** - Loop bounds must be known at parse time
- **Max 8 dimensions** - Index variables `_i0` through `_i7`
