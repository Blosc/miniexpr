# DSL Kernel Programming Guide

This guide explains how to use the miniexpr Domain-Specific Language (DSL) for writing multi-statement expression kernels.

## Overview

The DSL extends single expressions to full programs with:
- **Temporary variables** - Store intermediate results
- **Multi-statement programs** - Combine multiple computations
- **Element-wise conditionals** - Using `where(cond, true_val, false_val)`
- **Loop constructs** - `for` loops with `break` and `continue`
- **Index access** - Reference element positions via `_i0`, `_i1`, etc.

## Basic Syntax

### Statements

A DSL program consists of one or more statements, separated by semicolons or newlines:

```
temp = sin(x) ** 2
result = temp + cos(x) ** 2
```

The last statement's value becomes the output (or assign to `result` explicitly).

### Temporary Variables

Use any identifier to store intermediate values:

```
squared = x * x
cubed = squared * x
result = squared + cubed
```

Variables are element-wise arrays matching the input dimensions.

### Element-wise Conditionals

The `where()` function provides element-wise selection:

```
where(condition, value_if_true, value_if_false)
```

Example - clamping values:
```
clamped = where(x < 0, 0, where(x > 1, 1, x))
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
normalized = x * _i0 / _n0
```

### Loops

Basic loop syntax (Python-style indentation):
```
for i in range(n):
    accumulator = accumulator + i
```

With early exit:
```
for i in range(100):
    if any(converged > 0.99):
        break
    value = iterate(value)
```

With conditional continue:
```
for i in range(n):
    if any(mask == 0):
        continue
    result = result + compute(i)
```

Conditional blocks are only supported for loop control (`break`/`continue`), and
the condition must be scalar. Use reductions like `any()`/`all()` to turn an
element-wise predicate into a scalar condition.
Conditions that are uniform across the block (e.g., loop variables like `i`,
`_n*`, `_ndim`, or reductions) are accepted directly, so `if i == 2:` is valid.

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
for i in range(n):
    if any(startswith(tag, "pre")):
        break
    mask = contains(tag, "\u03B1")
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

## Examples

### Example 1: Polynomial Evaluation

Horner's method for `ax³ + bx² + cx + d`:
```
t1 = a * x + b
t2 = t1 * x + c
result = t2 * x + d
```

### Example 2: Softmax Normalization

```
max_val = max(x)
shifted = x - max_val
exp_vals = exp(shifted)
result = exp_vals / sum(exp_vals)
```

### Example 3: Distance from Center

```
cx = _i0 - _n0 / 2
cy = _i1 - _n1 / 2
result = sqrt(cx * cx + cy * cy)
```

### Example 4: Mandelbrot Iteration

```
zr = 0.0
zi = 0.0
for iter in range(100):
    zr_new = zr * zr - zi * zi + cr
    zi = 2 * zr * zi + ci
    zr = zr_new
    if any(zr * zr + zi * zi > 4.0):
        break
result = iter
```

### Example 5: Conditional Thresholding

```
magnitude = sqrt(x * x + y * y)
result = where(magnitude > threshold, magnitude, 0.0)
```

## API Usage

### Parsing DSL Programs

```c
#include "dsl_parser.h"

const char *source = "temp = sin(x)**2; result = temp + cos(x)**2";
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
    ME_DSL_STMT_EXPR,     // bare expression (result)
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
- **No user-defined functions** - Use built-in functions only
- **Max 8 dimensions** - Index variables `_i0` through `_i7`
