# miniexpr DSL Syntax (Canonical Reference)

This is the practical reference for the DSL accepted by `me_compile()`.
It focuses on what works today and the most common gotchas.
For usage walkthroughs and end-to-end examples, see `doc/dsl-usage.md`.

An experimental [portable profile 0.1](dsl-spec/0.1.md) specifies a smaller
language-independent subset with shared native/Python conformance fixtures.
This reference continues to describe the full native language.

## Quick start

A valid DSL program is one function:

```python
def kernel(x, y):
    temp = sin(x) ** 2
    return temp + cos(y) ** 2
```

Use Python-style indentation and always return a value on the paths you execute.

## Program shape

- Exactly one top-level `def ...:` function is expected.
- Leading blank lines and header comments are allowed.
- Any extra trailing content after the function is a parse error.
- Nested `def` inside the function body is not allowed.
- The first statement may be a single/double-quoted or triple-quoted docstring,
  including multiline strings and `r`/`u` prefixes. It is ignored at runtime.

## Header pragmas

Supported file-header pragmas:

- `# me:fp=strict|contract|fast`
- `# me:compiler=tcc|cc`

Notes:

- Pragma keys must be unique.
- Unknown `me:*` pragmas are errors.
- Malformed pragma values are errors.
- An explicit compiler pragma overrides the `ME_DSL_JIT_COMPILER` environment
  default. With no compiler pragma, that environment setting selects the default.
  `ME_JIT_OFF` disables JIT independently of compiler selection; unavailable JIT
  execution retains best-effort interpreter fallback.

## Function signature and inputs

- Parameters are positional names: `def kernel(a, b, c): ...`
- Parameter names must be unique.
- At compile time, DSL parameter names must match input variable names by set membership
  (order may differ, count must match).

## Statements

Supported statement forms:

- Assignment: `a = expr`
- Compound assignment: `+=`, `-=`, `*=`, `/=`, `//=`
- Expression statement: `expr`
- Return: `return expr`
- Print: `print(...)`
- Conditionals: `if` / `elif` / `else`
- While loop: `while cond:`
- For loop: `for i in range(...):`
- Loop control: `break`, `continue`
- No-op: `pass` (also valid as the sole statement in a branch or loop body)

General rules:

- Python-style indentation is required.
- Empty blocks are invalid.
- `elif`/`else` must belong to a matching `if`.
- Deprecated forms like `break if cond` / `continue if cond` are not part of DSL syntax.
- Simple statements may share a line, separated by `;`, including inside indented
  blocks. A trailing `;` is allowed; empty statements (`;;`) are not.
- Compound statements (`if`, `for`, `while`) require their own lines and indented
  bodies; they cannot follow a semicolon. Inline suites such as `if x: return x`
  are not supported.
- Expressions and calls can continue across lines inside parentheses, including
  comments. This does not add list literals or indexing to the expression grammar.
- **No reductions inside `if` / `for` / `while` bodies.** A reduction collapses
  the block to a scalar, which is meaningless under a per-element mask. They
  remain valid at top level and as a condition, which is the documented way to
  turn an element-wise predicate into a scalar one.

### Docstrings and semicolon-separated statements

```python
def kernel(x):
    """Transform each element.

    Documentation is not evaluated by the interpreter or JIT.
    """
    y = x + 1; z = y * y
    if z > 4:
        z -= 2; z *= 3
    return z
```

### `if` / `elif` / `else` example

```python
def kernel(x):
    if x > 0:
        y = x
    elif x == 0:
        y = 1
    else:
        y = -x
    return y
```

### `for` example

```python
def kernel(n):
    acc = 0
    for i in range(0, n, 1):
        acc += i
    return acc
```

### `while` example

```python
def kernel(x):
    i = 0
    y = x
    while i < 3:
        y = y * 2
        i += 1
    return y
```

## Expressions and function calls

Expressions are compiled by miniexpr with DSL checks.

### Numeric literals

Python-style digit separators are accepted in integers and floating-point
literals: `1_000`, `1_000.2_5`, `.1_25`, and `1e1_0`. Integer literals also accept
binary (`0b1010`), octal (`0o755`), and hexadecimal (`0xff`) prefixes, including
uppercase prefixes and separators such as `0x_FF` or `0b10_10`.

Malformed separators, invalid base digits, and nonzero decimal integers with
leading zeros are rejected. Prefixed integer magnitudes must fit in an unsigned
64-bit value; numeric evaluation retains the existing dtype and precision rules,
not Python's arbitrary-precision integer arithmetic. Strings and identifiers
(such as `x_1`) are not modified. Normalization occurs before expression
compilation, so interpreter and JIT backends see the same numeric text.

Commonly supported:

- Names and numeric constants
- Unary operators: `+`, `-`, logical not (`not` / `!`)
- Arithmetic and bitwise binary operators
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Function calls to supported miniexpr functions
- User-registered C functions/closures passed in `me_variable`

DSL expressions support Python-style chained comparisons such as `0 <= x < 10`
and `a < b <= c != d`. Operands are evaluated left-to-right, each at most once,
and later operands are skipped when an earlier comparison fails. This works in
assignments, nested expressions, conditions, and range arguments. A `while`
condition is re-evaluated on every iteration, including after `continue`.

The native DSL front end lowers chains to temporaries and guarded statements
before interpreter compilation or JIT IR construction. C callers can therefore
pass raw chain syntax to `me_compile()` or `me_compile_nd_jit()` inside a DSL
kernel (`def ...`), without Python rewriting. Generated temporaries avoid names
in the source and infer operand types independently of the output dtype.
Existing DSL `and`/`or` Boolean-result rules apply. The separate classic
expression API retains its existing comparison semantics.

Chaining does not expand the supported operand types or functions: existing
string-comparison and per-element reduction restrictions still apply.

Cast intrinsics:

- `int(expr)`
- `float(expr)`
- `bool(expr)`

Cast rules:

- Use function-call form only.
- Exactly one argument.
- `int(expr)` and `bool(expr)` consume the argument's evaluated value in its
  compiled dtype; they do not recompute its arithmetic in the cast's result
  dtype. Floating `int()` truncates toward zero; `bool()` tests nonzero truth.
  Conversion to the requested output dtype happens after this operation.
  Overflow/non-finite integer conversions and `float()`'s contextual evaluation
  rules remain part of the experimental portable numeric audit.

## Temporary variable type inference

In strict mode, leaf `sin(x)`/`cos(x)` calls on float32 variables evaluate with
scalar float operations at every block size. The float result rounds before
widening or comparison; approximate SIMD math must not change these strict
values or branch decisions. This bounded rule does not certify all other math
functions or nested arithmetic contexts in the experimental portable profile.

Local temporaries get their dtype from the expression assigned to them.

Example:

```python
def kernel(x):
    temp = sin(x) ** 2
    return temp + cos(x) ** 2
```

In this example, `temp` is inferred from `sin(x) ** 2` (typically a floating type).

Notes:

- You do not need to declare local variable types.
- Boolean output does not force numeric temporaries to Boolean: operand types
  are inferred independently, and the return value is converted to Boolean.
- If you assign a value with an incompatible dtype to the same local later, compilation fails.

## Loops

### `for ... in range(...)`

Supported forms:

```python
for i in range(stop):
for i in range(start, stop):
for i in range(start, stop, step):
```

Rules:

- `range` takes 1, 2, or 3 arguments.
- `step == 0` raises a runtime evaluation error.

### `while`

- `while` condition is a regular DSL expression.
- Runtime iteration cap is enforced to prevent runaway `while` loops.
- The host policy `ME_DSL_WHILE_MAX_ITERS` defaults to 10,000,000 body entries
  per loop invocation. Positive values cap execution; zero/negative values
  disable the cap. Invalid/out-of-range values retain the configured default.
- The condition is tested before enforcing the cap: exactly that many body
  entries may complete successfully. `continue` counts as a body entry;
  `break`/`return` on the last allowed entry succeeds. Nested/re-entered loops
  have independent counters. Lowered condition statements do not count as body
  entries.
- Interpreter and JIT cap errors report `ME_EVAL_ERR_INVALID_ARG`. Failed output
  contents are unspecified, and a JIT cap error is not retried in the interpreter.
  JIT captures the cap in its cache-keyed IR; changing the host cap after
  compilation selects interpreter execution under the new policy.
- The experimental portable audit still has an interpreter discrepancy for
  chained conditions with mixed active lanes; see `dsl-spec/numeric-audit-0.1.md`.

## `print(...)`

`print` is supported as a DSL statement.

Rules:

- At least one argument is required.
- First argument may be a format string.
- Placeholder count must match provided values.
- Printed expressions must be uniform/scalar for the block.

## Reserved names

Do not use these as user variable/function names in DSL:

- `print`, `int`, `float`, `bool`, `def`, `return`
- `_ndim`
- `_i<d>` and `_n<d>` (reserved ND symbols)
- `_flat_idx`

## ND reserved symbols

When referenced, these are synthesized by DSL compiler/runtime:

- `_i0`, `_i1`, ... (index per dimension)
- `_n0`, `_n1`, ... (shape per dimension)
- `_ndim`
- `_flat_idx` (global C-order linear index)

## Strings in DSL kernels

String locals and string-valued `return` statements work, for both `ME_STRING`
and `ME_BYTES` operands; the operation set is in `doc/strings.md`. The output
width is the widest of the kernel's `return` expressions, and narrower branches
are NUL-padded into it:

```python
def kernel(property_type, name):
    result = 'property_type=' + property_type
    desc = lower(name)
    if not contains(desc, ' with '):
        return result + ', room_type=' + removesuffix(desc, ' room')
    before = split_part(desc, ' with ', 0)
    after = split_part(desc, ' with ', 1)
    r2 = result + ', room_type=' + removesuffix(before, ' room')
    return r2 + ', amenity=' + after
```

Restrictions:

- A string local's width is fixed by its first assignment. Reassigning it to a
  **wider** value is a compile-time error, because statements already compiled
  captured the narrower width. Use a fresh name per step, as `r2` does above.
- There is no slicing syntax (`s[a:b]`); use `substr(s, start, len)`.
- String kernels are never JIT-compiled; they run on the interpreter.

## Typing and return behavior

- Reassigning incompatible dtypes to the same local is a compile-time error.
- Return dtype must be consistent across all `return` statements.
- Non-guaranteed return paths may compile; if execution reaches a missing return path, evaluation fails at runtime.
- Numeric kernels with non-guaranteed returns can also be JIT-compiled. A missing
  return on any executed element reports `ME_EVAL_ERR_INVALID_ARG`, without
  retrying that semantic failure in the interpreter. Output contents after a
  failed evaluation are unspecified; a kernel may already have written other
  elements before encountering the missing return.

## Compound assignment desugaring

- `a += b` -> `a = a + b`
- `a -= b` -> `a = a - b`
- `a *= b` -> `a = a * b`
- `a /= b` -> `a = a / b`
- `a //= b` -> `a = floor(a / b)`

## Compile-time vs runtime errors

Compile-time error examples:

- Invalid program shape or signature
- Unsupported statement forms
- Invalid `range(...)` arity
- Invalid cast intrinsic arity
- Reserved-name misuse
- Return dtype mismatch

Runtime error examples:

- `range(..., step=0)`
- Missing return on executed control path
- While-loop iteration cap exceeded

## Python syntax that is out of DSL scope

These Python features are not part of this DSL:

- Ternary expression: `a if cond else b`
- `for ... else` and `while ... else`
- Keyword-argument calls and other call forms outside the supported subset
