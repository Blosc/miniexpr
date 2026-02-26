# miniexpr DSL Syntax (Canonical Reference)

This document defines the currently supported DSL syntax for miniexpr.
When behavior changes in parser/compiler/tests, update this file in the same change.

## Scope and source of truth

The implementation source of truth is:

- Parser: `src/dsl_parser.c`, `src/dsl_parser.h`
- DSL compiler/runtime checks: `src/miniexpr.c`
- Behavioral tests: `tests/test_dsl_syntax.c`, `tests/test_nd.c`

This document describes **current behavior**, including compile-time and runtime errors.

## Program shape

A DSL program must be a single function definition:

```python
def kernel(x, y):
    return x + y
```

Rules:

- Exactly one top-level `def ...:` is expected.
- Leading blank lines and header comments are allowed.
- Trailing content after the function is rejected.

## Header pragmas

Supported pragmas at file header:

- `# me:fp=strict|contract|fast`
- `# me:compiler=tcc|cc`

Notes:

- Duplicate pragma keys are rejected.
- Unknown `me:*` pragmas are rejected.
- Malformed pragma values are rejected.

## Function signature

Supported:

- Positional parameters.

Rejected:

- Duplicate parameter names.
- Invalid signature syntax.

At DSL compile time, parameter names must match provided input variable names by set membership
(order may differ, counts must match).

## Statements

Supported statement kinds:

- Assignment: `a = expr`
- Augmented assignment: `+=`, `-=`, `*=`, `/=`, `//=`
- Expression statement: `expr`
- Return: `return expr`
- Print: `print(...)`
- If chain: `if` / `elif` / `else`
- While loop: `while cond:`
- For loop: `for i in range(...):`
- `break`, `continue`

Notes:

- Python-style indentation is required.
- Empty blocks are invalid.
- `def` nested inside function body is invalid.
- `elif`/`else` without matching `if` are invalid.

## Expressions

Expressions are parsed by miniexpr expression compiler with DSL constraints.

Commonly supported:

- Names and numeric constants
- Unary: `+`, `-`, logical not
- Binary arithmetic and bitwise operators
- Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`)
- Function calls supported by miniexpr

Notable current cast intrinsics:

- `int(expr)`
- `float(expr)`
- `bool(expr)`

Cast intrinsic constraints:

- Must be called form.
- Exactly one argument.

## Loops

### `for ... in range(...)`

Only this loop form is supported:

```python
for i in range(stop):
for i in range(start, stop):
for i in range(start, stop, step):
```

Range arity:

- 1, 2, or 3 arguments supported.
- Other arities rejected at compile time.

Runtime rule:

- `step == 0` is a runtime eval error.

### `while`

Supported with condition expression.

Runtime rule:

- Iteration cap is enforced (`ME_DSL_WHILE_MAX_ITERS` environment setting).

## Conditionals

Supported:

- `if` / `elif` / `else`

Rejected:

- Deprecated `break if ...` / `continue if ...` syntax.

## `print(...)`

Supported as DSL statement.

Rules:

- At least one argument.
- Optional first string format argument.
- Placeholder count must match argument count.
- Print arguments must be uniform expressions.

## Reserved names

The following cannot be used as user variable/function names in DSL context:

- `print`, `int`, `float`, `bool`, `def`, `return`
- `_ndim`
- `_i<d>` and `_n<d>` (reserved ND symbols)
- `_flat_idx`

## ND reserved symbols

Supported reserved ND symbols when referenced in expressions:

- `_i0`, `_i1`, ... (index per dimension)
- `_n0`, `_n1`, ... (shape per dimension)
- `_ndim`
- `_flat_idx` (global C-order linear index)

These are synthesized by DSL compiler/runtime when used.

## Assignment typing and returns

- Reassigning incompatible dtypes for the same DSL local is rejected.
- Return dtype must be consistent across all return statements.
- Programs with non-guaranteed return paths may compile, but missing return at runtime yields eval error.

## Compound assignment desugaring

- `a += b` -> `a = a + b`
- `a -= b` -> `a = a - b`
- `a *= b` -> `a = a * b`
- `a /= b` -> `a = a / b`
- `a //= b` -> floor-division desugaring (`floor(a / b)` semantics)

## Compile-time vs runtime errors

### Compile-time DSL errors (examples)

- Invalid program shape/signature.
- Unsupported statement forms.
- Invalid `range(...)` arity.
- Invalid cast intrinsic arity.
- Reserved-name misuse.
- Return dtype mismatch.

### Runtime DSL eval errors (examples)

- `range(..., step=0)`.
- Missing return on executed control path.
- While-loop iteration cap exceeded.

## Explicitly unsupported Python syntax (current)

These are not part of DSL syntax and should be rejected by frontends/validators:

- Python ternary expression: `a if cond else b`
- `for ... else` / `while ... else`
- Unsupported call targets and keyword-arg call forms (outside allowed subset)

## Versioning / maintenance policy

When parser/compiler/tests change DSL behavior:

1. Update this document in the same PR.
2. Update or add tests in `tests/test_dsl_syntax.c` / `tests/test_nd.c`.
3. Keep frontend validators (e.g. python-blosc2) aligned with this spec.
