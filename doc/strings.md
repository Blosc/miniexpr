# ME_STRING: Fixed-Size UCS4 Strings

This document describes the `ME_STRING` type for miniexpr and the supported
string operations.

## Representation

- Each element is a fixed-size array of UCS4 codepoints (`uint32_t`).
- Strings are NUL-terminated and **must not** contain embedded NULs.
- `itemsize` is the per-element byte size and must be a multiple of 4.
- Maximum string length in codepoints is `itemsize / 4 - 1`.

## API: Variables and Compilation

Use `me_variable_ex` to supply `itemsize` for string variables:

```c
uint32_t names[][8] = {
    {'a','l','p','h','a',0,0,0},
    {'b','e','t','a',0,0,0,0},
};

me_variable_ex vars[] = {
    {"name", ME_STRING, names, ME_VARIABLE, NULL, sizeof(names[0])}
};

me_expr *expr = NULL;
int err = 0;
if (me_compile_ex("contains(name, \"et\")", vars, 1, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
    /* handle error */
}
```

## Expressions

Supported operations:

- Comparisons: `==`, `!=` (string-to-string only)
- Predicates: `startswith(a, b)`, `endswith(a, b)`, `contains(a, b)`

String expressions always produce boolean output.

## String Literals

Literals are UTF-8 and can be written with double or single quotes. Supported
escapes:

- `\\`, `\"`, `\'`, `\n`, `\t`
- Unicode escapes: `\uXXXX`, `\UXXXXXXXX`

Example: `"\u03B1"` matches U+03B1 (Greek alpha).

## DSL Notes

String expressions are allowed inside DSL expressions and conditions, but
conditions must be scalar. Use `any()` / `all()` reductions to convert
element-wise string predicates to a scalar condition.
