# ME_STRING: Fixed-Size UCS4 Strings

This document describes the `ME_STRING` type for miniexpr and the supported
string operations.

## Representation

- Each element is a fixed-size array of UCS4 codepoints (`uint32_t`).
- `itemsize` is the per-element byte size and must be a multiple of 4.
- The layout is NumPy's `<Un`: a slot holds up to `itemsize / 4` codepoints and
  is NUL-padded when shorter. A value that exactly fills its slot carries **no**
  terminator, so the maximum length is `itemsize / 4`, not `itemsize / 4 - 1`.
- Embedded NULs are not supported: the first NUL ends the value.

## API: Variables and Compilation

Use `me_variable` to supply `itemsize` for string variables:

```c
uint32_t names[][8] = {
    {'a','l','p','h','a',0,0,0},
    {'b','e','t','a',0,0,0,0},
};

me_variable vars[] = {
    {"name", ME_STRING, names, ME_VARIABLE, NULL, sizeof(names[0])}
};

me_expr *expr = NULL;
int err = 0;
if (me_compile("contains(name, \"et\")", vars, 1, ME_BOOL, &err, &expr) != ME_COMPILE_SUCCESS) {
    /* handle error */
}
```

## Expressions

Predicates (boolean output):

- Comparisons: `==`, `!=` (string-to-string only)
- `startswith(a, b)`, `endswith(a, b)`, `contains(a, b)`

String-valued operations (string output):

| function | result |
|---|---|
| `a + b` | concatenation; either side may be a literal |
| `lower(s)`, `upper(s)` | full Unicode case mapping, as NumPy does |
| `strip(s)`, `lstrip(s)`, `rstrip(s)` | whitespace trimming |
| `removeprefix(s, p)`, `removesuffix(s, p)` | drop `p` when present |
| `replace(s, old, new)` | replace every occurrence |
| `substr(s, start, len)` | slice; negative `start` counts from the end |
| `split_part(s, sep, k)` | `k == 0` is the head, `k == 1` the remainder |

`split_part` stands in for Python's `s.split(sep, 1)` plus tuple unpacking.
With the separator absent it yields the whole string for `k == 0` and the empty
string for `k > 0`, where Python's unpack would raise. Guard with `contains()`
first if you need Python's behaviour.

`upper`/`lower` are **not** width-preserving: Python's full case mapping expands
(`ß` -> `SS`, `İ` -> two codepoints), so the bound is 3x for `upper` and 2x for
`lower`. This matches NumPy; a 1:1 table would keep the width but disagree.

`slen` is not implemented: it is the only operation here that takes a string and
returns a number, so it needs a path through the numeric evaluators that nothing
else needs. Ask if you want it.

## Output width inference

miniexpr computes a **conservative** width bound at compile time and publishes
it through `me_get_itemsize()`:

```c
me_expr *expr = NULL;
me_compile("'kind=' + a", vars, 1, ME_AUTO, &err, &expr);
size_t itemsize = me_get_itemsize(expr);   /* bytes per output element */
```

Callers must allocate the output with exactly that itemsize — `me_eval()` and
`me_eval_nd()` write `me_get_itemsize(expr)` bytes per element and never
allocate. The bound may be wider than the exact answer (a `replace` that grows,
a case mapping that expands); values are identical and nothing ever truncates.

An expression whose width cannot be bounded statically is rejected at compile
time rather than evaluated.

`me_get_itemsize()` returns `dtype_size()` for non-string expressions, so it is
the single call to size any output buffer.

## String Literals

Literals are UTF-8 and can be written with double or single quotes. Supported
escapes:

- `\\`, `\"`, `\'`, `\n`, `\t`
- Unicode escapes: `\uXXXX`, `\UXXXXXXXX`

Example: `"α"` matches U+03B1 (Greek alpha).

## DSL Notes

String locals and string-valued `return` statements are supported; see
`doc/dsl-syntax.md`. DSL conditions must still be scalar, so use `any()` /
`all()` to collapse an element-wise string predicate.

There is no JIT for string kernels (`me_jit_c_type()` returns NULL for
non-numeric types); they run on the DSL interpreter. Per-block parallelism from
the embedding library is unaffected.
