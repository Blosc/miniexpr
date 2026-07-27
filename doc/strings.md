# ME_STRING / ME_BYTES: Fixed-Width Strings

This document describes miniexpr's two fixed-width string types and the
supported string operations. They differ only in code-unit width:

| dtype | code unit | NumPy | case mapping |
|---|---|---|---|
| `ME_STRING` | 4 bytes (UCS4) | `<Un` | full Unicode, truncating |
| `ME_BYTES` | 1 byte | `Sn` | ASCII-only, 1:1 |

Everything below applies to both unless it says otherwise. The two **never mix**
in one expression — NumPy raises there too, and mixing is a compile error.

## Representation

- Each element is a fixed-size array of code units (UCS4 codepoints for
  `ME_STRING`, bytes for `ME_BYTES`).
- `itemsize` is the per-element byte size; for `ME_STRING` it must be a multiple
  of 4, for `ME_BYTES` any positive size.
- The layout is NumPy's: a slot holds up to `itemsize / unit` code units and is
  NUL-padded when shorter. A value that exactly fills its slot carries **no**
  terminator, so the maximum length is `itemsize / unit`, not
  `itemsize / unit - 1`.
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

`upper`/`lower` are **width-preserving**, as NumPy is. They apply Python's
full case mapping, which can expand (`ß` -> `SS`, `İ` -> two codepoints), and a
value whose mapping does not fit its slot is **truncated** — exactly what
`np.strings.upper` does, e.g. `"straße"` in `<U6` gives `"STRASS"`. Widen the
operand if you need the expansion to survive.

For `ME_BYTES` the mapping is ASCII-only, again as NumPy does for `S`, so it is
1:1 and bytes >= 0x80 are left alone.

`strip`/`lstrip`/`rstrip` trim Unicode whitespace for `ME_STRING` and
`bytes.strip()`'s ASCII set for `ME_BYTES`, again matching NumPy.

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
allocate. The bound may be wider than the exact answer (a `replace` that
grows); values are identical. The one operation that can lose data is a case
mapping that expands past its slot, which truncates — see above, and NumPy does
the same.

An expression whose width cannot be bounded statically is rejected at compile
time rather than evaluated.

`me_get_itemsize()` returns `dtype_size()` for non-string expressions, so it is
the single call to size any output buffer.

## Varlen output (Arrow offsets + byte blob)

The conservative bound above is what a *fixed-width* result costs. It is often
far wider than the values need: a `<U8` operand spends 32 bytes on a row
whether the value is 8 codepoints or 1, and a concat bound adds its operands'
widths whatever the values do.

`me_eval_varlen()` writes the same values in the Arrow varlen layout instead —
`int64` offsets plus a tight byte blob — so the bound is spent on scratch and
never on the result:

```c
size_t bound = me_varlen_data_bound(expr, nitems);   /* worst case, for sizing */
int64_t *offsets = malloc((nitems + 1) * sizeof(int64_t));
uint8_t *data = malloc(bound);
size_t used = 0;
me_eval_varlen(expr, vars_block, n_vars, nitems, offsets, data, bound, &used, NULL);
/* store offsets[0..nitems] and the first `used` bytes of data */
```

- `ME_STRING` yields UTF-8, i.e. Arrow `large_string`. Unpaired surrogates and
  out-of-range codepoints become U+FFFD.
- `ME_BYTES` copies its bytes verbatim, i.e. Arrow `large_binary`. Nothing
  validates them as UTF-8, matching what numpy `S` stores.
- Value length follows the same first-NUL rule as everywhere else: a
  slot-filling value carries no terminator.
- DSL kernels work unchanged — `me_eval()` dispatches the program itself.
- A capacity below what the values need returns `ME_EVAL_ERR_INVALID_ARG`
  rather than overrunning. Size with `me_varlen_data_bound()` and it cannot
  happen.

This packs after evaluation rather than threading varlen buffers through the
evaluator: intermediates stay fixed-width, since they are per-block scratch
that never reaches storage. The cost is one extra pass over a block that is
already in cache.

## String Literals

Literals are UTF-8 and can be written with double or single quotes, optionally
with a `b` prefix (`b'x'`), which is accepted so that Python's `repr()` of a
bytes scalar round-trips. The prefix does not change how the literal is stored:
literals are always UCS4 and are read code-unit-wise against the operand. A
literal with a **non-ASCII** codepoint against an `ME_BYTES` operand is a compile
error, since a codepoint and a byte only agree below 0x80.

Supported escapes:

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
