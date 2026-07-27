# String support for miniexpr / python-blosc2

## Status

Branch `dsl-string-support` in both repos.  Phases **1, 2 and 3 are done** — both acceptance forms
pass, and utf8 string expressions now run through the span-loop driver.  Phases 4 and 5 remain
gated on measurement and may never be built.

| step | state | commit |
|---|---|---|
| 1a–1c itemsize inference, guards, string evaluator | **done** | miniexpr `e10390a` |
| 1d string kernels + generated case table | **done** | miniexpr `9340d2a` |
| 1e `me_get_itemsize` | **done** | folded into `e10390a` |
| 1i DSL guards (`;`, reductions in control flow) | **done** | miniexpr `11fa57c` |
| NumPy fixed-width semantics fix | **done** | miniexpr `63491f6` |
| 1f–1g blosc2 plumbing + tests | **done** | blosc2 `4598b435` |
| 1h miniexpr: string locals + DSL output width | **done** | miniexpr `adefe02` |
| 1h blosc2: Python string syntax → DSL grammar | **done** | blosc2 `9a310e69` |
| **DSL string kernels end-to-end** | **done** | miniexpr `cad1456`, `efef0e9`; blosc2 `0f92c73b` |
| 1j documentation | **done** | miniexpr `9dffb0a`; blosc2 `9d5ec14e` |
| Phase 2 — bytes `S` / `ME_BYTES` | **done** | miniexpr `30267f1`, `665533a`; blosc2 `6f5c9fe9` |
| 3b prerequisite: `row["col"]` + control flow | **done** | blosc2 `729a2082` |
| 3b pandas 3 fixed-width `str` columns | **done** | blosc2 `729a2082` |
| 3, 3a, 3c: utf8 span driver, `utf8_array`, null policy | **done** | blosc2 `cd6317df`, `332ac400` |
| prefilter fix: operands wider than 255 bytes | **done** | blosc2 `cbcb15b5` |

**Phases 1, 2 and 3 are complete.** Suites green: miniexpr 35/35; python-blosc2 7703 passed /
22 skipped (full `tests/`). Acceptance form 1 passes — the blog kernel as a `@blosc2.dsl_kernel`
over `<U` NDArrays, byte-identical to the row-by-row Python version, with `strict_miniexpr=True`
(`tests/ndarray/test_string_output.py::test_blog_kernel_as_dsl_kernel`). Acceptance form 2
(pandas 3) passes too, via §3b below — the blog kernel runs unmodified through
`df.apply(..., axis=1, engine=blosc2.jit)`.

### The 255-byte prefilter bug (found while building §3)

Independent of strings, and it had been corrupting results silently since long before this branch.
c-blosc2 caps a typesize above `BLOSC_MAX_TYPESIZE` (255) to **1** in the chunk header so its split
machinery keeps working (`blosc2.c`, *"treat buffer as an 1-byte stream"*). `aux_miniexpr()` asked
`blosc2_getitem_ctx()` for each operand block in *element* units, which the chunk then read as a
**byte** range: block 0 came back with only its first few elements populated and every later block
was the untouched `malloc`'d buffer. `arr == "hello"` over 1200 rows of `<U64` matched 1 row instead
of 400, non-deterministically, with no error raised anywhere.

`<U64` is 256 bytes — the first fixed-width string that trips it, and exactly what §3's
power-of-two width bucketing produces. Phases 1 and 2 only ever tested widths up to `<U32`, which
is why it survived them. The fix converts the request through bytes using the typesize the header
actually records (`blosc1_cbuffer_metainfo`); identical arithmetic whenever the typesize is not
capped. Any dtype over 255 bytes was affected, not just strings.

### What the "Could not compress the data" blocker actually was

Four separate defects, none of them where the note above guessed:

1. **`me_eval_nd()` rejected `ME_STRING` outright** (`miniexpr.c`) — a leftover guard from before
   strings could be produced. This is the *chunked* entry point, i.e. the only one python-blosc2's
   prefilter uses, so every string expression failed there.
2. **Both nd evaluators sized the output with `dtype_size()`**, which returns 0 for `ME_STRING`, so
   they bailed with `ME_EVAL_ERR_INVALID_ARG` even once the guard was gone. Now `me_get_itemsize()`.
3. **Three places sized a DSL *variable* slot with `dtype_size()`** — string locals allocated 0
   bytes. Folded into one `dsl_var_item_size()` helper.
4. **Narrow returns were written at their own stride** into an output slot sized for the widest
   branch, so everything after element 0 landed at the wrong offset. `dsl_eval_expr_masked_copy()`
   now takes the destination width and NUL-pads into it, and `program->output_itemsize` is the
   widest return rather than the first.

On the blosc2 side, `lazyudf()` resolved a DSL kernel's dtype with `np.result_type` over the input
dtypes *before* `LazyUDF.__init__` could consult miniexpr, so the container was allocated at the
operand width and miniexpr wrote past it. `_set_pref_expr` now also verifies the compiled width
against the container instead of overrunning the block.

**The expression-level path had been silently falling back to NumPy all along.** The
`BLOSC_ME_JIT_TRACE` line is printed *before* the attempt, so `engine=miniexpr` in the trace proved
nothing; the `except` in `fast_eval` then swallowed the failure. The blosc2 string tests now all
pass `strict_miniexpr=True`, which is the only assertion that actually pins the engine.

### Known limitation introduced deliberately

Reassigning a string local to a **wider** value is rejected with a clear error, e.g.

```python
result = "a=" + prop
result += ", b=" + desc     # rejected
```

Statements already compiled captured the narrower width, so widening after the fact desyncs them.
Resolving it needs a fixed-point pass over the block: compile, note any string local whose inferred
width exceeded its registered one, widen and recompile until stable (widths only grow and are
bounded, so it terminates). Until then the workaround is a fresh variable per step, which is what
the reconstructed blog kernel in the tests does.

### Local build wiring (do not commit)

`python-blosc2/CMakeLists.txt` is modified locally to build against `../miniexpr`:

```cmake
FetchContent_Declare(miniexpr
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../miniexpr
)
```

`GIT_REPOSITORY`/`GIT_TAG` **must stay removed** while `SOURCE_DIR` is set. With all three present,
FetchContent clones the pinned tag *into* `SOURCE_DIR` and destroys the local checkout — this
already cost one full rebuild of Phase 1. Before pushing, revert this file and bump `GIT_TAG` to the
merged miniexpr SHA instead.

### Bugs found and fixed along the way

Four pre-existing defects surfaced; all block or corrupt the target workload.

1. **`dsl_is_candidate()` ignored string literals** — it scanned raw source for `=`, `;` and DSL
   keywords, so *any* expression containing a literal like `'property_type='` was misrouted to the
   DSL parser and failed to compile. Even a bare `'='` literal. Blocks the blog kernel outright.
2. **`;`-joined statements were silently discarded.** `parse_indented_block` skipped to the next
   line after each statement, so `a = 100; return a` inside an `if` compiled cleanly and never ran
   the second statement. This is what python-blosc2's `G1` guard was really protecting against.
3. **`dsl_compile_ctx` was field-by-field initialised**, so newly added fields read uninitialised
   stack — the same kernel compiled or not, run to run.
4. **`NDArray.__radd__` delegated to `__add__`**, reversing operands. Harmless while `+` meant only
   addition; wrong once it also means concatenation.

### Deviations from the plan as written

- **NumPy fixed-width semantics, not NUL-terminated.** An `<Un` slot holds `n` codepoints and is
  NUL-padded only when shorter; a full-width value has no terminator. miniexpr had assumed the last
  unit was always a terminator, costing a character of capacity everywhere and making the concat
  bound one short (`<U24` + `<U4` → `<U27`, where NumPy gives `<U28`). python-blosc2 passes numpy's
  `dtype.itemsize` straight through, so this was the real contract all along.
- **`upper`/`lower` are not width-preserving.** NumPy uses Python's *full* case mapping, which
  expands (`ß`→`SS`, `İ`→2 codepoints). Matching it costs a 3×/2× bound; a 1:1 table would have kept
  the width but silently disagreed with the numpy fallback for the same expression. Marked with a
  `ponytail:` comment naming the span-driver tightening as the upgrade path.
- **`slen` dropped.** It is the only op in the set that takes a string and returns a number, so it
  needs a path through the numeric evaluators that no other op here uses. Add it on demand.

## Context

python-blosc2 exposes three string flavours — fixed-width `string()` (`<Un`), `vlstring()`
(`ObjectArray`, msgpack blobs) and `utf8()` (`Utf8Array`, Arrow-style int64 offsets + UTF-8 blob) —
but string *expressions* barely reached miniexpr: it could consume `ME_STRING` and never produce it,
only `contains`/`startswith`/`endswith` existed, bytes silently fell back to numpy, and
utf8/vlstring never reached miniexpr at all.

Target: run the pandas-3 blog kernel (`datapythonista.me/blog/whats-new-in-pandas-3`) over blosc2
chunks —

```python
result = "property_type=" + row["property_type"]
desc = row["name"].lower()
if " with " not in desc: return result + ", room_type=" + desc.removesuffix(" room")
before, after = desc.split(" with ", 1)
...
```

Everything in it is **statically width-bounded**, which is why this is tractable without a
variable-length evaluator. The kernel must run **unmodified** — that is the point of
`df.apply(f, axis=1, engine=blosc2.jit)`. The `.method()` and tuple-unpacking rewrites landed in
§1h, and the kernel now runs as a `@blosc2.dsl_kernel` over `<U` NDArrays. Through
`df.apply(..., engine=blosc2.jit)` it still does not, for a reason unrelated to strings — see §3b.

## Decisions taken

- **Conservative output width.** miniexpr infers the bound at compile time; blosc2 probe-compiles
  with `ME_AUTO` (`blosc2_ext.me_output_dtype`) and reads it back. `.dtype` may be wider than
  numpy's exact answer; values are identical and nothing ever truncates.
- **No slicing syntax.** The parser has no `[` token. `substr(s, start, len)` instead of `s[a:b]`.
- **No JIT for strings.** `me_jit_c_type()` returns NULL for non-numeric; string kernels stay on the
  DSL interpreter. Parallelism still comes from blosc2's per-block threads.
- **utf8 is contagious.** A string-returning expression with at least one utf8 operand produces a
  utf8 result. A *container* decision in python-blosc2, not a miniexpr dtype — see Phase 3.
  `.dtype` reports `np.dtypes.StringDType()`.
- **vlstring / `ObjectArray` is out of scope.** Rows are msgpack blobs of arbitrary objects with no
  homogeneity guarantee. Only the `_is_utf8_column` branch of `ctable.py:12673` gets lifted.
- **pandas 3 string columns are core, not a follow-on.** Verified against pandas 3.0.3 / pyarrow 24
  / numpy 2.4.6:

  | fact | consequence |
  |---|---|
  | `Series.dtype` is `str` but `dtype.kind == 'O'` | gates must test `pd.api.types.is_string_dtype`, not kind |
  | `np.asarray(series).dtype` is `object` | the `__array__` route destroys the layout; bypass it |
  | backing is `ArrowStringArray`, type **`large_string`** | int64 offsets + UTF-8 blob — identical to `Utf8Array` and to Phase 4's `ME_UTF8` descriptor |
  | `Series.__arrow_c_stream__` exists | use the stream PyCapsule, never the private `._pa_array` |
  | `ChunkedArray` with a validity bitmap | must handle multi-chunk and nulls (§3c) |

- **Nulls propagate, path-sensitively; miniexpr stays null-unaware.** Nulls become `""` before any
  kernel runs and the driver re-applies nullity to string results. Predicates return plain `False`,
  which is what pandas 3 itself does. Full policy in §3c.

---

## Sequencing

Phase numbers are stable identifiers, **not** build order. Built **1 → 2**; Phase 3 is next. (The original order was 1 → 3 → 2; Phase 2 went first because it
was bounded and self-contained, and Phase 3 grew a prerequisite — see §3b.) Phase 3 is on the
critical path for the motivating workload; Phases 4 and 5 are gated on measurement and may never
be built.

---

## Phase 1 — string output for fixed-width `U` (mostly done)

### 1a–1e (done)

`infer_output_itemsize()` in `src/functions.c` computes the conservative compile-time width bound;
`retag_string_concat()` turns `str + str` into a `str_concat` tag stub from `apply_type_promotion`;
`eval_string_expr()` evaluates string trees bottom-up, giving computed children their own per-node
buffer the way the numeric evaluator does, with `string_view_at()` extended to read them.
`me_get_itemsize()` exposes the result. The three old guards (`me_eval`, `me_compile`,
`validate_string_usage`) now reject only *unbounded* widths rather than strings wholesale.

Builtins: `lower upper strip lstrip rstrip removeprefix removesuffix split_part replace substr`,
plus `+` as concat. Case table generated by `scripts/gen_unicode_case.py`.

`split_part(s, sep, k)` replaces Python's `str.split(sep, 1)` + unpacking: `k==0` is the head, `k==1`
the remainder. With the separator absent it yields the whole string for `k==0` and empty for `k>0`,
where Python raises on the unpack. Documented, not emulated — the blog kernel guards with an `in`
test first.

### 1f–1g (done)

`blosc2_ext.me_output_dtype(expression, operands)` probe-compiles with `ME_AUTO` and reports the
inferred dtype, returning `None` when miniexpr cannot type the expression so the caller keeps its
numpy fallback. `LazyExpr.dtype` consults it whenever a string operand is involved, ahead of the
dtype cached during expression building.

### 1h. Kernel syntax: methods and tuple unpacking — **outstanding**

The DSL parser has no `.method()` syntax and no tuple unpacking, so the target kernel does not
parse. **Implement as AST transforms in python-blosc2's `dsl_kernel.py`, not in the C parser** —
that file already runs an `ast.NodeTransformer` + `ast.unparse` pipeline (`_NumpyAttrCallRewriter`,
`_rewrite_numpy_attr_calls`), so both rewrites are a handful of lines there and the C grammar stays
function-call only.

- **Method syntax:** `recv.name(args)` → `name(recv, args)`, mirroring `_NumpyAttrCallRewriter`.
- **Tuple unpacking:** rewrite an `ast.Assign` whose target is an `ast.Tuple` into two `ast.Assign`
  statements; `ast.unparse` emits them on separate lines. **Do not emit `;`-joined source** — that
  path is now a hard error in miniexpr (§1i), and was silently wrong before.
- Restrict the unpack RHS to recognised splittable calls; a general N-ary protocol is YAGNI.
- `G1` (`_one_per_line`, `dsl_kernel.py:340`) can stay: the AST rewrite produces separate lines, so
  it never trips.

### 1i. DSL guards (done)

`;`-joined statements and reductions inside `if`/`for`/`while` bodies are now compile-time errors
rather than silently wrong. Reductions remain valid at top level and as conditions, where
`any()`/`all()` collapsing to a scalar is the documented idiom.

### 1j. Documentation (done)

Done: miniexpr `doc/strings.md` (output width inference, the builtins, NumPy fixed-width
semantics), `doc/data-types.md` (itemsize inference), `doc/dsl-syntax.md` (string locals, branch
widths, the §1i guards); python-blosc2 `RELEASE_NOTES.md` (new-features entry).

Deliberately **not** touched, because they are still accurate until Phase 3 lands: the `utf8()`
docstring in `schema.py` and the `ChoosingStringType` table (string-expression filters on utf8
columns really are unsupported), and `doc/guides/pandas_engine.md` (the pandas engine still rejects
string columns).

---

## Phase 2 — bytes `S` (**done**)

miniexpr `30267f1`, `665533a`; blosc2 `6f5c9fe9`. `ME_BYTES` is a 1-byte code
unit; the kernels are shared with `ME_STRING` and parametrised on the width rather than duplicated.
Reads go through an `sview` `{pointer, length, unit}` so a UCS4 literal can be matched against a
1-byte operand with no conversion buffer, and same-width copies stay a `memcpy`.

Verified against `np.strings`: `S8 + S6` -> `S14`, `upper(S8)` -> `S8` with byte 0xE9 untouched,
`strip`, `replace`, predicates, and a bytes DSL kernel with branches of different widths.

Where `S` genuinely differs from `U`, and all of it matches NumPy:

- case mapping is ASCII-only and 1:1, so `upper`/`lower` **keep** the width instead of paying the
  3x/2x Unicode expansion bound;
- `strip` uses `bytes.strip()`'s ASCII whitespace set, not the Unicode one;
- the two families do not mix in one expression (NumPy raises too) — `promote_types` returns
  `ME_AUTO` for the mismatch and `validate_string_usage()` rejects it;
- a **non-ASCII literal** against a bytes operand is a compile error: literals are stored as UCS4
  and read codepoint-against-byte, which is only exact below 0x80.

One thing the plan did not anticipate: python-blosc2 builds its expression strings with `repr()`,
so `arr + b"-x"` arrives as `(o0 + b'-x')`. The parser now accepts a `b` prefix on string literals.
The prefix does not change storage — it exists so the Python spelling round-trips.

blosc2 side: `kind == "S"` -> `ME_BYTES` in `_me_dtype_from_numpy_dtype`; both `v.dtype.num == 19`
itemsize gates now accept `num == 18`; `me_output_dtype` reports `Sn` back; the string-dtype gates
in `lazyexpr.py` test `kind in "US"`; and the DSL validator accepts `bytes` constants.

---

## Phase 3 — variable-width strings by materialisation: utf8 **and pandas 3** (**done**)

**No miniexpr changes**, as planned. Lifts the `NotImplementedError` on utf8 columns and delivers
the motivating pandas workload.

Source-agnostic driver: **variable-width source → span-local `<Un` → Phase 1 miniexpr**.
`CTable._utf8_span_eval()` (`ctable.py`) walks the column in row spans, materializes each span to a
fixed-width `<Un` array and hands that to miniexpr; `_lazyexpr_over_cols()` routes to it whenever an
expression touches a utf8 column, and `where()` / `sum(where=)` call that instead of
`blosc2.lazyexpr` directly.

- **Driver, not prefilter**, for the reason planned: `Utf8Array` offsets and data have independent
  chunk grids (`_OFFSETS_CHUNKS`, `_DATA_CHUNKS`, `utf8_array.py:57-58`).
- **Bucket the width** to a power of two, so a column costs a handful of compilations rather than
  one per span.
- **Container choice** follows the contagion rule. Bool-returning → plain bool NDArray.

### Deviations from the plan as written

- **`_read_persisted_span` does not return `(rel_offsets, data_slice)`** — it returns a decoded
  `StringDType` array. So the driver's span→`<Un` step is an `astype`, not an offsets walk. The
  zero-copy Arrow-buffer route stays available for Phase 4/5, which is where it would pay.
- **Span size is bounded by bytes, not just rows.** The bucketed width comes from the *longest*
  value in a span, so a single 4 KB row among short ones would materialize
  65536 × 4096 × 4 = 1 GiB. `_utf8_spans()` splits a nominal 65536-row span whenever the `<Un`
  buffer would exceed `_UTF8_EXPR_BUDGET` (64 MiB). The lengths come from the offsets
  (`Utf8Array._span_max_bytes`), so no span is read twice to size it.
- **Span operands are passed as blosc2 arrays, not NumPy ones.** With NumPy operands
  `blosc2.lazyexpr` evaluates through `slices_eval`, which never reaches miniexpr — the string
  kernels would have been bypassed entirely, for correct-looking results. This is the same trap the
  Phase 1 note describes. `_utf8_span_eval(strict=True)` is the assertion that pins it.
- **String-returning utf8 expressions have no caller, so nothing wraps the result in a
  `Utf8Array`.** The driver is dtype-agnostic and concatenates whatever `compute()` returns; the
  only consumer that would want a string result is a *computed column*, which needs a `LazyExpr`
  rather than a materialized value and is a separate job. Computed columns over utf8 still raise.
- **Nested (dotted) utf8 leaves still raise.** `_rewrite_nested_expression` aliases them away before
  the driver could find them; `_utf8_names_in` is therefore called on the original expression and
  rejects dotted names explicitly.

### 3a. Give `Utf8Array` a public face (**done**)

`blosc2.utf8_array(seq, spec=None, **kwargs)` lands as written; `Utf8Array` is exported too.
**Deliberately not** `blosc2.array(seq, dtype=blosc2.utf8())`: `blosc2.array` is annotated
`-> NDArray` and `Utf8Array` is not one, so that spelling makes the return *class* depend on a
kwarg *value*. Reserved future spelling, if the containers ever converge:
`blosc2.array(seq, dtype=np.dtypes.StringDType())`.

One wrinkle the plan did not anticipate: the public name shadowed the internal module
`blosc2.utf8_array`. `from blosc2.utf8_array import X` still resolved (the import machinery finds
the submodule), but attribute-path lookups landed on the function, breaking two
`monkeypatch.setattr("blosc2.utf8_array.…")` call sites in the tests. **The module was renamed to
`blosc2._utf8_array`** (blosc2 `2638a0dd`) — it was always internal, every reference to it is inside
blosc2 or its tests, and the underscore frees the plain name for the constructor. No public API
change.

### 3b. pandas 3 `str` columns — **fixed-width part done**

**A prerequisite the plan did not anticipate, and it was not string-specific.**
`df.apply(f, axis=1, engine=blosc2.jit)` could not run *any* kernel combining `row["colname"]`
with control flow, numeric ones included — tracing evaluated the `if` over a whole column
(`truth value ... is ambiguous`) and the DSL parser rejected the subscript. Fixed with a
`_RowSubscriptRewriter` alongside the other §1h rewrites: a single-parameter function whose every
use of that parameter is `param['literal']` has its signature rewritten to the referenced column
names. `DSLKernel` remembers the original labels (they need not be identifiers), `_jit_dsl_wrapper`
accepts the single row-proxy argument and pulls those columns out, and `_PandasRowProxy` hands back
raw arrays rather than `SimpleProxy` operands on that route.

**Acceptance form 2 passes** for fixed-width string columns: the blog kernel runs unmodified through
`df.apply(..., axis=1, engine=blosc2.jit)`, byte-identical to the plain call, reporting
`engine=miniexpr` (`tests/test_pandas_udf_engine.py::TestRowKernelsWithControlFlow`). The DSL route
raises rather than falling back, so that trace line is now trustworthy.

What was done here, and what is left:

- **`np.asarray` is bypassed** for string columns, as planned: `to_numpy(dtype=object).astype(str)`
  gives the fixed-width array the kernels want. The gates test `pd.api.types.is_string_dtype`,
  not `dtype.kind`, since pandas 3's `str` reports `kind == "O"`.
- **Not yet zero-copy.** The `__arrow_c_stream__` / `LargeStringArray.buffers()` route is still to
  do; it matters once the span-loop driver exists, which is the rest of §3.
- **Nulls are rejected, not propagated.** §3c's policy is for *column-wise* expressions
  (`.str.lower()`, `"x=" + s`), which propagate `NaN`. A row-wise kernel is different: pandas
  itself raises (`"p=" + row["x"]` is a `TypeError`, `row["x"].lower()` an `AttributeError`), so
  substituting `""` would invent a value pandas never produces. The error names the column and the
  `.fillna("")` fix. §3c's path-sensitive `null_out` machinery is still needed for the column-wise
  utf8 path.
- **utf8 columns are untouched**; that is the span-loop driver in §3 proper.


Reuses §3's span loop, bucketing and re-encode. What is new:

- **Bypass `np.asarray` for string columns.** The current handling (`proxy.py`, `0c73c21f`) is
  `hasattr(v, "__array__")` + `np.asarray(v)`, which yields an object array of `PyObject*` and
  destroys the Arrow layout. Detect string columns first and take `Series.__arrow_c_stream__` (or
  `pa.array(s.array)` → `LargeStringArray`).
- **Zero-copy buffers.** `LargeStringArray.buffers()` gives `[validity, offsets(int64), data(uint8)]`
  — exactly the pair `_read_persisted_span` produces, so it feeds the same path unconverted.
- **ChunkedArray.** May have `num_chunks > 1`; iterate chunks as spans.
- **Gates.** Widen `lazyexpr.py:1764` (`_miniexpr_eligible_operand`) and `_PandasRowProxy.__getitem__`
  (`proxy.py`), testing `pd.api.types.is_string_dtype` rather than `dtype.kind`.
- **Out of scope:** pandas 2 and object-dtype string columns; reject with a message pointing at
  `.astype("str")`.

### 3c. Null propagation — **done for utf8; the DSL half is not needed yet**

For a **utf8 column** nulls are a *sentinel string*, not `NaN`, so the policy collapses to two
lines in `_utf8_span_eval()`: build the sentinel mask per span, substitute `""` before any kernel
runs, and `&= ~nulls` the boolean result afterwards. A null then satisfies no predicate — not even
`name == '<NA>'` against the sentinel's own spelling — which is exactly what the operator form
(`Column._utf8_compare`) already did, and the tests assert the two forms agree row for row.

The **path-sensitive `null_out` machinery described below was not built**: its consumers are
string-returning DSL kernels over nullable columns, and neither utf8 (sentinel, so no nullity to
propagate separately) nor pandas row kernels (§3b rejects nulls, because pandas raises there too)
needs it. Build it when a string-returning kernel over a genuinely nullable source appears.

Ground truth, pandas 3.0.3 `str` dtype (nulls are `NaN`, a float — *not* `pd.NA`):

| operation | null row | dtype |
|---|---|---|
| `.str.lower()`, `.str.removesuffix()`, `"x=" + s` | `NaN` propagates | `str` |
| `s == 'x'`, `.str.contains/startswith/endswith` | **`False`** | **plain `bool`** |
| `df.apply(lambda r: r['name'].lower(), axis=1)` | raises `AttributeError` | — |

- **Materialise nulls to `""`** so no C kernel ever sees a null.
- **Bool results are never masked** → `False`, plain `bool`. This reproduces pandas exactly; no
  nullable-boolean container is needed anywhere.
- **String-returning column-wise expressions:** row-level mask (OR the operand validity bitmaps).
  No control flow, so operand nullity and data flow coincide.
- **String-returning DSL kernels: path-sensitive.** Accumulate a per-element `null_out` flag at each
  string-operand read, gated by the interpreter's active `run_mask`, so a row is null only if a null
  was read *on the path it took*. Row-level masking is wrong here:

  ```python
  def k(prop, name):
      if prop == "Entire home":
          return "whole-unit"          # never reads `name`
      return prop + ": " + lower(name)
  ```

  With `name = NaN` and `prop = "Entire home"` row-level masking returns `NaN`; the answer is
  `"whole-unit"`. Cheap because `dsl_eval.c:79-263` is already mask-based.
- **Do not reject null input.** pandas 3 columns carry nulls routinely, and the `.fillna("")`
  workaround yields `""` where pandas yields `NaN`, pushing re-masking onto the user.

Tests must cover: a null row on both branches, a null operand not read on the taken branch,
predicates over nulls returning `False`, and a `null_count == 0` fast path.

---

## Phase 4 — native utf8 predicates (optional; gate on measurement; ~1–1.5 weeks)

Fast path under Phase 3 for predicate-only expressions: `==`, `!=`, `contains`, `startswith`,
`endswith` return bool, so no variable-length *output* machinery is needed. UTF-8 is
self-synchronising, so all five are correct as raw-byte operations.

- New `ME_UTF8` input-only dtype, passed as `struct { const int64_t* offsets; const uint8_t* data; }`
  through the existing `const void*` slot. **Byte-identical to `Utf8Array`'s backing pair and to
  Arrow `large_string`**, so a pandas 3 column feeds it with zero conversion.
- `string_view_at` grows a third branch returning a `(bytes, len)` view.
- Reject `ME_UTF8` where the output would be a string: `infer_output_itemsize` returns 0 and the
  existing check fires. No new validation code.
- **Nulls:** taken only when `null_count == 0`; otherwise fall through to §3.
- **Driver selection:** any string-returning node in the tree → §3's driver; otherwise this one.

**Why gated:** its competition is not §3's decode but the *existing* numpy code —
`equal_mask_span` / `order_masks_span` (`utf8_array.py:552-624`) already compare raw bytes with no
decode. Build Phase 3, profile, and only pay for this if predicate filtering is demonstrably hot.

---

## Phase 5 — native Arrow varlen output in miniexpr (defer; ~2–3 months)

A rewrite of the evaluator's buffer model, not an addition.

1. **Intermediate buffers.** Every internal node is `malloc(nitems * sizeof(TYPE))` inside
   `DEFINE_ME_EVAL`, sized by the C type of the instantiation. Varlen needs a parallel
   `(offsets, data, capacity)` type threaded through all 12 instantiations plus the DSL
   interpreter's masked-copy machinery. This is the bulk of the work.
2. **Output contract.** `me_eval*` never allocates output. Varlen needs a new entry point.
3. **Easy part:** a per-block total-byte bound is computable in one pass over the input offsets, so
   bounded-preallocate works and dynamic growth is never needed.
4. **blosc2 driver.** The prefilter is unusable regardless, so the span-loop driver is needed either
   way — the same one Phase 3 builds.

~6–10 weeks miniexpr, ~2–3 weeks blosc2, against Phase 3's ~1.5–2. The entire payoff is one avoided
copy per span, for utf8 and pandas alike. Same measurement gate as Phase 4, stricter.

---

## Verification

```bash
# miniexpr
cd /Users/faltet/blosc/miniexpr && cmake -B build && cmake --build build -j && ctest --test-dir build
./build/tests/test_string_output && ./build/tests/test_dsl_guards
./check-whitespace.sh

# python-blosc2 (CMakeLists must point SOURCE_DIR at ../miniexpr, see above)
cd /Users/faltet/blosc/python-blosc2 && pip install -e . --no-build-isolation
pytest tests/ndarray/test_string_output.py tests/ndarray/test_lazyexpr.py -q
pytest tests/ndarray/test_stringarrays.py tests/ctable/test_utf8.py -q

# utf8 string expressions (Phase 3) + the 255-byte prefilter regression
pytest tests/ctable/test_utf8.py -q -k "where_expression or utf8_array"
pytest tests/ndarray/test_string_output.py -q -k wide_string_operands

# pandas 3 path (Phase 3b) -- requires pandas >= 3, pyarrow, numpy >= 2
pytest tests/test_pandas_udf_engine.py tests/ndarray/test_jit_dsl_dispatch.py -q

# confirm miniexpr is the engine, not the silent numpy fallback
BLOSC_ME_JIT_TRACE=1 python bench/ndarray/stringops_bench.py   # expect engine=miniexpr
```

End-to-end acceptance, two forms of the same kernel:

1. **blosc2 native** — the blog kernel as a DSL kernel over a `<U32` NDArray, byte-identical to
   `df.apply(format_room_info, axis=1)`, with `strict_miniexpr=True`.
2. **pandas 3** — `df.apply(format_room_info, axis=1, engine=blosc2.jit)` over a real `str`-dtype
   DataFrame including a null row, byte-identical to the same call without `engine=`.

Both must report `engine=miniexpr`; a silent numpy fallback producing correct values is a **failed**
acceptance.
