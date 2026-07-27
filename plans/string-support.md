# String support for miniexpr / python-blosc2

## Status

Branch `dsl-string-support` in both repos.  Phases **1, 2 and 3 are done** — both acceptance forms
pass, and utf8 string expressions now run through the span-loop driver.  **Phase 4 is closed
without building `ME_UTF8`**: its measurement gate came back against it, and the rewrite pass that
shipped instead reaches the same target 5–6× faster than before (see §4).  **Phase 5's output half
is done in miniexpr** — `me_eval_varlen()` emits Arrow offsets + a byte blob, 9.7× smaller than the
fixed-width result for a 12 % pack overhead, and its 6–10 week estimate turned out to rest on a
premise the code does not have (see §5).  **In blosc2 it does not pay and was reverted**:
`compute_varlen()` reached DuckDB's B/row exactly and still came out bigger and slower than the
compressed fixed-width result, because compression had already collected that win.  What *did*
pay is making `upper`/`lower` width-preserving (miniexpr `5a7de4f`), which then exposed that
**blosc2 should not SHUFFLE string dtypes** — see §5.  The finished thing
is benchmarked against numpy/pandas/polars/duckdb on real data — see §"Chicago Taxi string
benchmark".

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
| prefilter fix: operands wider than 255 bytes | **done** | blosc2 `cbcb15b5`, superseded by `9bb345d3`, `7cbd48b7` |
| 4 utf8 scalar predicates via raw-byte scan (`ME_UTF8` **not** built) | **done** | blosc2 `47b033c2` |
| eval-block stride fix: string output past element 4096 | **done** | miniexpr `a6a694d` |
| Chicago Taxi string benchmark vs numpy/pandas/polars/duckdb | **done** | blosc2 `bench/chicago-taxi/string-ops.py` |
| 5 miniexpr varlen output (`me_eval_varlen`, Arrow offsets + blob) | **done** | miniexpr `src/miniexpr_varlen.c` |
| 5 varlen intermediates in `eval_string_expr()` | **not built**, gated | — |
| 5 blosc2: `Utf8Array` results from the utf8 span driver | **done** | blosc2 `9cb490ac` |
| 5 blosc2: `compute_varlen()` — built, measured, **reverted** | **negative result** | blosc2 `f6b06438`, reverted in `4e364900` |
| `upper`/`lower` width-preserving (the `ponytail:` bound item) | **done** | miniexpr `5a7de4f` |
| 5 blosc2: computed columns over utf8 | **todo** — persistence caveat in §5 | — |

**Phases 1, 2 and 3 are complete.** Suites green: miniexpr 36/36; python-blosc2 7761 passed /
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
is why it survived them. Any dtype over 255 bytes was affected, not just strings.

The first fix (blosc2 `cbcb15b5`) was a local `getitem_span()` helper that converted the request
through bytes using the typesize the chunk header actually records
(`blosc1_cbuffer_metainfo`) — identical arithmetic whenever the typesize is not capped. The same
family turned up in `SChunk.get_slice()` and in the index sidecar reader, so `393a2004` swept all
of them through that one helper.

**Both are gone now: the real fix landed upstream** (blosc2 `9bb345d3`, `7cbd48b7`).

- Blosc/c-blosc2#796 fixed `blosc2_schunk_get_slice_buffer()` and the single-coordinate path of
  `blosc2_schunk_get_sparse_buffer()` to convert through the typesize chunks actually carry, so the
  `SChunk.get_slice()` workaround reverted to a plain `super()` call. Upstream also made partial
  getitem decodes return `BLOSC2_ERROR_DATA` instead of looking like success — that silence is how
  every downstream instance of this family survived so long. The three regression tests added with
  the workaround stayed and now exercise the C path: without the fix, 150 of 153 slice shapes at
  typesize 256 raise and 3 return wrong bytes, so their passing is what verifies the fix is live in
  the linked c-blosc2.
- c-blosc2 `bc074b22` added `blosc2_getitem_bytes_ctx()`, which counts in **bytes** at any typesize.
  All three partial-read sites moved to it (miniexpr prefilter, index sidecar reader, matmul
  prefilter) and `getitem_span()` was deleted rather than rewritten — once the cap rule moved back
  inside c-blosc2 it was just a multiply undoing a division its callers had already done. The cap
  rule now lives only in c-blosc2, which is where it belongs.

The new entry point requires `start` and `nbytes` to be multiples of the stored typesize; every
offset at these sites is derived from the real typesize, so this is exact below the cap and vacuous
above it.

Verified after the upstream switch: `arr == "hello"` over 1200 rows (chunks 400, blocks 100) matches
400/400 at `<U32`, `<U63`, `<U64`, `<U65` and `<U100`, slices are byte-exact across the cap
boundary, and a 320-byte structured dtype round-trips — i.e. the non-string half of the bug is gone
too.

**Pre-push follow-up — done.** `blosc2_getitem_bytes_ctx()` is a build-time dependency, so a system
c-blosc2 too old to carry it would pass the CMake gate and then fail at compile on the missing
symbol. `python-blosc2/CMakeLists.txt` now sets `BLOSC2_MIN_VERSION 3.3.0` and
`BLOSC2_BUNDLED_VERSION v3.3.0`; the unreleased `bc074b22` SHA is commented out beside it.

### The 4096-element eval-block bug (found while benchmarking)

A **fifth** instance of the `dtype_size()`-returns-0-for-strings family below, missed when the other
four were swept. `me_eval()` splits a block into `ME_EVAL_BLOCK_NITEMS` (4096) element chunks and
advances the output pointer per chunk:

```c
const size_t output_item_size = dtype_size(clone->dtype);   /* 0 for ME_STRING */
...
clone->output = (unsigned char*)output_block + (size_t)offset * output_item_size;
```

With a stride of 0 every chunk after the first wrote back onto element 0, and everything past
element 4096 of a block was left as `malloc`'d. Fixed to `me_get_itemsize(clone)`
(miniexpr `a6a694d`), matching what `miniexpr_eval_nd.c:199` and `miniexpr_eval_dsl_nd.c:111`
already did.

**Why the whole of Phases 1–3 missed it.** It needs a block *wider than 4096 elements*, and nothing
before the benchmark used one: the tests build small arrays whose auto-chosen blocks are well under
that, and §3's span driver buckets to `<Un` widths whose blocks land under it too. It is also
completely silent — elements 0..4095 of each block are correct, so a spot check of the head of the
array passes. At 1 M rows with 8192-element blocks, `"x=" + arr` returned garbage for 77 % of its
rows with no error raised anywhere.

Regression test: `tests/ndarray/test_string_output.py::test_block_larger_than_the_eval_block`,
which pins `blocks=(10000,)` explicitly — the assertion `arr.blocks[0] > 4096` is the part that
matters, since an auto-chosen geometry would silently stop exercising the path.

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

**Currently inactive**: `CMakeLists.txt` is pinned to `GIT_TAG a6a694d647f5633bfce639f41ac18c5bc4be64cf`
(the eval-block fix) with `SOURCE_DIR` commented out, which is the shape it should be pushed in.
Re-enable the local wiring only while iterating on miniexpr, and re-pin before pushing.

To build against `../miniexpr`:

```cmake
FetchContent_Declare(miniexpr
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../miniexpr
)
```

`GIT_REPOSITORY`/`GIT_TAG` **must stay removed** while `SOURCE_DIR` is set. With all three present,
FetchContent clones the pinned tag *into* `SOURCE_DIR` and destroys the local checkout — this
already cost one full rebuild of Phase 1. Before pushing, revert this file and bump `GIT_TAG` to the
merged miniexpr SHA instead.

The same file used to pin the bundled c-blosc2 to the unreleased SHA
`bc074b228968d6121b3c8c1a38c0afc0bbf923f6` carrying `blosc2_getitem_bytes_ctx()`. That is
**resolved**: it is on the `v3.3.0` release tag now, with the SHA left commented out beside it.

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
- **`upper`/`lower` were given a 3×/2× bound. That was wrong** — see §5: NumPy is
  width-preserving and truncates, so the reservation was wider than numpy rather than required by
  it. Closed in miniexpr `5a7de4f`. (The alternative rejected here, a 1:1 *mapping table*, really
  would disagree with numpy; truncating the full mapping does not.)
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

Phase numbers are stable identifiers, **not** build order. Built **1 → 2 → 3 → 4 → 5**. (The
original order was 1 → 3 → 2; Phase 2 went first because it was bounded and self-contained, and
Phase 3 grew a prerequisite — see §3b.) Both measurement gates fired as intended and in opposite
directions: Phase 4's said no and shipped a rewrite pass instead, Phase 5's said yes — and then
building it showed most of its estimated cost was not real.

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

## Phase 4 — native utf8 predicates: **measured, not built** (the gate said no)

**`ME_UTF8` was not added to miniexpr.** The plan made this phase conditional on measurement, and
the measurement came back against it. What shipped instead is a rewrite pass in python-blosc2
(blosc2 `47b033c2`) that reaches the same target for a fraction of the cost.

### The measurement

1M rows, `<U32`-class values, `t.where("name == 'x'")` on a utf8 column — the cost of Phase 3's
span driver, broken down:

| step | ms |
|---|---|
| decode (StringDType read) | 27.1 |
| `+ astype(<U32)` | 115.6 |
| `blosc2.asarray(span)` — feeding miniexpr | 45.7 |
| **miniexpr eval** | **19.6** |
| full expression path | 262.9 |
| operator form `t[t.name == "x"]` (raw bytes, no decode) | **53.0** |

miniexpr is 19.6 ms of 263. A native `ME_UTF8` would delete the decode and the astype and land at
roughly the operator form's 53 ms — which is exactly what `Column._utf8_scalar_mask` already does
today, in NumPy, for free. The 1–1.5 weeks of threading a new dtype through all 12 evaluator
instantiations buys what a regex rewrite pass buys. This is precisely the risk the gate was written
to catch.

### What shipped instead

`CTable._rewrite_utf8_predicates()` — mirroring the `_rewrite_dictionary_predicates` pass that
already sat next to it. A `utf8col <cmp> 'literal'` term (both operand orders, all six comparisons)
is answered by the raw-byte scan and substituted into the expression as a **boolean operand**, so
the rest of the expression stays one native expression. A utf8 name drops out of the span driver's
work list only when *every* occurrence was rewritten, so `startswith`/`contains`/`upper` still route
to §3, and a mixed expression like `startswith(name, 'x') | (name == 'zz')` rewrites the half it can.

| case | before | after | |
|---|---|---|---|
| 1M rows, short (~8 B) | 156.5 ms | 28.2 ms | 5.5× |
| 1M rows, medium (~31 B) | 267.5 ms | 56.3 ms | 4.8× |
| 200k rows, long (~105 B) | 97.7 ms | 16.0 ms | 6.1× |

The expression form now matches the operator form, which was Phase 4's actual target. Tests assert
which **route** an expression takes, not just its answer — correctness alone cannot tell the two
apart, which is the same trap as the `strict_miniexpr` one in Phase 1.

### When to revisit `ME_UTF8`

Three things would change the verdict, none of them true today:

- **`contains`/`startswith`/`endswith` on utf8 become hot.** They have no raw-byte helper, so they
  still pay the full decode → `<Un` → miniexpr trip. The cheap answer is another NumPy span helper
  next to `equal_mask_span`, not a miniexpr dtype — try that first.
- **Predicates need to fuse with heavy numeric work** in one pass over the data. The rewrite pass
  materializes each mask separately; only a real dtype would fuse them.
- **The decode itself shows up on a profile of a real workload**, rather than in a microbenchmark.

The original design is kept below for whoever revisits it.

### Original design (unbuilt)

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

---

## Phase 5 — native Arrow varlen output in miniexpr (**output half done**)

miniexpr `me_eval_varlen()` + `me_varlen_data_bound()` (`src/miniexpr_varlen.c`). String
expressions now emit the Arrow varlen layout — `int64` offsets plus a tight byte blob,
`large_string` for `ME_STRING` (UTF-8) and `large_binary` for `ME_BYTES` (verbatim). DSL kernels
included, since `me_eval()` dispatches the DSL program itself.

**The measured result, on the blog kernel's shape at 1 M rows**
(`bench/benchmark_varlen_output.c`):

| | time | result |
|---|---|---|
| fixed-width `me_eval` | 819 ms | **336 B/row** |
| `me_eval_varlen` | 914 ms | **34.8 B/row** |

9.7× smaller for a 12 % pack overhead — and 34.8 B/row is *below* DuckDB's 35.9 in the benchmark
table above, from `<U` operands, without touching the operand side. All three inflation factors
(UCS4, the `lower()` bound, the padding) are gone from the result at once, because the bound is now
spent on scratch only.

### The cost estimate below was wrong, and that is why this got built

**Strings never go through `DEFINE_ME_EVAL`'s buffer model.** Point 1 — "threaded through all 12
instantiations", priced as the bulk of ~6–10 weeks — describes code that does not handle strings.
String *production* lives entirely in `eval_string_expr()` (`functions.c:3084`), a separate
bottom-up evaluator that allocates its own per-node buffers at `functions.c:3108-3114`. The 12
numeric instantiations only ever *read* strings, through `string_view_at()`, for bool-returning
predicates. The estimate predates Phases 1–3 actually being built and was never revisited against
the code that resulted.

### What shipped, and what it deliberately skips

**Pack after evaluation, do not thread varlen buffers through the evaluator.** The intermediates
stay fixed-width. They are per-block scratch that never reaches storage, so widening them buys
nothing the measurement can see; what costs 336 B/row is the *stored* result, and packing on the way
out deletes that for one extra pass over a block already in cache. `src/miniexpr_varlen.c` is ~120
lines built only on the public API, with no coupling to `functions.c` at all.

That leaves points 1 and 2 of the original design — varlen *intermediates* and an evaluator that
never materialises a fixed-width block — unbuilt, and still gated. Their remaining payoff is the
12 % pack pass plus the scratch allocation, against a rewrite of `eval_string_expr()`. Build it only
if that pass shows up on a profile of a real workload; the `ponytail:` comment at the head of
`miniexpr_varlen.c` names the upgrade path.

Point 3 held as written: `me_varlen_data_bound()` is `nitems * itemsize`, exact for both dtypes
(a UCS4 slot spends 4 bytes per codepoint and UTF-8 never needs more), so bounded-preallocate works
and dynamic growth is never needed.

### The blosc2 side: driver done, caller not

**Done** (blosc2 `9cb490ac`): `_utf8_span_eval()` returns a **`Utf8Array`** for string-returning
expressions, extended span by span, so only one span's `<Un` block is ever live and each stored row
costs its own UTF-8 length. Bool and numeric results are untouched. Nulls keep §3c's policy — the
sentinel goes back into the values and the result spec carries it. This is the contagion rule from
§Decisions, and it closes the §3 deviation *"nothing wraps the result in a `Utf8Array`"*.

**`me_eval_varlen()` is deliberately not wired into the utf8 span driver.** Its only consumers are
`where()` and `sum(where=)`, both **boolean** — and a boolean result has no varlen output to speak
of.

### `blosc2.compute_varlen()` — built, measured, and it does *not* pay

Built (blosc2 `f6b06438`): `blosc2_ext.eval_varlen()` exposes `me_eval_varlen()`,
`Utf8Array.extend_encoded()` takes offsets+bytes in bulk without decoding to `str`, and
`blosc2.compute_varlen(expr)` runs a `LazyExpr` or a DSL-backed `LazyUDF` in row spans across a
thread pool (the Cython binding releases the GIL) into a `Utf8Array`. Values are identical to
`.compute()`, verified by the benchmark's cross-engine digest.

**The measurement refutes the rationale in §"The gate now has evidence".** 1 M rows, `transform`,
each engine run in its own process (running second costs ~40 % on this machine — the `filter` row
is literally the same function in both engines and moved 25 → 33 ms, which is how this was caught):

| | B/row | stored | time |
|---|---|---|---|
| fixed-width `<U66` | 264 | **0.81 MB** | **133 ms** |
| varlen | **34.2** | 1.14 MB | 149 ms |

The varlen blob lands at 34.2 B/row — right on DuckDB's 35.9, exactly the target — **and still
loses on both axes**:

- **Footprint: blosc2 stores results *compressed*, and that had already solved this.** The
  404 B/row figure this phase was ranked on is the *uncompressed* result; blosc2 never stored it.
  Worse, the fixed-width form's NUL padding compresses to almost nothing while a dense UTF-8 blob
  has nothing left to squeeze, so varlen comes out **bigger**: 1.14 MB vs 0.81. The benchmark table
  said this all along — blosc2's 0.9 MB against DuckDB's 34.2 — and the §5 comparison read
  blosc2's uncompressed B/row against everyone else's stored bytes.
- **Time: break-even is the ceiling.** Serial breakdown at 1 M rows: `eval_varlen` 290 ms,
  `extend_encoded` 70 ms, operand slicing 20 ms. Even with perfect threading and a free
  accumulator that is ~110 ms against the fixed path's 133. The prefilter runs in blosc2's own C
  thread pool *fused with compression*; varlen output has no fixed per-element stride, so it cannot
  use the prefilter at all and has to be driven from Python.

**Kept, but reframed as a representation feature**, since it is the only route from an expression
to an Arrow varlen result and is what a `Utf8Array`-typed computed column will need. Its docstring
says plainly that it is not for speed. In the benchmark it is off by default —
`--engines "blosc2,blosc2 (varlen)"`, one engine per process — because it is a representation
comparison, not a cross-engine one.

**What this means for the rest of Phase 5.** The miniexpr-side result stands on its own (9.7×
smaller output, `bench/benchmark_varlen_output.c`) — it is real for any consumer that stores what
miniexpr hands it. It just does not transfer to blosc2, whose compressor was already collecting
that win. Varlen *intermediates* — the 6–10 week item — are now firmly not worth building: their
payoff was a subset of this one.

### Still to do: computed columns over utf8

This is the caller. `_normalize_expression_transformer` (`ctable.py:9787`) calls
`_guard_scalar_expression(expr)` without `allow_utf8`, so `add_computed_column` over a utf8 column
raises. Findings from scoping it:

- **The "needs a `LazyExpr`, not a materialized value" objection is already answered in the file.**
  `_build_computed_lazy()` (`ctable.py:9937`) *eagerly materializes* its DSL branch —
  `lazyudf(...).compute()` on every access — precisely because the miniexpr DSL path cannot do
  partial-slice getitem. A utf8 entry follows that precedent: run the span driver, return the
  `Utf8Array`. Consumers slice it (`lazy[int]`, `lazy[a:b]`), which `Utf8Array` supports.
- **Persistence is the part that will bite.** `_schema_dict_with_computed()` (`ctable.py:9216`)
  writes `str(cc["dtype"])` and `_load_computed_cols_from_schema()` (`ctable.py:9445`) reads it back
  with `np.dtype(...)`. For a utf8 result the dtype is `np.dtypes.StringDType()`, and
  `np.dtype("StringDType()")` raises — so the column would save fine and then make the table
  **unopenable**. Serialize a sentinel (`"utf8"`) and map it back, and audit the
  `np.asarray(..., dtype=cc["dtype"])` call sites (`ctable.py:5229`, `9679`, `11287`) for the same
  assumption.
- Then: a new `kind: "utf8_expression"` descriptor through
  `_normalize_transformer` / `add_computed_column` / `_build_computed_lazy` / serialize / reload,
  `_readable_computed_expr`, `materialize_computed_column` (target spec should be `utf8()`), and
  `where()` over such a column.

Tests must pin the route, not just the values — the same trap as `strict_miniexpr` in Phase 1 — and
must include a save/reopen round trip, which is where the dtype problem shows up.

### `upper`/`lower` are width-preserving now (miniexpr `5a7de4f`)

The `ponytail:` item at `functions.c:2696` is closed, and the premise in §"Deviations" was wrong:
**NumPy does not reserve for case expansion either.** `np.strings.upper` on `<Un` returns `<Un` and
*truncates* — verified against numpy 2.4.6:

```
upper("straße") in <U6 -> "STRASS"      upper("ßßß") in <U3 -> "SSS"
lower("İ")      in <U1 -> "i"
```

So the old 3×/2× reservation was *wider* than numpy, not a compatibility requirement. The bound is
now the operand width; full case mapping still applies, only the slot no longer grows, which agrees
with numpy exactly. `string_case_map()` already bounded every write by the slot, so truncation
needed no code change. (The rejected alternative in §"Deviations" was a 1:1 *mapping table*,
i.e. simple case mapping — that really would disagree with numpy. Truncating full mapping does not.)

Chicago Taxi, full 24.3 M-row table, `kernel`: `<U101` → `<U54`.

| | before | after |
|---|---|---|
| uncompressed (`blosc2 (raw)`) | 10 881 MB, 6 128 ms | **5 766 MB, 4 458 ms** |
| compressed (`blosc2`) | 21.5 MB, 9 011 ms | 48.6 MB, 7 922 ms |

Uncompressed halves and gets 27 % faster, which is what the change was for. Against DuckDB the
`kernel` ratio goes from 3.2× to **2.68×** (7 922 ms vs 2 958 ms), still at 48.6 MB against 842.

### …which exposed a much bigger blosc2 problem: **SHUFFLE on string output**

The compressed column above got *worse* (21.5 → 48.6 MB). Not because narrower compresses worse —
it does not — but because both results crossed **below the 255-byte typesize cap**. Above 255,
c-blosc2 records typesize 1 ("treat as a 1-byte stream"), which silently disables SHUFFLE. `<U101`
is 404 B and `<U54` is 216 B, so tightening the bound turned SHUFFLE *on* — at a typesize that is
meaningless for text.

Turning it back off recovers everything and more (1 M rows):

| | before the bound change | after, default | after, `filters=[NOFILTER]` |
|---|---|---|---|
| `transform` | 134 ms / 0.81 MB | 183 ms / 2.01 MB | **117 ms / 0.78 MB** |
| `kernel` | 314 ms / 0.90 MB | 291 ms / 2.01 MB | **230 ms / 0.80 MB** |

**blosc2 should not shuffle string dtypes.** It only looked fine before because every string result
this workload produced happened to exceed 255 bytes and had its typesize capped. This is a
python-blosc2 default-cparams decision, not a miniexpr one, and it is worth more than anything left
in Phase 5.

**A second, unexplained anomaly found alongside it.** For the *same bytes* (verified byte-identical),
same dtype, same chunks/blocks/blocksize/splitmode/typesize and same cparams, the prefilter write
path compresses 3.2× worse than `blosc2.asarray()`: 2.01 MB vs 0.63 MB at 1 M rows. NOFILTER
narrows it (0.78 vs 0.63) but does not close it. Worth a look on its own — it applies to every
prefilter-written result, not just strings.

### Still unbuilt

- **Document `S` as the choice for ASCII/Latin-1 data** — zero work, still not written down
  anywhere as a performance decision.

### The gate now has evidence, and unlike Phase 4's it argues *for* building

**Source of every number in this section:**
`python-blosc2/bench/chicago-taxi/string-ops.py` — the `kernel` task, i.e. the blog kernel's shape
run as a `@blosc2.dsl_kernel`. Its README section (`bench/chicago-taxi/README.md`,
"String ops") carries the full-table results table and the reproduction commands; the plot is
`bench/chicago-taxi/string-ops.png`. To regenerate:

```bash
cd python-blosc2/bench/chicago-taxi
python string-ops.py                                   # full 24.3 M-row table
python string-ops.py --nrows 1000000                   # the profiling scale used below
python string-ops.py --engines "blosc2,blosc2 (raw)"   # compression cost only
```

At full scale that benchmark puts blosc2 at 3.2× DuckDB on the row-wise kernel. Profiling at 1 M
rows says **the gap is entirely the `<U` representation, not the evaluator**:

| same kernel, same code path | time | output |
|---|---|---|
| blosc2 `<U` operands | 300 ms | 404 B/row |
| **blosc2 `S` operands** | **114 ms** | **54 B/row** |
| duckdb | 111 ms | 35.9 B/row |
| polars | 137 ms | 39.7 B/row |

On `S` blosc2 is at DuckDB parity and ahead of polars, with the result still compressed to 2 MB.
Values verified identical between the two paths. Three multiplicative factors inflate `<U`:

1. **UCS4 — 4 bytes/codepoint.** The others hold UTF-8, ~1 B/char for this ASCII data.
2. **The `lower()` width bound — 2×.** On `U` it must reserve for full-case expansion, so
   `<U36`.lower() → `<U72` and the result is `<U101` where **54 suffices**. On `S`, case mapping is
   ASCII-only and 1:1, so the bound is exact — this is most of the `S` win, and it is the
   `ponytail:`-marked item from §"Deviations from the plan as written".
3. **Fixed-width padding.** Mean result length is 31.7 chars in a 101-char slot.

Everything else is secondary and was measured: operand decompression 23 ms; per-op interpreter cost
~10 ns/row/op (a `strip()`-chain sweep at constant width gives +9–11 ms per added op per M rows), so
~50 ms of the 300 for this 5-op kernel; `lazyudf` construction 0 (−6 ms, i.e. noise). Thread scaling
is 3.9× on 8 cores, consistent with being bandwidth-bound on the wide output. For scale, a numeric
DSL kernel over the same rows is 7.5 ms — strings are a scalar interpreter loop with no JIT, but at
10 ns/row/op that is not the cost here.

Compression itself costs **~45 %** of kernel time at full table (8.49 s vs 5.90 s at `clevel=0`) and
buys 506× the memory (21 MB vs 10 881 MB). Take that ratio from repeated runs, not one pair: the
`clevel=0` variant allocates 10.9 GB on a 24 GB machine and is the noisy one. Over five full-table
runs, compressed 8.2–9.4 s (median 8.5) and raw 5.7–6.0 s (median 5.9) after discarding a 7.7 s
first-run outlier.

> **Superseded — read §5 first.** The B/row column below is blosc2's *uncompressed* result set
> against everyone else's *stored* bytes. blosc2 stores results compressed, where the same
> `kernel` result is 0.9 MB against DuckDB's 34.2. Phase 5 was built on the strength of this
> comparison and the measurement went the other way: varlen output does reach 34.2 B/row, and it
> comes out **bigger and slower** than the compressed fixed-width result. The `S` findings and the
> `lower`/`upper` bound item below are unaffected.

Even on `S`, blosc2 moves 54 B/row against DuckDB's 35.9 (31.7 data + 4 offset + 0.1 validity) —
it pays the compile-time max on every row where they pay the mean plus an offset. **That residual
1.4× is exactly what Phase 5 deletes**, along with factors 1 and 3 above. Ranked by payoff/effort:

1. **Tighten the `lower`/`upper` bound on `U`** — ~1.5×, days not months. A 1:1 bound plus a slow
   path for the ~100 expanding codepoints, or a runtime max-length probe per block.
2. **Document `S` as the choice for ASCII/Latin-1 data** — zero work, already at DuckDB parity, and
   currently not mentioned as a performance decision anywhere.
3. **Phase 5** — removes UCS4, the bound and the padding together.

---

## Where the finished thing lands: the Chicago Taxi string benchmark

`python-blosc2/bench/chicago-taxi/string-ops.py` — the string counterpart of the numeric
`compare-query-methods.py` next to it, with its own section in `bench/chicago-taxi/README.md`.
The real dataset's two string columns, `company` (`<U44`, 35 distinct values) and `payment.type`
(`<U11`, 9), whole 24.3 M-row table by default. Every engine must agree or the run aborts —
verification is a streaming digest (per-row lengths plus every 97th row exactly, in 512 K-row
windows), because a 24 M-row `<U117` result is ~10 GB and nothing can hold a second copy.

| task | expression |
|---|---|
| `filter` | `startswith(company, 'Taxi') & (payment_type != 'Cash')` → bool |
| `transform` | `'co=' + company + '\|pay=' + lower(payment_type)` → str |
| `kernel` | the same, branching on whether the company is a cab company |

All three are timed; only `kernel` is plotted. `kernel` is the acceptance workload in miniature: the
blog kernel's shape (row-wise control flow, string locals, branches of different widths), run as a
`@blosc2.dsl_kernel`. Everyone else rewrites it as a mask plus two fully-evaluated branches.
`blosc2 (raw)` is the identical path at `clevel=0`, so compression is the only variable between the
two blosc2 rows.

Full table, Apple M-series (8 cores, 24 GB), warm, best of 3:

| | filter | transform | kernel | kernel result |
|---|---|---|---|---|
| **blosc2** | 648 ms | 3.86 s | 8.49 s | **21 MB** |
| blosc2 (raw) | 279 ms | 1.92 s | 5.69 s | 10 881 MB |
| pandas | 189 ms | 2.02 s | 5.19 s | 932 MB |
| polars | 91 ms | 1.72 s | 3.62 s | 932 MB |
| duckdb | 338 ms | 1.97 s | 2.95 s | 842 MB |

At 1 M rows, for the profiling scale used in §Phase 5: blosc2 300 ms / 0.9 MB, pandas 208 / 37.9,
polars 138 / 37.9, duckdb 114 / 34.2, numpy 708 / 417, pandas `.apply()` 21 400 ms.

What it says:

- **The DSL kernel is 69× `df.apply()`.** That is the whole proposition: the row-wise spelling stops
  being the slow spelling.
- **blosc2 holds everything compressed for a real but bounded price** — 21 MB against 842–932 MB,
  at 1.4–2.9× the time of the columnar engines. The `<U` representation, not the evaluator, is what
  costs the time; see §Phase 5 for the decomposition and for `S` reaching DuckDB parity.
- **`filter` is blosc2's weakest task** (648 ms vs polars' 91). A bool result is 1 byte per row, so
  there is no output-side compression win to offset the operand decompression. Consistent with
  Phase 4's finding that predicate throughput is dominated by getting bytes to the kernel, not by
  the kernel.
- **NumPy is off by default** — its `kernel` builds five full-width `<U` temporaries, ~10 GB each at
  24 M rows. At 1 M rows it loses on both time and memory.

Two measurement traps this benchmark hit, worth not re-learning: `polars.Series.estimated_size()`
reports the data buffer only and omits the 8 B/row offsets (~20 % low here), and converting each
engine's result to NumPy before measuring turns Arrow varlen into an object array, which both
mis-reports the footprint and charges the conversion to the engine's time.

The benchmark is what surfaced the eval-block bug above: it is the first thing in either repo to run
a string expression over blocks wider than 4096 elements.

---

## Verification

```bash
# miniexpr
cd /Users/faltet/blosc/miniexpr && cmake -B build && cmake --build build -j && ctest --test-dir build
./build/tests/test_string_output && ./build/tests/test_dsl_guards
./build/tests/test_varlen_output          # Phase 5 output half
./build/bench/benchmark_varlen_output     # its footprint/overhead numbers
./check-whitespace.sh

# python-blosc2 (CMakeLists is pinned to a pushed miniexpr SHA; point SOURCE_DIR at
# ../miniexpr only while iterating, see above)
cd /Users/faltet/blosc/python-blosc2 && pip install -e . --no-build-isolation
pytest tests/ndarray/test_string_output.py tests/ndarray/test_lazyexpr.py -q
pytest tests/ndarray/test_stringarrays.py tests/ctable/test_utf8.py -q

# utf8 string expressions (Phase 3) + the 255-byte prefilter regression
pytest tests/ctable/test_utf8.py -q -k "where_expression or utf8_array"
pytest tests/ndarray/test_string_output.py -q -k wide_string_operands

# the capped-typesize family; test_schunk_get_slice.py fails loudly unless the
# linked c-blosc2 carries the Blosc/c-blosc2#796 fix
pytest tests/test_schunk_get_slice.py tests/ctable/test_ctable_indexing.py -q

# pandas 3 path (Phase 3b) -- requires pandas >= 3, pyarrow, numpy >= 2
pytest tests/test_pandas_udf_engine.py tests/ndarray/test_jit_dsl_dispatch.py -q

# confirm miniexpr is the engine, not the silent numpy fallback
BLOSC_ME_JIT_TRACE=1 python bench/ndarray/stringops_bench.py   # expect engine=miniexpr

# blocks wider than one miniexpr eval block (4096 elements)
pytest tests/ndarray/test_string_output.py -q -k block_larger_than

# the Chicago Taxi string benchmark; cross-checks all five engines byte for byte,
# so it doubles as an end-to-end correctness run
cd bench/chicago-taxi && python string-ops.py --nrows 1000000
```

End-to-end acceptance, two forms of the same kernel:

1. **blosc2 native** — the blog kernel as a DSL kernel over a `<U32` NDArray, byte-identical to
   `df.apply(format_room_info, axis=1)`, with `strict_miniexpr=True`.
2. **pandas 3** — `df.apply(format_room_info, axis=1, engine=blosc2.jit)` over a real `str`-dtype
   DataFrame including a null row, byte-identical to the same call without `engine=`.

Both must report `engine=miniexpr`; a silent numpy fallback producing correct values is a **failed**
acceptance.
