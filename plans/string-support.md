# String support for miniexpr / python-blosc2

## Status

Branch `dsl-string-support` in both repos.  Build order **1 → 3 → 2 → 4 → 5**.

| step | state | commit |
|---|---|---|
| 1a–1c itemsize inference, guards, string evaluator | **done** | miniexpr `e10390a` |
| 1d string kernels + generated case table | **done** | miniexpr `9340d2a` |
| 1e `me_get_itemsize` | **done** | folded into `e10390a` |
| 1i DSL guards (`;`, reductions in control flow) | **done** | miniexpr `11fa57c` |
| NumPy fixed-width semantics fix | **done** | miniexpr `f0f3a1e` |
| 1f–1g blosc2 plumbing + tests | **done** | blosc2 `4598b435` |
| 1h method syntax + tuple unpacking | **not started** | |
| 1j documentation | **not started** | |

Suites green: miniexpr 34/34; python-blosc2 917 passed across `test_lazyexpr`,
`test_stringarrays`, `test_pandas_udf_engine`, `test_jit_dsl_dispatch`, plus 9 new checks in
`tests/ndarray/test_string_output.py`.

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
`df.apply(f, axis=1, engine=blosc2.jit)` — and today it does not parse: the DSL has no `.method()`
syntax and no tuple unpacking (§1h, still outstanding).

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

Phase numbers are stable identifiers, **not** build order. Build **1 → 3 → 2 → 4 → 5**. Phase 3 is
on the critical path for the motivating workload; Phase 2 blocks nothing; Phases 4 and 5 are gated
on measurement and may never be built.

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

### 1j. Documentation — **outstanding**

- `python-blosc2/src/blosc2/schema.py:849-857` — the `utf8()` docstring still says string-expression
  filters are unsupported. Also the `ChoosingStringType` reference.
- `python-blosc2/RELEASE_NOTES.md:111-114` — same known-gaps note.
- `python-blosc2/doc/guides/pandas_engine.md` — add the string kernel.
- miniexpr `doc/strings.md` (string output, new builtins, NumPy fixed-width semantics),
  `doc/data-types.md` (itemsize inference), `doc/dsl-syntax.md` (the §1i guards).

---

## Phase 2 — bytes `S` (~3–5 days)

Add `ME_BYTES` (1-byte code units). Cheap because the Phase 1 kernels are length-driven; parametrise
on code-unit width. Only `lower`/`upper`/`substr` care about the distinction (ASCII-only for bytes,
matching numpy).

- `promote_types` — `ME_BYTES` and `ME_STRING` do **not** mix (numpy raises too).
- `blosc2_ext.pyx:866-905` — `kind == "S"` → `ME_BYTES`; `:4191` and its duplicate at `:4282` both
  hardcode `v.dtype.num == 19` and must accept `num == 18` (`NPY_STRING`).
- `me_output_dtype` already returns `None` for `S`, so the fallback stays correct until this lands.

---

## Phase 3 — variable-width strings by materialisation: utf8 **and pandas 3** (~1.5–2 weeks)

**No miniexpr changes.** Lifts the `NotImplementedError` on utf8 columns and delivers the motivating
pandas workload. Handles both string- and bool-returning expressions, so it is complete on its own;
Phase 4 is only a fast path under it.

Source-agnostic driver: **variable-width source → span-local `<Un` → Phase 1 miniexpr → re-encode**.

- **Driver, not prefilter.** `Utf8Array` offsets and data have independent chunk grids
  (`_OFFSETS_CHUNKS = 2**17` rows, `_DATA_CHUNKS = 2**21` bytes, `utf8_array.py:57-58`), so the
  prefilter contract does not apply. Drive from `_read_persisted_span` (`:233`), which already
  produces `(rel_offsets, data_slice)`; hang it off `Column._utf8_chunked_bool` /
  `_utf8_chunked_bytes` (`ctable.py:2028-2062`), already looping in 65536-row spans. Thread the span
  loop explicitly — per-block blosc2 parallelism is not available here.
- **Bucket the width.** `n` is span-local and data-dependent, whereas `infer_output_itemsize` bakes
  widths in at compile time. Round `n` up to a power of two and cache compiled expressions by
  bucket, so a column costs a handful of compilations rather than one per span.
- **Container choice** follows the contagion rule. Bool-returning → plain bool NDArray.
- Lift only the `_is_utf8_column` branch of `ctable.py:12673`.

### 3a. Give `Utf8Array` a public face

`Utf8Array(blosc2.utf8())` + `.extend()` + `.flush()` is the only construction path today, and it is
not exported from `__init__.py`. Once a `lazyudf` can *return* one, that asymmetry needs closing:

```python
def utf8_array(seq, spec=None, **kwargs) -> Utf8Array:
    arr = Utf8Array(spec or blosc2.utf8(), **kwargs)
    arr.extend(seq)
    arr.flush()
    return arr
```

Export `utf8_array` and `Utf8Array`. **Deliberately not** `blosc2.array(seq, dtype=blosc2.utf8())`:
`blosc2.array` is annotated `-> NDArray` and `Utf8Array` is not one, so that spelling makes the
return *class* depend on a kwarg *value*. Reserved future spelling, if the containers ever converge:
`blosc2.array(seq, dtype=np.dtypes.StringDType())`.

### 3b. pandas 3 `str` columns as a span source

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

### 3c. Null propagation

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
