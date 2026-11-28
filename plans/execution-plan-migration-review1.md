# Execution-plan migration review 1

Review of:
- `plans/execution-plan-migration.md`
- `plans/execution-migration-contracts.md`
- `plans/execution-migration-status.md`

Cross-checked against current M1/M2 implementation anchors in `miniexpr.codeberg`,
`../c-blosc2.codeberg`, and `../python-blosc2.codeberg`.

## Summary assessment

The overall direction is sound:
- The three-layer split (`miniexpr` = typed kernel compiler/JIT, grid geometry,
  and shared numerical primitives; `C-Blosc2` = whole-array plans, schedules,
  traversal, callbacks, workers, and stages; `Python-Blosc2` = syntax/reference
  adaptation, interchange, persistence, and strong wrapper lifetimes) resolves
  the layering problem cleanly.
- Pivoting from operation-specific C pilots (`b2nd_min_i64_*`,
  `b2nd_map_min_i64_*`) to a single typed node-array boundary (`b2nd-exec.h`)
  in M1/M2 was the right decision.
- Core invariants—metadata-only admission, no execution replay on failure,
  immutable plans with borrowed schedules and invocation-local scratch,
  explicit old/new comparison selectors, and separate prepare/specialize/execute
  benchmarks—are strong and should be preserved.

However, several **plan-level gaps** and **early M1/M2 implementation issues**
should be amended before M3–M7 build on top of `b2nd-exec.h`.

---

## 1. Plan-level architectural gaps to amend before M3–M6

### 1.1 Unify graph-DAG plans (`b2nd-exec.h`) and direct `me_kernel` plans (`b2nd-kernel.h`)

- `../c-blosc2.codeberg/blosc/b2nd-exec.h` currently only prepares plans from
  an expression-node DAG (`b2nd_exec_descriptor`), while
  `../c-blosc2.codeberg/blosc/b2nd-kernel.h` takes a caller-prepared
  `const me_kernel *`.
- In M6, `MenudetLazyArray` must schedule user-authored Menudet 0.1/0.2 kernels
  with control flow, block reductions, ND context (`_i0`/`_n0`/`_ndim`/`_flat_idx`),
  complex dtypes, and fixed-width strings. Those cannot be represented as an
  expression-node DAG.
- **Action:** Specify how `b2nd_exec_plan` will accept *either* a typed graph
  descriptor *or* a direct `me_kernel` (prepared via `me_prepare_menudet` /
  `me_prepare_formula`) so M5/M6 share one C-Blosc2 schedule, traversal,
  source/sink, and worker-partitioning engine instead of maintaining two
  parallel schedulers.

### 1.2 Add an explicit `miniexpr` 1.0 core API prerequisite checklist

Section 3 of `plans/execution-plan-migration.md` rightly prohibits duplicating
miniexpr's numerical rules in C-Blosc2, but five primitives required by M3–M7
are not yet exported in `src/miniexpr_kernel.h` or `src/miniexpr_reduce.h`:

| Needed in | Missing core primitive | Current location / status |
| --- | --- | --- |
| M2/M3 | Portable strict-FP scope helper (nearest rounding, FTZ/DAZ check, AArch64/x86-64 fast register path, Emscripten/MSVC fallback) | Private `src/dsl_menudet_fp.h` (`dsl_menudet_fp_begin`/`end`) |
| M3 | Serial `sum`, `prod`, and NaN-checked `any`/`all` span reducers | Private `src/dsl_menudet_reduce.h`; `src/miniexpr_reduce.h` only exports pairwise sum, moments, arg-extrema, and extrema |
| M3 | Reduction accumulator dtype resolution (`int8 -> int64`, `uint8 -> uint64`, `bool -> int64`, compatibility rules) | Private `src/dsl_array.c` (`a_dtype`) |
| M4 | Actual backend introspection (`has_jit` on `me_kernel_info`, plus per-call actual execution mode on `me_eval_block` / `me_run_options`) | Not exposed in `src/miniexpr_kernel.h` |
| M4 | Conservative partitionability proof on `me_kernel_info` (or core promotion helper) | Private `g_partitionable()` in `src/dsl_graph.c` (depends on `dsl_numpy_numeric_promote` and a 40+ math-function allowlist) and `me_artifact_is_partitionable()` in `src/miniexpr_artifact.h` |
| M4 | `math_backend` policy (`auto`/`system`/`sleef`/`accelerate`) | `me_prepare_options` in `src/miniexpr_kernel.h` has no `math_backend` field; `src/dsl_graph.c` injects `# me:math=...` |

- **Action:** Add a "Miniexpr 1.0 core prerequisites" section/table to
  `plans/execution-plan-migration.md` so each primitive lands in `miniexpr`
  before the milestone that needs it, with feature macros / version checks for
  cross-repo builds.

### 1.3 Clarify M4/M5 thread-pool ownership, GIL release, and direct `b2nd_array_t` binding

The plan defines a synchronous host `fork_join` callback, but leaves three
important runtime questions unspecified:

- **Thread-pool ownership & GIL:** Whether C-Blosc2 drives M4/M5 parallelism
  with its own C thread pool (allowing Python-Blosc2 to release the GIL for
  resident and `b2nd` compressed paths) or delegates to a Python-managed pool.
- **Nested oversubscription (M5):** When `N` executor workers decompress
  compressed chunks in parallel, inner chunk decompression must use `nthreads=1`
  (or coordinate with C-Blosc2's pool) to avoid `N * nthreads` oversubscription.
- **Direct C storage vs. Python callbacks (M5 vs. M6):** Separate direct
  `b2nd_array_t *` C storage bindings in M5 (no Python callback overhead per
  block, GIL-free parallel workers) from generic Python source/sink callbacks
  in M6 (which require GIL discipline or serialization when callbacks are not
  thread-safe).

### 1.4 Decide the home of the wire-free restricted text frontend (`dsl_graph_text.c`)

- `plans/execution-plan-migration.md` (Section 3) says to preserve the
  restricted native text frontend by porting/reusing `src/dsl_graph_text.c`.
- Currently `src/dsl_graph_text.c` (lines 134–150) calls `yyjson_read()` to
  parse floating-point literals locale-independently. Under the M9
  `MINIEXPR_BUILD_ARTIFACT=OFF` gate (zero `yyjson`), porting that parser to C
  requires replacing `yyjson_read()` with a locale-independent decimal parser.
- **Action:** Confirm whether any C-only consumer in Blosc2 calls
  `me_graph_prepare_expression()`. If `NativeGraph.from_expression()` is purely
  Python-facing, moving the wire-free restricted text parser to Python (tested
  differentially against `dsl_graph_text.c`) avoids carrying a separate C parser
  and decimal-float parser in C-Blosc2.

### 1.5 Add missing dtype contracts and move MSVC/WASM portability checks earlier

- **Complex and fixed-width strings:** `plans/execution-migration-contracts.md`
  does not currently list `complex64`/`complex128` (handled by `src/dsl_array.c`)
  or fixed-width strings (`ME_STRING`/`ME_BYTES`, used by `MenudetLazyArray`
  in M6). Record both explicitly so admission rules do not drift.
- **Windows/MSVC and WASM/Emscripten:** `plans/blosc2-4.17-to-5.md` tracks
  WASM and cross-platform wheels, while `plans/execution-plan-migration.md`
  only gates ARM64/x86-64 at M9. Catching `<fenv.h>` and alignment portability
  in M2–M3 CI is far cheaper than discovering it at M9.

---

## 2. Concrete M1/M2 implementation issues to amend now

### 2.1 Avoid double-copy and unnecessary scratch on contiguous disjoint maps

- In `../c-blosc2.codeberg/blosc/b2nd-exec.c` (lines 338 and 452–470),
  non-reduction computed maps allocate a `mapped` tile buffer, run
  `me_eval_block` into `mapped`, and then `memcpy` `mapped` into
  `output.data + pos * width`.
- Because `b2nd_exec_execute` already enforces aligned, contiguous, disjoint
  output, contiguous pure maps should evaluate **directly into
  `output.data + pos * width`** with `mapped_bytes == 0`.
- Similarly, `../python-blosc2.codeberg/src/blosc2/typed_exec.py` (line 171)
  currently rejects `min(shape[0], tile) * 8 > 65536` even for identity
  `min(x)` (0 mapped bytes), narrow dtypes (`int16`), or direct-to-output maps.
  Tightening this avoids false fallbacks on common tile sizes (e.g., 16384 or
  32768) in M2/M3 benchmarks.

### 2.2 Fix host FP environment scoping in `b2nd_exec_execute`

`../c-blosc2.codeberg/blosc/b2nd-exec.c` (lines 423–484) hand-rolls
`feholdexcept(&saved)` / `fesetenv(&saved)` around float reductions:
- It does not set or verify `FE_TONEAREST` or check FTZ/DAZ (unlike
  `src/dsl_menudet_fp.h`), so a caller with a non-default rounding mode gets
  wrong reduction results.
- It omits the AArch64 `fpsr` / x86-64 `mxcsr` fast path (`feholdexcept` adds
  avoidable small-array overhead).
- On Emscripten (`__EMSCRIPTEN__`), it will fail or falsely report
  `fp.supported = true` (compare `src/miniexpr_kernel.c`, lines 495–501).
- For floating reductions, `raise_mask` is checked per-tile inside
  `me_eval_block`, whereas flags raised by `me_extrema_span` (e.g., sNaN) are
  only checked after the full tile loop finishes.
- **Action:** Replace this with a shared core FP-scope helper from `miniexpr`
  and check reducer FP flags consistently per tile when `raise_mask != 0`.

### 2.3 Descriptor and lowering limits to widen before M3

- **3-arity nodes:** `b2nd_exec_node.args` in
  `../c-blosc2.codeberg/blosc/b2nd-exec.h` (line 33) currently has size 2;
  `select` (`where`) and `fma` need 3 arguments (matching `src/dsl_graph_plan.h`).
- **Weak integer-power literal specialization:** `src/dsl_graph.c` (lines
  303–308) emits weak integer exponents as literal syntax (`**(2)`) so
  `integer_powers` multiply-chain specialization triggers. Plan for this when
  adding `B2ND_EXEC_POW` (either preserve literal emission in lowering or allow
  `me_prepare_menudet` to specialize weak `int64` exponent captures directly).
- **Growable bounded lowering buffer:** `../c-blosc2.codeberg/blosc/b2nd-exec.c`
  (line 208) uses a fixed `128 + d->nnodes * 96 + d->ninputs * 24` buffer. Once
  longer function names (`arctan2`, `copysign`, `where`, casts) are added, use
  a bounded growable buffer capped at `ME_GRAPH_MAX_BYTES`.
- **Menudet `"0.2"` vs. `"0.1"` and `# me:fp=strict`:** `b2nd-exec.c` lowers to
  Menudet `"0.2"` without emitting `# me:fp=strict`. That is deterministic while
  `inherit_environment == false`, but if `inherit_environment == true` is ever
  used, it would diverge from `src/dsl_graph.c` (which always emits
  `# me:fp=strict`). Either pin `# me:fp=strict` explicitly or remove
  `inherit_environment` from `b2nd_exec_options`.
- **Consolidate Python AST lowering and retire M2 pilots on schedule:**
  `../python-blosc2.codeberg/src/blosc2/typed_exec.py` introduces a separate
  Python AST walker alongside `expression_graph.py`. Consolidate onto one
  Python AST -> descriptor lowerer as M2 closes, and retire `b2nd_min_i64_*`
  and `b2nd_map_min_i64_*` at the M2 exit gate rather than carrying them to M9.

---

## 3. Recommended amendment order before starting M3

1. Add the **Miniexpr 1.0 core prerequisites** table to
   `plans/execution-plan-migration.md` (serial reducers + accumulator typing,
   FP-scope helper, `has_jit`/actual-backend + `partitionable`, `math_backend`).
2. Document the **graph-descriptor + direct-`me_kernel` unification** in
   `plans/execution-plan-migration.md` so M2–M6 share one `b2nd_exec_plan`
   engine.
3. Clarify **M4/M5 threading, GIL release, inner `nthreads=1`, and direct
   `b2nd_array_t` binding**, and settle the ownership of the wire-free text
   parser (`dsl_graph_text.c` without `yyjson`).
4. Fix the two M2 `b2nd-exec.c` issues now (direct-to-output contiguous maps
   with zero mapped scratch; shared strict-FP scope) and widen
   `b2nd_exec_node.args` to 3.
5. Update `plans/execution-migration-contracts.md` with complex/string rows and
   earlier MSVC/WASM portability checks, then close M2 by consolidating
   `typed_exec.py` lowering and retiring the specialized int64 pilots.
