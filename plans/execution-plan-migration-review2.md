# Execution-plan migration review 2

Review of:
- `plans/execution-migration-status.md`
- Recent M4 increments across:
  - `miniexpr.codeberg`: `1600d96`, `77580e9`, `aedd6cf`
  - `../c-blosc2.codeberg`: `6b518ffe`, `77d9c718`, `d72ae06e`
  - `../python-blosc2.codeberg`: `afadc881`, `81acaa8b`, `0fb0e4d8`

Plus three follow-up fixes landed and verified in this review pass (`8324ca4` in
`miniexpr.codeberg`, `16070bca` in `../c-blosc2.codeberg`, and `1579d370` in
`../python-blosc2.codeberg`).

---

## Summary assessment

**Verdict: Yes — we are heading in a very solid direction.**

The M4 work added in the reviewed commits strengthens the core/executor/Python
boundary without violating any of the migration invariants established in
`plans/execution-plan-migration.md` and `plans/execution-migration-contracts.md`:

1. **Append-only ABI hygiene (`miniexpr`):** `src/miniexpr_kernel.h` adds typed
   `math_backend` (`ME_MATH_*`) and `jit_compiler` (`ME_COMPILER_*`) policies as
   a v3 suffix (`ME_PREPARE_OPTIONS_V3_SIZE`), leaving `ME_PREPARE_OPTIONS_V1_SIZE`
   and `ME_PREPARE_OPTIONS_V2_SIZE` callers untouched and rejecting invalid enum
   values before compilation (`77580e9`).
2. **Honest per-invocation backend diagnostics:** Separating prepared
   availability (`me_kernel_has_jit`) from actual execution route tracking
   (`me_eval_block_backend` returning `ME_BACKEND_NONE`, `ME_BACKEND_INTERPRETER`,
   or `ME_BACKEND_JIT`, including on failure) avoids false JIT claims on
   identity traversals, empty blocks, or interpreter fallbacks (`1600d96`,
   `6b518ffe`, `afadc881`).
3. **Core-owned partitionability proof:** Moving partitionability proof into
   `me_kernel_is_partitionable` (`aedd6cf`, `d72ae06e`, `0fb0e4d8`) eliminated
   the unsafe C-Blosc2 assumption that all lane-local DAGs can partition. Narrow
   weak captures (where value-dependent range errors can occur) now retain serial
   execution while straight-line proven authored kernels (`b2nd_exec_prepare_kernel`)
   now safely partition through the same gate.
4. **Concurrence and failure-injection rigor:** Thread-start failure injection in
   `../c-blosc2.codeberg/tests/exec-plan/faults.c` (verifying all started
   threads join, counters remain exact, and reduction output stays unpublished),
   `ThreadPoolExecutor` shared-schedule concurrency tests, `EXEC_PLAN_TEST_JIT=ON`
   system-compiler testing, and adding policy options to `prepare_typed_plan`'s
   `@lru_cache` key (`81acaa8b`) are high-value M4 hardening steps.

---

## Fixes landed in this review pass

Three issues found during code review have been fixed, tested, and committed locally:

### 1. [High] Fixed unintended 64 KiB single-worker scratch limit on `NativeGraph` facade
- **Committed in:** `../python-blosc2.codeberg` (`1579d370`, *"Fix single-worker scratch budget default in typed schedule execute"*).
- **Root cause:** `77d9c718` began enforcing `jobs->max_worker_scratch` for
  single-worker calls when `jobs != NULL`, while `b2nd_exec_bridge.h` was updated
  to always pass `&jobs` (even at `workers=1`) and `b2nd_exec.pxi` defaulted
  `max_worker_scratch` to `65536`. Because `TypedGraphHandle.specialize` allows
  up to 64 MiB mapped tiles (`64 * 1024 * 1024`) and `TypedGraphSchedule.execute`
  did not override `max_worker_scratch`, any single-worker `NativeGraph(..., _executor="c-blosc2")`
  call exceeding 64 KiB mapped + gather scratch failed with `Aggregate worker scratch budget exceeded`.
- **Fix:**
  - Updated `b2_exec_execute_into` and `b2_exec_execute_alias` in
    `../python-blosc2.codeberg/src/blosc2/b2nd_exec_bridge.h` (both `#ifdef` and
    `#else` stubs) to accept `int has_worker_scratch`, passing `&jobs` only when
    `workers > 1 || has_worker_scratch`.
  - Changed `max_worker_scratch=None` on `_TypedExecSchedule.execute` and
    `_TypedExecSchedule.execute_alias` in `src/blosc2/b2nd_exec.pxi`: `None`
    retains an unbudgeted `NULL` jobs descriptor at `workers == 1` (while
    enforcing the 64 KiB default when `workers > 1` or honoring any explicit
    non-negative `int max_worker_scratch`).
  - Added regression test `test_typed_facade_large_tile_and_gather_at_workers_1`
    in `tests/test_b2nd_exec_plan.py` covering a 128 KiB mapped tile and >64 KiB
    strided gather at `workers=1`.

### 2. [Medium] Closed `me_kernel_is_partitionable` gap on weak integer arithmetic
- **Committed in:** `miniexpr.codeberg` (`8324ca4`, *"Require non-weak NumPy integer arithmetic for kernel partitionability"*) with Python regression in `../python-blosc2.codeberg` (`1579d370`).
- **Root cause:** In `src/dsl_menudet_expr.c`, `p_signed_op` / `p_unsigned_op`
  only use modular NumPy integer arithmetic when
  `(n->flags & ME_EXPR_FLAG_NUMPY_1_1) && !(n->flags & ME_EXPR_FLAG_WEAK_SCALAR)`;
  weak-scalar integer subexpressions (such as `c + c` or `-c` on weak `ME_INT64`
  captures holding `INT64_MAX`/`INT64_MIN`) evaluate with range-checked
  `dsl_menudet_signed_op` and can fail at runtime with `ME_EVAL_ERR_INVALID_ARG`.
  `kernel_partition_expr` in `src/miniexpr_kernel.c` previously checked only
  operand conversions without checking those flags on integer `+`, `-`, `*`.
- **Fix:** Required `(expr->flags & ME_EXPR_FLAG_NUMPY_1_1) && !(expr->flags & ME_EXPR_FLAG_WEAK_SCALAR)`
  for non-floating (`dtype != ME_FLOAT32 && dtype != ME_FLOAT64`) `+`/`-`/`*`
  nodes in `kernel_partition_expr`. Added C unit checks in
  `tests/test_kernel_api.c` (`partition_proof`) and Python checks in
  `tests/test_b2nd_exec_plan.py` (`test_typed_partition_proof_rejects_checked_weak_conversion`).

### 3. [Low] Unified `init_fp` support merge on partial thread-start failure
- **Committed in:** `../c-blosc2.codeberg` (`16070bca`, *"Unify init_fp support merge on partial thread-start failure"*).
- **Root cause:** When `pthread_create` failed partway in
  `../c-blosc2.codeberg/blosc/b2nd-exec.c`, the early error path checked
  `report->tiles` rather than `parts[i].fp_seen || parts[i].fp_scoped` when
  merging `init_fp.supported`, differing from the normal completion path if an
  earlier started worker observed FP status on a failing tile (`tiles == 0`).
- **Fix:** Extracted `exec_merge_init_fp(report, parts, count, init_fp_seen, init_fp)`
  and called it uniformly from both paths.

---

## Verification run after fixes

All suites were re-built and run cleanly after the three fixes:

- **Miniexpr (`/tmp/me-build-on`, including TinyCC):** `470/470` tests passed (`100% tests passed out of 470`). Artifact-off core rebuilt and installed into `/tmp/me-install-off` from `/tmp/me-build-off`.
- **C-Blosc2 ASan/UBSan (`/tmp/b2-exec-plan-asan`, `-Wall -Wextra -Werror`):** `2/2` tests passed (`exec_plan`, `exec_plan_faults`).
- **C-Blosc2 Installed Static Consumer (`/tmp/b2-installed-test` against `/tmp/b2-install-off` & `/tmp/me-install-off`):** `3/3` tests passed (`installed_miniexpr_consumer`, `installed_kernel_grid`, `installed_exec_plan`).
- **Python-Blosc2 stubs (`b2nd_exec_bridge.h` without `B2_HAVE_B2ND_EXEC`):** `cc -x c -fsyntax-only -Wall -Wextra -Werror -Wno-unused-function` passed.
- **Python-Blosc2 Ruff:** `ruff format` and `ruff check` passed cleanly on edited Python/test files.
- **Python-Blosc2 Isolated Opt-In Wheel (`/tmp/b2nd-exec-venv`, TCC enabled):** `754 passed in 2.73s` across all 9 regression files (`test_b2nd_exec_plan.py`, `test_native_graph_preparation.py`, `test_native_graph_maps.py`, `test_native_graph_extrema.py`, `test_native_graph_truth.py`, `test_native_graph_weak_captures.py`, `test_native_graph_output.py`, `test_expression_graph.py`, `test_expression_graph_lifetime.py`) with zero skips and zero failures.

*(Note: when convenient, update the validation section in `plans/execution-migration-status.md` from the previous `377/377` and `714 passed, 39 skipped` non-TCC snapshot to `470/470` and `754 passed`, and include `EXEC_PLAN_TEST_JIT=ON`.)*

---

## Recommended next steps for the other agent

1. **Refresh `plans/execution-migration-status.md` test counts** to reflect the full TCC-enabled run (`470/470` miniexpr, `754 passed` Python) and note the `8324ca4` / `16070bca` / `1579d370` fixes.
2. **Choose the next implementation milestone cleanly:**
   - Either close out the last two M3 resident items (`where` masks via 3-arity node/mask lowering + explicit reduction `dtype` accumulator casts) so resident M3 parity is completely finished;
   - Or begin **M5 (compressed/chunked C-Blosc2 storage adapters over `b2nd_exec_schedule`)** — e.g. `b2nd_exec_execute_arrays` / chunked reads and direct `b2nd_array_t` sinks with bounded tile/gather scratch and serial ordered-reduction partials — which is the primary remaining architectural milestone proving the scheduler beyond resident NumPy arrays.
3. Also keep **M4** items (broader non-default FP accuracy and math-function kernels under `jit=True`, and Windows/WASM parity checks) scoped explicitly as backend/platform gates rather than blocking M5 progress.
