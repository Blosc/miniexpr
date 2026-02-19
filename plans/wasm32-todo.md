# wasm32 TODO: DSL `int(...)` Cast Path

Last updated: 2026-02-19

## Context

In `python-blosc2` on wasm32/Pyodide, DSL kernels that use index symbols with `int(...)` casting were failing for this shape and are now passing after integration fixes:

- DSL kernel: `return int(_i0 * _n1 + _i1)`
- Prior integration symptom: `DSL kernels require miniexpr ... miniexpr compilation or execution failed`.
- Resolved integration detail (2026-02-19): python-blosc2 was mapping dtypes via `dtype.num` in `_set_pref_expr`, which is platform-dependent on wasm32/Pyodide.
- Current status in python-blosc2: task-7 kernel now succeeds on wasm32 with deterministic `int64` ramp output, and wasm expected-fail behavior was removed.

The remaining miniexpr work is now runtime-JIT parity for cast intrinsics on wasm32.

## Goal

Keep DSL `int(...)` casting reliable on wasm32 for ND/index-symbol expressions and complete runtime-JIT cast-intrinsic parity on wasm32.

## Work Items

1. Reproduce with a minimal miniexpr-only regression
- Add/extend a wasm-executed test that mirrors python-blosc2 usage:
  - ND expression with `_i0`, `_i1`, `_n1`
  - explicit output dtype `ME_INT64`
  - both no-input and with-input variants
- Candidate files:
  - `tests/test_nd.c`
  - `tests/test_dsl_syntax.c`

2. Confirm whether miniexpr core already passes the scenario
- If miniexpr tests pass but python-blosc2 fails, classify as integration issue and document expected call contract from miniexpr side.
- If miniexpr fails on wasm32, keep investigating below items.

3. Audit cast intrinsic lowering and dtype propagation
- Inspect and validate:
  - `dsl_cast_int_intrinsic`
  - `dsl_cast_int_target_dtype`
  - compile-time dtype selection for DSL statements and return value
  - ND output write path for integral dtypes
- Candidate source:
  - `src/miniexpr.c`

4. Audit wasm32 runtime behavior for cast-heavy DSL
- Verify interpreter path vs JIT path behavior parity for this kernel.
- Ensure reserved index symbol handling does not accidentally disable needed runtime path.
- Confirm no wasm-only narrowing/overflow bugs in writeback for `ME_INT64`.

5. Strengthen diagnostics for unsupported cast combos
- If a cast combination is intentionally unsupported on wasm32, return an explicit error reason (not generic execution failure).
- Add tests that assert the exact reason.

6. Add parity tests and CI coverage
- Native + wasm32 parity for:
  - `int(3.9)`
  - `int(_i0 * _n1 + _i1)`
  - `float(int(x)) + bool(x)`
  - mixed input dtype to integer output
- Ensure these tests run in the wasm32 CI lane.

## Session Update (2026-02-19)

### Completed in this session

- Added/extended cast parity coverage:
  - `tests/test_dsl_syntax.c`: interpreter/JIT parity for:
    - `int(3.9)`
    - `float(int(x)) + bool(x)`
    - `ME_FLOAT32` input to `ME_INT64` output via `int(x)`
  - `tests/test_nd.c`: ND mixed-input cast coverage:
    - `return int(x) + int(_i0 * _n1 + _i1)` with `ME_INT64` output and padding checks.
- Enabled wasm build/run of `test_dsl_jit_runtime_cache` when side-module wasm JIT is enabled:
  - `tests/CMakeLists.txt`.
- Wired wasm helper registration in runtime-cache test (same side-module style used by side-module test):
  - `tests/test_dsl_jit_runtime_cache.c`.
- Identified and fixed one JIT codegen issue in cast lowering:
  - `src/dsl_jit_cgen.c` now rewrites DSL cast intrinsics in JIT C codegen:
    - `int(...)` -> `ME_DSL_CAST_INT(...)`
    - `float(...)` -> `ME_DSL_CAST_FLOAT(...)`
    - `bool(...)` -> `ME_DSL_CAST_BOOL(...)`.
- Found an additional wasm-only runtime JIT failure for cast-heavy kernels at wasm module instantiation time.
- Added an explicit wasm runtime-JIT fallback policy for cast intrinsics:
  - `src/miniexpr.c` detects cast intrinsics in generated JIT C and skips wasm runtime JIT with a precise reason:
    - `wasm32 runtime JIT does not yet support DSL cast intrinsics`
  - preserved specific skip reason instead of always overwriting with generic failure text.
- Added wasm-specific test to enforce current behavior contract:
  - `tests/test_dsl_jit_runtime_cache.c`: cast-intrinsic kernel with `ME_DSL_JIT=1` must run correctly and currently fall back to interpreter (`me_expr_has_jit_kernel(expr) == false`).

### Current status after this session

- ND/index-symbol cast path (`int(_i0 * _n1 + _i1)`) is passing in miniexpr native and wasm interpreter paths.
- Cast parity tests pass on native and wasm for interpreter behavior.
- wasm runtime JIT now has explicit, deterministic behavior for cast intrinsics (fallback + precise reason), but full wasm runtime-JIT execution for cast-intrinsic kernels is not yet implemented.
- python-blosc2 integration still fails for task-7 kernel at compile/setup time before runtime-JIT choice, so runtime-JIT backend work is not the only remaining blocker.

## Session Update (2026-02-19, follow-up)

### Completed in this follow-up

- Completed Phase 1 / item 1 (exact contract reproduction) in miniexpr tests:
  - `tests/test_nd.c` now has python-blosc2 contract-parity cases using `me_compile_nd_jit(...)`:
    - with-input function-form DSL source:
      - `def kernel_index_ramp_int_cast(x): return int(_i0 * _n1 + _i1)`
    - no-input integration-style dummy-param form:
      - `def kernel_index_ramp_no_inputs(__me_dummy0): return int(_i0 * _n1 + _i1)`
    - each evaluated under JIT default / off / on compile+eval modes.
- Completed Phase 1 / item 2 triage:
  - Reproduced and verified current miniexpr behavior for the task-7 contract shape on both native and wasm (`test_nd` passes including the new contract tests).
  - Re-ran python-blosc2 task-7 target test natively:
    - `tests/ndarray/test_dsl_kernels.py::test_dsl_kernel_index_symbols_int_cast_matches_expected_ramp` (pass in local environment).
- Completed Phase 1 / item 3 diagnostics hardening on miniexpr side:
  - Added a thread-local diagnostic API:
    - `me_get_last_error_message()` in `src/miniexpr.h` / `src/miniexpr.c`.
  - Added explicit DSL compile/setup reasons for common rejection paths (e.g. parameter mismatch, reserved names, invalid cast intrinsic usage).
  - Added contract tests asserting reason substrings for intentional failures:
    - parameter mismatch
    - reserved input name
    - invalid cast usage (`int()` empty-arg form).

### Updated status after this follow-up

- Phase 1 compile/setup behavior in miniexpr is now exercised by dedicated contract tests and has explicit diagnostics.
- The former Phase 1 integration gap in python-blosc2 is now closed (diagnostics are surfaced and wasm-safe dtype mapping is in place).
- Full wasm runtime-JIT support for cast intrinsics remains Phase 2 follow-up work.

## Session Update (2026-02-19, Phase 2 takeover)

### Completed in this session

- Root-caused the wasm invalid-module failure for cast-heavy kernels by reproducing without the temporary skip gate:
  - `WebAssembly.Module(): ... i32.trunc_f32_u ... expected type f32, found ... f64`.
- Implemented wasm-safe cast lowering for runtime-JIT kernels in the wasm patch step:
  - `src/miniexpr.c` now rewrites JIT C cast macros to helper-call form in patched wasm source:
    - `ME_DSL_CAST_INT(x)` -> `me_wasm32_cast_int((double)(x))`
    - `ME_DSL_CAST_FLOAT(x)` -> `me_wasm32_cast_float((double)(x))`
    - `ME_DSL_CAST_BOOL(x)` -> `me_wasm32_cast_bool((double)(x))`
- Added explicit wasm import helpers in both wasm JIT JS import builders:
  - `src/me_jit_glue.js` (side-module path used in wasm tests)
  - `src/miniexpr.c` EM_JS instantiate helper (main-module path)
- Removed the temporary wasm cast-intrinsic runtime-JIT skip gate in `src/miniexpr.c`.
- Flipped wasm runtime-cache cast behavior contract to require JIT kernel creation:
  - `tests/test_dsl_jit_runtime_cache.c` test 8c now expects `me_expr_has_jit_kernel(expr) == true`.

### Validation in this session

- wasm targeted tests:
  - `ctest --test-dir build-wasm32-fast --output-on-failure -R "test_nd|test_dsl_syntax|test_dsl_jit_runtime_cache|test_dsl_jit_side_module"` (pass)
- native targeted regression check:
  - `ctest --test-dir build --output-on-failure -R test_dsl_jit_runtime_cache` (pass)

## Remaining Tasks

### Phase 1: task-7 unblock status (completed)

Completed on 2026-02-19 (cross-repo validation):

- miniexpr contract-parity tests pass for the exact task-7 expression shape on native and wasm.
- miniexpr diagnostics are exposed through `me_get_last_error_message()`.
- python-blosc2 integration now uses wasm-safe dtype mapping for miniexpr setup and task-7 passes on wasm.

No additional Phase 1 actions are pending in miniexpr for task-7 unblock.

### Phase 2: full wasm runtime-JIT cast support (follow-up)

1. Root-cause the wasm backend invalid-module failure for cast-heavy kernels.
- Completed in this session (reproduced concrete truncation opcode/type mismatch in wasm module compile).

2. Implement wasm-safe cast lowering for runtime-JIT kernels.
- Completed in this session via wasm-patched cast helper-call lowering for:
  - `int(...)`
  - `float(...)`
  - `bool(...)`

3. Remove the temporary wasm cast-intrinsic runtime-JIT skip gate.
- Completed in this session.

4. Flip wasm tests from fallback-expectation to JIT-expectation.
- Completed in this session (`test_dsl_jit_runtime_cache` test 8c).

5. Expand wasm runtime-cache coverage back to full parity where feasible.
- Reconcile runtime-cache test assumptions that are host-file-cache specific vs wasm side-module behavior.
- Ensure wasm CI covers the relevant runtime-JIT cast lanes, not only interpreter fallback.

## Fast Iteration Notes

- A faster local loop was configured with:
  - `build-wasm32-fast`
  - `MINIEXPR_BUILD_BENCH=OFF`
  - `MINIEXPR_BUILD_EXAMPLES=OFF`
  - targeted test builds (`test_dsl_jit_runtime_cache`, `test_dsl_jit_side_module`, `test_dsl_syntax`, `test_nd`).
- This setup is recommended while working on wasm cast runtime-JIT internals.

## Validation Commands (suggested)

```bash
# native
cmake -S . -B build -DMINIEXPR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -R "test_nd|test_dsl_syntax|test_dsl_jit_runtime_cache"

# wasm32
emcmake cmake -S . -B build-wasm32 -DMINIEXPR_BUILD_TESTS=ON
cmake --build build-wasm32 -j
ctest --test-dir build-wasm32 --output-on-failure -R "test_nd|test_dsl_syntax|test_dsl_jit_runtime_cache"
```

## Exit Criteria

- Task-7 unblock criterion is met: `int(_i0 * _n1 + _i1)` works on wasm32 in miniexpr and python-blosc2 integration, with deterministic behavior and tests.
- Remaining completion criterion for this plan: runtime-JIT cast intrinsics on wasm32 achieve parity (remove fallback gate and require JIT-kernel expectations in wasm tests).
