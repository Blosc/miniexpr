# wasm32 TODO: DSL `int(...)` Cast Path

Last updated: 2026-02-19

## Context

In `python-blosc2` on wasm32/Pyodide, DSL kernels that use index symbols with `int(...)` casting still fail in practice for this shape:

- DSL kernel: `return int(_i0 * _n1 + _i1)`
- Typical integration symptom: `DSL kernels require miniexpr ... miniexpr compilation or execution failed`
- Current status in python-blosc2: expected-to-fail behavior is still kept for wasm32.

This is likely either:
- a true miniexpr wasm32 cast/runtime gap, or
- an integration mismatch between python-blosc2's prefilter setup and miniexpr's supported dtype/cast combinations.

## Goal

Make DSL `int(...)` casting reliable on wasm32 for ND/index-symbol expressions (or explicitly declare and document it as unsupported with precise diagnostics).

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

- `int(_i0 * _n1 + _i1)` works on wasm32 in miniexpr and in python-blosc2 integration, with deterministic behavior and tests.
- OR behavior is explicitly unsupported, documented, and enforced by a precise, tested error path.
