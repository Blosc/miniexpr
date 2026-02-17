# Plan: External JS Glue (miniexpr scope only)

## Purpose

Track only the work that belongs in miniexpr for enabling wasm32 JIT when
consumed from side-module environments (for example Pyodide extensions).

This plan is derived from `../python-blosc2/plans/external-js-glue.md`, but
excludes python-blosc2 implementation details.

## Scope In

- JS glue extraction into miniexpr-managed file(s).
- Side-module-safe runtime call path in miniexpr (no direct `EM_JS` use).
- Public C API needed for host registration of JS helpers.
- Build and test coverage inside miniexpr.

## Scope Out

- Python-side loading of JS glue.
- Pyodide runtime-object discovery.
- Cython wrappers and wheel packaging in python-blosc2.

## Action Items

### 1. Extract JS glue into standalone source

- [x] Add `src/me_jit_glue.js` with host-callable functions:
  - `_meJitInstantiate(...)`
  - `_meJitFreeFn(...)`
- [x] Make glue runtime-agnostic by receiving a `runtime` object instead of
      assuming Emscripten globals as free variables.
- [x] Add a minimal syntax/smoke test (Node-compatible) that validates the
      glue file loads and basic entry points exist.

### 2. Add side-module-safe runtime indirection in miniexpr

- [x] In `src/miniexpr.c`, add function pointer types and static slots for:
  - instantiate helper
  - free helper
- [x] Add public registration function:
  - `me_register_wasm_jit_helpers(...)`
- [x] Gate current `EM_JS` path to main-module builds only
      (`ME_USE_WASM32_JIT && !ME_WASM32_SIDE_MODULE`).
- [x] In side-module builds (`ME_WASM32_SIDE_MODULE`):
  - call registered helper pointer for instantiate/free paths
  - gracefully skip JIT if helpers are not registered (`NULL` pointers)

### 3. Expose and document API contract

- [x] Declare `me_register_wasm_jit_helpers(...)` in `src/miniexpr.h`.
- [x] Document expected helper signatures and behavior (success/error codes,
      ownership/lifetime expectations).
- [x] Document fallback behavior when helpers are not registered.

### 4. Build-system integration

- [x] Ensure `src/me_jit_glue.js` is tracked as a first-class source artifact
      in miniexpr (install/package path only if miniexpr distributes it).
- [x] Add/adjust compile definitions for side-module gating, with clear CMake
      options and defaults.

### 5. miniexpr tests

- [x] Keep existing wasm32/main-module JIT behavior/tests passing.
- [x] Add a side-module-focused test target that:
  - simulates helper registration
  - exercises instantiate/free via indirect helper pointers
  - verifies clean fallback when helpers are not registered
- [x] Add regression test for helper-registration missing path:
  - JIT skipped
  - interpreter path remains correct

## Suggested Delivery Order

1. Land API + indirection in miniexpr (`src/miniexpr.c`, `src/miniexpr.h`).
2. Land `src/me_jit_glue.js`.
3. Land build wiring and tests.
4. Coordinate downstream adoption (python-blosc2) after miniexpr API is stable.

## Acceptance Criteria (miniexpr)

- Side-module builds compile without `EM_JS` linkage issues.
- JIT path can be driven through registered helper pointers.
- Missing-helper configuration degrades to interpreter execution without
  correctness regressions.
- Existing non-side-module behavior remains unchanged.
