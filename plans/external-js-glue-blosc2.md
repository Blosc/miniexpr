# Plan: python-blosc2 Next Steps for External JS Glue (Pyodide side-module)

## Difficulty Assessment

Testing the integration should be moderate, not high risk.

The main complexity is not JIT logic itself, but wiring between:
- Pyodide JS runtime objects (`wasmMemory`, `wasmTable`, stack helpers)
- JS `addFunction` trampolines
- C-level `me_register_wasm_jit_helpers(...)`

Once helper registration is in place, behavior should be straightforward to validate.

## Current Baseline

miniexpr side is now ready:
- `src/me_jit_glue.js` exists and is runtime-agnostic
- `me_register_wasm_jit_helpers(...)` is public
- side-module mode (`ME_WASM32_SIDE_MODULE`) supports helper-driven JIT and clean fallback
- side-module JIT/fallback tests pass in wasm32 builds

## Next Steps (python-blosc2)

### 1. Pin and consume updated miniexpr

- [ ] Update python-blosc2 to a miniexpr revision that includes:
  - `me_register_wasm_jit_helpers(...)`
  - `src/me_jit_glue.js`
  - side-module wasm32 helper path
- [ ] Verify wasm/Pyodide build config enables side-module mode for miniexpr usage.

### 2. Expose helper registration in Cython extension

- [ ] Add Cython declaration for `me_register_wasm_jit_helpers(...)`.
- [ ] Add an internal wrapper accepting integer function pointers (from JS `addFunction`).
- [ ] Cast pointer integers to expected function-pointer types in Cython/C.
- [ ] Keep this API internal/private to python-blosc2 runtime initialization.

### 3. Package and load JS glue in python-blosc2

- [ ] Include `me_jit_glue.js` in wheel/package data for Pyodide builds.
- [ ] On wasm import path, load and evaluate `me_jit_glue.js` exactly once.
- [ ] Ensure loading is idempotent (safe on repeated imports/reloads).

### 4. Build runtime object and register helpers

- [ ] In Pyodide initialization path, construct runtime object with required members:
  - `HEAPF32`, `HEAPF64`, `HEAPU8`
  - `wasmMemory`, `wasmTable`
  - `addFunction`, `removeFunction`
  - `stackSave`, `stackAlloc`, `stackRestore`
  - `lengthBytesUTF8`, `stringToUTF8`
  - `err`
- [ ] Create JS wrappers that call:
  - `_meJitInstantiate(runtime, wasmBytes, bridgeLookupFnIdx)`
  - `_meJitFreeFn(runtime, idx)`
- [ ] Register those wrappers via `addFunction` and pass resulting pointers to
      `me_register_wasm_jit_helpers(...)`.
- [ ] Keep returned function-pointer indexes alive for process lifetime.

### 5. Add integration tests in python-blosc2

- [ ] Add a wasm/Pyodide integration test that confirms helper registration succeeds.
- [ ] Add a test that triggers DSL JIT-capable expression path and verifies correct output.
- [ ] Add fallback test: skip/disable helper registration and verify interpreter correctness.
- [ ] Add an assertion/signal for JIT-path execution in wasm mode (trace marker or explicit counter).

### 6. CI updates

- [ ] Update wasm CI job to run the new integration tests.
- [ ] Ensure CI artifact includes `me_jit_glue.js` in installed package contents.
- [ ] Record expected behavior in CI logs:
  - with helpers: runtime JIT path active
  - without helpers: clean fallback, no functional regression

## Suggested Order

1. Miniexpr version bump in python-blosc2.
2. Cython wrapper for registration API.
3. JS glue packaging + one-time loader.
4. Runtime object and `addFunction` pointer registration.
5. Integration tests.
6. CI hardening and log assertions.

## Acceptance Criteria

- Pyodide side-module imports register JIT helpers successfully.
- JIT-capable expressions execute correctly with helpers registered.
- Missing-helper path falls back to interpreter with correct outputs.
- wasm CI covers the Pyodide integration path end-to-end.
