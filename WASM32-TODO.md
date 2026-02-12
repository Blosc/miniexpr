# WASM32 Test Wrap-Up And Next Steps

## Current CI Scope (wasm32)

- `minicc-wasm` lane:
  - Builds `wasm32-tcc` from minicc.
  - Runs minicc wasm backend tests (`minicc.wasm.*`).
  - Status: passing.

- `miniexpr-wasm32` lane:
  - Builds full `miniexpr` with Emscripten (`emcmake`, `node` as emulator).
  - Runs full wasm32-compatible `miniexpr` test suite.
  - Status: passing (`26/26`).

## Fixes Applied During wasm32 Bring-Up

- Resolved duplicate output rule (`libminiexpr.a`) by setting:
  - `MINIEXPR_BUILD_SHARED=OFF`
  - `MINIEXPR_BUILD_STATIC=ON`

- Removed Emscripten warning about unsupported shared test stub:
  - `me_jit_test_stub` is `STATIC` on Emscripten.

- Fixed wasm32 failures in tests:
  - `test_function_order`:
    - Embedded `src/functions.c` into wasm FS.
    - Added fallback open path `/src/functions.c` for Emscripten.
  - `test_nd`:
    - Added Emscripten link flags for test executables:
      - `-sALLOW_MEMORY_GROWTH=1`
      - `-sINITIAL_MEMORY=134217728`

- Added wasm trace visibility:
  - Introduced `MINIEXPR_DSL_TRACE_DEFAULT` compile option.
  - Enabled it in wasm CI so `[me-dsl]` traces are emitted by default.
  - Trace step now prints a compact normalized summary.

## Experimental wasm libtcc Runtime JIT Status

- Added `MINIEXPR_ENABLE_WASM_LIBTCC_JIT` (experimental, Emscripten-only).
- Added `MINIEXPR_BUILD_BUNDLED_LIBTCC` switch and disabled bundled libtcc in wasm CI to avoid unstable cross-build failures.
- Added CI configure sanity checks for:
  - `MINIEXPR_ENABLE_WASM_LIBTCC_JIT=ON`
  - `MINIEXPR_BUILD_BUNDLED_LIBTCC=OFF`
  - `MINIEXPR_USE_LIBTCC_FALLBACK=ON`

### What trace shows now

- JIT IR generation works (many `jit ir built` events).
- Runtime JIT does not activate on wasm32.
- Runtime skip reason:
  - `failed to load libtcc shared library: dynamic linking not enabled`

## Interpretation

- wasm32 support for `miniexpr` core + DSL interpreter path is healthy.
- Runtime libtcc JIT on wasm32 is currently blocked by dynamic loading constraints in this Emscripten mode.

## Suggested Next Experiments

1. Decide target mode for wasm32 release behavior.
- Option A: interpreter-only DSL on wasm32 (stable, simplest).
- Option B: true runtime libtcc JIT on wasm32 (experimental, higher complexity).

2. If choosing interpreter-only for now:
- Set `MINIEXPR_ENABLE_WASM_LIBTCC_JIT=OFF` in wasm CI.
- Keep trace summary and assert expected skip reasons.
- Add a CI assertion that no `jit runtime built` appears for wasm32.

3. If pursuing runtime JIT on wasm32:
- Prototype an Emscripten dynamic-linking build:
  - Main module with dynamic linking enabled.
  - libtcc built/packaged as a loadable side module.
- Rework loader path for Emscripten-specific dynamic loading semantics.
- Add a dedicated wasm JIT smoke assertion that expects `jit runtime built: ... compiler=tcc`.

4. Improve trace usability:
- Keep compact summary in logs.
- Optionally upload `build-wasm32/dsl-trace.log` as CI artifact for debugging.

5. Add wasm performance guardrails:
- Record wall-clock for `ctest` and selected heavy tests.
- Track memory usage trend for `test_nd` to catch regressions early.

## Useful Reference Signals

- Healthy interpreter-mode wasm run:
  - all wasm tests pass
  - trace includes `jit ir built`
  - trace includes `jit runtime skip ... dynamic linking not enabled`

- Healthy runtime-JIT-mode wasm run (future target):
  - trace includes `jit runtime built: ... compiler=tcc`
  - no fallback-only behavior for targeted smoke kernels
