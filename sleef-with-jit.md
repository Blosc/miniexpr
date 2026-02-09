# SLEEF With JIT Plan

## Rationale

Using SLEEF-backed math from JIT kernels is not just a link step; it requires
lowering and runtime integration work. The key reasons are:

- Linking makes symbols visible, but does not change generated JIT code shape.
  Current JIT lowering is mostly scalar C loops.
- SLEEF speedups come from vector/batch dispatch paths, so codegen must emit
  bulk calls (or equivalent pre/post transforms), not per-element scalar calls.
- JIT code should bind to a stable miniexpr bridge ABI (`me_jit_*`,
  `me_jit_vec_*`) rather than SLEEF internals directly.
- Backends differ:
  - `cc` shared-object JIT can resolve linked symbols at load/link time.
  - `libtcc` in-memory JIT requires explicit symbol registration
    (`tcc_add_symbol`).
- Only some IR patterns are safe and profitable for vector lowering; guarded
  pattern detection and scalar fallback are required to preserve semantics.
- Expression forms such as affine arguments (`f(x + c)`) need extra generated
  transforms before vector calls.
- Runtime cache keys/versioning must include lowering behavior changes to avoid
  stale/incompatible artifacts.
- Tests and numerical parity checks are needed to keep interpreter/JIT behavior
  aligned across dtypes, FP modes, and backends.

## Goal

Evaluate and stage support for using the same SLEEF-backed math path from DSL JIT kernels, while preserving current fallback behavior and portability.

## Non-goals (for initial work)

- No Windows JIT backend changes in this plan.
- No full IR redesign in phase 1.
- No behavior changes for existing interpreter path.

## Current State Summary

- Interpreter path can reach SLEEF-backed vector dispatch via `functions-simd.c`.
- JIT C path lowers from expression text into scalar C loops.
- JIT compiled kernels currently resolve math via toolchain/libm symbols, not miniexpr vector dispatch.

## Phased Plan

## Phase 0: Baseline and Guardrails

1. Add benchmark/test baselines for representative math kernels (`sin`, `exp`, `log`, `pow`, `hypot`, `atan2`, `sinpi`, `cospi`).
2. Record per-backend metrics:
   - compile latency (`cc`, `libtcc`)
   - warm runtime throughput
   - numerical diffs vs interpreter
3. Add a feature flag placeholder: `ME_DSL_JIT_USE_SLEEF_BRIDGE` (default off).

Acceptance:
- Baselines are reproducible in CI for Linux/macOS.

## Phase 1: Function Parity Bridge (Low Risk)

1. Add exported bridge symbols in miniexpr for JIT use:
   - scalar wrappers (alias coverage: `sinpi`, `cospi`, `exp10`, `logaddexp`, `where`)
   - vector bridge entrypoints mapped to existing `vec_*_dispatch`.
2. Extend JIT backends to resolve host symbols:
   - `cc` backend: link generated shared object against `libminiexpr`.
   - `libtcc` backend: register symbols with `tcc_add_symbol`.
3. Keep current scalar codegen unchanged; only symbol resolution improves.

Acceptance:
- DSL JIT accepts and runs kernels with math aliases currently missing or fragile in JIT.
- No regression in existing JIT/interpreter tests.

## Phase 2: Selective Vector Lowering (Medium Risk)

1. Add pattern detection in codegen for simple element kernels:
   - pure map-style unary/binary expressions over contiguous arrays
   - no control-flow divergence (`break`/`continue`/nested loop side effects).
2. Emit vector bridge calls for matched blocks.
3. Keep scalar lowering as fallback for unmatched patterns.

Acceptance:
- Matched kernels call vector bridge symbols.
- Runtime speedup over scalar JIT on math-heavy kernels is measurable.
- Numeric parity with interpreter remains within current tolerances.

## Phase 3: Structured JIT IR Upgrade (Optional, Larger Scope)

1. Move from text-only expression payloads to typed expression nodes/opcodes.
2. Implement lowering passes:
   - op selection
   - vectorizability analysis
   - scalar fallback insertion.
3. Expand vector lowering coverage to broader DSL subsets.

Acceptance:
- Clear improvement in coverage and maintainability versus text-rewrite approach.

## Phase 4: Packaging and Runtime Integration

1. Ensure `python-blosc2` packaging includes required runtime artifacts on Linux/macOS:
   - `libminiexpr`
   - `libtcc`
   - `libtcc1.a`
2. Validate runtime loader behavior in wheel/install contexts.
3. Keep environment toggles documented for support/debug.

Acceptance:
- JIT fallback behavior is stable in local dev and packaged runtime environments.

## Testing Strategy

1. Unit:
   - symbol registration/linking success
   - JIT acceptance/rejection paths by dialect/pattern
2. Integration:
   - DSL kernels using math aliases and two-arg functions
   - forced `libtcc` and default `cc` backends
3. Performance:
   - compile latency and warm runtime benchmarks across representative kernels
4. Numerical:
   - tolerance-based parity checks vs interpreter outputs

## Risk Register

1. ABI/symbol visibility issues between generated kernels and `libminiexpr`.
2. Backend divergence (`cc` vs `libtcc`) causing inconsistent behavior.
3. Overly broad vector lowering may introduce silent numeric drift.
4. Packaging/runtime loader differences across environments.

## Recommended Execution Order

1. Phase 0
2. Phase 1
3. Re-evaluate ROI after parity bridge results
4. Phase 2 only if speedup justifies complexity
5. Phase 3 only if long-term DSL compiler investment is approved

## Execution Status (2026-02-08)

- Started Phase 1 with a low-risk parity slice.
- Added JIT codegen function-name rewrites for DSL aliases:
  - `arctan2 -> atan2`
  - `exp10 -> me_jit_exp10`
  - `sinpi -> me_jit_sinpi`
  - `cospi -> me_jit_cospi`
  - `logaddexp -> me_jit_logaddexp`
  - `where -> me_jit_where`
- Added headerless math function prototypes and helper wrapper definitions to generated JIT C source.
- Added test coverage in `tests/test_dsl_jit_codegen.c` (`test_codegen_math_alias_rewrite`) to verify rewrite markers.
- Validation run:
  - `cmake --build build -j4`
  - `ctest --test-dir build -R 'test_dsl_jit_codegen|test_dsl_jit_runtime_cache' --output-on-failure`
- Started Phase 0 baseline harness:
  - Added `bench/benchmark_dsl_jit_math_kernels.c` for representative math kernels
    (`sin`, `exp`, `log`, `pow`, `hypot`, `atan2`, `sinpi`, `cospi`).
  - Benchmark reports per-kernel compile latency, warm throughput, and max-abs diff vs interpreter.
  - Added usage docs in `bench/README.md`, including default, forced-libTCC, and bridge-mode runs.
- Validation/measurement snapshot (`nitems=131072`, `repeats=4`):
  - `cc` backend: cold compile latency per kernel roughly `300-590 ms`.
  - forced `libtcc`: cold compile latency per kernel roughly `0.4-2.0 ms`.
  - forced `libtcc + bridge`: cold compile latency per kernel roughly `0.3-6.2 ms`.
  - Numerical parity stayed tight in all runs (`max_abs_diff` up to `4.441e-16`).
- Bridge runtime fix:
  - Fixed a forced-libTCC+bridge crash by registering bridge symbols **after**
    `tcc_set_output_type` in the libtcc compilation flow.
  - Kept bridge symbol errors specific (`tcc_add_symbol failed for <symbol>`).
- Started Phase 2 with a narrow selective-lowering slice:
  - Added pattern detection for simple element map kernels of the form
    `return sinpi(x)`, `return cospi(x)`, and `return exp10(x)`.
  - In runtime bridge mode, codegen now emits one bulk vector bridge call
    (no per-element loop) when the pattern matches.
  - Added libtcc bridge exports for vector entrypoints:
    - `me_jit_vec_{sinpi,cospi,exp10}_{f64,f32}`
  - Added codegen test coverage:
    - `test_codegen_runtime_math_bridge_vector_lowering`
- Phase 2 slice benchmark check (`nitems=262144`, `repeats=6`):
  - forced `libtcc`: `sinpi ~5.99 ns/elem`, `cospi ~6.03 ns/elem`
  - forced `libtcc + bridge`: `sinpi ~4.35 ns/elem`, `cospi ~4.30 ns/elem`
  - Result: bridge path moved from slower-than-interpreter to near-parity for
    these matched kernels; non-matched kernels remain unchanged.
- Expanded Phase 2 selective lowering slice:
  - Unary map patterns now include `sin`, `cos`, `exp`, `log`, `exp10`,
    `sinpi`, `cospi`.
  - Binary map patterns now include `atan2` and `hypot`.
  - Added vector bridge symbols and libtcc registration for:
    - unary: `me_jit_vec_{sin,cos,exp,log,exp10,sinpi,cospi}_{f64,f32}`
    - binary: `me_jit_vec_{atan2,hypot}_{f64,f32}`
  - Added codegen test coverage for:
    - unary vector lowering (`exp`)
    - binary vector lowering (`atan2(y, x)`)
- Expanded benchmark snapshot (`nitems=262144`, `repeats=6`):
  - forced `libtcc`:
    - `sin ~4.15 ns/elem`, `exp ~4.35`, `hypot ~3.05`, `atan2 ~7.10`
    - `sinpi ~6.25`, `cospi ~6.33`
  - forced `libtcc + bridge`:
    - `sin ~2.29 ns/elem`, `exp ~1.62`, `hypot ~0.89`, `atan2 ~2.89`
    - `sinpi ~4.55`, `cospi ~4.28`
  - Observation: strong improvements on most matched kernels; `log` remained
    slower in this snapshot (`~4.64 ns/elem` vs `~4.61` for non-bridge JIT and
    `~2.11` interpreter), so it should be treated as provisional.
- Unary affine lowering extension:
  - Extended unary matcher to accept simple affine forms:
    - `f(x + c)`, `f(x - c)`, `f(c + x)` for matched unary functions.
  - Implemented codegen path as:
    - prepass: `out[i] = in[i] + c`
    - bulk vector call: `vec_f(out, out, nitems)`
  - Added codegen test coverage for `log(x + 1.5)`.
- Affine `log` re-check (`nitems=262144`, `repeats=6`):
  - forced `libtcc`: `log ~4.59 ns/elem`
  - forced `libtcc + bridge`: `log ~4.48 ns/elem`
  - Result: slight improvement from bridge vector lowering with affine args, but
    still slower than interpreter in this benchmark (`~2.11 ns/elem`).
