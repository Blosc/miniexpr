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
  - `tcc` in-memory JIT requires explicit symbol registration
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
   - compile latency (`cc`, `tcc`)
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
   - `tcc` backend: register symbols with `tcc_add_symbol`.
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
   - forced `tcc` and default `cc` backends
3. Performance:
   - compile latency and warm runtime benchmarks across representative kernels
4. Numerical:
   - tolerance-based parity checks vs interpreter outputs

## Risk Register

1. ABI/symbol visibility issues between generated kernels and `libminiexpr`.
2. Backend divergence (`cc` vs `tcc`) causing inconsistent behavior.
3. Overly broad vector lowering may introduce silent numeric drift.
4. Packaging/runtime loader differences across environments.

## Recommended Execution Order

1. Phase 0
2. Phase 1
3. Re-evaluate ROI after parity bridge results
4. Phase 2 only if speedup justifies complexity
5. Phase 3 only if long-term DSL compiler investment is approved

## Execution Status (2026-02-09)

| Milestone | Status | Scope Delivered | Validation |
|---|---|---|---|
| Phase 0 baseline harness | Done | Added `bench/benchmark_dsl_jit_math_kernels.c` (`sin`, `exp`, `log`, `pow`, `hypot`, `atan2`, `sinpi`, `cospi`) with compile/warm/diff reporting and docs in `bench/README.md`. | Build + targeted tests pass; benchmark runnable in `cc`, forced `tcc`, and bridge modes. |
| Phase 1 parity bridge | Done | Alias rewrites (`arctan2`, `exp10`, `sinpi`, `cospi`, `logaddexp`, `where`), bridge declarations/definitions, runtime bridge env gate, tcc symbol registration, backend-tag cache separation. | `test_dsl_jit_codegen`, `test_dsl_jit_runtime_cache` pass; forced bridge runtime is stable. |
| Bridge runtime crash fix | Done | Fixed forced `tcc + bridge` crash by registering bridge symbols after `tcc_set_output_type`; added specific `tcc_add_symbol` error messages. | Reproduced crash, patched, validated via benchmarks and tests. |
| Phase 2 narrow lowering | Done | Selective lowering for simple element map unary kernels (`sinpi`, `cospi`, `exp10`) to bulk `me_jit_vec_*` calls. | Performance improved on matched kernels; tests updated. |
| Phase 2 expansion | Done | Added unary lowering for `sin`, `cos`, `exp`, `log` and binary lowering for `atan2`, `hypot`; added bridge exports for f64/f32 unary+binary vector entrypoints. | Codegen tests for unary+binary lowering pass; benchmark shows strong wins on most matched kernels. |
| Phase 2 binary `pow` lowering | Done | Added selective binary lowering for `pow(x, y)` and safe broadcast forms (`pow(x, c)`, `pow(c, x)`) to bulk `me_jit_vec_pow_*` bridge calls; registered tcc bridge symbols for f64/f32. | `test_dsl_jit_codegen` coverage includes map + broadcast `pow` lowering marker checks. |
| cc backend bridge-link path | Partial | Exposed real `me_jit_*` bridge symbols for `cc` runtime path and enabled bridge codegen when `ME_DSL_JIT_USE_SLEEF_BRIDGE=1` and symbols are dynamically visible; added cc-path runtime test with scalar fallback verification when bridge symbols are unavailable. | `test_dsl_jit_runtime_cache` includes cc bridge-path coverage; current local test binary falls back to scalar (no dynamic bridge symbols exported). |
| Unary affine lowering | Done | Added matcher/emitter support for `f(x + c)`, `f(x - c)`, `f(c + x)` using prepass + bulk vector call; added codegen test for `log(x + 1.5)`. | Tests pass; `log` improved vs non-bridge JIT but remains slower than interpreter in current benchmark. |

### Latest Metrics Snapshot (`nitems=262144`, `repeats=6`)

| Kernel | forced `tcc` ns/elem | forced `tcc + bridge` ns/elem | Notes |
|---|---:|---:|---|
| `sin` | ~7.77 | ~2.18 | improved |
| `exp` | ~5.83 | ~1.61 | improved |
| `log` (`log(x + 1.5)`) | ~5.39 | ~3.10 | improved vs non-bridge JIT, still slower than interpreter (~2.14) |
| `hypot` | ~2.78 | ~0.86 | improved |
| `atan2` | ~6.44 | ~2.78 | improved |
| `sinpi` | ~5.67 | ~4.37 | improved |
| `cospi` | ~5.86 | ~4.28 | improved |

### Open Items

1. Expand selective lowering coverage to remaining unary/binary/ternary math functions (`pow` map coverage is now included).
2. Add stronger pattern matching (constants, affine/broadcast forms) where safe (`pow` broadcast constants now covered).
3. Harden `cc` backend bridge-link path so bridge symbols are consistently visible in packaged/runtime link modes.
4. Keep extending benchmark matrix and parity tests as coverage grows.
