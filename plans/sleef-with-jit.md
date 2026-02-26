# Plan: Comprehensive SLEEF Integration for Runtime JIT

Last updated: 2026-02-26

## Context

Current status is a working bridge path where JIT-generated kernels can call runtime math bridge symbols (`me_jit_*`), and those bridge functions dispatch through miniexpr's SIMD/SLEEF machinery.

This is a good functional baseline, but it is not yet a full-performance design for all DSL/JIT math-heavy kernels.

## Goal

Make runtime JIT and SLEEF integration comprehensive across correctness, performance, and observability:

- Preserve strict semantics by default.
- Maximize SIMD/SLEEF use in JIT execution.
- Keep behavior deterministic across interpreter/JIT/native/wasm where feasible.
- Provide clear diagnostics and rollout controls.

## Non-goals (for this plan)

- Rewriting the whole JIT backend around LLVM/MLIR.
- Introducing behavior-changing fast-math defaults.
- Solving every transcendental in one phase.

## Problem Summary

The current scalar-bridge model fixes the main functional gap (JIT path no longer bypasses miniexpr runtime dispatch), but it still has structural limits:

1. Per-element scalar calls in JIT loops can leave performance on the table.
2. JIT codegen does not yet systematically emit vector-bridge calls for math DAGs.
3. Capability reporting is coarse (trace says SIMD init happened, but not exactly what each kernel used).
4. Cache/fingerprint and lowering policy controls need to be explicit as bridge behavior expands.

## Target Architecture

### A. Bridge ABI layers

1. Scalar bridge (already present)
- `me_jit_abs/sin/cos/exp/log/sqrt/...`
- Always available fallback.

2. Vector bridge (expand and standardize)
- Unary f64/f32: `me_jit_vec_<fn>_f64/f32`
- Binary f64/f32: `me_jit_vec_<fn>_f64/f32` with two inputs
- Selected ternary forms (`where`, fused helpers) as needed
- Stable naming/ABI contract for codegen and tests

### B. JIT lowering modes

1. Scalar lowering: always legal fallback.
2. Vector lowering: chosen when purity/shape/alignment constraints pass.
3. Hybrid lowering: vectorize safe subexpressions, scalar for the rest.

### C. Runtime capability model

Expose a compact capability struct/API (or equivalent internal contract) that includes:

- backend type (scalar/advsimd/avx2/avx512/etc)
- sleef enabled yes/no
- supported vector bridge ops
- strict vs relaxed mode flags

JIT codegen can use this for guarded lowering decisions and trace output.

## Phased Implementation Plan

## Progress snapshot (2026-02-25)

- Phase 0: mostly complete.
  - Done: bridge ABI contract header, signature/static checks, bridge ABI version in runtime cache metadata, cache rejection on ABI mismatch.
  - Done: codegen/runtime bridge declaration wiring moved to shared symbol contract.
- Phase 1: partially complete.
  - Done: direct vector lowering for simple single-return kernels (unary and selected binary), scalar fallback path remains automatic.
  - Done: staged assign+return pattern lowering (`tmp = fn(...); return tmp`) for supported unary/binary vector calls.
  - Done: unary vector lowering set includes baseline (`sin/cos/exp/log/sqrt/abs`) and extended unary bridge coverage.
  - In progress: runtime wiring for expression/lifted hybrid vector lowering is env-gated via `ME_DSL_JIT_HYBRID_EXPR_VEC_MATH=0|1` (default off).
  - In progress: branch-aware simple `if` assign/else lowering to branchless select form is env-gated via `ME_DSL_JIT_BRANCH_AWARE_IF=0|1` (default on).
  - Pending: general subexpression vector-lowering pass (current lowering is pattern-based, not full graph/hybrid).
- Phase 2: in progress.
  - Done: binary vector lowering for `atan2`, `hypot`, `pow`.
  - Done: binary vector lowering expanded to `fmax`/`fmin` where runtime dispatch already exists.
  - Done: selected special unary functions (`exp10`, `log1p`, `expm1`, `sinpi`, `cospi`, and others already exposed).
  - Done: mixed graph/broadcast support generalized for binary vector ops (constant + parameter) across supported binary set.
- Phase 3: partially complete.
  - Done: runtime bridge chunk-size heuristic via `ME_DSL_JIT_VEC_CHUNK_ITEMS` (default chunk cap) to avoid oversized bridge chunks.
  - Done: initial hybrid expression cost gating (plan-pass and expression/lifted-plan budgets, stricter when control flow is present) to avoid over-lowering regressions.
  - Pending: deeper temp-buffer scheduling/reuse and fused lowering.
- Phase 4: partially complete.
  - Done: explicit env-controlled default FP mode via `ME_DSL_FP_MODE` with `strict|contract|fast|relaxed` (`relaxed` mapped to fast), while pragma still takes precedence.
  - Pending: formal per-function ULP acceptance matrix for relaxed mode.
- Phase 5: mostly complete.
  - Done: per-kernel lowering diagnostics now reported (mode, vector op list, reason) from codegen into runtime trace.
  - Done: reason codes for non-vectorized cases (`runtime-math-bridge-disabled`, `vector-math-disabled`, `no-vector-lowering-match`).
- Phase 6: mostly complete.
  - Done: feature gates wired: `ME_DSL_JIT_MATH_BRIDGE` and `ME_DSL_JIT_VEC_MATH`.
  - Done: cache-key differentiation includes bridge/vector gate state.
  - Pending: release policy choice for default-on/default-off vector gate.
- Phase 7: expanded.
  - Done: codegen/runtime-cache test coverage added for new gates, diagnostics, staged lowering, and generalized binary broadcast lowering.
  - Done: math benchmark kernel set expanded (`fmax`, `fmin`, `black_scholes_like`).

## Phase 0: Baseline hardening

1. Freeze bridge symbol list v1 and document it in-code.
2. Add compile-time/static checks for bridge signature mismatches.
3. Ensure codegen version/fingerprint includes bridge ABI version.

Done criteria:
- Rebuilds cannot silently use stale JIT cache after bridge ABI changes.

## Phase 1: Vector-lowering infrastructure in codegen

1. Add a JIT math-lowering pass in `dsl_jit_cgen.c`:
- classify subexpressions as vectorizable (pure, elementwise, supported fn set)
- emit temp buffers for staged vector evaluation where needed

2. Support unary vector lowering first:
- `sin/cos/exp/log/sqrt/abs` for f64 and f32

3. Preserve strict evaluation order and branch semantics.

Done criteria:
- Generated C for eligible kernels contains `me_jit_vec_*` calls.
- Unsupported patterns auto-fallback to scalar lowering without failing compilation.

## Phase 2: Broaden function/operator coverage

1. Add binary vector lowering set:
- `pow`, `atan2`, `hypot`, min/max where already exposed in runtime layer

2. Add selected special functions:
- `exp10`, `log1p`, `expm1`, `sinpi`, `cospi` when stable

3. Add mixed graph support:
- vector + scalar constants/parameters with broadcast handling

Done criteria:
- Coverage matrix documented and tested by dtype/function/backend.

## Phase 3: Optimization and memory discipline

1. Reduce temporary-buffer pressure:
- expression scheduling to reuse scratch buffers
- avoid materializing intermediates when direct streaming is possible

2. Add chunk-size heuristics:
- pick vector block sizes balancing cache locality and call overhead

3. Optional fused lowering patterns:
- e.g. simple FMA-like chains where semantics remain strict-compatible

Done criteria:
- Throughput improvement over scalar-bridge baseline on representative kernels.

## Phase 4: Correctness/ULP policy and fast-math lane

1. Keep strict mode as default behavior.
2. Add explicit opt-in relaxed mode gate (env + API flag), if desired:
- e.g. `ME_DSL_FP_MODE=relaxed`

3. Define acceptance thresholds by function:
- strict: bitwise/parity where expected
- relaxed: ULP budgets per function/backend

Done criteria:
- Documented precision contract and tests per mode.

## Phase 5: Observability and diagnostics

1. Extend `ME_DSL_TRACE` per-kernel reporting with:
- lowering mode selected: scalar/vector/hybrid
- vector ops actually used (function list)
- backend + sleef usage confirmation for that kernel

2. Add optional verbose trace level:
- reason codes for non-vectorized subexpressions

Done criteria:
- One trace run is enough to answer "did this kernel use SLEEF/SIMD and where?"

## Phase 6: Rollout controls

1. Feature gates:
- `ME_DSL_JIT_MATH_BRIDGE=1` (existing/implicit baseline)
- `ME_DSL_JIT_VEC_MATH=0/1` for vector lowering
- optional per-function gate for quick bisect

2. Safe default rollout:
- start with vector lowering off-by-default in one release cycle
- enable-by-default after perf/correctness confidence

3. Maintain clean fallback chain:
- vector lowering fail -> scalar lowering
- JIT fail -> interpreter

Done criteria:
- No regressions in correctness when toggling gates.

## Phase 7: Test and benchmark expansion

1. Unit/codegen tests
- assert symbol rewrites and emitted bridge calls in generated C
- ensure cache invalidation on ABI/cgen version changes

2. Runtime parity tests
- interpreter vs JIT across random seeds and edge domains
- scalar-only backend vs SIMD/SLEEF backend consistency

3. Performance tests
- microbenchmarks for unary/binary math
- workload benchmarks (e.g. Black-Scholes-like kernels)
- report speedup vs:
  - interpreter
  - JIT scalar bridge
  - JIT vector bridge

Done criteria:
- Repeatable benchmark suite with pass/fail guardrails for regressions.

## Concrete code touchpoints (expected)

- `src/dsl_jit_cgen.c`
  - lowering pass, vector call emission, reason diagnostics
- `src/miniexpr.c`
  - bridge registration, runtime capability checks, cache metadata/versioning
- `src/functions-simd.c`
  - vector dispatcher coverage, capability data, trace details
- tests
  - `tests/test_dsl_jit_codegen.c`
  - `tests/test_dsl_jit_runtime_cache.c`
  - `tests/test_simd_math.c`
  - new perf harness under `bench/` if needed

## Risks and mitigations

1. Risk: semantic drift in vector lowering (branches/NaN/sign-zero behavior).
- Mitigation: strict-mode-first policy + exhaustive edge-case tests.

2. Risk: cache instability from evolving lowering choices.
- Mitigation: explicit ABI/cgen versioning in cache key + metadata checks.

3. Risk: temporary-buffer overhead erases SIMD gains.
- Mitigation: phased scheduler improvements and benchmark-driven tuning.

4. Risk: backend-specific differences become opaque.
- Mitigation: richer per-kernel trace and capability reporting.

## Suggested execution order

1. Phase 0 + Phase 5 (baseline observability/versioning)
2. Phase 1 (unary vector lowering)
3. Phase 7 initial benchmarks/parity loop
4. Phase 2 coverage expansion
5. Phase 3 optimization
6. Phase 4 optional relaxed-mode lane
7. Phase 6 default-on rollout

## Exit criteria

This plan is complete when all are true:

1. JIT kernels consistently use vector bridge for eligible math subgraphs.
2. Trace clearly reports actual per-kernel SIMD/SLEEF usage and lowering reasons.
3. Interpreter/JIT parity is validated under strict mode.
4. Benchmarks show sustained gains for math-heavy kernels over scalar-bridge baseline.
5. Rollout gates allow safe enable/disable without functional regressions.

## 2026-02-26 Implementation Plan (cc backend focus)

1. Wire runtime gate for expression/lifted hybrid lowering (safe default off).
- Add `ME_DSL_JIT_HYBRID_EXPR_VEC_MATH=0|1` gate.
- Include effective gate state in JIT runtime cache keying to prevent stale cache reuse across toggles.
- Surface gate state in trace output.

2. Expand runtime/cache validation.
- Add runtime-cache tests that toggle expression-hybrid gate and verify:
  - generated C differs for hybrid-lifted patterns when gate is enabled
  - cache metadata/key separation across gate states
  - numerical parity remains unchanged

3. Benchmark on Black-Scholes-like kernels with fixed env matrix.
- Measure `cc` backend with:
  - `ME_DSL_JIT_VEC_MATH=1`, `ME_DSL_JIT_SCALAR_MATH_BRIDGE=0|1`
  - `ME_DSL_JIT_HYBRID_EXPR_VEC_MATH=0|1`
- Capture lowering mode/op markers from trace and generated source.
- Treat wins as backend-dependent (SIMD/SLEEF on vs off) and keep default off until stable.

4. Follow-up lowering work if gate shows wins.
- Improve pattern coverage for nested expressions with temp reuse.
- Add branch/`where`-style lowering candidates for `if`-heavy kernels.
- Re-evaluate default-on policy after repeatable perf gains and parity coverage.
