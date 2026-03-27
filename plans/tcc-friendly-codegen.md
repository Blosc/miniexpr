# Plan: TCC-Friendly DSL JIT C Codegen

Last updated: 2026-03-02

## Checkpoint (completed)

- Kept only one proven low-risk optimization in active codegen:
  - for native `tcc`, emit `int nitems` and `for (int idx ...)` in the generated kernel entry/outer loop.
- Removed inactive experimental scaffolding from `src/dsl_jit_cgen.c`:
  - removed square-alias temporary extraction machinery and related context/options plumbing.
  - removed unused `enable_tcc_friendly_codegen` option from `me_dsl_jit_cgen_options`.
- Rationale:
  - keeps the codebase small and reviewable,
  - avoids carrying dead optimization paths,
  - preserves the measured Mandelbrot `tcc` speedup from the `nitems/idx` width change.

## Goal

Improve `tcc` JIT runtime performance for DSL kernels by emitting C that is easier for `minicc`/`tcc` to optimize, without changing DSL semantics or affecting `cc` codegen quality.

## Primary decision

Implement the low-risk path in `miniexpr` codegen, not in `minicc`.

Rationale:

- `miniexpr` already has compiler-specific JIT behavior (`tcc` vs `cc`).
- Scope stays limited to DSL JIT kernels.
- Lower regression risk than changing global C compilation behavior in `minicc`.
- Easier A/B benchmarking and rollback.

## Non-goals

- No global optimizer project in `minicc` (no SSA/global RA/LICM in compiler core).
- No DSL semantic changes.
- No architecture-specific codegen forks in this phase.

## Proposed scope

Continue with small, measurable source-shaping steps in DSL C codegen, gated to `tcc` through existing backend checks in `miniexpr` when needed.

Initial transforms (low risk):

1. Local temporary extraction for repeated arithmetic subexpressions inside basic blocks.
2. Expression linearization to reduce tree depth and register pressure.
3. Loop-invariant hoisting when source variables are loop-invariant by construction.
4. Explicit casts and typed constants to reduce implicit conversion churn.
5. Conservative control-flow simplification where it does not alter floating-point edge behavior.

## Guardrails

- Preserve strict/contract/fast FP mode behavior exactly.
- Preserve NaN/Inf behavior in strict mode.
- Do not reorder expressions if DSL semantics could observe side effects.
- Keep current `cc` path unchanged unless explicitly enabled.
- Preserve fallback behavior: if JIT generation/compile fails, interpreter path still works.

## Implementation phases

### Phase 0: Baseline and observability

- Capture baseline benchmark numbers for representative DSL kernels:
  - Mandelbrot
  - Black-Scholes
  - simple arithmetic-heavy kernels
- Extend/confirm trace logging for codegen mode and lowered shape.
- Document current `tcc` vs `cc` gaps for warm eval (`ns_per_elem`).

### Phase 1: Landed low-risk baseline

- Completed: `tcc`-native `int` loop/index width for kernel `nitems` and element loop index.
- Completed: removed inactive tcc-friendly temporary-rewrite scaffold.
- Result target met: measurable warm-eval improvement with no semantic change.

### Phase 2: Next pass candidates (ranked)

1. Reduce generated casts in the hottest scalar loop path where source/target dtype already match.
2. Emit narrower integer temporaries in loop-local control helpers when values are proven bounded.
3. Hoist repeated parameter pointer loads to loop preheader when safe and already immutable.
4. Re-evaluate minimal common-subexpression extraction only if it can be done structurally (IR-level), not string-rewrite.

### Phase 2 status (current)

- Attempted low-risk source-shaping experiments:
  - selective cast elision,
  - boolean condition simplification,
  - pointer-cursor scalar loop shaping,
  - register-hint shaping,
  - default `tcc` option fallback tuning.
- Result: no stable, repeatable Mandelbrot warm-eval improvement under current measurement noise.
- Action taken: reverted those exploratory changes and kept only the already-validated baseline (`tcc` native `int nitems/idx` path).

### Phase 3: Loop-focused shaping

- Hoist proven loop-invariant expressions out of generated loops.
- Normalize loop body to reduce repeated loads and repeated casts.
- Keep transformations local and conservative.

### Phase 4: Validation and rollout

- Add correctness tests comparing:
  - interpreted DSL vs `tcc` JIT output
  - interpreted DSL vs `cc` JIT output
- Add benchmark checks for regression visibility.
- Roll out behind environment gate first, then consider default-on for `tcc`.

## Validation plan

Correctness:

- Existing DSL syntax/runtime tests.
- New targeted tests for transformed patterns:
  - repeated subexpressions
  - loop invariants
  - strict-mode FP edge cases
  - mixed integer/float temporaries

Performance:

- `bench/benchmark_dsl_jit_mandelbrot`
- `bench/benchmark_black-scholes`
- at least one synthetic arithmetic stress kernel

Acceptance target for first rollout:

- measurable `tcc` warm-eval speedup (target range: 15-35% on arithmetic-heavy kernels),
- no correctness regressions,
- no `cc` performance regression.

## Risks and mitigations

- Risk: semantic drift from aggressive rewrites.
  - Mitigation: conservative transform rules and differential tests vs interpreter.
- Risk: code size growth from extra temporaries.
  - Mitigation: cap temp extraction heuristics and benchmark compile/eval tradeoff.
- Risk: `cc` path accidental impact.
  - Mitigation: strict option gating and dedicated tests for both compiler modes.

## Rough effort

- Phase 0-1: 2-3 days
- Phase 2: 1-2 weeks
- Phase 3: 1 week
- Phase 4: 3-5 days

Total: about 3-5 weeks for a solid low-risk first iteration.

## Future extensions

- Optional lightweight post-pass C rewriter for `tcc` kernels if codegen-side transforms are insufficient.
- Revisit selective `minicc` internal improvements only after measuring ceiling of source-shaping approach.
