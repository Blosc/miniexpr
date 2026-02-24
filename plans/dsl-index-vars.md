# Plan: JIT Support for DSL Reserved Index Variables

Last updated: 2026-02-24

## Goal

Enable runtime JIT for DSL kernels that reference reserved index symbols:

- `_i0`..`_i7`
- `_n0`..`_n7`
- `_ndim`
- `_global_linear_idx`

with behavior matching interpreter semantics for regular and ND/padded evaluation.

## Status snapshot

- Phase 0 (metadata groundwork): done
- Phase 1 (IR/codegen acceptance): done
- Phase 2 (runtime binding channel v1): done
- Phase 3 (ND/padding correctness hardening): done for current coverage
- Phase 4 (optional synthesis optimization): partial (non-ND + ND synthesis under current ABI; wasm ND synthesis still disabled)

## Implemented so far

### Phase 0: metadata groundwork (done)

- Added explicit JIT binding metadata (`kind`, `dim`, `var_index`) in `me_dsl_compiled_program`.
- Preserved deterministic parameter order: user params first, then reserved params.
- Included binding metadata in runtime cache key/fingerprint and cache metadata.
- Added `compile_ndims` metadata for compile-time context.

### Phase 1: IR/codegen acceptance (done)

- JIT IR now includes reserved symbols used by the DSL program.
- C codegen accepts reserved symbols as params.
- Runtime JIT path is no longer blanket-disabled for reserved index vars.
- Added rollout eligibility gate:
  - `ME_DSL_JIT_INDEX_VARS=0` disables JIT for reserved-index DSL kernels.
  - default behavior keeps this enabled.

### Phase 2: runtime binding channel v1 (done)

- Runtime dispatch builds JIT inputs from binding metadata:
  - user bindings from `vars_block`
  - reserved bindings from existing reserved buffers (`_i*`, `_n*`, `_ndim`, `_global_linear_idx`)
- Existing interpreter reserved-buffer preparation remains the source of truth.
- JIT remains best-effort with clean interpreter fallback on JIT runtime failure.

### Phase 3: ND/padding correctness hardening (done for current scope)

- Reused current ND semantics:
  - no-padding path (`valid_items == padded_items`)
  - packed/scatter path for padded blocks
- `_global_linear_idx` continues to map to global C-order positions as in interpreter path.
- No behavior change to padded write handling (existing zero-fill/scatter preserved).

### Phase 4: optional synthesis optimization (partial)

- Implemented non-ND synthesis in codegen while keeping the existing kernel ABI:
  - `_i0 = idx`, `_i(d>0) = 0`
  - `_n0 = nitems`, `_n(d>0) = 1`
  - `_ndim = 1`
  - `_global_linear_idx = idx`
- Implemented ND synthesis (native runtime JIT) under the same kernel ABI by passing a compact ND context pointer (`__me_nd_ctx`) and synthesizing:
  - `_i*` from `idx` decomposition + per-block base offsets
  - `_n*` from global shape
  - `_ndim` from ND context
  - `_global_linear_idx` from synthesized coordinates and global strides
- ND synthesis currently remains disabled for wasm32 builds.
- Synthesis is opt-in via:
  - `ME_DSL_JIT_INDEX_VARS_SYNTH=1`
  - default is off (buffer-passing path remains default for stability).
- Runtime input binding allows synthesized reserved params to be omitted from `inputs[]`.

## Rollout behavior (current)

- `ME_DSL_JIT=0`: disables runtime JIT globally.
- `ME_DSL_JIT_INDEX_VARS=0`: disables runtime JIT for DSL kernels using reserved index vars.
- `ME_DSL_JIT_INDEX_VARS_SYNTH=1`: enables non-ND synthesized reserved vars in JIT codegen.

## Validation completed

- Native tests passing:
  - `test_dsl_syntax`
  - `test_dsl_jit_codegen`
  - `test_dsl_jit_runtime_cache`
- Wasm tests passing:
  - `test_dsl_jit_side_module`
  - `test_dsl_jit_runtime_cache`
  - `test_dsl_syntax`
- Added explicit env-gate A/B test:
  - `test_reserved_index_vars_env_gate` in `tests/test_dsl_syntax.c`

## Remaining work

- True Phase 4 v2 ABI/context synthesis for ND contiguous cases (new context struct/ABI).
- ND synthesis path selection/safety hardening across all edge layouts and overflow-sensitive cases.
- Additional overflow-hardening review for global-linear-index arithmetic in all ND paths.
- Decide final long-term default for `ME_DSL_JIT_INDEX_VARS` and whether to retire the gate.

## Done criteria status

- JIT enabled for reserved-index DSL kernels: done (gate-controlled rollout).
- Interpreter/JIT parity for 1D + ND including padding and `_global_linear_idx`: done for current tests.
- Native + wasm32 lanes green with reserved-index coverage: done in current CI-relevant tests.
- Trace diagnostics for unsupported/disabled paths: done for current gates and runtime fallback.
