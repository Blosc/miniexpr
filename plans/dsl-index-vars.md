# Plan: JIT Support for DSL Reserved Index Variables

Last updated: 2026-02-23

## Goal

Enable runtime JIT for DSL kernels that reference reserved index symbols:

- `_i0`..`_i7`
- `_n0`..`_n7`
- `_ndim`
- `_global_linear_idx`

with behavior matching interpreter semantics for both regular and ND/padded evaluation.

## Why it is blocked today

Current runtime JIT only wires user-declared kernel params from `inputs[]` into generated C (`in_<param>[idx]`). Reserved index symbols are interpreter-injected synthetic values, not user params. The JIT pipeline therefore has no data channel for them, and runtime JIT is intentionally skipped when any reserved index symbol is used.

## Design principles

1. Correctness first: identical semantics vs interpreter before optimization.
2. Backward compatibility: do not break existing JIT kernel ABI abruptly.
3. Deterministic metadata: stable ordering for bindings and cache fingerprints.
4. Incremental rollout: pass values first, synthesize in-kernel later as an optimization.

## Proposed channel design

### Channel v1: Pass reserved buffers as synthetic JIT inputs

Use the existing kernel ABI:

`int me_dsl_jit_kernel(const void **inputs, void *output, int64_t nitems)`

and extend runtime binding metadata so `inputs[]` can include:

- user input arrays
- synthetic reserved arrays (`_i*`, `_n*`, `_ndim`, `_global_linear_idx`)

This keeps C codegen simple: reserved symbols are treated like normal params and read as `in_<name>[idx]`.

### Channel v2 (optional optimization): Synthesize indices in JIT kernel

Later, add an alternate ABI/context path where kernel receives ND context and computes reserved symbols on the fly (mainly to avoid materializing uniform/index buffers). Keep v1 as fallback for complex/padded layouts.

## Phase 0: Metadata groundwork

1. Add JIT param binding metadata in `me_dsl_compiled_program`:
   - binding kind (`user_input`, `reserved_i`, `reserved_n`, `reserved_ndim`, `reserved_global_linear_idx`)
   - auxiliary index (`d` for `_i<d>`/`_n<d>`)
   - stable param name
2. Keep user param order first, append reserved params in deterministic order.
3. Include new binding metadata in runtime cache key/fingerprint inputs.

## Phase 1: IR/codegen acceptance for reserved symbols

1. Build JIT IR param metadata from:
   - parsed user params
   - reserved symbols actually used by program
2. Ensure generated C declares `in_<reserved>` pointers for those symbols.
3. Remove blanket runtime-JIT skip for reserved symbols once runtime binding is implemented.

## Phase 2: Runtime binding path (v1 pass-through channel)

1. Introduce a shared helper to prepare reserved arrays for both interpreter and JIT dispatch:
   - `_i*`: use provided `idx_buffers[d]` when available, otherwise synthesize.
   - `_n*`: fill `int64_t` array with per-dimension shape value.
   - `_ndim`: fill `int64_t` array with ndim.
   - `_global_linear_idx`:
     - non-ND: `0..nitems-1`
     - ND contiguous/packed: use existing global-index computation semantics.
2. Build JIT `inputs[]` from binding metadata:
   - user bindings from `vars_block`
   - reserved bindings from prepared buffers
3. Attempt JIT with this expanded `inputs[]`.
4. Preserve interpreter fallback on any JIT runtime error.

## Phase 3: ND and padding correctness hardening

1. Reuse current ND semantics exactly:
   - `valid_items == padded_items`: direct valid traversal.
   - packed/scatter path: JIT runs on packed valid items only; scatter unchanged.
2. `_global_linear_idx` must map to global C-order position in original array for each valid element.
3. Keep integer overflow guards in global-index arithmetic (shape strides and index accumulation).
4. Ensure no writes occur to padded elements except existing zero-fill/scatter behavior.

## Phase 4: Optional synthesis optimization (v2)

1. Add optional JIT context struct/ABI for contiguous ND cases:
   - shape
   - chunk/block offsets
   - blockshape/valid layout metadata
2. Generate kernel-side formulas for `_i*` and `_global_linear_idx` when safe.
3. Select v2 only when layout permits simple arithmetic mapping.
4. Fallback to v1 (passed buffers) for complex/padded pack layouts.

## Safety and correctness requirements

1. Reserved index dtype remains `ME_INT64` everywhere.
2. Reserved names remain forbidden as user input names.
3. Any unsupported JIT lowering must emit explicit trace/diagnostic reason and fallback cleanly.
4. Memory ownership/lifetime for reserved buffers must be explicit and leak-free on all exit paths.

## Testing plan

1. Extend DSL JIT parity tests for each reserved symbol:
   - direct return of `_i0`, `_n0`, `_ndim`, `_global_linear_idx`
   - mixed arithmetic expressions
2. ND coverage:
   - no-padding and padding cases
   - packed/scatter path
   - multiple chunks/blocks
3. Control-flow coverage:
   - if/while/for kernels using reserved symbols
4. Backend matrix:
   - native (`cc`/`tcc` as available)
   - wasm32 runtime JIT path
5. Cache behavior:
   - ensure cache keys differ when reserved-binding metadata differs
6. Regression:
   - existing non-index JIT kernels unchanged in behavior/perf baseline.

## Rollout strategy

1. Land behind temporary env gate (example: `ME_DSL_JIT_INDEX_VARS=1`).
2. Keep detailed trace lines for:
   - binding construction
   - JIT eligibility
   - fallback reason
3. Remove gate after CI matrix is stable.

## Open decisions

1. Whether to implement v2 ABI/context now or after v1 parity lands.
2. Whether `_n*`/`_ndim` should stay materialized as arrays in v1, or be emitted as scalar uniforms in codegen.
3. Preferred deterministic reserved-param order for long-term cache stability.

## Done criteria

1. JIT is enabled for DSL programs using reserved index symbols.
2. Interpreter/JIT parity passes for 1D and ND (including padding and `_global_linear_idx`).
3. Native + wasm32 test lanes pass with no reserved-index regressions.
4. Trace diagnostics are explicit for any remaining unsupported edge case.
