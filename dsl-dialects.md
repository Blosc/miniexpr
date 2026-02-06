# DSL Dialects Proposal (Vector vs Element)

## Context

During Mandelbrot benchmarking, we hit a semantics/performance issue around escape conditions:

1. The current DSL/interpreter behavior treats control conditions at block level (effectively array-wise truth).
2. Mandelbrot needs per-element escape to benefit from early `break`/`continue`.
3. JIT codegen already runs an outer `idx` loop with implicit per-element loads (`cr`, `ci` are scalar values per `idx`), but current compiler restrictions reject the needed control flow.

This created a discussion point: if we enable per-element control flow in JIT only, the same DSL text would behave differently between interpreter and JIT.

## Decisions From Discussion

1. Explicit `cr[idx]` syntax is not required for per-element control flow in JIT, because indexing is already implicit in generated code.
2. If only JIT semantics change, we effectively end up with two DSL behaviors for the same source.
3. To make this explicit and manageable, introduce two DSL dialects selected by a kernel meta-label.
4. Use a source pragma comment (not decorators) for low parser risk and backward compatibility.
5. Default to current semantics when no pragma is provided.

## Proposed Dialect Label

Dialect is declared at top of DSL source:

```python
# me:dialect=vector
def kernel(cr, ci):
    ...
```

```python
# me:dialect=element
def kernel(cr, ci):
    ...
```

Accepted values:

1. `vector` (default if absent)
2. `element`

## Semantics

### `vector` dialect (current behavior)

1. Control conditions are uniform/block-level.
2. `if` / conditional `break` / conditional `continue` depend on array-level truth (e.g. via reductions like `any(...)`).
3. Keeps full backward compatibility with existing DSL kernels.

### `element` dialect (new behavior)

1. Control conditions inside loops are evaluated per element.
2. `break` and `continue` are per-element loop control.
3. `any(...)` and `all(...)` remain explicit reductions for global/block decisions.

This dialect directly addresses Mandelbrot escape behavior:

1. Each pixel exits its loop independently when `zr*zr + zi*zi > 4.0`.
2. Warm-run performance can improve because unnecessary iterations are skipped per element.

## Implementation Steps

1. Parser/metadata:
   1. Parse `# me:dialect=...` pragma from DSL source.
   2. Store dialect in compiled DSL program metadata.
   3. Emit clear error on unknown dialect value.
2. Compilation policy:
   1. Gate uniform-condition checks by dialect.
   2. Keep current checks in `vector`.
   3. Permit non-uniform loop control conditions in `element`.
3. JIT IR/codegen:
   1. Allow conditional `break`/`continue` in JIT IR for `element`.
   2. Lower conditional flow to scalar per-`idx` checks in emitted C.
   3. Keep current restrictions in `vector`.
4. Interpreter path:
   1. Add `element` execution semantics for loop control (per-item active state/mask or equivalent).
   2. Preserve existing behavior for `vector`.
   3. Ensure consistent numerical/control behavior with JIT for `element`.
5. Cache/versioning:
   1. Include dialect in IR fingerprint/runtime cache key.
   2. Bump codegen/runtime metadata version to avoid stale artifact reuse.
6. Diagnostics:
   1. Add dialect-aware compile diagnostics (unsupported construct, fallback reason).
   2. Include dialect in optional debug traces.
7. Tests:
   1. Parser tests for pragma detection/default.
   2. Unit tests for control-flow semantics in both dialects.
   3. JIT vs interpreter parity tests for `element`.
   4. Runtime cache-key tests including dialect differentiation.
8. Benchmarks/docs:
   1. Keep Mandelbrot benchmark in `element` mode to validate escape speedups.
   2. Document dialect guidance and migration notes.

## Rollout Notes

1. Keep default dialect as `vector` to avoid breaking existing DSL users.
2. Mark `element` as opt-in and initially experimental.
3. If parity risks remain, gate `element` behind an env flag during initial rollout.

## Effort Estimate

Estimate: **Large**.

Rationale:

1. Parser and metadata work is small.
2. JIT IR/codegen changes are medium.
3. Interpreter-side `element` semantics are the largest part (control-flow execution model update).
4. Significant testing is required to prevent semantic regressions and backend divergence.
