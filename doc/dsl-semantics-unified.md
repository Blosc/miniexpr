# Unified DSL Semantics (Draft)

Status: Draft
Scope: DSL language semantics and backend conformance
Goal: remove vector/element dialect split and define one behavior model

## Decision Summary (Proposed)

1. There is a single DSL semantics model. Dialect specific behavior is removed.
2. Conditions are interpreted as follows:
   - Scalar condition: broadcast to all currently active lanes.
   - Vector condition: evaluated per lane (masked control flow).
3. `if` / `elif` / `else` run per lane:
   - A lane enters the first branch whose condition is true for that lane.
   - `else` applies to lanes that did not match previous branches.
4. `for` loops have scalar limits only (`range(N)` where `N` is scalar).
5. `break` and `continue` are per-lane inside loops.
6. `return` is per-lane:
   - Returned lanes write output and become inactive for the rest of the kernel.
   - Kernel execution ends early when all lanes are inactive.
7. Reductions (`any/all/sum/mean/min/max/prod`) remain scalar expressions.
8. `print` arguments must be scalar/uniform. Vector prints are rejected.
9. Interpreter is the reference behavior. JIT backends must match it.
10. If JIT cannot represent a construct, compilation keeps the DSL program and runtime falls back to interpreter semantics.

## Normative Semantics

### Execution Domain

- Kernels execute over `nitems` logical lanes.
- Each statement is evaluated under an `active_mask`.
- Lane `i` participates in a statement only if `active_mask[i]` is true.

### Conditions

- Scalar condition:
  - Convert to boolean once.
  - If false, no active lane enters the branch.
  - If true, all active lanes enter.
- Vector condition:
  - Convert each lane value to boolean.
  - Branch membership is computed per lane.

### If Chains

- For each lane, `if`/`elif` conditions are tested in source order.
- First true condition wins for that lane.
- `else` executes for lanes with no prior match.

### Loops

- `for i in range(N)`:
  - `N` is scalar and evaluated once before loop entry.
  - If `N <= 0`, loop body is skipped for all lanes.
- Per iteration, only active lanes execute loop body.
- Early stop rule: if no lanes remain active, the loop exits.

### Break and Continue

- `break` affects only lanes that reach it and satisfy its optional condition.
- `continue` affects only lanes that reach it and satisfy its optional condition.
- Lanes that break become inactive for the current loop only.

### Return

- A lane that executes `return expr`:
  - evaluates `expr` for that lane,
  - stores result to output for that lane,
  - becomes inactive for remaining statements.
- Kernel is complete when all lanes returned or finished without returning.
- If any lane reaches function end without a return value, this is a runtime error.

### Truthiness

- Bool: `false` is false, `true` is true.
- Signed/unsigned integers: `0` is false, non-zero is true.
- Floating point: `0.0` and `-0.0` are false, all other values are true.
- Strings: non-empty true, empty false.

## Compile-Time Rules

1. Dialect pragma is accepted for now but ignored semantically.
2. Control-flow conditions no longer require explicit `any()`/`all()`.
3. `range()` with start/stop/step is still unsupported unless separately implemented.
4. New locals inside conditional branches remain disallowed unless separately relaxed.

## Backend Conformance

1. Interpreter defines canonical output and control-flow effects.
2. JIT must preserve lane-wise behavior exactly for control flow.
3. If JIT lowering is unavailable, runtime fallback to interpreter is mandatory.
4. Test parity is required for:
   - nested `if/elif/else`,
   - loop with mixed break/continue/return,
   - scalar condition and vector condition cases,
   - reduction in conditions and assignments.
