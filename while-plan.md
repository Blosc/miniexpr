# Python-Style `while` Loops for DSL Kernels: Implementation Plan

## Status (updated)

- [x] Parser AST + syntax support for `while`.
- [x] Interpreter compile/eval support for `while` with loop-control flow.
- [x] DSL docs updated with `while` syntax/examples.
- [x] Core DSL syntax tests added for `while` behavior and parity checks.
- [x] JIT IR now supports `while` lowering.
- [x] JIT C codegen now supports `while` lowering.
- [x] JIT IR/codegen tests updated for `while`.
- [x] Optional safety cap for non-terminating `while` loops.

## Current DSL loop syntax (from `doc/dsl-usage.md`)

- The DSL currently supports Python-style `for i in range(...)` loops.
- `range()` supports 1/2/3 arguments (`stop`, `start,stop`, `start,stop,step`).
- `break` and `continue` are supported inside loops.
- Loop control is element-wise for non-reduction conditions.
- Reduction conditions (`any`, `all`) are global to the active-element mask.

There is no `while` statement in the parser/AST today.

## Goal

Add Python-style `while` syntax:

```python
while <condition>:
    <indented block>
```

with semantics consistent with existing DSL control flow:

- Non-reduction condition: element-wise.
- Reduction condition (`any`/`all`): global over active elements.
- `break`/`continue` remain loop-only and element-wise.

## Scope and rollout

Recommended rollout is two phases to keep risk low:

1. Interpreter support (full functionality) + docs/tests.
2. JIT IR/codegen support later (or explicit JIT-IR rejection in phase 1).

Phase 1 is complete. Phase 2 JIT lowering is now implemented as well.

## Detailed changes

### 1. Parser AST and parsing (`src/dsl_parser.h`, `src/dsl_parser.c`)

- Add a new statement kind: `ME_DSL_STMT_WHILE`.
- Extend `me_dsl_stmt` union with:
  - `me_dsl_expr *cond`
  - `me_dsl_block body`
- Add `parse_while(...)` analogous to `parse_if(...)` header parsing + `parse_if_body(...)` block parsing.
- Update `parse_statement(...)` keyword dispatch to recognize `while`.
- Ensure `parse_while` calls `parse_if_body(..., in_loop=true, ...)` so `break/continue` remain valid inside `while` bodies.
- Update destructor paths (`dsl_stmt_free`) for `ME_DSL_STMT_WHILE`.

### 2. DSL candidate detection and reserved-name scanning (`src/miniexpr.c`)

- Update `dsl_is_candidate(...)` to treat `while` as DSL syntax so DSL parsing is attempted when present.
- Update `dsl_scan_reserved_usage_block(...)`:
  - Scan `while` condition text for `_i*`, `_n*`, `_ndim`.
  - Recurse into `while` body, same as existing `for` recursion.

### 3. Compiled DSL structures and compilation (`src/miniexpr.c`)

- Extend `me_dsl_compiled_stmt` union with a `while_loop` compiled form:
  - compiled condition (`me_dsl_compiled_expr cond`)
  - compiled body (`me_dsl_compiled_block body`)
- In `dsl_compile_block(...)`:
  - Add `ME_DSL_STMT_WHILE` case.
  - Compile condition via `dsl_compile_condition_expr(...)` (matches `if` behavior).
  - Increment/decrement `ctx->loop_depth` around body compilation (same as `for`).
- Update compiled-statement free logic for the new while fields.

### 4. Interpreter evaluation (`src/miniexpr.c`)

- Add `dsl_eval_while_element_loop(...)` similar to `dsl_eval_for_element_loop(...)`, but:
  - Re-evaluates condition each iteration on current active elements.
  - Handles both reduction/global and element-wise conditions via existing condition-mask helpers.
  - Executes body through `dsl_eval_block_element_loop(...)`.
  - Applies `break_mask`, `continue_mask`, and `return_mask` exactly like existing loop flow.
- Add `ME_DSL_STMT_WHILE` dispatch in `dsl_eval_block_element_loop(...)`.

### 5. Termination safety for `while`

Implemented:

- Interpreter `while` execution now enforces a high iteration cap per `while` statement.
- Default cap: `10,000,000`.
- Env override: `ME_DSL_WHILE_MAX_ITERS` (`<= 0` disables cap).
- On cap hit, evaluation returns `ME_EVAL_ERR_INVALID_ARG`.
- With `ME_DSL_TRACE=1`, runtime emits a trace line when the cap is hit.
- Behavior is documented in `doc/dsl-usage.md` and covered by `tests/test_dsl_syntax.c`.

### 6. JIT IR/codegen behavior (`src/dsl_jit_ir.h`, `src/dsl_jit_ir.c`, `src/dsl_jit_cgen.c`)

Implemented:

- Added `while` statement kind to JIT IR.
- Added JIT IR builder lowering for DSL `while` (`cond` + `body`).
- Added JIT C codegen emission for `while` truthiness checks and loop body emission.
- Updated JIT local collection, return-dtype traversal, free paths, and fingerprint hashing to include `while`.

### 7. Documentation (`doc/dsl-usage.md`)

- Add `while` syntax section and examples:
  - basic accumulator while loop
  - while + break/continue
  - reduction condition (`while any(mask): ...`)
- State condition semantics (element-wise vs reduction/global), matching `if`.
- Document loop-termination safety cap (if adopted).

### 8. Tests

Primary test file: `tests/test_dsl_syntax.c`.

Add tests for:

- Basic `while` loop with scalar induction variable.
- `while` with element-wise condition and per-element termination.
- `while` with reduction condition (`any`/`all`).
- `break` and `continue` inside `while`.
- Nested loops: `while` in `for`, `for` in `while`, nested `while`.
- Invalid syntax:
  - missing `:`
  - non-indented body
  - `break`/`continue` outside loops still rejected
- Runtime non-return behavior still unchanged (paths that never return remain eval-time error).
- If iteration cap is implemented: explicit test that infinite loop returns `ME_EVAL_ERR_INVALID_ARG`.

JIT tests:

- Validate JIT IR accepts `while`.
- Validate JIT C codegen for `while` emits compilable C.

## Suggested implementation order

1. Parser AST + parsing + free paths.
2. `miniexpr.c` compile-time structures and compile path.
3. Interpreter `while` evaluator + dispatch.
4. Reserved-scan + DSL candidate updates.
5. Tests (`test_dsl_syntax.c`) for happy path and errors.
6. JIT IR/codegen while lowering.
7. Optional loop-iteration safety cap.
8. Docs/tests final polish.

## Acceptance criteria

- DSL kernels with `while <cond>:` parse and execute correctly in interpreter mode.
- `break`/`continue` behavior inside `while` matches existing loop semantics.
- Existing `for` behavior remains unchanged.
- JIT path either supports `while` or rejects it with explicit reason and clean fallback.
- New syntax and semantics are documented in `doc/dsl-usage.md`.
