# DSL if/elif/else TODO (scalar conditions)

## Scope
- Add general `if/elif/else` clauses to the DSL.
- Conditions are **scalar/uniform** only (same rule as loop control).
- **No new locals** allowed inside branches.
- **All branches must assign `result`.**

## Collision avoidance
- Wait for the ongoing DSL C callback work to land before implementing.
- Expected overlapping files: `src/miniexpr.c`, `src/dsl_parser.{c,h}`.

## Design notes
- Introduce a new statement kind (e.g., `ME_DSL_STMT_IF`).
- Parse `if ...:` blocks with optional `elif ...:` and `else:` at same indent.
- Enforce:
  - Uniform/scalar condition via `dsl_expr_is_uniform()` + `dsl_any_nonzero()`.
  - No new locals in branches (reject assignments to new names).
  - All branches assign `result` (each branch block must contain a `result = ...`).

## Implementation plan (post-callback merge)
1. Parser (`src/dsl_parser.c`, `src/dsl_parser.h`)
   - Add `ME_DSL_STMT_IF` node with:
     - `cond` expression
     - `then` block
     - zero+ `elif` blocks
     - optional `else` block
   - Extend parsing to handle `elif`/`else` chains at the same indentation.

2. Compiler (`src/miniexpr.c`)
   - Compile IF blocks into compiled blocks.
   - Enforce scalar/uniform condition.
   - Reject assignments to new locals inside branches.
   - Require `result` assignment in each branch block.
   - Update reserved-usage scan to recurse into IF blocks.

3. Evaluator (`src/miniexpr.c`)
   - Evaluate the condition once per block (scalar) and execute the selected branch.

4. Docs/Tests
   - Update `doc/dsl-usage.md` to describe general conditionals.
   - Extend `tests/test_dsl_syntax.c` for if/elif/else (valid/invalid cases).
   - Ensure existing invalid conditional tests still match new rules.

## Open questions
- How strict should "all branches assign result" be?
  - Require single `result = ...` statement?
  - Allow other statements before/after so long as `result` is assigned?
