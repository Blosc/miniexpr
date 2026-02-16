# DSL Cast Intrinsics: Initial Design Draft

## Goal
Add explicit cast intrinsics to miniexpr DSL so users can write predictable typed expressions in kernels:
- `int(x)`
- `float(x)`
- `bool(x)`
- `str(x)` (deferred; see scope)

Primary motivation:
- Avoid implicit type surprises in DSL return values.
- Let users express intent directly, especially in mixed integer/float kernels and index-based expressions.

## Scope and Sequencing
Phase 1 (recommended):
- Implement `int(x)`, `float(x)`, `bool(x)` for scalar and element-wise array values.
- Keep casts available in all expression contexts (assignments, return expressions, branch expressions).

Phase 2 (defer):
- Evaluate `str(x)` feasibility after numeric/bool casts ship.

Rationale for deferring `str(x)`:
- String handling uses `ME_STRING` fixed itemsize semantics and stricter layout constraints.
- Returning string outputs from DSL currently has constraints in docs/runtime.
- Requires non-trivial API and memory model decisions.

## Non-Goals (Phase 1)
- Python-level fallback behavior changes outside DSL path.
- New user-visible string formatting functions.
- Cross-kernel global type inference redesign.

## User-Facing Semantics (Phase 1)

### `float(x)`
- Converts `x` to floating-point.
- Proposed target dtype rule:
  - If expression output context is explicitly floating (`ME_FLOAT32`/`ME_FLOAT64`), prefer that width.
  - Else default to `ME_FLOAT64`.

### `int(x)`
- Converts `x` to signed integer.
- Proposed target dtype rule:
  - Prefer context integer dtype if explicitly signed integer.
  - Else default to `ME_INT64`.
- Truncation semantics should match existing cast behavior in conversion nodes.

### `bool(x)`
- Converts `x` to boolean (`ME_BOOL`) using non-zero truth semantics.

### Arity and errors
- All cast intrinsics accept exactly 1 positional argument.
- Keyword arguments are invalid.
- Unsupported argument dtypes raise compile-time errors.

## Architecture Changes

### 1. Parser / DSL compile recognition
Current behavior treats `float(...)`/`int(...)`/`bool(...)` as ordinary function calls, and they fail if not in builtin function registry.

Planned change:
- In DSL expression compilation path, detect call targets `float`, `int`, `bool` as cast intrinsics.
- Compile directly into conversion nodes via existing conversion machinery (e.g. `create_conversion_node`/typed conversion op path).
- Preserve existing function-call path for all other call names.

Candidate areas:
- `dsl_compile_expr` pipeline and call-expression handling in `../miniexpr/src/miniexpr.c`.
- Reuse existing conversion helper(s) instead of introducing duplicate cast op implementations.

### 2. Type inference and return dtype consistency
Observed issue: DSL program output dtype can end up inferred from return expression, not honoring requested compile dtype in all cases.

Planned adjustment:
- Ensure final emitted program/output path respects requested compile dtype when provided (non-`ME_AUTO`), with explicit cast insertion if needed.
- Keep mixed-return-type validation strict, but normalize through conversion nodes where legal.

Candidate areas:
- `dsl_compile_program` / `ctx.return_dtype` / `program->output_dtype` selection logic in `../miniexpr/src/miniexpr.c`.

### 3. JIT codegen compatibility
- Ensure cast nodes lower identically for interpreter and JIT paths.
- If a cast form is unsupported in a specific JIT backend, fallback to interpreter path should remain safe and produce identical results.

Candidate areas:
- JIT IR lowering and C/TCC backend emitters in miniexpr DSL JIT path.

### 4. Documentation updates
Update `../miniexpr/doc/dsl-usage.md`:
- Add cast intrinsics under Available Functions / Type Conversion.
- Define explicit conversion semantics and defaults.
- Provide examples including index variable casting and mixed-type loops.

## Proposed Semantics Details

### Conversion table
Use existing conversion semantics already implemented for dtype conversions. Do not introduce separate cast rules unless required.

### NaN/Inf behavior
- For `float(x)`: preserve representable NaN/Inf values.
- For `int(float_nan_or_inf)`: follow existing conversion behavior (document exactly as implemented).
- For `bool(x)`: zero -> false, non-zero -> true.

### Complex numbers
- Decide explicitly for Phase 1:
  - `float(complex)` and `int(complex)`: either reject at compile-time or map to real part only.
- Recommendation: reject initially for clarity unless existing conversion path already has consistent behavior and tests.

## Test Plan

### A. Parser/compile tests
Add to DSL syntax tests (`../miniexpr/tests/test_dsl_syntax.c` or equivalent):
- `return float(3)` compiles.
- `return int(3.9)` compiles.
- `return bool(0)` compiles.
- Arity errors: `float()`, `float(x, y)` fail with clear compile error.
- Unknown cast-like names still fail as before.

### B. ND evaluation correctness
Add tests in `../miniexpr/tests/test_nd.c`:
- `def kernel(): return float(_i0 * _n1 + _i1)` with float output expected ramp.
- `def kernel(): return int(1.9)` yields integer truncation semantics.
- `def kernel(x): return bool(x)` with numeric arrays.
- Edge chunks/blocks (padding) to ensure cast path is consistent on tails.

### C. JIT parity tests
- Run same cast kernels with JIT on/off (`ME_JIT_ON`, `ME_JIT_OFF`).
- Assert bitwise/elementwise equality where appropriate.
- Include both `# me:compiler=tcc` and `# me:compiler=cc` where supported.

### D. Regression tests for python-blosc2 integration
In python-blosc2 tests (e.g. `tests/ndarray/test_dsl_kernels.py`):
- Kernel using `float(_i0)` no longer falls back to Python and returns correct typed output.
- Existing scalar-specialization behavior with `float(max_iter)` remains correct.
- Ensure failures in DSL compile produce actionable errors (if fallback policy is tightened later).

## Backward Compatibility
- Existing DSL kernels without casts remain unchanged.
- Existing scalar specialization in python-blosc2 (`float(constant)` folding) remains valid.
- New intrinsics only broaden accepted DSL syntax.

## Risks and Mitigations

Risk 1: Type ambiguity for cast target widths.
- Mitigation: define deterministic defaults (`float -> float64`, `int -> int64`) plus context-aware narrowing only when compile dtype is explicit.

Risk 2: Interpreter/JIT mismatch.
- Mitigation: parity tests across JIT modes and backends as part of CI.

Risk 3: Silent fallback masking compile errors in downstream integrations.
- Mitigation: in python-blosc2, consider surfacing DSL compile failures for DSLKernel paths instead of automatic Python fallback.

## Implementation Checklist
1. Add DSL cast intrinsic recognition for `int/float/bool` in compile path.
2. Lower to existing conversion node mechanism.
3. Align `program->output_dtype` selection with requested dtype policy.
4. Add parser and ND eval tests for casts.
5. Add JIT parity tests.
6. Update DSL docs (`dsl-usage.md`) with cast semantics and examples.
7. Add downstream python-blosc2 regression tests.

## Open Questions
1. Should `float(x)` default to `float32` when requested output dtype is `ME_FLOAT32`, or always `float64` unless explicitly narrowed later?
2. Should `int(x)` preserve unsigned context (`uint*`) when output dtype is unsigned, or always signed `int64` semantics?
3. For complex input casts, should we reject or map real-part semantics?
4. Should python-blosc2 DSLKernel path stop silent fallback on DSL compile errors?

## Suggested First Milestone
- Deliver Phase 1 cast intrinsics (`int/float/bool`) with interpreter correctness + unit tests.
- Include one integration test demonstrating `float(_i0)` kernel returning correct float ramp via python-blosc2.
- Defer `str()` to separate RFC after Phase 1 stabilization.
