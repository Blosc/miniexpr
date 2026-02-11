## Future Consideration: JIT Enablement for `any()` / `all()`

This section is non-normative and documents a minimal implementation path for later work.

Current state:
- JIT IR rejects reduction calls (`any/all/sum/mean/min/max/prod`) during subset validation.
- Kernels that use `any()` or `all()` therefore compile and run via interpreter fallback.

### Staged Proposal

1. Stage 1 (smallest, low-risk)
    - Allow `any()` / `all()` only in `if` / `elif` condition expressions.
    - Require the reduction argument to be compile-time uniform/scalar.
    - Lower to a scalar truthy condition in codegen.
    - Keep rejection for reductions in assignments/returns and for non-uniform arguments.
    - Note: this does not cover vector-style guards like `if all(active == 0): ...`.

2. Stage 2 (covers vector-style control-flow guards)
    - Extend JIT IR to represent `if` conditions that are reduction calls explicitly:
        - reduction kind: `any` or `all`
        - reduction argument expression
    - In codegen, add a reduction-capable path for those conditions:
        - evaluate reduction argument over element domain,
        - compute scalar reduction result with short-circuit behavior,
        - branch using that scalar result.
    - Keep conditional `break`/`continue` out of scope in this stage.

3. Stage 3 (optional follow-up)
    - Extend reduction-condition support to conditional `break` / `continue`.
    - Consider additional reductions (`sum/mean/min/max/prod`) only after `any/all` parity is stable.

### Suggested Code Touchpoints

- `src/dsl_jit_ir.c`
    - Make subset validation context-aware instead of rejecting all reductions unconditionally.
    - Allow only top-level `any(...)` / `all(...)` in condition context for staged support.
- `src/dsl_jit_ir.h`
    - Add optional reduction metadata for condition IR nodes (or equivalent explicit encoding).
- `src/dsl_jit_cgen.c`
    - Add emission logic for supported reduction conditions.
    - Keep existing fast scalar path unchanged when no reduction conditions are present.
- `src/miniexpr.c`
    - Keep JIT best-effort fallback semantics unchanged.
    - Emit clearer trace reason when reduction conditions are not yet in the supported JIT subset.
- `tests/test_dsl_jit_ir.c`
    - Add positive tests for accepted `any/all` condition forms.
    - Keep negative tests for unsupported reduction placements.
- `tests/test_dsl_jit_runtime_cache.c`
    - Add interpreter/JIT parity tests for `if any(...)` and `if all(...)` control flow.

### Parity Requirements

- Truthiness must match interpreter semantics exactly.
- `any/all` condition behavior must match interpreter control-flow effects for all elements.
- If a reduction form is unsupported, JIT must reject cleanly and runtime must fallback to interpreter.

## Migration Notes

- Existing scripts with explicit `any()`/`all()` keep working.
- Scripts written with element-wise semantics become baseline semantics.
- Scripts relying on scalar-condition control flow still run, but may now execute element-wise when condition is vector-valued.
