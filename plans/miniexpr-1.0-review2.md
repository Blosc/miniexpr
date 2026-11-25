# miniexpr 1.0 implementation review 2

Review date: 2026-11-25
Scope: outstanding working-tree follow-up changes on top of `fc46f66` addressing [`miniexpr-1.0-review1.md`](miniexpr-1.0-review1.md), summarized in [`miniexpr-1.0-core-api.md`](miniexpr-1.0-core-api.md) ("Review 1 follow-up").

This is a strong follow-up: almost every item from Review 1 landed cleanly, backed by targeted regression tests (`tests/check_kernel_no_compiler.cmake`, `tests/graph/grid-allocation.c`, `tests/cmake/embedded/`, plus SLEEF, bundled TCC, Emscripten, and both Blosc2 downstream suites).

## 1. What looks especially solid

- **Deferred JIT and single-pass typing:** `dsl_compile_menudet_deferred` in `src/dsl_compile.c` (lines 3723-3734) resolves four issues together: removes JSON `artifact_load`'s double-compile, removes `me_artifact_prepare_menudet`'s double-parse, guarantees rejected contracts never invoke a compiler (verified by `tests/check_kernel_no_compiler.cmake`), and shares weak-capture JIT specialization across both direct and JSON paths.
- **Padded grid hot path:** The suffix-merged contiguous-run odometer and single alignment-checked scratch allocation in `src/miniexpr_grid.c` (lines 112-224) are clean, overflow-safe, eliminate per-lane division/modulo, and handle 0D, empty elementwise, empty scalar, and mixed-width inputs cleanly.
- **Diagnostics, capabilities, and build cleanup:** Binding vs. source vs. invalid-arg classification is consistent, `me_artifact_prepare_menudet` derives capabilities accurately from the compiled program, dead `numpy_profile` branches are removed, `Makefile` delegates to CMake, and `miniexpr::core`/`miniexpr::artifact` plus embedded-OFF defaults are tested in-tree.

## 2. Subtle issues worth fixing

1. **ABI hazard with `*_V1_SIZE` excluding trailing padding on `me_signature` and `me_prepare_options`** (`src/miniexpr_kernel.h`, lines 76-98)
   - Both `me_signature` (ends with `bool integer_powers`, offset 69, `sizeof == 72` on 64-bit) and `me_prepare_options` (ends with `bool inherit_environment`, offset 13, `sizeof == 16` on 64-bit) have **3 bytes of unnamed trailing padding** after their last field.
   - `ME_SIGNATURE_INITIALIZER` and `ME_PREPARE_OPTIONS_INITIALIZER` set `struct_size = sizeof(...)` (`72` / `16`), not `V1_SIZE` (`69` / `13`).
   - If v1.1 appends a `bool` or `uint8_t` field after `integer_powers` or `inherit_environment`, the compiler places it in that trailing padding (offset 69 or 13), which is still `<= sizeof(v1_struct)`. A compiled v1.0 binary running against a v1.1 shared library will pass `struct_size = 72` / `16`, passing a `struct_size >= offsetof(new_field) + sizeof(new_field)` check and causing v1.1 to read **indeterminate v1.0 padding bytes** as the new field.
   - **Fix (before ABI freeze):** Add explicit zero-initialized reserved/padding bytes (or pack flags into `uint32_t`) so `V1_SIZE == sizeof(v1_struct)` with zero trailing padding—matching `me_kernel_info`, `me_block`, `me_run_options`, `me_grid_spec`, and `me_grid_block_info`—so any future appended field is guaranteed to start strictly past `sizeof(v1_struct)`.

2. **Formula `inherit_environment` snapshots `formula_math` but drops `ulp`** (`src/miniexpr_kernel.c`, lines 219-224 and 444-450)
   - `kernel_prepare` resolves `me_dsl_sleef_ulp ulp` via `dsl_math_policy_resolve`, then discards it instead of storing `kernel->formula_ulp` and calling `me_simd_accuracy_push`/`me_simd_accuracy_pop` in `me_eval_block` (unlike `dsl_eval_program_menudet` in `src/dsl_eval.c`, lines 2398-2402).
   - With `inherit_environment = true` and `ME_DSL_MATH=sleef` + `ME_DSL_SLEEF_ULP=u10`, later environment changes or worker threads will not stay pinned to the prepared formula's ULP mode. Saving `formula_ulp` on `me_kernel` and pushing/popping it around `dsl_menudet_eval_expr_masked` closes this gap.

3. **Minor cleanups**
   - In `src/dsl_artifact.c` (lines 844-865), wrap the entire `dsl_menudet_fp_begin` / `dsl_try_prepare_jit_runtime` block in `if (artifact->program->jit_request_mode == ME_JIT_ON)` (matching `src/miniexpr_kernel.c`, line 290) so `ME_JIT_OFF` loads do not touch FP state or enter `dsl_try_prepare_jit_runtime`.
   - `bench/benchmark_grid_api.c` is not wired into `bench/CMakeLists.txt` yet; adding it under `MINIEXPR_BUILD_BENCH=ON` keeps it building so it does not bitrot.

## 3. Verdict

Aside from those three items (especially #1 before freezing the 1.0 ABI, and #2 while `formula_environment` is fresh), the outstanding changes are clean and ready to land.
