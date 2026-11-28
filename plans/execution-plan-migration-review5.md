# Execution Migration Review 5 — M8 Artifact-Off Qualification & V5 FP Policy

**Context:** Review of the working-tree changes across `miniexpr.codeberg` (over `1cc0063`), `c-blosc2.codeberg` (over `2ac1978b`), and `python-blosc2.codeberg` (over `bd042255`), as recorded in [`plans/execution-migration-status.md`](execution-migration-status.md).

## Overall Assessment

We are **strongly on the right track**. The recent increments close the remaining M8 public-entry-point and floating-point policy gaps on top of the core Menudet decoupling—bringing the isolated `MINIEXPR_BUILD_ARTIFACT=OFF` wheel to `1764 passed, 83 skipped, 584 deselected` with `0` `yyjson_*`/`me_artifact_*`/`me_graph_*` symbols, a single core provider, and original 4.17.0 v1/v2 exchange fixtures executing without the legacy adapter.

Highlights that are particularly well engineered:

- **Shared loop/control-flow partition proof:** Extracting `src/dsl_partition_internal.h` removes ~285 lines of duplicate proof logic from `src/dsl_artifact.c` and lets `me_kernel_is_partitionable` in `src/miniexpr_kernel.c` share the exact bounded proof without source replay.
- **ABI-safe FP policy evolution (`ME_PREPARE_OPTIONS_V5_SIZE`):** Appending `fp_mode` and `math_accuracy` with explicit versioned size checks in `src/miniexpr_kernel.h` / `.c` and `B2ND_EXEC_FP_POLICY_API` / `CMakeLists.txt` header guards keeps V1–V4 callers untouched while catching mismatched local checkouts early.
- **Borrowed-kernel masks and explicit accumulators:** `b2nd_exec_prepare_borrowed_kernel_ex` in `c-blosc2/blosc/b2nd-exec.c` handles resident `where` masks and explicit reduction `accumulator` conversions cleanly without rewriting or mutating the borrowed `me_kernel`, and `exec_gather_tile` avoids reading discarded lanes for callback sources while `b2nd-exec-storage.c` rejects reader-backed masks upfront.
- **Isolated `-S` verification runner:** `python-blosc2/scripts/qualify_execution_migration.py` cleanly verifies the exact installed wheel without editable-checkout contamination, and `plans/execution-migration-status.md` accurately separates pre-V5 verification runs from V5 runs.

---

## Findings & Suggested Improvements

### 1. Math-backend precedence inversion & validation mismatch in `typed_graph_policy`

- **Location:** `python-blosc2/src/blosc2/typed_graph.py` (lines 102–148)
- **Observation:**
  1. **Precedence inversion:** Lines 108 and 122–125 let `ME_DSL_JIT_COMPILER` (environment) override `compute_options(jit_backend=...)`. Immediately below on lines 110–111:
     ```python
     math_backend = policy["math_backend"] or environment.pop("math", None)
     environment.pop("math", None)  # An explicit compute option overrides the environment.
     ```
     This does the **opposite**: `compute_options(math_backend=...)` overrides `ME_SIMD_MATH_BACKEND`, contradicting both `compiler` in `typed_graph_policy` and `menudet_execution_snapshot` (`menudet_kernel.py:483-494`) / MiniExpr (`dsl_math_policy.c`), where `env:` takes precedence.
  2. **Early validation mismatch:** Line 112 admits `"auto"` and `"scalar"` for `math_backend` and line 162 maps them in `{None: 0, "auto": 1, ..., "scalar": 5}`, yet line 130 still checks:
     ```python
     or policy["math_backend"] not in {None, "system", "sleef", "accelerate"}
     ```
     *before* line 145 updates `policy["math_backend"]`. Passing `compute_options(math_backend="auto")` or `"scalar"` therefore hits `NotImplementedError("Unsupported typed graph backend or environment policy")`.
  3. **`FPAccuracy.MEDIUM` math resolution:** Lines 133–142 auto-select SLEEF `u10` for `FPAccuracy.HIGH` when `math_backend in {None, "auto"}`, but leave `FPAccuracy.MEDIUM` at `accuracy=0` / default math instead of matching `preferred_expression_math` (`native_graph.py:1823-1841`, which selects Accelerate on macOS or SLEEF `u35` elsewhere).
- **Recommendation:** Clarify/align env vs `compute_options` precedence for `math_backend`, move the allowed-values check after resolution (or remove the duplicate `policy["math_backend"]` check on line 130), and confirm whether `FPAccuracy.MEDIUM` on `NativeGraph` should align with `preferred_expression_math`.

---

### 2. `ME_JIT_AUTO` (`jit_mode == 3`) misclassified in artifact-off `NativeGraphHandle`

- **Location:** `python-blosc2/src/blosc2/blosc2_ext.pyx` (`NativeGraphHandle.__cinit__` lines 2198–2201 and `NativeGraphHandle.from_expression` lines 2323–2328)
- **Observation:** Both artifact-off `TypedGraphHandle` delegation sites convert `jit_mode` with:
  ```cython
  jit=None if jit_mode == 0 else jit_mode == 1
  ```
  However, `menudet_auto_jit_mode()` returns `3` (`ME_JIT_AUTO`). Direct `NativeGraphHandle(..., jit_mode=3)` calls therefore evaluate `jit_mode == 1` as `False` instead of `None`, silently disabling auto-JIT heuristics and forcing interpreted execution.
- **Recommendation:** Change both sites to `jit=None if jit_mode in (0, 3) else jit_mode == 1` (and reject unknown modes consistently).

---

### 3. `b2nd_exec_staged_plan_describe` reports FP/math policy only from the final stage

- **Location:** `c-blosc2/blosc/b2nd-exec-stages.c` (lines 300–315)
- **Observation:** `b2nd_exec_staged_plan_describe` calls `b2nd_exec_describe(plan->children[plan->nstages - 1], out)` to populate `out->fp_mode`, `out->math_backend`, and `out->sleef_accuracy`, and then loops over all stages only updating `out->plan_metadata_bytes` and `out->has_jit`.
- **Impact:** For auto-split DAGs all stages share one `b2nd_exec_options`, so they agree; however, for explicitly assembled multi-stage plans (`b2nd_exec_staged_assemble`) where a producer Menudet stage uses e.g. `# me:fp=fast` or `# me:math=sleef` and the root stage is strict/system, plan metadata will report only the root stage's policy.
- **Recommendation:** Either document this as root-stage policy for heterogeneous assembled plans or combine/mark mixed policies across stages (e.g., matching how `has_jit` aggregates across stages).

---

### 4. Minor backend-counter mixing on borrowed kernels with explicit accumulator casts

- **Location:** `c-blosc2/blosc/b2nd-exec.c` (lines 1786–1792)
- **Observation:** When `plan->cast_kernel` is prepared via `me_prepare_formula("x", ...)`, it is always interpreted. Inside `exec_eval_tile`, its backend count (`ME_BACKEND_INTERPRETER`) is added to the same `job->interpreter_calls` counter as the main kernel, so a JIT-compiled borrowed kernel with an explicit reduction `accumulator` cast reports both `jit_calls > 0` and `interpreter_calls > 0`.
- **Recommendation:** Either track internal cast-kernel calls separately or note this in test assertions on borrowed reductions with explicit `dtype=` casts.

---

### 5. Unreachable `CMakeLists.txt` link branch

- **Location:** `python-blosc2/CMakeLists.txt` (lines 235–237)
- **Observation:** `elseif(TARGET miniexpr_static AND NOT B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE)` cannot be reached because lines 149–151 already emit `FATAL_ERROR` whenever `NOT MINIEXPR_BUILD_ARTIFACT AND NOT B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE`.
- **Recommendation:** Remove the dead `elseif` branch to keep link selection minimal.

---

### 6. Combined-tree re-verification & git hygiene before M9

- **Observation:**
  1. As accurately noted in `plans/execution-migration-status.md:64-67`, the V5 FP-policy changes (`ME_PREPARE_OPTIONS_V5_SIZE`, `cast_kernel`, sparse mask reads) have not yet been re-verified across the artifact-on wheel, standard non-opt-in wheel (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=OFF`), MiniExpr artifact-on C suite, installed C consumers (`6/6`), and ASan/UBSan suites (`exec-plan`, `exec-storage`, `exec-stages`). Running ASan/UBSan is especially worthwhile to exercise the new `cast_kernel` mapped-buffer offset and sparse mask reader paths.
  2. All changes across `miniexpr.codeberg`, `c-blosc2.codeberg`, and `python-blosc2.codeberg` are currently uncommitted working-tree modifications.
- **Recommendation:** Address items 1–3 (and 5), re-run the full local verification matrix on the combined working trees (including ASan/UBSan and installed consumers), and record clean local commits across all three repositories before starting M9 cross-platform/pinning/default-switch work.
