# miniexpr 1.0 implementation review 1

Review date: 2026-11-25
Scope: commits `eeb8766` ("Initial version for next major 1.0 alpha version"), `61fea03` ("Pilot typed kernel C API and Menudet 0.2 annotations"), and `fc46f66` ("Integrate direct preparation with native schedulers and padded grids") against [`miniexpr-1.0-cleanup.md`](miniexpr-1.0-cleanup.md) and [`miniexpr-1.0-core-api.md`](miniexpr-1.0-core-api.md).

Overall, the last three commits are moving in a strong direction: the new core stays JSON-free, compiler and evaluator internals are shared rather than duplicated, Menudet 0.1 remains untouched while 0.2 annotations are gated behind an explicit version string, and both downstream consumers (C-Blosc2 constructors and Python-Blosc2 `b2o` v2) are being exercised early before freezing the ABI.

## 1. Progress against the plans

| Plan area | Status | Notes |
| --- | --- | --- |
| **Core 5-call C API** (`miniexpr-1.0-core-api.md`) | **Piloted** | `src/miniexpr_kernel.h`, `src/miniexpr_kernel.c`, `doc/core-api.md`, `tests/test_kernel_api.c`. |
| **Menudet 0.2 annotations** (`doc/dsl-spec/0.2.md`) | **Implemented** | Parser, compiler, and validator support cleanly gated on `"0.2"`; return annotation converts last via `inferred_dtype`. |
| **Padded grid adapter** | **Piloted** | `src/miniexpr_grid.h`, `src/miniexpr_grid.c`, `tests/test_grid_api.c`. Handles non-contiguous live rows and whole-group block-scalar reductions. |
| **Scheduler bridge** | **Piloted** | `me_artifact_prepare_menudet` in `src/dsl_artifact.c` lets Python/array/graph schedulers adopt direct preparation without JSON. |
| **Cleanup step 2 (legacy deletion, artifact cleanup, build/docs)** (`miniexpr-1.0-cleanup.md`) | **Mostly pending** | `ME_DSL_PROFILE_LEGACY_DRAFT` was removed (`src/dsl_semantic_profile.h`), and core headers install without `MINIEXPR_BUILD_ARTIFACT`. Checked profile, `numpy_profile` branches, stale `Makefile`/docs, and `artifact_load` double-compile remain. |
| **Cleanup steps 5-6 (hot internals and ABI freeze)** | **Pending** | Ordinal slots/builtin op IDs (`/OPT:NOICF`), `me_eval_nd` sharing grid math, pairwise-sum extraction, and cross-platform CI (Linux/Windows/WASM/SLEEF/TCC). |

## 2. Concrete issues and gaps in the recent commits

### Correctness and parity

1. **Weak-capture JIT specialization is missing on the direct path.**
   In JSON loading, `src/dsl_artifact.c` (lines 855-868) calls `dsl_menudet_prepare_jit_constants()` for weak `ME_INT64`/`ME_BOOL` captures when initial JIT lowering could not emit C (e.g., constant loop bounds like `for i in range(n):`). `src/miniexpr_kernel.c` (lines 270-293) never calls it, so direct `me_prepare_menudet` (and `me_artifact_prepare_menudet`, which delegates to it) misses JIT or fails `ME_EXEC_REQUIRE_JIT` on kernels that JSON loading successfully JITs.
2. **Signature and annotation mismatches report `ME_ERR_SOURCE` instead of `ME_ERR_SIGNATURE`.**
   In `src/dsl_compile.c` (lines 3073-3097 and 3371-3443), parameter annotation mismatches, return-annotation conflicts, wrong input counts, and insufficient `context_ndim` all return `NULL` with `error_pos = -1`. `src/miniexpr_kernel.c` (lines 286-289) sends every `NULL` from compilation through `kernel_source_error()` -> `ME_ERR_SOURCE` (and `me_artifact_prepare_menudet` maps that to `ME_ARTIFACT_ERR_SOURCE` instead of `ME_ARTIFACT_ERR_BINDING`). `tests/test_kernel_api.c` (lines 111 and 124-125) currently only asserts `!= ME_SUCCESS`. Similarly, invalid `options` (`struct_size`/`execution`) in `src/miniexpr_kernel.c` (lines 117-126) return `ME_ERR_SIGNATURE` rather than `ME_ERR_INVALID_ARG`.
3. **`me_artifact_prepare_menudet` hardcodes all capability bits and double-parses.**
   - `src/dsl_artifact.c` (lines 1558-1560) sets all 6 `ME_ARTIFACT_CAP_*` flags unconditionally, so `me_artifact_capabilities()` reports `BLOCK_REDUCTIONS`, `ND_CONTEXT`, `FIXED_STRINGS`, and `FLEXIBLE_MATH` even on a strict numeric kernel. Derive these from `artifact->program` (reusing `artifact_block_capabilities()`).
   - It parses `source` once (line 1496) to read `entry_point` and `ME_JIT_AUTO` heuristics, then calls `me_prepare_menudet` which parses `source` a second time.
4. **Minor behavioral inconsistencies:**
   - `me_grid_eval_block` (`src/miniexpr_grid.c`, lines 142-168) does not initialize `options->fp_status` on grid preflight error or OOM, unlike `me_eval_block` (`src/miniexpr_kernel.c`, line 396).
   - `me_prepare_formula` silently ignores `options->inherit_environment` (`src/miniexpr_kernel.c`, line 410 always sets system math). Either document that formulas ignore it or honor it consistently.
   - `kernel_prepare` validates `sig->casting` in `kernel_result()` (`src/miniexpr_kernel.c`, line 302) *after* `dsl_compile_program_*` has already run JIT compilation; checking `inferred_dtype` before JIT avoids wasted work on rejected casts.
   - `kernel->info.context_ndim` is zeroed when source does not reference `_i*`/`_n*`/`_ndim`/`_flat_idx` (`src/miniexpr_kernel.c`, lines 298-300), so passing `block.ndim == sig.context_ndim` fails unless the caller checks `me_describe()`. That is reasonable (and matches "required coordinate rank"), but differs from `me_artifact_load` where `artifact->context_ndim` preserves the declared rank; spell this out explicitly in `doc/core-api.md`.

### Performance

5. **Padded grid gather/scatter does per-element division and `ninputs + 1` heap allocations.**
   In `src/miniexpr_grid.c` (lines 131-140 and 185-214), every partially padded block:
   - Calls `malloc` separately for `result` and each `scratch[i]`.
   - Calls `grid_physical_index()` (which performs `ndim` integer `%` and `/` operations) per element for every input and again on output scatter.
   - Calls `me_eval_block` even on fully padded (`valid_nitems == 0`) elementwise blocks instead of zeroing output directly after preflight.

   Recommended improvements:
   - Copy contiguous innermost runs (like `run_len` in `src/miniexpr_eval_nd.c`, lines 324-331) with an odometer increment over outer axes to eliminate per-element div/mod and turn per-element `memcpy` into row `memcpy`.
   - Coalesce `result` + `scratch[0..ninputs-1]` into one allocation (or a small stack buffer for small blocks).
   - Deduplicate `grid_range`/`grid_alignment`/`grid_overlap` with `kernel_range`/`kernel_alignment`/`kernel_overlap` via `src/miniexpr_kernel_internal.h`.
6. **All-active mask elision is missing in `me_eval_block`.**
   `src/dsl_artifact.c` (lines 1032-1042 and 1076) detects when a `valid_mask` is all `1`s and passes `NULL` downstream so SIMD/unmasked JIT runs. `src/miniexpr_kernel.c` (lines 354-356 and 417) already loops over the entire mask to validate `<= 1`, so tracking `all_active` there is essentially free and unlocks the unmasked fast path.
7. **JSON `artifact_load` still double-compiles for explicit outputs.**
   Now that `src/dsl_compile.c` (line 3566) records `program->inferred_dtype` directly during the discovery pass, `src/dsl_artifact.c` (lines 805-825) no longer needs to compile a throwaway `ME_AUTO` program just to check `dsl_numpy_can_cast()`.

### Build and CMake

8. **CMake version mismatch and embedded defaults:**
   - `CMakeLists.txt` declares `cmake_minimum_required(VERSION 3.16.3)` (line 1) and uses `CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR` (line 214), while line 248 uses `PROJECT_IS_TOP_LEVEL` (added in CMake 3.21). Make both consistent.
   - `cmake/MiniexprOptions.cmake` (lines 3-5) still defaults `MINIEXPR_BUILD_TESTS`, `MINIEXPR_BUILD_EXAMPLES`, and `MINIEXPR_BUILD_BENCH` to `ON` even when embedded via `add_subdirectory` (called out in `miniexpr-1.0-cleanup.md`).

## 3. Pre-ABI-freeze design questions to settle

Before locking down `miniexpr_kernel.h` and `miniexpr_grid.h`:

- **Struct extensibility rules:** All descriptors check `struct_size < sizeof(*s)`, which permits larger structs, but does not specify how unknown trailing bytes are handled (zero-must-be-ignored vs exact-version rejection). Pin that rule before 1.0.
- **Per-call interpreter fallback:** `miniexpr-1.0-cleanup.md` mentions allowing evaluation to select an available interpreter on a JIT-prepared kernel without recompiling; `me_run_options` does not currently have a field for that (easy to add now or reserve via `struct_size`).
- **Formula reductions and context:** Deferring them with `ME_ERR_UNSUPPORTED` in `me_prepare_formula` is the right call; keep them explicitly gated until their 1.0 semantics are pinned.

## 4. Suggested next steps (ordered)

1. **Fix direct-path parity and grid performance (small, high-value PR):**
   - Wire `dsl_menudet_prepare_jit_constants` into `me_prepare_menudet` and add a test with weak integer loop bounds + `ME_EXEC_REQUIRE_JIT`.
   - Distinguish `ME_ERR_SIGNATURE` vs `ME_ERR_SOURCE` (and `ME_ERR_INVALID_ARG` for options) and tighten assertions in `tests/test_kernel_api.c`.
   - Fix `me_artifact_prepare_menudet` capability reporting and double-parse; elide all-active masks in `me_eval_block`.
   - Optimize `me_grid_eval_block` to copy contiguous inner runs and coalesce scratch allocations.
2. **Execute cleanup step 2 ("Delete proven legacy") as separate, reviewable commits:**
   - Simplify `src/dsl_artifact.c` (lines 504-855): remove dead `numpy_profile` branches and eliminate the double-compile using `program->inferred_dtype`.
   - Retire `ME_DSL_PROFILE_MENUDET_CHECKED` after migrating/retiring cases in `tests/menudet-dsl/runner.c`, `tests/test_dsl_menudet_interp.c`, `tests/test_dsl_block_reduction_names.c`, and `src/dsl_menudet_expr.c` (line 725).
   - Clean up build and stale docs: replace/drop the stale `Makefile`, fix the 5-argument `me_eval` example in `src/miniexpr.h` (line 185), update `tests/menudet-artifacts/README.md`, default tests/examples/bench `OFF` when embedded, fix `PROJECT_IS_TOP_LEVEL`, and expose clean namespaced CMake targets (`miniexpr::core`, `miniexpr::artifact`).
3. **Broaden qualification before migrating more callers:**
   - Run `test_kernel_api`, `test_grid_api`, and downstream C-Blosc2/Python-Blosc2 tests across **Linux, Windows clang-cl, WASM, SLEEF ON, and bundled libtcc ON** (currently only macOS arm64 with SLEEF/TCC off is recorded).
4. **Move streaming pairwise-sum and share grid geometry (step 4):**
   - Split `me_pairwise_sum_*` out of `miniexpr_artifact` (`src/dsl_array.c`) into core (or a JSON-free target) so storage hosts can use it without `yyjson`.
   - Route `me_eval_nd` / `me_nd_valid_nitems` through shared grid geometry helpers once Python ND is migrated.
5. **Selective hot-path refactors and ABI freeze (steps 5-6):**
   - Replace `synthetic_var_addresses` with ordinal slots and replace function-pointer builtin identities with op IDs, then validate dropping `/OPT:NOICF` on Windows Release builds.
   - Audit exported symbols before tightening Windows visibility (`WINDOWS_EXPORT_ALL_SYMBOLS`), finalize the `b2o` v2 cross-language fixture matrix, and freeze the 1.0 headers.
