# miniexpr 1.0 implementation summary 1 (WASM32 JIT & Step 4)

Date: 2026-11-25
Base commit: `0cb6420` ("Harden kernel contracts and optimize padded-grid evaluation")
Scope: Local WASM32 JIT qualification and Step 4 (JSON-free streaming pairwise sums and shared ND grid geometry) across `miniexpr.codeberg` and sibling `../c-blosc2.codeberg` (`b2nd-constructors`).

## 1. Changes made

### WASM32 JIT qualification and coverage

- `tests/test_dsl_jit_menudet_wasm32.c`: Added `check_direct_kernel_api()` testing `me_prepare_menudet("0.2", ...)` with parameter/return annotations, a weak `ME_INT64` capture, and `ME_EXEC_REQUIRE_JIT`.
- `tests/CMakeLists.txt`: Added `kernel_api_require_wasm_jit` (`test_kernel_api --require-tcc`) when building Emscripten standalone WASM32 JIT (`MINIEXPR_WASM32_SIDE_MODULE=OFF`).

### JSON-free streaming pairwise sum (`miniexpr_reduce.h` + `miniexpr_pairwise_sum.c`)

- `src/miniexpr_reduce.h`: Moved `ME_PAIRWISE_SUM_API`, `ME_PAIRWISE_SUM_LEAF_ITEMS`, `me_pairwise_sum_state`, and `me_pairwise_sum_{init,update,append,leaves,finish}` (returning `int` with `ME_PAIRWISE_SUM_SUCCESS = 0` / `ME_PAIRWISE_SUM_ERR_INVALID_ARG = -4` for backward compatibility) into `miniexpr_reduce.h`. Added `extern "C"` guards, explicit `(const unsigned char *)` casts for C++11 compatibility, and `(uint64_t)count > UINT64_C(9007199254740992)` to clear the 32-bit WASM tautological comparison warning.
- `src/miniexpr_pairwise_sum.c` & `src/dsl_menudet_reduce.h`: Extracted the SSE2/AVX2/scalar pairwise sum implementation out of `src/dsl_array.c` and exposed internal `dsl_pairwise_sum_feed` / `dsl_pairwise_sum_accumulate` for `dsl_array.c`.
- `src/miniexpr_artifact.h`: Now includes `"miniexpr_reduce.h"` without duplicating pairwise declarations.
- `CMakeLists.txt`: Added `src/miniexpr_pairwise_sum.c` to `MINIEXPR_SOURCES` with strict FP compile flags (`-fno-fast-math;-ffp-contract=off` / `/fp:precise`) and moved `src/miniexpr_reduce.h` into the unconditional core `install(FILES ...)` list.
- `tests/test_pairwise_sum_core.c` & `tests/cmake/embedded/CMakeLists.txt`: Added core-only and embedded-consumer tests for `miniexpr_reduce.h`.
- Sibling `../c-blosc2.codeberg` (`b2nd-constructors`): Updated `bench/sum_linspace.c`, `bench/CMakeLists.txt`, and `README_DEVELOPERS.md` so `sum_linspace` links only the core `miniexpr` library without `miniexpr_artifact`.

### Shared ND grid geometry (`miniexpr_grid_internal.h`)

- `src/miniexpr_grid_internal.h`: Added shared inline helpers `me_grid_div_ceil`, `me_grid_axis_window`, `me_grid_copy_runs`, and `me_nd_locate_block`.
- `src/miniexpr_grid.c`, `src/miniexpr_eval_nd.c`, & `src/miniexpr_eval_dsl_nd.c`: Routed block window/stride computation and padded gather/scatter through the shared helpers, replacing duplicated coordinate decomposition and per-element gather/scatter loops.
- `doc/core-api.md` & `plans/miniexpr-1.0-core-api.md`: Documented the WASM32 JIT qualification, core `miniexpr_reduce.h` extraction/install, and shared ND grid geometry.

## 2. Verification and warnings

- **Native core-only (`MINIEXPR_BUILD_ARTIFACT=OFF`, `-Wall -Wextra`, bundled TCC ON, SLEEF OFF):** `433/433` tests passed (including `test_pairwise_sum_core`, `test_nd`, `test_logical_operands`, `test_grid_api`, `test_kernel_api`, `kernel_api_require_jit`, `kernel_api_require_tcc`).
- **C-Blosc2 against installed core-only prefix (no `miniexpr_artifact`):** `test_b2nd_constructors` (`1/1` passed) and `sum_linspace 1000000 4 2 2 256000 16000` passed (`b2nd_linspace (miniexpr)`, `sum = 500000 (expected 500000)`).
- **Native artifact-enabled (`MINIEXPR_BUILD_ARTIFACT=ON`, `-Wall -Wextra`):** `469/469` tests passed (including `graph_pairwise`, `grid_allocation`, `test_dsl_artifact`, `test_pairwise_sum_core`, `test_nd`).
- **C++11 header syntax check (`-std=c++11 -Wall -Wextra -Werror`):** Passed on installed `miniexpr.h`, `miniexpr_kernel.h`, `miniexpr_grid.h`, and `miniexpr_reduce.h`.
- **Embedded consumer (`tests/cmake/embedded`):** Core-only (`1/1` passed) and artifact-enabled (`2/2` passed).
- **ASan/UBSan focused suite:** `8/8` passed (`grid_allocation`, `graph_pairwise`, `test_dsl_artifact`, `test_grid_api`, `test_kernel_api`, `test_logical_operands`, `test_nd`, `test_pairwise_sum_core`).
- **WASM32 JIT (Emscripten + Node):**
  - Side-module (`MINIEXPR_WASM32_SIDE_MODULE=ON`): `12/12` focused tests passed (all eligible `test_dsl_jit_menudet_wasm32` cases reported `route=jit`, plus `test_pairwise_sum_core`, `graph_pairwise`, `test_nd`, `test_kernel_api`, `test_grid_api`, `grid_allocation`).
  - Standalone (`MINIEXPR_WASM32_SIDE_MODULE=OFF`): `3/3` passed (`test_dsl_jit_menudet_wasm32`, `test_kernel_api`, `kernel_api_require_wasm_jit`) prior to Step 2/3 edits; side-module was re-verified after all code edits.
- **Compiler warnings:** `0` warnings in all new/modified files across `-Wall -Wextra` and Emscripten builds; pre-existing warnings remain in untouched regions of `src/miniexpr.c` (`check_mixed_type_nested`, `ceil_div64`, `dsl_is_reserved_name`, `dsl_cast_int_intrinsic`), `src/dsl_array.c:30` (`a_component`), and `tests/test_bool_output.c`.

## 3. Remaining next steps

1. Review and commit the pending changes in `miniexpr.codeberg` and `../c-blosc2.codeberg` (and re-run `../python-blosc2.codeberg` tests if desired).
2. Execute Cleanup steps 2-3: migrate/retire `ME_DSL_PROFILE_MENUDET_CHECKED` and remaining legacy full-DSL/ND callers.
3. Qualify Linux and Windows `clang-cl` CI builds.
4. Settle per-call interpreter override, complete C/Python `b2o` v2 cross-language fixtures, audit symbol exports (`WINDOWS_EXPORT_ALL_SYMBOLS` / `/OPT:NOICF`), and freeze the 1.0 ABI.
