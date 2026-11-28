# Execution migration implementation status

This records implementation and local qualification, **not cross-platform completion of M0–M9**. The migration plan's
acceptance gates remain authoritative. No production default or persisted
format changed.

This records the locally qualified migration checkpoint; revision IDs in earlier
qualification sections identify their historical baselines. Release preparation
is deferred; local source overrides remain the development configuration.

| Milestone | Implemented / locally qualified work | Remaining gate / gap |
| --- | --- | --- |
| M0 | Route/test/capability inventory in `execution-migration-contracts.md`; explicit direct-handle baseline; separate benchmark preparation/specialization phases | Cross-platform performance qualification remains a later gate |
| M1 | C-Blosc2 internal `blosc/b2nd-exec.h` / `.c`; bounded typed nodes, exact captures, topology/slot/arity/width validation, copied ownership, borrowed metadata-only schedules; optional Cython conversion and strong Python schedule owner | Descriptor will need additional typed options/operations/stages as subsequent families are implemented; it is not installed/frozen |
| M2 | Fused multiple-input arithmetic maps, zero-scratch direct maps, inferred-width reduction scratch, core FP scopes and end-of-traversal raise masks; `GraphAdapter` lowering and direct authored-kernel handoff share the typed scheduler; Python canonical exports and `MenudetError`/`fp_status` compatibility now work without native artifact ownership | Wrapper dispatch is implemented (M8); final obsolete-pilot/wiring cleanup and cross-platform qualification remain |
| M3 | Scalar/ranked/N-D broadcasting; empty domains; strided, reversed, transposed, unaligned and non-native views with checked allocation extents and bounded gather scratch; whole/partial-axis reductions with `initial`, `keepdims`, pairwise sum, core accumulator typing and empty identities; disjoint strided outputs and separately budgeted transactional alias staging. Resident `where` masks and explicit accumulator casts now use typed plans, including borrowed kernels | Reader-backed masks and automatic conditional/masked multi-stage domains remain rejected; platform qualification remains |
| M4 | C-owned jobs enforce partition proofs, the 64-task cap, aggregate scratch bounds, per-job FP scopes, logical-order partial combination and joining before return. POSIX uses C-Blosc2's persistent pool with nesting guards; Windows remains serial. Backend reporting, submission failures and shared-schedule concurrency are tested. Core partition proofs cover supported straight-line numeric operations and bounded floating loops/control flow; typed compiler/FP/ULP policy is captured in preparation/cache identity | Live Windows/WASM and x86-64 qualification, including non-default FP policies, remains. Non-strict kernels and ordered sum/prod remain conservatively serial; broader proofs are optional extensions |
| M5 | Compressed storage execution (`b2nd_exec_execute_arrays` / `_ex`, `b2nd_exec_execute_to_array`): binds in-memory and frame-backed (`cframe` `contiguous=True` and `urlpath` on-disk, without mutating `current_nchunk`) chunked arrays across native numeric dtypes (`bool`, `int8..64`, `uint8..64`, `float32`, `float64`), right-aligned N-D broadcasting, partial-axis reductions (`axis`, `keepdims`, `initial`, `pairwise_sum`, parallel extrema, `argmin`/`argmax`, `var`/`std` with `ddof`), mixed compressed + resident inputs, and parallel partitionable jobs over the persistent shared pool with bounded 8-slot chunk LRU decode cache, `max_decode_bytes` task clamping, contiguous inner-block run copies in `storage_read`, and empty `0`-item / `chunkshape=0` storage validation; direct elementwise compressed sinks (`out` = `b2nd_array_t` / `NDArray`) evaluate blocks inside Blosc2's compression prefilter with matched-layout zero-copy block reads (`blosc2_getitem_ctx`, `gather_bytes=0`), 4-slot LRU block cache for mismatched/broadcast compressed inputs, resident box gathering, and padded live compaction without mutating output before validation. Explicit Python `NativeGraph(..., _executor="c-blosc2")` lowers `argmin`/`argmax`/`var`/`std` (`ddof`), executes resident/compressed/frame-backed inputs and sinks with the GIL released; Release ARM64 sweep confirms `1.00x` chunk decode amplification and 22x–60x speedups over Python adapter sources; no Python source/sink reads or whole-array operand materialization | Cross-platform Windows/WASM performance/FP qualification and x86-64 Release confirmation remain |
| M6 | Metadata-only generic source/sink callbacks (`b2nd_exec_execute_sources`), GIL-safe Cython trampolines, bounded index/gather/output buffers, and explicit persisted logical partitions (`b2nd_exec_prepare_borrowed_kernel`, `b2nd_exec_specialize_partitions`, `b2nd_exec_execute_partitions`) with `MenudetLazyArray` / `MenudetArtifactHandle.evaluate_array` & `evaluate_sources` handoff across real, complex, fixed-width string, block-scalar, 2-D context, 0-D scalar, and generator kernels | Cross-platform Windows/WASM qualification and M9 adapter retirement remain |
| M7 | General multi-stage typed graph planner and executor (`b2nd-exec-stages.h`/`.c`, up to `B2ND_EXEC_MAX_STAGES` = 32): direct node/slot remapping without child serialization; automatic region splitting at reductions and strong-dependent shared computed nodes; explicit `b2nd_exec_staged_assemble`/`_ex` composition (including authored Menudet kernels and single-stage resident masked reductions); inferred intermediate dtypes, casts, `initial`, `keepdims`, unary/binary/bitwise/comparison/`select`/`and`/`or`/`pos`/`function` ops; specialization-time intermediate sizing, `last_consumer` analysis, and upfront budget rejection before source reads; invocation-local last-consumer intermediate freeing; stage/node diagnostics across prepare, assemble, specialize, and execute (`b2nd_exec_stage_diagnostic`); `AND`-combined `has_jit` and `fp_observed` support; explicit compressed-stage, automatic conditional/masked multi-stage, shared scalar-only, and reader-backed mask rejections before reads; arbitrary non-contiguous / F-order staged `out=` views via `b2nd_exec_staged_execute_sources_into`; Python-owned canonical JSON (`to_json()`/`map_json()`) and stage import (`typed_stage_import.py`, `b2nd_exec_stages_bridge.h`) without consulting the native artifact/interchange parser | Compressed/parallel staged execution and cross-platform Windows/WASM qualification remain |
| M8 | Ordinary caller handoff (resident & large compressed `LazyExpr`, `_ConfiguredResidentSchedule`, authored `MenudetLazyArray` lane-local maps + `_cblosc2_partition_values`), explicit NativeGraph default dispatch, core Menudet ownership, Python-owned v1 import, compiler environment selection, resident masks/explicit accumulators, sparse participation reads, artifact-off direct `NativeGraphHandle` compatibility, comparison-only adapter gating, typed HIGH/MEDIUM plus strict/contract/fast FP preparation, and coordinated pilot & pilot-only native build retirement locally qualified | x86-64/Windows/WASM qualification remains; production build-default changes belong to M9 |
| M9 | Clean Release artifact-off candidate built/installed in isolation; original v1 and 4.17.0 v2 exchange fixtures execute; one core archive/provider, no native JSON/adapter/pilot symbols or downloads; installed C consumers `5/5` + superbuild `1/1`; independent compression-only C-Blosc2 `1681/1681`; stub/core-only syntax checks pass; C1–C3 public C-Blosc2 execution API and `python-blosc2` cutover complete locally on macOS ARM64 | C4 cross-platform installed-consumer / ABI gate (Linux x86-64, Windows/MSVC, WASM), exact published dependency pins, cross-platform Release/performance gates, standard non-experimental artifact-off default cutover, and non-opt-in compatibility helper removal remain |
| C1–C4 | Public self-contained [`include/b2nd_exec.h`](file:///Users/faltet/blosc/c-blosc2.codeberg/include/b2nd_exec.h) (`B2ND_EXEC_API_VERSION 1`) + optional [`include/b2nd_exec_kernel.h`](file:///Users/faltet/blosc/c-blosc2.codeberg/include/b2nd_exec_kernel.h), atomic-refcounted `b2nd_plan_t` / `b2nd_schedule_t` facade, one-shot `b2nd_eval` / `b2nd_eval_into`, always-built compression-only stubs, pure-C installed consumers (`7/7`, ASan+UBSan `7/7`), and `python-blosc2` C3 cutover with zero private `blosc/b2nd-exec*.h` includes locally qualified on macOS ARM64 (artifact-off, artifact-on, and `USE_SYSTEM_BLOSC2=ON` compression-only) | C4 cross-platform installed-consumer, compression-only link/stub, and ABI checks on Linux x86-64, Windows/MSVC, and WASM before freezing the public ABI in a C-Blosc2 release |

## CI qualification rollout (2026-11-28)

The dedicated workflow in Python-Blosc2 is
`.github/workflows/execution-migration.yml`, implementing
[`execution-ci-qualification.md`](execution-ci-qualification.md).
Development branches have been published to GitHub; shipping pins and production
defaults remain unchanged. CI records its native pair in
`.github/execution-native-refs.json`:

- MiniExpr `adfbdf4752fec9a6a59ab71461e824de36ea239d`.
- C-Blosc2 `9d7b9206f426bd979a5978c2333a31c8638abe7f`.
- Python-Blosc2 qualification candidate `828a7eb453b89f6294bbf82a77374905ef628005`.

The workflow builds isolated artifact-off/on wheels, checks installed headers,
exports/link inputs, compression-only stubs and external C/C++ consumers, and
retains CTest/pytest reports, capabilities and build logs. Linux has Release and
ASan/UBSan jobs; Windows has clang-cl numerical builds and independent MSVC
installed-C/DLL consumers plus compression-only builds. MiniExpr's native C99
complex implementation still requires clang-cl; this is not an MSVC numerical
core port. WASM runs installed static consumers and a separate Pyodide selection.

CI-exposed fixes include Release assertions in all installed fixtures, Windows
pool-cleanup guards, deterministic persistent-worker probes, portable frame paths,
Emscripten installed CMake exports and inline logical-job transport, SIMD cache
thread-exit cleanup, and capability-specific FP/JIT/worker-count expectations.
WASM's unavailable FP flags are tested explicitly: unsupported raise masks reject
without publishing output. The POSIX pool fixture is explicitly skipped on
Windows/WASM; this is not parallel-execution qualification on those platforms.

Evidence before the final candidate:

| Run / configuration | Result |
| --- | --- |
| [46145284552](https://github.com/Blosc/python-blosc2/actions/runs/46145284552), Python `ce61af49` | All native C4 jobs passed, including WASM installed static consumers and both Windows compiler configurations; all six native Python artifact-off/on jobs passed; Pyodide artifact-off passed. Artifact-on exposed two remaining FP-observation-only test assumptions. |
| [46148198120](https://github.com/Blosc/python-blosc2/actions/runs/46148198120), Python `c35c9892` | Both Pyodide artifact modes and all other completed jobs passed after those assumptions were corrected. The pending native WASM job was cancelled by the next push; this run is not an all-green matrix. |
| Local isolated artifact-off Python selection | `1802 passed, 87 skipped, 584 deselected`; optional native-semantic reference cases remain deselected. |
| Local installed native consumers | `7/7` passed; WASM `6` passed plus `1` explicit POSIX-launcher skip. |
| NumExpr-free subprocess regression | Simplified native-expression smoke test passes with installed wheels (`9/9` for its file) and an explicit split-editable/finder-only extension reproducer (`1/1`). |

The regular compatibility workflow exposed an editable-import isolation bug:
copying `sys.path` under `python -S` did not preserve the extension finder.
The final candidate explicitly selects the parent's package and compiled-module
locations while keeping NumExpr imports blocked, with no duplicated persistence
or kernel-roundtrip matrix.

Final-candidate CI:
[regular tests](https://github.com/Blosc/python-blosc2/actions/runs/46150819001)
passed all six jobs: Linux x86-64 (Python 3.12/3.14 and NumPy 1.26), Linux ARM64,
macOS ARM64 and Windows clang-cl. The NumExpr smoke test passes, and Windows
compatibility tests now retain the legacy adapter's parallel-worker expectations.
The [execution migration matrix](https://github.com/Blosc/python-blosc2/actions/runs/46150818902)
passed all 15 required jobs (the optional performance job was not selected).
This closes the recorded C4 installed-consumer/cross-platform correctness matrix,
not ABI freeze, stable-hardware performance acceptance or M9 default cutover.
These results supersede the platform-correctness gaps in the historical milestone
ledger above for the admitted routes; documented unsupported families remain
unsupported.

| Final-candidate configuration | Result |
| --- | --- |
| Linux x86-64 GCC Release / Clang ASan+UBSan | Core `434/434`, installed consumers `7/7`, public API/example `2/2`, numerical and stub shared consumers `1/1` each, compression-only `2123/2123` in each configuration |
| Linux x86-64 sanitizer owner fixtures | Executor `5/5`, storage `2/2`, stages `2/2`; native core and codec builds are also instrumented in this job |
| macOS ARM64 native Release | Core `434/434`, installed consumers `7/7`, public API/example `2/2`, numerical/stub shared consumers passed, compression-only `1682/1682` |
| Windows MSVC consumers / clang-cl consumers | Installed consumers `6` passed + `1` explicit POSIX-launcher skip in each configuration; public API/example `2/2`, strict C/C++ shared/DLL and stub consumers passed; compression-only `1698/1698` with each compiler; numerical core `341/341` built with clang-cl |
| WASM Emscripten installed static consumers | `6` passed + `1` explicit POSIX-launcher skip; public API/example `2/2`, numerical and compression-only stub consumers passed |
| Linux/macOS isolated Python artifact-off / artifact-on | `1806 passed, 87 skipped` / `1847 passed, 46 skipped` per platform |
| Windows isolated Python artifact-off / artifact-on | `1654 passed, 239 skipped` / `1695 passed, 198 skipped`; unavailable JIT capabilities remain explicit skips |
| Pyodide artifact-off / artifact-on | `143 passed, 80 skipped` / `176 passed, 47 skipped`, including explicit unavailable-FP and inline-logical-job checks |

Native Python selections still deselect the optional `584`-case semantic reference
matrix. WASM runtime selection is bounded rather than the full native Python
suite. Export/link audits, header version checks and capability reports are
retained in workflow artifacts, not copied into this repository.
Warnings remain visible in CI logs, including existing codec/Cython diagnostics,
WASM platform/compiler-option warnings, and an unused POSIX generic-pool wrapper
in the WASM build. Passing CI is not a warning-free-build claim.

The optional Linux x86-64 shared-runner performance diagnostic
[46153143461](https://github.com/Blosc/python-blosc2/actions/runs/46153143461)
completed successfully, including the repeated correctness matrix and performance
job. Its four-round interleaved results flag compressed `min(x + 1)` at `1.47x`
reference time; the other four representative cases are within the 5% threshold
or faster (`1.03x` resident map, `0.97x` resident min, `0.79x` compressed map,
`0.99x` compressed sink). The ordered compressed-sum cache probe reaches `1.00x`
decode amplification (`64` decodes for `64` chunks) with its eight-slot budget.
These are diagnostic observations, not stable-hardware performance acceptance.
The CI qualification implementation and its execution are complete; performance
acceptance, ABI freeze and M9's release/default changes remain open.
The interleaved performance runner completes locally against rebuilt isolated
candidate/reference wheels. Four rounds on macOS ARM64 reproduce a roughly 27%
slowdown for compressed `min(x + 1)` with four workers; one/eight-worker probes
are within 5%. Both paths use JIT; the typed path uses full-chunk decode/gather
scratch and decodes 65 chunks versus 62 prefilter fetches. This identifies a
route/memory-traffic investigation, not a proven single cause or accepted
regression. Retain the gate before any corresponding default cutover.

## Earlier qualification: C-Blosc2 public execution API (C1–C3) & C4/cross-platform handoff (2026-11-28)

### Current local branch/commit baseline (unpushed)

- **`miniexpr.codeberg` (`next`)**: `a4d4d80` (plans updated with C1–C3 rollout; code base `acf896f`; updated status commit follows this record)
- **`c-blosc2.codeberg` (`next-execution-b2nd`)**: `11af7649` (C1+C2 public `include/b2nd_exec.h` + `include/b2nd_exec_kernel.h`, refcounted facade `blosc/b2nd-exec-api.c`, compression-only stubs, `tests/b2nd/test_b2nd_exec_api.c`, `examples/b2nd/example_exec_eval.c`, `tests/installed-miniexpr/public-exec*.c`, plus C3-driven fixes to option wording `"Invalid execution options value"` and `b2nd_plan_from_graph` shared-node auto-staging)
- **`python-blosc2.codeberg` (`next`)**: `92b32372` (C3 public-header cutover in `CMakeLists.txt`, `src/blosc2/b2nd_exec_bridge.h`, `src/blosc2/b2nd_exec_stages_bridge.h`, `src/blosc2/b2nd_exec.pxi`, and `src/blosc2/typed_stage_import.py`, built on `8ff4cbfd` compressed-sink default worker scratch auto-lift)

### C1–C3 implementation & macOS ARM64 verification summary

| Target / Configuration | Result (macOS ARM64) |
| --- | --- |
| C-Blosc2 MiniExpr build + install | `test_b2nd_exec_api` + `b2nd_example_exec_eval` `2/2`; `tests/installed-miniexpr` `7/7` (including pure-C `installed_public_exec` with no MiniExpr headers/linkage and `installed_public_exec_kernel`); `nm -gU` exports 15 public symbols; `b2nd_exec.h` + `b2nd_exec_kernel.h` pass C99/C11/C++17 `-Wall -Wextra -Werror -fsyntax-only` |
| C-Blosc2 MiniExpr ASan + UBSan | `test_b2nd_exec_api` `1/1`; `tests/installed-miniexpr` `7/7` |
| C-Blosc2 compression-only (`BLOSC2_MINIEXPR_ROOT` unset) | `b2nd_exec.h` installed and `b2nd_exec_kernel.h` absent; `nm -gU` exports 14 stubs and `0` `b2nd_plan_from_kernel` / `_me_*` symbols; `test_b2nd_exec_api` + `b2nd_example_exec_eval` `2/2`; external shared consumer binary passed |
| Python-Blosc2 C3 artifact-off wheel (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`, `MINIEXPR_BUILD_ARTIFACT=OFF`) | Wheel installs `b2nd_exec.h` + `b2nd_exec_kernel.h`; `nm -gU` shows `0` `me_artifact\|yyjson\|me_graph` symbols; `scripts/qualify_execution_migration.py --artifact off` `1765 passed, 83 skipped, 584 deselected`; `test_execution_cleanup.py` + `test_resident_prefilter.py` `37 passed, 2 skipped` |
| Python-Blosc2 C3 artifact-on wheel (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`, `MINIEXPR_BUILD_ARTIFACT=ON`) | `test_b2nd_exec_plan.py`, `test_b2nd_exec_stages.py`, `test_b2nd_exec_m8.py`, `test_execution_cleanup.py`, `test_resident_prefilter.py` `210 passed, 1 skipped` |
| Python-Blosc2 `USE_SYSTEM_BLOSC2=ON` compression-only installed prefix (`blosc2.pc`) | Verified `_b2nd_exec_available() is False` and `menudet_legacy_adapter_available() is True`; `test_execution_cleanup.py`, `test_b2nd_exec_m8.py`, `test_resident_prefilter.py`, `test_native_first.py`, `test_storage_readers.py`, `test_lazyexpr*.py` `232 passed, 38 skipped` |
| Header / lint / private-include hygiene | `cc -std=c99 -Wall -Wextra -Wno-unused-function -Werror -fsyntax-only` clean on `b2nd_exec_bridge.h` + `b2nd_exec_stages_bridge.h` in stub and `-DB2_HAVE_B2ND_EXEC` installed-prefix modes (zero `c-blosc2/blosc/` paths); zero `b2nd-exec*` / `blosc2_SOURCE_DIR/blosc/` matches in `CMakeLists.txt` or `src/`; `ruff check` & `ruff format --check` clean on `typed_stage_import.py` |

### Handoff checklist for next agent (C4 & cross-platform qualification)

1. **C4 installed-consumer & ABI gate across platforms (Linux x86-64, Windows/MSVC, WASM):**
   - Run `test_b2nd_exec_api`, `b2nd_example_exec_eval`, `tests/installed-miniexpr` (`7/7`, including pure-C `installed_public_exec` without MiniExpr headers/linkage and `installed_public_exec_kernel`), and compression-only install/link/runtime stub checks on Linux x86-64 (Release + ASan/UBSan), Windows/MSVC (verifying `BLOSC_EXPORT` DLL import/export, `_InterlockedIncrement` / `_InterlockedDecrement` refcounting, serial launcher, and `Blosc2Config.cmake` / `blosc2.pc`), and WASM/Emscripten.
   - Verify `*_V1_SIZE` struct-size versioning and `B2ND_EXEC_API_VERSION 1` compatibility across compilers before freezing the public ABI.
2. **Cross-platform M0–M9 correctness & performance gates:**
   - Build and run `scripts/qualify_execution_migration.py` (artifact-off and artifact-on) and `scripts/qualify_execution_platform.py` on Linux x86-64, Windows, and WASM (including non-default FP policies and compressed/resident/staged routes).
   - Run Release performance gates on Linux x86-64 (`bench/execution_handoff.py`, `bench/compressed_reduction_plan.py`, compressed-to-compressed sink benchmarks) alongside macOS ARM64.
3. **Final M9 release cutover (after CI passes & commits are pushed):**
   - Push/publish exact MiniExpr and C-Blosc2 revisions, advance `B2_MINIEXPR_REVISION` and `BLOSC2_BUNDLED_VERSION` pins in `python-blosc2/CMakeLists.txt`, qualify clean no-source-override wheels, switch the production default to `B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON` + `MINIEXPR_BUILD_ARTIFACT=OFF`, and retire remaining non-opt-in compatibility helpers/adapter wiring.
   - Reminder for local development builds before pins are pushed: pass `-DFETCHCONTENT_SOURCE_DIR_MINIEXPR=/path/to/miniexpr` and `-DFETCHCONTENT_SOURCE_DIR_BLOSC2=/path/to/c-blosc2` in `CMAKE_ARGS`.

## Earlier qualification: ordinary-caller handoff & pilot retirement (2026-11-28)

- **Ordinary-caller handoffs (`python-blosc2`):**
  - `compute_native_graph` routes admitted typed plans directly to C-owned whole-domain schedules (`TypedGraphSchedule.execute` / `execute_arrays` / staged execution) with metadata-only admission via shared `resident_execution.typed_storage_inputs_eligible`, keeping compatibility readers for postfiltered, non-native-endian, and plugin codec/filter (`codec > ZSTD`, `filter > DELTA`) inputs.
  - `MenudetArtifactHandle.typed_plan()` and `resident_execution.prepare_kernel_schedule()` borrow compiled lane-local numeric kernels without recompiling compiler/FP policy, serving both `_ConfiguredResidentSchedule.execute` (checked after `self.partitionable`) and whole-domain `MenudetLazyArray._values` ahead of `_cblosc2_partition_values` and compatibility fallbacks.
  - Ownership-guard tests (`tests/test_execution_cleanup.py`) forbid Python pool submission (`resident_pool`, `run_dynamic`), `_parallel_*`, `_execute_resident_kernel`, `_prefilter_compressed_map`, and `evaluate_prefilter_numpy` on admitted typed routes; existing route/failure-injection assertions in `test_resident_prefilter.py`, `test_lazyexpr_output_dispatch.py`, `test_lazyexpr_resident.py`, `test_native_first.py`, and `test_storage_readers.py` are ported while preserving non-opt-in compatibility coverage.
- **Native fixes (`c-blosc2`):**
  - `blosc/b2nd-exec.c`: `plan->owns_kernel = true` is set immediately upon `me_prepare_menudet` success so post-prepare validation failures do not leak the kernel.
  - `blosc/b2nd-exec-storage.c`: an explicit `jobs->max_worker_scratch == 0` cap rejects scratch-requiring execution before decoding or sink writes instead of acting as unlimited.
  - `tests/installed-miniexpr/exec-plan.c` adds `#undef NDEBUG` so `assert(...)` checks execute under Release builds (including `superbuild`).
- **Coordinated single-case pilot & pilot-only native build retirement:**
  - Removed C-Blosc2 internal `blosc/b2nd-kernel.{h,c}`, `tests/installed-miniexpr/kernel-grid.c`, and their CMake entries (replacing `parent_kernel_grid` with `parent_exec_plan`); ported `int64` overflow wrap-before-min and 1-D logical partition `{7}/{3}` `block_sum` `{6, 15, 7}` + exact budget-1 rejection into `exec-plan.c` and `exec-storage.c`.
  - Removed Python-Blosc2 `src/blosc2/b2nd_kernel_pilot.h`, `B2_HAVE_B2ND_KERNEL_PILOT`, pilot-only `CMakeLists.txt` fallback (now requiring `b2nd-exec.h` whenever `B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`), `_ResidentMinI64Plan`, `_ResidentMapMinI64Plan`, `_evaluate_b2nd_kernel_pilot`, `_scalar_pilot_values`, `compute_c_blosc2_min_if_eligible`, `bench/resident_min_pilot.py`, and `tests/test_b2nd_kernel_pilot.py` (porting persisted-partition, multi-group FP status, and empty-domain contracts to `tests/test_execution_cleanup.py` and converting `test_native_graph_preparation.py` + `test_b2nd_exec_m8.py`).

Combined-tree local qualification (macOS ARM64, isolated `-S` wheel imports):

| Configuration | Result |
| --- | --- |
| Artifact-off shared-core wheel (`qualify_execution_migration.py`) | `1765 passed, 83 skipped, 584 deselected` |
| Artifact-off shared-core wheel (`test_execution_cleanup.py`, `test_resident_prefilter.py`) | `36 passed, 2 skipped` |
| Artifact-on shared-core wheel (`qualify_execution_migration.py`) | `1804 passed, 44 skipped, 584 deselected` |
| Artifact-on shared-core wheel (`test_execution_cleanup.py`, `test_resident_prefilter.py`) | `38 passed` |
| Standard non-opt-in wheel (`qualify_execution_platform.py` selection / 13-file ported compatibility selection) | `262 passed, 23 skipped` / `642 passed, 24 skipped` |
| MiniExpr artifact-off / artifact-on C suites | `343/343` / `470/470` |
| Installed artifact-off C consumers / superbuild `parent_exec_plan` | `5/5` / `1/1` |
| Native executor / storage / stages ASan+UBSan fixtures | `5/5` / `2/2` / `2/2` |
| Standalone compression-only C-Blosc2 (no MiniExpr) | `1681/1681` |

`nm` symbol audit on the artifact-off extension (`blosc2_ext.abi3.so`) confirms `0` `yyjson_*`, `me_artifact_*`, `me_graph_*`, `b2nd_min_i64*`, `b2nd_map_min_i64*`, `b2nd_eval_kernel_grid`, and `b2nd_eval_kernel_partitions_1d` symbols.
Ruff lint/format checks and C syntax checks (`cc -std=c99 -Wall -Wextra -Wno-unused-function -Werror -fsyntax-only` on `dsl_artifact_bridge.h`, `b2nd_exec_bridge.h`, and `b2nd_exec_stages_bridge.h` in stub and core-only modes) pass cleanly.

Release macOS ARM64 `bench/execution_handoff.py` (`16,000,019` float64 items, `workers=8`, JIT, artifact-off typed wheel vs standard non-opt-in wheel):

| Case | Artifact-off typed (`route`, `tasks`, `warmed_ms`) | Standard non-opt-in (`route`, `tasks`, `warmed_ms`) |
| --- | --- | --- |
| Resident `x + 1` | `c-blosc2-typed-resident`, `tasks=8`, `5.217 ms` | `parallel-resident`, `tasks=8`, `5.015 ms` |
| Resident `min(x + 1)` | `c-blosc2-typed-resident`, `tasks=8`, `3.364 ms` | `parallel-resident-reduction`, `tasks=1`, `3.450 ms` |
| Compressed `x + 1` | `c-blosc2-typed-storage`, `tasks=8`, `9.493 ms` | `menudet-prefilter`, `tasks=8`, `37.474 ms` (`3.95x` speedup) |
| Compressed `min(x + 1)` | `c-blosc2-typed-storage`, `tasks=8`, `7.377 ms` | `parallel-graph-bounded-readers`, `tasks=8`, `7.122 ms` |

## Earlier qualification: review 5 follow-ups (2026-11-28)

- Math backend environment selection now overrides compute options, matching
  compiler selection and Menudet execution snapshots. Environment-only `auto`
  and `scalar` values remain admitted; `compute_options` still accepts only its
  existing documented backend values. Explicit NativeGraph MEDIUM policy leaves
  automatic backend selection to the core; the ordinary-expression frontend's
  platform-specific preference is not newly imposed on explicit graph handles.
- Artifact-off direct handles preserve AUTO JIT instead of converting mode `3`
  to `False`; invalid modes are rejected. Typed preparation now passes
  `ME_EXEC_AUTO` through C-Blosc2 to native structural classification. Explicit
  environment JIT overrides are captured before cache lookup, preventing reuse
  of an interpreted plan after `ME_DSL_JIT` changes.
- Assembled-plan FP/math metadata is explicitly documented as root-stage policy,
  not a guarantee that producer policies agree. A heterogeneous fast/strict
  native-stage regression verifies this contract. Backend counters explicitly
  include internal accumulator-conversion calls; these are not hidden from the
  interpreter counts.
- Removed the unreachable non-shared artifact-off link branch. No dependency
  pins or production defaults changed.

Combined-tree local qualification (installed wheels use isolated `-S` imports):

| Configuration | Result |
| --- | --- |
| Artifact-off shared-core wheel | `1767 passed, 83 skipped, 584 deselected` |
| Artifact-on shared-core comparison wheel | `1806 passed, 44 skipped, 584 deselected` |
| Standard non-opt-in wheel, selected graph/Menudet compatibility tests | `262 passed, 23 skipped` |
| MiniExpr artifact-off / artifact-on suites | `343/343` / `377/377` |
| Installed artifact-off C consumers | `6/6` |
| Native executor / storage ASan+UBSan fixtures | `4/4` / `2/2` |
| Native staged ASan+UBSan fixture, including adapter differential | `3/3` |

Sanitizer instrumentation covers the fixture executor/storage/stage sources, not
the installed static core/codec dependencies or comparison-adapter archive.
The artifact-on wheel's additional skip relative to earlier runs
is the new artifact-off-only direct-handle delegation regression.
The standard wheel's first build attempt encountered a stale Make/Ninja
FetchContent cache; rebuilding in a fresh external build directory succeeded.
These runs replace the pre-V5-only evidence for the listed configurations, not
the still-outstanding cross-platform/performance gates.
Ruff lint/format, three-repository whitespace checks and the core-only bridge
syntax check with warnings treated as errors also pass.

## Earlier qualification: non-default graph FP policies (2026-11-28)

The artifact-off wheel now admits HIGH/MEDIUM accuracy and environment-selected
strict/contract/fast (`relaxed` alias) policy without adapter fallback:

- MiniExpr's append-only `ME_PREPARE_OPTIONS_V5_SIZE` adds typed FP mode and SLEEF
  accuracy preferences. V1–V4 callers retain their previous defaults; authored
  source policy retains precedence. Formula preparation remains strict-only.
- C-Blosc2 graph options pass those preferences to core preparation, including
  automatically split stages, rather than imposing a generated strict pragma.
  Native plan metadata reports effective FP mode, math backend and SLEEF accuracy.
- HIGH prefers available SLEEF u10; explicit SLEEF uses u10/u35 for HIGH/MEDIUM.
  An unavailable optional SLEEF preference keeps system math, while explicit
  backend requests remain authoritative. Capability probing is metadata-only
  and independent of the caller's environment/compiler mode.
- Runtime policy participates in typed plan cache identity and propagates through
  JSON import and canonical map export. Core v1 import now inherits the runtime
  environment after Python's deliberate policy-override normalization.
- TCC's existing strict-only restriction remains fail-closed. Contract/fast tests
  use the system compiler, including required-JIT execution; no adapter replay is
  used to hide an invalid compiler/FP combination.

The newly installed artifact-off wheel passes **`1764 passed, 83 skipped, 584
deselected`**, including resident/compressed input, staged execution, JSON/map
roundtrips, SLEEF u10/u35, required system JIT, unavailable-preference fallback,
cache separation and invalid TCC/non-strict policy tests. Ruff and native validation
results are recorded with the local qualification below.

**Release-only follow-up (deferred): qualify exact dependency revisions without source overrides.**
Neither MiniExpr HEAD `1cc0063539f8c0c0dcd374726906cd8b4f09f814` nor C-Blosc2 HEAD
`2ac1978b1a5b644db123ecf576dcf2ccd1e18844` contains the required FP policy API;
it currently exists in the working trees. The checked shared-core dependency pins
also predate this migration increment. Exact published revisions containing these
changes are required before any future pin advancement and clean no-source-override
wheel qualification. This is not a blocker to development or a request to release.
Build guards now reject older native FP APIs explicitly. Production pins/defaults
have not been changed.

The latest FP increment also passes the artifact-off MiniExpr C suite (`343/343`),
Ruff lint/format checks, diff whitespace checks and core-only bridge syntax checks
with warnings treated as errors. The older artifact-on wheel, standard wheel,
installed C consumer and sanitizer counts below predate the V5 FP-policy
increment; the review 5 section above records their combined-tree requalification.

## Earlier qualification: public-entry-point closure and clean artifact-off candidate

The remaining M8 public-route regressions are closed locally on macOS ARM64.
This section supersedes the historical M8/M9 gaps in the ledger above for compiler
environment selection, resident masks, explicit accumulators, direct handle
compatibility, comparison-test gating and local dependency/provider auditing.
Production build defaults and dependency pins remain unchanged.

- **MiniExpr:** formula inferred dtype is retained before final conversion.
  The adapter's bounded floating loop/control-flow partition proof is shared with
  the core, preserving prefilter eligibility for proven loops with final narrowing.
- **C-Blosc2:** `b2nd_exec_prepare_borrowed_kernel_ex` appends a resident Boolean
  participation slot without changing the borrowed kernel signature. Explicit
  accumulator conversions use a separate core conversion tile, counted in the
  mapped scratch budget. Callback readers read only participating runs; reader-backed
  masks still reject before callbacks. Both map and conversion FP flags are observed.
- **Python:** artifact-off direct `NativeGraphHandle` calls use typed ownership;
  auto admission preserves its original error when no adapter exists. Temporary
  legacy handles are acquired inside cleanup scope. `ME_DSL_JIT_COMPILER` is admitted
  as preparation policy. Unsupported typed operations retain both the public
  `MenudetError` contract and `NotImplementedError` admission classification;
  required-JIT failure remains a non-fallback `MenudetError`. JSON import rejects
  unreachable constants as well as unreachable computed nodes.
- **Test separation:** only tests explicitly exercising legacy comparisons,
  borrowed legacy artifacts or Python adapter pool submission require the adapter.
  The default-dispatch test runs artifact-off. Subprocess tests preserve the chosen
  installed package instead of processing unrelated editable-install startup hooks.

Validation:

| Configuration | Result |
| --- | --- |
| Clean Release artifact-off wheel, isolated installation | `1750 passed, 83 skipped, 584 deselected` |
| Artifact-on typed comparison wheel, isolated installation | `1790 passed, 43 skipped, 584 deselected` |
| Native artifact-off MiniExpr suite | `343/343` |
| Native artifact-on MiniExpr suite | `377/377` |
| Installed artifact-off C consumers | `6/6` |
| Native owner ASan/UBSan fixtures | executor `4/4`, storage `2/2`, stages `3/3` (including adapter differential) |
| Standalone compression-only C-Blosc2, no MiniExpr | `1688/1688` |
| Independent core-vs-legacy v1 import differential | `10/10` valid/invalid cases match values, inferred dtype, cardinality, FP status and error locations |
| Original 4.17.0 exchange directory | Original v1 carrier and v2 maps, grouped partitions, CFrame and table computed metadata reopen and execute without the adapter |

The Python selections use the existing default native-semantic deselection policy;
they are not claims that the optional 584-case reference matrix ran. The 40 additional
artifact-off skips are explicit adapter-only qualification tests.
The native owner sanitizer fixtures use the candidate's installed static core/codec
dependencies; those dependencies are not themselves sanitizer-instrumented.
Edited Python files pass Ruff lint/format checks, and bridge stubs plus core-only
headers pass `cc -fsyntax-only -Wall -Wextra -Werror` (unused static helpers excluded).

The clean artifact-off build has no yyjson source/download directory, no
`yyjson_*`/`me_artifact_*`/`me_graph_*` extension symbols, one core archive in the
extension link command, and one `me_prepare_formula` provider. Its dynamic dependencies
are only Accelerate and the system library. Compression-only C-Blosc2 is qualified
separately, not inferred from the numerical build.

Reproduction lives in Python-Blosc2's `scripts/qualify_execution_migration.py`,
`scripts/compare_menudet_v1.py` and `scripts/qualify_menudet_migration.py`.
Build candidate/reference wheels with the same local source overrides and install
with `pip --target`; the runners use conda Python with `-S`, assert package/extension
locations and do not modify the shared editable installation. Raw logs and wheels
remain outside the repositories.

**Still release gates:** the clean candidate uses explicit local source overrides,
not newly published dependency pins. Publish/qualify the exact native revisions
before advancing pins. x86-64/Windows/WASM qualification,
the production artifact-off default switch and final
superseded-wiring removal remain open. No platform/performance gate is waived.

## Earlier qualification: M8 Part 2 — Core Menudet decoupling & first artifact-off Python wheel

Following the M8 wrapper cutover, M8 Part 2 decoupled `MenudetArtifactHandle`, `MenudetKernel`,
Menudet prefilters (`MenudetResidentPrefilter`, `_execute_menudet_prefilter`), and
`_evaluate_b2nd_kernel_pilot` from `miniexpr_artifact` onto MiniExpr core `me_kernel` + `b2nd-exec`,
making `MINIEXPR_BUILD_ARTIFACT` an optional CMake toggle in `python-blosc2`:

- **MiniExpr Core API (`miniexpr.codeberg`)**:
  - Extended `src/miniexpr_kernel.h` and `src/miniexpr_kernel.c` with `ME_EXEC_AUTO = 3`,
    `ME_MENUDET_FEATURE_ND_CONTEXT` / `ME_MENUDET_FEATURE_FLEXIBLE_MATH`, `me_prepare_options.jit_options`
    (`ME_PREPARE_OPTIONS_V4_SIZE`), and the six `ME_KERNEL_METADATA_API` getters
    (`me_kernel_inferred_dtype`, `me_kernel_entry_point`, `me_kernel_execution_kind`,
    `me_kernel_math_backend`, `me_kernel_sleef_accuracy`, `me_kernel_fp_mode`).
  - Extended `tests/test_kernel_api.c` to cover `V3_SIZE`/`V4_SIZE` prefixes and the new feature restrictions.
- **Python-Blosc2 Core Menudet & Artifact-Off Build (`python-blosc2.codeberg`)**:
  - Added Python-owned Menudet 0.1 v1 manifest validation and pragma normalization (`src/blosc2/menudet_v1.py`),
    plus C-side `b2_core_kernel_*` wrappers (`src/blosc2/dsl_artifact_bridge.h`), `b2_exec_prepare_core_kernel`
    (`src/blosc2/b2nd_exec_bridge.h`, `b2nd_exec.pxi`), and `is_core` dispatch in `b2nd_kernel_pilot.h` and
    `blosc2_ext.pyx` (`MenudetArtifactHandle`, prefilters, and explicit legacy fallback guards for
    unmigrated masked `evaluate_array` / `evaluate_sources`).
  - Updated `CMakeLists.txt` (`option(MINIEXPR_BUILD_ARTIFACT ... ON)` with core `miniexpr` headers/linkage
    when `OFF`), added `menudet_legacy_adapter_available()`, allowed typed `where` graphs when the legacy
    adapter is absent (`src/blosc2/graph_plan.py`), and added v1 parity + core execution coverage in
    `tests/test_b2nd_exec_m8.py`.

Validation (macOS ARM64):

- **MiniExpr & C-Blosc2 C / ASan suites**: `test_kernel_api|kernel_` passed; installed artifact-off
  consumer tests `6/6` passed; ASan/UBSan `exec-plan` `4/4`, `exec-storage` `2/2`, `exec-stages` `3/3` passed.
- **C header stub & core-only syntax checks**: `dsl_artifact_bridge.h`, `b2nd_exec_bridge.h`,
  `b2nd_kernel_pilot.h`, and `b2nd_exec_stages_bridge.h` compiled cleanly with
  `cc -x c -std=c99 -Wall -Wextra -Wno-unused-function -Werror -fsyntax-only` both without
  `B2_HAVE_B2ND_EXEC` (`#else` stubs) and with `-DB2_HAVE_B2ND_EXEC=1` (without `B2_HAVE_MENUDET_ARTIFACT`).
- **Opt-in Artifact-ON Python wheel (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`, `MINIEXPR_BUILD_ARTIFACT=ON`)**:
  `1808 passed, 44 skipped` across the full Menudet/graph/b2nd-exec/lazyexpr/resident/pilot suite, plus
  `14 passed` on updated `tests/test_b2nd_exec_m8.py` and `tests/test_b2nd_exec_m6.py`.
- **Opt-in Artifact-OFF Python wheel (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`, `MINIEXPR_BUILD_ARTIFACT=OFF`)**:
  - `nm` symbol audit on `blosc2_ext.abi3.so` confirmed **0** `yyjson_*`, `me_artifact_*`, and `me_graph_*` symbols.
  - Isolated subprocess pytest on core Menudet / b2nd-exec / stages / pilot / resident / scalar-dispatch /
    expression-graph suites passed **`832 passed, 14 skipped` (`0` failures)**. (Full cross-suite run
    observed `1637 passed, 134 skipped`; remaining failures/errors are confined to legacy adapter comparison
    tests using `_executor="adapter"`, direct `NativeGraphHandle` tests, `ME_DSL_JIT_COMPILER` env tests
    falling back from `typed_graph_policy`, and masked `evaluate_array`/reader-mask legacy paths.)
- **Standard non-opt-in Python wheel (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=OFF`)**:
  confirmed `_b2nd_exec_available() is False` and `menudet_legacy_adapter_available() is True`;
  smoke suite `156 passed, 13 skipped` (`0` failures).
- **`ruff check` / `ruff format --check`**: clean across all edited/added Python files.

## Earlier qualification: M4 remainder, Release ARM64 sweep, M8 readiness & default cutover

Following M7 completion, this increment completed the M8 readiness gaps in MiniExpr/C-Blosc2,
cut `NativeGraph` over to `_executor="auto"` by default (preferring `"c-blosc2"` with
`NotImplementedError`-only fallback to `"adapter"`), retired the specialized single-case
pilots from normal dispatch when `_b2nd_exec_available()`, and qualified Release ARM64
performance against the adapter baseline.

Key additions:

- **M4 partitionability & static portability (`miniexpr.codeberg`, `c-blosc2.codeberg`)**:
  - Expanded `me_kernel_is_partitionable` (`src/miniexpr_kernel.c`) for lane-local math functions
    (`sin`, `cos`, `sqrt`, `fmod`, `atan2`, `fma`, unary predicates, `minimum`/`maximum`),
    division, comparisons/logical ops, bitwise ops, `where`, promoted NumPy integer math
    (`ME_EXPR_FLAG_NUMPY_1_1`), representable weak `float64 -> float32` constants/captures,
    and bounds-checked weak `int64` captures (`kernel_int_fits`) while keeping unchecked
    strong narrowing and non-lane integer ops (`//`, `%`, `**`, shifts) non-partitionable.
  - Audited `#ifdef _WIN32` / `#if defined(__EMSCRIPTEN__)` launcher paths in `blosc/b2nd-exec.c`
    and guarded standalone pthread fallback with `!defined(__EMSCRIPTEN__)`.
- **C-Blosc2 M8 readiness (`blosc/b2nd-exec.c`, `b2nd-exec-storage.c`, `b2nd-exec-stages.*`)**:
  - Explicit reduction `dtype=` accumulator casts across `ME_REDUCTION_SUM`/`PROD`/`MIN`/`MAX`
    (with `ME_BOOL` truth spans inside `exec_reduce_values`) via `b2nd_exec_plan.requested_dtype`.
  - Direct all-Fortran elementwise map fast path (`strides_fortran_contiguous`, zero gather/mapped scratch).
  - Non-contiguous / F-order explicit `out=` support across `b2nd_exec_execute_readers_into`,
    `b2nd_exec_execute_sources_into`, and `b2nd_exec_staged_execute_sources_into`.
  - Fixed empty (`0`-item / `chunkshape=0`) compressed storage source and sink validation.
- **Python M8 cutover & canonical JSON (`python-blosc2.codeberg`)**:
  - Removed `_compatibility_exporter()` from `TypedGraphHandle`; `to_json()` (including staged
    `menudet-staged-graph-1` splitting) and `map_json()` (`menudet-artifact` `0.1`) now generate
    canonical JSON purely in Python (`src/blosc2/typed_graph.py`), and `TypedGraphHandle`
    threads `integer_powers` optimization, `ME_DSL_JIT`/`BLOSC_ME_JIT` policy, bare `dtype=<name>`
    keywords, and `_prefilter_artifact()` for parallel compressed reductions.
  - Default `NativeGraph(..., _executor="auto")` cutover in `src/blosc2/graph_plan.py`; backward-compatible
    `NativeGraphSchedule.execute_sources` supporting M6 and legacy `ReaderState`/`_NativeSpanReader` signatures;
    `max_worker_scratch` automatic lift for `>32 KiB` per-worker scratch and strided outputs;
    JSON-reimport-free `resident_map_kernel` (`_retain_recipe=False`); retired `_scalar_pilot_values`
    (`menudet_lazy.py`) and `compute_c_blosc2_min_if_eligible` (`native_graph.py`) normal-caller dispatch
    when `_b2nd_exec_available()`.

Validation & Release ARM64 benchmark (macOS ARM64, M-series 12 cores, Release wheel with `B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`):

- **MiniExpr full C suite**: `470/470` passed (`test_kernel_api` expanded for math/ops/weak captures).
- **C-Blosc2 artifact-off installed consumers**: `6/6` passed.
- **C-Blosc2 ASan/UBSan suites**: `exec-plan` `4/4`, `exec-storage` `2/2`, `exec-stages` `3/3` passed.
- **Python opt-in suite (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`)**: `1696 passed, 43 skipped` (`0` failures across `test_b2nd_exec_*.py`, `test_native_graph_*.py`, `test_menudet_*.py`, `test_native_first.py`, `test_lazyexpr*.py`, `test_expression_graph*.py`, including `tests/test_b2nd_exec_m8.py`).
- **Python non-opt-in wheel (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=OFF`)**: `_b2nd_exec_available() == False`; `154 passed, 6 skipped` (`0` failures); `#else` C stubs pass `cc -fsyntax-only`.
- **`ruff check` / `ruff format --check`**: clean across all modified/added Python files.
- **Release ARM64 interleaved benchmark (`c-blosc2` vs `adapter` / `lazyexpr`)**:
  - `res map x*2.5+y n=2000000`: w=1 `2.215 ms` vs `2.305 ms` (`-3.9%`); w=4 `0.807 ms` vs `0.815 ms` (`-0.9%`); w=8 `0.559 ms` vs `0.599 ms` (`-6.6%`).
  - `res math sin+cos n=500000`: w=1 `2.476 ms` vs `2.482 ms` (`-0.2%`); w=4 `0.777 ms` vs `0.773 ms` (`+0.5%`, enabled by M4 math partitionability).
  - `res min(x*2+1) n=2M`: w=1 `2.504 ms` vs `2.568 ms` (`-2.5%`); w=4 `0.744 ms` vs `2.485 ms` (`-70.1%`, `3.34x` speedup).
  - `res sum(x, axis=0) (1000,1000)`: `4.456 ms` vs `6.791 ms` (`-34.4%`).
  - `cmp sum(x) (1001,1003)` (8-slot LRU, `1048576` B): `2.231 ms` vs `135.656 ms` (`-98.4%`, `60.8x` speedup; `bench/compressed_reduction_plan.py`: `162.156 ms -> 2.278 ms`, `8008 -> 64` decodes, `1.00x` amplification).
  - `cmp min(x) (1001,1003)` w=1: `2.286 ms` vs `51.143 ms` (`-95.5%`, `22.4x` speedup; explicit `w=4` without `max_worker_scratch` respects the default `65536` B worker scratch guard for `tile_items=16384`).
  - `cmp sink x*2.5+y vs lazyexpr`: w=1 `10.985 ms` vs `13.149 ms` (`-16.5%`); w=4 `3.178 ms` vs `7.402 ms` (`-57.1%`).
  - `staged` (`x-sum(x,axis=0)`, `sum(sum(x,axis=0)) int8`, `masked sum(sqrt(2x),where=m)`): `-34.9%`, `-35.7%`, `-4.8%` faster.

## Earlier qualification: M7 completion

M7 staged graph planning, composition, diagnostics and opt-in Python integration are
complete and verified across C-Blosc2 (`blosc/b2nd-exec-stages.*`, `b2nd-exec.*`,
`b2nd-exec-storage.c`) and Python-Blosc2 (`src/blosc2/typed_stage_import.py`,
`typed_graph.py`, `typed_exec.py`, `b2nd_exec_stages_bridge.h`, `b2nd_exec.pxi`).
Production defaults, dependency pins and persisted formats remain unchanged.

Key fixes completed for M7 closing:

- **Stage/node diagnostics**: added `b2nd_exec_stage_diagnostic` and `_ex` entry points
  (`b2nd_exec_staged_prepare_ex`, `b2nd_exec_staged_assemble_ex`,
  `b2nd_exec_staged_specialize_ex`) propagated through Cython as `MenudetError.stage`/`.node`.
- **Adapter parity**: changed plan `has_jit` aggregation to `AND` across stages,
  added `uses[where]` and strong-dependency checks rejecting shared scalar-only stage
  materialization, aligned automatic conditional/masked domain rejections (`select`,
  `and`, `or`, `where` when `nstages > 1`) with `dsl_graph.c`, rejected non-`auto`
  `output.dtype` on imported automatic staged `menudet-graph-1` graphs, added unary `pos`,
  and restored pre-M7 single-stage `_TypedExecSchedule.info()` keys.
- **Wide-record partition decode (`typesize > 255`)**: fixed `storage_validate_chunks`
  and `storage_read_box_cached` in `blosc/b2nd-exec-storage.c` for chunk headers with
  `typesize == 1` (e.g. `U100` strings), plus `ninputs == 0` source validation.
- **Trusted stage JSON import**: fixed `typed_stage_import.py` passing only `dtype`/`itemsize`
  into `recipe_dtype` for Menudet stage inputs/outputs without touching the native adapter.

Validation (macOS ARM64):

- **Artifact-off installed static consumers**: `6/6` passed (`installed_miniexpr_consumer`,
  `installed_kernel_grid`, `installed_exec_plan`, `installed_exec_storage`,
  `installed_exec_launcher`, `installed_exec_stages`). Note: `installed_exec_launcher`
  exhibited one earlier transient assertion abort at `exec-launch.c:139` before
  passing 5/5 standalone runs and the final `ctest` run.
- **ASan/UBSan native owner fixtures**: `exec-plan` `4/4`, `exec-storage` `2/2`,
  `exec-stages` `3/3` (including `exec_stages_faults` and `exec_stages_adapter`
  running `STAGE_ADAPTER_TEST` differential checks against `libminiexpr_artifact.a`).
  Owner fixtures compiled cleanly with `-Wall -Wextra -Werror`.
- **Opt-in Python (`B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE=ON`)**:
  - Focused M6/M7/staged selection (`tests/test_b2nd_exec_stages.py`,
    `tests/test_b2nd_exec_m6.py`, `tests/test_native_graph_stages.py`): `132 passed`.
  - Broader graph/menudet/executor/lazyexpr selection (`tests/test_*graph*.py`,
    `tests/test_*menudet*.py`, `tests/test_*b2nd_exec*.py`, `tests/test_lazyexpr*.py`):
    `1732 passed, 59 skipped` (0 failures; JIT-dependent cases skipped in this
    configuration).
- **Standard non-opt-in Python build**: clean wheel build with `_b2nd_exec_available() == False`;
  smoke check across `test_menudet_lazy.py`, `test_menudet_generator.py`,
  `test_menudet_string_prefilter.py` and `test_b2nd_exec_stages.py`: `44 passed, 76 skipped`;
  `cc -fsyntax-only -Wall -Wextra` passes on `b2nd_exec_bridge.h` `#else` stubs.
- **Lint & formatting**: `ruff check` and `ruff format --check` pass cleanly across all
  modified and added Python files. Pre-existing Cython/libtool empty-`threading.c.o`
  archive warnings remain.

## Earlier qualification: review 4 follow-ups and bounded chunk LRU

The same three working trees now address all six findings in
`execution-plan-migration-review4.md`. This remains an opt-in/native increment:
Python M7 integration, production defaults, dependency pins and persisted formats
are unchanged.

| Finding | Resolution |
| --- | --- |
| Sink mutex initialization | Shared checked initialization/cleanup for both storage entry points; injected failure at each frame-input slot proves cleanup, zero evaluation and unchanged compressed output |
| Admission versus execution failure | Separate native `b2nd_exec_admit_partitions` capability check, before output allocation or callbacks; only `admitted=false` may select the old logical loop; execution exceptions still propagate unchanged |
| Empty alias intervals | Nonempty capacities and logical output bytes guard overlap checks; nonempty alias errors match the general executor's unsupported classification |
| Producer allocation before preflight | Common validator explicitly accepts planned, address-less stage-owned allocations; both stages validate before producer allocation; actual allocated peak bytes remain truthful on every failure |
| FP support union | Internal `fp_observed` reporting distinguishes absent observation from unsupported FP; staged support is ANDed only across observing stages, with injected mixed-support tests |
| Stack footprint and shared inputs | Stage binding metadata is heap-allocated by actual child input counts; `x - sum(x)` covers shared callback reads, consumer failure, resident parity and reuse |

Wrapper regressions prove metadata-only fallback for BYTEDELTA, postfiltered
arrays and non-native Unicode, alongside late reader `NotImplementedError`
identity/no-replay and subsequent typed-route recovery. No zero-callback heuristic
or catch around execution is used to infer eligibility.

Resident-output compressed readers now use a per-operand/per-worker chunk LRU,
up to eight slots and no more than physical chunk count. Task admission reserves
the original one-slot-per-worker footprint first; only remaining aggregate
`max_decode_bytes` grows extra slots, round-robin. Worker geometry and strict
logical reduction order do not change. Default allocations may increase within
the same ceiling; `decode_bytes` reports actual owned allocation.

The original ordered-SUM benchmark now reports:

| Decode budget / actual cache bytes | Successful decodes | Physical chunks | Decoded bytes |
| ---: | ---: | ---: | ---: |
| 131072 | 8008 | 64 | 1049624576 |
| 1048576 | 64 | 64 | 8388608 |

This is the same `(1001,1003)` float64, `(128,128)` chunks, `(32,32)` blocks,
tile `1024`, one-worker case, with bitwise values and FP flags checked against
resident execution. It reaches `1.00x` decode amplification without changing
summation leaves. These are local Debug-build observations, not the
ARM64/x86-64 Release performance gate. The previous one-slot checkpoint below
is retained as historical evidence.

Validation:

- ASan/UBSan native owner fixtures: resident/launcher `4/4`, storage `2/2`,
  stages `3/3`, including the optional adapter differential target. The staged
  fixture now instruments both the stage owner and common executor. Installed
  numerical core, codec and comparison-adapter dependencies remain
  uninstrumented.
- Artifact-off installed static consumers: `6/6`.
- Isolated non-TCC opt-in Python selection: `1012 passed, 39 skipped`;
  focused M6/storage selection: `25 passed`. JIT-dependent skips and
  cross-platform qualification remain explicit gaps.
- Standard non-opt-in extension rebuild and isolated ordinary lazy smoke:
  `26 passed`, with typed execution confirmed unavailable.
- Ruff lint/format checks pass. Existing extension/compiler warnings remain;
  native owner fixtures compile with warnings treated as errors.

Native cache tests cover cancellation-sensitive sums, sNaN physical padding,
one/two/four/eight-slot budgets, exact decode traffic and cache-allocation OOM.
Python tests check frame-backed and chunked inputs, mixed/parallel cache
accounting and schedule reuse. Compressed sinks/logical partitions retain their
separate four-block cache; this increment does not silently change those readers.

## Earlier qualification: correctness follow-ups and first M7 slice

This increment uses the working trees based on miniexpr `0e58ed5`, C-Blosc2
`fdbf3a8d`, and Python-Blosc2 `c1acd238`. The C-Blosc2 and Python-Blosc2 trees
include the changes described below; the base revisions alone do not contain
them. No dependency pins, production defaults or persisted formats changed.

Implementation, normal-wrapper reachability and milestone qualification are
different gates:

| Slice | Implemented | Normal-wrapper reachability | Qualification / remaining gate |
| --- | --- | --- | --- |
| M5 frame cleanup | Zero-initialized mutex ownership flags; checked POSIX initialization with cleanup of earlier live mutexes | Explicit typed storage facade; ordinary default unchanged | Serial/mixed resident+frame, allocation failure, mutex-init failure and recovery covered; Windows runtime qualification pending |
| M6 partition source bridge | Rank rejected before copying source dimensions into native fixed-size arrays; callback `NotImplementedError` no longer means fallback after execution starts | Eligible `MenudetLazyArray` calls, conditional on the typed build and earlier route admission | Wrapper-level late-reader failure proves exception identity and zero replay; successful recovery asserts typed route; full M6 exit still requires broader qualification |
| M7 reduction-to-map | Internal `b2nd-exec-stages.h` / `.c`; one reduction feeding an arithmetic map; existing numerical executors reused | Native internal API and installed C consumer only; no Python facade or automatic dispatch | Artifact-off, fault injection and separate adapter differential fixture pass locally; this is not full staged-family migration |

The M7 native slice splits typed regions and remaps producer references without
serializing child graphs. It infers accumulator promotion, preserves captures,
`initial` and `keepdims`, and validates the producer's intermediate bytes and
last consumer before execution. A budget one byte too small rejects during
specialization. Both stages' metadata, outputs and gather budgets are validated
before the first callback. Producer storage is invocation-local and released on
every return; failures identify the original stage root. Final map prefix writes
retain the ordinary map contract. Reports union FP flags, while `raise_mask`
applies after each stage's traversal, matching the adapter: a flagged producer
must not enter its consumer. There is no callback/backend replay.

Local validation (macOS ARM64, non-TCC Python configuration):

- Native ASan/UBSan: resident/launcher `4/4`, storage `2/2`, stages `3/3`
  (two artifact-off stage fixtures plus a separate optional adapter comparison).
  New owner fixtures compile with `-Wall -Wextra -Werror`. Installed core,
  codec and comparison-adapter archives are not sanitizer-instrumented.
- Stage comparison checks bitwise results, dtype/rank, stage count and FP
  flags/support across tile widths and a late signaling NaN. Native tests also
  cover promoted integer sums, exact captures beyond double precision, shared
  producer references, fused producer maps, initial/keepdims, empty intermediate
  results, owner OOM, both stages' reader failures, no replay and recovery.
- Artifact-off installed static consumers: `6/6`, including staged execution
  linked against the actual installed C-Blosc2 archive.
- Isolated opt-in Python regression selection: `1007 passed, 39 skipped`
  across the M6 checkpoint's file selection plus the new regressions.
  The focused M6 wrapper selection is `7 passed`; package/extension locations
  were verified. JIT-dependent cases remain skipped in this configuration; no new TCC or
  cross-platform qualification is claimed.
- Standard non-opt-in extension build succeeds; isolated ordinary
  `test_menudet_lazy.py` smoke coverage is `26 passed`, with the typed executor
  confirmed unavailable. Ruff lint/format checks pass
  for edited Python/tests and the new benchmark. Existing Cython
  unreachable/declarator, integer `abs` width, const-pointer and empty Zstd
  threading-object warnings remain; no new owner warnings were emitted.

### Ordered compressed reduction performance checkpoint (historical one-slot baseline)

`Python-Blosc2/bench/compressed_reduction_plan.py` now provides a reproducible,
interleaved resident/storage `sum(x)` benchmark over the same typed plan.
Preparation/compression are outside timing; values, dtype/shape and FP flags
are checked against resident execution. Output includes requested decode budget,
actual cache allocation, successful decodes, physical chunks and decoded bytes.

For `(1001, 1003)` float64, chunks `(128, 128)`, blocks `(32, 32)`, tile `1024`,
one worker: both `131072`-byte and `1048576`-byte budgets allocate `131072`
cache bytes and perform `8008` decodes over `64` physical chunks (about `125x`
amplification), decoding `1049624576` bytes per invocation. Increasing the
budget alone does not enlarge the current one-chunk resident-output reader
cache. These are traffic observations from a Debug build, not a performance
gate or a claimed cache optimization. A bounded multi-chunk cache experiment
preserving logical reduction order is the next storage performance target.

## Admission and ownership

The detailed resident sections below retain the earlier increments' scope.
M5/M6 extensions and the latest native M7 slice are recorded above and in their
dated-by-increment validation sections; older exclusions are not claims that
those later explicit APIs remain unsupported.

Python-Blosc2's `B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE` build exposes the typed
boundary when built with the matching local C-Blosc2 header. Standard builds
compile unsupported stubs and keep existing dispatch. The checked-in dependency
pins have not been advanced to unpublished local commits.

The new ordinary path admits only explicitly interpreted, native/non-native
aligned/unaligned resident NumPy inputs across scalar, ranked and N-D
right-aligned broadcast shapes (up to rank 16), whole-domain and partial-axis
reductions (`axes`, `keepdims`, inline `initial`, plus opt-in fixed-leaf
`pairwise_sum` via `specialize(..., pairwise_sum=True)`), and whole-domain map
reads or explicit disjoint native NumPy outputs (including F-order, strided,
reversed, and transposed injective destinations) with numeric inputs and exact
typed/weak captures. Maps support add, subtract, multiply and negate. A 64 KiB
mapped-scratch admission bound uses the core-inferred width and applies to
computed maps feeding reductions as well as non-contiguous output scatter maps.
Non-direct inputs (broadcast, strided, reversed, unaligned, byte-swapped, or
non-suffix reductions) use bounded per-job gather scratch (`<= 64 KiB` aggregate
per-job mapped + gather admitted). Direct-output contiguous maps and contiguous
suffix identity reductions need no mapped or gather scratch. Reductions cover
serial `sum`/`prod` and partitionable `min`/`max`/`any`/`all`. Sum/product/truth
accumulator promotion, `initial` scalar conversions, pairwise summation,
non-short-circuiting truth spans, endian-safe scalar stores and empty
identities come from miniexpr core; map tile width stays separate from result
width. Transactional alias staging (`execute_alias`) evaluates into a separate
staging buffer bounded by `staging_budget` and commits to overlapping/strided
destinations only on success. Other syntax, layouts (`where` masks, explicit
reduction `dtype` overrides, `lazyexpr(..., initial=...)` / aliased `lazyexpr`
`out`), and backend policies retain their old route during metadata-only
admission. Once preparation or execution begins, errors propagate; no backend
retry is performed.

The native schedule retains no operand values. Each invocation validates
capacities, pointer overflow, alignment and disjoint output before evaluation.
Minimum publishes only on success; map output may contain earlier committed
tiles and a partially written failing tile after a later failure. Input arrays are not materialized for admission.
Reports distinguish map calls from direct identity traversal and separate
mapped bytes from reducer state; core/compiler allocations are not included
in that tile bound.

Authored formula/Menudet kernels can transfer ownership into a C plan without
source rewriting or a graph descriptor. Runtime slots are queried from the
typed core, not trusted caller duplicates. Context/block-scalar kernels can be
prepared and inspected, but resident specialization explicitly rejects them
until logical-group scheduling is implemented. This does not yet migrate
`MenudetLazyArray` or its Python wrappers.

The Python typed buffer bridge checks actual owning allocations (including
external buffer extents), not a view's possibly fictitious `nbytes`. Unknown
owners are rejected without copying. Forged contiguous `as_strided` input and
output extents are tested to fail before numerical evaluation or output writes.

`NativeGraph.from_expression(..., _executor="c-blosc2")` is an explicit
comparison selector, not a default. Its handle prepares a typed plan directly:
metadata queries, specialization and execution never borrow an artifact handle.
Its compatibility `to_json()`/`map_json()` deliberately instantiate the adapter
and are never reached from prepare/queries/specialize/execute. Restricted-text
literals are parsed inside the shared strict-FP scope with a fresh AST rather
than a cache keyed by expression text, so a caller's rounding mode cannot leak
into a cached literal. Unreachable input bindings are rejected in both executors.
The explicit facade additionally supports interpreted/preferred/required JIT
with typed `cc`/`tcc` and system/SLEEF/Accelerate mathematical policy. The
ordinary automatic route still admits only explicitly interpreted policy.
Admission is narrower than the adapter: no non-default FP accuracy or
environment overrides, no automatic-JIT heuristic, no partial selections, and no
stage diagnostics beyond the single resident stage. Evaluation failures raise
`MenudetError` with `evaluation_error` status and `fp_status`, matching the
adapter. Identity integer reductions still report `fp_supported=False` because
no FP status is observed; the adapter reports an observable environment instead.
They do not yet differ in published values or flags.

### Resident job partitioning (M4 slice)

`b2nd_exec_execute` accepts an optional `b2nd_exec_jobs` request. The executor,
not Python, partitions tiles, bounds scratch and combines partials:

- Ordered reducers (`sum`, `prod`) reject `workers > 1` before any read. Their
  strict serial leaves/combine order are not redefined by worker count.
- `min`/`max`/truth reducers and disjoint map tiles partition only when their
  numerical kernel has the core's sufficient proof. Identity traversal is proven.
  Lane locality alone no longer certifies a graph containing a potentially
  failing weak conversion. Narrow weak captures retain serial admission even
  when their values happen to fit; explicit parallel requests reject before
  reads/writes. Proven straight-line authored kernels can now partition too.
- The job count is capped so `tasks * mapped_bytes` stays within the aggregate
  budget; a budget below one job is rejected. Identity maps/reductions need none.
- Each job opens its own strict FP scope on its own thread and reports its own
  counters and flags; the union is taken in task order, and raise masks still
  apply only after successful traversal.
- Jobs are submitted only after validation and job-workspace allocation.
  Every accepted callback is drained before returning, including after a
  partial/non-prefix failure; no retry occurs.
- Windows keeps the serial path (no thread job support compiled).

Reports expose `tasks` and `worker_scratch_bytes` alongside the per-tile
`temporary_bytes`, keeping "one bounded tile" distinct from aggregate scratch.
`tasks` counts callbacks actually entered, not a presumed submitted prefix.

### Launcher boundary (historical review 3 increment)

The per-call-thread description below is historical, superseded for POSIX by
the persistent shared-pool increment recorded later in this document.

The production build selects `b2nd_exec_launch_host` at build/link time. It
uses the existing synchronous Blosc2 thread callback or portable
`blosc2_run_parallel` dispatcher. This is **not** codec shared-pool reuse:
without a host callback that supplies a pool, threads are still created per
call. No latency improvement or Windows parallel qualification is claimed.
Windows admission remains serial.

Numerical job geometry, scratch, strict per-job FP scopes, combination and
publication remain in `b2nd-exec.c`. All callbacks drain before return. Entered
jobs are counted explicitly, including non-prefix partial execution; a launcher
that returns success without running every job is rejected without replay.
The standalone backend and deterministic compile-time override remain independent
of the codec library. There is no process-global executor launcher setter.

Invocation-local job wrappers and thread-local depth tracking execute nested
executor jobs inline before any nested submission, avoiding recursive waits on
a bounded host callback pool. This does not qualify arbitrary nested codec jobs
or complete the future generic shared-pool backend.

### Actual backend diagnostics

The core's `me_kernel_has_jit` reports immutable prepared availability.
`me_eval_block_backend` reports the route actually entered, including failures;
empty elementwise blocks and rejected preflight report `ME_BACKEND_NONE`.
Interpreter overrides and empty scalar control-flow groups are not reported as
JIT merely because generated code exists. Backend reporting does not change
existing metadata/run-option layouts; compiler/math preparation policy is an
optional append-only suffix preserving the v1/v2 option prefixes.

C-Blosc2 reports `interpreter_calls` / `jit_calls` independently of successful
`map_calls`; identity traversal increments neither backend counter. Python
reports invocation `has_jit` from actual JIT calls, while plan `has_jit` reports
availability. The artifact-off fixture optionally requires system-compiler JIT
with `EXEC_PLAN_TEST_JIT=ON`; the default compiler selection remains unchanged.
This closes a diagnostic prerequisite, not the complete M4 backend gate.

## Validation performed

Review 2 follow-ups (`8324ca4`, C-Blosc2 `16070bca`, Python-Blosc2 `1579d370`)
fix checked weak-integer arithmetic proof, FP-support merging on partial task
submission failure, and omitted versus explicit single-worker scratch budgets.
The explicit facade's omitted single-worker aggregate budget is unbounded
(individual mapped/gather bounds still apply); explicit budgets remain enforced.

The reviewer reports an additional TCC-enabled configuration: 470/470 core tests,
754 Python passes without skips, artifact-off installed consumer 3/3, sanitizer
fixture 2/2, standard stubs and Ruff passing. This supplements the non-TCC
configuration below; it does not establish the optional required-system-JIT
fixture ran under sanitizers.

### First M5 storage increment (local, non-TCC configuration)

- `tests/exec-storage`: 2/2 passed with `-Wall -Wextra -Werror`; the same 2/2
  passed with ASan/UBSan on the new executor/storage owner (installed core and
  codec archives are not sanitizer-instrumented).
- Artifact-off installed static consumer: 4/4 passed, including the new
  `installed_exec_storage` target linked against the actual C-Blosc2 archive.
- Isolated opt-in Python: 759 passed, 39 skipped across the preceding nine
  regression files plus `test_native_graph_sources.py` and
  `test_b2nd_exec_storage.py`. Verified isolated package/extension locations;
  these counts are a different selection/configuration from the reviewer's
  TCC-enabled 754-pass run.
- Native storage tests compare the same plan/schedule with resident execution
  for map and sum/prod/min/max/any/all, multiple tile widths, differing padded
  chunk/block layouts and short tails. Signaling-NaN physical padding must never
  participate. Allocation/decode failure injection verifies no scalar
  publication and completed map-prefix writes, without replay.
- Python compares compressed execution to the explicitly selected adapter,
  blocks Python readers/adapter replay, checks preferred-system-JIT reporting,
  shared-schedule concurrent calls, late signaling-NaN failure/recovery, special
  zero chunks, budgets and unsupported dtype/frame-backed bindings.
- Decoder contexts and one-chunk-per-operand caches are invocation-local,
  use one codec thread and do not mutate shared `current_nchunk` or array caches.
  `decode_bytes` is cache allocation, `decoded_bytes` counts successful decode
  traffic (including revisits), and cache hits count logical item lookups.
  Codec-private allocation is not included. Compressed headers are metadata
  inspected during preflight; no payload is decoded before binding/budget checks.
- Standard bridge stubs pass syntax checking. Existing extension warnings and
  an empty Zstd threading-object libtool warning remain; new owner fixtures
  compile with warnings treated as errors.

### Launcher qualification (review 3)

- Standalone native suite: 5/5 passed, including the optional required-system-JIT
  fixture. ASan/UBSan suite: 4/4 passed without that optional JIT fixture.
- Tests cover reverse order, non-prefix partial submission, incomplete callback
  dispatch, worker failure, allocation failure, no replay, accurate entered-job
  counters and recovery. Host transport nesting is tested on the calling thread
  and a worker thread.
- Artifact-off installed static consumer: 5/5 passed. The added launcher consumer
  uses the actual archive and the real Blosc2 thread-callback API, including a
  nested executor call, incomplete-dispatch rejection and recovery.
- Isolated Python regression selection remains 759 passed, 39 skipped with the
  production launcher wired in; resident shared-schedule concurrency and the
  storage/adapter comparisons remain covered.
- Persistent generic shared-pool reuse, compressed sinks, run-copy optimization
  and Windows/WASM platform qualification are not implied by these results.

### Persistent shared pool & M5 storage increment 2

- POSIX `blosc2_run_parallel` now dispatches through C-Blosc2's persistent
  `blosc_shared_pool` (`struct blosc_job_group` `generic_dojob` + ref-counted
  pool handle) when `threads_callback` is unset, keeping a per-thread nesting
  depth guard (`shared_pool_worker_depth`) to run nested calls synchronously and
  prevent worker-pool deadlock. Tested with worker thread-ID reuse across calls,
  nested calls, concurrent submissions, and `blosc2_destroy()` / re-init.
- M5 storage execution (`b2nd_exec_execute_arrays_ex`) adds contiguous
  inner-block run copies in `storage_read`, native numeric dtypes (`bool`,
  `int8..64`, `uint8..64`, `float32`, `float64`), mixed compressed + resident
  views (`arrays[i] == NULL` -> `views[i]`), and parallel storage tasks over
  the persistent shared pool bounded by `max_decode_bytes` after contract checks.
- Local `-O2` measurement (`1001x1003` float64, `chunks={128,128}`,
  `blocks={32,32}`): serial `SUM` moved from `787.358 ms` to `778.337 ms`
  (row-order traversal still re-decodes 8 chunks per row = 8008 decodes across
  64 chunks with a 1-chunk cache, motivating future chunk-order traversal);
  parallel `MIN` improved from `780.103 ms` (`workers=1`) to `404.268 ms`
  (`workers=4`, `tasks=2`).
- Verification after edits: `exec_plan` ASan/UBSan 4/4 passed; `exec_storage`
  ASan/UBSan 2/2 passed (including `int16`, `float32`, mixed parallel, and
  ordered `sum` worker rejection even when `max_decode_bytes` fits 1 job);
  artifact-off installed static consumers 5/5 passed; `#else` bridge stubs
  `-fsyntax-only -Wall -Wextra -Werror -Wno-unused-function` passed; Ruff format
  and check passed; opt-in TCC-enabled Python wheel (`MINIEXPR_ENABLE_TCC_JIT=ON`)
  passed `802 passed` in 2.88s across the 11 regression/storage pytest files.

### M5 compressed sinks (prefilter increment 3)

- `b2nd_exec_execute_to_array` evaluates elementwise schedules directly into
  in-memory compressed `b2nd_array_t` destinations (`out=NDArray` in Python,
  reporting `route="c-blosc2-typed-storage-sink"`) inside Blosc2's compression
  prefilter (`blosc2_set_prefilter_full_write(compressor, true)`), validating
  output parameters/geometry, non-aliasing, input chunks/views, decode/gather/
  worker-scratch budgets, and `raise_mask` support before mutating the sink.
- Matched chunk/block inputs decode single blocks via `blosc2_getitem_ctx` with
  zero gather scratch (`gather_bytes == 0`) and in-place compaction on padded
  edge blocks (`memmove`); unpadded blocks write directly to the compressor block
  output buffer. Mismatched compressed inputs use logical box reads
  (`storage_read_box`), and resident views use `b2nd_exec_gather_box` (with
  arbitrary strides, byte-swapping, and right-aligned broadcasting).
- Release (`-DCMAKE_BUILD_TYPE=Release`) benchmark (`1001x1003` float64 `x + y` ->
  `b2nd_array_t`):
  - Matched (`chunks={128,128}`, `blocks={32,32}`): `w1 = 6.605 ms` ->
    `w4 = 1.850 ms` (`tasks=4`, `decodes=2048`, ~3.57x speedup).
  - Mismatched (`chunks={127,129}`, `blocks={31,33}`): `w1 = 150.197 ms` ->
    `w4 = 42.939 ms` (`tasks=4`, `decodes=15382`, ~3.50x speedup).
- Verification after edits: `exec_plan` ASan/UBSan 4/4 passed; `exec_storage`
  ASan/UBSan 2/2 passed (matched/mismatched/empty sinks, padded poisoned
  `sNaN`, late `sNaN` `ME_ERR_EVAL`, budget/aliasing/reduction preflight
  rejections); artifact-off installed static consumers 5/5 passed; `#else` bridge
  stubs `-fsyntax-only -Wall -Wextra -Werror -Wno-unused-function` passed; Ruff
  format/check passed; opt-in TCC-enabled Python wheel passed `804 passed` in
  3.01s across the 11 pytest files.

### M5 increment 4: 4-block LRU sink cache, compressed broadcast & partial-axis reductions

- Mismatched and broadcast compressed sink inputs now use a per-worker 4-slot
  LRU single-block cache (`STORAGE_BLOCK_CACHE_SLOTS = 4` via `blosc2_getitem_ctx`),
  sizing cache slots dynamically within `max_decode_bytes` after contract checks.
- Right-aligned N-D broadcasting and partial-axis reductions (`axes`, `keepdims`,
  `initial`, `pairwise_sum`, parallel extrema) are admitted for compressed
  arrays via `(group, pos)` coordinates in `storage_read` (`b2nd_exec_reader`)
  while preserving the linear fast path for contiguous same-shape maps/suffixes.
- Release (`-DCMAKE_BUILD_TYPE=Release`) benchmark (`1001x1003` float64 `x + y` ->
  `b2nd_array_t`):
  - Matched (`chunks={128,128}`, `blocks={32,32}`): `w1 = 6.801 ms` ->
    `w4 = 1.832 ms` (`tasks=4`, `decodes=2048`).
  - Mismatched (`chunks={127,129}`, `blocks={31,33}`) with 4-block LRU cache:
    `w1 = 7.867 ms` -> `w4 = 2.398 ms` (`tasks=4`, `decodes=4242`), ~19.1x /
    ~17.9x faster than the previous 1-chunk cache (`150.197 ms` / `42.939 ms`).
  - Broadcast sink (`(1001,1003) + (1003,)`): `w1 = 5.511 ms` ->
    `w4 = 1.686 ms` (`tasks=4`, `decodes=1152`).
- Verification after edits: `exec_plan` ASan/UBSan 4/4 passed; `exec_storage`
  ASan/UBSan 2/2 passed (compressed N-D broadcasting into resident/compressed
  sinks, partial-axis `sum`/`min` parity with `b2nd_exec_execute_views`, bounded
  block-cache budgets); artifact-off installed static consumers 5/5 passed; `#else`
  bridge stubs `-fsyntax-only -Wall -Wextra -Werror -Wno-unused-function`
  passed; Ruff format/check passed; opt-in TCC-enabled Python wheel passed
  `805 passed` in 2.89s across the 11 pytest files.

### M5 increment 5: Frame-backed operands/sinks, moments & arg-extrema

- `b2nd-exec-storage.c` now fetches chunk data via `storage_fetch_chunk` (per-input `frame_mutex` + `frame_get_chunk` / `sc->data`) for both in-memory (`cframe` `contiguous=True`) and on-disk (`urlpath`) frame-backed arrays without mutating shared `schunk->current_nchunk`, and admits frame-backed compressed sinks (`blosc2_schunk_append_chunk`) alongside chunked/cframe aliasing rejection (`use_dict=True` frames remain rejected).
- `b2nd-exec` (`b2nd-exec.h` / `b2nd-exec.c`), `b2nd_exec_bridge.h`, `b2nd_exec.pxi`, and `typed_exec.py` / `typed_graph.py` implement `argmin`/`argmax` (`ME_INT64`, earliest-tie, NaN-propagating) and `var`/`std` (`ddof`, `ME_FLOAT32`/`ME_FLOAT64` via `me_moments_span`/`me_moments_merge`/`me_moments_finish`), across whole-domain parallel and partial-axis resident and compressed arrays.
- Verification after edits: `exec_plan` ASan/UBSan `4/4` passed; `exec_storage` ASan/UBSan `2/2` passed; artifact-off installed consumers `5/5` passed; `#else` bridge stubs `-fsyntax-only -Wall -Wextra -Werror` passed; Ruff format/check passed; opt-in TCC-enabled Python wheel passed `960 passed` across the 11 pytest files.

### M6: Generic source/sink callbacks and persisted logical partitions

- `b2nd-exec-storage` adds `b2nd_exec_source`, `b2nd_exec_sink`, `b2nd_exec_callback_report`, and `b2nd_exec_execute_sources`: validates all source/sink metadata and workspace bounds (`max_gather_bytes`, `max_output_bytes`, `index_buffer_bytes`) before firing any callback, streams elementwise tiles into bounded sink buffers, emits reductions only after post-traversal validation/`raise_mask`, executes empty domains with zero callbacks, and stops immediately on source/sink failures with zero replay.
- `b2nd-exec` adds `b2nd_exec_prepare_borrowed_kernel` (borrowing caller-owned `me_kernel` without recompilation), `b2nd_exec_specialize_partitions` (`b2nd_exec_partition_spec`), and `b2nd_exec_execute_partitions` across in-memory/frame-backed `b2nd_array_t`, resident `b2nd_exec_view`, and generic `b2nd_exec_source` inputs; preserves full immutable logical partitions (`ME_BLOCK_SCALAR` vs. `ME_ELEMENTWISE`, N-D `me_block` context, short edge groups, rechunk invariance, partial rectangular selections, 0-D scalar domains, input-free generators, and complex/fixed-width string dtypes) without substituting storage blocks.
- Python bridges (`b2nd_exec_bridge.h`, `b2nd_exec.pxi`, `blosc2_ext.pyx`, `menudet_kernel.py`, `menudet_lazy.py`, `storage_readers.py`, `typed_graph.py`, `graph_plan.py`) wire GIL-safe source/sink trampolines, `_cblosc2_handle()` (reusing `core_owner` on `@blosc2.menudet`/`from_recipe` and caching lazily for `from_json`), and `MenudetLazyArray._cblosc2_partition_values` / `MenudetArtifactHandle.evaluate_array` & `evaluate_sources` / `evaluate_partitions`.
- Verification after edits: `exec_plan` ASan/UBSan `4/4` passed; `exec_storage` ASan/UBSan `2/2` passed (including OOM fault injection on callbacks and logical partitions); artifact-off installed consumers `5/5` passed; `#else` bridge stubs `-fsyntax-only -Wall -Wextra -Werror -Wno-unused-function` passed; Ruff format/check passed; opt-in TCC-enabled Python wheel passed `1044 passed` across `test_b2nd_exec_*.py`, `test_native_graph_*.py`, `test_expression_graph*.py`, `test_menudet_arrays.py`, `test_menudet_lazy.py`, `test_menudet_recipe.py`, and `test_menudet_descriptor_host.py`.

### Earlier resident qualification

- Artifact-off native descriptor/execution fixture, including exact int64
  captures beyond double precision, multiple inputs and width variants.
- Planner/specialization/execution allocation failure injection.
- ASan/UBSan builds of the new C-Blosc2 owner (the installed core dependency
  in this fixture was not sanitizer-instrumented).
- A C consumer linked the actual C-Blosc2 static archive with the artifact-off
  miniexpr archive and the existing platform framework dependency.
- Isolated opt-in and standard Python builds, with verified module locations;
  tests cover descriptor copying, plan/schedule lifetime, direct/computed maps,
  exact scalar transport, binding errors, changed values, shared DAGs, unreachable
  arithmetic, FP failure/recovery and no old-executor replay.
- Existing preparation/map/capture/output/source/stage Python regression suites.
- Named typed-graph parity against explicit adapter handles: shape, dtype,
  bitwise values, reduction metadata, single-stage metadata, canonical JSON,
  `MenudetError` status and `fp_status`, plus tests that block adapter
  preparation or artifact borrowing during metadata and execution.
- Native partitioned jobs under ASan/UBSan (`-Wall -Wextra -Werror`, 2/2 `tests/exec-plan` passed): identical results for 1..7 jobs, task cap at `B2ND_EXEC_MAX_TASKS`, capped scratch, rejected ordered sums and unproven direct kernels, multi-task float `min` with `sNaN`/`-0.0`, multi-worker `any`/`all` with late `sNaN`, narrow-integer `sum`/`prod` overflow beyond input width, N-D broadcast + strided/reversed/unaligned/non-native `b2nd_exec_execute_views`, `axes`/`keepdims`/`initial`/`pairwise_sum` (`b2nd_exec_specialize_ex`), disjoint strided/reversed `b2nd_exec_execute_into`, transactional `b2nd_exec_execute_alias` (with staging budget and `raise_mask` preservation), forged underflow/overflow rejection, and OOM fault injection across mapped/gather/scatter/staging/job allocations.
- Installed static consumer (`tests/installed-miniexpr` linked against `blosc2_static` and artifact-off `miniexpr::core`): 3/3 passed (`installed_miniexpr_consumer`, `installed_kernel_grid`, `installed_exec_plan`).
- Standard (non-opt-in) builds keep their existing dispatch; `b2nd_exec_bridge.h` `#else` stubs verified with `cc -fsyntax-only -Wall -Wextra -Werror -Wno-unused-function`.
- Earlier non-TCC artifact-on miniexpr suite: 377/377 passed, including interpreter and required-system-compiler backend diagnostics. The reviewer's TCC-enabled configuration has different test coverage/counts.
- Earlier non-TCC opt-in Python regression suite: `714 passed, 39 skipped` across `test_b2nd_exec_plan.py`, `test_native_graph_preparation.py`, `test_native_graph_maps.py`, `test_native_graph_extrema.py`, `test_native_graph_truth.py`, `test_native_graph_weak_captures.py`, `test_native_graph_output.py`, `test_expression_graph.py`, and `test_expression_graph_lifetime.py`. This includes concurrent calls sharing a schedule, actual backend counters, typed policy/cache separation, explicit single-worker scratch-budget rejection and conservative weak-conversion partition rejection.
- Artifact-off native fixture: 3/3 passed with `EXEC_PLAN_TEST_JIT=ON`, including required-system-compiler execution and failed submissions at each of three task-start positions; every earlier task is joined and reduction output stays unpublished.
- Ruff lint and formatting checks passed for edited Python/test files.

Some Python JIT-dependent tests are skipped in the interpreter-focused build.
System-compiler JIT is available and exercised by the native fixtures; disabling
TCC is not evidence that no JIT compiler exists. Core policy tests check typed
compiler selection, available/unavailable math policies, source pragma
precedence, invalid enum rejection and v1/v2 prefix compatibility. Existing
Python extension warnings include Cython unreachable/declarator warnings,
integer `abs` width warnings, const-pointer qualifier warnings and macOS CBLAS
deprecation warnings; the standalone new native owner builds with
`-Wall -Wextra -Werror`.

`bench/resident_typed_plan.py` compares frontend lowering, uncached preparation,
specialization and warmed execution through explicit handles. An initial
per-node-temporary lowering regression was identified and removed by fusing
single-use expressions while retaining shared DAG temporaries. Local Debug
smoke timings are not the ARM64/x86-64 production performance gate. Raw build
logs and timing results remain outside the repositories.

## Remaining increments

### Development and qualification

- **Qualification scheduling:** full emulated Linux x86-64 qualification was
  intentionally aborted at the user's request; the QEMU VM is stopped and its
  partial logs/build state are retained. Do not treat partial passes as a closed
  x86-64 gate or restart the full matrix before a separate decision to resume it.
  Prioritize the ordinary-caller handoffs in the cleanup audit first.
  The long-running WASM `exec-plan` diagnostic was also intentionally stopped;
  its logs are preserved and the WASM executor gate remains open. The completed
  MiniExpr WASM core runs (`98/98` with JIT, `68/68` artifact-off without JIT)
  do not establish full C-Blosc2/Python WASM migration qualification.
  Full emulated x86-64 and WASM matrices are not practical default workloads on
  this ARM64 development box. Prefer bounded, timeout-protected local probes;
  use native x86-64 hardware and a dedicated WASM test setup for full gates.
  Linux ARM64 is a practical routine local qualification target: the
  hardware-virtualized Lima guest completed core, wheel, installed-consumer,
  sanitizer and compression-only checks, including the affected reruns after
  fixes. Keep that evidence distinct from macOS ARM64 and performance gates.
- **Keep combined-tree regression current:** review 5 requalifies both shared-core
  wheel variants, the ordinary non-opt-in compatibility selection, installed C
  consumers and native owner sanitizer fixtures after the V5 FP-policy changes.
  Further code changes should rerun the affected configurations; older counts
  below remain historical rather than substitutes for these latest results.
- **Cross-platform M4–M8 gates:** x86-64 Release confirmation and live Windows/WASM
  execution remain. Check compiler/backend availability, FP flags and unsupported
  raise masks, alignment, concurrency, and the documented Windows serial path.
  HIGH/MEDIUM and strict/contract/fast policy plumbing is implemented locally;
  remaining work is broader qualification, not another initial implementation.
- **M5 performance matrix:** extend the completed ARM64 Release measurements to
  x86-64 and larger/differing working sets, multiple compressed inputs and worker
  counts. Preserve ordered reduction leaves and explicit scratch/decode budgets;
  investigate repeatable production-path regressions above the plan's threshold.
- **M8 ordinary-call-site handoff and cleanup:** the
  [cleanup audit](execution-migration-cleanup-audit.md) handoffs and coordinated
  pilot + pilot-only native build retirements are now complete locally with
  ownership guards (`tests/test_execution_cleanup.py`) and ARM64 Release handoff
  benchmarks (`bench/execution_handoff.py`). Non-opt-in standard/comparison
  compatibility routes remain intact until M9 switches the production build
  default.

### Capability extensions — not prerequisites unless a supported route needs them

- **M7 storage/parallel stages:** compressed and parallel staged execution remain
  unimplemented. General multi-stage Python integration, nested reductions,
  intermediate lifetimes/budgets, imported bindings and stage/node diagnostics
  are already present. Automatic conditional/masked multi-stage domains and
  reader-backed masks remain explicitly rejected before numerical reads.
- **Additional partition proofs:** extend conservative control-flow/conversion
  proofs only where needed. Straight-line numeric maps and bounded floating
  loop/control-flow programs already have proofs. Non-strict FP kernels remain
  conservatively non-partitionable; TCC remains strict-only. Changing those
  restrictions is separate from forwarding a supported FP preparation policy.
- **Cache tuning:** changing the existing eight-chunk/four-block LRU caps is an
  optional optimization requiring budget and decode-traffic evidence, not missing
  initial cache implementation.

### Deferred release preparation

- Select exact dependency revisions containing the qualified changes, advance
  pins only when those revisions are available, and rebuild without local source
  overrides. Existing pins do not yet describe the current working trees.
- After the migration plan's installation, compatibility and cross-platform
  performance gates pass, separately decide the standard non-experimental
  artifact-off default switch and final release cleanup. No release, publication,
  dependency-pin advancement or production-default change is currently requested.

## Implementation included in this checkpoint

These changes comprise the migration implementation, rather than an additional
list of unimplemented features:

| Repository | What the checkpoint changes do |
| --- | --- |
| MiniExpr | Add core metadata and AUTO preparation support; retain pre-conversion inferred dtype; expose append-only V5 FP/ULP preferences; share the bounded floating partition proof between core and adapter; add native API/prefix/policy tests. The large deletion in `dsl_artifact.c` moves the proof into `dsl_partition_internal.h`, rather than removing that behavior. |
| C-Blosc2 | Add borrowed-kernel resident masks and accumulator conversion tiles with truthful scratch accounting; perform sparse participation reads and preflight mask rejection; forward typed FP/ULP policy through graph preparation; report effective core policy; extend installed-consumer tests. |
| Python-Blosc2 | Decouple Menudet preparation/prefilters from the native artifact library; add Python-owned v1 manifest validation; support optional artifact-off builds and early native API guards; complete typed dispatch/direct-handle/error/ownership parity; carry FP policy through preparation, caches and interchange; add isolated-wheel and v1 differential runners and separate adapter-only tests. |
| Migration documentation | Record the migration plan and reviews, implemented behavior, configuration-specific test results, remaining qualification and deferred release work. Unrelated planning documents remain outside this checkpoint. |
