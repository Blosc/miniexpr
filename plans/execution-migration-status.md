# Execution migration implementation status

This is a checkpoint, **not completion of M0–M8**. The migration plan's
acceptance gates remain authoritative. No production default or persisted
format changed.

| Milestone | Landed work | Remaining gate / gap |
| --- | --- | --- |
| M0 | Route/test/capability inventory in `execution-migration-contracts.md`; explicit direct-handle baseline; separate benchmark preparation/specialization phases | Cross-platform performance qualification remains a later gate |
| M1 | C-Blosc2 internal `blosc/b2nd-exec.h` / `.c`; bounded typed nodes, exact captures, topology/slot/arity/width validation, copied ownership, borrowed metadata-only schedules; optional Cython conversion and strong Python schedule owner | Descriptor will need additional typed options/operations/stages as subsequent families are implemented; it is not installed/frozen |
| M2 | Serial multiple-input fused arithmetic maps and whole-domain minimum; zero-scratch direct maps; exact inferred-width reduction scratch; shared core strict FP scope and end-of-traversal raise masks; ordinary expression lowering reuses `GraphAdapter`; direct authored-kernel ownership transfer feeds the same scheduler; explicit `NativeGraph(..., _executor="c-blosc2")` facade whose metadata, specialization and execution use the typed plan, with adapter-backed compatibility exports and `MenudetError`/`fp_status` parity | Default wrapper dispatch and pilot API retirement are not complete; the facade covers only the typed resident family |
| M3 | Scalar/ranked and N-D right-aligned broadcast domains, empty maps and explicit disjoint NumPy outputs; strided, reversed, transposed, unaligned and non-native input views with bounded per-job gather scratch; allocation-owner extent checks (including underflow/overflow bounds); whole-domain and partial-axis reductions (`axes`, `keepdims`, `initial` with core conversion kernel, opt-in fixed-leaf `pairwise_sum` via `b2nd_exec_specialize_ex`); disjoint strided/transposed/reversed outputs (`b2nd_exec_execute_into`) and separately budgeted transactional alias staging (`b2nd_exec_execute_alias`); serial `sum`/`prod` and parallel `min`/`max`/`any`/`all` with authoritative core accumulator typing (`me_truth_span`, `me_reduce_store_scalar`, host-order `me_reduce_identity`, and accumulator-typed multi-task combine inside FP scope) and empty identities | `where` masks and explicit reduction `dtype` accumulator casts stay on the adapter |
| M4 | C-owned resident jobs: `partitionable` enforced in plan metadata and `b2nd_exec_execute_views` (unproven kernels and ordered `sum`/`prod` stay serial), job count capped at `B2ND_EXEC_MAX_TASKS` (64) and aggregate mapped + gather scratch budget (also enforced for explicit single-worker requests), partials combined in logical task order inside strict FP scope before `raise_mask`, each job owns its FP scope, started jobs always join before return, and Windows clamps `ntasks = 1` before partitioning. POSIX `blosc2_run_parallel` / `b2nd_exec_launch_host` dispatches via C-Blosc2's persistent shared pool (`blosc_shared_pool` + `generic_dojob`) when no user `threads_callback` is active, with nesting guard and clean `blosc2_destroy()` teardown. Core invocation backend reporting, interpreter/preferred/required-system-compiler fixture, submission-failure injection and simultaneous Python calls sharing a schedule are tested. Typed compiler/math preparation options feed core, C plans and explicit Python facade; policy is part of plan cache keys. `me_kernel_is_partitionable` supplies a conservative core proof for straight-line real arithmetic and proven conversions, including authored kernels | Richer control-flow/math/value-dependent conversion proofs, non-default FP accuracy, compressed sinks, and Windows/WASM backend qualification pending |
| M5 | Compressed storage execution (`b2nd_exec_execute_arrays` / `_ex`, `b2nd_exec_execute_to_array`): binds in-memory and frame-backed (`cframe` `contiguous=True` and `urlpath` on-disk, without mutating `current_nchunk`) chunked arrays across native numeric dtypes (`bool`, `int8..64`, `uint8..64`, `float32`, `float64`), right-aligned N-D broadcasting, partial-axis reductions (`axis`, `keepdims`, `initial`, `pairwise_sum`, parallel extrema, `argmin`/`argmax`, `var`/`std` with `ddof`), mixed compressed + resident inputs, and parallel partitionable jobs over the persistent shared pool with `max_decode_bytes` task clamping and contiguous inner-block run copies in `storage_read`; direct elementwise compressed sinks (`out` = `b2nd_array_t` / `NDArray`) evaluate blocks inside Blosc2's compression prefilter with matched-layout zero-copy block reads (`blosc2_getitem_ctx`, `gather_bytes=0`), 4-slot LRU block cache for mismatched/broadcast compressed inputs, resident box gathering, and padded live compaction without mutating output before validation. Explicit Python `NativeGraph(..., _executor="c-blosc2")` lowers `argmin`/`argmax`/`var`/`std` (`ddof`), executes resident/compressed/frame-backed inputs and sinks with the GIL released; no Python source/sink reads or whole-array operand materialization | Cross-platform Windows/WASM performance/FP qualification and ordinary default dispatch cutover (M8/M9) remain |
| M6 | Metadata-only generic source/sink callbacks (`b2nd_exec_execute_sources`), GIL-safe Cython trampolines, bounded index/gather/output buffers, and explicit persisted logical partitions (`b2nd_exec_prepare_borrowed_kernel`, `b2nd_exec_specialize_partitions`, `b2nd_exec_execute_partitions`) with `MenudetLazyArray` / `MenudetArtifactHandle.evaluate_array` & `evaluate_sources` handoff across real, complex, fixed-width string, block-scalar, 2-D context, 0-D scalar, and generator kernels | M7 staged planning and M8/M9 wrapper/default cutover remain |
| M7 | No staged family migrated | Stage planning, intermediate budgets and staged execution pending |
| M8 | No wrapper cutover | Existing `NativeGraphHandle`, `NativeGraphSchedule` and artifact compatibility handles remain |

## Admission and ownership

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

### Launcher boundary (review 3 increment)

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

- **M7 stages:** region splitting, lifetimes, intermediate budgets and staged
  diagnostics without an interchange tree.
- **M4 remainder:** broader backend/platform qualification (including Windows/WASM
  parallel dispatch and FP qualification), non-default accuracy, and richer
  control-flow/math/value-dependent conversion proofs.
- **M8 remainder:** default wrapper dispatch and retirement of the specialized
  int64 pilots once every migrated route has its gate.

Each needs its own differential, failure-injection and cross-platform work; none
is implied by the current checkpoint.
