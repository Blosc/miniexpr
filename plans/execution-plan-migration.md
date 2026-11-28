# Execution-plan migration to C-Blosc2

## Goal and scope

Move whole-array and graph preparation, specialization and execution ownership
from miniexpr's artifact adapter and Python orchestration into C-Blosc2. Python-
Blosc2 5.x will call those C plans directly. miniexpr remains the typed numerical
kernel/compiler library, not the whole-array scheduler.

This expands the execution phase of
[`blosc2-5-execution-handoff.md`](blosc2-5-execution-handoff.md), under the storage
and compatibility policy in [`blosc2-4.17-to-5.md`](blosc2-4.17-to-5.md).
It does not change `b2o`, invent another persisted graph format, or authorize
replacing legacy formula semantics with a Menudet kernel.

Resident M2–M4 slices have landed in the local opt-in build; see
[implementation status](execution-migration-status.md) for their remaining gaps.
The current storage increment is the first serial M5 binding over that same
schedule, not a claim that all earlier or later milestone gates are closed.
Do not continue the migration by adding a separate C API for each expression
or reducer. The specialized pilots are reference slices to absorb into the
general implementation once parity is proved.

## 1. Starting point before M0

The following is implemented, but is not a completed executor migration:

| Area | Current coverage | Still missing |
| --- | --- | --- |
| Shared core | Opt-in bundled Python build shares one miniexpr core with C-Blosc2; installed static consumers have been exercised. | Standard artifact-off wheel and complete link/dependency audit. |
| Compressed grid pilot | One-dimensional float64 maps and logical-partition block-scalar execution, including padding and FP flag union. | General operands, layouts, partial reads, backend qualification and full array scheduling. |
| Resident minimum | Whole-domain contiguous one-dimensional int64 `min(x)`. | General typed plan representation, remaining dtypes and reduction contracts. |
| Resident map/min | `min(x + integer)`, `min(x - integer)` and `min(x * integer)` with interpreter execution explicitly selected. C-Blosc2 owns the kernel, geometry, mapped tile and reducer loop. | General maps, multiple operands, axes, masks, JIT and parallel execution. |
| Typed graph preparation | miniexpr's private `dsl_graph_prepare_typed()` handles typed nodes and automatic staging without constructing interchange documents. | Independent C-Blosc2 planner/executor; preparation still links the artifact adapter. |
| Legacy importer | Python manifest decoding and differential checks exist. | Complete typed-core semantic validation and importer takeover. |

Current implementation anchors:

- miniexpr: `src/dsl_graph.c`, `src/dsl_graph_text.c`,
  `src/dsl_graph_plan.h`, `src/dsl_graph_internal.h`, `src/dsl_array.c`.
- Core interfaces that remain in miniexpr: `src/miniexpr_kernel.h`,
  `src/miniexpr_grid.h`, `src/miniexpr_reduce.h`.
- C-Blosc2: `blosc/b2nd-kernel.h`, `blosc/b2nd-kernel.c`,
  `tests/installed-miniexpr/`.
- Python-Blosc2: `src/blosc2/native_graph.py`, `graph_plan.py`,
  `menudet_lazy.py`, `blosc2_ext.pyx`, `dsl_graph_bridge.h`,
  `dsl_array_bridge.h`, `b2nd_kernel_pilot.h`.

The resident pilots pass local differential checks. Interleaved ARM64 timings
show lower specialized preparation overhead and approximately equal million-
item mapped execution time. These are not evidence that a general planner will
have the same preparation advantage, nor do they close the performance gate.

## 2. Target ownership and invariants

| Owner | Responsibilities |
| --- | --- |
| miniexpr core | Typed signatures and captures, kernel type checking, compilation/JIT, block evaluation, grid geometry helpers and shared numerical primitives. |
| C-Blosc2 | Graph validation/lowering into numerical regions, stage planning, shape specialization, traversal, source/sink callbacks, scratch, worker partitioning, reduction combination and execution reports. |
| Python-Blosc2 | Python expression adaptation, reference resolution, array/table/remote adapters, JSON interchange, legacy import and wrapper lifetime management. |

### Lifetime and concurrency

- A prepared C plan owns its kernels, copied descriptors, scalar payloads and
  immutable compilation policy. It does not own operand arrays or results.
- A specialized schedule borrows its C plan. The caller keeps the plan alive
  until all schedules are destroyed. Python schedules hold a strong plan owner.
- Existing `me_graph_*` schedules currently retain their native plans. Preserve
  that compatibility contract; do not change it to implement the new API.
- Specialization consumes metadata only. Preparation and specialization must
  not read source callbacks, resolve remote references, evaluate array values
  or create an output destination.
- Execution is synchronous. Views, buffers, callbacks, worker state and FP
  reports are invocation-local. Independent calls may share an immutable plan
  or schedule, but not mutable scratch.
- A host fork/join adapter must join every started task before return, including
  on submission failure. C-Blosc2 owns job boundaries and worker scratch, not
  Python loops over independently reduced chunks.

### Numerical and failure contracts

- Preserve native dtype inference, weak versus typed captures, width checks,
  casting policy and the distinction between map dtype and accumulator dtype.
- Preserve logical reduction order. Storage blocks and worker counts cannot
  redefine sum leaves, persisted logical groups, or combine order.
- Preserve participation masks and conditional map evaluation. A reduction
  `where` mask is not interchangeable with an elementwise selection node.
- Preserve requested and actual backend distinctions. A prepared JIT preference
  is not proof that every tile used JIT.
- Validate requests before execution. Unsupported requests may retain the old
  route only during metadata-only admission; execution failures must never be
  replayed through another backend.
- Specify publication per entry point: reductions may publish a final scalar
  only on success; ordinary map/sink execution may already have committed
  writes before a later callback fails. Transactional alias execution requires
  separately budgeted staging. Do not promise universal rollback.

## 3. Typed boundary to implement

Introduce an internal C-Blosc2 execution descriptor and opaque plan/schedule
handles. Names below describe roles, not a frozen public ABI. Choose final
header/file names in milestone M1 and keep them separate from `b2nd-kernel.h`
if that pilot header would otherwise become the general graph API.

### Graph descriptor

Use a bounded, topologically ordered node array:

- Operation enum, node ID and explicit argument IDs.
- Input slot IDs plus declared dtype/itemsize; input names are diagnostic or
  frontend metadata, not the runtime binding protocol.
- Exact scalar bytes, dtype, width and weak/typed category. Do not normalize
  integer or floating captures through a Python float or C `double`.
- Typed operation parameters: casts and casting policy, integer-power policy,
  reduction kind, axes, accumulator dtype, `keepdims`, optional initial scalar
  and reduction-mask reference.
- Root node ID and explicit stage/binding descriptions where imported staged
  graphs require them. Ordinary graphs may request automatic stage splitting.
- Lengths for every array or payload; bounded rank, depth, nodes, captures and
  stages. Start with limits compatible with the existing graph implementation.

Require argument IDs to refer to earlier nodes, valid roots and slots, valid
operation arity, unique bindings, supported dtypes and checked size arithmetic.
Copy all retained fields. Reject unsupported payloads before core compilation
or JIT; do not accidentally broaden numeric graph support to strings merely
because the kernel API can represent them.

The current `graph_node` is a useful extraction point, **not the finished
boundary**: it still has operation strings, transient workspace fields and
adapter-specific reduction structures. Do not install that private struct or
relocate its yyjson dependencies into C-Blosc2.

### Prepare, specialize and execute

1. **Prepare:** validate the descriptor; determine numerical regions and
   partitionability; lower maps/casts into typed core signatures; prepare owned
   kernels; copy immutable reduction/stage metadata.
2. **Specialize:** bind dtype/rank/shape metadata; resolve broadcast and reduction
   shapes; compute stage lifetimes, traversal and workspace bounds. No input
   pointers or callback invocations are retained.
3. **Execute:** validate current view/source/destination bindings; acquire
   invocation scratch; run C-owned jobs; combine results in logical order;
   publish according to the entry point's failure contract.
4. **Inspect/free:** expose input/output metadata, stage descriptions, capability
   proofs, budget estimates and diagnostics without exporting a wire document.
   Destruction releases only resources owned by that handle.

Keep core numerical typing authoritative. Where current graph preparation
depends on artifact-only helpers, expose the smallest typed core primitive
needed or port host orchestration; do not implement a second numerical type
system in Python or copy arithmetic implementations into C-Blosc2.

### Two preparation paths, one scheduler

The typed DAG prepares numerical regions. A second entry point should accept an
already prepared kernel for authored Menudet programs (including control flow,
ND context and block-scalar outputs). Do not reconstruct a DAG from kernel
source. Both paths must describe input slots and widths, output cardinality,
context rank and logical partition requirements to the same C-Blosc2 traversal,
source/sink and worker implementation. A block-scalar result is one item per
logical group, not necessarily a whole-domain reduction.

For the direct entry point, specify ownership before implementing M5/M6:
either transfer the kernel to the plan on successful preparation (the caller
keeps it on failure), or retain an explicit owner for the entire plan lifetime
when the kernel already belongs to another handle. A bare pointer borrowed for
the duration of preparation is insufficient. Schedules still borrow their plan.

### Miniexpr core prerequisites

These are capabilities needed by the host, not commitments to specific public
struct fields. Prefer small typed interfaces usable with the artifact build off;
feature-check them in cross-repository builds.

| Needed by | Core capability | C-Blosc2 responsibility |
| --- | --- | --- |
| M2 closeout | Shared strict FP scope and status collection, including restoration and unavailable-reporting behavior (`dsl_menudet_fp.h` today) | Apply it to host reducer arithmetic as well as observing kernel flags |
| M3 | Serial sum/product and truth span primitives with existing NaN behavior; authoritative accumulator dtype resolution (`dsl_menudet_reduce.h` and `dsl_array.c` today) | Traverse participating lanes in logical order; keep map and accumulator types separate |
| M4 | Prepared JIT availability and actual invocation backend; typed math/FP policy | Preserve requested policy in preparation/cache keys; report observed execution separately |
| M4 | Kernel facts needed by the existing conservative partitionability proof | Combine those facts with graph dependencies, conversions and failure ordering to decide which jobs may run concurrently |

### Frontends and compatibility exports

- Ordinary Python expressions lower to the descriptor, not JSON preparation.
  Python performs syntax/adaptation checks; native preparation owns numerical
  validation and inference.
- Explicit graph JSON imports are decoded and bounded on the Python side, then
  lower to the same descriptor. Preserve duplicate-key, UTF-8 and resource-limit
  checks rather than relying on plain permissive `json.loads()`.
- Preserve the restricted native text frontend through a wire-free lowering
  module, separate from the scheduler. Port/reuse its parser rather than add a
  new grammar. Test it against Python/declarative lowering before switching
  `NativeGraph.from_expression()`.
  The observed Blosc2 caller of `me_graph_prepare_expression()` is Python's
  `dsl_graph_bridge.h`; no call was found in the C-Blosc2 checkout. A Python-owned
  restricted frontend is therefore an option, not ordinary Python AST parsing.
  Gate the choice on differential grammar, literal (including locale-independent
  decimal floats), diagnostics and resource-limit tests. A C port must replace
  the existing `yyjson_read()` literal dependency. Preserve the independent
  miniexpr compatibility frontend.
- `to_json()`, canonical graph identity and explicit staged interchange remain
  frontend/compatibility concerns. Neither prepare nor execute should need an
  export/reparse round trip.
- Cache keys include descriptor semantics, exact captures, input metadata,
  optimization and compilation/backend policy. Cached objects never retain
  operand values or a previous invocation's result.

## 4. Implementation sequence

Each milestone lands as a testable opt-in increment. Keep the comparison adapter
until the corresponding acceptance gates pass.

### M0 — Freeze the comparison contracts

Before changing general dispatch:

1. Inventory routes in `native_graph.py`, Cython `NativeGraphHandle` and
   `NativeGraphSchedule`, and `MenudetLazyArray` array bridges.
2. Build a capability matrix covering syntax, dtype, layout, reducer, axis,
   backend, sources, output aliasing and error/publication behavior.
3. Map existing native tests to those contracts:
   `tests/graph/preparation.c`, `corpus.c`, `host.c`, `pairwise.c`,
   allocation/fault-injection tests, and existing array/kernel tests.
4. Retain explicit old/new test selectors in one build. Today's
   `_require_native=True` happens to reach the old graph adapter; do not rely on
   that continuing once wrappers change ownership.
5. Extend `bench/resident_min_pilot.py` or a general successor to separate
   frontend lowering, uncached native preparation, specialization and warmed
   execution. Comparing unlike preparation scope must be labelled.

**Exit:** every currently admitted route has a named owner, preservation tests
and an explicit baseline. Unsupported existing combinations remain unsupported.

### M1 — Define the C descriptor and lifetime boundary

Implement the internal header, descriptor validation and opaque plan/schedule
ownership in C-Blosc2. Add Python/Cython conversion helpers and error mapping,
but do not change ordinary dispatch.

Tests must cover invalid IDs/arity, cycles or forward references, duplicate
slots, malformed captures, excessive counts/rank, size overflow, truncated or
invalid option structures, preparation failure cleanup and Python owner
references. Mutating/freeing the caller's temporary descriptor after prepare
must not affect the plan.

**Exit:** a native C fixture prepares a typed input/map/reduction descriptor with
`MINIEXPR_BUILD_ARTIFACT=OFF`, without JSON or artifact headers. No new public
ABI or wire format is frozen.

### M2 — General resident map-to-reduction, serial first

Replace the expression-specific shape of the resident pilots with the M1 plan:

1. Support multiple int64 resident operands and exact captures; fuse a numerical
   map region into a core kernel. Implement both map output and whole-domain
   minimum through one schedule.
2. Reuse the direct-input fast path for proven identity maps. Keep a bounded
   mapped tile for computed maps; do not allocate the whole mapped domain.
3. Establish native output and accumulator typing; add representative
   float32/float64 and integer cases with their existing conversion rules.
4. Connect Python's ordinary expression lowering to typed prepare and
   specialization. Adapt `NativeGraph` metadata queries without requiring
   `map_json()` or a borrowed artifact handle.
5. Execute through C-Blosc2 even when the old graph executor is disabled in
   tests. Compilation must use the typed core, not reconstruct an artifact.

Start acceptance with `min(x + y * scale)` on same-shape contiguous arrays,
then prove dtype/capture variants natively. Preserve backend policy; expand
automatic admission only for combinations the new path actually supports.

Before closing M2:

- Write disjoint, aligned, contiguous pure maps directly to the validated
  destination; only a computed map feeding a reducer needs a mapped tile.
  Specialization must use the inferred map width and the actual traversal to
  decide the budget before execution, replacing Python's worst-case `items * 8`
  tile check. Test large tiles with identity min, direct output and narrow dtypes.
  Document the state of the failing output tile, not just earlier writes.
- Replace the host's independent FP scope with the shared strict core contract.
  Compare altered rounding state, subnormals/flush modes, signaling NaNs, flag
  union, raise masks and restoration on success and failure. First establish the
  adapter's observable failure timing; do not assume immediate per-tile raising.
  Keep generated graph FP policy explicit when environment inheritance is enabled.
- Consolidate ordinary Python expression lowering; query admitted `NativeGraph`
  metadata from the same typed plan used to specialize and execute. Preserve a
  separately selectable adapter for qualification before changing wrapper dispatch.
  Retire duplicate int64 pilots after their callers and report tests use this path.

**Exit:** ordinary Python map and map-to-min expressions use the same general
C plan in the opt-in build. Disabling the old executor proves no fallback.
Existing specialized int64 tests pass through this implementation before their
duplicate loops/APIs are retired.

### M3 — Resident layouts, outputs and complete reduction contracts

Move the general traversal currently embodied in `dsl_array.c` and its graph
callers, separating host iteration from core arithmetic.

- Broadcasting, scalars, arbitrary supported rank, empty dimensions and partial
  logical-domain selection where the old route supports it.
- Strided/reversed/unaligned/non-native inputs: use bounded gathering or reject
  according to existing contracts. Never materialize a whole operand just to
  enter the new executor.
- Destination descriptors, disjoint direct output, supported in-place maps and
  explicit transactional alias handling. Validate buffer extents and overflow,
  not only array shape and contiguity.
- `sum`, `prod`, `min`, `max`, `any`, `all`; map/accumulator conversions, axes,
  `keepdims`, empty identities/errors, `initial` and reduction masks.
- Serial strict sums and opt-in fixed-leaf pairwise sums with the same leaf
  boundaries regardless of iterator tile geometry.
- Actual FP flags and raise-mask behavior, including flags from different tiles
  and masked participation. Publish final reduction results only as specified.

Use shared miniexpr primitives for numerical behavior. Extend their core tests
when extraction is needed; do not duplicate exhaustive numerical matrices in
Python.

Extend node arity when the first ternary operation (`select` or `fma`) lands.
When adding power, preserve weak integer literal specialization and test it
against the adapter. Grow the lowering buffer with checked arithmetic and a
hard source-size limit when new operations outgrow today's bounded emitter.

**Exit:** resident map and reduction capability rows match the adapter, or
explicitly remain on it with a recorded gap. Bound gathering, mapped scratch
and alias staging separately; verify failed alias execution leaves the final
destination unchanged where that is the existing guarantee.

### M4 — Backend qualification and C-owned parallel jobs

Move task partitioning and workspace accounting from the current graph executor
and Python `_parallel_resident_*` paths into C-Blosc2.

1. Carry the existing conservative partitionability proof into the typed plan.
   Cardinality or JIT availability alone is not a safety proof.
2. Retain serial execution for checked conversions, errors or dependencies whose
   observable order would change under partitioning.
3. Qualify interpreter, preferred-JIT and required-JIT preparation; propagate
   failure without a hidden retry. Preserve numerical/math backend policy.
4. Allocate invocation-local worker buffers under an aggregate workspace budget.
   Combine ordered reduction partials using logical IDs, not completion order.
5. Keep host pool integration synchronous; test failed submission, worker errors
   and simultaneous calls sharing a schedule.
6. Expose actual execution metadata through a typed core diagnostic if current
   kernel metadata is insufficient; do not infer per-tile JIT use from a
    preparation preference.

Review-3 launcher sequence:

- Introduce a small synchronous transport boundary first. C-Blosc2's executor
  retains logical jobs, workspace, FP scopes and ordered combination; the
  launcher only runs job callbacks. Drain every accepted job on every return
  path, allow non-prefix execution order, and never replay failed submission.
- Retain an independently linked standalone launcher and deterministic override
  for allocation/submission/failure tests. Select the production launcher at
  build/link time, not through a mutable process-global executor hook.
- Integrate the existing portable `blosc2_run_parallel` callback/dispatch API
  without calling it persistent: without a host callback it currently creates
  and joins threads per call. Actual shared-pool reuse requires a separate
  generic-job extension; its current workers execute codec jobs.
  The existing shared-pool worker path is POSIX-specific; Windows codec
  threading uses different machinery, so it is not an already-qualified
  cross-platform generic pool service.
- Qualify simultaneous calls, nested invocation, callback integration and failure
  draining independently. Keep Windows execution admission conservative until
  numerical/FP tests pass; portable launch primitives alone do not close that gate.

Specify pool ownership and a single concurrency budget across executor jobs
and codec workers. Native resident and direct `b2nd_array_t` storage jobs
should run with the Python GIL released. Budget inner decompression threads
alongside outer jobs (one inner thread is one valid configuration). M5 should
bind native storage directly; M6 Python callbacks must acquire the GIL and be
serialized unless their adapter explicitly supports concurrent calls.

**Exit:** worker-count changes preserve the documented numerical contract and
bounded aggregate scratch. Python no longer owns the admitted resident job loop.

### M5 — Compressed maps and reductions

Build C-Blosc2 storage adapters over the general schedule, absorbing the grid
pilot where its contracts match:

- Start with a serial native `b2nd_array_t` binding over the existing schedule:
  differential `min(x + y)` on padded arrays with different chunk/block layouts
  and a short final span. Traverse logical C order, not physical padding; no
  whole-array operand materialization or independent storage reducer loop.
  Keep compressed output sinks and parallel storage admission separate until
  their own validation, publication and codec concurrency gates pass.
- Preflight decode scratch independently from gather/mapped scratch. Report
  decode allocations and observed decode/cache activity without claiming codec
  internals are included. Rejected budgets must cause no storage reads.
- Read padded/chunked arrays using C-Blosc2 callbacks and bounded decode buffers.
  Distinguish physical padding from logical participation.
- Support multiple compressed and mixed resident/compressed inputs without
  assuming identical physical layouts.
- Move prefilter and compressed-result sink execution, including source/destination
  validation before creating output storage.
  Prefilters are a compressed-sink optimization only: do not reintroduce dummy
  compression contexts for resident maps/reductions, storage reads or stages.
  Keep logical participation, padding, numerical order, FP union and publication
  contracts under the general executor. Zero intermediate chunk copies are an
  optimization target, not a guarantee for every graph/layout.
- Move compressed map/reduction task scheduling and ordered partial combination
  from `_parallel_compressed_*` and `_prefilter_graph_reduction()`.
- Migrate moments and arg-extrema orchestration separately from plain root
  reducers. Their centered scans, combine order, counts and earliest-index tie
  rules must use the existing shared numerical contracts.
- Keep block/chunk reuse and zero-copy/decode optimizations observable through
  reports; do not turn this into resident execution via `operand[:]`.

**Exit:** full and padded grids, differing layouts, short final spans and worker
counts agree with the adapter for values, dtype, callbacks, FP status and scratch.
Compressed maps can write directly through an admitted sink without a full
   uncompressed intermediate.

For the initial storage reader, retain the per-lane geometry path as a
correctness reference. Measure it before replacing the inner loop with maximal
contiguous runs bounded by logical, block and chunk edges. Preserve logical
reduction order, padding exclusion and report-counter meanings; rerun padding,
short-tail and fault tests after the optimization.

### M6 — Generic sources, sinks and logical partitions

Extend the same executor rather than create another remote/streaming scheduler:

1. Define metadata-only source/sink binding and synchronous span callbacks.
   Bridge Python storage/remote/table adapters, preserving their ownership.
2. Validate all bindings and workspace bounds before the first callback.
   Preserve unsupported reader-backed mask/staged combinations until explicitly
   implemented and tested.
3. Bound input gathering and output buffering independently of source size.
   Test callback failures, exact request coverage and output write reports.
4. Move `MenudetLazyArray`/Cython `evaluate_array*` whole-array traversal,
   source/sink and preflight orchestration onto C-Blosc2 plans.
5. Represent persisted logical partitions explicitly. Rechunking must not change
   block-scalar grouping or reduction leaves. Handle multidimensional partitions,
   short groups and admitted partial reads without substituting storage blocks.
6. Preserve block-scalar versus elementwise cardinality, context/rank and
   scalar-tail semantics. A block reduction kernel is not automatically a
   whole-array root reducer.

**Exit:** source-backed tests prove bounded memory with large logical inputs,
and persisted group recipes give the same results after rechunking. No remote or
strided operand is materialized as an admission shortcut.

### M7 — Staged graph planning and execution

Move region splitting, stage preparation/specialization and lifetime analysis
currently in `dsl_graph.c` into the C-Blosc2 typed planner.

- Remap typed node references and producer outputs directly; do not serialize
  child graphs to prepare them.
- Preserve inferred stage dtypes, casts, masks, initial values and explicit
  imported stage bindings.
- Calculate last consumers and intermediate sizes before execution; reject an
  insufficient intermediate budget before input reads.
- Release intermediates after their final consumer. Materialization is permitted
  only for explicitly planned, budgeted stage results, not as a hidden shortcut
  for source operands.
- Define source-backed/staged capabilities explicitly. Keep existing restrictions
  until equivalent traversal and failure behavior are implemented.
- Return stage/node diagnostics and stage-level execution reports without
  consulting an interchange tree.

**Exit:** automatic and explicit staged graphs agree with the adapter, including
nested and staged/masked reductions, allocation failures and budget rejection.
The C-Blosc2 native fixture runs these with the artifact build disabled.

### M8 — Python wrapper and compatibility cutover

Complete wrapper migration once each route has passed its gate:

1. `NativeGraphHandle` owns the C-Blosc2 plan; `NativeGraphSchedule` owns a strong
   reference to its plan wrapper and a C-Blosc2 schedule. Preserve Python methods,
   metadata, exception attributes and lifetime behavior.
2. Replace `MenudetArtifactHandle` uses that need only typed prepare/describe/
   eval with the core kernel path; array scheduling goes to C-Blosc2.
3. Complete Python v1 import differential coverage. Fixed manifest validation is
   Python-owned; source-dependent `requires` validation stays in typed core
   preparation and must run before JIT or reference reads.
4. Preserve explicit graph/artifact interchange and canonical cache identity
   without a mandatory native JSON parser.
5. Retain ordinary formula/full-DSL/legacy reduction compatibility paths where
   syntax or scalar-tail semantics differ. Track their dependencies separately;
   do not silently route them through Menudet or declare them migrated.
6. Retire migrated Python scheduling loops and temporary pilot bridges only
   after their normal callers, persisted fixtures and reports use the new route.
   Keep miniexpr's independent adapter compatibility available; any deprecation
   for other clients is a separate decision.

**Exit:** supported Python execution no longer depends on native artifact or
graph scheduling handles. The comparison implementation remains available in a
qualification configuration, not as an invisible runtime fallback.

### M9 — Artifact-off wheel, default switch and cleanup

- Build from an empty tree with `MINIEXPR_BUILD_ARTIFACT=OFF`; audit downloads
  and build inputs for unintended yyjson or adapter dependencies.
- Prove one core implementation/state provider: inspect static link inputs and
  dynamic dependencies, not just the number of final extension files.
- Install the standard wheel in isolation. Verify module/extension locations and
  subprocess imports so an editable checkout cannot contaminate results.
- Reopen original v1 and 4.17 v2 exchange fixtures, including CFrame and table
  metadata; exercise real maps, grouped arrays and staged graphs.
- Qualify the installed static C consumer and compression-only C-Blosc2 without
  miniexpr as separate configurations.
- Run the correctness/performance matrix below on ARM64 and x86-64.
- Switch defaults only after those gates pass. Remove superseded Blosc2 wiring
  and pilot-specific APIs, while preserving independently supported miniexpr
  compatibility entry points.

**Exit:** the standard Python 5.x execution path is C-Blosc2-owned and artifact-off,
with storage compatibility, installation and performance independently qualified.

## 5. Cross-milestone validation and reporting

### Test placement

Start Windows/MSVC and WASM/Emscripten compile and targeted FP/alignment checks
in M2–M3. In particular, verify unavailable FP reporting and raise-mask rejection
where the core cannot observe flags. These checks supplement the later
ARM64/x86-64 performance and wheel gates.

- **miniexpr:** numerical semantics, typed capture/cast behavior, kernel backend
  parity, FP behavior and shared reduction primitive tests.
- **C-Blosc2:** descriptors, ownership, shapes, scheduling, callbacks, storage
  integration, scratch/intermediate limits, task order and failure cleanup.
- **Python-Blosc2:** lowering/bindings, admission, wrapper lifetimes, output
  identity, original exceptions, persistence and representative end-to-end cases.

As contracts migrate, port or share existing scheduling tests with the new owner
before deleting old coverage. Keep numerical tests in the core rather than
copying their cross-products into Python.

For every new execution family:

- Keep validation configurations distinct: record artifact-on/off, compiler
  availability (TCC versus system compiler), sanitizers, installed consumers and
  optional required-JIT fixtures separately. A larger test count in one
  configuration does not supersede another configuration's evidence.
- Compare old/new in the same build/backend configuration using identical values,
  layouts, codecs, tile sizes, budgets and worker counts.
- Check dtype, shape, values and strict bitwise equality where guaranteed.
- Check FP flags/raise masks, callback counts/coverage, committed writes, actual
  backend and bounded scratch. Do not equate differently defined counters:
  an identity reduction may visit tiles without evaluating a map kernel.
- Inject preparation, specialization, allocation, callback and worker failures.
  Verify no replay, resource leaks or invocation state retained in cached plans.
- Run sanitizer builds of the new native owner. Preserve artifact-off and
  standard/compatibility build coverage as the implementation grows.

### Memory and report accounting

Specify separately:

1. Owned plan/kernel/compiler memory.
2. Specialized schedule metadata.
3. Per-invocation iterator/gather/decode/mapped-tile scratch.
4. Aggregate worker scratch and reducer state.
5. Planned staged intermediates and transactional alias staging.
6. Caller-owned input/output storage.

Report bounds and observed peaks with stable meanings. A bounded mapped tile
does not establish a bound on compiler allocations or total staged memory.
Avoid double-counting shared buffers; distinguish copies from direct input spans.

### Performance gate

Measure uncached preparation, specialization and warmed execution separately;
record frontend lowering separately where practical. Interleave old/new runs,
warm both paths and retain medians outside the repository. Cover:

- Small and large resident maps and direct/computed reductions.
- Compressed padded/chunked maps, direct sinks and reductions.
- Streamed sources and persisted logical groups across rechunking.
- Supported in-place/alias cases, whole-axis, pairwise and staged/masked reductions.
- Representative worker counts and equivalent kernel backends.

Investigate a repeatable regression above 5% on a representative production path
before switching that route's default. Never exchange bounded scratch, logical
grouping, reduction ordering or backend policy for a timing improvement.

## 6. Immediate work queue

1. Preserve the review-2 fixes and record validation by configuration, including
   TCC-enabled versus non-TCC runs and optional required-system-JIT fixtures.
2. Qualify the serial M5 native storage binding against the explicit adapter:
   differing padded layouts, short spans, decode budgets, FP/publication behavior
   and allocation/decode failures, without a second scheduler.
3. Qualify the generic persistent shared-pool backend separately from the
   implemented portable/callback launcher. Integrate compressed sinks through
   prefilters only after their validation, padding and publication gates.
   Frame-backed/mixed inputs and coordinated outer/codec workers remain distinct
   storage increments.
4. Track remaining resident masks, accumulator casts, partial selections, pilot
   retirement and wrapper gaps explicitly. Non-default accuracy, richer kernel
   proofs and Windows/WASM qualification remain M4/platform work; they do not
   prevent testing the restricted serial M5 family.

The migration is complete only when M9 passes. A specialized plan, a successful
native artifact-off fixture, or a fast microbenchmark alone is not completion.
