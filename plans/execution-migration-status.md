# Execution migration implementation status

This is a checkpoint, **not completion of M0–M8**. The migration plan's
acceptance gates remain authoritative. No production default or persisted
format changed.

| Milestone | Landed work | Remaining gate / gap |
| --- | --- | --- |
| M0 | Route/test/capability inventory in `execution-migration-contracts.md`; explicit direct-handle baseline; separate benchmark preparation/specialization phases | Cross-platform performance qualification remains a later gate |
| M1 | C-Blosc2 internal `blosc/b2nd-exec.h` / `.c`; bounded typed nodes, exact captures, topology/slot/arity/width validation, copied ownership, borrowed metadata-only schedules; optional Cython conversion and strong Python schedule owner | Descriptor will need additional typed options/operations/stages as subsequent families are implemented; it is not installed/frozen |
| M2 | Serial multiple-input fused arithmetic maps and whole-domain minimum; zero-scratch direct maps; exact inferred-width reduction scratch; shared core strict FP scope and end-of-traversal raise masks; ordinary expression lowering reuses `GraphAdapter`; direct authored-kernel ownership transfer feeds the same scheduler; explicit `NativeGraph(..., _executor="c-blosc2")` facade whose metadata, specialization and execution use the typed plan, with adapter-backed compatibility exports and `MenudetError`/`fp_status` parity | Default wrapper dispatch and pilot API retirement are not complete; the facade covers only the typed resident family |
| M3 | Scalar/ranked same-shape contiguous domains, empty maps and explicit disjoint NumPy outputs; allocation-owner extent checks; default serial sum/prod/min/max with authoritative core accumulator typing and empty sum/product identities | Broadcasting, strided/non-native gathering, aliases, axes/masks/initial/keepdims, any/all and fixed-leaf sums stay on the adapter |
| M4 | C-owned resident jobs: a request is capped to the aggregate mapped-scratch budget, only order-independent reducers/disjoint map tiles partition, partials combine by logical task ID, each job owns its FP scope, and started jobs always join before return | JIT/backend qualification, partitionability proof and storage/compressed jobs pending; ordered sums/products stay serial |
| M5 | No compressed family migrated | Existing grid pilot/adapter remain; general compressed maps/reductions pending |
| M6 | No generic source/sink family migrated | Source callbacks, logical partitions and array bridge handoff pending |
| M7 | No staged family migrated | Stage planning, intermediate budgets and staged execution pending |
| M8 | No wrapper cutover | Existing `NativeGraphHandle`, `NativeGraphSchedule` and artifact compatibility handles remain |

## Admission and ownership

Python-Blosc2's `B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE` build exposes the typed
boundary when built with the matching local C-Blosc2 header. Standard builds
compile unsupported stubs and keep existing dispatch. The checked-in dependency
pins have not been advanced to unpublished local commits.

The new ordinary path admits only explicitly interpreted, native aligned,
same-shape contiguous scalar/ranked domains (up to rank 16), whole-domain reductions
and whole-domain map reads or explicit disjoint NumPy outputs
with numeric inputs and exact typed/weak captures. Maps support add, subtract,
multiply and negate. A 64 KiB mapped-scratch admission bound uses the core-inferred
width and applies only to computed maps feeding reductions. Direct-output maps
and identity reductions need no mapped scratch. Reductions now include default
whole-domain serial sum/product/minimum/maximum. Sum/product accumulator promotion
and empty identities come from miniexpr core; map tile width stays separate from
result width. Other syntax, layouts, options and backend policies
retain their old route during metadata-only admission. Once preparation or
execution begins, errors propagate; no backend retry is performed.

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
Admission is narrower than the adapter: no JIT/backend/accuracy overrides, no
environment overrides, no partial selections, no parallel `workers`, and no
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
- `min`/`max`, identity reductions and disjoint map tiles partition freely.
- The job count is capped so `tasks * mapped_bytes` stays within the aggregate
  budget; a budget below one job is rejected. Identity maps/reductions need none.
- Each job opens its own strict FP scope on its own thread and reports its own
  counters and flags; the union is taken in task order, and raise masks still
  apply only after successful traversal.
- Threads are started only after every validation and allocation succeeded; a
  partially started set is joined before returning the error, with no retry.
- Windows keeps the serial path (no thread job support compiled).

Reports expose `tasks` and `worker_scratch_bytes` alongside the per-tile
`temporary_bytes`, keeping "one bounded tile" distinct from aggregate scratch.

## Validation performed

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
- Native partitioned jobs under ASan/UBSan: identical results for 1..7 jobs,
  capped scratch, rejected ordered sums, rejected short inputs, and a budget
  below one job. Python checks worker-count invariance against the adapter for
  min/max/maps, ordered-reduction rejection, and `tasks * scratch <= 65536`.
- Standard (non-opt-in) builds keep their existing dispatch; typed tests skip.
- Full miniexpr artifact-on core suite (376 tests) after extracting accumulator
  typing and serial reducers.
- Opt-in Python regression subset (771 passed, 39 JIT-dependent skips) covering
  typed plans, native graph preparation/maps/truth/extrema/captures/output/
  sources/stages and expression-graph lifetime.
- Ruff lint and formatting checks for edited Python/benchmark/test files.

JIT-dependent tests are skipped in these interpreter-focused builds. Existing
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

## Next increments (not started)

- **M5 compressed/storage:** C-Blosc2 storage adapters over the same schedule,
  padded/chunked reads, mixed resident/compressed operands, direct sinks and
  ordered compressed partials. Requires new C storage bindings; the grid pilot
  stays the comparison until then.
- **M6 sources/sinks and logical partitions:** metadata-only callbacks, GIL
  discipline, bounded gather buffers, `MenudetLazyArray`/`evaluate_array*`
  handoff, persisted partition recipes.
- **M7 stages:** region splitting, lifetimes, intermediate budgets and staged
  diagnostics without an interchange tree.
- **M4 remainder:** qualified JIT/math policy (this environment has no JIT
  compiler, so that path is untested), conservative partitionability facts for
  richer maps, and storage-backed jobs.
- **M8 remainder:** default wrapper dispatch and retirement of the specialized
  int64 pilots once every migrated route has its gate.

Each needs its own differential, failure-injection and cross-platform work; none
is implied by the current checkpoint.
