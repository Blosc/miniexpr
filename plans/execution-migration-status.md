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
| M4 | C-owned resident jobs: `partitionable` enforced in plan metadata and `b2nd_exec_execute_views` (unproven kernels and ordered `sum`/`prod` stay serial), job count capped at `B2ND_EXEC_MAX_TASKS` (64) and aggregate mapped + gather scratch budget (also enforced for explicit single-worker requests), partials combined in logical task order inside strict FP scope before `raise_mask`, each job owns its FP scope, started jobs always join before return, and Windows clamps `ntasks = 1` before partitioning. Core invocation backend reporting, interpreter/preferred/required-system-compiler fixture, submission-failure injection and simultaneous Python calls sharing a schedule are tested. Typed compiler/math preparation options feed core, C plans and explicit Python facade; policy is part of plan cache keys. `me_kernel_is_partitionable` supplies a conservative core proof for straight-line real arithmetic and proven conversions, including authored kernels | Richer control-flow/math/value-dependent conversion proofs, non-default FP accuracy, storage/compressed jobs and cross-platform backend qualification pending |
| M5 | No compressed family migrated | Existing grid pilot/adapter remain; general compressed maps/reductions pending |
| M6 | No generic source/sink family migrated | Source callbacks, logical partitions and array bridge handoff pending |
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
- Threads are started only after every validation and allocation succeeded; a
  partially started set is joined before returning the error, with no retry.
- Windows keeps the serial path (no thread job support compiled).

Reports expose `tasks` and `worker_scratch_bytes` alongside the per-tile
`temporary_bytes`, keeping "one bounded tile" distinct from aggregate scratch.

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
- Latest configured artifact-on miniexpr suite: 377/377 passed, including interpreter and required-system-compiler backend diagnostics. Other earlier configurations had different test counts.
- Latest isolated opt-in Python regression suite: `714 passed, 39 skipped` across `test_b2nd_exec_plan.py`, `test_native_graph_preparation.py`, `test_native_graph_maps.py`, `test_native_graph_extrema.py`, `test_native_graph_truth.py`, `test_native_graph_weak_captures.py`, `test_native_graph_output.py`, `test_expression_graph.py`, and `test_expression_graph_lifetime.py`. This includes concurrent calls sharing a schedule, actual backend counters, typed policy/cache separation, explicit single-worker scratch-budget rejection and conservative weak-conversion partition rejection.
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
- **M4 remainder:** broader backend/platform qualification, non-default accuracy,
  richer control-flow/math/value-dependent conversion proofs, and storage-backed
  jobs.
- **M8 remainder:** default wrapper dispatch and retirement of the specialized
  int64 pilots once every migrated route has its gate.

Each needs its own differential, failure-injection and cross-platform work; none
is implied by the current checkpoint.
