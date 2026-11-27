# Execution migration comparison contracts (M0)

The baseline is the artifact-backed `me_graph_*` executor, not NumPy and not
automatic `LazyExpr` dispatch. New capabilities must be admitted using metadata
before execution; a failure after admission must never replay the baseline.
This inventory does not widen either implementation's admission rules.

## Route owners and preservation coverage

Paths below are relative to the repository named in the first column.

| Repository / entry point | Current owner | Preservation coverage |
| --- | --- | --- |
| Python-Blosc2 `native_graph.prepare_resident`, `compute_native_graph`, `prepare_native_first` | Python admission/cache; `NativeGraphHandle` and miniexpr graph planner | `tests/test_native_graph_preparation.py`, `test_native_graph_maps.py`, `test_native_graph_weak_captures.py` |
| Python-Blosc2 `NativeGraph.from_expression`, `from_json` | Restricted native text / JSON frontends and miniexpr typed planner | `test_native_graph_preparation.py`, `test_expression_graph.py` |
| Python-Blosc2 `NativeGraphHandle.specialize`, `NativeGraphSchedule.execute` | miniexpr graph/array traversal; native schedules retain native plans | `test_expression_graph_lifetime.py`, `test_native_graph_output.py` |
| Python-Blosc2 `_parallel_resident_graph`, `_parallel_resident_reduction` | Python job partitioning and ordered combination over native schedules | `test_native_graph_maps.py`, `test_native_graph_extrema.py`, `test_native_graph_truth.py` |
| Python-Blosc2 `_parallel_compressed_map`, `_parallel_compressed_reduction`, `_prefilter_graph_reduction` | Python storage/task orchestration, miniexpr kernels/reducers | `test_native_graph_sources.py`, `test_native_graph_maps.py`, `test_native_graph_extrema.py` |
| Python-Blosc2 moments and arg-extrema prefilter routes | Python multi-scan orchestration and shared native primitives | `test_native_graph_extrema.py` |
| Python-Blosc2 `evaluate_graph_sources`, `NativeGraphSchedule._execute_sources_bound` | Python binding/adaptation; miniexpr source traversal | `test_native_graph_sources.py` |
| Python-Blosc2 automatic/explicit stages | miniexpr graph stage planning and intermediate ownership | `test_native_graph_stages.py` |
| Python-Blosc2 `MenudetLazyArray`, Cython `evaluate_array*`, prefilter bridges | Python adaptation; miniexpr artifact and array execution | `test_menudet_arrays.py`, `test_menudet_lazy.py`, `test_menudet_recipe.py`, `test_menudet_descriptor_host.py` |
| Python-Blosc2 resident min/map-min pilots | C-Blosc2 specialized resident loops, core arithmetic | `test_native_graph_preparation.py` resident pilot tests |
| Python-Blosc2 opt-in `typed_exec.py` resident maps/min | Python expression adaptation; `b2nd-exec.c` prepares the core kernel and owns specialization/traversal | `test_b2nd_exec_plan.py`, C-Blosc2 `tests/installed-miniexpr/exec-plan.c` and `tests/exec-plan/faults.c` |
| Python-Blosc2 explicit typed `NativeGraph.evaluate_sources` | C-Blosc2 `b2nd-exec-storage.c` binds native compressed arrays to the same schedule; `b2nd-exec.c` owns traversal and reduction | `test_b2nd_exec_storage.py`, C-Blosc2 `tests/exec-storage` and installed `exec-storage.c`; initial serial in-memory float64 slice only |
| Python-Blosc2 compressed grid/logical partition pilot | C-Blosc2 pilot traversal, core grid/kernel evaluation | `test_b2nd_kernel_pilot.py` |
| Python-Blosc2 formula/full-DSL/legacy and explicit artifact imports | Compatibility paths, not graph migration admission | `test_evaluate_menudet_routes.py`, `test_menudet_legacy.py`, `test_menudet_artifact.py`, `test_menudet_interop.py` |

## Capability matrix

“Adapter” below means retain the existing capability checks, not unconditional
support for the cross-product of these rows.

| Contract | Baseline / preservation | Original resident pilots (M0 baseline) |
| --- | --- | --- |
| Syntax | Restricted text and declarative graphs; explicit/automatic stages | `min(x)`, `min(x +/- integer)`, `min(x * integer)` only |
| Dtypes/captures | Native numeric inference, exact typed/weak scalar payloads and width checks | Native int64 input/output, weak signed int64 literal |
| Complex values | Preserve `complex64`/`complex128` only for operations and conversions the adapter admits; no blanket extrema support | Not admitted |
| Fixed-width strings | Preserve `ME_STRING`/`ME_BYTES` widths and authored Menudet array semantics on the direct-kernel path; do not widen numeric graph syntax | Not admitted |
| Layout/rank | Broadcast/scalar/ranked views; supported strides, endian and alignment conversion via bounded gathering | Nonempty aligned contiguous native 1-D |
| Reducers/options | sum/prod/min/max/any/all, admitted axes/keepdims/initial/where; separate moments and arg-extrema orchestration | Whole-domain min; no options |
| Reduction order | Strict serial or explicit fixed-leaf pairwise; logical order independent of tiles/workers | Serial min |
| Backend | Interpreter/preferred/required JIT and captured numerical policy; no replay on failure | Explicit interpreter only |
| Sources | Resident, compressed, mixed and reader-backed where admitted; source-backed masks/stages remain restricted | Resident only (separate grid pilot is not this plan) |
| Output | Shape/dtype/extent validation; admitted direct/in-place/transactional aliases | Scalar publication only on success, no caller output |
| Partial domains | Only adapter-admitted selections; persisted partitions are logical, not physical blocks | Full resident domain only |
| FP/errors | Union of actual evaluation flags, raise masks, original callback exceptions and committed-write reports | Completed map-call flags; no FP probe for direct integer min |
| Lifetime/cache | No operand/result caching; old native schedules retain plans | Immutable plans, invocation-local buffers |
| Memory | Separate gathering, mapped tiles, workers, alias staging and stage budgets | Direct min has no mapped scratch; computed map tile <= 64 KiB |

### M2 closeout contracts (acceptance targets)

These are the initial M2 family acceptance targets, not claims of complete
resident parity. Current coverage and remaining gaps live in the status document:

| Case | What to compare in the same build |
| --- | --- |
| Identity `min(x)` | No map call or mapped allocation, including large tiles; scalar published only on success |
| Computed map, disjoint contiguous destination | Direct kernel output and zero mapped scratch; validate extents/overlap before evaluation. Earlier writes may remain and the failing tile may be partially written. Transactional aliases require a separate contract. |
| Computed map feeding min | Bound mapped scratch by actual tile lanes times inferred map itemsize, excluding reducer/core storage. Test narrow dtypes at and just over budget, without replay or partial scalar publication. |
| Strict FP | Compare flags, raise timing, completed tiles and caller state after success/failure with altered rounding modes, subnormals and signaling NaNs. Match unavailable-reporting semantics on WASM; include Windows/MSVC and WASM FP/alignment checks in M2–M3. |
| Wrapper handoff | Metadata, specialization and execution for an admitted expression come from the same typed plan. Compare against an explicit adapter handle; block the old executor in new-path tests. |
| Future direct-kernel entry | Preserve input widths, context/rank, elementwise versus block-scalar cardinality and persisted group boundaries; test kernel owner lifetime independently of the DAG frontend. |

## Resident gaps and first storage gate (review 2)

Reduction masks describe participation, not just ternary map selection:
masked-out lanes must not evaluate a potentially failing/flag-raising map.
Masks and explicit accumulator `dtype` casts retain the adapter until their
individual gates pass. Partial selections and other recorded gaps also remain;
implementing masks and accumulator casts alone does not establish full M3 parity.

The first M5 gate is a serial storage binding over the existing typed schedule:
`min(x + y)` with independently padded/chunked native `b2nd_array_t` inputs,
different physical layouts, and a short final logical span. Compare to resident
execution of that same plan and, at Python qualification, the explicit adapter.
Physical padding never participates. No new storage-owned numerical reducer or
whole-array operand materialization is permitted.

Preflight storage metadata, all bindings and decode/gather/mapped budgets before
the first compressed read. Decode scratch, gather/mapped scratch and codec-private
allocations are distinct accounting categories. Report decode activity/cache
reuse honestly. Reductions publish only on success; storage failures propagate
without replay. Compressed sinks, mixed inputs, parallel storage and ordered
reducers each need explicit admission/tests rather than inheriting this first
slice's qualification.

Keep test evidence by configuration, including TCC/system-compiler availability,
artifact-on/off, optional required-JIT fixtures, sanitizers and installed static
consumers. A two-test sanitizer run does not establish the optional third
required-JIT test ran.

### Launcher and compressed-sink contracts (review 3)

The launcher transports invocation-local jobs; it does not choose partitions,
reduction leaves, workspace or FP policy. A synchronous return guarantees every
accepted callback has returned, including on failure. Jobs execute at most once,
may complete out of order, and a failed submission is never retried elsewhere.
Count actual entered jobs rather than assuming a contiguous started prefix.
No executor-specific mutable global launcher setter is introduced.

The production portable/callback dispatcher and a standalone/test override must
share this contract. `blosc2_run_parallel` is currently per-call thread dispatch
unless a host callback supplies its own executor, not a persistent-pool API.
Generic shared-pool reuse, nested-call deadlock prevention and Windows FP parity
remain separate qualification gates. Budget outer jobs and inner codec workers
together; one inner codec thread remains the initial storage policy.

Use prefilters only when optimizing an admitted compressed sink, never as fake
compression passes for resident/read scheduling. Preserve logical masks,
padding, FP reports and sink publication semantics. Run-copy reader optimization
must preserve logical order and counter definitions, with a measured scalar
baseline and the existing padding/failure suite retained.

## Native preservation suites

In miniexpr, `tests/graph/preparation.c` freezes validation, typing, captures and
metadata; `corpus.c` freezes representative expression/reduction semantics;
`host.c` freezes bindings, callbacks, budgets and reports; `pairwise.c` freezes
logical sum leaves. `allocation.c`, `compiler-allocation.c` and
`grid-allocation.c` exercise cleanup/fault injection. Existing kernel, array,
grid and reduction tests remain authoritative for numerical behavior.

In C-Blosc2, `tests/installed-miniexpr/kernel-grid.c` is the artifact-off pilot
comparison fixture. New descriptor/schedule tests belong with that native owner,
not as a duplicate Python numerical matrix.

## Explicit comparison selection and timing

`NativeGraph.from_expression(..., _executor="adapter")` explicitly selects the
baseline; `... _executor="c-blosc2"` selects the opt-in typed facade. The default
remains the adapter. `evaluate_sources` respects that selector too: the typed
storage slice rejects unsupported bindings rather than replaying the adapter.
`_TypedExecPlan` remains a direct low-level comparator.
Instantiate `_ResidentMinI64Plan` / `_ResidentMapMinI64Plan` directly to select
the **pilot explicitly**. Neither selector uses automatic dispatch.
`_require_native=True` is not an enduring old/new selector and must not be used
as evidence of ownership after dispatch changes.

`bench/resident_min_pilot.py` reports uncached adapter prepare and specialization
separately from warmed execution. The pilot still combines compilation and
geometry preparation: that scope is labelled, not presented as like-for-like
preparation. Ordinary compute is a separate dispatch measurement, not the
baseline ownership selector. Raw timings stay outside the repositories.
`bench/resident_typed_plan.py` additionally separates typed frontend lowering,
uncached preparation, specialization and warmed execution for the M2 family.

## Gates

M0 records the baseline; it does not qualify a general new executor. M1–M8
must preserve or explicitly record gaps for every row above. No pilot result
qualifies resident layouts, compressed execution, callbacks, stages or JIT.
