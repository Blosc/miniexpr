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
| Python-Blosc2 compressed grid/logical partition pilot | C-Blosc2 pilot traversal, core grid/kernel evaluation | `test_b2nd_kernel_pilot.py` |
| Python-Blosc2 formula/full-DSL/legacy and explicit artifact imports | Compatibility paths, not graph migration admission | `test_evaluate_menudet_routes.py`, `test_menudet_legacy.py`, `test_menudet_artifact.py`, `test_menudet_interop.py` |

## Capability matrix

“Adapter” below means retain the existing capability checks, not unconditional
support for the cross-product of these rows.

| Contract | Baseline / preservation | New resident pilots |
| --- | --- | --- |
| Syntax | Restricted text and declarative graphs; explicit/automatic stages | `min(x)`, `min(x +/- integer)`, `min(x * integer)` only |
| Dtypes/captures | Native numeric inference, exact typed/weak scalar payloads and width checks | Native int64 input/output, weak signed int64 literal |
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

Until M8, instantiate `NativeGraph.from_expression(..., jit=False)` and call
`specialize(...).execute(...)` to select the **adapter baseline explicitly**.
Instantiate `_ResidentMinI64Plan` / `_ResidentMapMinI64Plan` directly to select
the **pilot explicitly**. Neither selector uses automatic dispatch.
`_require_native=True` is not an enduring old/new selector and must not be used
as evidence of ownership after dispatch changes.

`bench/resident_min_pilot.py` reports uncached adapter prepare and specialization
separately from warmed execution. The pilot still combines compilation and
geometry preparation: that scope is labelled, not presented as like-for-like
preparation. Ordinary compute is a separate dispatch measurement, not the
baseline ownership selector. Raw timings stay outside the repositories.

## Gates

M0 records the baseline; it does not qualify a general new executor. M1–M8
must preserve or explicitly record gaps for every row above. No pilot result
qualifies resident layouts, compressed execution, callbacks, stages or JIT.
