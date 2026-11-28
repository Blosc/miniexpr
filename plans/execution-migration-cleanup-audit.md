# Execution migration cleanup audit

## Scope and conclusion (2026-11-28)

Audit the remaining Blosc2 scheduling/pilot wiring without removing compatibility
entry points or changing production defaults. Static inspection covers the three
repositories at the migration checkpoint; dynamic probes use the isolated,
artifact-off shared-core wheel built from Python-Blosc2 `c8e62603` and its native
checkpoint sources. The ongoing Linux/QEMU qualification is separate.

**Cleanup is not just deletion.** Explicit typed plans already use C-Blosc2
scheduling, but some ordinary `LazyExpr` JIT routes still orchestrate their work
in Python or use compression contexts without a compressed destination. Those
callers must hand off to the existing typed scheduler before their old machinery
can be removed.

## Observed routing

The probes instrumented `resident_execution.resident_pool().submit`, the
`native_graph._parallel_*` helpers and `_execute_resident_kernel`. They checked
values against NumPy in an isolated `-S` process, with no legacy native adapter.
Resident probes used four workers; the large compressed-map probe used eight
to satisfy its existing admission threshold. These are routing checks, not
performance measurements.

| Entry point | Observed ownership | Cleanup implication |
| --- | --- | --- |
| Ordinary interpreted resident `min(x + 1)` | Typed resident schedule, four C-owned tasks, no Python pool submissions | Already handed off |
| Ordinary JIT resident `min(x + 1)` | `_parallel_resident_reduction`, four Python pool submissions; typed child schedules | Migrate whole-domain scheduling/combination, not merely child evaluation |
| Ordinary JIT resident `x + 1` | `_parallel_resident_graph` → `_execute_resident_kernel`; resident prefilter context, discarded compressed output | Replace the dummy-compression route with direct typed resident execution |
| Explicit typed resident minimum | Four C-owned tasks, no old helper/pool calls | Existing target route |
| Ordinary large compressed `x + 1` | `_parallel_compressed_map` → `_prefilter_compressed_map`, eight Python pool submissions and discarded compressed output | Use native array binding for NumPy output instead of a fake compression sink |
| Ordinary large compressed minimum | `_parallel_compressed_reduction`; bounded Python readers/pool or the prefilter-reduction shortcut, depending on worker/layout gates | Hand off to typed storage execution while preserving logical reduction order |
| Explicit typed compressed map/minimum | `c-blosc2-typed-storage`, no old helper/pool calls | Existing target route; tile/aggregate scratch budgets still apply |

The compressed-map probe initially exceeded the default aggregate worker-scratch
budget with a large tile; a bounded 4096-lane tile was used for the successful
explicit comparator. That explicit comparator requested four workers; eight
workers were used for the ordinary compressed-map probe above.
This audit does not claim that every public default/budget
combination has been qualified. Small compressed requests below the ordinary
graph admission threshold continue to use a compatibility route.

Minimal resident reproduction, run against the isolated artifact-off package:

```python
import numpy as np
import blosc2
import blosc2.resident_execution as resident
from blosc2.graph_plan import NativeGraph

blosc2.set_nthreads(4)
x = np.arange(1_048_576, dtype=np.float64)
original_pool = resident.resident_pool

def forbid_python_jobs():
    raise AssertionError("Python scheduling still owns this route")

resident.resident_pool = forbid_python_jobs
try:
    plan = NativeGraph.from_expression("min(x + 1)", {"x": "float64"}, jit=True)
    assert plan.evaluate({"x": x}, workers=4) == 1
    # Currently raises the ownership assertion; no numerical backend replay.
    blosc2.lazyexpr("min(x + 1)", {"x": x}).compute(jit=True)
finally:
    resident.resident_pool = original_pool
```

## Remaining handoffs before deletion

1. **Ordinary graph dispatch:** in Python-Blosc2 `src/blosc2/native_graph.py`,
   `compute_native_graph` still calls `_parallel_resident_reduction`,
   `_parallel_compressed_reduction`, `_parallel_compressed_map` and
   `_parallel_resident_graph` without first dispatching typed plans directly to
   their C-owned whole-domain schedule. Preserve metadata-only admission and
   do not catch/replay execution failures when replacing these calls.
2. **Resident pseudo-prefilters:** `_execute_resident_kernel` and
   `_ConfiguredResidentSchedule` still lead to Cython's
   `MenudetResidentPrefilter`, which creates a compression context and discards
   compressed bytes. The plan expressly excludes this mechanism for resident
   execution. Retire its caller-local context cache only after configured
   compiler/FP policy and ownership are preserved on the typed route.
3. **Compressed readers/NumPy outputs:** `storage_readers.evaluate_prefilter_numpy`
   and `_dynamic_prefilter_reports` use dummy drivers and Python chunk claiming.
   They are not equivalent to a real compressed-output prefilter optimization.
   Separate these from genuine compressed sinks before removing shared helpers.
4. **Authored Menudet:** `MenudetLazyArray._values` still tries storage/resident
   shortcut helpers before `_cblosc2_partition_values`; `resident_execution`
   retains a Python `run_dynamic` block loop. Audit admission by dtype,
   partition/context and selection, then prefer the general borrowed-kernel
   traversal wherever equivalent. Keep unsupported/compatibility routes explicit.

For each handoff, add regression tests that forbid Python pool submission and
dummy prefilter entry for an admitted typed route, while checking values, FP
status, ordering, output identity, scratch and failure/publication behavior.
Existing tests that assert old route strings or inject Python-pool failures need
porting to the new owner; passing those tests is not evidence that their Python
scheduling loop has been retired. Keep explicit adapter-only differential tests.

### Budget and performance gates for the handoffs

- Size tiles and effective worker counts together using the inferred widths and
  separate mapped/gather/decode requirements. `max_worker_scratch` is an aggregate
  cap, not a per-worker allowance; do not silently multiply an explicit caller
  limit by the requested worker count. Preserve `max_decode_bytes` independently.
- Test budget boundaries and requested versus effective/entered workers for
  multiple inputs, padded/mismatched layouts and larger worker counts. A route
  that succeeds only by unexpectedly collapsing to one worker is not sufficient
  evidence of a performance-preserving handoff.
- Compare old/new Release throughput on real hardware before retiring
  `MenudetResidentPrefilter` / `_prefilter_compressed_map`. Measure preparation,
  specialization and warmed execution separately; investigate repeatable
  production-path regressions above the migration plan's 5% threshold.
  QEMU correctness results do not close this performance gate.

### Relationship to the public compute API

Complete these four ordinary-caller handoffs and coordinated pilot retirement
before C3 in [the public API plan](c-blosc2-public-exec-api.md). C3 should move one
qualified scheduling path from private headers onto installed `b2nd_exec.h` /
`b2nd_exec_kernel.h`, not port obsolete pilots and competing Python schedulers to
the new facade. This sequencing does not authorize public ABI freeze, dependency
pin changes or release/default switching.

## Pilot removal candidates

Normal typed builds already bypass the specialized pilots:

- Python-Blosc2 `native_graph.compute_c_blosc2_min_if_eligible` and
  `MenudetLazyArray._scalar_pilot_values` return before pilot execution when the
  general executor is available. `test_m8_retires_single_case_pilots` checks this.
- The remaining direct callers are pilot tests, `bench/resident_min_pilot.py`,
  and the guarded older-build dispatch branches.
- The coordinated retirement set is the `_ResidentMinI64Plan` /
  `_ResidentMapMinI64Plan` Cython classes, their availability/stub glue and
  `b2nd_kernel_pilot.h`, `NDArray._evaluate_b2nd_kernel_pilot`, plus C-Blosc2's
  internal `b2nd-kernel.h` / `.c` APIs and their CMake source entries.
- Port the corresponding `tests/installed-miniexpr/kernel-grid.c` contracts and
  direct pilot tests/benchmark selectors first. Decide explicitly whether
  pilot-only older native configurations must remain supported; do not remove
  those branches under the assumption that every build enables typed execution.

## Retain, or split by responsibility rather than delete

- `NativeGraphHandle` / `NativeGraphSchedule` are compatibility-facing names;
  artifact-off direct handles already delegate to typed execution.
- `MenudetArtifactHandle` now also owns core kernels. Its name does not establish
  a dependency on the native artifact library.
- `dsl_artifact_bridge.h` contains necessary core preparation/error/metadata
  glue; `dsl_graph_bridge.h` and `dsl_array_bridge.h` still serve ordinary
  non-opt-in and comparison builds. Optional native-adapter branches can be
  isolated, but wholesale removal would break those configurations.
- Genuine compressed-output prefilters, formula/full-DSL/legacy scalar-tail
  compatibility, Python interchange validation, and MiniExpr's independently
  supported adapter are not obsolete merely because typed graph execution exists.

No runtime code was removed by the initial audit. Since that audit, the
ordinary-caller handoffs (with `tests/test_execution_cleanup.py` ownership
guards and `bench/execution_handoff.py` Release verification) and coordinated
retirement of all single-case pilots and pilot-only native build paths
(`b2nd-kernel.{h,c}`, `kernel-grid.c`, `b2nd_kernel_pilot.h`,
`B2_HAVE_B2ND_KERNEL_PILOT`, `_ResidentMinI64Plan`, `_ResidentMapMinI64Plan`,
`_evaluate_b2nd_kernel_pilot`, `_scalar_pilot_values`,
`compute_c_blosc2_min_if_eligible`, `bench/resident_min_pilot.py`,
`tests/test_b2nd_kernel_pilot.py`) have been completed, with the superseded
Python loops retained only for non-opt-in/compatibility configurations.
