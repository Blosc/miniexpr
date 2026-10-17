# Native destinations and host executors

`me_artifact_eval_array_into()` and `me_graph_execute_into()` accept a versioned
`me_array_output`: dtype, logical shape, allocation base/capacity, origin offset
and signed byte strides. The allocation must remain alive and writable until
execution returns. Bounds describe the complete accessible allocation, not the
logical byte count of a sliced view.

Version 1 supports native-endian aligned numerical destinations with provably
non-overlapping elements. Ordinary F-order arrays, transposes, stepped views and
reversals are supported. Ambiguous overlapping layouts and input/output allocation
aliasing are rejected before execution. Existing contiguous pointer/capacity APIs
retain their behavior. The transactional alias API remains a separate full-result
staging contract.

Non-contiguous maps evaluate bounded contiguous tiles and scatter only successful
tiles. Serial reductions retain their established logical accumulation order and
address the destination only when storing each completed group. Earlier completed
tiles/groups may be visible after an execution error; ordinary execution is not
transactional and never replays a failure on another backend. Iterator scratch
includes output tile storage and is independent of total map extent.

## Host-owned parallelism

`me_graph_execution_capabilities()` exposes a conservative native-IR partitioning
proof. It currently admits floating maps and same-signature modular integer maps,
excluding casts, typed captures, scalar-only computed subtrees, shifts, staged
graphs and reductions. Rejection of parallel admission does not reject numerical
execution: `me_graph_execute_parallel_into()` uses the serial native path.

Pass a versioned `me_graph_executor` with a synchronous `fork_join` callback. The
host runs each supplied task index at most once and must join **all** started tasks
before returning, including on submission failure. A zero return means every task
completed. The callback and invocation pointers must not escape the call. This
interface does not require pthreads, Python, C-Blosc2 or a global native pool.

miniexpr owns first-axis partitioning, per-task binding/schedule metadata,
destination offsets, preflight, worker FP status and deterministic error selection
in logical task order. It specializes and preflights every task before dispatch.
Tasks have independent execution state; output injectivity is checked on the
complete destination before partitioning. Reports aggregate FP/copy/scratch data.

The executor's task budget is capped at 64. `minimum_task_items=0` defaults to
128 Ki elements; callers may override this performance policy. An aggregate
iterator-workspace budget reduces concurrency; a budget that cannot hold even
one planned parallel task is rejected. Compiler/interpreter-private allocations,
task metadata and staged intermediates are not iterator scratch. Serial fallback
for non-partitionable graphs retains the ordinary serial allocation contract.

The initial executor API accepts resident buffers only. Storage-source execution
supports destination descriptors serially; no existing reader callback is assumed
thread-safe. Storage hosts can continue caller-driven block execution on their
own workers, which is also the appropriate mode inside compression callbacks.
Avoid nesting a whole-array executor inside a host's parallel region.

## Compatibility and limitations

Destination scatter preserves logical C traversal; it does not yet choose an
F-friendly axis order or generate strided-store kernels. First-axis task counts
can be limited by short leading dimensions. These are performance limitations,
not reasons to allocate a full contiguous result. Native complex/string arrays,
general output casting and arbitrary in-place aliases are outside this contract.

Examples of descriptor construction and an out-of-order host executor are covered
by `tests/numpy-compat/arrays.c` and `tests/graph/preparation.c`; Python-Blosc2 also
tests a real host thread-pool adapter and join-on-submission-failure behavior.
