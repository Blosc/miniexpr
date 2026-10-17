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
proof. It currently admits floating maps, same-signature modular integer maps,
and real-math promotion/true division of integral arrays where conversion cannot fail,
excluding checked casts, typed captures, scalar-only computed subtrees, shifts, staged
graphs and reductions. Rejection of parallel admission does not reject numerical
execution: `me_graph_execute_parallel_into()` uses the serial native path.

Pass a versioned `me_graph_executor` with a synchronous `fork_join` callback. The
host runs each supplied task index at most once and must join **all** started tasks
before returning, including on submission failure. A zero return means every task
completed. The callback and invocation pointers must not escape the call. This
interface does not require pthreads, Python, C-Blosc2 or a global native pool.

miniexpr owns destination-aware axis partitioning, per-task binding/schedule metadata,
destination offsets, preflight, worker FP status and deterministic error selection
in logical task order. It specializes and preflights every task before dispatch.
Tasks have independent execution state; output injectivity is checked on the
complete destination before partitioning. Reports aggregate FP/copy/scratch data.

The executor's task budget is capped at 64. `minimum_task_items=0` defaults to
128 Ki elements for cheap maps and 16 Ki elements for admitted costly math maps;
small automatic math jobs use at most four tasks. Callers may override this
performance policy. Executor tiles default to at most 64 Ki elements, bounded by
the task extent, and can be overridden with `tile_items`. Retiling is restricted
to the proven lane-local subset. The automatic executor also retiles eligible
serial fallbacks, rather than paying a small-kernel invocation cost for every 1K
elements. An aggregate
iterator-workspace budget reduces concurrency; a budget that cannot hold even
one planned parallel task is rejected. Compiler/interpreter-private allocations,
task metadata and staged intermediates are not iterator scratch. Serial fallback
for non-partitionable graphs retains the ordinary serial allocation contract.

The initial executor API accepts resident buffers only. Storage-source execution
supports destination descriptors serially; no existing reader callback is assumed
thread-safe. Storage hosts can continue caller-driven block execution on their
own workers, which is also the appropriate mode inside compression callbacks.
Avoid nesting a whole-array executor inside a host's parallel region.

## No-JIT arithmetic interpretation

The portable interpreter recognizes bounded straight-line numerical recipes and
executes their instructions over 256-element vectors. Static C primitives permit
ordinary SIMD vectorization without runtime compilation. The initial subset
includes float32/float64 arithmetic, same-width strong modular integer arithmetic,
total integer-to-float/float conversions, and qualified unary math callbacks.
Each operation preserves its typed intermediate rounding; no fast-math or FMA
contraction is introduced.

Admission is metadata-only. Partial masks, possibly uninitialized locals,
checked casts/arithmetic, unsupported control flow/reductions, excessive plan
depth and ambiguous aliases retain scalar interpretation. An all-participating
mask can use the vector path. Execution of an admitted recipe cannot produce a
checked numerical error, so moving total operations across lanes preserves the
aggregate IEEE status contract. Exact same-width lane-local in-place arrays are
safe; other overlap retains scalar traversal. Numerical failures are never
replayed.

Workspace is invocation-local, at most 64 instruction slots of 256 doubles
(128 KiB), independent of result extent. Leaves can borrow input tiles, and
uniform scalar conversions are performed once per tile. This private interpreter
workspace is additional to iterator scratch, not part of the executor's iterator
workspace budget. Plans never retain operand arrays or share mutable workspaces.

## Compatibility and limitations

Certified resident maps reorder descriptor axes to favor destination locality;
matching F-order/permuted inputs and outputs can then use contiguous inner loops.
Partitioning uses a slow destination axis. Reductions, checked recipes and storage
callbacks retain their original logical traversal. Identity graphs bypass kernels
entirely for matching dense layouts; other identity layouts use bounded copies.
Mixed-layout traversal still needs performance work and no specialized strided-store
JIT kernels are generated. Native complex/string arrays,
general output casting and arbitrary in-place aliases are outside this contract.

Examples of descriptor construction and an out-of-order host executor are covered
by `tests/numpy-compat/arrays.c` and `tests/graph/preparation.c`; Python-Blosc2 also
tests a real host thread-pool adapter and join-on-submission-failure behavior.
