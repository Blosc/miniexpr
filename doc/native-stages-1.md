# Materialized native stages (G5)

This separately declared, opt-in subset extends native preparation without changing
artifact 1.0/1.1. The lifecycle and ownership rules of `miniexpr_graph.h` still apply.
There are no new dependencies, storage readers, callbacks or numerical fallback.

## Two equivalent frontends

An ordinary `menudet-graph-1` document may declare `requires:["numeric","staged"]`
and contain intermediate reductions or shared computed nodes. Native preparation
partitions these boundaries, infers each region and binds the resulting strong
typed dependencies. Native text and Python syntax adapters declare this capability
when they contain intermediate reductions, but do not choose stages or infer types.
Examples include `x - sum(x, axis=0)`, `sum(sum(x, axis=0))`, and shared array maps.

Automatic partitioning rejects every conditional/short-circuit/masked domain in a
graph requiring staging. It never eagerly hoists work out of an inactive branch.
Scalar-only shared computation also rejects: materializing a weak expression as a
strong stage input could change promotion. These rejections are deliberate scope
boundaries, not requests for a Python fallback. Explicit casts remain supported;
staged region result policy is `auto`, not a hidden final-conversion precision knob.

For explicit numerical boundaries, use `menudet-staged-graph-1` with the schema
`native-staged-graph-1.schema.json`. Root fields are `format`, `semantics`,
`requires`, `inputs`, `stages` and `root`. Stages have consecutive IDs, dependency
references point backwards and `root` is the last stage. Every stage and external
input must reach the result. Each stage's `inputs` object maps its local names to
`{"input":"external-name"}` or `{"stage":producer-id}`.

Graph stages contain an ordinary graph under `graph`. Local input `dtype:"auto"`
means **infer from the named dependency**, never inspect a runtime value. Explicit
types are checked against native inferred dependencies. Nested staged documents
are forbidden. Explicit stages are eager in their declared dependency order;
individual graph regions may have their own lazy selections or root masks, but
their independent producers are not conditional on a consumer's mask. A node may
share a previously materialized stage value with several consumers without
re-evaluating its producer. This differs intentionally from implicit masked sharing.

## Trusted portable regions

A stage with `kind:"portable"` contains a portable artifact object under `artifact`
and requires the exact contract:

```json
{"cardinality":"elementwise","context":"none",
 "effects":"ordered-lazy","mask":"none"}
```

Native artifact loading revalidates the schema/language 1.1 pair, numeric-only
capabilities, input/output signatures, elementwise cardinality and zero ND context.
Control-flow/reduction/context/string capabilities reject, as do block-scalar or
full-DSL recipes. Scalar constants, explicit precision and ordered lazy branches
remain the artifact's existing semantics. `mask:"none"` means that the stage itself
has no consumer participation mask; its lazy select operations remain lazy.
This is trusted, validated numerical composition, **not a sandbox** or permission
to embed arbitrary callbacks, user functions or machine resources.

## Resources, lifetimes and errors

Specialization determines each stage's concrete result shape/dtype/bytes and last
consumer. `me_graph_stage_output_*` and `me_graph_stage_last_consumer` expose them;
Python provides `schedule.stage_info(i)`. The last stage writes caller output.
All earlier results are independently materialized. Their **sum** is the reserved
intermediate bound (`me_graph_intermediate_bytes`), checked against the explicit
`intermediate_budget` (64 MiB default). This first scheduler preallocates every
dependency before numerical execution to preflight every region and fail OOM
without partially executing earlier stages. Results are released after their
last consumer; it does not claim a smaller arena-reuse peak or streamed storage IO.
Iterator scratch is separately the largest region iterator bound. Plan/schedule
metadata excludes compiler/artifact allocations and whole-process RSS.

Every region's bounds, capacity, strides, shapes, byte order, output alignment and
overlap are preflighted before *any* region executes. Signature and shape changes
reject deterministically; changed values reuse the same immutable schedule.
Runtime errors leave numerical output unspecified. FP status is per invocation and
aggregated over completed/failed active regions; raised errors identify the outer
stage. Required JIT fails closed. Reports count actual interpreted/JIT regions and
materializations rather than equating compilation preference with execution.
Reducers retain serial logical C grouping regardless of tile size or scheduling.

## Python example

```python
plan = blosc2.NativeGraph.from_expression(
    "x - sum(x, axis=0)", {"x": "float32"}
)
schedule = plan.specialize({"x": ("float32", (3, 4))}, intermediate_budget=16)
print(schedule.stage_info(0))  # (4,), float32, 16 bytes, last consumer 1
result, report = schedule.execute({"x": values})
imported = blosc2.NativeGraph.from_json(plan.to_json())
```

Native-required LazyExpr execution uses the same staging implementation. Partial
logical-domain reads of staged graphs reject; compressed storage materialization
remains a reported Python adapter. Elementwise artifact export rejects staged
graphs rather than pretending intermediate reductions are block-scalar kernels.

Tests cover inference/round-trip/broadcast consumers, all stage references,
resource rejection, rebinding, concurrent schedule ownership, trusted contracts,
hazardous inactive-domain rejection and FP raise/recovery. General conditional
stages, weak runtime stage inputs, arena reuse, parallel reassociation and streamed
compressed evaluation remain outside this intentionally qualified subset.
