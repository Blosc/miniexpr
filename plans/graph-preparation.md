# Native graph preparation for Menudet

Date: 2026-10-09.
Status: **implementation proposal; no new API, format, dependency or default is approved**.

## 1. Objective

Move portable numerical graph preparation into miniexpr so a C, WASM or other
non-Python host can validate, infer, plan and evaluate the same supported graphs.
Python remains an optional authoring/storage adapter, not a second semantic planner.

The intended deployment sequence is:

```text
expression or structured graph + operand metadata + scalar captures
    -> native validation and typed graph
    -> immutable numerical plan
    -> native shape specialization and execution schedule
    -> runtime buffer binding and native execution
```

Preparation must not require Python, NumPy, NumExpr, numerical input reads, or
execution of user code. Scalar captures are explicit preparation inputs, not array
values discovered by inspecting buffers. Their declared strength is semantic data.

This project does **not** initially move compressed storage reads, network access,
table partition selection or output persistence into miniexpr. A native host can
already supply in-memory buffers; C-Blosc2 storage orchestration is a later,
separately designed layer.

## 2. Starting architecture and reuse

### Native components

- `src/miniexpr_artifact.h`, `src/dsl_artifact.c`: versioned portable artifact
  loading, scalar captures, typed signatures, inferred dtype, status and ownership.
- `src/dsl_portable.c`, `src/dsl_portable_expr.c`, `src/dsl_portable_types.c`:
  authoritative 1.1 validation, promotion, casts, functions and diagnostics.
- `src/dsl_array.c`: checked logical-array views, broadcasting traversal, bounded
  gathers, shape operations and six logical reductions.
- `src/dsl_compile.c`, `src/dsl_portable_jit.c` and runtime backends: existing typed
  kernel compilation and actual-JIT reporting. Full-DSL semantics are not portable
  graph semantics and must not become an implicit alternate execution path.
- `tests/numpy-compat/`: native-owned arithmetic/function corpora, standalone C
  examples and layout/reduction tests. Reuse their semantic policies and case IDs.

### Current Python preparation to replace

In Python-Blosc2, `src/blosc2/native_graph.py` currently:

1. Parses the validated expression using Python AST tooling.
2. Recognizes a root reduction and separates its options.
3. Normalizes selected NumPy spellings and rejects unsupported attributes.
4. Classifies plain captures, typed scalars and array operands.
5. Generates a DSL function and exports it twice: first to discover inferred dtype,
   then to set the output signature.
6. Caches portable plans and arranges shapes, selections, storage reads and output.

Native execution already owns numerical evaluation. The first migration removes
Python's semantic recognition/source-generation/type-planning work, not the Python
object-to-descriptor adapter or compressed-storage frontend.

Read `doc/numpy-arithmetic-1.1.md`, `doc/numpy-functions-1.1.md` and
`doc/native-arrays-1.1.md` alongside the implementation. Some coverage statements are
historical; inspect the exact implementation/test revision rather than inferring
current support from an old signoff. Existing CI qualification remains independent.

## 3. First supported contract

### Include in the first end-to-end slice

- Portable 1.1 bool, signed/unsigned 8/16/32/64-bit integers and float32/64.
- Named typed array inputs, strong typed scalars and explicit weak scalar captures.
- Typed elementwise arithmetic, casts, comparisons, predicates, supported real
  functions, lazy selection and Boolean short-circuiting.
- Native broadcasting of runtime input shapes, including rank zero and empty axes.
- One optional root logical reduction: sum/prod/min/max/any/all, using the existing
  native accumulator, axes, keepdims, initial and participation contracts.
- One graph result; caller-owned aligned C-order output disjoint from inputs.
- Metadata/result queries before allocating or reading numerical buffers.
- Native interpretation everywhere qualified; optional eligible host JIT with
  explicit actual-route reporting, never a Python fallback.

Root reduction options belong to the graph/schedule, not an elementwise kernel
pretending to perform a logical reduction through a block-scalar return.

### Explicitly defer

- General Python syntax, attributes, object introspection, arbitrary callbacks,
  NumPy object/structured/string/datetime semantics and unsupported numeric types.
- Intermediate reductions and multi-stage DAG execution until the first slice is
  stable; list them as a separate milestone, not hidden first-slice requirements.
- Arbitrary mutable views, in-place outputs, advanced indexing/gathers/scatters.
- Table filtering/ordering/row partitions, proxies, remote operands and storage IO.
- New reduction families, generalized ufuncs, linear algebra and parallel reduction
  reassociation. No promise of universal NumPy bit parity or speed superiority.
- Embedding arbitrary full-DSL or user-defined artifact nodes. Trusted validated
  portable elementwise artifact nodes may be considered in a later milestone.

## 4. Semantic ownership and invariants

1. **One numerical authority.** Graph inference and kernel lowering reuse native
   portable rules. Do not introduce a copied promotion/function table or translate
   unsupported nodes into full-DSL evaluation to make them succeed.
2. **Output does not drive intermediates.** Infer numerical result dtype first;
   validate any requested final cast separately. Avoid the current export-twice
   frontend workaround.
3. **Metadata-only preparation.** No numerical buffer pointers are needed to infer
   types or shapes. Preparation never samples arrays, runs a kernel, or raises FP
   flags from constant evaluation. Explicit initial/capture values are copied data.
4. **Stable scalar categories.** Weak constants, typed scalars and strong 0-D inputs
   remain distinct through serialization, cache keys and execution.
5. **Rounding and participation are observable.** Preserve per-node precision,
   lazy masks, branch participation and specified reduction grouping. No fast-math,
   reassociation, FMA contraction or speculative evaluation of inactive operations.
6. **No result cache.** Immutable plans may own captures, never operand arrays,
   output buffers, numerical results or mutable cumulative FP status.
7. **Preflight before writes.** Format, capability, shape, signature, capacity and
   overlap failures reject before numerical execution. Errors arising from values
   during execution leave output unspecified, as in the existing array contract.
8. **Capabilities are explicit.** A valid graph need not JIT-compile. Required
   acceleration rejects when unavailable; ordinary preference may use native
   interpretation with an accurate report.
9. **No sandbox claim.** Bounded validation and no callback execution are not a
   general security proof or protection from all resource exhaustion.

## 5. Native graph representation

Use an immutable, topologically ordered internal DAG with stable node IDs and
source locations when available. Proposed node families:

| Node | Payload / constraints |
| --- | --- |
| Input | Binding name, explicit dtype and strong input category. |
| Constant | Weak or typed category, declared dtype/kind, lossless value bytes. |
| Unary/binary operation | Whitelisted portable opcode and ordered operand IDs. |
| Cast | Destination dtype and existing explicit-conversion rules. |
| Function | Canonical native function ID, arity and ordered operands. |
| Select / short-circuit | Condition and lazy branches with participation semantics. |
| Root reduction | Map value, axes/keepdims, accumulator policy, optional initial and Boolean participation mask. |
| Result conversion | Optional final dtype/casting policy, distinct from computation. |

Annotations include inferred dtype, scalar strength, shape relation, required
capabilities and whether evaluation can produce diagnostic/status effects. Treat
unknown effects conservatively. A shared value-producing node has a defined
once-per-active-logical-coordinate meaning; do not silently duplicate or globally
hoist it across different participation domains. If this cannot be preserved in
the first scheduler, reject that graph pattern explicitly.

Structured graph semantics must specify ordered operand evaluation and active
branch evaluation. A DAG edge is a value dependency, not permission to evaluate all
reachable nodes eagerly. Root reduction participation gates its map expression;
elementwise `where` is a separate lazy value-select operation.

### Two input frontends, one internal graph

1. **Versioned structured graph** is the language-independent interchange boundary.
   Start with JSON using the already available yyjson dependency. Do not require a
   Python parser, generated C or a new serialization dependency.
2. **Native expression text** is a convenience frontend into the identical DAG.
   Define a small portable grammar; do not claim it accepts Python source generally.
   Reuse native tokenizer/parser infrastructure where semantics fit, or add a small
   parser without widening the existing DSL grammar unintentionally.

Python initially sends structured graph nodes from its existing validated graph,
preserving scalar categories and argument order. That is syntactic adaptation,
not promotion/reduction/fusion planning. A later textual frontend can reduce this
adapter further. Normalizing Python method syntax to a graph opcode stays frontend
work; interpreting the opcode's type, axes and execution semantics is native work.

## 6. Public API lifecycle (proposed, not finalized ABI)

Introduce a dedicated `src/miniexpr_graph.h`, with C/C++ guards and versioned option
structs consistent with existing conventions. Tentative opaque handles:

```c
typedef struct me_graph_plan me_graph_plan;
typedef struct me_graph_schedule me_graph_schedule;
```

| Proposed operation | Responsibility |
| --- | --- |
| `me_graph_prepare_json` | Copy/validate graph, infer types, create immutable numerical plan from graph and input signatures. |
| `me_graph_prepare_expression` | Parse the restricted expression frontend, then use the same preparation pipeline. |
| `me_graph_specialize` | Infer concrete broadcast/output shapes and stage geometry from pointer-free input metadata. |
| Plan/schedule queries | Required bindings/capabilities, inferred result dtype/shape, stages and scratch bounds. |
| `me_graph_execute` | Validate actual `me_array_view` bindings against the specialization and execute using caller-owned output. |
| Export/import | Round-trip canonical declarative graph and semantic identity; never dump an in-memory compiled plan. |
| Free functions | Release owned resources; NULL-safe; document borrowed query lifetimes. |

The exact function signatures and names are an early design deliverable. Prefer
existing error/status conventions, extended with structured node/stage locations
where needed; version new structs instead of changing existing ABI layouts.

### Separation of plan, specialization and invocation

- **Plan:** owns normalized graph, typed signatures, immutable captures, semantic
  revision and portable kernel resources. Input shapes need not be fixed here.
- **Schedule:** owns concrete shapes, normalized axes, stage geometry, validated
  broadcast relationships and shape-dependent scratch estimates. It contains no
  caller storage pointers or mutable execution cursors.
- **Invocation:** validates bases, capacities, strides, byte order, masks and output
  overlap; owns scratch, cursors and per-call diagnostics. Layout-dependent direct
  versus gathered traversal may be selected here unless explicitly included in
  specialization metadata and subsequently revalidated.

A schedule must retain its plan or clearly require it to outlive the schedule;
prefer retention with explicit immutable ownership. Independent invocations can
share plan/schedule with disjoint caller buffers. Freeing a handle concurrently
with execution is unsupported unless a later API explicitly establishes lifetime
synchronization.

Changing shapes requires a new specialization, not stale cache reuse. Changing
values alone requires no preparation. Changing dtype or capture strength/value
requires a different plan. Zero-dimensional arrays remain runtime inputs, not
value-specialized captures.

## 7. Validation and resource limits

Before lowering or compiling, validate:

- Schema and numerical semantic identifiers, required capabilities and supported
  profile; reject unknown opcodes, arities, keywords and scalar encodings.
- Unique node/binding IDs, known references, topological ordering/cycle absence,
  valid root and no duplicate names. Decide whether unreachable nodes reject;
  initially reject to avoid ambiguous validation/diagnostic semantics.
- Types/categories, weak transport limits, cast policies and supported function
  signatures through the authoritative native rule implementation.
- Rank/shape extents, axis normalization/duplicates, accumulator/initial/mask
  compatibility and checked shape/itemsize/byte arithmetic.
- Bounded JSON size/depth, node/edge count, parser nesting, generated source bytes,
  stage count and compile work. Limits must cover demonstrated workloads and be
  tested near boundaries; do not depend on unbounded recursive C traversals.
- WASM32 size/address overflow separately from 64-bit native behavior.

Use explicit failure categories for malformed graph, unsupported operation,
signature, shape, binding, execution, OOM and requested unavailable capabilities.
Do not promise preparation-time success for value-dependent conditions that are
only knowable during evaluation, such as weak construction or exceptional casts.

## 8. Planning and lowering

### First slice: map with optional root reduction

1. Validate graph and resolve typed nodes natively without compiling twice.
2. Identify the map region and optional root reduction; reject nested reductions.
3. Infer shape constraints; specialize actual input shapes in native code.
4. Lower the typed map to the existing portable typed kernel representation.
   Prefer a private typed-IR builder shared with DSL compilation, not a second
   source/JSON serializer that loses categories or re-infers under different rules.
   If textual lowering is used temporarily, make it bounded and require structural
   dtype/participation equivalence tests before treating it as authoritative.
5. Reuse the logical-array iterator and accumulator, retaining output ownership,
   serial logical order and bounded gathers. Do not reimplement those semantics.
6. Query result metadata and invocation resource requirements before output writes.
7. Choose native interpreter/JIT per region and expose actual selection.

Type-rule extraction should be small and private where practical. Keep existing
artifact/DSL behavior stable; both frontend paths need regression coverage if
shared inference/compiler internals are changed.

### Conservative optimization policy

- Initial implementation is correct without optimization beyond existing kernel
  fusion. Record an optimization-disabled control for differential testing.
- Do not fold arithmetic constants if doing so suppresses runtime diagnostics or
  changes weak construction timing. Metadata-only literal parsing is not evaluation.
- Common-subexpression elimination, dead-node removal and algebraic simplification
  require a defined effect/participation policy; start disabled except transformations
  proven to preserve values, rounding, error ordering and active FP flags.
- Fuse only compatible map regions. Keep explicit casts, assignment precision,
  lazy branches, reduction boundaries and final conversions observable.
- Require exact map dtype and diagnostic agreement before enabling a rewrite.
  Optimization may not silently exchange strict semantics for performance.

### Later slice: staged DAGs and intermediate reductions

Examples such as `x - sum(x, axis=0)` require a reduction stage and a broadcast
consumer, not one per-lane expression. Define materialization, dependency lifetime
and active evaluation before implementing them.

- Topologically schedule stages; report intermediate shapes, bytes and lifetimes.
- Initially materialize required intermediates with checked allocation and an
  explicit budget; reject excess resources rather than recomputing with different
  rounding or falling back to Python. No general bounded-memory claim follows.
- Preserve reduction grouping and once-per-active-coordinate shared-node semantics.
- Reductions inside conditional domains require additional semantics; initially
  reject rather than eagerly hoist a possibly inactive stage.
- Extend later to streamed/chunked stages only where semantic and dependency
  analysis proves equivalence. Parallel scheduling is another explicit extension.

## 9. Serialization, capabilities and cache identity

Graph interchange is a **new versioned format**, independent of artifact schema
1.0/1.1 and product release numbering. Proposed first marker:
`menudet-graph-1`; finalize with reviewed examples/schema before implementation.

The document records numerical semantics, node order, explicit types/categories,
lossless constants, reduction defaults/axes and required capabilities. Persist
portable int64/uint64 accumulator defaults explicitly where needed; never inherit
deployment C `long`/pointer width. Do not embed host addresses, storage objects,
compiler binaries, JIT code or an authoritative cached inference result.

Revalidate and prepare on import. Serialized inferred metadata may be an audit aid,
but may not bypass validation or override current rules. Reject unsupported
semantic revisions instead of reinterpreting a graph silently.

Initially avoid a global graph cache: the reusable handle provides reuse. Optional
host caches key canonical graph, semantic revision, typed signatures and immutable
captures. Shape specializations additionally key concrete geometry/options.
Compiled resources key lowering/ABI/backend/compiler configuration as required by
existing JIT machinery. Never include numerical array values as an implicit cache
dependency or retain caller buffer identities/results. Cache eviction and handle
lifetimes must be safe for concurrent independent executions.

## 10. Python-Blosc2 migration

1. Add thin Cython bindings for prepare/specialize/query/execute/export and explicit
   resource ownership. No NumPy evaluation or Python fallback inside bindings.
2. Convert the existing validated expression graph to structured native nodes.
   Translate syntax only; native code accepts/rejects operation signatures, computes
   types/shapes, recognizes reductions and determines stage boundaries.
3. Preserve native-required eligibility preflight before compressed reads and
   destination creation. Unsupported Python storage capabilities remain explicit
   frontend rejections, distinct from numerical graph rejection.
4. Replace `NativeSyntax`, root semantic extraction, generated DSL export-twice and
   Python `np.broadcast_shapes` planning in `native_graph.py` with native queries.
   Keep array owner extraction and storage selection adapters until separate work.
5. Differential-test old/new routes at an explicit checkpoint, then retain one
   semantic preparation route for the eligible subset. Do not leave permanently
   divergent Python and C planners behind a hidden switch.
6. Keep `_require_native=True` opt-in; ordinary LazyExpr/default backend and existing
   persistence policies do not change. New graph persistence is explicit, not an
   automatic upgrade of portable elementwise/block recipes.

Python may remain a graph authoring frontend. The acceptance requirement is that
an equivalent declarative graph can be prepared entirely by miniexpr, not that C
understands live Python objects or can unpickle a LazyExpr.

## 11. Work milestones and acceptance

### G0 — contract and baseline

- Enumerate current native-required supported/rejected graphs and scalar categories.
- Pin exact native/Python/reference revisions and separate current CI fixes from
  graph work; don't attribute baseline failures to new graph APIs.
- Review graph schema/grammar, public lifecycle, participation/order semantics,
  resource limits and first-slice eligibility. Add fixture/schema examples.
- Baseline preparation, repeated execution and allocation behavior.

**Exit:** approved first-slice contract, executable cases and measurable baseline;
no new dependency, default or artifact compatibility change implied.

### G1 — native validation and type preparation

- Implement structured graph import, ownership, bounded validation and immutable DAG.
- Share/extract private portable annotation rules; prepare typed maps and optional
  root reduction without array reads or Python source generation.
- Expose input signatures, result dtype and required capability queries.

**Exit:** standalone C host prepares supported graphs; malformed/unsupported graphs
reject deterministically; arithmetic/function corpus-derived inference agrees.

### G2 — native specialization and execution

- Implement pointer-free metadata specialization, broadcast/output queries and
  checked runtime bindings. Reuse `dsl_array.c` traversal/reductions.
- Execute map/root-reduction slice through interpreter and eligible JIT paths.
- Add per-call status, route reporting, independent invocation scratch and lifecycle
  tests. Report iterator/intermediate/compiler allocations separately.

**Exit:** standalone C and Node/WASM hosts prepare and evaluate representative
graphs with no Python runtime; changed values/shapes and failure recovery are tested.

### G3 — thin Python integration

- Implement Cython bindings and structured graph adapter; switch native-required
  preparation to native type/shape/reduction planning.
- Remove duplicate semantic preparation for the supported route after differential
  qualification. Preserve unrelated frontend/backend/persistence behavior.

**Exit:** existing native-required tests pass; tests forbid Python/NumExpr numerical
fallback and Python promotion/broadcast/reduction planning on this route. Python
still handles storage adapters, explicitly reported rather than labeled native IO.

### G4 — native text convenience frontend and round-trip

- Implement restricted expression grammar into the same internal DAG.
- Support canonical graph export/import, version/capability rejection and inspectable
  prepared plan descriptions without serializing machine resources.

**Exit:** equivalent structured/textual/Python-authored graphs produce equivalent
native metadata and execution; exported graphs run in standalone hosts.

### G5 — optional multi-stage extension

- Implement intermediate reductions, shared stage values and broadcast consumers
  with explicit materialization/lifetime/resource budgets.
- Add trusted portable kernel nodes only with documented signature, cardinality,
  context, effect and mask contracts; don't admit arbitrary callback/full-DSL nodes.

**Exit:** a separately declared staged subset passes its own value/type/shape/status
and resource tests. This is not required to claim first-slice native preparation.

### G6 — qualification and publication

- Run exact-pair native/Python CI on promised Linux/macOS/Windows and WASM targets,
  including interpreter-only and qualified host-JIT configurations.
- Run sanitizers, distribution builds and existing artifact/DSL regressions.
- Publish API/format examples, capability matrix, divergences and measured costs.

**Exit:** scoped native-preparation claim backed by exact revisions/configurations,
not source inspection, request-on fallback counts or historical green jobs.

## 12. Test strategy

### Preparation / validation

- No input buffer pointers in prepare/specialize descriptors; test inaccessible
  numerical buffers at execution boundaries and instrumentation proving no reads.
- Snapshot caller fenv before/after successful and failed preparation, including
  potentially exceptional constants; no numeric constant-fold side effects.
- Duplicate/missing inputs, node cycles/forward references, invalid opcodes/arity,
  unknown revisions, malformed bytes, deep/large graphs and allocation failures.
- Strong dtype matrix, weak/typed/0-D distinctions, capture range limits and
  inference independent of requested output. Reuse stable corpus-derived cases.

### Shape / execution

- Scalar/0-D, singleton/empty broadcasting, incompatible shapes, maximum rank,
  extent/product overflow and WASM32 capacities.
- C/F/reversed/stepped/unaligned/byte-swapped views; direct/gather report accuracy;
  capacity and input/output overlap rejection before writes.
- All six root reductions: axes/negative axes/duplicates, keepdims, initial,
  accumulator dtype, masks, empty identities/errors and tile invariance.
- Signed integer extrema/wrapping, exact mixed comparisons, quiet/signaling NaNs,
  signed zero, domain/overflow/underflow and final cast errors.
- Lazy selections with hazardous inactive branches; shared nodes under differing
  masks; runtime status aggregation, caller fenv restoration and raise/recovery.
- Rebind changed values without cached results; re-specialize changed shapes;
  reject mismatched dtype/category; concurrent independent calls and handle lifetimes.
- Interpreter versus actual eligible JIT, with optimization disabled/enabled where
  available. Requested compilation is not proof of compiled execution.

### Deployment / integration

- C-only executable loads graph, prepares, queries output, binds buffers and runs;
  it does not link Python/NumPy. Standalone WASM has explicit fenv limitations.
- Python authoring -> canonical graph -> standalone native execution, plus native
  expression frontend -> identical contract. Check capture and reduction round-trip.
- Native-required Python tests trap semantic preparation helpers and numerical
  fallback; retain copying/materialization counters and unsupported storage checks.
- Full existing portable 1.0/1.1, full-DSL, persistence and nullable-table regression
  suites where shared internals change. Qualification must not broaden their claims.

Use deterministic property tests and minimized permanent graph fixtures. Do not
replace authoritative NumPy cases with hand-authored matching outputs or widen
accuracy tolerances to make a platform pass.

## 13. Performance and memory evidence

Measure separately: decode/parse, validation/inference, specialization, kernel
compilation, first execution, warm execution and rebinding. Compare current Python
preparation with structured native and native text frontends over identical graphs.

Include tiny/large arrays, float32/64 and integer/mixed signatures, graph node count,
shared expressions, broadcasts, layouts, masks and reductions. For staged graphs,
include intermediate bytes and liveness—not only fused arithmetic microbenchmarks.
Use interleaved/repeated subprocess measurements with exact revisions, compiler
versions, backend routes and warm/cold cache state recorded.

Report plan bytes, specialization bytes, invocation scratch, output/intermediates,
compiler memory and whole-process peak RSS separately. A bounded native iterator
does not establish bounded compressed frontend memory. Define workload budgets
after baseline measurements; do not require an invented universal speedup factor.

Expect the largest architectural gain to be **one shared semantic preparation
implementation and reusable native plans**, not necessarily faster execution of an
already compiled flat kernel. First-call compilation may dominate short workloads.

## 14. Scope decisions and risks to resolve early

- **Effectful DAG sharing:** define ordered active evaluation before CSE/fusion;
  otherwise status/error behavior can diverge despite matching finite values.
- **Typed-IR reuse:** assess a private native builder versus bounded textual lowering;
  avoid exposing unstable compiler internals as the public interchange ABI.
- **Shape polymorphism:** separate plan/schedule/invocation and explicit cache keys;
  don't bind a supposedly reusable plan to one caller's buffers or old geometry.
- **Parser compatibility:** restricted native text syntax versus Python authoring
  normalization; unsupported syntax must reject clearly, not be guessed.
- **Reduction serialization:** root options, mask domains, width defaults and initial
  encoding must be explicit; existing block reduction artifacts stay distinct.
- **Multistage memory:** materialized intermediates need visible limits; native
  graph support does not automatically yield streamed compressed traversal.
- **Release interaction:** this can land opt-in after the current M7 freeze rather
  than becoming an unbounded prerequisite to finishing NumPy-compat qualification.

## 15. First actionable implementation checkpoint

1. Build reviewed structured fixtures for `x * 2 + y`, lazy
   `where(x != 0, y / x, y)`, a broadcast map and a masked axis sum.
2. Specify their scalar strength, operand order, result dtype/shape, diagnostics,
   resource rejection cases and expected native deployment behavior.
3. Finalize minimal prepare/specialize/query/execute signatures and graph schema.
4. Implement native structured preparation and inference with no buffer access.
5. Execute through existing portable map + logical-array APIs in a standalone C
   host before changing Python integration.
6. Bind that slice into Python and replace the duplicated preparation steps only
   after same-graph differential tests pass.

Success means **supported numerical graphs can be prepared entirely in miniexpr
and evaluated from explicit native buffers without Python**. It does not mean all
LazyExpr objects, compressed-storage workflows or NumPy APIs have become portable.
