# Native numerical graph preparation (opt-in, graph format 1)

`src/miniexpr_graph.h` introduces a **new, experimental** interface, independent
of artifact schema/language 1.0/1.1 and of product version numbering. Build with
`MINIEXPR_BUILD_ARTIFACT=ON` and link `miniexpr_artifact` and its miniexpr dependency.
There are no new dependencies: JSON uses the existing yyjson library.

## Lifecycle and ownership

1. `me_graph_prepare_json` copies a declarative graph and captures, validates it,
   and infers numerical types using the portable **1.1** compiler. No array
   pointers, Python, numerical evaluation, callbacks or full-DSL fallback exist
   in this path. Caller FP flags and rounding are preserved, including on errors.
2. `me_graph_specialize` accepts names, dtypes, ranks and shapes only. It performs
   trailing-axis broadcasting (including rank zero, singleton-to-empty axes),
   native axis normalization and reduction result-shape/accumulator validation.
   Query output dtype/shape/bytes and scratch before allocation or storage reads.
3. `me_graph_execute` binds checked `me_array_view` buffers by name. Shapes and
   dtypes must exactly match the specialization. Layout, capacity, byte order,
   mask and overlap validation precede numerical execution. Changing shapes
   requires a new schedule; changing values does not require a new plan.
4. Free the schedule and plan with their NULL-safe free functions. A schedule
   retains its plan. Borrowed names, JSON and shapes last until their owning
   handle is freed. Independent invocations can share immutable handles; freeing
   a handle concurrently with its use or mutating shared inputs is unsupported.

Options have `struct_size=sizeof(options)` and `version=ME_GRAPH_VERSION` (1).
NULL options choose the interpreter, a 1024-item tile and ignore FP flags.
JIT preference and `require_jit` are explicit; required unavailable acceleration
rejects. Reports name the actual **map kernel** route; reductions still use the
serial logical accumulator. No optimization/reassociation/CSE is enabled, whether
`disable_optimization` is true or false.

`me_graph_export_json` returns canonical declarative JSON, with sorted object
keys and preserved node/operand order. Import is preparation and revalidation,
never deserialization of executable machine code or cached inference. Captures
retain categories and value bytes. No global native graph/result cache exists.
`me_graph_export_map_json` additionally exports eligible elementwise graphs as
ordinary typed 1.1 artifacts; reductions and final conversions are not silently
changed into block-scalar recipes.

## Format and supported subset

See `native-graph-1.schema.json` and executable fixtures in `tests/graph/fixtures`.
All root fields are required:

```json
{"format":"menudet-graph-1","semantics":"menudet-numpy-1.1",
 "requires":["numeric"],"nodes":[],"root":0,
 "output":{"dtype":"auto","casting":"unsafe"}}
```

The empty node array above illustrates fields only; it is not a valid graph.
Node IDs are exactly their zero-based positions. Edges reference earlier nodes.
Every node must be reachable from the result or reduction participation binding.
Duplicate JSON keys, unknown fields/revisions/capabilities/opcodes and malformed
scalar encodings reject. Binding names are ASCII identifiers of at most 63 bytes;
names colliding with generated `_me_capture_<node-id>` bindings reject.

| Node `op` | Fields in addition to `id`, `op` |
| --- | --- |
| `input` | `name`, numeric `dtype` (strong; zero-dimensional arrays stay inputs) |
| `constant` | `dtype`, `category`, `encoding`, `value` |
| Unary `neg`, `pos`, `not`, `invert` | `args`: one node |
| Binary `add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`, comparisons `eq/ne/lt/le/gt/ge`, bitwise `bitand/bitor/bitxor/lshift/rshift`, lazy `and/or` | `args`: two ordered nodes |
| `cast` | one `args` node, destination `dtype` |
| `function` | canonical portable `name`, ordered `args` (up to three) |
| `select` | three `args`: condition, active-true, active-false |
| Root `sum/prod/min/max/any/all` | one map `args` node, `axes`, `keepdims`, `dtype`, `initial`, `where` |

Numerical types are bool, signed/unsigned 8/16/32/64-bit integers and float32/64.
Function arities/signatures/promotions are validated by the existing portable
compiler, not a graph-owned promotion/function table. See
`numpy-arithmetic-1.1.md` and `numpy-functions-1.1.md` for exact numerical policies.

Weak constants use bool/int64/float64 transport; typed scalars retain their declared
dtype. Integers are canonical decimal **strings** (including full uint64 values).
Booleans use `encoding:"boolean"` and a JSON Boolean. Reals use
`encoding:"ieee754-hex"` and 8/16 lowercase hexadecimal digits in most-significant
byte order. Parsing copies bit patterns, including signed zero and typed NaNs,
without evaluating them. Weak construction errors may occur only at invocation.

Reductions use `axes:null` for every axis or an explicit integer list (including
empty and negative axes); `keepdims` is Boolean; `dtype:"auto"` selects fixed
int64/uint64 widening for sum/prod, never deployment C `long`. `initial:null`
selects the existing default, otherwise it is a scalar descriptor without `id/op`.
If needed, initial conversion is native and occurs per invocation, not during
preparation. Any/all forbid initial/dtype overrides. `where:null` selects all
coordinates, otherwise it references a strong Boolean input. A participation-only
input may broadcast **to** the map shape but cannot expand it. Computed participation
masks are not yet admitted. Initial/masked extrema and empty identities follow
`native-arrays-1.1.md`. Each group accumulates in serial logical C order;
tile size never changes grouping.

`output.dtype:"auto"` preserves the inferred result. A numeric final dtype with
`safe`, `same_kind` or `unsafe` casting is checked independently of map precision
and reduction accumulation. A differing final dtype creates an interpreted
conversion stage, not a request to widen intermediate arithmetic. The numerical
result is materialized with a checked budget (64 MiB by default;
`intermediate_budget` overrides it). Query intermediate bytes separately from
iterator scratch. Conversion errors leave the output unspecified. This limited
conversion stage is **not** general multi-stage DAG support.

### Participation, sharing and resource limits

Operand order is retained by bounded textual lowering into the authoritative typed
portable kernel. `select` and Boolean short-circuiting remain lazy. Reduction masks
gate the map before evaluation. All computed nodes are conservatively treated as
potentially effectful. Inputs and capture bindings may be shared; **shared computed
nodes reject**, rather than silently duplicate/hoist work across active domains.
Intermediate reductions and shared array computations require the separately
declared [staged subset](native-stages-1.md). Reductions inside conditional domains
and arbitrary full-DSL/callback nodes still reject.

Limits: 1 MiB JSON/expression/generated-buffer size; 256 nodes; 64 graph/parser
expression nesting; 16 JSON structural nesting; 32 object fields; `ME_MAX_VARS`
total map bindings; rank 16; three operands per function node; signed-32-bit tile
count. Shape/byte/scratch/stride arithmetic is checked against target `size_t`,
including WASM32. The graph tree cannot grow exponentially during lowering:
computed sharing is rejected first. This is bounded validation, **not a sandbox**.
OOM never falls back to another planner or backend.

## Restricted expression convenience grammar

`me_graph_prepare_expression` takes expression text plus named dtype signatures
(their shapes are ignored until specialization). It builds the identical graph.
Supported syntax: identifiers; weak decimal integer/finite real/Boolean literals;
parentheses; the operators above with Python-like precedence; canonical function
calls; dtype casts such as `float32(x)`; lazy `where(c,a,b)`; a root reduction with
literal `axis`, `keepdims`, `dtype`, `initial` and an input `where` keyword.
Negative literals preserve int64 extrema. **Not general Python:** attributes,
methods, subscripts, tuples as values, chained comparisons, arbitrary callbacks
and keyword elementwise functions reject. JSON is the lossless capture interface.
Examples:

```text
x * 2 + y
where(x != 0, y / x, y)
sum(sqrt(x), axis=[-1], keepdims=True, initial=2, where=mask)
```

## Standalone deployment and Python migration

`tests/graph/host.c` loads a fixture, prepares it, specializes only metadata,
queries/allocates the result and executes native buffers. It links no Python or
NumPy. CTest runs all five fixtures, lifecycle/layout/mask/stage tests and corpus-derived
single-return, constant-free inference cases with original corpus IDs. Node/WASM
uses the same C hosts with raw host-file access; WASM FP reporting is unavailable
and requested raise policies reject before execution. Interpretation is the native
portable route, not a JavaScript numerical fallback.

Python-Blosc2's `native_graph.py` now emits syntactic graph nodes and explicit
captures. Cython owns prepare/specialize/query/execute/export lifetimes. Python no
longer generates DSL twice, promotes graph types, or broadcasts graph shapes.
Storage selection/decompression and output persistence remain Python adapters;
`_require_native=True` is still opt-in. The 128-entry Python handle cache keys
declarative graph/captures/backend configuration, never numerical array values.
Elementwise `native_kernel()` preserves existing portable artifact persistence.
This migration requires a graph-enabled miniexpr build: older dependency revisions
reject explicitly rather than retaining a second hidden semantic planner.
For local exact-pair testing build Python with
`-Ccmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/path/to/miniexpr`.
Graph qualification runs must set `MENUDET_REQUIRE_GRAPH_RUNTIME=1`; ordinary
optional-dependency tests skip if the installed revision lacks graph preparation,
but qualification may not count those skips as passes. The default Python miniexpr
pin has not been changed to an unpublished native revision.

## Qualification boundaries and remaining work

The first slice is map plus an optional root logical reduction, with a distinct
optional final conversion. G5 now has a separate opt-in materialized subset for
intermediate reductions, shared strong stage values, broadcast consumers and
trusted validated portable 1.1 regions. See `native-stages-1.md` for its explicit
participation, lifetime and resource contracts; it does not broaden the first slice.
Current platform CI already runs graph hosts/corpus tests wherever artifact tests
run, including interpreter-only and Node/WASM configurations. Actual publication
qualification still requires fresh exact-revision Linux/macOS/Windows/WASM jobs
and broader compiler allocator qualification. Local distribution/install and
graph/JSON/materialization allocation checks pass; local green tests do not
certify those platforms or promise a universal speedup/NumPy reduction bit parity.

### Local checkpoint (2026-10-09)

Baseline native revision: `b0541bb54f65cbe8cdb7d9de2601372117a1ffaf`;
Python baseline: `cf9cf3596708d69f1318c739613e03b6fcb56d84` plus this graph work.
Reference NumPy 2.5.3 / Python 3.14.4, macOS arm64, interpreter and host JIT.
Graph inference checks 2,778 arithmetic and 854 function corpus cases; 415/209
cases outside the single-return/constant-free inference adapter are explicitly
skipped, not counted as graph conformance. Existing artifact corpora still cover
their own wider contracts. Graph/yyjson allocation failure tests now exercise
37 map, 278 automatic-stage, 38 final-conversion and 70 reduction-initial allocation
checkpoints, including schedule/execution and successful recovery. Compiler allocation
injection remains unqualified; these checks do not claim coverage of every allocator.
Standalone Node/WASM32 interpreter hosts pass the five fixtures, lifecycle and
inference tests; native ASan graph/array/artifact tests also pass.

Latest local G5 qualification: 439 native CTests passed; the paired Python graph,
array, JIT, portable artifact/DSL, persistence and nullable-table regression selection
passed 919 tests with 17 skips in 149 seconds. Graph runtime was required, so none
of those skips hide missing graph support. The Node/WASM graph selection passed
9 tests and the ASan graph/array/artifact selection passed 11. A standalone native
install linked and executed the staged host using installed headers/libraries; a
non-editable Python wheel built and its independently installed staged smoke test
passed. Paired CI now requires graph support and tests the non-editable wheel,
records both revisions and uploads distribution/configuration evidence. Remote
Linux/Windows/exact-pair publication qualification has **not** run locally.

`bench/benchmark_graph_preparation.c` reports C-only CPU-time phases and graph-owned
metadata/schedule bytes, iterator bounds/actual peak and output size separately.
Compiler/artifact resources and process RSS are **not** included in those metadata
queries. Python's `bench/ndarray/native-graph-preparation.py` interleaves the old
export-twice baseline, native JSON and native text preparation, specialization,
first/warm execution and value rebinding, with revision/configuration metadata.
One warm-process, interpreter-only float32 sample (10 repetitions, 10,000 items)
measured preparation medians of 137.0 / 16.8 / 22.8 microseconds respectively;
specialization 5.8 microseconds. Execution was approximately 22.7 milliseconds
on all native runs: this is a preparation architecture gain, not a claimed execution
speedup. Repeated subprocess/cold-compiler and broader workload/RSS studies remain
publication requirements, not extrapolations from this one sample.

### Materialized stage cost samples

`bench/ndarray/native-graph-stages.py` separates preparation (including requested
compilation), specialization, first/warm execution, rebinding, graph-owned metadata,
iterator bounds/actual scratch, output, reserved intermediates, last consumers and
whole-process peak RSS. Compiler-only memory is not inferred from RSS. It records
both revision IDs and dirty tracked-tree indicators; measurements below are working
tree G5 additions on native `5fd1a2b` / Python `225c3596`, not released-pair signoff.

Fresh subprocess samples of `x - sum(x, axis=0)`, warm imports and five execution
repetitions, on the same macOS arm64/Python 3.14.4/NumPy 2.5.3 environment:

| Input / route | Prepare (us) | Specialize (us) | First (us) | Warm median (us) | Intermediate / output bytes | Process peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 (1000,16), interpreter | 444 | 10.6 | 26674 | 17653 | 64 / 64000 | 79.2 MB |
| float64 (1000,16), 2 actual JIT regions | 9241 | 13.0 | 294 | 259 | 128 / 128000 | 78.5 MB |
| int64 (10,16), interpreter | 152 | 10.8 | 226 | 190 | 128 / 1280 | 76.3 MB |

These are workload/route cost examples, **not controlled cross-dtype speedup
comparisons**. Preparation includes compilation when requested; import cost is
excluded. Reduction grouping is unchanged. Metadata is roughly 59 KB per two-stage
plan and 57 KB per specialization because this experimental API currently uses
fixed bounded stage/input arrays. This overhead is visible and may matter more than
materialization for tiny graphs; no memory-efficiency superiority is claimed.
