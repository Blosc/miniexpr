# Blosc2 5.x execution handoff

This is the implementation sequence for the execution phase of
[`blosc2-4.17-to-5.md`](blosc2-4.17-to-5.md). The new execution API is not a
storage format and will not change `b2o`.

## Ownership

- The C-Blosc2 compute build selects one miniexpr 1.x core. For an installed
  static C-Blosc2, its CMake package must resolve the same core dependency.
  Keep compression-only builds usable without miniexpr. Python may arrange the
  bundled build, but must not link a second independent core alongside the
  one C-Blosc2 uses. An installed wheel must have one native implementation
  and one provider of compiler/runtime state.
- C-Blosc2 consumes the typed kernel, grid and numerical interfaces; it owns
  whole-array plans, schedules, source/sink callbacks, scratch, worker
  partitioning and the order in which partial results combine. Expose neither
  JSON nor MessagePack to the native scheduling entry points.
- A prepared C plan owns its miniexpr kernels and immutable metadata. A
  specialized schedule borrows its plan; its caller keeps the plan alive
  until all its schedules are destroyed. The Python schedule wrapper keeps a
  strong owner reference. Execution is synchronous and does not retain views,
  storage callbacks or output buffers past return.
- Python owns `b2o` and reference resolution, JSON interchange, and adaptation
  of Python arrays, remote references and tables. The optional miniexpr
  adapter remains the comparison implementation until parity is established;
  any independent-client deprecation is separate.

## Callers and order of work

| Current caller | Move or retain |
| --- | --- |
| C-Blosc2 `b2nd-constructors.c` | Existing miniexpr grid/block consumer; use it to prove the installed dependency boundary. |
| Python `MenudetArtifactHandle` preparation, block eval, prefilter and descriptor paths | Replace the compatibility handle with typed preparation and core block evaluation. Preserve actual backend and FP status reporting. |
| `MenudetLazyArray` and Cython `evaluate_array*`, source/sink and preflight bridges | Move whole-array orchestration. Keep logical groups immutable across rechunking and do not materialize remote or strided operands as a shortcut. |
| `NativeGraphHandle`, `NativeGraphSchedule`, `native_graph.py` resident/prefilter, compressed/source-backed and staged reductions | Move planner/schedule/executor together once the typed graph boundary can represent their contracts. Preserve bounded scratch, reduction ordering, alias handling and read/write reports. |
| Ordinary formula prefilter, full-DSL fallback and legacy formula reductions | Keep the compatibility path where it still owns distinct syntax or scalar-tail semantics. Do not substitute a Menudet kernel silently. |
| Explicit `MenudetKernel.from_json()` and persisted v1 Menudet objects/columns | Python decodes the legacy fixed manifest; C preparation still verifies source-dependent semantics. Validate before references are read or a JIT compiler is invoked. |

1. Qualify the installed C-Blosc2 static consumer with miniexpr's artifact
   build off. Introduce a C-Blosc2 array scheduling pilot behind a separate
   build/test path; compare with the current Python-facing array scheduler
   before replacing it. Include a contiguous input, padded grid and a
   source-backed case with a bounded tile. The initial internal grid pilot
   covers one compressed source, padding and a block-scalar result with bounded
   scheduler buffers. Its physical blocks are its groups; it is not a
   replacement for arbitrary persisted logical partitions.
2. Replace the scheduler's **input** with a typed, bounded, topologically
   ordered node array: operand slots, exact scalar captures and typed operation
   parameters; root and staged reduction references are node IDs. Copy the
   retained fields during prepare. The JSON and text frontends currently feed
   yyjson trees into the planner; each must instead lower to this descriptor.
   Map and cast kernels now use typed miniexpr signatures directly; the old map
    manifest remains an export, not a preparation input. The internal nodes
    now carry decoded dtypes, scalar bytes and mask references; map lowering
    and partitionability checks no longer consult the wire nodes. Automatic
    stage construction and child dtype resolution now pass in-memory trees
    instead of serializing and reparsing JSON text. The text frontend passes
    typed nodes directly to the shared planner, including reduction axes,
    accumulator dtype, masks and initial scalars. Its compatibility tree
    still supplies interchange exports. Automatic stage preparation remaps
    typed nodes and connects producer outputs directly. Explicit staged JSON
    imports decode bindings once into typed sources; inferred child input
    dtypes no longer require copying or rewriting a child tree. Keep
    explicit graph interchange and canonical cache identity as separate
    compatibility concerns.
3. Run the Python v1 importer in differential tests against the native adapter
   before making it the default. Preserve duplicate-key and UTF-8 rejection,
   JSON limits, binding order, weak/typed captures, context/rank, width and
   casting checks. Some legacy `requires` entries depend on compiled code
   (control flow, block reductions, fixed strings); validate those using the
   typed core before JIT, not a second implementation of the DSL in Python.
   `to_json()` remains explicit interchange; it need not round-trip through
   a native JSON parser.
4. Install a standard Python wheel with `MINIEXPR_BUILD_ARTIFACT=OFF`. Reopen
   the original v1 fixture and the 4.17 v2 exchange fixtures, including CFrame
   and table metadata. Prove real array/graph execution, not a fallback;
   audit an empty build tree for yyjson fetches and the wheel's link inputs for
   both archive and dynamic copies. Remove the old Blosc2 wiring only after
   this gate passes.

## Performance and correctness gates

Compare the old and new executor in the same build configuration on at least
ARM64 and x86-64. Run prepare-only and warmed execution separately. Cover
contiguous memory, compressed padded/chunked input, streamed sources, in-place
resident maps where supported, and whole-axis and staged/masked reductions
with different worker counts and partitions. Use identical generated values,
codecs, tile/scratch budgets and kernel backends. Compare dtype, values
(bitwise in strict mode where the contract guarantees it), FP flags, callback
counts and peak scratch. Record medians over interleaved runs; investigate a
repeatable regression exceeding 5% on a representative production path
before switching defaults. Do not trade bounded scratch or deterministic
reduction order for a benchmark improvement.

## Status

Python-Blosc2 `next` pins miniexpr `9eb1349` (typed map and conversion
preparation without an intermediate manifest). An isolated 5.0.0a1 wheel built
from that pin passed 596 of 597 focused Menudet, graph and expression tests.
The single failure was a local editable-install artifact in a subprocess test;
the same script passes when run against the wheel. The C-Blosc2 pilot at
`985a2676` passed its hosted job. The 4.17.0 storage release is tagged and live
on PyPI; the released wheel and the 5.x wheel exchanged the fixture in both
directions. The typed node array described in step 2 is the next implementation
step.
