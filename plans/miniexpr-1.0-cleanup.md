# miniexpr 1.0: a block execution API for embedding hosts

Discussion draft, 2026-11-25. Recommendation: **keep compile-once,
evaluate-many, but make a typed kernel over host-owned blocks the primary
contract.** Make JSON one way to describe that kernel. Keep array traversal and
graph scheduling above it. Delete the obsolete semantic profiles before
attempting to replace either live expression engine.

## Evidence from the consumers

| Tree inspected | What it tells us |
| --- | --- |
| miniexpr `next`, `eeb8766` | The package is 1.0 alpha; the frozen public Menudet language and artifact format are both **0.1**. Graph and staged graph have their own format identifiers. A library major bump need not invalidate stored computations. |
| C-Blosc2 `origin/next`, `a6799b5d` (verified against the remote) | No `miniexpr` integration found in that tree. There is no deployed next-major call sequence here to preserve or to optimize against yet. |
| C-Blosc2 `b2nd-constructors`, `fc2774d4` | `blosc/b2nd-constructors.c:ramp_compile` formats floating literals, repairs locale-dependent decimal points, builds a JSON manifest, and loads it with JIT off. `ramp_prefilter` finds live C-order rows of a padded physical block; `ramp_span` evaluates a zero-input kernel with a logical origin, extent and output byte capacity. The host handles padding, exact endpoints and byte order. `bench/sum_linspace.c` separately uses the streaming pairwise-sum API. The latter is a benchmark, not a production constructor dependency. |
| Python-Blosc2 `main`, `4fa88239` | `src/blosc2/blosc2_ext.pyx` uses ND compilation with JIT options, valid-item queries, ordinal input buffers, ND evaluation and WASM helpers. `dsl_artifact_bridge.h`, `dsl_array_bridge.h`, `dsl_graph_bridge.h` and `pairwise_sum_bridge.h` cover distinct adapter paths. For example, graph preparation, specialization, stage inspection and execution already have callers. Looking only at C-Blosc2 would misclassify live APIs as dead. |

These are usage samples, not an external ABI census. The useful division of
labor is visible in the ramp: Blosc2 knows where a block lives, when it is
decompressed and which thread owns it. Miniexpr knows which lanes participate,
how to compute `_flat_idx`, what dtype to write, and what numerical order the
language promises. Neither library needs to know the other's object layout.

## Persistence decision for C-Blosc2 (likely 4.0) and Python-Blosc2 5.0

Version the Menudet recipe, not the whole container. Python-Blosc2's
`b2objects.py` writes a `{kind, version}` marker in fixed `b2o` metadata
and a versioned MessagePack payload in variable-length `b2o` metadata.
Dispatch on `(kind, version)` and require matching marker and payload when
both are present. Handle inline recipes and MessagePack extension 43 through
the same dispatch; the extension decoder also checks version 1 today.

- `menudet` v1: retain the existing reader for the embedded 0.1 JSON artifact.
  Ship it in all standard C-Blosc2 and Python-Blosc2 builds. Standalone
  miniexpr can still be built without JSON. Do not reinterpret a concrete v1
  signature as an author-generic recipe; v1 cannot tell which types were inferred.
- `menudet` v2: write normalized source, explicit language version, captures
  (preserve exact dtype/value and weak versus typed scalar category), authored
  policy/cardinality/context, and the existing domain, partitions and operand
  references as MessagePack fields. Prepare directly after binding, with no
  JSON step. Proposed Menudet 0.2 adds optional input and return annotations;
  persist these in source, not the input dtypes observed on the first call.
  Python resolves explicit annotations such as `np.uint16` to canonical
  Menudet names (`uint16`); the native parser maps those names to `me_dtype`.
  Omitted annotations stay omitted. An input mismatch rejects at preparation;
  a return annotation requests final conversion under the casting policy.
- Keep unchanged kinds such as `lazyexpr` at v1. A nested recipe dispatches
  on its own kind/version. A build without the JSON adapter should read v1
  plain expressions and v2 kernels, and report a specific unsupported-feature
  error if it encounters a v1 Menudet artifact.
- Qualify with Python-write/C-read and C-write/Python-read fixtures, including
  extension and nested forms. Test the v2 path with the adapter absent and keep
  golden v1 reads with it enabled. CTable descriptors containing kernel JSON
  need their own migration; native graph JSON is a separate contract.

## What to delete, what to separate

1. **Remove the draft profile; retire the checked-arithmetic profile.**
   `ME_DSL_PROFILE_LEGACY_DRAFT` has no selector in the inspected source.
   `ME_DSL_PROFILE_MENUDET_CHECKED` is different: internal entry points and
   `tests/menudet-dsl/runner.c` actually select it. It is not a public 0.1
   artifact. Inventory the corpus by behavior: port control flow, masks, local
   initialization and diagnostics to 0.1; keep independent expected values for
   NumPy overflow and promotion; retire cases whose only requirement is the
   former checked result. Then remove the selector, checked-only typing/eval
   branches and unused convenience helpers. Do not delete checked *bounds* or
   shape arithmetic, or equate “typed” with “checked”.
2. **Simplify artifact loading around the single admitted version pair.**
   `dsl_artifact.c` tests `numpy_profile` throughout manifest, constants and
   cast handling, yet successful loads assign `ME_DSL_PROFILE_MENUDET_0_1`.
   Classify malformed versus unsupported version up front, then validate a
   single schema. Keep negative fixtures (including duplicate keys), weak
   versus strong constants and capability rejection. For explicit outputs the
   loader currently compiles once with `ME_AUTO` to check casting, frees that
   program, then compiles again. After the schema cleanup, separate inference
   from final conversion/JIT so the typed program can be reused. Measure
   preparation latency separately from evaluation; never let a requested output
   dtype change intermediate arithmetic.
3. **Target retirement of the raw/full kernel dialect in 1.0; treat classic
   formulas separately.** Python-Blosc2 v4.15.0 retired `@dsl_kernel` and
   `DSLKernel.from_source` and rejects historical source-backed recipes;
   v4.16.0 additionally retired the legacy ordinary-expression evaluator.
   Neither the old Python authoring API nor v1 Menudet JSON requires preserving
   full DSL. Identify and migrate internal generated-source calls and check
   known direct C clients before deleting dialect-specific paths; do not
   silently reinterpret an old kernel as Menudet. `dsl_compile.c:dsl_compile_expr` calls
   `private_compile_profile_ex` on statement expressions. Deleting the
   TinyExpr-derived parser would also remove infrastructure used by portable
   kernels. Conversely the restricted graph text frontend is not a drop-in
   replacement for `me_compile`. Make the source language explicit in new code:
   today `compile_with_jit` can wrap a plain formula as a DSL function when
   `ME_JIT_ON` is requested. Backend preference should not select syntax or
   change arithmetic. Preserve old routing in the old entry points until the
   accepted-language, result and performance differences are measured.
4. **Take callbacks out of ordinary input declarations.** `me_variable`
   combines name/dtype/itemsize, an address, arity/purity flags and closure
   context. `test_dsl_udf.c` exercises callbacks, and the compiler itself
   represents cast intrinsics as function entries. Give the embedding path a
   plain input signature. Isolate public UDF support in the compatibility API;
   remove it only if consumers agree. Internally give inputs ordinal slots
   instead of synthetic addresses; give builtins operation IDs instead of
   function-pointer identities. The latter has a concrete payoff: Windows
   currently propagates `/OPT:NOICF` to avoid merging distinct callbacks.
   Remove the linker workaround only after Release tests with folding enabled.
5. **Consolidate builds and correct misleading documentation now.** The root
   Makefile lists six core `.c` files and misses the split compiler/evaluators;
   CMake has the maintained source list. Replace the Makefile with thin CMake
   shortcuts or drop it. The artifact fixture README describes draft 1.0, and
   the core header contains an obsolete five-argument `me_eval` example.
   Separate historical plans from current specifications. Retain upstream
   license notices. Offer named CMake targets for the core and optional
   adapters with transitive link requirements; test both `add_subdirectory`
   and a small installed consumer. Default tests/examples/benchmarks off when
   embedded. The constructor branch currently discovers headers and two
   libraries independently; it should be able to link one target instead.

## A possible public API

The [minimal core proposal](miniexpr-1.0-core-api.md) refines the sketch below:
two preparation calls sharing describe, eval and free, with JSON import outside the stable core.

Start with the existing artifact block descriptor, not a universal execution
object. The following names are illustrative; they are not new symbols yet.

```c
me_status me_kernel_prepare(const me_kernel_spec *spec,
                            const me_prepare_options *options,
                            me_kernel **out, me_error *error);
me_status me_kernel_load_json(const char *json, size_t length,
                              const me_prepare_options *options,
                              me_kernel **out, me_error *error);
me_status me_kernel_describe(const me_kernel *kernel, me_kernel_info *out);
me_status me_kernel_eval_block(const me_kernel *kernel,
                               const me_buffer *inputs, size_t ninputs,
                               me_mut_buffer output, const me_block *block,
                               const me_run_options *options, me_error *error);
void me_kernel_free(me_kernel *kernel);
```

- `me_kernel_spec` initially describes **Menudet 0.1**: source, entry point,
  input signatures, typed constant captures (including scalar category),
  output signature/cardinality, context rank, capabilities and numerical
  policy. Direct prepare and JSON import feed the **same validation and typed
  compiler**, and return the same handle. No public `yyjson` types. A C ramp
  passes `start` and `step` as binary captures to
  `def ramp(start, step): return start + _flat_idx * step` (with a normal
  indented body); there are zero *runtime* inputs. No float printing, escaping,
  locale handling or JSON library on the direct path. Persisted artifacts still
  require a concrete output type; allow inference explicitly for preparation.
- Inputs and output are contiguous native buffers with **byte capacities**.
  Bind once by name at setup if necessary, then pass buffers by slot in the hot
  path. `describe` reports slot order, dtype, itemsize, output cardinality,
  context requirements and partitionability; returned metadata lives with the
  handle. Preparation copies the signature/captures. Buffers, masks and shape
  pointers are borrowed only during a synchronous call. Share handles across
  threads, never mutable worker scratch. Preserve checked alignment, overlap,
  capacity and valid-mask rules from `me_artifact_eval_ex`; preserve FP status
  collection as a per-call option. Use a caller-owned diagnostic with status,
  location and message; no second TLS query needed on the new API.
- `me_block` describes **logical** element count, optional mask, rank, shape,
  origin and extent. Preserve the existing empty-group behavior for each
  cardinality; a block-scalar result is not an array of `nitems` results.
  Keep a separate **physical grid adapter** for `(shape, chunkshape,
  blockshape, nchunk, nblock)`, with valid-item query and padded-block eval.
  It owns padding/coordinate conversion, not storage. A multidimensional edge
  can have holes between rows: a live count alone is insufficient. Never split
  a block reduction into row reductions merely because the physical rows are
  disjoint. Hosts already holding compact spans need no grid adapter.
- Initialize extensible structs with documented defaults and `struct_size`;
  distinguish API struct version, source profile and persisted schema.
  Resolve prepare policy once. Make explicit host options win on the new API,
  with environment inheritance opt-in; preserve existing precedence in shims.
  Allow evaluation to select an available interpreter instead of a prepared JIT
  kernel without recompiling or changing profile.

**Migration map.** `me_artifact_load*` becomes optional JSON import over
`prepare`; `me_artifact_eval[_ex/_status]` adapts named inputs to slots and
preserves old status codes. For the retained classic formula interface, collapse the
`me_compile[_nd[_jit[_options]]]` family internally to one options-based
implementation; keep `me_compile`/`me_eval` as short entry points. Route
`me_eval_nd` and `me_nd_valid_nitems` through shared grid geometry. Do not map
full DSL to frozen Menudet by changing a default. Keep `me_eval_varlen`
explicit: offsets plus a UTF-8/binary payload have different capacity and
ownership rules from a fixed-width output block.

**Keep above the block API:** array views/readers, graph prepare → specialize
→ execute, staged lifetimes, transactional alias execution and host fork/join.
They solve different problems; a compression prefilter should not construct a
graph schedule to fill a block. Consolidate resident/source and flat/descriptor
output variants behind existing general implementations where equivalent.
Keep named wrappers as long as the Python bridge uses them. Move streaming
pairwise-sum declarations/implementation out of the JSON/array adapter so a
storage host can use them without a manifest parser. Preserve complete ordered
leaves and boundary fragments; combining arbitrary per-chunk sums produces a
different floating-point result.

## Delivery sequence and exit criteria

| Step | Change and evidence required to finish |
| --- | --- |
| 1. Record the baseline | Pin both downstream consumers and versioned fixture sets. Record accepted/rejected source, diagnostics, dtype/itemsize, cardinality and output for classic, full and portable paths. Benchmark cold prepare, repeated small block eval, padded-edge eval, ramp construction with compression, and ordered sum separately. Include allocation counts and binary/dependency size for a core-only host. |
| 2. Delete proven legacy | Land Makefile/docs, draft profile, checked-profile corpus migration and single-version loader as independent reviews. Require the 0.1 conformance corpus and old invalid-manifest classifications to pass. Keep 0.x release tags as the historical reference instead of shipping a dormant dialect. |
| 3. Pilot the embedding contract | Extract an internal typed spec and handle; make JSON and direct prepare share it. Try the direct API in the constructor branch alongside the old route. Compare bytes for single/multiple workers, exact endpoint, padded final block, swapped endian and rank-zero cases. Compare failure/cleanup behavior. Gate adoption on measured block overhead and removal of host boilerplate, not on API aesthetics. |
| 4. Migrate real bindings | Add grid adapter and shims, migrate the Python ND path and artifact bridge one at a time. Exercise fixed/varlen strings, empty/masked reductions, `_i*`/`_n*`/`_flat_idx`, JIT off/on/fallback and independent concurrent evaluations. Keep graph reader/no-replay and pairwise numerical-order tests. Verify host build/link on Linux, macOS, Windows clang-cl and WASM side modules. |
| 5. Refactor hot internals selectively | Highest priority: ordinal slots and builtin identities. Then profile the classic `functions.c:me_eval` tree-clone/rebind path; if material, introduce invocation-local bindings/scratch, optionally reusable per worker. Compare allocations and throughput before changing dispatch. Separate registry metadata from scalar/vector kernels incrementally; keep strict FP compilation flags and interpreter/JIT parity tests. File size alone does not justify rewriting the SIMD or JIT engines. |
| 6. Freeze 1.0 | Publish supported headers/symbols, defaults, ownership, threading, errors, migration examples and build targets. Check for private cross-library and JIT bridge symbols before tightening visibility (Windows currently exports all core symbols). Drop old public wrappers only after downstream migration or explicit agreement to break them; otherwise ship a small compatibility header for a stated transition window. |

**Decisions still open before the API freeze:**

1. Which classic formula entry points (including C callbacks) are supported
   versus compatibility-only in 1.x? Inventory remaining dependencies on the
   full kernel dialect as migration work, not as a Python compatibility promise.
2. The ABI decision is settled: C/Cython consumers must rebuild for 0.x →
   1.0. The supported 1.0 C API is the stability baseline for 1.x. Existing
   v1 Menudet files remain readable in standard Blosc2 builds.
3. Which new APIs are stable in 1.0? Pilot direct prepare, block evaluation
   and grid handling with the constructor and Python ND paths before freezing
   descriptor layouts. Review the graph/array APIs separately.
4. What is the interoperable v2 subset for the first release? Specify exact
   capture encodings, supported operand references and rejection rules in a
   shared fixture set. Finalize the 0.2 annotation vocabulary, including
   whether fixed-string widths stay in binding descriptors for this release.
