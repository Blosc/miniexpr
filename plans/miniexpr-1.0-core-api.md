# miniexpr 1.0: proposed stable C core

Provisional design, now piloted in `src/miniexpr_kernel.h`. Design around one unit of work:
specialize source against a known signature, then repeatedly execute that
immutable specialization on logical blocks supplied by the caller.

Release boundary: C/Cython consumers rebuild for miniexpr 1.0 (C-Blosc2 4.0,
Python-Blosc2 5.0); 0.x binary layouts and symbols do not constrain the new
ABI. Freeze the supported 1.x interface after the downstream pilot. Defer
formula reduction semantics until characterized; do not let the pilot
implicitly freeze them.

```c
typedef struct me_kernel me_kernel;

me_status me_prepare_formula(const char *source, const me_signature *signature,
                             const me_prepare_options *options,
                             me_kernel **out, me_error *error);
me_status me_prepare_menudet(const char *version, const char *source,
                             const me_signature *signature,
                             const me_prepare_options *options,
                             me_kernel **out, me_error *error);

me_status me_describe(const me_kernel *kernel, me_kernel_info *out);

me_status me_eval_block(const me_kernel *kernel,
                        const me_buffer *inputs, size_t ninputs,
                        me_mut_buffer output, const me_block *block,
                        const me_run_options *options, me_error *error);

void me_kernel_free(me_kernel *kernel);
```

## Preparation

- Select the language by preparation function; pass an explicit language
  version for Menudet. Select language before
  selecting a backend. Formula denotes the supported one-line expression
  contract, not full-DSL `def` detection or rewriting under JIT. Pin that
  formula subset and numerical behavior with tests before freezing the API.
- `me_signature`: named runtime inputs in slot order (dtype and byte itemsize),
  named immutable scalar captures (dtype, itemsize, exact native value and weak
  versus typed category), logical context rank, result/casting contract.
  Permit an inferred output where the language supports it. Signatures carry
  metadata, never input array addresses. Zero runtime inputs is valid.
- In Menudet 0.2 a parameter annotation constrains the supplied dtype; it is
  not a request for an implicit input conversion. A declared return type is a
  final conversion. Conflicting explicit declarations reject. Version 0.1
  retains its current contract. Specialization does not modify recipe source.
- Keep authored semantic requirements (including math/FP requirements) distinct
  from host preparation options (e.g. interpreter, prefer JIT, require JIT).
  A backend must implement the selected semantics or decline before execution.
- Preparation copies all retained metadata and captures. On failure `*out` is
  NULL. `me_error` is an optional caller-owned structured diagnostic with a
  stable status and source location when available. No TLS query required.

## Execution

- `me_describe` returns the concrete result dtype, itemsize, and cardinality
  (per lane or per logical block), and required context. Input slot order is the
  order supplied at preparation. A name-based importer resolves slots once.
  The caller can size output without trial execution. Make this an O(1),
  allocation-free copy of prepared metadata into a caller-owned, size-tagged
  struct. It describes the actual output after final conversion, not the
  intermediate arithmetic dtype or accumulator dtype. It does not accept a
  block or compute a per-invocation shape, valid-lane count or scratch budget.
- Buffers are pointer + byte capacity, in native format with the required
  alignment. A fixed-width string's itemsize is bytes, not characters.
  Input and output must be disjoint. Validate before numerical dispatch.
- `me_block` carries lane count and optional logical shape/origin/extent and
  valid-lane mask, with the same coordinate/participation rules for a given
  Menudet profile as its existing checked evaluator. No physical chunk IDs.
  An all-valid contiguous block needs only its lane count. A mask changes
  participation, not the declared physical spacing of elements in a buffer.
  Empty elementwise blocks and empty scalar reductions are distinct contracts.
- The host owns output and concurrency. Evaluation is synchronous; no buffer
  is retained. Independent calls may share a kernel, not worker state. Optional
  run settings cover invocation-local FP reporting. Failures do not trigger a
  replay. `me_kernel_free(NULL)` is harmless.

## Boundaries and ABI

No recipe object, serialize method, storage reader, thread pool, mutable global
context or public scratch allocator is required for this core. The Blosc2
adapter resolves references and decodes `b2o`; the v1 JSON importer checks
manifest/schema requirements then uses the typed preparation implementation.
Neither can bypass semantic validation. Keep graph plans/schedules, physical
padded-grid adapters (including valid-item queries), streaming reductions and
offsets-plus-data varlen output as distinct interfaces. They need not masquerade
as ordinary contiguous blocks. Keep old short formula calls as wrappers where
their behavior can be preserved.

## Pilot implementation status

- Implemented the five calls in `miniexpr_kernel.h`/`.c`, using shared typed
  compiler/evaluator internals rather than a JSON wrapper. Captures are copied;
  input slots are ordinal. Added checked capacities, masks, overlap, Unicode
  input validation and invocation-local FP reporting.
- Added explicitly selected 0.2 header parsing and annotation checks without
  changing 0.1 acceptance. Return annotations convert last; output casting uses
  the inferred dtype from the existing discovery pass rather than a second
  complete compilation. Historical artifact imports remain unchanged.
- `tests/test_kernel_api.c` exercises the formula path, unused/reordered slots,
  captures, all canonical annotation names, signature rejection, return
  conversion precision, fixed bytes, zero-length calls, scalar blocks, masks,
  FP state restoration and shared-handle threads. Optional artifact-enabled
  coverage compares direct ramp results byte-for-byte with a v1 JSON import.
- The actual C-Blosc2 pilot is on the sibling checkout's `b2nd-constructors`
  branch, not `origin/next`: ramp preparation now uses binary captures and
  block evaluation. Constructor linkage no longer needs the artifact library.
  Its constructor suite passes against an artifact-disabled miniexpr build.
- Public behavior and limitations are documented in `doc/core-api.md`;
  `doc/dsl-spec/0.2.md` is the incremental annotation draft.

Next: migrate Python author normalization and native bindings, define/write
`b2o` v2 with cross-language fixtures, provide the shared padded-grid adapter,
then remove old entry points and full-DSL paths once callers have migrated.
The new formula API deliberately rejects reductions and context; it does not
resolve the deferred formula reduction question by silently inheriting old
behavior. Existing APIs and v1 readers still work.

Integration update:

- Python's `next` checkout now prepares reusable recipes through the typed
  core, retains canonical annotations, writes Menudet `b2o` v2, reads v1,
  and preserves existing graph/resident scheduling and provenance checks.
  Computed/generated table recipes have an analogous v2 field.
- `miniexpr_grid.h` now supplies the shared padded-grid adapter. The C-Blosc2
  constructor branch uses it for ND ramps. Native tests explicitly cover
  holes between live rows, context coordinates, completely padded groups,
  and a nonlinear scalar post-reduction that must see the whole group.
- Removed the unused draft semantic-profile selector and Python's obsolete
  float64-probe/JSON fallback for source preparation. Live legacy C paths are
  still being migrated; they have not been deleted merely because the new API
  exists. C-Blosc2 full b2o reference-closure import/export remains outstanding.

Local qualification (macOS arm64): the pre-change artifact-enabled baseline
passed 369 tests; the pilot passed all 370 tests, including the new core API
suite, with SLEEF/TCC disabled. Explicit host-CC required-JIT execution passed.
Five focused tests passed with ASan/UBSan and artifact support disabled.
The C-Blosc2 constructor suite passed with its existing ByteDelta plugin enabled,
using only the JSON-disabled core library. C99 warning-as-error syntax checking
and C++11 header inclusion also passed. No compiler warnings were reported.
At pilot time, Windows, Linux, WASM and bundled-TCC qualification remained outstanding.

The integration revision passes all 371 native tests and six focused
artifact-enabled ASan/UBSan tests. A core-only installation was also consumed
by a separate C99 grid test using installed public headers and no JSON library.

Before ABI freeze, fix the status numbering, descriptor field widths, layouts,
initializers, extension rules and lifetimes. Extensible structs cross the ABI
by pointer with size tags and documented defaults; pointer/byte-count buffer
views passed by value are fixed layout. Keep public dtype numbers stable.
Keep source-language, storage-format and C ABI versions independent.

Two acceptance clients before freezing: a zero-input C ramp with captures and
logical coordinates, and a Python-authored partly annotated kernel specialized
in C and evaluated concurrently on independent blocks. Also qualify the padded
adapter with non-contiguous live rows and a block-scalar kernel; splitting that
logical group by row would give the wrong result.

## Review 1 follow-up

The [implementation review](miniexpr-1.0-review1.md) identified real parity and
contract gaps. The follow-up addresses these without freezing the ABI:

- Direct Menudet and fixed JSON preparation now share deferred backend
  preparation. Casting, width and cardinality checks happen before runtime
  compiler/cache activity. JSON no longer compiles a throwaway inference
  program; dead `numpy_profile` branches are removed after version validation.
- Weak int64/Boolean capture specialization runs after annotation typing.
  Required host-CC and bundled-TCC tests cover captured loop bounds. Backend
  preparation still preserves caller rounding mode and exception flags.
- Signature/annotation/context mismatches have binding status, invalid options
  have argument status, and the adapter reports required capabilities rather
  than every supported capability. Entry-point ownership and AUTO policy no
  longer require a second parse.
- All-active masks use the unmasked evaluator path. Formulas snapshot inherited
  math policy when requested. Grid preflight/OOM failures initialize FP reports.
- Partial grids copy contiguous runs with an outer-axis odometer and one
  alignment-checked allocation. Tests cover multiple inputs with mixed widths,
  3D padding on each axis, whole-group scalar reductions and injected allocation
  failures. `bench/benchmark_grid_api.c` retains a reproducible timing harness,
  not machine-specific timing output.
- Size-tagged descriptors have fixed `*_V1_SIZE` baseline prefixes, ignored
  input suffixes and untouched output suffixes. Future fields must default when
  absent; future libraries must not reject the 1.x baseline using a new sizeof.
- CMake consistently uses a 3.16-compatible top-level check, defaults auxiliary
  targets OFF when embedded, and supplies build-tree `miniexpr::core` and
  `miniexpr::artifact` aliases. The Makefile delegates to CMake; stale API/fixture
  documentation is corrected. A checked-in embedded consumer tests the defaults.

Qualification: 375 native tests, eight focused ASan/UBSan tests, four Emscripten
tests (interpreter configuration), and nine SLEEF/bundled-TCC checks pass.
Python's Menudet/persistence/table/graph selection passes 1503 tests; the real
C-Blosc2 constructor suite also passes against the rebuilt JSON-free library.
Embedded core-only and artifact consumers and an installed C99 core-only
consumer are exercised separately.

Still separate work: migrate/retire the checked-profile corpus and live full-DSL
callers; extract streaming pairwise sums; consolidate legacy ND geometry; settle
per-call interpreter override; qualify Linux/Windows and WASM JIT; complete C
`b2o` cross-language fixtures. Formula reductions remain deliberately deferred.
No ABI freeze or Windows linker-workaround removal is implied by this follow-up.

## Review 2 follow-up

- Close the tail-padding ambiguity: `me_signature` and `me_prepare_options`
  have explicit zero-initialized reserved bytes, and their fixed v1 prefix
  includes that storage. Every new field must follow the entire previously
  released layout; reserved bytes are not new option slots. All size-tagged
  descriptor baseline sizes equal their current full sizes on qualified ABIs.
- An independent `tests/abi/kernel-v1.c` caller deliberately avoids current
  headers. Its bytes are checked against flat hypothetical extended layouts
  (not nested-prefix structs, which could hide the bug), including absent
  option defaults and actual preparation with the frozen representation.
- Formula evaluation pushes/pops the accuracy already retained in
  `formula_math.ulp`; no redundant policy field or new environment variable is
  introduced. A deterministic private dispatch probe tests conflicting ambient
  accuracy, successful/error returns and shared-kernel concurrent evaluation,
  including builds without SLEEF.
- JSON imports only enter the deferred backend/FP guard when the resolved JIT
  request is ON. Typed discovery still retains its own strict FP guard. The
  compiler-invocation probe also covers a valid explicitly JIT-OFF import.
- The grid benchmark was already discovered by the benchmark source glob.
  `CONFIGURE_DEPENDS` now refreshes that discovery when files are added, without
  registering another target; a fresh benchmark-enabled build exercises it.

These are pre-freeze corrections, not an ABI declaration for the alpha pilots.

Requalification passes the same 375 native tests, eight focused ASan/UBSan
tests, four WASM interpreter checks and nine SLEEF/bundled-TCC checks. The
independent caller/layout regression runs on both native 64-bit and WASM32.
The downstream selection again passes 1503 Python tests and the C-Blosc2
constructor suite; the benchmark builds/runs via its CMake target.
