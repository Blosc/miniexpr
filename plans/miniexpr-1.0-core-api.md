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

Local qualification (macOS arm64): the pre-change artifact-enabled baseline
passed 369 tests; the pilot passed all 370 tests, including the new core API
suite, with SLEEF/TCC disabled. Explicit host-CC required-JIT execution passed.
Five focused tests passed with ASan/UBSan and artifact support disabled.
The C-Blosc2 constructor suite passed with its existing ByteDelta plugin enabled,
using only the JSON-disabled core library. C99 warning-as-error syntax checking
and C++11 header inclusion also passed. No compiler warnings were reported.
Windows, Linux, WASM and bundled-TCC qualification remain outstanding.

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
