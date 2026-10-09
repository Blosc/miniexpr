# Logical arrays and native graph deployment (M5/M6)

The opt-in logical-array ABI (`ME_ARTIFACT_ARRAY_VERSION=1`) executes numeric,
elementwise, rank-zero portable 1.1 artifacts over a logical domain. It does not
reinterpret explicit block-scalar artifacts or ND-context partitions as logical
array reductions. Existing artifact language/schema semantics are unchanged.

## Descriptors, bounds and ownership

`me_array_view` records dtype, allocation base/capacity, byte offset, rank (0–16),
signed byte strides, shape, and native/little/big byte order. The allocation is
caller-owned and remains alive throughout execution. Bounds validation checks
the complete reachable byte interval with overflow-safe signed-stride arithmetic;
unaligned operands use bounded memcpy loads. Runtime bindings are by name and
must match immutable artifact dtype signatures. Arrays may broadcast by aligning
trailing axes: matching extents or singleton inputs are accepted, including 0-D
scalars, zero-sized domains and singleton-to-zero broadcasting. The host supplies
the logical domain; native validation rejects incompatible views before execution.

Output is caller-owned, aligned, native-endian C-order storage. Query its shape
and dtype with `me_array_result_shape()` before allocation. Capacity is checked.
Input allocations must be disjoint from output; in-place execution and overlapping
views are rejected even where particular element addresses do not coincide. As
with the descriptor API, caller metadata/status/error structures must not alias
array storage. Errors during execution leave output unspecified; bounds, options
and signature errors reject before numerical writes. Initial/mask value contents
are host-owned data; a malformed mask value can fail during traversal.

The native iterator is zero-copy for aligned, native-endian, matching C-contiguous
inputs when traversal is contiguous (elementwise and suffix-axis reduction groups).
Other supported layouts use **bounded per-tile gathers**, not normalized full-array
copies. This includes F-order, transposed, stepped, negative-stride, unaligned,
byte-swapped and broadcast views. `tile_items` defaults to 1024 and may not exceed
the native signed-32-bit lane limit. Allocation failure never switches backends.

`me_array_report` records peak iterator scratch (excluding output and interpreter
internals), cumulative gathered bytes, input zero-copy tile count, evaluated tile
count and aggregated floating status. These are measurements of this iterator,
not a comprehensive interpreter allocation bound. Host copying/decompression must
be measured separately; the Python adapter reports its owner-normalization bytes.
Views into a known owning NumPy allocation preserve their layout. Unsupported
external buffer owners use a copying adapter and report the copy, rather than
claiming zero-copy. Fictitious `as_strided` views outside their allocation reject.

## Reduction contract

Options select `sum`, `prod`, `min`, `max`, `any`, `all`, or elementwise execution.
Axis lists normalize negative indices and reject duplicates/out-of-range indices;
`naxes=-1` means all axes and zero means no axes. `keepdims` retains selected axes
as singleton dimensions. A broadcast Boolean `where` mask controls participation
before expression evaluation; masked lanes do not raise numerical flags.

Sum/prod widen Boolean/signed integer inputs to int64 and unsigned integer inputs
to uint64. Floating dtypes remain float32/float64. Any/all return bool; min/max
preserve expression output dtype. Explicit numeric accumulator dtypes are accepted
except floating-to-integer accumulation (currently unsupported); truth reductions
do not accept dtype/initial overrides. Initial is one native-endian scalar of the
selected accumulator dtype and participates once per group, not once per tile.
Empty sum/prod/any/all identities are 0/1/false/true. Empty/masked min/max require
initial. Integer accumulation and combination use fixed-width modular arithmetic,
including explicitly narrow accumulators, never signed-C overflow.

Each group visits its selected-axis coordinates in serial logical C order.
**Tile size and compressed storage chunking do not change grouping or order.**
Float32 rounds each accumulator step to float32; no unconstrained fast-math or
reassociation is allowed. This is not NumPy's layout/platform-dependent pairwise
sum, so universal bitwise floating-reduction parity is not promised. For finite
sum without overflow/underflow, use the standard forward-error criterion
`gamma_n * sum(abs(x))`, with `gamma_n = n*u/(1-n*u)` and unit roundoff `u`, plus
the initial term. Seeded finite tests check this against an independent `math.fsum`
reference and also require tile-invariant bytes. Nonfinite classifications and
extrema follow the existing M4 contract, including deterministic signed-zero ties.
Products may overflow/underflow in serial order even where another grouping would
not; no global relative-error promise is made near zero or overflow.

The logical scheduler is serial. Floating flags aggregate through scoped native
guards and caller fenv restoration; reporting is capability-qualified on WASM.
Independent calls/buffers on a shared immutable artifact may run concurrently;
same-output or same-storage mutation is not made safe automatically.

Mean/variance/std, arg/cumulative reductions, arbitrary gathers/scatters, tuple
results and mutable-view semantics are deferred, not implied by this ABI.

## Opt-in portable host JIT

`ME_JIT_ON` at 1.1 artifact load enables fail-closed typed-tree lowering for a
elementwise program, with or without ND context: float32/float64 arithmetic,
comparisons, Boolean selection, lazy `where`, straight-line locals and simple
`if`/`elif`/`else` branches with returns. Same-dtype signed and unsigned integer
`+`, `-`, `*` and negation use unsigned modular arithmetic and bit-copy results,
not overflowing signed C operations. Integral widening/narrowing conversions
lower to the same modular bit-copy; float-to-integer conversion stays interpreted
because it must report out-of-range/nonfinite values as errors. Block reductions
(`sum`, `min`, ...) remain on the interpreter.

Unary and binary float functions, the float `//`/`%`/`**` operators, the boolean
predicates (`isfinite`, `isinf`, `isnan`, `signbit`) and the integer
`%`/`//`/`<<`/`>>`/`&`/`|`/`^`/`~` operators reuse the authoritative native math
and checked-integer evaluator through private opaque-node bridges. The bridge
supplies already-evaluated operands, so promotion, libm choice, NaN rules, signed
divmod zero/sentinel handling, oversized shifts and IEEE exceptions are identical
to the interpreter. Check `me_artifact_has_jit()`;
an acceleration request alone is not evidence of compiled execution. Default
requests and checked 1.0 retain interpreter routing. TCC is the first backend;
`ME_DSL_JIT_COMPILER=cc` with `CC` selects GCC or another system compiler.

The generated loop uses a private participating-mask and separate
comparison/math/predicate/integer-operator bridge vectors, each a dispatcher
pointer followed by borrowed typed nodes. It preserves per-node dtype and final
conversion, avoids folding constant operations whose exceptions must be observed,
and scopes/restores fenv.
The comparison bridge preserves host NaN exception behavior; ordinary non-NaN
comparisons stay inline in generated code. For ND context the kernel recomputes
each lane's `_iN`/`_nN`/`_ndim`/`_flat_idx` from the logical shape/origin/extent
and applies the interpreter's range validation in-kernel; masked lanes skip the
preamble, and the interpreter's index buffers are not built on this path. The
math/operator bridges receive
already-evaluated operands, so a cheap per-lane call cannot diverge from the
interpreter's rounding or diagnostics. Volatile lane-local stores preserve
assignment precision and exceptions even for unused values. A bounded recursive
statement lowering rechecks definite assignment and rejects unsupported statements. Cache
identity includes semantic profile and lowering/ABI revision; runtime pointers
are supplied at invocation and are never serialized in generated code.

Integer division/shifts, mixed-width integer conversions, signed/unsigned
comparisons without explicit common promotion, floating floor division, other
functions, loops, block-scalar returns and ND context fall back to the portable
interpreter. Arbitrary compiler flags/options reject this route. WASM host-pointer
lowering is deliberately disabled. No explicit SIMD or cross-platform numerical
qualification is implied by local host tests. Logical reductions still use the
unchanged serial M5 accumulator; only their eligible map tiles are accelerated.

For exhaustive host conformance tests, the cc corpus runner bulk-compiles all
eligible generated kernels in one module, avoiding a compiler/linker process per
case. Each function has a unique symbol and retains its artifact's runtime
bindings; all corpus cases and value/status/recovery checks still execute.
This is test-only, not a production cache/compiler change. Set
`MENUDET_JIT_SERIAL=1` to exercise the original per-artifact compilation route.
Dedicated runtime/cache tests also retain that route. Host CTest includes full
arithmetic and function cc corpora with actual compilation required and a
60-second timeout.

## Shape operations

Native `me_array_reshape`, `me_array_transpose`, and `me_array_slice` create borrowed
metadata views with checked bounds. Reshape requires equal element count and C
contiguity; transpose validates a full axis permutation; slice takes a normalized
start, count and nonzero step. Negative-axis and negative-step views are supported.
These APIs do not allocate or transfer ownership. The evaluation result always
owns independent storage at the host level: NumPy returned-view alias guarantees
are not promised. Python can use normal array reshape/transpose/basic slices as
metadata before passing the resulting view into native evaluation.

## Deployment / graph boundary

`tests/numpy-compat/arrays.c` is a standalone C host that loads an artifact,
broadcasts, reduces axes, transforms views, varies tile sizes, checks bounds and
verifies modular combination. It has no Python, NumPy, NumExpr or compressed
storage dependency. CTest runs it in native and standalone Node/WASM builds.

Python's `PortableKernel.evaluate_array` wraps this scheduler.
Graph-enabled builds also provide [native metadata-only preparation](native-graphs-1.md)
through `miniexpr_graph.h`, independently of the artifact ABI described here.
`LazyExpr.compute(_require_native=True)` lowers an eligible safe numerical graph
to a declarative native graph with a fused portable 1.1 map and optional root logical reduction.
The frontend adapts syntax/owners and performs compressed storage reads; native code
owns numerical validation, type/shape preparation, values, casts, masks, traversal and reduction. Supported direct
operands are NumPy/Blosc2 arrays and plain/typed numeric scalars. Unsupported
graph capabilities reject before destination writes; no NumExpr or Python
numerical fallback is allowed. A bounded 128-entry immutable plan cache keys
canonical declarative graph, semantic profile, signatures and scalar captures, not input
 array identities or evaluated values. Mutation of inputs never reuses results.

`LazyExpr.compute(_require_native=True, jit=True)` explicitly requests this JIT
subset without changing default backend selection. Its execution report names
`portable-jit` only when the plan has a compiled kernel; unsupported plans retain
native interpretation. The same bounded graph-plan cache includes backend/compiler
configuration for JIT requests, never numerical input values.

Basic elementwise partial reads select operand views/storage before native
execution. Nested lazy/proxy/remote/table operands, table row/partition filtering,
 ordering, output aliases, reduction partial reads, custom backend/accuracy
overrides and unsupported functions are not eligible. Existing table-specific
partition semantics and safe/full persistence policies remain untouched.

`LazyExpr.native_kernel()` exports eligible **elementwise** artifacts. Their
existing portable lazy recipes retain native provenance validation and reject
unsupported kernels before creating carriers. Logical reduction options are a
host scheduler descriptor, not silently added to persisted block-scalar recipes;
deploy them explicitly with the logical-array C API.

NumExpr-unavailable subprocess tests cover imports, safe construction/metadata,
native-required execution/reduction, artifact import/export and persisted portable
recipe reload. NumExpr's packaging requirement remains unchanged: relaxing it for
all construction/persistence routes requires a wider clean-install audit.
