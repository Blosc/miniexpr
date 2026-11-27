# Direct kernel preparation (1.0 contract)

`miniexpr_kernel.h`, `miniexpr_grid.h` and `miniexpr_types.h` define the supported
1.0 embedding contract. `ME_KERNEL_API_VERSION` is 1. The library remains an
alpha release; a supported 1.x ABI begins at the 1.0 release, not at a 0.x or
alpha pilot. Classic/full-DSL entry points remain compatibility interfaces in
`miniexpr.h` through 1.x, preserving their existing routing rather than silently
reinterpreting source. New input signatures do not expose C callbacks.
No new dependencies are required. See [the API boundary audit](c-api-1.md).

The lifecycle is:

1. `me_prepare_formula` or `me_prepare_menudet` specializes source against
   concrete input types, snapshots captures, and returns an immutable kernel.
2. `me_describe` copies the final output dtype, byte itemsize, cardinality and
   required coordinate rank into a caller-owned, size-tagged descriptor.
3. `me_eval_block` executes on buffers in signature order, including unused
   declared inputs. Captures do not occupy runtime slots.
4. `me_kernel_free` releases the specialization.

Preparation copies retained metadata and captures, not runtime array addresses.
Evaluation is synchronous; independent calls may share a kernel. Never free
a kernel concurrently with its use.

### Descriptor compatibility

Initialize each size-tagged descriptor with its initializer. The documented
`*_V1_SIZE` prefix is the minimum accepted size throughout 1.x. On qualified
ABIs it equals the complete v1 descriptor size: signatures, preparation options
and run options end in explicit zero-initialized reserved bytes rather than unnamed
trailing padding. New fields must follow the entire v1 layout and must not reuse
its reserved bytes, otherwise old callers advertising `sizeof(v1)` could appear
to supply a new field. Every later released revision must likewise avoid unnamed
tail padding before advertising its size. Smaller descriptors reject without
accessing missing fields.
Larger input descriptors are accepted and unknown trailing bytes are ignored.
Output functions preserve `struct_size` and leave unknown trailing bytes
untouched. Any future appended fields must be read only when present, default
when absent, and retain these baseline minima rather than requiring a new
`sizeof`. This is not binary compatibility with the pre-1.0 APIs.

## Example

```c
#include "miniexpr_kernel.h"

me_input_spec inputs[] = {{"a", ME_FLOAT32, 0}, {"b", ME_FLOAT32, 0}};
me_signature signature = ME_SIGNATURE_INITIALIZER;
signature.inputs = inputs;
signature.ninputs = 2;

me_kernel *kernel = NULL;
me_error error;
me_status rc = me_prepare_formula("a + 2 * b", &signature, NULL, &kernel, &error);
if (rc != ME_SUCCESS) {
    /* error.status and error.message are available directly. */
    return rc;
}

me_kernel_info info = ME_KERNEL_INFO_INITIALIZER;
rc = me_describe(kernel, &info);
/* info: float32, sizeof(float), ME_ELEMENTWISE, no coordinate context. */

float a[] = {1, 2}, b[] = {3, 4}, result[2];
me_buffer buffers[] = {{a, sizeof(a)}, {b, sizeof(b)}};
me_block block = ME_BLOCK_INITIALIZER;
block.nitems = 2;
rc = me_eval_block(kernel, buffers, 2,
    (me_mut_buffer){result, sizeof(result)}, &block, NULL, &error);
me_kernel_free(kernel);
```

Formula evaluation uses typed numerical semantics, not the full-DSL route.
Execution preference never wraps a formula as a different language.
Formula reductions and ND context are unsupported pending the contract
discussion. Prefer-JIT may interpret; require-JIT fails for a formula until its
lowering is implemented.

## Captures and coordinates

For `def ramp(start, step): return start + _flat_idx * step` (with a normal
indented body), declare the constants as captures:

```c
double start = 1.25, step = 0.5;
me_capture captures[] = {
    {{"start", ME_FLOAT64, 0}, &start, ME_CAPTURE_TYPED},
    {{"step", ME_FLOAT64, 0}, &step, ME_CAPTURE_TYPED}
};
me_signature signature = ME_SIGNATURE_INITIALIZER;
signature.captures = captures;
signature.ncaptures = 2;
signature.context_ndim = 1;
```

There are zero runtime input slots. Evaluation supplies logical shape, origin
and extent. The compiler copies binary constants: no float printing, locale
conversion or JSON. Weak captures instead use `ME_CAPTURE_WEAK` and canonical
bool/int64/float64/complex128 representations. Scalar category affects promotion.

`me_describe().context_ndim` is the **required** coordinate rank, not necessarily
the rank supplied at preparation. Source without coordinate references reports
zero and must be evaluated with `block.ndim == 0`. The grid adapter supplies the
required context automatically. Legacy fixed JSON descriptors retain their
declared rank; the new core does not change that reader's contract.

Menudet source uses an explicit version. `"0.1"` preserves the unannotated
language; `"0.2"` enables [annotations](dsl-spec/0.2.md).
Unannotated inputs specialize to their bindings. Output dtype `ME_AUTO` infers;
a concrete result specification or return annotation converts only the final
result. Conflicting declarations reject. Casting policy checks the inferred
dtype before final conversion.

## Block and policy contracts

- Buffers are aligned, native endian, with byte capacities. Numeric itemsize
  can be zero in a signature; fixed-string itemsize is explicit bytes.
- Output is disjoint from inputs, mask and coordinate arrays. The host handles
  storage byte order, views and physical padding.
- A mask is one 0/1 byte per lane and does not compact output. Inactive output
  values are unspecified. A block-scalar result needs one item even for zero
  input lanes; an empty elementwise block needs none.
- Run options collect stable floating-status bits or raise selected flags.
  FP state is restored. WASM reports collection unavailable and rejects a
  nonempty raise mask before execution.
- `run.execution = ME_RUN_INTERPRET` selects the interpreter for that call,
  including a handle prepared with `ME_EXEC_REQUIRE_JIT`. It never recompiles,
  mutates a shared handle, or retries after execution. `ME_RUN_PREPARED` (zero)
  or NULL run options use the prepared backend. `execution` is in the v1 prefix.
- Preparation defaults to interpretation, strict FP and system math. Source
  pragmas can explicitly request another supported mathematical policy.
  `inherit_environment` is an explicit preparation opt-in for hosts that own
  process/scoped defaults. `integer_powers` in the signature preserves an
  authored multiply-chain policy; it is not inferred from input types.
  Formulas also snapshot the inherited math backend at preparation; later
  environment changes do not change their dispatch. Their interpreter may
  retain strict arithmetic even when relaxed FP optimizations are permitted.
  Evaluation scopes SIMD accuracy to the prepared policy and restores the caller's
  thread-local accuracy on success or failure. Backend inheritance uses
  `ME_SIMD_MATH_BACKEND`; this does not add a ULP environment variable.
  Optional JIT falls back before execution; required JIT rejects unavailable
  lowering. Numerical failures never cause execution replay.
- The optional v3 preparation-options suffix supplies typed `math_backend`
  (`DEFAULT`, `AUTO`, `SYSTEM`, `SLEEF`, `ACCELERATE`, `SCALAR`) and
  `jit_compiler` (`DEFAULT`, `TCC`, `CC`). Zero/default values preserve earlier
  behavior. Explicit selections override environment defaults; authored
  Menudet pragmas retain precedence. Invalid enum values are rejected before
  compilation. Older option prefixes omit these selections without reading the
  suffix. Formula compilation remains interpreted regardless of compiler policy.
- `me_kernel_has_jit` reports prepared availability, not actual execution.
  `me_eval_block_backend` additionally reports `NONE`, `INTERPRETER`, or `JIT`
  per invocation, including failed evaluations. Preflight rejection and empty
  elementwise blocks report `NONE`; forced interpretation and empty scalar
  control-flow fallbacks report `INTERPRETER`. No shared kernel state is mutated.
- A host may set `menudet_forbidden_features` to reject compiled control flow,
  block reductions or fixed strings before JIT. This is an optional
  preparation-options suffix; v1 sizes and a zero mask impose no restriction.
  These flags describe typed source, not container or JSON capability numbers.
- Invalid preparation/run options report `ME_ERR_INVALID_ARG`. Input-count,
  annotation, context-coverage and result-contract mismatches report
  `ME_ERR_SIGNATURE`; source-language errors remain `ME_ERR_SOURCE`.
  Contract checks precede runtime compiler/cache activity, including when JIT
  is required. Weak integer/Boolean capture specialization uses the capture's
  post-annotation scalar category.

Blosc2 recipes, JSON import, array/graph scheduling, varlen output and physical
padded-grid adaptation remain separate. Existing v1 JSON artifact loading is
unchanged; its schema remains 0.1.

## Padded grids

`miniexpr_grid.h` supplies an independent geometry adapter: create an immutable
grid, describe a physical `(nchunk, nblock)`, evaluate it, then free the grid.
Geometry and byte-capacity checks precede numerical execution. Partial blocks
gather live lanes in logical C order and scatter elementwise results back into
zero-padded physical output. Scalar kernels evaluate the entire live group once,
including all its rows. The adapter never implements a reduction by summing row
results. Rank-zero geometry denotes one scalar physical block; empty logical
domains contain no chunks.

Padded copies use contiguous runs and one invocation-local, alignment-checked
scratch allocation. Empty elementwise blocks validate physical buffers and run
options, then zero output without executing a kernel. Empty scalar groups still
execute their defined reduction. Both block and grid calls initialize FP reports
before geometry/buffer preflight, so such failures cannot leave stale flags.

The C-Blosc2 constructor branch uses this shared adapter while retaining host
ownership of compression, exact endpoints and endian conversion. Legacy ND
block evaluators (`me_eval_nd` / `me_nd_valid_nitems`) share the same underlying
window/run helpers, and `miniexpr_reduce.h` provides streaming pairwise sums
(`me_pairwise_sum_*`) without requiring the JSON adapter. A standalone
core-only installation installs `miniexpr.h`, `miniexpr_types.h`, `miniexpr_kernel.h`,
`miniexpr_grid.h` and `miniexpr_reduce.h` without JSON support.

## Build integration

Build-tree consumers can link `miniexpr::core` or `miniexpr::artifact` (the latter
includes the former transitively). Standalone native installs expose the same
targets with `find_package(miniexpr CONFIG REQUIRED COMPONENTS core)`. Add the
`artifact` component if the consumer needs the optional JSON adapter.
Embedded builds default tests, examples and
benchmarks OFF; explicit cache settings still win. Standalone builds retain
their previous defaults. The Makefile is now a thin CMake shortcut rather than
a separate, incomplete source list.

The embedded consumer fixture checks the defaults and exercises the grid and
pairwise-sum APIs:

```sh
cmake -S tests/cmake/embedded -B build-embedded \
  -DMINIEXPR_SOURCE="$PWD" -DMINIEXPR_USE_SLEEF=OFF \
  -DMINIEXPR_ENABLE_TCC_JIT=OFF
cmake --build build-embedded
ctest --test-dir build-embedded --output-on-failure
```

Repeat with `-DMINIEXPR_BUILD_ARTIFACT=ON` to exercise the transitive adapter
target. `bench/benchmark_grid_api.c` measures full and partial blocks with
preparation excluded; retain timing output outside the source tree.
