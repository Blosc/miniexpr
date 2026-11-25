# Direct kernel preparation (1.0 pilot)

The provisional `miniexpr_kernel.h` interface implements the proposed stable
core. It is not ABI-frozen: downstream migration and formula reduction
semantics remain to be qualified. Old entry points remain during migration.
No new dependencies are required.

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
Execution preference never wraps a formula as a different language. For this
pilot, formula reductions and ND context are unsupported pending the contract
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

Menudet source uses an explicit version. `"0.1"` preserves the unannotated
language; `"0.2"` enables proposed [annotations](dsl-spec/0.2.md).
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
- Preparation defaults to interpretation, strict FP and system math. Source
  pragmas can explicitly request another supported mathematical policy.
  Optional JIT falls back before execution; required JIT rejects unavailable
  lowering. Numerical failures never cause execution replay.

Blosc2 recipes, JSON import, array/graph scheduling, varlen output and physical
padded-grid adaptation remain separate. Existing v1 JSON artifact loading is
unchanged; its schema remains 0.1.
