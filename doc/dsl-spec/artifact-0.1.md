# Menudet artifact 0.1 beta — specification

Status: **beta**. The schema is expected to remain relatively stable, but may
change based on feedback before a 1.0 compatibility contract. The only accepted
schema and language version is `"0.1"`; the language name is `"menudet"`.

Retain standalone UTF-8 JSON, explicit language/schema versions, named typed
signatures, validated numerical policy, entry point, required capabilities and
exact typed constant encodings. Reject duplicate/unknown required fields, unavailable
capabilities, invalid types, malformed constants and incompatible source.

Add fixed-width string descriptors, explicit result cardinality, and context
requirements. Bytes constants use hex bytes; Unicode constants use canonical
fixed-endian 32-bit code-unit hex rather than assuming slots are valid UTF-8.
Native buffers are host-native/aligned; host adapters handle endian/stride
conversions. Keep range-checked decimal integer strings, IEEE floating hex and
JSON Boolean values for scalar constants.

The extended evaluation call supplies named inputs with dtype/width/capacity,
logical rank/shape, block origin/extent/valid lanes, and output capacity. Check
products, lengths, shapes, overflow, ownership and unsupported overlap before
execution. Distinguish empty elementwise calls from explicit empty scalar-block
calls. A scalar block returns one item; an elementwise block returns its valid
logical extent. Independent calls must not mutate a shared artifact handle.

Use the descriptor-based native API; `me_artifact_eval()` remains a rank-zero
numeric elementwise call adapter, not another wire format. Expose queries for result cardinality, output width,
required context and capabilities. Loading validates/compiles without executing;
context validation occurs when binding/evaluating. Unsupported optional JIT
falls back before execution; required-JIT reports failure explicitly.

Separate standalone kernel requirements from bound container execution metadata.
A persisted host record stores operand references, declared logical domain and
reduction partition (block-grid extents, origin, C-order traversal). Standalone
C consumers may supply those bindings/context directly without understanding
every Python container protocol. Neither artifact contains Python runtime
objects, machine code, compiler caches or implicit callback registrations.

Native descriptor execution and Python persistence share the same widths,
cardinality, ND coordinates, partial-read grouping and context requirements.

## Staged descriptor ABI

`me_artifact_eval_ex()` now accepts a versioned, size-tagged
`me_artifact_eval_descriptor` and named `me_artifact_buffer` inputs with dtype,
itemsize, and byte capacity. It checks required byte products, native alignment,
address-range overflow, and input/output overlap before dispatching. Cardinality
and input/output itemsize queries are available. The original `me_artifact_eval()`
ABI is unchanged.

Menudet 0.1 beta supports masks, single-item scalar outputs, empty reductions,
fixed strings and descriptor-based logical coordinates. The typed interpreter
defines semantics; eligible kernels may also use JIT. Unsupported JIT lowering
selects interpretation before execution unless JIT is required.

## Wire fields

Required top-level keys are `schema_version: "0.1"`,
`language: {"name":"menudet","version":"0.1"}`, `requires`, `source`,
`entry_point`, `inputs`, `constants`, `output`, `semantics` (with
`numeric: "numpy-2.5"` and `casting: "unsafe"|"same_kind"|"safe"`), and
`context: {"ndim": rank}`. Only object `metadata` is optional. Duplicate or
unknown fields reject. Rank is 0 without ND, otherwise 1 through the native limit.
Metadata is descriptive only; it grants no capabilities.

`semantics` may additionally declare an explicit math policy:
`fp: "strict"|"contract"|"fast"`, `math: "auto"|"system"|"sleef"|"accelerate"`,
and `ulp: "u10"|"u35"` (SLEEF only). Omitted fields request inheritance from the
preparation environment (`ME_DSL_FP_MODE`, `ME_SIMD_MATH_BACKEND`); an exporter
must not bake a machine-specific environment choice into an artifact. Artifacts
using any non-default math policy, including non-strict `fp`, must declare the
`flexible-math` capability; readers without that capability reject the artifact.
Explicit `fp: "strict"` remains strict. An unavailable explicit `sleef` or
`accelerate` backend fails at load/preparation instead of falling back.
Source pragmas are validated against the declared policy; conflicting explicit
choices reject.

Inputs are `{name,dtype}` for numeric types, adding byte `itemsize` for `bytes`
or `unicode32`. Constants add `encoding` and `value`: exact range-checked decimal
integer strings, IEEE hex floats, JSON Booleans, or `bytes-hex`/`unicode32be-hex`
with exactly twice `itemsize` lowercase hex digits. Unicode constants decode
canonical big-endian scalar code units into host-native aligned slots; surrogates
and values above U+10FFFF reject. All eleven numeric types are supported.

Output is `{dtype,contract}` with `elementwise` or `block_scalar`, adding
`itemsize` for strings. Native compilation derives cardinality/width and rejects
disagreements. Numeric constants declare `category: "weak"` or `"typed_scalar"`. Ordinary Python
captures are weak; explicit capture dtypes and NumPy scalars are strong typed
immutable uniform operands. Signatures/cardinality never change during evaluation.
Constant bindings retain scalar storage owned by the artifact. Both typed
interpretation and JIT read lane zero rather than materializing broadcast arrays;
concurrent evaluations borrow this immutable storage. Uniform local variables
still use their existing local storage and are not scalar external bindings.

Implemented requirements are `numeric` (mandatory), `control-flow`,
`block-reductions`, `fixed-strings`, `nd-context` and `flexible-math`. Names must
be unique/known; native tree validation rejects missing used capabilities. Unimplemented operations
and signatures reject explicitly. String indices accept typed integral expressions;
large indices clamp beyond the bounded subject extent before legacy index casts.
Dynamic substring length conservatively retains the subject width. `replace`
uses literal lengths when known and otherwise a checked worst-case width derived
from operand capacities; an empty needle errors in participating lanes.
String kernels reuse pinned Unicode 15.0.0 tables and ASCII mapping for bytes.

ND binding requires explicit logical shape, origin and extent. Extent product is
`nitems`; slots traverse in C order. `_iN` is origin plus slot coordinate, `_nN`
is logical shape, `_flat_idx` is the logical-domain row-major index. Products are
checked; padded out-of-domain slots must be invalid. Invalid slots neither execute
nor write. No physical-tile or implicit global context is synthesized.

Native queries expose schema, capabilities, context rank, cardinality and widths.
`me_artifact_eval` accepts numeric rank-zero elementwise handles; use
`me_artifact_eval_ex` for block scalars, strings and explicit ND context. Loading
never executes source. Optional JIT falls back before execution.
`me_artifact_load_inferred` (and its options variant) is the metadata-only
output-inference entry point: it accepts `output.dtype:"auto"`, resolves the
native result dtype under the declared input signatures and casting policy, and
returns a handle with concrete output dtype/width. Exporters must still persist
that concrete dtype; the auto spelling is an import request, not a new recipe.
Full beta certification and host persistence remain open. Requirement `jit-required` reports an unavailable-required-JIT diagnostic when
no eligible compiled backend is available.
## Targeted integer-power policy

NumPy-numeric artifacts may explicitly declare `semantics.integer_powers` as
`"multiply"`. This enables bounded constant-integer power chains for real
floating bases, independently of `semantics.fp`. Rounding and FP status may
differ from general power. Omission retains the default power policy (constant
squares use multiplication; other strict powers retain the math backend).
This optional field is preserved on export; unsupported values fail validation.
