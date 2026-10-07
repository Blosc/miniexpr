# Portable artifact 1.0 — implementation draft

Status: implemented staged wire schema, **implementation draft, not certified**.
Final compatibility guarantees await complete release gates.

Retain standalone UTF-8 JSON, explicit language/schema versions, named typed
signatures, strict semantics, entry point, required capabilities and exact typed
constant encodings. Reject duplicate/unknown required fields, unavailable
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

Do not finalize schema version 1.0 until the native descriptor runner and Python
save/load round trips agree on widths, cardinality, ND coordinates, partial-read
grouping and missing-context diagnostics. During beta, identify release-candidate
artifacts explicitly; no final compatibility guarantee attaches to draft data.

## Staged descriptor ABI

`me_artifact_eval_ex()` now accepts a versioned, size-tagged
`me_artifact_eval_descriptor` and named `me_artifact_buffer` inputs with dtype,
itemsize, and byte capacity. It checks required byte products, native alignment,
address-range overflow, and input/output overlap before dispatching. Cardinality
and input/output itemsize queries are available. The original `me_artifact_eval()`
ABI is unchanged.

Draft 1.0 loads through interpreter-only profile dispatch,
with masks, single-item scalar outputs, empty reductions, fixed strings, and
descriptor-based logical coordinates. Certification and host persistence remain.

## Implemented draft wire fields

Required top-level keys are `schema_version: "1.0"`,
`language: {"name":"miniexpr","version":"1.0"}`, `requires`, `source`,
`entry_point`, `inputs`, `constants`, `output`, `semantics: {"fp":"strict"}`,
and `context: {"ndim": rank}`. Only object `metadata` is optional. Duplicate or
unknown fields reject. Rank is 0 without ND, otherwise 1 through the native limit.
Metadata may label records `implementation-draft`; it grants no capabilities.

Inputs are `{name,dtype}` for numeric types, adding byte `itemsize` for `bytes`
or `unicode32`. Constants add `encoding` and `value`: exact range-checked decimal
integer strings, IEEE hex floats, JSON Booleans, or `bytes-hex`/`unicode32be-hex`
with exactly twice `itemsize` lowercase hex digits. Unicode constants decode
canonical big-endian scalar code units into host-native aligned slots; surrogates
and values above U+10FFFF reject. All eleven numeric types are supported.

Output is `{dtype,contract}` with `elementwise` or `block_scalar`, adding
`itemsize` for strings. Native compilation derives cardinality/width and rejects
disagreements. Constants are strong typed immutable uniform operands, not weak
source literals. Signatures/cardinality never change during evaluation.

Implemented requirements are `numeric` (mandatory), `control-flow`,
`block-reductions`, `fixed-strings`, and `nd-context`. Names must be unique/known;
native tree validation rejects missing used capabilities. Unimplemented operations
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
`me_artifact_eval` rejects draft 1.0 handles: use `me_artifact_eval_ex`. Loading
never executes source. Optional JIT falls back before execution. Full beta
certification and host persistence remain open. Requirement `jit-required`
explicitly rejects with an unavailable-required-JIT diagnostic in this draft.
