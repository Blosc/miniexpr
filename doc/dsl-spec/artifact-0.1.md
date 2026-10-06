# Portable miniexpr artifact 0.1 (experimental)

This is the candidate standalone interchange format for the draft
[portable language profile 0.1](0.1.md), not a package version or a Blosc2 storage
format. It is implemented by the optional native artifact adapter. The language
semantics must still pass their remaining audit before either contract is frozen.

## JSON representation

An artifact is one UTF-8 JSON object. Strict JSON syntax is required: no comments,
trailing commas, BOM, NaN numeric tokens, invalid UTF-8, or trailing non-whitespace
data. Escaped Unicode is accepted and decoded normally. Duplicate object keys
(including escaped-equivalent keys), raw or decoded NUL characters, and duplicate
capability/binding names are rejected. Member order and insignificant whitespace
are irrelevant. All fields below are required except `metadata`.

| Field | Required value/structure |
| --- | --- |
| `schema_version` | String `"0.1"` |
| `language` | `{"name": "miniexpr", "version": "0.1"}` |
| `requires` | Array containing `"core-scalar"` exactly once; no other capabilities in this version |
| `source` | Complete native source for one kernel, with retained compiler/FP pragmas |
| `entry_point` | Name of that kernel |
| `inputs` | Array of `{"name": "…", "dtype": "…"}` runtime bindings |
| `constants` | Array of named typed scalar bindings described below; empty is allowed |
| `output` | `{"dtype": "…", "contract": "scalar-per-element"}` |
| `semantics` | `{"fp": "strict"}` |
| `metadata` | Optional informational JSON object; never interpreted as execution configuration |

Unknown fields outside `metadata` are rejected, including unknown nested semantic
fields. `metadata` may contain arbitrary JSON values subject to the same structural
limits. Only the explicitly supported schema, language, capabilities, execution
contract, and FP requirements are accepted; no forward-compatibility inference.

The native implementation accepts at most 1 MiB of JSON, nesting depth 32, and
128 members/elements in each object/array. The combined number of inputs and
constants is at most `ME_MAX_VARS` (128). These are loader limits, not a sandbox.

## Signature and constant coverage

Dtypes are logical `bool`, `int32`, `int64`, `float32`, and `float64` values. The
draft portable profile requires all parameters, including constants, to share a
dtype; output can differ. Mixed input/constant types remain excluded for now.

Every source parameter must occur exactly once in either `inputs` or `constants`.
Overlaps, duplicate names, missing bindings, and unused bindings are errors.
`inputs` lists source parameters in declaration order with constants omitted.
Constants may be listed in any order. Name binding must not depend on the position
of a constant in the source or manifest. Names obey the portable profile.

A constant has exactly four fields: `name`, `dtype`, `encoding`, and `value`:

| Dtype | Encoding | Value |
| --- | --- | --- |
| `bool` | `boolean` | JSON `true` or `false` |
| `int32`, `int64` | `decimal` | Decimal string, range checked for the declared signed width |
| `float32` | `ieee754-hex` | Eight lowercase hexadecimal digits, no prefix |
| `float64` | `ieee754-hex` | Sixteen lowercase hexadecimal digits, no prefix |

Integers use canonical decimal notation: `0` or an optional minus followed by a
nonzero digit and remaining decimal digits; no plus, whitespace, leading zeros,
decimal points, exponent, or negative zero. JSON numeric values are not accepted
as integer constants, even when exactly representable in a particular parser.

Float hex strings spell IEEE-754 binary32/binary64 bits most-significant nibble
first, independent of host byte order. For example, float64 `2.0` is
`4000000000000000` and `-0.0` is `8000000000000000`. All bit patterns are
transportable, including infinities and NaNs; subsequent arithmetic need not
preserve a NaN payload or signaling state. The adapter requires a compatible
IEEE-754 host and decodes through integer bit patterns rather than JSON numbers.

## Semantics versus backend preferences

`semantics.fp` is a requirement. Source with a conflicting FP pragma is rejected.
When source omits an FP pragma, the adapter compiles an internal copy prefixed
with `# me:fp=strict`; the stored/interchange source remains unchanged. This makes
artifact execution independent of `ME_DSL_FP_MODE` defaults. No process environment
is changed. A `# me:compiler=tcc|cc` pragma survives and retains normal precedence
over compiler defaults. Lack of that backend does not invalidate an artifact;
normal best-effort interpreter fallback remains available. Host `ME_JIT_OFF`
can independently disable JIT preparation/execution.

`core-scalar` denotes the draft language profile, not a backend or hardware
capability. Input-range, overflow, initialized-local, conversion, loop, divisor,
and return-path constraints remain those of the language. The loader validates
source membership and typed compilation; it neither executes the kernel at load
time nor certifies arbitrary runtime values.

## Logical buffers, ownership, and execution

The C adapter is `miniexpr_artifact`, enabled with
`-DMINIEXPR_BUILD_ARTIFACT=ON`. It uses pinned yyjson only in this separate adapter;
the raw compiler remains usable without it. Public API: `miniexpr_artifact.h`.

`me_artifact_load()` copies source, names, and decoded constants into an owned
opaque handle and compiles it. The caller can immediately release the JSON input.
Query functions return borrowed strings, valid until `me_artifact_free()`, which
releases all owned resources and is NULL-safe. Metadata is discarded after load.

`me_artifact_eval()` accepts named runtime inputs in any order, each with the exact
declared dtype and common element count. No missing/extra/duplicate inputs are
allowed. It writes one output value per element; constant-only kernels are allowed
and use the host-provided output count. Empty buffers still validate bindings but
do not execute source or trigger potential runtime errors. Data/output pointers
may be NULL only for empty buffers.

Buffers are caller-owned, sufficiently sized, naturally aligned, contiguous
host-native values. Output must not overlap input buffers. Strides, endian/storage
conversion, and object ownership belong to host adapters, not the artifact. Caller
buffers must remain alive through evaluation. Constants are immutable and broadcast
in bounded per-call workspace, with native evaluation in tiles of at most 256
elements, never array-sized constant copies. Profile kernels have no cross-element
reductions, indexing, or ND symbols; tile boundaries have no observable successful
result semantics. Independent evaluations do not mutate the handle. Freeing a
handle concurrently with evaluation is invalid.

Failure categories distinguish malformed artifacts, unsupported requirements,
invalid source, binding errors, evaluation errors, and allocation failure. Native
status/source diagnostics are included when available; source locations may point
to expression starts rather than precise tokens. Output after failed evaluation is
unspecified, including elements written before a later tile failed. The API never
executes Python, loads artifact-specified libraries, or treats metadata as shell
configuration. JIT and general host execution still require appropriate trust and
resource isolation; the loader is not a security sandbox.

## Frozen candidate example

`tests/portable-artifacts/affine.json` is a hand-authored fixture. It captures
float64 scale `2.0` and bias `-1.0`; input `[0, 1, 2, 3]` must produce
`[-1, 1, 3, 5]`. Its constants intentionally differ from source parameter order.
It requires no originating Python module, preprocessing, or Python runtime.
