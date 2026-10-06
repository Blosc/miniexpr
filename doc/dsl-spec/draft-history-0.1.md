# Historical draft of portable miniexpr DSL profile 0.1

**Historical only:** Superseded by [the frozen conservative profile](0.1.md).
The following records the broader development draft, not release membership.
**Original status:** Experimental draft; not yet a published compatibility promise.
The executable conformance corpus covers elementwise kernels with `bool`,
`int32`, `int64`, `float32`, and `float64`. The tested matrix below is narrower
than all operations accepted natively; additional combinations must pass the
semantic audit before entering this profile.

## Relationship to the full language

The [canonical native DSL reference](../dsl-syntax.md) remains the reference for
the complete language accepted by `me_compile()`. This document defines a smaller
portable profile, not a replacement parser or a Python dialect. The
[usage guide](../dsl-usage.md) documents native C execution and runtime controls.

Before publication, incorporated normative rules must be frozen here or pinned
to an immutable revision. Links to the current reference are drafting references,
not a promise that future changes alter profile 0.1.

## First executable subset

- One top-level named `def`, with unique plain positional parameter names.
- Comments and native header pragmas.
- Local assignment, `return`, and `if`/`elif`/`else`.
- Finite decimal floating-point literals, `+`, `-`, `*`, and comparisons.
- Native chained comparisons, evaluated left-to-right, with each operand
  evaluated at most once and later operands skipped after a failed comparison.
- `for ... in range(...)`, `while`, `break`, `continue`, and `+=` in bounded
  conformance examples.
- Explicit fixed-width input and output types, caller-owned equal-length buffers.

### Currently verified type/operation matrix

| Inputs | Output | Verified operations |
| --- | --- | --- |
| `float64` | `float64` | Arithmetic, branches, chains, bounded loops, identity |
| `float64` | `float64` | `sin` on the bounded sample interval `[-1, 1]` |
| `float32` | `float32` | Homogeneous addition |
| `int32` | `int32` | Homogeneous addition without overflow |
| `int64` | `int64` | Identity, including exact signed extrema |
| `int64` | `bool` | Homogeneous comparisons, including values above `2**53` |
| `bool` | `bool` | Boolean `and` and `not` |
| `float64` | `bool` | Boolean-result `and`, not Python operand-return semantics |
| `float64` | `int64` | `int(x)` truncates toward zero for small finite values |
| All five supported dtypes | All five supported dtypes | Explicit output conversion on representable boundary values; exact shared 5×5 matrix |

The [numeric stabilization audit](numeric-audit-0.1.md) records conversion domains
and blocking discrepancies. Seven bounded division cases now have typed JIT
lowering; unsupported division contexts conservatively retain interpreter
execution. Float32 math intermediates and nested cast semantics remain unresolved,
despite being accepted by the experimental feature filter. Validation is not a
published guarantee for them.

Integer transport and comparison must not pass through floating-point values.
No wraparound or overflow rule is promised yet; integer arithmetic fixtures keep
all intermediates representable. Mixed input types, out-of-range narrowing, integer division,
and out-of-range/non-finite casts remain audit work, not implicitly admitted
matrix entries. The cast samples do not establish behavior for arbitrary
float-to-integer conversions. The `sin` samples do not establish a global
transcendental accuracy bound.

Identity of float64 values preserves signed zero and the classification/sign of
infinities. NaN identity preserves NaN classification; no payload guarantee is
made. These identity rules do not specify every special-value arithmetic case.

Each element produces one output independently of other elements. Reductions,
even in conditions, are excluded from this elementwise profile. So are reserved
ND symbols, `print`, external functions, strings/bytes, and frontend-specific
syntax. These exclusions do not imply the full DSL lacks those features.

Native variable binding is by name. Compilation variable order may differ from
source parameter order; evaluation pointers follow compilation variable order.
An artifact may choose a canonical order without changing this native rule.

Local types are inferred from expressions, not the requested output dtype.
Incompatible local reassignment and inconsistent return dtypes are compile-time
errors. Missing return on an executed path, zero `range` step, and exceeding the
native `while` iteration cap are runtime errors. Incomplete return coverage may
compile for either the interpreter or a supported JIT backend. All-returning
inputs succeed; if any executed element reaches a missing path, evaluation reports
`ME_EVAL_ERR_INVALID_ARG`. This semantic JIT failure must not rerun the kernel in
the interpreter. Output contents after a failed evaluation are unspecified.
The portable cap policy remains to be specified before publication.

## Execution policy
### Boolean arithmetic

Boolean operands participate in numeric arithmetic as exact zero/one values.
Convert the completed numeric result to Boolean by nonzero truth only at a
requested Boolean output or an explicit `bool()` call. Intermediate arithmetic,
locals, and fractional literals must not be converted to Boolean prematurely.
For example, Boolean `x` gives true for `x + 0.5` at either input value, and
`x * 0.5` preserves the truth of `x`. `(x + x) == 2` is true for true `x`.

The currently certified Boolean-context slice uses int64 computation for
Boolean-only arithmetic and integral-valued literals, and float64 computation
for fractional literals. Signed overflow and arbitrary mixed-type promotion
remain outside the completed audit; this rule is not a general promotion table.

### Backend and host policy

The native interpreter is the conformance baseline. TCC/CC are accelerators,
not prerequisites for portable execution. A source `# me:compiler=tcc|cc`
selects the preferred compiler ahead of the local compiler default, not a
required language capability. Preserve best-effort fallback when required
semantics can be honored. Explicit `ME_JIT_OFF` still disables JIT.

The host's while-loop resource cap is execution policy, not an artifact constant.
`ME_DSL_WHILE_MAX_ITERS` defaults to 10,000,000 body entries per loop invocation;
positive values limit execution, nonpositive values disable the limit, and
invalid values use the configured default. Conditions run before the cap check;
lowered condition prefixes are not body iterations. Cap failures report a native
evaluation error without JIT-to-interpreter retry, with failed outputs unspecified.
The compiled JIT cap participates in its IR/cache identity; a changed host cap
selects interpreter execution under the new policy. Mixed-lane chained while
conditions are covered by shared interpreter/TCC/CC regression fixtures.

Floating-point pragmas express separate semantic requirements. Their complete
profile mapping and transcendental accuracy rules remain audit work; finite
arithmetic conformance fixtures currently use relative and absolute tolerances
of `1e-12` for float64 and `1e-6` for float32. Exact integer/Boolean comparison and
signed-zero/non-finite checks are separate from those tolerances. Agreement alone
is insufficient: compare to specified expected values.

## Conformance and remaining work

The initial shared corpus is in `tests/portable-dsl/`. The standalone native
runner and Python-Blosc2 consume the same source and expected values without
reconstructing or executing Python functions.

Before freezing 0.1, specify the complete dtype/function/operator matrix, numeric
promotion, overflow and conversions, non-finite behavior, Boolean rules, loop
limits, empty-buffer behavior, and error categories. Extend negative and boundary
coverage. Artifact versioning and typed constants will be specified separately
in `artifact-0.1.md` after raw-source conformance is established.

## Experimental native validation contract

`me_validate_portable_dsl(source, version, inputs, ninputs, output_dtype, error)`
checks draft profile membership and native compilation without executing the
kernel or invoking TCC/CC. It is independent of Python. The language version is
explicitly `"0.1"`, not the miniexpr package version. The public declarations and
`ME_PORTABLE_DSL_VERSION` feature macro are in `miniexpr.h`.

The current feature boundary is intentionally conservative:

- Explicit `bool`, `int32`, `int64`, `float32`, or `float64` input/output types.
  All inputs share a dtype; the output may differ. No automatic, complex,
  string/bytes, or mixed-input types.
- Input descriptors are signature metadata: `ME_VARIABLE`, NULL address/context,
  itemsize 0, and unique ASCII names. Source and descriptor orders may differ;
  every parameter must be bound exactly once. No pointers or callbacks are used.
- Native assignments, expression statements, returns, conditionals, loops,
  `break`/`continue`, `pass`, comments, docstrings, and existing literal conveniences.
- Unary `+`/`-`, `not`/`!`, `and`/`or`, comparisons and chains, addition,
  subtraction, multiplication, and division for floating-input signatures.
  Integer division, powers, remainder, bitwise operators, indexing, attributes,
  and string literals remain outside this initial filter.
- Calls to `sin`, `cos`, `int`, `float`, and `bool`; `range` only in a for header.
  No reductions in any position, `print`, ND symbols, external functions,
  implicit globals, or implicit zero-argument native builtins. Every bare name
  must be an input, local, loop variable, or Boolean operator.
- Finite numeric literals; non-floating-input signatures additionally limit
  literal magnitudes to `2**53`. Exact wider integers can be passed as typed
  inputs instead. This does not promise arbitrary-precision literal arithmetic.
- Only absent or explicitly `strict` FP pragmas are admitted during drafting.
  Compiler pragmas remain preferences and do not require that compiler to exist.

These feature checks are broader than the sample operation matrix above. Passing
validation is not proof that arbitrary input data satisfy runtime constraints:
integer intermediates must remain representable, casts must remain in their
defined finite range, divisors and loop bounds must be appropriate, locals must
be initialized before being read on an executed path, and executed paths must
return. It is not a JIT-support or global numerical-accuracy certificate.
The remaining semantic audit must precede freezing the profile.

Validation checks the source's declared FP mode, not the host's execution policy.
For execution without an FP pragma, callers must ensure the required mode rather
than assuming an environment default honors it. Artifact execution will need to
make that requirement explicit. No sandbox or general resource isolation is
provided; parsing/compilation still require trusted or appropriately isolated
inputs. The validator has bounded name/nesting bookkeeping, not a new sandbox.

### Results and diagnostics

Return values distinguish success, unsupported version, invalid signature, invalid
source/native compilation, unsupported feature, and allocation failure. Optional
`me_portable_error` supplies a message and 1-based source location where available;
zero locations denote signature/version or otherwise non-source diagnostics.
Locations after native normalization/lowering may identify an expression start
rather than the exact original token. Success clears the diagnostic fields.

The standalone runner applies validation before its existing compile/evaluate
checks. Runtime-error fixtures must validate successfully: potential runtime
errors are not automatically invalid programs. Compile-error fixtures must fail
validation and compilation separately.
