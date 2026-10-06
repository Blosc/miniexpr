# Numeric stabilization audit for portable 0.1

**Status: incomplete; blocks publication.** These are checked conversions and
open discrepancies, not a frozen specification. The experimental validator
accepts more combinations than the audited matrix; validation alone does not
establish interpreter/JIT agreement.

## Output conversions

The shared `identity.dsl` and 25 `convert_INPUT_OUTPUT.txt` fixtures cover all
pairs of bool, int32, int64, float32, and float64 input/output types. Expected
results are independently specified. The `ok_exact` outcome requires exact finite
results, signed-zero agreement, and non-finite classification/sign agreement
(not NaN payload preservation).

- Boolean values convert to exact 0/1.
- Integer-to-Boolean conversion tests nonzero truth; integer identity/widening
  and representable narrowing are exact.
- Integer-to-float samples exercise nearest-even rounding, including int64
  values above `2**53`, under the tested default rounding mode.
- Floating-to-Boolean conversion tests nonzero truth, including NaN/infinities;
  either signed zero is false.
- Finite, representable floating-to-integer samples truncate toward zero.
- Floating conversions test exact widening, rounded narrowing, extrema,
  subnormals, signed zero, infinities, and NaNs.

Out-of-range narrowing, non-finite/out-of-range float-to-int casts, and signed
arithmetic overflow remain unaudited. No wraparound or error rule is inferred.
The native fixture reader accepts finite subnormal decoding with `ERANGE`, but
still rejects overflow. Exact comparisons prevent tolerances hiding a flushed
subnormal or incorrect rounding.

For non-floating signatures, normalized integer literal digits are now checked
exactly against `2**53`. A double-only comparison previously let `2**53 + 1`
through after rounding down. Regression tests include decimal, underscore, and
base-prefixed spellings. Typed inputs/constants can transport wider integers.

## Blocking expression gaps

The fixtures in `tests/portable-dsl/audit/` are reproducers, not passing JIT
conformance claims. Do not standardize their current backend discrepancies.

### Division promotion: first correction implemented

With floating input, `int(x) / 2` at `x = 1.25` yields `0.5` in the interpreter
but previously yielded `0` in both TCC and CC. Generated C retained integer operands and performed
integer division before the output cast. `bool(x) / 2` and literal-only `1 / 2`
showed the same issue.

Division-containing arithmetic now uses owned C text rendered from the actual
compiled interpreter tree, not source-token types or the output dtype alone.
It preserves optimized constants, leaf casts, floating operand conversions, and
intermediate rounding. Seven exact fixtures are promoted to required-JIT tests:
integer/Boolean leaf casts, literal folding, locals, loop indices, float32 results,
and a direct arithmetic branch condition. Both compiler preferences are checked.
The code-generation cache version is 14; typed text also enters the IR fingerprint.
Text-only hybrid expression plans must not bypass this typed lowering.

This is intentionally not a general typed-expression implementation. Unsupported
calls, comparisons containing division, nested cast arguments, integral/Boolean
arithmetic output, differing arithmetic intermediate dtypes, and inexact float32
division constants conservatively retain interpreter execution. Such kernels can
lose JIT acceleration; a compiler preference never licenses an incorrect result.
The full promotion/operator matrix still needs specification and certification.

Broader probing exposed nested-cast interpreter dispatch discrepancies. The
value-cast correction below covers `int(x + 0.25)`, including its use in floating
division. General nested cast semantics remain audit work; interpreter fallback
does not certify every accepted cast context. The separate `audit/division_nested`
sample is covered as a known-good fallback case in Python.

### Integer/Boolean cast arguments: value semantics correction implemented

`int()` and `bool()` now evaluate their argument in its compiled dtype before
applying truncation/nonzero truth. Previously the enclosing evaluator could
promote variables or recompute arithmetic in the cast's result dtype, changing
`int(x + 0.25)` and `bool(x + 0.25)` before the cast consumed the value. Native
DSL compilation marks these intrinsics; ordinary callbacks and `float()` retain
their existing policies. Variable-promotion traversal treats these casts as
boundaries, and nested writers still match their enclosing evaluator's width.

Integer arguments to `int()` retain their exact value without an intermediate
double conversion. `bool()` tests native nonzero truth, preserving NaN/infinity
truth and subnormals. Floating `int()` arguments truncate before output
conversion; unrepresentable/non-finite integer conversion remains outside the
certified domain. This does not settle contextual division or `float()` rules.

Five exact shared fixtures cover arithmetic cast arguments, nested Boolean/
integer casts, int64 values beyond `2**53` (including `INT64_MAX`), and non-finite
truth. They require interpreter/TCC/CC agreement. `audit/cast_argument_division`
checks the corrected interpreter path; nested-cast division still conservatively
falls back rather than using typed JIT lowering. Native tests exercise repeated
evaluation with counts 1/5/257 and both floating input/output widths; Python
artifact tests additionally exercise all five output dtypes for value casts.
Validation: 295 regular native tests and 233 AddressSanitizer tests passed;
484 focused Python tests passed with four existing math expected failures.

### Nested conversion buffer width: memory-safety correction implemented

AddressSanitizer isolated the anomalous results/abort for `audit/nested_cast.dsl`
(`float(int(x) / 2)`, float32 input/output): a nested conversion's declared target
was float64, but its enclosing float32 evaluator allocated a float32-sized scratch
buffer and dispatched that conversion through the float32 evaluator. Conversion
still selected an int64-to-float64 writer, overflowing that buffer.

The conversion writer now targets the actual typed evaluator's output
representation, including the Boolean special case, rather than a differing
declared nested target. Source evaluation still uses its recorded source dtype.
This is a buffer-width correction, not a general rewrite of cast inference or
promotion rules; typed JIT lowering still conservatively excludes nested casts.

`audit/nested_cast.txt` now provides exact interpreter regression values. A native
test checks float32/float64 input/output pairs, counts 1/5/257, all three JIT
policies, and repeated evaluation. Python artifact tests check the same dtype/count
matrix with interpreter and requested-JIT/fallback execution. All 217 tests in
the sanitizer build and all 274 regular native tests passed; 401 focused Python
tests passed with four remaining math expected failures.

Sanitizer validation also found that the existing mixed-type test allocated a
float32 output for an auto-inferred int32 + float32 expression, whose native
promotion is float64. That test now checks the compiled dtype and uses a matching
float64 buffer. This was a test allocation error, not a changed promotion rule.

### Float32 math intermediates and conditions

For float32 input and float64 output, interpreter `sin(1)` rounds to float32
before widening (`0.8414709568023681640625`). Scalar JIT math can retain the
double result, omitting intermediate rounding.

This can change control flow: for float32 `x = 0.0001`, rounded `cos(x)` equals
`1`, while the double result is below `1`. The standalone native interpreter
and Python-linked interpreter also differed on the direct condition in this
audit. Forcing interpreter mode therefore does not certify that case across
linked math engines. `math_condition` remains a reproducer, not a registered
passing standalone baseline.

The `math_widen` and corrected `nested_cast` interpreter audit cases are separately
registered in CTest. Python artifact tests preserve two math cases across two JIT
compilers as strict expected failures. The marker is applied only after successful
load, backend preparation, and execution; unrelated setup failures cannot hide.
An unexpected pass requires removing the marker and promoting the fixture.
Unavailable compilers are separate skips.

## Remaining publication gates

- Resolve expression gaps or explicitly narrow the profile.
- Audit literal/promotion rules, locals/intermediates, overflow/casts, loop caps,
  non-finite arithmetic, and linked scalar/vector math engines.
- Check additional platforms, not only this macOS host.
- Publish/integrate the native revision before changing Python's dependency pin
  or claiming adapter availability in ordinary distributed builds.
