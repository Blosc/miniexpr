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

### Division promotion

With floating input, `int(x) / 2` at `x = 1.25` yields `0.5` in the interpreter
but `0` in both TCC and CC. Generated C retains integer operands and performs
integer division before the output cast. `bool(x) / 2` and literal-only `1 / 2`
show the same issue. Typed expression generation needs correction, or these
contexts need explicit profile rejection; changing expected results is not a fix.
Integer-valued locals and loop indices also need coverage.

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

Four interpreter audit cases (three divisions and `math_widen`) are separately
registered in CTest. Python artifact tests preserve five cases across two JIT
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
