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

### Float32 leaf sin/cos: rounding correction implemented

For float32 input and float64 output, scalar interpreter `sin(1)` rounds to
float32 before widening (`0.8414709568023681640625`). Scalar JIT math previously
retained the double result. For float32 `x = 0.0001`, rounded `cos(x)` equals `1`,
while the double result is below `1`, so this difference can change control flow.

Wider probes isolated two additional interpreter issues: approximate SIMD
`sin(1)` produced the adjacent float32 value at count 257, and Boolean-expression
evaluation could pre-promote variables before evaluating math operands. The
Boolean path now leaves operand conversion to its existing preparation code,
rather than changing variable dtypes first. Strict DSL leaf `sin`/`cos` on a
float32 variable uses scalar `sinf`/`cosf` at every count, rounds before output
conversion, and forms a variable-promotion boundary. This deliberately trades
SIMD speed for count-independent strict values only for that bounded call shape;
non-strict math, classic expressions, and other functions keep their math policy.

Owned typed JIT text now lowers these leaf calls and their comparisons against
exactly representable float32 constants. It selects `sinf`/`cosf`, preserves the
float result before widening/comparison, bypasses text-only hybrid plans, and
participates in the IR fingerprint. Code-generation cache version is 15. This is
not a general lowering for nested math, arbitrary arithmetic intermediates,
other functions, or inexact comparison constants.

`math_widen` and `math_condition` have moved from `audit/` to required-JIT
interpreter/TCC/CC conformance, alongside `math_local_widen`. Python tests require
real JIT preparation and exact results with counts 1/2/257 and repeated evaluation;
the four former strict expected-failure markers are removed. A native regression
also checks signed zero and a reversed comparison at these counts in all JIT
policies, including sanitizer execution. General math accuracy across libraries
and platforms remains a publication gate, not certified by these samples.
Validation: all 307 regular native tests, 242 AddressSanitizer tests, and 525
focused Python tests passed, without expected failures or new build warnings.

### Pure float32 arithmetic: literal/intermediate rounding correction implemented

For float32 input/output and `x = 2**24`, `(x + 1.0) - x` previously returned
`0` in the interpreter but `1` in both JIT compilers. C's double literal widened
the inner addition, removing the native float32 rounding step. Decimal literals
expose another boundary: native `x - 0.1` first rounds the constant to float32,
so an input equal to that rounded value yields zero. A double-literal operation
followed only by an output cast instead yields a small nonzero result.

The new owned pure-arithmetic lowering preserves operand and intermediate
rounding for float32 `+`, `-`, `*`, and unary negation. Unlike native division's
scalar double callback, these operations convert scalar constants to the
evaluator's dtype before operating. Lowering uses the compiled computation
dtype rather than just the requested output dtype. Calls, casts, comparisons,
division, and mixed intermediate computation dtypes are deliberately outside
this slice; their existing paths are not certified by this correction.

The audit also confirms contextual literal typing: these floating literals use
the requested floating output context. The same `(x + 1.0) - x` expression with
float32 input and float64 output yields `1` at `x = 2**24`; it is not specified as
a float32 computation followed by widening. Explicit parameter/capture constants
have their declared dtype instead. This is documented observed native behavior,
not a completed promotion specification for every signature or local context.

Five exact shared fixtures cover exact/decimal literal rounding, scalar
subtraction, locals, and differing floating output context. Native and Python
artifact tests check scalar and vector counts 1/5/257, repeated evaluation, both
floating output widths, and addition/subtraction/multiplication. TCC and CC must
prepare real kernels for the shared fixtures and Python arithmetic matrix.
Typed assignments bypass text-only hybrid plans; ordinary float64 arithmetic
retains its existing hybrid optimizations. Owned text enters the IR fingerprint,
and code-generation cache version is 16.

Validation: all 328 regular native tests, 258 AddressSanitizer tests, and 632
focused Python tests passed, without expected failures or new build warnings.

### Same-dtype float32 `float()` arithmetic: corrected

The promotion audit found `(float(x) + 1.0) - float(x)` and
`float(x + 1.0) - float(x)` returning `1` in JIT instead of the interpreter's
`0` for float32 input/output at `x = 2**24`. The previous pure-arithmetic renderer
excluded calls, leaving C's double literals to remove intermediate rounding.

Typed pure-arithmetic lowering now admits `float()` when its result and argument
computation are both float32, recursively limited to the existing floating
leaves and `+`, `-`, `*`, and unary negation. Passing a float32 value through the
native double callback and back is an identity under the tested rounding policy;
rendering its argument preserves that argument's rounding before further
operations. This also covers a root `float()` assignment/return and repeated
same-dtype calls. It does not certify integral/mixed arguments, general calls,
comparisons, division, or other cast contexts. Ordinary float64 lowering remains
unchanged. Owned C text enters the fingerprint and bypasses incompatible hybrid
plans; code-generation cache version is 18.

Four exact shared fixtures cover leaf calls, nested arithmetic arguments, local
assignment, and float64 output context. The latter still returns `1` at `2**24`,
consistent with contextual floating literal typing, not float32 computation
followed by widening. Expanded native arithmetic tests cover repeated scalar
and vector evaluation. Required-JIT Python matrices check both compiler
preferences, output widths, counts 1/10/257, decimal constants, repeated calls,
negation, signed zero, subnormals, extrema, infinities, and NaNs. Exact value and
zero-sign checks do not promise NaN payload preservation.

Validation passed all 373 regular native tests, 292 AddressSanitizer tests, and
1019 focused Python tests without expected failures or new native build warnings.

## Remaining publication gates

### While-loop cap: JIT safety correction implemented

The interpreter enforced `ME_DSL_WHILE_MAX_ITERS`, but generated JIT `while`
loops previously had no counter. JIT now checks a scoped counter after the
condition and before each body entry. Exactly the cap's number of entries is
allowed; a false condition, `break`, or `return` at that boundary succeeds.
`continue` consumes an entry, while each nested/re-entered invocation resets its
own counter. Chained-condition lowering carries its condition-prefix statement
count into IR: that prefix executes before the cap check, not as a body entry.

The host cap defaults to 10,000,000, can be disabled with nonpositive values,
and falls back to its configured default on invalid/out-of-range text. The
normalized compile-time cap and presence of while loops enter the IR fingerprint;
code-generation cache version is 17. If the host changes the cap after loading,
native evaluation bypasses both direct and buffered JIT paths in favor of the
interpreter under the current policy. This requires no bridge/kernel ABI change.
JIT cap status propagates as `ME_EVAL_ERR_INVALID_ARG` without interpreter retry,
using the existing cleanup label to release hybrid temporaries.

Four shared fixtures check exact-limit success, mixed-lane cap failure,
`continue`, and hybrid failure with a cap of three. Python tests additionally
cover counts 1/5/257, repeated execution after errors, empty inputs, disabled and
invalid policies, changes after compilation (including an initially disabled
cap), cache isolation across caps, boundary `break`/`return`, nested/re-entered
counters, and inactive branches. Native runtime-stub tests verify both direct
and buffered semantic cap errors cannot trigger interpreter retry.

The audit exposed a separate interpreter discrepancy: `while 0 <= n < x` with
`x = [0, 2, 3]` unexpectedly hit the cap. This was initially retained as
`audit/while_cap_chain` with a strict expected failure; the following correction
resolves it and promotes the fixture to the passing corpus.
Validation: all 344 regular native tests and 270 AddressSanitizer tests passed;
727 focused Python tests passed with the one strict chained-while expected
failure. Ruff and whitespace checks passed; native builds emitted no new warnings.

### Masked local chain operands: corrected

Scalar evaluation of guarded chained-comparison operands used lane zero for
locals marked uniform by compile-time RHS analysis. Masked assignments and loop
updates can make those buffers nonuniform after lane zero exits. Local buffers
are full-width, so scalar operand evaluation now reads the active lane for all
locals; genuine non-local uniform bindings retain broadcast addressing.

Root reductions are an exception at the expression-evaluator boundary: they
write one scalar. DSL copies now broadcast that scalar across the destination
lanes, including masked copies, so reduction locals remain valid under per-lane
reads. A native compatibility regression covers `total = sum(x)` followed by
`0 < x < total`; reductions remain outside portable 0.1.

The promoted `while_cap_chain` and new `masked_local_chain`/`masked_bool_chain`
fixtures pass interpreter/TCC/CC execution. Python matrices cover numeric input
dtypes, counts 1/3/4/257, both lane orders, constant numeric/Boolean locals,
and repeated evaluation. A native regression additionally rotates all four
sample lanes and tests masked branch assignments followed by chained loops.
Validation passed all 357 regular native tests, 280 AddressSanitizer tests, and
895 focused Python tests without expected failures. Native builds emitted no
new warnings.

### Outstanding gates

- Resolve expression gaps or explicitly narrow the profile.
- Audit literal/promotion rules, locals/intermediates, overflow/casts, loop caps,
  non-finite arithmetic, and linked scalar/vector math engines.
- Check additional platforms, not only this macOS host.
- Publish/integrate the native revision before changing Python's dependency pin
  or claiming adapter availability in ordinary distributed builds.
