# Initial portable DSL conformance corpus

These are raw native source fixtures, not exported artifacts. Each `.txt` file
contains whitespace-delimited:

1. Expected outcome (`ok`, `ok_exact`, `compile_error`, or `eval_error`), input dtype, output
   dtype, element count, and input count.
2. Input names in compilation/evaluation order (possibly different from source).
3. One row per element: input values followed by the specified expected output.
   Error fixtures use zero as a placeholder for the output.

The runner limits cases to 4096 elements and 32 inputs. It supports `bool`, `int32`,
`int64`, `float32`, and `float64`. All inputs of a fixture currently share a dtype;
the output dtype may differ. Boolean tokens are `0`/`1`. Integers are parsed and
compared exactly, including int64 extrema and values above `2**53`. Floating-point
comparison uses `abs(actual - expected) <= tolerance + tolerance * abs(expected)`,
with `1e-6` for float32 and `1e-12` for float64. Signed zero must match; NaNs compare
by classification, not payload; infinities must match their sign.

`ok_exact` additionally compares finite floats exactly, preserving subnormal and
rounding checks. The 25 `convert_INPUT_OUTPUT.txt` fixtures share `identity.dsl`
and cover the supported 5×5 output-conversion matrix. The `audit/` sources preserve
unresolved numeric discrepancies; they are not required-JIT conformance claims.
See [the numeric audit](../../doc/dsl-spec/numeric-audit-0.1.md).

Seven exact `division_*` cases cover the first typed arithmetic lowering and are
required-JIT cases for TCC/CC. Unsupported typed contexts use interpreter fallback
rather than C token-level promotion. The `audit/nested_cast.dsl` source and its
exact fixture now guard the corrected nested-conversion buffer width in
interpreter mode; they do not certify general nested-cast/JIT semantics.
The `cast_argument_*`, `cast_integer_exact`, and `cast_nonfinite_truth` fixtures
require exact interpreter/TCC/CC agreement for value-cast arguments. The
`audit/cast_argument_division` fixture checks interpreter semantics while nested
casts in division remain outside the typed JIT slice.
The strict `math_widen`, `math_condition`, and `math_local_widen` fixtures now
require exact interpreter/TCC/CC agreement for leaf float32 sin/cos rounding
before widening or comparison. Other math contexts remain audit work.

Fixtures exercise bounded integer arithmetic, precise int64 comparisons, special
floating-point values, `break`/`continue`, unresolved names, unsupported indexing,
zero range steps, and executed missing-return paths. Additional fixtures check
Boolean-result semantics on numeric operands, small finite float-to-int casts,
and `sin` samples. They do not yet define overflow, mixed-input promotions,
arbitrary float-to-int casts, or global transcendental accuracy.
The fixture format is experimental test infrastructure, not an artifact
schema or a new public storage format.

Build miniexpr with tests enabled, then run:

```sh
build/tests/portable_dsl_runner tests/portable-dsl/affine.dsl tests/portable-dsl/affine.txt off
```

The final argument is `off` (require interpreter), `on` (require a prepared JIT
kernel), or `default` (allow the normal best-effort policy). The runner emits its
JIT status and computed values. It links only native miniexpr, never libpython.
CTest always runs interpreter cases and adds required-JIT cases when native TCC
is enabled, including kernels with incomplete return coverage. Missing-return
fixtures cover all-returning inputs, mixed successful/failing elements, returns
inside loops, and hybrid vector temporaries. A semantic missing-return error is
propagated without interpreter retry. Compile-error fixtures must fail at
compile time; runtime-error fixtures must first compile and then fail at evaluation.
On non-Windows hosts with a C compiler, CMake generates source variants that
change only the compiler pragma to `cc` and registers required-JIT CC cases.
These run even when bundled TCC is disabled. Both compilers must prepare a real
JIT kernel for the missing-return fixtures, including their successful paths.
CTest uses a build-local JIT cache and 30-second case timeouts, preventing
conformance runs from depending on a user's shared cache contents. A fresh-cache
retry resolved an observed float32 CC stall in the TCC-disabled configuration;
this isolation is not a claim to have diagnosed or fixed general cache behavior.
Python-Blosc2 consumes these files from the authoritative checkout
or the directory selected with `MINIEXPR_PORTABLE_CORPUS`.
