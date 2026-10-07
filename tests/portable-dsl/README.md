# Draft portable DSL 1.0 conformance

These raw native sources exercise [draft 1.0](../../doc/dsl-spec/1.0.md),
not Python authoring and not exported artifacts. The typed interpreter is the
admission and execution baseline. Optional compiler/JIT preferences must fall
back before execution; they are not required-JIT certification.

Each `.txt` fixture contains whitespace-delimited fields:

1. Outcome (`ok`, `ok_exact`, `compile_error`, `eval_error`), input dtype,
   output dtype, element count and input count.
2. Input names in evaluation order.
3. One row per element: input values followed by expected output. Error cases
   use zero output placeholders.

The runner supports up to 4096 lanes, 32 inputs, Boolean, int32/int64 and
float32/float64 transport. Inputs currently share a dtype; the output may differ.
Boolean tokens are 0/1 and integers are parsed exactly, including values above
2**53. Integral outputs compare exactly. Floating comparisons use relative plus
absolute tolerances of 1e-6 (float32) or 1e-12 (float64); `ok_exact` compares
finite values exactly. Signed zero, infinity sign and NaN classification are
checked. This transport subset is not the complete language dtype matrix.

The identity conversion fixtures cover output conversions; arithmetic/cast,
predicate, local, masked control-flow, loop cap and missing-return cases exercise
operand-driven computation followed by checked output conversion. Checked
overflow/domain, mixed signed/unsigned typing, reductions, fixed strings and ND
coverage also live in the native portable type/interpreter/artifact unit tests.
The `audit/` fixtures intentionally exercise different ordinary full-DSL behavior
and run separately with the final `native` argument; they are not portable gates.

```sh
build/tests/portable_dsl_runner tests/portable-dsl/affine.dsl tests/portable-dsl/affine.txt off
```

Policies `off`, `on` and `default` are preferences. Draft execution reports
`jit=0`. The runner links native miniexpr, never libpython. CTest supplies loop
caps and isolated caches; Python may opt into these fixtures with
`MINIEXPR_PORTABLE_CORPUS` and `MINIEXPR_PORTABLE_RUNNER`, never sibling inference.
Local results do not certify other platforms or universal numeric accuracy.
