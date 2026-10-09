# Portable 1.1 functions and floating status (M4)

This extends the experimental opt-in 1.1 numerical profile; checked 1.0 keeps
its existing rules. The native-owned `tests/numpy-compat/functions-v1.1.json`
enumerates function/type signatures, aliases, exceptional vectors and accuracy
policies. NumPy **2.5.3** is the reference. Native library extensions that have no
NumPy equivalent are explicitly marked `native-contract`, not NumPy passes.

## Function matrix

The corpus covers the advertised real numerical functions:

- `abs` / `absolute`, `fabs`, `square`, `sign`, `conj`, `real`, `imag`.
- `ceil`, `floor`, `trunc`, `rint`, one-argument `round`.
- `minimum`, `maximum`, `fmin`, `fmax`, `where`, `pow` / `power`.
- `isfinite`, `isinf`, `isnan`, `signbit`.
- `sqrt`, `cbrt`, `exp`, `exp2`, `expm1`, `log` / `ln`, `log2`, `log10`, `log1p`.
- `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, inverse forms with both C and
  NumPy `arc*` aliases; `atan2` / `arctan2`, `hypot`, `logaddexp`.
- `copysign`, `nextafter`, `fmod`, `remainder`, `ldexp`.
- Native extensions: `erf`, `erfc`, `lgamma`, `tgamma`, `exp10`, `sinpi`, `cospi`,
  `fdim`, explicit fused `fma`, checked `fac` / `ncr` / `npr`, constants `e` / `pi`.

Every spelling has arity/type/value coverage. `min`/`max` are reductions, not
elementwise extrema. String operations/reduction families are separate milestones.
Complex, half and extended floats remain unsupported. The corpus explicitly
rejects NumPy signatures whose result loop would require float16 (including
transcendentals/rint/fabs with bool/int8/uint8). These must not be silently mapped
to float64. Int16/uint16 real-math loops produce float32; larger integral inputs
produce float64. Float32 operations call float libm functions, not double libm
followed by narrowing. Integer-preserving `ceil`/`floor`/`trunc` and `round` follow
the pinned NumPy version; Boolean `square` and `conj` produce int8. `fabs` promotes
integrals to a floating loop whereas `abs`/`absolute` preserve their integer dtype.
Predicates return bool without narrowing large integers through float64.

In 1.1, `round` and `rint` use ties-even independent of caller rounding; `sign`
maps both floating zeros to positive zero and propagates NaN. `remainder` uses
NumPy's divisor-sign remainder, while `fmod` uses truncation/sign-of-dividend,
including integral loops. Legacy 1.0 `round`/`remainder` rules are retained.

`minimum`/`maximum` propagate NaN; `fmin`/`fmax` select the non-NaN operand, and
return NaN if both operands are NaN. **Defined divergence:** zero ties use
deterministic IEEE minimum (negative if either zero is negative) and maximum
(negative only if both are negative). NumPy's tied-zero results can differ with
array length, SIMD loop and platform; they are not a universal bitwise contract.
NaN payload/sign and signaling-NaN preservation are not guaranteed.

`where` remains **selected-lane lazy**: unselected branch values and their floating
exceptions do not participate. This differs from Python/NumPy's eager argument
evaluation. Branch selection is not an instruction to evaluate both branches and
mask the result afterward. Logical short-circuiting and explicit valid masks obey
the same active-operation rule. `ldexp` rejects exponents outside native C `int`
range rather than relying on implementation-defined narrowing.

Unsupported NumPy spellings and signatures are visible in the corpus, including
`float_power`, `logaddexp2`, `heaviside`, `spacing`, `frexp`, `modf`, `round` with
decimals, ufunc methods and tuple-valued functions. Unsupported arities reject
source validation, not evaluation via Python or an external callback.

## Accuracy

Exact/bitwise policies apply to discrete transforms, classification, casts,
selection and stepping primitives. NaNs compare by class, signed zero remains
exact except where a documented rule transforms it. Transcendentals and composed
real helpers have an explicit **8-ULP tested bound** against pinned NumPy and
independent 100-decimal-digit mpmath finite references; explicit `fma` has a fused
reference and exact tested output. Overflow/underflow, infinity, NaN and domain
boundaries use class-aware comparisons, not a broad absolute tolerance.

Independent certification includes 32 seeded finite samples per function/dtype
case. This is measured sample qualification, **not** a proof of correctly-rounded
libm or a uniform global bound over all real arguments. Large argument reduction,
near-boundary and cancellation examples are represented separately. Libm accuracy
on a new platform must pass its own matrix before claiming that platform qualified.

## C floating policy

`me_artifact_eval_status()` adds a **per-call**, native-host reporting API; it
does not change the artifact schema, numerical rules, compiled handle or process
policy. Its descriptor and buffers are the same as `me_artifact_eval_ex`:

```c
me_artifact_fp_status status;
me_artifact_status rc = me_artifact_eval_status(
    artifact, inputs, ninputs, output, &descriptor,
    ME_FP_INVALID | ME_FP_DIVIDE | ME_FP_OVERFLOW, &status, &error);
```

Stable flag bits are invalid **1**, divide-by-zero **2**, overflow **4**, underflow
**8**; platform `FE_*` values never escape the ABI. `status.flags` is cleared each
call and ORs flags raised by evaluated active operations, including intermediate
and output conversions. `status.supported` reports capability. A zero `raise_mask`
collects without turning flags into errors; nonzero masks select which aggregated
flags turn an otherwise completed call into `ME_ARTIFACT_ERR_EVAL`. Unknown bits,
legacy 1.0 artifacts and unavailable raising policies reject explicitly. Output
after an error is unspecified; status on an execution failure reflects work that
was evaluated before the failure, not unevaluated later operations.

Each expression/block evaluates under nontrapping nearest/ties-even, captures
exception flags before restoring the calling thread's previous rounding and
flags. A scoped thread-local collector preserves flags through nested expression
guards; it is restored on all exits and never lives on a shared compiled handle.
Independent buffers on one handle can be evaluated concurrently. Across blocks
or worker threads, the **caller explicitly ORs returned flags**; the library does
not retain cumulative hidden state. Caller-owned status/error metadata must not
alias inputs/output or each other.

**WASM capability limitation:** WASM exposes no IEEE exception flags through its
C fenv implementation. Value evaluation with collection returns `supported=0`,
`flags=0`, not a claim that no exception occurred. A nonzero raise mask rejects
before execution. Native status tests are capability-qualified; WASM still runs
all numerical, signature, mask, environment and policy-rejection tests.

This is an **IEEE-operation flag** policy, not full NumPy `seterr` emulation.
Inexact is intentionally omitted; there are no callbacks, warning emission or
per-ufunc warning counts. Library/internal operations may raise flags differently
from a NumPy vector loop. Portable integer wrapping and zero divisors follow their
value contract without synthesizing floating warning flags. No RSS or scaling
claim follows from these numerical tests.

## Python and backend qualification

`PortableKernel.evaluate(..., return_status=True)` and `evaluate_block` return
`(values, {"flags": int, "supported": bool})`. `fp_errors="ignore"` is default;
`fp_errors="raise"` selects all four bits and attaches `.fp_status` to a
`PortableArtifactError`. Other warning/callback policies reject. Lazy scheduling
continues the default ignore behavior; standalone block users explicitly aggregate
status. Old native builds reject status requests rather than silently ignoring them.

The portable profile still has **no eligible SIMD/JIT route**. Off/on corpus
requests explicitly report interpreter fallback; full-DSL TCC/GCC controls do not
certify these rules. CI registers the same corpus on native Linux/macOS/Windows
and standalone Node/WASM. Local macOS/WASM results are not remote CI results.
