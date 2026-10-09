# NumPy arithmetic profile 1.1

The **opt-in** artifact/schema and language pair `1.1` / `1.1` selects
`ME_DSL_PROFILE_PORTABLE_1_1`. Checked `1.0` / `1.0` remains the default and its
existing corpus is unchanged. Mixed version pairs, unknown numeric policies and
unsupported required capabilities reject; there is no automatic artifact upgrade.

The semantic reference is NumPy **2.5.3**. The advertised M3 matrix covers bool,
int8/16/32/64, uint8/16/32/64 and float32/64. Half, complex, extended floats,
arbitrary-precision Python integers and array broadcasting are not implemented.
This is an arithmetic profile, not a claim of full NumPy function or array parity.

## Artifact policy and scalar categories

```json
"schema_version": "1.1",
"language": {"name": "miniexpr", "version": "1.1"},
"semantics": {"fp": "strict", "numeric": "numpy-2.5", "casting": "unsafe"}
```

Numeric constants additionally require `category: "weak"` or `"typed_scalar"`.
Weak constants use bool/int64/float64 transport, but integer/float **kind**, not
transport width, determines promotion when paired with a strong array operand.
Integers outside the transport range reject. Strong typed scalar constants retain
their dtype. Runtime input descriptors are strong typed arrays; a host scalar or
0-D array input is a strong 0-D array, never implicitly a weak constant. Scalar
strength survives JSON serialization and is independent of the originating host.
String snapshots retain their 1.0 encoding and have no numeric scalar category.

Source numeric literals are weak. Integral literals cannot silently wrap during
operand construction; `uint8(x) + -1` is not rewritten into subtraction.
Weak out-of-range arithmetic literals reject source validation. Weak captured
integers and explicit weak scalar constructors reject at evaluation if they do
not fit the selected dtype. Comparisons can compare out-of-range weak integers
exactly without first narrowing them. Weak float + integer promotes to float64;
weak float + float32 remains float32, including overflow to infinity. Typed
scalars/0-D arrays participate in strong promotion, not value-dependent coercion.
Local weak scalar bindings retain strength; mixed static branch types join to a
strong common storage dtype. Weak-only integral expressions are checked within
their transport width rather than silently becoming arbitrary-precision arithmetic.

## Operators and dtype resolution

`arithmetic-v1.1.json` contains the enumerated strong promotion/casting tables and
executable cases. Promotion is metadata-only: it reads no input buffers and runs
no kernel. `me_artifact_inferred_dtype()` reports the result before final output
conversion; no requested output dtype controls arithmetic intermediates.

- Add/subtract/multiply/power and unsigned negation wrap at the **computation**
  width. Signed negation/absolute value of a signed minimum returns that minimum.
  Operations use unsigned bits and explicit two's-complement reconstruction;
  no C signed overflow or implementation-defined unsigned-to-signed narrowing.
- Bool addition/multiplication and bitwise operations retain bool (OR/AND as
  appropriate). Bool subtraction and unary plus/negation reject. Bool floor
  division/remainder/power and shifts use int8. Mixed Boolean/number operands
  promote to the numerical operand's dtype.
- Mixed signed/unsigned promotion uses a signed width that can hold both ranges;
  when none exists (including int64/uint64), arithmetic promotes to float64.
  Integer comparisons instead use exact sign/magnitude comparison. All six
  comparisons return bool and preserve unordered-NaN comparison rules.
- True division of integral/bool operands returns float64; float32 division
  remains float32. Floating operations use round-to-nearest/ties-even and restore
  the caller's environment. Expressions do not contract multiply/add into FMA.
- Integer floor division rounds toward negative infinity; remainder has the
  divisor's sign. Zero divisors return zero for integer floor division/remainder.
  Signed minimum divided by minus one returns the minimum; its remainder is zero.
  Negative integer power rejects. Floating division/remainder follow IEEE values;
  floor division applies NumPy's fmod-based quotient correction, not floor(x/y).
- Bitwise and shifts require an integral common dtype. Shift operands undergo
  normal promotion. Negative or oversized counts produce zero for left shifts,
  and zero/sign extension for right shifts. No invalid C shift is executed.

NumPy-like aggregated floating status/warnings are milestone 4 work. The profile
does not claim Python `seterr` callback/warning behavior. WASM has fixed nearest
rounding and no observable IEEE exception flags: alternate-rounding/flag probes
are native-host-only; value tests still execute on WASM.

## Conversion

Fixed-width unary casts `int8(x)` through `uint64(x)`, `float32(x)`, `float64(x)`
and `bool(x)` are native syntax. `int`/`float` use explicit int64/float64 widths.
Array integer narrowing is modular; Boolean conversion uses truth, including
NaNs and infinities. Float changes and integer-to-float conversions round directly
to the destination width. Weak scalar construction is checked, distinct from
array conversion and array arithmetic.

Final output conversion declares `safe`, `same_kind` or `unsafe`. The enumerated
dtype matrix agrees with `np.can_cast` for supported types. Disallowed policies
reject at import before any input data is inspected. Explicit unary casts are
unsafe array conversions (checked for weak scalar construction), not policy-bearing
calls. Neither a permitted policy nor unsafe conversion bypasses resource checks.

**Defined divergence:** floating-to-integer conversion truncates toward zero but
rejects nonfinite or truncated out-of-range results, including negative unsigned
results. NumPy's unstable host/compiler-dependent sentinels are not a universal
portable contract. This divergence is recorded and diagnostic-tested, not hidden
behind a widened tolerance or a machine-specific golden integer.

## Backend and validation

The profile currently has **no eligible portable JIT/SIMD route**. On/default
requests select the interpreter, and `jit-required` rejects. Full-DSL TCC/GCC
controls remain a different semantic profile. CI registers the same corpus for
Linux, macOS, Windows and standalone Node/WASM builds; explicit on/off tests never
count fallback as accelerated conformance.

Python authoring uses `DSLKernel.export(..., version="1.1", casting="safe")`;
ordinary Python scalar captures are weak, NumPy scalars or `capture_dtypes`
are strong. `PortableKernel.inferred_dtype` is native metadata. Saved lazy recipes
preserve their original artifact/profile. An older native dependency rejects
1.1; use an explicit updated native checkout for development builds rather than
silently loading a sibling library or changing 1.0 semantics.
