# NumPy compatibility: first checkpoint

Date: 2026-10-09. This is an inventory and conformance harness, **not a change to
arithmetic**, a release compatibility promise, or a JIT implementation.

## Reference, ownership and platform contract

The pinned generator reference is NumPy **2.5.3**. Starting branches are
`numpy-compat`: miniexpr `36770f87e10f1b92bfe0f9f92a7039eba2e976ea` and
Python-Blosc2 `063fd8b9a4c65ae75d76080f3497086a8e5a3e9d`. Python's established
native dependency pin is `3418cdce4b5c11e1661c8b6453e3d31d94b0b2b4`;
the native changes between that pin and the starting native SHA are documentation
only. The runtime C sources therefore have the same semantics at both revisions.
New local files are the vector corpus/schema, runner and its CTest registration.

Initial target set: Linux/macOS/Windows 64-bit native hosts and 32-bit WASM.
Only macOS arm64, little endian, Python 3.14.4, NumPy 2.5.3 is verified here.
NumPy 1.26 host support, secondary reference drift, Windows/Linux, WASM,
sanitizers and optional-feature build variants remain separate verification work.

Portable artifacts specify explicit widths. Existing `int()` means int64 and
`float()` means float64, independent of C `long`/pointer width. `intp` is not a
portable dtype. Future NumPy-default reductions must resolve the **authoring
platform** default once and persist an explicit accumulator width, never consult
the deployment compiler's `long`. Existing block reductions are not NumPy axis
reductions and must not be relabelled as such.

The authoritative reference corpus and machine-readable capability rows live in
`tests/numpy-compat/vectors.json`. Each row has operation, normalized artifact
signature, input/scalar categories, inherited reference configuration, expected
dtype/shape/bits, actual interpreter status/bits/backend and classification.
`matching` describes those explicit output signatures and particular values;
it does **not** certify inferred result dtype, every input or every backend.
The broader `tests/numpy-compat/inventory.json` and inventory below intentionally
include unverified/unsupported rows, with reference signatures, shapes and evidence.

## Dispatch and type/cast inventory

| Route | Existing computation | NumPy claim at checkpoint |
| --- | --- | --- |
| Full miniexpr expressions / DSL | `miniexpr.c`, `dsl_compile.c`, `dsl_eval.c`; common/full evaluation paths, typed leaves, output-context computation and specialized casts | unverified; not portable semantics |
| Portable compile/validate | `dsl_portable.c` -> `dsl_compile_program_profile` -> `dsl_portable_expr.c` typed annotation; metadata-only compilation | corpus validates explicit signatures |
| Portable execution | `dsl_eval.c` profile branch -> `dsl_portable_eval_expr[_masked]`; checked arithmetic in `dsl_portable_types.c` | observed interpreter baseline |
| Portable JIT/SIMD | draft profile disables full-DSL JIT IR/kernel selection; typed per-lane interpreter, not full math SIMD dispatch | unsupported acceleration, even when requested |
| Full JIT/SIMD | `dsl_jit_ir.c`, `dsl_jit_cgen.c`, backend files, `functions-simd.c` | unverified for NumPy; audit independently |
| Python export | `portable_kernel.py`: source literals retained; Python int/float captures become strong int64/float64; NumPy scalars keep explicit widths | no weak-capture category in artifacts |
| Python import/evaluate | installed `blosc2_ext.pyx` artifact handle -> native load/eval_ex; fixed output dtype, copying endian/stride adapter; no numerical fallback | same corpus status/bits as native |
| Ordinary LazyExpr / graph | NumExpr/full kernel routes and graph metadata; not automatically lowered to portable profile | unchanged, unverified here |

Native promotion: same signedness uses the wider integer; signed/unsigned uses a
signed width containing both ranges, rejecting the int64/uint64 pair. Integer
widths >16 mixed with float32 select float64; otherwise float32 is retained unless
float64 participates. Boolean arithmetic operands become strong int64 zero/one.
Comparisons can use exact signed/unsigned comparison instead of arithmetic
promotion. True integer division returns float64. Source literals are contextual
weak literals, but **integral float literals can adopt integer context**, unlike
NumPy weak Python floats. Strong typed captures do not become weak literals.

Portable intermediates have operation/operand types, not the output buffer type.
Output is a separate checked conversion. `int`, `float`, `bool` are the explicit
cast spellings: int64, float64, truth conversion. Integer narrowing and sign
changes reject out-of-range values; float-to-integer truncates finite in-range
values and rejects nonfinite/range failures. These are not NumPy `astype(unsafe)`.
No safe/same-kind/unsafe cast policy or general explicit-width cast syntax exists.

## Initial divergence register

All measured rows below use the paired starting revisions plus the new harness,
on the platform above. Stable case IDs in the corpus are the minimal reproducers.
Inventory-only rows name existing sources/tests; they are not conformance passes.

| ID / signature or category | NumPy 2.5.3 reference | Current native / adapter behavior; reproducer | Impact / cost | Decision |
| --- | --- | --- | --- | --- |
| D01 fixed-width integer arithmetic; values/diagnostics | array addition/multiply/negation/abs wrap at boundaries | checked eval error; `*-add-boundary`, `int8-multiply-boundary`, `int8-negate-min`, `int64-absolute-min` | common; moderate interpreter work, separate JIT audit | first slice: fix for 1.0 |
| D02 weak floating scalar; type/values | int8 + 1.0 computes float64 128 | literal adopts int8, overflows at 127; `int8-float-literal` | common authoring; moderate context/type work | next slice: fix |
| D03 signed/unsigned 64-bit arithmetic; type/capability | float64 common type | compile rejection; `int64-uint64-promote` | moderately common; low/moderate table cost | fix after integer slice |
| D04 Boolean arithmetic; type/values | bool + bool remains bool | int64 arithmetic; `bool-add` gives 2 rather than True | masks; moderate operator-specific promotion | fix with promotion slice |
| D05 integer zero divisor/min/-1; diagnostics/values | zero divisor gives zero; min/-1 wraps, floating warnings suppressed in corpus | checked eval error; `int64-zero-divide`, `int64-min-divide` | common exceptional paths; moderate diagnostics design | fix in integer slice, status policy separate |
| D06 large shift counts; values/diagnostics | out-of-width count gives zero for these vectors | checked eval error; `int8-large-shift` | less common; low interpreter cost | fix with bitwise slice |
| D07 narrowing/output casts; values/diagnostics | unsafe narrowing wraps; `int64-narrow-output` -> [127,-128,127] | checked output conversion rejects | common IO; moderate explicit policy/API cost | explicit cast-policy slice; do not silently equate output conversion with astype |
| D08 nonfinite float -> int; values/diagnostics | this machine returns INT64_MIN with invalid status suppressed | checked rejection; `float64-cast-infinity` | occasional; hardware-sensitive reference | investigate; sentinel bits are platform evidence, not universal contract |
| D09 scalar categories; types/artifacts | weak scalar, typed scalar and 0-D distinctions | literals weak; captures strong; no weak capture encoding; typed and 0-D corpus probes included | high authoring impact; moderate artifact/lowering changes | fix after revision transition |
| D10 half/complex/extended precision; capability | supported NumPy types | absent portable dtype support (`portable_width1`, Python descriptor rejects); no vectors | domain-specific; large extension cost | defer explicitly |
| D11 real functions/aliases; values/types/diagnostics | signature-specific promotion, NaNs/zeros and FP flags | advertised portable builtin list in `dsl_portable_expr.c`; only sin zero signs tested here; most signatures unverified | high; substantial certification cost | inventory first, incremental common functions |
| D12 broadcasting; shape/storage | singleton/scalar axes broadcast | Python adapter requires equal shapes; native elementwise adapter equal cardinality only | high; large native traversal work | milestone 5; unsupported here |
| D13 axis reductions; shapes/type/grouping | logical axes, keepdims, NumPy accumulator defaults | explicit block-local reductions, PortableLazy combines selected reductions; not general native axis scheduling | high; large native ownership/scheduling work | milestone 5; axis API unsupported |
| D14 layout/aliasing; storage | views/strides and overlapping outputs where permitted | Python normalizes to aligned contiguous host endian, allocates output; native explicit buffers prohibit overlap | high; functional copying available, zero-copy iterator costly | deliberate copying adapter; optimized iterator deferred |
| D15 FP diagnostics; diagnostics | np.seterr policy/warnings | strict numerical environment helpers, coarse eval status; no public aggregated NumPy-like floating flags | high for scientific hosts; threading/mask policy costly | design before broad exceptional-function claims |
| D16 inferred output dtype; types | ufunc result dtype derives from operands/op | artifact requires explicit output signature; equal output bytes can conceal intermediate mismatch | high; metadata inference integration moderate | unverified dimension; add native inference assertions in slice 3B |

## Vector schema and runner boundary

`tests/numpy-compat/schema.json` specifies `menudet-numpy-vectors-1`, deliberately
separate from artifact schema/language 1.0. Integers use fixed-width big-endian
two's-complement bytes (unsigned use ordinary binary), floats use their IEEE bits,
and bool uses byte 00/01. Hex strings are lossless: no JSON numeric rounding.
Shapes distinguish 0-D arrays from length-one arrays. Typed scalar operands are
listed separately and bind through the artifact constants; literals record weak
category and scalar kind. Layout is contiguous C only. Signed zero and NaN bits
are exact for this small corpus; no tolerance is silently applied. IDs are stable.
Generation is deterministic, not randomized (no seed). Generator revision is
`checkpoint-1`; the Python change revision is recorded with integration evidence.

`expected` is the pinned NumPy result. `baseline` is **observed current behavior**,
including native artifact status (-3 source rejection, -5 evaluation failure) and
output bits only on success. CTest compares to baseline, NOT a pretense that
divergent NumPy cases passed. The integration report retains underlying native
status and Python category. Changing semantics requires reviewing these failures,
not automatically accepting fresh golden data. Unsupported schemas/revisions and
malformed extents/encodings reject. The runner is reviewed test tooling, not a
general-purpose untrusted JSON execution service.

The native runner consumes artifact JSON and explicit aligned typed buffers with
no Python/NumPy runtime. Limits: 16 inputs, <=8 dimensions, <=4096 elements, numeric
elementwise output only. The schema is intentionally a **small milestone-2 slice**:
general diagnostics-as-reference, tolerance/ULP policies, noncontiguous recipes,
arbitrary operation graphs, capability skips, random generation/minimization and
WASM loading are follow-ups requiring a schema extension/version review.

To build/run independently (artifact support uses the existing yyjson dependency):

```sh
cmake -S . -B build-numpy -DMINIEXPR_BUILD_ARTIFACT=ON -DMINIEXPR_ENABLE_TCC_JIT=OFF
cmake --build build-numpy --target numpy_compat_runner
build-numpy/tests/numpy_compat_runner tests/numpy-compat/vectors.json off
build-numpy/tests/numpy_compat_runner tests/numpy-compat/vectors.json on
ctest --test-dir build-numpy -R numpy_compat --output-on-failure
```

Reports explicitly say interpreter for both requests. Native CPU compile/eval
microtimings use `clock()` (101 evaluations); they are not wall-time or JIT evidence.

## First arithmetic slice and artifact transition decision

Choose fixed-width **array add/subtract/multiply and unary negate/abs**, all signed
and unsigned widths, before general scalar promotion. The boundary corpus already
demonstrates failures across all eight integer widths; implementation is localized
and can use defined unsigned/bit operations, with clear validation/interpreter tests.
Keep division/zero-divisor and invalid-shift policy as subsequent small slices.
Do not fold scalar/promotion/cast redesign into wrapping: they need distinct tests.

Transition policy selected for this project: an incompatible NumPy-oriented draft
must use **language version 1.1 and artifact schema 1.1**, backed by a distinct
native semantic profile. Never reinterpret 1.0 checked artifacts. The next slice
must either retain the old 1.0 interpreter or explicitly reject 1.0 at import;
silent upgrade is prohibited. Migration is opt-in, re-exported from authoring.
There is no commitment to preserve every draft through release 1.0. Current
validators already reject 1.1; Python regression tests verify that boundary now.
The new versions/profile are a prerequisite to incompatible arithmetic, not
implemented runtime capability at this checkpoint. No permanent arithmetic
personalities or user-facing switch are introduced here.
