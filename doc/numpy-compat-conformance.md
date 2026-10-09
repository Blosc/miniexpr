# NumPy conformance harness (M2)

The authoritative corpus is `tests/numpy-compat/vectors-v2.json`, with
`schema-v2.json`. Version 1 remains supported for checkpoint reproducibility.
The vector schema version is independent of artifact/language versions: all
executable cases retain checked draft **1.0** semantics.

64 cases: **41 matching, 21 known divergences, 2 capability skips** on macOS
arm64 with NumPy 2.5.3. These counts are bounded evidence, not universal parity.
Complex arithmetic and array broadcasting skip with named missing capabilities.
An on request still selects the portable interpreter: no eligible portable JIT
exists. Full-DSL accelerated performance controls do not certify this profile.

## Transport and comparison contract

Inputs/outputs use explicit fixed-width dtype, logical shape, and big-endian hex
bytes. Integers never pass through JSON floating-point numbers. Weak scalar
literals use decimal strings with strength/type; typed captured scalars retain
their own dtype/bytes; zero-dimensional arrays have category and empty shape.
Float vectors preserve signed zero, subnormals, infinities, and NaN payloads.

Layout recipes specify C/F order, physical byte order, last-axis step, and one
reversed axis. C and Python reconstruct physical storage then normalize to the
contiguous native binding contract. This is a copying adapter, not native
strided/broadcast support. Zero-element input has non-NULL owned backing.

Every comparison first requires dtype and shape equality. `bitwise` checks
finite encodings; `exact` checks numerical equality; `ulp` uses monotonic IEEE
bit-distance; `tolerance` accepts absolute + relative error **or** the declared
ULP budget. NaN handling (`bits`/`equal`) and zero sign (`exact`/`ignore`) are
independent. Infinities must agree in sign. Budgets are reviewed per case, never
expanded automatically. Native diagnostics compare stable categories rather
than platform-dependent text. Native status is separately reported.

Both native and integrated readers distinguish matching, known divergence,
new mismatch, baseline regression, and skip. Observation mode cannot rewrite
baselines. Explicit Python recording rejects unreviewed divergences and changed
existing baselines. Repeated evaluation checks result stability, diagnostic
cleanup, same-handle overflow-to-success recovery, and native floating-environment
restoration with non-default rounding and pre-existing exception flags.

## Standalone use (no Python or NumPy)

Build with `MINIEXPR_BUILD_ARTIFACT=ON`, then run:

```sh
build/tests/numpy_compat_runner tests/numpy-compat/vectors-v2.json off
build/tests/numpy_compat_runner tests/numpy-compat/vectors-v2.json on
build/tests/numpy_compat_example
ctest --test-dir build -R numpy_compat --output-on-failure
```

`tests/numpy-compat/example.c` consumes a complete artifact with explicit int64
buffers. The runner returns nonzero for malformed transport or unexpected
conformance failures. Named case IDs accompany unexpected results.

Generation, integration, seeded properties, minimization and reference drift
reporting live in Python-Blosc2's `tools/menudet_conformance.py`. Corpus and native
runner paths are explicit: there is no sibling library discovery. PCG64 seeds
1729 and 20261009 each check 48 safe arithmetic cases and one known overflow
boundary; lane/scalar minimization reduces it to int8 `127 + 1`. The promoted
stable vector is `seed-1729-minimized-int8-overflow`.

The separate 284-sample mpmath certification remains enabled. NumPy/libm
agreement is not mathematical proof. Linux, Windows, WASM execution and NumPy
1.26 integration remain unverified; platform-qualified integer cast sentinels
must not be treated as universal semantics.
