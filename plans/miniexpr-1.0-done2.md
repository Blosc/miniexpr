# Step 4 review and Linux qualification

Reviewed the pending work described in [done1](miniexpr-1.0-done1.md), based on
`0cb6420`. The JSON-free pairwise extraction, shared ND geometry and WASM32 direct
API coverage are retained. Formula-reduction semantics and live legacy entry
points are unchanged.

## Corrections

- Pairwise core tests now advertise the real dtype-specific leaf-buffer capacity,
  check unaligned empty-result output with byte sentinels, and compare arbitrary
  fragments on cancellation-heavy data. Document that finish consumes the state.
- Linux Clang exposed a test-only use of glibc's conditionally available
  `CMPLX[F]` macros. Representation-based constructors preserve complex
  infinities/NaNs/signed zeros and work with both system compilers.
- Linux LeakSanitizer exposed four allocation-failure leaks in typed JIT source
  preparation. Release the uniform-setup builder on the affected failure paths,
  including WASM/ND fallbacks. Include that builder in source capacity estimates
  and reject truncated snprintf output.
- Format cached temporary identifiers directly, eliminating GCC's conservative
  format-truncation warnings without truncating arbitrary text.

## Verification

An isolated Lima VM uses Apple's Virtualization.framework and Ubuntu 24.04 arm64.
Source and builds are copied onto the guest's native ext4 filesystem, not built
on a macOS shared directory. The host checkout is not mounted in the guest.

| Configuration | Result |
| --- | --- |
| Linux GCC 13, artifact enabled, SLEEF/TCC disabled | 376/376 |
| Linux Clang 18, including Clang host-CC JIT | 376/376 |
| Linux GCC, SLEEF and bundled libtcc enabled | 469/469 |
| Linux ASan/UBSan, leak detection enabled | 8/8 focused |
| macOS artifact-enabled baseline | 376/376 |
| macOS ASan/UBSan | 9/9 focused |
| Python Menudet/persistence/table/graph integration | 1503 passed |
| WASM32 JIT side-module, including direct 0.2 API | 7/7 focused |
| WASM32 standalone JIT, after all Step 4 edits | 5/5 focused |
| C-Blosc2 constructors against JSON-free core | Passed |
| C-Blosc2 sum_linspace against installed core only | Correct sum |

The Linux failures were reproduced against the previous committed revision
before applying the corrections. These results do not claim Linux x86-64 or
Windows qualification. Raw logs, VM images and timing output remain outside
the repository.

Next: ensure CI covers `next`, full Linux Clang builds and leak detection;
then continue checked-profile/full-DSL caller migration, per-call interpreter
selection and cross-language recipe fixtures before freezing the ABI.
