# 1.0 cleanup and release boundary

This follows the reviewed Step 4 baseline (`51a7a8f`, `fea639f`) and the
GitHub CI configuration correction (`8d91cc8`). The corrected baseline passed
all routine hosted jobs, including Windows x64/ARM64, Linux x86-64/ARM64,
macOS, Clang sanitizers and WASM.

## Cleanup decisions and implementation

1. Publish the tested native branch to Codeberg and GitHub, with GitHub running
   qualification for `next`. Remove the suite-wide CC override: compiler choice
   must not invalidate tests that intentionally check default/TCC behavior.
2. Retire the unused checked-arithmetic profile and typing convenience function.
   Move portable source fixtures to the public prepare/eval API. Preserve their
   control-flow, masking, initialization and threading coverage with independent
   public numerical expectations. Remove checked-only typing branches, not
   required bounds, range/weak conversions or ordered-reduction overflow checks.
3. Add invocation-local `ME_RUN_INTERPRET` selection. It bypasses a prepared JIT
   without recompiling, mutating the shared handle or replaying execution.
   Absent appended fields default to the prepared policy. Test backend bypass,
   masked coordinate evaluation, old prefixes and invalid execution values.
4. Add a JSON-free C MessagePack fixture peer and C/Python v2 exchange tests.
   Cover exact captures, coordinate generators, nested annotated/unannotated
   recipes, extension 43 and Python carrier/file/CFrame reopening. The bounded
   fixture subset is explicit; no general storage resolver is added to the core.
5. Separate callback-free `miniexpr_types.h` from compatibility declarations.
   Fix dtype numbers explicitly, install the header, define the core API version,
   and publish the supported headers, symbols, extension rules and compatibility
   policy in `doc/c-api-1.md`. Finalize the 0.2 annotation vocabulary.

Python's ordinary formula prefilter now owns a typed kernel/grid handle instead
of an ND expression. String-width discovery and portable JIT probes no longer
use the old compiler. Update generator/UDF tests to observe direct recipe
preparation rather than JSON import, and retain scalar-return cardinality
discovery on the direct path.

## Deliberate retention, not unfinished deletion

Classic formulas, C callbacks and full-source/unresolved formula-reduction
compatibility calls remain under `miniexpr.h` through 1.x. They still have real
consumers. Removing them or reinterpreting source would be a behavior change,
not dead-code cleanup. The new embedding headers no longer expose them.
Formula reductions remain explicitly deferred.

The export audit also retains Windows `WINDOWS_EXPORT_ALL_SYMBOLS` (private
core-to-adapter calls) and `/OPT:NOICF` (builtin function-pointer identities).
Neither is safe to delete without a separately qualified implementation
refactor. Export visibility does not make private helper symbols a stable ABI.
The core contract is ready for the 1.0 freeze; this work does not publish a
release tag or promise compatibility with alpha-pilot layouts.

Raw logs, build trees and generated fixture payloads remain outside the source
trees. The reusable qualification fixtures and concise results belong here.
