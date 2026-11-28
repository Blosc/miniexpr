# Execution migration: CI qualification plan

**Implementation status:** the dedicated workflow and qualification runners are
implemented and published. The C4/native and Python correctness matrix passes at
Python-Blosc2 `828a7eb4` with the exact native pair recorded in
[`execution-migration-status.md`](execution-migration-status.md). Performance
acceptance and release/ABI/default-cutover approvals remain separate gates.

## Goal and agreed scope

Complete C4 and the outstanding cross-platform correctness gates for
[`execution-plan-migration.md`](execution-plan-migration.md), using the locally
qualified checkpoint in
[`execution-migration-status.md`](execution-migration-status.md).

Agreed decisions:

- Publish the matching MiniExpr and C-Blosc2 development branches upstream, then
  push Python-Blosc2 `next` to run qualification.
- Add a dedicated execution-migration workflow; preserve existing compatibility
  workflows and production build defaults.
- Qualify Windows with both MSVC and clang-cl.
- Monitor CI, fix failures, and rerun affected configurations before considering
  C4 closed. A green Python test suite alone does not close the native API gate.

This is development qualification, not authorization to publish releases, freeze
the public ABI, or switch production defaults.

## 1. Publish a reproducible three-repository checkpoint

1. Inspect repository instructions, working-tree changes, branches and remotes in
   all three repositories. Preserve unrelated work; verify the GitHub upstream
   destinations before pushing. Do not force-push.
2. Confirm the actual committed checkpoint against the status ledger:

   | Repository | Development branch | Recorded baseline |
   | --- | --- | --- |
   | MiniExpr | `next` | `a4d4d80` (status documentation may have advanced) |
   | C-Blosc2 | `next-execution-b2nd` | `11af7649` |
   | Python-Blosc2 | `next` | `92b32372` |

3. Push the native branches first. Record full commit IDs and verify those IDs
   are fetchable from GitHub before starting Python qualification.
4. Put exact native commit IDs in a qualification-only manifest or workflow
   configuration. CI must check out both dependencies at those IDs and pass
   source overrides to Python builds. Do not resolve floating branch heads.
5. Push Python's dedicated workflow and record its commit ID alongside the
   native pair. Every result must identify all three revisions.

Existing bundled pins predate this checkpoint. Do not silently qualify those
older dependencies or advance shipping pins merely to enable development CI.
Use fresh build directories and dependency-aware cache keys.

## 2. Dedicated workflow and execution order

Add `.github/workflows/execution-migration.yml` in Python-Blosc2:

- Trigger on pushes to `next` and on `workflow_dispatch`; support rerunning a
  recorded checkpoint without publishing wheels.
- Set minimal read permissions, explicit job timeouts, and matrix
  `fail-fast: false`.
- Cancel superseded runs for the same development branch, but retain the final
  run's logs and artifacts.
- Start with one representative supported Python version per platform. Existing
  workflows continue to cover broader Python/NumPy compatibility.
- Make configuration explicit: shared core enabled, artifact-off primary,
  artifact-on comparison. Record compiler, architecture, backend availability,
  FP observation support and actual execution backend.
- Assert the intended executor is available before testing; do not allow an
  all-skipped typed suite or accidental compatibility build to pass as evidence.

Stage the rollout to expose cheap failures first:

1. Header, build, install, export and native public-API smoke checks.
2. Installed native consumers and compression-only checks.
3. Isolated installed-wheel migration and platform qualification.
4. Native sanitizer coverage and bounded WASM runtime qualification.
5. Separate Release performance qualification on native x86-64 hardware.

Keep all required gates visible even if initially failing. Do not turn an
unsupported configuration into an unconditional skip or `continue-on-error`
without recording it as an open gate.

## 3. Qualification matrix

| Platform | Compiler / configuration | Required evidence |
| --- | --- | --- |
| Linux x86-64 | GCC, Release | Native C4, artifact-off and artifact-on installed Python wheels, platform correctness |
| Linux x86-64 | Clang, ASan + UBSan | Native public API, installed consumers, executor/storage/stages; report instrumentation scope |
| Windows x86-64 | MSVC, Release consumers/stubs | Compile installed public-C/DLL consumers with MSVC against the clang-cl numerical build; build compression-only C-Blosc2 with MSVC |
| Windows x86-64 | clang-cl, Release | Same migration configurations and native public-API coverage; retain existing compatibility job |
| macOS ARM64 | Apple Clang, Release | Reproduce public-API and installed-wheel smoke/qualification on CI; retain local evidence separately |
| WASM | Emscripten + bounded Node/Pyodide runtime | Native installed consumers/stubs where supported, artifact-off and artifact-on runtime correctness, explicit FP limitations |

Compression-only builds are independent native configurations, not inferred from
successful numerical builds. Linux ARM64 local qualification remains useful but
does not substitute for Linux x86-64.

**Toolchain boundary discovered during implementation:** MiniExpr and Python
explicitly require clang-cl on Windows because the numerical core uses native C99
complex types. MSVC qualification therefore checks installed C ABI consumers,
headers and compression-only code; it does not claim an MSVC-built numerical core
or Python extension. Those are built and tested with clang-cl in both artifact
modes. A full MSVC numerical-core port is separate work, not a removed build guard.

### WASM build prerequisite

The current Python CMake configuration restricts shared-core admission to native
bundled builds. Audit that guard and the Emscripten build/link path before claiming
WASM migration coverage. Make only the platform changes needed to support the
same executor and installed public headers; do not route WASM silently through
the old adapter.

Use bounded smoke tests first, then the dedicated runtime selection with explicit
timeouts. Full local emulated runs are not the default. Distinguish compile-only
evidence from actual runtime qualification.

## 4. Native C4 checks

For each applicable platform/compiler:

1. Build and install MiniExpr artifact-off and numerical C-Blosc2 into fresh
   prefixes. Build external consumers against the installed prefix, not private
   source headers.
2. Run `test_b2nd_exec_api`, `b2nd_example_exec_eval`, and
   `tests/installed-miniexpr` (current baseline: seven consumers), including:
   - Pure-C `installed_public_exec`, without MiniExpr headers or direct linkage.
   - Optional `installed_public_exec_kernel`.
   - Existing executor, storage, launcher and staged consumers.
3. Compile public headers as C99, C11 and C++ with strict warnings where supported.
   Exercise initializers, `*_V1_SIZE`, truncated/invalid sizes and API version 1.
   Do not require identical struct layouts across different platform ABIs.
4. Verify plan/schedule free-order safety and retained ownership, including
   composed plans. Run Release assertions with checks enabled.
5. Build compression-only C-Blosc2 independently:
   - `b2nd_exec.h` installed; optional kernel header absent.
   - External consumer compiles, links and runs against installed libraries.
   - Availability is false and entry points return the documented unsupported
     status.
   - No unintended MiniExpr/artifact dependency.
6. Audit exported public symbols using platform-appropriate tools. On Windows,
   explicitly build a shared DLL and external import-library consumer to test
   `BLOSC_EXPORT`; exercise MSVC atomic refcount and conservative serial execution.
7. Verify installed CMake package discovery and pkg-config metadata where
   supported. Record any platform-specific packaging limitation explicitly.

Keep standalone deterministic launcher/failure tests distinct from the production
launcher. POSIX concurrency results do not qualify Windows parallel execution;
Windows remains conservatively serial.

## 5. Python correctness and isolation

Build separate artifact-off and artifact-on wheels with the exact native source
overrides. Install into clean environments and verify package/extension locations;
do not run against an editable checkout.

Run:

- `scripts/qualify_execution_migration.py` with the appropriate artifact mode.
- `scripts/qualify_execution_platform.py`.
- Cleanup and ordinary-caller ownership regressions, including
  `test_execution_cleanup.py` and `test_resident_prefilter.py`.

Audit the runners for POSIX-only assumptions before using them on Windows/WASM.
Adapt the harness without weakening numerical, failure or admission contracts.

Required coverage includes:

- Resident, compressed, mixed/frame-backed, generic callback, logical-partition
  and staged routes within their documented admission limits.
- Dtype/shape/capture parity, broadcasting, unaligned/reversed/endian views,
  reduction order, empty domains, masks, explicit accumulators and output aliasing.
- FP flag union, raise masks, restoration and non-default FP/math policies.
  Where flag observation is unavailable, test truthful reporting and rejection
  of unsupported raise requests rather than accepting fabricated zero flags.
- Interpreter and available JIT backends, requested versus actual backend
  reporting, and fail-closed required-JIT failures. Separate optional backend
  skips from mandatory interpreter correctness.
- Metadata-only rejection before reads/writes, callback and allocation failures,
  no execution replay, publication contracts, schedule reuse and lifetime safety.
- Supported concurrent/nested calls, submission-failure draining and aggregate
  scratch budgets on platforms admitting parallel execution.
- Original persisted v1 / 4.17 v2 compatibility fixtures where available. Make
  fixture provenance explicit; retain any unavailable fixture as an open gate.

Artifact-on supplies the explicit comparison adapter; artifact-off must prove
that native artifact/graph/JSON dependencies are absent. Audit extension symbols,
link inputs and dynamic dependencies for one core provider and no unintended
adapter dependency using platform-appropriate tools.

## 6. Failure triage and evidence

Upload logs and machine-readable reports even on failure:

- Three full repository revisions and workflow run URL.
- OS/architecture, toolchain versions, build options and installed locations.
- CTest/pytest results, including skip reasons and selected/deselected counts.
- Header/export/link audits, backend/FP capabilities and sanitizer diagnostics.
- Build logs and small reproducers for failures; wheels where useful.

Triage failures as dependency/build, packaging/ABI, harness portability, numerical/
FP, scheduling/lifetime, or an explicitly unsupported capability. Reproduce with
the smallest targeted test, fix in the owning repository, publish the new native
revision if needed, and update the exact qualification pair.

After a fix, rerun its targeted checks and affected full configurations. Results
from different dependency pairs must not be merged into a single passing matrix.
Update `execution-migration-status.md` with configuration-specific evidence,
warnings and remaining gaps.

## 7. Performance and final cutover gates

Correctness CI does not establish stable performance. Run Release
`bench/execution_handoff.py`, `bench/compressed_reduction_plan.py`, and compressed
sink benchmarks on native Linux x86-64, alongside the macOS ARM64 baseline.
Separate lowering, preparation, specialization and warmed execution; interleave
equivalent old/new configurations and investigate repeatable regressions above 5%.
Shared-runner timings may guide triage but are not a production performance gate.

C4 closes only when the required installed-consumer, export/stub, ABI and runtime
checks pass across the recorded platform matrix. Compile-only WASM or clang-cl-only
Windows results are insufficient.

After C4 and correctness/performance qualification:

1. Review the final public API and approve its ABI freeze separately.
2. Advance shipping dependency pins to exact published revisions.
3. Qualify clean wheels without local/CI source overrides.
4. Switch the production shared-core/artifact-off defaults and remove remaining
   superseded wiring only under the M9 acceptance gates.

No release tags, package publication or production-default change is part of
the initial push-and-qualify phase.
