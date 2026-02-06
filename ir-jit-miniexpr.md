# IR + JIT in miniexpr (No LLVM)

## Context

Goal: for IR-compilable DSL kernels, dispatch a native JIT backend instead of the current miniexpr dense path.

Constraints:

1. Keep dependencies light.
2. Avoid LLVM as a required dependency.
3. Preserve current miniexpr behavior as a fallback.
4. Support current DSL syntax only (no broad language expansion in MVP).

## Current implementation status (as of 2026-02-06)

Implemented:

1. Milestone 1 complete:
   deterministic typed IR builder, rejection diagnostics, and tests.
2. Milestone 2 complete:
   C codegen for all currently supported non-complex dtypes, plus compile smoke tests.
3. Milestone 3 implemented (initial runtime cut):
   runtime compile/load/dispatch with fallback to interpreter on failure.
4. Runtime cache directory:
   `$TMPDIR/miniexpr-jit`.
5. Platform scope for runtime JIT:
   Linux/macOS only; Windows always falls back.
6. DSL permissiveness alignment with interpreter:
   `continue` accepted, truthy non-bool scalar conditions accepted.
7. Complex numbers:
   excluded from JIT IR/codegen in current implementation.
8. In-process negative cache for runtime JIT failures:
   keyed by runtime cache key, with failure class/timestamp and retry cooldown.
9. Process-wide positive in-memory cache for loaded JIT kernels:
   keyed by runtime cache key and reused across compilations in the process.
10. Disk cache metadata validation:
    runtime verifies cache metadata (target/ABI/codegen/compiler hash) before reusing artifacts.
11. Runtime cache tests:
    negative-cache cooldown, positive-cache reuse, and metadata mismatch rejection.
12. Rollout guardrail:
    `ME_DSL_JIT=0` disables runtime JIT and forces interpreter fallback.
13. Initial stabilization benchmarks:
    added Mandelbrot-style DSL benchmark for JIT cold/warm vs interpreter, plus optional numba baseline script.
14. Runtime compile-flag tuning hook:
    `ME_DSL_JIT_CFLAGS` is included in runtime compile command and cache metadata fingerprinting.

Current runtime limitations (known and intentional for now):

1. Runtime JIT dispatch currently targets regular per-element kernels.
2. Kernels using `_i*`, `_n*`, or `_ndim` use interpreter fallback.
3. Scalar-output DSL kernels use interpreter fallback.
4. Runtime cache (positive/negative) is process-local only (not persisted across processes).

## Proposed architecture

### 1. Frontend: parse DSL subset and validate

Input is current DSL source (function-like form). Build a strict AST validator for the supported subset:

1. Assignments to scalar temporaries.
2. Arithmetic and comparisons.
3. `if` / `elif` / `else`.
4. `for i in range(...)`.
5. `break`.
6. `return`.

Anything outside the subset is marked non-compilable and routes to normal miniexpr path.

### 2. Typed IR (compiler-internal)

Lower validated AST to a typed structured IR:

1. Scalar ops with explicit dtype.
2. Control flow nodes (`if`, loop, break).
3. Per-element load/store nodes.
4. Function signature: array inputs, scalar params, output pointer, length.

SSA is optional for MVP. A structured IR with explicit temporaries is sufficient.

### 3. C code generation backend

Lower IR to C source that computes element-by-element:

1. Outer loop over linear index `idx`.
2. For each input array, load `a = in_a[idx]`.
3. Execute kernel body with scalar temporaries and loop control (`break` maps naturally).
4. Write result into `out[idx]`.

This naturally supports early exit for kernels like Mandelbrot.

### 4. JIT compilation and dynamic loading

Use system C compiler (`cc` / `clang` / `gcc`) at runtime:

1. Generate source in cache directory.
2. Compile shared object with `-O3 -fPIC -shared`.
3. Load with `dlopen` and resolve entrypoint.
4. Store function pointer in process cache.

### 5. Dispatch in miniexpr evaluator

At evaluation time:

1. Check if expression is DSL and IR-compilable.
2. Build cache key `(dsl_hash, dtypes, ndim, target)`.
3. Lookup compiled kernel in memory/disk cache.
4. If available, run JIT kernel.
5. If unavailable and compile succeeds, run and cache.
6. If compile unavailable/fails, fallback to existing miniexpr path.

## Tradeoffs

### Pros

1. No LLVM dependency.
2. Native per-element loop performance with early break.
3. Smaller scope than full general-purpose compiler.
4. Works with current DSL semantics.

### Cons

1. Requires runtime C compiler availability.
2. Platform-specific compile/link details.
3. Extra complexity around caching and ABI stability.
4. Compile latency on first run.

## Why this over alternatives

### vs LLVM backend

1. Much lighter install and maintenance burden.
2. Lower integration complexity for MVP.
3. Easier operational story on developer machines.

### vs pure interpreter improvements

1. Interpreter stays slower for branch-heavy early-exit kernels.
2. JIT C path can approach Numba-like behavior on reruns.

### vs active-mask compaction runtime only

1. Compaction helps vectorized path but adds gather/scatter overhead.
2. Per-element JIT loop can avoid active-list management entirely.

## Cache design

### Key

Use the tuple suggested:

`(dsl_hash, dtypes, ndim, target)`

Where:

1. `dsl_hash`: hash of normalized DSL source and codegen version.
2. `dtypes`: input/output dtypes, including scalar param dtypes.
3. `ndim`: dimensionality and linearization strategy version.
4. `target`: CPU/ABI fingerprint (arch, compiler family/version, flags).

### Entries

Each cache entry should contain:

1. Shared library path.
2. Exported symbol name.
3. Metadata for ABI validation.
4. Build diagnostics (optional).

### Invalidation

Invalidate on:

1. DSL source or codegen version change.
2. Compiler flags change.
3. ABI/target mismatch.

## Runtime fallback policy

Fallback to current miniexpr path when:

1. DSL is not in compilable subset.
2. Compiler executable is missing.
3. Compilation or load fails.
4. Runtime safety checks fail.

Fallback should be explicit and optionally logged (debug mode).

## Safety and robustness notes

1. Generate C from validated IR only, never from raw unvalidated strings.
2. Compile in a private cache directory.
3. Sanitize symbol names and file names.
4. Gate with feature flag during rollout.

## Concrete MVP plan with milestones

### Milestone 1: IR-compilability gate + typed IR (no JIT yet)

Status: complete.

Deliverables:

1. DSL subset validator with clear rejection reasons.
2. Typed IR builder for accepted kernels.
3. Golden tests for accepted/rejected patterns.

Success criteria:

1. Deterministic IR for same input.
2. Accurate rejection for unsupported constructs.

### Milestone 2: C codegen + offline compile smoke tests

Status: complete.

Deliverables:

1. C code generator from typed IR.
2. Test harness that compiles generated C into shared object.
3. Unit tests validating numerical parity on small kernels.

Success criteria:

1. Generated C builds on Linux/macOS with `cc`.
2. Correctness parity for representative DSL kernels.

### Milestone 3: In-process JIT dispatch + fallback

Status: implemented for Linux/macOS in current runtime cut.

Deliverables:

1. Runtime compile/load/dispatch path in miniexpr.
2. Fallback path to existing miniexpr when unavailable.
3. Debug diagnostics for dispatch decisions.

Success criteria:

1. No regressions in existing behavior when JIT disabled or unavailable.
2. Automatic fallback works reliably.

### Milestone 4: Cache implementation

Status: complete.

Deliverables:

1. Cache key `(dsl_hash, dtypes, ndim, target)`.
2. Memory cache + disk cache.
3. Cache invalidation policy implementation.

Next workstream:

1. Milestone 5 performance benchmarking/tuning and rollout stabilization.

Success criteria:

1. First call compiles; repeated calls reuse cached artifact.
2. Wrong-target artifacts are not reused.

### Milestone 5: Performance and stabilization

Status: partial (benchmark + guardrail increment implemented).

Deliverables:

1. Benchmarks vs current miniexpr and numba for Mandelbrot and similar kernels.
2. Tuning of compiler flags and loop codegen.
3. Rollout guardrails and documentation.

Success criteria:

1. Significant speedup over current miniexpr on reruns for early-exit kernels.
2. Stable behavior across supported platforms.

## Open questions for later

1. Minimum supported compilers per platform.
2. Packaging story for environments without `cc`.
3. Whether to persist cache metadata in higher-level storage later (for example vlmeta), once ABI guarantees are defined.
4. Whether negative-cache entries should also be persisted to disk or remain process-local.
