# Plan: High-Level Codebase Simplification

Last updated: 2026-03-26

## Goal

Capture the current main components of the repo and list the highest-leverage cleanup ideas for later work, with an emphasis on reducing maintenance cost without changing the public API or current behavior.

## Current shape

The codebase still has a fairly clear product split, but the implementation has accumulated complexity in a few very large translation units and in cross-cutting platform/backend branches.

Current size signals from `src/`:

- `src/miniexpr.c`: about 11.4k lines
- `src/functions.c`: about 8.2k lines
- `src/functions-simd.c`: about 5.0k lines

Current support surface:

- `tests/`: 33 C test files
- `examples/`: 19 C example files
- `bench/`: 26 C benchmark files

Current build/runtime axes:

- scalar vs SIMD/SLEEF
- plain evaluator vs DSL
- no JIT vs `cc` JIT vs libtcc JIT vs wasm32 JIT
- POSIX vs Windows vs Emscripten
- 1D/block evaluation vs ND/padded evaluation

That combination is now the main source of maintenance pressure.

## Main components

### 1. Public API and classic expression engine

Primary files:

- `src/miniexpr.h`
- `src/miniexpr.c`
- `src/functions.h`
- `src/functions.c`

Responsibilities:

- public compile/eval/free API
- classic expression parsing and AST construction
- dtype inference and promotion
- scalar/vector evaluation
- reductions
- ND block/chunk evaluation
- string and complex-number semantics

### 2. Builtin function/type system layer

Primary files:

- `src/functions.c`
- `src/functions.h`

Responsibilities:

- builtin function registry and lookup
- comparison/reduction classification
- conversion and promotion helpers
- type-aware evaluation dispatch
- string-specific and complex-specific helpers

### 3. SIMD math backend

Primary files:

- `src/functions-simd.c`
- `src/functions-simd.h`
- `src/sleef_compat/*`

Responsibilities:

- runtime SIMD backend selection
- SLEEF integration
- scalar fallback
- vector dispatch for transcendentals and related math kernels

### 4. DSL frontend and JIT pipeline

Primary files:

- `src/dsl_parser.c`
- `src/dsl_parser.h`
- `src/dsl_jit_ir.c`
- `src/dsl_jit_ir.h`
- `src/dsl_jit_cgen.c`
- `src/dsl_jit_cgen.h`
- large DSL runtime sections inside `src/miniexpr.c`

Responsibilities:

- parse Python-like DSL kernels
- resolve DSL expression dtypes
- build JIT IR
- emit C source for JIT
- interpret DSL when JIT is unavailable or skipped
- handle reserved index variables and ND synthesis

### 5. Runtime JIT backends and platform glue

Primary files:

- `src/miniexpr.c`
- `src/me_jit_glue.js`
- `CMakeLists.txt`

Responsibilities:

- libtcc loading and execution
- host `cc` invocation
- wasm32 side-module/main-module glue
- temp-file and dynamic-loader handling
- runtime cache handling
- environment-driven JIT policy

### 6. Validation, examples, and planning surface

Primary files/directories:

- `tests/`
- `examples/`
- `bench/`
- `doc/`
- `plans/`

Responsibilities:

- correctness coverage
- user-facing examples
- performance exploration
- feature notes and pending design work

## Highest-leverage simplification venues

### 1. Split `src/miniexpr.c` by responsibility

This is the single biggest simplification opportunity.

Today `src/miniexpr.c` mixes:

- public API entry points
- core expression compile/eval helpers
- ND/padding helpers
- DSL compilation
- DSL interpretation
- JIT runtime policy
- platform-specific loader/compiler glue
- wasm-specific handling
- environment parsing and tracing

Suggested internal split:

- `src/expr_compile.c`
- `src/expr_eval.c`
- `src/expr_eval_nd.c`
- `src/dsl_compile.c`
- `src/dsl_eval.c`
- `src/jit_runtime.c`
- `src/jit_platform_posix.c`
- `src/jit_platform_windows.c`
- `src/jit_platform_wasm.c`

Even if the exact filenames differ, separating those concerns would shrink review scope and reduce merge conflicts immediately.

### 2. Give the DSL/JIT path a narrower internal boundary

The parser/IR/codegen files are already separate, but the runtime half of DSL support is still deeply interleaved with the classic evaluator.

Good target shape:

- core expression engine does not need to know backend-specific JIT details
- DSL compilation returns one internal program object
- DSL evaluation/JIT fallback is handled by a dedicated subsystem API

This would make it much easier to reason about DSL work without reopening the whole evaluator.

### 3. Centralize runtime policy and environment parsing

`src/miniexpr.c` currently reads many environment variables directly for:

- JIT enable/disable
- JIT backend tuning
- trace behavior
- floating-point mode
- cache behavior
- compiler/toolchain overrides

That policy should move into one lazily-built config/capabilities object with clear responsibilities:

- parse environment once
- normalize defaults
- expose capability flags
- expose diagnostics/tracing settings
- drive backend selection

This would remove hidden dependencies and make backend behavior easier to test.

### 4. Introduce an explicit internal JIT backend interface

Right now `cc`, libtcc, and wasm32 logic is mostly organized by conditionals inside shared flow.

A cleaner model would treat them as internal backends with common operations:

- compile
- load/bind
- execute
- release
- cache key contribution
- failure reporting

That would reduce `#if` sprawl and keep platform code out of general evaluator logic.

### 5. Normalize builtin function metadata into one source of truth

`src/functions.c` contains a lot of coupled knowledge:

- builtin names
- arities
- purity/flags
- comparison vs reduction behavior
- dtype support
- conversion behavior
- scalar evaluator entry points
- vector dispatch expectations

There is likely room to simplify this with a single metadata table or X-macro source of truth before considering any heavier code generation.

The benefit would be less drift between parsing, typing, evaluation, and documentation.

### 6. Break `src/functions-simd.c` into dispatch code plus backend-specific chunks

The current SIMD file combines:

- runtime backend selection
- trace/init policy
- scalar fallbacks
- architecture-specific wrappers
- SLEEF include tricks

Suggested shape:

- one small dispatch/init file
- one scalar fallback file
- one x86/SLEEF backend file
- one arm64/SLEEF backend file
- shared wrapper helpers

This would make SIMD changes more local and reduce architecture noise in generic code.

### 7. Simplify the build graph

The top-level `CMakeLists.txt` currently owns too much:

- feature options
- dependency fetching
- bundled libtcc sub-builds
- wasm host-tool bootstrapping
- platform linker quirks
- library assembly
- test/example/bench wiring

High-level cleanup:

- move dependency/JIT/wasm logic into `cmake/` modules
- keep the top-level file focused on product assembly
- replace `file(GLOB ...)` in `tests/`, `examples/`, and `bench/` with explicit lists once the tree stabilizes

That would make build behavior easier to audit.

### 8. Rationalize the support surface

The repo has a healthy amount of coverage and examples, but the maintenance cost is rising.

Possible cleanup directions:

- group examples and benchmarks by topic rather than one flat directory
- add shared helper sources for repeated test/bench setup
- distinguish smoke/unit/integration/perf targets more clearly in CMake
- move overlapping TODO/design notes under `plans/` or a small architecture docs area

This would lower review noise and clarify what is shipping code versus investigation scaffolding.

### 9. Tighten repository hygiene around generated and local files

The current worktree already shows local/editor/generated content outside `build/`, for example:

- `build-asan/`
- `build-asan2/`
- `bench/__pycache__/`
- `.idea/`
- ad hoc root-level scratch files

That suggests the repo boundary between source and generated material is too loose.

Low-risk cleanup:

- expand `.gitignore`
- reserve a `scratch/` or similar area for one-off experiments
- keep generated artifacts out of the repo root when possible

This is not the biggest design problem, but it would improve day-to-day cleanliness quickly.

## Suggested order of attack

### Phase 0: document and protect behavior

- write a short internal architecture note that names the intended module boundaries
- add characterization tests for DSL/JIT and ND behavior before moving code

### Phase 1: structural split without behavior change

- split `src/miniexpr.c`
- extract central policy/config parsing
- isolate platform-specific JIT helpers

### Phase 2: subsystem cleanup

- give DSL/JIT a narrower internal API
- introduce a backend interface for JIT implementations
- split SIMD dispatch from SIMD backend implementations

### Phase 3: metadata and build cleanup

- centralize builtin metadata
- simplify CMake ownership boundaries
- reduce support-surface duplication

## Guardrails

- preserve the public C API
- preserve current behavior unless a change is explicitly planned and tested
- keep non-JIT scalar builds simple
- avoid mixing structural refactors with semantic changes
- prefer characterization tests before moving DSL/JIT/ND code

## Optional later ideas

- If manual metadata cleanup stops paying off, consider generating parts of the builtin registry or SIMD wrapper boilerplate.
- If the DSL/JIT path keeps growing faster than the classic evaluator, consider making it an optional internal library target instead of keeping everything in one core library.
