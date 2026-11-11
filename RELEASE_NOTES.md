Release notes for miniexpr
==========================

Changes from 0.3.0 to 0.3.1
===========================

* Fixes binary real math (`hypot`, `atan2`, `logaddexp`, `nextafter` and
  `copysign`) on integer/Boolean inputs in classic expressions, including
  automatic dtype inference, comparisons and reductions (#27). These functions
  now compute in floating point; narrow inputs use float32 (also in place of
  unsupported float16), and 32-/64-bit integers use float64.

Changes from 0.2.0 to 0.3.0
===========================

* Introduces the feature-frozen Menudet 0.1 beta language profile, independently
  versioned from miniexpr. It defines NumPy-compatible arithmetic, promotion,
  casting and floating-point status rules, with native source validation via
  `me_validate_menudet()` and `me_validate_menudet_ex()`.
  - Adds chained comparisons, `pass`, Python-style numeric literals, docstrings,
    semicolon-separated statements and explicit block-scoped reductions.
  - Optional portable JSON artifacts are available with
    `MINIEXPR_BUILD_ARTIFACT=ON`; the core does not require a JSON dependency.
  - See `doc/menudet-intro.md`, `doc/menudet-syntax.md` and `doc/dsl-spec/0.1.md`.

* Source compatibility change: native expressions, kernels and saved artifacts
  no longer accept the inverse-function aliases `arcsin`, `arccos`, `arctan`,
  `arctan2`, `arcsinh`, `arccosh` and `arctanh`. Use the canonical names `asin`,
  `acos`, `atan`, `atan2`, `asinh`, `acosh` and `atanh` instead.
  See `doc/inverse-function-names.md` for migration details.

* Adds native numerical graph preparation and execution, including materialized
  stages, strided destinations, bounded storage readers, transactional output
  aliases and host-provided synchronous executors. Proven lane-independent maps
  can be partitioned and scheduled in parallel. See `doc/native-graphs-1.md`
  and `doc/native-execution.md`.

* Expands portable JIT coverage for typed operators, math functions, casts,
  loops, ND index variables and reductions, including wasm32 support. Bundled
  TinyCC compilation stays in memory, and installed compiler headers are
  relocatable. Per-call `me_jit_options` select compiler settings, diagnostics,
  cache directories and math backends without changing the environment.
  Unsupported kernels retain interpreter fallback.

* Improves numerical correctness and interpreter/JIT parity for exact 64-bit
  integer operations, Boolean arithmetic and reductions, float32 rounding and
  `nextafter`, complex powers, masked evaluation and checked conversions.
  Adds NumPy conformance corpora and backend qualification tests.

* Extends fixed-width strings and bytes (`ME_BYTES`) with string-valued
  expressions, DSL locals and returns, chunked evaluation and string kernels.
  Fixed-width behavior follows NumPy semantics rather than NUL termination;
  case conversion preserves the inferred output width.

* Menudet weak-scalar strength is preserved across local variable joins. A local
  initialized from a weak Python literal and later reassigned from a strong
  float32/int operand no longer widens to float64/int64: for example,
  `z = 0.0; z = z + x_float32` now keeps float32 storage and values, matching
  NumPy 2.x and the documented local-weak-binding rule. Adds a focused regression
  test for float32, float64, integer and widening/checking cases.

* Menudet JIT emits same-width typed integer bitwise operations directly instead
  of calling scalar interpreter helpers. Scalar integer/Boolean local declarations no
  longer use volatile rounding barriers, reducing stack traffic in hash kernels.
  Floating-point local barriers and checked-operation error semantics are unchanged.

* Portable interpretation batches complex64/128 addition, subtraction,
  multiplication, negation and conversions to complex over bounded vectors.
  Uniform captures are supported; complex division, other unsupported operations
  and partially masked expressions retain scalar evaluation. Scalar numerical
  semantics and aggregate IEEE status are preserved; no complex JIT is added.
  Input-free complex kernels with an immutable final-index return override
  handle the endpoint once per unmasked block, avoiding per-element condition
  interpretation while preserving skipped-lane errors and IEEE status.

* Windows (clang-cl) now supports complex64 and complex128 expressions and
  portable interpretation, using ABI-safe UCRT complex math wrappers.
  Complex DSL JIT kernels remain unsupported on all platforms.
  - API change: `ME_COMPILE_ERR_COMPLEX_UNSUPPORTED` has been removed; its
    numeric value (-4) remains unused. Consumers referencing the enum name
    must remove that reference when updating miniexpr.

* Flexible math libraries and floating-point modes for Menudet kernels.
  - New source pragmas `me:math=auto|system|sleef|accelerate` and
    `me:ulp=u10|u35`; `me:fp=strict|contract|fast` now reaches portable kernels.
  - Resolution precedence: source pragma, artifact declaration,
    `ME_SIMD_MATH_BACKEND`/`ME_DSL_FP_MODE`, built-in default.
  - `auto` no longer prefers SLEEF; it follows the system-math policy.
  - Explicit unavailable SLEEF/Accelerate requests fail during preparation.
  - Accelerated math uses a kernel-local immutable callback table and bounded
    vector interpretation; artifacts declare the `flexible-math` capability.

* New varlen output for string expressions: `me_eval_varlen()` writes the Arrow
  layout (int64 offsets + a tight byte blob) instead of fixed-width slots.
  - `ME_STRING` yields UTF-8 (`large_string`); `ME_BYTES` copies bytes verbatim
    (`large_binary`). DSL kernels work unchanged.
  - Size the data buffer with the new `me_varlen_data_bound()`.
  - The conservative compile-time width bound is then spent on scratch only and
    never on the result: 9.7x smaller output for a 12% pack overhead on
    `bench/benchmark_varlen_output.c`. See `doc/strings.md`.

* Classic expression SIMD math on macOS prefers SLEEF when available at build
  time; Menudet's separate `auto` policy follows system math as described above.
  - New CMake option: `MINIEXPR_USE_ACCELERATE=ON|OFF` (enabled by default on macOS, off elsewhere).
  - For classic expression SIMD math in runtime `auto` mode, macOS prefers SLEEF first, then falls back to Accelerate when enabled, then to scalar kernels.
  - Added `ME_SIMD_MATH_BACKEND=auto|sleef|accelerate|scalar` for backend forcing during debugging and benchmarking.
  - Updated SIMD math benchmarks to report backend-appropriate columns instead of implying fake `U10/U35` distinctions for Accelerate/scalar runs.

Changes from 0.1.0 to 0.2.0
===========================

* New DSL kernel support (major feature).
  - miniexpr now supports Python-like multi-statement kernels with assignments,
    temporaries, `if/elif/else`, `for`/`while`, `break`/`continue`, `print`,
    reductions, and ND index symbols.
  - For DSL language details, see:
    - `doc/dsl-syntax.md` (canonical syntax/behavior reference)
    - `doc/dsl-usage.md` (practical usage guide and examples)

* Runtime JIT for DSL kernels has been added and significantly expanded.
  - New DSL parser + IR + C code generation pipeline.
  - Runtime JIT compiler policy controls (`ME_JIT_DEFAULT`, `ME_JIT_ON`, `ME_JIT_OFF`),
    available both at compile-time (`me_compile_nd_jit`) and eval-time
    (`me_eval_params.jit_mode`).
  - JIT backend/compiler controls via DSL pragmas (`# me:fp=...`, `# me:compiler=...`)
    and environment/toolchain integration (`CC`, `CFLAGS`).
  - Improved runtime behavior for Linux/macOS/Windows/wasm32 paths, including cache
    handling and wasm helper registration APIs.

* New public diagnostics helper:
  - `me_get_last_error_message()` to retrieve thread-local human-readable failure details.

* Extended string support:
  - `ME_STRING` data type and string operations/predicates are now documented and covered.
  - See `doc/strings.md`.

* Quality, compatibility, and correctness improvements.
  - Multiple fixes in dtype inference/casting and interpreter/JIT parity.
  - ND/padded block and index-variable behavior improvements.
  - Broader test/benchmark coverage (DSL syntax, JIT IR/codegen/runtime cache, side-module
    glue smoke tests, dtype mismatch cases, string ops, and more).
  - CI/build updates and platform robustness improvements (especially Windows and wasm32).


Changes from 0.0.0 to 0.1.0
===========================

* Initial public beta release of miniexpr.
* Vectorized expression evaluation across multiple data types.
* Thread-safe evaluation and chunked/ND processing helpers.
* Optional SLEEF-accelerated SIMD math kernels (CMake toggle).
* Examples, tests, and benchmarks covering core usage patterns.
