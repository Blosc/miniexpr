Release notes for miniexpr
==========================

Changes from 0.2.0 to 0.2.1
===========================

** add blurb here **

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
