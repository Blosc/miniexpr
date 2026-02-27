# Announcing miniexpr 0.2.0
A small, efficient C library for parsing and evaluating mathematical
expressions and DSL kernels.

## What is new?

This release is centered on a major new capability: DSL kernels.

Highlights:

- New Python-like DSL kernel support for multi-statement programs.
  - Write kernels with temporaries, `if/elif/else`, `for`/`while`,
    `break`/`continue`, reductions, and ND index symbols.
  - Canonical syntax and behavior reference:
    - `doc/dsl-syntax.md`
  - Practical guide and examples:
    - `doc/dsl-usage.md`

- Runtime JIT support for DSL kernels.
  - New parser/IR/codegen pipeline.
  - JIT policy controls in API (`me_compile_nd_jit`, `me_eval_params.jit_mode`).
  - Compiler/floating-point pragma controls (`# me:compiler=...`, `# me:fp=...`).
  - Expanded Linux/macOS/Windows/wasm32 coverage and robustness.

- New diagnostics helper:
  - `me_get_last_error_message()` for clearer compile/setup failures.

- Additional improvements:
  - Extended string support (`ME_STRING`) and documentation.
  - Multiple correctness/performance fixes in DSL/JIT and dtype handling.
  - Many new tests and benchmarks (DSL syntax, JIT runtime/cache, strings, ND behavior).

For full details, see:

https://github.com/Blosc/miniexpr/blob/main/RELEASE_NOTES.md

## What is it?

miniexpr is designed to be embedded directly into larger projects. It provides
fast expression evaluation with support for multiple numeric types, vectorized
evaluation, thread-safe parallel processing, and now DSL kernels with optional JIT.

## Download sources

The GitHub repository is here:

https://github.com/Blosc/miniexpr

miniexpr is distributed under the BSD 3-Clause license, see LICENSE for details.

## Fosstodon feed

Please follow https://fosstodon.org/@Blosc2 to get informed about the latest
developments.


Enjoy Data!
- The Blosc Development Team
