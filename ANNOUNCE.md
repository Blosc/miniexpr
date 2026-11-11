# Announcing miniexpr 0.3.0
A small, efficient C library for parsing and evaluating mathematical
expressions and DSL kernels.

## What is new?

This release introduces the feature-frozen Menudet 0.1 beta language profile,
native numerical graphs, and expanded portable execution.

Highlights:

- Menudet 0.1 beta: a versioned Python-like language with defined NumPy-compatible
  arithmetic, promotion, casting and floating-point status rules.
  - Native source validation and optional portable JSON artifacts.
  - Chained comparisons, Python numeric literals and block-scoped reductions.
  - See `doc/menudet-intro.md` and `doc/menudet-syntax.md`.

- Native numerical graphs with materialized stages, strided destinations,
  bounded storage readers and host executors for parallel partitionable maps.

- Expanded portable JIT coverage, including wasm32, plus faster bounded vector
  interpretation and in-memory TinyCC compilation.
- Flexible math backends and floating-point modes, configurable per kernel or
  compilation call.
- Fixed-width string and byte kernels, plus Arrow-compatible variable-length
  output through `me_eval_varlen()`.
- Portable complex64/complex128 interpretation, now also supported on Windows
  with clang-cl. Complex JIT kernels remain unsupported on all platforms.

- Numerous numerical correctness and performance improvements, backed by NumPy
  conformance corpora and interpreter/JIT qualification tests.

Compatibility note: native inverse-function calls now require canonical names
such as `asin` rather than `arcsin`, and the obsolete
`ME_COMPILE_ERR_COMPLEX_UNSUPPORTED` enum name has been removed. See the release
notes for migration details.

For full details, see:

https://github.com/Blosc/miniexpr/blob/main/RELEASE_NOTES.md

## What is it?

miniexpr is designed to be embedded directly into larger projects. It provides
fast expression evaluation with support for multiple numeric types, vectorized
evaluation, thread-safe parallel processing, and Menudet kernels with optional JIT.

## Download sources

The GitHub repository is here:

https://github.com/Blosc/miniexpr

miniexpr is distributed under the BSD 3-Clause license, see LICENSE for details.

## Fosstodon feed

Please follow https://fosstodon.org/@Blosc2 to get informed about the latest
developments.


Enjoy Data!
- The Blosc Development Team
