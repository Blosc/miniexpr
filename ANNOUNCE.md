# Announcing miniexpr 0.3.1
A small, efficient C library for parsing and evaluating mathematical
expressions and DSL kernels.

## What is new?

This maintenance release fixes floating-point computation and dtype inference
for integer and Boolean inputs in classic expressions.

Highlights:

- True division (`/`) of integer and Boolean operands now computes and infers
  float64, including within nested arithmetic, comparisons and reductions.
  Zero divisors produce floating-point infinities/NaNs instead of integer
  division (issue #26).
- Binary real math (`hypot`, `atan2`, `logaddexp`, `nextafter` and `copysign`)
  now computes integer/Boolean inputs in floating point, with correct automatic
  dtype inference, comparisons and reductions (issue #27).
  Narrow inputs use float32, including where NumPy would use unsupported
  float16; 32-/64-bit integers use float64.
- New regression coverage for both fixes, including scalar/SIMD paths and
  sanitizer checks. Menudet's independently typed arithmetic is unchanged.

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
