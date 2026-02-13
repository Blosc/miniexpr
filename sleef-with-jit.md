# SLEEF With DSL JIT: Current State

## Purpose

This document tracks the current implementation status for SLEEF-backed math in
DSL JIT kernels, the supported lowering patterns, and the remaining gaps.

## What Is Implemented

## 1. Bridge ABI and symbol strategy

- JIT code uses stable miniexpr bridge symbols (`me_jit_*`, `me_jit_vec_*`),
  not direct SLEEF internals.
- Alias rewrites are supported in JIT expression lowering, including:
  - `arctan2` -> `atan2`
  - `exp10`, `sinpi`, `cospi`, `logaddexp`, `where` (bridge-backed helpers)
- Runtime bridge mode can be enabled with `ME_DSL_JIT_USE_SLEEF_BRIDGE=1`.

## 2. Backend integration

- `tcc` runtime path registers needed bridge symbols (`tcc_add_symbol`).
- `cc` runtime path supports bridge usage when symbols are available at runtime.
- Runtime cache keying includes backend/linking-relevant dimensions to avoid
  stale artifact reuse across incompatible settings.

## 3. Selective vector lowering

Lowering is selective and pattern-based, with scalar fallback preserved.

- Unary map patterns:
  - `sin`, `cos`, `exp`, `log`
  - `sinpi`, `cospi`, `exp10`
  - extended unary coverage in codegen tests (`abs`, `sqrt`, `log1p`,
    `exp2`, `log2`, `expm1`, `log10`, `sinh`, `cosh`, `tanh`, `asinh`,
    `acosh`, `atanh`)
- Binary map patterns:
  - `atan2`, `hypot`, `pow`
- Broadcast/constant forms:
  - `pow(x, c)`, `pow(c, x)` (safe broadcast lowering)
- Unary affine forms:
  - `f(x + c)`, `f(x - c)`, `f(c + x)` via prepass + vector call

If a kernel does not match a safe pattern, codegen falls back to scalar C.

## 4. DSL control flow and JIT subset

- DSL now supports `for` and `while` loops.
- JIT IR/C codegen supports `if`, `while`, `for`, `break`, and `continue`
  within the supported subset.
- Reductions and `print` are still intentionally rejected by JIT IR.
- Interpreter fallback remains the safety path when runtime JIT is unavailable
  or a kernel is rejected by JIT IR/codegen/runtime compile.

## 5. Safety behavior

- Interpreter `while` has an iteration cap:
  - default `10,000,000` per `while` statement
  - override via `ME_DSL_WHILE_MAX_ITERS`
  - `<= 0` disables the cap
  - cap hit returns `ME_EVAL_ERR_INVALID_ARG`

## 6. Test and benchmark coverage

- Benchmarks:
  - `bench/benchmark_dsl_jit_math_kernels.c`
- Key tests:
  - `tests/test_dsl_syntax.c`
  - `tests/test_dsl_jit_ir.c`
  - `tests/test_dsl_jit_codegen.c`
  - `tests/test_dsl_jit_runtime_cache.c`

These cover alias rewrites, bridge mode behavior, selective lowering markers,
control flow, and fallback behavior.

## Known Platform Behavior

- On wasm32, JIT IR may build successfully but runtime JIT compilation/loading
  can fail and be skipped; interpreter fallback is expected in that case.
- CI traces with `ME_DSL_TRACE=1` are useful to separate:
  - JIT IR built/rejected
  - runtime JIT built/skipped
  - fallback execution

## Important Remaining Work

1. Harden `cc` bridge-link path in packaged/runtime environments
   so bridge symbols are consistently visible across deployment modes.
2. Continue expanding selective lowering coverage where safe/profitable
   (without widening semantics risk).
3. Improve wasm32 runtime JIT reliability (or clearly scope wasm32 to
   interpreter fallback where runtime JIT is not yet robust).
4. Keep benchmark + parity tracking current as lowering coverage grows.

## Latest Benchmark Run (Local)

Run date: 2026-02-13  
Command:

- `ME_BENCH_COMPILER=tcc ./build-local/bench/benchmark_dsl_jit_math_kernels 262144 6`
- `ME_BENCH_COMPILER=cc ./build-local/bench/benchmark_dsl_jit_math_kernels 262144 6`

Notes:

- Snapshot supersedes earlier mixed-tree runs.
- `jit_warm`/`interp` are best single eval over repeats.

### `tcc` mode results (`nitems=262144`, `repeats=6`)

| Kernel | compile_ms | jit_ns_elem | interp_ns_elem | Max Abs |
|---|---:|---:|---:|---:|
| sin | 1.790 | 1.961 | 5.188 | 1.110e-16 |
| exp | 0.687 | 1.472 | 3.880 | 0.000e+00 |
| log | 0.914 | 3.391 | 3.845 | 0.000e+00 |
| pow | 0.811 | 11.147 | 11.963 | 0.000e+00 |
| hypot | 0.443 | 0.652 | 2.037 | 0.000e+00 |
| atan2 | 0.362 | 2.045 | 3.471 | 0.000e+00 |
| sinpi | 0.359 | 2.918 | 4.253 | 0.000e+00 |
| cospi | 0.366 | 2.804 | 4.189 | 0.000e+00 |

### `cc` mode results (`nitems=262144`, `repeats=6`)

| Kernel | compile_ms | jit_ns_elem | interp_ns_elem | Max Abs |
|---|---:|---:|---:|---:|
| sin | 341.261 | 1.831 | 5.424 | 1.110e-16 |
| exp | 330.707 | 3.220 | 3.349 | 0.000e+00 |
| log | 350.409 | 1.919 | 4.337 | 0.000e+00 |
| pow | 339.826 | 15.663 | 14.236 | 0.000e+00 |
| hypot | 326.106 | 1.976 | 2.941 | 0.000e+00 |
| atan2 | 332.817 | 2.769 | 5.310 | 0.000e+00 |
| sinpi | 328.903 | 3.872 | 5.642 | 0.000e+00 |
| cospi | 330.998 | 4.349 | 6.927 | 0.000e+00 |

### Discussion

1. `tcc` compile latency remains very low; `cc` remains much higher (~330-350 ms).
2. JIT warm runtime is faster than interpreter for most kernels in both modes.
3. `pow` remains the most expensive kernel and is still backend-sensitive.
4. Numerical parity remains strong (`max_abs` near zero across kernels).

## Practical Validation Checklist

For feature validation (including SLEEF/JIT behavior), run:

1. `test_dsl_syntax`
2. `test_dsl_jit_ir`
3. `test_dsl_jit_codegen`
4. `test_dsl_jit_runtime_cache` (where available)

and inspect `ME_DSL_TRACE=1` logs to confirm expected build/reject/skip/fallback
paths per backend (`tcc`, `cc`, wasm32).
