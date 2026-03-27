# Internal Architecture Notes

Last updated: 2026-03-27

## Goal

Name the current internal module boundaries so future structural refactors can move code without reopening the whole design discussion each time.

## Current boundaries

### Public API and classic evaluator

Primary files:

- `src/miniexpr.h`
- `src/miniexpr.c`
- `src/miniexpr_internal.h`
- `src/miniexpr_eval_nd.c`
- `src/miniexpr_eval_dsl_nd.c`
- `src/miniexpr_eval_reduce.c`
- `src/functions.h`
- `src/functions.c`

Responsibilities:

- public compile/eval/free entry points
- classic expression parsing and AST construction
- dtype inference and promotion
- scalar/vector evaluation
- reduction helpers and scalar conversion logic
- classic ND block/chunk evaluation
- DSL-specific ND block/chunk evaluation dispatch

### DSL frontend and runtime

Primary files:

- `src/dsl_parser.c`
- `src/dsl_compile.c`
- `src/dsl_compile_support.c`
- `src/dsl_eval.c`
- `src/dsl_jit_ir.c`
- `src/dsl_jit_cgen.c`

Responsibilities:

- parse DSL kernels
- resolve DSL dtypes
- build JIT IR
- emit JIT C source
- interpret DSL when JIT is skipped or unavailable
- provide shared DSL runtime helpers used by the ND DSL evaluator

### JIT policy and backend layer

Primary files:

- `src/dsl_config.h`
- `src/dsl_jit_runtime_host.c`
- `src/dsl_jit_runtime_nonhost.c`
- `src/dsl_jit_runtime_cache.c`
- `src/dsl_jit_backend_cc.c`
- `src/dsl_jit_backend_libtcc.c`
- `src/dsl_jit_backend_wasm32.c`

Responsibilities:

- central env/policy parsing helpers
- runtime cache and metadata
- host `cc`, libtcc, and wasm32 backend execution
- platform loader/compiler glue

### SIMD math backend

Primary files:

- `src/functions-simd.c`
- `src/functions-simd.h`
- `src/sleef_compat/*`

Responsibilities:

- runtime SIMD backend selection
- scalar fallback math paths
- architecture/SLEEF-specific vector backends

## Desired next boundary shifts

- separate more classic compile/eval helpers from public API glue in `src/miniexpr.c`
- narrow `src/miniexpr_internal.h` so it does not become a catch-all private header
- decide whether scalar conversion and reduction helpers deserve narrower private headers of their own
- keep backend-specific JIT logic behind a narrower runtime interface
- keep policy parsing centralized and avoid new direct env reads outside `src/dsl_config.h` unless clearly justified
