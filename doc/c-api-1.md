# 1.0 C API boundary and export audit

The stable embedding surface is intentionally smaller than the symbol table.
All clients rebuild for 0.x to 1.0. The library is still an alpha; these headers
define the contract to freeze at 1.0, not binary compatibility with alpha pilots.

| Header | Contract |
| --- | --- |
| `miniexpr_types.h` | Explicit dtype numbers 0–15, version macros and FP flags. |
| `miniexpr_kernel.h` | Typed signatures/captures, preparation, describe/eval/free, invocation-local execution policy. |
| `miniexpr_grid.h` | Storage-neutral geometry, block description, padded evaluation and destruction. |
| `miniexpr_reduce.h` | Streaming ordered pairwise sums and header-only embedding reduction helpers. |
| `miniexpr.h` | Compatibility formulas, C callbacks, full-DSL and ND entry points, retained through 1.x. |
| `miniexpr_artifact.h`, `miniexpr_graph.h` | Separately versioned JSON/array/graph adapters; do not infer their stability from the core ABI version. |

The core's external functions are `me_prepare_formula`, `me_prepare_menudet`,
`me_describe`, `me_eval_block`, `me_kernel_free`, `me_grid_create`,
`me_grid_describe_block`, `me_grid_eval_block`, `me_grid_free`,
`me_pairwise_sum_init`, `me_pairwise_sum_update`, `me_pairwise_sum_append`,
`me_pairwise_sum_leaves`, and `me_pairwise_sum_finish`.

`me_status` numbers are fixed at 0 and -1 through -6. Buffer views passed by
value have fixed pointer/size layouts. Extensible descriptors cross the ABI by
pointer, retain fixed minimum prefixes and default absent suffixes. Reserved
bytes must be initialized to zero and may not be reused as future fields.
Current and frozen-caller tests cover these rules on native and WASM32 builds.
Invocation-local interpreter selection is appended beyond the entire original
run-options prefix, with an explicit reserved tail.

## Compatibility is deliberate, not dead code

The unused checked-arithmetic profile and its convenience typing entry point
have been removed. Its control-flow, masking, initialization, diagnostic and
threading tests now use public Menudet semantics. Independent modular-overflow,
Boolean-promotion and rounding expectations replace historical checked results.
Bounds checks, weak-scalar/range conversions and ordered-reduction overflow
errors remain part of the public language; they are not obsolete profiles.

Full-DSL compilation still has compatibility consumers, including generated
source and classic reduction callers. Deleting that engine would change their
language and error behavior. It is therefore isolated behind `miniexpr.h`,
not silently relabelled Menudet. New formula/grid headers do not include that
callback-bearing header. Shared grid geometry and typed preparation are used by
the migrated constructor, recipe, discovery and ordinary formula paths.

Formula reductions remain explicitly unsupported by the new API. This release
does not choose their grouping or numerical semantics through a migration shim.

## Why exported private symbols remain

The JSON/array/graph adapter is a separate library and calls compiler/evaluator
internals in the core. Generated JIT bridges also use private helper addresses.
Hiding all `dsl_*`, `private_*` or SIMD symbols would break that arrangement;
their current visibility is not an ABI promise to embedding clients.
Windows `WINDOWS_EXPORT_ALL_SYMBOLS` is retained for those cross-library users.

Builtin dispatch also still uses function-pointer identities. Consequently
ClangCL Release builds retain `/OPT:NOICF`; removing it without replacing those
identities can merge distinct operations. Windows x64/ARM64 CI qualifies the
actual configuration. Symbol hiding or dropping the ICF guard is **not** a
safe cleanup-only change. A future private operation-ID refactor can tighten
exports without changing the supported embedding surface.

Source-language versions (0.1/0.2), JSON artifact format 0.1, b2o record versions,
and `ME_KERNEL_API_VERSION` are independent. Standard Blosc2 builds retain the
v1 Menudet JSON reader. The structured v2 fixture subset and its explicit limits
are documented in [recipe interoperability](recipe-interop.md).
