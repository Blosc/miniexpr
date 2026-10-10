# Bypassing bookkeeping for simple portable DSL expressions

## Recommendation

Feasible and worthwhile for a narrowly defined single-return block-scalar
program. A temporary 16-line dispatch prototype removed most of the remaining
DSL-versus-graph gap without implementing another reducer. The prototype was
removed after assessment; the narrow production shortcut is now implemented in
`src/dsl_eval.c` with a file-local eligibility helper and unchanged JIT precedence.

Start with numeric/Boolean, portable 1.1 programs with one return statement,
positive block length, no mask, no locals, and no reserved/context variables.
Leave all other programs on the existing route. Do not initially generalize
statement execution or try to eliminate all masks throughout the interpreter.

## Why the overhead exists

`src/dsl_eval.c` builds variable/local pointer tables and initialization storage
before calling `dsl_eval_block`. That routine allocates run, break, continue and
return masks even for a program consisting only of `return sum(x)`. It initializes
these arrays, executes the return, updates per-lane return/run masks, and scans
completion masks. `p_reduce` additionally checks full participation before using
the typed loop. This introduces allocations and several O(N) passes unrelated to
the actual numeric reduction.

The input identity graph already has a specialized execution contract and does
not need this statement machinery.

## Small implementation boundary

After the existing JIT dispatch, but before allocating interpreter variable
tables in `dsl_eval_program_impl`, recognize:

- Portable 1.1, descriptor present, scalar output, positive original group size.
- No supplied mask, no locals, and variable-table count equal to input count.
- No `uses_i_mask`, `uses_n_mask`, `uses_ndim` or `uses_flat_idx` requirements.
- Exactly one compiled statement, of kind `ME_DSL_STMT_RETURN`.
- Numeric/Boolean return expression with dtype and itemsize matching the output.

Then call `dsl_portable_eval_expr` with the original input pointers, item zero,
original group size, and NULL initialization/mask pointers. This is already the
function used by the scalar return interpreter. It retains strict FP environment
handling, expression/reduction caches, typed loops, checked DSL integer overflow,
NaN fallback and final scalar writing. Keep artifact/descriptor/buffer validation
outside this shortcut, and preserve JIT precedence.

The prototype used existing compiled metadata and required no new structures.
A small file-local eligibility helper would keep the production dispatch readable;
caching eligibility at compilation is optional, since these checks are O(1).
Do not fall back after a numerical execution error: return the original error.

## Local prototype evidence

Apple M4 Pro, 1,048,576 lanes, 9 warm samples, graph tiles of 1024, JIT off.
The same native benchmark and input data were used before and after changing
only the DSL dispatch. Every call verified its result. Preparation, Python and
compression were excluded. These are local median milliseconds, not portable
speedup guarantees.

| float64 reduction | DSL before | DSL prototype | Graph in prototype run |
| --- | ---: | ---: | ---: |
| sum | 2.531 | 0.545 | 0.524 |
| prod | 2.843 | 0.846 | 1.659 |
| min | 2.697 | 0.700 | 0.722 |
| max | 2.782 | 0.778 | 0.905 |
| any | 2.576 | 0.550 | 0.576 |
| all | 2.551 | 0.549 | 0.582 |

Integer arithmetic retains an expected difference: prototype int64 sum was
0.456 ms versus graph 0.088 ms, and product 0.789 ms versus graph 0.203 ms.
DSL checks overflow at every step; graph arithmetic is modular and permits more
compiler optimization. Removing bookkeeping does not erase that semantic cost.
The float product difference is a measurement, not an established explanation.

The temporary prototype passed 440 native tests and the focused Python suite
(974 passed, 17 skipped), including reduction edge/status/fallback tests. It was
not qualified on remote CI, Windows, sanitizers or WASM. An isolated strict C
syntax check encountered two existing conditional-JIT unused-variable errors in
`dsl_eval.c`; the normal CMake build completed successfully.

## Production coverage and remaining qualification

`tests/graph/compiler-allocation.c` now proves simple eligible returns run with
every instrumented allocation rejected. It also proves masks, empty groups,
locals and elementwise outputs retain allocating bookkeeping, failures recover,
and invalid output/input capacities, names, dtypes and masks are rejected before
numerical dispatch. This allocation-free assertion is for the tested expressions,
not all eligible expression trees.

Python's `tests/test_dsl_scalar_dispatch.py` compares direct returns with local
statement execution for multiple inputs, computed operands, repeated reductions,
scalar postprocessing, mean, Boolean outputs, output conversions, all-one/partial/
all-zero masks and empty blocks. It covers FP error policy, integer error recovery,
binding validation, context and elementwise fallbacks, statement control flow,
uninitialized locals, JIT preference and concurrent immutable-kernel reuse.
Remote platform qualification remains separate from local sanitizer/WASM tests.

The implementation passed 440 native tests, 1018 focused Python tests (17 skips),
12 sanitizer tests and 10 WASM tests locally. Allocation-injection route checks
also run in the sanitizer and WASM suites. A production float64 benchmark with
the same dimensions gave DSL/graph medians of 0.576/0.576 ms for sum,
0.703/0.737 ms for min and 0.555/0.574 ms for any. No new compiler warnings were
reported by the normal builds. Remote CI has not run for this change.

Elementwise single returns could later reuse `dsl_portable_eval_expr_masked`, but
still have per-lane expression dispatch; their benefit needs separate measurement.
Support for masks/empty scalar groups requires preserving participation and
uniform-expression rules. Supporting assignments or statement control flow would
be a substantially broader project and is unnecessary for this first shortcut.
