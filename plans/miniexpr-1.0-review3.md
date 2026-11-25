# Alpha 1 migration review

Review baseline: `41aad03` on `next`; downstream integrations reviewed against
C-Blosc2 `b6f58e92` and Python-Blosc2 `e91175b4`. This is an alpha review, not a
release or a change to the public ABI.

The migration put the boundary in the right place: C constructors can describe
and fill physical blocks without JSON, while Python can preserve an authored
recipe independently of a prepared specialization. The v2 C peer tests an
exchange format; resolving stored references remains a host responsibility.
Existing formula reductions and full-DSL callers still need the compatibility
interface. Moving them solely to eliminate a second call path would change
semantics.

## Corrections

- `dsl_eval.c`: check multiplication, alignment and addition when sizing
  interpreter locals and lane masks. Cache one stable owner in each POSIX
  thread or Windows fiber slot instead of registering a reallocatable buffer.
  Bookkeeping and destruction now refer to the same owner; failure to register
  a cache falls back to invocation-local allocations.
- `miniexpr_grid.c`: any zero logical extent makes the chunk count zero,
  even if a product of the other extents alone would overflow. Physical
  geometry is still validated. Added an axis-order regression.
- Installed CMake configuration: required `core`/`artifact` components now
  reflect the targets installed. Corrected the consumer guide and the
  run-options documentation to match the actual v1 prefix.

## Is another refactor worthwhile?

Yes, as two small private changes, independently reviewable:

1. **Share buffer preflight primitives.** `kernel_preflight()` and
   `me_artifact_eval_ex()` independently implement address range, overlap,
   component alignment and active-lane Unicode checks. Share these leaf
   predicates first. Keep named versus positional binding, diagnostics and
   descriptor-version policy in their respective entry points. Exercise both
   routes with the same padded strings, masks, empty blocks and complex
   alignment cases.
2. **Make scratch a private acquire/release facility.** `dsl_eval_program_impl()`
   still interleaves backend dispatch, frame setup and statement execution.
   Move only scratch ownership, sizing and cleanup out of that function;
   keep the nested-call and allocation-failure fallback. Protect the change
   with the allocator-fault tests, worker/fiber tests and a many-small-blocks
   benchmark. There is no reason to split the evaluator by line count.

The artifact adapter's borrowed program and binding pointers plus transferred
entry-point ownership are another seam to watch. Encapsulate that handoff if
another adapter needs it; don't introduce a general adapter framework for one
caller. Leave new operation IDs, Windows export policy, formula reduction
semantics, and a production C persistence resolver for separate projects.
