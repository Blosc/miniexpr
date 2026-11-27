# Execution migration implementation status

This is a checkpoint, **not completion of M0–M8**. The migration plan's
acceptance gates remain authoritative. No production default or persisted
format changed.

| Milestone | Landed work | Remaining gate / gap |
| --- | --- | --- |
| M0 | Route/test/capability inventory in `execution-migration-contracts.md`; explicit direct-handle baseline; separate benchmark preparation/specialization phases | Cross-platform performance qualification remains a later gate |
| M1 | C-Blosc2 internal `blosc/b2nd-exec.h` / `.c`; bounded typed nodes, exact captures, topology/slot/arity/width validation, copied ownership, borrowed metadata-only schedules; optional Cython conversion and strong Python schedule owner | Descriptor will need additional typed options/operations/stages as subsequent families are implemented; it is not installed/frozen |
| M2 | Serial multiple-input fused arithmetic maps and whole-domain minimum; zero-scratch direct maps; exact inferred-width reduction scratch; shared core strict FP scope and end-of-traversal raise masks; ordinary expression lowering reuses `GraphAdapter`; direct authored-kernel ownership transfer feeds the same scheduler | General `NativeGraph` metadata/frontend migration and pilot API retirement are not complete; this is the initial resident family, not the entire M2 wrapper handoff |
| M3 | Scalar/ranked same-shape contiguous domains, empty maps and explicit disjoint NumPy outputs; allocation-owner extent checks; default serial sum/prod/min/max with authoritative core accumulator typing and empty sum/product identities | Broadcasting, strided/non-native gathering, aliases, axes/masks/initial/keepdims, any/all and fixed-leaf sums stay on the adapter |
| M4 | No job/backend family migrated | C-owned parallel jobs, aggregate workspace and full backend qualification pending |
| M5 | No compressed family migrated | Existing grid pilot/adapter remain; general compressed maps/reductions pending |
| M6 | No generic source/sink family migrated | Source callbacks, logical partitions and array bridge handoff pending |
| M7 | No staged family migrated | Stage planning, intermediate budgets and staged execution pending |
| M8 | No wrapper cutover | Existing `NativeGraphHandle`, `NativeGraphSchedule` and artifact compatibility handles remain |

## Admission and ownership

Python-Blosc2's `B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE` build exposes the typed
boundary when built with the matching local C-Blosc2 header. Standard builds
compile unsupported stubs and keep existing dispatch. The checked-in dependency
pins have not been advanced to unpublished local commits.

The new ordinary path admits only explicitly interpreted, native aligned,
same-shape contiguous scalar/ranked domains (up to rank 16), whole-domain reductions
and whole-domain map reads or explicit disjoint NumPy outputs
with numeric inputs and exact typed/weak captures. Maps support add, subtract,
multiply and negate. A 64 KiB mapped-scratch admission bound uses the core-inferred
width and applies only to computed maps feeding reductions. Direct-output maps
and identity reductions need no mapped scratch. Reductions now include default
whole-domain serial sum/product/minimum/maximum. Sum/product accumulator promotion
and empty identities come from miniexpr core; map tile width stays separate from
result width. Other syntax, layouts, options and backend policies
retain their old route during metadata-only admission. Once preparation or
execution begins, errors propagate; no backend retry is performed.

The native schedule retains no operand values. Each invocation validates
capacities, pointer overflow, alignment and disjoint output before evaluation.
Minimum publishes only on success; map output may contain earlier committed
tiles and a partially written failing tile after a later failure. Input arrays are not materialized for admission.
Reports distinguish map calls from direct identity traversal and separate
mapped bytes from reducer state; core/compiler allocations are not included
in that tile bound.

Authored formula/Menudet kernels can transfer ownership into a C plan without
source rewriting or a graph descriptor. Runtime slots are queried from the
typed core, not trusted caller duplicates. Context/block-scalar kernels can be
prepared and inspected, but resident specialization explicitly rejects them
until logical-group scheduling is implemented. This does not yet migrate
`MenudetLazyArray` or its Python wrappers.

The Python typed buffer bridge checks actual owning allocations (including
external buffer extents), not a view's possibly fictitious `nbytes`. Unknown
owners are rejected without copying. Forged contiguous `as_strided` input and
output extents are tested to fail before numerical evaluation or output writes.

## Validation performed

- Artifact-off native descriptor/execution fixture, including exact int64
  captures beyond double precision, multiple inputs and width variants.
- Planner/specialization/execution allocation failure injection.
- ASan/UBSan builds of the new C-Blosc2 owner (the installed core dependency
  in this fixture was not sanitizer-instrumented).
- A C consumer linked the actual C-Blosc2 static archive with the artifact-off
  miniexpr archive and the existing platform framework dependency.
- Isolated opt-in and standard Python builds, with verified module locations;
  tests cover descriptor copying, plan/schedule lifetime, direct/computed maps,
  exact scalar transport, binding errors, changed values, shared DAGs, unreachable
  arithmetic, FP failure/recovery and no old-executor replay.
- Existing preparation/map/capture/output/source/stage Python regression suites.
- Ruff lint and formatting checks for edited Python/benchmark/test files.

JIT-dependent tests are skipped in these interpreter-focused builds. Existing
Python extension warnings include Cython unreachable/declarator warnings,
integer `abs` width warnings, const-pointer qualifier warnings and macOS CBLAS
deprecation warnings; the standalone new native owner builds with
`-Wall -Wextra -Werror`.

`bench/resident_typed_plan.py` compares frontend lowering, uncached preparation,
specialization and warmed execution through explicit handles. An initial
per-node-temporary lowering regression was identified and removed by fusing
single-use expressions while retaining shared DAG temporaries. Local Debug
smoke timings are not the ARM64/x86-64 production performance gate. Raw build
logs and timing results remain outside the repositories.
