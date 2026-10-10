# Portable JIT improvements: correctness and legacy parity

## Objective

Bring the portable Menudet 1.1 JIT to feature parity with the useful numeric
legacy DSL JIT routes before making portable DSL the default Python execution
path. Preserve portable interpreter semantics, diagnostics, lazy participation,
and safe fallback; compilation alone is not evidence of correctness or speed.

The first application-level acceptance case is
`python-blosc2/bench/ndarray/jit-dsl-mandelbrot.py`: its per-element loop and
escape control flow must execute through portable JIT rather than interpretation.

This document is an implementation plan, not a claim that the work is complete.
The existing release dependency pin and Python default routing remain unchanged
until the release gates below are met.

## Baseline and evidence

The current portable lowerer is `src/dsl_portable_jit.c`. It accepts numeric
elementwise assignments, returns, and conditional statements, but rejects loops
and scalar-output programs. Its expression lowering includes modular integer
arithmetic, strict floating-point bridges, and specialization of selected
immutable weak integer/Boolean captures.

Legacy references are `src/dsl_jit_ir.c`, `src/dsl_jit_ir.h`, and
`src/dsl_jit_cgen.c`. These are implementation references, not normative portable
semantics: copying their unchecked casts or arithmetic is not sufficient.

Local compilation probes using Clang and TCC established the following matrix.
These are sampled cases; phase P0 must turn them into a systematic inventory and
regression suite, including dtype and operand-category variants.

| Feature | Legacy JIT | Portable 1.1 JIT |
| --- | --- | --- |
| Numeric expressions, locals, conditionals, early returns | Supported | Supported |
| `for` with `range`, `while`, nested loops | Supported | Interpreter fallback |
| Loop `break`, `continue`, returns | Supported | Interpreter fallback |
| Float operand converted with `int` | Supported | Interpreter fallback |
| Integer-result `abs`, `sign`, `square`, `fac`, `ncr`, `npr`, named `pow` | Sampled cases supported | Interpreter fallback |
| Integer-input `floor`, `round`, `real`, `imag`, `conj` | Sampled cases supported | Interpreter fallback |
| `ldexp`, `fma` | Sampled cases supported | Interpreter fallback |
| Mixed signed/unsigned comparisons | Sampled case supported | Interpreter fallback |
| Block reductions | Interpreter fallback | Interpreter fallback/native helpers |
| Integer `**` | Sampled case falls back | Sampled case falls back |
| Standalone expression statements, `print` | Unsupported by legacy JIT | Interpreter fallback |

Additional restrictions:

- Unknown/computed weak operands and floating captures still have conversion
  fallback cases; immutable integer/Boolean leaf specialization is not general
  weak-scalar support.
- Portable source preparation is disabled under `__EMSCRIPTEN__`. Legacy has a
  WASM adapter, but that adapter has its own limitations.
- Both lowerers lack general complex/fixed-string JIT coverage. Complex is outside
  the portable language. These are not numeric legacy-parity requirements.
- Arbitrary user compiler flags remain an intentional fail-closed restriction
  until their effect on strict semantics can be qualified.

### Demonstrated correctness defects

1. An int64 comparison of `2**53 < 2**53 + 1` is true under interpretation but
   false under portable JIT. Integer operands must not pass through a `double`
   comparison ABI.
2. A computed weak integer intermediate `y + 1`, where `y = INT64_MAX`, reports
   overflow under interpretation but wraps under portable JIT. Typed modular
   arithmetic must not erase checked weak-scalar semantics.

Fix these before extending eligibility. Reproduce through production artifact
execution as well as lower-level compiled-program tests.

## Invariants and design constraints

- Treat the portable interpreter and its specified semantic profile as the
  reference, including dtype promotion, weak categories, overflow, FP status,
  rounding, and errors. Keep strong typed modular arithmetic distinct from
  checked weak arithmetic and checked integer reductions.
- Preserve operand evaluation count, ordering where specified, and conditional
  participation. Errors in unselected branches, skipped lanes, or unexecuted
  loop bodies must not become observable.
- Preserve logical ND context, uniform/capture inputs, masks, output dtype and
  cardinality, empty inputs, and non-contiguous/broadcast traversal behavior.
- Keep artifacts and plans pointer-free. Runtime bridge tables and compiled
  entrypoints remain process-local; never serialize host addresses.
- Keep preparation transactional. Allocation or backend compilation failure
  must leave a usable interpreter program, not partially installed JIT state.
- Do not introduce dependencies, fast-math, parallel reductions, or schema
  changes merely to obtain parity. Any ABI change needs a documented compatibility
  decision, cache invalidation, and all callers updated together.
- Require executed-route assertions. A JIT request or prepared source does not
  prove the kernel ran; tests must detect silent interpreter fallback.

## P0 — Inventory and executable parity contract

1. Enumerate portable operators, builtins, statements, dtypes, and operand
   categories against both lowerers. Classify each combination as supported,
   semantic fallback, backend limitation, language exclusion, or untested.
2. Add table-driven tests for the matrix above. Include representative float32,
   float64, bool, signed and unsigned widths, mixed categories, and captures.
3. Record compilation capability separately from actual execution and correctness.
   Compare portable JIT to portable interpretation, not to potentially different
   legacy arithmetic semantics.
4. Add reproducible route diagnostics for fallback reasons. Prefer existing trace
   facilities unless a public reporting API is separately justified. Do not make
   correctness tests depend on human-oriented log wording.

Acceptance: every planned feature has a failing or explicitly skipped coverage
case and a stated semantic owner; legacy support is not inferred from syntax alone.

## P1 — Fix existing correctness defects and establish checked execution

### Exact integer comparison

- Split integral comparison lowering from floating comparison bridges.
- For same-domain integral operands, emit exact typed comparisons without
  conversion to floating point. Audit all six comparison operators and bool.
- Implement mixed signed/unsigned comparisons using the interpreter's exact
  rules, including negative signed operands and unsigned values above INT64_MAX;
  do not rely on C's usual signed/unsigned conversions.
- Audit mixed integer/float comparisons independently against the portable
  semantic contract. Do not assume the same lowering is suitable for both.
- Retain floating NaN/FP-status handling on the qualified host bridge route.

Tests: adjacent values around `2**53`, signed and unsigned extrema, equal values,
negative-vs-unsigned operands, all widths/operators, branch conditions, masks,
and graph map expressions using the same lowerer.

### Checked weak arithmetic and execution status

- Carry the distinction between checked weak intermediates and modular typed
  intermediates into lowering eligibility and generated operations.
- First fail closed for the demonstrated unsafe case if a complete checked route
  is not ready. Add the regression before relaxing eligibility.
- Audit the existing JIT invocation/status ABI in `src/dsl_eval.c` and bridge
  functions in `src/dsl_portable_expr.c`. Specify a per-evaluation error mechanism
  that can represent conversion, arithmetic, and loop errors without global state.
- If the existing ABI cannot carry all required diagnostics, choose explicitly
  between an invocation-local status bridge/context and a versioned internal
  entrypoint. Do not silently overload existing return or input slots.
- Define failing-lane selection and partial-output behavior to match the existing
  interpreter contract. Stop evaluation where required; do not rerun a failed
  JIT invocation through interpretation as if it were a compilation fallback.
- Ensure checked operations evaluate operands once and do not invoke signed C
  overflow or out-of-range floating-to-integer casts.

Tests: weak overflow/underflow, strong modular controls, computed vs direct
captures, errors hidden by branches/masks, repeated and concurrent evaluations,
and error recovery on a later valid invocation.

Acceptance: demonstrated defects are fixed or safely excluded; interpreter/JIT
values and diagnostics agree on all new boundary tests.

## P2 — Per-element loop and control-flow JIT

### Lowering structure

- Extend `pj_block` with an explicit control-flow context: loop stack, unique
  labels, return target, and path-sensitive definite-assignment information.
- Replace the current return-as-`continue` emission with an explicit lane exit
  target or equivalent structured mechanism. Inside a user loop, `continue`
  would otherwise target that loop instead of completing the lane.
- Track normal, returned, broken, and continued paths separately. Merge only
  reachable paths; account for zero-iteration loops and locals assigned only
  inside a body. Never read uninitialized C locals.
- Keep loop counters and user locals lane-local. Preserve typed assignment
  rounding and the interpreter's lifetime/visibility rules for loop variables.

### `for` / `range`

- Implement every range form accepted by portable semantics, with start, stop,
  and step evaluated at the correct time and exactly once where required.
- Support positive/negative steps, empty ranges, nesting, and legal runtime
  bounds. Preserve zero-step errors and range conversion checks.
- Avoid signed overflow in counter advancement and termination calculations;
  match the interpreter's behavior at int64 boundaries.
- Preserve all portable loop/resource limits rather than assuming finite range
  syntax is sufficient protection.

### `while`, `break`, `continue`, and return

- Evaluate while conditions at the same points as interpretation and preserve
  short-circuit/error behavior.
- Implement the interpreter's iteration budget, including its exact boundary
  and exhaustion diagnostic. Do not add a different undocumented limit.
- Target `break` and `continue` to the innermost user loop; for-loop continue must
  still perform counter advancement, and while-loop continue must recheck its
  condition and respect its budget.
- Support returns within nested loops and conditionals without writing output
  twice or falling through into later statements.

Tests: nested loops, every exit kind, zero/one/many iterations, negative ranges,
budget boundaries, checked errors in conditions/bodies, masked failing lanes,
uniform captures, and ND context in loops. Explicitly test return-vs-continue
targeting and definite assignment after zero-iteration loops.

Acceptance: the Python Mandelbrot workload executes portable JIT and matches
portable interpretation and the established benchmark output. Equivalent native
portable-artifact tests must prove execution under TCC and available system CCs.
Retain the legacy native Mandelbrot benchmark as a reference; do not silently
change its API or interpretation baseline.

## P3 — Checked conversions and complete numeric builtin coverage

1. Implement float-to-integer conversion with finite/range checks before any C
   cast. Cover `int` and other accepted integer conversion forms according to
   their individual checked/modular contracts, including float32 source rounding.
2. Lower integer identity-like operations directly where their result dtype and
   semantics permit it: `floor`, `ceil`, `trunc`, `round`, `real`, `imag`, `conj`.
   Verify coverage rather than assuming every function has identical semantics.
3. Add exact integer-result `abs`, `sign`, and `square`. Handle signed minima and
   distinguish modular typed arithmetic from checked weak arithmetic.
4. Add `fac`, `ncr`, `npr`, and named `pow` using audited typed helpers or bridges
   where needed. Preserve domain, overflow, promotion, and diagnostic rules.
5. Add `ldexp` with exact exponent handling and `fma` with a qualified ternary
   route. Preserve fused semantics and FP flags; do not rewrite fma as multiply
   followed by add. Audit operand order/evaluation count in generated calls.
6. Extend the inventory to all remaining builtins, not just the sampled gaps.
   Treat integer `**` as an additional feature if both legacy and portable lack it.

Tests: dtype cross-products where practical, finite limits, infinities, NaNs,
signed zero, subnormals, boundary casts, FP flags, operand errors, lazy branches,
and executed JIT assertions. Share interpreter helpers only when the bridge
preserves exact categories and does not introduce lossy `double` transport.

Acceptance: each previously confirmed numeric fallback has either a qualified
JIT implementation or a specifically documented semantic/backend blocker.

## P4 — General weak operands, captures, and cache correctness

- Build on P1 checked execution to support computed and runtime weak operands;
  do not pretend they are immutable leaves.
- Add floating capture/conversion cases only after specifying range, rounding,
  and category preservation. Out-of-range values must retain portable errors.
- Keep artifact validation ahead of specialization. Include every embedded value,
  dtype/category, backend semantic option, and generated ABI version in relevant
  cache identity. Distinguish same-valued captures with different semantic types.
- Keep runtime-only values out of compile-time specialization unless immutability
  is proven; test reuse with different values, round trips, and concurrent calls.
- Avoid generated-source explosion; retain safe fallback for existing source,
  recursion, bridge-table, and variable limits until separately justified.

Acceptance: broader eligibility does not alter portable artifacts, leak pointers,
or reuse an incompatible compiled kernel. Graph and DSL tests cover shared lowering.

## P5 — Backend portability, including WASM

- Qualify each feature incrementally on TCC and available Clang/GCC backends;
  compiler-specific success must not enable unqualified routes globally.
- Audit Windows loading, bridge calling conventions, integer widths, and cache
  behavior. Keep separate platform projects such as Windows ARM64 TinyCC outside
  this plan's implicit scope; report unsupported combinations explicitly.
- Inventory the legacy WASM adapter ABI before enabling portable preparation.
  Design WASM-compatible numeric, mask, ND-context, checked-status, and math
  bridges rather than transplanting host function-pointer assumptions.
- Enable WASM in stages: simple elementwise programs, exact integer operations,
  checked conversions, control flow, and the Mandelbrot acceptance kernel.
- Add native/WASM differential tests, especially i64/u64 and float-to-int cases.
  Retain interpreter fallback for features the adapter cannot represent exactly.

Acceptance: advertised backend capabilities have executed tests; unsupported
backends fail closed with an actionable reason instead of partial ABI support.

## P6 — Optional reduction JIT, beyond legacy parity

This is a separate enhancement and is not required to close the Mandelbrot gap.
The current typed native reduction helpers remain the performance baseline.

- Start with eligible block-scalar sum/product/extrema/truth reductions, then
  consider expression reductions, multiple reductions, and locals/control flow.
- Preserve serial order, per-operation rounding, NaN behavior, checked integer
  overflow, empty-input behavior, mask semantics, and original block partitions.
- Preserve DSL-vs-graph arithmetic differences. Do not transfer graph modular
  accumulation into checked DSL reductions or claim graph tiling is equivalent
  to the DSL's original-group contract.
- Keep mean and unsupported axes/layouts on qualified fallback until explicitly
  implemented. Avoid reassociation, parallel reduction, and speculative evaluation.
- Benchmark against optimized interpreter helpers before accepting complexity;
  whole-block JIT may not improve simple direct-input reductions.

Acceptance: numerical/diagnostic parity plus measured benefit for stated cases;
no universal claim that JIT, graph, or DSL is fastest.

## Qualification and release gates

### Native tests

- Add focused coverage to `tests/graph/preparation.c`, portable interpreter and
  validation tests, artifact tests, and dedicated control-flow tests as needed.
- Extend `tests/graph/compiler-allocation.c` sweeps to new source builders,
  metadata, bridge contexts, and cleanup paths. Cover compiler/backend failure.
- Run targeted tests during each phase, then the full native suite, sanitizers,
  strict changed-C checks, and the WASM suite where relevant.
- Test that errors, masks, and failed compilation do not pollute subsequent
  evaluations or process floating-point state.

### Python integration

- Use the `blosc2` conda environment and an explicit local/native revision
  override for integration builds; do not change the release pin prematurely.
- Add source and exported-artifact tests for loops, checked conversions, weak
  captures, graph maps, errors, partitions, and concurrency.
- Exercise the existing `@blosc2.jit` benchmark through a deliberately selected
  portable path before changing the default; record the actual execution route.
- Run targeted tests and the full relevant Python suite; include exact-pair CI
  qualification when publication is separately authorized.

### Performance

- Keep all existing backend columns in
  `bench/benchmark_dsl_interpreter_vs_jit.c`; add representative loop and newly
  supported math cases without conflating graph and DSL capability.
- Measure cold preparation/compilation and warm execution separately. Report
  sizes, repetitions, compiler/backend, execution route, and result validation.
- Measure Mandelbrot at multiple shapes and iteration limits, including early
  escape and long-running pixels. Compare portable interpreter, portable JIT,
  legacy JIT, and the NumPy reference where applicable.
- Set performance targets from reproducible baselines rather than inventing
  thresholds. Investigate material regressions before default-route migration.

### Completion criteria

P1-P4 plus qualification of the intended release backends are the core native
numeric parity milestone. P5 WASM and P6 reductions are separately reported
milestones, not hidden prerequisites or implied accomplishments.

Before a Python default switch: resolve the demonstrated correctness defects,
qualify Mandelbrot and the parity matrix, document all remaining fallback cases,
verify safe operation without a JIT compiler, and pass the release platform and
exact native/Python revision matrix. Keep execution/fallback reporting available
so downstream benchmarks cannot silently claim JIT performance while interpreting.

## Suggested implementation order

1. P0 regression inventory plus immediate fail-closed protection for P1 defects.
2. P1 exact comparisons and checked execution/status design.
3. P2 range loops and exits, then while loops and full Mandelbrot qualification.
4. P3 checked casts, inexpensive integer builtins, then multi-operand math.
5. P4 general weak operands and specialization/cache qualification.
6. Complete intended host backend/release gates; pursue P5 WASM separately.
7. Evaluate P6 reductions only after parity work and performance evidence.
