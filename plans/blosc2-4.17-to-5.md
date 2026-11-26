# Python-Blosc2 4.17 to 5.x: storage first, execution second

Status: the storage backport passed candidate qualification and the final
4.17.0 wheel run. `v4.17.0` is tagged. The execution handoff is in
development, not enabled in standard 5.x builds; Python-Blosc2 `next` now pins
miniexpr's typed graph preparation and passed an isolated-wheel integration
slice.

## Local checkout convention

| Workstream | Local checkouts (siblings under the same parent) |
| --- | --- |
| Production: miniexpr 0.4.x, Python-Blosc2 4.17 and its C-Blosc2 dependency | `../miniexpr`, `../python-blosc2`, `../c-blosc2` |
| Development: miniexpr 1.0, Python-Blosc2 5.x and C-Blosc2 4.x | `../miniexpr.codeberg`, `../python-blosc2.codeberg`, `../c-blosc2.codeberg` |

Do the 0.4.x release cut and 4.17 backports, builds and wheel qualification
from the production checkouts. Bring selected development commits into those
checkouts; do not switch the development trees to release work. For the
cross-version gate, generate the 4.17 fixture in the production environment
and reopen it in the development `next` environment. Keeping GitHub `main`
and Codeberg `main` in sync does not change this local split.

## 1. Give 4.17 a maintainable miniexpr baseline

- Select a qualified miniexpr `next` revision, review the merge against `main`,
  and merge it into `main`. Set the library headers and installed CMake package
  version to `0.4.0` on `main` only. Do not renumber the kernel API, Menudet
  language, JSON schema, or Blosc2 storage format.
- Use the release-candidate commit to build and test the 4.17 backport. Tag
  `0.4.0` only after those gates pass; publish the same tag on GitHub and
  Codeberg. Keep the production GitHub fetch URL and pin the tagged commit
  SHA. Check that dependency diagnostics refer to 4.17, not 5.0.
- Maintain `0.4.x` on `main` for Python-Blosc2 4.x. Keep `next` on the
  miniexpr 1.0 line. Cherry-pick shared fixes, not the release-version commit.
  Before making
  1.0 the default release, branch `maint/0.4.x` so 4.x fixes remain possible.

## 2. Ship the storage migration in Python-Blosc2 4.17

- Backport structured MessagePack `b2o` v2 Menudet recipes to the 4.x line.
  Use the [Menudet v2 wire contract](../../python-blosc2.codeberg/doc/development/menudet-recipe-v2.md):
  container version 2, recipe layout 1, and independent language version 0.1 or 0.2.
  Normal Menudet saves write only v2. Unrelated object kinds retain their
  current versions. New source is Menudet 0.2, with optional annotations;
  there is no authoring option to select 0.1. Give existing callers passing
  an explicit `version="0.1"` on the authoring path an actionable diagnostic.
  Audit indirect authors too: LazyUDF persistence (`menudet_from_lazyudf`),
  `prepare_menudet_kernel` callers, and capture normalization. Test that each
  new-source route emits 0.2; do not mechanically change version defaults on
  explicit JSON export or validation of existing source.
- Read legacy v1 JSON artifacts and v2 MessagePack recipes. Opening data never
  upgrades it in place. Explicitly resaving a v1 kernel writes v2 but retains
  its declared 0.1 language, fixed binding contract, scalar categories and
  numerical policy. Never synthesize annotations from a specialization.
  Explicit legacy JSON interchange is distinct from normal persistence. Neither
  that interchange nor supported v1 reads warrant a deprecation warning.
  For CTable, ordinary schema writes preserve v1 entries; an explicit
  `CTable.save()` upgrades the destination copy only, including both computed
  columns and materialized generated-column transformers.
- Keep `miniexpr_artifact` enabled in the standard 4.17 wheels, including its
  yyjson dependency. Reuse the working legacy reader and native schedulers.
  Replacing the reader with Python `json` is not a prerequisite for 4.17.
  Do not introduce a dependency on C-Blosc2 4.x for this migration.
- Release notes: older 4.x readers cannot be relied on to open new v2 records.
  For shared stores, upgrade all readers before 4.17 writers start saving.

### 4.17 release gates

1. The released Python-Blosc2 4.16.1 wheel writes the v1 fixture (generator and
   files in `../python-blosc2.codeberg/tests/data/legacy_menudet_v1`). 4.17 opens
   it without changing the carrier, evaluates it, explicitly saves v2, and
   Python-Blosc2 `next` reopens it with
   the same bindings, cardinality, dtypes and results. Assert the decoded recipe
   preserves weak versus typed captures, fixed input constraints and output
   contract, and explicit semantic overrides (e.g. `integer_powers: "multiply"`).
   Compare the contract as well as outputs: one set of values can hide a lost
   constraint or a change in promotion. Assert that a converted recipe rejects
   a changed input dtype, and record the exact Python-Blosc2 `next` SHA used
   for the cross-version check.
2. 4.17 and `next` exchange newly authored annotated 0.2 recipes and reject
   unsupported versions. Cover captures, nested operands, CFrames/files and
   CTable computed and generated columns; retain the bounded C-peer
   interoperability tests. Rebind an unconstrained parameter after reopening.
   Reject non-integer and unsupported Menudet envelope versions and
   marker/payload disagreement, including nested version failures before
   resolving sibling operands.
3. Run normal wheel/platform qualification against the exact miniexpr release
   candidate, including existing native graph and array callers. Verify the
   installed CMake version and C header agree with the proposed tag.

The local wheel gate is reproducible with
`../python-blosc2.codeberg/scripts/qualify_menudet_migration.py`: use the 4.17
wheel to `write` a directory and the `next` wheel to `check` it, then reverse
the roles with a fresh directory.

The first candidate passed that exchange on macOS arm64: Python-Blosc2
`7a31db15b14eb1f3ee6826f997ad377becbd88e7` (4.17) and
`518adb33d52181f3835b7af473b85857a294006b` (`next`).
The 4.17 wheel uses miniexpr `64e5d180904f87513cead89cbb15bff09d065cab`
and C-Blosc2 3.8. The miniexpr candidate's
[hosted qualification](https://github.com/Blosc/miniexpr/actions/runs/45776437750)
passed, and `v0.4.0` points to that commit on both remotes. The
[candidate wheel matrix](https://github.com/Blosc/python-blosc2/actions/runs/45776669778),
[native tests](https://github.com/Blosc/python-blosc2/actions/runs/45776660663),
and [WASM tests](https://github.com/Blosc/python-blosc2/actions/runs/45776660763)
passed. The
[final wheel run](https://github.com/Blosc/python-blosc2/actions/runs/45779775014)
qualified `a32d05852972009736c7a6a1a380a15088f049c7` without uploading to PyPI,
and `v4.17.0` is tagged at that commit on both remotes.

## 3. Change execution ownership for 5.x without changing storage again

- Miniexpr 1.0 `next` owns formula and Menudet preparation, block evaluation,
  grid geometry and numerical primitives. Move whole-array scheduling and
  graph planning/execution needed by Blosc2 into C-Blosc2 4.x; Python-Blosc2
  5.x consumes that implementation. A general native `b2o` operand resolver
  is host work, not a requirement for the 4.17 writer.
- Keep reading 4.17 records unchanged. For a JSON-free standard 5.x build,
  decode and validate legacy JSON in Python before native preparation.
  Check rejection behavior as well as valid values against released fixtures.
- Qualify results across chunkings, padding and reduction group boundaries,
  backend policies, and representative performance workloads. Build with
  `MINIEXPR_BUILD_ARTIFACT=OFF` and verify yyjson is neither fetched nor linked.
  Test the installed wheel, not just standalone miniexpr: reopen the released
  v1 and 4.17 v2 fixtures and run representative array and graph paths before
  removing the old scheduler.
- Audit independent C users of the optional adapter before retiring it from
  the 1.0 distribution. Its retirement is not a 4.17 release gate.

## Decisions still needed during implementation

The 5.x [execution handoff](blosc2-5-execution-handoff.md) records the dependency
boundary, current callers, implementation sequence and proposed acceptance
gates. Moving the executor must not make yyjson a required C-Blosc2 dependency.
