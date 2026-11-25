# Portable artifact fixtures

These hand-authored Menudet 0.1 / artifact schema 0.1 fixtures are owned by miniexpr and need
no Python source/module, frontend rewriting, or Python interpreter. The normative
format is [artifact-0.1.md](../../doc/dsl-spec/artifact-0.1.md).

Enable the separately linked JSON adapter:

```sh
cmake -S . -B build-artifacts -DMINIEXPR_BUILD_ARTIFACT=ON
cmake --build build-artifacts --target portable_artifact_runner test_dsl_artifact
ctest --test-dir build-artifacts -R artifact --output-on-failure
build-artifacts/tests/portable_artifact_runner tests/menudet-artifacts/affine.json off
```

The demonstration runner supplies float64 input `[0, 1, 2, 3]`, verifies output
`[-1, 1, 3, 5]`, and reports interpreter execution. `on` requests optional JIT,
with interpreter fallback if lowering or a backend is unavailable.
Tests generate a CC-preference variant without changing the source computation.

`test_dsl_artifact.c` covers scalar widths/boundaries, non-finite floats, signed
zero, multi-tile broadcast, empty and constant-only kernels, ownership after JSON
release, reordered runtime bindings, invalid field/type/version/capability data,
duplicate keys (also in metadata), malformed UTF-8/JSON, invalid encodings, source
rejection, and missing-return runtime errors. The raw DSL corpus remains separate.

The adapter target is available in the build tree as `miniexpr::artifact`
(`miniexpr_artifact` remains available). Standalone installs include its archive
and public headers. yyjson is not linked into
the core language library and is not fetched when artifact support is disabled.
