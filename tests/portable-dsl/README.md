# Initial portable DSL conformance corpus

These are raw native source fixtures, not exported artifacts. All inputs and
outputs currently use `float64`. Each `.txt` file contains whitespace-delimited:

1. Element count and input count.
2. Input names in compilation/evaluation order (possibly different from source).
3. One row per element: input values followed by the specified expected output.

The initial runner limits cases to 4096 elements and 32 inputs. Numeric comparison
uses `abs(actual - expected) <= 1e-12 + 1e-12 * abs(expected)`.

Build miniexpr with tests enabled, then run:

```sh
build/tests/portable_dsl_runner tests/portable-dsl/affine.dsl tests/portable-dsl/affine.txt off
```

The final argument is `off` (require interpreter), `on` (require a prepared JIT
kernel), or `default` (allow the normal best-effort policy). The runner emits its
JIT status and computed values. It links only native miniexpr, never libpython.
CTest always runs interpreter cases and adds required-JIT cases when native TCC
is enabled. Python-Blosc2 consumes these files from the authoritative checkout
or the directory selected with `MINIEXPR_PORTABLE_CORPUS`.
