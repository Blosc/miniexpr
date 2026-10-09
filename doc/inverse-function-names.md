# Canonical inverse-function names

Native miniexpr expressions and Menudet kernels use the Array API/C inverse
function names. NumPy names remain Python frontend syntax, not native aliases.

| Removed native spelling | Canonical spelling |
| --- | --- |
| `arcsin` | `asin` |
| `arccos` | `acos` |
| `arctan` | `atan` |
| `arctan2` | `atan2` |
| `arcsinh` | `asinh` |
| `arccosh` | `acosh` |
| `arctanh` | `atanh` |

This is a deliberate source compatibility change. Full-expression and DSL
compilation rejects the removed builtin calls. Portable artifact loading also
rejects them for both checked 1.0 and numerical 1.1 profiles. Artifact import does
not rewrite source, change the saved artifact, or switch profiles. Syntax removal
does not change the values/promotion/diagnostics of the retained functions.

For existing native source, replace function-call names using the table, leaving
string literals, comments and unrelated identifiers untouched. For saved artifacts,
prefer updating original authoring and re-exporting with the same profile and
signature. Review provenance if manually modifying JSON; replacing source alone
does not certify an externally signed/hashed artifact. No automatic migration or
backward acceptance of these spellings is promised.

Python-Blosc2 keeps its public NumPy spellings and normalizes eligible frontend
calls before native compilation. Newly exported artifacts use canonical names;
raw `PortableKernel.from_json` remains a validation boundary, not a migration API.

The native function corpus retains all canonical inverse-function numerical
signature/exceptional tests and adds explicit removed-spelling rejections. Its
generator reference remains pinned NumPy 2.5.3: canonical names map to NumPy's
`arc*` ufuncs only in the reference generator. Historical corpus counts/signoff
reports remain historical, not current alias-support evidence.
