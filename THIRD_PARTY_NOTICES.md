## Third-Party Notices

This project includes or depends on third-party components with separate licenses.

### yyjson (optional artifact adapter only)

- Component: strict UTF-8 JSON reader for portable DSL artifacts
- Upstream: https://github.com/ibireme/yyjson
- Version: 0.12.0 (`8b4a38dc994a110abaec8a400615567bd996105f`)
- License: MIT
- The upstream `LICENSE` is installed as `LICENSE-YYJSON` when
  `MINIEXPR_BUILD_ARTIFACT` is enabled. It is not a dependency of the raw DSL compiler.

### TinyExpr

- Component: parser/evaluator base design and code portions
- Upstream: https://github.com/codeplea/tinyexpr
- License: zlib
- Local license file: `LICENSE-TINYEXPR`

### SLEEF

- Component: SIMD math kernels
- Upstream: https://github.com/shibatch/sleef
- License: Boost Software License 1.0
- Local license file: `LICENSE-SLEEF`

### TinyCC / libtcc

- Component: DSL JIT in-memory compiler backend (`tcc`, powered by `libtcc`)
- Upstream: https://repo.or.cz/tinycc.git
- License: GNU LGPL v2.1 or later
- Local license file: `LICENSE-LIBTCC`

For installed binaries, the corresponding TinyCC source and license are staged at:

- https://repo.or.cz/w/tinycc.git`
- https://repo.or.cz/tinycc.git/blob/HEAD:/COPYING
