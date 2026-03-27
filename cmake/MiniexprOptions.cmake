option(MINIEXPR_BUILD_SHARED "Build shared library" ON)
option(MINIEXPR_BUILD_STATIC "Build static library" ON)
option(MINIEXPR_BUILD_TESTS "Build tests" ON)
option(MINIEXPR_BUILD_EXAMPLES "Build examples" ON)
option(MINIEXPR_BUILD_BENCH "Build benchmarks" ON)
option(MINIEXPR_USE_SLEEF "Enable SLEEF SIMD acceleration" ON)
option(MINIEXPR_ENABLE_TCC_JIT "Enable TCC-based JIT backends (libtcc/wasm32)" ON)
option(MINIEXPR_BUILD_BUNDLED_LIBTCC "Build bundled libtcc from minicc sources when TCC JIT is enabled" ON)
option(MINIEXPR_DSL_TRACE_DEFAULT "Enable DSL trace logs by default when ME_DSL_TRACE is unset" OFF)

if(EMSCRIPTEN)
  # wasm32 JIT uses a static libtcc build; bundled host shared libtcc is not used.
  set(MINIEXPR_BUILD_BUNDLED_LIBTCC OFF)
  if(MINIEXPR_ENABLE_TCC_JIT)
    set(MINIEXPR_USE_WASM32_JIT ON)
  else()
    set(MINIEXPR_USE_WASM32_JIT OFF)
  endif()
elseif(NOT DEFINED MINIEXPR_USE_WASM32_JIT)
  set(MINIEXPR_USE_WASM32_JIT OFF)
endif()

# TCC JIT is only available for architectures that minicc has a native backend
# for. On everything else fall back silently to the non-JIT evaluator so the
# build succeeds without requiring a working libtcc.
if(NOT EMSCRIPTEN AND MINIEXPR_ENABLE_TCC_JIT)
  # Resolve the actual target processor, honouring CMAKE_OSX_ARCHITECTURES on
  # Apple (conda-forge cross-builds set that rather than CMAKE_SYSTEM_PROCESSOR).
  set(_miniexpr_proc "")
  if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(GET CMAKE_OSX_ARCHITECTURES 0 _miniexpr_osx_arch)
    string(TOLOWER "${_miniexpr_osx_arch}" _miniexpr_proc)
    unset(_miniexpr_osx_arch)
  endif()
  if(_miniexpr_proc STREQUAL "")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _miniexpr_proc)
  endif()
  # Supported minicc native backends: i386, x86_64, arm, arm64, riscv64.
  if(NOT _miniexpr_proc MATCHES "^(x86_64|amd64|i[3-6]86|x86|aarch64|arm64|arm|riscv64)$")
    message(STATUS
      "MiniExpr: TCC JIT is not supported on '${CMAKE_SYSTEM_PROCESSOR}'. "
      "Falling back to non-JIT evaluation.")
    set(MINIEXPR_ENABLE_TCC_JIT OFF CACHE BOOL
        "Enable TCC-based JIT backends (libtcc/wasm32)" FORCE)
    set(MINIEXPR_BUILD_BUNDLED_LIBTCC OFF CACHE BOOL
        "Build bundled libtcc from minicc sources when TCC JIT is enabled" FORCE)
  endif()
  unset(_miniexpr_proc)
endif()

if(NOT DEFINED MINIEXPR_WASM32_SIDE_MODULE)
  if(EMSCRIPTEN)
    set(MINIEXPR_WASM32_SIDE_MODULE ON)
  else()
    set(MINIEXPR_WASM32_SIDE_MODULE OFF)
  endif()
endif()
set(MINIEXPR_WASM32_SIDE_MODULE "${MINIEXPR_WASM32_SIDE_MODULE}" CACHE BOOL
    "Route wasm32 JIT instantiate/free via host-registered helpers (side-module mode).")

set(MINIEXPR_NEEDS_BUNDLED_TINYCC OFF)
if(MINIEXPR_ENABLE_TCC_JIT AND MINIEXPR_BUILD_BUNDLED_LIBTCC)
  set(MINIEXPR_NEEDS_BUNDLED_TINYCC ON)
endif()
set(MINIEXPR_NEEDS_TINYCC OFF)
if(MINIEXPR_NEEDS_BUNDLED_TINYCC OR MINIEXPR_USE_WASM32_JIT)
  set(MINIEXPR_NEEDS_TINYCC ON)
endif()
