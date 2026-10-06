if(MINIEXPR_USE_SLEEF OR MINIEXPR_NEEDS_TINYCC OR MINIEXPR_BUILD_ARTIFACT)
  include(FetchContent)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
endif()

if(MINIEXPR_BUILD_ARTIFACT)
  FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 8b4a38dc994a110abaec8a400615567bd996105f # release 0.12.0
  )
  # Compile the reader in the adapter, not in the language/compiler library.
  FetchContent_Populate(yyjson)
endif()

if(MINIEXPR_USE_SLEEF)
  FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    #GIT_TAG 3.9.0
    GIT_TAG 7623d6cfa2712462880fa63a4d0f0b5f775d1a83  # latest commit as of Jan 2025
    GIT_SHALLOW TRUE
    # in case you want to use a local copy of c-blosc2 for development, uncomment the line below
    # SOURCE_DIR "/Users/faltet/blosc/sleef"
  )
  FetchContent_Populate(sleef)
endif()

if(MINIEXPR_NEEDS_TINYCC)
  # Shared tinycc source declaration for both host fallback and wasm32 JIT builds.
  FetchContent_Declare(
    tinycc
    # GIT_REPOSITORY https://repo.or.cz/tinycc.git
    # GIT_TAG 4597a9621e70a337b241d424f4ab4729cb75b426  # latest commit as of Feb 2025
    # minicc is a fork of tinycc with support for wasm32 and shared library builds
    GIT_REPOSITORY https://github.com/Blosc/minicc.git
    GIT_TAG ffbb8ebff5586b89f545826142a911a932b95315
    # SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../minicc"
  )
  FetchContent_Populate(tinycc)
endif()
