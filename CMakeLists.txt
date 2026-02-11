cmake_minimum_required(VERSION 3.16.3)
project(miniexpr LANGUAGES C)
include(GNUInstallDirs)

if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  message(FATAL_ERROR "MSVC is not supported. Use clang-cl (e.g., cmake -G \"Visual Studio 17 2022\" -T ClangCL).")
endif()
if(WIN32 AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "Windows builds require clang-cl.")
endif()
if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang"
   AND DEFINED CMAKE_C_COMPILER_FRONTEND_VARIANT
   AND NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  message(FATAL_ERROR "Windows builds require clang-cl (MSVC frontend variant).")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MINIEXPR_BUILD_SHARED "Build shared library" ON)
option(MINIEXPR_BUILD_STATIC "Build static library" ON)
option(MINIEXPR_BUILD_TESTS "Build tests" ON)
option(MINIEXPR_BUILD_EXAMPLES "Build examples" ON)
option(MINIEXPR_BUILD_BENCH "Build benchmarks" ON)
option(MINIEXPR_USE_SLEEF "Enable SLEEF SIMD acceleration" ON)

if(WIN32)
  set(MINIEXPR_USE_LIBTCC_FALLBACK OFF)
else()
  set(MINIEXPR_USE_LIBTCC_FALLBACK ON)
endif()

set(MINIEXPR_SOURCES
  src/miniexpr.c
  src/dsl_parser.c
  src/dsl_jit_ir.c
  src/dsl_jit_cgen.c
  src/functions.c
  src/functions-simd.c
)

if(MINIEXPR_USE_SLEEF OR MINIEXPR_USE_LIBTCC_FALLBACK)
  include(FetchContent)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
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

if(MINIEXPR_USE_LIBTCC_FALLBACK AND NOT WIN32)
  FetchContent_Declare(
    tinycc
    # GIT_REPOSITORY https://repo.or.cz/tinycc.git
    # GIT_TAG 4597a9621e70a337b241d424f4ab4729cb75b426  # latest commit as of Feb 2025
    #GIT_REPOSITORY https://github.com/Blosc/minicc.git
    #GIT_TAG 20a9b2a0d55ff2866776ba82c40965bb307ada24
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../minicc"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/minicc-build"
  )
  FetchContent_Populate(tinycc)

  if(APPLE)
    set(MINIEXPR_TINYCC_SHARED_NAME "libtcc.dylib")
  else()
    set(MINIEXPR_TINYCC_SHARED_NAME "libtcc.so")
  endif()
  set(MINIEXPR_TINYCC_SHARED_PATH "${tinycc_BINARY_DIR}/${MINIEXPR_TINYCC_SHARED_NAME}")
  set(MINIEXPR_TINYCC_STAGED_SHARED_PATH "${CMAKE_CURRENT_BINARY_DIR}/${MINIEXPR_TINYCC_SHARED_NAME}")
  set(MINIEXPR_TINYCC_CONFIG_STAMP "${tinycc_BINARY_DIR}/.miniexpr_tinycc_configured")

  add_custom_command(
    OUTPUT "${MINIEXPR_TINYCC_CONFIG_STAMP}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${tinycc_BINARY_DIR}"
    COMMAND "${CMAKE_COMMAND}"
            -S "${tinycc_SOURCE_DIR}"
            -B "${tinycc_BINARY_DIR}"
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DMINICC_BUILD_SHARED_LIBTCC=ON"
            "-DMINICC_BUILD_WASM32_TCC=OFF"
            "-DMINICC_ENABLE_TESTING=OFF"
    COMMAND "${CMAKE_COMMAND}" -E touch "${MINIEXPR_TINYCC_CONFIG_STAMP}"
    DEPENDS "${tinycc_SOURCE_DIR}/CMakeLists.txt"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${MINIEXPR_TINYCC_SHARED_PATH}"
    COMMAND "${CMAKE_COMMAND}" --build "${tinycc_BINARY_DIR}" --target minicc_libtcc_shared
    DEPENDS "${MINIEXPR_TINYCC_CONFIG_STAMP}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${MINIEXPR_TINYCC_SHARED_PATH}" "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}"
    DEPENDS "${MINIEXPR_TINYCC_SHARED_PATH}"
    VERBATIM
  )

  add_custom_target(miniexpr_tinycc ALL
    DEPENDS "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}")

  install(FILES "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(FILES "${tinycc_SOURCE_DIR}/libtcc.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(FILES "${tinycc_SOURCE_DIR}/COPYING"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/miniexpr/third_party/tinycc")
  install(DIRECTORY "${tinycc_SOURCE_DIR}/"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/miniexpr/third_party/tinycc/source"
          PATTERN ".git" EXCLUDE)
endif()

function(miniexpr_setup_target target_name)
  target_sources(${target_name} PRIVATE ${MINIEXPR_SOURCES})
  target_include_directories(${target_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_features(${target_name} PUBLIC c_std_99)
  if(MINIEXPR_USE_SLEEF)
    target_compile_definitions(${target_name} PRIVATE ME_USE_SLEEF=1)
    target_include_directories(${target_name} PRIVATE
      # SLEEF .c includes pull helper headers multiple times; shims add include guards.
      "${CMAKE_CURRENT_SOURCE_DIR}/src/sleef_compat"
      "${sleef_SOURCE_DIR}"
      "${sleef_SOURCE_DIR}/src"
      "${sleef_SOURCE_DIR}/src/common"
      "${sleef_SOURCE_DIR}/src/arch"
    )
  else()
    target_compile_definitions(${target_name} PRIVATE ME_USE_SLEEF=0)
  endif()
  if(MINIEXPR_USE_LIBTCC_FALLBACK)
    target_compile_definitions(${target_name} PRIVATE ME_USE_LIBTCC_FALLBACK=1)
    if(NOT WIN32 AND DEFINED MINIEXPR_TINYCC_STAGED_SHARED_PATH)
      target_compile_definitions(${target_name}
        PRIVATE ME_DSL_JIT_LIBTCC_DEFAULT_PATH="${MINIEXPR_TINYCC_STAGED_SHARED_PATH}")
    endif()
  else()
    target_compile_definitions(${target_name} PRIVATE ME_USE_LIBTCC_FALLBACK=0)
  endif()
  if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_link_libraries(${target_name} PUBLIC clang_rt.builtins-x86_64.lib)
  endif()
  if(NOT WIN32)
    target_link_libraries(${target_name} PRIVATE m)
    if(UNIX AND NOT APPLE)
      target_link_libraries(${target_name} PRIVATE dl)
    endif()
  endif()
endfunction()

if(MINIEXPR_BUILD_SHARED)
  add_library(miniexpr SHARED)
  miniexpr_setup_target(miniexpr)
  if(MINIEXPR_USE_LIBTCC_FALLBACK AND NOT WIN32)
    add_dependencies(miniexpr miniexpr_tinycc)
  endif()
  if(WIN32)
    set_target_properties(miniexpr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
endif()

if(MINIEXPR_BUILD_STATIC)
  add_library(miniexpr_static STATIC)
  miniexpr_setup_target(miniexpr_static)
  if(MINIEXPR_USE_LIBTCC_FALLBACK AND NOT WIN32)
    add_dependencies(miniexpr_static miniexpr_tinycc)
  endif()
  set_target_properties(miniexpr_static PROPERTIES OUTPUT_NAME miniexpr)
endif()

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE-TINYEXPR"
  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE-SLEEF"
  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE-LIBTCC"
  "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/miniexpr/licenses")

find_package(Threads)
if(Threads_FOUND)
  set(MINIEXPR_THREADS_LIB Threads::Threads)
endif()

if(MINIEXPR_BUILD_SHARED)
  set(MINIEXPR_DEFAULT_LIB miniexpr)
elseif(MINIEXPR_BUILD_STATIC)
  set(MINIEXPR_DEFAULT_LIB miniexpr_static)
endif()

if(MINIEXPR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MINIEXPR_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(MINIEXPR_BUILD_BENCH)
  add_subdirectory(bench)
endif()
