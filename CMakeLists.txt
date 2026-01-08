cmake_minimum_required(VERSION 3.16.3)
project(miniexpr LANGUAGES C)

if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  message(FATAL_ERROR "MSVC is not supported. Use clang-cl (e.g., cmake -G \"Visual Studio 17 2022\" -T ClangCL).")
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

set(MINIEXPR_SOURCES
  src/miniexpr.c
  src/functions.c
  src/functions-simd.c
)

if(MINIEXPR_USE_SLEEF)
  include(FetchContent)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
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
  if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_link_libraries(${target_name} PRIVATE clang_rt.builtins-x86_64.lib)
  endif()
  if(NOT WIN32)
    target_link_libraries(${target_name} PRIVATE m)
  endif()
endfunction()

if(MINIEXPR_BUILD_SHARED)
  add_library(miniexpr SHARED)
  miniexpr_setup_target(miniexpr)
endif()

if(MINIEXPR_BUILD_STATIC)
  add_library(miniexpr_static STATIC)
  miniexpr_setup_target(miniexpr_static)
  set_target_properties(miniexpr_static PROPERTIES OUTPUT_NAME miniexpr)
endif()

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
