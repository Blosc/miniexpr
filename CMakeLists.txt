cmake_minimum_required(VERSION 3.16.3)
project(miniexpr LANGUAGES C)
include(GNUInstallDirs)

if(WIN32 AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "Windows builds require clang-cl.")
endif()
if(WIN32
   AND DEFINED CMAKE_C_COMPILER_FRONTEND_VARIANT
   AND NOT CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  message(FATAL_ERROR "Windows builds require clang-cl (MSVC frontend variant).")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/MiniexprOptions.cmake)

set(MINIEXPR_SOURCES
  src/miniexpr.c
  src/dsl_compile.c
  src/dsl_compile_support.c
  src/dsl_eval.c
  src/dsl_parser.c
  src/dsl_jit_ir.c
  src/dsl_jit_cgen.c
  src/dsl_jit_backend_cc.c
  src/dsl_jit_backend_libtcc.c
  src/dsl_jit_backend_wasm32.c
  src/dsl_jit_runtime_cache.c
  src/dsl_jit_runtime_host.c
  src/dsl_jit_runtime_nonhost.c
  src/functions.c
  src/functions-simd.c
)

include(cmake/MiniexprFetchDeps.cmake)
include(cmake/MiniexprTinyccTargets.cmake)

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
  if(MINIEXPR_ENABLE_TCC_JIT)
    target_compile_definitions(${target_name} PRIVATE ME_USE_LIBTCC_FALLBACK=1)
    if(DEFINED MINIEXPR_TINYCC_STAGED_SHARED_PATH)
      target_compile_definitions(${target_name}
        PRIVATE ME_DSL_JIT_LIBTCC_DEFAULT_PATH="${MINIEXPR_TINYCC_STAGED_SHARED_PATH}")
    endif()
  else()
    target_compile_definitions(${target_name} PRIVATE ME_USE_LIBTCC_FALLBACK=0)
  endif()
  if(MINIEXPR_DSL_TRACE_DEFAULT)
    target_compile_definitions(${target_name} PRIVATE ME_DSL_TRACE_DEFAULT=1)
  endif()
  if(MINIEXPR_USE_WASM32_JIT)
    target_compile_definitions(${target_name} PRIVATE ME_USE_WASM32_JIT=1)
    if(MINIEXPR_WASM32_SIDE_MODULE)
      target_compile_definitions(${target_name} PRIVATE ME_WASM32_SIDE_MODULE=1)
    else()
      target_compile_definitions(${target_name} PRIVATE ME_WASM32_SIDE_MODULE=0)
    endif()
    target_include_directories(${target_name} PRIVATE "${tinycc_SOURCE_DIR}")
    target_link_libraries(${target_name} PRIVATE miniexpr_wasm_libtcc)
  endif()
  if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(_miniexpr_builtins_names)
    if(CMAKE_VS_PLATFORM_NAME STREQUAL "ARM64"
       OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|AARCH64|arm64|aarch64)$")
      list(APPEND _miniexpr_builtins_names
        clang_rt.builtins-aarch64
        clang_rt.builtins-arm64)
    elseif(CMAKE_VS_PLATFORM_NAME STREQUAL "x64"
           OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|X86_64|amd64|x86_64)$")
      list(APPEND _miniexpr_builtins_names clang_rt.builtins-x86_64)
    endif()
    if(_miniexpr_builtins_names)
      unset(_miniexpr_builtins_lib CACHE)
      execute_process(
        COMMAND "${CMAKE_C_COMPILER}" -print-resource-dir
        OUTPUT_VARIABLE _miniexpr_clang_resource_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(_miniexpr_clang_resource_dir)
        foreach(_miniexpr_builtins_name IN LISTS _miniexpr_builtins_names)
          find_file(_miniexpr_builtins_lib
            NAMES "${_miniexpr_builtins_name}.lib"
            PATHS "${_miniexpr_clang_resource_dir}/lib/windows"
            NO_DEFAULT_PATH
          )
          if(_miniexpr_builtins_lib)
            break()
          endif()
        endforeach()
      endif()
      if(NOT _miniexpr_builtins_lib)
        find_library(_miniexpr_builtins_lib NAMES ${_miniexpr_builtins_names})
      endif()
      if(_miniexpr_builtins_lib)
        target_link_libraries(${target_name} PUBLIC "${_miniexpr_builtins_lib}")
      else()
        message(STATUS
          "MiniExpr: clang_rt builtins library not found for ${CMAKE_VS_PLATFORM_NAME}; using default toolchain runtime.")
      endif()
      unset(_miniexpr_builtins_lib CACHE)
      unset(_miniexpr_builtins_name)
      unset(_miniexpr_clang_resource_dir)
    endif()
    unset(_miniexpr_builtins_names)
  endif()
  if(NOT WIN32)
    target_link_libraries(${target_name} PRIVATE m)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_libraries(${target_name} PRIVATE dl)
    endif()
  endif()
endfunction()

function(miniexpr_add_library_target target_name target_type)
  add_library(${target_name} ${target_type})
  miniexpr_setup_target(${target_name})
  if(TARGET miniexpr_tinycc)
    add_dependencies(${target_name} miniexpr_tinycc)
  endif()
endfunction()

if(MINIEXPR_BUILD_SHARED)
  miniexpr_add_library_target(miniexpr SHARED)
  if(WIN32)
    set_target_properties(miniexpr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
endif()

if(MINIEXPR_BUILD_STATIC)
  miniexpr_add_library_target(miniexpr_static STATIC)
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
