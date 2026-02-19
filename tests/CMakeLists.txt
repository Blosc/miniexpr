file(GLOB MINIEXPR_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "jit_kernel_stub\\.c$")
if(EMSCRIPTEN)
  add_library(me_jit_test_stub STATIC "${CMAKE_CURRENT_SOURCE_DIR}/jit_kernel_stub.c")
else()
  add_library(me_jit_test_stub SHARED "${CMAKE_CURRENT_SOURCE_DIR}/jit_kernel_stub.c")
endif()
target_compile_features(me_jit_test_stub PUBLIC c_std_99)
if(NOT WIN32)
  target_link_libraries(me_jit_test_stub PRIVATE m)
endif()
if(WIN32)
  list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_threadsafe_chunk\\.c$")
  # MSVC/clang-cl complex literal handling breaks this test; skip on Windows.
  list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_regressions\\.c$")
endif()
if(EMSCRIPTEN)
  # These rely on host runtime features that are not available in plain wasm32 CI.
  list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_threadsafe_chunk\\.c$")
  list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_dsl_jit_codegen\\.c$")
  if(NOT (MINIEXPR_USE_WASM32_JIT AND MINIEXPR_WASM32_SIDE_MODULE))
    list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_dsl_jit_runtime_cache\\.c$")
  endif()
endif()
if(NOT (EMSCRIPTEN AND MINIEXPR_USE_WASM32_JIT AND MINIEXPR_WASM32_SIDE_MODULE))
  list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_dsl_jit_side_module\\.c$")
endif()

foreach(test_src IN LISTS MINIEXPR_TEST_SOURCES)
  get_filename_component(test_name "${test_src}" NAME_WE)
  add_executable("${test_name}" "${test_src}")
  target_compile_definitions("${test_name}" PRIVATE MINIEXPR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_compile_definitions("${test_name}"
    PRIVATE ME_DSL_JIT_TEST_STUB_SO_PATH="$<TARGET_FILE:me_jit_test_stub>")
  target_link_libraries("${test_name}" PRIVATE ${MINIEXPR_DEFAULT_LIB} ${MINIEXPR_THREADS_LIB})
  if(EMSCRIPTEN)
    target_link_options("${test_name}" PRIVATE
      "-sALLOW_MEMORY_GROWTH=1"
      "-sINITIAL_MEMORY=134217728")
    if(MINIEXPR_USE_WASM32_JIT)
      target_link_options("${test_name}" PRIVATE
        "-sALLOW_TABLE_GROWTH=1"
        "-sEXPORTED_RUNTIME_METHODS=addFunction,removeFunction")
    endif()
  endif()
  if(NOT WIN32)
    target_link_libraries("${test_name}" PRIVATE m)
  endif()
  if(EMSCRIPTEN AND test_name STREQUAL "test_function_order")
    target_link_options("${test_name}" PRIVATE
      "SHELL:--embed-file ${CMAKE_SOURCE_DIR}/src/functions.c@/src/functions.c")
  endif()
  if(EMSCRIPTEN AND test_name STREQUAL "test_dsl_jit_side_module")
    target_link_options("${test_name}" PRIVATE
      "SHELL:--pre-js ${CMAKE_SOURCE_DIR}/src/me_jit_glue.js"
      "-sEXPORTED_RUNTIME_METHODS=addFunction,removeFunction,stackSave,stackAlloc,stackRestore,lengthBytesUTF8,stringToUTF8")
  endif()
  if(EMSCRIPTEN AND test_name STREQUAL "test_dsl_jit_runtime_cache")
    target_link_options("${test_name}" PRIVATE
      "SHELL:--pre-js ${CMAKE_SOURCE_DIR}/src/me_jit_glue.js"
      "-sEXPORTED_RUNTIME_METHODS=addFunction,removeFunction,stackSave,stackAlloc,stackRestore,lengthBytesUTF8,stringToUTF8")
  endif()
  if(test_name STREQUAL "test_dsl_jit_runtime_cache")
    add_dependencies("${test_name}" me_jit_test_stub)
  endif()
  add_test(NAME "${test_name}" COMMAND "${test_name}")
  if(NOT test_name MATCHES "^test_dsl_jit_")
    set_tests_properties("${test_name}" PROPERTIES ENVIRONMENT "ME_DSL_JIT=0")
  endif()
endforeach()

set(MINIEXPR_JS_GLUE_SMOKE_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_me_jit_glue_smoke.js")
find_program(MINIEXPR_NODE_EXE node)
if(MINIEXPR_NODE_EXE AND EXISTS "${MINIEXPR_JS_GLUE_SMOKE_TEST}")
  add_test(NAME test_me_jit_glue_smoke
           COMMAND "${MINIEXPR_NODE_EXE}" "${MINIEXPR_JS_GLUE_SMOKE_TEST}")
endif()
