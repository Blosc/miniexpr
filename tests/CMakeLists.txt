if(NOT EMSCRIPTEN)
  # The first portable runner uses host files; wasm corpus adapters are follow-up work.
  add_executable(portable_dsl_runner portable-dsl/runner.c)
  target_link_libraries(portable_dsl_runner PRIVATE ${MINIEXPR_DEFAULT_LIB})
  target_compile_features(portable_dsl_runner PRIVATE c_std_99)
  if(NOT WIN32)
    target_link_libraries(portable_dsl_runner PRIVATE m)
    find_program(MINIEXPR_PORTABLE_CC NAMES cc clang gcc)
  endif()
  set(MINIEXPR_PORTABLE_CASES affine chains loops int32 int64 float32 bool compare special_floats flow
                     boolean_results cast_int math
                     division_int division_bool division_literals division_locals division_loop division_float32
                     division_condition
                     cast_argument_int cast_argument_bool cast_argument_nested
                     cast_integer_exact cast_nonfinite_truth
                     math_widen math_condition math_local_widen
                     arithmetic_round arithmetic_decimal arithmetic_scalar_sub arithmetic_local_round
                     arithmetic_widen
                     arithmetic_float_cast arithmetic_float_nested arithmetic_float_local arithmetic_float_widen
                     bool_float_sum bool_float_neg bool_float_product bool_float_difference
                     bool_float_local_neg
                     while_cap_exact while_cap_exceeded while_cap_continue
                     while_cap_hybrid
                     while_cap_chain
                     masked_local_chain
                     masked_bool_chain
                    unknown_name invalid_index zero_step missing_return missing_return_ok
                     missing_return_mixed missing_return_loop missing_return_loop_ok
                     missing_return_hybrid missing_return_hybrid_ok)
  foreach(input_type bool int32 int64 float32 float64)
    foreach(output_type bool int32 int64 float32 float64)
      list(APPEND MINIEXPR_PORTABLE_CASES convert_${input_type}_${output_type})
    endforeach()
  endforeach()
  foreach(case_name IN LISTS MINIEXPR_PORTABLE_CASES)
    if(case_name MATCHES "^convert_")
      set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/portable-dsl/identity.dsl")
    else()
      set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/portable-dsl/${case_name}.dsl")
    endif()
    foreach(policy off default)
      add_test(NAME portable_dsl_${case_name}_${policy}
        COMMAND portable_dsl_runner
          "${source_path}"
          "${CMAKE_CURRENT_SOURCE_DIR}/portable-dsl/${case_name}.txt" "${policy}")
      set_tests_properties(portable_dsl_${case_name}_${policy} PROPERTIES TIMEOUT 30
        ENVIRONMENT "ME_DSL_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/portable-jit-cache")
    endforeach()
    if(MINIEXPR_ENABLE_TCC_JIT)
      add_test(NAME portable_dsl_${case_name}_on
        COMMAND portable_dsl_runner
          "${source_path}"
          "${CMAKE_CURRENT_SOURCE_DIR}/portable-dsl/${case_name}.txt" on)
      set_tests_properties(portable_dsl_${case_name}_on PROPERTIES TIMEOUT 30
        ENVIRONMENT "ME_DSL_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/portable-jit-cache")
    endif()
    if(MINIEXPR_PORTABLE_CC)
      # Same language fixture, with only the execution preference changed.
      file(READ "${source_path}" cc_source)
      string(REPLACE "# me:compiler=tcc\n" "" cc_source "${cc_source}")
      set(cc_path "${CMAKE_CURRENT_BINARY_DIR}/portable-dsl/${case_name}.cc.dsl")
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/portable-dsl")
      file(WRITE "${cc_path}" "# me:compiler=cc\n${cc_source}")
      add_test(NAME portable_dsl_${case_name}_cc
        COMMAND portable_dsl_runner "${cc_path}"
          "${CMAKE_CURRENT_SOURCE_DIR}/portable-dsl/${case_name}.txt" on)
      set_tests_properties(portable_dsl_${case_name}_cc PROPERTIES TIMEOUT 30
        ENVIRONMENT "CC=${MINIEXPR_PORTABLE_CC};ME_DSL_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/portable-jit-cache")
    endif()
  endforeach()
  # These preserve the interpreter baseline for unresolved typed-codegen gaps.
  # Do not promote them to required-JIT conformance before fixing the gaps.
  # math_condition also differs between linked interpreter math engines; keep
  # that source as an audit reproducer rather than a passing baseline case.
  get_property(portable_tests DIRECTORY PROPERTY TESTS)
  foreach(test_name IN LISTS portable_tests)
    if(test_name MATCHES "^portable_dsl_while_cap_")
      set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "ME_DSL_WHILE_MAX_ITERS=3")
    endif()
  endforeach()

  foreach(case_name nested_cast cast_argument_division)
    add_test(NAME portable_audit_${case_name}_off COMMAND portable_dsl_runner
      "${CMAKE_CURRENT_SOURCE_DIR}/portable-dsl/audit/${case_name}.dsl"
      "${CMAKE_CURRENT_SOURCE_DIR}/portable-dsl/audit/${case_name}.txt" off)
    set_tests_properties(portable_audit_${case_name}_off PROPERTIES TIMEOUT 30)
  endforeach()
endif()

if(MINIEXPR_BUILD_ARTIFACT AND NOT EMSCRIPTEN)
  add_executable(portable_artifact_runner portable-artifacts/runner.c)
  target_link_libraries(portable_artifact_runner PRIVATE miniexpr_artifact)
  foreach(policy off default)
    add_test(NAME portable_artifact_affine_${policy} COMMAND portable_artifact_runner
      "${CMAKE_CURRENT_SOURCE_DIR}/portable-artifacts/affine.json" "${policy}")
    set_tests_properties(portable_artifact_affine_${policy} PROPERTIES TIMEOUT 30
      ENVIRONMENT "ME_DSL_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/artifact-jit-cache")
  endforeach()
  if(MINIEXPR_ENABLE_TCC_JIT)
    add_test(NAME portable_artifact_affine_on COMMAND portable_artifact_runner
      "${CMAKE_CURRENT_SOURCE_DIR}/portable-artifacts/affine.json" on)
    set_tests_properties(portable_artifact_affine_on PROPERTIES TIMEOUT 30
      ENVIRONMENT "ME_DSL_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/artifact-jit-cache")
  endif()
  if(MINIEXPR_PORTABLE_CC)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/portable-artifacts/affine.json" cc_artifact)
    string(REPLACE "# me:compiler=tcc" "# me:compiler=cc" cc_artifact "${cc_artifact}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/affine.cc.json" "${cc_artifact}")
    add_test(NAME portable_artifact_affine_cc COMMAND portable_artifact_runner
      "${CMAKE_CURRENT_BINARY_DIR}/affine.cc.json" on)
    set_tests_properties(portable_artifact_affine_cc PROPERTIES TIMEOUT 30
      ENVIRONMENT "CC=${MINIEXPR_PORTABLE_CC};ME_DSL_FP_MODE=fast;ME_DSL_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/artifact-jit-cache")
  endif()
  add_test(NAME portable_artifact_affine_fallback COMMAND portable_artifact_runner
    "${CMAKE_CURRENT_SOURCE_DIR}/portable-artifacts/affine.json" default)
  set_tests_properties(portable_artifact_affine_fallback PROPERTIES TIMEOUT 30
    ENVIRONMENT "ME_DSL_JIT=0;ME_DSL_FP_MODE=fast;ME_DSL_JIT_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/artifact-jit-cache")
endif()

file(GLOB MINIEXPR_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
if(NOT MINIEXPR_BUILD_ARTIFACT)
  list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_dsl_artifact\\.c$")
endif()
if(NOT MINIEXPR_ENABLE_TCC_JIT OR EMSCRIPTEN)
  list(FILTER MINIEXPR_TEST_SOURCES EXCLUDE REGEX "test_dsl_jit_libtcc\\.c$")
endif()
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
  if(test_name STREQUAL "test_dsl_artifact")
    target_link_libraries("${test_name}" PRIVATE miniexpr_artifact)
    target_compile_definitions("${test_name}" PRIVATE
      MINIEXPR_ARTIFACT_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/portable-artifacts/affine.json")
  endif()
  if(EMSCRIPTEN)
    if(MINIEXPR_USE_WASM32_JIT)
      target_compile_definitions("${test_name}" PRIVATE ME_USE_WASM32_JIT=1)
      if(MINIEXPR_WASM32_SIDE_MODULE)
        target_compile_definitions("${test_name}" PRIVATE ME_WASM32_SIDE_MODULE=1)
      else()
        target_compile_definitions("${test_name}" PRIVATE ME_WASM32_SIDE_MODULE=0)
      endif()
    endif()
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
  if(test_name STREQUAL "test_dsl_masked_locals")
    set_property(TEST "${test_name}" APPEND PROPERTY ENVIRONMENT "ME_DSL_WHILE_MAX_ITERS=3")
    set_tests_properties("${test_name}" PROPERTIES TIMEOUT 30)
  endif()
endforeach()

set(MINIEXPR_JS_GLUE_SMOKE_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_me_jit_glue_smoke.js")
find_program(MINIEXPR_NODE_EXE node)
if(MINIEXPR_NODE_EXE AND EXISTS "${MINIEXPR_JS_GLUE_SMOKE_TEST}")
  add_test(NAME test_me_jit_glue_smoke
           COMMAND "${MINIEXPR_NODE_EXE}" "${MINIEXPR_JS_GLUE_SMOKE_TEST}")
endif()
