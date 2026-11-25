# Isolate the cache so an accidental compile cannot be hidden by a cache hit.
string(RANDOM LENGTH 12 ALPHABET abcdef0123456789 suffix)
set(directory "${TEST_DIR}/${suffix}")
file(MAKE_DIRECTORY "${directory}")
set(marker "${directory}/invoked")
set(compiler "${directory}/compiler.sh")
file(WRITE "${compiler}" "#!/bin/sh\n: > '${marker}'\nexit 99\n")
execute_process(COMMAND chmod +x "${compiler}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Could not create compiler invocation probe")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "CC=${compiler}" "CFLAGS="
  "ME_DSL_JIT_CACHE_DIR=${directory}/cache" "${TEST_EXE}" --reject-jit-contracts
  RESULT_VARIABLE result)
if(NOT result EQUAL 0 OR EXISTS "${marker}")
  message(FATAL_ERROR "Rejected kernel contract invoked a compiler or returned the wrong status")
endif()
