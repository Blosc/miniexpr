if(NOT DEFINED src1 OR NOT DEFINED src2 OR NOT DEFINED dst)
  message(FATAL_ERROR "copy_first_existing.cmake requires src1, src2 and dst")
endif()

# Normalize accidental surrounding quotes from -Dsrc="..." style arguments.
foreach(_v src1 src2 dst)
  string(REGEX REPLACE "^\"(.*)\"$" "\\1" ${_v} "${${_v}}")
endforeach()

set(_picked "")
if(EXISTS "${src1}")
  set(_picked "${src1}")
elseif(EXISTS "${src2}")
  set(_picked "${src2}")
else()
  message(FATAL_ERROR "Neither source exists: '${src1}' nor '${src2}'")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_picked}" "${dst}"
  RESULT_VARIABLE _copy_rc
)
if(NOT _copy_rc EQUAL 0)
  message(FATAL_ERROR "Failed copying '${_picked}' to '${dst}'")
endif()
