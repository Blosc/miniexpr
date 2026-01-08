file(GLOB MINIEXPR_EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

foreach(example_src IN LISTS MINIEXPR_EXAMPLE_SOURCES)
  get_filename_component(example_name "${example_src}" NAME_WE)
  add_executable("${example_name}" "${example_src}")
  target_link_libraries("${example_name}" PRIVATE ${MINIEXPR_DEFAULT_LIB} ${MINIEXPR_THREADS_LIB})
endforeach()
