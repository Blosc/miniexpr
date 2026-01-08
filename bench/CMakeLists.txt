file(GLOB MINIEXPR_BENCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

foreach(bench_src IN LISTS MINIEXPR_BENCH_SOURCES)
  get_filename_component(bench_name "${bench_src}" NAME_WE)
  add_executable("${bench_name}" "${bench_src}")
  target_link_libraries("${bench_name}" PRIVATE ${MINIEXPR_DEFAULT_LIB} ${MINIEXPR_THREADS_LIB})
endforeach()
