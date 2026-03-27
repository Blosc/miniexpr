if(MINIEXPR_NEEDS_BUNDLED_TINYCC)
  if(APPLE)
    set(MINIEXPR_TINYCC_SHARED_NAME "libtcc.dylib")
  elseif(WIN32)
    set(MINIEXPR_TINYCC_SHARED_NAME "tcc.dll")
  else()
    set(MINIEXPR_TINYCC_SHARED_NAME "libtcc.so")
  endif()
  if(CMAKE_CONFIGURATION_TYPES)
    set(MINIEXPR_TINYCC_BUILT_SHARED_PATH "${tinycc_BINARY_DIR}/$<CONFIG>/${MINIEXPR_TINYCC_SHARED_NAME}")
  else()
    set(MINIEXPR_TINYCC_BUILT_SHARED_PATH "${tinycc_BINARY_DIR}/${MINIEXPR_TINYCC_SHARED_NAME}")
  endif()
  set(MINIEXPR_TINYCC_STAGED_SHARED_PATH "${CMAKE_CURRENT_BINARY_DIR}/${MINIEXPR_TINYCC_SHARED_NAME}")
  set(MINIEXPR_TINYCC_CONFIG_STAMP "${tinycc_BINARY_DIR}/.miniexpr_tinycc_configured")
  set(MINIEXPR_TINYCC_GENERATOR_ARGS)
  if(CMAKE_GENERATOR)
    list(APPEND MINIEXPR_TINYCC_GENERATOR_ARGS -G "${CMAKE_GENERATOR}")
  endif()
  if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND MINIEXPR_TINYCC_GENERATOR_ARGS -A "${CMAKE_GENERATOR_PLATFORM}")
  endif()
  if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND MINIEXPR_TINYCC_GENERATOR_ARGS -T "${CMAKE_GENERATOR_TOOLSET}")
  endif()
  set(MINIEXPR_TINYCC_CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DMINICC_BUILD_SHARED_LIBTCC=ON"
    "-DMINICC_BUILD_WASM32_TCC=OFF"
    "-DMINICC_ENABLE_TESTING=OFF"
  )
  # When cross-compiling, forward target system info to the minicc sub-build
  # so minicc selects the correct backend and triggers its host-tool build logic.
  if(CMAKE_CROSSCOMPILING)
    if(CMAKE_SYSTEM_NAME)
      list(APPEND MINIEXPR_TINYCC_CMAKE_ARGS "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR)
      list(APPEND MINIEXPR_TINYCC_CMAKE_ARGS "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(CMAKE_TOOLCHAIN_FILE)
      list(APPEND MINIEXPR_TINYCC_CMAKE_ARGS "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
    endif()
  endif()
  # Always forward OSX architectures; affects target arch on Apple even on native builds.
  if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(APPEND MINIEXPR_TINYCC_CMAKE_ARGS "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
  endif()
  set(MINIEXPR_TINYCC_BUILD_CMD
    "${CMAKE_COMMAND}" --build "${tinycc_BINARY_DIR}" --target minicc_libtcc_shared
  )
  if(CMAKE_CONFIGURATION_TYPES)
    set(MINIEXPR_TINYCC_BUILD_CMD
      "${CMAKE_COMMAND}" --build "${tinycc_BINARY_DIR}" --config "$<CONFIG>" --target minicc_libtcc_shared
    )
  endif()

  add_custom_command(
    OUTPUT "${MINIEXPR_TINYCC_CONFIG_STAMP}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${tinycc_BINARY_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${tinycc_BINARY_DIR}/CMakeCache.txt"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${tinycc_BINARY_DIR}/CMakeFiles"
    COMMAND "${CMAKE_COMMAND}"
            ${MINIEXPR_TINYCC_GENERATOR_ARGS}
            -S "${tinycc_SOURCE_DIR}"
            -B "${tinycc_BINARY_DIR}"
            ${MINIEXPR_TINYCC_CMAKE_ARGS}
    COMMAND "${CMAKE_COMMAND}" -E touch "${MINIEXPR_TINYCC_CONFIG_STAMP}"
    DEPENDS "${tinycc_SOURCE_DIR}/CMakeLists.txt"
    VERBATIM
  )

  if(WIN32)
    # Windows generators differ in where DLL artifacts are placed.
    # Use a helper that copies the first existing candidate.
    add_custom_command(
      OUTPUT "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}"
      COMMAND ${MINIEXPR_TINYCC_BUILD_CMD}
      COMMAND "${CMAKE_COMMAND}"
              -Dsrc1="${tinycc_BINARY_DIR}/$<CONFIG>/${MINIEXPR_TINYCC_SHARED_NAME}"
              -Dsrc2="${tinycc_BINARY_DIR}/${MINIEXPR_TINYCC_SHARED_NAME}"
              -Ddst="${MINIEXPR_TINYCC_STAGED_SHARED_PATH}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/copy_first_existing.cmake"
      DEPENDS "${MINIEXPR_TINYCC_CONFIG_STAMP}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/copy_first_existing.cmake"
      VERBATIM
    )
  else()
    add_custom_command(
      OUTPUT "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}"
      COMMAND ${MINIEXPR_TINYCC_BUILD_CMD}
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${MINIEXPR_TINYCC_BUILT_SHARED_PATH}" "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}"
      DEPENDS "${MINIEXPR_TINYCC_CONFIG_STAMP}"
      VERBATIM
    )
  endif()

  add_custom_target(miniexpr_tinycc ALL
    DEPENDS "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}")

  install(FILES "${MINIEXPR_TINYCC_STAGED_SHARED_PATH}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(FILES "${tinycc_SOURCE_DIR}/libtcc.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(FILES "${tinycc_SOURCE_DIR}/COPYING"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/miniexpr/third_party/tinycc")
endif()

# --- wasm32 static libtcc (Emscripten cross-compile, targets wasm32) --------
if(MINIEXPR_USE_WASM32_JIT)
  # Build the c2str host tool with the native compiler so it can run during
  # the cross-compilation build.  Emscripten sets CMAKE_CROSSCOMPILING, so
  # we invoke the system C compiler directly.
  find_program(MINIEXPR_HOST_CC cc REQUIRED)
  set(MINIEXPR_WASM_LIBTCC_INCDIR "${CMAKE_CURRENT_BINARY_DIR}/wasm-libtcc-include")
  file(MAKE_DIRECTORY "${MINIEXPR_WASM_LIBTCC_INCDIR}")
  set(MINIEXPR_C2STR_BIN "${CMAKE_CURRENT_BINARY_DIR}/minicc-c2str-host")
  set(MINIEXPR_WASM_TCCDEFS "${MINIEXPR_WASM_LIBTCC_INCDIR}/tccdefs_.h")
  add_custom_command(
    OUTPUT "${MINIEXPR_C2STR_BIN}"
    COMMAND "${MINIEXPR_HOST_CC}" -DC2STR=1
            "${tinycc_SOURCE_DIR}/conftest.c"
            -o "${MINIEXPR_C2STR_BIN}"
    DEPENDS "${tinycc_SOURCE_DIR}/conftest.c"
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${MINIEXPR_WASM_TCCDEFS}"
    COMMAND "${MINIEXPR_C2STR_BIN}"
            "${tinycc_SOURCE_DIR}/include/tccdefs.h"
            "${MINIEXPR_WASM_TCCDEFS}"
    DEPENDS "${MINIEXPR_C2STR_BIN}" "${tinycc_SOURCE_DIR}/include/tccdefs.h"
    VERBATIM
  )
  add_custom_target(miniexpr_wasm_tccdefs DEPENDS "${MINIEXPR_WASM_TCCDEFS}")

  # Generate config.h for the wasm32-targeting libtcc.
  file(READ "${tinycc_SOURCE_DIR}/VERSION" _minicc_ver)
  string(STRIP "${_minicc_ver}" _minicc_ver)
  file(WRITE "${MINIEXPR_WASM_LIBTCC_INCDIR}/config.h"
    "/* Generated for wasm32-targeting libtcc under Emscripten */\n"
    "#define TCC_VERSION \"${_minicc_ver}\"\n"
    "#define CC_NAME CC_clang\n"
    "#define GCC_MAJOR 0\n"
    "#define GCC_MINOR 0\n"
    "#define CONFIG_TCC_PREDEFS 1\n"
    "#define CONFIG_NEW_DTAGS 1\n"
    "#ifndef CONFIG_TCCDIR\n"
    "#define CONFIG_TCCDIR \"/usr/local/lib/tcc\"\n"
    "#endif\n"
  )

  # Build a static libtcc that compiles C to wasm32 modules.
  # Uses ONE_SOURCE=1 so libtcc.c pulls in all other .c files via #include.
  add_library(miniexpr_wasm_libtcc STATIC "${tinycc_SOURCE_DIR}/libtcc.c")
  set_target_properties(miniexpr_wasm_libtcc PROPERTIES OUTPUT_NAME "tcc_wasm32")
  target_include_directories(miniexpr_wasm_libtcc PRIVATE
    "${MINIEXPR_WASM_LIBTCC_INCDIR}"   # config.h, tccdefs_.h
    "${tinycc_SOURCE_DIR}"
  )
  target_compile_definitions(miniexpr_wasm_libtcc PRIVATE
    ONE_SOURCE=1
    TCC_TARGET_WASM32
  )
  add_dependencies(miniexpr_wasm_libtcc miniexpr_wasm_tccdefs)
endif()
