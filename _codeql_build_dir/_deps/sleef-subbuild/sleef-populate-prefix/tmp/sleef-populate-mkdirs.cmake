# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-src")
  file(MAKE_DIRECTORY "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-src")
endif()
file(MAKE_DIRECTORY
  "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-build"
  "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-subbuild/sleef-populate-prefix"
  "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-subbuild/sleef-populate-prefix/tmp"
  "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-subbuild/sleef-populate-prefix/src/sleef-populate-stamp"
  "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-subbuild/sleef-populate-prefix/src"
  "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-subbuild/sleef-populate-prefix/src/sleef-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-subbuild/sleef-populate-prefix/src/sleef-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/runner/work/miniexpr/miniexpr/_codeql_build_dir/_deps/sleef-subbuild/sleef-populate-prefix/src/sleef-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
