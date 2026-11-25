# Convenience entry points only: CMake owns all source lists and dependencies.
# Override BUILDDIR, JOBS, CMAKE_ARGS or CTEST_ARGS as needed.
CMAKE ?= cmake
CTEST ?= ctest
BUILDDIR ?= build
JOBS ?= 8
CMAKE_ARGS ?=
CTEST_ARGS ?=
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: all lib bench examples configure test debug debug-test clean help

all lib bench examples: configure
	$(CMAKE) --build "$(BUILDDIR)" --parallel $(JOBS)

configure:
	$(CMAKE) -S "$(ROOT)" -B "$(BUILDDIR)" $(CMAKE_ARGS)

test: all
	$(CMAKE) -E chdir "$(BUILDDIR)" $(CTEST) --output-on-failure $(CTEST_ARGS)

debug:
	$(CMAKE) -S "$(ROOT)" -B "$(BUILDDIR)-debug" $(CMAKE_ARGS) -DCMAKE_BUILD_TYPE=Debug
	$(CMAKE) --build "$(BUILDDIR)-debug" --parallel $(JOBS)

debug-test: debug
	$(CMAKE) -E chdir "$(BUILDDIR)-debug" $(CTEST) --output-on-failure $(CTEST_ARGS)

# Keep old individual-target spellings, with executable placement owned by CMake.
$(BUILDDIR)/%: configure
	$(CMAKE) --build "$(BUILDDIR)" --parallel $(JOBS) --target "$*"

# Clean configured targets without discarding the cache or downloaded sources.
clean:
	@if test -f "$(BUILDDIR)/CMakeCache.txt"; then \
	  $(CMAKE) --build "$(BUILDDIR)" --target clean; \
	fi

help:
	@echo "CMake shortcuts: all lib bench examples test debug debug-test clean"
	@echo "all/lib/bench/examples build the targets enabled by CMake configuration."
	@echo "Use CMAKE_ARGS to select features; BUILDDIR and JOBS select build location and parallelism."
	@echo "Individual targets: make build/test_kernel_api (executables live under build/tests)."
