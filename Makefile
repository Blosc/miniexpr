# MiniExpr Makefile
# Organized structure: src/, bench/, tests/

CC ?= gcc
SLEEF_DIR ?= ../sleef
ifneq ("$(wildcard $(SLEEF_DIR)/src)","")
  SLEEF_CFLAGS = -Isrc -Isrc/sleef_compat -I$(SLEEF_DIR) -I$(SLEEF_DIR)/src -I$(SLEEF_DIR)/src/common -I$(SLEEF_DIR)/src/arch
  SLEEF_DEFS =
else
  SLEEF_CFLAGS = -Isrc
  SLEEF_DEFS = -DME_USE_SLEEF=0
endif
ifeq ($(OS),Windows_NT)
  # Check if we are using clang-cl
  ifneq (,$(findstring clang-cl,$(CC)))
    CFLAGS = -O2 -DNDEBUG $(SLEEF_CFLAGS) $(SLEEF_DEFS)
    DEBUG_CFLAGS = -O0 -g $(SLEEF_CFLAGS) $(SLEEF_DEFS)
    LDFLAGS = clang_rt.builtins-x86_64.lib
  else
    CFLAGS = -Wall -Wshadow -Wno-unknown-pragmas -Wno-unused-function -O2 -DNDEBUG $(SLEEF_CFLAGS) $(SLEEF_DEFS)
    DEBUG_CFLAGS = -Wall -Wshadow -Wno-unknown-pragmas -Wno-unused-function -O0 -g $(SLEEF_CFLAGS) $(SLEEF_DEFS)
    LDFLAGS = -lm
  endif
else
  CFLAGS = -Wall -Wshadow -Wno-unknown-pragmas -Wno-unused-function -O2 -DNDEBUG $(SLEEF_CFLAGS) $(SLEEF_DEFS)
  DEBUG_CFLAGS = -Wall -Wshadow -Wno-unknown-pragmas -Wno-unused-function -O0 -g $(SLEEF_CFLAGS) $(SLEEF_DEFS)
  LDFLAGS = -lm
endif

# Directories
SRCDIR = src
BENCHDIR = bench
TESTDIR = tests
EXAMPLEDIR = examples
BUILDDIR = build

# Source files
LIB_SRCS = $(SRCDIR)/miniexpr.c $(SRCDIR)/functions.c
LIB_OBJS = $(BUILDDIR)/miniexpr.o $(BUILDDIR)/functions.o
LIB_HDR = $(SRCDIR)/miniexpr.h

ifeq ($(OS),Windows_NT)
  EXE = .exe
else
  EXE =
endif

# Benchmark sources
BENCH_SRCS = $(wildcard $(BENCHDIR)/*.c)
BENCH_BINS = $(patsubst $(BENCHDIR)/%.c,$(BUILDDIR)/%$(EXE),$(BENCH_SRCS))

# Test sources
TEST_SRCS = $(wildcard $(TESTDIR)/*.c)
# Exclude tests with pthreads on windows
ifeq ($(OS),Windows_NT)
  TEST_SRCS := $(filter-out $(TESTDIR)/test_threadsafe_chunk.c,$(TEST_SRCS))
endif
TEST_BINS = $(patsubst $(TESTDIR)/%.c,$(BUILDDIR)/%$(EXE),$(TEST_SRCS))

# Example sources
EXAMPLE_SRCS = $(wildcard $(EXAMPLEDIR)/*.c)
EXAMPLE_BINS = $(patsubst $(EXAMPLEDIR)/%.c,$(BUILDDIR)/%$(EXE),$(EXAMPLE_SRCS))

.PHONY: all lib bench test examples clean help debug debug-test

# Default target
all: lib bench

# Help target
help:
	@echo "MiniExpr Build System"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  make lib          - Build library object file (release mode)"
	@echo "  make bench        - Build all benchmarks"
	@echo "  make test         - Build and run all tests"
	@echo "  make examples     - Build all examples"
	@echo "  make debug        - Build library in debug mode (-g -O0, asserts enabled)"
	@echo "  make debug-test   - Build and run tests in debug mode"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Environment overrides:"
	@echo "  SLEEF_DIR=../sleef  - Path to SLEEF source tree (optional)"
	@echo ""
	@echo "Individual benchmarks:"
	@echo "  make $(BUILDDIR)/benchmark_all_types"
	@echo "  make $(BUILDDIR)/benchmark_types"
	@echo ""
	@echo "Individual tests:"
	@echo "  make $(BUILDDIR)/test_all_types"
	@echo "  make $(BUILDDIR)/test_bytecode"

# Create build directory
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

# Build library object
lib: $(LIB_OBJS)

$(LIB_OBJS): $(LIB_SRCS) $(LIB_HDR) | $(BUILDDIR)
	@echo "Building library..."
	$(CC) $(CFLAGS) -c $(SRCDIR)/miniexpr.c -o $(BUILDDIR)/miniexpr.o
	$(CC) $(CFLAGS) -c $(SRCDIR)/functions.c -o $(BUILDDIR)/functions.o
	@echo "✓ Library built: $(LIB_OBJS)"

# Build library in debug mode
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: clean lib
	@echo "✓ Debug build complete (asserts enabled, -g -O0)"

# Build and run tests in debug mode
debug-test: CFLAGS = $(DEBUG_CFLAGS)
debug-test: clean test
	@echo "✓ Debug tests complete"

# Build all benchmarks
bench: $(BENCH_BINS)

$(BUILDDIR)/%$(EXE): $(BENCHDIR)/%.c $(LIB_OBJS) | $(BUILDDIR)
	@echo "Building benchmark: $@"
	$(CC) $(CFLAGS) -I$(SRCDIR) $< $(LIB_OBJS) -o $@ $(LDFLAGS)
	@echo "✓ Built: $@"

# Build and run all tests
test: $(TEST_BINS)
	@echo ""
	@echo "Running tests..."
	@echo "================"
	@for test in $(TEST_BINS); do \
		echo ""; \
		echo "Running $$test..."; \
		echo "-----------------------------------"; \
		./$$test || exit 1; \
	done
	@echo ""
	@echo "✅ All tests passed!"

$(BUILDDIR)/%$(EXE): $(TESTDIR)/%.c $(LIB_OBJS) | $(BUILDDIR)
	@echo "Building test: $@"
	$(CC) $(CFLAGS) -I$(SRCDIR) $< $(LIB_OBJS) -o $@ $(LDFLAGS)
	@echo "✓ Built: $@"

# Build all examples
examples: $(EXAMPLE_BINS)

$(BUILDDIR)/%$(EXE): $(EXAMPLEDIR)/%.c $(LIB_OBJS) | $(BUILDDIR)
	@echo "Building example: $@"
	$(CC) $(CFLAGS) -I$(SRCDIR) $< $(LIB_OBJS) -o $@ $(LDFLAGS) -lpthread
	@echo "✓ Built: $@"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILDDIR)
	rm -f *.o *.exe
	@echo "✓ Clean complete"

# Pattern rule for object files (if needed for future extensions)
$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(LIB_HDR) | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Check for trailing whitespace
check-whitespace:
	@./check-whitespace.sh

# Fix trailing whitespace
fix-whitespace:
	@./check-whitespace.sh --fix
