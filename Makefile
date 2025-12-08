# MiniExpr Makefile
# Organized structure: src/, bench/, tests/

CC = gcc
CFLAGS = -Wall -Wshadow -Wno-unknown-pragmas -Wno-unused-function -O2
LDFLAGS = -lm

# Directories
SRCDIR = src
BENCHDIR = bench
TESTDIR = tests
BUILDDIR = build

# Source files
LIB_SRC = $(SRCDIR)/miniexpr.c
LIB_OBJ = $(BUILDDIR)/miniexpr.o
LIB_HDR = $(SRCDIR)/miniexpr.h

# Benchmark sources
BENCH_SRCS = $(wildcard $(BENCHDIR)/*.c)
BENCH_BINS = $(patsubst $(BENCHDIR)/%.c,$(BUILDDIR)/%,$(BENCH_SRCS))

# Test sources
TEST_SRCS = $(wildcard $(TESTDIR)/*.c)
TEST_BINS = $(patsubst $(TESTDIR)/%.c,$(BUILDDIR)/%,$(TEST_SRCS))

.PHONY: all lib bench test clean help

# Default target
all: lib bench

# Help target
help:
	@echo "MiniExpr Build System"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  make lib       - Build library object file"
	@echo "  make bench     - Build all benchmarks"
	@echo "  make test      - Build and run all tests"
	@echo "  make clean     - Remove all build artifacts"
	@echo "  make help      - Show this help"
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
lib: $(LIB_OBJ)

$(LIB_OBJ): $(LIB_SRC) $(LIB_HDR) | $(BUILDDIR)
	@echo "Building library..."
	$(CC) $(CFLAGS) -c $(LIB_SRC) -o $@
	@echo "✓ Library built: $@"

# Build all benchmarks
bench: $(BENCH_BINS)

$(BUILDDIR)/%: $(BENCHDIR)/%.c $(LIB_OBJ) | $(BUILDDIR)
	@echo "Building benchmark: $@"
	$(CC) $(CFLAGS) -I$(SRCDIR) $< $(LIB_OBJ) -o $@ $(LDFLAGS)
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

$(BUILDDIR)/%: $(TESTDIR)/%.c $(LIB_OBJ) | $(BUILDDIR)
	@echo "Building test: $@"
	$(CC) $(CFLAGS) -I$(SRCDIR) $< $(LIB_OBJ) -o $@ $(LDFLAGS)
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
