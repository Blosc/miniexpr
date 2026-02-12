# MiniExpr DSL Development Logbook

This document records design decisions, experiments, and lessons learned during the development of the MiniExpr DSL and expression engine.

---

## 2026-01-31: Bytecode VM Experiment

### Motivation

The existing AST interpreter walks the expression tree recursively for each evaluation. We hypothesized that a bytecode VM could reduce interpretation overhead, especially for:
- Complex expressions with many operations
- Nested transcendental function chains
- Repeated evaluations on large arrays

### Implementation

Created a complete bytecode VM infrastructure:

**Files created (~1500 lines total):**
- `src/vm_opcodes.h` - ~150 opcodes covering arithmetic, transcendentals, comparisons, logic, loops
- `src/vm_types.h` - VM program and execution context structures
- `src/vm_exec.h/c` - Bytecode interpreter with SIMD dispatch
- `src/vm_compiler.h/c` - AST to bytecode compiler

**Design decisions:**
- 8-byte instruction format: `{opcode, dtype, arg, reserved}` for alignment
- Stack-based VM operating on block buffers (not individual values)
- Thread-local buffer pools to avoid malloc per operation
- Reuse existing SLEEF SIMD dispatch functions for transcendentals
- Integrated sincos caching (`vec_sin_cached`/`vec_cos_cached`)

### Benchmarks

**Single-threaded results (8M elements):**

| Expression | AST GB/s | VM GB/s | VM Speedup |
|------------|----------|---------|------------|
| `sin(a)` | 4.72 | 4.57 | 0.97x |
| `sin(a)**2 + cos(a)**2` | 3.78 | 2.11 | 0.56x |
| `a + b` | 57.98 | 42.05 | 0.73x |

**Multi-threaded results (8M elements, sin²+cos²):**

| Threads | AST GB/s | VM GB/s | VM Speedup |
|---------|----------|---------|------------|
| 1 | 3.78 | 2.11 | 0.56x |
| 4 | 13.59 | 4.70 | 0.35x |
| 8 | ~15 | ~1 | 0.09x |

**Small block results (4096 elements, nested transcendentals):**

| Expression | AST Time | VM Time | VM Speedup |
|------------|----------|---------|------------|
| `sin(cos(tan(a)))` | 20.5ms | 17.4ms | **1.18x** |
| `sin(a)+cos(b)` | 28.8ms | 21.0ms | **1.37x** |

### Analysis

**Why VM lost in most cases:**

1. **Final memcpy overhead** - VM stores results to scratch buffer, then copies to output. AST writes directly.

2. **Poor thread scaling** - Buffer pool reallocation when chunk sizes differ between threads caused severe degradation.

3. **AST is already highly optimized:**
   - Direct function pointer dispatch (no bytecode interpretation)
   - Sincos caching computes sin+cos together when sharing input
   - No intermediate scratch buffers for simple operations
   - Better memory locality

4. **VM overhead not amortized** - For transcendentals, the SIMD computation dominates; for arithmetic, the overhead dominates.

**Where VM won:**
- Nested transcendental chains on small blocks where bytecode dispatch overhead is tiny compared to computation time.

### Decision: DISCARDED

The VM was removed because:
- Adds ~1500 lines of complexity
- Slower in nearly all real-world scenarios
- Terrible multi-threaded scaling
- Only marginal wins in niche cases

### Lessons Learned

1. **Profile before optimizing** - The AST interpreter was already near-optimal for this workload.

2. **Bytecode VMs need JIT** - Interpretation overhead is significant; a true win would require JIT compilation to native code.

3. **Memory allocation patterns matter** - Thread-local pools helped but weren't enough; the fundamental issue was too many intermediate buffers.

4. **Sincos fusion is critical** - The AST's sincos caching (computing both in one SLEEF call) provides ~1.4x speedup that the VM initially missed.

---

## Architecture Notes

### Current Expression Engine

The miniexpr engine uses an AST-based approach:

1. **Parsing** - Expressions parsed into `me_expr` tree nodes
2. **Compilation** - `me_compile()` builds AST, resolves variables, applies optimizations
3. **Evaluation** - `me_eval()` walks AST recursively, dispatching to SIMD functions

**Key optimizations in place:**
- SLEEF SIMD for transcendentals (AVX2, NEON)
- Sincos caching (thread-local, keyed by input pointer)
- Type promotion and constant folding
- Block-wise evaluation for cache efficiency

### DSL Layer

The DSL (`dsl_parser.c`) extends single expressions to multi-statement programs:
- Temporary variables (`$temp = expr`)
- Result assignment (`result = expr`)
- Conditionals planned but not yet implemented

---

## Already Implemented Features

1. **Index variables** ✅
   - `_i0`, `_i1`, `_i2`, etc. for element indices in n-dimensional arrays
   - `_n0`, `_n1`, `_n2` for dimension sizes, `_ndim` for number of dimensions
   - Enable position-dependent computations where each element knows its location
   - Examples:
     - `result = a * _i0 / _n0` - Weight by normalized position (0.0 to ~1.0)
     - `result = sin(2 * pi * _i0 / _n0)` - Generate sine wave based on position
     - `result = where(_i0 < _n0 / 2, a, b)` - Split array in half with different operations

2. **Loop constructs** ✅
   - `for var in range(n):` syntax implemented
   - Enables iterative algorithms (e.g., Mandelbrot set, Newton-Raphson)
   - See `examples/13_mandelbrot_dsl.c` for demonstration

3. **Conditionals in DSL** ✅
   - Element-wise conditions via `where(cond, true_val, false_val)`
   - Block-level conditionals: `if/elif/else` with scalar conditions and `result` assignment
   - Flow-only loop control: `if/elif` chains with `break`/`continue`
   - Early exit optimizations: `if all(escape_iter != 100.0): break`

4. **Basic string support** ✅
   - String constants (UTF-8 literals, decoded to UCS4)
   - String comparisons for categorical data filtering (`==`, `!=`)
   - String predicates (`startswith`, `endswith`, `contains`)
   - Format strings for debugging (`print("value = {}", x)`)
   - Note: Full string manipulation is out of scope; current support is fixed-size UCS4 only

5. **Reduction operations** ✅
   - `sum()`, `mean()`, `min()`, `max()`, `any()`, `all()`, `prod()`

6. **User-defined functions** ✅
   - Register custom C functions callable from DSL
   - Enable domain-specific extensions without modifying core

---

## Future Work

### High Priority

1. **Emit `while` loops instead of ternary-`for` in JIT C codegen**
   - Current codegen emits: `for (i = start; ((step > 0) ? (i < stop) : (i > stop)); i += step)`
   - The ternary condition creates TCC's "jump into loop" CFG pattern (nearly 2× more basic blocks)
   - This forces the wasm32 backend to use the slower switch-loop dispatch instead of structured control flow
   - When step direction is known at codegen time (positive or negative), emit a simple `while (i < stop)` or `while (i > stop)` instead
   - Expected benefit: enables wasm32 Stackifier path (native `block`/`loop`/`br_if`), fewer basic blocks, and potentially better V8 JIT optimization
   - See `minicc/wasm32-opts.md` and the structured control flow code in `minicc/tccwasm.c`

### Medium Priority
1. **String interop with Blosc2 variable-length types**
   - interop with Blosc2 variable-length string types

2. **Partial reductions along axes**
   - reductions along axes for N-D data

### Low Priority / Research

7. **JIT compilation**
   - If bytecode is revisited, would need LLVM or similar
   - Significant complexity; only justified for very complex expressions
   - May be better to rely on compiler autovectorization

8. **GPU offload**
   - CUDA/OpenCL kernels for transcendental-heavy workloads
   - Would require significant infrastructure
   - Blosc2 GPU support would be prerequisite

9. **Expression caching/memoization**
   - Cache compiled expressions by string hash
   - Avoid re-parsing identical expressions
   - Trade memory for compilation time

10. **Symbolic differentiation**
    - Automatic gradient computation
    - Would enable optimization algorithms
    - Complex to implement correctly

---

## References

- `dsl_instructions.md` - Original DSL design document
- `dsl_instructions-opus.md` - Detailed DSL specification with examples
- SLEEF library - SIMD transcendental functions
- Blosc2 - Compressed array storage that miniexpr targets
