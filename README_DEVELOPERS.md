# Developer Notes

This document collects developer-facing build options, internal execution semantics, runtime JIT tuning knobs, and advanced numerical graph interfaces.

---

## Evaluation & API Semantics

The primary user-facing API reference lives in [README.md](README.md) and [src/miniexpr.h](src/miniexpr.h).

### `dtype` Compilation Modes
The `dtype` argument to `me_compile()` has two mutually exclusive modes:
1. **Uniform Type**: Provide an explicit type (e.g. `ME_FLOAT64`). All variables must use `ME_AUTO`.
2. **Mixed Types / Type Inference**: Set `dtype` to `ME_AUTO`. All variables must have explicit types, and the result type is inferred via NumPy 2.x promotion rules.

Mixing modes causes compilation to fail.

### ND / Padded Block Evaluation
- `shape`, `chunkshape`, and `blockshape` are in C-contiguous order.
- `me_eval_nd()` computes only valid (unpadded) elements for chunk `nchunk` and block `nblock`, zero-filling the padded tail in non-scalar output buffers.
- `me_nd_valid_nitems()` reports the count of valid elements in a given block.

### Variable-Length Arrow Output (`me_eval_varlen`)
For string (`ME_STRING` / UCS-4) or byte (`ME_BYTES`) outputs, `me_eval_varlen()` packs results into an Arrow-compatible layout:
- `offsets`: `int64_t` buffer of `(block_nitems + 1)` entries (`offsets[0] == 0`).
- `data`: Contiguous byte buffer containing UTF-8 (Arrow `large_string`) or verbatim bytes (Arrow `large_binary`).
- `me_varlen_data_bound()` calculates the exact upper bound needed for the destination buffer, avoiding the memory overhead of worst-case fixed-width padding in storage.

### Scalar Reductions
Available scalar reductions: `sum`, `prod`, `min`, `max`, `any`, `all`.
- Reductions collapse an operand block/chunk to a scalar and broadcast across the expression.
- Constant subexpressions are folded at compile time (`sum(x * -2)` folds `-2`).
- Result dtypes: integer `sum`/`prod` promote to 64-bit; `float32` accumulates in `float64` and rounds back to `float32`; `any`/`all` yield `bool`.
- NaNs propagate in `min`/`max` if any lane is NaN.

---

## Building & CMake Options

### Core CMake Flags

| CMake Option | Default | Description |
| :--- | :--- | :--- |
| `-DMINIEXPR_BUILD_SHARED=ON\|OFF` | `ON` | Build shared library |
| `-DMINIEXPR_BUILD_STATIC=ON\|OFF` | `ON` | Build static library |
| `-DMINIEXPR_BUILD_TESTS=ON\|OFF` | `ON` | Build test suite |
| `-DMINIEXPR_BUILD_EXAMPLES=ON\|OFF` | `ON` | Build example programs |
| `-DMINIEXPR_BUILD_BENCH=ON\|OFF` | `ON` | Build benchmark executables |
| `-DMINIEXPR_BUILD_ARTIFACT=ON\|OFF` | `OFF` | Build `miniexpr_artifact` (JSON artifacts & numerical graphs) |
| `-DMINIEXPR_USE_SLEEF=ON\|OFF` | `ON` | Fetch & enable SLEEF SIMD math acceleration |
| `-DMINIEXPR_USE_ACCELERATE=ON\|OFF` | `ON` | macOS Accelerate/vForce backend |
| `-DMINIEXPR_ENABLE_TCC_JIT=ON\|OFF` | `ON` | Enable in-memory TinyCC JIT backend |

### Build Invocations
- **Standard (CMake + Ninja)**:
  ```bash
  cmake -S . -B build -G Ninja
  cmake --build build -j
  ctest --test-dir build
  ```
- **Windows (`clang-cl`)**:
  CMake automatically selects the newest installed Visual Studio:
  ```bash
  cmake -S . -B build -T ClangCL
  cmake --build build
  ```
- **Run targeted tests**:
  ```bash
  ctest --test-dir build -R <pattern>
  ```

---

## Native Numerical Graphs & Host Executors

For query engines and chunked storage frameworks (such as [C-Blosc2](https://github.com/Blosc/c-blosc2)), miniexpr provides an opt-in graph execution engine ([`src/miniexpr_graph.h`](src/miniexpr_graph.h)) enabled with `-DMINIEXPR_BUILD_ARTIFACT=ON`:

1. **Declarative Graphs**: Prepares and validates multi-node computational graphs from JSON specifications without invoking Python.
2. **Specialization**: Binds input ranks, shapes, and strides, performing trailing-axis broadcasting and reduction shape inference.
3. **Host-Owned Parallelism & Scheduling**:
   - Analyzes execution graph partitionability to admit lane-local maps for parallel execution.
   - Dispatches work to host-owned worker threads via a synchronous `fork_join` callback.
   - Dynamic work queues tile domains (defaulting to 32 Ki elements) and interleave initial memory claims to prevent first-touch memory allocation serialization.
   - Preserves deterministic error reporting and IEEE floating-point status across workers.

For full architectural details, see [doc/native-graphs-1.md](doc/native-graphs-1.md) and [doc/native-execution.md](doc/native-execution.md).

---

## JIT Runtime Internals & Tuning Knobs

### Execution Backends
- **Bundled TinyCC (`# me:compiler=tcc`)**: Compiles directly in memory with zero disk cache. On Linux, it uses anonymous `memfd`-backed RW/RX mappings.
- **System CC (`# me:compiler=cc`)**: Invokes an external compiler (`cc` or MSVC) to produce cached shared libraries in `TMPDIR`.
- **Call-Local Options**: Embedders can pass `me_jit_options` to `me_compile_nd_jit_options()` to customize the compiler command, CFLAGS, cache directory, and tracing without modifying global environment variables.

### Internal Environment Variables (Debugging & Benchmarking)

| Variable | Values / Default | Description |
| :--- | :--- | :--- |
| `ME_SIMD_MATH_BACKEND` | `auto\|system\|sleef\|accelerate\|scalar` | Force math backend (`auto`/`system` select system math; `scalar` forces scalar) |
| `ME_DSL_JIT_COMPILER` | `tcc\|cc` | Override default JIT backend compiler |
| `ME_DSL_FP_MODE` | `strict\|contract\|fast` | Default floating-point contraction mode |
| `ME_DSL_WHILE_MAX_ITERS` | Integer | Override safety iteration cap for `while` loops |
| `ME_DSL_JIT_MATH_BRIDGE` | `0\|1` (default `1`) | Enable/disable runtime math-bridge lowering |
| `ME_DSL_JIT_VEC_MATH` | `0\|1` (default `1`) | Enable/disable vector math lowering |
| `ME_DSL_JIT_BRANCH_AWARE_IF` | `0\|1` (default `1`) | Enable branch-aware `if` lowering |
| `ME_DSL_JIT_DEBUG_CC` | `0\|1` (default `0`) | Print compiler output during runtime `cc` JIT builds |
| `ME_DSL_TRACE` | `0\|1` | Print compiler and JIT diagnostic traces to stderr |

---

## Contributing

Install pre-commit Git hooks after cloning:

```bash
./scripts/install-hooks.sh
```

Follow the K&R braces style, ensure all tests pass (`ctest --test-dir build`), and run `./check-whitespace.sh` before submitting changes.
