# Introduction to Menudet

**Menudet** is a little language for portable computation on arrays and tables.

While **Menudet** defines the portable language specification, syntax, and computational contract, **miniexpr** is its native C runtime implementation and JIT compiler.

---

## Status: Beta (Feature & API Freeze)

Menudet is currently at version **0.1 (beta)**. 

Entering beta means that the core language grammar, public C APIs ([`me_validate_menudet()`](../src/miniexpr.h)), artifact formats, and numerical semantics are **frozen** for the 0.1 release line. Ongoing work during the beta period is dedicated to:
- Edge-case testing and numerical validation against NumPy 2.x.
- Performance optimization and vectorization.
- Multi-platform qualification across Linux, macOS, Windows (clang-cl), and WebAssembly (`wasm32`).

Once proven stable across all target environments, Menudet will transition to **1.0 (General Availability)** with long-term backward-compatibility commitments.

---

## What Does "Portable" Mean?

In Menudet, portability is realized across four fundamental dimensions:

### 1. Cross-Platform Numerical Determinism
The exact same Menudet program computes bit-accurate, identical numerical results regardless of the host operating system or CPU architecture (x86_64, aarch64, or wasm32):
- Conforms to **NumPy 2.x** numerical promotion and computation rules.
- Eliminates host-compiler and hardware quirks (such as C undefined behaviors and CPU division traps).
- Enforces strict IEEE 754 rules: ties-even rounding for `round()` and `rint()`, deterministic signed zero handling, and defined NaN propagation.

### 2. Storage & Ecosystem Portability (Serialized Artifacts)
Menudet kernels can be serialized into standalone, self-contained JSON computational artifacts (see [doc/dsl-spec/artifact-0.1.md](dsl-spec/artifact-0.1.md)):
- Artifacts package the source code, typed signatures, constant bindings, and execution requirements.
- They can be embedded directly inside compressed data containers (such as **C-Blosc2** and **Python-Blosc2** arrays and tables).
- A kernel authored in Python can be saved inside a Blosc2 container and later evaluated natively in a pure C service, an embedded device, or in a web browser via WebAssembly without needing a Python runtime.

### 3. Backend-Agnostic Execution (Interpreter vs. JIT Parity)
- The **portable native interpreter** is the normative specification of the language.
- JIT compilers (TinyCC, system `cc`, or WASM) are optional acceleration engines that must produce output semantically identical to the interpreter.
- If JIT compilation is unavailable (for example, on locked-down environments that deny executable memory allocations / W^X, or on mobile/WASM sandboxes), Menudet kernels transparently fall back to the interpreter with no changes in results.

### 4. Sandboxing & Safety
Unlike unrestricted native C extensions, Menudet execution is safe:
- No arbitrary pointer dereferencing or unmanaged memory allocation.
- No I/O, networking, or host environment side-effects.
- Guaranteed loop boundaries (e.g. safety iteration caps on `while` loops).

---

## Robust Arithmetic & Defined Edge Cases

In standard C (ISO C99/C11), many common arithmetic edge cases trigger **Undefined Behavior (UB)** or hardware CPU faults (`SIGFPE`). In Menudet, every arithmetic operation is explicitly defined:

### Exact Integer Wrapping
- Signed integer overflow is undefined behavior in C (which permits compilers to optimize away overflow checks).
- Menudet guarantees exact **two's-complement modular arithmetic** across all integer types (`int8`, `int16`, `int32`, `int64` and unsigned types). Overflow wraps cleanly and deterministically.

### Defined Shift Behavior (`<<`, `>>`)
- In C, shift counts that are negative or $\ge$ bit width are UB (and on x86 hardware, the CPU masks shift counts modulo 32/64). Left-shifting negative numbers is also UB in C99.
- Menudet guarantees:
  - Shift counts $< 0$ or $\ge$ width produce `0` (for left shifts) or sign-extended `0` / `-1` (for right shifts).
  - Right shifts on signed negative numbers implement exact mathematical floor division by $2^{\text{count}}$, independent of compiler choices.

### Division-by-Zero & `INT_MIN / -1` Trap Safety
- **True division (`/`)**: Dividing integers with `/` promotes to `float64` (e.g. `5 / 2 == 2.5`), matching Python 3 and NumPy. Division by zero yields IEEE floating-point $\pm\infty$ or $\text{NaN}$ without terminating the process.
- **Floor division (`//`) and remainder (`%`)**: Dividing by zero returns `0`, matching NumPy array evaluation semantics.
- **The `INT_MIN / -1` trap**: In two's complement, `-INT_MIN` overflows positive range and triggers a CPU `SIGFPE` exception on x86. Menudet guards this boundary: `INT_MIN // -1` evaluates safely to `INT_MIN` with remainder `0`.
- **Divisor-sign modulo**: Menudet's `%` operator follows the Python/NumPy divisor-sign convention (e.g., `-5 % 3 == 1`), rather than C's truncation toward zero (which yields `-2`).

---

## Quick Start Example

A Menudet program consists of a single top-level `def` function:

```python
def normalize(x, mean, std):
    """Normalize input array, falling back to zero when variance is negligible."""
    if std > 1e-6:
        return (x - mean) / std
    return 0.0
```

### Validating and Compiling in C

```c
#include "miniexpr.h"
#include <stdio.h>

const char *source =
    "def normalize(x, mean, std):\n"
    "    if std > 1e-6:\n"
    "        return (x - mean) / std\n"
    "    return 0.0\n";

me_variable sig[] = {
    {"x", ME_FLOAT64},
    {"mean", ME_FLOAT64},
    {"std", ME_FLOAT64}
};

// 1. Validate signature against the frozen Menudet 0.1 profile
me_menudet_error diag;
if (me_validate_menudet(source, ME_MENUDET_DSL_VERSION, sig, 3, ME_FLOAT64, &diag) != ME_MENUDET_SUCCESS) {
    fprintf(stderr, "Validation error at line %d: %s\n", diag.line, diag.message);
    return -1;
}

// 2. Compile for evaluation
int err;
me_expr *expr = NULL;
if (me_compile(source, sig, 3, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
    fprintf(stderr, "Compile error: %s\n", me_get_last_error_message());
    return -1;
}

// 3. Evaluate over data buffers
double x_data[] = {10.0, 20.0, 30.0};
double mean_val = 20.0;
double std_val = 5.0;
double out[3];
const void *ptrs[] = {x_data, &mean_val, &std_val};

me_eval(expr, ptrs, 3, out, 3, NULL);
// out: [-2.0, 0.0, 2.0]

me_free(expr);
```

---

## Documentation Map

- **Language & Programming Guides**:
  - [doc/dsl-usage.md](dsl-usage.md): Practical guide for writing kernels, configuring JIT, and integrating UDFs.
  - [doc/dsl-syntax.md](dsl-syntax.md): Canonical reference for grammar, statements, pragmas, and expressions.
  - [doc/strings.md](strings.md): Working with UCS-4 strings (`ME_STRING`), bytes (`ME_BYTES`), and Arrow variable-length output (`me_eval_varlen()`).

- **Menudet Specifications**:
  - [doc/dsl-spec/0.1.md](dsl-spec/0.1.md): Formal Menudet 0.1 language specification.
  - [doc/dsl-spec/artifact-0.1.md](dsl-spec/artifact-0.1.md): JSON artifact serialization schema and contract.
  - [doc/dsl-spec/features-0.1.json](dsl-spec/features-0.1.json): Machine-readable feature matrix.

- **NumPy Conformance**:
  - [doc/numpy-functions-0.1.md](numpy-functions-0.1.md): Function matrix and transcendental status rules.
  - [doc/numpy-arithmetic-0.1.md](numpy-arithmetic-0.1.md): Type promotion, casting, and arithmetic rules.
  - [doc/numpy-compat-checkpoint.md](numpy-compat-checkpoint.md): Conformance checkpoints and test methodology.

- **Arrays, Graphs & Native Execution**:
  - [doc/chunk-processing.md](chunk-processing.md): Chunked array evaluation patterns.
  - [doc/parallel-processing.md](parallel-processing.md): Thread-safe evaluation and multithreading.
  - [doc/native-execution.md](native-execution.md): Host executors and native execution pipeline.
  - [doc/native-graphs-1.md](native-graphs-1.md): Multi-stage numerical computation graphs.
