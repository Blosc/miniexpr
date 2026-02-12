# Plan: wasm32 DSL JIT – Compile and Dynamically Load Kernels

## Problem

MiniExpr's DSL JIT pipeline already works on Linux (cc/gcc/clang → .so → dlopen),
macOS (cc → .dylib → dlopen), and Windows (libtcc → in-memory).
wasm32/Emscripten is the only major platform without runtime JIT.

## Approach: Statically-Linked libtcc Targeting wasm32

Instead of dynamically loading libtcc at runtime (impossible in Emscripten), we
**statically link** a wasm32-targeting libtcc into the miniexpr library itself.

The JIT flow on wasm32:

```
DSL source → IR → C source → libtcc (linked in) → .wasm bytes (in memory)
  → WebAssembly.instantiate() (via EM_JS) → addFunction() → callable fn ptr
```

The JIT wasm module imports the host's linear memory (via binary patching of
the .wasm output) so kernel code can read/write arrays directly.

## Work Plan

### Phase 1: Build libtcc-wasm32 as a static library ✅ DONE

- [x] CMake: when `EMSCRIPTEN`, FetchContent minicc sources, compile with
      `-DTCC_TARGET_WASM32 -DONE_SOURCE=0` to produce `libtcc_wasm32.a`.
- [x] Link static libtcc into miniexpr on Emscripten.
- [x] Build compiles cleanly with emcc.

### Phase 2: wasm32 in-memory compile path ✅ DONE

- [x] `dsl_jit_compile_wasm32()` — compiles C source via statically-linked
      libtcc, outputs .wasm to Emscripten memfs, reads bytes back.
- [x] `me_wasm_jit_instantiate` (EM_JS) — binary-patches TCC .wasm to import
      host's `wasmMemory`, strips memory section/export, instantiates module,
      returns function table index via `addFunction()`.
- [x] `me_wasm_jit_free_fn` (EM_JS) — frees function table slot via
      `removeFunction()`.
- [x] `dsl_wasm32_patch_source()` — patches generated C: replaces `int64_t`
      with `int` (avoids TCC wasm32 64-bit comparison bugs), splits `||`
      guard into separate `if` statements.
- [x] `dsl_wasm32_source_calls_math()` — scans C source for actual math
      function calls (not just extern/static declarations); skips JIT when
      math bridge is not yet available.
- [x] Wired into `dsl_try_prepare_jit_runtime()` under `ME_USE_WASM32_JIT`.
- [x] Emscripten link flags: `-sALLOW_TABLE_GROWTH=1`,
      `-sEXPORTED_RUNTIME_METHODS=addFunction,removeFunction`,
      `-sALLOW_MEMORY_GROWTH=1`, `-sINITIAL_MEMORY=134217728` for both
      tests and benchmarks.

#### TCC wasm32 backend fixes (in minicc repo)

- [x] **gv() pointer store fix** (`tccgen.c`): `vtop->c.i = 0` after
      `vtop->r = r` — prevents stale offset leak into pointer stores.
- [x] **gen_opl 64-bit comparison fix** (`tccgen.c`): wasm32-specific path
      duplicates high words and uses real `gen_op(TOK_NE)` instead of
      synthetic `vset_VT_CMP(TOK_NE)` which doesn't emit an actual comparison
      on wasm32.
- [x] **is_cond_bool disabled for wasm32** (`tccgen.c`): ternary optimization
      that combines jump targets from branches with different comparison types
      is incompatible with wasm32's single `local_cmp` model.
- [x] **wasm_cmp_invert relaxed** (`wasm-gen.c`): returns 0 on op mismatch
      instead of `tcc_error` — correct because switch-loop dispatch means only
      one branch executes at runtime.
- [x] **tcc_error → tcc_error_noabort** (`tccwasm.c`): unresolved direct call
      errors no longer abort; `nb_errors` checked after each function body
      emission to return -1 early from `tcc_output_wasm`.
- [x] **tcc_exit_state NULL crash fix** (`tccwasm.c`): `tcc_error_noabort`
      calls `tcc_exit_state` which sets `tcc_state = NULL`; passed `TCCState*`
      directly to `wasm_emit_function_body` to avoid NULL dereference.
- [x] **tcc_set_wasm_data_base API** (`tcc.h`, `libtcc.h`, `libtcc.c`):
      relocates JIT module's data/stack to a safe memory region.

#### Memory management

- [x] JIT scratch memory (256KB, 64KB-aligned) kept alive in
      `program->jit_dl_handle` for the lifetime of the compiled kernel —
      the wasm module's `__stack_pointer` and data section reference this
      region.  Freed on program cleanup.

### Phase 3: Math bridge for wasm32 kernels ✅ DONE

- [x] Register math functions (sin, cos, round, exp, pow, etc.) as wasm
      imports so JIT kernels can call them.
- [x] Extend `me_wasm_jit_instantiate` JS to provide math functions in the
      import object: `{ env: { memory: wasmMemory, sin: Math.sin, ... } }`.
- [x] Add TCC symbol registration for bridge functions so the wasm linker
      can resolve the calls (via `tcc_add_symbol()` or a function index table
      in `tccwasm.c`).
- [x] Remove `dsl_wasm32_source_calls_math()` skip once bridge is working.
- [x] Handle `me_jit_exp10`, `me_jit_sinpi`, `me_jit_cospi`, `me_jit_where`,
      `me_jit_logaddexp` — either bridge to JS implementations or inline.
- [x] Handle static inline helpers in non-bridge codegen path (currently
      these emit `pow()`, `sin()`, `cos()` calls inside static functions
      that TCC compiles even when unused).

#### Phase 3 progress (in this branch)

- [x] `me_wasm_jit_instantiate` now merges/patches an existing import section
      (if present), ensures `env.memory` import is available, and provides a
      JS math/bridge import object (scalar + `me_jit_*` + `me_jit_vec_*`).
- [x] Added wasm32-side best-effort `tcc_add_symbol()` registration for
      required math and bridge symbols detected in generated C source.
- [x] Removed hard skip on `dsl_wasm32_source_calls_math()` so math kernels
      now attempt wasm32 JIT compilation.
- [x] Patched `../minicc/tccwasm.c` to represent unresolved direct calls as
      wasm imports (env module), emit import section entries, and offset
      function/export indices accordingly. Probe kernels using `sin()+exp()`
      now report `has_jit=1` for both `# me:compiler=tcc` and
      `# me:compiler=cc`.
- [x] Added wasm32 bridge-symbol lookup + import binding preference in
      `me_wasm_jit_instantiate`: imports now resolve to host wasm bridge
      functions first, with JS math/vector loops kept as fallback.

**Current behavior:** On wasm32, both math and non-math DSL kernels now JIT
 successfully. Math operations are resolved through host-provided imports,
 while non-math kernels run through the existing JIT path unchanged.

Math benchmark on wasm32 (`bench/benchmark_dsl_jit_math_kernels.js`,
`nitems=262144`, `repeats=6`, strict):

| Kernel | JIT ns/elem | Interp ns/elem | Relative |
|--------|-------------|----------------|----------|
| sin    | 2.816       | 8.689          | 3.09x faster |
| exp    | 2.180       | 3.365          | 1.54x faster |
| log    | 4.043       | 3.854          | 0.95x (slower) |
| pow    | 4.362       | 5.614          | 1.29x faster |
| hypot  | 3.244       | 4.255          | 1.31x faster |
| atan2  | 6.344       | 7.339          | 1.16x faster |
| sinpi  | 4.946       | 6.440          | 1.30x faster |
| cospi  | 4.915       | 6.492          | 1.32x faster |

### Phase 4: Testing ✅ DONE (for current scope)

- [x] All 26 wasm32 ctest tests pass (100%).
- [x] JIT kernels compile and execute correctly for non-math kernels
      (tests 1–3, 11–12, 14, 16, 18–19).
- [x] Math-using kernels now JIT on wasm32 via imports/bridge binding; no
      forced interpreter fallback for standard math kernels.
- [x] UDF tests gracefully skip JIT (unresolved external calls), pass.
- [x] ND-index kernels (tests 4–7) skip JIT at compile time (`_i0`
      undeclared), interpreter fallback works.
- [x] Mandelbrot benchmark: correct checksums at all sizes, JIT compiles
      and runs (0.8× speedup — see Phase 6 below).
- [x] All 9 minicc native tests pass (no regressions from TCC fixes).

### Phase 5: Cleanup — IN PROGRESS

- [x] Removed debug trace (`dsl_tracef("jit wasm32: patched source:...")`).
- [ ] Update `WASM32-TODO.md` with final status.
- [ ] Update `README.md` build instructions if needed.
- [ ] Consolidate minicc patches into proper commits in the minicc repo.

### Phase 6: wasm32 JIT performance optimization — IN PROGRESS

The wasm32 TCC JIT currently **underperforms** the interpreter for
compute-heavy kernels due to the switch-loop dispatch model.  Mandelbrot
benchmark (1200×800):

| Backend | Eval (ms) | Speedup |
|---------|-----------|---------|
| Interpreter | 3742 | baseline |
| TCC wasm32 JIT warm | 4555 | 0.8× |
| TCC native arm64 | 453 | 5.4× |

A detailed optimization plan with three options is in
`../minicc/wasm32-opts.md`:

Recent progress: bridge import binding now prefers wasm-native host bridge
functions over JS per-element loops, which significantly improved math-kernel
JIT throughput (see Phase 3 benchmark table above).

- [ ] **Option 1 — Basic block coalescing** (~300 lines, expected 2–4×):
      merge consecutive non-branching IR ops into single br_table cases.
- [ ] **Option 3a — Peephole: redundant load/store elimination** (~150 lines,
      +15–25%): chain register values on wasm stack between adjacent ops.
- [ ] **Option 2 — Structured control flow** (~2000 lines, expected 5–8×):
      full relooper/stackifier to emit native wasm block/loop/if.
- [ ] **Alternative — Binaryen post-processing**: pipe .wasm through
      `wasm-opt -O2` for zero-effort optimization (external dependency).

## Summary: What Works Now

| Feature | Status |
|---------|--------|
| Build libtcc-wasm32 as static lib | ✅ |
| Compile DSL kernels to .wasm | ✅ |
| Load and run JIT kernels dynamically | ✅ |
| Shared memory (host ↔ JIT module) | ✅ |
| Correct numerical results | ✅ |
| Graceful fallback for unsupported kernels | ✅ |
| Math function bridge | ✅ |
| JIT performance (vs interpreter) | ⚠️ mixed: math mostly >1x, Mandelbrot still <1x |

## What's Left

1. **Performance optimization (Phase 6)** — the switch-loop dispatch model
   makes JIT slower than interpreter for loop-heavy kernels.  Basic block
   coalescing (Option 1) is the recommended first step.

2. **Documentation (Phase 5 remainder)** — update WASM32-TODO.md, README.

3. **Minicc patch consolidation** — the TCC fixes are currently in the local
   `../minicc` working tree.  Need proper commits pushed to the minicc repo.
