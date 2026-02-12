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

### Phase 3: Math bridge for wasm32 kernels — NOT STARTED

- [ ] Register math functions (sin, cos, round, exp, pow, etc.) as wasm
      imports so JIT kernels can call them.
- [ ] Extend `me_wasm_jit_instantiate` JS to provide math functions in the
      import object: `{ env: { memory: wasmMemory, sin: Math.sin, ... } }`.
- [ ] Add TCC symbol registration for bridge functions so the wasm linker
      can resolve the calls (via `tcc_add_symbol()` or a function index table
      in `tccwasm.c`).
- [ ] Remove `dsl_wasm32_source_calls_math()` skip once bridge is working.
- [ ] Handle `me_jit_exp10`, `me_jit_sinpi`, `me_jit_cospi`, `me_jit_where`,
      `me_jit_logaddexp` — either bridge to JS implementations or inline.
- [ ] Handle static inline helpers in non-bridge codegen path (currently
      these emit `pow()`, `sin()`, `cos()` calls inside static functions
      that TCC compiles even when unused).

**Current behavior:** kernels that call math functions are detected by
`dsl_wasm32_source_calls_math()` and gracefully skip JIT, falling back
to the interpreter.  Tests pass.

### Phase 4: Testing ✅ DONE (for current scope)

- [x] All 26 wasm32 ctest tests pass (100%).
- [x] JIT kernels compile and execute correctly for non-math kernels
      (tests 1–3, 11–12, 14, 16, 18–19).
- [x] Math-using kernels (test 10) gracefully skip JIT, interpreter fallback
      produces correct results.
- [x] UDF tests gracefully skip JIT (unresolved external calls), pass.
- [x] ND-index kernels (tests 4–7) skip JIT at compile time (`_i0`
      undeclared), interpreter fallback works.
- [x] Mandelbrot benchmark: correct checksums at all sizes, JIT compiles
      and runs (0.8× speedup — see Phase 6 below).
- [x] All 9 minicc native tests pass (no regressions from TCC fixes).

### Phase 5: Cleanup ✅ DONE

- [x] Removed debug trace (`dsl_tracef("jit wasm32: patched source:...")`).
- [ ] Update `WASM32-TODO.md` with final status.
- [ ] Update `README.md` build instructions if needed.
- [ ] Consolidate minicc patches into proper commits in the minicc repo.

### Phase 6: wasm32 JIT performance optimization — NOT STARTED

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
| Math function bridge | ❌ not started |
| JIT performance (vs interpreter) | ⚠️ 0.8× (needs optimization) |

## What's Left

1. **Math bridge (Phase 3)** — required for kernels using sin/cos/round/etc.
   Currently these gracefully fall back to the interpreter.

2. **Performance optimization (Phase 6)** — the switch-loop dispatch model
   makes JIT slower than interpreter for loop-heavy kernels.  Basic block
   coalescing (Option 1) is the recommended first step.

3. **Documentation (Phase 5 remainder)** — update WASM32-TODO.md, README.

4. **Minicc patch consolidation** — the TCC fixes are currently in the local
   `../minicc` working tree.  Need proper commits pushed to the minicc repo.
