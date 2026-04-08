#ifndef DSL_JIT_TEST_H
#define DSL_JIT_TEST_H

#define ME_DSL_JIT_TEST_NEG_CACHE_FLAG "-me_intentional_bad_flag_for_neg_cache"

#ifndef ME_DSL_JIT_WASM_POS_CACHE_SLOTS
#define ME_DSL_JIT_WASM_POS_CACHE_SLOTS 64
#endif

#if defined(__EMSCRIPTEN__)
void dsl_jit_wasm_pos_cache_reset_for_tests(void);
#endif

#endif
