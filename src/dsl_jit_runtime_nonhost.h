/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_JIT_RUNTIME_NONHOST_H
#define MINIEXPR_DSL_JIT_RUNTIME_NONHOST_H

static void dsl_try_prepare_jit_runtime(me_dsl_compiled_program *program) {
#if ME_USE_WASM32_JIT
    if (!program || !program->jit_c_source || program->jit_kernel_fn) {
        return;
    }
    if (!dsl_jit_runtime_enabled()) {
        return;
    }
#if ME_WASM32_SIDE_MODULE
    if (!me_wasm_jit_helpers_available()) {
        dsl_tracef("jit runtime skip: side-module wasm32 helpers are not registered");
        return;
    }
#endif
    uint64_t key = dsl_jit_runtime_cache_key(program);
    if (dsl_jit_wasm_pos_cache_bind_program(program, key)) {
        dsl_tracef("jit runtime hit: fp=%s source=wasm-process-cache key=%016llx",
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
        return;
    }
    if (dsl_jit_compile_wasm32(program, key)) {
        dsl_tracef("jit runtime built: fp=%s compiler=%s key=%016llx",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler),
                   (unsigned long long)key);
        return;
    }
    if (program->jit_c_error[0] == '\0') {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime wasm32 compilation failed");
    }
    dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s",
               dsl_fp_mode_name(program->fp_mode),
               dsl_compiler_name(program->compiler),
               program->jit_c_error);
#elif ME_USE_LIBTCC_FALLBACK
    if (!program || !program->jit_c_source || program->jit_kernel_fn) {
        return;
    }
    if (!dsl_jit_runtime_enabled()) {
        return;
    }
    if (program->compiler != ME_DSL_COMPILER_LIBTCC) {
        return;
    }
    if (dsl_jit_compile_libtcc_in_memory(program)) {
        dsl_tracef("jit runtime built: fp=%s compiler=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler));
        return;
    }
    snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
             "jit runtime tcc compilation failed");
    dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s detail=%s",
               dsl_fp_mode_name(program->fp_mode),
               dsl_compiler_name(program->compiler),
               program->jit_c_error,
               dsl_jit_libtcc_error_message());
#else
    (void)program;
#endif
}

#endif
