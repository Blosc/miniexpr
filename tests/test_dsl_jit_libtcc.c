/* Real libtcc diagnostics and state lifetime; no disk-cache helpers are needed. */
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/dsl_jit_runtime_internal.h"

int main(void) {
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    me_dsl_compiled_program program;
    memset(&program, 0, sizeof(program));
    program.fp_mode = ME_DSL_FP_STRICT;
    program.jit_c_source = "int me_dsl_jit_kernel(const void **in, void *out, long long n) { return 0; }";
    setenv("TMPDIR", __FILE__, 1); /* An existing regular file, not a directory. */
    setenv("CC", "/no/system/compiler", 1);
    setenv("ME_DSL_JIT_TCC_OPTIONS", "-invalid-safer-jit-option", 1);
    assert(!dsl_jit_compile_libtcc_in_memory(&program));
    assert(strstr(program.jit_c_error, "invalid-safer-jit-option"));
    assert(!program.jit_tcc_state && !program.jit_kernel_fn);
    unsetenv("ME_DSL_JIT_TCC_OPTIONS");
    program.jit_c_source = "int broken( {";
    assert(!dsl_jit_compile_libtcc_in_memory(&program));
    assert(strstr(program.jit_c_error, "error"));
    assert(!program.jit_tcc_state);
    program.jit_c_source = "int me_dsl_jit_kernel(const void **in, void *out, long long n) { return 0; }";
    assert(dsl_jit_compile_libtcc_in_memory(&program));
    assert(program.jit_tcc_state && program.jit_kernel_fn);
    assert(program.jit_kernel_fn(NULL, NULL, 0) == 0);
    assert(!program.jit_c_error[0]);
    dsl_jit_libtcc_delete_state(program.jit_tcc_state);
#endif
    return 0;
}
