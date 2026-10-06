/* Source compiler preferences override environment defaults, like FP pragmas. */
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include "../src/dsl_compile_internal.h"

static void set_compiler(const char *value) {
#ifdef _WIN32
    assert(_putenv_s("ME_DSL_JIT_COMPILER", value) == 0);
#else
    assert(setenv("ME_DSL_JIT_COMPILER", value, 1) == 0);
#endif
}

static void check(const char *source, me_dsl_compiler expected) {
    me_dsl_error error;
    me_dsl_program *parsed = me_dsl_parse(source, &error);
    assert(parsed);
    char reason[256] = {0};
    me_dsl_compiled_program *program = dsl_compiled_program_alloc(
        parsed, source, 1, reason, sizeof(reason));
    assert(program);
    assert(program->compiler == expected);
    dsl_compiled_program_free(program);
    me_dsl_program_free(parsed);
}

int main(void) {
    set_compiler("cc");
    check("def kernel(x):\n    return x\n", ME_DSL_COMPILER_CC);
    check("\n# ordinary comment\n# me:compiler=tcc\ndef kernel(x):\n    return x\n",
          ME_DSL_COMPILER_LIBTCC);
    set_compiler("tcc");
    check("def kernel(x):\n    return x\n", ME_DSL_COMPILER_LIBTCC);
    check("# me:compiler=cc\ndef kernel(x):\n    return x\n", ME_DSL_COMPILER_CC);
    return 0;
}
