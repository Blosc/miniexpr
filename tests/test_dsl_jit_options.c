/* Options are call-local and must be restored on both failure and success. */
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"
#include "../src/dsl_config.h"

static void check_restored(void) {
    const char *actual = me_jit_option_value("CC");
    const char *expected = getenv("CC");
    assert((!actual && !expected) || (actual && expected && !strcmp(actual, expected)));
}

int main(void) {
    me_jit_options options = {"/no/compiler", "-O1", NULL, 0, 0};
    int64_t shape[] = {8};
    int32_t chunks[] = {8}, blocks[] = {8};
    me_expr *expr = NULL;
    int error = 0;
    int rc = me_compile_nd_jit_options("def broken( :", NULL, 0, ME_FLOAT64, 1,
        shape, chunks, blocks, ME_JIT_ON, &options, &error, &expr);
    assert(rc != ME_COMPILE_SUCCESS && !expr);
    check_restored();
    rc = me_compile_nd_jit_options("1 + 2", NULL, 0, ME_FLOAT64, 1,
        shape, chunks, blocks, ME_JIT_OFF, &options, &error, &expr);
    assert(rc == ME_COMPILE_SUCCESS && expr);
    me_free(expr);
    check_restored();
    return 0;
}
