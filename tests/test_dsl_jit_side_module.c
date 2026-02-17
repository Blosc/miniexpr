/*
 * Side-module wasm32 JIT helper registration coverage test.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/miniexpr.h"
#include "minctest.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>

EM_JS(int, test_wasm_side_instantiate,
      (const unsigned char *wasm_bytes, int wasm_len, int bridge_lookup_fn_idx), {
    if (typeof _meJitInstantiate !== "function") {
        err("[test-side-module] missing _meJitInstantiate");
        return 0;
    }
    var runtime = {
        HEAPF64: HEAPF64,
        HEAPF32: HEAPF32,
        wasmMemory: wasmMemory,
        wasmTable: wasmTable,
        stackSave: stackSave,
        stackAlloc: stackAlloc,
        stackRestore: stackRestore,
        lengthBytesUTF8: lengthBytesUTF8,
        stringToUTF8: stringToUTF8,
        addFunction: addFunction,
        err: err
    };
    var src = HEAPU8.subarray(wasm_bytes, wasm_bytes + wasm_len);
    return _meJitInstantiate(runtime, src, bridge_lookup_fn_idx) | 0;
});

EM_JS(void, test_wasm_side_free, (int idx), {
    if (typeof _meJitFreeFn === "function") {
        _meJitFreeFn({ removeFunction: removeFunction }, idx);
        return;
    }
    if (idx) {
        removeFunction(idx);
    }
});
#endif

static int eval_simple_kernel(const char *src, int expect_jit, double expected_offset) {
    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: compile error at %d\n", err);
        me_free(expr);
        return 1;
    }

    int has_jit = me_expr_has_jit_kernel(expr) ? 1 : 0;
    if (has_jit != expect_jit) {
        printf("  FAILED: expected has_jit=%d got %d\n", expect_jit, has_jit);
        me_free(expr);
        return 1;
    }

    double in[4] = {0.0, 1.0, 2.0, 3.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs[] = {in};
    if (me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: eval failed\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    for (int i = 0; i < 4; i++) {
        double expected = in[i] + expected_offset;
        if (fabs(out[i] - expected) > 1e-12) {
            printf("  FAILED: mismatch at %d got %.12f expected %.12f\n", i, out[i], expected);
            return 1;
        }
    }
    return 0;
}

int main(void) {
#if !defined(__EMSCRIPTEN__)
    printf("Skipping side-module wasm32 helper test (requires Emscripten).\n");
    return 0;
#else
    const char *src =
        "def kernel(x):\n"
        "    y = x + 5\n"
        "    return y\n";

    printf("=== Side-module wasm32 JIT helper registration test ===\n");

    me_register_wasm_jit_helpers(test_wasm_side_instantiate, test_wasm_side_free);
    if (eval_simple_kernel(src, 1, 5.0) != 0) {
        me_register_wasm_jit_helpers(NULL, NULL);
        return 1;
    }

    me_register_wasm_jit_helpers(NULL, NULL);
    if (eval_simple_kernel(src, 0, 5.0) != 0) {
        return 1;
    }

    printf("PASS: side-module helper registration and fallback behavior verified.\n");
    return 0;
#endif
}
