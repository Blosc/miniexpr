/*
 * JIT C codegen smoke tests for DSL kernels.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "../src/dsl_jit_cgen.h"
#include "../src/dsl_jit_ir.h"
#include "../src/dsl_parser.h"
#include "minctest.h"

typedef struct {
    me_dtype value_dtype;
} dtype_resolve_ctx;

static bool mock_resolve_dtype(void *ctx, const me_dsl_expr *expr, me_dtype *out_dtype) {
    dtype_resolve_ctx *dctx = (dtype_resolve_ctx *)ctx;
    if (!dctx || !expr || !expr->text || !out_dtype) {
        return false;
    }
    const char *text = expr->text;
    if (strcmp(text, "4") == 0 || strcmp(text, "1") == 0 || strcmp(text, "0") == 0) {
        *out_dtype = ME_INT64;
        return true;
    }
    if (strstr(text, "==") || strstr(text, "!=") || strstr(text, "<=") ||
        strstr(text, ">=") || strstr(text, "<") || strstr(text, ">")) {
        *out_dtype = ME_BOOL;
        return true;
    }
    *out_dtype = dctx->value_dtype;
    return true;
}

#if !defined(_WIN32) && !defined(_WIN64)
static int compile_generated_source(const char *source) {
    if (!source) {
        return 1;
    }
    char dir_template[] = "/tmp/me_jit_codegen_XXXXXX";
    char *dir = mkdtemp(dir_template);
    if (!dir) {
        return 1;
    }

    char src_path[512];
    char so_path[512];
    char cmd[1600];
    snprintf(src_path, sizeof(src_path), "%s/kernel.c", dir);
    snprintf(so_path, sizeof(so_path), "%s/kernel.so", dir);

    FILE *f = fopen(src_path, "wb");
    if (!f) {
        rmdir(dir);
        return 1;
    }
    size_t n = strlen(source);
    if (fwrite(source, 1, n, f) != n) {
        fclose(f);
        remove(src_path);
        rmdir(dir);
        return 1;
    }
    fclose(f);

    snprintf(cmd, sizeof(cmd),
             "cc -std=c99 -O2 -fPIC -shared -o \"%s\" \"%s\" >/dev/null 2>&1",
             so_path, src_path);
    int rc = system(cmd);

    remove(so_path);
    remove(src_path);
    rmdir(dir);
    return rc == 0 ? 0 : 1;
}
#endif

static int test_codegen_all_noncomplex_dtypes(void) {
    printf("\n=== JIT C Codegen Test 1: all non-complex dtypes ===\n");

    const me_dtype dtypes[] = {
        ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
        ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64,
        ME_FLOAT32, ME_FLOAT64
    };
    const int ndtypes = (int)(sizeof(dtypes) / sizeof(dtypes[0]));

    const char *src =
        "def kernel(x):\n"
        "    acc = x\n"
        "    for i in range(4):\n"
        "        if i == 1:\n"
        "            continue\n"
        "        acc = acc + x\n"
        "        if i == 3:\n"
        "            break\n"
        "    if acc:\n"
        "        return acc\n"
        "    return x\n";

    for (int i = 0; i < ndtypes; i++) {
        me_dsl_error parse_error;
        me_dsl_program *program = me_dsl_parse(src, &parse_error);
        if (!program) {
            printf("  FAILED: parse error at %d:%d (%s)\n",
                   parse_error.line, parse_error.column, parse_error.message);
            return 1;
        }

        const char *param_names[] = {"x"};
        me_dtype param_dtypes[] = {dtypes[i]};
        dtype_resolve_ctx rctx;
        rctx.value_dtype = dtypes[i];

        me_dsl_error ir_error;
        me_dsl_jit_ir_program *ir = NULL;
        bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                                      mock_resolve_dtype, &rctx, &ir, &ir_error);
        me_dsl_program_free(program);
        if (!ok || !ir) {
            printf("  FAILED: ir build rejected dtype %d at %d:%d (%s)\n",
                   (int)dtypes[i], ir_error.line, ir_error.column, ir_error.message);
            me_dsl_jit_ir_free(ir);
            return 1;
        }

        me_dsl_error cg_error;
        me_dsl_jit_cgen_options options;
        memset(&options, 0, sizeof(options));
        options.symbol_name = "me_dsl_jit_kernel";
        char *c_source = NULL;
        ok = me_dsl_jit_codegen_c(ir, dtypes[i], &options, &c_source, &cg_error);
        me_dsl_jit_ir_free(ir);
        if (!ok || !c_source) {
            printf("  FAILED: codegen rejected dtype %d at %d:%d (%s)\n",
                   (int)dtypes[i], cg_error.line, cg_error.column, cg_error.message);
            free(c_source);
            return 1;
        }

#if !defined(_WIN32) && !defined(_WIN64)
        if (compile_generated_source(c_source) != 0) {
            printf("  FAILED: generated C did not compile for dtype %d\n", (int)dtypes[i]);
            free(c_source);
            return 1;
        }
#endif
        free(c_source);
    }

    printf("  PASSED\n");
    return 0;
}

static int test_codegen_rejects_unsupported_expression_ops(void) {
    printf("\n=== JIT C Codegen Test 2: unsupported expression ops ===\n");

    const char *src =
        "def kernel(x):\n"
        "    return x % 2\n";

    me_dsl_error parse_error;
    me_dsl_program *program = me_dsl_parse(src, &parse_error);
    if (!program) {
        printf("  FAILED: parse error at %d:%d (%s)\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    dtype_resolve_ctx rctx;
    rctx.value_dtype = ME_FLOAT64;

    me_dsl_error ir_error;
    me_dsl_jit_ir_program *ir = NULL;
    bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                                  mock_resolve_dtype, &rctx, &ir, &ir_error);
    me_dsl_program_free(program);
    if (!ok || !ir) {
        printf("  FAILED: ir build rejected setup at %d:%d (%s)\n",
               ir_error.line, ir_error.column, ir_error.message);
        me_dsl_jit_ir_free(ir);
        return 1;
    }

    me_dsl_error cg_error;
    char *c_source = NULL;
    ok = me_dsl_jit_codegen_c(ir, ME_FLOAT64, NULL, &c_source, &cg_error);
    me_dsl_jit_ir_free(ir);
    if (ok) {
        printf("  FAILED: codegen accepted unsupported %% operator\n");
        free(c_source);
        return 1;
    }

    free(c_source);
    printf("  PASSED\n");
    return 0;
}

static int test_codegen_element_loop_control(void) {
    printf("\n=== JIT C Codegen Test 3: element loop control ===\n");

    const char *src =
        "# me:dialect=element\n"
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(8):\n"
        "        if i == 0:\n"
            "            continue\n"
        "        if x > i:\n"
        "            acc = acc + i\n"
        "        else:\n"
        "            break\n"
        "    return acc\n";

    me_dsl_error parse_error;
    me_dsl_program *program = me_dsl_parse(src, &parse_error);
    if (!program) {
        printf("  FAILED: parse error at %d:%d (%s)\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    dtype_resolve_ctx rctx;
    rctx.value_dtype = ME_INT64;

    me_dsl_error ir_error;
    me_dsl_jit_ir_program *ir = NULL;
    bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                                  mock_resolve_dtype, &rctx, &ir, &ir_error);
    me_dsl_program_free(program);
    if (!ok || !ir) {
        printf("  FAILED: ir build rejected element loop-control source at %d:%d (%s)\n",
               ir_error.line, ir_error.column, ir_error.message);
        me_dsl_jit_ir_free(ir);
        return 1;
    }

    me_dsl_error cg_error;
    me_dsl_jit_cgen_options options;
    memset(&options, 0, sizeof(options));
    options.symbol_name = "me_dsl_jit_kernel";
    char *c_source = NULL;
    ok = me_dsl_jit_codegen_c(ir, ME_INT64, &options, &c_source, &cg_error);
    me_dsl_jit_ir_free(ir);
    if (!ok || !c_source) {
        printf("  FAILED: codegen rejected element loop-control source at %d:%d (%s)\n",
               cg_error.line, cg_error.column, cg_error.message);
        free(c_source);
        return 1;
    }

#if !defined(_WIN32) && !defined(_WIN64)
    if (compile_generated_source(c_source) != 0) {
        printf("  FAILED: generated C did not compile for element loop-control source\n");
        free(c_source);
        return 1;
    }
#endif

    free(c_source);
    printf("  PASSED\n");
    return 0;
}

static int test_codegen_math_alias_rewrite(void) {
    printf("\n=== JIT C Codegen Test 4: math alias rewrite ===\n");

    const char *src =
        "# me:dialect=element\n"
        "def kernel(x):\n"
        "    t0 = sinpi(x) + cospi(x)\n"
        "    t1 = exp10(x) + logaddexp(x, 1.0)\n"
        "    t2 = where(1, t0, t1)\n"
        "    return arctan2(t2, 1.0)\n";

    me_dsl_error parse_error;
    me_dsl_program *program = me_dsl_parse(src, &parse_error);
    if (!program) {
        printf("  FAILED: parse error at %d:%d (%s)\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    dtype_resolve_ctx rctx;
    rctx.value_dtype = ME_FLOAT64;

    me_dsl_error ir_error;
    me_dsl_jit_ir_program *ir = NULL;
    bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                                  mock_resolve_dtype, &rctx, &ir, &ir_error);
    me_dsl_program_free(program);
    if (!ok || !ir) {
        printf("  FAILED: ir build rejected source at %d:%d (%s)\n",
               ir_error.line, ir_error.column, ir_error.message);
        me_dsl_jit_ir_free(ir);
        return 1;
    }

    me_dsl_error cg_error;
    char *c_source = NULL;
    ok = me_dsl_jit_codegen_c(ir, ME_FLOAT64, NULL, &c_source, &cg_error);
    me_dsl_jit_ir_free(ir);
    if (!ok || !c_source) {
        printf("  FAILED: codegen rejected source at %d:%d (%s)\n",
               cg_error.line, cg_error.column, cg_error.message);
        free(c_source);
        return 1;
    }

    if (!strstr(c_source, "me_jit_sinpi(") ||
        !strstr(c_source, "me_jit_cospi(") ||
        !strstr(c_source, "me_jit_exp10(") ||
        !strstr(c_source, "me_jit_logaddexp(") ||
        !strstr(c_source, "me_jit_where(") ||
        !strstr(c_source, "atan2(") ||
        strstr(c_source, "arctan2(")) {
        printf("  FAILED: expected math alias rewrite markers not found in generated source\n");
        free(c_source);
        return 1;
    }

    free(c_source);
    printf("  PASSED\n");
    return 0;
}

static int test_codegen_runtime_math_bridge_emission(void) {
    printf("\n=== JIT C Codegen Test 5: runtime math bridge emission ===\n");

    const char *src =
        "# me:dialect=element\n"
        "def kernel(x):\n"
        "    return sinpi(x) + exp10(x) + where(1, x, 0)\n";

    me_dsl_error parse_error;
    me_dsl_program *program = me_dsl_parse(src, &parse_error);
    if (!program) {
        printf("  FAILED: parse error at %d:%d (%s)\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    dtype_resolve_ctx rctx;
    rctx.value_dtype = ME_FLOAT64;

    me_dsl_error ir_error;
    me_dsl_jit_ir_program *ir = NULL;
    bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                                  mock_resolve_dtype, &rctx, &ir, &ir_error);
    me_dsl_program_free(program);
    if (!ok || !ir) {
        printf("  FAILED: ir build rejected source at %d:%d (%s)\n",
               ir_error.line, ir_error.column, ir_error.message);
        me_dsl_jit_ir_free(ir);
        return 1;
    }

    me_dsl_error cg_error;
    me_dsl_jit_cgen_options options;
    memset(&options, 0, sizeof(options));
    options.use_runtime_math_bridge = true;
    char *c_source = NULL;
    ok = me_dsl_jit_codegen_c(ir, ME_FLOAT64, &options, &c_source, &cg_error);
    me_dsl_jit_ir_free(ir);
    if (!ok || !c_source) {
        printf("  FAILED: codegen rejected source at %d:%d (%s)\n",
               cg_error.line, cg_error.column, cg_error.message);
        free(c_source);
        return 1;
    }

    if (!strstr(c_source, "extern double me_jit_exp10(double);") ||
        !strstr(c_source, "extern double me_jit_sinpi(double);") ||
        !strstr(c_source, "extern double me_jit_where(double, double, double);") ||
        strstr(c_source, "static double me_jit_exp10(") ||
        strstr(c_source, "static double me_jit_sinpi(") ||
        strstr(c_source, "static double me_jit_where(")) {
        printf("  FAILED: runtime bridge codegen markers not emitted as expected\n");
        free(c_source);
        return 1;
    }

    free(c_source);
    printf("  PASSED\n");
    return 0;
}

static int test_codegen_runtime_math_bridge_vector_lowering(void) {
    printf("\n=== JIT C Codegen Test 6: runtime math bridge vector lowering ===\n");

    const char *src =
        "# me:dialect=element\n"
        "def kernel(x):\n"
        "    return exp(x)\n";

    me_dsl_error parse_error;
    me_dsl_program *program = me_dsl_parse(src, &parse_error);
    if (!program) {
        printf("  FAILED: parse error at %d:%d (%s)\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    dtype_resolve_ctx rctx;
    rctx.value_dtype = ME_FLOAT64;

    me_dsl_error ir_error;
    me_dsl_jit_ir_program *ir = NULL;
    bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                                  mock_resolve_dtype, &rctx, &ir, &ir_error);
    me_dsl_program_free(program);
    if (!ok || !ir) {
        printf("  FAILED: ir build rejected source at %d:%d (%s)\n",
               ir_error.line, ir_error.column, ir_error.message);
        me_dsl_jit_ir_free(ir);
        return 1;
    }

    me_dsl_error cg_error;
    me_dsl_jit_cgen_options options;
    memset(&options, 0, sizeof(options));
    options.use_runtime_math_bridge = true;
    char *c_source = NULL;
    ok = me_dsl_jit_codegen_c(ir, ME_FLOAT64, &options, &c_source, &cg_error);
    me_dsl_jit_ir_free(ir);
    if (!ok || !c_source) {
        printf("  FAILED: codegen rejected source at %d:%d (%s)\n",
               cg_error.line, cg_error.column, cg_error.message);
        free(c_source);
        return 1;
    }

    if (!strstr(c_source, "me_jit_vec_exp_f64(in_x, out, nitems);") ||
        strstr(c_source, "for (int64_t idx = 0; idx < nitems; idx++) {")) {
        printf("  FAILED: vector bridge lowering markers not emitted as expected\n");
        free(c_source);
        return 1;
    }

    free(c_source);
    printf("  PASSED\n");
    return 0;
}

static int test_codegen_runtime_math_bridge_vector_lowering_binary(void) {
    printf("\n=== JIT C Codegen Test 7: runtime math bridge vector binary lowering ===\n");

    const char *src =
        "# me:dialect=element\n"
        "def kernel(x, y):\n"
        "    return atan2(y, x)\n";

    me_dsl_error parse_error;
    me_dsl_program *program = me_dsl_parse(src, &parse_error);
    if (!program) {
        printf("  FAILED: parse error at %d:%d (%s)\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }

    const char *param_names[] = {"x", "y"};
    me_dtype param_dtypes[] = {ME_FLOAT64, ME_FLOAT64};
    dtype_resolve_ctx rctx;
    rctx.value_dtype = ME_FLOAT64;

    me_dsl_error ir_error;
    me_dsl_jit_ir_program *ir = NULL;
    bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 2,
                                  mock_resolve_dtype, &rctx, &ir, &ir_error);
    me_dsl_program_free(program);
    if (!ok || !ir) {
        printf("  FAILED: ir build rejected source at %d:%d (%s)\n",
               ir_error.line, ir_error.column, ir_error.message);
        me_dsl_jit_ir_free(ir);
        return 1;
    }

    me_dsl_error cg_error;
    me_dsl_jit_cgen_options options;
    memset(&options, 0, sizeof(options));
    options.use_runtime_math_bridge = true;
    char *c_source = NULL;
    ok = me_dsl_jit_codegen_c(ir, ME_FLOAT64, &options, &c_source, &cg_error);
    me_dsl_jit_ir_free(ir);
    if (!ok || !c_source) {
        printf("  FAILED: codegen rejected source at %d:%d (%s)\n",
               cg_error.line, cg_error.column, cg_error.message);
        free(c_source);
        return 1;
    }

    if (!strstr(c_source, "me_jit_vec_atan2_f64(in_y, in_x, out, nitems);") ||
        strstr(c_source, "for (int64_t idx = 0; idx < nitems; idx++) {")) {
        printf("  FAILED: vector binary bridge lowering markers not emitted as expected\n");
        free(c_source);
        return 1;
    }

    free(c_source);
    printf("  PASSED\n");
    return 0;
}

int main(void) {
    int fail = 0;
    fail |= test_codegen_all_noncomplex_dtypes();
    fail |= test_codegen_rejects_unsupported_expression_ops();
    fail |= test_codegen_element_loop_control();
    fail |= test_codegen_math_alias_rewrite();
    fail |= test_codegen_runtime_math_bridge_emission();
    fail |= test_codegen_runtime_math_bridge_vector_lowering();
    fail |= test_codegen_runtime_math_bridge_vector_lowering_binary();
    return fail;
}
