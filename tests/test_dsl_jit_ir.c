/*
 * JIT IR builder tests for DSL kernels.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/dsl_jit_ir.h"
#include "../src/dsl_parser.h"
#include "minctest.h"

static bool mock_resolve_dtype(void *ctx, const me_dsl_expr *expr, me_dtype *out_dtype) {
    (void)ctx;
    if (!expr || !expr->text || !out_dtype) {
        return false;
    }

    const char *text = expr->text;
    size_t len = strlen(text);
    bool all_digits = (len > 0);
    for (size_t i = 0; i < len; i++) {
        if (text[i] < '0' || text[i] > '9') {
            all_digits = false;
            break;
        }
    }
    if (all_digits) {
        *out_dtype = ME_INT64;
        return true;
    }

    if (strstr(text, "==") || strstr(text, "!=") || strstr(text, "<=") ||
        strstr(text, ">=") || strstr(text, "<") || strstr(text, ">")) {
        *out_dtype = ME_BOOL;
        return true;
    }

    *out_dtype = ME_FLOAT64;
    return true;
}

static int test_ir_accepts_supported_subset(void) {
    printf("\n=== JIT IR Test 1: supported subset ===\n");

    const char *src =
        "def kernel(x):\n"
        "    acc = 0.0\n"
        "    for i in range(8):\n"
        "        if i == 2:\n"
        "            continue\n"
        "        acc = acc + x\n"
        "        if i == 4:\n"
        "            break\n"
        "    if acc:\n"
        "        return acc\n"
        "    else:\n"
        "        return 1.0\n";

    me_dsl_error parse_error;
    me_dsl_program *program = me_dsl_parse(src, &parse_error);
    if (!program) {
        printf("  FAILED: parse error at %d:%d (%s)\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    me_dsl_error build_error;
    me_dsl_jit_ir_program *ir = NULL;

    bool ok = me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                                  mock_resolve_dtype, NULL, &ir, &build_error);
    me_dsl_program_free(program);
    if (!ok || !ir) {
        printf("  FAILED: jit ir build rejected supported kernel at %d:%d (%s)\n",
               build_error.line, build_error.column, build_error.message);
        me_dsl_jit_ir_free(ir);
        return 1;
    }

    uint64_t fp = me_dsl_jit_ir_fingerprint(ir);
    if (fp == 0) {
        printf("  FAILED: jit ir fingerprint should be non-zero\n");
        me_dsl_jit_ir_free(ir);
        return 1;
    }
    me_dsl_jit_ir_free(ir);
    printf("  PASSED\n");
    return 0;
}

static int test_ir_rejects_unsupported_statements(void) {
    printf("\n=== JIT IR Test 2: unsupported statements ===\n");

    const char *src_expr_stmt =
        "def kernel(x):\n"
        "    x + 1\n"
        "    return x\n";

    const char *src_print =
        "def kernel(x):\n"
        "    print(x)\n"
        "    return x\n";

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    me_dsl_error error;
    me_dsl_jit_ir_program *ir = NULL;

    me_dsl_program *program = me_dsl_parse(src_expr_stmt, &error);
    if (!program) {
        printf("  FAILED: parse error for expression-statement source\n");
        return 1;
    }
    if (me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                            mock_resolve_dtype, NULL, &ir, &error)) {
        printf("  FAILED: expression statement should be rejected by jit ir subset\n");
        me_dsl_jit_ir_free(ir);
        me_dsl_program_free(program);
        return 1;
    }
    me_dsl_program_free(program);

    program = me_dsl_parse(src_print, &error);
    if (!program) {
        printf("  FAILED: parse error for print source\n");
        return 1;
    }
    if (me_dsl_jit_ir_build(program, param_names, param_dtypes, 1,
                            mock_resolve_dtype, NULL, &ir, &error)) {
        printf("  FAILED: print should be rejected by jit ir subset\n");
        me_dsl_jit_ir_free(ir);
        me_dsl_program_free(program);
        return 1;
    }
    me_dsl_program_free(program);

    printf("  PASSED\n");
    return 0;
}

static int test_ir_fingerprint_is_deterministic(void) {
    printf("\n=== JIT IR Test 3: deterministic fingerprint ===\n");

    const char *src =
        "def kernel(x):\n"
        "    y = x + 1\n"
        "    if y > 3:\n"
        "        return y\n"
        "    return 3\n";

    me_dsl_error error;
    me_dsl_program *program_a = me_dsl_parse(src, &error);
    if (!program_a) {
        printf("  FAILED: parse error for program A\n");
        return 1;
    }
    me_dsl_program *program_b = me_dsl_parse(src, &error);
    if (!program_b) {
        printf("  FAILED: parse error for program B\n");
        me_dsl_program_free(program_a);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    me_dsl_jit_ir_program *ir_a = NULL;
    me_dsl_jit_ir_program *ir_b = NULL;

    if (!me_dsl_jit_ir_build(program_a, param_names, param_dtypes, 1,
                             mock_resolve_dtype, NULL, &ir_a, &error) || !ir_a) {
        printf("  FAILED: jit ir build failed for program A\n");
        me_dsl_program_free(program_a);
        me_dsl_program_free(program_b);
        me_dsl_jit_ir_free(ir_a);
        return 1;
    }

    if (!me_dsl_jit_ir_build(program_b, param_names, param_dtypes, 1,
                             mock_resolve_dtype, NULL, &ir_b, &error) || !ir_b) {
        printf("  FAILED: jit ir build failed for program B\n");
        me_dsl_program_free(program_a);
        me_dsl_program_free(program_b);
        me_dsl_jit_ir_free(ir_a);
        me_dsl_jit_ir_free(ir_b);
        return 1;
    }

    uint64_t fp_a = me_dsl_jit_ir_fingerprint(ir_a);
    uint64_t fp_b = me_dsl_jit_ir_fingerprint(ir_b);
    if (fp_a != fp_b) {
        printf("  FAILED: fingerprint mismatch (%llu vs %llu)\n",
               (unsigned long long)fp_a, (unsigned long long)fp_b);
        me_dsl_program_free(program_a);
        me_dsl_program_free(program_b);
        me_dsl_jit_ir_free(ir_a);
        me_dsl_jit_ir_free(ir_b);
        return 1;
    }

    me_dsl_program_free(program_a);
    me_dsl_program_free(program_b);
    me_dsl_jit_ir_free(ir_a);
    me_dsl_jit_ir_free(ir_b);
    printf("  PASSED\n");
    return 0;
}

static int test_parser_dialect_pragma(void) {
    printf("\n=== JIT IR Test 4: parser dialect pragma ===\n");

    me_dsl_error error;

    const char *src_default =
        "def kernel(x):\n"
        "    return x\n";
    me_dsl_program *program = me_dsl_parse(src_default, &error);
    if (!program) {
        printf("  FAILED: parse error for default dialect source\n");
        return 1;
    }
    if (program->dialect != ME_DSL_DIALECT_VECTOR) {
        printf("  FAILED: default dialect should be vector\n");
        me_dsl_program_free(program);
        return 1;
    }
    me_dsl_program_free(program);

    const char *src_element =
        "# me:dialect=element\n"
        "def kernel(x):\n"
        "    return x\n";
    program = me_dsl_parse(src_element, &error);
    if (!program) {
        printf("  FAILED: parse error for element dialect source\n");
        return 1;
    }
    if (program->dialect != ME_DSL_DIALECT_ELEMENT) {
        printf("  FAILED: element dialect pragma not detected\n");
        me_dsl_program_free(program);
        return 1;
    }
    me_dsl_program_free(program);

    const char *src_unknown =
        "# me:dialect=unknown\n"
        "def kernel(x):\n"
        "    return x\n";
    program = me_dsl_parse(src_unknown, &error);
    if (program) {
        printf("  FAILED: unknown dialect pragma should fail parse\n");
        me_dsl_program_free(program);
        return 1;
    }

    printf("  PASSED\n");
    return 0;
}

static int test_ir_fingerprint_includes_dialect(void) {
    printf("\n=== JIT IR Test 5: fingerprint includes dialect ===\n");

    const char *src_vector =
        "# me:dialect=vector\n"
        "def kernel(x):\n"
        "    y = x + 1\n"
        "    return y\n";
    const char *src_element =
        "# me:dialect=element\n"
        "def kernel(x):\n"
        "    y = x + 1\n"
        "    return y\n";

    me_dsl_error error;
    me_dsl_program *program_vector = me_dsl_parse(src_vector, &error);
    if (!program_vector) {
        printf("  FAILED: parse error for vector source\n");
        return 1;
    }
    me_dsl_program *program_element = me_dsl_parse(src_element, &error);
    if (!program_element) {
        printf("  FAILED: parse error for element source\n");
        me_dsl_program_free(program_vector);
        return 1;
    }

    const char *param_names[] = {"x"};
    me_dtype param_dtypes[] = {ME_FLOAT64};
    me_dsl_jit_ir_program *ir_vector = NULL;
    me_dsl_jit_ir_program *ir_element = NULL;

    if (!me_dsl_jit_ir_build(program_vector, param_names, param_dtypes, 1,
                             mock_resolve_dtype, NULL, &ir_vector, &error) || !ir_vector) {
        printf("  FAILED: jit ir build failed for vector program\n");
        me_dsl_program_free(program_vector);
        me_dsl_program_free(program_element);
        me_dsl_jit_ir_free(ir_vector);
        return 1;
    }
    if (!me_dsl_jit_ir_build(program_element, param_names, param_dtypes, 1,
                             mock_resolve_dtype, NULL, &ir_element, &error) || !ir_element) {
        printf("  FAILED: jit ir build failed for element program\n");
        me_dsl_program_free(program_vector);
        me_dsl_program_free(program_element);
        me_dsl_jit_ir_free(ir_vector);
        me_dsl_jit_ir_free(ir_element);
        return 1;
    }

    uint64_t fp_vector = me_dsl_jit_ir_fingerprint(ir_vector);
    uint64_t fp_element = me_dsl_jit_ir_fingerprint(ir_element);
    if (fp_vector == fp_element) {
        printf("  FAILED: fingerprints should differ by dialect\n");
        me_dsl_program_free(program_vector);
        me_dsl_program_free(program_element);
        me_dsl_jit_ir_free(ir_vector);
        me_dsl_jit_ir_free(ir_element);
        return 1;
    }

    me_dsl_program_free(program_vector);
    me_dsl_program_free(program_element);
    me_dsl_jit_ir_free(ir_vector);
    me_dsl_jit_ir_free(ir_element);
    printf("  PASSED\n");
    return 0;
}

int main(void) {
    int fail = 0;
    fail |= test_ir_accepts_supported_subset();
    fail |= test_ir_rejects_unsupported_statements();
    fail |= test_ir_fingerprint_is_deterministic();
    fail |= test_parser_dialect_pragma();
    fail |= test_ir_fingerprint_includes_dialect();
    return fail;
}
