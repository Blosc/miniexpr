/* Tests for string-valued expression output (concatenation and width inference) */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"
#include "minctest.h"

#define ITEMS 4

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

/* Fill a fixed-width UCS4 slot from an ASCII literal. */
static void put(uint32_t *base, size_t units, int idx, const char *ascii) {
    uint32_t *slot = base + (size_t)idx * units;
    memset(slot, 0, units * sizeof(uint32_t));
    for (size_t i = 0; ascii[i] && i + 1 < units; i++) {
        slot[i] = (uint32_t)(unsigned char)ascii[i];
    }
}

static void expect_slot(const uint32_t *base, size_t units, int idx,
                        const char *ascii, const char *label) {
    const uint32_t *slot = base + (size_t)idx * units;
    size_t i = 0;
    for (; ascii[i]; i++) {
        if (i >= units || slot[i] != (uint32_t)(unsigned char)ascii[i]) {
            printf("  FAIL %s at [%d]: mismatch on char %zu\n", label, idx, i);
            tests_failed++;
            return;
        }
    }
    /* A value that exactly fills the slot carries no terminator (NumPy does
     * the same); a shorter one must be NUL-padded. */
    if (i < units && slot[i] != 0) {
        printf("  FAIL %s at [%d]: missing NUL padding\n", label, idx);
        tests_failed++;
        return;
    }
}

static void expect_itemsize(const me_expr *expr, size_t want, const char *label) {
    size_t got = me_get_itemsize(expr);
    if (got != want) {
        printf("  FAIL %s: itemsize %zu, expected %zu\n", label, got, want);
        tests_failed++;
    }
}

static void test_concat_var_var(void) {
    TEST("a + b with differing itemsizes");
    const size_t au = 8, bu = 6;
    uint32_t a[ITEMS * 8], b[ITEMS * 6];
    put(a, au, 0, "foo");     put(b, bu, 0, "bar");
    put(a, au, 1, "");        put(b, bu, 1, "x");
    put(a, au, 2, "abcdefg"); put(b, bu, 2, "");
    put(a, au, 3, "hi");      put(b, bu, 3, "there");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
        {"b", ME_STRING, b, ME_VARIABLE, NULL, bu * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + b", vars, 2, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    if (me_get_dtype(expr) != ME_STRING) {
        printf("  FAIL: output dtype is not ME_STRING\n");
        tests_failed++;
        me_free(expr);
        return;
    }
    /* NumPy widths add directly: 8 + 6 = 14 units = 56 bytes */
    const size_t ou = 14;
    expect_itemsize(expr, ou * sizeof(uint32_t), "a + b");

    uint32_t out[ITEMS * 14];
    memset(out, 0xFF, sizeof(out));
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, ptrs, 2, out, ITEMS);

    expect_slot(out, ou, 0, "foobar", "a + b");
    expect_slot(out, ou, 1, "x", "a + b");
    expect_slot(out, ou, 2, "abcdefg", "a + b");
    expect_slot(out, ou, 3, "hithere", "a + b");
    printf("  PASS a + b\n");

    me_free(expr);
}

static void test_concat_literal(void) {
    TEST("'kind=' + a");
    const size_t au = 8;
    uint32_t a[ITEMS * 8];
    put(a, au, 0, "home");
    put(a, au, 1, "room");
    put(a, au, 2, "");
    put(a, au, 3, "loft");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("'kind=' + a", vars, 1, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    /* literal is 5 chars = 5 units; 5 + 8 = 13 units */
    const size_t ou = 13;
    expect_itemsize(expr, ou * sizeof(uint32_t), "literal concat");

    uint32_t out[ITEMS * 13];
    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, out, ITEMS);

    expect_slot(out, ou, 0, "kind=home", "literal concat");
    expect_slot(out, ou, 1, "kind=room", "literal concat");
    expect_slot(out, ou, 2, "kind=", "literal concat");
    expect_slot(out, ou, 3, "kind=loft", "literal concat");
    printf("  PASS literal concat\n");

    me_free(expr);
}

static void test_concat_nested(void) {
    TEST("a + '-' + b (nested concat)");
    const size_t au = 6, bu = 6;
    uint32_t a[ITEMS * 6], b[ITEMS * 6];
    for (int i = 0; i < ITEMS; i++) {
        put(a, au, i, "ab");
        put(b, bu, i, "cd");
    }

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
        {"b", ME_STRING, b, ME_VARIABLE, NULL, bu * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + '-' + b", vars, 2, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    /* (6 + 1) = 7 units, then 7 + 6 = 13 units */
    const size_t ou = 13;
    expect_itemsize(expr, ou * sizeof(uint32_t), "nested concat");

    uint32_t out[ITEMS * 13];
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, ptrs, 2, out, ITEMS);

    for (int i = 0; i < ITEMS; i++) {
        expect_slot(out, ou, i, "ab-cd", "nested concat");
    }
    printf("  PASS nested concat\n");

    me_free(expr);
}

static void test_concat_fills_bound(void) {
    TEST("concat exactly filling the inferred bound");
    const size_t au = 4, bu = 4; /* 3 usable chars each */
    uint32_t a[ITEMS * 4], b[ITEMS * 4];
    for (int i = 0; i < ITEMS; i++) {
        put(a, au, i, "xyz");
        put(b, bu, i, "pqr");
    }

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
        {"b", ME_STRING, b, ME_VARIABLE, NULL, bu * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + b", vars, 2, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    /* 4 + 4 = 8 units: exactly the 6 chars, with 2 NULs of padding */
    const size_t ou = 8;
    expect_itemsize(expr, ou * sizeof(uint32_t), "bound-filling concat");

    uint32_t out[ITEMS * 8];
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(expr, ptrs, 2, out, ITEMS);

    for (int i = 0; i < ITEMS; i++) {
        expect_slot(out, ou, i, "xyzpqr", "bound-filling concat");
    }
    printf("  PASS bound-filling concat\n");

    me_free(expr);
}

static void test_non_ascii(void) {
    TEST("concat preserves non-ASCII codepoints");
    const size_t au = 6;
    uint32_t a[ITEMS * 6];
    for (int i = 0; i < ITEMS; i++) {
        memset(a + (size_t)i * au, 0, au * sizeof(uint32_t));
        a[i * au + 0] = 0x00E9; /* e-acute */
        a[i * au + 1] = 0x4F60; /* CJK ni */
    }

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("a + '!'", vars, 1, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    const size_t ou = 7; /* 6 + 1 = 7 units */
    uint32_t out[ITEMS * 7];
    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, out, ITEMS);

    for (int i = 0; i < ITEMS; i++) {
        const uint32_t *slot = out + (size_t)i * ou;
        if (slot[0] != 0x00E9 || slot[1] != 0x4F60 ||
            slot[2] != (uint32_t)'!' || slot[3] != 0) {
            printf("  FAIL non-ascii at [%d]\n", i);
            tests_failed++;
            me_free(expr);
            return;
        }
    }
    printf("  PASS non-ascii concat\n");

    me_free(expr);
}

static void test_numeric_add_unaffected(void) {
    TEST("numeric + still evaluates arithmetically");
    double x[ITEMS] = {1.0, 2.0, 3.0, 4.0};
    me_variable vars[] = {
        {"x", ME_FLOAT64, x, ME_VARIABLE, NULL, 0},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("x + 10", vars, 1, ME_FLOAT64, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    double out[ITEMS];
    const void *ptrs[] = {x};
    ME_EVAL_CHECK(expr, ptrs, 1, out, ITEMS);
    for (int i = 0; i < ITEMS; i++) {
        if (out[i] != x[i] + 10.0) {
            printf("  FAIL numeric add at [%d]: got %f\n", i, out[i]);
            tests_failed++;
            me_free(expr);
            return;
        }
    }
    printf("  PASS numeric add unaffected\n");

    me_free(expr);
}

/* Run a single-operand string expression over `in` and check each result. */
static void run_unary(const char *expr_str, size_t in_units, const char *const *inputs,
                      const char *const *expected, size_t out_units) {
    TEST(expr_str);
    uint32_t *a = calloc(ITEMS * in_units, sizeof(uint32_t));
    uint32_t *out = calloc(ITEMS * out_units, sizeof(uint32_t));
    for (int i = 0; i < ITEMS; i++) {
        put(a, in_units, i, inputs[i]);
    }

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, in_units * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile(expr_str, vars, 1, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        free(a); free(out);
        return;
    }

    size_t got_units = me_get_itemsize(expr) / sizeof(uint32_t);
    if (got_units > out_units) {
        printf("  FAIL: inferred %zu units exceeds test buffer %zu\n", got_units, out_units);
        tests_failed++;
        me_free(expr); free(a); free(out);
        return;
    }

    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, out, ITEMS);
    for (int i = 0; i < ITEMS; i++) {
        expect_slot(out, got_units, i, expected[i], expr_str);
    }
    printf("  PASS %s\n", expr_str);

    me_free(expr);
    free(a);
    free(out);
}

static void test_case_mapping(void) {
    static const char *in[ITEMS] = {"Hello", "WORLD", "MiXeD", ""};
    static const char *lo[ITEMS] = {"hello", "world", "mixed", ""};
    static const char *up[ITEMS] = {"HELLO", "WORLD", "MIXED", ""};
    run_unary("lower(a)", 8, in, lo, 32);
    run_unary("upper(a)", 8, in, up, 48);
}

static void test_case_expansion(void) {
    /* NumPy (and Python) expand these; a 1:1 table would silently disagree. */
    TEST("upper() expands eszett like NumPy");
    const size_t au = 8;
    uint32_t a[ITEMS * 8];
    const uint32_t src[] = {'s', 't', 'r', 'a', 0x00DF, 'e'};
    for (int i = 0; i < ITEMS; i++) {
        memset(a + (size_t)i * au, 0, au * sizeof(uint32_t));
        memcpy(a + (size_t)i * au, src, sizeof(src));
    }

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("upper(a)", vars, 1, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    const size_t ou = me_get_itemsize(expr) / sizeof(uint32_t);
    uint32_t *out = calloc(ITEMS * ou, sizeof(uint32_t));
    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, out, ITEMS);
    for (int i = 0; i < ITEMS; i++) {
        expect_slot(out, ou, i, "STRASSE", "upper eszett");
    }
    printf("  PASS upper() expands eszett\n");

    free(out);
    me_free(expr);
}

static void test_strip_family(void) {
    static const char *in[ITEMS] = {"  pad  ", "left  ", "  right", "none"};
    static const char *both[ITEMS] = {"pad", "left", "right", "none"};
    static const char *lft[ITEMS] = {"pad  ", "left  ", "right", "none"};
    static const char *rgt[ITEMS] = {"  pad", "left", "  right", "none"};
    run_unary("strip(a)", 10, in, both, 10);
    run_unary("lstrip(a)", 10, in, lft, 10);
    run_unary("rstrip(a)", 10, in, rgt, 10);
}

static void test_prefix_suffix(void) {
    static const char *in[ITEMS] = {"double room", "suite", "single room", "room"};
    static const char *nosuf[ITEMS] = {"double", "suite", "single", "room"};
    run_unary("removesuffix(a, ' room')", 14, in, nosuf, 14);

    static const char *in2[ITEMS] = {"pre_a", "pre_b", "c", "pre_"};
    static const char *nopre[ITEMS] = {"a", "b", "c", ""};
    run_unary("removeprefix(a, 'pre_')", 8, in2, nopre, 8);
}

static void test_split_part(void) {
    static const char *in[ITEMS] = {"loft with view", "plain", "a with b with c", " with x"};
    static const char *head[ITEMS] = {"loft", "plain", "a", ""};
    static const char *tail[ITEMS] = {"view", "", "b with c", "x"};
    run_unary("split_part(a, \' with \', 0)", 18, in, head, 18);
    run_unary("split_part(a, \' with \', 1)", 18, in, tail, 18);
}

static void test_replace_and_substr(void) {
    static const char *in[ITEMS] = {"a-b-c", "no", "---", ""};
    static const char *rep[ITEMS] = {"a+b+c", "no", "+++", ""};
    run_unary("replace(a, '-', '+')", 8, in, rep, 24);

    static const char *in2[ITEMS] = {"abcdef", "ab", "", "xyz"};
    static const char *sub[ITEMS] = {"bcd", "b", "", "yz"};
    run_unary("substr(a, 1, 3)", 8, in2, sub, 8);

    static const char *tailc[ITEMS] = {"ef", "ab", "", "yz"};
    run_unary("substr(a, -2, 2)", 8, in2, tailc, 8);
}

static void test_blog_kernel_shape(void) {
    /* The pandas-3 blog kernel's core expression, in function-call form. */
    TEST("nested lower/removesuffix/concat");
    const size_t au = 32;
    uint32_t a[ITEMS * 32];
    put(a, au, 0, "Cozy Loft With City View");
    put(a, au, 1, "Small Single Room");
    put(a, au, 2, "Studio");
    put(a, au, 3, "Double Room");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    int rc = me_compile("'room_type=' + removesuffix(lower(a), ' room')",
                        vars, 1, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }

    const size_t ou = me_get_itemsize(expr) / sizeof(uint32_t);
    uint32_t *out = calloc(ITEMS * ou, sizeof(uint32_t));
    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, out, ITEMS);

    expect_slot(out, ou, 0, "room_type=cozy loft with city view", "blog kernel");
    expect_slot(out, ou, 1, "room_type=small single", "blog kernel");
    expect_slot(out, ou, 2, "room_type=studio", "blog kernel");
    expect_slot(out, ou, 3, "room_type=double", "blog kernel");
    printf("  PASS nested lower/removesuffix/concat\n");

    free(out);
    me_free(expr);
}

static void test_literal_with_dsl_chars(void) {
    /* Regression: dsl_is_candidate() used to scan the raw source for '=', ';'
     * and DSL keywords without skipping string literals, so any expression
     * containing e.g. 'property_type=' was misrouted to the DSL parser. */
    TEST("string literals containing DSL syntax characters");
    const char *cases[] = {"'='", "'a=b'", "'x;y'", "'if'", "'def f'", NULL};

    static uint32_t a[ITEMS * 8];
    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, 8 * sizeof(uint32_t)},
    };

    for (int i = 0; cases[i]; i++) {
        int err;
        me_expr *expr = NULL;
        int rc = me_compile(cases[i], vars, 1, ME_AUTO, &err, &expr);
        if (rc != ME_COMPILE_SUCCESS) {
            printf("  FAIL: %s did not compile (rc=%d)\n", cases[i], rc);
            tests_failed++;
            continue;
        }
        me_free(expr);
    }
    printf("  PASS literals with DSL characters\n");
}

/* The chunked entry point (me_compile_nd/me_eval_nd) is what python-blosc2's
 * prefilter drives; it used to reject string output outright and to size
 * buffers with dtype_size(), which is 0 for ME_STRING. */
static void test_nd_string_output(void) {
    TEST("me_eval_nd with string output (expression and DSL)");
    const size_t au = 8;
    static uint32_t a[ITEMS * 8];
    put(a, au, 0, "foo");
    put(a, au, 1, "bar");
    put(a, au, 2, "");
    put(a, au, 3, "abcdefg");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };
    const int64_t shape[1] = {ITEMS};
    const int32_t chunkshape[1] = {ITEMS};
    const int32_t blockshape[1] = {ITEMS};
    const char *sources[] = {"'p=' + a", "def k(a):\n    r = 'p=' + a\n    return r", NULL};

    for (int c = 0; sources[c]; c++) {
        int err;
        me_expr *expr = NULL;
        int rc = me_compile_nd(sources[c], vars, 1, ME_AUTO, 1, shape, chunkshape,
                               blockshape, &err, &expr);
        if (rc != ME_COMPILE_SUCCESS) {
            printf("  FAIL: case %d did not compile (rc=%d)\n", c, rc);
            tests_failed++;
            continue;
        }
        const size_t ou = me_get_itemsize(expr) / sizeof(uint32_t);
        expect_itemsize(expr, (au + 2) * sizeof(uint32_t), "nd concat width");
        uint32_t *out = calloc(ITEMS, me_get_itemsize(expr));
        const void *ptrs[] = {a};
        rc = me_eval_nd(expr, ptrs, 1, out, ITEMS, 0, 0, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            printf("  FAIL: case %d did not evaluate (rc=%d)\n", c, rc);
            tests_failed++;
        }
        else {
            expect_slot(out, ou, 0, "p=foo", "nd concat");
            expect_slot(out, ou, 1, "p=bar", "nd concat");
            expect_slot(out, ou, 2, "p=", "nd concat");
            expect_slot(out, ou, 3, "p=abcdefg", "nd concat");
        }
        free(out);
        me_free(expr);
    }
    printf("  PASS me_eval_nd string output\n");
}

/* A DSL kernel whose branches return strings of different widths: the output
 * slot is the widest of them, so the narrow branch must be NUL-padded into it
 * rather than written at its own stride. */
static void test_dsl_branch_widths(void) {
    TEST("DSL kernel with returns of differing widths");
    const size_t au = 8;
    static uint32_t a[ITEMS * 8];
    put(a, au, 0, "yes x");
    put(a, au, 1, "no");
    put(a, au, 2, "yes y");
    put(a, au, 3, "no");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };

    int err;
    me_expr *expr = NULL;
    const char *src = "def k(a):\n"
                      "    if contains(a, 'yes'):\n"
                      "        return a\n"
                      "    return 'long-prefix-' + a\n";
    int rc = me_compile(src, vars, 1, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return;
    }
    /* 'long-prefix-' is 12 units, so the wide branch is 8 + 12 = 20 units */
    const size_t ou = 20;
    expect_itemsize(expr, ou * sizeof(uint32_t), "branch widths");

    uint32_t *out = calloc(ITEMS, ou * sizeof(uint32_t));
    const void *ptrs[] = {a};
    ME_EVAL_CHECK(expr, ptrs, 1, out, ITEMS);
    expect_slot(out, ou, 0, "yes x", "branch widths");
    expect_slot(out, ou, 1, "long-prefix-no", "branch widths");
    expect_slot(out, ou, 2, "yes y", "branch widths");
    expect_slot(out, ou, 3, "long-prefix-no", "branch widths");
    printf("  PASS DSL branch widths\n");

    free(out);
    me_free(expr);
}

int main(void) {
    printf("=== String output tests ===\n\n");

    test_concat_var_var();
    test_concat_literal();
    test_concat_nested();
    test_concat_fills_bound();
    test_non_ascii();
    test_numeric_add_unaffected();
    test_case_mapping();
    test_case_expansion();
    test_strip_family();
    test_prefix_suffix();
    test_split_part();
    test_replace_and_substr();
    test_blog_kernel_shape();
    test_literal_with_dsl_chars();
    test_nd_string_output();
    test_dsl_branch_widths();

    printf("\n=== %d tests run, %d failures ===\n", tests_run, tests_failed);
    return tests_failed != 0;
}
