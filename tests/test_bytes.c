/* Tests for ME_BYTES: fixed-width 1-byte-unit strings (numpy 'S').
 *
 * The kernels are shared with ME_STRING and parametrised on the code-unit
 * width, so these check the places where 'S' must differ from 'U': the width
 * bounds (no case-mapping expansion), ASCII-only case mapping and stripping,
 * and the refusal to mix the two families. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"

#define ITEMS 4
#define AU 8
#define BU 6

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

static char a_data[ITEMS * AU];
static char b_data[ITEMS * BU];

static void put(char *base, size_t units, int idx, const char *ascii) {
    char *slot = base + (size_t)idx * units;
    memset(slot, 0, units);
    /* NumPy `Sn` semantics: NUL-padded, no terminator when the value fills it. */
    strncpy(slot, ascii, units);
}

static void fill_inputs(void) {
    put(a_data, AU, 0, "foo");
    put(a_data, AU, 1, "Hello");
    put(a_data, AU, 2, " pad  ");
    put(a_data, AU, 3, "abcdefgh");
    put(b_data, BU, 0, "bar");
    put(b_data, BU, 1, "x");
    put(b_data, BU, 2, "");
    put(b_data, BU, 3, "there");
}

static void expect_slot(const char *out, size_t units, int idx, const char *want,
                        const char *label) {
    const char *slot = out + (size_t)idx * units;
    const size_t wlen = strlen(want);
    if (wlen > units || memcmp(slot, want, wlen) != 0) {
        printf("  FAIL %s at [%d]: got '%.*s', want '%s'\n", label, idx, (int)units, slot, want);
        tests_failed++;
        return;
    }
    for (size_t i = wlen; i < units; i++) {
        if (slot[i] != 0) {
            printf("  FAIL %s at [%d]: missing NUL padding\n", label, idx);
            tests_failed++;
            return;
        }
    }
}

/* Compile *src*, evaluate over a_data (and b_data when nvars == 2), and compare
 * the four results.  want_itemsize of 0 skips the width check. */
static void check_string_case(const char *src, int nvars, size_t want_itemsize,
                              const char *w0, const char *w1, const char *w2, const char *w3) {
    me_variable vars[2] = {{0}, {0}};
    vars[0].name = "a"; vars[0].dtype = ME_BYTES; vars[0].address = a_data; vars[0].itemsize = AU;
    vars[1].name = "b"; vars[1].dtype = ME_BYTES; vars[1].address = b_data; vars[1].itemsize = BU;

    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile(src, vars, nvars, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL %s: compile rc=%d (%s)\n", src, rc, me_get_last_error_message());
        tests_failed++;
        return;
    }
    if (me_get_dtype(expr) != ME_BYTES) {
        printf("  FAIL %s: output dtype is %d, expected ME_BYTES\n", src, me_get_dtype(expr));
        tests_failed++;
        me_free(expr);
        return;
    }
    const size_t itemsize = me_get_itemsize(expr);
    if (want_itemsize && itemsize != want_itemsize) {
        printf("  FAIL %s: itemsize %zu, expected %zu\n", src, itemsize, want_itemsize);
        tests_failed++;
    }

    char *out = calloc(ITEMS, itemsize);
    const void *ptrs[] = {a_data, b_data};
    rc = me_eval(expr, ptrs, nvars, out, ITEMS, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL %s: eval rc=%d\n", src, rc);
        tests_failed++;
    }
    else {
        const char *want[4] = {w0, w1, w2, w3};
        for (int i = 0; i < ITEMS; i++) {
            expect_slot(out, itemsize, i, want[i], src);
        }
    }
    free(out);
    me_free(expr);
}

static void check_predicate(const char *src, bool e0, bool e1, bool e2, bool e3) {
    me_variable vars[1] = {{0}};
    vars[0].name = "a"; vars[0].dtype = ME_BYTES; vars[0].address = a_data; vars[0].itemsize = AU;

    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile(src, vars, 1, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL %s: compile rc=%d\n", src, rc);
        tests_failed++;
        return;
    }
    bool out[ITEMS] = {false};
    const void *ptrs[] = {a_data};
    rc = me_eval(expr, ptrs, 1, out, ITEMS, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL %s: eval rc=%d\n", src, rc);
        tests_failed++;
    }
    else {
        const bool want[4] = {e0, e1, e2, e3};
        for (int i = 0; i < ITEMS; i++) {
            if (out[i] != want[i]) {
                printf("  FAIL %s at [%d]: got %d, want %d\n", src, i, out[i], want[i]);
                tests_failed++;
            }
        }
    }
    me_free(expr);
}

static void test_string_ops(void) {
    TEST("string-valued ops over ME_BYTES");
    check_string_case("a + b", 2, AU + BU, "foobar", "Hellox", " pad  ", "abcdefghthere");
    check_string_case("'kind=' + a", 1, 5 + AU, "kind=foo", "kind=Hello", "kind= pad  ",
                      "kind=abcdefgh");
    /* No 3x/2x case-expansion bound: NumPy's `S` mapping is ASCII-only and 1:1. */
    check_string_case("upper(a)", 1, AU, "FOO", "HELLO", " PAD  ", "ABCDEFGH");
    check_string_case("lower(a)", 1, AU, "foo", "hello", " pad  ", "abcdefgh");
    check_string_case("strip(a)", 1, AU, "foo", "Hello", "pad", "abcdefgh");
    check_string_case("removesuffix(a, 'gh')", 1, AU, "foo", "Hello", " pad  ", "abcdef");
    check_string_case("replace(a, 'o', '0')", 1, AU, "f00", "Hell0", " pad  ", "abcdefgh");
    check_string_case("substr(a, 1, 3)", 1, 3, "oo", "ell", "pad", "bcd");
    check_string_case("split_part(a, 'l', 1)", 1, AU, "", "lo", "", "");
    printf("  PASS string-valued ops\n");
}

static void test_predicates(void) {
    TEST("predicates over ME_BYTES");
    check_predicate("a == 'foo'", true, false, false, false);
    check_predicate("a != 'foo'", false, true, true, true);
    check_predicate("contains(a, 'ell')", false, true, false, false);
    check_predicate("startswith(a, 'H')", false, true, false, false);
    check_predicate("endswith(a, 'd  ')", false, false, true, false);
    printf("  PASS predicates\n");
}

static void test_families_do_not_mix(void) {
    TEST("`S` and `U` operands are rejected together");
    static uint32_t u_data[ITEMS * 4];
    me_variable vars[2] = {{0}, {0}};
    vars[0].name = "a"; vars[0].dtype = ME_BYTES; vars[0].address = a_data; vars[0].itemsize = AU;
    vars[1].name = "u"; vars[1].dtype = ME_STRING; vars[1].address = u_data;
    vars[1].itemsize = 4 * sizeof(uint32_t);

    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile("a + u", vars, 2, ME_AUTO, &err, &expr);
    if (rc == ME_COMPILE_SUCCESS) {
        printf("  FAIL: 'a + u' compiled; NumPy rejects mixing 'S' and 'U' too\n");
        tests_failed++;
        me_free(expr);
        return;
    }
    printf("  PASS mixed families rejected\n");
}

static void test_non_ascii_literal_rejected(void) {
    /* Literals are stored as UCS4 and read codepoint-by-codepoint against byte
     * operands, which is only exact below 0x80. */
    TEST("non-ASCII literal against a bytes operand is rejected");
    me_variable vars[1] = {{0}};
    vars[0].name = "a"; vars[0].dtype = ME_BYTES; vars[0].address = a_data; vars[0].itemsize = AU;

    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile("a + '\\u00e9'", vars, 1, ME_AUTO, &err, &expr);
    if (rc == ME_COMPILE_SUCCESS) {
        printf("  FAIL: non-ASCII literal compiled against a bytes operand\n");
        tests_failed++;
        me_free(expr);
        return;
    }
    printf("  PASS non-ASCII literal rejected\n");
}

static void test_nd_and_dsl(void) {
    TEST("me_eval_nd and DSL kernels over ME_BYTES");
    me_variable vars[1] = {{0}};
    vars[0].name = "a"; vars[0].dtype = ME_BYTES; vars[0].itemsize = AU;

    const int64_t shape[1] = {ITEMS};
    const int32_t chunkshape[1] = {ITEMS};
    const int32_t blockshape[1] = {ITEMS};

    struct { const char *src; size_t itemsize; const char *want[ITEMS]; } cases[] = {
        {"'p=' + a", 2 + AU, {"p=foo", "p=Hello", "p= pad  ", "p=abcdefgh"}},
        {"def k(a):\n    r = 'p=' + upper(a)\n    return r",
         2 + AU, {"p=FOO", "p=HELLO", "p= PAD  ", "p=ABCDEFGH"}},
        /* Branches of different widths: the narrow one is padded into the slot. */
        {"def k(a):\n    if contains(a, 'o'):\n        return a\n    return 'long-' + a\n",
         5 + AU, {"foo", "Hello", "long- pad  ", "long-abcdefgh"}},
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int err = 0;
        me_expr *expr = NULL;
        int rc = me_compile_nd(cases[c].src, vars, 1, ME_AUTO, 1, shape, chunkshape,
                               blockshape, &err, &expr);
        if (rc != ME_COMPILE_SUCCESS) {
            printf("  FAIL case %zu: compile rc=%d (%s)\n", c, rc, me_get_last_error_message());
            tests_failed++;
            continue;
        }
        const size_t itemsize = me_get_itemsize(expr);
        if (itemsize != cases[c].itemsize) {
            printf("  FAIL case %zu: itemsize %zu, expected %zu\n", c, itemsize,
                   cases[c].itemsize);
            tests_failed++;
        }
        char *out = calloc(ITEMS, itemsize);
        const void *ptrs[] = {a_data};
        rc = me_eval_nd(expr, ptrs, 1, out, ITEMS, 0, 0, NULL);
        if (rc != ME_EVAL_SUCCESS) {
            printf("  FAIL case %zu: eval rc=%d\n", c, rc);
            tests_failed++;
        }
        else {
            for (int i = 0; i < ITEMS; i++) {
                expect_slot(out, itemsize, i, cases[c].want[i], cases[c].src);
            }
        }
        free(out);
        me_free(expr);
    }
    printf("  PASS nd and DSL over bytes\n");
}

int main(void) {
    printf("=== ME_BYTES tests ===\n\n");
    fill_inputs();

    test_string_ops();
    test_predicates();
    test_families_do_not_mix();
    test_non_ascii_literal_rejected();
    test_nd_and_dsl();

    printf("\n=== %d tests run, %d failures ===\n", tests_run, tests_failed);
    return tests_failed != 0;
}
