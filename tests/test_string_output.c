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
    if (i >= units || slot[i] != 0) {
        printf("  FAIL %s at [%d]: missing NUL terminator\n", label, idx);
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
    /* 32 + 24 - 4 = 52 bytes = 13 units */
    const size_t ou = 13;
    expect_itemsize(expr, ou * sizeof(uint32_t), "a + b");

    uint32_t out[ITEMS * 13];
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

    /* literal is 5 chars -> 24 bytes; 24 + 32 - 4 = 52 = 13 units */
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

    /* (24 + 8 - 4) = 28, then 28 + 24 - 4 = 48 bytes = 12 units */
    const size_t ou = 12;
    expect_itemsize(expr, ou * sizeof(uint32_t), "nested concat");

    uint32_t out[ITEMS * 12];
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

    /* 16 + 16 - 4 = 28 bytes = 7 units: exactly 6 chars plus NUL */
    const size_t ou = 7;
    expect_itemsize(expr, ou * sizeof(uint32_t), "bound-filling concat");

    uint32_t out[ITEMS * 7];
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

    const size_t ou = 7; /* 24 + 8 - 4 = 28 */
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

int main(void) {
    printf("=== String output tests ===\n\n");

    test_concat_var_var();
    test_concat_literal();
    test_concat_nested();
    test_concat_fills_bound();
    test_non_ascii();
    test_numeric_add_unaffected();
    test_literal_with_dsl_chars();

    printf("\n=== %d tests run, %d failures ===\n", tests_run, tests_failed);
    return tests_failed != 0;
}
