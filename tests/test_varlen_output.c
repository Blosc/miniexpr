/* Tests for me_eval_varlen(): Arrow offsets + tight byte blob out of a string
 * expression, i.e. that the compile-time width bound is spent on scratch only
 * and never on the result. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"

#define ITEMS 4

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

static void fail(const char *fmt, const char *a) {
    printf("  FAIL ");
    printf(fmt, a);
    printf("\n");
    tests_failed++;
}

/* Fill a fixed-width UCS4 slot from an ASCII literal. */
static void put(uint32_t *base, size_t units, int idx, const char *ascii) {
    uint32_t *slot = base + (size_t)idx * units;
    memset(slot, 0, units * sizeof(uint32_t));
    for (size_t i = 0; ascii[i] && i < units; i++) {
        slot[i] = (uint32_t)(unsigned char)ascii[i];
    }
}

/* Check that value `idx` of the packed blob is exactly `want`. */
static void expect_value(const int64_t *offsets, const uint8_t *data, int idx,
                         const char *want, const char *label) {
    const int64_t from = offsets[idx], to = offsets[idx + 1];
    const size_t len = strlen(want);
    if (to < from) {
        fail("%s: offsets go backwards", label);
        return;
    }
    if ((size_t)(to - from) != len) {
        printf("  FAIL %s at [%d]: length %d, expected %zu\n", label, idx,
               (int)(to - from), len);
        tests_failed++;
        return;
    }
    if (len && memcmp(data + from, want, len) != 0) {
        printf("  FAIL %s at [%d]: bytes differ\n", label, idx);
        tests_failed++;
    }
}

static me_expr *compile_or_fail(const char *src, const me_variable *vars, int nvars) {
    int err;
    me_expr *expr = NULL;
    const int rc = me_compile(src, vars, nvars, ME_AUTO, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compilation error %d at %d\n", rc, err);
        tests_failed++;
        return NULL;
    }
    return expr;
}

/* The headline case: a concat whose result is far shorter than its bound. */
static void test_concat_packs_tight(void) {
    TEST("'kind=' + a packs to Arrow offsets + UTF-8");
    const size_t au = 8;
    uint32_t a[ITEMS * 8];
    put(a, au, 0, "home");
    put(a, au, 1, "room");
    put(a, au, 2, "");
    put(a, au, 3, "loft");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };
    me_expr *expr = compile_or_fail("'kind=' + a", vars, 1);
    if (!expr) return;

    /* 5 + 8 = 13 codepoints, i.e. 52 bytes per row of fixed-width slot. */
    const size_t bound = me_varlen_data_bound(expr, ITEMS);
    if (bound != ITEMS * 13 * sizeof(uint32_t)) {
        fail("%s: unexpected data bound", "concat");
        me_free(expr);
        return;
    }

    int64_t offsets[ITEMS + 1];
    uint8_t *data = malloc(bound);
    size_t used = 0;
    const void *ptrs[] = {a};
    const int rc = me_eval_varlen(expr, ptrs, 1, ITEMS, offsets, data, bound, &used, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL: me_eval_varlen returned %d\n", rc);
        tests_failed++;
        free(data);
        me_free(expr);
        return;
    }

    if (offsets[0] != 0) fail("%s: offsets[0] must be 0", "concat");
    expect_value(offsets, data, 0, "kind=home", "concat");
    expect_value(offsets, data, 1, "kind=room", "concat");
    expect_value(offsets, data, 2, "kind=", "concat");
    expect_value(offsets, data, 3, "kind=loft", "concat");

    /* 9 + 9 + 5 + 9 = 32 bytes of payload against a 208-byte fixed block. */
    if (used != 32 || (size_t)offsets[ITEMS] != used) {
        printf("  FAIL concat: used %zu bytes, expected 32\n", used);
        tests_failed++;
    }
    else {
        printf("  PASS concat packs %zu bytes where fixed-width needs %zu\n", used, bound);
    }

    free(data);
    me_free(expr);
}

/* The reason this is worth having: lower() reserves a 2x expansion bound on
 * 'U', so the fixed-width result is 8x the bytes the values actually use. */
static void test_case_bound_is_scratch_only(void) {
    TEST("lower(a) pays its 2x bound on scratch, not on the result");
    const size_t au = 8;
    uint32_t a[ITEMS * 8];
    put(a, au, 0, "HOME");
    put(a, au, 1, "ROOM");
    put(a, au, 2, "LOFT");
    put(a, au, 3, "SUITE");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };
    me_expr *expr = compile_or_fail("lower(a)", vars, 1);
    if (!expr) return;

    const size_t bound = me_varlen_data_bound(expr, ITEMS);
    int64_t offsets[ITEMS + 1];
    uint8_t *data = malloc(bound);
    size_t used = 0;
    const void *ptrs[] = {a};
    const int rc = me_eval_varlen(expr, ptrs, 1, ITEMS, offsets, data, bound, &used, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL: me_eval_varlen returned %d\n", rc);
        tests_failed++;
        free(data);
        me_free(expr);
        return;
    }

    expect_value(offsets, data, 0, "home", "lower");
    expect_value(offsets, data, 1, "room", "lower");
    expect_value(offsets, data, 2, "loft", "lower");
    expect_value(offsets, data, 3, "suite", "lower");

    /* itemsize is 8 units * 2 (case bound) * 4 (UCS4) = 64 bytes per row. */
    if (me_get_itemsize(expr) != 64) {
        fail("%s: expected a 64-byte fixed slot", "lower");
    }
    if (used != 17) {
        printf("  FAIL lower: used %zu bytes, expected 17\n", used);
        tests_failed++;
    }
    else {
        printf("  PASS lower packs %zu bytes where fixed-width needs %zu\n", used, bound);
    }

    free(data);
    me_free(expr);
}

/* UCS4 -> UTF-8, so a codepoint outside ASCII must widen on the way out. */
static void test_non_ascii_becomes_utf8(void) {
    TEST("non-ASCII codepoints encode as UTF-8");
    const size_t au = 4;
    uint32_t a[ITEMS * 4];
    memset(a, 0, sizeof(a));
    a[0 * au + 0] = 0x00E9;  /* e-acute, 2 bytes  */
    a[1 * au + 0] = 0x20AC;  /* euro sign, 3 bytes */
    a[2 * au + 0] = 0x1F600; /* grinning face, 4 bytes */
    a[3 * au + 0] = 'z';     /* 1 byte */

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };
    me_expr *expr = compile_or_fail("a + '!'", vars, 1);
    if (!expr) return;

    const size_t bound = me_varlen_data_bound(expr, ITEMS);
    int64_t offsets[ITEMS + 1];
    uint8_t *data = malloc(bound);
    size_t used = 0;
    const void *ptrs[] = {a};
    const int rc = me_eval_varlen(expr, ptrs, 1, ITEMS, offsets, data, bound, &used, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL: me_eval_varlen returned %d\n", rc);
        tests_failed++;
        free(data);
        me_free(expr);
        return;
    }

    expect_value(offsets, data, 0, "\xC3\xA9!", "utf8");
    expect_value(offsets, data, 1, "\xE2\x82\xAC!", "utf8");
    expect_value(offsets, data, 2, "\xF0\x9F\x98\x80!", "utf8");
    expect_value(offsets, data, 3, "z!", "utf8");
    if (used != 3 + 4 + 5 + 2) {
        printf("  FAIL utf8: used %zu bytes, expected 14\n", used);
        tests_failed++;
    }
    else {
        printf("  PASS UTF-8 encoding across all four widths\n");
    }

    free(data);
    me_free(expr);
}

/* 'S' is large_binary: bytes go out verbatim, including ones that are not
 * valid UTF-8 on their own.  This matches what numpy 'S' stores. */
static void test_bytes_verbatim(void) {
    TEST("ME_BYTES copies its bytes verbatim");
    const size_t au = 6;
    uint8_t a[ITEMS * 6];
    memset(a, 0, sizeof(a));
    memcpy(a + 0 * au, "caf\xE9", 4); /* latin-1 e-acute: a lone 0xE9 */
    memcpy(a + 1 * au, "bar", 3);
    memcpy(a + 2 * au, "abcdef", 6); /* fills the slot, so no terminator */
    /* row 3 stays empty */

    me_variable vars[] = {
        {"a", ME_BYTES, a, ME_VARIABLE, NULL, au},
    };
    me_expr *expr = compile_or_fail("a + b'-x'", vars, 1);
    if (!expr) return;

    const size_t bound = me_varlen_data_bound(expr, ITEMS);
    if (bound != ITEMS * (au + 2)) {
        fail("%s: unexpected data bound", "bytes");
        me_free(expr);
        return;
    }

    int64_t offsets[ITEMS + 1];
    uint8_t *data = malloc(bound);
    size_t used = 0;
    const void *ptrs[] = {a};
    const int rc = me_eval_varlen(expr, ptrs, 1, ITEMS, offsets, data, bound, &used, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL: me_eval_varlen returned %d\n", rc);
        tests_failed++;
        free(data);
        me_free(expr);
        return;
    }

    expect_value(offsets, data, 0, "caf\xE9-x", "bytes");
    expect_value(offsets, data, 1, "bar-x", "bytes");
    expect_value(offsets, data, 2, "abcdef-x", "bytes");
    expect_value(offsets, data, 3, "-x", "bytes");
    if (used != 6 + 5 + 8 + 2) {
        printf("  FAIL bytes: used %zu bytes, expected 21\n", used);
        tests_failed++;
    }
    else {
        printf("  PASS bytes pass through unconverted\n");
    }

    free(data);
    me_free(expr);
}

/* A DSL kernel with branches of differing widths goes through the same path,
 * since me_eval() dispatches the DSL program itself. */
static void test_dsl_kernel(void) {
    TEST("DSL kernel with mixed branch widths");
    const size_t au = 8;
    uint32_t a[ITEMS * 8];
    put(a, au, 0, "home");
    put(a, au, 1, "room");
    put(a, au, 2, "home");
    put(a, au, 3, "loft");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };
    me_expr *expr = compile_or_fail(
        "def k(a):\n"
        "    if a == 'home':\n"
        "        return 'whole-unit'\n"
        "    return 'shared:' + a\n",
        vars, 1);
    if (!expr) return;

    const size_t bound = me_varlen_data_bound(expr, ITEMS);
    int64_t offsets[ITEMS + 1];
    uint8_t *data = malloc(bound);
    size_t used = 0;
    const void *ptrs[] = {a};
    const int rc = me_eval_varlen(expr, ptrs, 1, ITEMS, offsets, data, bound, &used, NULL);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL: me_eval_varlen returned %d\n", rc);
        tests_failed++;
        free(data);
        me_free(expr);
        return;
    }

    expect_value(offsets, data, 0, "whole-unit", "dsl");
    expect_value(offsets, data, 1, "shared:room", "dsl");
    expect_value(offsets, data, 2, "whole-unit", "dsl");
    expect_value(offsets, data, 3, "shared:loft", "dsl");
    printf("  PASS DSL kernel packs %zu bytes where fixed-width needs %zu\n", used, bound);

    free(data);
    me_free(expr);
}

static void test_rejects_bad_args(void) {
    TEST("argument and capacity guards");
    const size_t au = 8;
    uint32_t a[ITEMS * 8];
    put(a, au, 0, "home");
    put(a, au, 1, "room");
    put(a, au, 2, "loft");
    put(a, au, 3, "flat");

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, au * sizeof(uint32_t)},
    };
    me_expr *expr = compile_or_fail("'kind=' + a", vars, 1);
    if (!expr) return;

    int64_t offsets[ITEMS + 1];
    uint8_t data[16];
    size_t used = 0;
    const void *ptrs[] = {a};

    /* 4 rows of "kind=xxxx" need 36 bytes; 16 must be refused, not overrun. */
    if (me_eval_varlen(expr, ptrs, 1, ITEMS, offsets, data, sizeof(data), &used, NULL)
        != ME_EVAL_ERR_INVALID_ARG) {
        fail("%s: a too-small data buffer must be rejected", "guards");
    }
    if (me_eval_varlen(NULL, ptrs, 1, ITEMS, offsets, data, sizeof(data), &used, NULL)
        != ME_EVAL_ERR_NULL_EXPR) {
        fail("%s: a NULL expression must be rejected", "guards");
    }
    if (me_eval_varlen(expr, ptrs, 1, ITEMS, NULL, data, sizeof(data), &used, NULL)
        != ME_EVAL_ERR_INVALID_ARG) {
        fail("%s: NULL offsets must be rejected", "guards");
    }
    if (me_varlen_data_bound(expr, 0) != 0) {
        fail("%s: a zero item count has no bound", "guards");
    }
    me_free(expr);

    /* A numeric expression has no varlen form. */
    double n[ITEMS] = {1, 2, 3, 4};
    me_variable nvars[] = {{"n", ME_FLOAT64, n, ME_VARIABLE, NULL, 0}};
    me_expr *nexpr = compile_or_fail("n * 2", nvars, 1);
    if (nexpr) {
        const void *nptrs[] = {n};
        if (me_varlen_data_bound(nexpr, ITEMS) != 0) {
            fail("%s: numeric output has no varlen bound", "guards");
        }
        if (me_eval_varlen(nexpr, nptrs, 1, ITEMS, offsets, data, sizeof(data), &used, NULL)
            != ME_EVAL_ERR_INVALID_ARG) {
            fail("%s: numeric output must be rejected", "guards");
        }
        me_free(nexpr);
    }
    printf("  PASS guards\n");
}

int main(void) {
    printf("=== Varlen (Arrow) string output tests ===\n\n");

    test_concat_packs_tight();
    test_case_bound_is_scratch_only();
    test_non_ascii_becomes_utf8();
    test_bytes_verbatim();
    test_dsl_kernel();
    test_rejects_bad_args();

    printf("\n=== %d tests run, %d failures ===\n", tests_run, tests_failed);
    return tests_failed != 0;
}
