/* Guards against two DSL constructs that used to compile but misbehave:
 *   - ';'-joined statements were silently discarded (worst inside an `if`)
 *   - reductions inside per-element control flow give wrong results past
 *     element 0, because the reduction collapses the whole block while the
 *     surrounding mask is per-element
 * Both must now be rejected at compile time rather than mislead. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "../src/miniexpr.h"
#include "minctest.h"

int tests_run = 0;
int tests_failed = 0;

#define TEST(name) \
    printf("Testing: %s\n", name); \
    tests_run++;

static double kx[4] = {1.0, 2.0, 3.0, 4.0};

static void expect(const char *label, const char *src, bool want_ok) {
    TEST(label);
    me_variable vars[] = {{"x", ME_FLOAT64, kx, ME_VARIABLE, NULL, 0}};
    int err = 0;
    me_expr *expr = NULL;
    int rc = me_compile(src, vars, 1, ME_FLOAT64, &err, &expr);
    const bool ok = (rc == ME_COMPILE_SUCCESS);
    if (ok != want_ok) {
        printf("  FAIL: expected %s, got rc=%d\n", want_ok ? "success" : "rejection", rc);
        tests_failed++;
    }
    else {
        printf("  PASS\n");
    }
    if (expr) me_free(expr);
}

int main(void) {
    printf("=== DSL guard tests ===\n\n");

    /* Semicolons: a bare trailing one is harmless, a joined statement is not. */
    expect("plain newline-separated body",
           "def k(x):\n    a = x * 2\n    b = a + 1\n    return b\n", true);
    expect("trailing semicolon accepted",
           "def k(x):\n    a = x * 2;\n    return a\n", true);
    expect("';'-joined statements rejected",
           "def k(x):\n    a = x * 2; b = a + 1\n    return b\n", false);
    expect("';'-joined inside if rejected",
           "def k(x):\n    if x > 2:\n        a = 100; return a\n    return 0\n", false);

    /* Reductions: fine at top level and as a condition, not inside a body. */
    expect("reduction at top level accepted",
           "def k(x):\n    y = sum(x)\n    return y\n", true);
    expect("any() as condition accepted",
           "def k(x):\n    if any(x > 2):\n        return 1\n    return 0\n", true);
    expect("reduction in if body rejected",
           "def k(x):\n    if x > 2:\n        y = sum(x)\n        return y\n    return 0\n", false);
    expect("reduction in for body rejected",
           "def k(x):\n    y = 0\n    for i in range(3):\n        y = max(x)\n    return y\n", false);

    printf("\n=== %d tests run, %d failures ===\n", tests_run, tests_failed);
    return tests_failed != 0;
}
