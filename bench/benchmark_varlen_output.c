/* me_eval_varlen() vs fixed-width string output.
 *
 * The conservative width bound is what makes fixed-width string results
 * expensive: UCS4 spends 4 bytes per codepoint, lower()/upper() reserve a 2x
 * case-expansion bound on top of that, and every row is padded out to the
 * worst case.  This measures what packing to the Arrow varlen layout on the
 * way out costs, and what it saves.
 *
 * Shape is the pandas-3 blog kernel's: row-wise control flow, string locals,
 * branches of differing widths.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "miniexpr.h"

#define NROWS 1000000
#define AU 36 /* codepoints per operand slot, i.e. numpy <U36 */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Real Chicago Taxi company names, which is where the width spread comes from. */
static const char *COMPANIES[] = {
    "Flash Cab",
    "Taxi Affiliation Services",
    "Medallion Leasin",
    "Yellow Cab",
    "Chicago Carriage Cab Corp",
    "Sun Taxi",
    "Blue Ribbon Taxi Association Inc.",
    "Choice Taxi Association",
    "Globe Taxi",
    "Dispatch Taxi Affiliation",
};
#define NCOMPANIES ((int)(sizeof(COMPANIES) / sizeof(COMPANIES[0])))

int main(void) {
    printf("=== Varlen vs fixed-width string output (%d rows) ===\n\n", NROWS);

    uint32_t *a = calloc((size_t)NROWS * AU, sizeof(uint32_t));
    if (!a) return 1;
    for (int i = 0; i < NROWS; i++) {
        const char *s = COMPANIES[i % NCOMPANIES];
        uint32_t *slot = a + (size_t)i * AU;
        for (size_t k = 0; s[k] && k < AU; k++) {
            slot[k] = (uint32_t)(unsigned char)s[k];
        }
    }

    me_variable vars[] = {
        {"a", ME_STRING, a, ME_VARIABLE, NULL, AU * sizeof(uint32_t)},
    };
    const char *src = "def k(a):\n"
                      "    if contains(a, 'Taxi'):\n"
                      "        return 'co=' + lower(a) + '|kind=cab'\n"
                      "    return 'co=' + lower(a)\n";
    int err;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 1, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("compile failed at position %d\n", err);
        free(a);
        return 1;
    }

    const size_t itemsize = me_get_itemsize(expr);
    const void *ptrs[] = {a};

    unsigned char *fixed = malloc((size_t)NROWS * itemsize);
    if (!fixed) return 1;
    double t0 = now_sec();
    int rc = me_eval(expr, ptrs, 1, fixed, NROWS, NULL);
    const double t_fixed = now_sec() - t0;
    if (rc != ME_EVAL_SUCCESS) {
        printf("me_eval failed: %d\n", rc);
        return 1;
    }

    const size_t bound = me_varlen_data_bound(expr, NROWS);
    int64_t *offsets = malloc(((size_t)NROWS + 1) * sizeof(int64_t));
    uint8_t *data = malloc(bound);
    if (!offsets || !data) return 1;
    size_t used = 0;
    t0 = now_sec();
    rc = me_eval_varlen(expr, ptrs, 1, NROWS, offsets, data, bound, &used, NULL);
    const double t_varlen = now_sec() - t0;
    if (rc != ME_EVAL_SUCCESS) {
        printf("me_eval_varlen failed: %d\n", rc);
        return 1;
    }

    const double fixed_bpr = (double)itemsize;
    const double varlen_bpr = (double)used / NROWS + (double)sizeof(int64_t);
    printf("fixed  me_eval          %8.1f ms   %6.1f B/row  (itemsize %zu)\n",
           t_fixed * 1e3, fixed_bpr, itemsize);
    printf("varlen me_eval_varlen   %8.1f ms   %6.1f B/row  (%zu data bytes + offsets)\n",
           t_varlen * 1e3, varlen_bpr, used);
    printf("\npack overhead           %8.1f ms   %5.0f%% of eval\n",
           (t_varlen - t_fixed) * 1e3, 100.0 * (t_varlen - t_fixed) / t_fixed);
    printf("result footprint        %8.1fx smaller\n", fixed_bpr / varlen_bpr);

    free(data);
    free(offsets);
    free(fixed);
    me_free(expr);
    free(a);
    return 0;
}
