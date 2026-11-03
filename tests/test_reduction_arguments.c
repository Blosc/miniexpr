/*
 * Reductions (sum, prod, mean, min, max, any, all) over every numeric dtype and
 * several kinds of argument: a plain variable, an argument with a constant
 * subexpression (sum(x * -2), max(x + sin(1.0)), sum(where(x > 0, -1, 1))),
 * comparisons including those of two bools (sum(p == q)), and a constant
 * argument (sum(-2)).
 *
 * Every case is evaluated three ways: me_eval over the whole array, and
 * me_eval_nd block by block (as python-blosc2 drives reductions) for a shape
 * with padded edge blocks and for one without padding. Each block result and
 * the combined result are compared with a plain C loop: exactly for integer
 * and bool results, to a tight tolerance for floating-point ones.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"

typedef enum { R_SUM, R_PROD, R_MEAN, R_MIN, R_MAX, R_ANY, R_ALL, R_COUNT } red_kind;
static const char *red_names[R_COUNT] = {"sum", "prod", "mean", "min", "max", "any", "all"};

static const me_dtype all_dtypes[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
                                      ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64,
                                      ME_FLOAT32, ME_FLOAT64};
#define N_DTYPES ((int)(sizeof(all_dtypes) / sizeof(all_dtypes[0])))

static int failures = 0;
static int checks = 0;

static const char *dtype_name(me_dtype dt) {
    switch (dt) {
    case ME_BOOL: return "bool";
    case ME_INT8: return "int8";
    case ME_INT16: return "int16";
    case ME_INT32: return "int32";
    case ME_INT64: return "int64";
    case ME_UINT8: return "uint8";
    case ME_UINT16: return "uint16";
    case ME_UINT32: return "uint32";
    case ME_UINT64: return "uint64";
    case ME_FLOAT32: return "float32";
    case ME_FLOAT64: return "float64";
    default: return "?";
    }
}

static bool is_signed_int(me_dtype dt) { return dt >= ME_INT8 && dt <= ME_INT64; }
static bool is_unsigned_int(me_dtype dt) { return dt >= ME_UINT8 && dt <= ME_UINT64; }
static bool is_float(me_dtype dt) { return dt == ME_FLOAT32 || dt == ME_FLOAT64; }

static void store(void *buf, me_dtype dt, int64_t i, double v) {
    switch (dt) {
    case ME_BOOL: ((bool *)buf)[i] = v != 0.0; break;
    case ME_INT8: ((int8_t *)buf)[i] = (int8_t)v; break;
    case ME_INT16: ((int16_t *)buf)[i] = (int16_t)v; break;
    case ME_INT32: ((int32_t *)buf)[i] = (int32_t)v; break;
    case ME_INT64: ((int64_t *)buf)[i] = (int64_t)v; break;
    case ME_UINT8: ((uint8_t *)buf)[i] = (uint8_t)v; break;
    case ME_UINT16: ((uint16_t *)buf)[i] = (uint16_t)v; break;
    case ME_UINT32: ((uint32_t *)buf)[i] = (uint32_t)v; break;
    case ME_UINT64: ((uint64_t *)buf)[i] = (uint64_t)v; break;
    case ME_FLOAT32: ((float *)buf)[i] = (float)v; break;
    case ME_FLOAT64: ((double *)buf)[i] = v; break;
    default: break;
    }
}

/* Every value used here (inputs and integer results) is exact in double. */
static double load(const void *buf, me_dtype dt, int64_t i) {
    switch (dt) {
    case ME_BOOL: return ((const bool *)buf)[i] ? 1.0 : 0.0;
    case ME_INT8: return ((const int8_t *)buf)[i];
    case ME_INT16: return ((const int16_t *)buf)[i];
    case ME_INT32: return ((const int32_t *)buf)[i];
    case ME_INT64: return (double)((const int64_t *)buf)[i];
    case ME_UINT8: return ((const uint8_t *)buf)[i];
    case ME_UINT16: return ((const uint16_t *)buf)[i];
    case ME_UINT32: return ((const uint32_t *)buf)[i];
    case ME_UINT64: return (double)((const uint64_t *)buf)[i];
    case ME_FLOAT32: return ((const float *)buf)[i];
    case ME_FLOAT64: return ((const double *)buf)[i];
    default: return NAN;
    }
}

/* Class of the elementwise result, which fixes the reduction output dtype the
 * way NumPy (and python-blosc2) would ask for it. */
typedef enum {
    RES_SAME,  /* dtype of the input */
    RES_FLOAT, /* float32 for float32 input, else float64 */
    RES_INT,   /* int64 (python int literals only) */
    RES_BOOL   /* comparison */
} res_class;

static me_dtype elem_dtype(res_class rc, me_dtype in) {
    switch (rc) {
    case RES_SAME: return in;
    case RES_FLOAT: return in == ME_FLOAT32 ? ME_FLOAT32 : ME_FLOAT64;
    case RES_INT: return ME_INT64;
    default: return ME_BOOL;
    }
}

static me_dtype reduction_dtype(red_kind r, me_dtype elem) {
    switch (r) {
    case R_SUM:
    case R_PROD:
        if (elem == ME_BOOL || is_signed_int(elem)) return ME_INT64;
        if (is_unsigned_int(elem)) return ME_UINT64;
        return elem;
    case R_MEAN: return ME_FLOAT64;
    case R_MIN:
    case R_MAX: return elem;
    default: return ME_BOOL;
    }
}

typedef double (*ref_fn)(double x, double y);

static double f_x(double x, double y) { (void)y; return x; }
static double f_mul_m2(double x, double y) { (void)y; return x * -2.0; }
static double f_sub_m05(double x, double y) { (void)y; return x - -0.5; }
static double f_sub_m1(double x, double y) { (void)y; return x - -1.0; }
static double f_neg(double x, double y) { (void)y; return -x; }
static double f_div_m2(double x, double y) { (void)y; return x / -2.0; }
static double f_pow_m1(double x, double y) { (void)y; return pow(x, -1.0); }
static double f_mul_sqrt4(double x, double y) { (void)y; return x * sqrt(4.0); }
static double f_add_sin1(double x, double y) { (void)y; return x + sin(1.0); }
static double f_mul_abs3(double x, double y) { (void)y; return x * 3.0; }
static double f_gt_m2(double x, double y) { (void)y; return x > -2.0; }
static double f_gt_1(double x, double y) { (void)y; return x > 1.0; }
static double f_where_m1(double x, double y) { (void)y; return x > 0.0 ? -1.0 : 1.0; }
static double f_where_xy(double x, double y) { return x > 0.0 ? x : y; }
static double f_eq(double x, double y) { return x == y; }
static double f_ne(double x, double y) { return x != y; }
static double f_lt(double x, double y) { return x < y; }
static double f_le(double x, double y) { return x <= y; }
static double f_gt(double x, double y) { return x > y; }
static double f_ge(double x, double y) { return x >= y; }

#define ANY_DTYPE 0
#define SIGNED_OR_FLOAT 1 /* negative literals: keep unsigned/bool typing out of it */
#define FLOAT_ONLY 2

typedef struct {
    const char *expr;
    ref_fn ref;
    res_class res;
    int dtypes;
    int nvars;
} test_case;

static const test_case cases[] = {
    /* A plain variable: strided (padded-block) and contiguous paths. */
    {"x", f_x, RES_SAME, ANY_DTYPE, 1},
    /* Constant subexpressions: negate(c), sin(c), sqrt(c), abs(negate(c)),
     * where() with constant branches. */
    {"x * -2", f_mul_m2, RES_SAME, SIGNED_OR_FLOAT, 1},
    {"-2 * x", f_mul_m2, RES_SAME, SIGNED_OR_FLOAT, 1},
    {"(-2) * x", f_mul_m2, RES_SAME, SIGNED_OR_FLOAT, 1},
    {"x * (0 - 2)", f_mul_m2, RES_SAME, SIGNED_OR_FLOAT, 1},
    {"x - -1", f_sub_m1, RES_SAME, SIGNED_OR_FLOAT, 1},
    {"x - -0.5", f_sub_m05, RES_FLOAT, FLOAT_ONLY, 1},
    {"-x", f_neg, RES_SAME, SIGNED_OR_FLOAT, 1},
    {"x / -2", f_div_m2, RES_FLOAT, FLOAT_ONLY, 1},
    {"x ** -1", f_pow_m1, RES_FLOAT, FLOAT_ONLY, 1},
    {"x * sqrt(4.0)", f_mul_sqrt4, RES_FLOAT, ANY_DTYPE, 1},
    {"x + sin(1.0)", f_add_sin1, RES_FLOAT, ANY_DTYPE, 1},
    {"x * abs(-3)", f_mul_abs3, RES_SAME, SIGNED_OR_FLOAT, 1},
    {"x > -2", f_gt_m2, RES_BOOL, SIGNED_OR_FLOAT, 1},
    {"where(x > 0, -1, 1)", f_where_m1, RES_INT, SIGNED_OR_FLOAT, 1},
    {"where(x > 0, x, y)", f_where_xy, RES_SAME, ANY_DTYPE, 2},
    /* Comparisons, including of two bool operands (and of a bool with a
     * literal, which a bool-typed reduction types as bool). */
    {"x > 1", f_gt_1, RES_BOOL, ANY_DTYPE, 1},
    {"x == y", f_eq, RES_BOOL, ANY_DTYPE, 2},
    {"x != y", f_ne, RES_BOOL, ANY_DTYPE, 2},
    {"x < y", f_lt, RES_BOOL, ANY_DTYPE, 2},
    {"x <= y", f_le, RES_BOOL, ANY_DTYPE, 2},
    {"x > y", f_gt, RES_BOOL, ANY_DTYPE, 2},
    {"x >= y", f_ge, RES_BOOL, ANY_DTYPE, 2},
};
#define N_CASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* Small values, no zeros for floats (x ** -1); prod gets values whose product
 * stays exact in int64 and double. */
static double input_value(me_dtype dt, int64_t i, int which, red_kind r) {
    const int64_t k = which ? (i * 5 + 1) : (i * 7 + 3);
    if (dt == ME_BOOL) return (k % 3) == 0 ? 1.0 : 0.0;
    if (r == R_PROD) {
        if (k % 5 == 0) return 2.0;
        if (!is_unsigned_int(dt) && k % 3 == 0) return -1.0;
        return 1.0;
    }
    if (is_unsigned_int(dt)) return (double)(k % 5);
    if (is_signed_int(dt)) return (double)(k % 5 - 2);
    return (double)(k % 10 - 5) * 0.25 + 0.125;
}

typedef struct {
    double d;   /* float results, mean */
    int64_t i;  /* integer results (sum/prod/min/max) */
    bool b;     /* any/all */
    bool init;
} acc_t;

static void acc_init(acc_t *a, red_kind r) {
    memset(a, 0, sizeof(*a));
    if (r == R_PROD) {
        a->d = 1.0;
        a->i = 1;
    }
    a->b = (r == R_ALL);
}

static void acc_push(acc_t *a, red_kind r, double v, bool float_result) {
    switch (r) {
    case R_SUM:
        a->d += v;
        a->i += (int64_t)v;
        break;
    case R_PROD:
        a->d *= v;
        a->i *= (int64_t)v;
        break;
    case R_MEAN: a->d += v; break;
    case R_MIN:
        if (!a->init || v < a->d) a->d = v;
        break;
    case R_MAX:
        if (!a->init || v > a->d) a->d = v;
        break;
    case R_ANY: a->b = a->b || v != 0.0; break;
    case R_ALL: a->b = a->b && v != 0.0; break;
    default: break;
    }
    if ((r == R_MIN || r == R_MAX) && !float_result) a->i = (int64_t)a->d;
    a->init = true;
}

static bool close_to(double got, double expected, me_dtype dt) {
    const double tol = (dt == ME_FLOAT32) ? 1e-5 : 1e-12;
    if (isinf(expected)) return got == expected;
    return fabs(got - expected) <= tol * (1.0 + fabs(expected));
}

/* Compare a reduction output of dtype `out_dt` with the reference. */
static bool expect(const char *label, const void *out, me_dtype out_dt, red_kind r,
                   const acc_t *ref, int64_t n, me_dtype elem_dt) {
    bool ok;
    char got_s[64], exp_s[64];
    checks++;
    if (r == R_ANY || r == R_ALL) {
        /* Read the byte: a bool result must also be canonical (0 or 1). */
        const uint8_t got = ((const uint8_t *)out)[0];
        ok = got == (ref->b ? 1 : 0);
        snprintf(got_s, sizeof(got_s), "%u", (unsigned)got);
        snprintf(exp_s, sizeof(exp_s), "%d", (int)ref->b);
    }
    else if (r == R_MEAN) {
        const double got = ((const double *)out)[0];
        const double expected = ref->d / (double)n;
        ok = close_to(got, expected, elem_dt);
        snprintf(got_s, sizeof(got_s), "%.17g", got);
        snprintf(exp_s, sizeof(exp_s), "%.17g", expected);
    }
    else if (is_float(out_dt)) {
        const double got = load(out, out_dt, 0);
        ok = close_to(got, ref->d, out_dt);
        snprintf(got_s, sizeof(got_s), "%.17g", got);
        snprintf(exp_s, sizeof(exp_s), "%.17g", ref->d);
    }
    else if (out_dt == ME_BOOL) {
        /* min/max of a bool argument */
        const uint8_t got = ((const uint8_t *)out)[0];
        ok = got == (ref->i != 0 ? 1 : 0);
        snprintf(got_s, sizeof(got_s), "%u", (unsigned)got);
        snprintf(exp_s, sizeof(exp_s), "%d", ref->i != 0 ? 1 : 0);
    }
    else {
        const double got = load(out, out_dt, 0);
        ok = got == (double)ref->i;
        snprintf(got_s, sizeof(got_s), "%.17g", got);
        snprintf(exp_s, sizeof(exp_s), "%lld", (long long)ref->i);
    }
    if (!ok) {
        printf("  FAILED %s: got %s, expected %s\n", label, got_s, exp_s);
        failures++;
    }
    return ok;
}

typedef struct {
    int64_t shape[2];
    int32_t chunk[2];
    int32_t block[2];
} geometry;

/* Padded edge chunks and blocks / an exact tiling. */
static const geometry geoms[2] = {
    {{37, 41}, {20, 25}, {8, 10}},
    {{16, 64}, {16, 64}, {8, 32}},
};
/* Smaller arrays for prod, so that the product stays exact. */
static const geometry prod_geoms[2] = {
    {{3, 5}, {2, 4}, {1, 3}},
    {{2, 8}, {2, 8}, {1, 4}},
};

/* Evaluate a compiled reduction block by block, check every block result
 * against a reference over that block's valid items (stopping at the first
 * mismatch), and fold the block results into `result`. */
static int eval_by_blocks(const me_expr *expr, const geometry *g, void *const *data,
                          const test_case *tc, me_dtype in_dt, me_dtype el_dt, me_dtype out_dt,
                          red_kind r, const char *label, void *result) {
    const int nvars = tc->nvars;
    bool block_failed = false;
    const int64_t nch0 = (g->shape[0] + g->chunk[0] - 1) / g->chunk[0];
    const int64_t nch1 = (g->shape[1] + g->chunk[1] - 1) / g->chunk[1];
    const int nb0 = (g->chunk[0] + g->block[0] - 1) / g->block[0];
    const int nb1 = (g->chunk[1] + g->block[1] - 1) / g->block[1];
    const int block_nitems = g->block[0] * g->block[1];
    void *bufs[2] = {NULL, NULL};
    acc_t acc;
    int64_t total = 0;
    int rc = ME_EVAL_SUCCESS;

    acc_init(&acc, r);
    for (int v = 0; v < nvars; v++) {
        bufs[v] = malloc((size_t)block_nitems * 8);
        if (!bufs[v]) {
            rc = ME_EVAL_ERR_OOM;
            goto done;
        }
    }
    for (int64_t c0 = 0; c0 < nch0; c0++) {
        for (int64_t c1 = 0; c1 < nch1; c1++) {
            int64_t clen0 = g->shape[0] - c0 * g->chunk[0];
            int64_t clen1 = g->shape[1] - c1 * g->chunk[1];
            if (clen0 > g->chunk[0]) clen0 = g->chunk[0];
            if (clen1 > g->chunk[1]) clen1 = g->chunk[1];
            for (int b0 = 0; b0 < nb0; b0++) {
                for (int b1 = 0; b1 < nb1; b1++) {
                    const int64_t nchunk = c0 * nch1 + c1;
                    const int64_t nblock = (int64_t)b0 * nb1 + b1;
                    int64_t valid = 0;
                    unsigned char out[16];
                    const void *ptrs[2];
                    acc_t bref;

                    if (me_nd_valid_nitems(expr, nchunk, nblock, &valid) != ME_EVAL_SUCCESS) {
                        rc = ME_EVAL_ERR_INVALID_ARG;
                        goto done;
                    }
                    if (valid == 0) continue;
                    acc_init(&bref, r);
                    for (int v = 0; v < nvars; v++) {
                        for (int i0 = 0; i0 < g->block[0]; i0++) {
                            for (int i1 = 0; i1 < g->block[1]; i1++) {
                                const int64_t l0 = (int64_t)b0 * g->block[0] + i0;
                                const int64_t l1 = (int64_t)b1 * g->block[1] + i1;
                                const int64_t gi = (c0 * g->chunk[0] + l0) * g->shape[1] +
                                    c1 * g->chunk[1] + l1;
                                const bool in_block = l0 < clen0 && l1 < clen1;
                                const double val = in_block ? load(data[v], in_dt, gi) : 0.0;
                                store(bufs[v], in_dt, i0 * g->block[1] + i1, val);
                                if (in_block && v == nvars - 1) {
                                    const double x = load(data[0], in_dt, gi);
                                    const double y = nvars > 1 ? load(data[1], in_dt, gi) : 0.0;
                                    double ev = tc->ref(x, y);
                                    if (el_dt == ME_FLOAT32) ev = (double)(float)ev;
                                    acc_push(&bref, r, ev, is_float(el_dt));
                                }
                            }
                        }
                        ptrs[v] = bufs[v];
                    }
                    memset(out, 0xA5, sizeof(out));
                    rc = me_eval_nd(expr, ptrs, nvars, out, block_nitems, nchunk, nblock, NULL);
                    if (rc != ME_EVAL_SUCCESS) goto done;
                    if (!block_failed) {
                        char blabel[256];
                        snprintf(blabel, sizeof(blabel), "%s, chunk %lld block %lld", label,
                                 (long long)nchunk, (long long)nblock);
                        block_failed = !expect(blabel, out, out_dt, r, &bref, valid, el_dt);
                    }
                    if (r == R_MEAN) {
                        acc_push(&acc, R_SUM, ((const double *)out)[0] * (double)valid, true);
                    }
                    else if (r == R_ANY || r == R_ALL) {
                        acc_push(&acc, r, ((const bool *)out)[0] ? 1.0 : 0.0, false);
                    }
                    else if (out_dt == ME_BOOL) {
                        acc_push(&acc, r, ((const uint8_t *)out)[0], false);
                    }
                    else {
                        acc_push(&acc, r, load(out, out_dt, 0), is_float(out_dt));
                    }
                    total += valid;
                }
            }
        }
    }
    if (r == R_ANY || r == R_ALL) ((bool *)result)[0] = acc.b;
    else if (r == R_MEAN) ((double *)result)[0] = acc.d / (double)total;
    else if (out_dt == ME_BOOL) ((bool *)result)[0] = acc.d != 0.0;
    else if (is_float(out_dt)) store(result, out_dt, 0, acc.d);
    else store(result, out_dt, 0, (double)((r == R_MIN || r == R_MAX) ? (int64_t)acc.d : acc.i));
done:
    free(bufs[0]);
    free(bufs[1]);
    return rc;
}

static void run_case(const test_case *tc, me_dtype in_dt, red_kind r) {
    const me_dtype el_dt = elem_dtype(tc->res, in_dt);
    const me_dtype out_dt = reduction_dtype(r, el_dt);
    const bool float_result = is_float(el_dt);
    const me_variable vars[2] = {{"x", in_dt}, {"y", in_dt}};
    char rexpr[128];
    char label[192];

    snprintf(rexpr, sizeof(rexpr), "%s(%s)", red_names[r], tc->expr);
    for (int gi = 0; gi < 2; gi++) {
        const geometry *g = (r == R_PROD) ? &prod_geoms[gi] : &geoms[gi];
        const int64_t n = g->shape[0] * g->shape[1];
        void *data[2] = {NULL, NULL};
        acc_t ref;
        me_expr *expr = NULL;
        int err = 0;

        for (int v = 0; v < tc->nvars; v++) {
            data[v] = malloc((size_t)n * 8);
            if (!data[v]) {
                printf("  FAILED %s: out of memory\n", rexpr);
                failures++;
                free(data[0]);
                return;
            }
            for (int64_t i = 0; i < n; i++) {
                store(data[v], in_dt, i, input_value(in_dt, i, v, r));
            }
        }
        acc_init(&ref, r);
        for (int64_t i = 0; i < n; i++) {
            const double x = load(data[0], in_dt, i);
            const double y = tc->nvars > 1 ? load(data[1], in_dt, i) : 0.0;
            double v = tc->ref(x, y);
            if (el_dt == ME_FLOAT32) v = (double)(float)v;
            acc_push(&ref, r, v, float_result);
        }

        /* Whole array through me_eval (first geometry only). */
        if (gi == 0) {
            snprintf(label, sizeof(label), "%s [%s] me_eval", rexpr, dtype_name(in_dt));
            if (me_compile(rexpr, vars, tc->nvars, out_dt, &err, &expr) != ME_COMPILE_SUCCESS) {
                printf("  FAILED %s: compile error at %d\n", label, err);
                failures++;
            }
            else {
                unsigned char out[16];
                const void *ptrs[2] = {data[0], data[1]};
                memset(out, 0xA5, sizeof(out));
                if (me_eval(expr, ptrs, tc->nvars, out, (int)n, NULL) != ME_EVAL_SUCCESS) {
                    printf("  FAILED %s: eval error\n", label);
                    failures++;
                }
                else {
                    expect(label, out, out_dt, r, &ref, n, el_dt);
                }
                me_free(expr);
                expr = NULL;
            }
        }

        snprintf(label, sizeof(label), "%s [%s] me_eval_nd geometry %d", rexpr,
                 dtype_name(in_dt), gi);
        if (me_compile_nd(rexpr, vars, tc->nvars, out_dt, 2, g->shape, g->chunk, g->block,
                          &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("  FAILED %s: compile error at %d\n", label, err);
            failures++;
        }
        else {
            unsigned char result[16] = {0};
            if (eval_by_blocks(expr, g, data, tc, in_dt, el_dt, out_dt, r, label, result) !=
                ME_EVAL_SUCCESS) {
                printf("  FAILED %s: eval error\n", label);
                failures++;
            }
            else {
                expect(label, result, out_dt, r, &ref, n, el_dt);
            }
            me_free(expr);
        }
        free(data[0]);
        free(data[1]);
    }
}

static void test_expression_matrix(void) {
    printf("\n=== reductions by dtype and argument kind ===\n");
    const int before = failures;
    for (int c = 0; c < N_CASES; c++) {
        for (int d = 0; d < N_DTYPES; d++) {
            const me_dtype dt = all_dtypes[d];
            if (cases[c].dtypes == FLOAT_ONLY && !is_float(dt)) continue;
            if (cases[c].dtypes == SIGNED_OR_FLOAT && !(is_float(dt) || is_signed_int(dt))) {
                continue;
            }
            for (int r = 0; r < R_COUNT; r++) {
                run_case(&cases[c], dt, (red_kind)r);
            }
        }
    }
    printf("  %s\n", failures == before ? "ok" : "FAILED");
}

/* Reductions whose whole argument is constant, including constant-only
 * function nodes, over n items. */
static void test_constant_argument(void) {
    printf("\n=== reductions of a constant argument ===\n");
    static const struct {
        const char *expr;
        me_dtype dtype;
        double expected;
    } cc[] = {
        {"sum(2)", ME_INT64, 20.0},
        {"sum(-2)", ME_INT64, -20.0},
        {"sum(sqrt(4.0))", ME_FLOAT64, 20.0},
        {"sum(-2.5)", ME_FLOAT64, -25.0},
        {"prod(-2)", ME_INT64, 1024.0},
        {"min(-2)", ME_INT64, -2.0},
        {"max(sin(1.0))", ME_FLOAT64, 0.8414709848078965},
        {"mean(-2.5)", ME_FLOAT64, -2.5},
        {"any(0 - 0)", ME_BOOL, 0.0},
        {"all(-1)", ME_BOOL, 1.0},
        {"sum(where(1 > 0, -1, 1))", ME_INT64, -10.0},
    };
    const int before = failures;
    for (size_t c = 0; c < sizeof(cc) / sizeof(cc[0]); c++) {
        me_expr *expr = NULL;
        int err = 0;
        unsigned char out[16];
        checks++;
        if (me_compile(cc[c].expr, NULL, 0, cc[c].dtype, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("  FAILED %s: compile error at %d\n", cc[c].expr, err);
            failures++;
            continue;
        }
        memset(out, 0xA5, sizeof(out));
        if (me_eval(expr, NULL, 0, out, 10, NULL) != ME_EVAL_SUCCESS) {
            printf("  FAILED %s: eval error\n", cc[c].expr);
            failures++;
        }
        else {
            const double got = load(out, cc[c].dtype, 0);
            const bool ok = (cc[c].dtype == ME_BOOL) ? (out[0] == (cc[c].expected != 0.0)) :
                close_to(got, cc[c].expected, cc[c].dtype);
            if (!ok) {
                printf("  FAILED %s: got %.17g, expected %.17g\n", cc[c].expr, got, cc[c].expected);
                failures++;
            }
        }
        me_free(expr);
    }
    printf("  %s\n", failures == before ? "ok" : "FAILED");
}

/* Reductions inside a larger expression keep their constant operands too. */
static void test_reduction_in_expression(void) {
    printf("\n=== reductions with constants inside larger expressions ===\n");
    double x[10];
    double out[10];
    const void *ptrs[1] = {x};
    me_variable vars[] = {{"x", ME_FLOAT64}};
    me_expr *expr = NULL;
    int err = 0;
    const int before = failures;
    for (int i = 0; i < 10; i++) x[i] = (double)i;
    checks++;
    /* sum(x * -2) = -90, max(x - -1) = 10, sum(-2) over 10 items = -20 */
    if (me_compile("sum(x * -2) + max(x - -1) + sum(-2)", vars, 1, ME_FLOAT64, &err, &expr) !=
        ME_COMPILE_SUCCESS) {
        printf("  FAILED compile error at %d\n", err);
        failures++;
        return;
    }
    if (me_eval(expr, ptrs, 1, out, 10, NULL) != ME_EVAL_SUCCESS || out[0] != -100.0) {
        printf("  FAILED got %.17g, expected -100\n", out[0]);
        failures++;
    }
    me_free(expr);
    printf("  %s\n", failures == before ? "ok" : "FAILED");
}

int main(void) {
    test_expression_matrix();
    test_constant_argument();
    test_reduction_in_expression();
    printf("\n%d of %d checks failed\n", failures, checks);
    return failures ? 1 : 0;
}
