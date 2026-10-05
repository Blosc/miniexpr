/* Raw C callers must get the same chains as Python, without source rewriting. */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "../src/miniexpr.h"

typedef struct {
    double values[4];
    int calls[16], count;
} probe_context;

static double probe(void *opaque, double index) {
    probe_context *ctx = opaque;
    int i = (int)index;
    if (ctx->count < 16) {
        ctx->calls[ctx->count++] = i;
    }
    return i >= 0 && i < 4 ? ctx->values[i] : NAN;
}

static int evaluate_count(const char *source, const me_variable *vars, int nvars,
                    const void **inputs, int ninputs, me_dtype dtype, void *out,
                    int jit, bool *has_jit, int count) {
    int64_t shape[] = {count};
    int32_t chunks[] = {count}, blocks[] = {count};
    int error = 0;
    me_expr *expr = NULL;
    int rc = me_compile_nd_jit(source, vars, nvars, dtype, 1, shape, chunks, blocks,
                               jit, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS || !expr) {
        printf("Compile failed at %d: %s\n", error, source);
        return 1;
    }
    if (has_jit) {
        *has_jit = me_expr_has_jit_kernel(expr);
    }
    rc = me_eval_nd(expr, inputs, ninputs, out, count, 1, 0, NULL);
    me_free(expr);
    if (rc != ME_EVAL_SUCCESS) {
        printf("Evaluation failed: %s\n", source);
        return 1;
    }
    return 0;
}

static int evaluate(const char *source, const me_variable *vars, int nvars,
                    const void **inputs, int ninputs, me_dtype dtype, void *out,
                    int jit, bool *has_jit) {
    return evaluate_count(source, vars, nvars, inputs, ninputs, dtype, out, jit, has_jit, 1);
}

int main(void) {
    const char *expressions[] = {
        "0 <= x < 5", "5 > x >= 0", "x == 2 != 3", "x != 2 == 2",
        "-2 < x <= 4 != 3", "(0 < x < 5) + (2 < x < 7)",
        "not (0 < x < 5)", "(x < 0) or (0 < x < 5)",
        "(x > 0) and (0 < x < 5)", "where(0 < x < 5, 1, 2)",
        "(x < 0) == (0 < x < 5)", "0 < x < 12 / x",
        "0.25 < x < 0.75",
    };
    const double values[] = {-3, -1, 0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, NAN, INFINITY};
    for (int backend = 0; backend < 2; backend++) {
        const char *pragma = backend ? "# me:compiler=cc\n" : "# me:compiler=tcc\n";
        for (int jit = ME_JIT_OFF; jit <= ME_JIT_ON; jit++) {
            double x = 1, baseline = 0;
            const void *inputs[] = {&x};
            me_variable vars[] = {{"x", ME_FLOAT64}};
            bool available = false;
            char source[2048];
            snprintf(source, sizeof(source), "%sdef k(x):\n    return x + 1\n", pragma);
            if (evaluate(source, vars, 1, inputs, 1, ME_FLOAT64, &baseline, jit, &available)) {
                return 1;
            }
            for (size_t v = 0; v < sizeof(values) / sizeof(values[0]); v++) {
                x = values[v];
                double expected[] = {
                    0 <= x && x < 5, 5 > x && x >= 0, x == 2 && 2 != 3, x != 2 && 2 == 2,
                    -2 < x && x <= 4 && 4 != 3,
                    (int)(0 < x && x < 5) + (int)(2 < x && x < 7),
                    !(0 < x && x < 5), (x < 0) || (0 < x && x < 5),
                    (x > 0) && (0 < x && x < 5), (0 < x && x < 5) ? 1 : 2,
                    (x < 0) == (0 < x && x < 5), 0 < x && x < 12 / x,
                    0.25 < x && x < 0.75,
                };
                for (size_t e = 0; e < sizeof(expressions) / sizeof(expressions[0]); e++) {
                    snprintf(source, sizeof(source), "%sdef k(x):\n    return %s\n", pragma, expressions[e]);
                    double out = -99;
                    bool has_jit = false;
                    if (evaluate(source, vars, 1, inputs, 1, ME_FLOAT64, &out, jit, &has_jit) ||
                        out != expected[e] || (available && !has_jit)) {
                        printf("Mismatch: %s at %g: %g != %g, jit=%d\n", expressions[e], x, out, expected[e], has_jit);
                        return 1;
                    }
                    if (e != 5 && e != 9) {
                        bool bool_out = false;
                        if (evaluate(source, vars, 1, inputs, 1, ME_BOOL, &bool_out, jit, NULL) ||
                            bool_out != (bool)expected[e]) {
                            printf("Boolean mismatch: %s at %g\n", expressions[e], x);
                            return 1;
                        }
                    }
                }
            }
            probe_context ctx = {0};
            me_variable probe_vars[] = {
                {"x", ME_FLOAT64}, {"probe", ME_FLOAT64, probe, ME_CLOSURE1, &ctx},
            };
            const double sequences[][4] = {{0, 1, 2, 3}, {2, 1, 2, 3}, {0, 2, 1, 3}};
            const int counts[] = {4, 2, 3};
            for (int s = 0; s < 3; s++) {
                memcpy(ctx.values, sequences[s], sizeof(ctx.values));
                ctx.count = 0;
                snprintf(source, sizeof(source), "%sdef k(x):\n    return probe(0) < probe(1) < probe(2) < probe(3)\n", pragma);
                bool out = false;
                if (evaluate(source, probe_vars, 2, inputs, 1, ME_BOOL, &out, jit, NULL) ||
                    ctx.count != counts[s] || out != (s == 0)) {
                    printf("Probe count/result mismatch: %d in mode %d\n", ctx.count, jit);
                    return 1;
                }
                for (int i = 0; i < ctx.count; i++) {
                    if (ctx.calls[i] != i) {
                        printf("Operand order mismatch\n");
                        return 1;
                    }
                }
            }
            double masked_x[] = {-1, 0, 1, 2};
            const void *masked_inputs[] = {masked_x};
            bool masked_out[4] = {false};
            ctx.values[1] = ctx.values[2] = 10;
            ctx.count = 0;
            snprintf(source, sizeof(source), "%sdef k(x):\n    return 0 < x < probe(x)\n", pragma);
            if (evaluate_count(source, probe_vars, 2, masked_inputs, 1, ME_BOOL, masked_out, jit, NULL, 4) ||
                ctx.count != 2 || ctx.calls[0] != 1 || ctx.calls[1] != 2 ||
                masked_out[0] || masked_out[1] || !masked_out[2] || !masked_out[3]) {
                printf("Inactive callback operands were evaluated or misread\n");
                return 1;
            }
            snprintf(source, sizeof(source),
                "%sdef k(x):\n    y = x\n    while 0 <= y < 3:\n        y += 1\n        continue\n"
                "    if 3 <= y < 5:\n        y += 10\n    elif -5 <= y < 0:\n        y -= 10\n"
                "    else:\n        y += 0 < y < 10\n    return y\n", pragma);
            for (int i = -3; i <= 7; i++) {
                x = i;
                double expected = i;
                while (0 <= expected && expected < 3) {
                    expected++;
                }
                if (3 <= expected && expected < 5) {
                    expected += 10;
                } else if (-5 <= expected && expected < 0) {
                    expected -= 10;
                } else {
                    expected += 0 < expected && expected < 10;
                }
                double out = -99;
                if (evaluate(source, vars, 1, inputs, 1, ME_FLOAT64, &out, jit, NULL) || out != expected) {
                    printf("Loop/elif mismatch at %d: %g != %g\n", i, out, expected);
                    return 1;
                }
            }
            double batch[] = {-1, 0, 1, 2, 3, 4, 5, 6};
            const void *batch_inputs[] = {batch};
            double batch_out[8];
            snprintf(source, sizeof(source), "%sdef k(x):\n    return (0 < x < 5) + (2 < x < 7)\n", pragma);
            if (evaluate_count(source, vars, 1, batch_inputs, 1, ME_FLOAT64, batch_out, jit, NULL, 8)) {
                return 1;
            }
            for (int i = 0; i < 8; i++) {
                int expected = (int)(0 < batch[i] && batch[i] < 5) + (int)(2 < batch[i] && batch[i] < 7);
                if (batch_out[i] != expected) {
                    printf("Batched chain arithmetic mismatch at %d: %g != %d\n", i, batch_out[i], expected);
                    return 1;
                }
            }
            snprintf(source, sizeof(source), "%sdef k(x):\n    y = 0\n"
                     "    for i in range(int(0 < x < 5), 3):\n        y += 1\n    return y\n", pragma);
            if (evaluate_count(source, vars, 1, batch_inputs, 1, ME_FLOAT64, batch_out, jit, NULL, 8)) {
                return 1;
            }
            for (int i = 0; i < 8; i++) {
                int expected = 3 - (int)(0 < batch[i] && batch[i] < 5);
                if (batch_out[i] != expected) {
                    printf("Range-argument chain mismatch at %d: %g != %d\n", i, batch_out[i], expected);
                    return 1;
                }
            }
        }
    }
    const char *invalid[] = {"0 < < x", "0 < x <", "0 < x < (2 +", "0 < x < unknown(x)",
                             "0 < x < sin(x,)", "0 < x < 'unterminated"};
    me_variable vars[] = {{"x", ME_FLOAT64}};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        char source[1024];
        snprintf(source, sizeof(source), "def k(x):\n    return %s\n", invalid[i]);
        me_expr *expr = NULL;
        int error = 0;
        int rc = me_compile(source, vars, 1, ME_BOOL, &error, &expr);
        if (rc == ME_COMPILE_SUCCESS || expr || error <= 0) {
            printf("Malformed/undefined chain accepted or unlocated: %s (%d)\n", invalid[i], error);
            me_free(expr);
            return 1;
        }
    }
    /* The non-ND C API consumes raw chain syntax too. */
    double x[] = {-1, 0, 1, 2};
    const void *inputs[] = {x};
    bool out[4] = {false};
    me_expr *expr = NULL;
    int error = 0;
    if (me_compile("def k(x):\n    return 0 < x < 2\n", vars, 1, ME_BOOL, &error, &expr) != ME_COMPILE_SUCCESS ||
        me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS ||
        out[0] || out[1] || !out[2] || out[3]) {
        printf("Non-ND raw C chain evaluation failed\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    return 0;
}
