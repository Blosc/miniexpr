/* Canonical Array API inverse trig names; NumPy aliases are frontend syntax. */
#include "../src/miniexpr.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    const char *names[] = {"acos", "asin", "atan", "atan2", "acosh", "asinh", "atanh"};
    const char *aliases[] = {"arccos", "arcsin", "arctan", "arctan2",
                             "arccosh", "arcsinh", "arctanh"};
    double x[] = {0.0, 0.5, 0.707, 0.866, 1.0};
    double out[5];
    me_variable vars[] = {{"x", ME_FLOAT64}};
    const void *inputs[] = {x};
    int failures = 0;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        char source[64];
        int error = 0;
        me_expr *expr = NULL;
        snprintf(source, sizeof(source), "%s(x%s)", aliases[i], i == 3 ? ", 1.0" : "");
        int rc = me_compile(source, vars, 1, ME_FLOAT64, &error, &expr);
        if (rc == ME_COMPILE_SUCCESS || expr != NULL) {
            fprintf(stderr, "removed alias accepted: %s\n", source);
            failures++;
        }
        me_free(expr);
        expr = NULL;
        snprintf(source, sizeof(source), "%s(x%s)", names[i], i == 3 ? ", 1.0" : "");
        rc = me_compile(source, vars, 1, ME_FLOAT64, &error, &expr);
        if (rc != ME_COMPILE_SUCCESS || !expr) {
            fprintf(stderr, "canonical name rejected: %s\n", source);
            failures++;
        } else if (i < 4) {
            if (me_eval(expr, inputs, 1, out, 5, NULL) != ME_EVAL_SUCCESS) {
                failures++;
            } else {
                for (int j = 0; j < 5; j++) {
                    double expected = i == 0 ? acos(x[j]) : i == 1 ? asin(x[j]) :
                                      i == 2 ? atan(x[j]) : atan2(x[j], 1.0);
                    if (fabs(out[j] - expected) > 1e-9) failures++;
                }
            }
        }
        me_free(expr);
    }

    /* Preserve the Windows int32/ME_AUTO and small-chunk regression coverage. */
    int32_t integers[] = {-1, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    me_variable ivars[] = {{"x", ME_INT32}};
    int error = 0;
    me_expr *expr = NULL;
    if (me_compile("acos(x)", ivars, 1, ME_AUTO, &error, &expr) != ME_COMPILE_SUCCESS || !expr) {
        failures++;
    } else {
        for (int chunk = 3; chunk <= 10; chunk += 7) {
            for (int offset = 0; offset < 10; offset += chunk) {
                int count = 10 - offset < chunk ? 10 - offset : chunk;
                double result[10];
                const void *values[] = {integers + offset};
                if (me_eval(expr, values, 1, result, count, NULL) != ME_EVAL_SUCCESS) {
                    failures++;
                } else {
                    for (int j = 0; j < count; j++) {
                        if (fabs(result[j] - acos(integers[offset + j])) > 1e-9) failures++;
                    }
                }
            }
        }
    }
    me_free(expr);
    printf("Canonical inverse trig / removed alias checks: %d failures\n", failures);
    return failures ? 1 : 0;
}
