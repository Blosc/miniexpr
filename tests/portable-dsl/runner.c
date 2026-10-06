/* Standalone raw-source conformance runner. This is not an artifact loader. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "miniexpr.h"

#define MAX_ITEMS 4096
#define MAX_INPUTS 32
#define MAX_SOURCE 65536

int main(int argc, char **argv) {
    int result = 1, count = 0, nvars = 0, error = 0;
    FILE *file = NULL;
    me_expr *expr = NULL;
    char *source = NULL;
    double *data = NULL, *output = NULL, *expected = NULL;
    char names[MAX_INPUTS][128];
    me_variable variables[MAX_INPUTS] = {0};
    const void *inputs[MAX_INPUTS] = {0};
    me_jit_mode mode = ME_JIT_DEFAULT;

    if (argc != 4) {
        fprintf(stderr, "usage: %s source.dsl case.txt off|on|default\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[3], "off")) {
        mode = ME_JIT_OFF;
    } else if (!strcmp(argv[3], "on")) {
        mode = ME_JIT_ON;
    } else if (strcmp(argv[3], "default")) {
        fprintf(stderr, "invalid JIT policy\n");
        return 2;
    }

    source = calloc(MAX_SOURCE + 1, 1);
    data = calloc(MAX_ITEMS * MAX_INPUTS, sizeof(double));
    output = calloc(MAX_ITEMS, sizeof(double));
    expected = calloc(MAX_ITEMS, sizeof(double));
    if (!source || !data || !output || !expected) {
        fprintf(stderr, "out of memory\n");
        goto cleanup;
    }
    file = fopen(argv[1], "rb");
    if (!file) {
        perror(argv[1]);
        goto cleanup;
    }
    size_t length = fread(source, 1, MAX_SOURCE, file);
    if (ferror(file) || fgetc(file) != EOF || memchr(source, '\0', length)) {
        fprintf(stderr, "invalid or oversized source\n");
        goto cleanup;
    }
    fclose(file);
    file = fopen(argv[2], "r");
    if (!file) {
        perror(argv[2]);
        goto cleanup;
    }
    if (fscanf(file, "%d %d", &count, &nvars) != 2 || count < 1 ||
        count > MAX_ITEMS || nvars < 1 || nvars > MAX_INPUTS) {
        fprintf(stderr, "invalid fixture dimensions\n");
        goto cleanup;
    }
    for (int v = 0; v < nvars; v++) {
        if (fscanf(file, "%127s", names[v]) != 1) {
            fprintf(stderr, "missing fixture input name\n");
            goto cleanup;
        }
        variables[v].name = names[v];
        variables[v].dtype = ME_FLOAT64;
        inputs[v] = data + v * MAX_ITEMS;
    }
    for (int i = 0; i < count; i++) {
        for (int v = 0; v < nvars; v++) {
            if (fscanf(file, "%lf", &data[v * MAX_ITEMS + i]) != 1) {
                fprintf(stderr, "missing fixture input value\n");
                goto cleanup;
            }
        }
        if (fscanf(file, "%lf", &expected[i]) != 1 || !isfinite(expected[i])) {
            fprintf(stderr, "invalid fixture expected value\n");
            goto cleanup;
        }
    }
    char trailing[2];
    if (fscanf(file, "%1s", trailing) != EOF || ferror(file)) {
        fprintf(stderr, "unexpected trailing fixture content\n");
        goto cleanup;
    }
    int64_t shape[] = {count};
    int32_t grid[] = {count};
    int rc = me_compile_nd_jit(source, variables, nvars, ME_FLOAT64,
                              1, shape, grid, grid, mode, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        const char *message = me_get_last_error_message();
        fprintf(stderr, "compile error %d at %d: %s\n", rc, error,
                message ? message : "no diagnostic");
        goto cleanup;
    }
    bool jit = me_expr_has_jit_kernel(expr);
    printf("jit=%d\n", (int)jit);
    if ((mode == ME_JIT_OFF && jit) || (mode == ME_JIT_ON && !jit)) {
        fprintf(stderr, "requested execution backend was not prepared\n");
        goto cleanup;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.jit_mode = mode;
    rc = me_eval(expr, inputs, nvars, output, count, &params);
    if (rc != ME_EVAL_SUCCESS) {
        fprintf(stderr, "evaluation error %d\n", rc);
        goto cleanup;
    }
    for (int i = 0; i < count; i++) {
        if (!isfinite(output[i]) ||
            fabs(output[i] - expected[i]) > 1e-12 + 1e-12 * fabs(expected[i])) {
            fprintf(stderr, "element %d: expected %.17g, got %.17g\n",
                    i, expected[i], output[i]);
            goto cleanup;
        }
        printf("%.17g\n", output[i]);
    }
    result = 0;

cleanup:
    if (file) {
        fclose(file);
    }
    me_free(expr);
    free(source);
    free(data);
    free(output);
    free(expected);
    return result;
}
