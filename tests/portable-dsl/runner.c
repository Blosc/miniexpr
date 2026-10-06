/* Standalone raw-source conformance runner. This is not an artifact loader. */
#include <math.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "miniexpr.h"

#define MAX_ITEMS 4096
#define MAX_INPUTS 32
#define MAX_SOURCE 65536

static me_dtype parse_dtype(const char *name) {
    if (!strcmp(name, "bool")) {
        return ME_BOOL;
    }
    if (!strcmp(name, "int32")) {
        return ME_INT32;
    }
    if (!strcmp(name, "int64")) {
        return ME_INT64;
    }
    if (!strcmp(name, "float32")) {
        return ME_FLOAT32;
    }
    if (!strcmp(name, "float64")) {
        return ME_FLOAT64;
    }
    return ME_AUTO;
}

static size_t dtype_size(me_dtype dtype) {
    switch (dtype) {
        case ME_BOOL: return sizeof(bool);
        case ME_INT32: return sizeof(int32_t);
        case ME_INT64: return sizeof(int64_t);
        case ME_FLOAT32: return sizeof(float);
        case ME_FLOAT64: return sizeof(double);
        default: return 0;
    }
}

/* Parse integers exactly, never through double (including values above 2**53). */
static bool read_value(FILE *file, me_dtype dtype, void *buffer, int index) {
    char token[128], *end = NULL;
    if (fscanf(file, "%127s", token) != 1) {
        return false;
    }
    errno = 0;
    switch (dtype) {
        case ME_BOOL:
            if (strcmp(token, "0") && strcmp(token, "1")) {
                return false;
            }
            ((bool *)buffer)[index] = token[0] == '1';
            return true;
        case ME_INT32:
        case ME_INT64: {
            intmax_t value = strtoimax(token, &end, 10);
            if (errno || end == token || *end || value < INT64_MIN || value > INT64_MAX) {
                return false;
            }
            if (dtype == ME_INT32) {
                if (value < INT32_MIN || value > INT32_MAX) {
                    return false;
                }
                ((int32_t *)buffer)[index] = (int32_t)value;
            } else {
                ((int64_t *)buffer)[index] = (int64_t)value;
            }
            return true;
        }
        case ME_FLOAT32:
            ((float *)buffer)[index] = strtof(token, &end);
            break;
        case ME_FLOAT64:
            ((double *)buffer)[index] = strtod(token, &end);
            break;
        default:
            return false;
    }
    return !errno && end != token && !*end;
}

static bool compare_value(me_dtype dtype, const void *actual, const void *expected, int index) {
    if (dtype == ME_BOOL) {
        return ((const bool *)actual)[index] == ((const bool *)expected)[index];
    }
    if (dtype == ME_INT32) {
        return ((const int32_t *)actual)[index] == ((const int32_t *)expected)[index];
    }
    if (dtype == ME_INT64) {
        return ((const int64_t *)actual)[index] == ((const int64_t *)expected)[index];
    }
    double a = dtype == ME_FLOAT32 ? ((const float *)actual)[index] : ((const double *)actual)[index];
    double e = dtype == ME_FLOAT32 ? ((const float *)expected)[index] : ((const double *)expected)[index];
    if (isnan(e)) {
        return isnan(a);
    }
    if (isinf(e)) {
        return a == e;
    }
    if (a == 0 && e == 0) {
        return !!signbit(a) == !!signbit(e);
    }
    double tolerance = dtype == ME_FLOAT32 ? 1e-6 : 1e-12;
    return isfinite(a) && fabs(a - e) <= tolerance + tolerance * fabs(e);
}

static void print_value(me_dtype dtype, const void *buffer, int index) {
    switch (dtype) {
        case ME_BOOL: printf("%d\n", (int)((const bool *)buffer)[index]); break;
        case ME_INT32: printf("%" PRId32 "\n", ((const int32_t *)buffer)[index]); break;
        case ME_INT64: printf("%" PRId64 "\n", ((const int64_t *)buffer)[index]); break;
        case ME_FLOAT32: printf("%.9g\n", (double)((const float *)buffer)[index]); break;
        case ME_FLOAT64: printf("%.17g\n", ((const double *)buffer)[index]); break;
        default: break;
    }
}

int main(int argc, char **argv) {
    int result = 1, count = 0, nvars = 0, error = 0;
    FILE *file = NULL;
    me_expr *expr = NULL;
    char *source = NULL;
    unsigned char *data = NULL;
    void *output = NULL, *expected = NULL;
    char outcome[16], input_type[16], output_type[16];
    me_dtype input_dtype = ME_AUTO, output_dtype = ME_AUTO;
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
    if (fscanf(file, "%15s %15s %15s %d %d", outcome, input_type, output_type, &count, &nvars) != 5 ||
        (strcmp(outcome, "ok") && strcmp(outcome, "compile_error") && strcmp(outcome, "eval_error")) || count < 1 ||
        count > MAX_ITEMS || nvars < 1 || nvars > MAX_INPUTS) {
        fprintf(stderr, "invalid fixture header\n");
        goto cleanup;
    }
    input_dtype = parse_dtype(input_type);
    output_dtype = parse_dtype(output_type);
    if (input_dtype == ME_AUTO || output_dtype == ME_AUTO ||
        dtype_size(input_dtype) > sizeof(double) || dtype_size(output_dtype) > sizeof(double)) {
        fprintf(stderr, "unsupported fixture dtype\n");
        goto cleanup;
    }
    for (int v = 0; v < nvars; v++) {
        if (fscanf(file, "%127s", names[v]) != 1) {
            fprintf(stderr, "missing fixture input name\n");
            goto cleanup;
        }
        variables[v].name = names[v];
        variables[v].dtype = input_dtype;
        inputs[v] = data + v * MAX_ITEMS * sizeof(double);
    }
    for (int i = 0; i < count; i++) {
        for (int v = 0; v < nvars; v++) {
            if (!read_value(file, input_dtype, data + v * MAX_ITEMS * sizeof(double), i)) {
                fprintf(stderr, "missing fixture input value\n");
                goto cleanup;
            }
        }
        if (!read_value(file, output_dtype, expected, i)) {
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
    me_portable_error profile_error;
    me_portable_status profile = me_validate_portable_dsl(source, ME_PORTABLE_DSL_VERSION,
        variables, nvars, output_dtype, &profile_error);
    bool expect_compile_error = !strcmp(outcome, "compile_error");
    if ((expect_compile_error && profile == ME_PORTABLE_SUCCESS) ||
        (!expect_compile_error && profile != ME_PORTABLE_SUCCESS)) {
        fprintf(stderr, "unexpected portable validation status %d at %d:%d: %s\n",
                profile, profile_error.line, profile_error.column, profile_error.message);
        goto cleanup;
    }
    int rc = me_compile_nd_jit(source, variables, nvars, output_dtype,
                              1, shape, grid, grid, mode, &error, &expr);
    if (!strcmp(outcome, "compile_error")) {
        if (rc == ME_COMPILE_SUCCESS || rc == ME_COMPILE_ERR_OOM) {
            fprintf(stderr, "expected a semantic compilation failure\n");
            goto cleanup;
        }
        printf("compile_error\n");
        result = 0;
        goto cleanup;
    }
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
    if (!strcmp(outcome, "eval_error")) {
        if (rc != ME_EVAL_ERR_INVALID_ARG) {
            fprintf(stderr, "expected an invalid-argument evaluation error, got %d\n", rc);
            goto cleanup;
        }
        printf("eval_error\n");
        result = 0;
        goto cleanup;
    }
    if (rc != ME_EVAL_SUCCESS) {
        fprintf(stderr, "evaluation error %d\n", rc);
        goto cleanup;
    }
    for (int i = 0; i < count; i++) {
        if (!compare_value(output_dtype, output, expected, i)) {
            fprintf(stderr, "element %d differs from expected %s value\n", i, output_type);
            goto cleanup;
        }
        print_value(output_dtype, output, i);
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
