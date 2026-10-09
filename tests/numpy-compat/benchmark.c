/* Native-only baseline; full DSL controls are NOT portable conformance evidence. */
#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include "miniexpr_artifact.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    return 1e9 * (double)ticks.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e9 * ts.tv_sec + ts.tv_nsec;
#endif
}

static int compare_double(const void *a, const void *b) {
    double left = *(const double *)a, right = *(const double *)b;
    return (left > right) - (left < right);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s portable|interpreter|tcc|cc float32|float64|int64 count repeats\n", argv[0]);
        return 2;
    }
    bool portable = !strcmp(argv[1], "portable"), off = !strcmp(argv[1], "interpreter");
    if (!portable && !off && strcmp(argv[1], "tcc") && strcmp(argv[1], "cc")) return 2;
    me_dtype dtype = !strcmp(argv[2], "float32") ? ME_FLOAT32 :
                     !strcmp(argv[2], "float64") ? ME_FLOAT64 :
                     !strcmp(argv[2], "int64") ? ME_INT64 : ME_AUTO;
    char *end;
    long count = strtol(argv[3], &end, 10);
    if (*end || count < 1 || count > 1048576 || dtype == ME_AUTO) return 2;
    long repeats = strtol(argv[4], &end, 10);
    if (*end || repeats < 1 || repeats > 101) return 2;
    size_t width = dtype == ME_FLOAT32 ? 4 : 8;
    void *x = calloc((size_t)count, width), *output = calloc((size_t)count, width);
    if (!x || !output) {
        free(x);
        free(output);
        return 2;
    }
    for (long i = 0; i < count; i++) {
        if (dtype == ME_FLOAT32) ((float *)x)[i] = (float)i;
        else if (dtype == ME_FLOAT64) ((double *)x)[i] = (double)i;
        else ((int64_t *)x)[i] = i;
    }
    char source[256], json[1024], reason[256] = "";
    snprintf(source, sizeof(source), "# me:compiler=%s\n# me:fp=strict\ndef k(x):\n    return x * 2 + 1\n",
             !strcmp(argv[1], "cc") ? "cc" : "tcc");
    snprintf(json, sizeof(json),
        "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\"],\"source\":\"def k(x):\\n    return x * 2 + 1\\n\","
        "\"entry_point\":\"k\",\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},\"semantics\":{\"fp\":\"strict\"},"
        "\"context\":{\"ndim\":0},\"metadata\":{}}", argv[2], argv[2]);
    me_variable variable = {.name = "x", .dtype = dtype, .type = ME_VARIABLE};
    me_artifact *artifact = NULL;
    me_dsl_compiled_program *program = NULL, *second = NULL;
    me_artifact_error error = {0};
    me_jit_mode mode = off ? ME_JIT_OFF : ME_JIT_ON;
    double start = now_ns();
    bool is_dsl;
    int position;
    if (portable) {
        if (me_artifact_load(json, strlen(json), mode, &artifact, &error)) goto fail;
    }
    else {
        program = dsl_compile_program_profile(source, &variable, 1, dtype, 0, mode,
            ME_DSL_PROFILE_FULL, &position, &is_dsl, reason, sizeof(reason));
        if (!program) goto fail;
    }
    double compile_ns = now_ns() - start;
    const char *backend = "interpreter";
    if (program && program->jit_kernel_fn) {
        backend = program->jit_tcc_state ? "tcc" : "cc";
    }
    /* Refuse to call a fallback a compiler measurement. */
    if ((!portable && !off && strcmp(backend, argv[1])) ||
        (portable && me_artifact_has_jit(artifact))) goto fail;
    double recompile_ns = 0;
    if (program) {
        start = now_ns();
        second = dsl_compile_program_profile(source, &variable, 1, dtype, 0, mode,
            ME_DSL_PROFILE_FULL, &position, &is_dsl, reason, sizeof(reason));
        recompile_ns = now_ns() - start;
        if (!second || (!!second->jit_kernel_fn != !!program->jit_kernel_fn)) goto fail;
    }
    const void *inputs[] = {x};
    me_artifact_input binding = {"x", dtype, x, (size_t)count};
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.jit_mode = mode;
    double samples[102];
    for (long repeat = 0; repeat <= repeats; repeat++) {
        int batch = repeat && count == 16 ? 1000 : 1;
        start = now_ns();
        for (int i = 0; i < batch; i++) {
            int rc = portable ? me_artifact_eval(artifact, &binding, 1, output, (size_t)count, &error) :
                dsl_eval_program(program, inputs, 1, output, (int)count, &params, 0, NULL, NULL, NULL, NULL);
            if (rc) goto fail;
        }
        samples[repeat] = (now_ns() - start) / batch;
        for (long i = 0; i < count; i++) {
            double actual = dtype == ME_FLOAT32 ? ((float *)output)[i] :
                            dtype == ME_FLOAT64 ? ((double *)output)[i] : (double)((int64_t *)output)[i];
            if (actual != 2 * i + 1) goto fail;
        }
    }
    double first_ns = samples[0];
    qsort(samples + 1, (size_t)repeats, sizeof(double), compare_double);
    printf("{\"requested\":\"%s\",\"backend\":\"%s\",\"dtype\":\"%s\",\"nitems\":%ld,"
           "\"compile_ns\":%.0f,\"same_process_recompile_ns\":%.0f,\"first_ns\":%.0f,\"warm_median_ns\":%.0f}"
           "\n", argv[1], backend, argv[2], count, compile_ns, recompile_ns, first_ns, samples[1 + repeats / 2]);
    dsl_compiled_program_free(second);
    dsl_compiled_program_free(program);
    me_artifact_free(artifact);
    free(x);
    free(output);
    return 0;
fail:
    fprintf(stderr, "compile/backend/value verification failed: %s %s\n", reason, error.message);
    dsl_compiled_program_free(second);
    dsl_compiled_program_free(program);
    me_artifact_free(artifact);
    free(x);
    free(output);
    return 1;
}
