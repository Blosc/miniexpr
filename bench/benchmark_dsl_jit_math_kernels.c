/*
 * DSL JIT math-kernel baseline benchmark.
 *
 * Reports per-kernel:
 * - JIT cold compile latency
 * - JIT warm runtime throughput
 * - Interpreter throughput
 * - max-abs numerical diff (JIT warm vs interpreter)
 *
 * Usage:
 *   ./benchmark_dsl_jit_math_kernels [nitems] [repeats]
 */

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <dirent.h>
#include <unistd.h>
#endif

#include "miniexpr.h"

typedef struct {
    const char *name;
    const char *expr;
} math_kernel_def;

typedef struct {
    double compile_ms;
    double eval_ms_best;
    double ns_per_elem_best;
    double checksum;
} mode_result;

typedef struct {
    mode_result jit_cold;
    mode_result jit_warm;
    mode_result interp;
    double max_abs_diff;
} kernel_result;

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static bool parse_positive_int(const char *text, int *out_value) {
    if (!text || !text[0] || !out_value) {
        return false;
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v <= 0 || v > INT_MAX) {
        return false;
    }
    *out_value = (int)v;
    return true;
}

static char *dup_env_value(const char *name) {
    const char *value = getenv(name);
    if (!value) {
        return NULL;
    }
    size_t n = strlen(value) + 1;
    char *copy = malloc(n);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, value, n);
    return copy;
}

static void restore_env_value(const char *name, const char *value) {
    if (!name) {
        return;
    }
    if (value) {
        (void)setenv(name, value, 1);
    }
    else {
        (void)unsetenv(name);
    }
}

#if !defined(_WIN32) && !defined(_WIN64)
static void remove_files_in_dir(const char *dir_path) {
    if (!dir_path) {
        return;
    }
    DIR *dir = opendir(dir_path);
    if (!dir) {
        return;
    }
    struct dirent *ent = NULL;
    char path[1024];
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        if (snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name) >= (int)sizeof(path)) {
            continue;
        }
        (void)remove(path);
    }
    closedir(dir);
}
#endif

static const char *current_dsl_compiler_label(void) {
    const char *compiler = getenv("ME_BENCH_COMPILER");
    if (!compiler || compiler[0] == '\0') {
        return "tcc-default";
    }
    if (strcmp(compiler, "tcc") == 0) {
        return "tcc";
    }
    if (strcmp(compiler, "cc") == 0) {
        return "cc";
    }
    return "invalid";
}

static bool build_dsl_source(char *out, size_t out_size, const char *expr) {
    if (!out || out_size == 0 || !expr || expr[0] == '\0') {
        return false;
    }
    const char *compiler = getenv("ME_BENCH_COMPILER");
    bool use_compiler_pragma = false;
    const char *compiler_value = NULL;
    if (compiler && compiler[0] != '\0') {
        if (strcmp(compiler, "tcc") == 0) {
            use_compiler_pragma = true;
            compiler_value = "tcc";
        }
        else if (strcmp(compiler, "cc") == 0) {
            use_compiler_pragma = true;
            compiler_value = "cc";
        }
        else {
            fprintf(stderr, "invalid ME_BENCH_COMPILER=%s (expected tcc or cc)\n", compiler);
            return false;
        }
    }
    int n = 0;
    if (use_compiler_pragma) {
        n = snprintf(out, out_size,
                     "# me:compiler=%s\n"
                     "# me:fp=strict\n"
                     "def kernel(x, y):\n"
                     "    return %s\n",
                     compiler_value, expr);
    }
    else {
        n = snprintf(out, out_size,
                     "# me:fp=strict\n"
                     "def kernel(x, y):\n"
                     "    return %s\n",
                     expr);
    }
    return n > 0 && (size_t)n < out_size;
}

static int run_mode(const char *source,
                    const double *in_x,
                    const double *in_y,
                    int nitems,
                    int repeats,
                    const char *jit_env_value,
                    mode_result *result,
                    double *out_values) {
    if (!source || !in_x || !in_y || nitems <= 0 || repeats <= 0 || !result) {
        return 1;
    }

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    if (jit_env_value) {
        if (setenv("ME_DSL_JIT", jit_env_value, 1) != 0) {
            free(saved_jit);
            return 1;
        }
    }
    else {
        (void)unsetenv("ME_DSL_JIT");
    }

    me_variable vars[] = {
        {"x", ME_FLOAT64},
        {"y", ME_FLOAT64}
    };
    const void *inputs[] = {in_x, in_y};

    int err = 0;
    me_expr *expr = NULL;
    uint64_t t0 = monotonic_ns();
    int rc_compile = me_compile(source, vars, 2, ME_FLOAT64, &err, &expr);
    uint64_t t1 = monotonic_ns();
    if (rc_compile != ME_COMPILE_SUCCESS || !expr) {
        fprintf(stderr, "compile failed (jit=%s err=%d rc=%d)\n",
                jit_env_value ? jit_env_value : "<unset>", err, rc_compile);
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        me_free(expr);
        return 1;
    }

    double *out = malloc((size_t)nitems * sizeof(*out));
    if (!out) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        me_free(expr);
        return 1;
    }

    uint64_t eval_ns_best = UINT64_MAX;
    for (int r = 0; r < repeats; r++) {
        uint64_t run_start = monotonic_ns();
        int rc_eval = me_eval(expr, inputs, 2, out, nitems, NULL);
        uint64_t run_end = monotonic_ns();
        if (rc_eval != ME_EVAL_SUCCESS) {
            fprintf(stderr, "eval failed (jit=%s rc=%d)\n",
                    jit_env_value ? jit_env_value : "<unset>", rc_eval);
            free(out);
            restore_env_value("ME_DSL_JIT", saved_jit);
            free(saved_jit);
            me_free(expr);
            return 1;
        }
        uint64_t run_ns = run_end - run_start;
        if (run_ns < eval_ns_best) {
            eval_ns_best = run_ns;
        }
    }

    double checksum = 0.0;
    int stride = nitems / 19;
    if (stride < 1) {
        stride = 1;
    }
    for (int i = 0; i < nitems; i += stride) {
        checksum += out[i];
    }

    result->compile_ms = (double)(t1 - t0) / 1.0e6;
    result->eval_ms_best = (double)eval_ns_best / 1.0e6;
    result->ns_per_elem_best = (double)eval_ns_best / (double)nitems;
    result->checksum = checksum;

    if (out_values) {
        memcpy(out_values, out, (size_t)nitems * sizeof(*out));
    }

    free(out);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);
    me_free(expr);
    return 0;
}

static int run_kernel(const math_kernel_def *kernel,
                      const double *in_x,
                      const double *in_y,
                      int nitems,
                      int repeats,
                      kernel_result *out_result) {
    if (!kernel || !kernel->name || !kernel->expr || !in_x || !in_y ||
        nitems <= 0 || repeats <= 0 || !out_result) {
        return 1;
    }

    char source[512];
    if (!build_dsl_source(source, sizeof(source), kernel->expr)) {
        return 1;
    }

    double *jit_out = malloc((size_t)nitems * sizeof(*jit_out));
    double *interp_out = malloc((size_t)nitems * sizeof(*interp_out));
    if (!jit_out || !interp_out) {
        free(jit_out);
        free(interp_out);
        return 1;
    }

    memset(out_result, 0, sizeof(*out_result));

    if (run_mode(source, in_x, in_y, nitems, 1, "1", &out_result->jit_cold, NULL) != 0) {
        free(jit_out);
        free(interp_out);
        return 1;
    }
    if (run_mode(source, in_x, in_y, nitems, repeats, "1", &out_result->jit_warm, jit_out) != 0) {
        free(jit_out);
        free(interp_out);
        return 1;
    }
    if (run_mode(source, in_x, in_y, nitems, repeats, "0", &out_result->interp, interp_out) != 0) {
        free(jit_out);
        free(interp_out);
        return 1;
    }

    double max_abs_diff = 0.0;
    for (int i = 0; i < nitems; i++) {
        double d = fabs(jit_out[i] - interp_out[i]);
        if (d > max_abs_diff) {
            max_abs_diff = d;
        }
    }
    out_result->max_abs_diff = max_abs_diff;

    free(jit_out);
    free(interp_out);
    return 0;
}

static void fill_inputs(double *x, double *y, int nitems) {
    if (!x || !y || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        double t = ((double)i + 0.5) / (double)nitems;
        x[i] = -0.9 + 1.8 * t;
        y[i] = 0.1 + 1.6 * t;
    }
}

int main(int argc, char **argv) {
#if defined(_WIN32) || defined(_WIN64)
    (void)argc;
    (void)argv;
    printf("benchmark_dsl_jit_math_kernels: skipped on Windows\n");
    return 0;
#else
    int nitems = 1 << 18;
    int repeats = 6;

    if (argc > 1) {
        if (!parse_positive_int(argv[1], &nitems)) {
            fprintf(stderr, "invalid nitems: %s\n", argv[1]);
            return 1;
        }
    }
    if (argc > 2) {
        if (!parse_positive_int(argv[2], &repeats)) {
            fprintf(stderr, "invalid repeats: %s\n", argv[2]);
            return 1;
        }
    }
    if (argc > 3) {
        fprintf(stderr, "usage: %s [nitems] [repeats]\n", argv[0]);
        return 1;
    }

    static const math_kernel_def kernels[] = {
        {"sin", "sin(x)"},
        {"exp", "exp(x)"},
        {"log", "log(x + 1.5)"},
        {"pow", "pow(x, y)"},
        {"fmax", "fmax(x, y)"},
        {"fmin", "fmin(x, y)"},
        {"hypot", "hypot(x, y)"},
        {"atan2", "atan2(y, x)"},
        {"sinpi", "sinpi(x)"},
        {"cospi", "cospi(x)"},
        {"black_scholes_like",
         "(x + 1.5) * (0.5 + 0.5 * erf((log((x + 1.5) / (y + 1.5)) + 0.03) / sqrt(0.2))) - "
         "(y + 1.5) * exp(-0.01) * (0.5 + 0.5 * erf((log((x + 1.5) / (y + 1.5)) - 0.02) / sqrt(0.2)))"}
    };
    const int nkernels = (int)(sizeof(kernels) / sizeof(kernels[0]));

    double *x = malloc((size_t)nitems * sizeof(*x));
    double *y = malloc((size_t)nitems * sizeof(*y));
    kernel_result *results = calloc((size_t)nkernels, sizeof(*results));
    if (!x || !y || !results) {
        fprintf(stderr, "allocation failed\n");
        free(x);
        free(y);
        free(results);
        return 1;
    }
    fill_inputs(x, y, nitems);

    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");

    char tmp_template[] = "/tmp/me_jit_math_bench_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    if (!tmp_root) {
        fprintf(stderr, "mkdtemp failed\n");
        free(saved_tmpdir);
        free(saved_pos_cache);
        free(x);
        free(y);
        free(results);
        return 1;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0 || setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        fprintf(stderr, "setenv failed\n");
        restore_env_value("TMPDIR", saved_tmpdir);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_tmpdir);
        free(saved_pos_cache);
        free(x);
        free(y);
        free(results);
        (void)rmdir(tmp_root);
        return 1;
    }

    int failed = 0;
    for (int i = 0; i < nkernels; i++) {
        if (run_kernel(&kernels[i], x, y, nitems, repeats, &results[i]) != 0) {
            fprintf(stderr, "kernel run failed: %s\n", kernels[i].name);
            failed = 1;
            break;
        }
    }

    if (!failed) {
        printf("benchmark_dsl_jit_math_kernels\n");
        printf("compiler=%s nitems=%d repeats=%d\n", current_dsl_compiler_label(), nitems, repeats);
        printf("fp_pragma=strict\n");
        printf("timing: jit-warm/interp report best single eval over repeats\n");
        printf("%-8s %12s %14s %14s %14s %14s %12s %12s\n",
               "kernel", "compile_ms", "jit_warm_ms", "interp_ms", "jit_ns_elem",
               "interp_ns_elem", "max_abs", "checksum");
        for (int i = 0; i < nkernels; i++) {
            printf("%-8s %12.3f %14.3f %14.3f %14.3f %14.3f %12.3e %12.3f\n",
                   kernels[i].name,
                   results[i].jit_cold.compile_ms,
                   results[i].jit_warm.eval_ms_best,
                   results[i].interp.eval_ms_best,
                   results[i].jit_warm.ns_per_elem_best,
                   results[i].interp.ns_per_elem_best,
                   results[i].max_abs_diff,
                   results[i].interp.checksum);
        }
    }

    char cache_dir[1024];
    cache_dir[0] = '\0';
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) < (int)sizeof(cache_dir)) {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_pos_cache);
    free(x);
    free(y);
    free(results);
    (void)rmdir(tmp_root);

    return failed;
#endif
}
