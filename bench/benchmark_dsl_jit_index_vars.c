/*
 * DSL reserved-index-vars A/B benchmark.
 *
 * Exercises reserved symbols in one kernel:
 *   _i0, _i1, _n0, _n1, _ndim, _global_linear_idx
 *
 * Compares:
 * - interp      : ME_DSL_JIT=0
 * - jit-buffer  : ME_DSL_JIT=1, ME_DSL_JIT_INDEX_VARS=1, ME_DSL_JIT_INDEX_VARS_SYNTH=0
 * - jit-synth   : ME_DSL_JIT=1, ME_DSL_JIT_INDEX_VARS=1, ME_DSL_JIT_INDEX_VARS_SYNTH=1
 * - jit-gateoff : ME_DSL_JIT=1, ME_DSL_JIT_INDEX_VARS=0 (control)
 *
 * Usage:
 *   ./benchmark_dsl_jit_index_vars [nitems] [repeats]
 *
 * Optional:
 *   ME_BENCH_COMPILER=tcc|cc
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
    const char *jit;
    const char *index_vars;
    const char *synth;
    bool expect_jit;
} mode_def;

typedef struct {
    double compile_ms;
    double eval_ms_best;
    double ns_per_elem_best;
    double checksum;
    bool has_jit;
    double max_abs_diff_vs_interp;
} mode_result;

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

static bool set_or_unset_env(const char *name, const char *value) {
    if (!name) {
        return false;
    }
    if (value) {
        return setenv(name, value, 1) == 0;
    }
    return unsetenv(name) == 0;
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

static void clear_jit_cache_dir(void) {
    char dir[1024];
    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir && tmpdir[0] != '\0') {
        if (snprintf(dir, sizeof(dir), "%s/miniexpr-jit", tmpdir) >= (int)sizeof(dir)) {
            return;
        }
    }
    else {
        if (snprintf(dir, sizeof(dir), "/tmp/miniexpr-jit-%lu", (unsigned long)getuid()) >= (int)sizeof(dir)) {
            return;
        }
    }
    remove_files_in_dir(dir);
}
#else
static void clear_jit_cache_dir(void) {
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

static bool build_dsl_source(char *out, size_t out_size) {
    if (!out || out_size == 0) {
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
                     "def kernel():\n"
                     "    return _global_linear_idx + _i0 + _i1 + _n0 + _n1 + _ndim\n",
                     compiler_value);
    }
    else {
        n = snprintf(out, out_size,
                     "# me:fp=strict\n"
                     "def kernel():\n"
                     "    return _global_linear_idx + _i0 + _i1 + _n0 + _n1 + _ndim\n");
    }
    return n > 0 && (size_t)n < out_size;
}

static bool verify_expected_formula(const double *out, int nitems) {
    if (!out || nitems <= 0) {
        return false;
    }
    int samples[6];
    samples[0] = 0;
    samples[1] = 1;
    samples[2] = (nitems > 2) ? 2 : 1;
    samples[3] = nitems / 3;
    samples[4] = nitems / 2;
    samples[5] = nitems - 1;
    for (int i = 0; i < 6; i++) {
        int idx = samples[i];
        if (idx < 0 || idx >= nitems) {
            continue;
        }
        double expected = (double)(2 * idx + nitems + 2);
        if (fabs(out[idx] - expected) > 1e-12) {
            fprintf(stderr,
                    "formula mismatch at idx=%d: got=%.17g expected=%.17g\n",
                    idx, out[idx], expected);
            return false;
        }
    }
    return true;
}

static double compute_max_abs_diff(const double *a, const double *b, int nitems) {
    if (!a || !b || nitems <= 0) {
        return 0.0;
    }
    double max_diff = 0.0;
    for (int i = 0; i < nitems; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > max_diff) {
            max_diff = d;
        }
    }
    return max_diff;
}

static int run_mode(const mode_def *mode,
                    const char *source,
                    int nitems,
                    int repeats,
                    mode_result *out_result,
                    double *out_values) {
    if (!mode || !source || nitems <= 0 || repeats <= 0 || !out_result) {
        return 1;
    }

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    char *saved_index_vars = dup_env_value("ME_DSL_JIT_INDEX_VARS");
    char *saved_synth = dup_env_value("ME_DSL_JIT_INDEX_VARS_SYNTH");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");

    if (!set_or_unset_env("ME_DSL_JIT", mode->jit) ||
        !set_or_unset_env("ME_DSL_JIT_INDEX_VARS", mode->index_vars) ||
        !set_or_unset_env("ME_DSL_JIT_INDEX_VARS_SYNTH", mode->synth) ||
        !set_or_unset_env("ME_DSL_JIT_POS_CACHE", "0")) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_INDEX_VARS_SYNTH", saved_synth);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_synth);
        free(saved_pos_cache);
        return 1;
    }

    clear_jit_cache_dir();

    int err = 0;
    me_expr *expr = NULL;
    uint64_t t0 = monotonic_ns();
    int rc_compile = me_compile(source, NULL, 0, ME_FLOAT64, &err, &expr);
    uint64_t t1 = monotonic_ns();
    if (rc_compile != ME_COMPILE_SUCCESS || !expr) {
        fprintf(stderr, "compile failed mode=%s err=%d rc=%d\n", mode->name, err, rc_compile);
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_INDEX_VARS_SYNTH", saved_synth);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_synth);
        free(saved_pos_cache);
        return 1;
    }

    out_result->has_jit = me_expr_has_jit_kernel(expr);
    if (mode->expect_jit != out_result->has_jit) {
        fprintf(stderr, "mode=%s expected has_jit=%d got=%d\n",
                mode->name, mode->expect_jit ? 1 : 0, out_result->has_jit ? 1 : 0);
    }

    double *out = malloc((size_t)nitems * sizeof(*out));
    if (!out) {
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_INDEX_VARS_SYNTH", saved_synth);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_synth);
        free(saved_pos_cache);
        return 1;
    }

    uint64_t best_eval_ns = UINT64_MAX;
    for (int r = 0; r < repeats; r++) {
        uint64_t rs = monotonic_ns();
        int rc_eval = me_eval(expr, NULL, 0, out, nitems, NULL);
        uint64_t re = monotonic_ns();
        if (rc_eval != ME_EVAL_SUCCESS) {
            fprintf(stderr, "eval failed mode=%s rc=%d\n", mode->name, rc_eval);
            free(out);
            me_free(expr);
            restore_env_value("ME_DSL_JIT", saved_jit);
            restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
            restore_env_value("ME_DSL_JIT_INDEX_VARS_SYNTH", saved_synth);
            restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
            free(saved_jit);
            free(saved_index_vars);
            free(saved_synth);
            free(saved_pos_cache);
            return 1;
        }
        uint64_t run_ns = re - rs;
        if (run_ns < best_eval_ns) {
            best_eval_ns = run_ns;
        }
    }

    if (!verify_expected_formula(out, nitems)) {
        free(out);
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_INDEX_VARS_SYNTH", saved_synth);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_synth);
        free(saved_pos_cache);
        return 1;
    }

    int stride = nitems / 23;
    if (stride < 1) {
        stride = 1;
    }
    double checksum = 0.0;
    for (int i = 0; i < nitems; i += stride) {
        checksum += out[i];
    }

    out_result->compile_ms = (double)(t1 - t0) / 1.0e6;
    out_result->eval_ms_best = (double)best_eval_ns / 1.0e6;
    out_result->ns_per_elem_best = (double)best_eval_ns / (double)nitems;
    out_result->checksum = checksum;
    out_result->max_abs_diff_vs_interp = 0.0;

    if (out_values) {
        memcpy(out_values, out, (size_t)nitems * sizeof(*out));
    }

    free(out);
    me_free(expr);
    restore_env_value("ME_DSL_JIT", saved_jit);
    restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
    restore_env_value("ME_DSL_JIT_INDEX_VARS_SYNTH", saved_synth);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_jit);
    free(saved_index_vars);
    free(saved_synth);
    free(saved_pos_cache);
    return 0;
}

static void print_row(const mode_def *mode,
                      const mode_result *result,
                      double interp_ns_per_elem) {
    char speedup[32] = "-";
    if (interp_ns_per_elem > 0.0 && result->ns_per_elem_best > 0.0) {
        (void)snprintf(speedup, sizeof(speedup), "%.2fx", interp_ns_per_elem / result->ns_per_elem_best);
    }
    printf("%-12s %7s %12.3f %12.3f %13.3f %12.3f %10.3g %10s\n",
           mode->name,
           result->has_jit ? "yes" : "no",
           result->compile_ms,
           result->eval_ms_best,
           result->ns_per_elem_best,
           result->checksum,
           result->max_abs_diff_vs_interp,
           speedup);
}

int main(int argc, char **argv) {
    int nitems = 1 << 20;
    int repeats = 9;
    if (argc >= 2 && !parse_positive_int(argv[1], &nitems)) {
        fprintf(stderr, "invalid nitems: %s\n", argv[1]);
        return 1;
    }
    if (argc >= 3 && !parse_positive_int(argv[2], &repeats)) {
        fprintf(stderr, "invalid repeats: %s\n", argv[2]);
        return 1;
    }

    char source[1024];
    if (!build_dsl_source(source, sizeof(source))) {
        return 1;
    }

    mode_def modes[] = {
        {"interp", "0", "1", "0", false},
        {"jit-buffer", "1", "1", "0", true},
        {"jit-synth", "1", "1", "1", true},
        {"jit-gateoff", "1", "0", "1", false}
    };
    enum {
        NMODES = (int)(sizeof(modes) / sizeof(modes[0]))
    };
    mode_result results[NMODES];
    memset(results, 0, sizeof(results));

    double *interp_values = malloc((size_t)nitems * sizeof(*interp_values));
    double *tmp_values = malloc((size_t)nitems * sizeof(*tmp_values));
    if (!interp_values || !tmp_values) {
        free(interp_values);
        free(tmp_values);
        return 1;
    }

    for (int i = 0; i < NMODES; i++) {
        double *store = (i == 0) ? interp_values : tmp_values;
        if (run_mode(&modes[i], source, nitems, repeats, &results[i], store) != 0) {
            free(interp_values);
            free(tmp_values);
            return 1;
        }
        if (i > 0) {
            results[i].max_abs_diff_vs_interp = compute_max_abs_diff(interp_values, tmp_values, nitems);
        }
    }

    const char *compiler_label = current_dsl_compiler_label();
    printf("benchmark_dsl_jit_index_vars\n");
    printf("compiler=%s nitems=%d repeats=%d\n", compiler_label, nitems, repeats);
    printf("kernel: _global_linear_idx + _i0 + _i1 + _n0 + _n1 + _ndim\n");
    printf("\n");
    printf("%-12s %7s %12s %12s %13s %12s %10s %10s\n",
           "mode", "has_jit", "compile_ms", "eval_ms", "ns_per_elem", "checksum", "max_diff", "speedup");
    printf("%-12s %7s %12s %12s %13s %12s %10s %10s\n",
           "------------", "-------", "------------", "------------",
           "-------------", "------------", "----------", "----------");

    double interp_ns_per_elem = results[0].ns_per_elem_best;
    for (int i = 0; i < NMODES; i++) {
        print_row(&modes[i], &results[i], interp_ns_per_elem);
    }

    printf("\n");
    printf("A/B notes:\n");
    printf("  A=jit-buffer  (ME_DSL_JIT_INDEX_VARS_SYNTH=0)\n");
    printf("  B=jit-synth   (ME_DSL_JIT_INDEX_VARS_SYNTH=1)\n");
    printf("  gate-off ctrl (ME_DSL_JIT_INDEX_VARS=0)\n");

    free(interp_values);
    free(tmp_values);
    return 0;
}
