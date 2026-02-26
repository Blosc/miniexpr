/*
 * DSL JIT benchmark for a Black-Scholes kernel close to the notebook version.
 *
 * Usage:
 *   ./benchmark_black-scholes [nitems] [repeats]
 *
 * Optional env:
 *   ME_BENCH_COMPILER=tcc|cc
 *   ME_DSL_TRACE=1
 *   ME_DSL_JIT_SCALAR_MATH_BRIDGE=0|1
 *   ME_DSL_JIT_VEC_MATH=0|1
 *   ME_DSL_JIT_HYBRID_EXPR_VEC_MATH=0|1
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
    bool has_vec_call;
    bool has_scalar_bridge_call;
    bool has_scalar_loop;
} bench_result;

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

static int count_kernel_files_with_suffix(const char *dir_path, const char *suffix,
                                          char *out_path, size_t out_path_cap) {
    if (out_path && out_path_cap > 0) {
        out_path[0] = '\0';
    }
    if (!dir_path || !suffix) {
        return 0;
    }
    DIR *dir = opendir(dir_path);
    if (!dir) {
        return 0;
    }
    int count = 0;
    struct dirent *ent = NULL;
    size_t suffix_len = strlen(suffix);
    while ((ent = readdir(dir)) != NULL) {
        size_t n = strlen(ent->d_name);
        if (n < suffix_len) {
            continue;
        }
        if (strcmp(ent->d_name + n - suffix_len, suffix) != 0) {
            continue;
        }
        count++;
        if (count == 1 && out_path && out_path_cap > 0) {
            if (snprintf(out_path, out_path_cap, "%s/%s", dir_path, ent->d_name) >= (int)out_path_cap) {
                out_path[0] = '\0';
            }
        }
    }
    closedir(dir);
    return count;
}

static bool file_contains_text(const char *path, const char *needle) {
    if (!path || !needle || needle[0] == '\0') {
        return false;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    long len = ftell(f);
    if (len < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return false;
    }
    size_t nr = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[nr] = '\0';
    bool found = strstr(buf, needle) != NULL;
    free(buf);
    return found;
}

static bool file_has_line_pattern(const char *path,
                                  const char *must_a,
                                  const char *must_b,
                                  const char *must_not) {
    if (!path || !must_a || must_a[0] == '\0') {
        return false;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    char line[2048];
    while (fgets(line, sizeof(line), f) != NULL) {
        if (!strstr(line, must_a)) {
            continue;
        }
        if (must_b && must_b[0] != '\0' && !strstr(line, must_b)) {
            continue;
        }
        if (must_not && must_not[0] != '\0' && strstr(line, must_not)) {
            continue;
        }
        fclose(f);
        return true;
    }
    fclose(f);
    return false;
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
                     "def kernel(S, X, T, R, V):\n"
                     "    A1 = 0.31938153\n"
                     "    A2 = -0.356563782\n"
                     "    A3 = 1.781477937\n"
                     "    A4 = -1.821255978\n"
                     "    A5 = 1.330274429\n"
                     "    RSQRT2PI = 0.39894228040143267793994605993438\n"
                     "    sqrtT = sqrt(T)\n"
                     "    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)\n"
                     "    d2 = d1 - V * sqrtT\n"
                     "    K = 1.0 / (1.0 + 0.2316419 * abs(d1))\n"
                     "    ret_val = (RSQRT2PI * exp(-0.5 * d1 * d1) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
                     "    if d1 > 0:\n"
                     "        cndd1 = 1.0 - ret_val\n"
                     "    else:\n"
                     "        cndd1 = ret_val\n"
                     "    K = 1.0 / (1.0 + 0.2316419 * abs(d2))\n"
                     "    ret_val = (RSQRT2PI * exp(-0.5 * d2 * d2) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
                     "    if d2 > 0:\n"
                     "        cndd2 = 1.0 - ret_val\n"
                     "    else:\n"
                     "        cndd2 = ret_val\n"
                     "    expRT = exp((-1.0 * R) * T)\n"
                     "    callResult = (S * cndd1 - X * expRT * cndd2)\n"
                     "    return callResult\n",
                     compiler_value);
    }
    else {
        n = snprintf(out, out_size,
                     "# me:fp=strict\n"
                     "def kernel(S, X, T, R, V):\n"
                     "    A1 = 0.31938153\n"
                     "    A2 = -0.356563782\n"
                     "    A3 = 1.781477937\n"
                     "    A4 = -1.821255978\n"
                     "    A5 = 1.330274429\n"
                     "    RSQRT2PI = 0.39894228040143267793994605993438\n"
                     "    sqrtT = sqrt(T)\n"
                     "    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)\n"
                     "    d2 = d1 - V * sqrtT\n"
                     "    K = 1.0 / (1.0 + 0.2316419 * abs(d1))\n"
                     "    ret_val = (RSQRT2PI * exp(-0.5 * d1 * d1) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
                     "    if d1 > 0:\n"
                     "        cndd1 = 1.0 - ret_val\n"
                     "    else:\n"
                     "        cndd1 = ret_val\n"
                     "    K = 1.0 / (1.0 + 0.2316419 * abs(d2))\n"
                     "    ret_val = (RSQRT2PI * exp(-0.5 * d2 * d2) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))\n"
                     "    if d2 > 0:\n"
                     "        cndd2 = 1.0 - ret_val\n"
                     "    else:\n"
                     "        cndd2 = ret_val\n"
                     "    expRT = exp((-1.0 * R) * T)\n"
                     "    callResult = (S * cndd1 - X * expRT * cndd2)\n"
                     "    return callResult\n");
    }
    return n > 0 && (size_t)n < out_size;
}

static void fill_inputs(double *s, double *x, double *t, double *r, double *v, int nitems) {
    if (!s || !x || !t || !r || !v || nitems <= 0) {
        return;
    }
    for (int i = 0; i < nitems; i++) {
        double f = ((double)i + 0.5) / (double)nitems;
        s[i] = 60.0 + 80.0 * f;
        t[i] = 0.05 + 2.0 * f;
        x[i] = 100.0;
        r[i] = 0.02;
        v[i] = 0.30;
    }
}

static int run_mode(const char *source,
                    const double *s,
                    const double *x,
                    const double *t,
                    const double *r,
                    const double *v,
                    int nitems,
                    int repeats,
                    const char *jit_env_value,
                    mode_result *result,
                    double *out_values) {
    if (!source || !s || !x || !t || !r || !v || nitems <= 0 ||
        repeats <= 0 || !result) {
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
        {"S", ME_FLOAT64},
        {"X", ME_FLOAT64},
        {"T", ME_FLOAT64},
        {"R", ME_FLOAT64},
        {"V", ME_FLOAT64}
    };
    const void *inputs[] = {s, x, t, r, v};

    int err = 0;
    me_expr *expr = NULL;
    uint64_t t0 = monotonic_ns();
    int rc_compile = me_compile(source, vars, 5, ME_FLOAT64, &err, &expr);
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
    for (int i = 0; i < repeats; i++) {
        uint64_t run_start = monotonic_ns();
        int rc_eval = me_eval(expr, inputs, 5, out, nitems, NULL);
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
    int stride = nitems / 23;
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

int main(int argc, char **argv) {
#if defined(_WIN32) || defined(_WIN64)
    (void)argc;
    (void)argv;
    printf("benchmark_black-scholes: skipped on Windows\n");
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

    double *s = malloc((size_t)nitems * sizeof(*s));
    double *x = malloc((size_t)nitems * sizeof(*x));
    double *t = malloc((size_t)nitems * sizeof(*t));
    double *r = malloc((size_t)nitems * sizeof(*r));
    double *v = malloc((size_t)nitems * sizeof(*v));
    double *jit_out = malloc((size_t)nitems * sizeof(*jit_out));
    double *interp_out = malloc((size_t)nitems * sizeof(*interp_out));
    if (!s || !x || !t || !r || !v || !jit_out || !interp_out) {
        fprintf(stderr, "allocation failed\n");
        free(s);
        free(x);
        free(t);
        free(r);
        free(v);
        free(jit_out);
        free(interp_out);
        return 1;
    }
    fill_inputs(s, x, t, r, v, nitems);

    char source[4096];
    if (!build_dsl_source(source, sizeof(source))) {
        fprintf(stderr, "failed to build benchmark DSL source\n");
        free(s);
        free(x);
        free(t);
        free(r);
        free(v);
        free(jit_out);
        free(interp_out);
        return 1;
    }

    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    char tmp_template[] = "/tmp/me_jit_black_scholes_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    if (!tmp_root) {
        fprintf(stderr, "mkdtemp failed\n");
        free(saved_tmpdir);
        free(saved_pos_cache);
        free(s);
        free(x);
        free(t);
        free(r);
        free(v);
        free(jit_out);
        free(interp_out);
        return 1;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0 || setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        fprintf(stderr, "setenv TMPDIR/ME_DSL_JIT_POS_CACHE failed\n");
        restore_env_value("TMPDIR", saved_tmpdir);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_tmpdir);
        free(saved_pos_cache);
        free(s);
        free(x);
        free(t);
        free(r);
        free(v);
        free(jit_out);
        free(interp_out);
        (void)rmdir(tmp_root);
        return 1;
    }

    bench_result result;
    memset(&result, 0, sizeof(result));
    if (run_mode(source, s, x, t, r, v, nitems, 1, "1", &result.jit_cold, NULL) != 0 ||
        run_mode(source, s, x, t, r, v, nitems, repeats, "1", &result.jit_warm, jit_out) != 0 ||
        run_mode(source, s, x, t, r, v, nitems, repeats, "0", &result.interp, interp_out) != 0) {
        fprintf(stderr, "benchmark runs failed\n");
        restore_env_value("TMPDIR", saved_tmpdir);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_tmpdir);
        free(saved_pos_cache);
        free(s);
        free(x);
        free(t);
        free(r);
        free(v);
        free(jit_out);
        free(interp_out);
        (void)rmdir(tmp_root);
        return 1;
    }

    for (int i = 0; i < nitems; i++) {
        double d = fabs(jit_out[i] - interp_out[i]);
        if (d > result.max_abs_diff) {
            result.max_abs_diff = d;
        }
    }

    char cache_dir[1024];
    cache_dir[0] = '\0';
    char c_path[1200];
    c_path[0] = '\0';
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) < (int)sizeof(cache_dir)) {
        if (count_kernel_files_with_suffix(cache_dir, ".c", c_path, sizeof(c_path)) > 0 && c_path[0] != '\0') {
            result.has_vec_call = file_has_line_pattern(c_path, "me_jit_vec_", "nitems);", "extern ");
            result.has_scalar_bridge_call =
                file_has_line_pattern(c_path, "me_jit_exp(", NULL, "extern ") ||
                file_has_line_pattern(c_path, "me_jit_log(", NULL, "extern ") ||
                file_has_line_pattern(c_path, "me_jit_sqrt(", NULL, "extern ") ||
                file_has_line_pattern(c_path, "me_jit_abs(", NULL, "extern ");
            result.has_scalar_loop = file_contains_text(c_path, "for (int64_t idx = 0; idx < nitems; idx++) {");
        }
    }

    const char *scalar_bridge_env = getenv("ME_DSL_JIT_SCALAR_MATH_BRIDGE");
    const char *vec_math_env = getenv("ME_DSL_JIT_VEC_MATH");
    const char *expr_vec_math_env = getenv("ME_DSL_JIT_HYBRID_EXPR_VEC_MATH");
    printf("benchmark_black-scholes\n");
    printf("compiler=%s nitems=%d repeats=%d\n", current_dsl_compiler_label(), nitems, repeats);
    printf("ME_DSL_JIT_SCALAR_MATH_BRIDGE=%s ME_DSL_JIT_VEC_MATH=%s ME_DSL_JIT_HYBRID_EXPR_VEC_MATH=%s\n",
           scalar_bridge_env ? scalar_bridge_env : "<unset>",
           vec_math_env ? vec_math_env : "<unset>",
           expr_vec_math_env ? expr_vec_math_env : "<unset>");
    printf("%-16s %12s %14s %14s %14s %14s %12s %12s\n",
           "kernel", "compile_ms", "jit_warm_ms", "interp_ms", "jit_ns_elem",
           "interp_ns_elem", "max_abs", "checksum");
    printf("%-16s %12.3f %14.3f %14.3f %14.3f %14.3f %12.3e %12.3f\n",
           "black_scholes",
           result.jit_cold.compile_ms,
           result.jit_warm.eval_ms_best,
           result.interp.eval_ms_best,
           result.jit_warm.ns_per_elem_best,
           result.interp.ns_per_elem_best,
           result.max_abs_diff,
           result.interp.checksum);
    printf("markers: vec_call=%s scalar_bridge_call=%s scalar_loop=%s\n",
           result.has_vec_call ? "yes" : "no",
           result.has_scalar_bridge_call ? "yes" : "no",
           result.has_scalar_loop ? "yes" : "no");

    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_pos_cache);
    free(s);
    free(x);
    free(t);
    free(r);
    free(v);
    free(jit_out);
    free(interp_out);
    (void)rmdir(tmp_root);
    return 0;
#endif
}
