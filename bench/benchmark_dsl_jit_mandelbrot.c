/*
 * DSL Mandelbrot benchmark for JIT cold/warm vs interpreted baseline.
 *
 * Uses an element-wise escape loop via `if ...: break`.
 *
 * The kernel matches the python-blosc2 notebook variant and returns
 * element-wise escape iteration counts.
 *
 * Usage:
 *   ./benchmark_dsl_jit_mandelbrot [widthxheight | width height] [repeats] [max_iter]
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>

#include "miniexpr.h"

typedef struct {
    double compile_ms;
    double eval_ms_best;
    double ns_per_elem_best;
    double checksum;
} bench_result;

typedef struct {
    bench_result jit_cold;
    bench_result jit_warm;
} kernel_result;

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
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

static bool parse_dims_arg(const char *arg, int *out_width, int *out_height) {
    if (!arg || !out_width || !out_height) {
        return false;
    }
    const char *sep = strchr(arg, 'x');
    if (!sep) {
        sep = strchr(arg, 'X');
    }
    if (!sep) {
        return false;
    }
    if (sep == arg || sep[1] == '\0') {
        return false;
    }
    size_t left_len = (size_t)(sep - arg);
    if (left_len >= 32) {
        return false;
    }
    char left[32];
    memcpy(left, arg, left_len);
    left[left_len] = '\0';
    int width = 0;
    int height = 0;
    if (!parse_positive_int(left, &width) || !parse_positive_int(sep + 1, &height)) {
        return false;
    }
    *out_width = width;
    *out_height = height;
    return true;
}

static void fill_inputs(float *cr, float *ci, int width, int height) {
    if (!cr || !ci || width <= 0 || height <= 0) {
        return;
    }
    for (int y = 0; y < height; y++) {
        float ty = (height > 1) ? ((float)y / (float)(height - 1)) : 0.0f;
        float imag = 1.1f - 2.2f * ty;
        for (int x = 0; x < width; x++) {
            float tx = (width > 1) ? ((float)x / (float)(width - 1)) : 0.0f;
            float real = -2.0f + 2.6f * tx;
            int idx = y * width + x;
            cr[idx] = real;
            ci[idx] = imag;
        }
    }
}

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

static bool valid_fp_mode(const char *fp_mode) {
    if (!fp_mode) {
        return false;
    }
    return strcmp(fp_mode, "strict") == 0 ||
           strcmp(fp_mode, "contract") == 0 ||
           strcmp(fp_mode, "fast") == 0;
}

static bool valid_compiler_mode(const char *compiler_mode) {
    if (!compiler_mode) {
        return true;
    }
    return strcmp(compiler_mode, "cc") == 0 ||
           strcmp(compiler_mode, "tcc") == 0;
}

static bool build_dsl_source(char *out, size_t out_size, int max_iter,
                             const char *fp_mode, const char *compiler_mode) {
    if (!out || out_size == 0 || max_iter <= 0) {
        return false;
    }
    if (!valid_fp_mode(fp_mode)) {
        return false;
    }
    if (!valid_compiler_mode(compiler_mode)) {
        return false;
    }
    char prefix[96];
    int n = 0;
    if (compiler_mode && compiler_mode[0] != '\0') {
        n = snprintf(prefix, sizeof(prefix),
                     "# me:fp=%s\n# me:compiler=%s\n",
                     fp_mode, compiler_mode);
    }
    else {
        n = snprintf(prefix, sizeof(prefix), "# me:fp=%s\n", fp_mode);
    }
    if (n <= 0 || (size_t)n >= sizeof(prefix)) {
        return false;
    }
    n = snprintf(out, out_size,
                 "%s"
                 "def kernel(cr, ci):\n"
                 "    zr = 0.0\n"
                 "    zi = 0.0\n"
                 "    escape_iter = %.1f\n"
                 "    for i in range(%d):\n"
                 "        if zr * zr + zi * zi > 4.0:\n"
                 "            escape_iter = i + 0.0\n"
                 "            break\n"
                 "        zr_new = zr * zr - zi * zi + cr\n"
                 "        zi = 2.0 * zr * zi + ci\n"
                 "        zr = zr_new\n"
                 "    return escape_iter\n",
                 prefix,
                 (double)max_iter,
                 max_iter);
    return n > 0 && (size_t)n < out_size;
}

static void print_mode_result_line(const char *compiler_label,
                                   const char *mode, const bench_result *result,
                                   const char *speedup_label) {
    if (!compiler_label || !mode || !result || !speedup_label) {
        return;
    }
    printf("%-10s %-18s %12.3f %14.3f %12.3f %12.3f %14s\n",
           compiler_label, mode,
           result->compile_ms, result->eval_ms_best,
           result->ns_per_elem_best, result->checksum, speedup_label);
}

static void format_speedup_vs_baseline(char *out, size_t out_size,
                                       double baseline_ns_per_elem,
                                       const bench_result *result) {
    if (!out || out_size == 0 || !result ||
        baseline_ns_per_elem <= 0.0 || result->ns_per_elem_best <= 0.0) {
        return;
    }
    (void)snprintf(out, out_size, "%.1fx",
                   baseline_ns_per_elem / result->ns_per_elem_best);
}

static int run_mode(const char *mode_name, const char *jit_env_value,
                    const char *source,
                    const float *cr, const float *ci, int nitems, int repeats,
                    bench_result *result, bool *jit_kernel_ready) {
    if (!mode_name || !source || !cr || !ci || nitems <= 0 || repeats <= 0 || !result) {
        return 1;
    }
    if (jit_kernel_ready) {
        *jit_kernel_ready = false;
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
        {"cr", ME_FLOAT32},
        {"ci", ME_FLOAT32}
    };
    const void *inputs[] = {cr, ci};

    int err = 0;
    me_expr *expr = NULL;
    uint64_t t0 = monotonic_ns();
    int rc_compile = me_compile(source, vars, 2, ME_FLOAT32, &err, &expr);
    uint64_t t1 = monotonic_ns();
    if (rc_compile != ME_COMPILE_SUCCESS || !expr) {
        fprintf(stderr, "compile failed for mode %s (err=%d, rc=%d)\n", mode_name, err, rc_compile);
        restore_env_value("ME_DSL_JIT", saved_jit);
        free(saved_jit);
        me_free(expr);
        return 1;
    }
    if (jit_kernel_ready) {
        *jit_kernel_ready = me_expr_has_jit_kernel(expr);
    }

    float *out = malloc((size_t)nitems * sizeof(*out));
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
            fprintf(stderr, "eval failed for mode %s (rc=%d)\n", mode_name, rc_eval);
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
    int stride = nitems / 17;
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

    free(out);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);
    me_free(expr);
    return 0;
}

static int run_kernel(const char *compiler_label, const char *source,
                      const float *cr, const float *ci, int nitems, int repeats,
                      double baseline_ns_per_elem, kernel_result *result) {
    if (!compiler_label || !source || !cr || !ci ||
        nitems <= 0 || repeats <= 0 || baseline_ns_per_elem <= 0.0 || !result) {
        return 1;
    }

    memset(result, 0, sizeof(*result));
    bool cold_has_jit = false;
    bool warm_has_jit = false;
    char speedup[32];

    if (run_mode("jit-cold", "1", source, cr, ci, nitems, 1,
                 &result->jit_cold, &cold_has_jit) != 0) {
        return 1;
    }
    const char *cold_mode = cold_has_jit ? "jit-cold" : "jit-fallback-cold";
    snprintf(speedup, sizeof(speedup), "-");
    format_speedup_vs_baseline(speedup, sizeof(speedup), baseline_ns_per_elem,
                               &result->jit_cold);
    print_mode_result_line(compiler_label, cold_mode, &result->jit_cold, speedup);
    if (run_mode("jit-warm", "1", source, cr, ci, nitems, repeats,
                 &result->jit_warm, &warm_has_jit) != 0) {
        return 1;
    }
    const char *warm_mode = warm_has_jit ? "jit-warm" : "jit-fallback-warm";
    snprintf(speedup, sizeof(speedup), "-");
    format_speedup_vs_baseline(speedup, sizeof(speedup), baseline_ns_per_elem,
                               &result->jit_warm);
    print_mode_result_line(compiler_label, warm_mode, &result->jit_warm, speedup);
    return 0;
}

int main(int argc, char **argv) {
    int width = 1200;
    int height = 800;
    int repeats = 3;
    int max_iter = 200;
    int argi = 1;
    const char *fp_mode = getenv("ME_BENCH_FP_MODE");
    if (!fp_mode || fp_mode[0] == '\0') {
        fp_mode = "strict";
    }
    if (!valid_fp_mode(fp_mode)) {
        fprintf(stderr, "invalid ME_BENCH_FP_MODE=%s (expected strict|contract|fast)\n", fp_mode);
        return 1;
    }

    if (argc > 1) {
        if (strchr(argv[1], 'x') || strchr(argv[1], 'X')) {
            if (!parse_dims_arg(argv[1], &width, &height)) {
                fprintf(stderr, "invalid size arg: %s (use widthxheight)\n", argv[1]);
                return 1;
            }
            argi = 2;
        }
        else {
            if (argc < 3 || !parse_positive_int(argv[1], &width) || !parse_positive_int(argv[2], &height)) {
                fprintf(stderr, "invalid size args: expected width height or widthxheight\n");
                return 1;
            }
            argi = 3;
        }
    }
    if (argc > argi) {
        repeats = atoi(argv[argi]);
    }
    if (argc > argi + 1) {
        max_iter = atoi(argv[argi + 1]);
    }
    if (argc > argi + 2) {
        fprintf(stderr, "too many args\n");
        return 1;
    }
    int64_t nitems64 = (int64_t)width * (int64_t)height;
    if (width <= 0 || height <= 0 || nitems64 <= 0 || nitems64 > INT_MAX ||
        repeats <= 0 || max_iter <= 0) {
        fprintf(stderr,
                "invalid args: width=%d height=%d repeats=%d max_iter=%d "
                "(size via widthxheight or width height; max_iter > 0)\n",
                width, height, repeats, max_iter);
        return 1;
    }
    int nitems = (int)nitems64;

    float *cr = malloc((size_t)nitems * sizeof(*cr));
    float *ci = malloc((size_t)nitems * sizeof(*ci));
    if (!cr || !ci) {
        fprintf(stderr, "allocation failed\n");
        free(cr);
        free(ci);
        return 1;
    }
    fill_inputs(cr, ci, width, height);

    char *saved_tmpdir = dup_env_value("TMPDIR");
    char tmp_template[] = "/tmp/me_jit_bench_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    if (!tmp_root) {
        fprintf(stderr, "mkdtemp failed\n");
        free(cr);
        free(ci);
        free(saved_tmpdir);
        return 1;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        fprintf(stderr, "cache path too long\n");
        free(cr);
        free(ci);
        free(saved_tmpdir);
        (void)rmdir(tmp_root);
        return 1;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        fprintf(stderr, "setenv TMPDIR failed\n");
        free(cr);
        free(ci);
        free(saved_tmpdir);
        (void)rmdir(tmp_root);
        return 1;
    }

    kernel_result element_tcc_res;
    kernel_result element_cc_strict_res;
    kernel_result element_cc_fast_res;
    bench_result interpreted_baseline;
    char element_tcc_source[1024];
    char element_cc_strict_source[1024];
    char element_cc_fast_source[1024];

    if (!build_dsl_source(element_tcc_source, sizeof(element_tcc_source),
                          max_iter, fp_mode, "tcc") ||
        !build_dsl_source(element_cc_strict_source, sizeof(element_cc_strict_source),
                          max_iter, "strict", "cc") ||
        !build_dsl_source(element_cc_fast_source, sizeof(element_cc_fast_source),
                          max_iter, "fast", "cc")) {
        fprintf(stderr, "failed to build benchmark DSL source\n");
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }

    printf("width=%d height=%d repeats=%d max_iter=%d\n",
           width, height, repeats, max_iter);
    printf("%-10s %-18s %12s %14s %12s %12s %14s\n",
           "compiler", "mode", "compile_ms", "eval_ms_best", "ns_per_elem", "checksum", "speedup");

    if (run_mode("interpreted DSL", "0", element_tcc_source,
                 cr, ci, nitems, 1, &interpreted_baseline, NULL) != 0) {
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }
    print_mode_result_line("n/a", "interpreted DSL", &interpreted_baseline, "baseline");

    if (run_kernel("tcc", element_tcc_source,
                   cr, ci, nitems, repeats,
                   interpreted_baseline.ns_per_elem_best, &element_tcc_res) != 0) {
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }

    printf("fp_pragma=strict\n");
    if (run_kernel("cc", element_cc_strict_source,
                   cr, ci, nitems, repeats,
                   interpreted_baseline.ns_per_elem_best, &element_cc_strict_res) != 0) {
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }

    printf("fp_pragma=fast\n");
    if (run_kernel("cc", element_cc_fast_source,
                   cr, ci, nitems, repeats,
                   interpreted_baseline.ns_per_elem_best, &element_cc_fast_res) != 0) {
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }

    restore_env_value("TMPDIR", saved_tmpdir);
    free(saved_tmpdir);
    free(cr);
    free(ci);
    remove_files_in_dir(cache_dir);
    (void)rmdir(cache_dir);
    (void)rmdir(tmp_root);
    return 0;
}
