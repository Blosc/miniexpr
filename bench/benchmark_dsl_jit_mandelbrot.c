/*
 * DSL Mandelbrot benchmark for JIT cold/warm vs interpreter fallback.
 *
 * Uses element dialect with per-item escape (`if ...: break`) to model
 * Mandelbrot early-exit behavior directly.
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
    double eval_ms_total;
    double ns_per_elem;
    double checksum;
} bench_result;

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

static void fill_inputs(double *cr, double *ci, int width, int height) {
    if (!cr || !ci || width <= 0 || height <= 0) {
        return;
    }
    for (int y = 0; y < height; y++) {
        double ty = (height > 1) ? ((double)y / (double)(height - 1)) : 0.0;
        double imag = 1.5 - 3.0 * ty;
        for (int x = 0; x < width; x++) {
            double tx = (width > 1) ? ((double)x / (double)(width - 1)) : 0.0;
            double real = -2.2 + 3.2 * tx;
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

static bool build_dsl_source(char *out, size_t out_size, int max_iter,
                             bool element_dialect, bool use_any_break) {
    if (!out || out_size == 0 || max_iter <= 0) {
        return false;
    }
    int n = 0;
    const char *prefix = element_dialect ? "# me:dialect=element\n" : "";
    if (use_any_break) {
        n = snprintf(out, out_size,
                     "%s"
                     "def kernel(cr, ci):\n"
                     "    zr = 0.0\n"
                     "    zi = 0.0\n"
                     "    acc = 0.0\n"
                     "    for i in range(%d):\n"
                     "        zr2 = 0.5 * (zr * zr - zi * zi + cr)\n"
                     "        zi = 0.5 * (2.0 * zr * zi + ci)\n"
                     "        zr = zr2\n"
                     "        acc = acc + zr\n"
                     "        if any(zr * zr + zi * zi > 4.0):\n"
                     "            break\n"
                     "    return acc\n",
                     prefix,
                     max_iter);
    }
    else {
        n = snprintf(out, out_size,
                     "%s"
                     "def kernel(cr, ci):\n"
                     "    zr = 0.0\n"
                     "    zi = 0.0\n"
                     "    acc = 0.0\n"
                     "    for i in range(%d):\n"
                     "        zr2 = 0.5 * (zr * zr - zi * zi + cr)\n"
                     "        zi = 0.5 * (2.0 * zr * zi + ci)\n"
                     "        zr = zr2\n"
                     "        acc = acc + zr\n"
                     "        if zr * zr + zi * zi > 4.0:\n"
                     "            break\n"
                     "    return acc\n",
                     prefix,
                     max_iter);
    }
    return n > 0 && (size_t)n < out_size;
}

static int run_mode(const char *mode_name, const char *jit_env_value,
                    const char *source,
                    const double *cr, const double *ci, int nitems, int repeats,
                    bench_result *result) {
    if (!mode_name || !source || !cr || !ci || nitems <= 0 || repeats <= 0 || !result) {
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
        {"cr", ME_FLOAT64},
        {"ci", ME_FLOAT64}
    };
    const void *inputs[] = {cr, ci};

    int err = 0;
    me_expr *expr = NULL;
    uint64_t t0 = monotonic_ns();
    int rc_compile = me_compile(source, vars, 2, ME_FLOAT64, &err, &expr);
    uint64_t t1 = monotonic_ns();
    if (rc_compile != ME_COMPILE_SUCCESS || !expr) {
        fprintf(stderr, "compile failed for mode %s (err=%d, rc=%d)\n", mode_name, err, rc_compile);
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

    uint64_t eval_start = monotonic_ns();
    for (int r = 0; r < repeats; r++) {
        int rc_eval = me_eval(expr, inputs, 2, out, nitems, NULL);
        if (rc_eval != ME_EVAL_SUCCESS) {
            fprintf(stderr, "eval failed for mode %s (rc=%d)\n", mode_name, rc_eval);
            free(out);
            restore_env_value("ME_DSL_JIT", saved_jit);
            free(saved_jit);
            me_free(expr);
            return 1;
        }
    }
    uint64_t eval_end = monotonic_ns();

    double checksum = 0.0;
    int stride = nitems / 17;
    if (stride < 1) {
        stride = 1;
    }
    for (int i = 0; i < nitems; i += stride) {
        checksum += out[i];
    }

    result->compile_ms = (double)(t1 - t0) / 1.0e6;
    result->eval_ms_total = (double)(eval_end - eval_start) / 1.0e6;
    result->ns_per_elem = (double)(eval_end - eval_start) / (double)((uint64_t)nitems * (uint64_t)repeats);
    result->checksum = checksum;

    free(out);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);
    me_free(expr);
    return 0;
}

int main(int argc, char **argv) {
    int width = 1200;
    int height = 800;
    int repeats = 6;
    int max_iter = 200;
    int argi = 1;

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

    double *cr = malloc((size_t)nitems * sizeof(*cr));
    double *ci = malloc((size_t)nitems * sizeof(*ci));
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

    bench_result jit_cold;
    bench_result jit_warm;
    bench_result interp;
    char jit_source[1024];
    char interp_source[1024];
    memset(&jit_cold, 0, sizeof(jit_cold));
    memset(&jit_warm, 0, sizeof(jit_warm));
    memset(&interp, 0, sizeof(interp));
    if (!build_dsl_source(jit_source, sizeof(jit_source), max_iter, true, false) ||
        !build_dsl_source(interp_source, sizeof(interp_source), max_iter, true, false)) {
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

    if (run_mode("jit-cold", "1", jit_source, cr, ci, nitems, 1, &jit_cold) != 0) {
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }
    if (run_mode("jit-warm", "1", jit_source, cr, ci, nitems, repeats, &jit_warm) != 0) {
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }
    if (run_mode("interp", "0", interp_source, cr, ci, nitems, repeats, &interp) != 0) {
        restore_env_value("TMPDIR", saved_tmpdir);
        free(saved_tmpdir);
        free(cr);
        free(ci);
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
        (void)rmdir(tmp_root);
        return 1;
    }

    printf("benchmark_dsl_jit_mandelbrot\n");
    printf("width=%d height=%d repeats=%d max_iter=%d\n",
           width, height, repeats, max_iter);
    printf("kernel=element-dialect-per-item-break\n");
    printf("%-12s %12s %14s %12s %12s\n",
           "mode", "compile_ms", "eval_ms_total", "ns_per_elem", "checksum");
    printf("%-12s %12.3f %14.3f %12.3f %12.3f\n",
           "jit-cold", jit_cold.compile_ms, jit_cold.eval_ms_total, jit_cold.ns_per_elem, jit_cold.checksum);
    printf("%-12s %12.3f %14.3f %12.3f %12.3f\n",
           "jit-warm", jit_warm.compile_ms, jit_warm.eval_ms_total, jit_warm.ns_per_elem, jit_warm.checksum);
    printf("%-12s %12.3f %14.3f %12.3f %12.3f\n",
           "interp", interp.compile_ms, interp.eval_ms_total, interp.ns_per_elem, interp.checksum);

    restore_env_value("TMPDIR", saved_tmpdir);
    free(saved_tmpdir);
    free(cr);
    free(ci);
    remove_files_in_dir(cache_dir);
    (void)rmdir(cache_dir);
    (void)rmdir(tmp_root);
    return 0;
}
