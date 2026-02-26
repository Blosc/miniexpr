/*
 * DSL global-linear-index benchmark (ND).
 *
 * Exercises a kernel that uses only _flat_idx with constant offsets:
 *   START_CONST + _flat_idx + STEP_CONST
 *
 * Compares:
 * - interp      : ME_DSL_JIT=0
 * - jit-indexvars: ME_DSL_JIT=1, ME_DSL_JIT_INDEX_VARS=1
 * - jit-gateoff : ME_DSL_JIT=1, ME_DSL_JIT_INDEX_VARS=0 (control)
 *
 * ND scenarios:
 * - seq-true-origin       : contiguous linear walk from origin
 * - seq-true-outer-offset : contiguous linear walk with non-zero outer offset
 * - seq-false-inner-offset: non-contiguous walk due to inner-dim offset
 * - seq-false-inner-tail  : non-contiguous walk with inner-dim tail padding
 *
 * Usage:
 *   ./benchmark_dsl_jit_flat_idx [target_nitems] [repeats]
 *
 * Optional:
 *   ME_BENCH_COMPILER=tcc|cc
 */

#if defined(_WIN32) || defined(_WIN64)
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

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
#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if !defined(_WIN32) && !defined(_WIN64)
#include <dirent.h>
#include <unistd.h>
#endif

#include "miniexpr.h"

#define ME_STR_HELPER(x) #x
#define ME_STR(x) ME_STR_HELPER(x)
#define START_CONST 17
#define STEP_CONST 5

typedef struct {
    const char *name;
    const char *jit;
    const char *index_vars;
    bool expect_jit;
} mode_def;

typedef struct {
    const char *name;
    int64_t shape[2];
    int32_t chunkshape[2];
    int32_t blockshape[2];
    int64_t nchunk;
    int64_t nblock;
    int padded_items;
    int valid_items;
    bool seq_expected;
} nd_case;

typedef struct {
    double compile_ms;
    double eval_ms_best;
    double ns_per_elem_best;
    double checksum;
    bool has_jit;
    double max_abs_diff_vs_interp;
} mode_result;

static uint64_t monotonic_ns(void) {
#if defined(_WIN32) || defined(_WIN64)
    static LARGE_INTEGER freq;
    static bool freq_ready = false;
    LARGE_INTEGER now;
    if (!freq_ready) {
        if (!QueryPerformanceFrequency(&freq)) {
            return 0;
        }
        freq_ready = true;
    }
    if (!QueryPerformanceCounter(&now)) {
        return 0;
    }
    return (uint64_t)((now.QuadPart * 1000000000ULL) / (uint64_t)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static int setenv_compat(const char *name, const char *value) {
#if defined(_WIN32) || defined(_WIN64)
    return _putenv_s(name, value ? value : "");
#else
    return setenv(name, value, 1);
#endif
}

static int unsetenv_compat(const char *name) {
#if defined(_WIN32) || defined(_WIN64)
    return _putenv_s(name, "");
#else
    return unsetenv(name);
#endif
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
#if defined(_WIN32) || defined(_WIN64)
    char *copy = NULL;
    size_t n = 0;
    if (_dupenv_s(&copy, &n, name) != 0 || !copy || n == 0) {
        free(copy);
        return NULL;
    }
    return copy;
#else
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
#endif
}

static void restore_env_value(const char *name, const char *value) {
    if (!name) {
        return;
    }
    if (value) {
        (void)setenv_compat(name, value);
    }
    else {
        (void)unsetenv_compat(name);
    }
}

static bool set_or_unset_env(const char *name, const char *value) {
    if (!name) {
        return false;
    }
    if (value) {
        return setenv_compat(name, value) == 0;
    }
    return unsetenv_compat(name) == 0;
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
                     "    return " ME_STR(START_CONST) " + _flat_idx + " ME_STR(STEP_CONST) "\n",
                     compiler_value);
    }
    else {
        n = snprintf(out, out_size,
                     "# me:fp=strict\n"
                     "def kernel():\n"
                     "    return " ME_STR(START_CONST) " + _flat_idx + " ME_STR(STEP_CONST) "\n");
    }
    return n > 0 && (size_t)n < out_size;
}

static int64_t ceil_div64(int64_t a, int64_t b) {
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

static bool compute_case_layout_2d(const nd_case *sc,
                                   int64_t out_base_idx[2],
                                   int64_t out_valid_len[2]) {
    if (!sc || !out_base_idx || !out_valid_len) {
        return false;
    }
    int64_t shape0 = sc->shape[0];
    int64_t shape1 = sc->shape[1];
    int64_t chunk0 = sc->chunkshape[0];
    int64_t chunk1 = sc->chunkshape[1];
    int64_t block0 = sc->blockshape[0];
    int64_t block1 = sc->blockshape[1];
    if (shape0 <= 0 || shape1 <= 0 || chunk0 <= 0 || chunk1 <= 0 || block0 <= 0 || block1 <= 0) {
        return false;
    }

    int64_t nchunks0 = ceil_div64(shape0, chunk0);
    int64_t nchunks1 = ceil_div64(shape1, chunk1);
    int64_t nblocks0 = ceil_div64(chunk0, block0);
    int64_t nblocks1 = ceil_div64(chunk1, block1);
    if (nchunks0 <= 0 || nchunks1 <= 0 || nblocks0 <= 0 || nblocks1 <= 0) {
        return false;
    }

    int64_t chunk_idx[2];
    int64_t block_idx[2];
    int64_t tmp = sc->nchunk;
    chunk_idx[1] = tmp % nchunks1;
    tmp /= nchunks1;
    chunk_idx[0] = tmp % nchunks0;
    tmp /= nchunks0;
    if (tmp != 0) {
        return false;
    }
    tmp = sc->nblock;
    block_idx[1] = tmp % nblocks1;
    tmp /= nblocks1;
    block_idx[0] = tmp % nblocks0;
    tmp /= nblocks0;
    if (tmp != 0) {
        return false;
    }

    out_base_idx[0] = chunk_idx[0] * chunk0 + block_idx[0] * block0;
    out_base_idx[1] = chunk_idx[1] * chunk1 + block_idx[1] * block1;

    int64_t chunk_start0 = chunk_idx[0] * chunk0;
    int64_t chunk_start1 = chunk_idx[1] * chunk1;
    int64_t chunk_len0 = shape0 - chunk_start0;
    int64_t chunk_len1 = shape1 - chunk_start1;
    if (chunk_len0 > chunk0) {
        chunk_len0 = chunk0;
    }
    if (chunk_len1 > chunk1) {
        chunk_len1 = chunk1;
    }

    int64_t block_start0 = block_idx[0] * block0;
    int64_t block_start1 = block_idx[1] * block1;
    if (block_start0 >= chunk_len0) {
        out_valid_len[0] = 0;
    }
    else {
        int64_t remain0 = chunk_len0 - block_start0;
        out_valid_len[0] = (remain0 < block0) ? remain0 : block0;
    }
    if (block_start1 >= chunk_len1) {
        out_valid_len[1] = 0;
    }
    else {
        int64_t remain1 = chunk_len1 - block_start1;
        out_valid_len[1] = (remain1 < block1) ? remain1 : block1;
    }
    return true;
}

static bool finalize_case(nd_case *sc) {
    if (!sc) {
        return false;
    }
    int64_t base_idx[2];
    int64_t valid_len[2];
    if (!compute_case_layout_2d(sc, base_idx, valid_len)) {
        return false;
    }
    sc->padded_items = sc->blockshape[0] * sc->blockshape[1];
    sc->valid_items = (int)(valid_len[0] * valid_len[1]);
    sc->seq_expected = (base_idx[1] == 0 && valid_len[1] == sc->shape[1]);
    return sc->padded_items > 0 && sc->valid_items >= 0;
}

static bool build_cases(int target_nitems, nd_case out_cases[4]) {
    int side = (int)ceil(sqrt((double)target_nitems));
    if (side < 4) {
        side = 4;
    }
    int inner_half = side / 2;
    if (inner_half < 1) {
        inner_half = 1;
    }

    out_cases[0].name = "seq-true-origin";
    out_cases[0].shape[0] = side;
    out_cases[0].shape[1] = side;
    out_cases[0].chunkshape[0] = side;
    out_cases[0].chunkshape[1] = side;
    out_cases[0].blockshape[0] = side;
    out_cases[0].blockshape[1] = side;
    out_cases[0].nchunk = 0;
    out_cases[0].nblock = 0;
    if (!finalize_case(&out_cases[0])) {
        return false;
    }

    out_cases[1].name = "seq-true-outer-offset";
    out_cases[1].shape[0] = 2 * side;
    out_cases[1].shape[1] = side;
    out_cases[1].chunkshape[0] = side;
    out_cases[1].chunkshape[1] = side;
    out_cases[1].blockshape[0] = side;
    out_cases[1].blockshape[1] = side;
    out_cases[1].nchunk = 1;
    out_cases[1].nblock = 0;
    if (!finalize_case(&out_cases[1])) {
        return false;
    }

    out_cases[2].name = "seq-false-inner-offset";
    out_cases[2].shape[0] = side;
    out_cases[2].shape[1] = side;
    out_cases[2].chunkshape[0] = side;
    out_cases[2].chunkshape[1] = side;
    out_cases[2].blockshape[0] = side;
    out_cases[2].blockshape[1] = inner_half;
    out_cases[2].nchunk = 0;
    out_cases[2].nblock = 1;
    if (!finalize_case(&out_cases[2])) {
        return false;
    }

    out_cases[3].name = "seq-false-inner-tail";
    out_cases[3].shape[0] = side;
    out_cases[3].shape[1] = side - 3;
    out_cases[3].chunkshape[0] = side;
    out_cases[3].chunkshape[1] = side;
    out_cases[3].blockshape[0] = side;
    out_cases[3].blockshape[1] = inner_half;
    out_cases[3].nchunk = 0;
    out_cases[3].nblock = 1;
    if (!finalize_case(&out_cases[3])) {
        return false;
    }
    return true;
}

static bool verify_expected_formula_2d(const double *out, const nd_case *sc) {
    if (!out || !sc) {
        return false;
    }
    int64_t base_idx[2];
    int64_t valid_len[2];
    if (!compute_case_layout_2d(sc, base_idx, valid_len)) {
        return false;
    }
    int64_t n0 = sc->shape[0];
    int64_t n1 = sc->shape[1];
    int b0 = sc->blockshape[0];
    int b1 = sc->blockshape[1];
    for (int i0 = 0; i0 < b0; i0++) {
        for (int i1 = 0; i1 < b1; i1++) {
            int off = i0 * b1 + i1;
            double expected = 0.0;
            if ((int64_t)i0 < valid_len[0] && (int64_t)i1 < valid_len[1]) {
                int64_t global_i0 = base_idx[0] + i0;
                int64_t global_i1 = base_idx[1] + i1;
                if (global_i0 < n0 && global_i1 < n1) {
                    int64_t global = global_i0 * n1 + global_i1;
                    expected = (double)(global + START_CONST + STEP_CONST);
                }
            }
            if (fabs(out[off] - expected) > 1e-12) {
                fprintf(stderr,
                        "formula mismatch case=%s at off=%d (%d,%d): got=%.17g expected=%.17g "
                        "base=(%" PRId64 ",%" PRId64 ") valid=(%" PRId64 ",%" PRId64 ")\n",
                        sc->name, off, i0, i1, out[off], expected,
                        base_idx[0], base_idx[1], valid_len[0], valid_len[1]);
                return false;
            }
        }
    }
    return true;
}

static bool case_seq_flag_matches(const nd_case *sc) {
    if (!sc) {
        return false;
    }
    int64_t base_idx[2];
    int64_t valid_len[2];
    if (!compute_case_layout_2d(sc, base_idx, valid_len)) {
        return false;
    }
    bool seq_actual = (base_idx[1] == 0 && valid_len[1] == sc->shape[1]);
    if (seq_actual != sc->seq_expected) {
        fprintf(stderr,
                "seq flag mismatch case=%s expected=%d actual=%d base=(%" PRId64 ",%" PRId64 ") "
                "valid=(%" PRId64 ",%" PRId64 ") shape=(%" PRId64 ",%" PRId64 ")\n",
                sc->name, sc->seq_expected ? 1 : 0, seq_actual ? 1 : 0,
                base_idx[0], base_idx[1], valid_len[0], valid_len[1], sc->shape[0], sc->shape[1]);
        return false;
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
                    int repeats,
                    const nd_case *sc,
                    mode_result *out_result,
                    double *out_values) {
    if (!mode || !source || repeats <= 0 || !sc || !out_result) {
        return 1;
    }

    char *saved_jit = dup_env_value("ME_DSL_JIT");
    char *saved_index_vars = dup_env_value("ME_DSL_JIT_INDEX_VARS");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");

    if (!set_or_unset_env("ME_DSL_JIT", mode->jit) ||
        !set_or_unset_env("ME_DSL_JIT_INDEX_VARS", mode->index_vars) ||
        !set_or_unset_env("ME_DSL_JIT_POS_CACHE", "0")) {
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_pos_cache);
        return 1;
    }

    clear_jit_cache_dir();

    int err = 0;
    me_expr *expr = NULL;
    uint64_t t0 = monotonic_ns();
    int rc_compile = me_compile_nd(source, NULL, 0, ME_FLOAT64, 2,
                                   sc->shape, sc->chunkshape, sc->blockshape, &err, &expr);
    uint64_t t1 = monotonic_ns();
    if (rc_compile != ME_COMPILE_SUCCESS || !expr) {
        fprintf(stderr, "compile_nd failed mode=%s case=%s err=%d rc=%d\n",
                mode->name, sc->name, err, rc_compile);
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_pos_cache);
        return 1;
    }

    out_result->has_jit = me_expr_has_jit_kernel(expr);
    if (mode->expect_jit != out_result->has_jit) {
        fprintf(stderr, "mode=%s case=%s expected has_jit=%d got=%d\n",
                mode->name, sc->name, mode->expect_jit ? 1 : 0, out_result->has_jit ? 1 : 0);
    }

    double *out = malloc((size_t)sc->padded_items * sizeof(*out));
    if (!out) {
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_pos_cache);
        return 1;
    }

    uint64_t best_eval_ns = UINT64_MAX;
    for (int r = 0; r < repeats; r++) {
        uint64_t rs = monotonic_ns();
        int rc_eval = me_eval_nd(expr, NULL, 0, out, sc->padded_items, sc->nchunk, sc->nblock, NULL);
        uint64_t re = monotonic_ns();
        if (rc_eval != ME_EVAL_SUCCESS) {
            fprintf(stderr, "eval_nd failed mode=%s case=%s rc=%d\n", mode->name, sc->name, rc_eval);
            free(out);
            me_free(expr);
            restore_env_value("ME_DSL_JIT", saved_jit);
            restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
            restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
            free(saved_jit);
            free(saved_index_vars);
            free(saved_pos_cache);
            return 1;
        }
        uint64_t run_ns = re - rs;
        if (run_ns < best_eval_ns) {
            best_eval_ns = run_ns;
        }
    }

    if (!verify_expected_formula_2d(out, sc)) {
        free(out);
        me_free(expr);
        restore_env_value("ME_DSL_JIT", saved_jit);
        restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
        restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
        free(saved_jit);
        free(saved_index_vars);
        free(saved_pos_cache);
        return 1;
    }

    int stride = sc->padded_items / 23;
    if (stride < 1) {
        stride = 1;
    }
    double checksum = 0.0;
    for (int i = 0; i < sc->padded_items; i += stride) {
        checksum += out[i];
    }

    out_result->compile_ms = (double)(t1 - t0) / 1.0e6;
    out_result->eval_ms_best = (double)best_eval_ns / 1.0e6;
    out_result->ns_per_elem_best = (double)best_eval_ns / (double)sc->valid_items;
    out_result->checksum = checksum;
    out_result->max_abs_diff_vs_interp = 0.0;

    if (out_values) {
        memcpy(out_values, out, (size_t)sc->padded_items * sizeof(*out));
    }

    free(out);
    me_free(expr);
    restore_env_value("ME_DSL_JIT", saved_jit);
    restore_env_value("ME_DSL_JIT_INDEX_VARS", saved_index_vars);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_jit);
    free(saved_index_vars);
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
    printf("%-13s %7s %12.3f %12.3f %13.3f %12.3f %10.3g %10s\n",
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
    int target_nitems = 1 << 20;
    int repeats = 9;
    if (argc >= 2 && !parse_positive_int(argv[1], &target_nitems)) {
        fprintf(stderr, "invalid target_nitems: %s\n", argv[1]);
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

    nd_case cases[4];
    if (!build_cases(target_nitems, cases)) {
        fprintf(stderr, "failed to build benchmark cases\n");
        return 1;
    }

    mode_def modes[] = {
        {"interp", "0", "1", false},
        {"jit-indexvars", "1", "1", true},
        {"jit-gateoff", "1", "0", false}
    };
    enum {
        NMODES = (int)(sizeof(modes) / sizeof(modes[0]))
    };
    mode_result results[NMODES];

    const char *compiler_label = current_dsl_compiler_label();
    printf("benchmark_dsl_jit_flat_idx\n");
    printf("compiler=%s target_nitems=%d repeats=%d\n", compiler_label, target_nitems, repeats);
    printf("kernel: %d + _flat_idx + %d\n", START_CONST, STEP_CONST);

    enum {
        NCASES = (int)(sizeof(cases) / sizeof(cases[0]))
    };
    for (int c = 0; c < NCASES; c++) {
        const nd_case *sc = &cases[c];
        if (!case_seq_flag_matches(sc)) {
            return 1;
        }
        memset(results, 0, sizeof(results));

        double *interp_values = malloc((size_t)sc->padded_items * sizeof(*interp_values));
        double *tmp_values = malloc((size_t)sc->padded_items * sizeof(*tmp_values));
        if (!interp_values || !tmp_values) {
            free(interp_values);
            free(tmp_values);
            return 1;
        }

        for (int i = 0; i < NMODES; i++) {
            double *store = (i == 0) ? interp_values : tmp_values;
            if (run_mode(&modes[i], source, repeats, sc, &results[i], store) != 0) {
                free(interp_values);
                free(tmp_values);
                return 1;
            }
            if (i > 0) {
                results[i].max_abs_diff_vs_interp =
                    compute_max_abs_diff(interp_values, tmp_values, sc->padded_items);
            }
        }

        printf("\n");
        printf("case=%s seq=%s shape=(%" PRId64 ",%" PRId64 ") chunk=(%d,%d) block=(%d,%d) "
               "nchunk=%" PRId64 " nblock=%" PRId64 " valid=%d padded=%d\n",
               sc->name,
               sc->seq_expected ? "true" : "false",
               sc->shape[0], sc->shape[1],
               sc->chunkshape[0], sc->chunkshape[1],
               sc->blockshape[0], sc->blockshape[1],
               sc->nchunk, sc->nblock,
               sc->valid_items, sc->padded_items);
        printf("%-13s %7s %12s %12s %13s %12s %10s %10s\n",
               "mode", "has_jit", "compile_ms", "eval_ms", "ns_per_elem", "checksum", "max_diff", "speedup");
        printf("%-13s %7s %12s %12s %13s %12s %10s %10s\n",
               "-------------", "-------", "------------", "------------",
               "-------------", "------------", "----------", "----------");

        double interp_ns_per_elem = results[0].ns_per_elem_best;
        for (int i = 0; i < NMODES; i++) {
            print_row(&modes[i], &results[i], interp_ns_per_elem);
        }

        free(interp_values);
        free(tmp_values);
    }

    printf("\n");
    printf("notes:\n");
    printf("  jit-indexvars: ME_DSL_JIT=1, ME_DSL_JIT_INDEX_VARS=1\n");
    printf("  gate-off ctrl: ME_DSL_JIT=1, ME_DSL_JIT_INDEX_VARS=0\n");
    return 0;
}
