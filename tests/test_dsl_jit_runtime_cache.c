/*
 * Runtime JIT cache behavior tests.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <limits.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

#if !defined(_WIN32) && !defined(_WIN64)
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "../src/miniexpr.h"
#include "../src/dsl_parser.h"
#include "../src/dsl_jit_test.h"
#include "minctest.h"

#ifndef ME_DSL_JIT_TEST_STUB_SO_PATH
#define ME_DSL_JIT_TEST_STUB_SO_PATH ""
#endif

#if defined(__EMSCRIPTEN__)
EM_JS(int, test_wasm_runtime_cache_instantiate,
      (const unsigned char *wasm_bytes, int wasm_len, int bridge_lookup_fn_idx), {
    if (typeof _meJitInstantiate !== "function") {
        err("[test-runtime-cache] missing _meJitInstantiate");
        return 0;
    }
    if (typeof globalThis.__meJitInstantiateCount !== "number") {
        globalThis.__meJitInstantiateCount = 0;
    }
    globalThis.__meJitInstantiateCount = (globalThis.__meJitInstantiateCount + 1) | 0;
    var runtime = {
        HEAPF64: HEAPF64,
        HEAPF32: HEAPF32,
        wasmMemory: wasmMemory,
        wasmTable: wasmTable,
        stackSave: stackSave,
        stackAlloc: stackAlloc,
        stackRestore: stackRestore,
        lengthBytesUTF8: lengthBytesUTF8,
        stringToUTF8: stringToUTF8,
        addFunction: addFunction,
        err: err
    };
    var src = HEAPU8.subarray(wasm_bytes, wasm_bytes + wasm_len);
    return _meJitInstantiate(runtime, src, bridge_lookup_fn_idx) | 0;
});

EM_JS(void, test_wasm_runtime_cache_free, (int idx), {
    if (typeof _meJitFreeFn === "function") {
        _meJitFreeFn({ removeFunction: removeFunction }, idx);
        return;
    }
    if (idx) {
        removeFunction(idx);
    }
});

EM_JS(void, test_wasm_runtime_cache_reset_instantiate_count, (), {
    globalThis.__meJitInstantiateCount = 0;
});

EM_JS(int, test_wasm_runtime_cache_get_instantiate_count, (), {
    if (typeof globalThis.__meJitInstantiateCount !== "number") {
        return 0;
    }
    return globalThis.__meJitInstantiateCount | 0;
});
#endif

static void configure_jit_stub_env(void) {
#if !defined(_WIN32) && !defined(_WIN64)
    if (ME_DSL_JIT_TEST_STUB_SO_PATH[0] != '\0') {
        (void)setenv("ME_DSL_JIT_TEST_STUB_SO", ME_DSL_JIT_TEST_STUB_SO_PATH, 1);
    }
#endif
}

#if !defined(_WIN32) && !defined(_WIN64)
static bool has_suffix(const char *s, const char *suffix) {
    if (!s || !suffix) {
        return false;
    }
    size_t s_len = strlen(s);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > s_len) {
        return false;
    }
    return strcmp(s + (s_len - suffix_len), suffix) == 0;
}

static int count_kernel_files_with_suffix(const char *dir_path, const char *suffix,
                                          char *first_path, size_t first_path_size) {
    if (!dir_path) {
        return -1;
    }
    DIR *dir = opendir(dir_path);
    if (!dir) {
        return 0;
    }
    int count = 0;
    struct dirent *ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        if (strncmp(ent->d_name, "kernel_", 7) != 0) {
            continue;
        }
        if (!has_suffix(ent->d_name, suffix)) {
            continue;
        }
        if (count == 0 && first_path && first_path_size > 0) {
            snprintf(first_path, first_path_size, "%s/%s", dir_path, ent->d_name);
        }
        count++;
    }
    closedir(dir);
    return count;
}

static bool file_exists(const char *path) {
    return path && access(path, F_OK) == 0;
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

static int compile_and_eval_simple_dsl(const char *src, double expected_offset) {
    if (!src) {
        return 1;
    }
    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: compile error at %d\n", err);
        me_free(expr);
        return 1;
    }

    double in[4] = {0.0, 1.0, 2.0, 3.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs[] = {in};
    if (me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: eval failed\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    for (int i = 0; i < 4; i++) {
        if (out[i] != in[i] + expected_offset) {
            printf("  FAILED: eval mismatch at %d\n", i);
            return 1;
        }
    }
    return 0;
}

static int compile_and_eval_dsl_values(const char *src, const double *in, int nitems, double *out) {
    if (!src || !in || !out || nitems <= 0) {
        return 1;
    }
    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: compile error at %d\n", err);
        me_free(expr);
        return 1;
    }

    const void *inputs[] = {in};
    if (me_eval(expr, inputs, 1, out, nitems, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: eval failed\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);
    return 0;
}

static int test_negative_cache_skips_immediate_retry(void) {
    printf("\n=== DSL JIT Runtime Cache Test 1: negative cache cooldown ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_neg_cache_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char first_src_path[1200];
    first_src_path[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_cflags = dup_env_value("CFLAGS");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    y = x + 11\n"
        "    return y\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("CFLAGS", "-me_intentional_bad_flag_for_neg_cache", 1) != 0) {
        printf("  FAILED: setenv CFLAGS failed\n");
        goto cleanup;
    }
    (void)unsetenv("ME_DSL_JIT_POS_CACHE");

    if (compile_and_eval_simple_dsl(src, 11.0) != 0) {
        goto cleanup;
    }

    int n_first = count_kernel_files_with_suffix(cache_dir, ".c", first_src_path, sizeof(first_src_path));
    if (n_first != 1 || first_src_path[0] == '\0') {
        printf("  FAILED: expected one generated source file after first compile attempt (got %d)\n", n_first);
        goto cleanup;
    }
    if (remove(first_src_path) != 0) {
        printf("  FAILED: could not remove first generated source file\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src, 11.0) != 0) {
        goto cleanup;
    }

    int n_second = count_kernel_files_with_suffix(cache_dir, ".c", NULL, 0);
    if (n_second != 0) {
        printf("  FAILED: second attempt regenerated source despite negative cache (count=%d)\n", n_second);
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("CFLAGS", saved_cflags);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_cflags);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_positive_cache_reuses_loaded_kernel(void) {
    printf("\n=== DSL JIT Runtime Cache Test 2: positive cache reuse ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_pos_cache_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    y = x + 7\n"
        "    return y\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src, 7.0) != 0) {
        goto cleanup;
    }

    remove_files_in_dir(cache_dir);
    if (setenv("CC", "me_missing_cc_for_pos_cache_test", 1) != 0) {
        printf("  FAILED: setenv CC (missing compiler) failed\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src, 7.0) != 0) {
        goto cleanup;
    }

    int n_files = 0;
    n_files += count_kernel_files_with_suffix(cache_dir, ".c", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".so", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".dylib", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_files != 0) {
        printf("  FAILED: positive cache did not short-circuit runtime compile path\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int tamper_file_first_byte(const char *path) {
    if (!path) {
        return 1;
    }
    FILE *f = fopen(path, "r+b");
    if (!f) {
        return 1;
    }
    int c = fgetc(f);
    if (c == EOF) {
        fclose(f);
        return 1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 1;
    }
    unsigned char bad = (unsigned char)((unsigned char)c ^ 0x5aU);
    if (fwrite(&bad, 1, 1, f) != 1) {
        fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        return 1;
    }
    return 0;
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
    long n = ftell(f);
    if (n <= 0 || n > (8L * 1024L * 1024L)) {
        fclose(f);
        return false;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return false;
    }
    size_t nr = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[nr] = '\0';
    bool found = strstr(buf, needle) != NULL;
    free(buf);
    return found;
}

static bool file_contains_texts_in_order(const char *path, const char *const *needles, int nneedles) {
    if (!path || !needles || nneedles <= 0) {
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
    long n = ftell(f);
    if (n <= 0 || n > (8L * 1024L * 1024L)) {
        fclose(f);
        return false;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return false;
    }
    size_t nr = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[nr] = '\0';

    const char *cursor = buf;
    bool ok = true;
    for (int i = 0; i < nneedles; i++) {
        if (!needles[i] || needles[i][0] == '\0') {
            ok = false;
            break;
        }
        const char *pos = strstr(cursor, needles[i]);
        if (!pos) {
            ok = false;
            break;
        }
        cursor = pos + strlen(needles[i]);
    }
    free(buf);
    return ok;
}

static int test_rejects_metadata_mismatch_artifact(void) {
    printf("\n=== DSL JIT Runtime Cache Test 3: metadata mismatch rejection ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_meta_cache_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char meta_path[1200];
    meta_path[0] = '\0';
    char src_path[1200];
    src_path[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    y = x + 13\n"
        "    return y\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    (void)unsetenv("ME_DSL_JIT_POS_CACHE");

    if (compile_and_eval_simple_dsl(src, 13.0) != 0) {
        goto cleanup;
    }

    if (count_kernel_files_with_suffix(cache_dir, ".meta", meta_path, sizeof(meta_path)) != 1 || meta_path[0] == '\0') {
        printf("  FAILED: expected one cache metadata file\n");
        goto cleanup;
    }
    if (count_kernel_files_with_suffix(cache_dir, ".c", src_path, sizeof(src_path)) != 1 || src_path[0] == '\0') {
        printf("  FAILED: expected one generated source file\n");
        goto cleanup;
    }
    if (tamper_file_first_byte(meta_path) != 0) {
        printf("  FAILED: could not tamper metadata file\n");
        goto cleanup;
    }
    if (remove(src_path) != 0) {
        printf("  FAILED: could not remove generated source file\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src, 13.0) != 0) {
        goto cleanup;
    }
    if (!file_exists(src_path)) {
        printf("  FAILED: metadata mismatch did not force recompilation\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_jit_disable_env_guardrail(void) {
    printf("\n=== DSL JIT Runtime Cache Test 4: JIT disable guardrail ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_disable_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    y = x + 19\n"
        "    return y\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "me_missing_cc_for_disable_guardrail_test", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT failed\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src, 19.0) != 0) {
        goto cleanup;
    }

    int n_files = 0;
    n_files += count_kernel_files_with_suffix(cache_dir, ".c", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".so", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".dylib", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_files != 0) {
        printf("  FAILED: JIT disable guardrail still generated runtime cache files\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_jit);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_default_tcc_skips_cc_backend(void) {
    printf("\n=== DSL JIT Runtime Cache Test 5: default tcc backend ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_force_tcc_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src =
        "def kernel(x):\n"
        "    y = x + 29\n"
        "    return y\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src, 29.0) != 0) {
        goto cleanup;
    }

    int n_files = 0;
    n_files += count_kernel_files_with_suffix(cache_dir, ".c", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".so", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".dylib", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_files != 0) {
        printf("  FAILED: default tcc path unexpectedly used cc-backed cache path\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_unknown_me_pragma_is_rejected(void) {
    printf("\n=== DSL JIT Runtime Cache Test 6: unknown me pragma rejected ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_unknown_pragma_cache_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src_base =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    y = x + 23\n"
        "    return y\n";
    const char *src_unknown_pragma =
        "# me:compiler=cc\n"
        "# me:bogus=element\n"
        "def kernel(x):\n"
        "    y = x + 23\n"
        "    return y\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src_base, 23.0) != 0) {
        goto cleanup;
    }

    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src_unknown_pragma, vars, 1, ME_FLOAT64, &err, &expr) == ME_COMPILE_SUCCESS) {
        printf("  FAILED: unknown me pragma should be rejected\n");
        me_free(expr);
        goto cleanup;
    }

    int n_meta = count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_meta != 1) {
        printf("  FAILED: rejected unknown me pragma should not create a second cache entry (meta=%d)\n",
               n_meta);
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_cache_key_differentiates_fp_mode(void) {
    printf("\n=== DSL JIT Runtime Cache Test 7: fp mode cache key differentiation ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_fp_cache_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src_strict =
        "# me:compiler=cc\n"
        "# me:fp=strict\n"
        "def kernel(x):\n"
        "    y = x + 23\n"
        "    return y\n";
    const char *src_fast =
        "# me:compiler=cc\n"
        "# me:fp=fast\n"
        "def kernel(x):\n"
        "    y = x + 23\n"
        "    return y\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (compile_and_eval_simple_dsl(src_strict, 23.0) != 0) {
        goto cleanup;
    }
    if (compile_and_eval_simple_dsl(src_fast, 23.0) != 0) {
        goto cleanup;
    }

    int n_meta = count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_meta != 2) {
        printf("  FAILED: expected 2 cache metadata files for strict+fast fp modes (got %d)\n", n_meta);
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_reserved_index_cache_key_and_param_order(void) {
    printf("\n=== DSL JIT Runtime Cache Test 8: reserved index cache key + param ordering ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_reserved_cache_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char src_path[1200];
    src_path[0] = '\0';
    char meta_path_first[1200];
    meta_path_first[0] = '\0';
    char meta_path_second[1200];
    meta_path_second[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src_order_a =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    return x + _global_linear_idx + _i0 + _n0 + _ndim\n";
    const char *src_order_b =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    return x + _ndim + _n0 + _i0 + _global_linear_idx\n";
    const char *src_global_only =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    return x + _global_linear_idx\n";
    const char *src_i0_only =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    return x + _i0\n";
    const double in[4] = {1.0, 1.0, 1.0, 1.0};
    const double expected_reserved_mix[4] = {6.0, 8.0, 10.0, 12.0};
    const double expected_linear[4] = {1.0, 2.0, 3.0, 4.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const char *const expected_param_lines[] = {
        "const double *in_x = (const double *)inputs[0];",
        "const int64_t *in__i0 = (const int64_t *)inputs[1];",
        "const int64_t *in__n0 = (const int64_t *)inputs[2];",
        "const int64_t *in__ndim = (const int64_t *)inputs[3];",
        "const int64_t *in__global_linear_idx = (const int64_t *)inputs[4];"
    };

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (compile_and_eval_dsl_values(src_order_a, in, 4, out) != 0) {
        goto cleanup;
    }
    for (int i = 0; i < 4; i++) {
        if (out[i] != expected_reserved_mix[i]) {
            printf("  FAILED: reserved mix eval mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected_reserved_mix[i]);
            goto cleanup;
        }
    }
    if (count_kernel_files_with_suffix(cache_dir, ".c", src_path, sizeof(src_path)) != 1 || src_path[0] == '\0') {
        printf("  FAILED: expected one generated source file for reserved order test A\n");
        goto cleanup;
    }
    if (!file_contains_texts_in_order(src_path, expected_param_lines,
                                      (int)(sizeof(expected_param_lines) / sizeof(expected_param_lines[0])))) {
        printf("  FAILED: generated source param declaration order mismatch for reserved order test A\n");
        goto cleanup;
    }

    remove_files_in_dir(cache_dir);
    src_path[0] = '\0';
    if (compile_and_eval_dsl_values(src_order_b, in, 4, out) != 0) {
        goto cleanup;
    }
    for (int i = 0; i < 4; i++) {
        if (out[i] != expected_reserved_mix[i]) {
            printf("  FAILED: reserved mix eval mismatch (reordered expr) at %d (%.17g vs %.17g)\n",
                   i, out[i], expected_reserved_mix[i]);
            goto cleanup;
        }
    }
    if (count_kernel_files_with_suffix(cache_dir, ".c", src_path, sizeof(src_path)) != 1 || src_path[0] == '\0') {
        printf("  FAILED: expected one generated source file for reserved order test B\n");
        goto cleanup;
    }
    if (!file_contains_texts_in_order(src_path, expected_param_lines,
                                      (int)(sizeof(expected_param_lines) / sizeof(expected_param_lines[0])))) {
        printf("  FAILED: generated source param declaration order mismatch for reserved order test B\n");
        goto cleanup;
    }

    remove_files_in_dir(cache_dir);
    meta_path_first[0] = '\0';
    meta_path_second[0] = '\0';

    if (compile_and_eval_dsl_values(src_global_only, in, 4, out) != 0) {
        goto cleanup;
    }
    for (int i = 0; i < 4; i++) {
        if (out[i] != expected_linear[i]) {
            printf("  FAILED: global-linear eval mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected_linear[i]);
            goto cleanup;
        }
    }
    if (count_kernel_files_with_suffix(cache_dir, ".meta", meta_path_first, sizeof(meta_path_first)) != 1 ||
        meta_path_first[0] == '\0') {
        printf("  FAILED: expected one metadata file after first _global_linear_idx compile\n");
        goto cleanup;
    }

    if (compile_and_eval_dsl_values(src_global_only, in, 4, out) != 0) {
        goto cleanup;
    }
    if (count_kernel_files_with_suffix(cache_dir, ".meta", meta_path_second, sizeof(meta_path_second)) != 1 ||
        meta_path_second[0] == '\0') {
        printf("  FAILED: expected stable metadata entry after repeated _global_linear_idx compile\n");
        goto cleanup;
    }
    if (strcmp(meta_path_first, meta_path_second) != 0) {
        printf("  FAILED: repeated _global_linear_idx compile changed cache key unexpectedly\n");
        goto cleanup;
    }

    if (compile_and_eval_dsl_values(src_i0_only, in, 4, out) != 0) {
        goto cleanup;
    }
    for (int i = 0; i < 4; i++) {
        if (out[i] != expected_linear[i]) {
            printf("  FAILED: _i0 eval mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected_linear[i]);
            goto cleanup;
        }
    }
    if (count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0) != 2) {
        printf("  FAILED: reserved symbol set did not produce distinct runtime cache key entries\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_element_interpreter_jit_parity(void) {
    printf("\n=== DSL JIT Runtime Cache Test 9: element interpreter/JIT parity ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_element_parity_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(6):\n"
        "        if x > i:\n"
        "            acc = acc + i\n"
        "        else:\n"
        "            break\n"
        "    return acc\n";

    const double in[4] = {0.0, 2.0, 7.0, -1.0};
    double out_interp[4] = {0.0, 0.0, 0.0, 0.0};
    double out_jit[4] = {0.0, 0.0, 0.0, 0.0};

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=0 failed\n");
        goto cleanup;
    }
    if (compile_and_eval_dsl_values(src, in, 4, out_interp) != 0) {
        goto cleanup;
    }

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=1 failed\n");
        goto cleanup;
    }
    if (compile_and_eval_dsl_values(src, in, 4, out_jit) != 0) {
        goto cleanup;
    }

    for (int i = 0; i < 4; i++) {
        if (out_interp[i] != out_jit[i]) {
            printf("  FAILED: interpreter/JIT mismatch at %d (%.17g vs %.17g)\n",
                   i, out_interp[i], out_jit[i]);
            goto cleanup;
        }
    }

    int n_meta = count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_meta < 1) {
        printf("  FAILED: JIT parity test did not generate runtime cache metadata\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    free(saved_jit);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int run_cast_interpreter_jit_parity_for_compiler(const char *compiler_tag) {
    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_cast_parity_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src = NULL;
    const char *src_cc =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    return float(int(x)) + bool(x)\n";
    const char *src_tcc =
        "# me:compiler=tcc\n"
        "def kernel(x):\n"
        "    return float(int(x)) + bool(x)\n";
    const double in[6] = {0.0, 0.2, 1.0, 1.9, 2.0, 3.2};
    const double expected[6] = {0.0, 1.0, 2.0, 2.0, 3.0, 4.0};
    double out_interp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double out_jit[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    if (strcmp(compiler_tag, "cc") == 0) {
        src = src_cc;
    }
    else if (strcmp(compiler_tag, "tcc") == 0) {
        src = src_tcc;
    }
    else {
        printf("  FAILED: unknown compiler tag '%s'\n", compiler_tag ? compiler_tag : "(null)");
        goto cleanup;
    }

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }

    if (setenv("ME_DSL_JIT", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=0 failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }
    if (compile_and_eval_dsl_values(src, in, 6, out_interp) != 0) {
        printf("  FAILED: interpreter execution failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=1 failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }
    if (compile_and_eval_dsl_values(src, in, 6, out_jit) != 0) {
        printf("  FAILED: JIT execution failed for %s cast parity test\n", compiler_tag);
        goto cleanup;
    }

    for (int i = 0; i < 6; i++) {
        if (out_interp[i] != out_jit[i]) {
            printf("  FAILED: %s cast parity mismatch at %d (interp=%.17g jit=%.17g)\n",
                   compiler_tag, i, out_interp[i], out_jit[i]);
            goto cleanup;
        }
        if (fabs(out_jit[i] - expected[i]) > 1e-12) {
            printf("  FAILED: %s cast output mismatch at %d (got=%.17g exp=%.17g)\n",
                   compiler_tag, i, out_jit[i], expected[i]);
            goto cleanup;
        }
    }

    rc = 0;

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    free(saved_jit);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_cast_interpreter_jit_parity_compilers(void) {
    printf("\n=== DSL JIT Runtime Cache Test 8b: cast interpreter/JIT parity (cc+tcc) ===\n");

    int rc = 0;
    rc |= run_cast_interpreter_jit_parity_for_compiler("cc");
    rc |= run_cast_interpreter_jit_parity_for_compiler("tcc");
    if (rc != 0) {
        return 1;
    }

    printf("  PASSED\n");
    return 0;
}

static int test_wasm_cast_intrinsics_jit_enabled(void) {
#if !defined(__EMSCRIPTEN__)
    return 0;
#else
    printf("\n=== DSL JIT Runtime Cache Test 8c: wasm cast intrinsics runtime-JIT enabled ===\n");

    int rc = 1;
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    return float(int(x)) + bool(x)\n";
    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    double in[4] = {0.0, 0.2, 1.9, 3.2};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const double expected[4] = {0.0, 1.0, 2.0, 4.0};
    const void *inputs[] = {in};

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=1 failed\n");
        goto cleanup;
    }

    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: compile error at %d\n", err);
        me_free(expr);
        goto cleanup;
    }

    if (!me_expr_has_jit_kernel(expr)) {
        printf("  FAILED: expected wasm runtime JIT kernel for cast intrinsics\n");
        me_free(expr);
        goto cleanup;
    }

    if (me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: eval failed\n");
        me_free(expr);
        goto cleanup;
    }
    me_free(expr);

    for (int i = 0; i < 4; i++) {
        if (out[i] != expected[i]) {
            printf("  FAILED: output mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected[i]);
            goto cleanup;
        }
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);
    return rc;
#endif
}

static int test_wasm_reserved_index_vars_jit_parity(void) {
#if !defined(__EMSCRIPTEN__)
    return 0;
#else
    printf("\n=== DSL JIT Runtime Cache Test 8d: wasm reserved-index runtime-JIT parity ===\n");

    int rc = 1;
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src_1d =
        "# me:compiler=cc\n"
        "def kernel():\n"
        "    return _global_linear_idx + _i0 + _n0 + _ndim\n";
    const double expected_1d[6] = {7.0, 9.0, 11.0, 13.0, 15.0, 17.0};
    double out_1d[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    me_expr *expr = NULL;
    int err = 0;

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=1 failed\n");
        goto cleanup;
    }

    if (me_compile(src_1d, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: 1D compile error at %d\n", err);
        me_free(expr);
        goto cleanup;
    }
    if (!me_expr_has_jit_kernel(expr)) {
        printf("  FAILED: expected wasm runtime JIT kernel for 1D reserved-index kernel\n");
        me_free(expr);
        goto cleanup;
    }
    if (me_eval(expr, NULL, 0, out_1d, 6, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: 1D eval failed\n");
        me_free(expr);
        goto cleanup;
    }
    me_free(expr);
    expr = NULL;
    for (int i = 0; i < 6; i++) {
        if (out_1d[i] != expected_1d[i]) {
            printf("  FAILED: 1D output mismatch at %d (%.17g vs %.17g)\n",
                   i, out_1d[i], expected_1d[i]);
            goto cleanup;
        }
    }

    const char *src_nd =
        "# me:compiler=cc\n"
        "def kernel():\n"
        "    return _global_linear_idx + _i0 + _i1 + _n0 + _n1 + _ndim\n";
    int64_t shape[2] = {3, 5};
    int32_t chunks[2] = {2, 4};
    int32_t blocks[2] = {2, 3};
    const double expected_nd[6] = {18.0, 0.0, 0.0, 24.0, 0.0, 0.0};
    double out_nd[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    err = 0;
    if (me_compile_nd(src_nd, NULL, 0, ME_FLOAT64, 2, shape, chunks, blocks, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: ND compile error at %d\n", err);
        me_free(expr);
        goto cleanup;
    }
    if (!me_expr_has_jit_kernel(expr)) {
        printf("  FAILED: expected wasm runtime JIT kernel for ND reserved-index kernel\n");
        me_free(expr);
        goto cleanup;
    }
    if (me_eval_nd(expr, NULL, 0, out_nd, 6, 1, 0, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: ND eval failed\n");
        me_free(expr);
        goto cleanup;
    }
    me_free(expr);
    expr = NULL;
    for (int i = 0; i < 6; i++) {
        if (out_nd[i] != expected_nd[i]) {
            printf("  FAILED: ND output mismatch at %d (%.17g vs %.17g)\n",
                   i, out_nd[i], expected_nd[i]);
            goto cleanup;
        }
    }

    /* wasm32 runtime JIT has keyed in-process caching; disk cache is not used.
       This test focuses on reserved-index JIT enablement + parity. */
    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);
    return rc;
#endif
}

static int test_wasm_reserved_index_cache_key_differentiation(void) {
#if !defined(__EMSCRIPTEN__)
    return 0;
#else
    printf("\n=== DSL JIT Runtime Cache Test 8e: wasm reserved-index cache-key differentiation ===\n");

    int rc = 1;
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src_global =
        "# me:compiler=cc\n"
        "def kernel():\n"
        "    return _global_linear_idx + 123\n";
    const char *src_i0 =
        "# me:compiler=cc\n"
        "def kernel():\n"
        "    return _i0 + 123\n";
    const double expected[4] = {123.0, 124.0, 125.0, 126.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    int err = 0;
    me_expr *expr = NULL;

    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=1 failed\n");
        goto cleanup;
    }
    test_wasm_runtime_cache_reset_instantiate_count();

    if (me_compile(src_global, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: first _global_linear_idx compile error at %d\n", err);
        me_free(expr);
        goto cleanup;
    }
    if (!me_expr_has_jit_kernel(expr)) {
        printf("  FAILED: expected runtime JIT kernel for first _global_linear_idx compile\n");
        me_free(expr);
        goto cleanup;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: first _global_linear_idx eval failed\n");
        me_free(expr);
        goto cleanup;
    }
    me_free(expr);
    expr = NULL;
    for (int i = 0; i < 4; i++) {
        if (out[i] != expected[i]) {
            printf("  FAILED: first _global_linear_idx output mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected[i]);
            goto cleanup;
        }
    }
    if (test_wasm_runtime_cache_get_instantiate_count() != 1) {
        printf("  FAILED: expected 1 wasm instantiation after first _global_linear_idx compile\n");
        goto cleanup;
    }

    err = 0;
    if (me_compile(src_global, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: second _global_linear_idx compile error at %d\n", err);
        me_free(expr);
        goto cleanup;
    }
    if (!me_expr_has_jit_kernel(expr)) {
        printf("  FAILED: expected runtime JIT kernel for second _global_linear_idx compile\n");
        me_free(expr);
        goto cleanup;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: second _global_linear_idx eval failed\n");
        me_free(expr);
        goto cleanup;
    }
    me_free(expr);
    expr = NULL;
    if (test_wasm_runtime_cache_get_instantiate_count() != 1) {
        printf("  FAILED: second identical _global_linear_idx compile did not reuse wasm cache entry\n");
        goto cleanup;
    }

    err = 0;
    if (me_compile(src_i0, NULL, 0, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: _i0 compile error at %d\n", err);
        me_free(expr);
        goto cleanup;
    }
    if (!me_expr_has_jit_kernel(expr)) {
        printf("  FAILED: expected runtime JIT kernel for _i0 compile\n");
        me_free(expr);
        goto cleanup;
    }
    if (me_eval(expr, NULL, 0, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: _i0 eval failed\n");
        me_free(expr);
        goto cleanup;
    }
    me_free(expr);
    expr = NULL;
    for (int i = 0; i < 4; i++) {
        if (out[i] != expected[i]) {
            printf("  FAILED: _i0 output mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected[i]);
            goto cleanup;
        }
    }
    if (test_wasm_runtime_cache_get_instantiate_count() != 2) {
        printf("  FAILED: distinct reserved symbol set did not create a second wasm cache entry\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_jit);
    return rc;
#endif
}

static int test_missing_return_skips_runtime_jit(void) {
    printf("\n=== DSL JIT Runtime Cache Test 9: missing return skips runtime JIT ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_missing_return_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    if any(x > 0):\n"
        "        return 1\n";

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=1 failed\n");
        goto cleanup;
    }

    me_variable vars[] = {{"x", ME_FLOAT64}};
    double in[4] = {-1.0, -2.0, -3.0, -4.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs[] = {in};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: compile error at %d\n", err);
        me_free(expr);
        goto cleanup;
    }

    int eval_rc = me_eval(expr, inputs, 1, out, 4, NULL);
    me_free(expr);
    if (eval_rc != ME_EVAL_ERR_INVALID_ARG) {
        printf("  FAILED: expected runtime missing-return error, got %d\n", eval_rc);
        goto cleanup;
    }

    int n_files = 0;
    n_files += count_kernel_files_with_suffix(cache_dir, ".c", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".so", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".dylib", NULL, 0);
    n_files += count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_files != 0) {
        printf("  FAILED: missing-return kernel should not emit runtime JIT artifacts (count=%d)\n",
               n_files);
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    free(saved_jit);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_range_start_stop_step_jit_lowering(void) {
    printf("\n=== DSL JIT Runtime Cache Test 10: range(start, stop, step) JIT lowering ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_range_step_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    char *saved_jit = dup_env_value("ME_DSL_JIT");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    acc = 0\n"
        "    for i in range(1, 8, 3):\n"
        "        acc = acc + i\n"
        "    return x + acc\n";
    const double in[4] = {0.0, 1.0, 2.0, 3.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT=1 failed\n");
        goto cleanup;
    }

    if (compile_and_eval_dsl_values(src, in, 4, out) != 0) {
        goto cleanup;
    }
    for (int i = 0; i < 4; i++) {
        double expected = in[i] + 12.0;
        if (fabs(out[i] - expected) > 1e-12) {
            printf("  FAILED: output mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected);
            goto cleanup;
        }
    }

    int n_meta = count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_meta < 1) {
        printf("  FAILED: expected JIT cache metadata for range(start, stop, step) kernel\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    restore_env_value("ME_DSL_JIT", saved_jit);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    free(saved_jit);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}

static int test_cc_backend_bridge_path(void) {
    printf("\n=== DSL JIT Runtime Cache Test 11: cc backend bridge path ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_cc_bridge_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char c_path[1200];
    c_path[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src =
        "# me:compiler=cc\n"
        "def kernel(x):\n"
        "    return exp(x)\n";
    const double in[4] = {0.0, 0.5, 1.0, -1.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};

    if (!tmp_root) {
        printf("  FAILED: mkdtemp failed\n");
        goto cleanup;
    }
    if (snprintf(cache_dir, sizeof(cache_dir), "%s/miniexpr-jit", tmp_root) >= (int)sizeof(cache_dir)) {
        printf("  FAILED: cache path too long\n");
        goto cleanup;
    }
    if (setenv("TMPDIR", tmp_root, 1) != 0) {
        printf("  FAILED: setenv TMPDIR failed\n");
        goto cleanup;
    }
    if (setenv("CC", "cc", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_POS_CACHE", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_POS_CACHE failed\n");
        goto cleanup;
    }

    if (compile_and_eval_dsl_values(src, in, 4, out) != 0) {
        goto cleanup;
    }
    for (int i = 0; i < 4; i++) {
        double expected = exp(in[i]);
        if (fabs(out[i] - expected) > 1e-12) {
            printf("  FAILED: exp parity mismatch at %d (%.17g vs %.17g)\n",
                   i, out[i], expected);
            goto cleanup;
        }
    }

    int n_c = count_kernel_files_with_suffix(cache_dir, ".c", c_path, sizeof(c_path));
    if (n_c != 1 || c_path[0] == '\0' || !file_exists(c_path)) {
        printf("  FAILED: expected one generated source file for cc bridge path (got %d)\n", n_c);
        goto cleanup;
    }
    bool has_bridge = file_contains_text(c_path, "me_jit_vec_exp_f64(");
    bool has_scalar = file_contains_text(c_path, "for (int64_t idx = 0; idx < nitems; idx++) {");
    if (has_bridge) {
        printf("  NOTE: cc backend bridge lowering active\n");
    }
    else if (has_scalar) {
        printf("  NOTE: cc backend bridge unavailable in current binary; scalar fallback active\n");
    }
    else {
        printf("  FAILED: generated source had neither bridge nor scalar markers\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_pos_cache);
    if (cache_dir[0] != '\0') {
        remove_files_in_dir(cache_dir);
        (void)rmdir(cache_dir);
    }
    if (tmp_root) {
        (void)rmdir(tmp_root);
    }
    return rc;
}
#endif

int main(void) {
#if defined(_WIN32) || defined(_WIN64)
    printf("\n=== DSL JIT Runtime Cache Test 1: Windows compiler=tcc smoke ===\n");

    const char *src =
        "# me:compiler=tcc\n"
        "def kernel(x):\n"
        "    y = x + 31\n"
        "    return y\n";

    me_dsl_error parse_error;
    memset(&parse_error, 0, sizeof(parse_error));
    me_dsl_program *parsed = me_dsl_parse(src, &parse_error);
    if (!parsed) {
        printf("  FAILED: parse error at %d:%d: %s\n",
               parse_error.line, parse_error.column, parse_error.message);
        return 1;
    }
    if (parsed->compiler != ME_DSL_COMPILER_LIBTCC) {
        printf("  FAILED: compiler pragma did not select tcc backend\n");
        me_dsl_program_free(parsed);
        return 1;
    }
    me_dsl_program_free(parsed);

    me_variable vars[] = {{"x", ME_FLOAT64}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile(src, vars, 1, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAILED: compile error at %d\n", err);
        me_free(expr);
        return 1;
    }

    double in[4] = {0.0, 1.0, 2.0, 3.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};
    const void *inputs[] = {in};
    if (me_eval(expr, inputs, 1, out, 4, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAILED: eval failed\n");
        me_free(expr);
        return 1;
    }
    me_free(expr);

    for (int i = 0; i < 4; i++) {
        if (out[i] != in[i] + 31.0) {
            printf("  FAILED: eval mismatch at %d\n", i);
            return 1;
        }
    }

    printf("  PASSED\n");
    return 0;
#else
#if defined(__EMSCRIPTEN__)
    me_register_wasm_jit_helpers(test_wasm_runtime_cache_instantiate,
                                 test_wasm_runtime_cache_free);
#endif
    configure_jit_stub_env();
    int fail = 0;
#if defined(__EMSCRIPTEN__)
    fail |= test_jit_disable_env_guardrail();
    fail |= test_default_tcc_skips_cc_backend();
    fail |= test_cast_interpreter_jit_parity_compilers();
    fail |= test_wasm_cast_intrinsics_jit_enabled();
    fail |= test_wasm_reserved_index_vars_jit_parity();
    fail |= test_wasm_reserved_index_cache_key_differentiation();
    fail |= test_missing_return_skips_runtime_jit();
#else
    fail |= test_negative_cache_skips_immediate_retry();
    fail |= test_positive_cache_reuses_loaded_kernel();
    fail |= test_rejects_metadata_mismatch_artifact();
    fail |= test_jit_disable_env_guardrail();
    fail |= test_default_tcc_skips_cc_backend();
    fail |= test_unknown_me_pragma_is_rejected();
    fail |= test_cache_key_differentiates_fp_mode();
    fail |= test_reserved_index_cache_key_and_param_order();
    fail |= test_element_interpreter_jit_parity();
    fail |= test_cast_interpreter_jit_parity_compilers();
    fail |= test_missing_return_skips_runtime_jit();
    fail |= test_range_start_stop_step_jit_lowering();
    fail |= test_cc_backend_bridge_path();
#endif
#if defined(__EMSCRIPTEN__)
    me_register_wasm_jit_helpers(NULL, NULL);
#endif
    return fail;
#endif
}
