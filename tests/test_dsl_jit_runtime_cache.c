/*
 * Runtime JIT cache behavior tests.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(_WIN64)
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "../src/miniexpr.h"
#include "minctest.h"

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
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src =
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
    if (setenv("CC", "me_missing_cc_for_neg_cache_test", 1) != 0) {
        printf("  FAILED: setenv CC failed\n");
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

static int test_force_libtcc_gate_skips_cc_backend(void) {
    printf("\n=== DSL JIT Runtime Cache Test 5: force-libtcc gate ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_force_libtcc_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_force = dup_env_value("ME_DSL_JIT_FORCE_LIBTCC");
    char *saved_libtcc = dup_env_value("ME_DSL_JIT_LIBTCC");
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
    if (setenv("ME_DSL_JIT_FORCE_LIBTCC", "1", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_FORCE_LIBTCC failed\n");
        goto cleanup;
    }
    if (setenv("ME_DSL_JIT_LIBTCC", "0", 1) != 0) {
        printf("  FAILED: setenv ME_DSL_JIT_LIBTCC failed\n");
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
        printf("  FAILED: forced libtcc gate unexpectedly used cc-backed cache path\n");
        goto cleanup;
    }

    rc = 0;
    printf("  PASSED\n");

cleanup:
    restore_env_value("TMPDIR", saved_tmpdir);
    restore_env_value("CC", saved_cc);
    restore_env_value("ME_DSL_JIT_FORCE_LIBTCC", saved_force);
    restore_env_value("ME_DSL_JIT_LIBTCC", saved_libtcc);
    restore_env_value("ME_DSL_JIT_POS_CACHE", saved_pos_cache);
    free(saved_tmpdir);
    free(saved_cc);
    free(saved_force);
    free(saved_libtcc);
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

static int test_cache_key_differentiates_dialect(void) {
    printf("\n=== DSL JIT Runtime Cache Test 6: dialect cache key differentiation ===\n");

    int rc = 1;
    char tmp_template[] = "/tmp/me_jit_dialect_cache_XXXXXX";
    char *tmp_root = mkdtemp(tmp_template);
    char cache_dir[1024];
    cache_dir[0] = '\0';
    char *saved_tmpdir = dup_env_value("TMPDIR");
    char *saved_cc = dup_env_value("CC");
    char *saved_pos_cache = dup_env_value("ME_DSL_JIT_POS_CACHE");
    const char *src_vector =
        "def kernel(x):\n"
        "    y = x + 23\n"
        "    return y\n";
    const char *src_element =
        "# me:dialect=element\n"
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

    if (compile_and_eval_simple_dsl(src_vector, 23.0) != 0) {
        goto cleanup;
    }
    if (compile_and_eval_simple_dsl(src_element, 23.0) != 0) {
        goto cleanup;
    }

    int n_meta = count_kernel_files_with_suffix(cache_dir, ".meta", NULL, 0);
    if (n_meta != 2) {
        printf("  FAILED: expected 2 cache metadata files for vector+element dialects (got %d)\n", n_meta);
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
        "# me:fp=strict\n"
        "def kernel(x):\n"
        "    y = x + 23\n"
        "    return y\n";
    const char *src_fast =
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

static int test_element_interpreter_jit_parity(void) {
    printf("\n=== DSL JIT Runtime Cache Test 8: element interpreter/JIT parity ===\n");

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
        "# me:dialect=element\n"
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
#endif

int main(void) {
#if defined(_WIN32) || defined(_WIN64)
    printf("\n=== DSL JIT Runtime Cache Test: skipped on Windows ===\n");
    return 0;
#else
    int fail = 0;
    (void)setenv("ME_DSL_JIT_LIBTCC", "0", 1);
    fail |= test_negative_cache_skips_immediate_retry();
    fail |= test_positive_cache_reuses_loaded_kernel();
    fail |= test_rejects_metadata_mismatch_artifact();
    fail |= test_jit_disable_env_guardrail();
    fail |= test_force_libtcc_gate_skips_cc_backend();
    fail |= test_cache_key_differentiates_dialect();
    fail |= test_cache_key_differentiates_fp_mode();
    fail |= test_element_interpreter_jit_parity();
    return fail;
#endif
}
