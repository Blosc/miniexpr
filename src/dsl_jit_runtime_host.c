/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_runtime_internal.h"
#include "dsl_jit_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)

/* In-process negative cache for recent JIT runtime failures. */
#define ME_DSL_JIT_NEG_CACHE_SLOTS 64
#define ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET 2
#define ME_DSL_JIT_NEG_CACHE_SHORT_COOLDOWN_SEC 10
#define ME_DSL_JIT_NEG_CACHE_LONG_COOLDOWN_SEC 120
#define ME_DSL_JIT_POS_CACHE_SLOTS 64

typedef enum {
    ME_DSL_JIT_NEG_FAIL_CACHE_DIR = 1,
    ME_DSL_JIT_NEG_FAIL_PATH = 2,
    ME_DSL_JIT_NEG_FAIL_WRITE = 3,
    ME_DSL_JIT_NEG_FAIL_COMPILE = 4,
    ME_DSL_JIT_NEG_FAIL_LOAD = 5,
    ME_DSL_JIT_NEG_FAIL_METADATA = 6
} me_dsl_jit_neg_failure_class;

typedef struct {
    bool valid;
    uint64_t key;
    uint64_t last_failure_at;
    uint64_t retry_after_at;
    uint8_t retries_left;
    uint8_t failure_class;
} me_dsl_jit_neg_cache_entry;

typedef struct {
    bool valid;
    uint64_t key;
    void *handle;
    me_dsl_jit_kernel_fn kernel_fn;
} me_dsl_jit_pos_cache_entry;

static me_dsl_jit_neg_cache_entry g_dsl_jit_neg_cache[ME_DSL_JIT_NEG_CACHE_SLOTS];
static int g_dsl_jit_neg_cache_cursor = 0;
static me_dsl_jit_pos_cache_entry g_dsl_jit_pos_cache[ME_DSL_JIT_POS_CACHE_SLOTS];

static uint64_t dsl_jit_now_seconds(void) {
    time_t now = time(NULL);
    if (now < 0) {
        return 0;
    }
    return (uint64_t)now;
}

static int dsl_jit_neg_cache_find_slot(uint64_t key) {
    for (int i = 0; i < ME_DSL_JIT_NEG_CACHE_SLOTS; i++) {
        if (g_dsl_jit_neg_cache[i].valid && g_dsl_jit_neg_cache[i].key == key) {
            return i;
        }
    }
    return -1;
}

static int dsl_jit_neg_cache_alloc_slot(void) {
    for (int i = 0; i < ME_DSL_JIT_NEG_CACHE_SLOTS; i++) {
        if (!g_dsl_jit_neg_cache[i].valid) {
            return i;
        }
    }
    int slot = g_dsl_jit_neg_cache_cursor;
    g_dsl_jit_neg_cache_cursor = (g_dsl_jit_neg_cache_cursor + 1) % ME_DSL_JIT_NEG_CACHE_SLOTS;
    return slot;
}

static bool dsl_jit_neg_cache_should_skip(uint64_t key) {
    int slot = dsl_jit_neg_cache_find_slot(key);
    if (slot < 0) {
        return false;
    }
    me_dsl_jit_neg_cache_entry *entry = &g_dsl_jit_neg_cache[slot];
    uint64_t now = dsl_jit_now_seconds();
    if (now < entry->retry_after_at) {
        return true;
    }
    if (entry->retries_left == 0) {
        entry->retries_left = ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET;
    }
    return false;
}

static void dsl_jit_neg_cache_record_failure(uint64_t key, me_dsl_jit_neg_failure_class failure_class) {
    int slot = dsl_jit_neg_cache_find_slot(key);
    if (slot < 0) {
        slot = dsl_jit_neg_cache_alloc_slot();
    }
    me_dsl_jit_neg_cache_entry *entry = &g_dsl_jit_neg_cache[slot];
    if (!entry->valid || entry->key != key) {
        memset(entry, 0, sizeof(*entry));
        entry->key = key;
        entry->valid = true;
        entry->retries_left = ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET;
    }
    if (entry->retries_left > 0) {
        entry->retries_left--;
    }
    uint64_t now = dsl_jit_now_seconds();
    uint64_t cooldown = (entry->retries_left == 0)
        ? ME_DSL_JIT_NEG_CACHE_LONG_COOLDOWN_SEC
        : ME_DSL_JIT_NEG_CACHE_SHORT_COOLDOWN_SEC;
    entry->last_failure_at = now;
    entry->retry_after_at = now + cooldown;
    entry->failure_class = (uint8_t)failure_class;
}

static void dsl_jit_neg_cache_clear(uint64_t key) {
    int slot = dsl_jit_neg_cache_find_slot(key);
    if (slot < 0) {
        return;
    }
    memset(&g_dsl_jit_neg_cache[slot], 0, sizeof(g_dsl_jit_neg_cache[slot]));
}

static int dsl_jit_pos_cache_find_slot(uint64_t key) {
    for (int i = 0; i < ME_DSL_JIT_POS_CACHE_SLOTS; i++) {
        if (g_dsl_jit_pos_cache[i].valid && g_dsl_jit_pos_cache[i].key == key) {
            return i;
        }
    }
    return -1;
}

static int dsl_jit_pos_cache_find_free_slot(void) {
    for (int i = 0; i < ME_DSL_JIT_POS_CACHE_SLOTS; i++) {
        if (!g_dsl_jit_pos_cache[i].valid) {
            return i;
        }
    }
    return -1;
}

static void dsl_jit_pos_cache_evict(uint64_t key) {
    int slot = dsl_jit_pos_cache_find_slot(key);
    if (slot < 0) {
        return;
    }
    if (g_dsl_jit_pos_cache[slot].handle) {
        dlclose(g_dsl_jit_pos_cache[slot].handle);
    }
    memset(&g_dsl_jit_pos_cache[slot], 0, sizeof(g_dsl_jit_pos_cache[slot]));
}

static bool dsl_jit_pos_cache_bind_program(me_dsl_compiled_program *program, uint64_t key) {
    if (!program) {
        return false;
    }
    int slot = dsl_jit_pos_cache_find_slot(key);
    if (slot < 0) {
        return false;
    }
    program->jit_dl_handle = g_dsl_jit_pos_cache[slot].handle;
    program->jit_kernel_fn = g_dsl_jit_pos_cache[slot].kernel_fn;
    program->jit_runtime_key = key;
    program->jit_dl_handle_cached = true;
    return true;
}

static bool dsl_jit_pos_cache_store_program(me_dsl_compiled_program *program, uint64_t key) {
    if (!program || !program->jit_dl_handle || !program->jit_kernel_fn) {
        return false;
    }

    int slot = dsl_jit_pos_cache_find_slot(key);
    if (slot >= 0) {
        if (program->jit_dl_handle != g_dsl_jit_pos_cache[slot].handle) {
            dlclose(program->jit_dl_handle);
            program->jit_dl_handle = g_dsl_jit_pos_cache[slot].handle;
            program->jit_kernel_fn = g_dsl_jit_pos_cache[slot].kernel_fn;
        }
        program->jit_runtime_key = key;
        program->jit_dl_handle_cached = true;
        return true;
    }

    slot = dsl_jit_pos_cache_find_free_slot();
    if (slot < 0) {
        program->jit_runtime_key = key;
        program->jit_dl_handle_cached = false;
        return false;
    }

    g_dsl_jit_pos_cache[slot].valid = true;
    g_dsl_jit_pos_cache[slot].key = key;
    g_dsl_jit_pos_cache[slot].handle = program->jit_dl_handle;
    g_dsl_jit_pos_cache[slot].kernel_fn = program->jit_kernel_fn;
    program->jit_runtime_key = key;
    program->jit_dl_handle_cached = true;
    return true;
}

void dsl_try_prepare_jit_runtime(me_dsl_compiled_program *program) {
    if (!program || !program->jit_ir || !program->jit_c_source) {
        return;
    }
    if (program->output_is_scalar) {
        dsl_tracef("jit runtime skip: fp=%s reason=scalar output",
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (program->jit_nparams != program->jit_ir->nparams) {
        dsl_tracef("jit runtime skip: fp=%s reason=parameter metadata mismatch",
                   dsl_fp_mode_name(program->fp_mode));
        return;
    }
    if (!dsl_jit_runtime_enabled()) {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime disabled by environment");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (program->compiler == ME_DSL_COMPILER_CC && program->jit_use_runtime_math_bridge) {
#if defined(__linux__)
        if (!dsl_jit_cc_promote_self_symbols_global()) {
            dsl_tracef("jit runtime note: failed to promote self symbols globally for cc bridge");
        }
#endif
    }

    uint64_t key = dsl_jit_runtime_cache_key(program);
    if (dsl_jit_pos_cache_enabled() && dsl_jit_pos_cache_bind_program(program, key)) {
        dsl_tracef("jit runtime hit: fp=%s source=process-cache key=%016llx",
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
        dsl_jit_neg_cache_clear(key);
        return;
    }
    if (dsl_jit_neg_cache_should_skip(key)) {
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime skipped after recent failure");
        dsl_tracef("jit runtime skip: fp=%s reason=%s key=%016llx",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error,
                   (unsigned long long)key);
        return;
    }

    char cache_dir[1024];
    if (!dsl_jit_get_cache_dir(cache_dir, sizeof(cache_dir))) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_CACHE_DIR);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime cache directory unavailable");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }

    const char *ext = "so";
#if defined(__APPLE__)
    ext = "dylib";
#endif
    char src_path[1300];
    char so_path[1300];
    char meta_path[1300];
    if (snprintf(src_path, sizeof(src_path), "%s/kernel_%016llx.c",
                 cache_dir, (unsigned long long)key) >= (int)sizeof(src_path) ||
        snprintf(so_path, sizeof(so_path), "%s/kernel_%016llx.%s",
                 cache_dir, (unsigned long long)key, ext) >= (int)sizeof(so_path) ||
        snprintf(meta_path, sizeof(meta_path), "%s/kernel_%016llx.meta",
                 cache_dir, (unsigned long long)key) >= (int)sizeof(meta_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_PATH);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime cache path too long");
        dsl_tracef("jit runtime skip: reason=%s", program->jit_c_error);
        return;
    }

    me_dsl_jit_cache_meta expected_meta;
    dsl_jit_fill_cache_meta(&expected_meta, program, key);

    bool so_exists = (access(so_path, F_OK) == 0);
    bool meta_matches = so_exists && dsl_jit_meta_file_matches(meta_path, &expected_meta);
    if (so_exists && !meta_matches) {
        /* Evict stale positive-cache entry so the old dlopen handle is
           closed before we overwrite the .so file on disk. */
        dsl_jit_pos_cache_evict(key);
    }
    if (meta_matches) {
        char load_detail[256];
        if (dsl_jit_load_kernel(program, so_path, load_detail, sizeof(load_detail))) {
            if (dsl_jit_pos_cache_enabled()) {
                (void)dsl_jit_pos_cache_store_program(program, key);
            }
            dsl_tracef("jit runtime hit: fp=%s source=disk-cache key=%016llx",
                       dsl_fp_mode_name(program->fp_mode),
                       (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
        dsl_tracef("jit runtime cache reload failed: fp=%s key=%016llx detail=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key,
                   load_detail[0] ? load_detail : "-");
    }

    if (program->compiler == ME_DSL_COMPILER_LIBTCC) {
        if (dsl_jit_compile_libtcc_in_memory(program)) {
            dsl_tracef("jit runtime built: fp=%s compiler=%s key=%016llx",
                       dsl_fp_mode_name(program->fp_mode),
                       dsl_compiler_name(program->compiler),
                       (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime tcc compilation failed");
        dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s detail=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler),
                   program->jit_c_error,
                   dsl_jit_libtcc_error_message());
        return;
    }

    if (!dsl_jit_write_text_file(src_path, program->jit_c_source)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_WRITE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write source");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    const char *jit_stub_path = getenv("ME_DSL_JIT_TEST_STUB_SO");
    if (jit_stub_path && jit_stub_path[0] != '\0') {
        const char *cflags = getenv("CFLAGS");
        if (cflags && strstr(cflags, ME_DSL_JIT_TEST_NEG_CACHE_FLAG)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime compilation failed");
            dsl_tracef("jit runtime skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        if (!dsl_jit_copy_file(jit_stub_path, so_path)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime stub copy failed");
            dsl_tracef("jit runtime skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        if (!dsl_jit_write_meta_file(meta_path, &expected_meta)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_METADATA);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime failed to write cache metadata");
            dsl_tracef("jit runtime skip: fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        char load_detail[256];
        if (!dsl_jit_load_kernel(program, so_path, load_detail, sizeof(load_detail))) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime shared object load failed");
            dsl_tracef("jit runtime skip: fp=%s reason=%s detail=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error,
                       load_detail[0] ? load_detail : "-");
            return;
        }
        if (dsl_jit_pos_cache_enabled()) {
            (void)dsl_jit_pos_cache_store_program(program, key);
        }
        dsl_tracef("jit runtime stubbed: fp=%s key=%016llx",
                   dsl_fp_mode_name(program->fp_mode),
                   (unsigned long long)key);
        dsl_jit_neg_cache_clear(key);
        return;
    }
    if (!dsl_jit_c_compiler_available()) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime c compiler unavailable");
        dsl_tracef("jit runtime skip: fp=%s compiler=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler),
                   program->jit_c_error);
        return;
    }
    if (!dsl_jit_compile_shared(program, src_path, so_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime compilation failed");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (!dsl_jit_write_meta_file(meta_path, &expected_meta)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_METADATA);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write cache metadata");
        dsl_tracef("jit runtime skip: fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    char load_detail[256];
    if (!dsl_jit_load_kernel(program, so_path, load_detail, sizeof(load_detail))) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime shared object load failed");
        dsl_tracef("jit runtime skip: fp=%s reason=%s detail=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error,
                   load_detail[0] ? load_detail : "-");
        return;
    }
    if (dsl_jit_pos_cache_enabled()) {
        (void)dsl_jit_pos_cache_store_program(program, key);
    }
    dsl_tracef("jit runtime built: fp=%s key=%016llx",
               dsl_fp_mode_name(program->fp_mode),
               (unsigned long long)key);
    dsl_jit_neg_cache_clear(key);
}

#endif
