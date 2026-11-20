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

#include <pthread.h>

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
    bool retired;
    bool is_tcc;
    uint64_t key;
    uint64_t cache_id;
    void *handle;
    me_dsl_jit_kernel_fn kernel_fn;
    unsigned int ref_count;
} me_dsl_jit_pos_cache_entry;

static void dsl_jit_pos_cache_close_handle(void *handle, bool is_tcc) {
    if (!handle) {
        return;
    }
#if ME_USE_LIBTCC_FALLBACK
    if (is_tcc) {
        dsl_jit_libtcc_delete_state(handle);
        return;
    }
#else
    (void)is_tcc;
#endif
    dlclose(handle);
}

static pthread_mutex_t g_dsl_jit_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_dsl_jit_tmp_seq = 0;
static uint64_t g_dsl_jit_pos_cache_seq = 0;
#define ME_JIT_CACHE_LOCK() pthread_mutex_lock(&g_dsl_jit_cache_lock)
#define ME_JIT_CACHE_UNLOCK() pthread_mutex_unlock(&g_dsl_jit_cache_lock)

static me_dsl_jit_neg_cache_entry g_dsl_jit_neg_cache[ME_DSL_JIT_NEG_CACHE_SLOTS];
static int g_dsl_jit_neg_cache_cursor = 0;
static me_dsl_jit_pos_cache_entry g_dsl_jit_pos_cache[ME_DSL_JIT_POS_CACHE_SLOTS];

static uint64_t dsl_jit_next_tmp_seq(void) {
    ME_JIT_CACHE_LOCK();
    uint64_t seq = ++g_dsl_jit_tmp_seq;
    ME_JIT_CACHE_UNLOCK();
    return seq;
}

static uint64_t dsl_jit_now_seconds(void) {
    time_t now = time(NULL);
    if (now < 0) {
        return 0;
    }
    return (uint64_t)now;
}

static int dsl_jit_neg_cache_find_slot_locked(uint64_t key) {
    for (int i = 0; i < ME_DSL_JIT_NEG_CACHE_SLOTS; i++) {
        if (g_dsl_jit_neg_cache[i].valid && g_dsl_jit_neg_cache[i].key == key) {
            return i;
        }
    }
    return -1;
}

static int dsl_jit_neg_cache_alloc_slot_locked(void) {
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
    ME_JIT_CACHE_LOCK();
    int slot = dsl_jit_neg_cache_find_slot_locked(key);
    if (slot < 0) {
        ME_JIT_CACHE_UNLOCK();
        return false;
    }
    me_dsl_jit_neg_cache_entry *entry = &g_dsl_jit_neg_cache[slot];
    uint64_t now = dsl_jit_now_seconds();
    if (now < entry->retry_after_at) {
        ME_JIT_CACHE_UNLOCK();
        return true;
    }
    if (entry->retries_left == 0) {
        entry->retries_left = ME_DSL_JIT_NEG_CACHE_RETRY_BUDGET;
    }
    ME_JIT_CACHE_UNLOCK();
    return false;
}

static void dsl_jit_neg_cache_record_failure(uint64_t key, me_dsl_jit_neg_failure_class failure_class) {
    ME_JIT_CACHE_LOCK();
    int slot = dsl_jit_neg_cache_find_slot_locked(key);
    if (slot < 0) {
        slot = dsl_jit_neg_cache_alloc_slot_locked();
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
    ME_JIT_CACHE_UNLOCK();
}

static void dsl_jit_neg_cache_clear(uint64_t key) {
    ME_JIT_CACHE_LOCK();
    int slot = dsl_jit_neg_cache_find_slot_locked(key);
    if (slot >= 0) {
        memset(&g_dsl_jit_neg_cache[slot], 0, sizeof(g_dsl_jit_neg_cache[slot]));
    }
    ME_JIT_CACHE_UNLOCK();
}

static int dsl_jit_pos_cache_find_slot_locked(uint64_t key) {
    for (int i = 0; i < ME_DSL_JIT_POS_CACHE_SLOTS; i++) {
        if (g_dsl_jit_pos_cache[i].valid && g_dsl_jit_pos_cache[i].key == key) {
            return i;
        }
    }
    return -1;
}

static int dsl_jit_pos_cache_find_free_slot_locked(void) {
    for (int i = 0; i < ME_DSL_JIT_POS_CACHE_SLOTS; i++) {
        if (!g_dsl_jit_pos_cache[i].valid && !g_dsl_jit_pos_cache[i].retired &&
            g_dsl_jit_pos_cache[i].handle == NULL) {
            return i;
        }
    }
    return -1;
}

static void dsl_jit_pos_cache_evict(uint64_t key) {
    void *to_close = NULL;
    bool to_close_is_tcc = false;
    ME_JIT_CACHE_LOCK();
    int slot = dsl_jit_pos_cache_find_slot_locked(key);
    if (slot >= 0) {
        if (g_dsl_jit_pos_cache[slot].ref_count == 0) {
            to_close = g_dsl_jit_pos_cache[slot].handle;
            to_close_is_tcc = g_dsl_jit_pos_cache[slot].is_tcc;
            memset(&g_dsl_jit_pos_cache[slot], 0, sizeof(g_dsl_jit_pos_cache[slot]));
        }
        else {
            /* Keep the handle loaded while existing programs still execute it. */
            g_dsl_jit_pos_cache[slot].valid = false;
            g_dsl_jit_pos_cache[slot].retired = true;
        }
    }
    ME_JIT_CACHE_UNLOCK();
    if (to_close) {
        dsl_jit_pos_cache_close_handle(to_close, to_close_is_tcc);
    }
}

static bool dsl_jit_pos_cache_bind_program(me_dsl_compiled_program *program, uint64_t key) {
    if (!program) {
        return false;
    }
    bool want_tcc = (program->compiler == ME_DSL_COMPILER_LIBTCC);
    ME_JIT_CACHE_LOCK();
    int slot = dsl_jit_pos_cache_find_slot_locked(key);
    if (slot < 0 || g_dsl_jit_pos_cache[slot].is_tcc != want_tcc) {
        ME_JIT_CACHE_UNLOCK();
        return false;
    }
    if (want_tcc) {
        program->jit_tcc_state = g_dsl_jit_pos_cache[slot].handle;
        program->jit_dl_handle = NULL;
    }
    else {
        program->jit_dl_handle = g_dsl_jit_pos_cache[slot].handle;
    }
    program->jit_kernel_fn = g_dsl_jit_pos_cache[slot].kernel_fn;
    program->jit_runtime_key = key;
    program->jit_pos_cache_id = g_dsl_jit_pos_cache[slot].cache_id;
    program->jit_dl_handle_cached = true;
    g_dsl_jit_pos_cache[slot].ref_count++;
    ME_JIT_CACHE_UNLOCK();
    return true;
}

static bool dsl_jit_pos_cache_store_program(me_dsl_compiled_program *program, uint64_t key) {
    bool is_tcc = (program && program->jit_tcc_state != NULL);
    void *handle = is_tcc ? program->jit_tcc_state : (program ? program->jit_dl_handle : NULL);
    if (!program || !handle || !program->jit_kernel_fn) {
        return false;
    }

    void *dup_handle = NULL;
    ME_JIT_CACHE_LOCK();
    int slot = dsl_jit_pos_cache_find_slot_locked(key);
    if (slot >= 0) {
        if (g_dsl_jit_pos_cache[slot].is_tcc != is_tcc) {
            ME_JIT_CACHE_UNLOCK();
            return false;
        }
        dup_handle = handle;
        if (is_tcc) {
            program->jit_tcc_state = g_dsl_jit_pos_cache[slot].handle;
            program->jit_dl_handle = NULL;
        }
        else {
            program->jit_dl_handle = g_dsl_jit_pos_cache[slot].handle;
        }
        program->jit_kernel_fn = g_dsl_jit_pos_cache[slot].kernel_fn;
        program->jit_runtime_key = key;
        program->jit_pos_cache_id = g_dsl_jit_pos_cache[slot].cache_id;
        program->jit_dl_handle_cached = true;
        g_dsl_jit_pos_cache[slot].ref_count++;
        ME_JIT_CACHE_UNLOCK();
        dsl_jit_pos_cache_close_handle(dup_handle, is_tcc);
        return true;
    }

    slot = dsl_jit_pos_cache_find_free_slot_locked();
    if (slot < 0) {
        program->jit_runtime_key = key;
        program->jit_pos_cache_id = 0;
        program->jit_dl_handle_cached = false;
        ME_JIT_CACHE_UNLOCK();
        return false;
    }

    uint64_t cache_id = ++g_dsl_jit_pos_cache_seq;
    g_dsl_jit_pos_cache[slot].valid = true;
    g_dsl_jit_pos_cache[slot].retired = false;
    g_dsl_jit_pos_cache[slot].is_tcc = is_tcc;
    g_dsl_jit_pos_cache[slot].key = key;
    g_dsl_jit_pos_cache[slot].cache_id = cache_id;
    g_dsl_jit_pos_cache[slot].handle = handle;
    g_dsl_jit_pos_cache[slot].kernel_fn = program->jit_kernel_fn;
    g_dsl_jit_pos_cache[slot].ref_count = 1;
    program->jit_runtime_key = key;
    program->jit_pos_cache_id = cache_id;
    program->jit_dl_handle_cached = true;
    ME_JIT_CACHE_UNLOCK();
    return true;
}

void dsl_jit_pos_cache_release_program(me_dsl_compiled_program *program) {
    if (!program || !program->jit_dl_handle_cached) {
        return;
    }
    void *handle = program->jit_tcc_state ? program->jit_tcc_state : program->jit_dl_handle;
    if (!handle) {
        return;
    }
    void *to_close = NULL;
    bool to_close_is_tcc = false;
    ME_JIT_CACHE_LOCK();
    for (int i = 0; i < ME_DSL_JIT_POS_CACHE_SLOTS; i++) {
        me_dsl_jit_pos_cache_entry *entry = &g_dsl_jit_pos_cache[i];
        if ((entry->valid || entry->retired) &&
            entry->cache_id == program->jit_pos_cache_id &&
            entry->handle == handle) {
            if (entry->ref_count > 0) {
                entry->ref_count--;
            }
            if (entry->retired && entry->ref_count == 0) {
                to_close = entry->handle;
                to_close_is_tcc = entry->is_tcc;
                memset(entry, 0, sizeof(*entry));
            }
            break;
        }
    }
    ME_JIT_CACHE_UNLOCK();
    if (to_close) {
        dsl_jit_pos_cache_close_handle(to_close, to_close_is_tcc);
    }
}

static void dsl_jit_publish_and_load(me_dsl_compiled_program *program,
                                     uint64_t key,
                                     const me_dsl_jit_cache_meta *expected_meta,
                                     const char *tmp_so_path,
                                     const char *so_path,
                                     const char *tmp_meta_path,
                                     const char *meta_path,
                                     const char *trace_label) {
    if (rename(tmp_so_path, so_path) != 0) {
        (void)unlink(tmp_so_path);
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to publish shared object");
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    if (!dsl_jit_write_meta_file(tmp_meta_path, expected_meta) ||
        rename(tmp_meta_path, meta_path) != 0) {
        (void)unlink(tmp_meta_path);
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_METADATA);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write cache metadata");
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    char load_detail[256];
    if (!dsl_jit_load_kernel(program, so_path, load_detail, sizeof(load_detail))) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_LOAD);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime shared object load failed");
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s detail=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error,
                   load_detail[0] ? load_detail : "-");
        return;
    }
    if (dsl_jit_pos_cache_enabled()) {
        (void)dsl_jit_pos_cache_store_program(program, key);
    }
    dsl_tracef("jit runtime %s: fp=%s key=%016llx",
               trace_label,
               dsl_fp_mode_name(program->fp_mode),
               (unsigned long long)key);
    dsl_jit_neg_cache_clear(key);
}

void dsl_try_prepare_jit_runtime(me_dsl_compiled_program *program) {
    if (program && dsl_menudet_typed_profile(program->semantic_profile) &&
        program->semantic_profile != ME_DSL_PROFILE_MENUDET_0_1) return;
    if (!program || !program->jit_ir || !program->jit_c_source) {
        return;
    }
    /* Menudet source preparation audits its supported scalar reductions. The
     * legacy/full lowerer still has an exclusively elementwise kernel contract. */
    if (program->output_is_scalar && program->semantic_profile != ME_DSL_PROFILE_MENUDET_0_1) {
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
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
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
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s key=%016llx",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error,
                   (unsigned long long)key);
        return;
    }

    /* TCC kernels are owned by their compiled program, never by the disk cache.
       An unusable TMPDIR must not prevent in-memory JIT execution. */
    if (program->compiler == ME_DSL_COMPILER_LIBTCC) {
        if (dsl_jit_compile_libtcc_in_memory(program)) {
            if (dsl_jit_pos_cache_enabled()) {
                (void)dsl_jit_pos_cache_store_program(program, key);
            }
            dsl_tracef("jit runtime built: fp=%s compiler=tcc key=%016llx",
                       dsl_fp_mode_name(program->fp_mode), (unsigned long long)key);
            dsl_jit_neg_cache_clear(key);
            return;
        }
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        dsl_tracef("jit runtime fallback: interpreter fp=%s compiler=tcc reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }

    char cache_dir[1024];
    if (!dsl_jit_get_cache_dir(cache_dir, sizeof(cache_dir))) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_CACHE_DIR);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime cache directory unavailable");
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
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
    char tmp_src_path[1400];
    char tmp_so_path[1400];
    char tmp_meta_path[1400];
    unsigned long pid = (unsigned long)getpid();
    unsigned long long seq = (unsigned long long)dsl_jit_next_tmp_seq();
    int n_src = snprintf(src_path, sizeof(src_path), "%s/kernel_%016llx.c",
                         cache_dir, (unsigned long long)key);
    int n_so = snprintf(so_path, sizeof(so_path), "%s/kernel_%016llx.%s",
                        cache_dir, (unsigned long long)key, ext);
    int n_meta = snprintf(meta_path, sizeof(meta_path), "%s/kernel_%016llx.meta",
                          cache_dir, (unsigned long long)key);
    int n_tmp_src = snprintf(tmp_src_path, sizeof(tmp_src_path), "%s/.tmp_%016llx_%lu_%llu.c",
                             cache_dir, (unsigned long long)key, pid, seq);
    int n_tmp_so = snprintf(tmp_so_path, sizeof(tmp_so_path), "%s/.tmp_%016llx_%lu_%llu.%s",
                            cache_dir, (unsigned long long)key, pid, seq, ext);
    int n_tmp_meta = snprintf(tmp_meta_path, sizeof(tmp_meta_path), "%s/.tmp_%016llx_%lu_%llu.meta",
                              cache_dir, (unsigned long long)key, pid, seq);
    if (n_src < 0 || n_src >= (int)sizeof(src_path) ||
        n_so < 0 || n_so >= (int)sizeof(so_path) ||
        n_meta < 0 || n_meta >= (int)sizeof(meta_path) ||
        n_tmp_src < 0 || n_tmp_src >= (int)sizeof(tmp_src_path) ||
        n_tmp_so < 0 || n_tmp_so >= (int)sizeof(tmp_so_path) ||
        n_tmp_meta < 0 || n_tmp_meta >= (int)sizeof(tmp_meta_path)) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_PATH);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime cache path too long");
        dsl_tracef("jit runtime fallback: interpreter reason=%s", program->jit_c_error);
        return;
    }

    me_dsl_jit_cache_meta expected_meta;
    dsl_jit_fill_cache_meta(&expected_meta, program, key);

    bool so_exists = (access(so_path, F_OK) == 0);
    bool meta_exists = (access(meta_path, F_OK) == 0);
    bool meta_matches = so_exists && meta_exists && dsl_jit_meta_file_matches(meta_path, &expected_meta);
    if (so_exists && meta_exists && !meta_matches) {
        /* Evict stale positive-cache entry so the old handle is retired or
           closed before we replace the .so file on disk. */
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

    if (!dsl_jit_write_text_file(tmp_src_path, program->jit_c_source) ||
        rename(tmp_src_path, src_path) != 0) {
        (void)unlink(tmp_src_path);
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_WRITE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime failed to write source");
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    const char *jit_stub_path = getenv("ME_DSL_JIT_TEST_STUB_SO");
    if (jit_stub_path && jit_stub_path[0] != '\0') {
        const char *cflags = me_jit_option_value("CFLAGS");
        if (cflags && strstr(cflags, ME_DSL_JIT_TEST_NEG_CACHE_FLAG)) {
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime compilation failed");
            dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        if (!dsl_jit_copy_file(jit_stub_path, tmp_so_path)) {
            (void)unlink(tmp_so_path);
            dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
            snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                     "jit runtime stub copy failed");
            dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
                       dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
            return;
        }
        dsl_jit_publish_and_load(program, key, &expected_meta,
                                 tmp_so_path, so_path, tmp_meta_path, meta_path, "stubbed");
        return;
    }
    if (!dsl_jit_c_compiler_available()) {
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime c compiler unavailable");
        dsl_tracef("jit runtime fallback: interpreter fp=%s compiler=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode),
                   dsl_compiler_name(program->compiler),
                   program->jit_c_error);
        return;
    }
    if (!dsl_jit_compile_shared(program, src_path, tmp_so_path)) {
        (void)unlink(tmp_so_path);
        dsl_jit_neg_cache_record_failure(key, ME_DSL_JIT_NEG_FAIL_COMPILE);
        snprintf(program->jit_c_error, sizeof(program->jit_c_error), "%s",
                 "jit runtime compilation failed");
        dsl_tracef("jit runtime fallback: interpreter fp=%s reason=%s",
                   dsl_fp_mode_name(program->fp_mode), program->jit_c_error);
        return;
    }
    dsl_jit_publish_and_load(program, key, &expected_meta,
                             tmp_so_path, so_path, tmp_meta_path, meta_path, "built");
}

#endif
