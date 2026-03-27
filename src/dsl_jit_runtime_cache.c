/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_runtime_internal.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static uint64_t dsl_jit_hash_bytes(uint64_t h, const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t dsl_jit_hash_i32(uint64_t h, int v) {
    return dsl_jit_hash_bytes(h, &v, sizeof(v));
}

static uint64_t dsl_jit_hash_u64(uint64_t h, uint64_t v) {
    return dsl_jit_hash_bytes(h, &v, sizeof(v));
}

static int dsl_jit_target_tag(void) {
#if defined(__APPLE__)
    return 1;
#elif defined(__linux__)
    return 2;
#else
    return 3;
#endif
}

static int dsl_jit_backend_tag(const me_dsl_compiled_program *program) {
    if (!program) {
        return 1;
    }
    if (program->compiler == ME_DSL_COMPILER_CC) {
        return program->jit_use_runtime_math_bridge ? 3 : 2;
    }
    return 1;
}

uint64_t dsl_jit_runtime_cache_key(const me_dsl_compiled_program *program) {
    uint64_t h = 1469598103934665603ULL;
    if (!program) {
        return h;
    }
    h = dsl_jit_hash_u64(h, program->jit_ir_fingerprint);
    h = dsl_jit_hash_i32(h, (int)program->output_dtype);
    h = dsl_jit_hash_i32(h, (int)program->fp_mode);
    h = dsl_jit_hash_i32(h, program->jit_nparams);
    if (program->jit_ir) {
        for (int i = 0; i < program->jit_ir->nparams; i++) {
            h = dsl_jit_hash_i32(h, (int)program->jit_ir->param_dtypes[i]);
        }
    }
    for (int i = 0; i < program->jit_nparams; i++) {
        if (!program->jit_param_bindings) {
            h = dsl_jit_hash_i32(h, -1);
            h = dsl_jit_hash_i32(h, -1);
            h = dsl_jit_hash_i32(h, -1);
            continue;
        }
        h = dsl_jit_hash_i32(h, (int)program->jit_param_bindings[i].kind);
        h = dsl_jit_hash_i32(h, program->jit_param_bindings[i].dim);
        h = dsl_jit_hash_i32(h, program->jit_param_bindings[i].var_index);
    }
    int synth_mode = 0;
    if (program->jit_synth_reserved_non_nd) {
        synth_mode = 1;
    }
    else if (program->jit_synth_reserved_nd) {
        synth_mode = 2;
    }
    h = dsl_jit_hash_i32(h, synth_mode);
    if (program->jit_synth_reserved_nd) {
        h = dsl_jit_hash_i32(h, program->compile_ndims);
    }
    h = dsl_jit_hash_i32(h, (int)sizeof(void *));
    h = dsl_jit_hash_i32(h, ME_DSL_JIT_CGEN_VERSION);
    h = dsl_jit_hash_i32(h, ME_DSL_JIT_BRIDGE_ABI_VERSION);
    h = dsl_jit_hash_i32(h, dsl_jit_target_tag());
    h = dsl_jit_hash_i32(h, dsl_jit_backend_tag(program));
    h = dsl_jit_hash_i32(h, program->jit_use_runtime_math_bridge ? 1 : 0);
    h = dsl_jit_hash_i32(h, program->jit_scalar_math_bridge_enabled ? 1 : 0);
    h = dsl_jit_hash_i32(h, program->jit_vec_math_enabled ? 1 : 0);
    h = dsl_jit_hash_i32(h, program->jit_branch_aware_if_lowering_enabled ? 1 : 0);
    h = dsl_jit_hash_i32(h, program->jit_hybrid_expr_vec_math_enabled ? 1 : 0);
    return h;
}

static uint64_t dsl_jit_hash_cstr(uint64_t h, const char *s) {
    if (!s) {
        return dsl_jit_hash_i32(h, 0);
    }
    return dsl_jit_hash_bytes(h, s, strlen(s));
}

static uint64_t dsl_jit_toolchain_hash(const me_dsl_compiled_program *program) {
    if (!program) {
        return 1469598103934665603ULL;
    }
    if (program->compiler == ME_DSL_COMPILER_LIBTCC) {
        const char *tcc_opts = getenv("ME_DSL_JIT_TCC_OPTIONS");
        uint64_t h = dsl_jit_hash_cstr(1469598103934665603ULL, "tcc");
        return dsl_jit_hash_cstr(h, tcc_opts ? tcc_opts : "");
    }
    const char *cc = getenv("CC");
    const char *cflags = getenv("CFLAGS");
    const char *fp_cflags = dsl_jit_fp_mode_cflags(program->fp_mode);
    if (!cc || cc[0] == '\0') {
        cc = "cc";
    }
    if (!cflags) {
        cflags = "";
    }
    if (!fp_cflags) {
        fp_cflags = "";
    }
    uint64_t h = dsl_jit_hash_cstr(1469598103934665603ULL, cc);
    h = dsl_jit_hash_cstr(h, fp_cflags);
    return dsl_jit_hash_cstr(h, cflags);
}

void dsl_jit_fill_cache_meta(me_dsl_jit_cache_meta *meta,
                             const me_dsl_compiled_program *program,
                             uint64_t key) {
    if (!meta) {
        return;
    }
    memset(meta, 0, sizeof(*meta));
    meta->magic = ME_DSL_JIT_META_MAGIC;
    meta->version = ME_DSL_JIT_META_VERSION;
    meta->cgen_version = ME_DSL_JIT_CGEN_VERSION;
    meta->bridge_abi_version = ME_DSL_JIT_BRIDGE_ABI_VERSION;
    meta->target_tag = (uint32_t)dsl_jit_target_tag();
    meta->ptr_size = (uint32_t)sizeof(void *);
    meta->cache_key = key;
    if (!program) {
        return;
    }
    meta->ir_fingerprint = program->jit_ir_fingerprint;
    meta->output_dtype = (int32_t)program->output_dtype;
    meta->fp_mode = (int32_t)program->fp_mode;
    meta->compiler = (int32_t)program->compiler;
    meta->nparams = (int32_t)program->jit_nparams;
    for (int i = 0; i < ME_MAX_VARS; i++) {
        meta->param_dtypes[i] = -1;
        meta->binding_kinds[i] = -1;
        meta->binding_dims[i] = -1;
        meta->binding_var_indices[i] = -1;
    }
    if (program->jit_ir && program->jit_nparams > 0) {
        int n = program->jit_nparams;
        if (n > ME_MAX_VARS) {
            n = ME_MAX_VARS;
        }
        for (int i = 0; i < n; i++) {
            meta->param_dtypes[i] = (int32_t)program->jit_ir->param_dtypes[i];
            if (program->jit_param_bindings) {
                meta->binding_kinds[i] = (int32_t)program->jit_param_bindings[i].kind;
                meta->binding_dims[i] = (int32_t)program->jit_param_bindings[i].dim;
                meta->binding_var_indices[i] = (int32_t)program->jit_param_bindings[i].var_index;
            }
        }
    }
    meta->toolchain_hash = dsl_jit_toolchain_hash(program);
}

bool dsl_jit_write_meta_file(const char *path, const me_dsl_jit_cache_meta *meta) {
    if (!path || !meta) {
        return false;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    bool ok = (fwrite(meta, 1, sizeof(*meta), f) == sizeof(*meta));
    if (fclose(f) != 0) {
        ok = false;
    }
    return ok;
}

static bool dsl_jit_read_meta_file(const char *path, me_dsl_jit_cache_meta *out_meta) {
    if (!path || !out_meta) {
        return false;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    bool ok = (fread(out_meta, 1, sizeof(*out_meta), f) == sizeof(*out_meta));
    if (fclose(f) != 0) {
        ok = false;
    }
    return ok;
}

bool dsl_jit_meta_file_matches(const char *path, const me_dsl_jit_cache_meta *expected) {
    if (!path || !expected) {
        return false;
    }
    me_dsl_jit_cache_meta actual;
    if (!dsl_jit_read_meta_file(path, &actual)) {
        return false;
    }
    return memcmp(&actual, expected, sizeof(actual)) == 0;
}

static bool dsl_jit_ensure_dir(const char *path) {
    if (!path || !path[0]) {
        return false;
    }
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    if (mkdir(path, 0700) == 0) {
        return true;
    }
    if (errno == EEXIST) {
        return true;
    }
    return false;
}

bool dsl_jit_get_cache_dir(char *out, size_t out_size) {
    if (!out || out_size == 0) {
        return false;
    }
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || tmpdir[0] == '\0') {
        /* Avoid cross-user permission conflicts when TMPDIR is not set. */
        if (snprintf(out, out_size, "/tmp/miniexpr-jit-%lu", (unsigned long)getuid()) >= (int)out_size) {
            return false;
        }
        return dsl_jit_ensure_dir(out);
    }
    if (snprintf(out, out_size, "%s/miniexpr-jit", tmpdir) >= (int)out_size) {
        return false;
    }
    return dsl_jit_ensure_dir(out);
}

bool dsl_jit_write_text_file(const char *path, const char *text) {
    if (!path || !text) {
        return false;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    size_t n = strlen(text);
    bool ok = (fwrite(text, 1, n, f) == n);
    if (fclose(f) != 0) {
        ok = false;
    }
    return ok;
}

bool dsl_jit_copy_file(const char *src, const char *dst) {
    if (!src || !dst) {
        return false;
    }
    FILE *fin = fopen(src, "rb");
    if (!fin) {
        return false;
    }
    FILE *fout = fopen(dst, "wb");
    if (!fout) {
        fclose(fin);
        return false;
    }
    unsigned char buf[4096];
    bool ok = true;
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), fin)) > 0) {
        if (fwrite(buf, 1, n, fout) != n) {
            ok = false;
            break;
        }
    }
    if (ferror(fin)) {
        ok = false;
    }
    if (fclose(fin) != 0) {
        ok = false;
    }
    if (fclose(fout) != 0) {
        ok = false;
    }
    return ok;
}
