/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "dsl_jit_bridge_contract.h"
#include "dsl_jit_runtime_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
#define ME_DSL_JIT_BRIDGE_NAME_ENTRY(pub_sym, bridge_fn, sig_type, decl) #pub_sym,
static const char *const dsl_jit_bridge_symbol_names[] = {
    ME_DSL_JIT_BRIDGE_SYMBOL_CONTRACT(ME_DSL_JIT_BRIDGE_NAME_ENTRY)
};
#undef ME_DSL_JIT_BRIDGE_NAME_ENTRY
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
static bool dsl_jit_extract_command_name(const char *cmd, char *out, size_t out_size) {
    if (!cmd || !out || out_size == 0) {
        return false;
    }
    const char *p = cmd;
    while (*p && isspace((unsigned char)*p)) {
        p++;
    }
    if (*p == '\0') {
        return false;
    }
    char quote = '\0';
    if (*p == '"' || *p == '\'') {
        quote = *p++;
    }
    size_t n = 0;
    while (*p) {
        if (quote) {
            if (*p == quote) {
                break;
            }
        }
        else if (isspace((unsigned char)*p)) {
            break;
        }
        if (n + 1 >= out_size) {
            return false;
        }
        out[n++] = *p++;
    }
    out[n] = '\0';
    return n > 0;
}

static bool dsl_jit_command_exists(const char *cmd) {
    char tool[512];
    if (!dsl_jit_extract_command_name(cmd, tool, sizeof(tool))) {
        return false;
    }
    if (strchr(tool, '/')) {
        return access(tool, X_OK) == 0;
    }
    const char *path = getenv("PATH");
    if (!path || path[0] == '\0') {
        return false;
    }
    char candidate[1024];
    const char *seg = path;
    while (seg && seg[0] != '\0') {
        const char *next = strchr(seg, ':');
        size_t seg_len = next ? (size_t)(next - seg) : strlen(seg);
        if (seg_len == 0) {
            seg = next ? next + 1 : NULL;
            continue;
        }
        if (seg_len + 1 + strlen(tool) + 1 < sizeof(candidate)) {
            memcpy(candidate, seg, seg_len);
            candidate[seg_len] = '/';
            strcpy(candidate + seg_len + 1, tool);
            if (access(candidate, X_OK) == 0) {
                return true;
            }
        }
        seg = next ? next + 1 : NULL;
    }
    return false;
}

bool dsl_jit_c_compiler_available(void) {
    const char *cc = getenv("CC");
    if (!cc || cc[0] == '\0') {
        cc = "cc";
    }
    return dsl_jit_command_exists(cc);
}
#else
bool dsl_jit_c_compiler_available(void) {
    return false;
}
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)

static bool dsl_jit_module_path_from_symbol(const void *symbol, char *out, size_t out_size) {
    if (!symbol || !out || out_size == 0) {
        return false;
    }
    Dl_info info;
    if (dladdr(symbol, &info) == 0 || !info.dli_fname || info.dli_fname[0] == '\0') {
        return false;
    }
    int n = snprintf(out, out_size, "%s", info.dli_fname);
    return n > 0 && (size_t)n < out_size;
}

bool dsl_jit_cc_math_bridge_available(void) {
    for (size_t i = 0; i < sizeof(dsl_jit_bridge_symbol_names) / sizeof(dsl_jit_bridge_symbol_names[0]); i++) {
        if (dlsym(RTLD_DEFAULT, dsl_jit_bridge_symbol_names[i]) == NULL) {
            return false;
        }
    }
    return true;
}

#if defined(__linux__)
bool dsl_jit_cc_promote_self_symbols_global(void) {
    char self_path[PATH_MAX];
    if (!dsl_jit_module_path_from_symbol((const void *)&dsl_jit_cc_promote_self_symbols_global,
                                         self_path, sizeof(self_path))) {
        return false;
    }
#if defined(RTLD_NOLOAD)
    dlerror();
    void *handle = dlopen(self_path, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (!handle) {
        dlerror();
        handle = dlopen(self_path, RTLD_NOW | RTLD_GLOBAL);
    }
#else
    dlerror();
    void *handle = dlopen(self_path, RTLD_NOW | RTLD_GLOBAL);
#endif
    if (!handle) {
        return false;
    }
    dlclose(handle);
    return true;
}
#endif

static void dsl_jit_cc_add_library_flag_if_exists(char *flags, size_t flags_size, const char *path) {
    if (!flags || flags_size == 0 || !path || path[0] == '\0') {
        return;
    }
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        return;
    }
    size_t len = strlen(flags);
    if (len >= flags_size - 1) {
        return;
    }
    int n = snprintf(flags + len, flags_size - len, " -L%s", path);
    if (n <= 0 || (size_t)n >= (flags_size - len)) {
        flags[flags_size - 1] = '\0';
    }
}

static void dsl_jit_cc_add_multiarch_library_flags(char *flags, size_t flags_size) {
    if (!flags || flags_size == 0) {
        return;
    }
    flags[0] = '\0';
#if defined(__linux__)
    const char *paths[] = {
#if defined(__x86_64__) || defined(__amd64__)
        "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
#elif defined(__aarch64__)
        "/usr/lib/aarch64-linux-gnu", "/lib/aarch64-linux-gnu",
#elif defined(__arm__)
        "/usr/lib/arm-linux-gnueabihf", "/lib/arm-linux-gnueabihf",
        "/usr/lib/arm-linux-gnueabi", "/lib/arm-linux-gnueabi",
#elif defined(__riscv) && (__riscv_xlen == 64)
        "/usr/lib/riscv64-linux-gnu", "/lib/riscv64-linux-gnu",
#elif defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
        "/usr/lib/powerpc64le-linux-gnu", "/lib/powerpc64le-linux-gnu",
#elif defined(__s390x__)
        "/usr/lib/s390x-linux-gnu", "/lib/s390x-linux-gnu",
#elif defined(__i386__)
        "/usr/lib/i386-linux-gnu", "/lib/i386-linux-gnu",
#endif
        "/usr/lib64", "/lib64",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        dsl_jit_cc_add_library_flag_if_exists(flags, flags_size, paths[i]);
    }
#else
    (void)flags_size;
#endif
}

bool dsl_jit_compile_shared(const me_dsl_compiled_program *program,
                            const char *src_path, const char *so_path) {
    if (!program || !src_path || !so_path) {
        return false;
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
    const char *debug_cc = getenv("ME_DSL_JIT_DEBUG_CC");
    bool show_cc_output = (debug_cc && debug_cc[0] != '\0' && strcmp(debug_cc, "0") != 0);
    const char *bridge_ldflags = "";
    char multiarch_ldflags[512];
    const char *math_ldflags = "";
    dsl_jit_cc_add_multiarch_library_flags(multiarch_ldflags, sizeof(multiarch_ldflags));
#if defined(__APPLE__)
    if (program->jit_use_runtime_math_bridge) {
        bridge_ldflags = " -Wl,-undefined,dynamic_lookup";
    }
#else
    math_ldflags = " -lm";
#endif
    char cmd[2048];
#if defined(__APPLE__)
    int n = snprintf(cmd, sizeof(cmd),
                     "%s -std=c99 -O3 -fPIC %s %s -dynamiclib -o \"%s\" \"%s\"%s%s%s%s",
                     cc, fp_cflags, cflags, so_path, src_path, bridge_ldflags,
                     multiarch_ldflags, math_ldflags,
                     show_cc_output ? "" : " >/dev/null 2>&1");
#else
    int n = snprintf(cmd, sizeof(cmd),
                     "%s -std=c99 -O3 -fPIC %s %s -shared -o \"%s\" \"%s\"%s%s%s%s",
                     cc, fp_cflags, cflags, so_path, src_path, bridge_ldflags,
                     multiarch_ldflags, math_ldflags,
                     show_cc_output ? "" : " >/dev/null 2>&1");
#endif
    if (n <= 0 || (size_t)n >= sizeof(cmd)) {
        return false;
    }
    int rc = system(cmd);
    return rc == 0;
}

bool dsl_jit_load_kernel(me_dsl_compiled_program *program, const char *shared_path,
                         char *detail, size_t detail_cap) {
    if (!program || !shared_path) {
        return false;
    }
    if (detail && detail_cap > 0) {
        detail[0] = '\0';
    }
    dlerror();
    void *handle = dlopen(shared_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        if (detail && detail_cap > 0) {
            const char *err = dlerror();
            if (!err || err[0] == '\0') {
                err = "unknown";
            }
            (void)snprintf(detail, detail_cap, "dlopen failed: %s", err);
        }
        return false;
    }
    dlerror();
    void *sym = dlsym(handle, ME_DSL_JIT_SYMBOL_NAME);
    if (!sym) {
        if (detail && detail_cap > 0) {
            const char *err = dlerror();
            if (!err || err[0] == '\0') {
                err = "unknown";
            }
            (void)snprintf(detail, detail_cap, "dlsym(%s) failed: %s",
                           ME_DSL_JIT_SYMBOL_NAME, err);
        }
        dlclose(handle);
        return false;
    }
    program->jit_dl_handle = handle;
    program->jit_kernel_fn = (me_dsl_jit_kernel_fn)sym;
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    return true;
}

#else

bool dsl_jit_cc_math_bridge_available(void) {
    return false;
}

bool dsl_jit_compile_shared(const me_dsl_compiled_program *program,
                            const char *src_path, const char *so_path) {
    (void)program;
    (void)src_path;
    (void)so_path;
    return false;
}

bool dsl_jit_load_kernel(me_dsl_compiled_program *program, const char *shared_path,
                         char *detail, size_t detail_cap) {
    (void)program;
    (void)shared_path;
    if (detail && detail_cap > 0) {
        detail[0] = '\0';
    }
    return false;
}

#endif
