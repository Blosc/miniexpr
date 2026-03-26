/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#include "dsl_jit_bridge_contract.h"
#include "dsl_jit_runtime_internal.h"

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

#if ME_USE_LIBTCC_FALLBACK

typedef struct TCCState me_tcc_state;

typedef me_tcc_state *(*me_tcc_new_fn)(void);
typedef void (*me_tcc_delete_fn)(me_tcc_state *s);
typedef int (*me_tcc_set_output_type_fn)(me_tcc_state *s, int output_type);
typedef int (*me_tcc_compile_string_fn)(me_tcc_state *s, const char *buf);
typedef int (*me_tcc_relocate_fn)(me_tcc_state *s);
typedef void *(*me_tcc_get_symbol_fn)(me_tcc_state *s, const char *name);
typedef int (*me_tcc_set_options_fn)(me_tcc_state *s, const char *str);
typedef int (*me_tcc_add_library_path_fn)(me_tcc_state *s, const char *path);
typedef int (*me_tcc_add_library_fn)(me_tcc_state *s, const char *libraryname);
typedef int (*me_tcc_add_symbol_fn)(me_tcc_state *s, const char *name, const void *val);
typedef void (*me_tcc_set_lib_path_fn)(me_tcc_state *s, const char *path);

typedef struct {
    bool attempted;
    bool available;
    void *handle;
    me_tcc_new_fn tcc_new_fn;
    me_tcc_delete_fn tcc_delete_fn;
    me_tcc_set_output_type_fn tcc_set_output_type_fn;
    me_tcc_compile_string_fn tcc_compile_string_fn;
    me_tcc_relocate_fn tcc_relocate_fn;
    me_tcc_get_symbol_fn tcc_get_symbol_fn;
    me_tcc_set_options_fn tcc_set_options_fn;
    me_tcc_add_library_path_fn tcc_add_library_path_fn;
    me_tcc_add_library_fn tcc_add_library_fn;
    me_tcc_add_symbol_fn tcc_add_symbol_fn;
    me_tcc_set_lib_path_fn tcc_set_lib_path_fn;
    char error[160];
} me_dsl_tcc_api;

static me_dsl_tcc_api g_dsl_tcc_api;

const char *dsl_jit_libtcc_error_message(void) {
    if (g_dsl_tcc_api.error[0]) {
        return g_dsl_tcc_api.error;
    }
    return "tcc backend unavailable";
}

static bool dsl_jit_module_path_from_symbol(const void *symbol, char *out, size_t out_size) {
    if (!symbol || !out || out_size == 0) {
        return false;
    }
#if defined(_WIN32) || defined(_WIN64)
    HMODULE module = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)symbol, &module)) {
        return false;
    }
    DWORD n = GetModuleFileNameA(module, out, (DWORD)out_size);
    if (n == 0 || n >= (DWORD)out_size) {
        return false;
    }
    out[n] = '\0';
    return true;
#else
    Dl_info info;
    if (dladdr(symbol, &info) == 0 || !info.dli_fname || info.dli_fname[0] == '\0') {
        return false;
    }
    int n = snprintf(out, out_size, "%s", info.dli_fname);
    return n > 0 && (size_t)n < out_size;
#endif
}

static void *dsl_jit_dynlib_open(const char *path) {
#if defined(_WIN32) || defined(_WIN64)
    return (void *)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *dsl_jit_dynlib_symbol(void *handle, const char *name) {
#if defined(_WIN32) || defined(_WIN64)
    return (void *)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

static void dsl_jit_dynlib_close(void *handle) {
    if (!handle) {
        return;
    }
#if defined(_WIN32) || defined(_WIN64)
    (void)FreeLibrary((HMODULE)handle);
#else
    dlclose(handle);
#endif
}

static const char *dsl_jit_dynlib_last_error(void) {
#if defined(_WIN32) || defined(_WIN64)
    static char err[160];
    DWORD code = GetLastError();
    if (code == 0) {
        err[0] = '\0';
        return err;
    }
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             NULL, code, 0, err, (DWORD)sizeof(err), NULL);
    if (n == 0) {
        snprintf(err, sizeof(err), "Win32 error %lu", (unsigned long)code);
        return err;
    }
    while (n > 0 && (err[n - 1] == '\r' || err[n - 1] == '\n' ||
                     err[n - 1] == ' ' || err[n - 1] == '\t')) {
        err[n - 1] = '\0';
        n--;
    }
    return err;
#else
    const char *err = dlerror();
    return err ? err : "";
#endif
}

static bool dsl_jit_path_dirname(const char *path, char *out, size_t out_size) {
    if (!path || !out || out_size == 0) {
        return false;
    }
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    if (!slash || (backslash && backslash > slash)) {
        slash = backslash;
    }
    if (!slash) {
        if (out_size < 2) {
            return false;
        }
        out[0] = '.';
        out[1] = '\0';
        return true;
    }
    if (slash == path) {
        if (out_size < 2) {
            return false;
        }
        out[0] = '/';
        out[1] = '\0';
        return true;
    }
    size_t len = (size_t)(slash - path);
    if (len + 1 > out_size) {
        return false;
    }
    memcpy(out, path, len);
    out[len] = '\0';
    return true;
}

static bool dsl_jit_libtcc_path_near_self(char *out, size_t out_size) {
    if (!out || out_size == 0) {
        return false;
    }
    char module_path[PATH_MAX];
    if (!dsl_jit_module_path_from_symbol((const void *)&dsl_jit_libtcc_path_near_self,
                                         module_path, sizeof(module_path))) {
        return false;
    }
    char dir[PATH_MAX];
    if (!dsl_jit_path_dirname(module_path, dir, sizeof(dir))) {
        return false;
    }
#if defined(_WIN32) || defined(_WIN64)
    const char *name = "tcc.dll";
#elif defined(__APPLE__)
    const char *name = "libtcc.dylib";
#else
    const char *name = "libtcc.so";
#endif
    char sep = '/';
    if (strchr(dir, '\\')) {
        sep = '\\';
    }
    int n = snprintf(out, out_size, "%s%c%s", dir, sep, name);
    return n > 0 && (size_t)n < out_size;
}

static bool dsl_jit_libtcc_runtime_dir(char *out, size_t out_size) {
    if (!out || out_size == 0) {
        return false;
    }
    const char *env = getenv("ME_DSL_JIT_TCC_LIB_PATH");
    if (env && env[0] != '\0') {
        int n = snprintf(out, out_size, "%s", env);
        return n > 0 && (size_t)n < out_size;
    }
    if (!g_dsl_tcc_api.tcc_new_fn) {
        return false;
    }
    char module_path[PATH_MAX];
    if (!dsl_jit_module_path_from_symbol((const void *)g_dsl_tcc_api.tcc_new_fn,
                                         module_path, sizeof(module_path))) {
        return false;
    }
    return dsl_jit_path_dirname(module_path, out, out_size);
}

static void dsl_jit_libtcc_add_library_path_if_exists(me_tcc_state *state, const char *path) {
    if (!state || !g_dsl_tcc_api.tcc_add_library_path_fn || !path || path[0] == '\0') {
        return;
    }
#if defined(_WIN32) || defined(_WIN64)
    DWORD attrs = GetFileAttributesA(path);
    if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        (void)g_dsl_tcc_api.tcc_add_library_path_fn(state, path);
    }
#else
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        (void)g_dsl_tcc_api.tcc_add_library_path_fn(state, path);
    }
#endif
}

static void dsl_jit_libtcc_add_multiarch_paths(me_tcc_state *state) {
#if defined(__linux__)
    if (!state || !g_dsl_tcc_api.tcc_add_library_path_fn) {
        return;
    }
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
        dsl_jit_libtcc_add_library_path_if_exists(state, paths[i]);
    }
#else
    (void)state;
#endif
}

static void *dsl_jit_self_dynlib_handle(void) {
    static bool initialized = false;
    static void *handle = NULL;
    if (initialized) {
        return handle;
    }
    initialized = true;
#if defined(_WIN32) || defined(_WIN64)
    HMODULE module = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&dsl_jit_self_dynlib_handle, &module)) {
        handle = (void *)module;
    }
#else
    char module_path[PATH_MAX];
    if (dsl_jit_module_path_from_symbol((const void *)&dsl_jit_self_dynlib_handle,
                                        module_path, sizeof(module_path))) {
        dlerror();
        handle = dlopen(module_path, RTLD_NOW | RTLD_LOCAL);
    }
#endif
    return handle;
}

static const void *dsl_jit_lookup_self_symbol(const char *name) {
    if (!name || name[0] == '\0') {
        return NULL;
    }
    void *handle = dsl_jit_self_dynlib_handle();
    if (!handle) {
        return NULL;
    }
    return dsl_jit_dynlib_symbol(handle, name);
}

#define ME_DSL_JIT_BRIDGE_NAME_ENTRY(pub_sym, bridge_fn, sig_type, decl) #pub_sym,
static const char *const dsl_jit_math_bridge_symbol_names[] = {
    ME_DSL_JIT_BRIDGE_SYMBOL_CONTRACT(ME_DSL_JIT_BRIDGE_NAME_ENTRY)
};
#undef ME_DSL_JIT_BRIDGE_NAME_ENTRY

static bool dsl_jit_libtcc_register_math_bridge(me_tcc_state *state) {
    if (!state) {
        return false;
    }
    if (!g_dsl_tcc_api.tcc_add_symbol_fn) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc backend missing required symbol tcc_add_symbol for math bridge");
        return false;
    }
    for (size_t i = 0; i < sizeof(dsl_jit_math_bridge_symbol_names) / sizeof(dsl_jit_math_bridge_symbol_names[0]); i++) {
        const char *name = dsl_jit_math_bridge_symbol_names[i];
        const void *addr = dsl_jit_lookup_self_symbol(name);
        if (!addr) {
            snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error),
                     "jit bridge symbol unavailable: %s", name);
            return false;
        }
        if (g_dsl_tcc_api.tcc_add_symbol_fn(state, name, addr) < 0) {
            snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error),
                     "tcc_add_symbol failed for %s", name);
            return false;
        }
    }
    return true;
}

static bool dsl_jit_libtcc_load_api(void) {
    if (g_dsl_tcc_api.attempted) {
        return g_dsl_tcc_api.available;
    }
    g_dsl_tcc_api.attempted = true;

    const char *env_path = getenv("ME_DSL_JIT_LIBTCC_PATH");
    const char *default_path = ME_DSL_JIT_LIBTCC_DEFAULT_PATH;
    const char *candidates[12];
    char self_candidate[PATH_MAX];
    int ncandidates = 0;
    if (env_path && env_path[0] != '\0') {
        candidates[ncandidates++] = env_path;
    }
    if (default_path && default_path[0] != '\0') {
        candidates[ncandidates++] = default_path;
    }
    if (dsl_jit_libtcc_path_near_self(self_candidate, sizeof(self_candidate))) {
        candidates[ncandidates++] = self_candidate;
    }
#if defined(_WIN32) || defined(_WIN64)
    candidates[ncandidates++] = "tcc.dll";
    candidates[ncandidates++] = "libtcc.dll";
#elif defined(__APPLE__)
    candidates[ncandidates++] = "libtcc.dylib";
    candidates[ncandidates++] = "libtcc.so";
    candidates[ncandidates++] = "libtcc.so.1";
#else
    candidates[ncandidates++] = "libtcc.so";
    candidates[ncandidates++] = "libtcc.so.1";
#endif
    candidates[ncandidates] = NULL;

    void *handle = NULL;
    for (int i = 0; i < ncandidates; i++) {
        handle = dsl_jit_dynlib_open(candidates[i]);
        if (handle) {
            break;
        }
    }
    if (!handle) {
        const char *err = dsl_jit_dynlib_last_error();
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error),
                 "failed to load libtcc shared library%s%s",
                 (err && err[0]) ? ": " : "", (err && err[0]) ? err : "");
        return false;
    }

#define ME_LOAD_TCC_SYM(field, sym_name, fn_type) \
    do { \
        g_dsl_tcc_api.field = (fn_type)dsl_jit_dynlib_symbol(handle, sym_name); \
        if (!g_dsl_tcc_api.field) { \
            dsl_jit_dynlib_close(handle); \
            snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), \
                     "libtcc missing required symbol %s", sym_name); \
            return false; \
        } \
    } while (0)

    ME_LOAD_TCC_SYM(tcc_new_fn, "tcc_new", me_tcc_new_fn);
    ME_LOAD_TCC_SYM(tcc_delete_fn, "tcc_delete", me_tcc_delete_fn);
    ME_LOAD_TCC_SYM(tcc_set_output_type_fn, "tcc_set_output_type", me_tcc_set_output_type_fn);
    ME_LOAD_TCC_SYM(tcc_compile_string_fn, "tcc_compile_string", me_tcc_compile_string_fn);
    ME_LOAD_TCC_SYM(tcc_relocate_fn, "tcc_relocate", me_tcc_relocate_fn);
    ME_LOAD_TCC_SYM(tcc_get_symbol_fn, "tcc_get_symbol", me_tcc_get_symbol_fn);
#undef ME_LOAD_TCC_SYM

    g_dsl_tcc_api.tcc_set_options_fn = (me_tcc_set_options_fn)dsl_jit_dynlib_symbol(handle, "tcc_set_options");
    g_dsl_tcc_api.tcc_add_library_path_fn = (me_tcc_add_library_path_fn)dsl_jit_dynlib_symbol(handle, "tcc_add_library_path");
    g_dsl_tcc_api.tcc_add_library_fn = (me_tcc_add_library_fn)dsl_jit_dynlib_symbol(handle, "tcc_add_library");
    g_dsl_tcc_api.tcc_add_symbol_fn = (me_tcc_add_symbol_fn)dsl_jit_dynlib_symbol(handle, "tcc_add_symbol");
    g_dsl_tcc_api.tcc_set_lib_path_fn = (me_tcc_set_lib_path_fn)dsl_jit_dynlib_symbol(handle, "tcc_set_lib_path");
    g_dsl_tcc_api.handle = handle;
    g_dsl_tcc_api.available = true;
    g_dsl_tcc_api.error[0] = '\0';
    return true;
}

void dsl_jit_libtcc_delete_state(void *state) {
    if (!state) {
        return;
    }
    if (!dsl_jit_libtcc_load_api() || !g_dsl_tcc_api.tcc_delete_fn) {
        return;
    }
    g_dsl_tcc_api.tcc_delete_fn((me_tcc_state *)state);
}

bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program) {
    if (!program || !program->jit_c_source) {
        return false;
    }
    if (program->fp_mode != ME_DSL_FP_STRICT) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc backend supports only strict fp mode");
        return false;
    }
    if (!dsl_jit_libtcc_load_api()) {
        return false;
    }
    me_tcc_state *state = g_dsl_tcc_api.tcc_new_fn();
    if (!state) {
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_new failed");
        return false;
    }

    char tcc_lib_dir[PATH_MAX];
    if (g_dsl_tcc_api.tcc_set_lib_path_fn &&
        dsl_jit_libtcc_runtime_dir(tcc_lib_dir, sizeof(tcc_lib_dir))) {
        g_dsl_tcc_api.tcc_set_lib_path_fn(state, tcc_lib_dir);
    }

    const char *tcc_opts = getenv("ME_DSL_JIT_TCC_OPTIONS");
    if (g_dsl_tcc_api.tcc_set_options_fn && tcc_opts && tcc_opts[0] != '\0') {
        (void)g_dsl_tcc_api.tcc_set_options_fn(state, tcc_opts);
    }
    if (g_dsl_tcc_api.tcc_set_output_type_fn(state, 1) < 0) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_set_output_type failed");
        return false;
    }
    dsl_jit_libtcc_add_multiarch_paths(state);
#if !defined(__APPLE__) && !defined(_WIN32) && !defined(_WIN64)
    if (g_dsl_tcc_api.tcc_add_library_fn) {
        (void)g_dsl_tcc_api.tcc_add_library_fn(state, "m");
    }
#endif
    if (program->jit_use_runtime_math_bridge &&
        !dsl_jit_libtcc_register_math_bridge(state)) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        return false;
    }
    if (g_dsl_tcc_api.tcc_compile_string_fn(state, program->jit_c_source) < 0) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_compile_string failed");
        return false;
    }
    if (g_dsl_tcc_api.tcc_relocate_fn(state) < 0) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_relocate failed");
        return false;
    }
    void *sym = g_dsl_tcc_api.tcc_get_symbol_fn(state, ME_DSL_JIT_SYMBOL_NAME);
    if (!sym) {
        g_dsl_tcc_api.tcc_delete_fn(state);
        snprintf(g_dsl_tcc_api.error, sizeof(g_dsl_tcc_api.error), "%s",
                 "tcc_get_symbol failed");
        return false;
    }

    if (program->jit_tcc_state) {
        dsl_jit_libtcc_delete_state(program->jit_tcc_state);
    }
    program->jit_tcc_state = state;
    program->jit_kernel_fn = (me_dsl_jit_kernel_fn)sym;
    program->jit_c_error_line = 0;
    program->jit_c_error_column = 0;
    program->jit_c_error[0] = '\0';
    program->jit_runtime_key = 0;
    program->jit_dl_handle_cached = false;
    return true;
}

#else

const char *dsl_jit_libtcc_error_message(void) {
    return "tcc backend not built";
}

void dsl_jit_libtcc_delete_state(void *state) {
    (void)state;
}

bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program) {
    (void)program;
    return false;
}

#endif
