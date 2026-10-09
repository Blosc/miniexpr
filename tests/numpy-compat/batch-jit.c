/* Test-only bulk compilation: every eligible typed kernel is still emitted and
 * executed. Amortize external compiler startup/linking, not numerical coverage.
 * Per-kernel source differs from production only in its exported symbol name. */
#include "batch-jit.h"
#include "dsl_compile_internal.h"
#include "dsl_jit_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#include <unistd.h>
#endif

bool numpy_compat_batch_enabled(me_jit_mode mode) {
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
    (void)mode;
    return false;
#else
    const char *backend = getenv("ME_DSL_JIT_COMPILER");
    const char *serial = getenv("MENUDET_JIT_SERIAL");
    return mode == ME_JIT_ON && backend && !strcmp(backend,"cc") &&
        !(serial && !strcmp(serial,"1"));
#endif
}

int numpy_compat_batch_compile(me_artifact **artifacts, size_t count, void **handle) {
    *handle = NULL;
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
    (void)artifacts;
    (void)count;
    return 1;
#else
    char directory[1024], source_path[1300], library_path[1300];
    if (!dsl_jit_get_cache_dir(directory,sizeof(directory))) return 1;
    snprintf(source_path,sizeof(source_path),"%s/conformance_%ld.c",directory,(long)getpid());
    snprintf(library_path,sizeof(library_path),"%s/conformance_%ld.so",directory,(long)getpid());
    FILE *file = fopen(source_path,"wb");
    if (!file) return 1;
    const me_dsl_compiled_program *first = NULL;
    const char *header = NULL;
    size_t header_size = 0, kernels = 0;
    int failed = 0;
    for (size_t i = 0; i < count && !failed; i++) {
        me_dsl_compiled_program *p = dsl_artifact_program_for_tests(artifacts[i]);
        if (!p) continue;
        p->jit_request_mode = ME_JIT_ON;
        dsl_portable_prepare_jit_source(p);
        if (!p->jit_c_source) continue; /* identical production eligibility */
        const char *entry = strstr(p->jit_c_source,"int " ME_DSL_JIT_SYMBOL_NAME "(");
        if (!entry) {
            failed = 1;
            break;
        }
        size_t prefix = (size_t)(entry-p->jit_c_source);
        if (!first) {
            first = p;
            header = p->jit_c_source;
            header_size = prefix;
            if (fwrite(header,1,prefix,file) != prefix) {
                failed = 1;
                break;
            }
        }
        /* Refuse to combine differing lowering/ABI/helper headers or flags. */
        else if (prefix != header_size || memcmp(header,p->jit_c_source,prefix) ||
                 p->compiler != first->compiler || p->fp_mode != first->fp_mode) {
            failed = 1;
            break;
        }
        const char *body = entry + strlen("int " ME_DSL_JIT_SYMBOL_NAME);
        if (fprintf(file,"int %s_%zu%s\n",ME_DSL_JIT_SYMBOL_NAME,i,body) < 0) failed = 1;
        kernels++;
    }
    if (fclose(file)) failed = 1;
    if (getenv("MENUDET_REQUIRE_JIT") && !kernels) failed = 1;
    if (!failed && first && !dsl_jit_compile_shared(first,source_path,library_path)) failed = 1;
    if (!failed && first) {
        *handle = dlopen(library_path,RTLD_NOW | RTLD_LOCAL);
        if (!*handle) failed = 1;
    }
    for (size_t i = 0; i < count && !failed && *handle; i++) {
        me_dsl_compiled_program *p = dsl_artifact_program_for_tests(artifacts[i]);
        if (!p || !p->jit_c_source) continue;
        char symbol[128];
        snprintf(symbol,sizeof(symbol),"%s_%zu",ME_DSL_JIT_SYMBOL_NAME,i);
        p->jit_kernel_fn = (me_dsl_jit_kernel_fn)dlsym(*handle,symbol);
        if (!p->jit_kernel_fn) failed = 1;
        /* The harness owns the library until every borrowed program is freed. */
    }
    remove(source_path);
    remove(library_path);
    fprintf(stderr,"conformance bulk JIT: kernels=%zu compiler_invocations=%d status=%s\n",
            kernels,first ? 1 : 0,failed ? "failed" : "ok");
    return failed;
#endif
}

void numpy_compat_batch_close(void *handle) {
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    if (handle) dlclose(handle);
#else
    (void)handle;
#endif
}
