#ifndef NUMPY_COMPAT_BATCH_JIT_H
#define NUMPY_COMPAT_BATCH_JIT_H
#include "miniexpr_artifact.h"
bool numpy_compat_batch_enabled(me_jit_mode mode);
int numpy_compat_batch_compile(me_artifact **artifacts, size_t count, void **handle);
void numpy_compat_batch_close(void *handle);
#endif
