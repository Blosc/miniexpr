/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_CONFIG_H
#define MINIEXPR_DSL_CONFIG_H

#include "dsl_parser.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ME_DSL_TRACE_DEFAULT
#define ME_DSL_TRACE_DEFAULT 0
#endif

#ifndef ME_DSL_WHILE_MAX_ITERS_DEFAULT
#define ME_DSL_WHILE_MAX_ITERS_DEFAULT 10000000LL
#endif

static inline bool dsl_env_flag_enabled(const char *name, bool default_value) {
    const char *env = getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    return strcmp(env, "0") != 0;
}

static inline int64_t dsl_env_i64(const char *name, int64_t default_value,
                                  int64_t min_value, int64_t max_value) {
    const char *env = getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    errno = 0;
    char *end = NULL;
    long long value = strtoll(env, &end, 10);
    if (errno != 0 || end == env) {
        return default_value;
    }
    while (*end && isspace((unsigned char)*end)) {
        end++;
    }
    if (*end != '\0') {
        return default_value;
    }
    if ((int64_t)value < min_value || (int64_t)value > max_value) {
        return default_value;
    }
    return (int64_t)value;
}

static inline me_dsl_fp_mode dsl_default_fp_mode_from_env(void) {
    const char *env = getenv("ME_DSL_FP_MODE");
    if (!env || env[0] == '\0') {
        return ME_DSL_FP_STRICT;
    }
    if (strcmp(env, "strict") == 0) {
        return ME_DSL_FP_STRICT;
    }
    if (strcmp(env, "contract") == 0) {
        return ME_DSL_FP_CONTRACT;
    }
    if (strcmp(env, "fast") == 0 || strcmp(env, "relaxed") == 0) {
        return ME_DSL_FP_FAST;
    }
    return ME_DSL_FP_STRICT;
}

static inline const char *dsl_jit_fp_mode_cflags(me_dsl_fp_mode fp_mode) {
    switch (fp_mode) {
    case ME_DSL_FP_STRICT:
        return "-fno-fast-math -ffp-contract=off";
    case ME_DSL_FP_CONTRACT:
        return "-fno-fast-math -ffp-contract=fast";
    case ME_DSL_FP_FAST:
        return "-ffast-math";
    default:
        return "-fno-fast-math -ffp-contract=off";
    }
}

static inline bool dsl_trace_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_TRACE", ME_DSL_TRACE_DEFAULT != 0);
}

static inline void dsl_tracef(const char *fmt, ...) {
    if (!fmt || !dsl_trace_enabled()) {
        return;
    }

    fprintf(stderr, "[me-dsl] ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static inline int64_t dsl_while_max_iters(void) {
    return dsl_env_i64("ME_DSL_WHILE_MAX_ITERS",
                       (int64_t)ME_DSL_WHILE_MAX_ITERS_DEFAULT,
                       LLONG_MIN, LLONG_MAX);
}

static inline bool dsl_jit_pos_cache_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT_POS_CACHE", true);
}

static inline bool dsl_jit_runtime_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT", true);
}

static inline bool dsl_jit_math_bridge_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT_MATH_BRIDGE", true);
}

static inline bool dsl_jit_scalar_math_bridge_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT_SCALAR_MATH_BRIDGE", false);
}

static inline bool dsl_jit_vec_math_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT_VEC_MATH", true);
}

static inline bool dsl_jit_branch_aware_if_lowering_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT_BRANCH_AWARE_IF", true);
}

static inline bool dsl_jit_hybrid_expr_vector_math_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT_HYBRID_EXPR_VEC_MATH", false);
}

static inline bool dsl_jit_index_vars_enabled(void) {
    return dsl_env_flag_enabled("ME_DSL_JIT_INDEX_VARS", true);
}

static inline int dsl_jit_bridge_chunk_items(void) {
    return (int)dsl_env_i64("ME_DSL_JIT_VEC_CHUNK_ITEMS", 1 << 20, 1, INT_MAX);
}

#endif
