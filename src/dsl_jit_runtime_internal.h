/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025-2026  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_DSL_JIT_RUNTIME_INTERNAL_H
#define MINIEXPR_DSL_JIT_RUNTIME_INTERNAL_H

#include "dsl_config.h"
#include "dsl_jit_ir.h"
#include "miniexpr.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef ME_DSL_MAX_NDIM
#define ME_DSL_MAX_NDIM 8
#endif

#ifndef ME_DSL_JIT_SYMBOL_NAME
#define ME_DSL_JIT_SYMBOL_NAME "me_dsl_jit_kernel"
#endif

#ifndef ME_DSL_JIT_CGEN_VERSION
#define ME_DSL_JIT_CGEN_VERSION 8
#endif

#ifndef ME_DSL_JIT_BRIDGE_ABI_VERSION
/* Runtime math bridge ABI for JIT-generated symbols (frozen list v1). */
#define ME_DSL_JIT_BRIDGE_ABI_VERSION 1
#endif

#define ME_DSL_JIT_META_MAGIC 0x4d454a49544d4554ULL
#define ME_DSL_JIT_META_VERSION 7

#if ME_USE_WASM32_JIT
typedef int (*me_dsl_jit_kernel_fn)(const void **inputs, void *output, int nitems);
#else
typedef int (*me_dsl_jit_kernel_fn)(const void **inputs, void *output, int64_t nitems);
#endif

typedef struct {
    me_expr *expr;
    int *var_indices;
    int n_vars;
} me_dsl_compiled_expr;

typedef struct me_dsl_compiled_stmt me_dsl_compiled_stmt;

typedef struct {
    me_dsl_compiled_stmt **stmts;
    int nstmts;
    int capacity;
} me_dsl_compiled_block;

typedef struct {
    me_dsl_compiled_expr cond;
    me_dsl_compiled_block block;
} me_dsl_compiled_if_branch;

struct me_dsl_compiled_stmt {
    me_dsl_stmt_kind kind;
    int line;
    int column;
    union {
        struct {
            int local_slot;
            me_dsl_compiled_expr value;
        } assign;
        struct {
            me_dsl_compiled_expr expr;
        } expr_stmt;
        struct {
            me_dsl_compiled_expr expr;
        } return_stmt;
        struct {
            char *format;
            me_dsl_compiled_expr *args;
            int nargs;
        } print_stmt;
        struct {
            me_dsl_compiled_expr cond;
            me_dsl_compiled_block then_block;
            me_dsl_compiled_if_branch *elif_branches;
            int n_elifs;
            int elif_capacity;
            me_dsl_compiled_block else_block;
            bool has_else;
        } if_stmt;
        struct {
            int loop_var_slot;
            me_dsl_compiled_expr start;
            me_dsl_compiled_expr stop;
            me_dsl_compiled_expr step;
            me_dsl_compiled_block body;
        } for_loop;
        struct {
            me_dsl_compiled_expr cond;
            me_dsl_compiled_block body;
        } while_loop;
        struct {
            me_dsl_compiled_expr cond;
        } flow;
    } as;
};

typedef struct {
    char **names;
    me_dtype *dtypes;
    size_t *itemsizes;
    bool *uniform;
    int count;
    int capacity;
} me_dsl_var_table;

typedef enum {
    ME_DSL_JIT_BIND_USER_INPUT = 0,
    ME_DSL_JIT_BIND_RESERVED_I = 1,
    ME_DSL_JIT_BIND_RESERVED_N = 2,
    ME_DSL_JIT_BIND_RESERVED_NDIM = 3,
    ME_DSL_JIT_BIND_RESERVED_GLOBAL_LINEAR_IDX = 4,
    ME_DSL_JIT_BIND_SYNTH_ND_CTX = 5
} me_dsl_jit_param_binding_kind;

typedef struct {
    int var_index;
    int dim;
    me_dsl_jit_param_binding_kind kind;
} me_dsl_jit_param_binding;

typedef struct {
    me_dsl_compiled_block block;
    me_dsl_var_table vars;
    int n_inputs;
    int n_locals;
    int *local_var_indices;
    int *local_slots;
    int idx_ndim;
    int idx_flat_idx;
    int idx_i[ME_DSL_MAX_NDIM];
    int idx_n[ME_DSL_MAX_NDIM];
    int uses_i_mask;
    int uses_n_mask;
    bool uses_ndim;
    bool uses_flat_idx;
    int compile_ndims;
    me_dsl_fp_mode fp_mode;
    me_dsl_compiler compiler;
    bool guaranteed_return;
    bool output_is_scalar;
    me_dtype output_dtype;
    me_dsl_jit_ir_program *jit_ir;
    uint64_t jit_ir_fingerprint;
    int jit_ir_error_line;
    int jit_ir_error_column;
    char jit_ir_error[128];
    char *jit_c_source;
    bool jit_use_runtime_math_bridge;
    bool jit_scalar_math_bridge_enabled;
    bool jit_synth_reserved_non_nd;
    bool jit_synth_reserved_nd;
    bool jit_vec_math_enabled;
    bool jit_branch_aware_if_lowering_enabled;
    bool jit_hybrid_expr_vec_math_enabled;
    int jit_c_error_line;
    int jit_c_error_column;
    char jit_c_error[128];
    char jit_lowering_mode[16];
    char jit_vector_ops[128];
    char jit_lowering_reason[128];
    me_dsl_jit_param_binding *jit_param_bindings;
    int jit_nparams;
    me_dsl_jit_kernel_fn jit_kernel_fn;
    void *jit_dl_handle;
    void *jit_tcc_state;
    uint64_t jit_runtime_key;
    bool jit_dl_handle_cached;
} me_dsl_compiled_program;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t cgen_version;
    uint32_t bridge_abi_version;
    uint32_t target_tag;
    uint32_t ptr_size;
    uint64_t cache_key;
    uint64_t ir_fingerprint;
    int32_t output_dtype;
    int32_t fp_mode;
    int32_t compiler;
    int32_t nparams;
    int32_t param_dtypes[ME_MAX_VARS];
    int32_t binding_kinds[ME_MAX_VARS];
    int32_t binding_dims[ME_MAX_VARS];
    int32_t binding_var_indices[ME_MAX_VARS];
    uint64_t toolchain_hash;
} me_dsl_jit_cache_meta;

static inline const char *dsl_fp_mode_name(me_dsl_fp_mode fp_mode) {
    switch (fp_mode) {
    case ME_DSL_FP_STRICT:
        return "strict";
    case ME_DSL_FP_CONTRACT:
        return "contract";
    case ME_DSL_FP_FAST:
        return "fast";
    default:
        return "unknown";
    }
}

static inline const char *dsl_compiler_name(me_dsl_compiler compiler) {
    switch (compiler) {
    case ME_DSL_COMPILER_LIBTCC:
        return "tcc";
    case ME_DSL_COMPILER_CC:
        return "cc";
    default:
        return "unknown";
    }
}

bool dsl_jit_compile_libtcc_in_memory(me_dsl_compiled_program *program);
const char *dsl_jit_libtcc_error_message(void);
void dsl_jit_libtcc_delete_state(void *state);
bool dsl_jit_cc_math_bridge_available(void);
uint64_t dsl_jit_runtime_cache_key(const me_dsl_compiled_program *program);
void dsl_jit_fill_cache_meta(me_dsl_jit_cache_meta *meta,
                             const me_dsl_compiled_program *program,
                             uint64_t key);
bool dsl_jit_write_meta_file(const char *path, const me_dsl_jit_cache_meta *meta);
bool dsl_jit_meta_file_matches(const char *path, const me_dsl_jit_cache_meta *expected);
bool dsl_jit_get_cache_dir(char *out, size_t out_size);
bool dsl_jit_write_text_file(const char *path, const char *text);
bool dsl_jit_copy_file(const char *src_path, const char *dst_path);

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__EMSCRIPTEN__)
bool dsl_jit_c_compiler_available(void);
bool dsl_jit_compile_shared(const me_dsl_compiled_program *program,
                            const char *src_path, const char *so_path);
bool dsl_jit_load_kernel(me_dsl_compiled_program *program, const char *shared_path,
                         char *detail, size_t detail_cap);
#if defined(__linux__)
bool dsl_jit_cc_promote_self_symbols_global(void);
#endif
#endif

#if ME_USE_WASM32_JIT
bool dsl_jit_compile_wasm32(me_dsl_compiled_program *program, uint64_t key);
bool dsl_jit_wasm_pos_cache_bind_program(me_dsl_compiled_program *program, uint64_t key);
#if ME_WASM32_SIDE_MODULE
bool me_wasm_jit_helpers_available(void);
#endif
#endif

void dsl_try_prepare_jit_runtime(me_dsl_compiled_program *program);

#endif
