/* Portable 1.1 scalar C lowering, revision 30. Never translate source
 * text: the typed tree includes promotion and final output conversions. The
 * private kernel ABI appends a participating mask and host comparison/math bindings;
 * legacy/full kernels retain their unchanged three-argument ABI. */
#include "dsl_compile_internal.h"
#include "dsl_math_dispatch.h"
#include "dsl_portable_expr.h"
#include "dsl_portable_types.h"
#include "dsl_portable_batch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

static const char *pj_type(me_dtype dtype) {
    switch (dtype) {
        case ME_FLOAT32: return "float";
        case ME_FLOAT64: return "double";
        case ME_BOOL: return "_Bool";
        case ME_INT8: return "int8_t";
        case ME_INT16: return "int16_t";
        case ME_INT32: return "int32_t";
        case ME_INT64: return "int64_t";
        case ME_UINT8: return "uint8_t";
        case ME_UINT16: return "uint16_t";
        case ME_UINT32: return "uint32_t";
        case ME_UINT64: return "uint64_t";
        default: return NULL;
    }
}

static bool pj_integer(me_dtype d) {
    return d != ME_BOOL && d != ME_FLOAT32 && d != ME_FLOAT64 && pj_type(d);
}

static bool pj_unsigned(me_dtype d) {
    return d == ME_UINT8 || d == ME_UINT16 || d == ME_UINT32 || d == ME_UINT64;
}

static bool pj_float(me_dtype d) {
    return d == ME_FLOAT32 || d == ME_FLOAT64;
}

static bool pj_integral(me_dtype d) {
    return d == ME_BOOL || pj_integer(d);
}

typedef struct {
    const int64_t *values;
    const bool *known;
    int count;
    bool scalar_phase;
    const char *nd_preamble;
} pj_constants;

typedef struct {
    char *text;
    size_t used, capacity;
} pj_text;

typedef struct {
    const me_expr *node;
    uint64_t hash;
    int uses;
    char value[32];
} pj_cse_entry;

typedef struct {
    pj_cse_entry entries[64];
    int count;
} pj_cse_context;

static bool pj_append(pj_text *s, const char *format, ...);

static const char *pj_unsigned_type(me_dtype dtype) {
    switch (dtype) {
        case ME_INT8: case ME_UINT8: return "uint8_t";
        case ME_INT16: case ME_UINT16: return "uint16_t";
        case ME_INT32: case ME_UINT32: return "uint32_t";
        case ME_INT64: case ME_UINT64: return "uint64_t";
        default: return NULL;
    }
}

static const char *pj_arithmetic_carrier(me_dtype dtype) {
    return dtype_size(dtype) <= 2 ? "uint32_t" : "uint64_t";
}

static bool pj_literal_integer_bits(const me_expr *n, me_dtype dtype, uint64_t *out) {
    if (!n || TYPE_MASK(n->type) != ME_CONSTANT || !(n->flags & ME_EXPR_FLAG_INTEGER_LITERAL) ||
        !pj_integer(n->dtype) || !pj_integer(dtype)) return false;
    int bits = (int)(dtype_size(dtype)*8);
    bool negative = (n->flags & ME_EXPR_FLAG_NEGATIVE_LITERAL) != 0;
    uint64_t limit = pj_unsigned(dtype) ? (negative ? 0 : UINT64_MAX >> (64-bits)) :
        negative ? UINT64_C(1) << (bits-1) : UINT64_MAX >> (65-bits);
    if (n->integer_magnitude > limit) return false;
    *out = negative ? 0ULL-n->integer_magnitude : n->integer_magnitude;
    return true;
}

/* Projection to the low N bits commutes with +,-,*,&,|,^ and unary -/~.
 * Keep a whole same-width strong integer tree unsigned until its consumer,
 * instead of materializing a signed representation after every node. Never
 * absorb checked weak subtrees, casts, shifts, comparisons or mixed widths. */
static bool pj_integer_tree(const me_dsl_compiled_program *p, const me_expr *n,
    const bool *defined, const pj_constants *constants, me_dtype dtype, pj_text *out, int depth) {
    if (!n || n->dtype != dtype || !pj_integer(dtype) || depth > 128) return false;
    if (TYPE_MASK(n->type) == ME_CONSTANT && (n->flags & ME_EXPR_FLAG_INTEGER_LITERAL)) {
        uint64_t bits;
        if (!pj_literal_integer_bits(n,dtype,&bits)) return false;
        return pj_append(out,"%lluULL",(unsigned long long)bits);
    }
    if (IS_FUNCTION(n->type) && !n->function && ARITY(n->type) == 1) {
        const me_expr *arg = n->parameters[0];
        uint64_t bits, original;
        if (!arg || !(arg->flags & (ME_EXPR_FLAG_WEAK_LITERAL | ME_EXPR_FLAG_WEAK_SCALAR))) return false;
        if (pj_literal_integer_bits(arg,arg->dtype,&original) && pj_literal_integer_bits(arg,dtype,&bits))
            return pj_append(out,"%lluULL",(unsigned long long)bits);
        if (!constants || !constants->known || TYPE_MASK(arg->type) != ME_VARIABLE ||
            !is_synthetic_address(arg->bound) || (arg->dtype != ME_INT64 && arg->dtype != ME_BOOL)) return false;
        int index = (int)((const char *)arg->bound-synthetic_var_addresses);
        if (index < 0 || index >= constants->count || !constants->known[index] || !p->vars.uniform[index]) return false;
        /* Artifact-owned weak captures are immutable. Only absorb a conversion
         * after checking its exact target range; never fold scalar arithmetic or
         * an out-of-range leaf, whose error still depends on lane participation. */
        if (pj_unsigned(dtype)) {
            if (dsl_portable_signed_to_unsigned(dtype,constants->values[index],&bits)) return false;
        } else {
            int64_t value;
            if (dsl_portable_signed_to_signed(dtype,constants->values[index],&value)) return false;
            bits = (uint64_t)value;
        }
        return pj_append(out,"%lluULL",(unsigned long long)bits);
    }
    if (TYPE_MASK(n->type) == ME_VARIABLE && is_synthetic_address(n->bound)) {
        int index = (int)((const char *)n->bound-synthetic_var_addresses);
        if (index < 0 || index >= p->vars.count) return false;
        if (index < p->n_inputs) return pj_append(out,"((uint64_t)pj_input_%d[%s])",index,
            p->vars.uniform[index] ? "0" : "i");
        if (defined && p->local_slots && p->local_slots[index] >= 0 && defined[index] && !p->output_is_scalar)
            return pj_append(out,"((uint64_t)local_%d)",index);
        return false;
    }
    if (!IS_FUNCTION(n->type) || !n->function || (n->flags & ME_EXPR_FLAG_WEAK_SCALAR) ||
        !(n->flags & ME_EXPR_FLAG_NUMPY_1_1)) return false;
    int arity = ARITY(n->type);
    const char *op = me_portable_operator(n);
    if (!op || (arity == 1 ? (strcmp(op,"-") && strcmp(op,"+") && strcmp(op,"~")) :
        arity != 2 || (strcmp(op,"+") && strcmp(op,"-") && strcmp(op,"*") &&
                       strcmp(op,"&") && strcmp(op,"|") && strcmp(op,"^")))) return false;
    if (!pj_append(out,"(%s",arity == 1 ? (!strcmp(op,"-") ? "0ULL-" : op) : "")) return false;
    if (!pj_integer_tree(p,n->parameters[0],defined,constants,dtype,out,depth+1)) return false;
    if (arity == 2 && (!pj_append(out," %s ",op) ||
        !pj_integer_tree(p,n->parameters[1],defined,constants,dtype,out,depth+1))) return false;
    return pj_append(out,")");
}

static bool pj_supported_reduction(const me_expr *n) {
    if (!n || !is_reduction_node(n) || !pj_type(n->dtype)) return false;
    const me_expr *arg = n->parameters[0];
    if (!arg || !pj_type(arg->dtype)) return false;
    me_reduce_kind kind = reduction_kind(n->function);
    if (kind == ME_REDUCE_MIN || kind == ME_REDUCE_MAX ||
        kind == ME_REDUCE_ANY || kind == ME_REDUCE_ALL) return true;
    if (kind == ME_REDUCE_SUM || kind == ME_REDUCE_PROD) return true;
    return pj_float(n->dtype) && (pj_float(arg->dtype) || pj_integral(arg->dtype)) &&
        (kind == ME_REDUCE_SUM || kind == ME_REDUCE_PROD || kind == ME_REDUCE_MEAN);
}

/* Only immutable integral capture leaves can be specialized. Validate the exact
 * checked weak conversion; out-of-range captures keep interpretation, including
 * its lazy participation/error timing. Never alter the shared typed tree. */
static char *pj_weak_conversion(const me_expr *n, const pj_constants *constants) {
    const me_expr *arg = n->parameters[0];
    if (!constants || !constants->known || TYPE_MASK(arg->type) != ME_VARIABLE ||
        !is_synthetic_address(arg->bound) || (arg->dtype != ME_INT64 && arg->dtype != ME_BOOL)) return NULL;
    int index = (int)((const char *)arg->bound - synthetic_var_addresses);
    if (index < 0 || index >= constants->count || !constants->known[index]) return NULL;
    int64_t value = constants->values[index];
    uint64_t bits;
    if (pj_unsigned(n->dtype)) {
        if (dsl_portable_signed_to_unsigned(n->dtype, value, &bits)) return NULL;
    } else {
        int64_t converted;
        if (dsl_portable_signed_to_signed(n->dtype, value, &converted)) return NULL;
        bits = (uint64_t)converted;
    }
    char literal[128];
    snprintf(literal, sizeof(literal), "pj_%s(UINT64_C(%llu))", pj_type(n->dtype), (unsigned long long)bits);
    return strdup(literal);
}

/* Prove runtime participation only through floating arithmetic and array inputs.
 * In particular, a constant-only subtree of a larger array expression is not
 * runtime arithmetic and must retain its exception-preserving literal loads. */
static bool pj_runtime_float(const me_dsl_compiled_program *p, const me_expr *n, int depth) {
    if (!n || depth > 128 || !pj_float(n->dtype)) return false;
    if (TYPE_MASK(n->type) == ME_VARIABLE && is_synthetic_address(n->bound)) {
        int index = (int)((const char *)n->bound - synthetic_var_addresses);
#ifdef __EMSCRIPTEN__
        return index >= 0 && index < p->vars.count;
#else
        return index >= 0 && index < p->n_inputs && !p->vars.uniform[index];
#endif
    }
    if (!IS_FUNCTION(n->type) || is_reduction_node(n)) return false;
    const char *op = me_portable_operator(n);
    int arity = ARITY(n->type);
    bool arithmetic = op && arity == 2 && (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*") || !strcmp(op,"/"));
    bool math = !arithmetic && (dsl_portable_resolve_unary_math(n) || dsl_portable_resolve_binary_math(n) || dsl_portable_resolve_ternary_math(n));
    if (!arithmetic && !math) return false;
    for (int i=0; i<arity; i++) if (pj_runtime_float(p,n->parameters[i],depth+1)) return true;
    return false;
}

static bool pj_plain_literal(const me_expr *n, const char *op) {
    if (!n || TYPE_MASK(n->type) != ME_CONSTANT || !pj_float(n->dtype)) return false;
    double value;
    if (n->flags & ME_EXPR_FLAG_INTEGER_LITERAL) {
        /* Conservatively accept only the contiguous exact-integer range. Do not
         * fold a potentially inexact integer-to-float conversion into a literal. */
        uint64_t limit = UINT64_C(1) << (n->dtype == ME_FLOAT32 ? 24 : 53);
        if (n->integer_magnitude > limit) return false;
        value = (double)n->integer_magnitude;
    } else {
        value = n->dtype == ME_FLOAT32 ? (double)n->literal_f32 : n->value;
    }
    /* Identities can erase the runtime operation, including signaling-NaN
     * exceptions. Keep their barriers even though the literal itself is exact. */
    return isfinite(value) && value != 0 &&
        ((strcmp(op,"*") && strcmp(op,"/")) || fabs(value) != 1);
}

static bool pj_cse_key(const me_dsl_compiled_program *p, const me_expr *n, uint64_t *hash,
                       int depth) {
    if (!n || depth > 32 || n->flags & (ME_EXPR_FLAG_WEAK_LITERAL | ME_EXPR_FLAG_WEAK_SCALAR))
        return false;
    uint64_t h = UINT64_C(1469598103934665603);
#define PJ_CSE_MIX(v) do { uint64_t pj_v_ = (uint64_t)(v); \
    for (int pj_b_ = 0; pj_b_ < 8; pj_b_++) { h ^= (unsigned char)(pj_v_ >> (pj_b_ * 8)); \
    h *= UINT64_C(1099511628211); } } while (0)
    PJ_CSE_MIX(n->type); PJ_CSE_MIX(n->dtype); PJ_CSE_MIX(n->flags);
    int kind = TYPE_MASK(n->type), arity = ARITY(n->type);
    if (kind == ME_VARIABLE) {
        if (!is_synthetic_address(n->bound)) return false;
        int index = (int)((const char *)n->bound - synthetic_var_addresses);
        if (index < 0 || index >= p->vars.count ||
            (p->output_is_scalar && !p->vars.uniform[index])) return false;
        PJ_CSE_MIX(index);
    } else if (kind == ME_CONSTANT) {
        if (pj_float(n->dtype)) {
            uint64_t bits = 0;
            if (n->dtype == ME_FLOAT32) memcpy(&bits, &n->literal_f32, sizeof(float));
            else memcpy(&bits, &n->value, sizeof(double));
            PJ_CSE_MIX(bits);
        } else {
            PJ_CSE_MIX(n->integer_magnitude);
        }
    } else {
        const char *op = me_portable_operator(n);
        if (!op || (strcmp(op,"+") && strcmp(op,"-") &&
            strcmp(op,"*") && strcmp(op,"/")) || arity < 1 || arity > 2) return false;
        for (int i = 0; i < arity; i++) {
            uint64_t child;
            if (!pj_cse_key(p,(const me_expr *)n->parameters[i],&child,depth+1)) return false;
            PJ_CSE_MIX(child);
        }
    }
    *hash = h;
#undef PJ_CSE_MIX
    return true;
}

static bool pj_cse_equal(const me_dsl_compiled_program *p, const me_expr *a,
                         const me_expr *b, int depth) {
    if (!a || !b || depth > 32 || a->type != b->type || a->dtype != b->dtype ||
        a->flags != b->flags) return false;
    int kind = TYPE_MASK(a->type), arity = ARITY(a->type);
    if (kind == ME_VARIABLE) return a->bound == b->bound;
    if (kind == ME_CONSTANT) {
        if (pj_float(a->dtype)) return a->dtype == ME_FLOAT32 ?
            memcmp(&a->literal_f32,&b->literal_f32,sizeof(float)) == 0 :
            memcmp(&a->value,&b->value,sizeof(double)) == 0;
        return a->integer_magnitude == b->integer_magnitude;
    }
    if (a->function != b->function || !me_portable_operator(a) || arity < 1 || arity > 2)
        return false;
    for (int i = 0; i < arity; i++)
        if (!pj_cse_equal(p,(const me_expr *)a->parameters[i],
                          (const me_expr *)b->parameters[i],depth+1)) return false;
    (void)p;
    return true;
}

static void pj_cse_prescan(const me_dsl_compiled_program *p, const me_expr *n,
                           pj_cse_context *cse, int depth) {
    if (!n || depth > 32 || !IS_FUNCTION(n->type) || is_reduction_node(n)) return;
    uint64_t hash = 0;
    if (pj_cse_key(p, n, &hash, 0)) {
        for (int i = 0; i < cse->count; i++) {
            if (cse->entries[i].hash == hash && pj_cse_equal(p, cse->entries[i].node, n, 0)) {
                cse->entries[i].uses++;
                return;
            }
        }
        if (cse->count < (int)(sizeof(cse->entries) / sizeof(cse->entries[0]))) {
            pj_cse_entry *entry = &cse->entries[cse->count++];
            entry->node = n;
            entry->hash = hash;
            entry->uses = 1;
            entry->value[0] = '\0';
        }
    }
    const char *op = me_portable_operator(n);
    if (op && (!strcmp(op, "where") || !strcmp(op, "and") || !strcmp(op, "or"))) return;
    int arity = ARITY(n->type);
    for (int i = 0; i < arity && i < 3; i++) {
        pj_cse_prescan(p, (const me_expr *)n->parameters[i], cse, depth + 1);
    }
}

static char *pj_expr_impl(me_dsl_compiled_program *p, const me_expr *n, const bool *defined, int depth,
    const pj_constants *constants, pj_text *s, int *next_temp, bool plain_literal,
    pj_cse_context *cse);

static char *pj_expr(me_dsl_compiled_program *p, const me_expr *n, const bool *defined, int depth,
    const pj_constants *constants, pj_text *s, int *next_temp) {
#ifdef __EMSCRIPTEN__
    /* Nested lazy expressions call this wrapper recursively. Keep the CSE table
     * off the wasm stack, whose default budget is much smaller than a host's. */
    pj_cse_context *cse = calloc(1,sizeof(*cse));
    if (!cse) return NULL;
    pj_cse_prescan(p, n, cse, 0);
    char *result = pj_expr_impl(p,n,defined,depth,constants,s,next_temp,false,cse);
    free(cse);
    return result;
#else
    pj_cse_context cse = {0};
    pj_cse_prescan(p, n, &cse, 0);
    return pj_expr_impl(p,n,defined,depth,constants,s,next_temp,false,&cse);
#endif
}

static char *pj_expr_impl(me_dsl_compiled_program *p, const me_expr *n, const bool *defined, int depth,
    const pj_constants *constants, pj_text *s, int *next_temp, bool plain_literal,
    pj_cse_context *cse) {
    if (!n || depth > 128 || !pj_type(n->dtype)) return NULL;
    uint64_t cse_hash = 0;
    bool cse_eligible = !plain_literal && pj_cse_key(p,n,&cse_hash,0);
    pj_cse_entry *cse_slot = NULL;
    if (cse_eligible) for (int i = 0; i < cse->count; i++) {
        pj_cse_entry *entry = &cse->entries[i];
        if (entry->hash == cse_hash && pj_cse_equal(p,entry->node,n,0)) {
            if (entry->value[0]) return strdup(entry->value);
            cse_slot = entry;
            break;
        }
    }
    const char *type = pj_type(n->dtype);
    char leaf[256];
    if (IS_FUNCTION(n->type) && pj_integer(n->dtype)) {
        pj_text integer = {0};
        if (pj_integer_tree(p,n,defined,constants,n->dtype,&integer,0)) {
            size_t length = integer.used + 64;
            char *result = malloc(length);
            if (result) snprintf(result,length,"pj_%s(%s)",type,integer.text);
            free(integer.text);
            return result;
        }
        free(integer.text);
    }
    if (is_reduction_node(n) && constants && constants->scalar_phase) {
        me_reduce_kind kind = reduction_kind(n->function);
        const me_expr *arg = n->parameters[0];
        if (!pj_supported_reduction(n)) return NULL;
        bool extrema = kind == ME_REDUCE_MIN || kind == ME_REDUCE_MAX;
        bool truth = kind == ME_REDUCE_ANY || kind == ME_REDUCE_ALL;
        bool integer = (kind == ME_REDUCE_SUM || kind == ME_REDUCE_PROD || kind == ME_REDUCE_MEAN) &&
            pj_integral(arg->dtype);
        bool bridged = extrema || truth || integer;
        int id = (*next_temp)++;
        const char *acc_type = pj_type(arg->dtype);
        int bridge = -1;
        if (bridged) {
            if (p->portable_jit_nchecked == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) return NULL;
            bridge = p->portable_jit_nchecked++;
            p->portable_jit_checked[bridge] = n;
            acc_type = "uint64_t";
        }
        if (bridged && TYPE_MASK(arg->type) == ME_VARIABLE && is_synthetic_address(arg->bound)) {
            int index = (int)((const char *)arg->bound - synthetic_var_addresses);
            if (index >= 0 && index < p->n_inputs && !p->vars.uniform[index]) {
                const char *decode = n->dtype == ME_FLOAT32 ? "pj_unpack_f32" :
                    n->dtype == ME_FLOAT64 ? "pj_unpack_f64" : NULL;
                char integer_decode[64];
                if (!decode) {
                    snprintf(integer_decode,sizeof(integer_decode),"pj_%s",type);
                    decode = n->dtype == ME_BOOL ? "(_Bool)" : integer_decode;
                }
                if (!pj_append(s,"uint64_t pj_bits_%d = ((pj_checked)inputs[%d])(inputs[%d],"
                    "(uint64_t)(uintptr_t)inputs[%d],(uint64_t)(uintptr_t)mask,"
                    "(uint64_t)count | (UINT64_C(1) << 63),&pj_status); if (pj_status) return pj_status;\n"
                    "volatile %s pj_reduced_%d = %s(pj_bits_%d);\n",
                    id,p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF+1+bridge,index,type,id,decode,id)) return NULL;
                snprintf(leaf,sizeof(leaf),"pj_reduced_%d",id);
                return strdup(leaf);
            }
        }
        bool ok = pj_append(s,"%s pj_acc_%d = %d; int64_t pj_count_%d = 0;\n"
            "for (int64_t i=0; i<count; i++) { if (mask && !mask[i]) continue;\n%s",
            acc_type,id,kind == ME_REDUCE_PROD || kind == ME_REDUCE_ALL ? 1 : 0,id,
            constants->nd_preamble ? constants->nd_preamble : "");
        pj_constants mapped = *constants;
        mapped.scalar_phase = false;
        char *value = ok ? pj_expr(p,arg,defined,depth+1,&mapped,s,next_temp) : NULL;
        if (!value) return NULL;
        if (bridged) {
            const char *pack = arg->dtype == ME_FLOAT32 ? "pj_pack_f32" :
                arg->dtype == ME_FLOAT64 ? "pj_pack_f64" : "(uint64_t)";
            ok = pj_append(s,"pj_acc_%d = ((pj_checked)inputs[%d])(inputs[%d],pj_acc_%d,%s(%s),"
                "pj_count_%d != 0,&pj_status); if (pj_status) return pj_status; pj_count_%d++;\n}\n",
                id,p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF,
                p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF+1+bridge,id,pack,value,id,id);
        } else ok = pj_append(s,"pj_acc_%d = (%s)(pj_acc_%d %s (%s)); pj_count_%d++;\n}\n",
            id,acc_type,id,kind == ME_REDUCE_PROD ? "*" : "+",value,id);
        free(value);
        if (!ok) return NULL;
        if (bridged) {
            if (extrema && !pj_append(s,"if (!pj_count_%d) return -5;\n",id)) return NULL;
            /* Integer mean retains a checked integer accumulator. Finalize in
             * the host bridge so uint64 conversion/rounding matches p_reduce. */
            if (integer && kind == ME_REDUCE_MEAN && !pj_append(s,
                "pj_acc_%d = ((pj_checked)inputs[%d])(inputs[%d],pj_acc_%d,"
                "(uint64_t)pj_count_%d,UINT64_C(1) << 62,&pj_status); if (pj_status) return pj_status;\n",
                id,p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF,
                p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF+1+bridge,id,id)) return NULL;
            const char *decode = n->dtype == ME_FLOAT32 ? "pj_unpack_f32" :
                n->dtype == ME_FLOAT64 ? "pj_unpack_f64" : NULL;
            char integer_decode[64];
            if (!decode) {
                snprintf(integer_decode,sizeof(integer_decode),"pj_%s",type);
                decode = n->dtype == ME_BOOL ? "(_Bool)" : integer_decode;
            }
            if (!pj_append(s,"volatile %s pj_reduced_%d = %s(pj_acc_%d);\n",type,id,decode,id)) return NULL;
        } else if (kind == ME_REDUCE_MEAN) {
            const char *nan = arg->dtype == ME_FLOAT32 ?
                "pj_unpack_f32(UINT64_C(0x7fc00000))" : "pj_unpack_f64(UINT64_C(0x7ff8000000000000))";
            if (!pj_append(s,"volatile %s pj_reduced_%d = pj_count_%d ? pj_acc_%d / (%s)pj_count_%d : %s;\n",
                type,id,id,id,acc_type,id,nan)) return NULL;
        } else if (!pj_append(s,"volatile %s pj_reduced_%d = pj_acc_%d;\n",type,id,id)) return NULL;
        snprintf(leaf,sizeof(leaf),"pj_reduced_%d",id);
        return strdup(leaf);
    }
    if (TYPE_MASK(n->type) == ME_CONSTANT) {
        if (n->flags & ME_EXPR_FLAG_INTEGER_LITERAL) {
            if (pj_integer(n->dtype)) {
                snprintf(leaf,sizeof(leaf),"pj_%s(%s%lluULL)",type,
                    n->flags & ME_EXPR_FLAG_NEGATIVE_LITERAL ? "0ULL-" : "",
                    (unsigned long long)n->integer_magnitude);
            }
            else {
                snprintf(leaf,sizeof(leaf),"((%s)(%s(%s)%lluULL))",type,
                    n->flags & ME_EXPR_FLAG_NEGATIVE_LITERAL ? "-" : "",type,
                    (unsigned long long)n->integer_magnitude);
            }
        }
        else {
            if (pj_integer(n->dtype)) return NULL;
            double value = n->dtype == ME_FLOAT32 ? (double)n->literal_f32 : n->value;
            if (!isfinite(value)) return NULL;
            snprintf(leaf,sizeof(leaf),"((%s)%a)",type,value);
        }
        /* Integer literal construction has no observable FP exceptions. Keep
         * any checked conversions in the typed tree, but do not force an
         * otherwise constant integer through a volatile load on every lane. */
        if (plain_literal || pj_integral(n->dtype)) return strdup(leaf);
        char literal[300];
        /* A volatile scalar compound literal prevents arithmetic on constants
         * being folded away together with its observable IEEE exceptions. */
        snprintf(literal, sizeof(literal), "((volatile %s){%s})", type, leaf);
        return strdup(literal);
    }
    if (TYPE_MASK(n->type) == ME_VARIABLE) {
        if (!is_synthetic_address(n->bound)) return NULL;
        int index = (int)((const char *)n->bound - synthetic_var_addresses);
        if (index < 0 || index >= p->vars.count) return NULL;
        if (constants && constants->scalar_phase &&
            (!p->vars.uniform[index] || (index >= p->n_inputs && !defined[index]))) return NULL;
        /* ND reserved symbols are recomputed per lane in the kernel preamble. */
        if (p->idx_ndim >= 0 && index == p->idx_ndim) return strdup("nd[0]");
        if (p->idx_flat_idx >= 0 && index == p->idx_flat_idx) return strdup("flat");
        for (int d = 0; d < p->compile_ndims && d < ME_DSL_MAX_NDIM; d++) {
            if (p->idx_i[d] >= 0 && index == p->idx_i[d]) {
                snprintf(leaf,sizeof(leaf),"coord_%d",d);
                return strdup(leaf);
            }
            if (p->idx_n[d] >= 0 && index == p->idx_n[d]) {
                snprintf(leaf,sizeof(leaf),"nd[%d]",1+d);
                return strdup(leaf);
            }
        }
        if (index < p->n_inputs) snprintf(leaf,sizeof(leaf),"pj_input_%d[%s]",index,
            p->vars.uniform[index] || (constants && constants->scalar_phase) ? "0" : "i");
        else if (p->local_slots && p->local_slots[index] >= 0 && defined[index]) {
            snprintf(leaf,sizeof(leaf),p->output_is_scalar && !p->vars.uniform[index] ?
                "local_%d[i]" : "local_%d",index);
        }
        else return NULL;
        return strdup(leaf);
    }
    if (!IS_FUNCTION(n->type) || is_reduction_node(n)) return NULL;
    /* Checked weak intermediates are not modular typed arithmetic. Only audited
     * operators may use the invocation-local checked integer bridge. */
    bool checked_weak = pj_integer(n->dtype) && (n->flags & ME_EXPR_FLAG_WEAK_SCALAR);
    int arity = ARITY(n->type);
    const char *op = me_portable_operator(n);
    const me_expr *arg0 = arity > 0 ? (const me_expr *)n->parameters[0] : NULL;
    const me_expr *arg1 = arity > 1 ? (const me_expr *)n->parameters[1] : NULL;
    bool float_result = pj_float(n->dtype);
    /* Float/Boolean conversions and integer identities are safe C casts.
     * Integral narrowing/widening is modular in 1.1. Float-to-integer conversion
     * uses the checked scalar bridge rather than an unchecked C cast. */
    bool conversion = !n->function && arity == 1;
    bool weak_operand = arg0 &&
        (arg0->flags & (ME_EXPR_FLAG_WEAK_LITERAL | ME_EXPR_FLAG_WEAK_SCALAR)) != 0;
    if (conversion && pj_integer(n->dtype) && weak_operand) {
        uint64_t original, converted;
        if (pj_literal_integer_bits(arg0,arg0->dtype,&original) && pj_literal_integer_bits(arg0,n->dtype,&converted)) {
            char literal[128];
            snprintf(literal,sizeof(literal),"pj_%s(%lluULL)",type,(unsigned long long)converted);
            return strdup(literal);
        }
    }
    bool int_cast = conversion && (n->flags & ME_EXPR_FLAG_NUMPY_1_1) && pj_integer(n->dtype) && pj_integral(arg0->dtype) &&
        arg0->dtype != n->dtype && !weak_operand;
    if (conversion && pj_integer(n->dtype) && arg0->dtype != n->dtype && weak_operand) {
        /* Artifact capture leaves are immutable, but their values arrive only
         * after validation. Defer preparation so constant specialization keeps
         * its fast path and out-of-range immutable leaves keep safe fallback. */
        if ((!constants || !constants->known) && TYPE_MASK(arg0->type) == ME_VARIABLE && is_synthetic_address(arg0->bound) &&
            (arg0->dtype == ME_INT64 || arg0->dtype == ME_BOOL)) {
            int index = (int)((const char *)arg0->bound - synthetic_var_addresses);
            if (index >= 0 && index < p->vars.count && p->vars.uniform[index]) return NULL;
        }
        char *specialized = pj_weak_conversion(n, constants);
        if (specialized) return specialized;
        /* Preserve fail-closed behavior for known out-of-range immutable integer
         * captures. Unknown/computed/floating operands use runtime checks. */
        if (constants && TYPE_MASK(arg0->type) == ME_VARIABLE && is_synthetic_address(arg0->bound) &&
            (arg0->dtype == ME_INT64 || arg0->dtype == ME_BOOL)) {
            int index = (int)((const char *)arg0->bound - synthetic_var_addresses);
            if (index >= 0 && index < constants->count && constants->known[index]) return NULL;
        }
    }
    bool checked_conversion = conversion && pj_integer(n->dtype) &&
        ((arg0->dtype != n->dtype && !int_cast) || weak_operand);
    bool where = op && !strcmp(op,"where") && arity == 3;
    bool comparison = op && is_comparison_node(n) && arity == 2;
    bool integer_comparison = comparison && pj_integral(arg0->dtype) && pj_integral(arg1->dtype);
    /* Keep mixed-domain comparisons on the authoritative exact-bit operation
     * bridge. Host ARM64 system-CC qualification exposed a boundary mismatch
     * in generated mixed comparison code; no floating transport is involved. */
    bool checked_comparison = integer_comparison &&
        pj_unsigned(arg0->dtype) != pj_unsigned(arg1->dtype);
#ifdef __EMSCRIPTEN__
    bool comparison_bridge = false;
#else
    bool comparison_bridge = comparison && !integer_comparison;
#endif
    bool arithmetic = op && float_result &&
        (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*") || !strcmp(op,"/")) &&
        (arity == 2 || (arity == 1 && !strcmp(op,"-")));
    bool integer_arithmetic = op && pj_integer(n->dtype) &&
        (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*")) &&
        (arity == 2 || (arity == 1 && !strcmp(op,"-")));
    bool checked_arithmetic = checked_weak && n->dtype == ME_INT64 && integer_arithmetic;
    /* Unary/binary math, predicates and the divmod/power operators are lowered
     * through host bridges that replay the exact interpreter scalar path. */
    const char *math = me_portable_math_name(n);
    bool predicate = math && arity == 1 && n->dtype == ME_BOOL && arg0 && pj_float(arg0->dtype);
    bool integer_predicate = math && arity == 1 && n->dtype == ME_BOOL && arg0 && pj_integral(arg0->dtype) &&
        (!strcmp(math,"isfinite") || !strcmp(math,"isinf") || !strcmp(math,"isnan") || !strcmp(math,"signbit"));
    bool integer_math = math && arity == 1 && !checked_weak && arg0 && pj_integral(n->dtype) && arg0->dtype == n->dtype &&
        (!strcmp(math,"fabs") || !strcmp(math,"sign") || !strcmp(math,"square") || !strcmp(math,"real") ||
         !strcmp(math,"imag") || !strcmp(math,"conj") || !strcmp(math,"floor") || !strcmp(math,"ceil") ||
          !strcmp(math,"trunc") || !strcmp(math,"round") || !strcmp(math,"rint"));
    /* Typed integral extrema require neither conversion nor checked arithmetic.
     * Keep them in the generated loop instead of rebuilding an interpreter
     * operation per lane. Mixed domains/weak computed results retain the bridge. */
    bool integer_extrema = math && arity == 2 && !checked_weak && arg0 && arg1 &&
        pj_integral(n->dtype) && arg0->dtype == n->dtype && arg1->dtype == n->dtype &&
        (!strcmp(math,"minimum") || !strcmp(math,"maximum") ||
         !strcmp(math,"fmin") || !strcmp(math,"fmax"));
    bool integer_fmod = math && !strcmp(math,"fmod") && arity == 2 && !checked_weak &&
        (n->flags & ME_EXPR_FLAG_NUMPY_1_1) && arg0 && arg1 && pj_integer(n->dtype) &&
        arg0->dtype == n->dtype && arg1->dtype == n->dtype;
    bool native_power = op && !strcmp(op,"**") && arity == 2 && !checked_weak && pj_integer(n->dtype) &&
        (n->flags & ME_EXPR_FLAG_NUMPY_1_1) && arg0 && arg1 && arg0->dtype == n->dtype && arg1->dtype == n->dtype;
    bool unary_math = math && arity == 1 && !predicate && float_result && arg0 && pj_float(arg0->dtype);
    bool binary_math = math && arity == 2 && float_result && arg0 && arg1 &&
        pj_float(arg0->dtype) && pj_float(arg1->dtype);
    bool exponent_math = math && !strcmp(math,"ldexp") && arity == 2 && float_result && arg0 && arg1 &&
        pj_float(arg0->dtype) && arg1->dtype == ME_INT64;
    bool ternary_math = math && !strcmp(math,"fma") && arity == 3 && float_result && arg0 && arg1 &&
        pj_float(arg0->dtype) && pj_float(arg1->dtype) && pj_float(((const me_expr *)n->parameters[2])->dtype);
    bool checked_math = (math && arity >= 1 && arity <= 3 &&
        !predicate && !integer_predicate && !integer_math && !integer_extrema && !integer_fmod && !unary_math && !binary_math && !exponent_math && !ternary_math) ||
        (op && !strcmp(op,"**") && arity == 2 && pj_integer(n->dtype) && !native_power);
    bool float_operator = op && arity == 2 && float_result && arg0 && arg1 &&
        (!strcmp(op,"//") || !strcmp(op,"%") || !strcmp(op,"**")) &&
        pj_float(arg0->dtype) && pj_float(arg1->dtype);
    bool integer_operator = op && pj_integer(n->dtype) &&
        ((arity == 2 && (!strcmp(op,"%") || !strcmp(op,"//") || !strcmp(op,"<<") ||
                         !strcmp(op,">>") || !strcmp(op,"&") || !strcmp(op,"|") || !strcmp(op,"^"))) ||
          (arity == 1 && !strcmp(op,"~")));
    bool boolean_bitwise = op && n->dtype == ME_BOOL &&
        ((arity == 2 && (!strcmp(op,"&") || !strcmp(op,"|") || !strcmp(op,"^") ||
                        !strcmp(op,"+") || !strcmp(op,"*"))) ||
         (arity == 1 && !strcmp(op,"~")));
    bool native_shift = integer_operator && !checked_weak && (n->flags & ME_EXPR_FLAG_NUMPY_1_1) &&
        arity == 2 && (!strcmp(op,"<<") || !strcmp(op,">>"));
    bool native_divmod = integer_operator && !checked_weak && (n->flags & ME_EXPR_FLAG_NUMPY_1_1) &&
        arity == 2 && (!strcmp(op,"//") || !strcmp(op,"%"));
    if (checked_weak) {
        if (!integer_arithmetic && !integer_operator && !checked_math && !checked_conversion && !where) return NULL;
        if (!checked_math && !checked_conversion) { integer_operator = true; integer_arithmetic = false; }
    }
    bool logical = op && ((!strcmp(op,"not") && arity == 1) ||
        ((!strcmp(op,"and") || !strcmp(op,"or")) && arity == 2));
    if (!conversion && !where && !comparison && !arithmetic && !integer_arithmetic &&
        !logical && !predicate && !integer_predicate && !unary_math && !binary_math && !float_operator && !exponent_math && !ternary_math &&
         !integer_operator && !checked_math && !boolean_bitwise && !integer_math && !integer_extrema && !integer_fmod && !native_power) return NULL;
    /* Emit selected operands in their branch, never as eager C call arguments.
     * Every other operation gets sequenced temporaries and an immediate status
     * check, so a failed left operand prevents right-operand participation. */
    if (where || (logical && arity == 2)) {
        char *condition = pj_expr(p,arg0,defined,depth+1,constants,s,next_temp);
        if (!condition) return NULL;
        int id = (*next_temp)++;
        bool is_or = logical && !strcmp(op,"or");
        bool ok = pj_append(s,"%s pj_expr_%d; if (%s(%s)) {\n",type,id,is_or ? "!" : "",condition);
        free(condition);
        if (!ok) return NULL;
        char *selected = pj_expr(p,arg1,defined,depth+1,constants,s,next_temp);
        if (!selected) return NULL;
        ok = pj_append(s,"pj_expr_%d = (%s)(%s); } else {\n",id,type,selected);
        free(selected);
        if (!ok) return NULL;
        if (where) {
            selected = pj_expr(p,n->parameters[2],defined,depth+1,constants,s,next_temp);
            if (!selected) return NULL;
            ok = pj_append(s,"pj_expr_%d = (%s)(%s); }\n",id,type,selected);
            free(selected);
        } else ok = pj_append(s,"pj_expr_%d = %d; }\n",id,is_or ? 1 : 0);
        if (!ok) return NULL;
        snprintf(leaf,sizeof(leaf),"pj_expr_%d",id);
        return strdup(leaf);
    }
    char *args[3] = {0};
    for (int j = 0; j < arity; j++) {
        bool plain = arithmetic && arity == 2 && pj_plain_literal(n->parameters[j],op) &&
            pj_runtime_float(p,n->parameters[1-j],0);
#ifdef __EMSCRIPTEN__
        plain = plain || (comparison && arity == 2 &&
            TYPE_MASK(((const me_expr *)n->parameters[j])->type) == ME_CONSTANT &&
            pj_runtime_float(p,n->parameters[1-j],0));
#endif
        /* Runtime-bound callbacks cannot be folded into compile-time libm
         * answers. Exact nonzero call literals therefore need no barrier. */
        plain = plain || ((predicate || unary_math || binary_math || exponent_math || ternary_math) &&
            pj_plain_literal(n->parameters[j],"+"));
        args[j] = pj_expr_impl(p,n->parameters[j],defined,depth+1,constants,s,next_temp,plain,cse);
        if (!args[j]) {
            for (int k = 0; k < arity; k++) free(args[k]);
            return NULL;
        }
    }
    size_t capacity = 512;
    for (int j = 0; j < arity; j++) capacity += 8*strlen(args[j]);
    char *out = malloc(capacity);
    if (out) {
        if (integer_predicate) {
            if (!strcmp(math,"isfinite")) snprintf(out,capacity,"((_Bool)1)");
            else if (strcmp(math,"signbit") || pj_unsigned(arg0->dtype) || arg0->dtype == ME_BOOL)
                snprintf(out,capacity,"((_Bool)0)");
            else snprintf(out,capacity,"((_Bool)((%s) < 0))",args[0]);
        }
        else if (integer_math) {
            if (!strcmp(math,"imag")) snprintf(out,capacity,"((%s)0)",type);
            else if (n->dtype == ME_BOOL) snprintf(out,capacity,"((_Bool)(%s))",args[0]);
            else if (!strcmp(math,"sign")) {
                if (pj_unsigned(n->dtype)) snprintf(out,capacity,"((%s)((%s) != 0))",type,args[0]);
                else snprintf(out,capacity,"((%s)(((%s) > 0) - ((%s) < 0)))",type,args[0],args[0]);
            }
            else if (!strcmp(math,"square")) snprintf(out,capacity,"pj_%s((uint64_t)(%s)*(uint64_t)(%s))",type,args[0],args[0]);
            else if (!strcmp(math,"fabs") && !pj_unsigned(n->dtype)) snprintf(out,capacity,
                "pj_%s((%s) < 0 ? 0ULL-(uint64_t)(%s) : (uint64_t)(%s))",type,args[0],args[0],args[0]);
            else snprintf(out,capacity,"((%s)(%s))",type,args[0]);
        }
        else if (integer_extrema) {
            snprintf(out,capacity,"((%s)((%s) %s (%s) ? (%s) : (%s)))",type,args[0],
                !strcmp(math,"minimum") || !strcmp(math,"fmin") ? "<" : ">",args[1],args[0],args[1]);
        }
        else if (native_power) {
            int id = (*next_temp)++;
            bool ok = pj_append(s,"uint64_t pj_power_base_%d = (uint64_t)(%s), pj_power_exp_%d = (uint64_t)(%s), pj_power_result_%d = 1;\n",
                id,args[0],id,args[1],id);
            if (!pj_unsigned(n->dtype)) ok = ok && pj_append(s,"if ((%s) < 0) return -5;\n",args[1]);
            ok = ok && pj_append(s,"while (pj_power_exp_%d) { if (pj_power_exp_%d & 1) pj_power_result_%d *= pj_power_base_%d;\n"
                "pj_power_exp_%d >>= 1; if (pj_power_exp_%d) pj_power_base_%d *= pj_power_base_%d; }\n",
                id,id,id,id,id,id,id,id);
            if (!ok) { free(out); out = NULL; }
            else snprintf(out,capacity,"pj_%s(pj_power_result_%d)",type,id);
        }
        else if (checked_conversion || checked_math || checked_comparison) {
            if (p->portable_jit_nchecked == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_nchecked++;
                p->portable_jit_checked[id] = n;
                char *packed[3] = {0};
                bool ok = true;
                for (int j = 0; j < arity; j++) {
                    const me_expr *arg = n->parameters[j];
                    size_t size = strlen(args[j]) + 64;
                    packed[j] = malloc(size);
                    if (!packed[j]) { ok = false; break; }
                    snprintf(packed[j],size,"%s(%s)",arg->dtype == ME_FLOAT32 ? "pj_pack_f32" :
                        arg->dtype == ME_FLOAT64 ? "pj_pack_f64" : "(uint64_t)",args[j]);
                }
                if (ok) {
                    const char *decode = n->dtype == ME_FLOAT32 ? "pj_unpack_f32" :
                        n->dtype == ME_FLOAT64 ? "pj_unpack_f64" : n->dtype == ME_BOOL ? "(_Bool)" : NULL;
                    char integer_decode[64];
                    if (!decode) { snprintf(integer_decode,sizeof(integer_decode),"pj_%s",type); decode = integer_decode; }
                    snprintf(out,capacity,"%s(((pj_checked)inputs[%d])(inputs[%d],%s,%s,%s,&pj_status))",
                        decode,p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF,
                        p->n_inputs+ME_DSL_PORTABLE_JIT_CHECKED_OFF+1+id,
                        packed[0],arity > 1 ? packed[1] : "0ULL",arity > 2 ? packed[2] : "0ULL");
                } else { free(out); out = NULL; }
                for (int j = 0; j < arity; j++) free(packed[j]);
            }
        }
        else if (boolean_bitwise) {
            if (arity == 1) snprintf(out,capacity,"((_Bool)!(%s))",args[0]);
            else snprintf(out,capacity,"((_Bool)((%s) %s (%s)))",args[0],
                !strcmp(op,"+") ? "|" : !strcmp(op,"*") ? "&" : op,args[1]);
        }
        else if (where) snprintf(out,capacity,"((%s)((%s) ? (%s) : (%s)))",type,args[0],args[1],args[2]);
        else if (int_cast) snprintf(out,capacity,"pj_%s((uint64_t)(%s))",type,args[0]);
        else if (conversion) snprintf(out,capacity,"((%s)(%s))",type,args[0]);
        else if (integer_comparison) {
            /* Mixed signed/unsigned comparisons were routed to the checked
             * bridge above. Same-domain integer C comparisons need no helper. */
            snprintf(out,capacity,"((%s) %s (%s))",args[0],op,args[1]);
        }
        else if (checked_arithmetic) {
            int id = (*next_temp)++;
            const char *left = arity == 1 ? "0ULL" : args[0];
            const char *right = arity == 1 ? args[0] : args[1];
            bool ok = pj_append(s,"uint64_t pj_weak_a_%d = (uint64_t)(%s), pj_weak_b_%d = (uint64_t)(%s);\n",
                id,left,id,right);
            if (!strcmp(op,"*")) {
                ok = ok && pj_append(s,"int64_t pj_weak_x_%d = pj_int64_t(pj_weak_a_%d), pj_weak_y_%d = pj_int64_t(pj_weak_b_%d);\n"
                    "if (pj_weak_x_%d > 0 ? (pj_weak_y_%d > 0 ? pj_weak_x_%d > INT64_MAX/pj_weak_y_%d : "
                    "pj_weak_y_%d < INT64_MIN/pj_weak_x_%d) : (pj_weak_x_%d < 0 && "
                    "(pj_weak_y_%d > 0 ? pj_weak_x_%d < INT64_MIN/pj_weak_y_%d : pj_weak_y_%d < INT64_MAX/pj_weak_x_%d))) return -5;\n",
                    id,id,id,id,id,id,id,id,id,id,id,id,id,id,id,id);
            }
            ok = ok && pj_append(s,"uint64_t pj_weak_r_%d = pj_weak_a_%d %s pj_weak_b_%d;\n",id,id,op,id);
            if (strcmp(op,"*")) {
                ok = ok && pj_append(s,!strcmp(op,"+") ?
                    "if ((~(pj_weak_a_%d ^ pj_weak_b_%d) & (pj_weak_a_%d ^ pj_weak_r_%d)) >> 63) return -5;\n" :
                    "if (((pj_weak_a_%d ^ pj_weak_b_%d) & (pj_weak_a_%d ^ pj_weak_r_%d)) >> 63) return -5;\n",
                    id,id,id,id);
            }
            if (!ok) { free(out); out = NULL; }
            else snprintf(out,capacity,"pj_int64_t(pj_weak_r_%d)",id);
        }
        else if (predicate) {
            if (p->portable_jit_npreds == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_npreds++;
                p->portable_jit_preds[id] = n;
                p->portable_jit_pred_fn[id] = dsl_portable_resolve_predicate(n);
                if (p->portable_jit_pred_fn[id]) snprintf(out,capacity,"((_Bool)(((%s)inputs[%d])((%s)(%s))))",
                    arg0->dtype == ME_FLOAT32 ? "pj_predf" : "pj_predd",
                    p->n_inputs+ME_DSL_PORTABLE_JIT_PRED_FN_OFF+id,pj_type(arg0->dtype),args[0]);
                else snprintf(out,capacity,"((_Bool)(((pj_pred)inputs[%d])(inputs[%d],(double)(%s))))",
                    p->n_inputs+ME_DSL_PORTABLE_JIT_PRED_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_PRED_OFF+1+id,args[0]);
            }
        }
        else if (exponent_math) {
            if (p->portable_jit_nmath2 == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_nmath2++, exponent = (*next_temp)++;
                p->portable_jit_math2[id] = n;
                p->portable_jit_math2_fn[id] = dsl_portable_resolve_binary_math(n);
                if (!p->portable_jit_math2_fn[id] || !pj_append(s,
                    "int64_t pj_exponent_%d = (int64_t)(%s); if (pj_exponent_%d < INT_MIN || pj_exponent_%d > INT_MAX) return -5;\n",
                    exponent,args[1],exponent,exponent)) { free(out); out = NULL; }
                else snprintf(out,capacity,"((%s)(((%s)inputs[%d])((%s)(%s),(int)pj_exponent_%d)))",
                    type,n->dtype == ME_FLOAT32 ? "pj_ldexpf" : "pj_ldexpd",
                    p->n_inputs+ME_DSL_PORTABLE_JIT_MATH2_FN_OFF+id,type,args[0],exponent);
            }
        }
        else if (ternary_math) {
            if (p->portable_jit_nmath3 == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_nmath3++;
                p->portable_jit_math3[id] = n;
                p->portable_jit_math3_fn[id] = dsl_portable_resolve_ternary_math(n);
                if (!p->portable_jit_math3_fn[id]) { free(out); out = NULL; }
                else snprintf(out,capacity,"((%s)(((%s)inputs[%d])((%s)(%s),(%s)(%s),(%s)(%s))))",
                    type,n->dtype == ME_FLOAT32 ? "pj_libm3f" : "pj_libm3",
                    p->n_inputs+ME_DSL_PORTABLE_JIT_MATH3_FN_OFF+id,type,args[0],type,args[1],type,args[2]);
            }
        }
        else if (unary_math) {
            if (p->portable_jit_nmath == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_nmath++;
                p->portable_jit_math[id] = n;
                p->portable_jit_math_fn[id] = dsl_portable_resolve_unary_math(n);
                if (p->portable_jit_math_fn[id]) {
                    snprintf(out,capacity,"((%s)(((%s)inputs[%d])((%s)(%s))))",
                        type,n->dtype == ME_FLOAT32 ? "pj_libmf" : "pj_libm",
                        p->n_inputs+ME_DSL_PORTABLE_JIT_MATH1_FN_OFF+id,type,args[0]);
                } else snprintf(out,capacity,"((%s)(((pj_math)inputs[%d])(inputs[%d],(double)(%s))))",
                    type,p->n_inputs+ME_DSL_PORTABLE_JIT_MATH1_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_MATH1_OFF+1+id,args[0]);
            }
        }
        else if (binary_math || float_operator) {
            if (p->portable_jit_nmath2 == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_nmath2++;
                p->portable_jit_math2[id] = n;
                p->portable_jit_math2_fn[id] = dsl_portable_resolve_binary_math(n);
                if (p->portable_jit_math2_fn[id]) {
                    snprintf(out,capacity,"((%s)(((%s)inputs[%d])((%s)(%s),(%s)(%s))))",
                        type,n->dtype == ME_FLOAT32 ? "pj_libm2f" : "pj_libm2",
                        p->n_inputs+ME_DSL_PORTABLE_JIT_MATH2_FN_OFF+id,type,args[0],type,args[1]);
                } else snprintf(out,capacity,"((%s)(((pj_math2)inputs[%d])(inputs[%d],(double)(%s),(double)(%s))))",
                    type,p->n_inputs+ME_DSL_PORTABLE_JIT_MATH2_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_MATH2_OFF+1+id,args[0],args[1]);
            }
        }
        else if (native_shift) {
            int bits = (int)(dtype_size(n->dtype)*8);
            bool right = !strcmp(op,">>");
            if (pj_unsigned(n->dtype)) {
                snprintf(out,capacity,"pj_%s((uint64_t)(%s) >= %d ? 0ULL : ((uint64_t)(%s) %s (uint64_t)(%s)))",
                    type,args[1],bits,args[0],op,args[1]);
            } else if (!right) {
                snprintf(out,capacity,"pj_%s((uint64_t)(%s) >= %d ? 0ULL : ((uint64_t)(%s) << (uint64_t)(%s)))",
                    type,args[1],bits,args[0],args[1]);
            } else {
                snprintf(out,capacity,"pj_%s((uint64_t)(%s) >= %d ? ((%s) < 0 ? UINT64_MAX : 0ULL) : "
                    "((%s) < 0 ? ~((~(uint64_t)(%s)) >> (uint64_t)(%s)) : (uint64_t)(%s) >> (uint64_t)(%s)))",
                    type,args[1],bits,args[0],args[0],args[0],args[1],args[0],args[1]);
            }
        }
        else if (native_divmod || integer_fmod) {
            int id = (*next_temp)++;
            bool modulo = integer_fmod || !strcmp(op,"%");
            bool ok;
            if (pj_unsigned(n->dtype)) {
                ok = pj_append(s,"uint64_t pj_div_left_%d = (uint64_t)(%s), pj_div_right_%d = (uint64_t)(%s);\n"
                    "uint64_t pj_div_bits_%d = pj_div_right_%d ? pj_div_left_%d %s pj_div_right_%d : 0ULL;\n",
                    id,args[0],id,args[1],id,id,id,modulo ? "%" : "/",id);
            } else {
                int bits = (int)(dtype_size(n->dtype)*8);
                char overflow[64];
                if (modulo) strcpy(overflow,"0ULL");
                else snprintf(overflow,sizeof(overflow),"(uint64_t)pj_div_left_%d",id);
                ok = pj_append(s,"int64_t pj_div_left_%d = (int64_t)(%s), pj_div_right_%d = (int64_t)(%s);\n"
                    "uint64_t pj_div_bits_%d = 0;\n",id,args[0],id,args[1],id);
                ok = ok && pj_append(s,
                    "if (pj_div_right_%d) { if ((uint64_t)pj_div_left_%d == (UINT64_MAX << %d) && pj_div_right_%d == -1) "
                    "pj_div_bits_%d = %s; else {\n",id,id,bits-1,id,id,overflow);
                ok = ok && pj_append(s,
                    "int64_t pj_div_q_%d = pj_div_left_%d / pj_div_right_%d;\n"
                    "int64_t pj_div_r_%d = pj_div_left_%d %% pj_div_right_%d;\n",id,id,id,id,id,id);
                if (!integer_fmod) ok = ok && pj_append(s,
                    "if (pj_div_r_%d && ((pj_div_left_%d < 0) != (pj_div_right_%d < 0))) "
                    "{ pj_div_q_%d--; pj_div_r_%d += pj_div_right_%d; }\n",id,id,id,id,id,id);
                ok = ok && pj_append(s,"pj_div_bits_%d = (uint64_t)pj_div_%s_%d; } }\n",id,modulo ? "r" : "q",id);
            }
            if (!ok) { free(out); out = NULL; }
            else snprintf(out,capacity,"pj_%s(pj_div_bits_%d)",type,id);
        }
        else if (integer_operator) {
            if (p->portable_jit_niops == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_niops++;
                p->portable_jit_iops[id] = n;
                snprintf(out,capacity,"pj_%s((uint64_t)(((pj_iop)inputs[%d])(inputs[%d],(uint64_t)(%s),(uint64_t)(%s),&pj_status)))",
                    type,p->n_inputs+ME_DSL_PORTABLE_JIT_IOP_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_IOP_OFF+1+id,args[0],arity > 1 ? args[1] : "0ULL");
            }
        }
        else if (integer_arithmetic) {
            const char *carrier = pj_arithmetic_carrier(n->dtype);
            if (arity == 1) snprintf(out,capacity,"pj_%s((%s)0-(%s)(%s))",
                type,carrier,carrier,args[0]);
            else snprintf(out,capacity,"pj_%s((%s)(%s) %s (%s)(%s))",
                type,carrier,args[0],op,carrier,args[1]);
        }
        else if (comparison_bridge) {
            if (p->portable_jit_ncomparisons == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_ncomparisons++;
                p->portable_jit_comparisons[id] = n;
                int temporary = (*next_temp)++;
                bool ok = pj_append(s,
                    "union { double f; uint64_t u; } pj_cmp_a_%d, pj_cmp_b_%d;\n"
                    "pj_cmp_a_%d.f = (double)(%s); pj_cmp_b_%d.f = (double)(%s);\n"
                    "_Bool pj_cmp_result_%d;\n",temporary,temporary,temporary,args[0],temporary,args[1],temporary);
                ok = ok && pj_append(s,
                    "if ((pj_cmp_a_%d.u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000) || "
                    "(pj_cmp_b_%d.u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000))\n"
                    "pj_cmp_result_%d = ((pj_cmp)inputs[%d])(inputs[%d],pj_cmp_a_%d.f,pj_cmp_b_%d.f);\n"
                    "else pj_cmp_result_%d = pj_cmp_a_%d.f %s pj_cmp_b_%d.f;\n",
                    temporary,temporary,temporary,p->n_inputs+ME_DSL_PORTABLE_JIT_CMP_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_CMP_OFF+1+id,temporary,temporary,
                    temporary,temporary,op,temporary);
                if (!ok) { free(out); out = NULL; }
                else snprintf(out,capacity,"pj_cmp_result_%d",temporary);
            }
        }
        else if (arity == 1) snprintf(out,capacity,"((%s)(%s(%s)))",type,logical ? "!" : "-",args[0]);
        else snprintf(out,capacity,"((%s)((%s) %s (%s)))",type,args[0],
            logical ? (!strcmp(op,"and") ? "&&" : "||") : op,args[1]);
    }
    for (int j = 0; j < arity; j++) free(args[j]);
    if (!out) return NULL;
#ifdef __EMSCRIPTEN__
    if (!cse_slot || cse_slot->uses <= 1) {
        return out;
    }
#endif
    int id = (*next_temp)++;
    bool ok = pj_append(s,"%s pj_expr_%d = %s; PJ_STATUS_CHECK();\n",type,id,out);
    free(out);
    if (!ok) return NULL;
    snprintf(leaf,sizeof(leaf),"pj_expr_%d",id);
    if (cse_slot) {
        snprintf(cse_slot->value,sizeof(cse_slot->value),"%s",leaf);
    }
    else if (cse_eligible && cse->count < (int)(sizeof(cse->entries)/sizeof(cse->entries[0]))) {
        pj_cse_entry *entry = &cse->entries[cse->count++];
        entry->node = n; entry->hash = cse_hash; entry->uses = 1;
        snprintf(entry->value,sizeof(entry->value),"%s",leaf);
    }
    return strdup(leaf);
}

static bool pj_append(pj_text *s, const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    va_list copy;
    va_copy(copy, ap);
    int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0 || s->used + (size_t)needed > 1048576) {
        va_end(ap);
        return false;
    }
    size_t capacity = s->used + (size_t)needed + 1;
    if (capacity > s->capacity) {
        char *next = realloc(s->text, capacity);
        if (!next) {
            va_end(ap);
            return false;
        }
        s->text = next;
        s->capacity = capacity;
    }
    vsnprintf(s->text+s->used, s->capacity-s->used, format, ap);
    va_end(ap);
    s->used += (size_t)needed;
    return true;
}

/* Lane-local statements preserve assignment rounding and observable exceptions.
 * Definite assignment is checked again here; unsupported flow remains interpreted. */
typedef struct {
    int *next_label;
    int *next_temp;
    int loop_label;
} pj_flow;

/* Only paths reaching the join constrain definite assignment. Loops may execute
 * zero times; their exits never establish definitions after the loop. */
static bool pj_falls_through(const me_dsl_compiled_block *block, int depth) {
    if (depth > 64) return true;
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_compiled_stmt *stmt = block->stmts[i];
        if (stmt->kind == ME_DSL_STMT_RETURN) return false;
        if ((stmt->kind == ME_DSL_STMT_BREAK || stmt->kind == ME_DSL_STMT_CONTINUE) &&
            !stmt->as.flow.cond.expr) return false;
        if (stmt->kind == ME_DSL_STMT_IF && stmt->as.if_stmt.has_else) {
            bool live = pj_falls_through(&stmt->as.if_stmt.then_block,depth+1) ||
                pj_falls_through(&stmt->as.if_stmt.else_block,depth+1);
            for (int j = 0; j < stmt->as.if_stmt.n_elifs; j++) {
                live |= pj_falls_through(&stmt->as.if_stmt.elif_branches[j].block,depth+1);
            }
            if (!live) return false;
        }
    }
    return true;
}

static bool pj_expr_uses_var(const me_expr *n, int var_index, int depth) {
    if (!n || depth > 128) return true;
    if (TYPE_MASK(n->type) == ME_VARIABLE) {
        return n->bound == synthetic_var_addresses + var_index;
    }
    if (!IS_FUNCTION(n->type)) return false;
    int arity = ARITY(n->type);
    for (int i = 0; i < arity && i < 3; i++) {
        if (pj_expr_uses_var((const me_expr *)n->parameters[i], var_index, depth + 1)) return true;
    }
    return false;
}

/* A terminal same-width modular operation can store its unsigned bit pattern
 * directly. Corresponding signed/unsigned types may alias in C99. This avoids
 * per-lane representation temporaries with non-inlining compilers such as TCC;
 * checked conversions, computed operands and uniform captures use normal lowering. */
static int pj_direct_integer_return(const me_dsl_compiled_program *p, const me_expr *n,
                                    const bool *defined, const pj_constants *constants, pj_text *s) {
    if (!n || !pj_integer(n->dtype) || n->dtype != p->output_dtype) return 0;
    pj_text integer = {0};
    if (!pj_integer_tree(p,n,defined,constants,n->dtype,&integer,0)) { free(integer.text); return 0; }
    const char *type = pj_unsigned_type(n->dtype);
    bool ok = pj_append(s,"((%s *)output)[i] = (%s)(%s); goto pj_lane_done;\n",type,type,integer.text);
    free(integer.text);
    return ok ? 1 : -1;
}

static bool pj_block(me_dsl_compiled_program *p, const me_dsl_compiled_block *block,
                     bool *defined, pj_text *s, int depth, const pj_constants *constants,
                     pj_flow flow) {
    if (depth > 64) return false;
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_compiled_stmt *stmt = block->stmts[i];
        if (stmt->kind == ME_DSL_STMT_ASSIGN || stmt->kind == ME_DSL_STMT_RETURN) {
            bool assignment = stmt->kind == ME_DSL_STMT_ASSIGN;
            const me_expr *node = assignment ? stmt->as.assign.value.expr : stmt->as.return_stmt.expr.expr;
            if (!assignment && !(constants && constants->scalar_phase)) {
                int direct = pj_direct_integer_return(p,node,defined,constants,s);
                if (direct) return direct > 0;
            }
            if (assignment && constants && constants->scalar_phase) {
                int index = p->local_var_indices[stmt->as.assign.local_slot];
                if (!p->vars.uniform[index]) {
                    /* Materialize whole-participation array assignments in
                     * statement order. Branch/loop-local arrays need per-lane
                     * initialization tracking and deliberately retain fallback. */
                    if (depth || node->dtype != p->vars.dtypes[index]) return false;
                    pj_constants mapped = *constants;
                    mapped.scalar_phase = false;
                    if (!pj_append(s,"for (int64_t i=0; i<count; i++) { if (mask && !mask[i]) continue;\n%s",
                        constants->nd_preamble ? constants->nd_preamble : "")) return false;
                    char *lane = pj_expr(p,node,defined,0,&mapped,s,flow.next_temp);
                    if (!lane) return false;
                    bool ok = pj_append(s,"local_%d[i] = %s; PJ_STATUS_CHECK();\n}\n",index,lane);
                    free(lane);
                    if (!ok) return false;
                    defined[index] = true;
                    continue;
                }
            }
            char *value = pj_expr(p,node,defined,0,constants,s,flow.next_temp);
            if (!value) {
                dsl_tracef("portable jit ineligible: expression lowering failed dtype=%d",(int)node->dtype);
                return false;
            }
            bool ok;
            if (assignment) {
                int index = p->local_var_indices[stmt->as.assign.local_slot];
                ok = node->dtype == p->vars.dtypes[index] &&
                    pj_append(s,"local_%d = %s; PJ_STATUS_CHECK();\n",index,value);
                defined[index] = true;
            }
            else if (constants && constants->scalar_phase) {
                ok = pj_append(s,"((%s *)output)[0] = %s; PJ_STATUS_CHECK(); return 0;\n",
                    pj_type(p->output_dtype),value);
            } else {
                ok = pj_append(s,"((%s *)output)[i] = %s; PJ_STATUS_CHECK(); goto pj_lane_done;\n",
                    pj_type(p->output_dtype),value);
            }
            free(value);
            if (!ok) return false;
            if (!assignment) return true;
        }
        else if (stmt->kind == ME_DSL_STMT_IF) {
            bool merged[ME_MAX_VARS], branch[ME_MAX_VARS];
            memcpy(merged,defined,sizeof(merged));
            int alternatives = 1 + stmt->as.if_stmt.n_elifs;
            bool live = false;
            for (int j = 0; j <= alternatives; j++) {
                const me_dsl_compiled_block *body;
                if (j < alternatives) {
                    const me_expr *cond = j == 0 ? stmt->as.if_stmt.cond.expr :
                        stmt->as.if_stmt.elif_branches[j-1].cond.expr;
                    /* Nest else branches so condition preludes execute only
                     * after preceding conditions are false. */
                    if (j && !pj_append(s,"else {\n")) return false;
                    char *value = pj_expr(p,cond,defined,0,constants,s,flow.next_temp);
                    if (!value) return false;
                    /* Test status after evaluating the condition, before either
                     * branch participates. Keep else-if chains syntactically intact. */
                    bool ok = pj_append(s,"if (%s) { PJ_STATUS_CHECK();\n",value);
                    free(value);
                    if (!ok) return false;
                    body = j == 0 ? &stmt->as.if_stmt.then_block : &stmt->as.if_stmt.elif_branches[j-1].block;
                }
                else {
                    if (!pj_append(s,"else { PJ_STATUS_CHECK();\n")) return false;
                    body = &stmt->as.if_stmt.else_block;
                }
                memcpy(branch,defined,sizeof(branch));
                if (!pj_block(p,body,branch,s,depth+1,constants,flow) || !pj_append(s,"}\n")) return false;
                if (pj_falls_through(body,depth+1)) {
                    if (!live) memcpy(merged,branch,sizeof(merged));
                    else for (int k = 0; k < p->vars.count; k++) merged[k] &= branch[k];
                    live = true;
                }
            }
            for (int j = 1; j < alternatives; j++) if (!pj_append(s,"}\n")) return false;
            memcpy(defined,merged,sizeof(merged));
            if (!live) return true;
        }
        else if (stmt->kind == ME_DSL_STMT_FOR) {
            const me_expr *nodes[] = {stmt->as.for_loop.start.expr,
                stmt->as.for_loop.stop.expr, stmt->as.for_loop.step.expr};
            int index = p->local_var_indices[stmt->as.for_loop.loop_var_slot];
            uint64_t ubounds[3] = {0};
            if (p->vars.dtypes[index] == ME_INT64 &&
                nodes[0] && pj_integral(nodes[0]->dtype) && !pj_unsigned(nodes[0]->dtype) &&
                nodes[1] && pj_integral(nodes[1]->dtype) && !pj_unsigned(nodes[1]->dtype) &&
                nodes[2] && pj_integral(nodes[2]->dtype) && !pj_unsigned(nodes[2]->dtype) &&
                pj_literal_integer_bits(nodes[0],ME_INT64,&ubounds[0]) &&
                pj_literal_integer_bits(nodes[1],ME_INT64,&ubounds[1]) &&
                pj_literal_integer_bits(nodes[2],ME_INT64,&ubounds[2])) {
                int64_t cstart = (int64_t)ubounds[0], cstop = (int64_t)ubounds[1], cstep = (int64_t)ubounds[2];
                if (cstep != 0) {
                    int trips = 0;
                    int64_t iter = cstart, last_iter = cstart;
                    while (cstep > 0 ? iter < cstop : iter > cstop) {
                        last_iter = iter;
                        trips++;
                        if (trips > 64) break;
                        if (cstep > 0 ? iter > INT64_MAX - cstep : iter < INT64_MIN - cstep) break;
                        iter += cstep;
                    }
                    if (trips == 0) continue;
                    if (trips <= 64 && !(constants && constants->scalar_phase) &&
                        stmt->as.for_loop.body.nstmts == 1 &&
                        stmt->as.for_loop.body.stmts[0] &&
                        stmt->as.for_loop.body.stmts[0]->kind == ME_DSL_STMT_ASSIGN) {
                        const me_dsl_compiled_stmt *only = stmt->as.for_loop.body.stmts[0];
                        int target_idx = p->local_var_indices[only->as.assign.local_slot];
                        const me_expr *rhs = only->as.assign.value.expr;
                        const char *op = rhs && IS_FUNCTION(rhs->type) && ARITY(rhs->type) == 2 ?
                            me_portable_operator(rhs) : NULL;
                        if (target_idx >= 0 && target_idx < p->vars.count &&
                            target_idx != index && defined[target_idx] &&
                            !(p->output_is_scalar && !p->vars.uniform[target_idx]) &&
                            rhs && pj_float(rhs->dtype) && rhs->dtype == p->vars.dtypes[target_idx] &&
                            op && (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*") || !strcmp(op,"/"))) {
                            const me_expr *a0 = (const me_expr *)rhs->parameters[0];
                            const me_expr *a1 = (const me_expr *)rhs->parameters[1];
                            bool left_is_target = TYPE_MASK(a0->type) == ME_VARIABLE &&
                                a0->bound == synthetic_var_addresses + target_idx &&
                                a0->dtype == rhs->dtype;
                            bool right_is_target = !left_is_target &&
                                TYPE_MASK(a1->type) == ME_VARIABLE &&
                                a1->bound == synthetic_var_addresses + target_idx &&
                                a1->dtype == rhs->dtype;
                            const me_expr *other = left_is_target ? a1 : (right_is_target ? a0 : NULL);
                            if (other && other->dtype == rhs->dtype &&
                                !pj_expr_uses_var(other,target_idx,0) &&
                                !pj_expr_uses_var(other,index,0)) {
                                char *other_val = pj_expr(p,other,defined,0,constants,s,flow.next_temp);
                                if (!other_val) return false;
                                int step_id = (*flow.next_temp)++;
                                const char *ttype = pj_type(rhs->dtype);
                                bool next_is_return = (i + 1 < block->nstmts &&
                                    block->stmts[i + 1] &&
                                    block->stmts[i + 1]->kind == ME_DSL_STMT_RETURN);
                                const me_expr *ret_expr = next_is_return ?
                                    block->stmts[i + 1]->as.return_stmt.expr.expr : NULL;
                                bool fuse_return = next_is_return && ret_expr &&
                                    TYPE_MASK(ret_expr->type) == ME_VARIABLE &&
                                    ret_expr->bound == synthetic_var_addresses + target_idx &&
                                    ret_expr->dtype == p->output_dtype &&
                                    ret_expr->dtype == rhs->dtype;
                                bool need_loop_var = !next_is_return ||
                                    (ret_expr && pj_expr_uses_var(ret_expr,index,0));
                                bool ok = pj_append(s,"const %s pj_for_rhs_%d = %s; PJ_STATUS_CHECK();\n",
                                    ttype,step_id,other_val);
                                free(other_val);
                                if (!ok) return false;
                                if (need_loop_var &&
                                    !pj_append(s,"local_%d = %lldLL;\n",index,(long long)last_iter)) {
                                    return false;
                                }
                                if (fuse_return ? !pj_append(s,"((%s *)output)[i] = ",ttype) :
                                                  !pj_append(s,"local_%d = ",target_idx)) {
                                    return false;
                                }
                                for (int t = 0; t < trips; t++) {
                                    if (!pj_append(s,left_is_target ? "((%s)(" : "((%s)(pj_for_rhs_%d %s ",
                                        ttype,step_id,op)) return false;
                                }
                                if (!pj_append(s,"local_%d",target_idx)) return false;
                                for (int t = 0; t < trips; t++) {
                                    if (!pj_append(s,left_is_target ? " %s pj_for_rhs_%d))" : "))",
                                        op,step_id)) return false;
                                }
                                if (fuse_return) {
                                    return pj_append(s,"; goto pj_lane_done;\n");
                                }
                                if (!pj_append(s,";\n")) return false;
                                continue;
                            }
                        }
                    }
                }
            }
            char *values[3] = {0};
            bool ok = true;
            for (int j = 0; j < 3; j++) {
                /* The compiler inserts authoritative checked int64 conversions
                 * for floating/unsigned bounds before lowering. */
                if (!nodes[j] || !pj_integral(nodes[j]->dtype) || pj_unsigned(nodes[j]->dtype)) {
                    ok = false;
                    break;
                }
                values[j] = pj_expr(p,nodes[j],defined,0,constants,s,flow.next_temp);
                if (!values[j]) { ok = false; break; }
            }
            int label = (*flow.next_label)++;
            ok = ok && p->vars.dtypes[index] == ME_INT64 && pj_append(s,
                "{ int64_t pj_iter_%d = %s; int64_t pj_stop_%d = %s; int64_t pj_step_%d = %s;\n"
                "if (pj_status) return pj_status; if (!pj_step_%d) return -5;\n"
                "while (pj_step_%d > 0 ? pj_iter_%d < pj_stop_%d : pj_iter_%d > pj_stop_%d) {\n"
                "local_%d = pj_iter_%d;\n",
                label,values[0],label,values[1],label,values[2],label,
                label,label,label,label,label,index,label);
            for (int j = 0; j < 3; j++) free(values[j]);
            if (!ok) return false;
            bool body_defined[ME_MAX_VARS];
            memcpy(body_defined,defined,sizeof(body_defined));
            body_defined[index] = true;
            pj_flow loop_flow = {flow.next_label,flow.next_temp,label};
            if (!pj_block(p,&stmt->as.for_loop.body,body_defined,s,depth+1,constants,loop_flow)) return false;
            /* Continue performs advancement; overflow exhausts the range just as
             * in the interpreter, without executing overflowing signed addition. */
            if (!pj_append(s,
                "pj_continue_%d: ;\n"
                "if (pj_step_%d > 0 ? pj_iter_%d > INT64_MAX-pj_step_%d : pj_iter_%d < INT64_MIN-pj_step_%d) break;\n"
                "pj_iter_%d += pj_step_%d;\n} pj_break_%d: ; }\n",
                label,label,label,label,label,label,label,label,label)) return false;
            /* A range can be empty. Newly initialized locals cannot be assumed
             * defined afterwards; retain safe fallback for such later reads. */
        }
        else if (stmt->kind == ME_DSL_STMT_WHILE) {
            int label = (*flow.next_label)++;
            pj_flow loop_flow = {flow.next_label,flow.next_temp,label};
            bool body_defined[ME_MAX_VARS];
            memcpy(body_defined,defined,sizeof(body_defined));
            if (!pj_append(s,"{ uint64_t pj_count_%d = 0; while (1) {\npj_continue_%d: ;\n",label,label)) return false;
            me_dsl_compiled_block prefix = stmt->as.while_loop.body;
            prefix.nstmts = stmt->as.while_loop.condition_nstmts;
            if (!pj_block(p,&prefix,body_defined,s,depth+1,constants,loop_flow)) return false;
            char *condition = pj_expr(p,stmt->as.while_loop.cond.expr,body_defined,0,constants,s,flow.next_temp);
            if (!condition) return false;
            bool ok = pj_append(s,
                "int pj_cond_%d = (%s); if (pj_status) return pj_status; if (!pj_cond_%d) break;\n"
                "if (pj_cap > 0 && pj_count_%d >= (uint64_t)pj_cap) return -5;\n"
                "pj_count_%d++;\n",label,condition,label,label,label);
            free(condition);
            if (!ok) return false;
            me_dsl_compiled_block body = stmt->as.while_loop.body;
            body.stmts += prefix.nstmts;
            body.nstmts -= prefix.nstmts;
            if (!pj_block(p,&body,body_defined,s,depth+1,constants,loop_flow) ||
                !pj_append(s,"} pj_break_%d: ; }\n",label)) return false;
        }
        else if (stmt->kind == ME_DSL_STMT_BREAK || stmt->kind == ME_DSL_STMT_CONTINUE) {
            if (flow.loop_label < 0) return false;
            const char *target = stmt->kind == ME_DSL_STMT_BREAK ? "break" : "continue";
            if (stmt->as.flow.cond.expr) {
                char *condition = pj_expr(p,stmt->as.flow.cond.expr,defined,0,constants,s,flow.next_temp);
                if (!condition) return false;
                bool ok = pj_append(s,"if (%s) { if (pj_status) return pj_status; goto pj_%s_%d; }\n"
                    "if (pj_status) return pj_status;\n",condition,target,flow.loop_label);
                free(condition);
                if (!ok) return false;
            }
            else {
                if (!pj_append(s,"goto pj_%s_%d;\n",target,flow.loop_label)) return false;
                return true;
            }
        }
        else return false;
    }
    return true;
}

/* Qualified numeric reductions in a scalar recipe. Reduction loops are
 * emitted at their expression position, not hoisted ahead of checked/lazy
 * operands. Independent scans are intentionally not fused. Scalar-phase input
 * reads must be uniform captures or previously initialized scalar locals. */
static bool pj_count_scalar_reductions(const me_expr *n, int *count, int depth) {
    if (!n || depth > 128) return false;
    if (!IS_FUNCTION(n->type)) return true;
    if (is_reduction_node(n)) {
        ++*count;
        if (!pj_supported_reduction(n)) return false;
    }
    int arity = ARITY(n->type);
    if (arity < 1 || arity > 3) return false;
    for (int j = 0; j < arity; j++) {
        if (!pj_count_scalar_reductions(n->parameters[j],count,depth+1)) return false;
    }
    return true;
}

static bool pj_scalar_block(const me_dsl_compiled_block *block, int *count, int depth) {
    if (depth > 64) return false;
    for (int j = 0; j < block->nstmts; j++) {
        const me_dsl_compiled_stmt *stmt = block->stmts[j];
        const me_expr *expr;
        if (stmt->kind == ME_DSL_STMT_ASSIGN) expr = stmt->as.assign.value.expr;
        else if (stmt->kind == ME_DSL_STMT_RETURN) expr = stmt->as.return_stmt.expr.expr;
        else if (stmt->kind == ME_DSL_STMT_IF) {
            if (!pj_count_scalar_reductions(stmt->as.if_stmt.cond.expr,count,0) ||
                !pj_scalar_block(&stmt->as.if_stmt.then_block,count,depth+1) ||
                !pj_scalar_block(&stmt->as.if_stmt.else_block,count,depth+1)) return false;
            for (int k = 0; k < stmt->as.if_stmt.n_elifs; k++) {
                if (!pj_count_scalar_reductions(stmt->as.if_stmt.elif_branches[k].cond.expr,count,0) ||
                    !pj_scalar_block(&stmt->as.if_stmt.elif_branches[k].block,count,depth+1)) return false;
            }
            continue;
        }
        else if (stmt->kind == ME_DSL_STMT_FOR) {
            if (!pj_count_scalar_reductions(stmt->as.for_loop.start.expr,count,0) ||
                !pj_count_scalar_reductions(stmt->as.for_loop.stop.expr,count,0) ||
                !pj_count_scalar_reductions(stmt->as.for_loop.step.expr,count,0) ||
                !pj_scalar_block(&stmt->as.for_loop.body,count,depth+1)) return false;
            continue;
        }
        else if (stmt->kind == ME_DSL_STMT_WHILE) {
            if (!pj_count_scalar_reductions(stmt->as.while_loop.cond.expr,count,0) ||
                !pj_scalar_block(&stmt->as.while_loop.body,count,depth+1)) return false;
            continue;
        }
        else if (stmt->kind == ME_DSL_STMT_BREAK || stmt->kind == ME_DSL_STMT_CONTINUE) {
            expr = stmt->as.flow.cond.expr;
            if (!expr) continue;
        }
        else return false;
        if (!pj_count_scalar_reductions(expr,count,0)) return false;
    }
    return true;
}

#ifndef __EMSCRIPTEN__

static bool pj_arith_bridge_enabled(void) {
    const char *env = getenv("ME_DSL_JIT_ARITH_BRIDGE");
    if (env && strcmp(env, "0") == 0) return false;
    return true;
}

typedef enum {
    PJ_ARITH_NODE_VAR,
    PJ_ARITH_NODE_UNIFORM_VAR,
    PJ_ARITH_NODE_LITERAL,
    PJ_ARITH_NODE_UNARY,
    PJ_ARITH_NODE_BINARY
} pj_arith_node_kind;

typedef struct {
    const me_expr *expr;
    pj_arith_node_kind kind;
    int op;
    int left;
    int right;
    bool uniform;
    int uses;
    int temp_slot;
    int var_index;
    int scalar_id;
} pj_arith_node;

typedef struct {
    pj_arith_node nodes[64];
    int count;
    int root;
    int ntemps;
    int nnodes;
    int uniform_ops;
    me_dtype dtype;
    const char *carrier;
    const char *carrier_type;
    const char *var_type;
    int strip;
} pj_arith_dag;

static int pj_arith_find_or_add_node(pj_arith_dag *dag, const me_dsl_compiled_program *p,
                                     const me_expr *n, const char **reason) {
    if (!n) {
        *reason = "unsupported-leaf";
        return -1;
    }
    for (int i = 0; i < dag->count; i++) {
        if (pj_cse_equal(p, dag->nodes[i].expr, n, 0)) {
            dag->nodes[i].uses++;
            return i;
        }
    }
    if (dag->count >= 64) {
        *reason = "too-many-temps";
        return -1;
    }
    if (n->dtype != dag->dtype) {
        *reason = "mixed-dtype";
        return -1;
    }


    int kind = TYPE_MASK(n->type);
    if (kind == ME_VARIABLE) {
        if (!is_synthetic_address(n->bound)) {
            *reason = "unsupported-leaf";
            return -1;
        }
        int var_idx = (int)((const char *)n->bound - synthetic_var_addresses);
        if (var_idx < 0 || var_idx >= p->vars.count) {
            *reason = "unsupported-leaf";
            return -1;
        }
        if (dag->count >= 64) {
            *reason = "too-many-temps";
            return -1;
        }
        int idx = dag->count++;
        pj_arith_node *node = &dag->nodes[idx];
        node->expr = n;
        node->uses = 1;
        node->temp_slot = -1;
        node->left = -1;
        node->right = -1;
        node->scalar_id = -1;
        node->var_index = var_idx;
        if (p->vars.uniform[var_idx] || (n->flags & ME_EXPR_FLAG_UNIFORM_INPUT)) {
            node->kind = PJ_ARITH_NODE_UNIFORM_VAR;
            node->uniform = true;
        } else {
            node->kind = PJ_ARITH_NODE_VAR;
            node->uniform = false;
        }
        return idx;
    }

    if (kind == ME_CONSTANT) {
        if (pj_float(dag->dtype)) {
            double val = (dag->dtype == ME_FLOAT32) ? (double)n->literal_f32 : n->value;
            if (!isfinite(val)) {
                *reason = "unsupported-leaf";
                return -1;
            }
        } else if (pj_integer(dag->dtype)) {
            if (!(n->flags & ME_EXPR_FLAG_INTEGER_LITERAL)) {
                *reason = "unsupported-leaf";
                return -1;
            }
        } else {
            *reason = "unsupported-leaf";
            return -1;
        }
        if (dag->count >= 64) {
            *reason = "too-many-temps";
            return -1;
        }
        int idx = dag->count++;
        pj_arith_node *node = &dag->nodes[idx];
        node->expr = n;
        node->uses = 1;
        node->temp_slot = -1;
        node->left = -1;
        node->right = -1;
        node->scalar_id = -1;
        node->kind = PJ_ARITH_NODE_LITERAL;
        node->uniform = true;
        return idx;
    }

    if (IS_FUNCTION(n->type)) {
        if (!n->function) {
            *reason = "conversion";
            return -1;
        }
        if (is_reduction_node(n)) {
            *reason = "unsupported-op";
            return -1;
        }
        if (n->flags & ME_EXPR_FLAG_WEAK_SCALAR) {
            *reason = "conversion";
            return -1;
        }

        const char *op = me_portable_operator(n);
        if (!op) {
            *reason = "unsupported-op";
            return -1;
        }
        int arity = ARITY(n->type);
        if (arity == 1) {
            if (strcmp(op, "-") != 0) {
                *reason = "unsupported-op";
                return -1;
            }
            int left_idx = pj_arith_find_or_add_node(dag, p, (const me_expr *)n->parameters[0], reason);
            if (left_idx < 0) return -1;
            if (dag->count >= 64) {
                *reason = "too-many-temps";
                return -1;
            }
            int idx = dag->count++;
            pj_arith_node *node = &dag->nodes[idx];
            node->expr = n;
            node->uses = 1;
            node->temp_slot = -1;
            node->right = -1;
            node->scalar_id = -1;
            node->kind = PJ_ARITH_NODE_UNARY;
            node->op = 0; /* neg */
            node->left = left_idx;
            node->uniform = dag->nodes[left_idx].uniform;
            return idx;
        }
        if (arity == 2) {
            int op_code;
            if (strcmp(op, "+") == 0) op_code = 0;
            else if (strcmp(op, "-") == 0) op_code = 1;
            else if (strcmp(op, "*") == 0) op_code = 2;
            else if (strcmp(op, "/") == 0) {
                if (pj_integer(dag->dtype)) {
                    *reason = "unsupported-op";
                    return -1;
                }
                op_code = 3;
            } else {
                *reason = "unsupported-op";
                return -1;
            }
            int left_idx = pj_arith_find_or_add_node(dag, p, (const me_expr *)n->parameters[0], reason);
            if (left_idx < 0) return -1;
            int right_idx = pj_arith_find_or_add_node(dag, p, (const me_expr *)n->parameters[1], reason);
            if (right_idx < 0) return -1;
            if (dag->count >= 64) {
                *reason = "too-many-temps";
                return -1;
            }
            int idx = dag->count++;
            pj_arith_node *node = &dag->nodes[idx];
            node->expr = n;
            node->uses = 1;
            node->temp_slot = -1;
            node->scalar_id = -1;
            node->kind = PJ_ARITH_NODE_BINARY;
            node->op = op_code;
            node->left = left_idx;
            node->right = right_idx;
            node->uniform = dag->nodes[left_idx].uniform && dag->nodes[right_idx].uniform;
            return idx;
        }
    }

    *reason = "unsupported-leaf";
    return -1;
}

static bool pj_match_arith_tree(pj_arith_dag *dag, const me_dsl_compiled_program *p,
                                const me_expr *root, const char **reason) {
    if (!p || !root) {
        *reason = "unsupported-leaf";
        return false;
    }
    memset(dag, 0, sizeof(*dag));
    dag->dtype = root->dtype;
    if (dag->dtype == ME_FLOAT64) {
        dag->carrier = "f64";
        dag->carrier_type = "double";
        dag->var_type = "double";
        dag->strip = 512;
    } else if (dag->dtype == ME_FLOAT32) {
        dag->carrier = "f32";
        dag->carrier_type = "float";
        dag->var_type = "float";
        dag->strip = 1024;
    } else if (dag->dtype == ME_INT64 || dag->dtype == ME_UINT64) {
        dag->carrier = "u64";
        dag->carrier_type = "uint64_t";
        dag->var_type = pj_type(dag->dtype);
        dag->strip = 512;
    } else if (dag->dtype == ME_INT32 || dag->dtype == ME_UINT32) {
        dag->carrier = "u32";
        dag->carrier_type = "uint32_t";
        dag->var_type = pj_type(dag->dtype);
        dag->strip = 1024;
    } else if (dag->dtype == ME_INT16 || dag->dtype == ME_UINT16) {
        dag->carrier = "u16";
        dag->carrier_type = "uint16_t";
        dag->var_type = pj_type(dag->dtype);
        dag->strip = 2048;
    } else if (dag->dtype == ME_INT8 || dag->dtype == ME_UINT8) {
        dag->carrier = "u8";
        dag->carrier_type = "uint8_t";
        dag->var_type = pj_type(dag->dtype);
        dag->strip = 4096;
    } else {
        *reason = "mixed-dtype";
        return false;
    }

    if (p->output_dtype != dag->dtype) {
        *reason = "mixed-dtype";
        return false;
    }

    int root_idx = pj_arith_find_or_add_node(dag, p, root, reason);
    if (root_idx < 0) {
        return false;
    }
    dag->root = root_idx;
    dag->nnodes = dag->count;

    int n_literals = 0;
    int n_uniform_subs = 0;
    for (int i = 0; i < dag->count; i++) {
        pj_arith_node *node = &dag->nodes[i];
        if (node->kind == PJ_ARITH_NODE_LITERAL) {
            node->scalar_id = n_literals++;
        } else if (node->uniform && (node->kind == PJ_ARITH_NODE_UNARY || node->kind == PJ_ARITH_NODE_BINARY)) {
            node->scalar_id = n_uniform_subs++;
        }
    }
    dag->uniform_ops = n_uniform_subs;

    /* Temporary slot allocation */
    int free_mask = 0x0F;
    int peak_temps = 0;
    int rem_uses[64];
    for (int i = 0; i < dag->count; i++) {
        rem_uses[i] = dag->nodes[i].uses;
    }

    for (int i = 0; i < dag->count; i++) {
        pj_arith_node *node = &dag->nodes[i];
        if (node->uniform || node->kind == PJ_ARITH_NODE_VAR) {
            continue;
        }
        if (i == dag->root) {
            node->temp_slot = -1;
        } else {
            int slot = -1;
            for (int s = 0; s < 4; s++) {
                if (free_mask & (1 << s)) {
                    slot = s;
                    break;
                }
            }
            if (slot < 0) {
                *reason = "too-many-temps";
                return false;
            }
            node->temp_slot = slot;
            free_mask &= ~(1 << slot);
            if (slot + 1 > peak_temps) peak_temps = slot + 1;
        }

        if (node->left >= 0) {
            pj_arith_node *child = &dag->nodes[node->left];
            if (!child->uniform && child->temp_slot >= 0) {
                rem_uses[node->left]--;
                if (rem_uses[node->left] <= 0) {
                    free_mask |= (1 << child->temp_slot);
                }
            }
        }
        if (node->right >= 0) {
            pj_arith_node *child = &dag->nodes[node->right];
            if (!child->uniform && child->temp_slot >= 0) {
                rem_uses[node->right]--;
                if (rem_uses[node->right] <= 0) {
                    free_mask |= (1 << child->temp_slot);
                }
            }
        }
    }
    dag->ntemps = peak_temps;
    return true;
}

static void pj_arith_node_scalar_name(const pj_arith_dag *dag, int idx, char *buf, size_t cap) {
    const pj_arith_node *node = &dag->nodes[idx];
    if (node->kind == PJ_ARITH_NODE_LITERAL) {
        snprintf(buf, cap, "pj_c%d", node->scalar_id);
    } else if (node->kind == PJ_ARITH_NODE_UNIFORM_VAR) {
        snprintf(buf, cap, "pj_input_%d[0]", node->var_index);
    } else if (node->scalar_id >= 0) {
        snprintf(buf, cap, "pj_u%d", node->scalar_id);
    } else {
        snprintf(buf, cap, "0");
    }
}

static void pj_arith_emit_uniform_expr(const pj_arith_dag *dag, int idx, char *buf, size_t cap) {
    const pj_arith_node *node = &dag->nodes[idx];
    bool integer = pj_integer(dag->dtype);
    if (node->kind == PJ_ARITH_NODE_UNARY) {
        char child_name[64];
        pj_arith_node_scalar_name(dag, node->left, child_name, sizeof(child_name));
        if (integer) {
            snprintf(buf, cap, "pj_%s(0ULL-(uint64_t)(%s))", dag->var_type, child_name);
        } else {
            snprintf(buf, cap, "-%s", child_name);
        }
    } else if (node->kind == PJ_ARITH_NODE_BINARY) {
        char left_name[64], right_name[64];
        pj_arith_node_scalar_name(dag, node->left, left_name, sizeof(left_name));
        pj_arith_node_scalar_name(dag, node->right, right_name, sizeof(right_name));
        const char *op_sym = (node->op == 0) ? "+" : (node->op == 1) ? "-" : (node->op == 2) ? "*" : "/";
        if (integer) {
            snprintf(buf, cap, "pj_%s((uint64_t)(%s) %s (uint64_t)(%s))",
                dag->var_type, left_name, op_sym, right_name);
        } else {
            snprintf(buf, cap, "%s %s %s", left_name, op_sym, right_name);
        }
    } else {
        pj_arith_node_scalar_name(dag, idx, buf, cap);
    }
}

static bool pj_emit_arith_bridge(const pj_arith_dag *dag, const me_dsl_compiled_program *p, pj_text *out) {
    if (!pj_append(out, "static int pj_unmasked(const void *const *inputs, void *restrict output, int64_t count) {\n"))
        return false;
    if (!pj_append(out, "    if (count <= 0) return 0;\n"))
        return false;
    if (!pj_append(out, "    const me_pj_arith_table *pj_t = (const me_pj_arith_table *)inputs[%d];\n",
                   p->n_inputs + ME_DSL_PORTABLE_JIT_ARITH_OFF))
        return false;
    for (int i = 0; i < p->n_inputs; i++) {
        const char *type = pj_type(p->vars.dtypes[i]);
        if (!pj_append(out, "    const %s *pj_input_%d = (const %s *)inputs[%d];\n", type, i, type, i))
            return false;
    }
    if (!pj_append(out, "    %s *out = (%s *)output;\n", dag->var_type, dag->var_type))
        return false;

    /* Literals */
    for (int i = 0; i < dag->count; i++) {
        const pj_arith_node *node = &dag->nodes[i];
        if (node->kind == PJ_ARITH_NODE_LITERAL) {
            char val_str[128];
            const me_expr *n = node->expr;
            if (pj_float(dag->dtype)) {
                double val = (dag->dtype == ME_FLOAT32) ? (double)n->literal_f32 : n->value;
                snprintf(val_str, sizeof(val_str), "((%s)%a)", dag->var_type, val);
            } else {
                if (n->flags & ME_EXPR_FLAG_NEGATIVE_LITERAL) {
                    snprintf(val_str, sizeof(val_str), "pj_%s(0ULL-%lluULL)", dag->var_type, (unsigned long long)n->integer_magnitude);
                } else {
                    snprintf(val_str, sizeof(val_str), "pj_%s(%lluULL)", dag->var_type, (unsigned long long)n->integer_magnitude);
                }
            }
            if (!pj_append(out, "    const %s pj_c%d = %s;\n", dag->var_type, node->scalar_id, val_str))
                return false;
        }
    }

    /* Uniform subtrees */
    for (int i = 0; i < dag->count; i++) {
        const pj_arith_node *node = &dag->nodes[i];
        if (node->uniform && (node->kind == PJ_ARITH_NODE_UNARY || node->kind == PJ_ARITH_NODE_BINARY)) {
            char expr_str[256];
            pj_arith_emit_uniform_expr(dag, i, expr_str, sizeof(expr_str));
            if (!pj_append(out, "    const %s pj_u%d = ((volatile %s){ %s });\n",
                           dag->var_type, node->scalar_id, dag->var_type, expr_str))
                return false;
        }
    }

    /* Temporaries */
    for (int t = 0; t < dag->ntemps; t++) {
        if (!pj_append(out, "    %s pj_tmp%d[%d];\n", dag->carrier_type, t, dag->strip))
            return false;
    }

    /* Strip loop */
    if (!pj_append(out, "    for (int64_t s = 0; s < count; s += %d) {\n", dag->strip))
        return false;
    if (!pj_append(out, "        int n = (int)(count - s < %d ? count - s : %d);\n", dag->strip, dag->strip))
        return false;

    /* Inside strip loop */
    if (dag->nodes[dag->root].uniform) {
        char scalar_name[64];
        pj_arith_node_scalar_name(dag, dag->root, scalar_name, sizeof(scalar_name));
        if (!pj_append(out, "        pj_t->%s_fill((%s)(%s), (%s *)(out + s), n);\n",
                       dag->carrier, dag->carrier_type, scalar_name, dag->carrier_type))
            return false;
    } else {
        for (int i = 0; i < dag->count; i++) {
            const pj_arith_node *node = &dag->nodes[i];
            if (node->uniform) continue;

            if (node->kind == PJ_ARITH_NODE_VAR) {
                if (i == dag->root) {
                    if (!pj_append(out, "        memcpy(out + s, pj_input_%d + s, (size_t)n * sizeof(%s));\n",
                                   node->var_index, dag->var_type))
                        return false;
                }
                continue;
            }

            char dst[64];
            if (i == dag->root) {
                snprintf(dst, sizeof(dst), "(%s *)(out + s)", dag->carrier_type);
            } else {
                snprintf(dst, sizeof(dst), "pj_tmp%d", node->temp_slot);
            }

            if (node->kind == PJ_ARITH_NODE_UNARY) {
                char src[64];
                const pj_arith_node *child = &dag->nodes[node->left];
                if (child->kind == PJ_ARITH_NODE_VAR) {
                    snprintf(src, sizeof(src), "(const %s *)(pj_input_%d + s)", dag->carrier_type, child->var_index);
                } else {
                    snprintf(src, sizeof(src), "pj_tmp%d", child->temp_slot);
                }
                if (!pj_append(out, "        pj_t->%s_neg(%s, %s, n);\n", dag->carrier, src, dst))
                    return false;
            } else if (node->kind == PJ_ARITH_NODE_BINARY) {
                const char *op_name;
                switch (node->op) {
                    case 0: op_name = "PJ_ARITH_ADD"; break;
                    case 1: op_name = "PJ_ARITH_SUB"; break;
                    case 2: op_name = "PJ_ARITH_MUL"; break;
                    case 3: op_name = "PJ_ARITH_DIV"; break;
                    default: return false;
                }
                const pj_arith_node *l = &dag->nodes[node->left];
                const pj_arith_node *r = &dag->nodes[node->right];

                if (!l->uniform && !r->uniform) {
                    char left_src[64], right_src[64];
                    if (l->kind == PJ_ARITH_NODE_VAR) {
                        snprintf(left_src, sizeof(left_src), "(const %s *)(pj_input_%d + s)", dag->carrier_type, l->var_index);
                    } else {
                        snprintf(left_src, sizeof(left_src), "pj_tmp%d", l->temp_slot);
                    }
                    if (r->kind == PJ_ARITH_NODE_VAR) {
                        snprintf(right_src, sizeof(right_src), "(const %s *)(pj_input_%d + s)", dag->carrier_type, r->var_index);
                    } else {
                        snprintf(right_src, sizeof(right_src), "pj_tmp%d", r->temp_slot);
                    }
                    if (!pj_append(out, "        pj_t->%s_vv[%s](%s, %s, %s, n);\n",
                                   dag->carrier, op_name, left_src, right_src, dst))
                        return false;
                } else if (!l->uniform && r->uniform) {
                    char left_src[64], right_scalar[64];
                    if (l->kind == PJ_ARITH_NODE_VAR) {
                        snprintf(left_src, sizeof(left_src), "(const %s *)(pj_input_%d + s)", dag->carrier_type, l->var_index);
                    } else {
                        snprintf(left_src, sizeof(left_src), "pj_tmp%d", l->temp_slot);
                    }
                    pj_arith_node_scalar_name(dag, node->right, right_scalar, sizeof(right_scalar));
                    if (!pj_append(out, "        pj_t->%s_vs[%s](%s, (%s)(%s), %s, n);\n",
                                   dag->carrier, op_name, left_src, dag->carrier_type, right_scalar, dst))
                        return false;
                } else if (l->uniform && !r->uniform) {
                    char left_scalar[64], right_src[64];
                    pj_arith_node_scalar_name(dag, node->left, left_scalar, sizeof(left_scalar));
                    if (r->kind == PJ_ARITH_NODE_VAR) {
                        snprintf(right_src, sizeof(right_src), "(const %s *)(pj_input_%d + s)", dag->carrier_type, r->var_index);
                    } else {
                        snprintf(right_src, sizeof(right_src), "pj_tmp%d", r->temp_slot);
                    }
                    if (!pj_append(out, "        pj_t->%s_sv[%s]((%s)(%s), %s, %s, n);\n",
                                   dag->carrier, op_name, dag->carrier_type, left_scalar, right_src, dst))
                        return false;
                }
            }
        }
    }

    if (!pj_append(out, "    }\n    return 0;\n}\n"))
        return false;
    return true;
}

static bool pj_uniform_tree(const me_dsl_compiled_program *p, const me_expr *n, int depth) {
    if (!n || depth > 128) return false;
    int kind = TYPE_MASK(n->type);
    if (kind == ME_CONSTANT) return true;
    if (kind == ME_VARIABLE && is_synthetic_address(n->bound)) {
        int index = (int)((const char *)n->bound - synthetic_var_addresses);
        return index >= 0 && index < p->vars.count &&
            (p->vars.uniform[index] || (n->flags & ME_EXPR_FLAG_UNIFORM_INPUT));
    }
    if (!IS_FUNCTION(n->type) || is_reduction_node(n)) return false;
    for (int i=0; i<ARITY(n->type); i++) {
        if (!pj_uniform_tree(p, n->parameters[i], depth+1)) return false;
    }
    return true;
}

static bool pj_flat_index_node(const me_dsl_compiled_program *p, const me_expr *n) {
    if (!n || TYPE_MASK(n->type) != ME_VARIABLE || !is_synthetic_address(n->bound)) return false;
    return (int)((const char *)n->bound - synthetic_var_addresses) == p->idx_flat_idx;
}

static bool pj_flat_index_expr(const me_dsl_compiled_program *p, const me_expr *n, int depth) {
    if (pj_flat_index_node(p, n)) return true;
    if (!n || depth > 8 || !IS_FUNCTION(n->type) || ARITY(n->type) != 1 ||
        me_portable_operator(n) != NULL) return false;
    return pj_flat_index_expr(p, n->parameters[0], depth+1);
}

/* Match only the affine form used by the ramp constructors. Restricting the
 * initial bridge to unmasked strict kernels keeps global-index semantics
 * and fallback behavior straightforward. */
static bool pj_match_ramp(const me_dsl_compiled_program *p, const me_expr *root,
                          const me_expr **start, const me_expr **step) {
    if (!root || p->output_is_scalar || !p->uses_flat_idx || p->compile_ndims < 1 ||
        p->uses_i_mask || p->uses_n_mask || p->uses_ndim || p->fp_mode != ME_DSL_FP_STRICT ||
        p->block.nstmts != 1 || p->block.stmts[0]->kind != ME_DSL_STMT_RETURN) return false;
    /* Unwrap outer conversion to output_dtype if present (e.g. float32 conversion from float64 promotion). */
    if (!root->function && ARITY(root->type) == 1 && root->dtype == p->output_dtype) {
        root = (const me_expr *)root->parameters[0];
        if (!root) return false;
    }
    const char *root_op = me_portable_operator(root);
    if (!root_op || strcmp(root_op, "+") || ARITY(root->type) != 2) return false;
    const me_expr *base = root->parameters[0];
    const me_expr *product = root->parameters[1];
    const char *product_op = me_portable_operator(product);
    if (!product_op || strcmp(product_op, "*") || ARITY(product->type) != 2) {
        base = root->parameters[1];
        product = root->parameters[0];
        product_op = me_portable_operator(product);
        if (!product_op || strcmp(product_op, "*") || ARITY(product->type) != 2) return false;
    }
    const me_expr *index = product->parameters[0];
    const me_expr *stride = product->parameters[1];
    if (!pj_flat_index_expr(p, index, 0)) {
        index = product->parameters[1];
        stride = product->parameters[0];
    }
    bool flat_index = pj_flat_index_expr(p, index, 0);
    bool uniform_base = pj_uniform_tree(p, base, 0);
    bool uniform_stride = pj_uniform_tree(p, stride, 0);
    if (!flat_index || !uniform_base || !uniform_stride) return false;
    if (p->output_dtype == ME_INT64) {
        if (root->dtype != ME_INT64 || product->dtype != ME_INT64 ||
            base->dtype != ME_INT64 || stride->dtype != ME_INT64) return false;
    } else if (p->output_dtype == ME_FLOAT64) {
        if (root->dtype != ME_FLOAT64 || product->dtype != ME_FLOAT64 ||
            (base->dtype != ME_FLOAT64 && base->dtype != ME_INT64) ||
            (stride->dtype != ME_FLOAT64 && stride->dtype != ME_INT64)) return false;
    } else if (p->output_dtype == ME_FLOAT32) {
        /* The f32 AOT routine rounds intermediates to float. A final f32 cast
         * of an f64 expression must retain its f64 arithmetic in generic JIT. */
        if (root->dtype != ME_FLOAT32 || product->dtype != ME_FLOAT32 ||
            (base->dtype != ME_FLOAT32 && base->dtype != ME_INT64) ||
            (stride->dtype != ME_FLOAT32 && stride->dtype != ME_INT64)) return false;
    } else return false;
    *start = base;
    *step = stride;
    return true;
}

static bool pj_emit_ramp_bridge(const me_dsl_compiled_program *p, const char *start,
                                const char *step, const char *setup, pj_text *out) {
    const char *fn, *type;
    if (p->output_dtype == ME_INT64) {
        fn = "ramp_i64";
        type = "int64_t";
    } else if (p->output_dtype == ME_FLOAT32) {
        fn = "ramp_f32";
        type = "float";
    } else if (p->output_dtype == ME_FLOAT64) {
        fn = "ramp_f64";
        type = "double";
    } else return false;
    bool ok = pj_append(out,
        "static int pj_unmasked(const void *const *inputs, void *restrict output, int64_t count) {\n"
        "if (count <= 0) return 0;\n"
        "const me_pj_arith_table *pj_t = (const me_pj_arith_table *)inputs[%d];\n"
        "const int64_t *nd = (const int64_t *)inputs[%d];\n",
        p->n_inputs + ME_DSL_PORTABLE_JIT_ARITH_OFF,
        p->n_inputs + ME_DSL_PORTABLE_JIT_ND_OFF);
    if (!ok) return false;
    /* Scalar constructor arguments are represented as uniform input slots. */
    for (int i=0; i<p->n_inputs; i++) {
        const char *input_type = pj_type(p->vars.dtypes[i]);
        if (!input_type || !pj_append(out,
            "const %s *pj_input_%d = (const %s *)inputs[%d];\n",
            input_type, i, input_type, i)) return false;
    }
    if (setup && *setup) {
        ok = ok && pj_append(out,
            "#ifndef PJ_STATUS_CHECK\n"
            "#define PJ_STATUS_CHECK() do { if (pj_status) return pj_status; } while (0)\n"
            "#endif\n"
            "int pj_status = 0;\n%s", setup);
        if (!ok) return false;
    }
    if (p->compile_ndims <= 1) {
        return pj_append(out,
            "int64_t pj_origin = nd ? nd[2] : 0;\n"
            "%s *out = (%s *)output;\n"
            "for (int64_t s=0; s<count; s+=256) {\n"
            "int n = (int)(count-s < 256 ? count-s : 256);\n"
            "if (nd && (s > nd[1]-pj_origin || n > nd[1]-pj_origin-s)) return -2;\n"
            "pj_t->%s(pj_origin+s, (%s)(%s), (%s)(%s), out+s, n);\n"
            "}\nreturn 0;\n}\n",
            type, type, fn, type, start, type, step);
    }
    return pj_append(out,
        "int64_t pj_origin = 0, pj_stride = 1;\n"
        "_Bool pj_contig = 1;\n"
        "if (nd) {\n"
        "int64_t pj_nd = nd[0];\n"
        "for (int64_t d = pj_nd - 1; d >= 0; d--) {\n"
        "pj_origin += nd[1 + pj_nd + d] * pj_stride;\n"
        "pj_stride *= nd[1 + d];\n"
        "if (d > 0 && nd[1 + 2 * pj_nd + d] != nd[1 + d]) pj_contig = 0;\n"
        "}\n"
        "if (count > pj_stride - pj_origin) return -2;\n"
        "}\n"
        "%s *out = (%s *)output;\n"
        "if (pj_contig) {\n"
        "for (int64_t s=0; s<count; s+=256) {\n"
        "int n = (int)(count-s < 256 ? count-s : 256);\n"
        "pj_t->%s(pj_origin+s, (%s)(%s), (%s)(%s), out+s, n);\n"
        "}\nreturn 0;\n}\n"
        /* Traverse compact output one innermost row at a time. Decode outer
         * coordinates once per row, not once per lane, for every logical rank. */
        "if (nd && nd[0] >= 2) {\n"
        "int64_t rank = nd[0], cols = nd[1 + 3*rank - 1], rows = count / cols;\n"
        "for (int64_t r = 0; r < rows; r++) {\n"
        "int64_t row_origin = pj_origin, offset = r, stride = nd[rank];\n"
        "for (int64_t d = rank - 2; d >= 0; d--) {\n"
        "int64_t extent = nd[1 + 2*rank + d];\n"
        "row_origin += (offset %% extent) * stride;\n"
        "offset /= extent; stride *= nd[1 + d];\n"
        "}\n"
        "%s *row_out = out + r * cols;\n"
        "for (int64_t s = 0; s < cols; s += 256) {\n"
        "int n = (int)(cols - s < 256 ? cols - s : 256);\n"
        "pj_t->%s(row_origin + s, (%s)(%s), (%s)(%s), row_out + s, n);\n"
        "}\n}\nreturn 0;\n}\n"
        "return -1;\n}\n",
        type, type, fn, type, start, type, step,
        type, fn, type, start, type, step);
}

#endif

static void pj_prepare_source(me_dsl_compiled_program *p, const pj_constants *constants) {
    if (p) dsl_tracef("portable jit prepare: request=%d profile=%d",p->jit_request_mode,p->semantic_profile);
    /* Elementwise statements, lane-local loops, and audited float reductions. ND logical
     * context is supported through lane-local reserved index recomputation. */
    if (!p || p->semantic_profile != ME_DSL_PROFILE_PORTABLE_1_1 ||
        p->jit_request_mode != ME_JIT_ON ||
        p->compile_ndims < 0 || p->compile_ndims > ME_DSL_MAX_NDIM ||
        !p->guaranteed_return || p->vars.count > ME_MAX_VARS ||
        !pj_type(p->output_dtype)) {
        if (p) dsl_tracef("portable jit ineligible: request=%d statements=%d rank=%d",p->jit_request_mode,p->block.nstmts,p->compile_ndims);
        return;
    }
    if (p->math_backend == ME_DSL_MATH_SLEEF || p->math_backend == ME_DSL_MATH_ACCELERATE) {
        /* Generated kernels are fused per lane; their scalar callback bridge
         * would pay one lane-packing dispatch per element. The bounded vector
         * interpreter already batches the same math policy over tiles and is the
         * documented optional-JIT fallback for accelerated backends. */
        dsl_tracef("portable jit ineligible: accelerated math uses bounded vector interpretation");
        return;
    }
    bool scalar = p->output_is_scalar;
    int reduction_count = 0;
    if (scalar && (!pj_scalar_block(&p->block,&reduction_count,0) || !reduction_count)) {
        dsl_tracef("portable jit ineligible: unsupported scalar recipe");
        return;
    }
    for (int i = 0; i < p->n_inputs; i++) if (!pj_type(p->vars.dtypes[i])) {
        dsl_tracef("portable jit ineligible: unsupported input dtype"); return;
    }
    /* User compiler switches can override strict flags, so fail closed rather
     * than claim qualification under arbitrary host toolchain configuration. */
    const char *flags = me_jit_option_value("CFLAGS"), *tcc = getenv("ME_DSL_JIT_TCC_OPTIONS");
    if ((flags && *flags) || (tcc && *tcc)) {
        dsl_tracef("portable jit ineligible: explicit compiler flags");
        return;
    }
    bool defined[ME_MAX_VARS] = {false};
    pj_text body = {0};
    p->portable_jit_ncomparisons = p->portable_jit_nmath = 0;
    p->portable_jit_nmath2 = p->portable_jit_npreds = p->portable_jit_niops = 0;
    p->portable_jit_nmath3 = 0;
    p->portable_jit_nchecked = 0;
    for (int i = 0; i < p->n_locals; i++) {
        int index = p->local_var_indices[i];
        if (scalar && !p->vars.uniform[index]) {
            if (!pj_type(p->vars.dtypes[index]) || !pj_append(&body,
                "volatile %s *local_%d = (%s *)((void *const *)inputs[%d])[%d];\n",
                pj_type(p->vars.dtypes[index]),index,pj_type(p->vars.dtypes[index]),
                p->n_inputs+ME_DSL_PORTABLE_JIT_LOCALS_OFF,i)) {
                free(body.text);
                dsl_tracef("portable jit ineligible: scalar scratch declaration");
                return;
            }
            continue;
        }
        if (!pj_type(p->vars.dtypes[index]) ||
#ifdef __EMSCRIPTEN__
            !pj_append(&body,"%s local_%d;\n",pj_type(p->vars.dtypes[index]),index)) {
#else
            !pj_append(&body,"volatile %s local_%d;\n",pj_type(p->vars.dtypes[index]),index)) {
#endif
            free(body.text);
            dsl_tracef("portable jit ineligible: local declaration");
            return;
        }
    }
    int next_label = 0, next_temp = 0;
    pj_flow flow = {&next_label,&next_temp,-1};
    bool lowered;
    /* Scalar loops need the ND preamble; lower them after constructing it. */
    lowered = scalar || pj_block(p,&p->block,defined,&body,0,constants,flow);
    if (!lowered) {
        free(body.text);
        dsl_tracef("portable jit ineligible: unsupported typed statements");
        return;
    }
#ifdef __EMSCRIPTEN__
    /* The initial wasm ABI supports only elementwise source with all operations
     * emitted inline. Host callbacks and pointer-table scratch state are not
     * portable across the independently instantiated wasm module. */
    if (scalar || p->uses_i_mask || p->uses_n_mask || p->uses_ndim ||
        p->uses_flat_idx || p->portable_jit_nchecked || p->portable_jit_niops ||
        p->portable_jit_ncomparisons || p->portable_jit_npreds || p->portable_jit_nmath ||
        p->portable_jit_nmath2 || p->portable_jit_nmath3) {
        free(body.text);
        dsl_tracef("portable jit ineligible: wasm32 first slice requires bridge-free elementwise kernel");
        return;
    }
#endif
    /* Lane-local ND index recomputation, range-checked like the interpreter's
     * logical traversal. Masked lanes are skipped before this preamble. */
    pj_text preamble = {0};
    char nd_decl[96] = "";
    bool uses_nd = (p->uses_i_mask || p->uses_n_mask || p->uses_ndim || p->uses_flat_idx);
    if (uses_nd) {
        snprintf(nd_decl,sizeof(nd_decl),
            "const int64_t *nd = (const int64_t *)inputs[%d];\n",
            p->n_inputs + ME_DSL_PORTABLE_JIT_ND_OFF);
        bool ok = pj_append(&preamble, "int64_t jit_bad = 0;\n");
        for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
            if (p->uses_i_mask & (1 << d)) ok = ok && pj_append(&preamble, "int64_t coord_%d = 0;\n", d);
        }
        if (p->uses_flat_idx) ok = ok && pj_append(&preamble, "int64_t flat = 0;\n");
        /* A one-dimensional flat index is just origin + lane. Avoid the
         * general mixed-radix coordinate loop (including division/modulo)
         * for the common ramp kernel. Keep the generic checked path for
         * every other reserved-index combination and rank. */
        bool flat_1d = p->uses_flat_idx && p->compile_ndims == 1 &&
            !p->uses_i_mask && !p->uses_n_mask && !p->uses_ndim;
        if (flat_1d) {
            ok = ok && pj_append(&preamble,
                "if (nd) { flat = nd[2] + i; if (i >= nd[1] - nd[2]) jit_bad = 1; } else flat = 0;\n"
                "if (jit_bad) return -2;\n");
        } else {
            ok = ok && pj_append(&preamble,
                "if (nd) { int64_t jit_off = i, jit_stride = 1; int jit_nd = (int)nd[0];\n"
                "for (int jit_d = jit_nd - 1; jit_d >= 0; jit_d--) {\n"
                "int64_t jit_ext = nd[1+2*jit_nd+jit_d];\n"
                "int64_t jit_rem = nd[1+jit_d]-nd[1+jit_nd+jit_d];\n"
                "int64_t jit_rel = jit_ext > 0 ? jit_off %% jit_ext : 0;\n"
                "jit_off = jit_ext > 0 ? jit_off / jit_ext : 0;\n"
                "int64_t jit_coord = nd[1+jit_nd+jit_d] + jit_rel;\n"
                "if (jit_rel >= jit_rem) jit_bad = 1;\n");
            for (int d = 0; d < ME_DSL_MAX_NDIM; d++) {
                if (p->uses_i_mask & (1 << d)) {
                    ok = ok && pj_append(&preamble, "if (jit_d == %d) coord_%d = jit_coord;\n", d, d);
                }
            }
            if (p->uses_flat_idx) {
                ok = ok && pj_append(&preamble, "flat += jit_coord * jit_stride; jit_stride *= nd[1+jit_d];\n");
            }
            ok = ok && pj_append(&preamble, "}\n}\nif (jit_bad) return -2;\n");
        }
        if (!ok) {
            dsl_tracef("portable jit ineligible: ND preamble");
            free(preamble.text);
            free(body.text);
            return;
        }
    }
    if (scalar) {
        pj_constants scalar_constants = constants ? *constants : (pj_constants){0};
        scalar_constants.scalar_phase = true;
        scalar_constants.nd_preamble = preamble.text;
        if (!pj_block(p,&p->block,defined,&body,0,&scalar_constants,flow)) {
            free(preamble.text);
            free(body.text);
            dsl_tracef("portable jit ineligible: unsupported scalar recipe");
            return;
        }
    }
    pj_text input_declarations = {0}, unmasked_body = {0};
    bool declarations_ok = true;
    for (int i=0; i<p->n_inputs; i++) {
        const char *type = pj_type(p->vars.dtypes[i]);
        declarations_ok = declarations_ok && type && pj_append(&input_declarations,
            "const %s *pj_input_%d = (const %s *)inputs[%d];\n",type,i,type,i);
    }
    /* Pointer-table reads must be outside byte-output loops: a char store may
     * otherwise alias inputs[] and force a reload on every lane. Separate
     * unmasked/masked functions also make vectorization independent of loop
     * unswitching heuristics, without duplicating function-scoped goto labels. */
    const char *skip = "if (mask && !mask[i]) continue;\n";
    const char *scan = body.text ? body.text : "", *found;
    while ((found=strstr(scan,skip)) != NULL) {
        declarations_ok = declarations_ok && pj_append(&unmasked_body,"%.*s",(int)(found-scan),scan);
        scan = found+strlen(skip);
    }
    declarations_ok = declarations_ok && pj_append(&unmasked_body,"%s",scan);
    if (!declarations_ok) {
        dsl_tracef("portable jit ineligible: input declarations");
        free(input_declarations.text); free(unmasked_body.text); free(body.text); free(preamble.text);
        return;
    }
    bool use_arith_bridge = false;
    bool use_ramp_bridge = false;
    pj_text unmasked_bridge = {0};
    const char *arith_rejection_reason = NULL;

#ifndef __EMSCRIPTEN__
    if ((p->compiler == ME_DSL_COMPILER_LIBTCC || p->compiler == ME_DSL_COMPILER_CC) && !scalar &&
        p->block.nstmts == 1 && p->block.stmts[0]->kind == ME_DSL_STMT_RETURN) {
        const me_expr *ramp_start = NULL, *ramp_step = NULL;
        const me_expr *root = p->block.stmts[0]->as.return_stmt.expr.expr;
        if (pj_match_ramp(p, root, &ramp_start, &ramp_step)) {
            pj_text scalar_setup = {0};
            int scalar_temp = 0;
            char *start_expr = pj_expr(p, ramp_start, defined, 0, constants, &scalar_setup, &scalar_temp);
            char *step_expr = pj_expr(p, ramp_step, defined, 0, constants, &scalar_setup, &scalar_temp);
            if (start_expr && step_expr &&
                pj_emit_ramp_bridge(p, start_expr, step_expr, scalar_setup.text, &unmasked_bridge)) {
                use_ramp_bridge = true;
                dsl_tracef("portable jit affine ramp bridge: dtype=%s compiler=host",
                    pj_type(p->output_dtype));
            }
            free(start_expr);
            free(step_expr);
            free(scalar_setup.text);
        }
    }
    pj_arith_dag arith_dag = {0};
    if (p->compiler == ME_DSL_COMPILER_LIBTCC && !scalar &&
        p->block.nstmts == 1 && p->block.stmts[0]->kind == ME_DSL_STMT_RETURN &&
        p->n_locals == 0 && !p->uses_ndim && !p->uses_flat_idx &&
        !p->uses_i_mask && !p->uses_n_mask) {
        if (!pj_arith_bridge_enabled()) {
            arith_rejection_reason = "disabled";
        } else if (p->fp_mode != ME_DSL_FP_STRICT) {
            arith_rejection_reason = "non-strict-fp";
        } else {
            const me_expr *ret_expr = p->block.stmts[0]->as.return_stmt.expr.expr;
            if (pj_match_arith_tree(&arith_dag, p, ret_expr, &arith_rejection_reason)) {
                if (pj_emit_arith_bridge(&arith_dag, p, &unmasked_bridge)) {
                    use_arith_bridge = true;
                }
            }
        }
        if (use_arith_bridge) {
            dsl_tracef("portable jit arithmetic bridge: dtype=%s nodes=%d temps=%d uniform_ops=%d compiler=tcc",
                pj_type(arith_dag.dtype), arith_dag.nnodes, arith_dag.ntemps, arith_dag.uniform_ops);
        } else if (arith_rejection_reason) {
            dsl_tracef("portable jit arithmetic bridge rejected: reason=%s", arith_rejection_reason);
        }
    }
    p->portable_jit_has_arith_bridge = use_arith_bridge;
    p->portable_jit_has_ramp_bridge = use_ramp_bridge;
    p->portable_jit_arith_ntemps = use_arith_bridge ? (uint8_t)arith_dag.ntemps : 0;
    p->portable_jit_arith_nnodes = use_arith_bridge ? (uint16_t)arith_dag.nnodes : 0;
#else
    p->portable_jit_has_arith_bridge = false;
    p->portable_jit_has_ramp_bridge = false;
    p->portable_jit_arith_ntemps = 0;
    p->portable_jit_arith_nnodes = 0;
#endif

    size_t capacity = 2*(body.used + preamble.used + input_declarations.used + unmasked_bridge.used) + 32768;
    char *source = malloc(capacity);
    me_dsl_jit_ir_program *ir = calloc(1,sizeof(*ir));
    if (!source || !ir) {
        free(preamble.text);
        free(body.text);
        free(source);
        free(ir);
        free(input_declarations.text);
        free(unmasked_body.text);
        free(unmasked_bridge.text);
        return;
    }
    ir->nparams = p->n_inputs;
    ir->param_dtypes = calloc(p->n_inputs ? p->n_inputs : 1,sizeof(me_dtype));
    ir->params = calloc(p->n_inputs ? p->n_inputs : 1,sizeof(char *));
    if (!ir->param_dtypes || !ir->params) {
        free(preamble.text);
        free(body.text);
        free(source);
        me_dsl_jit_ir_free(ir);
        free(input_declarations.text);
        free(unmasked_body.text);
        free(unmasked_bridge.text);
        return;
    }
    for (int i = 0; i < p->n_inputs; i++) ir->param_dtypes[i] = p->vars.dtypes[i];
    const char *entry_count_type = "int64_t";
    const char *unmasked_loop = "for (int64_t i=0; i<count; i++) {\n";
    const char *masked_loop = "for (int64_t i=0; i<count; i++) { if (mask && !mask[i]) continue;\n";
    const char *loop_close = "pj_lane_done: ; }";
#ifdef __EMSCRIPTEN__
    /* Keep counts and address induction on wasm32 i32. Use a single-header
     * while loop so TCC's wasm32 backend emits structured loop/br_if control
     * flow instead of falling back to the br_table state-machine dispatcher. */
    entry_count_type = "int";
    unmasked_loop = "{ int i=0; while (i<count) {\n";
    masked_loop = "{ int i=0; while (i<count) { if (mask && !mask[i]) goto pj_lane_done;\n";
    loop_close = "pj_lane_done: i++; } }";
#endif

    const char *status_macro = (p->portable_jit_nchecked || p->portable_jit_niops || p->portable_jit_ncomparisons ||
        p->portable_jit_npreds || p->portable_jit_nmath || p->portable_jit_nmath2 ||
        p->portable_jit_nmath3) ?
        "#undef PJ_STATUS_CHECK\n#define PJ_STATUS_CHECK() do { if (pj_status) return pj_status; } while (0)\n" :
        "#undef PJ_STATUS_CHECK\n#define PJ_STATUS_CHECK() ((void)0)\n";

    pj_text unmasked_func = {0};
    if (use_arith_bridge || use_ramp_bridge) {
        pj_append(&unmasked_func, "%s", unmasked_bridge.text);
    } else {
        pj_append(&unmasked_func,
            "static int pj_unmasked(const void *const *inputs, void *restrict output, %s count) {\n"
            "const unsigned char *mask = 0;\n"
            "const int64_t pj_cap = *(const int64_t *)inputs[%d];\n"
            "int pj_status = 0;\n"
            "%s"
            "%s"
            "%s"
            "%s"
            "%s"
            "%s %s return 0; }\n",
            entry_count_type, p->n_inputs + ME_DSL_PORTABLE_JIT_CAP_OFF,
            status_macro,
            input_declarations.text ? input_declarations.text : "", nd_decl,
            scalar ? "" : unmasked_loop,
            scalar ? "" : (preamble.text ? preamble.text : ""), unmasked_body.text,
            scalar ? "" : loop_close);
    }

    int used = snprintf(source,capacity,
        "/* portable-1.1 lowering-r31 integral-math arith-bridge abi-r12 */\n"
        "#include <stdint.h>\n"
        "#include <limits.h>\n"
        "#include <string.h>\n"
        "typedef _Bool (*pj_cmp)(const void *,double,double);\n"
        "typedef double (*pj_math)(const void *,double);\n"
        "typedef double (*pj_math2)(const void *,double,double);\n"
        "typedef double (*pj_libm)(double);\n"
        "typedef double (*pj_libm2)(double,double);\n"
        "typedef float (*pj_libmf)(float);\n"
        "typedef float (*pj_libm2f)(float,float);\n"
        "typedef double (*pj_libm3)(double,double,double);\n"
        "typedef float (*pj_libm3f)(float,float,float);\n"
        "typedef double (*pj_ldexpd)(double,int);\n"
        "typedef float (*pj_ldexpf)(float,int);\n"
        "typedef _Bool (*pj_predf)(float);\n"
        "typedef _Bool (*pj_predd)(double);\n"
        "typedef _Bool (*pj_pred)(const void *,double);\n"
        "typedef uint64_t (*pj_iop)(const void *,uint64_t,uint64_t,int *);\n"
        "typedef uint64_t (*pj_checked)(const void *,uint64_t,uint64_t,uint64_t,int *);\n"
        "typedef struct me_pj_arith_table {\n"
        "    void (*f64_vv[4])(const double *, const double *, double *restrict, int);\n"
        "    void (*f64_vs[4])(const double *, double, double *restrict, int);\n"
        "    void (*f64_sv[4])(double, const double *, double *restrict, int);\n"
        "    void (*f64_neg)(const double *, double *restrict, int);\n"
        "    void (*f64_fill)(double, double *restrict, int);\n"
        "    void (*f32_vv[4])(const float *, const float *, float *restrict, int);\n"
        "    void (*f32_vs[4])(const float *, float, float *restrict, int);\n"
        "    void (*f32_sv[4])(float, const float *, float *restrict, int);\n"
        "    void (*f32_neg)(const float *, float *restrict, int);\n"
        "    void (*f32_fill)(float, float *restrict, int);\n"
        "    void (*u8_vv[3])(const uint8_t *, const uint8_t *, uint8_t *restrict, int);\n"
        "    void (*u8_vs[3])(const uint8_t *, uint8_t, uint8_t *restrict, int);\n"
        "    void (*u8_sv[3])(uint8_t, const uint8_t *, uint8_t *restrict, int);\n"
        "    void (*u8_neg)(const uint8_t *, uint8_t *restrict, int);\n"
        "    void (*u8_fill)(uint8_t, uint8_t *restrict, int);\n"
        "    void (*u16_vv[3])(const uint16_t *, const uint16_t *, uint16_t *restrict, int);\n"
        "    void (*u16_vs[3])(const uint16_t *, uint16_t, uint16_t *restrict, int);\n"
        "    void (*u16_sv[3])(uint16_t, const uint16_t *, uint16_t *restrict, int);\n"
        "    void (*u16_neg)(const uint16_t *, uint16_t *restrict, int);\n"
        "    void (*u16_fill)(uint16_t, uint16_t *restrict, int);\n"
        "    void (*u32_vv[3])(const uint32_t *, const uint32_t *, uint32_t *restrict, int);\n"
        "    void (*u32_vs[3])(const uint32_t *, uint32_t, uint32_t *restrict, int);\n"
        "    void (*u32_sv[3])(uint32_t, const uint32_t *, uint32_t *restrict, int);\n"
        "    void (*u32_neg)(const uint32_t *, uint32_t *restrict, int);\n"
        "    void (*u32_fill)(uint32_t, uint32_t *restrict, int);\n"
        "    void (*u64_vv[3])(const uint64_t *, const uint64_t *, uint64_t *restrict, int);\n"
        "    void (*u64_vs[3])(const uint64_t *, uint64_t, uint64_t *restrict, int);\n"
        "    void (*u64_sv[3])(uint64_t, const uint64_t *, uint64_t *restrict, int);\n"
        "    void (*u64_neg)(const uint64_t *, uint64_t *restrict, int);\n"
        "    void (*u64_fill)(uint64_t, uint64_t *restrict, int);\n"
        "    void (*ramp_i64)(int64_t, int64_t, int64_t, int64_t *restrict, int);\n"
        "    void (*ramp_f32)(int64_t, float, float, float *restrict, int);\n"
        "    void (*ramp_f64)(int64_t, double, double, double *restrict, int);\n"
        "} me_pj_arith_table;\n"
        "#define PJ_ARITH_ADD 0\n"
        "#define PJ_ARITH_SUB 1\n"
        "#define PJ_ARITH_MUL 2\n"
        "#define PJ_ARITH_DIV 3\n"
        "#define pj_pack_f32(x) ((uint64_t)((union { float f; uint32_t u; }){ .f = (x) }).u)\n"
        "#define pj_pack_f64(x) (((union { double f; uint64_t u; }){ .f = (x) }).u)\n"
        "#define pj_unpack_f32(x) (((union { uint32_t u; float f; }){ .u = (uint32_t)(x) }).f)\n"
        "#define pj_unpack_f64(x) (((union { uint64_t u; double f; }){ .u = (uint64_t)(x) }).f)\n"
        "#define PJ_SIGNED(S,U,x) (((union { U u; S s; }){ .u = (U)(x) }).s)\n"
        "#define pj_int8_t(x) PJ_SIGNED(int8_t,uint8_t,x)\n"
        "#define pj_int16_t(x) PJ_SIGNED(int16_t,uint16_t,x)\n"
        "#define pj_int32_t(x) PJ_SIGNED(int32_t,uint32_t,x)\n"
        "#define pj_int64_t(x) PJ_SIGNED(int64_t,uint64_t,x)\n"
        "#define pj_uint8_t(x) ((uint8_t)(x))\n"
        "#define pj_uint16_t(x) ((uint16_t)(x))\n"
        "#define pj_uint32_t(x) ((uint32_t)(x))\n"
        "#define pj_uint64_t(x) ((uint64_t)(x))\n"
        "%s"
        "static int pj_masked(const void *const *inputs, void *restrict output, %s count) {\n"
        "const unsigned char *mask = inputs[%d];\n"
        "const int64_t pj_cap = *(const int64_t *)inputs[%d];\nint pj_status = 0;\n"
        "%s"
        "%s%s%s%s%s %s return 0; }\n"
        "int %s(const void *const *inputs, void *output, %s count) {\n"
        "if (count < 0) return -1;\n"
        "return inputs[%d] ? pj_masked(inputs,output,count) : pj_unmasked(inputs,output,count); }\n",
        unmasked_func.text ? unmasked_func.text : "",
        entry_count_type,p->n_inputs,p->n_inputs+ME_DSL_PORTABLE_JIT_CAP_OFF,
        status_macro,
        input_declarations.text ? input_declarations.text : "",nd_decl,
        scalar ? "" : masked_loop,
        scalar ? "" : (preamble.text ? preamble.text : ""),body.text,
        scalar ? "" : loop_close,ME_DSL_JIT_SYMBOL_NAME,entry_count_type,p->n_inputs);
    free(unmasked_func.text);
    free(unmasked_bridge.text);
    if (used < 0) {
        dsl_tracef("portable jit ineligible: generated source exceeds capacity");
        free(source); me_dsl_jit_ir_free(ir); return;
    }
    p->jit_ir = ir;
    p->jit_c_source = source;
#ifdef __EMSCRIPTEN__
    p->jit_portable_wasm_source = true;
#endif
    p->jit_nparams = p->n_inputs;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *c = (const unsigned char *)source; *c; c++) {
        hash ^= *c; hash *= UINT64_C(1099511628211);
    }
    p->jit_ir_fingerprint = hash;
}

void dsl_portable_prepare_jit_source(me_dsl_compiled_program *p) {
    const me_math_dispatch *previous = me_math_dispatch_set(p ? &p->math_dispatch : NULL);
    pj_prepare_source(p, NULL);
    me_math_dispatch_set(previous);
}

void dsl_portable_prepare_jit_constants(me_dsl_compiled_program *p,
    const int64_t *values, const bool *known) {
    if (!p || !values || !known || p->jit_c_source || p->jit_ir) return;
    pj_constants constants = {.values = values, .known = known, .count = p->n_inputs};
    const me_math_dispatch *previous = me_math_dispatch_set(&p->math_dispatch);
    pj_prepare_source(p, &constants);
    me_math_dispatch_set(previous);
    dsl_try_prepare_jit_runtime(p);
}

void dsl_portable_prepare_jit(me_dsl_compiled_program *p) {
    dsl_portable_prepare_jit_source(p);
    dsl_try_prepare_jit_runtime(p);
}
