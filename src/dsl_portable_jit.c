/* Portable 1.1 scalar C lowering, revision 18. Never translate source
 * text: the typed tree includes promotion and final output conversions. The
 * private kernel ABI appends a participating mask and host comparison/math bindings;
 * legacy/full kernels retain their unchanged three-argument ABI. */
#include "dsl_compile_internal.h"
#include "dsl_portable_expr.h"
#include "dsl_portable_types.h"
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
} pj_constants;

typedef struct {
    char *text;
    size_t used, capacity;
} pj_text;

static bool pj_append(pj_text *s, const char *format, ...);

/* Only immutable integral capture leaves can be specialized. Validate the exact
 * checked weak conversion; out-of-range captures keep interpretation, including
 * its lazy participation/error timing. Never alter the shared typed tree. */
static char *pj_weak_conversion(const me_expr *n, const pj_constants *constants) {
    const me_expr *arg = n->parameters[0];
    if (!constants || TYPE_MASK(arg->type) != ME_VARIABLE ||
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

static char *pj_expr(me_dsl_compiled_program *p, const me_expr *n, const bool *defined, int depth,
    const pj_constants *constants, pj_text *s, int *next_temp) {
    if (!n || depth > 128 || !pj_type(n->dtype)) return NULL;
    const char *type = pj_type(n->dtype);
    char leaf[256];
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
        if (index < p->n_inputs) snprintf(leaf,sizeof(leaf),"((const %s *)inputs[%d])[i]",type,index);
        else if (p->local_slots && p->local_slots[index] >= 0 && defined[index]) {
            snprintf(leaf,sizeof(leaf),"local_%d",index);
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
    bool int_cast = conversion && pj_integer(n->dtype) && pj_integral(arg0->dtype) &&
        arg0->dtype != n->dtype && !weak_operand;
    if (conversion && pj_integer(n->dtype) && arg0->dtype != n->dtype && weak_operand) {
        /* Artifact capture leaves are immutable, but their values arrive only
         * after validation. Defer preparation so constant specialization keeps
         * its fast path and out-of-range immutable leaves keep safe fallback. */
        if (!constants && TYPE_MASK(arg0->type) == ME_VARIABLE && is_synthetic_address(arg0->bound) &&
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
    bool comparison_bridge = comparison && !integer_comparison;
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
    bool unary_math = math && arity == 1 && !predicate && float_result && arg0 && pj_float(arg0->dtype);
    bool binary_math = math && arity == 2 && float_result && arg0 && arg1 &&
        pj_float(arg0->dtype) && pj_float(arg1->dtype);
    bool checked_math = (math && arity >= 1 && arity <= 3 &&
        !predicate && !unary_math && !binary_math) ||
        (op && !strcmp(op,"**") && arity == 2 && pj_integer(n->dtype));
    bool float_operator = op && arity == 2 && float_result && arg0 && arg1 &&
        (!strcmp(op,"//") || !strcmp(op,"%") || !strcmp(op,"**")) &&
        pj_float(arg0->dtype) && pj_float(arg1->dtype);
    bool integer_operator = op && pj_integer(n->dtype) &&
        ((arity == 2 && (!strcmp(op,"%") || !strcmp(op,"//") || !strcmp(op,"<<") ||
                         !strcmp(op,">>") || !strcmp(op,"&") || !strcmp(op,"|") || !strcmp(op,"^"))) ||
          (arity == 1 && !strcmp(op,"~")));
    if (checked_weak) {
        if (!integer_arithmetic && !integer_operator && !checked_math && !checked_conversion && !where) return NULL;
        if (!checked_math && !checked_conversion) { integer_operator = true; integer_arithmetic = false; }
    }
    bool logical = op && ((!strcmp(op,"not") && arity == 1) ||
        ((!strcmp(op,"and") || !strcmp(op,"or")) && arity == 2));
    if (!conversion && !where && !comparison && !arithmetic && !integer_arithmetic &&
        !logical && !predicate && !unary_math && !binary_math && !float_operator &&
        !integer_operator && !checked_math) return NULL;
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
        args[j] = pj_expr(p,n->parameters[j],defined,depth+1,constants,s,next_temp);
        if (!args[j]) {
            for (int k = 0; k < arity; k++) free(args[k]);
            return NULL;
        }
    }
    size_t capacity = 256;
    for (int j = 0; j < arity; j++) capacity += strlen(args[j]);
    char *out = malloc(capacity);
    if (out) {
        if (checked_conversion || checked_math || checked_comparison) {
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
        else if (where) snprintf(out,capacity,"((%s)((%s) ? (%s) : (%s)))",type,args[0],args[1],args[2]);
        else if (int_cast) snprintf(out,capacity,"pj_%s((uint64_t)(%s))",type,args[0]);
        else if (conversion) snprintf(out,capacity,"((%s)(%s))",type,args[0]);
        else if (integer_comparison) {
            snprintf(out,capacity,"(pj_icmp((uint64_t)(%s),(uint64_t)(%s),%d,%d) %s 0)",
                args[0],args[1],pj_unsigned(arg0->dtype),pj_unsigned(arg1->dtype),op);
        }
        else if (checked_arithmetic) {
            const char *name = !strcmp(op,"+") ? "add" : !strcmp(op,"-") ? "sub" : "mul";
            snprintf(out,capacity,"pj_int64_t(pj_weak_%s((uint64_t)(%s),(uint64_t)(%s),&pj_status))",
                name,arity == 1 ? "0ULL" : args[0],arity == 1 ? args[0] : args[1]);
        }
        else if (predicate) {
            if (p->portable_jit_npreds == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_npreds++;
                p->portable_jit_preds[id] = n;
                snprintf(out,capacity,"((_Bool)(((pj_pred)inputs[%d])(inputs[%d],(double)(%s))))",
                    p->n_inputs+ME_DSL_PORTABLE_JIT_PRED_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_PRED_OFF+1+id,args[0]);
            }
        }
        else if (unary_math) {
            if (p->portable_jit_nmath == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_nmath++;
                p->portable_jit_math[id] = n;
                snprintf(out,capacity,"((%s)(((pj_math)inputs[%d])(inputs[%d],(double)(%s))))",
                    type,p->n_inputs+ME_DSL_PORTABLE_JIT_MATH1_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_MATH1_OFF+1+id,args[0]);
            }
        }
        else if (binary_math || float_operator) {
            if (p->portable_jit_nmath2 == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_nmath2++;
                p->portable_jit_math2[id] = n;
                snprintf(out,capacity,"((%s)(((pj_math2)inputs[%d])(inputs[%d],(double)(%s),(double)(%s))))",
                    type,p->n_inputs+ME_DSL_PORTABLE_JIT_MATH2_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_MATH2_OFF+1+id,args[0],args[1]);
            }
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
            if (arity == 1) snprintf(out,capacity,"pj_%s(0ULL-(uint64_t)(%s))",type,args[0]);
            else snprintf(out,capacity,"pj_%s((uint64_t)(%s) %s (uint64_t)(%s))",type,args[0],op,args[1]);
        }
        else if (comparison_bridge) {
            if (p->portable_jit_ncomparisons == ME_DSL_PORTABLE_JIT_BRIDGE_LIMIT) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_ncomparisons++;
                p->portable_jit_comparisons[id] = n;
                const char *name = !strcmp(op, "==") ? "eq" : !strcmp(op, "!=") ? "ne" :
                    !strcmp(op, "<") ? "lt" : !strcmp(op, "<=") ? "le" :
                    !strcmp(op, ">") ? "gt" : "ge";
                snprintf(out,capacity,"pj_%s((pj_cmp)inputs[%d],inputs[%d],(double)(%s),(double)(%s))",
                    name,p->n_inputs+ME_DSL_PORTABLE_JIT_CMP_OFF,
                    p->n_inputs+ME_DSL_PORTABLE_JIT_CMP_OFF+1+id,args[0],args[1]);
            }
        }
        else if (arity == 1) snprintf(out,capacity,"((%s)(%s(%s)))",type,logical ? "!" : "-",args[0]);
        else snprintf(out,capacity,"((%s)((%s) %s (%s)))",type,args[0],
            logical ? (!strcmp(op,"and") ? "&&" : "||") : op,args[1]);
    }
    for (int j = 0; j < arity; j++) free(args[j]);
    if (!out) return NULL;
    int id = (*next_temp)++;
    bool ok = pj_append(s,"%s pj_expr_%d = %s; if (pj_status) return pj_status;\n",type,id,out);
    free(out);
    if (!ok) return NULL;
    snprintf(leaf,sizeof(leaf),"pj_expr_%d",id);
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

static bool pj_block(me_dsl_compiled_program *p, const me_dsl_compiled_block *block,
                     bool *defined, pj_text *s, int depth, const pj_constants *constants,
                     pj_flow flow) {
    if (depth > 64) return false;
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_compiled_stmt *stmt = block->stmts[i];
        if (stmt->kind == ME_DSL_STMT_ASSIGN || stmt->kind == ME_DSL_STMT_RETURN) {
            bool assignment = stmt->kind == ME_DSL_STMT_ASSIGN;
            const me_expr *node = assignment ? stmt->as.assign.value.expr : stmt->as.return_stmt.expr.expr;
            char *value = pj_expr(p,node,defined,0,constants,s,flow.next_temp);
            if (!value) return false;
            bool ok;
            if (assignment) {
                int index = p->local_var_indices[stmt->as.assign.local_slot];
                ok = node->dtype == p->vars.dtypes[index] &&
                    pj_append(s,"local_%d = %s; if (pj_status) return pj_status;\n",index,value);
                defined[index] = true;
            }
            else ok = pj_append(s,"((%s *)output)[i] = %s; if (pj_status) return pj_status; goto pj_lane_done;\n",pj_type(p->output_dtype),value);
            free(value);
            if (!ok) return false;
        }
        else if (stmt->kind == ME_DSL_STMT_IF) {
            bool merged[ME_MAX_VARS], branch[ME_MAX_VARS];
            memcpy(merged,defined,sizeof(merged));
            int alternatives = 1 + stmt->as.if_stmt.n_elifs;
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
                    bool ok = pj_append(s,"if (%s) { if (pj_status) return pj_status;\n",value);
                    free(value);
                    if (!ok) return false;
                    body = j == 0 ? &stmt->as.if_stmt.then_block : &stmt->as.if_stmt.elif_branches[j-1].block;
                }
                else {
                    if (!pj_append(s,"else { if (pj_status) return pj_status;\n")) return false;
                    body = &stmt->as.if_stmt.else_block;
                }
                memcpy(branch,defined,sizeof(branch));
                if (!pj_block(p,body,branch,s,depth+1,constants,flow) || !pj_append(s,"}\n")) return false;
                if (j == 0) memcpy(merged,branch,sizeof(merged));
                else for (int k = 0; k < p->vars.count; k++) merged[k] &= branch[k];
            }
            for (int j = 1; j < alternatives; j++) if (!pj_append(s,"}\n")) return false;
            memcpy(defined,merged,sizeof(merged));
        }
        else if (stmt->kind == ME_DSL_STMT_FOR) {
            const me_expr *nodes[] = {stmt->as.for_loop.start.expr,
                stmt->as.for_loop.stop.expr, stmt->as.for_loop.step.expr};
            char *values[3] = {0};
            bool ok = true;
            for (int j = 0; j < 3; j++) {
                /* Range's checked float/u64 conversions require a checked bridge.
                 * Signed integral bounds are exactly representable in int64. */
                if (!nodes[j] || !pj_integral(nodes[j]->dtype) || pj_unsigned(nodes[j]->dtype)) {
                    ok = false;
                    break;
                }
                values[j] = pj_expr(p,nodes[j],defined,0,constants,s,flow.next_temp);
                if (!values[j]) { ok = false; break; }
            }
            int label = (*flow.next_label)++;
            int index = p->local_var_indices[stmt->as.for_loop.loop_var_slot];
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
            if (!pj_append(s,"{ uint64_t pj_count_%d = 0; for (;;) {\npj_continue_%d: ;\n",label,label)) return false;
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
            else if (!pj_append(s,"goto pj_%s_%d;\n",target,flow.loop_label)) return false;
        }
        else return false;
    }
    return true;
}

/* First scalar-JIT scope: one float sum/product return, optionally mapped and
 * finally converted to another floating dtype. Accumulation remains serial;
 * integer overflow, mean, extrema/truth, multiple reductions, and statement
 * programs retain their qualified native interpreter paths. */
static const me_expr *pj_scalar_reduction(const me_dsl_compiled_program *p) {
    if (p->block.nstmts != 1 || p->block.stmts[0]->kind != ME_DSL_STMT_RETURN ||
        !pj_float(p->output_dtype)) return NULL;
    const me_expr *node = p->block.stmts[0]->as.return_stmt.expr.expr;
    if (node && IS_FUNCTION(node->type) && !node->function && ARITY(node->type) == 1) {
        node = node->parameters[0];
    }
    if (!node || !is_reduction_node(node) || !pj_float(node->dtype)) return NULL;
    me_reduce_kind kind = reduction_kind(node->function);
    return kind == ME_REDUCE_SUM || kind == ME_REDUCE_PROD ? node : NULL;
}

static void pj_prepare_source(me_dsl_compiled_program *p, const pj_constants *constants) {
#ifdef __EMSCRIPTEN__
    /* Host function-pointer comparison/mask ABI is not the WASM adapter ABI. */
    dsl_tracef("portable jit ineligible: wasm adapter lacks exact i64 and typed bridge ABI");
    (void)p;
    (void)constants;
    return;
#endif
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
    const me_expr *scalar = p->output_is_scalar ? pj_scalar_reduction(p) : NULL;
    if (p->output_is_scalar && !scalar) return;
    for (int i = 0; i < p->n_inputs; i++) if (!pj_type(p->vars.dtypes[i])) {
        dsl_tracef("portable jit ineligible: unsupported input dtype"); return;
    }
    /* User compiler switches can override strict flags, so fail closed rather
     * than claim qualification under arbitrary host toolchain configuration. */
    const char *flags = me_jit_option_value("CFLAGS"), *tcc = getenv("ME_DSL_JIT_TCC_OPTIONS");
    if ((flags && *flags) || (tcc && *tcc)) return;
    bool defined[ME_MAX_VARS] = {false};
    pj_text body = {0};
    p->portable_jit_ncomparisons = p->portable_jit_nmath = 0;
    p->portable_jit_nmath2 = p->portable_jit_npreds = p->portable_jit_niops = 0;
    p->portable_jit_nchecked = 0;
    for (int i = 0; i < p->n_locals; i++) {
        int index = p->local_var_indices[i];
        if (!pj_type(p->vars.dtypes[index]) ||
            !pj_append(&body,"volatile %s local_%d;\n",pj_type(p->vars.dtypes[index]),index)) {
            free(body.text);
            return;
        }
    }
    int next_label = 0, next_temp = 0;
    pj_flow flow = {&next_label,&next_temp,-1};
    char scalar_init[128] = "", scalar_store[128] = "";
    bool lowered;
    if (scalar) {
        char *value = pj_expr(p,scalar->parameters[0],defined,0,constants,&body,&next_temp);
        me_reduce_kind kind = reduction_kind(scalar->function);
        lowered = value && pj_append(&body,"pj_acc = (%s)(pj_acc %s (%s));\n",
            pj_type(scalar->dtype),kind == ME_REDUCE_SUM ? "+" : "*",value);
        free(value);
        snprintf(scalar_init,sizeof(scalar_init),"%s pj_acc = %d;\n",
            pj_type(scalar->dtype),kind == ME_REDUCE_SUM ? 0 : 1);
        snprintf(scalar_store,sizeof(scalar_store),"((%s *)output)[0] = (%s)pj_acc;\n",
            pj_type(p->output_dtype),pj_type(p->output_dtype));
    } else lowered = pj_block(p,&p->block,defined,&body,0,constants,flow);
    if (!lowered) {
        free(body.text);
        dsl_tracef("portable jit ineligible: unsupported typed statements");
        return;
    }
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
        if (!ok) {
            free(preamble.text);
            free(body.text);
            return;
        }
    }
    size_t capacity = body.used + preamble.used + 8192;
    char *source = malloc(capacity);
    me_dsl_jit_ir_program *ir = calloc(1,sizeof(*ir));
    if (!source || !ir) {
        free(preamble.text);
        free(body.text);
        free(source);
        free(ir);
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
        return;
    }
    for (int i = 0; i < p->n_inputs; i++) ir->param_dtypes[i] = p->vars.dtypes[i];
    snprintf(source,capacity,
        "/* portable-1.1 lowering-r18 exact-mixed-comparison-bridge abi-r8 */\n"
        "#include <stdint.h>\n"
        "#include <string.h>\n"
        "typedef _Bool (*pj_cmp)(const void *,double,double);\n"
        "typedef double (*pj_math)(const void *,double);\n"
        "typedef double (*pj_math2)(const void *,double,double);\n"
        "typedef _Bool (*pj_pred)(const void *,double);\n"
        "typedef uint64_t (*pj_iop)(const void *,uint64_t,uint64_t,int *);\n"
        "typedef uint64_t (*pj_checked)(const void *,uint64_t,uint64_t,uint64_t,int *);\n"
        "static inline uint64_t pj_pack_f32(float x) { uint32_t u; memcpy(&u,&x,4); return u; }\n"
        "static inline uint64_t pj_pack_f64(double x) { uint64_t u; memcpy(&u,&x,8); return u; }\n"
        "static inline float pj_unpack_f32(uint64_t x) { uint32_t u=(uint32_t)x; float f; memcpy(&f,&u,4); return f; }\n"
        "static inline double pj_unpack_f64(uint64_t x) { double f; memcpy(&f,&x,8); return f; }\n"
        /* Unsigned arithmetic followed by a bit copy implements modular signed
         * arithmetic without signed overflow or out-of-range signed casts. */
        "#define PJ_INT(S,U) static inline S pj_##S(uint64_t x) { U u=(U)x; S s; memcpy(&s,&u,sizeof(s)); return s; }\n"
        "PJ_INT(int8_t,uint8_t)\nPJ_INT(int16_t,uint16_t)\nPJ_INT(int32_t,uint32_t)\nPJ_INT(int64_t,uint64_t)\n"
        "PJ_INT(uint8_t,uint8_t)\nPJ_INT(uint16_t,uint16_t)\nPJ_INT(uint32_t,uint32_t)\nPJ_INT(uint64_t,uint64_t)\n"
        /* Overflow is detected in unsigned representations, never by executing
         * overflowing signed C arithmetic. Multiplication checks bounds before
         * producing the modular bit pattern. Weak int64 stays checked. */
        "static inline uint64_t pj_weak_add(uint64_t a,uint64_t b,int *status) {\n"
        "uint64_t r=a+b; if ((~(a^b)&(a^r))>>63) *status=-5; return r; }\n"
        "static inline uint64_t pj_weak_sub(uint64_t a,uint64_t b,int *status) {\n"
        "uint64_t r=a-b; if (((a^b)&(a^r))>>63) *status=-5; return r; }\n"
        "static inline uint64_t pj_weak_mul(uint64_t a,uint64_t b,int *status) {\n"
        "int64_t x=pj_int64_t(a), y=pj_int64_t(b);\n"
        "if (x>0 ? (y>0 ? x>INT64_MAX/y : y<INT64_MIN/x) :\n"
        "    (x<0 && (y>0 ? x<INT64_MIN/y : y<INT64_MAX/x))) *status=-5;\n"
        "return a*b; }\n"
        /* Signed values are sign-extended before transport. Only compare their
         * unsigned representations once both operands are known nonnegative. */
        "static inline int pj_icmp(uint64_t a,uint64_t b,int au,int bu) {\n"
        "int an=!au && pj_int64_t(a)<0, bn=!bu && pj_int64_t(b)<0;\n"
        "if (an!=bn) return an ? -1 : 1;\n"
        "return (a>b)-(a<b); }\n"
        /* Inspect representation without executing any floating comparison on
         * a NaN (especially sNaN). Arguments are evaluated once, preserving lazy
         * branch participation and the tree's float32->float64 widening. GCC and
         * Clang inline/specialize these helpers; TCC may emit local calls. */
        "#define PJ_COMPARE(name, op) \\\n"
        "static inline _Bool pj_##name(pj_cmp slow, const void *node, double x, double y) { \\\n"
        "union { double f; uint64_t u; } a, b; a.f=x; b.f=y; \\\n"
        "if ((a.u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000) || \\\n"
        "    (b.u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000)) \\\n"
        "    return slow(node,x,y); \\\n"
        "return x op y; }\n"
        "PJ_COMPARE(eq, ==)\nPJ_COMPARE(ne, !=)\nPJ_COMPARE(lt, <)\n"
        "PJ_COMPARE(le, <=)\nPJ_COMPARE(gt, >)\nPJ_COMPARE(ge, >=)\n"
        "int %s(const void *const *inputs, void *output, int64_t count) {\n"
        "const unsigned char *mask = inputs[%d];\n"
        "const int64_t pj_cap = *(const int64_t *)inputs[%d];\n"
        "int pj_status = 0;\n"
        "%s"
        "%s"
        "for (int64_t i=0; i<count; i++) { if (mask && !mask[i]) continue;\n"
        "%s"
        "%s pj_lane_done: ; } %s return 0; }\n",
        ME_DSL_JIT_SYMBOL_NAME,p->n_inputs,p->n_inputs+ME_DSL_PORTABLE_JIT_CAP_OFF,
        nd_decl,scalar_init,preamble.text ? preamble.text : "",body.text,scalar_store);
    free(preamble.text);
    free(body.text);
    p->jit_ir = ir;
    p->jit_c_source = source;
    p->jit_nparams = p->n_inputs;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *c = (const unsigned char *)source; *c; c++) {
        hash ^= *c; hash *= UINT64_C(1099511628211);
    }
    p->jit_ir_fingerprint = hash;
}

void dsl_portable_prepare_jit_source(me_dsl_compiled_program *p) {
    pj_prepare_source(p, NULL);
}

void dsl_portable_prepare_jit_constants(me_dsl_compiled_program *p,
    const int64_t *values, const bool *known) {
    if (!p || !values || !known || p->jit_c_source || p->jit_ir) return;
    pj_constants constants = {values, known, p->n_inputs};
    pj_prepare_source(p, &constants);
    dsl_try_prepare_jit_runtime(p);
}

void dsl_portable_prepare_jit(me_dsl_compiled_program *p) {
    dsl_portable_prepare_jit_source(p);
    dsl_try_prepare_jit_runtime(p);
}
