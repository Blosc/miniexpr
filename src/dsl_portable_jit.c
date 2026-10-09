/* Portable 1.1 scalar C lowering, revision 8. Never translate source
 * text: the typed tree includes promotion and final output conversions. The
 * private kernel ABI appends a participating mask and host comparison/math bindings;
 * legacy/full kernels retain their unchanged three-argument ABI. */
#include "dsl_compile_internal.h"
#include "dsl_portable_expr.h"
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

static char *pj_expr(me_dsl_compiled_program *p, const me_expr *n, const bool *defined, int depth) {
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
        if (index < p->n_inputs) snprintf(leaf,sizeof(leaf),"((const %s *)inputs[%d])[i]",type,index);
        else if (p->local_slots && p->local_slots[index] >= 0 && defined[index]) {
            snprintf(leaf,sizeof(leaf),"local_%d",index);
        }
        else return NULL;
        return strdup(leaf);
    }
    if (!IS_FUNCTION(n->type) || is_reduction_node(n)) return NULL;
    int arity = ARITY(n->type);
    const char *op = me_portable_operator(n);
    const me_expr *arg0 = arity > 0 ? (const me_expr *)n->parameters[0] : NULL;
    const me_expr *arg1 = arity > 1 ? (const me_expr *)n->parameters[1] : NULL;
    bool float_result = pj_float(n->dtype);
    /* Float/Boolean conversions and integer identities are safe C casts.
     * Integral narrowing/widening is modular in 1.1, so it lowers through the
     * same bit-copy helpers. Float-to-integer checks still reject out-of-range
     * values and must stay on the interpreter where they can report an error. */
    bool conversion = !n->function && arity == 1;
    bool weak_operand = arg0 &&
        (arg0->flags & (ME_EXPR_FLAG_WEAK_LITERAL | ME_EXPR_FLAG_WEAK_SCALAR)) != 0;
    bool int_cast = conversion && pj_integer(n->dtype) && pj_integral(arg0->dtype) &&
        arg0->dtype != n->dtype && !weak_operand;
    if (conversion && pj_integer(n->dtype) && arg0->dtype != n->dtype && !int_cast) return NULL;
    bool where = op && !strcmp(op,"where") && arity == 3;
    bool comparison = op && is_comparison_node(n) && arity == 2;
    if (comparison) {
        /* Weak negative literals can intentionally remain signed beside uint64.
         * C's usual conversions would turn -1 into UINT64_MAX: fail closed. */
        if (pj_integer(arg0->dtype) && pj_integer(arg1->dtype) &&
            pj_unsigned(arg0->dtype) != pj_unsigned(arg1->dtype)) return NULL;
    }
    bool comparison_bridge = comparison && (arg0->dtype != ME_BOOL || arg1->dtype != ME_BOOL);
    bool arithmetic = op && float_result &&
        (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*") || !strcmp(op,"/")) &&
        (arity == 2 || (arity == 1 && !strcmp(op,"-")));
    bool integer_arithmetic = op && pj_integer(n->dtype) &&
        (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*")) &&
        (arity == 2 || (arity == 1 && !strcmp(op,"-")));
    /* Unary/binary math, predicates and the divmod/power operators are lowered
     * through host bridges that replay the exact interpreter scalar path. */
    const char *math = me_portable_math_name(n);
    bool predicate = math && arity == 1 && n->dtype == ME_BOOL && arg0 && pj_float(arg0->dtype);
    bool unary_math = math && arity == 1 && !predicate && float_result && arg0 && pj_float(arg0->dtype);
    bool binary_math = math && arity == 2 && float_result && arg0 && arg1 &&
        pj_float(arg0->dtype) && pj_float(arg1->dtype);
    bool float_operator = op && arity == 2 && float_result && arg0 && arg1 &&
        (!strcmp(op,"//") || !strcmp(op,"%") || !strcmp(op,"**")) &&
        pj_float(arg0->dtype) && pj_float(arg1->dtype);
    bool integer_operator = op && pj_integer(n->dtype) &&
        ((arity == 2 && (!strcmp(op,"%") || !strcmp(op,"//") || !strcmp(op,"<<") ||
                         !strcmp(op,">>") || !strcmp(op,"&") || !strcmp(op,"|") || !strcmp(op,"^"))) ||
         (arity == 1 && !strcmp(op,"~")));
    bool logical = op && ((!strcmp(op,"not") && arity == 1) ||
        ((!strcmp(op,"and") || !strcmp(op,"or")) && arity == 2));
    if (!conversion && !where && !comparison_bridge && !arithmetic && !integer_arithmetic &&
        !logical && !predicate && !unary_math && !binary_math && !float_operator &&
        !integer_operator) return NULL;
    char *args[3] = {0};
    for (int j = 0; j < arity; j++) {
        args[j] = pj_expr(p,n->parameters[j],defined,depth+1);
        if (!args[j]) {
            for (int k = 0; k < arity; k++) free(args[k]);
            return NULL;
        }
    }
    size_t capacity = 128;
    for (int j = 0; j < arity; j++) capacity += strlen(args[j]);
    char *out = malloc(capacity);
    if (out) {
        if (where) snprintf(out,capacity,"((%s)((%s) ? (%s) : (%s)))",type,args[0],args[1],args[2]);
        else if (int_cast) snprintf(out,capacity,"pj_%s((uint64_t)(%s))",type,args[0]);
        else if (conversion) snprintf(out,capacity,"((%s)(%s))",type,args[0]);
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
                snprintf(out,capacity,"pj_%s((uint64_t)(((pj_iop)inputs[%d])(inputs[%d],(uint64_t)(%s),(uint64_t)(%s))))",
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
    return out;
}

typedef struct {
    char *text;
    size_t used, capacity;
} pj_text;

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
static bool pj_block(me_dsl_compiled_program *p, const me_dsl_compiled_block *block,
                     bool *defined, pj_text *s, int depth) {
    if (depth > 64) return false;
    for (int i = 0; i < block->nstmts; i++) {
        const me_dsl_compiled_stmt *stmt = block->stmts[i];
        if (stmt->kind == ME_DSL_STMT_ASSIGN || stmt->kind == ME_DSL_STMT_RETURN) {
            bool assignment = stmt->kind == ME_DSL_STMT_ASSIGN;
            const me_expr *node = assignment ? stmt->as.assign.value.expr : stmt->as.return_stmt.expr.expr;
            char *value = pj_expr(p,node,defined,0);
            if (!value) return false;
            bool ok;
            if (assignment) {
                int index = p->local_var_indices[stmt->as.assign.local_slot];
                ok = node->dtype == p->vars.dtypes[index] &&
                    pj_append(s,"local_%d = %s;\n",index,value);
                defined[index] = true;
            }
            else ok = pj_append(s,"((%s *)output)[i] = %s; continue;\n",pj_type(p->output_dtype),value);
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
                    char *value = pj_expr(p,cond,defined,0);
                    if (!value) return false;
                    bool ok = pj_append(s,"%sif (%s) {\n",j ? "else " : "",value);
                    free(value);
                    if (!ok) return false;
                    body = j == 0 ? &stmt->as.if_stmt.then_block : &stmt->as.if_stmt.elif_branches[j-1].block;
                }
                else {
                    if (!pj_append(s,"else {\n")) return false;
                    body = &stmt->as.if_stmt.else_block;
                }
                memcpy(branch,defined,sizeof(branch));
                if (!pj_block(p,body,branch,s,depth+1) || !pj_append(s,"}\n")) return false;
                if (j == 0) memcpy(merged,branch,sizeof(merged));
                else for (int k = 0; k < p->vars.count; k++) merged[k] &= branch[k];
            }
            memcpy(defined,merged,sizeof(merged));
        }
        else return false;
    }
    return true;
}

void dsl_portable_prepare_jit_source(me_dsl_compiled_program *p) {
#ifdef __EMSCRIPTEN__
    /* Host function-pointer comparison/mask ABI is not the WASM adapter ABI. */
    (void)p;
    return;
#endif
    /* Bounded elementwise statements only; no loops, reductions or ND context. */
    if (!p || p->semantic_profile != ME_DSL_PROFILE_PORTABLE_1_1 ||
        p->jit_request_mode != ME_JIT_ON || p->output_is_scalar || p->compile_ndims ||
        !p->guaranteed_return || p->vars.count > ME_MAX_VARS ||
        !pj_type(p->output_dtype)) {
        if (p) dsl_tracef("portable jit ineligible: request=%d statements=%d rank=%d",p->jit_request_mode,p->block.nstmts,p->compile_ndims);
        return;
    }
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
    for (int i = 0; i < p->n_locals; i++) {
        int index = p->local_var_indices[i];
        if (!pj_type(p->vars.dtypes[index]) ||
            !pj_append(&body,"volatile %s local_%d;\n",pj_type(p->vars.dtypes[index]),index)) {
            free(body.text);
            return;
        }
    }
    if (!pj_block(p,&p->block,defined,&body,0)) {
        free(body.text);
        dsl_tracef("portable jit ineligible: unsupported typed statements");
        return;
    }
    size_t capacity = body.used + 4096;
    char *source = malloc(capacity);
    me_dsl_jit_ir_program *ir = calloc(1,sizeof(*ir));
    if (!source || !ir) {
        free(body.text);
        free(source);
        free(ir);
        return;
    }
    ir->nparams = p->n_inputs;
    ir->param_dtypes = calloc(p->n_inputs ? p->n_inputs : 1,sizeof(me_dtype));
    ir->params = calloc(p->n_inputs ? p->n_inputs : 1,sizeof(char *));
    if (!ir->param_dtypes || !ir->params) {
        free(body.text);
        free(source);
        me_dsl_jit_ir_free(ir);
        return;
    }
    for (int i = 0; i < p->n_inputs; i++) ir->param_dtypes[i] = p->vars.dtypes[i];
    snprintf(source,capacity,
        "/* portable-1.1 lowering-r9 mask-compare-math-abi-r5 */\n"
        "#include <stdint.h>\n"
        "#include <string.h>\n"
        "typedef _Bool (*pj_cmp)(const void *,double,double);\n"
        "typedef double (*pj_math)(const void *,double);\n"
        "typedef double (*pj_math2)(const void *,double,double);\n"
        "typedef _Bool (*pj_pred)(const void *,double);\n"
        "typedef uint64_t (*pj_iop)(const void *,uint64_t,uint64_t);\n"
        /* Unsigned arithmetic followed by a bit copy implements modular signed
         * arithmetic without signed overflow or out-of-range signed casts. */
        "#define PJ_INT(S,U) static inline S pj_##S(uint64_t x) { U u=(U)x; S s; memcpy(&s,&u,sizeof(s)); return s; }\n"
        "PJ_INT(int8_t,uint8_t)\nPJ_INT(int16_t,uint16_t)\nPJ_INT(int32_t,uint32_t)\nPJ_INT(int64_t,uint64_t)\n"
        "PJ_INT(uint8_t,uint8_t)\nPJ_INT(uint16_t,uint16_t)\nPJ_INT(uint32_t,uint32_t)\nPJ_INT(uint64_t,uint64_t)\n"
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
        "for (int64_t i=0; i<count; i++) { if (mask && !mask[i]) continue;\n"
        "%s } return 0; }\n",
        ME_DSL_JIT_SYMBOL_NAME,p->n_inputs,body.text);
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

void dsl_portable_prepare_jit(me_dsl_compiled_program *p) {
    dsl_portable_prepare_jit_source(p);
    dsl_try_prepare_jit_runtime(p);
}
