/* Portable 1.1 scalar C lowering, revision 7. Never translate source
 * text: the typed tree includes promotion and final output conversions. The
 * private kernel ABI appends a participating mask and host comparison bindings;
 * legacy/full kernels retain their unchanged three-argument ABI. */
#include "dsl_compile_internal.h"
#include "dsl_portable_expr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *pj_type(me_dtype dtype) {
    return dtype == ME_FLOAT32 ? "float" : dtype == ME_FLOAT64 ? "double" :
           dtype == ME_BOOL ? "_Bool" : NULL;
}

static char *pj_expr(me_dsl_compiled_program *p, const me_expr *n, int depth) {
    if (!n || depth > 128 || !pj_type(n->dtype)) return NULL;
    const char *type = pj_type(n->dtype);
    char leaf[256];
    if (TYPE_MASK(n->type) == ME_CONSTANT) {
        if (n->flags & ME_EXPR_FLAG_INTEGER_LITERAL) {
            snprintf(leaf,sizeof(leaf),"((%s)(%s(%s)%lluULL))",type,
                n->flags & ME_EXPR_FLAG_NEGATIVE_LITERAL ? "-" : "",type,
                (unsigned long long)n->integer_magnitude);
        }
        else {
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
        if (index < 0 || index >= p->n_inputs) return NULL;
        snprintf(leaf,sizeof(leaf),"((const %s *)inputs[%d])[i]",type,index);
        return strdup(leaf);
    }
    if (!IS_FUNCTION(n->type) || is_reduction_node(n)) return NULL;
    int arity = ARITY(n->type);
    const char *op = me_portable_operator(n);
    /* Only implicit float/bool conversion nodes are supported; integer and
     * explicit cast callbacks must not accidentally enter this subset. */
    bool conversion = !n->function && arity == 1;
    bool where = op && !strcmp(op,"where") && arity == 3;
    bool comparison = op && is_comparison_node(n) && arity == 2;
    bool arithmetic = op && (n->dtype == ME_FLOAT32 || n->dtype == ME_FLOAT64) &&
        (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*") || !strcmp(op,"/")) &&
        (arity == 2 || (arity == 1 && !strcmp(op,"-")));
    bool logical = op && ((!strcmp(op,"not") && arity == 1) ||
        ((!strcmp(op,"and") || !strcmp(op,"or")) && arity == 2));
    if (!conversion && !where && !comparison && !arithmetic && !logical) return NULL;
    char *args[3] = {0};
    for (int j = 0; j < arity; j++) {
        args[j] = pj_expr(p,n->parameters[j],depth+1);
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
        else if (conversion) snprintf(out,capacity,"((%s)(%s))",type,args[0]);
        else if (comparison && (((const me_expr *)n->parameters[0])->dtype != ME_BOOL ||
                                ((const me_expr *)n->parameters[1])->dtype != ME_BOOL)) {
            if (p->portable_jit_ncomparisons == 128) { free(out); out = NULL; }
            else {
                int id = p->portable_jit_ncomparisons++;
                p->portable_jit_comparisons[id] = n;
                const char *name = !strcmp(op, "==") ? "eq" : !strcmp(op, "!=") ? "ne" :
                    !strcmp(op, "<") ? "lt" : !strcmp(op, "<=") ? "le" :
                    !strcmp(op, ">") ? "gt" : "ge";
                snprintf(out,capacity,"pj_%s((pj_cmp)inputs[%d],inputs[%d],(double)(%s),(double)(%s))",
                    name,p->n_inputs+1,p->n_inputs+2+id,args[0],args[1]);
            }
        }
        else if (arity == 1) snprintf(out,capacity,"((%s)(%s(%s)))",type,logical ? "!" : "-",args[0]);
        else snprintf(out,capacity,"((%s)((%s) %s (%s)))",type,args[0],
            logical ? (!strcmp(op,"and") ? "&&" : "||") : op,args[1]);
    }
    for (int j = 0; j < arity; j++) free(args[j]);
    return out;
}

void dsl_portable_prepare_jit(me_dsl_compiled_program *p) {
#ifdef __EMSCRIPTEN__
    /* Host function-pointer comparison/mask ABI is not the WASM adapter ABI. */
    (void)p;
    return;
#endif
    /* First slice: one elementwise return, no ND context, integer signatures,
     * local initialization or control-flow statements. Unsupported stays native
     * interpreter even under an acceleration request. */
    if (!p || p->semantic_profile != ME_DSL_PROFILE_PORTABLE_1_1 ||
        p->jit_request_mode != ME_JIT_ON || p->output_is_scalar || p->compile_ndims ||
        p->block.nstmts != 1 || p->block.stmts[0]->kind != ME_DSL_STMT_RETURN ||
        !pj_type(p->output_dtype)) {
        if (p) dsl_tracef("portable jit ineligible: request=%d statements=%d rank=%d",p->jit_request_mode,p->block.nstmts,p->compile_ndims);
        return;
    }
    for (int i = 0; i < p->n_inputs; i++) if (!pj_type(p->vars.dtypes[i])) {
        dsl_tracef("portable jit ineligible: non-floating/non-Boolean input"); return;
    }
    /* User compiler switches can override strict flags, so fail closed rather
     * than claim qualification under arbitrary host toolchain configuration. */
    const char *flags = me_jit_option_value("CFLAGS"), *tcc = getenv("ME_DSL_JIT_TCC_OPTIONS");
    if ((flags && *flags) || (tcc && *tcc)) return;
    char *expression = pj_expr(p,p->block.stmts[0]->as.return_stmt.expr.expr,0);
    if (!expression) { dsl_tracef("portable jit ineligible: unsupported typed expression"); return; }
    size_t capacity = strlen(expression) + 2048;
    char *source = malloc(capacity);
    me_dsl_jit_ir_program *ir = calloc(1,sizeof(*ir));
    if (!source || !ir) { free(expression); free(source); free(ir); return; }
    ir->nparams = p->n_inputs;
    ir->param_dtypes = calloc(p->n_inputs ? p->n_inputs : 1,sizeof(me_dtype));
    ir->params = calloc(p->n_inputs ? p->n_inputs : 1,sizeof(char *));
    if (!ir->param_dtypes || !ir->params) {
        free(expression); free(source); me_dsl_jit_ir_free(ir); return;
    }
    for (int i = 0; i < p->n_inputs; i++) ir->param_dtypes[i] = p->vars.dtypes[i];
    snprintf(source,capacity,
        "/* portable-1.1 lowering-r7 mask-compare-abi-r3 */\n"
        "#include <stdint.h>\n"
        "typedef _Bool (*pj_cmp)(const void *,double,double);\n"
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
        "((%s *)output)[i] = %s; } return 0; }\n",
        ME_DSL_JIT_SYMBOL_NAME,p->n_inputs,pj_type(p->output_dtype),expression);
    free(expression);
    p->jit_ir = ir;
    p->jit_c_source = source;
    p->jit_nparams = p->n_inputs;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *c = (const unsigned char *)source; *c; c++) {
        hash ^= *c; hash *= UINT64_C(1099511628211);
    }
    p->jit_ir_fingerprint = hash;
    dsl_try_prepare_jit_runtime(p);
}
