/* Shared bounded abstract interpreter for lane-local floating programs.
 * This is metadata analysis, not source execution. Checked integer control
 * arithmetic must be proved exact and small before changing invocation domains.
 */
#ifndef DSL_PARTITION_INTERNAL_H
#define DSL_PARTITION_INTERNAL_H
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "dsl_compile_internal.h"
#include "dsl_menudet_expr.h"
#include "dsl_menudet_types.h"

typedef struct {
    bool valid, known_integer, known_float;
    int64_t integer;
    double floating;
} artifact_proof_value;

typedef struct {
    const me_dsl_compiled_program *program;
    artifact_proof_value values[ME_MAX_VARS];
    bool counters[ME_MAX_VARS];
    int *budget;
} artifact_proof;

static artifact_proof_value artifact_proof_integer(int64_t value) {
    if (value < -1000000 || value > 1000000) return (artifact_proof_value){0};
    return (artifact_proof_value){.valid=true,.known_integer=true,.integer=value};
}

static artifact_proof_value artifact_proof_floating(me_dtype dtype, const void *data) {
    double value;
    if (dtype == ME_FLOAT32) {
        uint32_t bits; memcpy(&bits,data,sizeof(bits));
        if ((bits & UINT32_C(0x7fffffff)) >= UINT32_C(0x7f800000)) return (artifact_proof_value){0};
        float single; memcpy(&single,data,sizeof(single)); value=single;
    } else {
        uint64_t bits; memcpy(&bits,data,sizeof(bits));
        if ((bits & UINT64_C(0x7fffffffffffffff)) >= UINT64_C(0x7ff0000000000000)) return (artifact_proof_value){0};
        memcpy(&value,data,sizeof(value));
    }
    return (artifact_proof_value){.valid=true,.known_float=true,.floating=value};
}

static artifact_proof_value artifact_proof_expr(artifact_proof *proof, const me_expr *n, int depth) {
    artifact_proof_value unknown={.valid=true}, invalid={0}, args[3];
    if (!n || depth > 128 || --*proof->budget < 0 || is_reduction_node(n)) return invalid;
    int kind=TYPE_MASK(n->type);
    if (kind == ME_VARIABLE) {
        int index=dsl_synthetic_var_index(n->bound);
        if (index >= 0) return index < proof->program->vars.count ? proof->values[index] : invalid;
    }
    if (kind == ME_CONSTANT) {
        if (n->dtype == ME_BOOL) return artifact_proof_integer(n->value != 0);
        if (n->flags & ME_EXPR_FLAG_INTEGER_LITERAL) {
            if (n->integer_magnitude > 1000000) return invalid;
            return artifact_proof_integer((n->flags & ME_EXPR_FLAG_NEGATIVE_LITERAL) ?
                -(int64_t)n->integer_magnitude : (int64_t)n->integer_magnitude);
        }
        if (n->dtype == ME_FLOAT32 || n->dtype == ME_FLOAT64) {
            return artifact_proof_floating(n->dtype,n->dtype == ME_FLOAT32 ?
                (const void *)&n->literal_f32 : (const void *)&n->value);
        }
        return invalid;
    }
    if (!IS_FUNCTION(n->type)) return invalid;
    int arity=ARITY(n->type);
    if (arity < 1 || arity > 3) return invalid;
    for (int i=0; i<arity; i++) {
        args[i]=artifact_proof_expr(proof,n->parameters[i],depth+1);
        if (!args[i].valid) return invalid;
    }
    if (!n->function) {
        if (n->dtype == ME_FLOAT32 || n->dtype == ME_FLOAT64) {
            const me_expr *arg=n->parameters[0];
            if ((arg->flags & (ME_EXPR_FLAG_WEAK_SCALAR | ME_EXPR_FLAG_WEAK_LITERAL)) &&
                args[0].known_float &&
                fabs(args[0].floating) > (n->dtype == ME_FLOAT32 ? FLT_MAX : DBL_MAX)) return invalid;
            return unknown;
        }
        if (n->dtype == ME_BOOL) return unknown;
        return n->dtype == ME_INT64 && args[0].known_integer ? args[0] : invalid;
    }
    const char *op=me_menudet_operator(n), *math=me_menudet_math_name(n);
    if (op && !strcmp(op,"where") && arity == 3) {
        if (args[0].known_integer) return args[args[0].integer ? 1 : 2];
        if (n->dtype == ME_FLOAT32 || n->dtype == ME_FLOAT64 || n->dtype == ME_BOOL) return unknown;
        return args[1].known_integer && args[2].known_integer && args[1].integer == args[2].integer ? args[1] : invalid;
    }
    if (op && is_comparison_node(n)) {
        if (!args[0].known_integer || !args[1].known_integer) return unknown;
        int64_t a=args[0].integer, b=args[1].integer;
        bool result=!strcmp(op,"==") ? a == b : !strcmp(op,"!=") ? a != b :
            !strcmp(op,"<") ? a < b : !strcmp(op,"<=") ? a <= b :
            !strcmp(op,">") ? a > b : a >= b;
        return artifact_proof_integer(result);
    }
    if (op && (n->dtype == ME_FLOAT32 || n->dtype == ME_FLOAT64) &&
        (!strcmp(op,"+") || !strcmp(op,"-") || !strcmp(op,"*") || !strcmp(op,"/") ||
         !strcmp(op,"**") || !strcmp(op,"%") || !strcmp(op,"//"))) return unknown;
    if (math) {
        for (int i=0; i<arity; i++) {
            const me_expr *arg=n->parameters[i];
            if (arg->dtype != ME_FLOAT32 && arg->dtype != ME_FLOAT64) return invalid;
        }
        if (!((arity == 1 && (dsl_menudet_resolve_unary_math(n) || dsl_menudet_resolve_predicate(n))) ||
            (arity == 2 && dsl_menudet_resolve_binary_math(n)) ||
            (arity == 3 && dsl_menudet_resolve_ternary_math(n)))) return invalid;
        return n->dtype == ME_FLOAT32 || n->dtype == ME_FLOAT64 || n->dtype == ME_BOOL ? unknown : invalid;
    }
    if (!op || (n->dtype != ME_INT64 && n->dtype != ME_BOOL) ||
        !args[0].known_integer || (arity > 1 && !args[1].known_integer)) return invalid;
    int64_t a=args[0].integer, b=arity > 1 ? args[1].integer : 0, result;
    if (!strcmp(op,"+")) result=arity == 1 ? a : a+b;
    else if (!strcmp(op,"-")) result=arity == 1 ? -a : a-b;
    else if (!strcmp(op,"*")) result=a*b;
    else if (!strcmp(op,"not")) result=!a;
    else if (!strcmp(op,"and")) result=a && b;
    else if (!strcmp(op,"or")) result=a || b;
    else if ((!strcmp(op,"%") || !strcmp(op,"//")) && b) {
        int64_t remainder=a%b, quotient=a/b;
        if (remainder && ((remainder < 0) != (b < 0))) { remainder+=b; quotient--; }
        result=!strcmp(op,"%") ? remainder : quotient;
    } else return invalid;
    return artifact_proof_integer(result);
}

static artifact_proof_value artifact_proof_merge(artifact_proof_value a, artifact_proof_value b) {
    if (!a.valid || !b.valid) return (artifact_proof_value){0};
    if (a.known_integer && b.known_integer && a.integer == b.integer) return a;
    return (artifact_proof_value){.valid=true};
}

typedef enum {
    ARTIFACT_FLOW_NEXT = 0,
    ARTIFACT_FLOW_BREAK,
    ARTIFACT_FLOW_CONTINUE
} artifact_proof_flow;

static void artifact_proof_merge_into(artifact_proof *dest, const artifact_proof *src) {
    if (!dest->program) {
        *dest = *src;
        return;
    }
    for (int j = 0; j < dest->program->vars.count; j++) {
        dest->values[j] = artifact_proof_merge(dest->values[j], src->values[j]);
    }
}

static bool artifact_proof_block(artifact_proof *proof, const me_dsl_compiled_block *block,
    int depth, int loop_depth, artifact_proof_flow *flow,
    artifact_proof *break_proof, artifact_proof *continue_proof) {
    if (depth > 16) return false;
    const me_dsl_compiled_program *p=proof->program;
    if (flow) *flow = ARTIFACT_FLOW_NEXT;
    for (int i=0; i<block->nstmts; i++) {
        if (--*proof->budget < 0) return false;
        const me_dsl_compiled_stmt *s=block->stmts[i];
        if (s->kind == ME_DSL_STMT_ASSIGN) {
            int index=p->local_var_indices[s->as.assign.local_slot];
            if (proof->counters[index]) return false;
            artifact_proof_value value=artifact_proof_expr(proof,s->as.assign.value.expr,0);
            if (!value.valid) return false;
            proof->values[index]=value;
        } else if (s->kind == ME_DSL_STMT_RETURN) {
            if (depth || i != block->nstmts-1) return false;
            const me_expr *expr=s->as.return_stmt.expr.expr;
            artifact_proof_value value=artifact_proof_expr(proof,expr,0);
            if (!value.valid) return false;
            if (expr->dtype != p->output_dtype &&
                (expr->flags & (ME_EXPR_FLAG_WEAK_SCALAR | ME_EXPR_FLAG_WEAK_LITERAL)) &&
                value.known_float &&
                fabs(value.floating) > (p->output_dtype == ME_FLOAT32 ? FLT_MAX : DBL_MAX)) return false;
        } else if (s->kind == ME_DSL_STMT_BREAK) {
            if (loop_depth <= 0) return false;
            if (s->as.flow.cond.expr) {
                artifact_proof_value cond = artifact_proof_expr(proof, s->as.flow.cond.expr, 0);
                if (!cond.valid) return false;
                if (cond.known_integer) {
                    if (!cond.integer) continue;
                } else return false;
            }
            if (flow) *flow = ARTIFACT_FLOW_BREAK;
            return true;
        } else if (s->kind == ME_DSL_STMT_CONTINUE) {
            if (loop_depth <= 0) return false;
            if (s->as.flow.cond.expr) {
                artifact_proof_value cond = artifact_proof_expr(proof, s->as.flow.cond.expr, 0);
                if (!cond.valid) return false;
                if (cond.known_integer) {
                    if (!cond.integer) continue;
                } else return false;
            }
            if (flow) *flow = ARTIFACT_FLOW_CONTINUE;
            return true;
        } else if (s->kind == ME_DSL_STMT_IF) {
            if (s->as.if_stmt.n_elifs) return false;
            artifact_proof_value condition=artifact_proof_expr(proof,s->as.if_stmt.cond.expr,0);
            if (!condition.valid) return false;
            if (condition.known_integer) {
                if (!artifact_proof_block(proof,condition.integer ? &s->as.if_stmt.then_block :
                    &s->as.if_stmt.else_block,depth+1,loop_depth,flow,break_proof,continue_proof)) return false;
                if (flow && *flow != ARTIFACT_FLOW_NEXT) return true;
            } else {
                artifact_proof left=*proof, right=*proof;
                artifact_proof_flow left_flow=ARTIFACT_FLOW_NEXT, right_flow=ARTIFACT_FLOW_NEXT;
                if (!artifact_proof_block(&left,&s->as.if_stmt.then_block,depth+1,loop_depth,
                    &left_flow,break_proof,continue_proof)) return false;
                if (!artifact_proof_block(&right,&s->as.if_stmt.else_block,depth+1,loop_depth,
                    &right_flow,break_proof,continue_proof)) return false;
                if (left_flow == ARTIFACT_FLOW_BREAK && right_flow == ARTIFACT_FLOW_BREAK) {
                    if (!break_proof) return false;
                    artifact_proof_merge_into(break_proof, &left);
                    artifact_proof_merge_into(break_proof, &right);
                    if (flow) *flow = ARTIFACT_FLOW_BREAK;
                    return true;
                }
                if (left_flow == ARTIFACT_FLOW_CONTINUE && right_flow == ARTIFACT_FLOW_CONTINUE) {
                    if (!continue_proof) return false;
                    artifact_proof_merge_into(continue_proof, &left);
                    artifact_proof_merge_into(continue_proof, &right);
                    if (flow) *flow = ARTIFACT_FLOW_CONTINUE;
                    return true;
                }
                if (left_flow == ARTIFACT_FLOW_BREAK) {
                    if (!break_proof) return false;
                    artifact_proof_merge_into(break_proof, &left);
                    *proof = right;
                    if (right_flow != ARTIFACT_FLOW_NEXT) {
                        if (flow) *flow = right_flow;
                        return true;
                    }
                } else if (right_flow == ARTIFACT_FLOW_BREAK) {
                    if (!break_proof) return false;
                    artifact_proof_merge_into(break_proof, &right);
                    *proof = left;
                    if (left_flow != ARTIFACT_FLOW_NEXT) {
                        if (flow) *flow = left_flow;
                        return true;
                    }
                } else if (left_flow == ARTIFACT_FLOW_CONTINUE) {
                    if (!continue_proof) return false;
                    artifact_proof_merge_into(continue_proof, &left);
                    *proof = right;
                } else if (right_flow == ARTIFACT_FLOW_CONTINUE) {
                    if (!continue_proof) return false;
                    artifact_proof_merge_into(continue_proof, &right);
                    *proof = left;
                } else {
                    for (int j=0; j<p->vars.count; j++) proof->values[j]=artifact_proof_merge(left.values[j],right.values[j]);
                }
            }
        } else if (s->kind == ME_DSL_STMT_FOR) {
            artifact_proof_value start=artifact_proof_expr(proof,s->as.for_loop.start.expr,0);
            artifact_proof_value stop=artifact_proof_expr(proof,s->as.for_loop.stop.expr,0);
            artifact_proof_value step=artifact_proof_expr(proof,s->as.for_loop.step.expr,0);
            int index=p->local_var_indices[s->as.for_loop.loop_var_slot];
            if (!start.known_integer || !stop.known_integer || !step.known_integer || !step.integer || proof->counters[index]) return false;
            proof->counters[index]=true;
            artifact_proof loop_break={0};
            bool has_survivor=true;
            for (int64_t value=start.integer; step.integer > 0 ? value < stop.integer : value > stop.integer; value+=step.integer) {
                if (--*proof->budget < 0) return false;
                proof->values[index]=artifact_proof_integer(value);
                if (!proof->values[index].valid) return false;
                artifact_proof_flow iter_flow=ARTIFACT_FLOW_NEXT;
                artifact_proof loop_continue={0};
                if (!artifact_proof_block(proof,&s->as.for_loop.body,depth+1,loop_depth+1,
                    &iter_flow,&loop_break,&loop_continue)) return false;
                if (loop_continue.program) {
                    if (iter_flow == ARTIFACT_FLOW_NEXT) {
                        for (int j=0; j<p->vars.count; j++) proof->values[j]=artifact_proof_merge(proof->values[j],loop_continue.values[j]);
                    } else {
                        *proof=loop_continue;
                        iter_flow=ARTIFACT_FLOW_NEXT;
                    }
                }
                if (iter_flow == ARTIFACT_FLOW_BREAK) {
                    has_survivor=false;
                    break;
                }
            }
            if (loop_break.program) {
                if (has_survivor) artifact_proof_merge_into(&loop_break,proof);
                *proof=loop_break;
            }
            proof->counters[index]=false;
            proof->values[index]=(artifact_proof_value){0};
        } else return false;
    }
    return true;
}
#endif
