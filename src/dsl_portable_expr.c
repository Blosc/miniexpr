/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "dsl_portable_expr.h"
#include "dsl_portable_types.h"
#include "dsl_portable_fp.h"
#include "miniexpr_internal.h"

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _MSC_VER
static __declspec(thread) unsigned *p_status_collector;
#else
static _Thread_local unsigned *p_status_collector;
#endif

unsigned *dsl_portable_status_begin(unsigned *flags) {
    unsigned *previous = p_status_collector;
    p_status_collector = flags;
    return previous;
}

void dsl_portable_status_end(unsigned *previous) {
    p_status_collector = previous;
}

void dsl_portable_status_capture(void) {
#ifndef __EMSCRIPTEN__
    if (!p_status_collector) return;
    int raised = fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
    *p_status_collector |= (raised & FE_INVALID ? 1u : 0u) | (raised & FE_DIVBYZERO ? 2u : 0u) |
                          (raised & FE_OVERFLOW ? 4u : 0u) | (raised & FE_UNDERFLOW ? 8u : 0u);
#endif
}

static bool p_unsigned(me_dtype d) {
    return d == ME_UINT8 || d == ME_UINT16 || d == ME_UINT32 || d == ME_UINT64;
}

static bool p_float(me_dtype d) {
    return d == ME_FLOAT32 || d == ME_FLOAT64;
}

static bool p_numeric(me_dtype d) {
    return dsl_portable_numeric_promote(d, d) != ME_AUTO;
}

static bool p_numpy(const me_expr *expr) {
    return (expr->flags & ME_EXPR_FLAG_NUMPY_1_1) != 0;
}

static bool p_weak(const me_expr *expr) {
    return (expr->flags & (ME_EXPR_FLAG_WEAK_LITERAL | ME_EXPR_FLAG_WEAK_SCALAR)) != 0;
}

static bool p_predicate(const char *name) {
    return name && (!strcmp(name, "isfinite") || !strcmp(name, "isinf") ||
                    !strcmp(name, "isnan") || !strcmp(name, "signbit"));
}

static me_dtype p_numpy_math_dtype(me_dtype dtype) {
    if (dtype == ME_BOOL || dtype == ME_INT8 || dtype == ME_UINT8) return ME_AUTO; /* float16 loop */
    if (dtype == ME_INT16 || dtype == ME_UINT16 || dtype == ME_FLOAT32) return ME_FLOAT32;
    return ME_FLOAT64;
}

static me_dtype p_promote(bool numpy, me_dtype left, me_dtype right) {
    return numpy ? dsl_numpy_numeric_promote(left, right) : dsl_portable_numeric_promote(left, right);
}

static me_portable_numeric_status p_signed_op(const me_expr *n, me_portable_integer_op op,
                                              int64_t left, int64_t right, int64_t *out) {
    return p_numpy(n) && !(n->flags & ME_EXPR_FLAG_WEAK_SCALAR) ? dsl_numpy_signed_op(n->dtype, op, left, right, out) :
                        dsl_portable_signed_op(n->dtype, op, left, right, out);
}

static me_portable_numeric_status p_unsigned_op(const me_expr *n, me_portable_integer_op op,
                                                uint64_t left, uint64_t right, uint64_t *out) {
    return p_numpy(n) ? dsl_numpy_unsigned_op(n->dtype, op, left, right, out) :
                        dsl_portable_unsigned_op(n->dtype, op, left, right, out);
}

static bool p_truth(me_dtype d, const me_scalar *v) {
    if (d == ME_BYTES) return v->string && *(const uint8_t *)v->string != 0;
    if (d == ME_STRING) return v->string && *(const uint32_t *)v->string != 0;
    if (d == ME_BOOL) return v->b;
    if (d == ME_FLOAT32) return v->f32 != 0.0f;
    if (d == ME_FLOAT64) return v->f64 != 0.0;
    if (p_unsigned(d)) return v->u64 != 0;
    return v->i64 != 0;
}

#ifdef _MSC_VER
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static double p_widen_float(float value) {
    return value;
}

static double p_double(me_dtype d, const me_scalar *v) {
    if (d == ME_BOOL) return v->b ? 1.0 : 0.0;
    /* Do not speculate float32 widening from an inactive union member: arbitrary
     * float64 low bits can encode a signaling float32 NaN and pollute status. */
    if (d == ME_FLOAT32) return p_widen_float(v->f32);
    if (d == ME_FLOAT64) return v->f64;
    if (p_unsigned(d)) return (double)v->u64;
    return (double)v->i64;
}

static me_portable_numeric_status p_convert(me_dtype from, const me_scalar *v,
                                           me_dtype to, me_scalar *out) {
    if (to == ME_BOOL) {
        out->b = p_truth(from, v);
        return ME_PORTABLE_NUMERIC_OK;
    }
    if (p_float(to)) {
        /* Integer -> float32 rounds directly, not through float64. */
        if (to == ME_FLOAT32) {
            if (p_unsigned(from)) out->f32 = (float)v->u64;
            else if (!p_float(from) && from != ME_BOOL) out->f32 = (float)v->i64;
            else out->f32 = (float)p_double(from, v);
        }
        else out->f64 = p_double(from, v);
        return ME_PORTABLE_NUMERIC_OK;
    }
    if (p_unsigned(to)) {
        if (p_float(from)) return dsl_portable_float_to_unsigned(to, p_double(from, v), &out->u64);
        if (p_unsigned(from)) return dsl_portable_unsigned_to_unsigned(to, v->u64, &out->u64);
        return dsl_portable_signed_to_unsigned(to, from == ME_BOOL ? (int64_t)v->b : v->i64,
                                               &out->u64);
    }
    if (p_float(from)) return dsl_portable_float_to_signed(to, p_double(from, v), &out->i64);
    if (p_unsigned(from)) return dsl_portable_unsigned_to_signed(to, v->u64, &out->i64);
    return dsl_portable_signed_to_signed(to, from == ME_BOOL ? (int64_t)v->b : v->i64, &out->i64);
}

/* Explicit/output conversion is different from literal construction. Integer
 * narrowing is modular, but unstable nonfinite/out-of-range float casts reject. */
static me_portable_numeric_status p_numpy_convert(me_dtype from, const me_scalar *v,
                                                  me_dtype to, me_scalar *out) {
    if (!p_float(from) && to != ME_BOOL && !p_float(to)) {
        uint64_t raw = from == ME_BOOL ? (uint64_t)v->b : p_unsigned(from) ? v->u64 : (uint64_t)v->i64;
        if (p_unsigned(to)) out->u64 = dsl_numpy_unsigned_bits(to, raw);
        else out->i64 = dsl_numpy_signed_bits(to, raw);
        return ME_PORTABLE_NUMERIC_OK;
    }
    return p_convert(from, v, to, out);
}

static bool p_fail(char *reason, size_t cap, const char *message) {
    if (reason && cap) snprintf(reason, cap, "%s", message);
    return false;
}

static bool p_literal_value(const me_expr *n, me_scalar *out) {
    bool negative = (n->flags & ME_EXPR_FLAG_NEGATIVE_LITERAL) != 0;
    if (n->flags & ME_EXPR_FLAG_INTEGER_LITERAL) {
        uint64_t magnitude = n->integer_magnitude;
        if (p_float(n->dtype)) {
            if (n->dtype == ME_FLOAT32) out->f32 = negative ? -(float)magnitude : (float)magnitude;
            else out->f64 = negative ? -(double)magnitude : (double)magnitude;
            return true;
        }
        if (n->dtype == ME_BOOL) {
            if (negative || magnitude > 1) return false;
            out->b = magnitude != 0;
            return true;
        }
        if (!negative) {
            return p_convert(ME_UINT64, &(me_scalar){.u64 = magnitude}, n->dtype, out) == 0;
        }
        if (magnitude > UINT64_C(9223372036854775808)) return false;
        int64_t value = magnitude == UINT64_C(9223372036854775808) ? INT64_MIN : -(int64_t)magnitude;
        return p_convert(ME_INT64, &(me_scalar){.i64 = value}, n->dtype, out) == 0;
    }
    if (n->dtype == ME_FLOAT32) {
        out->f32 = n->literal_f32;
        return p_numpy(n) ? isfinite(n->value) : isfinite(out->f32);
    }
    if (n->dtype == ME_FLOAT64) {
        out->f64 = n->value;
        return isfinite(out->f64);
    }
    /* Contextual floating literals may adopt an integer only if integral. */
    if (trunc(n->value) != n->value) return false;
    return p_convert(ME_FLOAT64, &(me_scalar){.f64 = n->value}, n->dtype, out) == 0;
}

static bool p_context_literal(me_expr *literal, me_dtype context) {
    if (!(literal->flags & ME_EXPR_FLAG_WEAK_LITERAL)) return true;
    if (p_numpy(literal) && !(literal->flags & ME_EXPR_FLAG_INTEGER_LITERAL) && !p_float(context)) {
        literal->dtype = ME_FLOAT64;
        return true;
    }
    if (p_numpy(literal) && context == ME_BOOL) {
        /* Python int/float is higher-kind than bool, regardless of its value. */
        context = literal->flags & ME_EXPR_FLAG_INTEGER_LITERAL ? ME_INT64 : ME_FLOAT64;
    }
    if (!(literal->flags & ME_EXPR_FLAG_INTEGER_LITERAL) &&
        !p_float(context) && trunc(literal->value) != literal->value) return true;
    literal->dtype = context;
    me_scalar value;
    return p_literal_value(literal, &value);
}

static bool p_context_string_literal(me_expr *literal, me_dtype family) {
    if (TYPE_MASK(literal->type) != ME_STRING_CONSTANT) return literal->dtype == family;
    if (family == ME_BYTES) {
        const uint32_t *text = literal->bound;
        for (size_t i = 0; i < literal->str_len; i++) if (text[i] > 127) return false;
    }
    literal->dtype = family;
    literal->itemsize = (literal->str_len ? literal->str_len : 1) * dtype_code_unit(family);
    return true;
}

static bool p_wrap(me_expr **expr, me_dtype to) {
    if ((*expr)->dtype == to) return true;
    me_expr *conv = NEW_EXPR(ME_FUNCTION1 | ME_FLAG_PURE, *expr);
    if (!conv) return false;
    conv->function = NULL;
    conv->input_dtype = (*expr)->dtype;
    conv->dtype = to;
    conv->flags |= ME_EXPR_FLAG_PORTABLE_1;
    conv->flags |= (*expr)->flags & ME_EXPR_FLAG_NUMPY_1_1;
    *expr = conv;
    return true;
}

bool dsl_portable_convert_expr(me_expr **expr, me_dtype dtype) {
    return expr && *expr && p_numeric(dtype) && p_wrap(expr, dtype);
}

static bool p_type(me_expr **slot, char *reason, size_t cap, int depth, bool numpy) {
    me_expr *n = *slot;
    if (!n || depth > 128) return p_fail(reason, cap, "portable expression nesting limit exceeded");
    n->flags |= ME_EXPR_FLAG_PORTABLE_1;
    if (numpy) n->flags |= ME_EXPR_FLAG_NUMPY_1_1;
    int kind = TYPE_MASK(n->type);
    if (kind == ME_VARIABLE || kind == ME_CONSTANT || kind == ME_STRING_CONSTANT) {
        if (is_string_dtype(n->dtype) || kind == ME_STRING_CONSTANT) {
            if (kind == ME_STRING_CONSTANT) {
                n->dtype = ME_STRING;
                n->itemsize = (n->str_len ? n->str_len : 1) * sizeof(uint32_t);
            }
            return (n->itemsize && n->itemsize <= 1024 * 1024 && n->itemsize % dtype_code_unit(n->dtype) == 0) ||
                p_fail(reason, cap, "invalid fixed-string width");
        }
        return p_numeric(n->dtype) || p_fail(reason, cap, "unsupported portable operand dtype");
    }
    if (!IS_FUNCTION(n->type) || IS_CLOSURE(n->type)) {
        return p_fail(reason, cap, "strings and external closures are not implemented in the staged 1.0 path");
    }
    const char *op = me_portable_operator(n);
    const char *math_name = me_portable_math_name(n);
    if (numpy && op && !strcmp(op, "%")) math_name = NULL;
    if (!numpy && math_name && (p_predicate(math_name) || !strcmp(math_name, "minimum") ||
        !strcmp(math_name, "maximum") || !strcmp(math_name, "floating_abs"))) {
        return p_fail(reason, cap, "function requires portable profile 1.1");
    }
    me_reduce_kind reduce = reduction_kind(n->function);
    me_dtype cast = dsl_portable_cast_dtype(n);
    if (!op && !math_name && !reduce && cast == ME_AUTO && !me_portable_string_operation(n)) {
        return p_fail(reason, cap, "operation is not implemented in the staged portable 1.0 interpreter");
    }
    int arity = ARITY(n->type);
    for (int i = 0; i < arity; i++) {
        if (!p_type((me_expr **)&n->parameters[i], reason, cap, depth + 1, numpy)) return false;
    }
    if (op && !strcmp(op, "+") && is_string_dtype(((me_expr *)n->parameters[0])->dtype)) retag_string_concat(n);
    if (me_portable_string_operation(n)) {
        return me_portable_string_validate(n) || p_fail(reason, cap, "invalid or unbounded fixed-string operation");
    }
    for (int i = 0; i < arity; i++) {
        const me_expr *arg = n->parameters[i];
        if (is_string_dtype(arg->dtype) && cast != ME_BOOL &&
            !(op && (!strcmp(op, "not") || !strcmp(op, "and") || !strcmp(op, "or") || !strcmp(op, "where")))) {
            return p_fail(reason, cap, "string operand is invalid for this numeric operation");
        }
    }
    if (math_name && arity == 0) {
        n->type = ME_CONSTANT;
        n->dtype = ME_FLOAT64;
        n->value = !strcmp(math_name, "pi") ? 0x1.921fb54442d18p1 : 0x1.5bf0a8b145769p1;
        return true;
    }
    me_expr *a = n->parameters[0];
    if (numpy && p_predicate(math_name)) {
        n->dtype = ME_BOOL;
        return true;
    }
    if (numpy && op && arity == 1 && !strcmp(op, "+")) {
        if (a->dtype == ME_BOOL) return p_fail(reason, cap, "Boolean unary plus is unsupported");
        n->parameters[0] = NULL;
        me_free(n);
        *slot = a;
        return true;
    }
    if (reduce) {
        if (is_string_dtype(a->dtype)) return p_fail(reason, cap, "reduce numeric or Boolean lanes, not strings");
        if (contains_reduction(a)) return p_fail(reason, cap, "nested reductions are not supported");
        n->dtype = reduce == ME_REDUCE_ANY || reduce == ME_REDUCE_ALL ? ME_BOOL :
                   reduce == ME_REDUCE_MIN || reduce == ME_REDUCE_MAX ? a->dtype :
                   reduce == ME_REDUCE_MEAN ? (a->dtype == ME_FLOAT32 ? ME_FLOAT32 : ME_FLOAT64) :
                   p_float(a->dtype) ? a->dtype : p_unsigned(a->dtype) ? ME_UINT64 : ME_INT64;
        return true;
    }
    /* Fold only a unary source-literal sign, exactly. This is not full-DSL
     * optimization: no arithmetic is executed and no error is suppressed. */
    if (op && arity == 1 && !strcmp(op, "-") && (a->flags & ME_EXPR_FLAG_WEAK_LITERAL)) {
        a->flags ^= ME_EXPR_FLAG_NEGATIVE_LITERAL;
        a->value = -a->value;
        a->literal_f32 = -a->literal_f32;
        n->parameters[0] = NULL;
        me_free(n);
        *slot = a;
        return true;
    }
    if (cast != ME_AUTO) {
        if (is_string_dtype(a->dtype) && cast != ME_BOOL) return p_fail(reason, cap, "explicit string numeric casts are unsupported");
        n->function = NULL;
        n->dtype = cast;
        n->input_dtype = a->dtype;
        return true;
    }
    if (math_name && !strcmp(math_name, "fma")) {
        me_dtype common = ME_AUTO;
        for (int i = 0; i < 3; i++) {
            const me_expr *arg = n->parameters[i];
            if (arg->flags & ME_EXPR_FLAG_WEAK_LITERAL) continue;
            common = common == ME_AUTO ? arg->dtype : p_promote(numpy, common, arg->dtype);
            if (common == ME_AUTO) return p_fail(reason, cap, "unsupported fma operand promotion");
        }
        if (common == ME_AUTO) common = ME_FLOAT64;
        for (int i = 0; i < 3; i++) {
            if (!p_context_literal(n->parameters[i], common)) return p_fail(reason, cap, "fma literal is out of range");
        }
        n->dtype = common == ME_FLOAT32 ? ME_FLOAT32 : ME_FLOAT64;
        for (int i = 0; i < 3; i++) {
            if (!p_wrap((me_expr **)&n->parameters[i], n->dtype)) return false;
        }
        return true;
    }
    if (math_name && !strcmp(math_name, "ldexp")) {
        const me_expr *exponent = n->parameters[1];
        if (p_float(exponent->dtype)) return p_fail(reason, cap, "ldexp exponent must be integral");
        if (numpy && exponent->dtype == ME_UINT64) return p_fail(reason, cap, "ldexp uint64 exponent is unsupported");
        n->dtype = numpy ? p_numpy_math_dtype(a->dtype) : a->dtype == ME_FLOAT32 ? ME_FLOAT32 : ME_FLOAT64;
        if (n->dtype == ME_AUTO) return p_fail(reason, cap, "NumPy ldexp loop requires unsupported float16");
        return p_wrap((me_expr **)&n->parameters[0], n->dtype) &&
               p_wrap((me_expr **)&n->parameters[1], ME_INT64);
    }
    if (math_name && arity == 1) {
        if (!strcmp(math_name, "fac")) {
            if (p_float(a->dtype)) return p_fail(reason, cap, "factorial operand must be integral");
            n->dtype = a->dtype == ME_BOOL ? ME_INT64 : a->dtype;
            return p_wrap((me_expr **)&n->parameters[0], n->dtype);
        }
        bool preserving = !strcmp(math_name, "fabs") || !strcmp(math_name, "ceil") ||
                          !strcmp(math_name, "floor") || !strcmp(math_name, "rint") ||
                          !strcmp(math_name, "round") || !strcmp(math_name, "trunc") ||
                          !strcmp(math_name, "square") || !strcmp(math_name, "sign") ||
                          !strcmp(math_name, "conj") || !strcmp(math_name, "real") ||
                            !strcmp(math_name, "imag");
        if (numpy) {
            if (!strcmp(math_name, "sign") && a->dtype == ME_BOOL) return p_fail(reason, cap, "Boolean sign is unsupported");
            if ((!strcmp(math_name, "square") || !strcmp(math_name, "conj")) && a->dtype == ME_BOOL) {
                n->dtype = ME_INT8;
                return p_wrap((me_expr **)&n->parameters[0], ME_INT8);
            }
            if (!strcmp(math_name, "rint") || !strcmp(math_name, "floating_abs")) preserving = false;
            if (!strcmp(math_name, "round") && a->dtype == ME_BOOL) preserving = false;
            n->dtype = preserving ? a->dtype : p_numpy_math_dtype(a->dtype);
            if (n->dtype == ME_AUTO) return p_fail(reason, cap, "NumPy function loop requires unsupported float16");
            return p_wrap((me_expr **)&n->parameters[0], n->dtype);
        }
        if (numpy && preserving && p_weak(a)) n->flags |= ME_EXPR_FLAG_WEAK_SCALAR;
        n->dtype = preserving ? (a->dtype == ME_BOOL && !numpy ? ME_INT64 : a->dtype) :
                   a->dtype == ME_FLOAT32 ? ME_FLOAT32 : ME_FLOAT64;
        return p_wrap((me_expr **)&n->parameters[0], n->dtype);
    }
    if (!op) op = math_name;
    if (!strcmp(op, "not") || !strcmp(op, "and") || !strcmp(op, "or")) {
        n->dtype = ME_BOOL;
        return true;
    }
    if (arity == 1) {
        if (numpy && p_weak(a)) n->flags |= ME_EXPR_FLAG_WEAK_SCALAR;
        if (numpy && a->dtype == ME_BOOL && !strcmp(op, "-")) return p_fail(reason, cap, "Boolean negation is unsupported; use logical not");
        n->dtype = a->dtype == ME_BOOL && strcmp(op, "~") ? ME_INT64 : a->dtype;
        if (!strcmp(op, "~") && p_float(n->dtype)) return p_fail(reason, cap, "bitwise operand must be integral");
        return p_wrap((me_expr **)&n->parameters[0], n->dtype);
    }
    int left_index = op && !strcmp(op, "where") ? 1 : 0;
    a = n->parameters[left_index];
    me_expr *b = n->parameters[left_index + 1];
    if (left_index && is_string_dtype(a->dtype) && is_string_dtype(b->dtype) && a->dtype != b->dtype) {
        if (TYPE_MASK(a->type) == ME_STRING_CONSTANT) {
            if (!p_context_string_literal(a, b->dtype)) return p_fail(reason, cap, "incompatible string literal family");
        }
        else if (TYPE_MASK(b->type) == ME_STRING_CONSTANT) {
            if (!p_context_string_literal(b, a->dtype)) return p_fail(reason, cap, "incompatible string literal family");
        }
    }
    if (left_index && is_string_dtype(a->dtype) && a->dtype == b->dtype) {
        n->dtype = a->dtype;
        n->itemsize = a->itemsize > b->itemsize ? a->itemsize : b->itemsize;
        return true;
    }
    bool weak_a = p_weak(a);
    bool weak_b = p_weak(b);
    bool numeric_bool = math_name || (strcmp(op, "where") && !is_comparison_node(n) &&
                        strcmp(op, "&") && strcmp(op, "|") && strcmp(op, "^"));
    me_dtype context_a = !numpy && numeric_bool && b->dtype == ME_BOOL ? ME_INT64 : b->dtype;
    me_dtype context_b = !numpy && numeric_bool && a->dtype == ME_BOOL ? ME_INT64 : a->dtype;
    me_dtype original_a = a->dtype, original_b = b->dtype;
    bool literal_ok = true;
    if (weak_a && !weak_b && !p_context_literal(a, context_a)) {
        if (numpy && is_comparison_node(n)) a->dtype = original_a;
        else literal_ok = false;
    }
    if (weak_b && !weak_a && !p_context_literal(b, context_b)) {
        if (numpy && is_comparison_node(n)) b->dtype = original_b;
        else literal_ok = false;
    }
    if (!literal_ok) {
        return p_fail(reason, cap, "source literal is out of range for its operand context");
    }
    if (is_comparison_node(n) && !p_float(a->dtype) && !p_float(b->dtype)) {
        n->dtype = ME_BOOL; /* Mixed int64/uint64 comparison stays exact. */
        return true;
    }
    if (!numpy && (!strcmp(op, "<<") || !strcmp(op, ">>"))) {
        if (p_float(a->dtype) || p_float(b->dtype)) return p_fail(reason, cap, "shift operands must be integral");
        n->dtype = a->dtype == ME_BOOL ? ME_INT64 : a->dtype;
        return p_wrap((me_expr **)&n->parameters[0], n->dtype);
    }
    me_dtype common = p_promote(numpy, a->dtype, b->dtype);
    if (numpy && weak_a != weak_b) {
        const me_expr *weak = weak_a ? a : b, *strong = weak_a ? b : a;
        if (weak->flags & ME_EXPR_FLAG_WEAK_SCALAR) {
            common = p_float(weak->dtype) && !p_float(strong->dtype) ? ME_FLOAT64 :
                     strong->dtype == ME_BOOL && weak->dtype != ME_BOOL ? weak->dtype : strong->dtype;
        }
    }
    if (numpy && weak_a && weak_b) n->flags |= ME_EXPR_FLAG_WEAK_SCALAR;
    if (numpy && math_name && (strcmp(math_name, "minimum") && strcmp(math_name, "maximum") &&
        strcmp(math_name, "fmin") && strcmp(math_name, "fmax") && strcmp(math_name, "fmod") && strcmp(math_name, "ncr") && strcmp(math_name, "npr"))) {
        common = p_numpy_math_dtype(common);
        if (common == ME_AUTO) return p_fail(reason, cap, "NumPy function loop requires unsupported float16");
    }
    bool combinatoric = math_name && (!strcmp(math_name, "ncr") || !strcmp(math_name, "npr"));
    if (combinatoric && p_float(common)) return p_fail(reason, cap, "combinatoric operands must be integral");
    if (numpy && combinatoric && common == ME_BOOL) common = ME_INT64;
    bool ordered = math_name && (!strcmp(math_name, "minimum") || !strcmp(math_name, "maximum") ||
                                  !strcmp(math_name, "fmin") || !strcmp(math_name, "fmax") || !strcmp(math_name, "fmod"));
    if (numpy && math_name && !strcmp(math_name, "fmod") && common == ME_BOOL) common = ME_INT8;
    if (math_name && !combinatoric && common != ME_AUTO && !(numpy && ordered)) common = common == ME_FLOAT32 ? ME_FLOAT32 : ME_FLOAT64;
    if (!strcmp(op, "&") || !strcmp(op, "|") || !strcmp(op, "^")) {
        if (a->dtype == ME_BOOL && b->dtype == ME_BOOL) common = ME_BOOL;
        if (p_float(common)) return p_fail(reason, cap, "bitwise operands must be integral");
    }
    if (numpy && (!strcmp(op, "<<") || !strcmp(op, ">>")) && (p_float(common) || common == ME_BOOL)) {
        if (common == ME_BOOL) common = ME_INT8;
        else return p_fail(reason, cap, "shift operands require an integral common dtype");
    }
    if (numpy && common == ME_BOOL) {
        if (!strcmp(op, "-")) return p_fail(reason, cap, "Boolean subtraction is unsupported");
        if (!strcmp(op, "//") || !strcmp(op, "%") || !strcmp(op, "**")) common = ME_INT8;
    }
    if (!strcmp(op, "/")) common = numpy ? (p_float(common) ? common : ME_FLOAT64) : dsl_portable_division_dtype(a->dtype, b->dtype);
    if (common == ME_AUTO) return p_fail(reason, cap, "unsupported implicit numeric promotion; use an explicit cast");
    n->input_dtype = common;
    n->dtype = is_comparison_node(n) ? ME_BOOL : common;
    return p_wrap((me_expr **)&n->parameters[left_index], common) &&
           p_wrap((me_expr **)&n->parameters[left_index + 1], common);
}

static bool p_validate_literals(const me_expr *n) {
    if (TYPE_MASK(n->type) == ME_CONSTANT) {
        me_scalar value;
        return p_literal_value(n, &value);
    }
    for (int i = 0; i < ARITY(n->type); i++) {
        if (!p_validate_literals(n->parameters[i])) return false;
    }
    return true;
}

bool dsl_portable_type_expr(me_expr **expr, me_dtype output_dtype, char *reason, size_t cap) {
    return dsl_portable_type_expr_profile(expr, output_dtype, ME_DSL_PROFILE_PORTABLE_1_0, reason, cap);
}

bool dsl_portable_type_expr_profile(me_expr **expr, me_dtype output_dtype, me_dsl_semantic_profile profile,
                                    char *reason, size_t cap) {
    if (!expr || !*expr) return p_fail(reason, cap, "missing portable expression");
    if (!p_type(expr, reason, cap, 0, profile == ME_DSL_PROFILE_PORTABLE_1_1)) return false;
    if (!p_validate_literals(*expr)) return p_fail(reason, cap, "source literal is outside its computation dtype");
    if (output_dtype != ME_AUTO) {
        if (is_string_dtype(output_dtype)) return p_context_string_literal(*expr, output_dtype) || p_fail(reason, cap, "incompatible string output family");
        if (!p_numeric(output_dtype) || is_string_dtype((*expr)->dtype)) return p_fail(reason, cap, "unsupported portable output dtype");
        if (!p_wrap(expr, output_dtype)) return p_fail(reason, cap, "cannot allocate portable output conversion");
    }
    return true;
}

/* Independent of the ambient host rounding mode. Preserve signed zero. */
static double p_round_even(double x) {
    if (!isfinite(x) || fabs(x) >= 0x1p52) return x;
    double magnitude = fabs(x), base = floor(magnitude), fraction = magnitude - base;
    if (fraction > 0.5 || (fraction == 0.5 && fmod(base, 2.0) != 0.0)) base += 1.0;
    return copysign(base, x);
}

/* NumPy divmod correction: fmod determines the quotient before rounding.
 * floor(x/y) alone is wrong for e.g. float64 1 // 0.1 and infinite inputs.
 * Each float32 step is performed in float32, never narrowed from a double loop. */
#define P_DIVMOD_IMPL(suffix, type, fmod_fn, floor_fn, copysign_fn) \
static type p_floor_divide##suffix(type x, type y) { \
    if (y == 0) return x / y; \
    type rem = fmod_fn(x, y); \
    type div = (x - rem) / y; \
    /* Ordered correction comparisons on a quiet NaN can spuriously raise \
     * invalid on some compiler paths. Preserve arithmetic diagnostics only. */ \
    if (isnan(div)) return div; \
    if (rem != 0 && ((y < 0) != (rem < 0))) div -= 1; \
    if (div != 0) { \
        type floored = floor_fn(div); \
        if (div - floored > (type)0.5) floored += 1; \
        return floored; \
    } \
    return copysign_fn((type)0, x / y); \
}
P_DIVMOD_IMPL(f, float, fmodf, floorf, copysignf)
P_DIVMOD_IMPL(d, double, fmod, floor, copysign)
#undef P_DIVMOD_IMPL

static uint64_t p_gcd(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

static int p_combinatoric(const me_expr *node, const char *name, const me_scalar *a,
                          const me_scalar *b, me_scalar *out) {
    me_scalar n_value, k_value;
    if (p_convert(node->dtype, a, ME_UINT64, &n_value) != 0) return ME_EVAL_ERR_INVALID_ARG;
    uint64_t n = n_value.u64, k = n;
    if (strcmp(name, "fac")) {
        if (p_convert(node->dtype, b, ME_UINT64, &k_value) != 0) return ME_EVAL_ERR_INVALID_ARG;
        k = k_value.u64;
        if (k > n) return ME_EVAL_ERR_INVALID_ARG;
    }
    bool choose = !strcmp(name, "ncr");
    if (choose && k > n - k) k = n - k;
    uint64_t acc = 1;
    for (uint64_t i = 0; i < k; i++) {
        uint64_t numerator = choose ? n - k + i + 1 : n - i;
        if (choose) {
            uint64_t denominator = i + 1;
            uint64_t divisor = p_gcd(acc, denominator);
            acc /= divisor;
            denominator /= divisor;
            numerator /= denominator; /* Remaining factor divides exactly. */
        }
        if (dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_MUL, acc, numerator, &acc) != 0 ||
            p_convert(ME_UINT64, &(me_scalar){.u64 = acc}, node->dtype, out) != 0) return ME_EVAL_ERR_INVALID_ARG;
    }
    return p_convert(ME_UINT64, &(me_scalar){.u64 = acc}, node->dtype, out) == 0 ? 0 : ME_EVAL_ERR_INVALID_ARG;
}

static int p_math(const me_expr *n, const char *name, const me_scalar *a,
                    const me_scalar *b, me_scalar *out) {
    if (!strcmp(name, "minimum") || !strcmp(name, "maximum") ||
        (p_numpy(n) && (!strcmp(name, "fmin") || !strcmp(name, "fmax")))) {
        bool minimum = !strcmp(name, "minimum") || !strcmp(name, "fmin");
        if (p_float(n->dtype)) {
            double x = p_double(n->dtype, a), y = p_double(n->dtype, b);
            bool propagate = !strcmp(name, "minimum") || !strcmp(name, "maximum");
            if (isnan(x)) *out = propagate ? *a : *b;
            else if (isnan(y)) *out = propagate ? *b : *a;
            else if (x == 0 && y == 0) {
                bool negative = minimum ? signbit(x) || signbit(y) : signbit(x) && signbit(y);
                if (n->dtype == ME_FLOAT32) out->f32 = negative ? -0.0f : 0.0f;
                else out->f64 = negative ? -0.0 : 0.0;
            }
            else *out = (minimum ? x < y : x > y) ? *a : *b; /* ties: second operand */
        }
        else if (n->dtype == ME_BOOL) out->b = minimum ? a->b && b->b : a->b || b->b;
        else if (p_unsigned(n->dtype)) out->u64 = (minimum ? a->u64 < b->u64 : a->u64 > b->u64) ? a->u64 : b->u64;
        else out->i64 = (minimum ? a->i64 < b->i64 : a->i64 > b->i64) ? a->i64 : b->i64;
        return 0;
    }
    if (!strcmp(name, "fac") || !strcmp(name, "ncr") || !strcmp(name, "npr")) return p_combinatoric(n, name, a, b, out);
    if (!strcmp(name, "ldexp")) {
        if (b->i64 < INT_MIN || b->i64 > INT_MAX) return ME_EVAL_ERR_INVALID_ARG;
        if (n->dtype == ME_FLOAT32) out->f32 = ldexpf(a->f32, (int)b->i64);
        else out->f64 = ldexp(a->f64, (int)b->i64);
        return 0;
    }
    if (!p_float(n->dtype)) {
        *out = *a; /* Integer ceil/floor/trunc/round/rint are identities. */
        if (p_numpy(n) && !strcmp(name, "fmod")) {
            if (p_unsigned(n->dtype)) out->u64 = b->u64 ? a->u64 % b->u64 : 0;
            else {
                int bits = (int)(dtype_size(n->dtype) * 8);
                int64_t minimum = bits == 64 ? INT64_MIN : -(INT64_C(1) << (bits - 1));
                out->i64 = !b->i64 || (a->i64 == minimum && b->i64 == -1) ? 0 : a->i64 % b->i64;
            }
            return 0;
        }
        if (n->dtype == ME_BOOL) {
            if (!strcmp(name, "imag")) out->b = false;
            return 0;
        }
        if (!strcmp(name, "square")) {
            me_portable_numeric_status status = p_unsigned(n->dtype) ?
                p_unsigned_op(n, ME_PORTABLE_MUL, a->u64, a->u64, &out->u64) :
                p_signed_op(n, ME_PORTABLE_MUL, a->i64, a->i64, &out->i64);
            return status == 0 ? 0 : ME_EVAL_ERR_INVALID_ARG;
        }
        if (!strcmp(name, "imag")) {
            if (p_unsigned(n->dtype)) out->u64 = 0;
            else out->i64 = 0;
        }
        if (!strcmp(name, "sign")) {
            if (p_unsigned(n->dtype)) out->u64 = a->u64 != 0;
            else out->i64 = (a->i64 > 0) - (a->i64 < 0);
        }
        if (!strcmp(name, "fabs") && !p_unsigned(n->dtype) && a->i64 < 0) {
            return p_signed_op(n, ME_PORTABLE_SUB, 0, a->i64, &out->i64) == 0 ?
                   0 : ME_EVAL_ERR_INVALID_ARG;
        }
        return 0;
    }
    double x = p_double(n->dtype, a);
    if (!strcmp(name, "logaddexp")) {
        double y = p_double(n->dtype, b);
        if (isnan(x) || isnan(y)) {
            if (n->dtype == ME_FLOAT32) out->f32 = NAN;
            else out->f64 = NAN;
        }
        else if (isinf(x) && x == y) *out = *a;
        else if (n->dtype == ME_FLOAT32) {
            float high = fmaxf(a->f32, b->f32), low = fminf(a->f32, b->f32);
            float difference = low - high;
            float term = log1pf(expf(difference));
            out->f32 = high + term;
        }
        else {
            double high = fmax(x, y), low = fmin(x, y);
            out->f64 = high + log1p(exp(low - high));
        }
        return 0;
    }
    if (!strcmp(name, "square") || !strcmp(name, "sign") || !strcmp(name, "conj") ||
        !strcmp(name, "real") || !strcmp(name, "imag")) {
        if (n->dtype == ME_FLOAT32) {
            out->f32 = !strcmp(name, "square") ? a->f32 * a->f32 : !strcmp(name, "imag") ? 0.0f :
                        !strcmp(name, "sign") && a->f32 != 0.0f && !isnan(a->f32) ? copysignf(1.0f, a->f32) :
                        p_numpy(n) && !strcmp(name, "sign") && a->f32 == 0 ? 0.0f : a->f32;
        }
        else out->f64 = !strcmp(name, "square") ? x * x : !strcmp(name, "imag") ? 0.0 :
                        !strcmp(name, "sign") && x != 0.0 && !isnan(x) ? copysign(1.0, x) :
                        p_numpy(n) && !strcmp(name, "sign") && x == 0 ? 0.0 : x;
        return 0;
    }
    if (!strcmp(name, "sinpi") || !strcmp(name, "cospi")) {
        /* Reduce in units of pi, not radians: multiplication first loses the
         * fractional part of large inputs and gives nonzero integer sinpi. */
        bool cosine = !strcmp(name, "cospi");
        double reduced = remainder(x, 2.0);
        double magnitude = fabs(reduced), result;
        if (!isfinite(x)) result = NAN;
        else if (cosine && (magnitude == 0.0 || magnitude == 1.0)) result = magnitude == 0.0 ? 1.0 : -1.0;
        else if (cosine && magnitude == 0.5) result = 0.0;
        else if (!cosine && (magnitude == 0.0 || magnitude == 1.0)) result = copysign(0.0, x);
        else if (!cosine && magnitude == 0.5) result = copysign(1.0, reduced);
        else {
            /* Reflect into [0, 1/4] before evaluating. Subtraction near an
             * integer/half-integer is exact; no large radian argument remains. */
            bool use_cosine;
            double fraction, sign = 1.0;
            if (cosine) {
                sign = magnitude > 0.5 ? -1.0 : 1.0;
                fraction = magnitude > 0.5 ? 1.0 - magnitude : magnitude;
                use_cosine = fraction <= 0.25;
                if (!use_cosine) fraction = 0.5 - fraction;
            }
            else {
                sign = copysign(1.0, reduced);
                fraction = magnitude > 0.5 ? 1.0 - magnitude : magnitude;
                use_cosine = fraction > 0.25;
                if (use_cosine) fraction = 0.5 - fraction;
            }
            if (n->dtype == ME_FLOAT32) {
                float angle = 0x1.921fb6p1f * (float)fraction;
                result = sign * (use_cosine ? cosf(angle) : sinf(angle));
            }
            else {
                double angle = 0x1.921fb54442d18p1 * fraction;
                result = sign * (use_cosine ? cos(angle) : sin(angle));
            }
        }
        if (n->dtype == ME_FLOAT32) out->f32 = (float)result;
        else out->f64 = result;
        return 0;
    }
    if (!strcmp(name, "exp10")) {
        if (n->dtype == ME_FLOAT32) {
            out->f32 = powf(10.0f, a->f32);
        }
        else {
            out->f64 = pow(10.0, x);
        }
        return 0;
    }
    if (!strcmp(name, "rint") || (p_numpy(n) && !strcmp(name, "round"))) {
        if (n->dtype == ME_FLOAT32) out->f32 = (float)p_round_even(x);
        else out->f64 = p_round_even(x);
        return 0;
    }
    if (!strcmp(name, "floating_abs")) {
        if (n->dtype == ME_FLOAT32) out->f32 = fabsf(a->f32);
        else out->f64 = fabs(a->f64);
        return 0;
    }
    if (!strcmp(name, "fmin") || !strcmp(name, "fmax")) {
        double y = p_double(n->dtype, b);
        if (isnan(x)) { *out = *b; return 0; }
        if (isnan(y)) { *out = *a; return 0; }
        if (x == 0.0 && y == 0.0) {
            bool negative = !strcmp(name, "fmin") ? signbit(x) || signbit(y) : signbit(x) && signbit(y);
            if (n->dtype == ME_FLOAT32) out->f32 = negative ? -0.0f : 0.0f;
            else out->f64 = negative ? -0.0 : 0.0;
            return 0;
        }
    }
#define P_UNARY(fn) if (!strcmp(name, #fn)) { \
    if (n->dtype == ME_FLOAT32) out->f32 = fn##f(a->f32); \
    else out->f64 = fn(a->f64); return 0; }
    P_UNARY(acos); P_UNARY(acosh); P_UNARY(asin); P_UNARY(asinh);
    P_UNARY(atan); P_UNARY(atanh); P_UNARY(cbrt); P_UNARY(cos); P_UNARY(cosh);
    P_UNARY(erf); P_UNARY(erfc); P_UNARY(exp); P_UNARY(exp2); P_UNARY(expm1);
    P_UNARY(lgamma); P_UNARY(log); P_UNARY(log10); P_UNARY(log1p); P_UNARY(log2);
    P_UNARY(sin); P_UNARY(sinh); P_UNARY(sqrt); P_UNARY(tan); P_UNARY(tanh);
    P_UNARY(tgamma); P_UNARY(fabs); P_UNARY(ceil); P_UNARY(floor); P_UNARY(round); P_UNARY(trunc);
#undef P_UNARY
#define P_BINARY(fn) if (!strcmp(name, #fn)) { \
    if (n->dtype == ME_FLOAT32) out->f32 = fn##f(a->f32, b->f32); \
    else out->f64 = fn(a->f64, b->f64); return 0; }
    P_BINARY(atan2); P_BINARY(copysign); P_BINARY(fdim); P_BINARY(fmax); P_BINARY(fmin);
    P_BINARY(hypot); P_BINARY(nextafter); P_BINARY(remainder); P_BINARY(fmod);
#undef P_BINARY
    return ME_EVAL_ERR_INVALID_ARG;
}

typedef struct p_eval_context p_eval_context;

typedef struct {
    const me_expr *expr;
    int alternative;
    p_eval_context *context;
} p_branch_cache;

typedef struct {
    const me_expr *expr;
    me_scalar value;
} p_reduce_cache;

struct p_eval_context {
    const void *const *vars;
    const uint8_t *const *initialized;
    int nvars;
    int nitems;
    const uint8_t *mask;
    /* Per-expression-evaluation cache. The mask/bindings cannot change while
     * this workspace lives, and nothing is retained on the shared program. */
    p_reduce_cache reductions[64];
    int nreductions;
    p_branch_cache branches[64];
    int nbranches;
    struct p_string_allocation *strings;
};

typedef struct p_string_allocation {
    struct p_string_allocation *next;
    unsigned char data[];
} p_string_allocation;

static void *p_string_alloc(p_eval_context *ctx, size_t size) {
    p_string_allocation *value = calloc(1, sizeof(*value) + size);
    if (!value) return NULL;
    value->next = ctx->strings;
    ctx->strings = value;
    return value->data;
}

static int p_eval(const me_expr *n, p_eval_context *ctx, int item, me_scalar *out);

static void p_context_free(p_eval_context *ctx) {
    while (ctx->strings) {
        p_string_allocation *value = ctx->strings;
        ctx->strings = value->next;
        free(value);
    }
    for (int i = 0; i < ctx->nbranches; i++) {
        p_eval_context *child = ctx->branches[i].context;
        p_context_free(child);
        free((void *)child->mask);
        free(child);
    }
    ctx->nbranches = 0;
}

static int p_eval_branch(const me_expr *node, int alternative, bool selected_truth,
                         p_eval_context *parent, int item, me_scalar *out) {
    const me_expr *branch = node->parameters[alternative];
    if (!contains_reduction(branch)) return p_eval(branch, parent, item, out);
    for (int i = 0; i < parent->nbranches; i++) {
        p_branch_cache *entry = &parent->branches[i];
        if (entry->expr == node && entry->alternative == alternative) return p_eval(branch, entry->context, item, out);
    }
    p_eval_context *child = calloc(1, sizeof(*child));
    uint8_t *mask = calloc(parent->nitems ? (size_t)parent->nitems : 1, 1);
    if (!child || !mask) {
        free(child);
        free(mask);
        return ME_EVAL_ERR_OOM;
    }
    child->vars = parent->vars;
    child->initialized = parent->initialized;
    child->nvars = parent->nvars;
    child->nitems = parent->nitems;
    child->mask = mask;
    const me_expr *condition = node->parameters[0];
    for (int i = 0; i < parent->nitems; i++) {
        if (parent->mask && !parent->mask[i]) continue;
        me_scalar value;
        int rc = p_eval(condition, parent, i, &value);
        if (rc) {
            free(child);
            free(mask);
            return rc;
        }
        mask[i] = p_truth(condition->dtype, &value) == selected_truth;
    }
    if (parent->nbranches < 64) {
        parent->branches[parent->nbranches++] = (p_branch_cache){node, alternative, child};
        return p_eval(branch, child, item, out);
    }
    int rc = p_eval(branch, child, item, out);
    p_context_free(child);
    free(mask);
    free(child);
    return rc;
}

static int p_reduce(const me_expr *n, p_eval_context *ctx, me_scalar *out) {
    me_reduce_kind kind = reduction_kind(n->function);
    const me_expr *arg = n->parameters[0];
    me_dtype acc_dtype = n->dtype;
    if (kind == ME_REDUCE_MEAN) {
        acc_dtype = p_float(arg->dtype) ? arg->dtype : p_unsigned(arg->dtype) ? ME_UINT64 : ME_INT64;
    }
    me_scalar acc = {0};
    bool truth = kind == ME_REDUCE_ALL;
    int count = 0;
    if (kind == ME_REDUCE_PROD) {
        if (acc_dtype == ME_FLOAT32) acc.f32 = 1.0f;
        else if (acc_dtype == ME_FLOAT64) acc.f64 = 1.0;
        else if (p_unsigned(acc_dtype)) acc.u64 = 1;
        else acc.i64 = 1;
    }
    for (int i = 0; i < ctx->nitems; i++) {
        if (ctx->mask && !ctx->mask[i]) continue;
        me_scalar value, converted;
        int rc = p_eval(arg, ctx, i, &value);
        if (rc) return rc;
        if (kind == ME_REDUCE_ALL || kind == ME_REDUCE_ANY) {
            bool lane = p_truth(arg->dtype, &value);
            truth = kind == ME_REDUCE_ALL ? truth && lane : truth || lane;
            count++;
            continue;
        }
        if (p_convert(arg->dtype, &value, acc_dtype, &converted) != 0) return ME_EVAL_ERR_INVALID_ARG;
        if (kind == ME_REDUCE_MIN || kind == ME_REDUCE_MAX) {
            bool minimum = kind == ME_REDUCE_MIN;
            if (!count) acc = converted;
            else if (p_float(acc_dtype)) {
                double x = p_double(acc_dtype, &acc), y = p_double(acc_dtype, &converted);
                if (isnan(y) || (!isnan(x) && (minimum ? y < x : y > x)) ||
                    (!isnan(x) && x == 0.0 && y == 0.0 && (minimum ? signbit(y) : !signbit(y)))) acc = converted;
            }
            else if (acc_dtype == ME_BOOL) acc.b = minimum ? acc.b && converted.b : acc.b || converted.b;
            else if (p_unsigned(acc_dtype)) {
                if (minimum ? converted.u64 < acc.u64 : converted.u64 > acc.u64) acc = converted;
            }
            else if (minimum ? converted.i64 < acc.i64 : converted.i64 > acc.i64) acc = converted;
        }
        else if (acc_dtype == ME_FLOAT32) {
            acc.f32 = kind == ME_REDUCE_PROD ? acc.f32 * converted.f32 : acc.f32 + converted.f32;
        }
        else if (acc_dtype == ME_FLOAT64) {
            acc.f64 = kind == ME_REDUCE_PROD ? acc.f64 * converted.f64 : acc.f64 + converted.f64;
        }
        else {
            me_portable_integer_op op = kind == ME_REDUCE_PROD ? ME_PORTABLE_MUL : ME_PORTABLE_ADD;
            me_portable_numeric_status status = p_unsigned(acc_dtype) ?
                dsl_portable_unsigned_op(acc_dtype, op, acc.u64, converted.u64, &acc.u64) :
                dsl_portable_signed_op(acc_dtype, op, acc.i64, converted.i64, &acc.i64);
            if (status) return ME_EVAL_ERR_INVALID_ARG;
        }
        count++;
    }
    if (kind == ME_REDUCE_ANY || kind == ME_REDUCE_ALL) out->b = truth;
    else if (kind == ME_REDUCE_MEAN) {
        if (n->dtype == ME_FLOAT32) out->f32 = count ? acc.f32 / (float)count : NAN;
        else out->f64 = count ? p_double(acc_dtype, &acc) / (double)count : NAN;
    }
    else if (!count && (kind == ME_REDUCE_MIN || kind == ME_REDUCE_MAX)) return ME_EVAL_ERR_INVALID_ARG;
    else *out = acc;
    return 0;
}

static int p_eval(const me_expr *n, p_eval_context *ctx, int item, me_scalar *out) {
    if (TYPE_MASK(n->type) == ME_STRING_CONSTANT) {
        void *value = p_string_alloc(ctx, n->itemsize);
        if (!value) return ME_EVAL_ERR_OOM;
        if (n->dtype == ME_BYTES) {
            const uint32_t *text = n->bound;
            for (size_t i = 0; i < n->str_len; i++) ((uint8_t *)value)[i] = (uint8_t)text[i];
        }
        else memcpy(value, n->bound, n->str_len * sizeof(uint32_t));
        out->string = value;
        return 0;
    }
    if (TYPE_MASK(n->type) == ME_CONSTANT) {
        return p_literal_value(n, out) ? ME_EVAL_SUCCESS : ME_EVAL_ERR_INVALID_ARG;
    }
    if (TYPE_MASK(n->type) == ME_VARIABLE) {
        if (!is_synthetic_address(n->bound)) return ME_EVAL_ERR_INVALID_ARG;
        int index = (int)((const char *)n->bound - synthetic_var_addresses);
        if (index < 0 || index >= ctx->nvars || !ctx->vars[index]) return ME_EVAL_ERR_VAR_MISMATCH;
        if (ctx->initialized && ctx->initialized[index] && !ctx->initialized[index][item]) return ME_EVAL_ERR_INVALID_ARG;
        if (is_string_dtype(n->dtype)) out->string = (const unsigned char *)ctx->vars[index] + (size_t)item * n->itemsize;
        else read_scalar((const unsigned char *)ctx->vars[index] + (size_t)item * dtype_size(n->dtype), n->dtype, out);
        return ME_EVAL_SUCCESS;
    }
    if (me_portable_string_operation(n)) {
        const void *values[7] = {0};
        me_scalar args[7];
        double indices[7];
        for (int i = 0; i < ARITY(n->type); i++) {
            const me_expr *arg = n->parameters[i];
            int rc = p_eval(arg, ctx, item, &args[i]);
            if (rc) return rc;
            if (is_string_dtype(arg->dtype)) values[i] = args[i].string;
            else {
                indices[i] = p_double(arg->dtype, &args[i]);
                values[i] = &indices[i];
            }
        }
        if (is_string_dtype(n->dtype)) {
            void *value = p_string_alloc(ctx, n->itemsize);
            if (!value) return ME_EVAL_ERR_OOM;
            if (!me_portable_string_execute(n, values, value)) return ME_EVAL_ERR_INVALID_ARG;
            out->string = value;
        }
        else if (!me_portable_string_execute(n, values, &out->b)) return ME_EVAL_ERR_INVALID_ARG;
        return 0;
    }
    if (is_reduction_node(n)) {
        for (int i = 0; i < ctx->nreductions; i++) {
            if (ctx->reductions[i].expr == n) {
                *out = ctx->reductions[i].value;
                return 0;
            }
        }
        int rc = p_reduce(n, ctx, out);
        if (!rc && ctx->nreductions < 64) {
            ctx->reductions[ctx->nreductions++] = (p_reduce_cache){n, *out};
        }
        return rc;
    }
    me_expr *a_node = n->parameters[0];
    me_scalar a, b;
    int rc = p_eval(a_node, ctx, item, &a);
    if (rc) return rc;
    if (!n->function) {
        return (p_numpy(n) && !p_weak(a_node) ? p_numpy_convert(a_node->dtype, &a, n->dtype, out) :
                             p_convert(a_node->dtype, &a, n->dtype, out)) == 0 ? 0 : ME_EVAL_ERR_INVALID_ARG;
    }
    const char *op = me_portable_operator(n);
    const char *math_name = me_portable_math_name(n);
    if (p_numpy(n) && op && !strcmp(op, "%")) math_name = NULL;
    if (p_predicate(math_name)) {
        double x = p_float(a_node->dtype) ? p_double(a_node->dtype, &a) : 0;
        out->b = !strcmp(math_name, "isfinite") ? !p_float(a_node->dtype) || isfinite(x) :
                 !strcmp(math_name, "isnan") ? isnan(x) : !strcmp(math_name, "isinf") ? isinf(x) :
                 p_float(a_node->dtype) ? signbit(x) != 0 : a_node->dtype != ME_BOOL && !p_unsigned(a_node->dtype) && a.i64 < 0;
        return 0;
    }
    if (math_name && ARITY(n->type) == 1) return p_math(n, math_name, &a, NULL, out);
    if (math_name && !strcmp(math_name, "fma")) {
        me_scalar c;
        rc = p_eval(n->parameters[1], ctx, item, &b);
        if (rc) return rc;
        rc = p_eval(n->parameters[2], ctx, item, &c);
        if (rc) return rc;
        if (n->dtype == ME_FLOAT32) out->f32 = fmaf(a.f32, b.f32, c.f32);
        else out->f64 = fma(a.f64, b.f64, c.f64);
        return 0;
    }
    if (!op) op = math_name;
    if (!op) return ME_EVAL_ERR_INVALID_ARG;
    if (!strcmp(op, "not")) {
        out->b = !p_truth(a_node->dtype, &a);
        return 0;
    }
    if (!strcmp(op, "where")) {
        bool selected = p_truth(a_node->dtype, &a);
        int alternative = selected ? 1 : 2;
        int status = p_eval_branch(n, alternative, selected, ctx, item, out);
        if (!status && is_string_dtype(n->dtype)) {
            const me_expr *chosen = n->parameters[alternative];
            void *value = p_string_alloc(ctx, n->itemsize);
            if (!value) return ME_EVAL_ERR_OOM;
            memcpy(value, out->string, chosen->itemsize);
            out->string = value;
        }
        return status;
    }
    bool logical = !strcmp(op, "and") || !strcmp(op, "or");
    if (logical && ((!strcmp(op, "and") && !p_truth(a_node->dtype, &a)) ||
                    (!strcmp(op, "or") && p_truth(a_node->dtype, &a)))) {
        out->b = !strcmp(op, "or");
        return 0;
    }
    if (ARITY(n->type) == 1) {
        if (p_float(n->dtype)) {
            if (n->dtype == ME_FLOAT32) out->f32 = -a.f32;
            else out->f64 = -a.f64;
            return 0;
        }
        if (n->dtype == ME_BOOL) {
            out->b = !a.b;
            return 0;
        }
        me_portable_numeric_status status;
        if (p_unsigned(n->dtype)) {
            status = p_unsigned_op(n, !strcmp(op, "~") ? ME_PORTABLE_XOR : ME_PORTABLE_SUB,
                                              !strcmp(op, "~") ? UINT64_MAX >> (64 - 8 * dtype_size(n->dtype)) : 0,
                                              a.u64, &out->u64);
        }
        else status = p_signed_op(n, !strcmp(op, "~") ? ME_PORTABLE_XOR : ME_PORTABLE_SUB,
                                             !strcmp(op, "~") ? -1 : 0, a.i64, &out->i64);
        return status == 0 ? 0 : ME_EVAL_ERR_INVALID_ARG;
    }
    const me_expr *b_node = n->parameters[1];
    rc = logical ? p_eval_branch(n, 1, !strcmp(op, "and"), ctx, item, &b) : p_eval(b_node, ctx, item, &b);
    if (rc) return rc;
    if (math_name) return p_math(n, math_name, &a, &b, out);
    if (logical) {
        out->b = p_truth(b_node->dtype, &b);
        return 0;
    }
    if (is_comparison_node(n)) {
        int cmp = 0;
        bool unordered = false;
        if (p_float(a_node->dtype)) {
            double x = p_double(a_node->dtype, &a), y = p_double(b_node->dtype, &b);
            unordered = isnan(x) || isnan(y);
            cmp = (x > y) - (x < y);
        }
        else if (p_unsigned(a_node->dtype) && !p_unsigned(b_node->dtype)) {
            cmp = -dsl_portable_compare_signed_unsigned(b_node->dtype == ME_BOOL ? (int64_t)b.b : b.i64, a.u64);
        }
        else if (!p_unsigned(a_node->dtype) && p_unsigned(b_node->dtype)) {
            cmp = dsl_portable_compare_signed_unsigned(a_node->dtype == ME_BOOL ? (int64_t)a.b : a.i64, b.u64);
        }
        else if (p_unsigned(a_node->dtype)) cmp = (a.u64 > b.u64) - (a.u64 < b.u64);
        else {
            int64_t x = a_node->dtype == ME_BOOL ? (int64_t)a.b : a.i64;
            int64_t y = b_node->dtype == ME_BOOL ? (int64_t)b.b : b.i64;
            cmp = (x > y) - (x < y);
        }
        me_cmp_kind kind = comparison_kind(n->function);
        out->b = kind == ME_CMP_NE ? (unordered || cmp != 0) :
                 !unordered && (kind == ME_CMP_EQ ? cmp == 0 : kind == ME_CMP_LT ? cmp < 0 :
                 kind == ME_CMP_LE ? cmp <= 0 : kind == ME_CMP_GT ? cmp > 0 : cmp >= 0);
        return 0;
    }
    if (p_float(n->dtype)) {
        if (n->dtype == ME_FLOAT32) {
            float x = a.f32, y = b.f32;
            if (!strcmp(op, "+")) out->f32 = x + y;
            else if (!strcmp(op, "-")) out->f32 = x - y;
            else if (!strcmp(op, "*")) out->f32 = x * y;
            else if (!strcmp(op, "/")) out->f32 = x / y;
            else if (!strcmp(op, "//")) out->f32 = p_numpy(n) ? p_floor_dividef(x, y) : floorf(x / y);
            else if (!strcmp(op, "%")) {
                float rem = fmodf(x, y);
                if (rem != 0.0f && signbit(rem) != signbit(y)) rem += y;
                out->f32 = rem == 0.0f ? copysignf(0.0f, y) : rem;
            }
            else out->f32 = powf(x, y);
        }
        else {
            double x = a.f64, y = b.f64;
            if (!strcmp(op, "+")) out->f64 = x + y;
            else if (!strcmp(op, "-")) out->f64 = x - y;
            else if (!strcmp(op, "*")) out->f64 = x * y;
            else if (!strcmp(op, "/")) out->f64 = x / y;
            else if (!strcmp(op, "//")) out->f64 = p_numpy(n) ? p_floor_divided(x, y) : floor(x / y);
            else if (!strcmp(op, "%")) {
                double rem = fmod(x, y);
                if (rem != 0.0 && signbit(rem) != signbit(y)) rem += y;
                out->f64 = rem == 0.0 ? copysign(0.0, y) : rem;
            }
            else out->f64 = pow(x, y);
        }
        return 0;
    }
    if (n->dtype == ME_BOOL) {
        out->b = !strcmp(op, "&") || !strcmp(op, "*") ? a.b && b.b :
                 !strcmp(op, "|") || !strcmp(op, "+") ? a.b || b.b : a.b != b.b;
        return 0;
    }
    me_portable_integer_op integer_op;
    if (!strcmp(op, "+")) integer_op = ME_PORTABLE_ADD;
    else if (!strcmp(op, "-")) integer_op = ME_PORTABLE_SUB;
    else if (!strcmp(op, "*")) integer_op = ME_PORTABLE_MUL;
    else if (!strcmp(op, "%")) integer_op = ME_PORTABLE_MOD;
    else if (!strcmp(op, "//")) integer_op = ME_PORTABLE_FLOORDIV;
    else if (!strcmp(op, "**")) integer_op = ME_PORTABLE_POW;
    else if (!strcmp(op, "<<")) integer_op = ME_PORTABLE_SHL;
    else if (!strcmp(op, ">>")) integer_op = ME_PORTABLE_SHR;
    else if (!strcmp(op, "&")) integer_op = ME_PORTABLE_AND;
    else if (!strcmp(op, "|")) integer_op = ME_PORTABLE_OR;
    else integer_op = ME_PORTABLE_XOR;
    if (!p_numpy(n) && (integer_op == ME_PORTABLE_SHL || integer_op == ME_PORTABLE_SHR)) {
        me_scalar count;
        if (p_convert(b_node->dtype, &b, ME_INT64, &count) != 0 || count.i64 < 0 ||
            count.i64 >= (int64_t)(8 * dtype_size(n->dtype))) return ME_EVAL_ERR_INVALID_ARG;
        if (p_unsigned(n->dtype)) b.u64 = (uint64_t)count.i64;
        else b.i64 = count.i64;
    }
    me_portable_numeric_status status = p_unsigned(n->dtype) ?
        p_unsigned_op(n, integer_op, a.u64, b.u64, &out->u64) :
        p_signed_op(n, integer_op, a.i64, b.i64, &out->i64);
    return status == 0 ? 0 : ME_EVAL_ERR_INVALID_ARG;
}

/* Private JIT bridges replay the evaluator's scalar routines so the host
 * compiler never has to reproduce promotion, libm choice or exception policy. */
double dsl_portable_jit_unary_math(const void *node, double x) {
    const me_expr *n = node;
    me_scalar a = {0}, result = {0};
    if (n->dtype == ME_FLOAT32) a.f32 = (float)x;
    else a.f64 = x;
    p_math(n, me_portable_math_name(n), &a, NULL, &result);
    return n->dtype == ME_FLOAT32 ? (double)result.f32 : result.f64;
}

double dsl_portable_jit_binary_math(const void *node, double x, double y) {
    const me_expr *n = node;
    /* Probe the operator first: divmod/power are not math names, and the
     * math-name lookup is comparatively expensive for these per-lane calls. */
    const char *op = me_portable_operator(n);
    bool divmod_pow = op && (!strcmp(op, "//") || !strcmp(op, "%") || !strcmp(op, "**"));
    me_scalar result = {0};
    if (divmod_pow) {
        if (n->dtype == ME_FLOAT32) {
            float xf = (float)x, yf = (float)y;
            if (!strcmp(op, "//")) result.f32 = p_numpy(n) ? p_floor_dividef(xf, yf) : floorf(xf / yf);
            else if (!strcmp(op, "%")) {
                float rem = fmodf(xf, yf);
                if (rem != 0.0f && signbit(rem) != signbit(yf)) rem += yf;
                result.f32 = rem == 0.0f ? copysignf(0.0f, yf) : rem;
            }
            else result.f32 = powf(xf, yf);
        }
        else {
            if (!strcmp(op, "//")) result.f64 = p_numpy(n) ? p_floor_divided(x, y) : floor(x / y);
            else if (!strcmp(op, "%")) {
                double rem = fmod(x, y);
                if (rem != 0.0 && signbit(rem) != signbit(y)) rem += y;
                result.f64 = rem == 0.0 ? copysign(0.0, y) : rem;
            }
            else result.f64 = pow(x, y);
        }
        return n->dtype == ME_FLOAT32 ? (double)result.f32 : result.f64;
    }
    me_scalar a = {0}, b = {0};
    if (n->dtype == ME_FLOAT32) { a.f32 = (float)x; b.f32 = (float)y; }
    else { a.f64 = x; b.f64 = y; }
    p_math(n, me_portable_math_name(n), &a, &b, &result);
    return n->dtype == ME_FLOAT32 ? (double)result.f32 : result.f64;
}

bool dsl_portable_jit_predicate(const void *node, double x) {
    const char *name = me_portable_math_name((const me_expr *)node);
    return !strcmp(name, "isfinite") ? isfinite(x) :
           !strcmp(name, "isnan") ? isnan(x) :
           !strcmp(name, "isinf") ? isinf(x) : signbit(x) != 0;
}

/* Integer remainder/floor-division/shift/bitwise reuse the exact checked
 * modular routines; operands arrive as raw bit patterns. */
uint64_t dsl_portable_jit_int_op(const void *node, uint64_t a, uint64_t b) {
    const me_expr *n = node;
    const char *op = me_portable_operator(n);
    bool complement = op && !strcmp(op, "~");
    me_portable_integer_op iop;
    if (op && !strcmp(op, "%")) iop = ME_PORTABLE_MOD;
    else if (op && !strcmp(op, "//")) iop = ME_PORTABLE_FLOORDIV;
    else if (op && !strcmp(op, "<<")) iop = ME_PORTABLE_SHL;
    else if (op && !strcmp(op, ">>")) iop = ME_PORTABLE_SHR;
    else if (op && !strcmp(op, "&")) iop = ME_PORTABLE_AND;
    else if (op && !strcmp(op, "|")) iop = ME_PORTABLE_OR;
    else if (op && !strcmp(op, "^")) iop = ME_PORTABLE_XOR;
    else if (complement) iop = ME_PORTABLE_XOR;
    else return 0;
    me_scalar result = {0};
    if (p_unsigned(n->dtype)) {
        uint64_t right = complement ? (UINT64_MAX >> (64 - 8 * (int)dtype_size(n->dtype))) : b;
        p_unsigned_op(n, iop, a, right, &result.u64);
        return result.u64;
    }
    int64_t right = complement ? -1 : (int64_t)b;
    p_signed_op(n, iop, (int64_t)a, right, &result.i64);
    return (uint64_t)result.i64;
}

static bool p_jit_nan_compare(const void *node, double x, double y);

bool dsl_portable_float_compare(const void *node, double x, double y) {
    /* Execute the exact host comparison path: NaN exception instructions differ
     * between host/JIT compilers even when Boolean values are identical. Only
     * the already-computed operands enter this bridge, not their expression trees. */
    const me_expr *original = node;
    uint64_t xb, yb;
    memcpy(&xb,&x,sizeof(xb)); memcpy(&yb,&y,sizeof(yb));
    if ((xb & UINT64_C(0x7fffffffffffffff)) <= UINT64_C(0x7ff0000000000000) &&
        (yb & UINT64_C(0x7fffffffffffffff)) <= UINT64_C(0x7ff0000000000000)) {
        /* Finite/infinite comparisons cannot raise. Keep the rare NaN path on
         * the exact evaluator instruction sequence; avoid recursive dispatch
         * for ordinary lanes without changing classification/exception policy. */
        int cmp = (x > y) - (x < y);
        me_cmp_kind kind = comparison_kind(original->function);
        return kind == ME_CMP_NE ? cmp != 0 : kind == ME_CMP_EQ ? cmp == 0 :
            kind == ME_CMP_LT ? cmp < 0 : kind == ME_CMP_LE ? cmp <= 0 :
            kind == ME_CMP_GT ? cmp > 0 : cmp >= 0;
    }
    return p_jit_nan_compare(node,x,y);
}

#ifdef _MSC_VER
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static bool p_jit_nan_compare(const void *node, double x, double y) {
    const me_expr *original = node;
    /* Leaves are allocated only through the header, not sizeof(me_expr), whose
     * trailing parameters[1] slot need not exist on a variable/constant. */
    me_expr left = {0}, right = {0};
    memcpy(&left, original->parameters[0], offsetof(me_expr, parameters));
    memcpy(&right, original->parameters[1], offsetof(me_expr, parameters));
    /* p_literal_value intentionally rejects NaNs; use typed variables instead. */
    me_scalar values[2] = {{0},{0}};
    if (left.dtype == ME_FLOAT32) values[0].f32 = (float)x;
    else if (left.dtype == ME_BOOL) values[0].b = x != 0;
    else values[0].f64 = x;
    if (right.dtype == ME_FLOAT32) values[1].f32 = (float)y;
    else if (right.dtype == ME_BOOL) values[1].b = y != 0;
    else values[1].f64 = y;
    const void *vars[2] = {&values[0],&values[1]};
    left.type = right.type = ME_VARIABLE;
    left.bound = synthetic_var_addresses; right.bound = synthetic_var_addresses + 1;
    /* me_expr has trailing operand storage; allocate sufficient stack space. */
    union { me_expr expr; unsigned char bytes[sizeof(me_expr)+2*sizeof(void *)]; } storage;
    memcpy(&storage.expr,original,offsetof(me_expr, parameters));
    storage.expr.parameters[0] = &left; storage.expr.parameters[1] = &right;
    p_eval_context ctx = {.vars=vars,.nvars=2,.nitems=1}; me_scalar result;
    return p_eval(&storage.expr,&ctx,0,&result) == 0 && result.b;
}

int dsl_portable_eval_expr(const me_expr *expr, const void *const *vars, int nvars,
                           const uint8_t *const *initialized,
                           int item, int nitems, const uint8_t *mask, void *out) {
    if (!expr || !out || item < 0 || nitems < 0 || nvars < 0 || (!vars && nvars)) return ME_EVAL_ERR_INVALID_ARG;
    p_eval_context ctx = {.vars = vars, .initialized = initialized, .nvars = nvars, .nitems = nitems, .mask = mask};
    fenv_t saved;
    if (!dsl_portable_fp_begin(&saved)) return ME_EVAL_ERR_INVALID_ARG;
    me_scalar value;
    int rc = p_eval(expr, &ctx, item, &value);
    if (!rc) {
        if (is_string_dtype(expr->dtype)) memcpy(out, value.string, expr->itemsize);
        else write_scalar(out, expr->dtype, expr->dtype, &value);
    }
    p_context_free(&ctx);
    if (!dsl_portable_fp_end(&saved)) return ME_EVAL_ERR_INVALID_ARG;
    return rc;
}

int dsl_portable_eval_expr_masked(const me_expr *expr, const void *const *vars, int nvars,
                                  const uint8_t *const *initialized,
                                  int nitems, const uint8_t *mask, size_t stride, void *out) {
    if (!expr || !out || nitems < 0 || nvars < 0 || (!vars && nvars) ||
        stride < dtype_size(expr->dtype)) return ME_EVAL_ERR_INVALID_ARG;
    p_eval_context ctx = {.vars = vars, .initialized = initialized, .nvars = nvars, .nitems = nitems, .mask = mask};
    fenv_t saved;
    if (!dsl_portable_fp_begin(&saved)) return ME_EVAL_ERR_INVALID_ARG;
    int status = ME_EVAL_SUCCESS;
    for (int i = 0; i < nitems; i++) {
        if (mask && !mask[i]) continue;
        me_scalar value;
        int rc = p_eval(expr, &ctx, i, &value);
        if (rc) {
            status = rc;
            break;
        }
        if (is_string_dtype(expr->dtype)) {
            if (stride < expr->itemsize) { status = ME_EVAL_ERR_INVALID_ARG; break; }
            memset((unsigned char *)out + (size_t)i * stride, 0, stride);
            memcpy((unsigned char *)out + (size_t)i * stride, value.string, expr->itemsize);
        }
        else write_scalar((unsigned char *)out + (size_t)i * stride, expr->dtype, expr->dtype, &value);
    }
    p_context_free(&ctx);
    if (!dsl_portable_fp_end(&saved)) return ME_EVAL_ERR_INVALID_ARG;
    return status;
}
