/*********************************************************************
Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2025  Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifndef MINIEXPR_FUNCTIONS_H
#define MINIEXPR_FUNCTIONS_H

#include "miniexpr.h"
#include <stdbool.h>
#include <stddef.h>

typedef double (*me_fun2)(double, double);

enum { ME_CONSTANT = 1 };

typedef struct state {
    const char* start;
    const char* next;
    int type;

    union {
        double value;
        const double* bound;
        const void* function;
    };

    void* context;
    me_dtype dtype;
    me_dtype target_dtype;

    const me_variable* lookup;
    int lookup_len;
} state;

/* Internal definition of me_expr (opaque to users). */
struct me_expr {
    int type;

    union {
        double value;
        const void* bound;
        const void* function;
    };

    void* output;
    int nitems;
    me_dtype dtype;
    me_dtype input_dtype;
    void* bytecode;
    int ncode;
    void* parameters[1];
};

enum {
    TOK_NULL = ME_CLOSURE7 + 1, TOK_ERROR, TOK_END, TOK_SEP,
    TOK_OPEN, TOK_CLOSE, TOK_NUMBER, TOK_VARIABLE, TOK_INFIX,
    TOK_BITWISE, TOK_SHIFT, TOK_COMPARE, TOK_POW
};

#define TYPE_MASK(TYPE) ((TYPE)&0x0000001F)
#define IS_PURE(TYPE) (((TYPE) & ME_FLAG_PURE) != 0)
#define IS_FUNCTION(TYPE) (((TYPE) & ME_FUNCTION0) != 0)
#define IS_CLOSURE(TYPE) (((TYPE) & ME_CLOSURE0) != 0)
#define ARITY(TYPE) ( ((TYPE) & (ME_FUNCTION0 | ME_CLOSURE0)) ? ((TYPE) & 0x00000007) : 0 )
#define NEW_EXPR(type, ...) new_expr((type), (const me_expr*[]){__VA_ARGS__})
#define CHECK_NULL(ptr, ...) if ((ptr) == NULL) { __VA_ARGS__; return NULL; }

me_expr* new_expr(const int type, const me_expr* parameters[]);
void apply_type_promotion(me_expr* node);
me_dtype infer_output_type(const me_expr* n);
me_dtype infer_result_type(const me_expr* n);
void me_free_parameters(me_expr* n);

bool is_reduction_node(const me_expr* n);
bool is_comparison_node(const me_expr* n);
bool is_float_math_function(const void* func);
size_t dtype_size(me_dtype dtype);
bool has_complex_input_types(const me_expr* n);
bool has_unsupported_complex_function(const me_expr* n);
me_dtype reduction_output_dtype(me_dtype dt, const void* func);
double min_reduce(double x);
double max_reduce(double x);

void optimize(me_expr* n);
void next_token(state* s);
me_expr* list(state* s);

double imag_wrapper(double x);
double real_wrapper(double x);
double where_scalar(double c, double x, double y);

#if defined(_WIN32) || defined(_WIN64)
bool has_complex_node(const me_expr* n);
bool has_complex_input(const me_expr* n);
#endif

#endif
