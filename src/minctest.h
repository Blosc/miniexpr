#ifndef MINCTEST_H
#define MINCTEST_H

#include <stdio.h>
#include <stdlib.h>
#include "miniexpr.h"

#ifndef ME_EVAL_CHECK
#define ME_EVAL_CHECK(expr, vars, n, out, count) \
    do { \
        int _rc = me_eval((expr), (vars), (n), (out), (count)); \
        if (_rc != ME_EVAL_SUCCESS) { \
            fprintf(stderr, "me_eval failed: %d\n", _rc); \
            exit(1); \
        } \
    } while (0)
#endif

#ifndef ME_COMPILE_CHECK
#define ME_COMPILE_CHECK(expr_str, vars, n, dtype, errp, outp) \
    do { \
        int _rc = me_compile((expr_str), (vars), (n), (dtype), (errp), (outp)); \
        if (_rc != ME_COMPILE_SUCCESS) { \
            fprintf(stderr, "me_compile failed: %d\n", _rc); \
            exit(1); \
        } \
    } while (0)
#endif

#endif
