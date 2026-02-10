#ifndef MINCTEST_H
#define MINCTEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "miniexpr.h"

#if defined(_WIN32) || defined(_WIN64)
static int minctest_setenv(const char *name, const char *value, int overwrite) {
    if (!name || !value || name[0] == '\0' || strchr(name, '=') != NULL) {
        return -1;
    }
    if (!overwrite) {
        char *existing = NULL;
        size_t existing_len = 0;
        if (_dupenv_s(&existing, &existing_len, name) == 0 && existing != NULL) {
            free(existing);
            return 0;
        }
        free(existing);
    }
    return _putenv_s(name, value);
}

static int minctest_unsetenv(const char *name) {
    if (!name || name[0] == '\0' || strchr(name, '=') != NULL) {
        return -1;
    }
    return _putenv_s(name, "");
}

#ifndef setenv
#define setenv(name, value, overwrite) minctest_setenv((name), (value), (overwrite))
#endif
#ifndef unsetenv
#define unsetenv(name) minctest_unsetenv((name))
#endif
#endif

#ifndef ME_EVAL_CHECK
#define ME_EVAL_CHECK(expr, vars, n, out, count) \
    do { \
        int _rc = me_eval((expr), (vars), (n), (out), (count), NULL); \
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
