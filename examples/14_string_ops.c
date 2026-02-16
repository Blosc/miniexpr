/*
 * Example: ME_STRING usage with UCS4 strings
 */
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "../src/miniexpr.h"



#define NAMES_COUNT 4

static int ucs4_to_utf8(uint32_t cp, char *out, int cap) {
    if (cap < 1) return 0;
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF && cap >= 2) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF && cap >= 3) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF && cap >= 4) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

static void ucs4_to_utf8_string(const uint32_t *s, char *out, int cap) {
    int pos = 0;
    if (cap <= 0) return;
    for (int i = 0; s[i] != 0 && pos + 4 < cap; i++) {
        char tmp[4];
        int wrote = ucs4_to_utf8(s[i], tmp, (int)sizeof(tmp));
        if (wrote <= 0 || pos + wrote >= cap) {
            break;
        }
        for (int j = 0; j < wrote; j++) {
            out[pos++] = tmp[j];
        }
    }
    out[pos] = '\0';
}

static void print_bool_array(const char *label, const uint32_t names[][8], const bool *values, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++) {
        char utf8[32];
        ucs4_to_utf8_string(names[i], utf8, (int)sizeof(utf8));
        if (utf8[0] == '\0') {
            snprintf(utf8, sizeof(utf8), "<empty>");
        }
        printf("%d:%s=%s%s", i, utf8, values[i] ? "true" : "false",
               (i + 1 == n) ? "" : ", ");
    }
    printf("]\n");
}

static int run_expr(const char *expr_str, const void **vars, int nvars,
                    bool *out, int nitems) {
    me_expr *expr = NULL;
    int err = 0;

    me_variable variables[] = {
        {"name", ME_STRING, vars[0], ME_VARIABLE, NULL, sizeof(uint32_t) * 8}
    };

    int rc = me_compile(expr_str, variables, nvars, ME_BOOL, &err, &expr);
    if (rc != ME_COMPILE_SUCCESS) {
        printf("compile failed (%d) at %d for: %s\n", rc, err, expr_str);
        return rc;
    }

    if (me_eval(expr, vars, nvars, out, nitems, NULL) != ME_EVAL_SUCCESS) {
        printf("eval failed for: %s\n", expr_str);
        me_free(expr);
        return -1;
    }

    me_free(expr);
    return 0;
}

int main(void) {
    uint32_t names[NAMES_COUNT][8] = {
        {0x0063,0x0061,0x0066,0x00E9,0,0,0,0},    /* café */
        {0x03B2,0x03AD,0x03C4,0x03B1,0,0,0,0},    /* βέτα */
        {0x6C49,0x5B57,0,0,0,0,0,0},              /* 汉字 */
        {0,0,0,0,0,0,0,0}
    };
    const void *vars[] = {names};
    bool out[NAMES_COUNT] = {0};

    if (run_expr("name == \"café\"", vars, 1, out, NAMES_COUNT) == 0) {
        print_bool_array("name == \"café\"", names, out, NAMES_COUNT);
    }
    if (run_expr("name != \"café\"", vars, 1, out, NAMES_COUNT) == 0) {
        print_bool_array("name != \"café\"", names, out, NAMES_COUNT);
    }
    if (run_expr("startswith(name, \"caf\")", vars, 1, out, NAMES_COUNT) == 0) {
        print_bool_array("startswith(name, \"caf\")", names, out, NAMES_COUNT);
    }
    if (run_expr("endswith(name, \"α\")", vars, 1, out, NAMES_COUNT) == 0) {
        print_bool_array("endswith(name, \"α\")", names, out, NAMES_COUNT);
    }
    if (run_expr("contains(name, \"汉\")", vars, 1, out, NAMES_COUNT) == 0) {
        print_bool_array("contains(name, \"汉\")", names, out, NAMES_COUNT);
    }
    if (run_expr("name == \"汉字\"", vars, 1, out, NAMES_COUNT) == 0) {
        print_bool_array("name == \"汉字\"", names, out, NAMES_COUNT);
    }

    return 0;
}
