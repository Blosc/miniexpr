/*
 * Sanity check: ensure builtin function list stays alphabetically sorted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MINIEXPR_SOURCE_DIR
#error "MINIEXPR_SOURCE_DIR must be defined"
#endif

int main(void) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/src/functions.c", MINIEXPR_SOURCE_DIR);

    FILE *fp = fopen(path, "r");
    if (!fp) {
        printf("Failed to open %s\n", path);
        return 1;
    }

    int in_functions = 0;
    char line[512];
    char prev_name[128] = {0};
    int failures = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (!in_functions) {
            if (strstr(line, "static const me_variable functions[]") ||
                strstr(line, "static const me_variable_ex functions[]")) {
                in_functions = 1;
            }
            continue;
        }

        if (strstr(line, "{0, 0, 0, 0, 0}")) {
            break;
        }

        char *start = strchr(line, '"');
        if (!start) {
            continue;
        }
        char *end = strchr(start + 1, '"');
        if (!end) {
            continue;
        }

        size_t len = (size_t)(end - start - 1);
        if (len == 0 || len >= sizeof(prev_name)) {
            continue;
        }

        char name[128];
        memcpy(name, start + 1, len);
        name[len] = '\0';

        if (prev_name[0] != '\0' && strcmp(prev_name, name) > 0) {
            printf("Out of order: \"%s\" before \"%s\"\n", prev_name, name);
            failures++;
        }
        snprintf(prev_name, sizeof(prev_name), "%s", name);
    }

    fclose(fp);

    if (!in_functions) {
        printf("Did not find builtin function list\n");
        return 1;
    }

    if (failures) {
        printf("Function order check failed: %d issue(s)\n", failures);
        return 1;
    }

    printf("Function order check PASS\n");
    return 0;
}
