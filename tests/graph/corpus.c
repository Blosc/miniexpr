/* Corpus-derived graph inference, retaining native-owned case IDs. This checks
 * single-return, constant-free cases; DSL control flow/artifact captures retain
 * their existing independent suites, not a falsely broadened graph claim. */
#include "miniexpr_graph.h"
#include "yyjson.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static me_dtype dtype(const char *name) {
    const char *names[] = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"};
    const me_dtype types[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    if (name) for (int i = 0; i < 11; i++) if (!strcmp(name, names[i])) return types[i];
    return ME_AUTO;
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 2;
    fseek(file, 0, SEEK_END); long length = ftell(file); rewind(file);
    if (length < 0 || length > 64 * 1024 * 1024) return 2;
    char *json = malloc((size_t)length);
    if (!json || fread(json, 1, (size_t)length, file) != (size_t)length) return 2;
    fclose(file);
    yyjson_doc *doc = yyjson_read(json, (size_t)length, 0); free(json);
    if (!doc) return 2;
    yyjson_val *cases = yyjson_obj_get(yyjson_doc_get_root(doc), "cases"), *value;
    size_t index, max; int checked = 0, skipped = 0, failed = 0;
    yyjson_arr_foreach(cases, index, max, value) {
        const char *encoded = yyjson_get_str(yyjson_obj_get(value, "artifact"));
        if (!encoded) { skipped++; continue; }
        yyjson_doc *artifact = yyjson_read(encoded, strlen(encoded), 0);
        yyjson_val *root = artifact ? yyjson_doc_get_root(artifact) : NULL;
        const char *source = yyjson_get_str(yyjson_obj_get(root, "source"));
        const char *expected = yyjson_get_str(yyjson_obj_get(value, "inferred_dtype"));
        const char *expression = source ? strstr(source, "\n    return ") : NULL;
        if (!expression || yyjson_arr_size(yyjson_obj_get(root, "constants")) || !expected || !strcmp(expected, "unsupported")) {
            skipped++; yyjson_doc_free(artifact); continue;
        }
        expression += strlen("\n    return "); size_t n = strcspn(expression, "\n");
        if (expression[n] && expression[n + 1]) { skipped++; yyjson_doc_free(artifact); continue; }
        me_graph_input_metadata inputs[ME_MAX_VARS] = {0};
        yyjson_val *bindings = yyjson_obj_get(root, "inputs");
        int declared = (int)yyjson_arr_size(bindings), count = 0;
        if (declared > ME_MAX_VARS) return 2;
        for (int i = 0; i < declared; i++) {
            yyjson_val *binding = yyjson_arr_get(bindings, (size_t)i);
            const char *name = yyjson_get_str(yyjson_obj_get(binding, "name"));
            bool used = false;
            for (const char *at = expression; (at = strstr(at, name)) != NULL; at++) {
                char before = at == expression ? 0 : at[-1], after = at[strlen(name)];
                if (!isalnum((unsigned char)before) && before != '_' && !isalnum((unsigned char)after) && after != '_') used = true;
            }
            /* Graphs deliberately reject unreachable input nodes. The corpus
             * has unused x signatures for native zero-arity constants e/pi. */
            if (!used) continue;
            inputs[count].name = name;
            inputs[count++].dtype = dtype(yyjson_get_str(yyjson_obj_get(binding, "dtype")));
        }
        me_graph_plan *plan = NULL; me_graph_error error;
        int rc = me_graph_prepare_expression(expression, n, inputs, count, NULL, &plan, &error);
        checked++;
        if (rc || me_graph_inferred_dtype(plan) != dtype(expected)) {
            fprintf(stderr, "%s: inference mismatch: %s (rc=%d)\n",
                yyjson_get_str(yyjson_obj_get(value, "id")), error.native.message, rc);
            failed++;
        }
        me_graph_plan_free(plan); yyjson_doc_free(artifact);
    }
    yyjson_doc_free(doc);
    printf("graph corpus inference: checked=%d skipped=%d failed=%d\n", checked, skipped, failed);
    return failed || !checked;
}
