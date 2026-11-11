/* Standalone native/WASM deployment example. No Python or NumPy linkage. */
#include "miniexpr_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: graph_host graph.json\n"); return 2; }
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 2;
    fseek(file, 0, SEEK_END); long length = ftell(file); rewind(file);
    if (length <= 0 || length > ME_GRAPH_MAX_BYTES) { fclose(file); return 2; }
    char *json = malloc((size_t)length);
    if (!json || fread(json, 1, (size_t)length, file) != (size_t)length) { free(json); fclose(file); return 2; }
    fclose(file);
    me_graph_plan *plan = NULL; me_graph_schedule *schedule = NULL; me_graph_error error;
    int rc = me_graph_prepare_json(json, (size_t)length, NULL, &plan, &error); free(json);
    if (rc) { fprintf(stderr, "%s\n", error.native.message); return 1; }
    me_graph_input_metadata metadata[3] = {0}; me_array_view bindings[3] = {0};
    float x[] = {0, 2}, y[] = {3, 4, 5}; bool mask[] = {true, false, true};
    int count = me_graph_ninputs(plan);
    if (count > 3) { me_graph_plan_free(plan); return 2; }
    for (int i = 0; i < count; i++) {
        const char *name = me_graph_input_name(plan, i);
        metadata[i].name = name; metadata[i].dtype = me_graph_input_dtype(plan, i);
        bindings[i].name = name; bindings[i].dtype = metadata[i].dtype;
        if (!strcmp(name, "x") && metadata[i].dtype == ME_FLOAT32) {
            metadata[i].rank = 2; metadata[i].shape[0] = 2; metadata[i].shape[1] = 1;
            bindings[i].base = x; bindings[i].capacity = sizeof(x); bindings[i].strides[0] = 4; bindings[i].strides[1] = 4;
        } else if (!strcmp(name, "y") && metadata[i].dtype == ME_FLOAT32) {
            metadata[i].rank = 1; metadata[i].shape[0] = 3;
            bindings[i].base = y; bindings[i].capacity = sizeof(y); bindings[i].strides[0] = 4;
        } else if (!strcmp(name, "mask") && metadata[i].dtype == ME_BOOL) {
            metadata[i].rank = 1; metadata[i].shape[0] = 3;
            bindings[i].base = mask; bindings[i].capacity = sizeof(mask); bindings[i].strides[0] = 1;
        } else { me_graph_plan_free(plan); return 2; }
        bindings[i].rank = metadata[i].rank;
        memcpy(bindings[i].shape, metadata[i].shape, sizeof(bindings[i].shape));
    }
    /* Metadata is sufficient to query output before touching a numerical buffer. */
    rc = me_graph_specialize(plan, metadata, count, NULL, &schedule, &error);
    size_t bytes = me_graph_output_bytes(schedule);
    void *output = rc ? NULL : malloc(bytes ? bytes : 1);
    if (!rc && !output) rc = ME_GRAPH_ERR_OOM;
    me_graph_report report = {0};
    if (!rc) rc = me_graph_execute(schedule, bindings, count, output, bytes, NULL, &report, &error);
    if (rc) fprintf(stderr, "%s\n", error.native.message);
    else {
        printf("bindings=%d rank=%d output_bytes=%zu stages=%zu jit=%d fp=%u\n", count,
            me_graph_output_rank(schedule), bytes, report.stages, report.has_jit, report.array.fp_flags);
        if (me_graph_output_dtype(schedule) == ME_FLOAT32) {
            for (size_t i = 0; i < bytes / sizeof(float); i++) printf("%s%.7g", i ? " " : "", (double)((float *)output)[i]);
            puts("");
        }
    }
    free(output); me_graph_schedule_free(schedule); me_graph_plan_free(plan); return rc != 0;
}
