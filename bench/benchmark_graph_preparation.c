/* CPU-time phase measurements. Run repeated/interleaved subprocesses; these
 * numbers are workload evidence, not a promised speedup or total RSS bound. */
#include "miniexpr_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double elapsed(clock_t start) { return 1e6 * (double)(clock() - start) / CLOCKS_PER_SEC; }
static void check(int rc, const me_graph_error *e) {
    if (rc) { fprintf(stderr, "%s\n", e->native.message); exit(1); }
}
int main(int argc, char **argv) {
    size_t count = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 10000;
    int repeats = argc > 2 ? atoi(argv[2]) : 20;
    bool jit = argc > 3 && !strcmp(argv[3], "jit");
    if (!count || count > 10000000 || repeats < 1 || repeats > 10000) return 2;
    const char *source = "x * 2 + y";
    me_graph_input_metadata inputs[] = {{"x", ME_FLOAT64, 1, {0}}, {"y", ME_FLOAT64, 0, {0}}};
    inputs[0].shape[0] = (int64_t)count;
    me_graph_prepare_options options = {sizeof(options), ME_GRAPH_VERSION, jit ? ME_JIT_ON : ME_JIT_OFF, false, true};
    me_graph_error error; me_graph_plan *p = NULL, *json_plan = NULL; me_graph_schedule *s = NULL;
    clock_t start = clock();
    check(me_graph_prepare_expression(source, strlen(source), inputs, 2, &options, &p, &error), &error);
    double text_us = elapsed(start);
    size_t n; const char *json = me_graph_export_json(p, &n);
    start = clock(); check(me_graph_prepare_json(json, n, &options, &json_plan, &error), &error);
    double json_us = elapsed(start); me_graph_plan_free(json_plan);
    start = clock(); check(me_graph_specialize(p, inputs, 2, NULL, &s, &error), &error);
    double specialize_us = elapsed(start);
    double *x = malloc(count * sizeof(*x)), y = 3, *output = malloc(me_graph_output_bytes(s));
    if (!x || !output) return 2;
    for (size_t i = 0; i < count; i++) x[i] = (double)i;
    me_array_view views[2] = {0};
    for (int i = 0; i < 2; i++) { views[i].name = inputs[i].name; views[i].dtype = inputs[i].dtype; views[i].rank = inputs[i].rank; }
    views[0].base = x; views[0].capacity = count * sizeof(*x); views[0].shape[0] = (int64_t)count; views[0].strides[0] = sizeof(*x);
    views[1].base = &y; views[1].capacity = sizeof(y);
    me_graph_report report;
    start = clock(); check(me_graph_execute(s, views, 2, output, me_graph_output_bytes(s), NULL, &report, &error), &error);
    double first_us = elapsed(start);
    start = clock();
    for (int i = 0; i < repeats; i++) check(me_graph_execute(s, views, 2, output, me_graph_output_bytes(s), NULL, &report, &error), &error);
    double warm_us = elapsed(start) / repeats;
    y = 4; start = clock();
    check(me_graph_execute(s, views, 2, output, me_graph_output_bytes(s), NULL, &report, &error), &error);
    double rebind_us = elapsed(start);
    if (output[0] != 4 || output[count - 1] != 2 * (double)(count - 1) + 4) return 1;
    puts("items,repeats,jit,text_cpu_us,json_cpu_us,specialize_cpu_us,first_cpu_us,warm_cpu_us,rebind_cpu_us,graph_metadata_bytes,schedule_bytes,iterator_bound_bytes,iterator_peak_bytes,output_bytes");
    printf("%zu,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%zu,%zu,%zu,%zu,%zu\n", count, repeats,
        report.has_jit, text_us, json_us, specialize_us, first_us, warm_us, rebind_us,
        me_graph_plan_bytes(p), me_graph_schedule_bytes(s), me_graph_scratch_bytes(s), report.array.temporary_bytes, me_graph_output_bytes(s));
    free(x); free(output); me_graph_schedule_free(s); me_graph_plan_free(p); return 0;
}
