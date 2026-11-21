/* Opt-in floating summation: accuracy, bounded readers and tile invariance. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "miniexpr_graph.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    me_dtype dtype;
    size_t calls, maximum;
} ramp_reader;

static int read_ramp(void *owner, const size_t *indices, size_t count,
    void *destination, size_t capacity) {
    ramp_reader *r = owner;
    size_t width = r->dtype == ME_FLOAT32 ? sizeof(float) : sizeof(double);
    assert(capacity >= count * width);
    r->calls++;
    if (count > r->maximum) r->maximum = count;
    for (size_t i = 0; i < count; i++) {
        if (r->dtype == ME_FLOAT32) ((float *)destination)[i] = (float)indices[i];
        else ((double *)destination)[i] = (double)indices[i];
    }
    return 0;
}

static void ramp(size_t count, me_dtype dtype, bool large) {
    me_graph_input_metadata metadata = {"x", dtype, 1, {(int64_t)count}};
    me_graph_plan *plan = NULL;
    me_graph_error error;
    const char *expression = "sum(x)";
    assert(!me_graph_prepare_expression(expression, strlen(expression),
        &metadata, 1, NULL, &plan, &error));
    ramp_reader reader = {dtype, 0, 0};
    me_array_source source = {0};
    source.view.name = "x";
    source.view.dtype = dtype;
    source.view.rank = 1;
    source.view.shape[0] = (int64_t)count;
    source.read = read_ramp;
    source.owner = &reader;
    uint64_t baseline = 0;
    /* Odd tiles split both the eight accumulators and the 128-lane leaves. */
    size_t tiles[] = {1, 7, 127, 129, 1024, 64000};
    for (size_t t = large ? 4 : 0; t < sizeof(tiles)/sizeof(*tiles); t++) {
        me_graph_specialize_options options = {0};
        options.struct_size = sizeof(options);
        options.version = ME_GRAPH_VERSION;
        options.tile_items = tiles[t];
        options.pairwise_sum = true;
        me_graph_schedule *schedule = NULL;
        assert(!me_graph_specialize(plan, &metadata, 1, &options, &schedule, &error));
        uint64_t output = 0;
        me_graph_report report;
        assert(!me_graph_execute_sources(schedule, &source, 1, &output,
            sizeof(output), NULL, &report, &error));
        double exact = (double)count * (double)(count - (count != 0)) / 2;
        if (dtype == ME_FLOAT64) {
            double value; memcpy(&value, &output, sizeof(value));
            assert(value == exact);
            if (large) printf("ramp %zu tile %zu: %.17g (exact)\n", count, tiles[t], value);
        } else {
            float value; memcpy(&value, &output, sizeof(value));
            assert(value == (float)exact);
        }
        if (t != (large ? 4u : 0u)) assert(output == baseline);
        baseline = output;
        assert(reader.maximum <= tiles[t]);
        assert(report.array.temporary_bytes <= me_graph_scratch_bytes(schedule));
        reader.maximum = 0;
        me_graph_schedule_free(schedule);
    }
    me_graph_plan_free(plan);
}

static void contracts(void) {
    me_graph_input_metadata metadata = {"x", ME_FLOAT64, 1, {1031}};
    me_graph_plan *plan = NULL;
    me_graph_error error;
    assert(!me_graph_prepare_expression("sum(x)", 6, &metadata, 1, NULL, &plan, &error));
    double values[1031];
    for (int i = 0; i < 1031; i++) values[i] = i % 3 == 0 ? 1e16 : i % 3 == 1 ? 1 : -1e16;
    me_array_view input = {0};
    input.name = "x"; input.dtype = ME_FLOAT64;
    input.base = values; input.capacity = sizeof(values);
    input.rank = 1; input.shape[0] = 1031; input.strides[0] = sizeof(double);
    double reference = 0;
    size_t tiles[] = {1, 7, 127, 129, 1024, 64000};
    for (int pairwise = 0; pairwise <= 1; pairwise++) {
        double first = 0;
        for (size_t t = 0; t < sizeof(tiles)/sizeof(*tiles); t++) {
            me_graph_specialize_options options = {0};
            options.struct_size = sizeof(options); options.version = ME_GRAPH_VERSION;
            options.tile_items = tiles[t]; options.pairwise_sum = pairwise != 0;
            me_graph_schedule *schedule = NULL;
            assert(!me_graph_specialize(plan, &metadata, 1, &options, &schedule, &error));
            double output;
            assert(!me_graph_execute(schedule, &input, 1, &output, sizeof(output), NULL, NULL, &error));
            if (!t) first = output;
            else assert(!memcmp(&first, &output, sizeof(output)));
            me_graph_schedule_free(schedule);
        }
        if (!pairwise) {
            for (size_t i = 0; i < 1031; i++) reference += values[i];
            assert(first == reference);
        }
    }
    /* Source-compatible old-size specialization options still select serial. */
    me_graph_specialize_options old = {0};
    old.struct_size = offsetof(me_graph_specialize_options, pairwise_sum);
    old.version = ME_GRAPH_VERSION;
    me_graph_schedule *schedule = NULL;
    assert(!me_graph_specialize(plan, &metadata, 1, &old, &schedule, &error));
    me_graph_schedule_free(schedule);
    me_graph_plan_free(plan);
    plan = NULL;
    assert(!me_graph_prepare_expression("prod(x)", 7, &metadata, 1, NULL, &plan, &error));
    old.struct_size = sizeof(old);
    old.pairwise_sum = true;
    schedule = NULL;
    assert(me_graph_specialize(plan, &metadata, 1, &old, &schedule, &error) == ME_GRAPH_ERR_UNSUPPORTED);
    assert(!schedule);
    me_graph_plan_free(plan);
}

static void array_contracts(void) {
    me_graph_input_metadata metadata = {"x", ME_FLOAT64, 1, {771}};
    me_graph_plan *plan = NULL;
    me_graph_error error;
    assert(!me_graph_prepare_expression("sum(x)", 6, &metadata, 1, NULL, &plan, &error));
    const me_artifact *map = me_graph_map_artifact(plan);
    assert(map);
    double values[771];
    bool mask[771];
    for (size_t i = 0; i < 771; i++) {
        values[i] = (double)i;
        mask[i] = i % 3 != 0;
    }
    me_array_view input = {0}, participation = {0};
    input.name = me_artifact_input_name(map, 0); input.dtype = ME_FLOAT64;
    input.base = values; input.capacity = sizeof(values);
    input.rank = 2; input.shape[0] = 3; input.shape[1] = 257;
    /* Reverse each row: the gathered path must use the same logical leaves. */
    input.strides[0] = 257 * sizeof(double); input.strides[1] = -(int64_t)sizeof(double);
    input.offset = 256 * sizeof(double);
    participation = input;
    participation.dtype = ME_BOOL; participation.base = mask;
    participation.capacity = sizeof(mask); participation.offset = 256;
    participation.strides[0] = 257; participation.strides[1] = -1;
    double initial = 3.5, first[3];
    size_t tiles[] = {1, 7, 129, 64000};
    for (size_t t = 0; t < sizeof(tiles)/sizeof(*tiles); t++) {
        me_array_options options = {0};
        options.version = ME_ARTIFACT_ARRAY_VERSION;
        options.reduction = ME_ARRAY_SUM_PAIRWISE;
        options.naxes = 1; options.axes[0] = -1; options.keepdims = true;
        options.tile_items = tiles[t]; options.initial = &initial; options.where = &participation;
        double output[3];
        me_artifact_error native;
        me_array_report report;
        assert(!me_artifact_eval_array(map, &input, 1, 2, input.shape, &options,
            output, sizeof(output), &report, &native));
        for (size_t row = 0; row < 3; row++) {
            double expected = initial;
            for (size_t col = 0; col < 257; col++) {
                size_t i = row * 257 + col;
                if (mask[i]) expected += values[i];
            }
            assert(output[row] == expected);
        }
        if (!t) memcpy(first, output, sizeof(first));
        else assert(!memcmp(first, output, sizeof(first)));
    }
    /* Nonfinite classifications and invalid flags come from the new arithmetic,
     * while masked NaNs must not participate. No NaN payload parity is promised. */
    input.rank = 1; input.shape[0] = 771; input.offset = 0;
    input.strides[0] = sizeof(double);
    participation.rank = 1; participation.shape[0] = 771;
    participation.offset = 0; participation.strides[0] = 1;
    me_array_options options = {0};
    options.version = ME_ARTIFACT_ARRAY_VERSION;
    options.reduction = ME_ARRAY_SUM_PAIRWISE; options.naxes = -1;
    values[0] = NAN;
    options.where = &participation;
    double output;
    assert(!me_artifact_eval_array(map, &input, 1, 1, input.shape, &options,
        &output, sizeof(output), NULL, NULL));
    assert(isfinite(output));
    options.where = NULL;
    assert(!me_artifact_eval_array(map, &input, 1, 1, input.shape, &options,
        &output, sizeof(output), NULL, NULL));
    assert(isnan(output));
    values[0] = INFINITY; values[1] = -INFINITY;
    me_array_report report;
    assert(!me_artifact_eval_array(map, &input, 1, 1, input.shape, &options,
        &output, sizeof(output), &report, NULL));
    assert(isnan(output));
    if (report.fp_supported) assert(report.fp_flags & ME_FP_INVALID);
    /* Unsupported explicit integer accumulation fails before execution. */
    options.accumulator = ME_INT64;
    assert(me_artifact_eval_array(map, &input, 1, 1, input.shape, &options,
        &output, sizeof(output), NULL, NULL) == ME_ARTIFACT_ERR_UNSUPPORTED);
    options.accumulator = ME_AUTO;
    options.initial = &initial;
    input.shape[0] = 0;
    assert(!me_artifact_eval_array(map, &input, 1, 1, input.shape, &options,
        &output, sizeof(output), NULL, NULL));
    assert(output == initial);
    me_graph_plan_free(plan);
}

static void resident_timing(void) {
    const size_t count = 16000000;
    double *values = malloc(count * sizeof(*values));
    assert(values);
    for (size_t i = 0; i < count; i++) values[i] = 1000000000.0 + (double)i;
    me_graph_input_metadata metadata = {"x", ME_FLOAT64, 1, {(int64_t)count}};
    me_graph_plan *plan = NULL;
    me_graph_error error;
    assert(!me_graph_prepare_expression("sum(x)", 6, &metadata, 1, NULL, &plan, &error));
    me_array_view input = {0};
    input.name = "x"; input.dtype = ME_FLOAT64; input.base = values;
    input.capacity = count * sizeof(*values); input.rank = 1;
    input.shape[0] = (int64_t)count; input.strides[0] = sizeof(*values);
    double exact = (double)count * (2000000000.0 + (double)count - 1) / 2;
    for (int pairwise = 0; pairwise <= 1; pairwise++) {
        me_graph_specialize_options options = {0};
        options.struct_size = sizeof(options); options.version = ME_GRAPH_VERSION;
        options.tile_items = 64000; options.pairwise_sum = pairwise != 0;
        me_graph_schedule *schedule = NULL;
        assert(!me_graph_specialize(plan, &metadata, 1, &options, &schedule, &error));
        double output = 0;
        clock_t started = clock();
        for (int repeat = 0; repeat < 5; repeat++)
            assert(!me_graph_execute(schedule, &input, 1, &output, sizeof(output), NULL, NULL, &error));
        printf("%s: %.3f ms, error %.17g\n", pairwise ? "pairwise" : "serial",
            1000.0 * (double)(clock() - started) / CLOCKS_PER_SEC / 5, output - exact);
        if (pairwise) assert(output == exact);
        me_graph_schedule_free(schedule);
    }
    me_graph_plan_free(plan);
    free(values);
}

static void streaming_interface(void) {
    for (int f32 = 0; f32 < 2; f32++) {
        me_dtype dtype = f32 ? ME_FLOAT32 : ME_FLOAT64;
        size_t width = f32 ? sizeof(float) : sizeof(double);
        union { double f64[1031]; float f32[1031]; } values;
        for (size_t i = 0; i < 1031; i++) {
            if (f32) values.f32[i] = i % 3 == 0 ? 1e8f : i % 3 == 1 ? 1.0f : -1e8f;
            else values.f64[i] = i % 3 == 0 ? 1e16 : i % 3 == 1 ? 1.0 : -1e16;
        }
        me_pairwise_sum_state state;
        assert(!me_pairwise_sum_init(&state, dtype));
        assert(!me_pairwise_sum_update(&state, &values, 1031));
        uint64_t baseline = 0;
        assert(!me_pairwise_sum_finish(&state, &baseline));
        size_t blocks[] = {1, 5, 100, 128, 129, 256, 1000};
        for (size_t b = 0; b < sizeof(blocks)/sizeof(*blocks); b++) {
            assert(!me_pairwise_sum_init(&state, dtype));
            for (size_t start = 0; start < 1031;) {
                size_t count = blocks[b] < 1031 - start ? blocks[b] : 1031 - start;
                size_t prefix = (128 - start % 128) % 128;
                if (prefix > count) prefix = count;
                size_t full = (count - prefix) / 128 * 128;
                size_t suffix = count - prefix - full;
                unsigned char *data = (unsigned char *)&values + start * width;
                if (prefix) assert(!me_pairwise_sum_update(&state, data, prefix));
                if (full) {
                    double leaves[8];
                    assert(!me_pairwise_sum_leaves(dtype, data + prefix * width,
                        full, leaves, sizeof(leaves)));
                    assert(!me_pairwise_sum_append(&state, leaves, full / 128));
                }
                if (suffix) assert(!me_pairwise_sum_update(&state, data + (prefix + full) * width, suffix));
                start += count;
            }
            uint64_t output = 0;
            assert(!me_pairwise_sum_finish(&state, &output));
            assert(output == baseline);
        }
        assert(!me_pairwise_sum_init(&state, dtype));
        assert(!me_pairwise_sum_update(&state, &values, 1));
        assert(me_pairwise_sum_append(&state, &values, 1) == ME_ARTIFACT_ERR_BINDING);
        assert(me_pairwise_sum_leaves(dtype, &values, 127, &values, sizeof(values)) == ME_ARTIFACT_ERR_BINDING);
        assert(me_pairwise_sum_leaves(dtype, &values, 128, &values, 0) == ME_ARTIFACT_ERR_BINDING);
        assert(!me_pairwise_sum_init(&state, dtype));
        assert(me_pairwise_sum_append(&state, &values, SIZE_MAX) == ME_ARTIFACT_ERR_BINDING);
        /* Exercise all 1/8/16-leaf append alignments and tail lengths against serial update. */
        double dense_f64[48 * 128];
        float dense_f32[48 * 128];
        for (size_t i = 0; i < 48 * 128; i++) {
            double v = ((int)(i % 19) - 9) * 1.125 + (i % 5 == 0 ? 1e6 : i % 5 == 1 ? -1e6 : 0.00390625);
            dense_f64[i] = v;
            dense_f32[i] = (float)v;
        }
        const void *dense = f32 ? (const void *)dense_f32 : (const void *)dense_f64;
        double all_leaves[48];
        assert(!me_pairwise_sum_leaves(dtype, dense, 48 * 128, all_leaves, sizeof(all_leaves)));
        for (size_t first_n = 0; first_n <= 19; first_n++) {
            for (size_t second_n = 0; second_n <= 48 - first_n; second_n++) {
                size_t total_leaves = first_n + second_n;
                me_pairwise_sum_state ref_state, fast_state;
                uint64_t ref_out = 0, fast_out = 0;
                assert(!me_pairwise_sum_init(&ref_state, dtype));
                assert(!me_pairwise_sum_update(&ref_state, dense, total_leaves * 128));
                assert(!me_pairwise_sum_finish(&ref_state, &ref_out));
                assert(!me_pairwise_sum_init(&fast_state, dtype));
                if (first_n) assert(!me_pairwise_sum_append(&fast_state, all_leaves, first_n));
                if (second_n) assert(!me_pairwise_sum_append(&fast_state,
                    (const unsigned char *)all_leaves + first_n * width, second_n));
                assert(!me_pairwise_sum_finish(&fast_state, &fast_out));
                assert(fast_out == ref_out);
            }
        }
    }
}

int main(int argc, char **argv) {
    (void)argv;
    contracts();
    array_contracts();
    streaming_interface();
    ramp(0, ME_FLOAT64, false);
    ramp(1031, ME_FLOAT64, false);
    ramp(1031, ME_FLOAT32, false);
    ramp(1000000, ME_FLOAT32, true);
    if (argc > 1) {
        resident_timing();
        ramp(100000000, ME_FLOAT64, true);
        ramp(1000000000, ME_FLOAT64, true);
    }
    puts("pairwise summation passed");
    return 0;
}
