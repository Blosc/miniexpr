/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE)
**********************************************************************/
#undef NDEBUG
#include "miniexpr_reduce.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_dtype(me_dtype dtype) {
    enum { N = 48 * ME_PAIRWISE_SUM_LEAF_ITEMS + 17 };
    size_t width = dtype == ME_FLOAT32 ? sizeof(float) : sizeof(double);
    double dense64[N];
    float dense32[N];
    double leaves64[49];
    float leaves32[49];
    void *dense = dtype == ME_FLOAT32 ? (void *)dense32 : (void *)dense64;
    void *leaves = dtype == ME_FLOAT32 ? (void *)leaves32 : (void *)leaves64;
    size_t capacity = dtype == ME_FLOAT32 ? sizeof(leaves32) : sizeof(leaves64);

    for (size_t i = 0; i < N; i++) {
        if (dtype == ME_FLOAT32) dense32[i] = (float)(i & 15);
        else dense64[i] = (double)(i + 1);
    }

    me_pairwise_sum_state ref_state;
    assert(me_pairwise_sum_init(&ref_state, dtype) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_update(&ref_state, dense, N) == ME_PAIRWISE_SUM_SUCCESS);
    uint64_t ref_out = 0;
    assert(me_pairwise_sum_finish(&ref_state, &ref_out) == ME_PAIRWISE_SUM_SUCCESS);

    double expected = dtype == ME_FLOAT32 ?
        (double)(N / 16) * 120.0 + (double)((N % 16) * ((N % 16) - 1) / 2) :
        (double)N * (double)(N + 1) / 2.0;
    if (dtype == ME_FLOAT32) {
        float got;
        memcpy(&got, &ref_out, sizeof(got));
        assert(got == (float)expected);
    }
    else {
        double got;
        memcpy(&got, &ref_out, sizeof(got));
        assert(got == expected);
    }

    /* Split across aligned prefix leaves + me_pairwise_sum_leaves/append + tail suffix. */
    size_t prefixes[] = {
        0,
        ME_PAIRWISE_SUM_LEAF_ITEMS,
        7 * ME_PAIRWISE_SUM_LEAF_ITEMS,
        17 * ME_PAIRWISE_SUM_LEAF_ITEMS
    };
    for (size_t p = 0; p < sizeof(prefixes) / sizeof(prefixes[0]); p++) {
        size_t prefix = prefixes[p];
        size_t full = ((N - prefix) / ME_PAIRWISE_SUM_LEAF_ITEMS) * ME_PAIRWISE_SUM_LEAF_ITEMS;
        size_t suffix = N - prefix - full;
        me_pairwise_sum_state split_state;
        assert(me_pairwise_sum_init(&split_state, dtype) == ME_PAIRWISE_SUM_SUCCESS);
        if (prefix) {
            assert(me_pairwise_sum_update(&split_state, dense, prefix) == ME_PAIRWISE_SUM_SUCCESS);
        }
        if (full) {
            assert(me_pairwise_sum_leaves(dtype, (const unsigned char *)dense + prefix * width,
                full, leaves, capacity) == ME_PAIRWISE_SUM_SUCCESS);
            assert(me_pairwise_sum_append(&split_state, leaves,
                full / ME_PAIRWISE_SUM_LEAF_ITEMS) == ME_PAIRWISE_SUM_SUCCESS);
        }
        if (suffix) {
            assert(me_pairwise_sum_update(&split_state,
                (const unsigned char *)dense + (prefix + full) * width, suffix) == ME_PAIRWISE_SUM_SUCCESS);
        }
        uint64_t split_out = 0;
        assert(me_pairwise_sum_finish(&split_state, &split_out) == ME_PAIRWISE_SUM_SUCCESS);
        assert(split_out == ref_out);
    }

    /* Zero-element and error contracts. */
    me_pairwise_sum_state empty_state;
    assert(me_pairwise_sum_init(&empty_state, dtype) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_update(&empty_state, NULL, 0) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_append(&empty_state, NULL, 0) == ME_PAIRWISE_SUM_SUCCESS);
    unsigned char zero[sizeof(double) + 2];
    memset(zero, 0xa5, sizeof(zero));
    assert(me_pairwise_sum_finish(&empty_state, zero + 1) == ME_PAIRWISE_SUM_SUCCESS);
    double empty = -1;
    if (dtype == ME_FLOAT32) {
        float value;
        memcpy(&value, zero + 1, sizeof(value));
        empty = value;
    }
    else memcpy(&empty, zero + 1, sizeof(empty));
    assert(empty == 0 && zero[0] == 0xa5 && zero[width + 1] == 0xa5);

    assert(me_pairwise_sum_init(&empty_state, dtype) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_update(&empty_state, dense, 1) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_append(&empty_state, leaves, 1) == ME_PAIRWISE_SUM_ERR_INVALID_ARG);
    assert(me_pairwise_sum_leaves(dtype, dense, 127, leaves, capacity) == ME_PAIRWISE_SUM_ERR_INVALID_ARG);
    assert(me_pairwise_sum_leaves(dtype, dense, 128, leaves, 0) == ME_PAIRWISE_SUM_ERR_INVALID_ARG);
    assert(me_pairwise_sum_append(&empty_state, leaves, SIZE_MAX) == ME_PAIRWISE_SUM_ERR_INVALID_ARG);

    /* Non-exact, cancellation-heavy data: arbitrary raw fragments must preserve
     * the same leaf boundaries and rounding order as one continuous update. */
    for (size_t i = 0; i < N; i++) {
        double value = i % 8 == 0 ? 1e20 : i % 8 == 2 ? -1e20 : (double)(i % 7) / 10;
        if (dtype == ME_FLOAT32) dense32[i] = (float)value;
        else dense64[i] = value;
    }
    me_pairwise_sum_state fragmented;
    assert(me_pairwise_sum_init(&ref_state, dtype) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_init(&fragmented, dtype) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_update(&ref_state, dense, N) == ME_PAIRWISE_SUM_SUCCESS);
    for (size_t offset = 0; offset < N;) {
        size_t count = offset % 193 + 1;
        if (count > N - offset) count = N - offset;
        assert(me_pairwise_sum_update(&fragmented, (const unsigned char *)dense + offset * width,
            count) == ME_PAIRWISE_SUM_SUCCESS);
        offset += count;
    }
    uint64_t fragmented_out = 0;
    ref_out = 0;
    assert(me_pairwise_sum_finish(&ref_state, &ref_out) == ME_PAIRWISE_SUM_SUCCESS);
    assert(me_pairwise_sum_finish(&fragmented, &fragmented_out) == ME_PAIRWISE_SUM_SUCCESS);
    assert(fragmented_out == ref_out);
}

int main(void) {
    assert(me_pairwise_sum_init(NULL, ME_FLOAT64) == ME_PAIRWISE_SUM_ERR_INVALID_ARG);
    me_pairwise_sum_state bad;
    assert(me_pairwise_sum_init(&bad, ME_INT64) == ME_PAIRWISE_SUM_ERR_INVALID_ARG);
    assert(me_pairwise_sum_finish(NULL, &bad) == ME_PAIRWISE_SUM_ERR_INVALID_ARG);
    test_dtype(ME_FLOAT32);
    test_dtype(ME_FLOAT64);
    puts("Core pairwise sum passed");
    return 0;
}
