#include "miniexpr_reduce.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <fenv.h>
#include <stdio.h>

#define CHECK_TYPE(TYPE, DTYPE) do { \
    TYPE values[] = {3, 1, 1, 4, 4}; \
    for (int minimum = 0; minimum <= 1; minimum++) { \
        uint64_t whole, split_value; \
        assert(me_extrema_span(DTYPE, values, 5, 0, false, minimum, &whole)); \
        assert(me_extrema_span(DTYPE, values, 2, 0, false, minimum, &split_value)); \
        assert(me_extrema_span(DTYPE, values + 2, 3, split_value, true, minimum, &split_value)); \
        assert(whole == split_value); \
        me_argextrema_state serial = {0}, split = {0}, tail = {0}; \
        assert(me_argextrema_update(DTYPE, values, 5, 100, minimum, &serial)); \
        assert(serial.index == (minimum ? 101 : 103)); \
        assert(me_argextrema_update(DTYPE, values + 2, 3, 102, minimum, &tail)); \
        /* Deliberately merge in reverse position order. */ \
        assert(me_argextrema_merge(DTYPE, &split, tail.value, tail.index, minimum)); \
        assert(me_argextrema_update(DTYPE, values, 2, 100, minimum, &split)); \
        assert(split.index == serial.index && split.value == serial.value); \
    } \
} while (0)

static void test_integers(void) {
    CHECK_TYPE(int8_t, ME_INT8);
    CHECK_TYPE(int16_t, ME_INT16);
    CHECK_TYPE(int32_t, ME_INT32);
    CHECK_TYPE(int64_t, ME_INT64);
    CHECK_TYPE(uint8_t, ME_UINT8);
    CHECK_TYPE(uint16_t, ME_UINT16);
    CHECK_TYPE(uint32_t, ME_UINT32);
    CHECK_TYPE(uint64_t, ME_UINT64);
    CHECK_TYPE(float, ME_FLOAT32);
    CHECK_TYPE(double, ME_FLOAT64);
    uint64_t large[] = {UINT64_MAX - 1, UINT64_MAX, UINT64_MAX};
    me_argextrema_state state = {0};
    assert(me_argextrema_update(ME_UINT64, large, 3, 0, false, &state));
    assert(state.index == 1 && state.value == UINT64_MAX);
    int64_t signed_values[] = {INT64_MIN, INT64_MAX, INT64_MIN};
    state.initialized = false;
    assert(me_argextrema_update(ME_INT64, signed_values, 3, 0, true, &state));
    assert(state.index == 0);
    bool truth[] = {false, true, true};
    state.initialized = false;
    assert(me_argextrema_update(ME_BOOL, truth, 3, 0, false, &state));
    assert(state.index == 1);
    uint8_t noncanonical[] = {2, 1, 3};
    state.initialized = false;
    assert(me_argextrema_update(ME_BOOL, noncanonical, 3, 0, true, &state));
    assert(state.index == 0);
}

static void test_floating(me_dtype dtype) {
    for (int minimum = 0; minimum <= 1; minimum++) {
        uint64_t result = 0;
        float f[] = {-0.0f, 0.0f, -0.0f};
        double d[] = {-0.0, 0.0, -0.0};
        const void *values = dtype == ME_FLOAT32 ? (void *)f : (void *)d;
        assert(me_extrema_span(dtype, values, 3, 0, false, minimum, &result));
        if (dtype == ME_FLOAT32) {
            float v; memcpy(&v, &result, 4);
            assert(!!signbit(v) == !!minimum);
        } else {
            double v; memcpy(&v, &result, 8);
            assert(!!signbit(v) == !!minimum);
        }
        me_argextrema_state state = {0};
        assert(me_argextrema_update(dtype, values, 3, 0, minimum, &state));
        assert(state.index == 0);  /* Zero sign is NOT an arg-extrema tie-breaker. */
        if (dtype == ME_FLOAT32) {
            uint32_t bits[] = {0x3f800000u, 0xffc00011u, 0x7fc00022u};
            memcpy(f, bits, sizeof(f));
        } else {
            uint64_t bits[] = {UINT64_C(0x3ff0000000000000),
                              UINT64_C(0xfff8000000000011), UINT64_C(0x7ff8000000000022)};
            memcpy(d, bits, sizeof(d));
        }
        state.initialized = false;
        assert(me_argextrema_update(dtype, values, 3, 0, minimum, &state));
        assert(state.index == 1);
        assert(me_extrema_span(dtype, values, 3, 0, false, minimum, &result));
        assert(result == state.value);
        /* Unaligned input and nonzero initial state use the same rules. */
        unsigned char unaligned[25] = {0};
        size_t width = me_extrema_itemsize(dtype);
        memcpy(unaligned + 1, values, 3 * width);
        uint64_t again = 0;
        assert(me_extrema_float(dtype, unaligned + 1, 3, result, true, minimum, &again));
        assert(again == result);
    }
}

static void test_signaling_nan(void) {
    uint64_t raw[] = {UINT64_C(0x3ff0000000000000),
                      UINT64_C(0xfff0000000000011), UINT64_C(0x7ff0000000000022)};
    uint64_t out = 0;
    assert(me_extrema_float(ME_FLOAT64, raw, 3, 0, false, true, &out));
    assert(out == UINT64_C(0xfff8000000000011));
    uint32_t raw32[] = {0x3f800000u, 0xff800011u, 0x7f800022u};
    assert(me_extrema_float(ME_FLOAT32, raw32, 3, 0, false, false, &out));
    assert((uint32_t)out == 0xffc00011u);
    me_argextrema_state state = {0};
    assert(me_argextrema_update(ME_FLOAT64, raw, 3, 0, true, &state));
    assert(state.index == 1);
    /* A NaN in the earlier partition wins regardless of completion order. */
    me_argextrema_state merged = {0};
    assert(me_argextrema_update(ME_FLOAT64, raw + 2, 1, 2, true, &merged));
    assert(me_argextrema_merge(ME_FLOAT64, &merged, state.value, state.index, true));
    assert(merged.index == 1);
}

int main(void) {
    test_integers();
    test_floating(ME_FLOAT32);
    test_floating(ME_FLOAT64);
    test_signaling_nan();
    me_argextrema_state state = {0};
    assert(!me_argextrema_update(ME_AUTO, NULL, 1, 0, true, &state));
    assert(!me_argextrema_merge(ME_AUTO, &state, 0, 0, true));
    assert(!me_argextrema_update(ME_FLOAT64, NULL, 1, INT64_MAX, true, &state));
    puts("shared extrema and arg-extrema: passed");
    return 0;
}
