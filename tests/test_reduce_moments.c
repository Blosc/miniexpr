#include "miniexpr_reduce.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>

static void test_partition(me_dtype dtype) {
    double values[259];
    float narrow[259];
    double origin = dtype == ME_FLOAT32 ? 1000000 : 1e12;
    long double sum = 0, m2 = 0;
    for (size_t i = 0; i < 259; i++) {
        values[i] = origin + (double)(i % 17) * 0.25;
        narrow[i] = (float)values[i];
        sum += values[i] - origin;
    }
    long double mean = sum / 259;
    for (size_t i = 0; i < 259; i++) {
        long double delta = (values[i] - origin) - mean;
        m2 += delta * delta;
    }
    size_t width = dtype == ME_FLOAT32 ? sizeof(float) : sizeof(double);
    const void *input = dtype == ME_FLOAT32 ? (void *)narrow : (void *)values;
    unsigned char unaligned[sizeof(values) + 1];
    memcpy(unaligned + 1, input, 259 * width);
    for (size_t step = 1; step <= 259; step += 7) {
        double state[4] = {0}, result;
        for (size_t i = 0; i < 259; i += step) {
            double part[4];
            size_t n = 259 - i < step ? 259 - i : step;
            assert(me_moments_span(dtype, unaligned + 1 + i * width, n, part));
            assert(me_moments_merge(state, part));
        }
        assert(state[0] == 259);
        assert(fabsl((state[1] - origin) + state[2] - mean) < 1e-12L);
        for (int ddof = -1; ddof <= 2; ddof++) {
            double expected = (double)(m2 / (259 - ddof));
            assert(me_moments_finish(state, ddof, false, &result));
            assert(fabs(result - expected) < 1e-8);
            assert(me_moments_finish(state, ddof, true, &result));
            assert(fabs(result - sqrt(expected)) < 1e-8);
        }
        assert(!me_moments_finish(state, 259, false, &result));
        assert(!me_moments_finish(state, NAN, false, &result));
        double reverse[4] = {0};
        for (size_t end = 259; end > 0;) {
            size_t begin = end > step ? end - step : 0;
            double part[4];
            assert(me_moments_span(dtype, unaligned + 1 + begin * width, end - begin, part));
            assert(me_moments_merge(reverse, part));
            end = begin;
        }
        assert(me_moments_finish(reverse, 0, false, &result));
        assert(fabs(result - (double)(m2 / 259)) < 1e-8);
    }
}

static void test_special(void) {
    double part[4], state[4] = {0}, result;
    assert(me_moments_span(ME_FLOAT64, NULL, 0, part));
    assert(me_moments_merge(state, part));
    assert(!me_moments_finish(state, 0, false, &result));
    assert(!me_moments_span(ME_INT32, NULL, 0, part));
    double constant[] = {1e150, 1e150, 1e150};
    assert(me_moments_span(ME_FLOAT64, constant, 3, part));
    assert(me_moments_finish(part, 0, false, &result) && result == 0);
    double zeros[] = {-0.0, 0.0, -0.0};
    assert(me_moments_span(ME_FLOAT64, zeros, 3, part));
    assert(me_moments_finish(part, 0, false, &result) && result == 0 && !signbit(result));
    double nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < 3; i++) {
        double values[] = {1, nonfinite[i], 2};
        assert(me_moments_span(ME_FLOAT64, values, 3, part));
        assert(me_moments_finish(part, 0, false, &result) && isnan(result));
    }
    double too_many[] = {9007199254740992.0, 1, 0, 0};
    double one[] = {1, 1, 0, 0};
    assert(!me_moments_merge(too_many, one));
    double extreme[] = {-1e308, 1e308};
    assert(me_moments_span(ME_FLOAT64, extreme, 2, part));
    assert(part[1] + part[2] == 0);
    assert(me_moments_finish(part, 0, false, &result) && isinf(result));
    memset(state, 0, sizeof(state));
    for (size_t i = 0; i < 2; i++) {
        assert(me_moments_span(ME_FLOAT64, extreme + i, 1, part));
        assert(me_moments_merge(state, part));
    }
    assert(state[1] + state[2] == 0);
    assert(me_moments_finish(state, 0, false, &result) && isinf(result));
}

int main(void) {
    test_partition(ME_FLOAT32);
    test_partition(ME_FLOAT64);
    test_special();
    puts("moments primitives passed");
    return 0;
}
