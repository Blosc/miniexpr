/* Shared direct-input reductions. Callers validate bindings and establish the strict
 * floating environment. Preserve serial order and rounding at each arithmetic step;
 * do not use tile partial sums or reassociation. memcpy permits unaligned DSL
 * buffers and is optimized to scalar loads by the host compiler. */
#ifndef MINIEXPR_DSL_PORTABLE_REDUCE_H
#define MINIEXPR_DSL_PORTABLE_REDUCE_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "miniexpr.h"

static inline float dsl_portable_sum_f32(float sum, const void *values, size_t count) {
    const unsigned char *data = values;
    for (size_t i = 0; i < count; i++) {
        float value; memcpy(&value, data + i * sizeof(value), sizeof(value));
        sum += value;
    }
    return sum;
}
static inline double dsl_portable_sum_f64(double sum, const void *values, size_t count) {
    const unsigned char *data = values;
    for (size_t i = 0; i < count; i++) {
        double value; memcpy(&value, data + i * sizeof(value), sizeof(value));
        sum += value;
    }
    return sum;
}
/* Integer sums accumulate in 64 bits. checked preserves the DSL reducer's
 * range errors; graph reductions use modulo arithmetic. Signed values are
 * sign-extended then added as unsigned bits, never overflowing signed C. */
static inline bool dsl_portable_sum_integer(me_dtype source, const void *values,
    size_t count, uint64_t initial, bool checked, uint64_t *out) {
    const unsigned char *data = values;
    uint64_t sum = initial;
#define SUM_SIGNED(type, expression) do { \
    for (size_t i = 0; i < count; i++) { \
        type raw; memcpy(&raw, data + i * sizeof(raw), sizeof(raw)); \
        int64_t value = (expression); \
        if (checked) { \
            int64_t current; memcpy(&current, &sum, sizeof(current)); \
            if ((value > 0 && current > INT64_MAX - value) || \
                (value < 0 && current < INT64_MIN - value)) return false; \
        } \
        sum += (uint64_t)value; \
    } \
} while (0)
#define SUM_UNSIGNED(type) do { \
    for (size_t i = 0; i < count; i++) { \
        type raw; memcpy(&raw, data + i * sizeof(raw), sizeof(raw)); \
        uint64_t value = raw; \
        if (checked && sum > UINT64_MAX - value) return false; \
        sum += value; \
    } \
} while (0)
    switch (source) {
    case ME_BOOL: SUM_SIGNED(bool, raw ? 1 : 0); break;
    case ME_INT8: SUM_SIGNED(int8_t, raw); break;
    case ME_INT16: SUM_SIGNED(int16_t, raw); break;
    case ME_INT32: SUM_SIGNED(int32_t, raw); break;
    case ME_INT64: SUM_SIGNED(int64_t, raw); break;
    case ME_UINT8: SUM_UNSIGNED(uint8_t); break;
    case ME_UINT16: SUM_UNSIGNED(uint16_t); break;
    case ME_UINT32: SUM_UNSIGNED(uint32_t); break;
    case ME_UINT64: SUM_UNSIGNED(uint64_t); break;
    default: return false;
    }
#undef SUM_SIGNED
#undef SUM_UNSIGNED
    *out = sum;
    return true;
}
static inline float dsl_portable_prod_f32(float product, const void *values, size_t count) {
    const unsigned char *data = values;
    for (size_t i = 0; i < count; i++) {
        float value; memcpy(&value, data + i * sizeof(value), sizeof(value));
        product *= value;
    }
    return product;
}
static inline double dsl_portable_prod_f64(double product, const void *values, size_t count) {
    const unsigned char *data = values;
    for (size_t i = 0; i < count; i++) {
        double value; memcpy(&value, data + i * sizeof(value), sizeof(value));
        product *= value;
    }
    return product;
}
static inline bool dsl_portable_product_checked(int64_t left, int64_t right) {
#if defined(__GNUC__) || defined(__clang__)
    int64_t result;
    return !__builtin_mul_overflow(left, right, &result);
#else
    bool negative = (left < 0) != (right < 0);
    uint64_t a = left < 0 ? (uint64_t)(-(left + 1)) + 1 : (uint64_t)left;
    uint64_t b = right < 0 ? (uint64_t)(-(right + 1)) + 1 : (uint64_t)right;
    uint64_t limit = negative ? UINT64_C(1) << 63 : INT64_MAX;
    return !b || a <= limit / b;
#endif
}
static inline bool dsl_portable_prod_integer(me_dtype source, const void *values,
    size_t count, uint64_t initial, bool checked, uint64_t *out) {
    const unsigned char *data = values;
    uint64_t product = initial;
#define PROD_SIGNED(type, expression) do { \
    for (size_t i = 0; i < count; i++) { \
        type raw; memcpy(&raw, data + i * sizeof(raw), sizeof(raw)); \
        int64_t value = (expression), current; \
        memcpy(&current, &product, sizeof(current)); \
        if (checked && !dsl_portable_product_checked(current, value)) return false; \
        product *= (uint64_t)value; \
    } \
} while (0)
#define PROD_UNSIGNED(type) do { \
    for (size_t i = 0; i < count; i++) { \
        type value; memcpy(&value, data + i * sizeof(value), sizeof(value)); \
        if (checked && value && product > UINT64_MAX / value) return false; \
        product *= (uint64_t)value; \
    } \
} while (0)
    switch (source) {
    case ME_BOOL: PROD_SIGNED(bool, raw ? 1 : 0); break;
    case ME_INT8: PROD_SIGNED(int8_t, raw); break;
    case ME_INT16: PROD_SIGNED(int16_t, raw); break;
    case ME_INT32: PROD_SIGNED(int32_t, raw); break;
    case ME_INT64: PROD_SIGNED(int64_t, raw); break;
    case ME_UINT8: PROD_UNSIGNED(uint8_t); break;
    case ME_UINT16: PROD_UNSIGNED(uint16_t); break;
    case ME_UINT32: PROD_UNSIGNED(uint32_t); break;
    case ME_UINT64: PROD_UNSIGNED(uint64_t); break;
    default: return false;
    }
#undef PROD_SIGNED
#undef PROD_UNSIGNED
    *out = product;
    return true;
}
/* Scan by representation, without quieting signaling NaNs or raising flags.
 * NaN extrema/truth cases retain their original engine-specific behavior. */
static inline bool dsl_portable_no_nan(me_dtype dtype, const void *values, size_t count) {
    const unsigned char *data = values;
    if (dtype == ME_FLOAT32) {
        for (size_t i = 0; i < count; i++) {
            uint32_t bits; memcpy(&bits, data + i * 4, 4);
            if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) return false;
        }
    } else if (dtype == ME_FLOAT64) {
        for (size_t i = 0; i < count; i++) {
            uint64_t bits; memcpy(&bits, data + i * 8, 8);
            if ((bits & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000)) return false;
        }
    }
    return true;
}
static inline bool dsl_portable_extrema(me_dtype dtype, const void *values, size_t count,
    uint64_t initial, bool initialized, bool minimum, uint64_t *out) {
    const unsigned char *data = values;
#define EXTREMA(type, expression) do { \
    uint64_t best = initial; \
    for (size_t i = 0; i < count; i++) { \
        type raw; memcpy(&raw, data + i * sizeof(raw), sizeof(raw)); \
        expression value = raw, current; memcpy(&current, &best, sizeof(current)); \
        if (!initialized || (minimum ? value < current : value > current)) best = (uint64_t)value; \
        initialized = true; \
    } \
    *out = best; \
} while (0)
    switch (dtype) {
    case ME_BOOL: EXTREMA(bool, uint64_t); break;
    case ME_INT8: EXTREMA(int8_t, int64_t); break;
    case ME_INT16: EXTREMA(int16_t, int64_t); break;
    case ME_INT32: EXTREMA(int32_t, int64_t); break;
    case ME_INT64: EXTREMA(int64_t, int64_t); break;
    case ME_UINT8: EXTREMA(uint8_t, uint64_t); break;
    case ME_UINT16: EXTREMA(uint16_t, uint64_t); break;
    case ME_UINT32: EXTREMA(uint32_t, uint64_t); break;
    case ME_UINT64: EXTREMA(uint64_t, uint64_t); break;
    case ME_FLOAT32: {
        if (!dsl_portable_no_nan(dtype, values, count) ||
            (initialized && !dsl_portable_no_nan(dtype, &initial, 1))) return false;
        float best; memcpy(&best, &initial, sizeof(best));
        for (size_t i = 0; i < count; i++) {
            float value; memcpy(&value, data + i * sizeof(value), sizeof(value));
            if (!initialized || (minimum ? value < best : value > best) ||
                (value == 0 && best == 0 && (minimum ? signbit(value) : !signbit(value)))) best = value;
            initialized = true;
        }
        *out = 0; memcpy(out, &best, sizeof(best)); break;
    }
    case ME_FLOAT64: {
        if (!dsl_portable_no_nan(dtype, values, count) ||
            (initialized && !dsl_portable_no_nan(dtype, &initial, 1))) return false;
        double best; memcpy(&best, &initial, sizeof(best));
        for (size_t i = 0; i < count; i++) {
            double value; memcpy(&value, data + i * sizeof(value), sizeof(value));
            if (!initialized || (minimum ? value < best : value > best) ||
                (value == 0 && best == 0 && (minimum ? signbit(value) : !signbit(value)))) best = value;
            initialized = true;
        }
        memcpy(out, &best, sizeof(best)); break;
    }
    default: return false;
    }
#undef EXTREMA
    return initialized;
}
static inline bool dsl_portable_truth_reduce(me_dtype dtype, const void *values, size_t count,
    bool initial, bool all, bool *out) {
    if (!dsl_portable_no_nan(dtype, values, count)) return false;
    const unsigned char *data = values;
    bool result = initial;
#define TRUTH(type) do { \
    for (size_t i = 0; i < count; i++) { \
        type value; memcpy(&value, data + i * sizeof(value), sizeof(value)); \
        bool lane = value != 0; result = all ? result && lane : result || lane; \
    } \
} while (0)
    switch (dtype) {
    case ME_BOOL: TRUTH(bool); break;
    case ME_INT8: TRUTH(int8_t); break;
    case ME_INT16: TRUTH(int16_t); break;
    case ME_INT32: TRUTH(int32_t); break;
    case ME_INT64: TRUTH(int64_t); break;
    case ME_UINT8: TRUTH(uint8_t); break;
    case ME_UINT16: TRUTH(uint16_t); break;
    case ME_UINT32: TRUTH(uint32_t); break;
    case ME_UINT64: TRUTH(uint64_t); break;
    case ME_FLOAT32: TRUTH(float); break;
    case ME_FLOAT64: TRUTH(double); break;
    default: return false;
    }
#undef TRUTH
    *out = result;
    return true;
}
#endif
