/* Shared scalar extrema and mergeable arg-extrema for embedding schedulers.
 * Callers validate buffers and own the floating environment. No allocation,
 * alignment requirement, or dependency on the graph executor.
 */
#ifndef MINIEXPR_REDUCE_H
#define MINIEXPR_REDUCE_H
#include "miniexpr.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define ME_EXTREMA_REDUCE_API 1

typedef struct {
    uint64_t value;
    int64_t index;
    bool initialized;
} me_argextrema_state;

/* Equal values (including signed zeros) choose the earliest logical position.
 * NaNs outrank finite values; among NaNs the earliest position wins. Comparing
 * floating values through double matches the native graph's float32 widening.
 * Index reductions preserve the selected value representation unchanged.
 */
static inline bool me_argextrema_merge(me_dtype dtype, me_argextrema_state *state,
    uint64_t value, int64_t index, bool minimum) {
    if (dtype < ME_BOOL || dtype > ME_FLOAT64 || index < 0) return false;
    bool better = !state->initialized;
    bool equal = false;
#define ME_ARG_COMPARE(TYPE) do { \
    TYPE a, b; \
    memcpy(&a, &state->value, sizeof(a)); \
    memcpy(&b, &value, sizeof(b)); \
    better = minimum ? b < a : b > a; \
    equal = b == a; \
} while (0)
    if (state->initialized) {
        switch (dtype) {
        case ME_BOOL: {
            bool a = state->value != 0, b = value != 0;
            better = minimum ? b < a : b > a;
            equal = b == a;
            break;
        }
        case ME_UINT8: ME_ARG_COMPARE(uint8_t); break;
        case ME_UINT16: ME_ARG_COMPARE(uint16_t); break;
        case ME_UINT32: ME_ARG_COMPARE(uint32_t); break;
        case ME_UINT64: ME_ARG_COMPARE(uint64_t); break;
        case ME_INT8: ME_ARG_COMPARE(int8_t); break;
        case ME_INT16: ME_ARG_COMPARE(int16_t); break;
        case ME_INT32: ME_ARG_COMPARE(int32_t); break;
        case ME_INT64: ME_ARG_COMPARE(int64_t); break;
        case ME_FLOAT32:
        case ME_FLOAT64: {
            double a, b;
            if (dtype == ME_FLOAT32) {
                float x, y;
                memcpy(&x, &state->value, sizeof(x));
                memcpy(&y, &value, sizeof(y));
                a = x; b = y;
            } else {
                memcpy(&a, &state->value, sizeof(a));
                memcpy(&b, &value, sizeof(b));
            }
            bool an = isnan(a), bn = isnan(b);
            better = bn ? !an : (!an && (minimum ? b < a : b > a));
            equal = (an && bn) || a == b;
            break;
        }
        default: return false;
        }
    }
#undef ME_ARG_COMPARE
    if (better || (equal && index < state->index)) {
        state->value = value;
        state->index = index;
        state->initialized = true;
    }
    return true;
}

static inline size_t me_extrema_itemsize(me_dtype dtype) {
    switch (dtype) {
    case ME_BOOL: case ME_INT8: case ME_UINT8: return 1;
    case ME_INT16: case ME_UINT16: return 2;
    case ME_INT32: case ME_UINT32: case ME_FLOAT32: return 4;
    case ME_INT64: case ME_UINT64: case ME_FLOAT64: return 8;
    default: return 0;
    }
}

static inline bool me_argextrema_update(me_dtype dtype, const void *values,
    size_t count, int64_t origin, bool minimum, me_argextrema_state *state) {
    size_t width = me_extrema_itemsize(dtype);
    if (!width || origin < 0 || count > (uint64_t)(INT64_MAX - origin)) return false;
    if (!count) return true;
    const unsigned char *data = values;
#define ME_ARG_SCAN(TYPE, FLOATING) do { \
    TYPE best; \
    size_t begin = 0; \
    int64_t best_index = state->index; \
    if (state->initialized) memcpy(&best, &state->value, sizeof(best)); \
    else { memcpy(&best, data, sizeof(best)); best_index = origin; begin = 1; } \
    for (size_t i = begin; i < count; i++) { \
        TYPE value; memcpy(&value, data + i * sizeof(value), sizeof(value)); \
        int64_t index = origin + (int64_t)i; \
        if ((minimum ? value < best : value > best) || \
            ((FLOATING) && isnan((double)value) && !isnan((double)best)) || \
            (index < best_index && (value == best || \
                ((FLOATING) && isnan((double)value) && isnan((double)best))))) { \
            best = value; best_index = index; \
        } \
    } \
    state->value = 0; memcpy(&state->value, &best, sizeof(best)); \
    state->index = best_index; state->initialized = true; \
} while (0)
    /* Dispatch once per span, not once per lane. */
    switch (dtype) {
    case ME_BOOL: {
        bool best = state->initialized ? state->value != 0 : data[0] != 0;
        int64_t index = state->initialized ? state->index : origin;
        for (size_t i = state->initialized ? 0 : 1; i < count; i++) {
            bool value = data[i] != 0;
            int64_t position = origin + (int64_t)i;
            if ((minimum ? value < best : value > best) || (value == best && position < index)) {
                best = value; index = position;
            }
        }
        state->value = best ? 1 : 0;
        state->index = index; state->initialized = true;
        break;
    }
    case ME_UINT8: ME_ARG_SCAN(uint8_t, 0); break;
    case ME_UINT16: ME_ARG_SCAN(uint16_t, 0); break;
    case ME_UINT32: ME_ARG_SCAN(uint32_t, 0); break;
    case ME_UINT64: ME_ARG_SCAN(uint64_t, 0); break;
    case ME_INT8: ME_ARG_SCAN(int8_t, 0); break;
    case ME_INT16: ME_ARG_SCAN(int16_t, 0); break;
    case ME_INT32: ME_ARG_SCAN(int32_t, 0); break;
    case ME_INT64: ME_ARG_SCAN(int64_t, 0); break;
    case ME_FLOAT32: ME_ARG_SCAN(float, 1); break;
    case ME_FLOAT64: ME_ARG_SCAN(double, 1); break;
    default: return false;
    }
#undef ME_ARG_SCAN
    return true;
}

/* Float extrema retain the first NaN payload (quieting signaling NaNs),
 * prefer -0 for min and +0 for max, and inspect every lane for FP status.
 * Integer callers use the existing typed shared reducer.
 */
static inline bool me_extrema_float(me_dtype dtype, const void *values,
    size_t count, uint64_t initial, bool initialized, bool minimum, uint64_t *out) {
    const unsigned char *data = values;
    double best = 0;
    if (initialized) {
        if (dtype == ME_FLOAT32) {
            float value; memcpy(&value, &initial, sizeof(value)); best = value;
        } else if (dtype == ME_FLOAT64) {
            memcpy(&best, &initial, sizeof(best));
        } else return false;
    }
    if (dtype != ME_FLOAT32 && dtype != ME_FLOAT64) return false;
    for (size_t i = 0; i < count; i++) {
        double value;
        if (dtype == ME_FLOAT32) {
            float raw; memcpy(&raw, data + i * sizeof(raw), sizeof(raw)); value = raw;
        } else {
            memcpy(&value, data + i * sizeof(value), sizeof(value));
        }
        if (!initialized) best = value;
        else if (isnan(best)) { /* Keep first NaN, but still inspect each lane. */
            (void)isnan(value);
        } else if (isnan(value) || (minimum ? value < best : value > best) ||
                   (value == 0 && best == 0 && (minimum ? signbit(value) : !signbit(value)))) {
            best = value;
        }
        initialized = true;
    }
    if (!initialized) return false;
    /* Quiet the selected NaN without replacing its payload. */
    if (isnan(best)) best = best + best;
    *out = 0;
    if (dtype == ME_FLOAT32) {
        float value = (float)best; memcpy(out, &value, sizeof(value));
    } else memcpy(out, &best, sizeof(best));
    return true;
}
#endif
