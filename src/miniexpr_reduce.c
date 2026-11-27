/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "miniexpr_reduce.h"
#include "dsl_menudet_reduce.h"

static bool reduce_float(me_dtype dtype) {
    return dtype == ME_FLOAT32 || dtype == ME_FLOAT64;
}

static bool reduce_complex(me_dtype dtype) {
    return dtype == ME_COMPLEX64 || dtype == ME_COMPLEX128;
}

me_dtype me_reduce_accumulator_dtype(me_dtype source, me_reduction_kind kind, me_dtype requested) {
    if (source < ME_BOOL || source > ME_COMPLEX128 ||
        kind < ME_REDUCTION_NONE || kind > ME_REDUCTION_ALL) {
        return ME_AUTO;
    }
    if (kind == ME_REDUCTION_ANY || kind == ME_REDUCTION_ALL) {
        return requested == ME_AUTO ? ME_BOOL : ME_AUTO;
    }
    if (requested != ME_AUTO) {
        bool compatible = reduce_float(source) ? reduce_float(requested) :
            reduce_complex(source) ? (reduce_float(requested) || reduce_complex(requested)) : true;
        return requested >= ME_BOOL && requested <= ME_COMPLEX128 && compatible ? requested : ME_AUTO;
    }
    if ((kind == ME_REDUCTION_SUM || kind == ME_REDUCTION_PROD) &&
        !reduce_float(source) && !reduce_complex(source)) {
        return source >= ME_UINT8 && source <= ME_UINT64 ? ME_UINT64 : ME_INT64;
    }
    return source;
}

bool me_reduce_identity(me_reduction_kind kind, me_dtype accumulator, uint64_t *out) {
    size_t width = me_extrema_itemsize(accumulator);
    if (!out || !width || (kind != ME_REDUCTION_SUM && kind != ME_REDUCTION_PROD &&
        kind != ME_REDUCTION_ANY && kind != ME_REDUCTION_ALL)) {
        return false;
    }
    uint64_t result = 0;
    if (kind == ME_REDUCTION_PROD || kind == ME_REDUCTION_ALL) {
        if (accumulator == ME_FLOAT32) {
            float one = 1;
            memcpy(&result, &one, sizeof(one));
        } else if (accumulator == ME_FLOAT64) {
            double one = 1;
            memcpy(&result, &one, sizeof(one));
        } else {
            result = 1;
        }
    }
    *out = result;
    return true;
}
static bool serial_span(me_dtype source, const void *values, size_t count,
                        uint64_t initial, bool checked, bool product, uint64_t *out) {
    if (!out || (count && !values)) {
        return false;
    }
    uint64_t result = 0;
    if (source == ME_FLOAT32) {
        float acc;
        memcpy(&acc, &initial, sizeof(acc));
        acc = product ? dsl_menudet_prod_f32(acc, values, count) :
            dsl_menudet_sum_f32(acc, values, count);
        memcpy(&result, &acc, sizeof(acc));
    } else if (source == ME_FLOAT64) {
        double acc;
        memcpy(&acc, &initial, sizeof(acc));
        acc = product ? dsl_menudet_prod_f64(acc, values, count) :
            dsl_menudet_sum_f64(acc, values, count);
        memcpy(&result, &acc, sizeof(acc));
    } else {
        bool success = product ?
            dsl_menudet_prod_integer(source, values, count, initial, checked, &result) :
            dsl_menudet_sum_integer(source, values, count, initial, checked, &result);
        if (!success) {
            return false;
        }
    }
    *out = result;
    return true;
}

bool me_sum_span(me_dtype source, const void *values, size_t count,
                 uint64_t initial, bool checked, uint64_t *out) {
    return serial_span(source, values, count, initial, checked, false, out);
}

bool me_prod_span(me_dtype source, const void *values, size_t count,
                  uint64_t initial, bool checked, uint64_t *out) {
    return serial_span(source, values, count, initial, checked, true, out);
}

bool me_truth_span(me_dtype source, const void *values, size_t count,
                   bool initial, bool all, bool *out) {
    if (!out || (count && !values) || source < ME_BOOL || source > ME_FLOAT64) {
        return false;
    }
    const unsigned char *data = (const unsigned char *)values;
    bool result = initial;
#define TRUTH_SPAN(TYPE) do { \
    for (size_t i = 0; i < count; i++) { \
        TYPE value; \
        memcpy(&value, data + i * sizeof(value), sizeof(value)); \
        bool lane = value != 0; \
        result = all ? (result && lane) : (result || lane); \
    } \
} while (0)
    switch (source) {
    case ME_BOOL:
        for (size_t i = 0; i < count; i++) {
            bool lane = data[i] != 0;
            result = all ? (result && lane) : (result || lane);
        }
        break;
    case ME_INT8: TRUTH_SPAN(int8_t); break;
    case ME_INT16: TRUTH_SPAN(int16_t); break;
    case ME_INT32: TRUTH_SPAN(int32_t); break;
    case ME_INT64: TRUTH_SPAN(int64_t); break;
    case ME_UINT8: TRUTH_SPAN(uint8_t); break;
    case ME_UINT16: TRUTH_SPAN(uint16_t); break;
    case ME_UINT32: TRUTH_SPAN(uint32_t); break;
    case ME_UINT64: TRUTH_SPAN(uint64_t); break;
    case ME_FLOAT32:
        for (size_t i = 0; i < count; i++) {
            float raw;
            memcpy(&raw, data + i * sizeof(raw), sizeof(raw));
            double value = raw;
            bool lane = value != 0.0;
            result = all ? (result && lane) : (result || lane);
        }
        break;
    case ME_FLOAT64:
        for (size_t i = 0; i < count; i++) {
            double value;
            memcpy(&value, data + i * sizeof(value), sizeof(value));
            bool lane = value != 0.0;
            result = all ? (result && lane) : (result || lane);
        }
        break;
    default:
        return false;
    }
#undef TRUTH_SPAN
    *out = result;
    return true;
}

bool me_reduce_store_scalar(me_dtype dtype, uint64_t value, void *out) {
    size_t width = me_extrema_itemsize(dtype);
    if (!out || !width) {
        return false;
    }
    if (dtype == ME_FLOAT32 || dtype == ME_FLOAT64) {
        memcpy(out, &value, width);
        return true;
    }
    uint16_t endian = 1;
    size_t offset = *(const unsigned char *)&endian ? 0 : sizeof(value) - width;
    memcpy(out, (const unsigned char *)&value + offset, width);
    return true;
}
