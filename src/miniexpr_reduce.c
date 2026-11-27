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
            uint64_t one = 1;
            uint16_t endian = 1;
            memcpy(&result, (const unsigned char *)&one + (*(unsigned char *)&endian ? 0 : 8 - width), width);
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
