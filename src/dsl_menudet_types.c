/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "dsl_menudet_types.h"
#include <limits.h>
#include <math.h>
#include <string.h>

static bool menudet_signed_bounds(me_dtype dtype, int64_t *minimum, int64_t *maximum);

static int menudet_integer_bits(me_dtype dtype) {
    switch (dtype) {
    case ME_INT8: case ME_UINT8: return 8;
    case ME_INT16: case ME_UINT16: return 16;
    case ME_INT32: case ME_UINT32: return 32;
    case ME_INT64: case ME_UINT64: return 64;
    default: return 0;
    }
}

static bool menudet_unsigned(me_dtype dtype) {
    return dtype == ME_UINT8 || dtype == ME_UINT16 ||
           dtype == ME_UINT32 || dtype == ME_UINT64;
}

static me_dtype menudet_integer_type(int bits, bool is_unsigned) {
    if (bits <= 8) return is_unsigned ? ME_UINT8 : ME_INT8;
    if (bits <= 16) return is_unsigned ? ME_UINT16 : ME_INT16;
    if (bits <= 32) return is_unsigned ? ME_UINT32 : ME_INT32;
    if (bits <= 64) return is_unsigned ? ME_UINT64 : ME_INT64;
    return ME_AUTO;
}

bool dsl_menudet_is_complex(me_dtype dtype) {
    return dtype == ME_COMPLEX64 || dtype == ME_COMPLEX128;
}

me_dtype dsl_menudet_complex_component(me_dtype dtype) {
    if (dtype == ME_COMPLEX64) return ME_FLOAT32;
    if (dtype == ME_COMPLEX128) return ME_FLOAT64;
    return ME_AUTO;
}

/* Component width that a value contributes to a complex promotion. C matches
 * NumPy: 16-bit integers fit complex64's 24-bit significand, wider ones do not. */
static me_dtype menudet_complex_component(me_dtype dtype) {
    if (dsl_menudet_is_complex(dtype)) return dsl_menudet_complex_component(dtype);
    if (dtype == ME_BOOL) return ME_FLOAT32;
    if (dtype == ME_FLOAT32 || dtype == ME_FLOAT64) return dtype;
    if (menudet_integer_bits(dtype) <= 16) return ME_FLOAT32;
    return menudet_integer_bits(dtype) ? ME_FLOAT64 : ME_AUTO;
}

me_dtype dsl_menudet_numeric_promote(me_dtype left, me_dtype right) {
    /* Truth conversion is separate from arithmetic. Numeric Boolean values
     * are strong int64 zero/one operands, not Boolean temporaries. */
    if (left == ME_BOOL) left = ME_INT64;
    if (right == ME_BOOL) right = ME_INT64;
    if (dsl_menudet_is_complex(left) || dsl_menudet_is_complex(right)) {
        me_dtype left_component = menudet_complex_component(left);
        me_dtype right_component = menudet_complex_component(right);
        if (left_component == ME_AUTO || right_component == ME_AUTO) return ME_AUTO;
        /* Complex arithmetic keeps the width needed for both real components. */
        return left_component == ME_FLOAT64 || right_component == ME_FLOAT64 ?
               ME_COMPLEX128 : ME_COMPLEX64;
    }
    int left_bits = menudet_integer_bits(left);
    int right_bits = menudet_integer_bits(right);
    bool left_float = left == ME_FLOAT32 || left == ME_FLOAT64;
    bool right_float = right == ME_FLOAT32 || right == ME_FLOAT64;
    if ((!left_bits && !left_float) || (!right_bits && !right_float)) {
        return ME_AUTO;
    }
    if (left_float || right_float) {
        if (left == ME_FLOAT64 || right == ME_FLOAT64 ||
            left_bits > 16 || right_bits > 16) {
            return ME_FLOAT64;
        }
        return ME_FLOAT32;
    }
    bool left_unsigned = menudet_unsigned(left);
    bool right_unsigned = menudet_unsigned(right);
    if (left_unsigned == right_unsigned) {
        return menudet_integer_type(left_bits > right_bits ? left_bits : right_bits,
                                     left_unsigned);
    }
    int signed_bits = left_unsigned ? right_bits : left_bits;
    int unsigned_bits = left_unsigned ? left_bits : right_bits;
    /* A signed common type needs one additional bit for the unsigned range. */
    int required_bits = signed_bits > unsigned_bits ? signed_bits : unsigned_bits + 1;
    return menudet_integer_type(required_bits, false);
}

me_dtype dsl_menudet_division_dtype(me_dtype left, me_dtype right) {
    me_dtype promoted = dsl_menudet_numeric_promote(left, right);
    if (promoted == ME_AUTO || promoted == ME_FLOAT32 || promoted == ME_FLOAT64 ||
        dsl_menudet_is_complex(promoted)) {
        return promoted;
    }
    return ME_FLOAT64;
}

me_dtype dsl_numpy_numeric_promote(me_dtype left, me_dtype right) {
    if (left == ME_BOOL && right == ME_BOOL) return ME_BOOL;
    if (left == ME_BOOL) return dsl_menudet_numeric_promote(right, right);
    if (right == ME_BOOL) return dsl_menudet_numeric_promote(left, left);
    me_dtype common = dsl_menudet_numeric_promote(left, right);
    /* int64/uint64 has no lossless integer common type in NumPy. */
    if (common == ME_AUTO && menudet_integer_bits(left) && menudet_integer_bits(right)) return ME_FLOAT64;
    return common;
}

bool dsl_numpy_can_cast(me_dtype from, me_dtype to, const char *policy) {
    if (!policy || !strcmp(policy, "unsafe")) return dsl_numpy_numeric_promote(from, to) != ME_AUTO;
    if (!strcmp(policy, "safe")) return dsl_numpy_numeric_promote(from, to) == to;
    if (strcmp(policy, "same_kind")) return false;
    int a = from == ME_BOOL ? 0 : menudet_unsigned(from) ? 1 : menudet_integer_bits(from) ? 2 :
            dsl_menudet_is_complex(from) ? 4 : 3;
    int b = to == ME_BOOL ? 0 : menudet_unsigned(to) ? 1 : menudet_integer_bits(to) ? 2 :
            dsl_menudet_is_complex(to) ? 4 : 3;
    return dsl_numpy_numeric_promote(from, to) != ME_AUTO && a <= b;
}

uint64_t dsl_numpy_unsigned_bits(me_dtype dtype, uint64_t raw) {
    int bits = menudet_integer_bits(dtype);
    return bits == 64 ? raw : raw & ((UINT64_C(1) << bits) - 1);
}

int64_t dsl_numpy_signed_bits(me_dtype dtype, uint64_t raw) {
    int bits = menudet_integer_bits(dtype);
    uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    raw &= mask;
    return raw & (UINT64_C(1) << (bits - 1)) ? -1 - (int64_t)(mask - raw) : (int64_t)raw;
}

me_menudet_numeric_status dsl_numpy_unsigned_op(me_dtype dtype, me_menudet_integer_op op,
                                                uint64_t left, uint64_t right, uint64_t *out) {
    int bits = menudet_integer_bits(dtype);
    if (!out || !bits || !menudet_unsigned(dtype)) return ME_MENUDET_NUMERIC_TYPE;
    uint64_t raw;
    switch (op) {
    case ME_MENUDET_ADD: raw = left + right; break;
    case ME_MENUDET_SUB: raw = left - right; break;
    case ME_MENUDET_MUL: raw = left * right; break;
    case ME_MENUDET_FLOORDIV: raw = right ? left / right : 0; break;
    case ME_MENUDET_MOD: raw = right ? left % right : 0; break;
    case ME_MENUDET_SHL: raw = right >= (uint64_t)bits ? 0 : left << right; break;
    case ME_MENUDET_SHR: raw = right >= (uint64_t)bits ? 0 : left >> right; break;
    case ME_MENUDET_AND: raw = left & right; break;
    case ME_MENUDET_OR: raw = left | right; break;
    case ME_MENUDET_XOR: raw = left ^ right; break;
    case ME_MENUDET_POW:
        raw = 1;
        while (right) {
            if (right & 1) raw *= left;
            right >>= 1;
            if (right) left *= left;
        }
        break;
    default: return ME_MENUDET_NUMERIC_TYPE;
    }
    *out = dsl_numpy_unsigned_bits(dtype, raw);
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_numpy_signed_op(me_dtype dtype, me_menudet_integer_op op,
                                              int64_t left, int64_t right, int64_t *out) {
    int64_t minimum, maximum;
    if (!out || !menudet_signed_bounds(dtype, &minimum, &maximum)) return ME_MENUDET_NUMERIC_TYPE;
    int bits = menudet_integer_bits(dtype);
    if (op == ME_MENUDET_FLOORDIV || op == ME_MENUDET_MOD) {
        if (!right) *out = 0;
        else if (left == minimum && right == -1) *out = op == ME_MENUDET_MOD ? 0 : minimum;
        else {
            int64_t q = left / right, r = left % right;
            if (r && ((left < 0) != (right < 0))) { q--; r += right; }
            *out = op == ME_MENUDET_MOD ? r : q;
        }
        return ME_MENUDET_NUMERIC_OK;
    }
    if (op == ME_MENUDET_SHL || op == ME_MENUDET_SHR) {
        if (right < 0 || right >= bits) *out = op == ME_MENUDET_SHR && left < 0 ? -1 : 0;
        else if (op == ME_MENUDET_SHL) *out = dsl_numpy_signed_bits(dtype, (uint64_t)left << right);
        else *out = left < 0 ? -1 - (int64_t)((uint64_t)(-(left + 1)) >> right) :
                              (int64_t)((uint64_t)left >> right);
        return ME_MENUDET_NUMERIC_OK;
    }
    if (op == ME_MENUDET_POW && right < 0) return ME_MENUDET_NUMERIC_RANGE;
    uint64_t raw;
    me_dtype unsigned_dtype = menudet_integer_type(bits, true);
    me_menudet_numeric_status status = dsl_numpy_unsigned_op(unsigned_dtype, op, (uint64_t)left, (uint64_t)right, &raw);
    if (status == ME_MENUDET_NUMERIC_OK) *out = dsl_numpy_signed_bits(dtype, raw);
    return status;
}

static bool menudet_signed_bounds(me_dtype dtype, int64_t *minimum, int64_t *maximum) {
    switch (dtype) {
    case ME_INT8: *minimum = INT8_MIN; *maximum = INT8_MAX; return true;
    case ME_INT16: *minimum = INT16_MIN; *maximum = INT16_MAX; return true;
    case ME_INT32: *minimum = INT32_MIN; *maximum = INT32_MAX; return true;
    case ME_INT64: *minimum = INT64_MIN; *maximum = INT64_MAX; return true;
    default: return false;
    }
}

static uint64_t menudet_magnitude(int64_t value) {
    /* -(INT64_MIN) is not representable; -(value + 1) always is. */
    return value < 0 ? (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;
}

static me_menudet_numeric_status menudet_signed_product(int64_t left, int64_t right,
                                                           int64_t minimum, int64_t maximum,
                                                           int64_t *out) {
    bool negative = (left < 0) != (right < 0);
    uint64_t a = menudet_magnitude(left);
    uint64_t b = menudet_magnitude(right);
    uint64_t limit = negative ? menudet_magnitude(minimum) : (uint64_t)maximum;
    if (b && a > limit / b) return ME_MENUDET_NUMERIC_RANGE;
    uint64_t product = a * b;
    if (negative && product == (UINT64_C(1) << 63)) {
        *out = INT64_MIN;
    }
    else {
        *out = negative ? -(int64_t)product : (int64_t)product;
    }
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_signed_op(me_dtype dtype, me_menudet_integer_op op,
                                                 int64_t left, int64_t right, int64_t *out) {
    int64_t minimum, maximum;
    if (!out || !menudet_signed_bounds(dtype, &minimum, &maximum)) return ME_MENUDET_NUMERIC_TYPE;
    if (left < minimum || left > maximum) return ME_MENUDET_NUMERIC_RANGE;
    bool shift = op == ME_MENUDET_SHL || op == ME_MENUDET_SHR;
    if (!shift && (right < minimum || right > maximum)) return ME_MENUDET_NUMERIC_RANGE;
    int64_t result;
    switch (op) {
    case ME_MENUDET_ADD:
        if ((right > 0 && left > maximum - right) ||
            (right < 0 && left < minimum - right)) return ME_MENUDET_NUMERIC_RANGE;
        result = left + right;
        break;
    case ME_MENUDET_SUB:
        if ((right < 0 && left > maximum + right) ||
            (right > 0 && left < minimum + right)) return ME_MENUDET_NUMERIC_RANGE;
        result = left - right;
        break;
    case ME_MENUDET_MUL:
        return menudet_signed_product(left, right, minimum, maximum, out);
    case ME_MENUDET_FLOORDIV:
    case ME_MENUDET_MOD:
        if (!right) return ME_MENUDET_NUMERIC_ZERO;
        if (left == minimum && right == -1) {
            if (op == ME_MENUDET_FLOORDIV) return ME_MENUDET_NUMERIC_RANGE;
            result = 0;
        }
        else {
            int64_t quotient = left / right;
            int64_t remainder = left % right;
            if (remainder && ((left < 0) != (right < 0))) {
                quotient--;
                remainder += right;
            }
            result = op == ME_MENUDET_FLOORDIV ? quotient : remainder;
        }
        break;
    case ME_MENUDET_SHL:
    case ME_MENUDET_SHR: {
        int bits = menudet_integer_bits(dtype);
        if (right < 0 || right >= bits) return ME_MENUDET_NUMERIC_SHIFT;
        if (op == ME_MENUDET_SHL) {
            /* Define left shift as checked multiplication, including negative
             * operands. Multiplication by two avoids casting 2**63 to int64. */
            result = left;
            for (int64_t i = 0; i < right; i++) {
                me_menudet_numeric_status status = menudet_signed_product(result, 2, minimum, maximum, &result);
                if (status != ME_MENUDET_NUMERIC_OK) return status;
            }
        }
        else {
            /* Floor division by 2**count, independent of C's implementation-
             * defined signed right shift. Negative rounding uses -(x+1). */
            result = left < 0 ? -1 - (int64_t)((uint64_t)(-(left + 1)) >> right)
                              : (int64_t)((uint64_t)left >> right);
        }
        break;
    }
    case ME_MENUDET_POW: {
        if (right < 0) return ME_MENUDET_NUMERIC_RANGE;
        result = 1;
        int64_t base = left;
        uint64_t exponent = (uint64_t)right;
        while (exponent) {
            me_menudet_numeric_status status;
            if (exponent & 1) {
                status = menudet_signed_product(result, base, minimum, maximum, &result);
                if (status != ME_MENUDET_NUMERIC_OK) return status;
            }
            exponent >>= 1;
            if (exponent) {
                status = menudet_signed_product(base, base, minimum, maximum, &base);
                if (status != ME_MENUDET_NUMERIC_OK) return status;
            }
        }
        break;
    }
    case ME_MENUDET_AND:
    case ME_MENUDET_OR:
    case ME_MENUDET_XOR: {
        uint64_t raw = op == ME_MENUDET_AND ? (uint64_t)left & (uint64_t)right :
                       op == ME_MENUDET_OR ? (uint64_t)left | (uint64_t)right :
                                             (uint64_t)left ^ (uint64_t)right;
        int bits = menudet_integer_bits(dtype);
        uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
        raw &= mask;
        /* Interpret a fixed-width two's-complement result without an
         * implementation-defined unsigned-to-signed narrowing cast. */
        result = raw & (UINT64_C(1) << (bits - 1)) ? -1 - (int64_t)(mask - raw) : (int64_t)raw;
        break;
    }
    default: return ME_MENUDET_NUMERIC_TYPE;
    }
    *out = result;
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_unsigned_op(me_dtype dtype, me_menudet_integer_op op,
                                                   uint64_t left, uint64_t right, uint64_t *out) {
    int bits = menudet_integer_bits(dtype);
    if (!out || !bits || !menudet_unsigned(dtype)) return ME_MENUDET_NUMERIC_TYPE;
    uint64_t maximum = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    if (left > maximum) return ME_MENUDET_NUMERIC_RANGE;
    bool shift = op == ME_MENUDET_SHL || op == ME_MENUDET_SHR;
    if (!shift && right > maximum) return ME_MENUDET_NUMERIC_RANGE;
    uint64_t result;
    switch (op) {
    case ME_MENUDET_ADD:
        if (left > maximum - right) return ME_MENUDET_NUMERIC_RANGE;
        result = left + right;
        break;
    case ME_MENUDET_SUB:
        if (left < right) return ME_MENUDET_NUMERIC_RANGE;
        result = left - right;
        break;
    case ME_MENUDET_MUL:
        if (right && left > maximum / right) return ME_MENUDET_NUMERIC_RANGE;
        result = left * right;
        break;
    case ME_MENUDET_FLOORDIV:
    case ME_MENUDET_MOD:
        if (!right) return ME_MENUDET_NUMERIC_ZERO;
        result = op == ME_MENUDET_FLOORDIV ? left / right : left % right;
        break;
    case ME_MENUDET_SHL:
    case ME_MENUDET_SHR:
        if (right >= (uint64_t)bits) return ME_MENUDET_NUMERIC_SHIFT;
        if (op == ME_MENUDET_SHL && left > (maximum >> right)) return ME_MENUDET_NUMERIC_RANGE;
        result = op == ME_MENUDET_SHL ? left << right : left >> right;
        break;
    case ME_MENUDET_POW: {
        result = 1;
        uint64_t base = left;
        uint64_t exponent = right;
        while (exponent) {
            if (exponent & 1) {
                if (base && result > maximum / base) return ME_MENUDET_NUMERIC_RANGE;
                result *= base;
            }
            exponent >>= 1;
            if (exponent) {
                if (base && base > maximum / base) return ME_MENUDET_NUMERIC_RANGE;
                base *= base;
            }
        }
        break;
    }
    case ME_MENUDET_AND: result = left & right; break;
    case ME_MENUDET_OR: result = left | right; break;
    case ME_MENUDET_XOR: result = left ^ right; break;
    default: return ME_MENUDET_NUMERIC_TYPE;
    }
    *out = result;
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_float_to_signed(me_dtype dtype, double value, int64_t *out) {
    int64_t minimum, maximum;
    if (!out || !menudet_signed_bounds(dtype, &minimum, &maximum)) return ME_MENUDET_NUMERIC_TYPE;
    if (!isfinite(value)) return ME_MENUDET_NUMERIC_RANGE;
    double integral = trunc(value);
    double limit = ldexp(1.0, menudet_integer_bits(dtype) - 1);
    /* An exclusive power-of-two upper bound remains exact for int64; converting
     * INT64_MAX to double would round it to the first invalid integer. */
    if (integral < -limit || integral >= limit) return ME_MENUDET_NUMERIC_RANGE;
    *out = (int64_t)integral;
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_float_to_unsigned(me_dtype dtype, double value, uint64_t *out) {
    int bits = menudet_integer_bits(dtype);
    if (!out || !bits || !menudet_unsigned(dtype)) return ME_MENUDET_NUMERIC_TYPE;
    if (!isfinite(value)) return ME_MENUDET_NUMERIC_RANGE;
    double integral = trunc(value);
    if (integral < 0 || integral >= ldexp(1.0, bits)) return ME_MENUDET_NUMERIC_RANGE;
    *out = (uint64_t)integral;
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_signed_to_signed(me_dtype dtype, int64_t value, int64_t *out) {
    int64_t minimum, maximum;
    if (!out || !menudet_signed_bounds(dtype, &minimum, &maximum)) return ME_MENUDET_NUMERIC_TYPE;
    if (value < minimum || value > maximum) return ME_MENUDET_NUMERIC_RANGE;
    *out = value;
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_unsigned_to_signed(me_dtype dtype, uint64_t value, int64_t *out) {
    int64_t minimum, maximum;
    if (!out || !menudet_signed_bounds(dtype, &minimum, &maximum)) return ME_MENUDET_NUMERIC_TYPE;
    if (value > (uint64_t)maximum) return ME_MENUDET_NUMERIC_RANGE;
    *out = (int64_t)value;
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_unsigned_to_unsigned(me_dtype dtype, uint64_t value, uint64_t *out) {
    int bits = menudet_integer_bits(dtype);
    if (!out || !bits || !menudet_unsigned(dtype)) return ME_MENUDET_NUMERIC_TYPE;
    uint64_t maximum = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    if (value > maximum) return ME_MENUDET_NUMERIC_RANGE;
    *out = value;
    return ME_MENUDET_NUMERIC_OK;
}

me_menudet_numeric_status dsl_menudet_signed_to_unsigned(me_dtype dtype, int64_t value, uint64_t *out) {
    /* Validate the destination even when value is negative, so an unsupported
     * type is consistently distinguished from a value-domain failure. */
    if (!out || !menudet_unsigned(dtype)) return ME_MENUDET_NUMERIC_TYPE;
    if (value < 0) return ME_MENUDET_NUMERIC_RANGE;
    return dsl_menudet_unsigned_to_unsigned(dtype, (uint64_t)value, out);
}

int dsl_menudet_compare_signed_unsigned(int64_t left, uint64_t right) {
    if (left < 0) return -1;
    uint64_t unsigned_left = (uint64_t)left;
    return unsigned_left < right ? -1 : unsigned_left > right ? 1 : 0;
}
