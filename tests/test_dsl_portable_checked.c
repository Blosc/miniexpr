/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "dsl_portable_types.h"
#include <math.h>
#include <stdio.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int signed_boundaries(void) {
    int64_t out = 17;
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_ADD, INT64_MAX, 1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(out == 17);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SUB, INT64_MIN, 1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SUB, INT64_MAX, -1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_ADD, INT64_MIN, -1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_MUL, INT64_MIN, -1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_MUL, INT64_MAX, 2, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_MUL, INT64_MIN, 1, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT64_MIN);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_MUL, INT64_MIN, 0, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == 0);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_FLOORDIV, INT64_MIN, -1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_MOD, INT64_MIN, -1, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == 0);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_FLOORDIV, INT64_MIN, 3, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT64_C(-3074457345618258603));
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_MOD, INT64_MIN, 3, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == 1);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SHL, -1, 63, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT64_MIN);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SHL, 1, 63, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SHR, INT64_MIN, 63, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == -1);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SHR, -3, 1, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == -2);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SHR, 1, -1, &out) == ME_PORTABLE_NUMERIC_SHIFT);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_SHR, 1, 64, &out) == ME_PORTABLE_NUMERIC_SHIFT);
    CHECK(dsl_portable_signed_op(ME_INT16, ME_PORTABLE_ADD, INT16_MAX, 1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT32, ME_PORTABLE_MUL, INT32_MIN, -1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_op(ME_INT32, ME_PORTABLE_ADD, INT32_MAX, 0, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT32_MAX);
    CHECK(dsl_portable_signed_op(ME_BOOL, ME_PORTABLE_ADD, 0, 1, &out) == ME_PORTABLE_NUMERIC_TYPE);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_ADD, 0, 1, NULL) == ME_PORTABLE_NUMERIC_TYPE);
    return 0;
}

static int unsigned_boundaries(void) {
    uint64_t out = 17;
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_ADD, UINT64_MAX, 1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(out == 17);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_SUB, 0, 1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_MUL, UINT64_MAX, 2, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_MUL, UINT64_MAX, 1, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == UINT64_MAX);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_SHL, 1, 63, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == (UINT64_C(1) << 63));
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_SHL, 2, 63, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_SHR, UINT64_MAX, 64, &out) == ME_PORTABLE_NUMERIC_SHIFT);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_FLOORDIV, 0, 0, &out) == ME_PORTABLE_NUMERIC_ZERO);
    CHECK(dsl_portable_unsigned_op(ME_UINT16, ME_PORTABLE_ADD, UINT16_MAX, 1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_op(ME_UINT32, ME_PORTABLE_MUL, UINT32_MAX, 2, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_op(ME_UINT32, ME_PORTABLE_ADD, UINT32_MAX, 0, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == UINT32_MAX);
    CHECK(dsl_portable_unsigned_op(ME_INT64, ME_PORTABLE_ADD, 0, 1, &out) == ME_PORTABLE_NUMERIC_TYPE);
    return 0;
}

static int conversions(void) {
    int64_t signed_out = 17;
    uint64_t unsigned_out = 17;
    CHECK(dsl_portable_float_to_signed(ME_INT64, 0x1p63, &signed_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(signed_out == 17);
    CHECK(dsl_portable_float_to_signed(ME_INT64, -0x1p63, &signed_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(signed_out == INT64_MIN);
    CHECK(dsl_portable_float_to_signed(ME_INT64, nextafter(0x1p63, 0), &signed_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(signed_out == INT64_MAX - 1023);
    CHECK(dsl_portable_float_to_signed(ME_INT64, nextafter(-0x1p63, -INFINITY), &signed_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_float_to_unsigned(ME_UINT64, 0x1p64, &unsigned_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(unsigned_out == 17);
    CHECK(dsl_portable_float_to_unsigned(ME_UINT64, nextafter(0x1p64, 0), &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == UINT64_MAX - 2047);
    CHECK(dsl_portable_float_to_unsigned(ME_UINT64, -0.9, &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == 0);
    CHECK(dsl_portable_float_to_unsigned(ME_UINT64, -1, &unsigned_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_float_to_signed(ME_INT8, -128.9, &signed_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(signed_out == -128);
    CHECK(dsl_portable_float_to_signed(ME_INT8, 127.9, &signed_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(signed_out == 127);
    CHECK(dsl_portable_float_to_signed(ME_INT8, 128, &signed_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_float_to_signed(ME_INT8, -129, &signed_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_float_to_unsigned(ME_UINT8, 255.9, &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == 255);
    CHECK(dsl_portable_float_to_unsigned(ME_UINT8, 256, &unsigned_out) == ME_PORTABLE_NUMERIC_RANGE);
    const double invalid[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        CHECK(dsl_portable_float_to_signed(ME_INT64, invalid[i], &signed_out) == ME_PORTABLE_NUMERIC_RANGE);
        CHECK(dsl_portable_float_to_unsigned(ME_UINT64, invalid[i], &unsigned_out) == ME_PORTABLE_NUMERIC_RANGE);
    }
    CHECK(dsl_portable_float_to_signed(ME_FLOAT64, 1, &signed_out) == ME_PORTABLE_NUMERIC_TYPE);
    CHECK(dsl_portable_float_to_unsigned(ME_INT64, 1, &unsigned_out) == ME_PORTABLE_NUMERIC_TYPE);
    return 0;
}

/* Exhaust small integer domains using wide reference calculations. This
 * exercises boundary sign combinations without a Python parametrization grid. */
static int exhaustive_small(void) {
    for (int a = -128; a <= 127; a++) {
        for (int b = -128; b <= 127; b++) {
            for (int op = ME_PORTABLE_ADD; op <= ME_PORTABLE_MOD; op++) {
                int reference = 0;
                me_portable_numeric_status expected = ME_PORTABLE_NUMERIC_OK;
                switch (op) {
                case ME_PORTABLE_ADD: reference = a + b; break;
                case ME_PORTABLE_SUB: reference = a - b; break;
                case ME_PORTABLE_MUL: reference = a * b; break;
                default:
                    if (!b) expected = ME_PORTABLE_NUMERIC_ZERO;
                    else {
                        int quotient = a / b;
                        if (a % b && ((a < 0) != (b < 0))) quotient--;
                        reference = op == ME_PORTABLE_FLOORDIV ? quotient : a - quotient * b;
                    }
                    break;
                }
                if (expected == ME_PORTABLE_NUMERIC_OK && (reference < -128 || reference > 127)) {
                    expected = ME_PORTABLE_NUMERIC_RANGE;
                }
                int64_t out = 999;
                CHECK(dsl_portable_signed_op(ME_INT8, (me_portable_integer_op)op, a, b, &out) == expected);
                CHECK(out == (expected == ME_PORTABLE_NUMERIC_OK ? reference : 999));
            }
        }
    }
    for (unsigned a = 0; a <= 255; a++) {
        for (unsigned b = 0; b <= 255; b++) {
            for (int op = ME_PORTABLE_ADD; op <= ME_PORTABLE_MOD; op++) {
                int reference = 0;
                me_portable_numeric_status expected = ME_PORTABLE_NUMERIC_OK;
                switch (op) {
                case ME_PORTABLE_ADD: reference = (int)a + (int)b; break;
                case ME_PORTABLE_SUB: reference = (int)a - (int)b; break;
                case ME_PORTABLE_MUL: reference = (int)a * (int)b; break;
                default:
                    if (!b) expected = ME_PORTABLE_NUMERIC_ZERO;
                    else reference = op == ME_PORTABLE_FLOORDIV ? (int)(a / b) : (int)(a % b);
                    break;
                }
                if (expected == ME_PORTABLE_NUMERIC_OK && (reference < 0 || reference > 255)) {
                    expected = ME_PORTABLE_NUMERIC_RANGE;
                }
                uint64_t out = 999;
                CHECK(dsl_portable_unsigned_op(ME_UINT8, (me_portable_integer_op)op, a, b, &out) == expected);
                CHECK(out == (expected == ME_PORTABLE_NUMERIC_OK ? (uint64_t)reference : 999));
            }
        }
    }
    return 0;
}

static int powers_bits_and_narrowing(void) {
    int64_t out = 17;
    uint64_t unsigned_out = 17;
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_POW, -2, 63, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT64_MIN);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_POW, 2, 63, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(out == INT64_MIN);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_POW, 0, 0, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == 1);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_POW, -1, INT64_MAX, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == -1);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_POW, 2, -1, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_POW, 2, 63, &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == (UINT64_C(1) << 63));
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_POW, 2, 64, &unsigned_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_POW, 1, UINT64_MAX, &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == 1);
    CHECK(dsl_portable_signed_op(ME_INT64, ME_PORTABLE_XOR, INT64_MIN, -1, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT64_MAX);
    CHECK(dsl_portable_signed_op(ME_INT8, ME_PORTABLE_AND, -128, -1, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == -128);
    CHECK(dsl_portable_signed_op(ME_INT8, ME_PORTABLE_OR, -128, 127, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == -1);
    CHECK(dsl_portable_unsigned_op(ME_UINT64, ME_PORTABLE_XOR, UINT64_MAX, 1, &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == UINT64_MAX - 1);
    CHECK(dsl_portable_signed_to_signed(ME_INT8, 128, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_signed_to_signed(ME_INT64, INT64_MIN, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT64_MIN);
    CHECK(dsl_portable_unsigned_to_signed(ME_INT64, UINT64_MAX, &out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(out == INT64_MIN);
    CHECK(dsl_portable_unsigned_to_signed(ME_INT64, INT64_MAX, &out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(out == INT64_MAX);
    CHECK(dsl_portable_unsigned_to_unsigned(ME_UINT8, 256, &unsigned_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(dsl_portable_unsigned_to_unsigned(ME_UINT64, UINT64_MAX, &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == UINT64_MAX);
    CHECK(dsl_portable_signed_to_unsigned(ME_UINT64, -1, &unsigned_out) == ME_PORTABLE_NUMERIC_RANGE);
    CHECK(unsigned_out == UINT64_MAX);
    CHECK(dsl_portable_signed_to_unsigned(ME_UINT64, INT64_MAX, &unsigned_out) == ME_PORTABLE_NUMERIC_OK);
    CHECK(unsigned_out == INT64_MAX);
    CHECK(dsl_portable_compare_signed_unsigned(-1, 0) == -1);
    CHECK(dsl_portable_compare_signed_unsigned(INT64_MAX, UINT64_MAX) == -1);
    CHECK(dsl_portable_compare_signed_unsigned(INT64_MAX, INT64_MAX) == 0);
    CHECK(dsl_portable_compare_signed_unsigned(INT64_MAX, INT64_MAX - 1) == 1);
    CHECK(dsl_portable_compare_signed_unsigned(INT64_C(9007199254740993), UINT64_C(9007199254740992)) == 1);
    return 0;
}

static int shifts_and_bits_small(void) {
    for (int a = -128; a <= 127; a++) {
        for (int count = -1; count <= 8; count++) {
            int64_t out = 999;
            me_portable_numeric_status expected = ME_PORTABLE_NUMERIC_SHIFT;
            int reference = 999;
            if (count >= 0 && count < 8) {
                reference = a * (1 << count);
                expected = reference < -128 || reference > 127 ? ME_PORTABLE_NUMERIC_RANGE : ME_PORTABLE_NUMERIC_OK;
            }
            CHECK(dsl_portable_signed_op(ME_INT8, ME_PORTABLE_SHL, a, count, &out) == expected);
            CHECK(out == (expected == ME_PORTABLE_NUMERIC_OK ? reference : 999));
            out = 999;
            if (count >= 0 && count < 8) {
                int divisor = 1 << count;
                reference = a / divisor;
                if (a < 0 && a % divisor) reference--;
                expected = ME_PORTABLE_NUMERIC_OK;
            }
            CHECK(dsl_portable_signed_op(ME_INT8, ME_PORTABLE_SHR, a, count, &out) == expected);
            CHECK(out == (expected == ME_PORTABLE_NUMERIC_OK ? reference : 999));
        }
        for (int b = -128; b <= 127; b++) {
            unsigned ua = (unsigned)a & 255;
            unsigned ub = (unsigned)b & 255;
            for (int op = ME_PORTABLE_AND; op <= ME_PORTABLE_XOR; op++) {
                unsigned raw = op == ME_PORTABLE_AND ? ua & ub : op == ME_PORTABLE_OR ? ua | ub : ua ^ ub;
                int reference = raw >= 128 ? (int)raw - 256 : (int)raw;
                int64_t out = 999;
                CHECK(dsl_portable_signed_op(ME_INT8, (me_portable_integer_op)op, a, b, &out) == ME_PORTABLE_NUMERIC_OK);
                CHECK(out == reference);
            }
        }
    }
    for (unsigned a = 0; a <= 255; a++) {
        for (unsigned count = 0; count <= 8; count++) {
            uint64_t out = 999;
            unsigned reference = a << count;
            me_portable_numeric_status expected = count == 8 ? ME_PORTABLE_NUMERIC_SHIFT :
                reference > 255 ? ME_PORTABLE_NUMERIC_RANGE : ME_PORTABLE_NUMERIC_OK;
            CHECK(dsl_portable_unsigned_op(ME_UINT8, ME_PORTABLE_SHL, a, count, &out) == expected);
            CHECK(out == (expected == ME_PORTABLE_NUMERIC_OK ? reference : 999));
        }
    }
    return 0;
}

int main(void) {
    if (signed_boundaries() || unsigned_boundaries() || conversions() || exhaustive_small() ||
        powers_bits_and_narrowing() || shifts_and_bits_small()) return 1;
    puts("portable checked integer foundation passed");
    return 0;
}
