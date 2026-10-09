/* Shared test transport. Fixed-width bytes, never integer transport via double. */
#ifndef MINIEXPR_NUMPY_VECTOR_IO_H
#define MINIEXPR_NUMPY_VECTOR_IO_H
#include "miniexpr_artifact.h"
#include "yyjson.h"
#include <stdlib.h>
#include <string.h>

static const char *text(yyjson_val *object, const char *key) {
    return yyjson_get_str(yyjson_obj_get(object, key));
}

static me_dtype dtype(const char *name) {
    static const char *names[] = {"bool", "int8", "int16", "int32", "int64",
                                  "uint8", "uint16", "uint32", "uint64", "float32", "float64"};
    static const me_dtype types[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
                                    ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    for (size_t i = 0; name && i < sizeof(types) / sizeof(types[0]); i++) {
        if (!strcmp(name, names[i])) return types[i];
    }
    return ME_AUTO;
}

static size_t vector_width(me_dtype type) {
    switch (type) {
    case ME_BOOL: case ME_INT8: case ME_UINT8: return 1;
    case ME_INT16: case ME_UINT16: return 2;
    case ME_INT32: case ME_UINT32: case ME_FLOAT32: return 4;
    case ME_INT64: case ME_UINT64: case ME_FLOAT64: return 8;
    default: return 0;
    }
}

static int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int little_endian(void) {
    const uint16_t one = 1;
    return *(const unsigned char *)&one;
}

static int decode(const char *hex, unsigned char *out, size_t width, size_t count) {
    if (!hex || !width || strlen(hex) != width * count * 2) return 0;
    for (size_t i = 0; i < width * count; i++) {
        int high = nibble(hex[2 * i]), low = nibble(hex[2 * i + 1]);
        if (high < 0 || low < 0) return 0;
        size_t offset = little_endian() ? (i / width) * width + width - 1 - i % width : i;
        out[offset] = (unsigned char)(high * 16 + low);
    }
    return 1;
}

static void encode(const unsigned char *data, size_t width, size_t count, char *out) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < width * count; i++) {
        size_t offset = little_endian() ? (i / width) * width + width - 1 - i % width : i;
        out[2 * i] = digits[data[offset] >> 4];
        out[2 * i + 1] = digits[data[offset] & 15];
    }
    out[2 * width * count] = 0;
}
#endif
