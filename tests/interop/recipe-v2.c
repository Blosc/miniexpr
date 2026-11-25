/* JSON-free C peer for the b2o v2 cross-language fixtures.
 * This deliberately small test codec is NOT a general persistence API. Its
 * interoperable subset is documented in doc/recipe-interop.md. */
#include "miniexpr_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LIMIT 65536
#define CHECK(test) do { if (!(test)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #test); return 0; \
} } while (0)

typedef struct {
    const unsigned char *start, *data, *end;
    size_t count;
    unsigned tag;
} value;

/* Bounded MessagePack reader. No executable objects, JSON, allocation based on
 * wire lengths, host-endian numbers, or file/reference resolution. */
static int take(const unsigned char **cursor, const unsigned char *end, value *out, int depth) {
    CHECK(depth <= 32 && *cursor < end);
    const unsigned char *p = *cursor;
    out->start = p;
    unsigned tag = *p++;
    size_t n = 0;
    unsigned bytes = 0;
    out->tag = tag;
    if (tag <= 0x7f) n = tag;
    else if ((tag & 0xe0) == 0xa0) {
        n = tag & 31;
        out->tag = 0xa0;
    }
    else if ((tag & 0xf0) == 0x80 || (tag & 0xf0) == 0x90) {
        n = tag & 15;
        out->tag = tag & 0xf0;
    }
    else {
        switch (tag) {
        case 0xc0: case 0xc2: case 0xc3: break;
        case 0xcc: case 0xd0: bytes = 1; break;
        case 0xcd: case 0xd1: bytes = 2; break;
        case 0xce: case 0xd2: bytes = 4; break;
        case 0xcf: case 0xd3: bytes = 8; break;
        case 0xd9: case 0xc7: bytes = 1; break;
        case 0xda: case 0xdc: case 0xde: case 0xc8: bytes = 2; break;
        default: CHECK(0);
        }
        CHECK((size_t)(end - p) >= bytes);
        for (unsigned i = 0; i < bytes; i++) {
            CHECK(n <= (SIZE_MAX >> 8));
            n = (n << 8) | *p++;
        }
        if (tag == 0xd9 || tag == 0xda) out->tag = 0xa0;
        if (tag == 0xdc) out->tag = 0x90;
        if (tag == 0xde) out->tag = 0x80;
        if (tag == 0xc7 || tag == 0xc8) {
            CHECK(p < end && *p++ == 43);
            out->tag = 43;
        }
    }
    out->data = p;
    out->count = n;
    if (out->tag == 0xa0 || out->tag == 43) {
        CHECK(n <= (size_t)(end - p));
        p += n;
    }
    else if (out->tag == 0x80 || out->tag == 0x90) {
        CHECK(n <= 128);
        size_t children = out->tag == 0x80 ? 2 * n : n;
        for (size_t i = 0; i < children; i++) {
            value child;
            CHECK(take(&p, end, &child, depth + 1));
        }
    }
    out->end = p;
    *cursor = p;
    return 1;
}

static int text(value v, const char *s) {
    return v.tag == 0xa0 && v.count == strlen(s) && !memcmp(v.data, s, v.count);
}

static int field(value map, const char *name, value *out) {
    CHECK(map.tag == 0x80);
    const unsigned char *p = map.data;
    int found = 0;
    for (size_t i = 0; i < map.count; i++) {
        value key, item;
        CHECK(take(&p, map.end, &key, 0) && key.tag == 0xa0);
        CHECK(take(&p, map.end, &item, 0));
        if (text(key, name)) {
            CHECK(!found);
            *out = item;
            found = 1;
        }
    }
    return found;
}

static int string(value v, char *out, size_t capacity) {
    CHECK(v.tag == 0xa0 && v.count < capacity && !memchr(v.data, 0, v.count));
    memcpy(out, v.data, v.count);
    out[v.count] = 0;
    return 1;
}

static int number(value v, size_t expected) {
    return v.tag <= 0x7f && v.count == expected;
}

/* Four-lane elementwise fixture domain, two independent logical partitions. */
static int geometry(value record) {
    const char *names[] = {"shape", "partitions"};
    for (int i = 0; i < 2; i++) {
        value a, size;
        CHECK(field(record, names[i], &a) && a.tag == 0x90 && a.count == 1);
        const unsigned char *p = a.data;
        CHECK(take(&p, a.end, &size, 0) && number(size, i ? 2 : 4));
    }
    return 1;
}

static int evaluate(value record, double out[4], int depth) {
    CHECK(depth <= 8);
    if (record.tag == 43) {
        const unsigned char *p = record.data;
        value nested;
        CHECK(take(&p, record.end, &nested, 0) && p == record.end);
        return evaluate(nested, out, depth + 1);
    }
    value item, recipe, names, constants, operands;
    CHECK(field(record, "kind", &item) && text(item, "menudet"));
    CHECK(field(record, "version", &item) && number(item, 2));
    CHECK(geometry(record));
    CHECK(field(record, "recipe", &recipe));
    CHECK(field(recipe, "version", &item) && number(item, 1));
    CHECK(field(recipe, "cardinality", &item) && text(item, "elementwise"));
    CHECK(field(recipe, "casting", &item) && text(item, "unsafe"));
    CHECK(field(recipe, "ndim", &item) && (number(item, 0) || number(item, 1)));
    int ndim = (int)item.count;
    char source[4096], version[16];
    CHECK(field(recipe, "source", &item) && string(item, source, sizeof(source)));
    CHECK(field(recipe, "language_version", &item) && string(item, version, sizeof(version)));
    CHECK(field(recipe, "inputs", &names) && names.tag == 0x90 && names.count <= 4);
    CHECK(field(recipe, "constants", &constants) && constants.tag == 0x90 && constants.count <= 4);
    CHECK(field(record, "operands", &operands) && operands.count == names.count);
    me_input_spec slots[4];
    me_capture captures[4];
    double input[4][4], capture_values[4];
    char input_names[4][64], capture_names[4][64];
    const unsigned char *p = names.data;
    for (size_t i = 0; i < names.count; i++) {
        CHECK(take(&p, names.end, &item, 0) && string(item, input_names[i], sizeof(input_names[i])));
        CHECK(field(operands, input_names[i], &item));
        CHECK(evaluate(item, input[i], depth + 1));
        slots[i] = (me_input_spec){input_names[i], ME_FLOAT64, sizeof(double)};
    }
    p = constants.data;
    for (size_t i = 0; i < constants.count; i++) {
        value constant, bits, category;
        char hex[17];
        CHECK(take(&p, constants.end, &constant, 0));
        CHECK(field(constant, "name", &item) && string(item, capture_names[i], sizeof(capture_names[i])));
        CHECK(field(constant, "dtype", &item) && text(item, "float64"));
        CHECK(field(constant, "encoding", &item) && text(item, "ieee754-hex"));
        CHECK(field(constant, "value", &bits) && string(bits, hex, sizeof(hex)) && strlen(hex) == 16);
        uint64_t raw = 0;
        for (size_t j = 0; j < 16; j++) {
            const char *digit = strchr("0123456789abcdef", hex[j]);
            CHECK(digit != NULL);
            raw = (raw << 4) | (uint64_t)(digit - "0123456789abcdef");
        }
        memcpy(&capture_values[i], &raw, sizeof(raw));
        CHECK(field(constant, "category", &category));
        CHECK(text(category, "weak") || text(category, "typed_scalar"));
        captures[i] = (me_capture){{capture_names[i], ME_FLOAT64, sizeof(double)},
            &capture_values[i], text(category, "weak") ? ME_CAPTURE_WEAK : ME_CAPTURE_TYPED};
    }
    me_signature signature = ME_SIGNATURE_INITIALIZER;
    signature.inputs = slots;
    signature.ninputs = names.count;
    signature.captures = captures;
    signature.ncaptures = constants.count;
    signature.context_ndim = ndim;
    signature.cardinality = ME_ELEMENTWISE;
    me_kernel *kernel = NULL;
    me_error error;
    int rc = me_prepare_menudet(version, source, &signature, NULL, &kernel, &error);
    if (rc) fprintf(stderr, "%s\n", error.message);
    CHECK(rc == ME_SUCCESS);
    me_kernel_info info = ME_KERNEL_INFO_INITIALIZER;
    rc = me_describe(kernel, &info);
    if (rc || info.output_dtype != ME_FLOAT64 || info.context_ndim > ndim) {
        me_kernel_free(kernel);
        CHECK(0);
    }
    int64_t shape = 4, extent = 2;
    for (int64_t origin = 0; origin < 4 && rc == ME_SUCCESS; origin += 2) {
        me_buffer buffers[4];
        for (size_t i = 0; i < names.count; i++) buffers[i] = (me_buffer){input[i] + origin, 2 * sizeof(double)};
        me_block block = ME_BLOCK_INITIALIZER;
        block.nitems = 2;
        block.ndim = info.context_ndim;
        block.logical_shape = info.context_ndim ? &shape : NULL;
        block.origin = info.context_ndim ? &origin : NULL;
        block.extent = info.context_ndim ? &extent : NULL;
        rc = me_eval_block(kernel, buffers, names.count,
            (me_mut_buffer){out + origin, 2 * sizeof(double)}, &block, NULL, &error);
    }
    me_kernel_free(kernel);
    CHECK(rc == ME_SUCCESS);
    return 1;
}

typedef struct { unsigned char data[LIMIT]; size_t size; } writer;
static void byte(writer *w, unsigned char b) {
    if (w->size >= LIMIT) abort();
    w->data[w->size++] = b;
}
static void str(writer *w, const char *s) {
    size_t n = strlen(s);
    if (n < 32) byte(w, (unsigned char)(0xa0 | n));
    else {
        if (n > 255) abort();
        byte(w, 0xd9);
        byte(w, (unsigned char)n);
    }
    for (size_t i = 0; i < n; i++) byte(w, (unsigned char)s[i]);
}
static void pair(writer *w, const char *key, const char *v) {
    str(w, key);
    str(w, v);
}

static void wrap_ext43(writer *w, const writer *child) {
    byte(w, 0xc8);
    byte(w, (unsigned char)(child->size >> 8));
    byte(w, (unsigned char)child->size);
    byte(w, 43);
    for (size_t i = 0; i < child->size; i++) {
        byte(w, child->data[i]);
    }
}

static void encode(writer *w, int nested) {
    byte(w, 0x86);
    pair(w, "kind", "menudet");
    str(w, "version"); byte(w, 2);
    str(w, "shape"); byte(w, 0x91); byte(w, 4);
    str(w, "partitions"); byte(w, 0x91); byte(w, 2);
    str(w, "recipe"); byte(w, 0x8a);
    str(w, "version"); byte(w, 1);
    pair(w, "language_version", "0.2");
    pair(w, "source", nested ?
        "def twice(x: float64) -> float64:\n    return 2 * x\n" :
        "def ramp(start: float64) -> float64:\n    return start + _flat_idx\n");
    pair(w, "entry_point", nested ? "twice" : "ramp");
    str(w, "inputs"); byte(w, nested ? 0x91 : 0x90);
    if (nested) str(w, "x");
    str(w, "constants"); byte(w, nested ? 0x90 : 0x91);
    if (!nested) {
        byte(w, 0x85);
        pair(w, "name", "start");
        pair(w, "dtype", "float64");
        pair(w, "encoding", "ieee754-hex");
        pair(w, "value", "3ff0000000000000");
        pair(w, "category", "typed_scalar");
    }
    str(w, "output"); byte(w, 0xc0);
    pair(w, "cardinality", "elementwise");
    str(w, "ndim"); byte(w, nested ? 0 : 1);
    pair(w, "casting", "unsafe");
    str(w, "operands"); byte(w, nested ? 0x81 : 0x80);
    if (nested) {
        str(w, "x");
        if (nested == 2) {
            writer child = {{0}, 0};
            encode(&child, 0);
            wrap_ext43(w, &child);
        }
        else encode(w, 0);
    }
}

int main(int argc, char **argv) {
    if (argc == 1) {
        for (int mode = 0; mode < 4; mode++) {
            writer w = {{0}, 0};
            if (mode == 3) {
                writer child = {{0}, 0};
                encode(&child, 1);
                wrap_ext43(&w, &child);
            }
            else encode(&w, mode);
            const unsigned char *p = w.data;
            value record;
            double out[4];
            if (!take(&p, w.data + w.size, &record, 0) || p != w.data + w.size ||
                !evaluate(record, out, 0)) return 1;
            for (int i = 0; i < 4; i++) if (out[i] != (i + 1) * (mode ? 2 : 1)) return 1;
        }
        return 0;
    }
    if (argc < 3) return 2;
    if (!strcmp(argv[1], "write")) {
        writer w = {{0}, 0};
        int mode = argc > 3 ? atoi(argv[3]) : 0;
        if (mode == 2) {
            writer child = {{0}, 0};
            encode(&child, 1);
            wrap_ext43(&w, &child);
        }
        else encode(&w, mode);
        FILE *file = fopen(argv[2], "wb");
        if (!file) return 1;
        int ok = fwrite(w.data, 1, w.size, file) == w.size;
        if (fclose(file)) ok = 0;
        return ok ? 0 : 1;
    }
    if (strcmp(argv[1], "read")) return 2;
    unsigned char bytes[LIMIT + 1];
    FILE *file = fopen(argv[2], "rb");
    if (!file) return 1;
    size_t n = fread(bytes, 1, sizeof(bytes), file);
    int ok = !ferror(file) && n <= LIMIT;
    fclose(file);
    if (!ok) return 1;
    const unsigned char *p = bytes;
    value record;
    double out[4];
    if (!take(&p, bytes + n, &record, 0) || p != bytes + n || !evaluate(record, out, 0)) return 1;
    for (int i = 0; i < 4; i++) printf("%.17g\n", out[i]);
    return 0;
}
