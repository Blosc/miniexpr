/* Native logical traversal. Storage/decompression is a host concern, grouping is
 * not: each reduction visits C logical coordinates in a fixed serial order. */
#include "miniexpr_artifact.h"
#include "dsl_graph_internal.h"
#include "dsl_menudet_fp.h"
#include "dsl_menudet_reduce.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Enable SSE2 baseline kernels and runtime-dispatched AVX2 kernels on x86-64. */
#if (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)) && !defined(_M_ARM64EC)
#define ME_PAIRWISE_X86_64 1
#include <emmintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#define ME_PAIRWISE_HAVE_AVX2_DISPATCH 1
#define ME_PAIRWISE_AVX2_TARGET __attribute__((target("avx2")))
#define ME_PAIRWISE_NOINLINE __attribute__((noinline))
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif
#endif

static int a_error(me_artifact_error *e, int rc, const char *message) {
    if (e) { memset(e, 0, sizeof(*e)); snprintf(e->message, sizeof(e->message), "%s", message); }
    return rc;
}
static size_t a_width(me_dtype d) {
    switch (d) {
        case ME_BOOL: case ME_INT8: case ME_UINT8: return 1;
        case ME_INT16: case ME_UINT16: return 2;
        case ME_INT32: case ME_UINT32: case ME_FLOAT32: return 4;
        case ME_INT64: case ME_UINT64: case ME_FLOAT64: return 8;
        case ME_COMPLEX64: return 8;
        case ME_COMPLEX128: return 16;
        default: return 0;
    }
}
static bool a_complex(me_dtype d) { return d == ME_COMPLEX64 || d == ME_COMPLEX128; }
static me_dtype a_component(me_dtype d) {
    return d == ME_COMPLEX64 ? ME_FLOAT32 : d == ME_COMPLEX128 ? ME_FLOAT64 : ME_AUTO;
}
static bool a_unsigned(me_dtype d) {
    return d == ME_BOOL || d == ME_UINT8 || d == ME_UINT16 || d == ME_UINT32 || d == ME_UINT64;
}
static bool a_float(me_dtype d) { return d == ME_FLOAT32 || d == ME_FLOAT64; }
static bool a_product(int rank, const int64_t *shape, size_t *n) {
    if (rank < 0 || rank > ME_ARRAY_MAX_RANK || (rank && !shape)) return false;
    *n = 1;
    for (int i = 0; i < rank; i++) {
        if (shape[i] < 0 || (shape[i] && *n > SIZE_MAX / (uint64_t)shape[i])) return false;
        *n *= (size_t)shape[i];
    }
    return true;
}
static bool a_add(size_t base, int64_t stride, uint64_t count, size_t *out) {
    uint64_t magnitude = stride < 0 ? (uint64_t)(-(stride + 1)) + 1 : (uint64_t)stride;
    if (magnitude && count > SIZE_MAX / magnitude) return false;
    size_t bytes = (size_t)(magnitude * count);
    if (stride < 0) { if (bytes > base) return false; *out = base - bytes; }
    else { if (base > SIZE_MAX - bytes) return false; *out = base + bytes; }
    return true;
}
static bool a_valid(const me_array_view *v) {
    size_t n, width = a_width(v->dtype), low = v->offset, high = v->offset;
    if (!width || !a_product(v->rank, v->shape, &n) || v->byte_order > 2 || v->offset > v->capacity) return false;
    if (v->capacity && (!v->base || (uintptr_t)v->base > UINTPTR_MAX - v->capacity)) return false;
    if (!n) return true;
    for (int i = 0; i < v->rank; i++) {
        if (!a_add(v->strides[i] < 0 ? low : high, v->strides[i], (uint64_t)v->shape[i] - 1,
                   v->strides[i] < 0 ? &low : &high)) return false;
    }
    return v->base && high <= v->capacity && width <= v->capacity - high &&
           (uintptr_t)v->base <= UINTPTR_MAX - v->capacity;
}
static bool a_overlap(const void *a, size_t na, const void *b, size_t nb) {
    return na && nb && (uintptr_t)a < (uintptr_t)b + nb && (uintptr_t)b < (uintptr_t)a + na;
}
static bool a_native(const me_array_view *v) {
    uint16_t x = 1;
    return !v->byte_order || v->byte_order == (*(uint8_t *)&x ? 1u : 2u);
}
static bool a_contiguous(const me_array_view *v) {
    size_t stride = a_width(v->dtype);
    for (int i = v->rank - 1; i >= 0; i--) {
        if (v->shape[i] > 1 && (v->strides[i] < 0 || (uint64_t)v->strides[i] != stride)) return false;
        if (v->shape[i] && stride > INT64_MAX / (uint64_t)v->shape[i]) return false;
        stride *= (size_t)v->shape[i];
    }
    return true;
}
static me_array_view a_output_view(const me_array_output *output) {
    me_array_view view = {0};
    view.dtype = output->dtype; view.base = output->base;
    view.capacity = output->capacity; view.offset = output->offset;
    view.rank = output->rank;
    memcpy(view.shape, output->shape, sizeof(view.shape));
    memcpy(view.strides, output->strides, sizeof(view.strides));
    return view;
}
me_artifact_status dsl_array_output_validate(const me_array_output *output,
    int rank, const int64_t *shape, me_dtype dtype, me_artifact_error *error) {
    if (!output || output->struct_size != sizeof(*output) ||
        output->version != ME_ARRAY_OUTPUT_VERSION || output->rank != rank ||
        rank < 0 || rank > ME_ARRAY_MAX_RANK || (rank && !shape) || output->dtype != dtype)
        return a_error(error, ME_ARTIFACT_ERR_BINDING, "invalid output descriptor or signature");
    me_array_view view = a_output_view(output);
    size_t count, width = a_width(dtype);
    if (!a_valid(&view) || !a_product(rank, shape, &count) ||
        (rank && memcmp(shape, view.shape, (size_t)rank * sizeof(*shape))))
        return a_error(error, ME_ARTIFACT_ERR_BINDING, "invalid output bounds or shape");
    if (!count) return ME_ARTIFACT_SUCCESS;
    if (((uintptr_t)view.base + view.offset) % width)
        return a_error(error, ME_ARTIFACT_ERR_BINDING, "unaligned output origin");
    /* Sort nontrivial axes by absolute stride. Each slower axis must clear the
     * entire faster footprint. Conservative by design: no overlap solver. */
    uint64_t strides[ME_ARRAY_MAX_RANK], extents[ME_ARRAY_MAX_RANK];
    int axes = 0;
    for (int i = 0; i < rank; i++) if (shape[i] > 1) {
        int64_t stride = view.strides[i];
        uint64_t magnitude = stride < 0 ? (uint64_t)(-(stride + 1)) + 1 : (uint64_t)stride;
        if (!magnitude || magnitude % width)
            return a_error(error, ME_ARTIFACT_ERR_BINDING, "overlapping or unaligned output strides");
        int j = axes++;
        while (j && strides[j - 1] > magnitude) {
            strides[j] = strides[j - 1]; extents[j] = extents[j - 1]; j--;
        }
        strides[j] = magnitude; extents[j] = (uint64_t)shape[i];
    }
    size_t span = width;
    for (int i = 0; i < axes; i++) {
        if (strides[i] < span || strides[i] > SIZE_MAX ||
            extents[i] - 1 > (SIZE_MAX - span) / strides[i])
            return a_error(error, ME_ARTIFACT_ERR_BINDING, "output layout is not provably disjoint");
        span += (size_t)((extents[i] - 1) * strides[i]);
    }
    return ME_ARTIFACT_SUCCESS;
}
static bool a_broadcast(const me_array_view *v, int rank, const int64_t *shape) {
    if (!a_valid(v) || v->rank > rank) return false;
    for (int i = 0; i < v->rank; i++) {
        if (v->shape[i] != 1 && v->shape[i] != shape[rank - v->rank + i]) return false;
    }
    return true;
}
static size_t a_offset(const me_array_view *v, int rank, const int64_t *coordinates) {
    size_t offset = v->offset;
    for (int i = 0; i < v->rank; i++) {
        if (v->shape[i] != 1) a_add(offset, v->strides[i], (uint64_t)coordinates[rank - v->rank + i], &offset);
    }
    return offset;
}
static void a_coordinates(int rank, const int64_t *shape, const bool *axes,
    bool reduction, size_t group, size_t lane, int64_t *coordinates) {
    for (int i = rank - 1; i >= 0; i--) {
        size_t *index = reduction && !axes[i] ? &group : &lane;
        coordinates[i] = (int64_t)(*index % (size_t)shape[i]);
        *index /= (size_t)shape[i];
    }
}
static size_t a_source_index(const me_array_view *v, int rank, const int64_t *coordinates) {
    size_t index = 0;
    for (int i = 0; i < v->rank; i++) {
        index *= (size_t)v->shape[i];
        if (v->shape[i] != 1) index += (size_t)coordinates[rank - v->rank + i];
    }
    return index;
}
static bool a_reader_valid(const me_array_view *v, int rank, const int64_t *shape) {
    size_t count;
    if (!a_width(v->dtype) || !a_product(v->rank, v->shape, &count) ||
        v->rank > rank || v->base || v->capacity || v->offset || v->byte_order) return false;
    for (int i = 0; i < v->rank; i++) {
        if (v->shape[i] != 1 && v->shape[i] != shape[rank - v->rank + i]) return false;
    }
    return true;
}
static void a_load(const me_array_view *v, size_t offset, void *out) {
    size_t width = a_width(v->dtype);
    const uint8_t *p = (const uint8_t *)v->base + offset;
    if (a_native(v)) memcpy(out, p, width);
    else for (size_t i = 0; i < width; i++) ((uint8_t *)out)[i] = p[width - i - 1];
}
static bool a_axes(int rank, const me_array_options *o, bool *axes) {
    memset(axes, 0, ME_ARRAY_MAX_RANK * sizeof(bool));
    if (!o || o->version != ME_ARTIFACT_ARRAY_VERSION || o->reduction < ME_ARRAY_NONE || o->reduction > ME_ARRAY_SUM_PAIRWISE ||
        o->naxes < -1 || o->naxes > rank) return false;
    if (o->reduction == ME_ARRAY_NONE) return o->naxes == 0 && !o->initial && o->accumulator == ME_AUTO && !o->where;
    if (o->initial && (o->reduction == ME_ARRAY_ANY || o->reduction == ME_ARRAY_ALL)) return false;
    if (o->naxes == -1) { for (int i = 0; i < rank; i++) axes[i] = true; }
    else for (int i = 0; i < o->naxes; i++) {
        int axis = o->axes[i] < 0 ? o->axes[i] + rank : o->axes[i];
        if (axis < 0 || axis >= rank || axes[axis]) return false;
        axes[axis] = true;
    }
    return true;
}
static me_dtype a_dtype(me_dtype source, const me_array_options *o) {
    if (o->reduction == ME_ARRAY_ANY || o->reduction == ME_ARRAY_ALL) return o->accumulator == ME_AUTO ? ME_BOOL : ME_AUTO;
    if (o->accumulator != ME_AUTO) {
        bool compatible = a_float(source) ? a_float(o->accumulator) :
                          a_complex(source) ? (a_float(o->accumulator) || a_complex(o->accumulator)) :
                          true;
        return a_width(o->accumulator) && compatible ? o->accumulator : ME_AUTO;
    }
    if ((o->reduction == ME_ARRAY_SUM || o->reduction == ME_ARRAY_SUM_PAIRWISE || o->reduction == ME_ARRAY_PROD) &&
        !a_float(source) && !a_complex(source)) {
        return a_unsigned(source) && source != ME_BOOL ? ME_UINT64 : ME_INT64;
    }
    return source;
}
me_artifact_status me_array_result_shape(const me_artifact *a, int rank, const int64_t *shape,
    const me_array_options *o, int *out_rank, int64_t *out_shape, me_dtype *dtype, me_artifact_error *error) {
    if (error) memset(error,0,sizeof(*error));
    bool axes[ME_ARRAY_MAX_RANK]; size_t n;
    if (!a || !out_rank || !out_shape || !dtype || !a_product(rank, shape, &n) || !a_axes(rank, o, axes)) return a_error(error, ME_ARTIFACT_ERR_BINDING, "invalid logical shape, axes or options");
    if (strcmp(me_artifact_schema_version(a), "0.1") || me_artifact_context_ndim(a) ||
        me_artifact_result_cardinality(a) != ME_ARTIFACT_ELEMENTWISE || !a_width(me_artifact_output_dtype(a))) return a_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "logical arrays require numeric elementwise rank-zero Menudet 0.1 beta");
    *dtype = a_dtype(me_artifact_output_dtype(a), o);
    if (o->reduction == ME_ARRAY_SUM_PAIRWISE &&
        (!a_float(*dtype) || *dtype != me_artifact_output_dtype(a)))
        return a_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "pairwise sum requires matching floating map and accumulator dtypes");
    if (*dtype == ME_AUTO) return a_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "unsupported accumulator signature");
    *out_rank = 0;
    for (int i = 0; i < rank; i++) {
        if (!axes[i] || o->keepdims) out_shape[(*out_rank)++] = axes[i] ? 1 : shape[i];
    }
    return ME_ARTIFACT_SUCCESS;
}
static uint64_t a_integer(me_dtype dtype, const void *p) {
    switch (dtype) {
        case ME_BOOL: case ME_UINT8: { uint8_t x; memcpy(&x,p,1); return x; }
        case ME_INT8: { int8_t x; memcpy(&x,p,1); return (uint64_t)x; }
        case ME_UINT16: { uint16_t x; memcpy(&x,p,2); return x; }
        case ME_INT16: { int16_t x; memcpy(&x,p,2); return (uint64_t)x; }
        case ME_UINT32: { uint32_t x; memcpy(&x,p,4); return x; }
        case ME_INT32: { int32_t x; memcpy(&x,p,4); return (uint64_t)x; }
        default: { uint64_t x; memcpy(&x,p,8); return x; }
    }
}
static bool a_extrema_valid(const me_array_options *o, size_t red_count, size_t out_count, bool mask) {
    return !((o->reduction == ME_ARRAY_MIN || o->reduction == ME_ARRAY_MAX) &&
        !o->initial && (!red_count || mask) && out_count);
}
me_artifact_status dsl_array_validate_domain(const me_artifact *a, int rank,
    const int64_t *shape, const me_array_options *o, bool participating, me_artifact_error *error) {
    int out_rank; int64_t out_shape[ME_ARRAY_MAX_RANK]; me_dtype dtype;
    int rc = me_array_result_shape(a, rank, shape, o, &out_rank, out_shape, &dtype, error);
    if (rc) return rc;
    bool axes[ME_ARRAY_MAX_RANK]; a_axes(rank, o, axes);
    size_t out_count, red_count = 1;
    if (!a_product(out_rank, out_shape, &out_count)) return a_error(error, ME_ARTIFACT_ERR_BINDING, "output extent overflow");
    for (int i = 0; i < rank; i++) if (axes[i]) {
        if (shape[i] && red_count > SIZE_MAX / (uint64_t)shape[i]) return a_error(error, ME_ARTIFACT_ERR_BINDING, "reduction extent overflow");
        red_count *= (size_t)shape[i];
    }
    if (!a_extrema_valid(o, red_count, out_count, participating))
        return a_error(error, ME_ARTIFACT_ERR_BINDING, "empty/masked extrema require initial");
    return ME_ARTIFACT_SUCCESS;
}
static double a_real(me_dtype d, const void *p) {
    if (d == ME_FLOAT32) { float x; memcpy(&x,p,4); return x; }
    if (d == ME_FLOAT64) { double x; memcpy(&x,p,8); return x; }
    if (a_complex(d)) {
        /* Explicit real/int casts discard the imaginary component. */
        if (d == ME_COMPLEX64) { float parts[2]; memcpy(parts,p,sizeof(parts)); return parts[0]; }
        double parts[2]; memcpy(parts,p,sizeof(parts)); return parts[0];
    }
    uint64_t raw = a_integer(d,p);
    if (a_unsigned(d)) return (double)raw;
    int64_t x; memcpy(&x,&raw,8); return (double)x;
}
static void a_complex_parts(me_dtype d, const void *p, double *real, double *imag) {
    if (d == ME_COMPLEX64) {
        float parts[2]; memcpy(parts,p,sizeof(parts));
        *real = parts[0]; *imag = parts[1];
    }
    else {
        double parts[2]; memcpy(parts,p,sizeof(parts));
        *real = parts[0]; *imag = parts[1];
    }
}
static void a_complex_store(me_dtype d, double real, double imag, void *out) {
    if (d == ME_COMPLEX64) {
        float parts[2] = {(float)real, (float)imag}; memcpy(out,parts,sizeof(parts));
    }
    else {
        double parts[2] = {real, imag}; memcpy(out,parts,sizeof(parts));
    }
}
static void a_cast(me_dtype source, const void *p, me_dtype target, void *out) {
    if (target == ME_FLOAT32) { float x = (float)a_real(source,p); memcpy(out,&x,4); }
    else if (target == ME_FLOAT64) { double x = a_real(source,p); memcpy(out,&x,8); }
    else if (a_complex(target)) {
        if (a_complex(source)) {
            double real, imag; a_complex_parts(source,p,&real,&imag);
            a_complex_store(target,real,imag,out);
        }
        else a_complex_store(target,a_real(source,p),0.0,out);
    }
    else {
        uint64_t raw;
        if (target == ME_BOOL) {
            if (a_complex(source)) {
                double real, imag;
                a_complex_parts(source, p, &real, &imag);
                raw = real != 0.0 || imag != 0.0;
            }
            else raw = a_real(source, p) != 0;
        }
        else raw = a_integer(source,p);
        size_t width = a_width(target); uint16_t little = 1;
        if (*(uint8_t *)&little) memcpy(out,&raw,width);
        else memcpy(out,(uint8_t *)&raw + 8 - width,width);
    }
}
static void a_combine(me_dtype dtype, int op, void *acc, const void *value) {
    if (a_complex(dtype)) {
        double ar, ai, vr, vi;
        a_complex_parts(dtype, acc, &ar, &ai);
        a_complex_parts(dtype, value, &vr, &vi);
        if (op == ME_ARRAY_SUM) a_complex_store(dtype, ar + vr, ai + vi, acc);
        else if (op == ME_ARRAY_PROD) a_complex_store(dtype, ar * vr - ai * vi, ar * vi + ai * vr, acc);
        else if (isnan(vr) || isnan(vi)) a_complex_store(dtype, vr, vi, acc);
        else if (!(isnan(ar) || isnan(ai))) {
            bool less = op == ME_ARRAY_MIN ? (vr < ar || (vr == ar && vi < ai)) :
                                             (vr > ar || (vr == ar && vi > ai));
            if (less) a_complex_store(dtype, vr, vi, acc);
        }
        return;
    }
    if (a_float(dtype)) {
        double x = a_real(dtype,acc), y = a_real(dtype,value), z;
        if (op == ME_ARRAY_SUM) z = x + y;
        else if (op == ME_ARRAY_PROD) z = x * y;
        else if (isnan(x)) z = x;
        else if (isnan(y)) z = y;
        else if (x == 0 && y == 0) z = op == ME_ARRAY_MIN ? (signbit(x) || signbit(y) ? -0.0 : 0.0) : (signbit(x) && signbit(y) ? -0.0 : 0.0);
        else z = op == ME_ARRAY_MIN ? (x < y ? x : y) : (x > y ? x : y);
        /* A float32 accumulator rounds every step, independent of tile size. */
        if (dtype == ME_FLOAT32) { float f = (float)z; memcpy(acc,&f,4); }
        else memcpy(acc,&z,8);
    }
    else {
        uint64_t x = a_integer(dtype,acc), y = a_integer(dtype,value), z;
        if (op == ME_ARRAY_SUM) z = dtype == ME_BOOL ? x || y : x + y;
        else if (op == ME_ARRAY_PROD) z = dtype == ME_BOOL ? x && y : x * y;
        else if (op == ME_ARRAY_ANY) z = x || y;
        else if (op == ME_ARRAY_ALL) z = x && y;
        else {
            bool less;
            if (a_unsigned(dtype)) less = x < y;
            else { int64_t sx,sy; memcpy(&sx,&x,8); memcpy(&sy,&y,8); less = sx < sy; }
            z = (op == ME_ARRAY_MIN ? less : !less) ? x : y;
        }
        a_cast(ME_UINT64,&z,dtype,acc);
    }
}
/* A fixed 128-lane logical leaf, with eight independent accumulators. Iterator
 * and reader boundaries never reset a leaf. Completed leaves enter a binary
 * carry tree; its depth is bounded by the number of bits in size_t. */
typedef me_pairwise_sum_state a_pairwise_sum;
static inline void a_pairwise_push_f64_level(a_pairwise_sum *s, double value,
    size_t level, size_t delta_leaves) {
    size_t occupied = s->occupied;
    size_t bit = (size_t)1 << level;
    while (occupied & bit) {
        double prev;
        memcpy(&prev, &s->tree[level], sizeof(prev));
        value = prev + value;
        occupied &= ~bit;
        level++;
        bit <<= 1;
    }
    memcpy(&s->tree[level], &value, sizeof(value));
    s->occupied = occupied | bit;
    s->leaves += delta_leaves;
}
static inline float a_pairwise_add_f32(float a, float b) {
    /* Match a_combine(ME_FLOAT32, ME_ARRAY_SUM) rounding and FP flags. */
    return (float)((double)a + (double)b);
}
static inline void a_pairwise_push_f32_level(a_pairwise_sum *s, float value,
    size_t level, size_t delta_leaves) {
    size_t occupied = s->occupied;
    size_t bit = (size_t)1 << level;
    while (occupied & bit) {
        float prev;
        memcpy(&prev, &s->tree[level], sizeof(prev));
        value = a_pairwise_add_f32(prev, value);
        occupied &= ~bit;
        level++;
        bit <<= 1;
    }
    uint64_t slot = 0;
    memcpy(&slot, &value, sizeof(value));
    s->tree[level] = slot;
    s->occupied = occupied | bit;
    s->leaves += delta_leaves;
}
static inline double a_pairwise_fold8_f64(const double v[8]) {
    return ((v[0] + v[1]) + (v[2] + v[3])) +
           ((v[4] + v[5]) + (v[6] + v[7]));
}
static inline float a_pairwise_fold8_f32(const float v[8]) {
    float p0 = a_pairwise_add_f32(v[0], v[1]);
    float p1 = a_pairwise_add_f32(v[2], v[3]);
    float p2 = a_pairwise_add_f32(v[4], v[5]);
    float p3 = a_pairwise_add_f32(v[6], v[7]);
    return a_pairwise_add_f32(a_pairwise_add_f32(p0, p1), a_pairwise_add_f32(p2, p3));
}
static void a_pairwise_leaf(a_pairwise_sum *s, me_dtype dtype) {
    if (dtype == ME_FLOAT64) {
        double *v = s->lanes.f64;
        double sum = ((v[0] + v[1]) + (v[2] + v[3])) +
                     ((v[4] + v[5]) + (v[6] + v[7]));
        a_pairwise_push_f64_level(s, sum, 0, 1);
    } else {
        float *v = s->lanes.f32;
        float sum = ((v[0] + v[1]) + (v[2] + v[3])) +
                    ((v[4] + v[5]) + (v[6] + v[7]));
        a_pairwise_push_f32_level(s, sum, 0, 1);
    }
    memset(&s->lanes, 0, sizeof(s->lanes));
    s->used = 0;
}
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
/* Probe CPUID/OSXSAVE/XGETBV once and cache whether AVX2 is usable at runtime. */
static int me_pairwise_cpu_supports_avx2(void) {
    /* Fast path: return cached detection result without re-running CPUID. */
    static int cached = -1;
    int val = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (val >= 0) {
        return val;
    }
    int supported = 0;
#if defined(_MSC_VER)
    /* Check leaf 7 support, OS XSAVE/YMM state (XCR0 bits 1:2), and AVX2 bit. */
    int info[4] = {0, 0, 0, 0};
    __cpuidex(info, 0, 0);
    if (info[0] >= 7) {
        __cpuidex(info, 1, 0);
        int osxsave = (info[2] & (1 << 27)) != 0;
        int avx = (info[2] & (1 << 28)) != 0;
        if (osxsave && avx && ((_xgetbv(0) & 0x6) == 0x6)) {
            __cpuidex(info, 7, 0);
            supported = (info[1] & (1 << 5)) != 0;
        }
    }
#else
    /* Query max CPUID leaf, then verify OSXSAVE, AVX, XCR0 YMM state, and AVX2. */
    unsigned int a = 0, b = 0, c = 0, d = 0;
    __asm__ __volatile__("cpuid"
                         : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                         : "a"(0), "c"(0));
    if (a >= 7) {
        __asm__ __volatile__("cpuid"
                             : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                             : "a"(1), "c"(0));
        int osxsave = (c & (1u << 27)) != 0;
        int avx = (c & (1u << 28)) != 0;
        if (osxsave && avx) {
            unsigned int xcr0_lo = 0, xcr0_hi = 0;
            __asm__ __volatile__("xgetbv"
                                 : "=a"(xcr0_lo), "=d"(xcr0_hi)
                                 : "c"(0));
            if ((xcr0_lo & 0x6u) == 0x6u) {
                __asm__ __volatile__("cpuid"
                                     : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                                     : "a"(7), "c"(0));
                supported = (b & (1u << 5)) != 0;
            }
        }
    }
#endif
    __atomic_store_n(&cached, supported, __ATOMIC_RELAXED);
    return supported;
}

/* Accumulate full 8-double octets into the 8 independent lanes with AVX2. */
static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void a_pairwise_feed_octets_f64_avx2(
    double lanes[8], const uint8_t *src, size_t n_octets) {
    __m256d acc0 = _mm256_loadu_pd(lanes);
    __m256d acc1 = _mm256_loadu_pd(lanes + 4);
    for (size_t i = 0; i < n_octets; i++, src += 8 * sizeof(double)) {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd((const double *)src));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd((const double *)(src + 4 * sizeof(double))));
    }
    _mm256_storeu_pd(lanes, acc0);
    _mm256_storeu_pd(lanes + 4, acc1);
}

/* Accumulate full 8-float octets into the 8 independent lanes with AVX2. */
static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void a_pairwise_feed_octets_f32_avx2(
    float lanes[8], const uint8_t *src, size_t n_octets) {
    __m256 acc = _mm256_loadu_ps(lanes);
    for (size_t i = 0; i < n_octets; i++, src += 8 * sizeof(float)) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)src));
    }
    _mm256_storeu_ps(lanes, acc);
}

/* Reduce contiguous 128-double leaves using AVX2 with exact pairwise order. */
static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void a_pairwise_leaves_f64_avx2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    /* Interleave two 128-item leaves in parallel (4 independent vaddpd chains)
     * to hide FP add latency while preserving exact per-leaf IEEE-754 order. */
    size_t leaf = 0;
    const size_t leaf_bytes = ME_PAIRWISE_SUM_LEAF_ITEMS * sizeof(double);
    for (; leaf + 2 <= nleaves; leaf += 2) {
        /* Accumulate 8 lanes per leaf across two leaves simultaneously. */
        const uint8_t *s0 = src;
        const uint8_t *s1 = src + leaf_bytes;
        __m256d a0 = _mm256_setzero_pd();
        __m256d a1 = _mm256_setzero_pd();
        __m256d b0 = _mm256_setzero_pd();
        __m256d b1 = _mm256_setzero_pd();
#if defined(__GNUC__) && !defined(__clang__)
        /* Keep the 2-leaf loop rolled so GCC does not serialize leaf 0 before leaf 1. */
        #pragma GCC unroll 2
#endif
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 8) {
            a0 = _mm256_add_pd(a0, _mm256_loadu_pd((const double *)(s0 + 0)));
            a1 = _mm256_add_pd(a1, _mm256_loadu_pd((const double *)(s0 + 32)));
            b0 = _mm256_add_pd(b0, _mm256_loadu_pd((const double *)(s1 + 0)));
            b1 = _mm256_add_pd(b1, _mm256_loadu_pd((const double *)(s1 + 32)));
            s0 += 64;
            s1 += 64;
        }
        /* Fold both leaves in exact ((0+1)+(2+3))+((4+5)+(6+7)) order and store 2 sums. */
        __m256d ap = _mm256_add_pd(_mm256_unpacklo_pd(a0, a1), _mm256_unpackhi_pd(a0, a1));
        __m256d bp = _mm256_add_pd(_mm256_unpacklo_pd(b0, b1), _mm256_unpackhi_pd(b0, b1));
        __m128d aq = _mm_add_pd(_mm256_castpd256_pd128(ap), _mm256_extractf128_pd(ap, 1));
        __m128d bq = _mm_add_pd(_mm256_castpd256_pd128(bp), _mm256_extractf128_pd(bp, 1));
        __m128d sums = _mm_add_pd(_mm_unpacklo_pd(aq, bq), _mm_unpackhi_pd(aq, bq));
        _mm_storeu_pd((double *)dst, sums);
        src += 2 * leaf_bytes;
        dst += 2 * sizeof(double);
    }
    /* Handle a single trailing leaf when nleaves is odd. */
    for (; leaf < nleaves; leaf++) {
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 16) {
            acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd((const double *)(src + 0)));
            acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd((const double *)(src + 32)));
            acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd((const double *)(src + 64)));
            acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd((const double *)(src + 96)));
            src += 128;
        }
        /* Fold in exact ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)) order. */
        __m256d lo2 = _mm256_unpacklo_pd(acc0, acc1); /* [a0, a4, a2, a6] */
        __m256d hi2 = _mm256_unpackhi_pd(acc0, acc1); /* [a1, a5, a3, a7] */
        __m256d p = _mm256_add_pd(lo2, hi2);          /* [a0+a1, a4+a5, a2+a3, a6+a7] */
        __m128d q = _mm_add_pd(_mm256_castpd256_pd128(p), _mm256_extractf128_pd(p, 1));
        double sum = _mm_cvtsd_f64(q) + _mm_cvtsd_f64(_mm_unpackhi_pd(q, q));
        memcpy(dst, &sum, sizeof(sum));
        dst += sizeof(sum);
    }
}

/* Reduce contiguous 128-float leaves using AVX2 with exact pairwise order. */
static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void a_pairwise_leaves_f32_avx2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    /* Interleave four 128-item leaves in parallel (4 independent vaddps chains). */
    size_t leaf = 0;
    const size_t leaf_bytes = ME_PAIRWISE_SUM_LEAF_ITEMS * sizeof(float);
    for (; leaf + 4 <= nleaves; leaf += 4) {
        /* Accumulate 8 float lanes per leaf across four leaves simultaneously. */
        const uint8_t *s0 = src;
        const uint8_t *s1 = src + leaf_bytes;
        const uint8_t *s2 = src + 2 * leaf_bytes;
        const uint8_t *s3 = src + 3 * leaf_bytes;
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 8) {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps((const float *)s0));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps((const float *)s1));
            a2 = _mm256_add_ps(a2, _mm256_loadu_ps((const float *)s2));
            a3 = _mm256_add_ps(a3, _mm256_loadu_ps((const float *)s3));
            s0 += 32;
            s1 += 32;
            s2 += 32;
            s3 += 32;
        }
        /* Fold each leaf via even/odd pairs and scalar final adds to avoid extra FP flags. */
        __m256 accs[4] = {a0, a1, a2, a3};
        for (int j = 0; j < 4; j++) {
            __m128 lo = _mm256_castps256_ps128(accs[j]);
            __m128 hi = _mm256_extractf128_ps(accs[j], 1);
            __m128 p = _mm_add_ps(_mm_shuffle_ps(lo, hi, 0x88), _mm_shuffle_ps(lo, hi, 0xDD));
            float t0 = _mm_cvtss_f32(p);
            float t1 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x01));
            float t2 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x02));
            float t3 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x03));
            float sum = (t0 + t1) + (t2 + t3);
            memcpy(dst + j * sizeof(float), &sum, sizeof(sum));
        }
        src += 4 * leaf_bytes;
        dst += 4 * sizeof(float);
    }
    /* Handle the remaining 0..3 trailing leaves one by one. */
    for (; leaf < nleaves; leaf++) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 16) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(src + 0)));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(src + 32)));
            src += 64;
        }
        /* Fold 8 lanes into 4 even+odd pairs, then finish in exact scalar tree order. */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 ev = _mm_shuffle_ps(lo, hi, 0x88); /* [a0, a2, a4, a6] */
        __m128 od = _mm_shuffle_ps(lo, hi, 0xDD); /* [a1, a3, a5, a7] */
        __m128 p = _mm_add_ps(ev, od);            /* [a0+a1, a2+a3, a4+a5, a6+a7] */
        float t0 = _mm_cvtss_f32(p);
        float t1 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x01));
        float t2 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x02));
        float t3 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x03));
        float sum = (t0 + t1) + (t2 + t3);
        memcpy(dst, &sum, sizeof(sum));
        dst += sizeof(sum);
    }
}
#endif /* ME_PAIRWISE_HAVE_AVX2_DISPATCH */

#if defined(ME_PAIRWISE_X86_64)
/* Baseline SSE2 fallback: accumulate 8-double octets across 4 XMM registers. */
static void a_pairwise_feed_octets_f64_sse2(
    double lanes[8], const uint8_t *src, size_t n_octets) {
    __m128d a0 = _mm_loadu_pd(lanes + 0);
    __m128d a1 = _mm_loadu_pd(lanes + 2);
    __m128d a2 = _mm_loadu_pd(lanes + 4);
    __m128d a3 = _mm_loadu_pd(lanes + 6);
    for (size_t i = 0; i < n_octets; i++, src += 8 * sizeof(double)) {
        a0 = _mm_add_pd(a0, _mm_loadu_pd((const double *)(src + 0)));
        a1 = _mm_add_pd(a1, _mm_loadu_pd((const double *)(src + 16)));
        a2 = _mm_add_pd(a2, _mm_loadu_pd((const double *)(src + 32)));
        a3 = _mm_add_pd(a3, _mm_loadu_pd((const double *)(src + 48)));
    }
    _mm_storeu_pd(lanes + 0, a0);
    _mm_storeu_pd(lanes + 2, a1);
    _mm_storeu_pd(lanes + 4, a2);
    _mm_storeu_pd(lanes + 6, a3);
}

/* Baseline SSE2 fallback: accumulate 8-float octets across 2 XMM registers. */
static void a_pairwise_feed_octets_f32_sse2(
    float lanes[8], const uint8_t *src, size_t n_octets) {
    __m128 a0 = _mm_loadu_ps(lanes + 0);
    __m128 a1 = _mm_loadu_ps(lanes + 4);
    for (size_t i = 0; i < n_octets; i++, src += 8 * sizeof(float)) {
        a0 = _mm_add_ps(a0, _mm_loadu_ps((const float *)(src + 0)));
        a1 = _mm_add_ps(a1, _mm_loadu_ps((const float *)(src + 16)));
    }
    _mm_storeu_ps(lanes + 0, a0);
    _mm_storeu_ps(lanes + 4, a1);
}

/* Baseline SSE2 fallback: reduce 128-double leaves in exact pairwise order. */
static void a_pairwise_leaves_f64_sse2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    for (size_t leaf = 0; leaf < nleaves; leaf++) {
        /* Accumulate 8 lanes in 4 XMM vectors across the 128-item leaf. */
        __m128d a0 = _mm_setzero_pd();
        __m128d a1 = _mm_setzero_pd();
        __m128d a2 = _mm_setzero_pd();
        __m128d a3 = _mm_setzero_pd();
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 8) {
            a0 = _mm_add_pd(a0, _mm_loadu_pd((const double *)(src + 0)));
            a1 = _mm_add_pd(a1, _mm_loadu_pd((const double *)(src + 16)));
            a2 = _mm_add_pd(a2, _mm_loadu_pd((const double *)(src + 32)));
            a3 = _mm_add_pd(a3, _mm_loadu_pd((const double *)(src + 48)));
            src += 64;
        }
        /* Fold in exact ((0+1)+(2+3))+((4+5)+(6+7)) pairwise tree order. */
        __m128d p01 = _mm_add_pd(_mm_unpacklo_pd(a0, a1), _mm_unpackhi_pd(a0, a1));
        __m128d p23 = _mm_add_pd(_mm_unpacklo_pd(a2, a3), _mm_unpackhi_pd(a2, a3));
        __m128d q = _mm_add_pd(_mm_unpacklo_pd(p01, p23), _mm_unpackhi_pd(p01, p23));
        double sum = _mm_cvtsd_f64(q) + _mm_cvtsd_f64(_mm_unpackhi_pd(q, q));
        memcpy(dst, &sum, sizeof(sum));
        dst += sizeof(sum);
    }
}

/* Baseline SSE2 fallback: reduce 128-float leaves in exact pairwise order. */
static void a_pairwise_leaves_f32_sse2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    for (size_t leaf = 0; leaf < nleaves; leaf++) {
        /* Accumulate 8 lanes in 2 XMM vectors across the 128-item leaf. */
        __m128 lo = _mm_setzero_ps();
        __m128 hi = _mm_setzero_ps();
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 8) {
            lo = _mm_add_ps(lo, _mm_loadu_ps((const float *)(src + 0)));
            hi = _mm_add_ps(hi, _mm_loadu_ps((const float *)(src + 16)));
            src += 32;
        }
        /* Fold even+odd pairs and finish with scalar adds to avoid unused-lane FP flags. */
        __m128 ev = _mm_shuffle_ps(lo, hi, 0x88);
        __m128 od = _mm_shuffle_ps(lo, hi, 0xDD);
        __m128 p = _mm_add_ps(ev, od);
        float t0 = _mm_cvtss_f32(p);
        float t1 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x01));
        float t2 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x02));
        float t3 = _mm_cvtss_f32(_mm_shuffle_ps(p, p, 0x03));
        float sum = (t0 + t1) + (t2 + t3);
        memcpy(dst, &sum, sizeof(sum));
        dst += sizeof(sum);
    }
}
#endif /* ME_PAIRWISE_X86_64 */

static void a_pairwise_feed(a_pairwise_sum *s, me_dtype dtype, const void *values,
    const uint8_t *mask, size_t count) {
    size_t width = a_width(dtype), offset = 0;
    while (offset < count) {
        /* Keep full unmasked octets free of dynamic indexing and scalar
         * memcpy calls. Independent lanes permit ordinary SIMD/SLP without
         * fast-math or reassociation of any accumulator's additions. */
        if (!mask && !(s->used % 8) && count - offset >= 8) {
#if defined(ME_PAIRWISE_X86_64)
            /* Dispatch full aligned octets up to the leaf boundary to AVX2 or SSE2. */
            size_t end = count - offset;
            if (end > ME_PAIRWISE_SUM_LEAF_ITEMS - s->used) end = ME_PAIRWISE_SUM_LEAF_ITEMS - s->used;
            size_t n_octets = end / 8;
            const uint8_t *src = (const uint8_t *)values + offset * width;
            if (dtype == ME_FLOAT64) {
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
                if (me_pairwise_cpu_supports_avx2()) {
                    a_pairwise_feed_octets_f64_avx2(s->lanes.f64, src, n_octets);
                } else
#endif
                {
                    a_pairwise_feed_octets_f64_sse2(s->lanes.f64, src, n_octets);
                }
            } else {
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
                if (me_pairwise_cpu_supports_avx2()) {
                    a_pairwise_feed_octets_f32_avx2(s->lanes.f32, src, n_octets);
                } else
#endif
                {
                    a_pairwise_feed_octets_f32_sse2(s->lanes.f32, src, n_octets);
                }
            }
            offset += n_octets * 8;
            s->used += n_octets * 8;
#else
#define PAIRWISE_OCTETS(type, member) do { \
    size_t end = count - offset; \
    if (end > ME_PAIRWISE_SUM_LEAF_ITEMS - s->used) end = ME_PAIRWISE_SUM_LEAF_ITEMS - s->used; \
    end = offset + (end & ~(size_t)7); \
    type a0 = s->lanes.member[0], a1 = s->lanes.member[1]; \
    type a2 = s->lanes.member[2], a3 = s->lanes.member[3]; \
    type a4 = s->lanes.member[4], a5 = s->lanes.member[5]; \
    type a6 = s->lanes.member[6], a7 = s->lanes.member[7]; \
    size_t used = s->used; \
    for (; offset < end; offset += 8, used += 8) { \
        type v[8]; \
        memcpy(v, (const uint8_t *)values + offset * sizeof(type), sizeof(v)); \
        a0 += v[0]; a1 += v[1]; \
        a2 += v[2]; a3 += v[3]; \
        a4 += v[4]; a5 += v[5]; \
        a6 += v[6]; a7 += v[7]; \
    } \
    s->lanes.member[0] = a0; s->lanes.member[1] = a1; \
    s->lanes.member[2] = a2; s->lanes.member[3] = a3; \
    s->lanes.member[4] = a4; s->lanes.member[5] = a5; \
    s->lanes.member[6] = a6; s->lanes.member[7] = a7; \
    s->used = used; \
} while (0)
            if (dtype == ME_FLOAT64) PAIRWISE_OCTETS(double, f64);
            else PAIRWISE_OCTETS(float, f32);
#undef PAIRWISE_OCTETS
#endif
            if (s->used == ME_PAIRWISE_SUM_LEAF_ITEMS) a_pairwise_leaf(s, dtype);
            continue;
        }
        size_t n = 8 - s->used % 8;
        if (n > count - offset) n = count - offset;
        size_t first = s->used % 8;
        if (dtype == ME_FLOAT64) {
            for (size_t k = 0; k < n; k++) {
                double value = 0;
                if (!mask || mask[offset + k])
                    memcpy(&value, (const uint8_t *)values + (offset + k)*width, width);
                s->lanes.f64[first + k] += value;
            }
        } else {
            for (size_t k = 0; k < n; k++) {
                float value = 0;
                if (!mask || mask[offset + k])
                    memcpy(&value, (const uint8_t *)values + (offset + k)*width, width);
                s->lanes.f32[first + k] += value;
            }
        }
        offset += n;
        s->used += n;
        if (s->used == ME_PAIRWISE_SUM_LEAF_ITEMS) a_pairwise_leaf(s, dtype);
    }
}
static void a_pairwise_finish(a_pairwise_sum *s, me_dtype dtype, void *acc) {
    if (s->used) a_pairwise_leaf(s, dtype);
    /* Fold the newest (smallest) suffix into older, larger subtrees. */
    uint64_t tail = 0;
    bool found = false;
    for (size_t level = 0; level < sizeof(size_t)*CHAR_BIT; level++) {
        if (!(s->occupied & ((size_t)1 << level))) continue;
        if (found) a_combine(dtype, ME_ARRAY_SUM, &s->tree[level], &tail);
        tail = s->tree[level];
        found = true;
    }
    if (found) a_combine(dtype, ME_ARRAY_SUM, acc, &tail);
}
me_artifact_status me_pairwise_sum_init(me_pairwise_sum_state *s, me_dtype dtype) {
    if (!s || !a_float(dtype)) return ME_ARTIFACT_ERR_BINDING;
    memset(s, 0, sizeof(*s));
    s->dtype = dtype;
    return ME_ARTIFACT_SUCCESS;
}
me_artifact_status me_pairwise_sum_update(me_pairwise_sum_state *s, const void *values, size_t count) {
    if (!s || !a_float(s->dtype) || (count && !values) ||
        count > SIZE_MAX / a_width(s->dtype)) return ME_ARTIFACT_ERR_BINDING;
    if (s->leaves > SIZE_MAX / ME_PAIRWISE_SUM_LEAF_ITEMS || s->used >= ME_PAIRWISE_SUM_LEAF_ITEMS ||
        count > SIZE_MAX - s->leaves * ME_PAIRWISE_SUM_LEAF_ITEMS - s->used)
        return ME_ARTIFACT_ERR_BINDING;
    a_pairwise_feed(s, s->dtype, values, NULL, count);
    return ME_ARTIFACT_SUCCESS;
}
me_artifact_status me_pairwise_sum_append(me_pairwise_sum_state *s, const void *leaves, size_t count) {
    if (!s || !a_float(s->dtype) || s->used || (count && !leaves)) return ME_ARTIFACT_ERR_BINDING;
    if (s->leaves > SIZE_MAX / ME_PAIRWISE_SUM_LEAF_ITEMS ||
        count > SIZE_MAX / ME_PAIRWISE_SUM_LEAF_ITEMS - s->leaves)
        return ME_ARTIFACT_ERR_BINDING;
    if (s->dtype == ME_FLOAT64) {
        const uint8_t *src = (const uint8_t *)leaves;
        size_t i = 0;
        while (i < count && (s->occupied & (size_t)7)) {
            double v;
            memcpy(&v, src + i * sizeof(double), sizeof(v));
            a_pairwise_push_f64_level(s, v, 0, 1);
            i++;
        }
        if ((s->occupied & (size_t)8) && count - i >= 8) {
            double v[8];
            memcpy(v, src + i * sizeof(double), sizeof(v));
            a_pairwise_push_f64_level(s, a_pairwise_fold8_f64(v), 3, 8);
            i += 8;
        }
        while (count - i >= 16) {
            double v0[8], v1[8];
            memcpy(v0, src + i * sizeof(double), sizeof(v0));
            memcpy(v1, src + (i + 8) * sizeof(double), sizeof(v1));
            a_pairwise_push_f64_level(s, a_pairwise_fold8_f64(v0) + a_pairwise_fold8_f64(v1), 4, 16);
            i += 16;
        }
        if (count - i >= 8) {
            double v[8];
            memcpy(v, src + i * sizeof(double), sizeof(v));
            a_pairwise_push_f64_level(s, a_pairwise_fold8_f64(v), 3, 8);
            i += 8;
        }
        while (i < count) {
            double v;
            memcpy(&v, src + i * sizeof(double), sizeof(v));
            a_pairwise_push_f64_level(s, v, 0, 1);
            i++;
        }
        return ME_ARTIFACT_SUCCESS;
    }
    const uint8_t *src = (const uint8_t *)leaves;
    size_t i = 0;
    while (i < count && (s->occupied & (size_t)7)) {
        float v;
        memcpy(&v, src + i * sizeof(float), sizeof(v));
        a_pairwise_push_f32_level(s, v, 0, 1);
        i++;
    }
    if ((s->occupied & (size_t)8) && count - i >= 8) {
        float v[8];
        memcpy(v, src + i * sizeof(float), sizeof(v));
        a_pairwise_push_f32_level(s, a_pairwise_fold8_f32(v), 3, 8);
        i += 8;
    }
    while (count - i >= 16) {
        float v0[8], v1[8];
        memcpy(v0, src + i * sizeof(float), sizeof(v0));
        memcpy(v1, src + (i + 8) * sizeof(float), sizeof(v1));
        a_pairwise_push_f32_level(s,
            a_pairwise_add_f32(a_pairwise_fold8_f32(v0), a_pairwise_fold8_f32(v1)), 4, 16);
        i += 16;
    }
    if (count - i >= 8) {
        float v[8];
        memcpy(v, src + i * sizeof(float), sizeof(v));
        a_pairwise_push_f32_level(s, a_pairwise_fold8_f32(v), 3, 8);
        i += 8;
    }
    while (i < count) {
        float v;
        memcpy(&v, src + i * sizeof(float), sizeof(v));
        a_pairwise_push_f32_level(s, v, 0, 1);
        i++;
    }
    return ME_ARTIFACT_SUCCESS;
}
me_artifact_status me_pairwise_sum_leaves(me_dtype dtype, const void *values,
    size_t count, void *leaves, size_t capacity) {
    size_t width = a_width(dtype);
    if (!a_float(dtype) || count % ME_PAIRWISE_SUM_LEAF_ITEMS || count > SIZE_MAX / width ||
        count / ME_PAIRWISE_SUM_LEAF_ITEMS > capacity / width ||
        (count && (!values || !leaves))) return ME_ARTIFACT_ERR_BINDING;
    size_t nleaves = count / ME_PAIRWISE_SUM_LEAF_ITEMS;
    if (dtype == ME_FLOAT64) {
        const uint8_t *src = (const uint8_t *)values;
        uint8_t *dst = (uint8_t *)leaves;
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
        /* Prefer AVX2 leaf kernel at runtime when supported by CPU and OS. */
        if (me_pairwise_cpu_supports_avx2()) {
            a_pairwise_leaves_f64_avx2(src, dst, nleaves);
            return ME_ARTIFACT_SUCCESS;
        }
#endif
#if defined(ME_PAIRWISE_X86_64)
        /* Fall back to baseline SSE2 leaf kernel on x86-64 without AVX2. */
        a_pairwise_leaves_f64_sse2(src, dst, nleaves);
        return ME_ARTIFACT_SUCCESS;
#else
        for (size_t leaf = 0; leaf < nleaves; leaf++) {
            double a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 8) {
                double v[8];
                memcpy(v, src, sizeof(v));
                src += sizeof(v);
                a0 += v[0]; a1 += v[1];
                a2 += v[2]; a3 += v[3];
                a4 += v[4]; a5 += v[5];
                a6 += v[6]; a7 += v[7];
            }
            double sum = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            memcpy(dst, &sum, sizeof(sum));
            dst += sizeof(sum);
        }
        return ME_ARTIFACT_SUCCESS;
#endif
    }
    const uint8_t *src = (const uint8_t *)values;
    uint8_t *dst = (uint8_t *)leaves;
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
    /* Prefer AVX2 float32 leaf kernel at runtime when supported by CPU and OS. */
    if (me_pairwise_cpu_supports_avx2()) {
        a_pairwise_leaves_f32_avx2(src, dst, nleaves);
        return ME_ARTIFACT_SUCCESS;
    }
#endif
#if defined(ME_PAIRWISE_X86_64)
    /* Fall back to baseline SSE2 float32 leaf kernel on x86-64 without AVX2. */
    a_pairwise_leaves_f32_sse2(src, dst, nleaves);
    return ME_ARTIFACT_SUCCESS;
#else
    for (size_t leaf = 0; leaf < nleaves; leaf++) {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 8) {
            float v[8];
            memcpy(v, src, sizeof(v));
            src += sizeof(v);
            a0 += v[0]; a1 += v[1];
            a2 += v[2]; a3 += v[3];
            a4 += v[4]; a5 += v[5];
            a6 += v[6]; a7 += v[7];
        }
        float sum = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        memcpy(dst, &sum, sizeof(sum));
        dst += sizeof(sum);
    }
    return ME_ARTIFACT_SUCCESS;
#endif
}
me_artifact_status me_pairwise_sum_finish(me_pairwise_sum_state *s, void *output) {
    if (!s || !a_float(s->dtype) || !output) return ME_ARTIFACT_ERR_BINDING;
    memset(output, 0, a_width(s->dtype));
    a_pairwise_finish(s, s->dtype, output);
    return ME_ARTIFACT_SUCCESS;
}
static me_artifact_status a_run(const me_artifact *a, const me_array_view *inputs, int ninputs,
    int rank, const int64_t *shape, const me_array_options *o, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error, bool execute, bool identity_map,
    const me_array_source *sources, const me_array_output *destination_view) {
    int out_rank; int64_t out_shape[ME_ARRAY_MAX_RANK]; me_dtype dtype;
    me_array_report local = {0}; if (!report) report = &local; memset(report,0,sizeof(*report));
    int rc = me_array_result_shape(a,rank,shape,o,&out_rank,out_shape,&dtype,error);
    if (rc) return rc;
    bool pairwise = o->reduction == ME_ARRAY_SUM_PAIRWISE;
    me_array_options normalized = *o;
    if (pairwise) normalized.reduction = ME_ARRAY_SUM;
    o = &normalized;
    me_array_view destination = {0};
    bool scatter = false;
    if (destination_view) {
        rc = dsl_array_output_validate(destination_view, out_rank, out_shape, dtype, error);
        if (rc) return rc;
        destination = a_output_view(destination_view);
        output = destination_view->base;
        capacity = destination_view->capacity;
        scatter = !a_contiguous(&destination);
    }
    size_t total, out_count, red_count = 1, width = a_width(dtype), source_width = me_artifact_output_itemsize(a);
    a_product(rank,shape,&total); a_product(out_rank,out_shape,&out_count);
    bool axes[ME_ARRAY_MAX_RANK]; a_axes(rank,o,axes);
    for (int i = 0; i < rank; i++) if (axes[i]) {
        if (shape[i] && red_count > SIZE_MAX / (uint64_t)shape[i]) return a_error(error,ME_ARTIFACT_ERR_BINDING,"reduction extent overflow");
        red_count *= (size_t)shape[i];
    }
    if (ninputs < 0 || ninputs > 128 || ninputs != me_artifact_ninputs(a) || (ninputs && !inputs) ||
        out_count > SIZE_MAX / width || capacity < out_count * width ||
        (capacity && (!output || (uintptr_t)output > UINTPTR_MAX - capacity)) ||
        (out_count && (!output || (!destination_view && (uintptr_t)output % width)))) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid output capacity or bindings");
    int binding[128]; bool direct[128], source_linear[128]; bool readers = false;
    bool suffix = true, seen = false;
    for (int i = 0; i < rank; i++) { if (axes[i]) seen = true; else if (seen) suffix = false; }
    for (int b = 0; b < ninputs; b++) {
        binding[b] = -1;
        for (int i = 0; i < ninputs; i++) if (inputs[i].name && !strcmp(inputs[i].name,me_artifact_input_name(a,b))) {
            if (binding[b] != -1) return a_error(error,ME_ARTIFACT_ERR_BINDING,"duplicate array input");
            binding[b] = i;
        }
        if (binding[b] < 0) return a_error(error,ME_ARTIFACT_ERR_BINDING,"missing array input");
        const me_array_view *v = &inputs[binding[b]];
        bool reader = sources && sources[binding[b]].read;
        readers |= reader;
        /* Matching shapes and suffix reductions traverse storage linearly;
         * broadcasting/non-suffix reductions still require coordinate mapping. */
        source_linear[b] = suffix && v->rank == rank &&
            (!rank || !memcmp(v->shape,shape,(size_t)rank*sizeof(int64_t)));
        if (v->dtype != me_artifact_input_dtype(a,b) ||
            !(reader ? a_reader_valid(v,rank,shape) : a_broadcast(v,rank,shape)) ||
            a_overlap(v->base,v->capacity,output,capacity)) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid bounds, broadcast, dtype or overlap");
        direct[b] = !reader && suffix && a_native(v) && a_contiguous(v) && v->rank == rank &&
                    (!rank || !memcmp(v->shape,shape,(size_t)rank*sizeof(int64_t))) &&
                    !(((uintptr_t)v->base + v->offset) % a_width(v->dtype));
    }
    if (o->where && (o->where->dtype != ME_BOOL || !a_broadcast(o->where,rank,shape) || a_overlap(o->where->base,o->where->capacity,output,capacity))) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid participating mask");
    if (o->initial && a_overlap(o->initial,width,output,capacity)) return a_error(error,ME_ARTIFACT_ERR_BINDING,"initial overlaps output");
    if (!a_extrema_valid(o, red_count, out_count, o->where != NULL)) return a_error(error,ME_ARTIFACT_ERR_BINDING,"empty/masked extrema require initial");
    size_t tile = o->tile_items ? o->tile_items : 1024;
    if (tile > INT32_MAX || tile > SIZE_MAX / source_width) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid tile size");
    if (tile > red_count && o->reduction != ME_ARRAY_NONE) tile = red_count;
    if (tile > total && o->reduction == ME_ARRAY_NONE) tile = total;
    if (!tile) tile = 1;
    if (readers && tile > SIZE_MAX / sizeof(size_t)) return a_error(error,ME_ARTIFACT_ERR_BINDING,"reader index scratch overflow");
    if (!execute) return ME_ARTIFACT_SUCCESS;
    /* Identity maps need no interpreter/JIT or floating-state transition.
     * This shortcut follows the complete shared validation, including bounds,
     * dtype, shape, overlap and tile options. Hosts can parallelize disjoint
     * regions through the same native executor regardless of JIT availability. */
    if (identity_map && o->reduction == ME_ARRAY_NONE && ninputs == 1 &&
        direct[0] && !scatter && inputs[binding[0]].dtype == dtype) {
        const me_array_view *input = &inputs[binding[0]];
        if (total) memcpy((uint8_t *)output + destination.offset,
            (const uint8_t *)input->base + input->offset,total*width);
        report->zero_copy_tiles = total ? 1 : 0;
#ifndef __EMSCRIPTEN__
        report->fp_supported = 1;
#endif
        return ME_ARTIFACT_SUCCESS;
    }
    if (identity_map && o->reduction == ME_ARRAY_NONE && ninputs == 1 && destination_view) {
        const me_array_view *input = &inputs[binding[0]];
        bool same = !(sources && sources[binding[0]].read) && a_native(input) && input->dtype == dtype && input->rank == out_rank &&
            (!out_rank || !memcmp(input->shape,out_shape,(size_t)out_rank*sizeof(*out_shape)));
        size_t low = destination.offset, high = destination.offset, source_low = input->offset;
        for (int i = 0; same && i < out_rank; i++) if (out_shape[i] > 1) {
            same = input->strides[i] == destination.strides[i];
            if (!same) break;
            int64_t stride = destination.strides[i];
            a_add(stride < 0 ? low : high,stride,(uint64_t)out_shape[i]-1,stride < 0 ? &low : &high);
            if (stride < 0) a_add(source_low,stride,(uint64_t)out_shape[i]-1,&source_low);
        }
        /* Matching dense permutations/reversals have identical physical value
         * order. Copy their interval without visiting logical coordinates. */
        if (same && (!total || high-low+width == total*width)) {
            if (total) memcpy((uint8_t *)output+low,(const uint8_t *)input->base+source_low,total*width);
            report->zero_copy_tiles = total ? 1 : 0;
#ifndef __EMSCRIPTEN__
            report->fp_supported = 1;
#endif
            return ME_ARTIFACT_SUCCESS;
        }
    }
    /* Keep the complete shared preflight above. Only native contiguous tiles
     * without masks bypass map evaluation; other layouts use the original path. */
    bool direct_reduction = identity_map && o->reduction != ME_ARRAY_NONE &&
        ninputs == 1 && (direct[0] || (sources && sources[binding[0]].read)) && !o->where &&
        inputs[binding[0]].dtype == me_artifact_output_dtype(a);
    void *owned[128] = {0}; me_artifact_buffer buffers[128];
    size_t *indices = readers ? malloc(tile * sizeof(size_t)) : NULL;
    uint8_t *mask = o->where ? malloc(tile) : NULL;
    bool need_scratch = (o->reduction != ME_ARRAY_NONE && !direct_reduction) || scatter;
    void *scratch = need_scratch ? malloc(tile * source_width) : NULL;
    if ((readers && !indices) || (o->where && !mask) || (need_scratch && !scratch)) { rc = ME_ARTIFACT_ERR_OOM; goto cleanup; }
    report->temporary_bytes = (indices ? tile * sizeof(size_t) : 0) + (mask ? tile : 0) + (scratch ? tile * source_width : 0);
    if (pairwise) report->temporary_bytes += sizeof(a_pairwise_sum);
    for (int b = 0; b < ninputs; b++) {
        const me_array_view *v = &inputs[binding[b]]; size_t w = a_width(v->dtype);
        if (!direct[b]) {
            if (tile > SIZE_MAX / w || !(owned[b] = malloc(tile*w))) { rc = ME_ARTIFACT_ERR_OOM; goto cleanup; }
            report->temporary_bytes += tile*w;
        }
        buffers[b] = (me_artifact_buffer){v->name,v->dtype,w,NULL,0};
    }
    dsl_menudet_fp_state saved;
    if (!dsl_menudet_fp_begin(&saved)) { rc = ME_ARTIFACT_ERR_UNSUPPORTED; goto cleanup; }
    unsigned *previous = dsl_menudet_status_begin(&report->fp_flags);
#ifndef __EMSCRIPTEN__
    report->fp_supported = 1;
#endif
    size_t groups = o->reduction == ME_ARRAY_NONE ? 1 : out_count;
    size_t lanes = o->reduction == ME_ARRAY_NONE ? total : red_count;
    bool gather = mask != NULL;
    for (int b = 0; b < ninputs; b++) gather |= !direct[b] && !(sources && sources[binding[b]].read);
    for (size_t g = 0; g < groups && !rc; g++) {
        a_pairwise_sum pairwise_state;
        if (pairwise) memset(&pairwise_state, 0, sizeof(pairwise_state));
        uint64_t acc = 0; bool initialized = false;
        unsigned char acc_complex[16] = {0};
        if (o->initial) { memcpy(&acc,o->initial,width); initialized = true; }
        else if (o->reduction != ME_ARRAY_MIN && o->reduction != ME_ARRAY_MAX) {
            uint64_t identity = o->reduction == ME_ARRAY_PROD || o->reduction == ME_ARRAY_ALL;
            a_cast(ME_UINT64,&identity,dtype,&acc); initialized = true;
        }
        for (size_t begin = 0; begin < lanes && !rc;) {
            size_t count = lanes - begin < tile ? lanes - begin : tile;
            for (size_t lane = 0; gather && lane < count; lane++) {
                int64_t coordinates[ME_ARRAY_MAX_RANK];
                a_coordinates(rank,shape,axes,o->reduction != ME_ARRAY_NONE,g,begin + lane,coordinates);
                if (mask) { a_load(o->where,a_offset(o->where,rank,coordinates),&mask[lane]); if (mask[lane] > 1) { rc = ME_ARTIFACT_ERR_BINDING; break; } report->gathered_bytes++; }
                for (int b = 0; b < ninputs; b++) if (!direct[b] && !(sources && sources[binding[b]].read)) {
                    const me_array_view *v = &inputs[binding[b]];
                    a_load(v,a_offset(v,rank,coordinates),(uint8_t *)owned[b] + lane*buffers[b].itemsize);
                    report->gathered_bytes += buffers[b].itemsize;
                }
            }
            if (rc) break;
            for (int b = 0; b < ninputs && !rc; b++) if (sources && sources[binding[b]].read) {
                const me_array_source *source = &sources[binding[b]];
                if (mask) memset(owned[b],0,count*buffers[b].itemsize);
                size_t participating = 0;
                for (size_t lane = 0; lane < count; lane++) {
                    if (mask && !mask[lane]) continue;
                    if (source_linear[b]) {
                        indices[participating++] = g * lanes + begin + lane;
                        continue;
                    }
                    int64_t coordinates[ME_ARRAY_MAX_RANK];
                    a_coordinates(rank,shape,axes,o->reduction != ME_ARRAY_NONE,g,begin + lane,coordinates);
                    indices[participating++] = a_source_index(&source->view,rank,coordinates);
                }
                if (!participating) continue;
                /* Storage callbacks do not contribute numerical FP diagnostics. */
                fenv_t reader_environment;
                if (fegetenv(&reader_environment)) {
                    rc = a_error(error,ME_ARTIFACT_ERR_UNSUPPORTED,"cannot save reader floating environment");
                    break;
                }
                unsigned reader_flags = 0;
                unsigned *reader_previous = dsl_menudet_status_begin(&reader_flags);
                int read_status = source->read(source->owner,indices,participating,owned[b],participating*buffers[b].itemsize);
                dsl_menudet_status_end(reader_previous);
                int restore_status = fesetenv(&reader_environment);
                if (restore_status) rc = a_error(error,ME_ARTIFACT_ERR_UNSUPPORTED,"cannot restore reader floating environment");
                else if (read_status) rc = a_error(error,ME_ARTIFACT_ERR_EVAL,"storage reader failed");
                else {
                    report->gathered_bytes += participating*buffers[b].itemsize;
                    /* Scatter backward so compact reads can share the gather
                     * buffer without overwriting not-yet-scattered values. */
                    if (mask) for (size_t lane = count; lane-- > 0;) if (mask[lane]) {
                        participating--;
                        memmove((uint8_t *)owned[b] + lane*buffers[b].itemsize,
                            (uint8_t *)owned[b] + participating*buffers[b].itemsize,buffers[b].itemsize);
                    }
                }
            }
            if (rc) break;
            for (int b = 0; b < ninputs; b++) {
                const me_array_view *v = &inputs[binding[b]];
                buffers[b].data = direct[b] ? (const uint8_t *)v->base + v->offset + (g*lanes + begin)*buffers[b].itemsize : owned[b];
                buffers[b].capacity = count*buffers[b].itemsize;
                if (direct[b]) report->zero_copy_tiles++;
            }
            if (!direct_reduction) {
                void *tile_output = scratch ? scratch : (uint8_t *)output + destination.offset + begin*source_width;
                me_artifact_eval_descriptor descriptor = {sizeof(descriptor),ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,count,count*source_width,mask,mask ? count : 0,0,NULL,NULL,NULL};
                if (identity_map && o->reduction == ME_ARRAY_NONE && ninputs == 1) {
                    memcpy(tile_output,buffers[0].data,count*source_width);
                } else {
                    rc = me_artifact_eval_ex(a,buffers,ninputs,tile_output,&descriptor,error);
                    report->evaluated_tiles++;
                }
                if (!rc && scatter && o->reduction == ME_ARRAY_NONE) {
                    for (size_t lane = 0; lane < count; lane++) {
                        int64_t coordinates[ME_ARRAY_MAX_RANK];
                        a_coordinates(out_rank,out_shape,axes,false,0,begin + lane,coordinates);
                        memcpy((uint8_t *)output + a_offset(&destination,out_rank,coordinates),
                            (uint8_t *)scratch + lane*source_width,source_width);
                    }
                }
            }
            const void *values = direct_reduction ? buffers[0].data : o->reduction == ME_ARRAY_NONE ? NULL : scratch;
            bool handled = false;
            if (!rc && pairwise && values) {
                a_pairwise_feed(&pairwise_state, dtype, values, mask, count);
                handled = true;
            }
            /* Computed maps produce the same typed, contiguous tile as direct
             * inputs. Without a participation mask, reuse the serial typed
             * reducers instead of casting/dispatching once per mapped lane. */
            if (!rc && !handled && !mask && values &&
                (o->reduction == ME_ARRAY_MIN || o->reduction == ME_ARRAY_MAX) &&
                dtype == me_artifact_output_dtype(a)) {
                uint64_t normalized = acc, result;
                if (!a_float(dtype)) normalized = a_integer(dtype, &acc);
                handled = dsl_menudet_extrema(dtype, values, count, normalized,
                    initialized, o->reduction == ME_ARRAY_MIN, &result);
                if (handled) {
                    if (a_float(dtype)) acc = result;
                    else a_cast(ME_UINT64, &result, dtype, &acc);
                    initialized = true;
                }
            } else if (!rc && !mask && values && dtype == ME_BOOL &&
                (o->reduction == ME_ARRAY_ANY || o->reduction == ME_ARRAY_ALL)) {
                bool truth;
                handled = dsl_menudet_truth_reduce(me_artifact_output_dtype(a), values,
                    count, a_integer(dtype, &acc) != 0, o->reduction == ME_ARRAY_ALL, &truth);
                if (handled) { uint64_t result = truth; a_cast(ME_UINT64, &result, dtype, &acc); }
            }
            if (!rc && !handled && !mask && values &&
                (o->reduction == ME_ARRAY_SUM || o->reduction == ME_ARRAY_PROD) &&
                dtype == me_artifact_output_dtype(a) && a_float(dtype)) {
                /* Fixed serial order, with one float32 rounding per operation.
                 * No reassociation, vector reduction or tile partial sums. */
                if (dtype == ME_FLOAT64) {
                    double sum; memcpy(&sum, &acc, sizeof(sum));
                    sum = o->reduction == ME_ARRAY_PROD ? dsl_menudet_prod_f64(sum, values, count) :
                        dsl_menudet_sum_f64(sum, values, count);
                    memcpy(&acc, &sum, sizeof(sum));
                } else {
                    float sum; memcpy(&sum, &acc, sizeof(sum));
                    sum = o->reduction == ME_ARRAY_PROD ? dsl_menudet_prod_f32(sum, values, count) :
                        dsl_menudet_sum_f32(sum, values, count);
                    memcpy(&acc, &sum, sizeof(sum));
                }
            } else if (!rc && !handled && !mask && values &&
                (o->reduction == ME_ARRAY_SUM || o->reduction == ME_ARRAY_PROD) &&
                !a_float(me_artifact_output_dtype(a)) &&
                ((dtype == ME_UINT64 && a_unsigned(me_artifact_output_dtype(a)) &&
                  me_artifact_output_dtype(a) != ME_BOOL) ||
                 (dtype == ME_INT64 && (!a_unsigned(me_artifact_output_dtype(a)) ||
                  me_artifact_output_dtype(a) == ME_BOOL)))) {
                bool ok = o->reduction == ME_ARRAY_PROD ?
                    dsl_menudet_prod_integer(me_artifact_output_dtype(a), values, count, acc, false, &acc) :
                    dsl_menudet_sum_integer(me_artifact_output_dtype(a), values, count, acc, false, &acc);
                if (!ok) rc = ME_ARTIFACT_ERR_EVAL;
            } else if (!rc && !handled && values && a_complex(dtype)) {
                for (size_t lane = 0; lane < count; lane++) {
                    if (mask && !mask[lane]) continue;
                    unsigned char value[16] = {0};
                    a_cast(me_artifact_output_dtype(a),(const uint8_t *)values + lane*source_width,
                           dtype,value);
                    if (!initialized) { memcpy(acc_complex,value,width); initialized = true; }
                    else a_combine(dtype,o->reduction,acc_complex,value);
                }
            } else if (!rc && !handled && values) for (size_t lane = 0; lane < count; lane++) {
                if (mask && !mask[lane]) continue;
                uint64_t value = 0; a_cast(me_artifact_output_dtype(a),(const uint8_t *)values + lane*source_width,dtype,&value);
                if (!initialized) { memcpy(&acc,&value,width); initialized = true; }
                else a_combine(dtype,o->reduction,&acc,&value);
            }
            begin += count;
        }
        if (!rc && o->reduction != ME_ARRAY_NONE) {
            if (pairwise) a_pairwise_finish(&pairwise_state, dtype, &acc);
            size_t offset = destination.offset + g*width;
            if (scatter) {
                int64_t coordinates[ME_ARRAY_MAX_RANK];
                a_coordinates(out_rank,out_shape,axes,false,0,g,coordinates);
                offset = a_offset(&destination,out_rank,coordinates);
            }
            const void *acc_source = a_complex(dtype) ? (const void *)acc_complex : (const void *)&acc;
            memcpy((uint8_t *)output + offset, acc_source, width);
        }
    }
    if (!dsl_menudet_fp_end(&saved) && !rc) rc = a_error(error,ME_ARTIFACT_ERR_UNSUPPORTED,"cannot restore floating environment");
    dsl_menudet_status_end(previous);
cleanup:
    for (int b = 0; b < ninputs; b++) free(owned[b]);
    free(indices); free(mask); free(scratch);
    if (rc && (!error || !error->message[0])) return a_error(error,rc,"native array traversal failed");
    return rc;
}
me_artifact_status me_artifact_eval_array(const me_artifact *a, const me_array_view *inputs, int ninputs,
    int rank, const int64_t *shape, const me_array_options *o, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error) {
    return a_run(a, inputs, ninputs, rank, shape, o, output, capacity, report, error, true, false, NULL, NULL);
}
me_artifact_status dsl_array_eval_direct_input(const me_artifact *a, const me_array_view *inputs,
    int ninputs, int rank, const int64_t *shape, const me_array_options *o, void *output,
    size_t capacity, me_array_report *report, me_artifact_error *error) {
    return a_run(a, inputs, ninputs, rank, shape, o, output, capacity, report, error, true, true, NULL, NULL);
}
me_artifact_status dsl_array_preflight(const me_artifact *a, const me_array_view *inputs, int ninputs,
    int rank, const int64_t *shape, const me_array_options *o, void *output, size_t capacity,
    me_artifact_error *error) {
    return a_run(a, inputs, ninputs, rank, shape, o, output, capacity, NULL, error, false, false, NULL, NULL);
}
static me_artifact_status a_run_sources(const me_artifact *a, const me_array_source *sources,
    int ninputs, int rank, const int64_t *shape, const me_array_options *o,
    void *output, size_t capacity, me_array_report *report, me_artifact_error *error,
    bool execute, bool identity_map) {
    if (report) memset(report,0,sizeof(*report));
    if (ninputs < 0 || ninputs > 128 || (ninputs && !sources))
        return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid source bindings");
    me_array_view views[128];
    for (int i = 0; i < ninputs; i++) views[i] = sources[i].view;
    return a_run(a,views,ninputs,rank,shape,o,output,capacity,report,error,execute,identity_map,sources,NULL);
}
me_artifact_status dsl_array_run_into(const me_artifact *a,
    const me_array_view *inputs, const me_array_source *sources, int ninputs,
    int rank, const int64_t *shape, const me_array_options *o,
    const me_array_output *output, me_array_report *report, me_artifact_error *error,
    bool execute, bool identity_map) {
    me_array_view views[128];
    if (report) memset(report,0,sizeof(*report));
    if (!output || ninputs < 0 || ninputs > 128 || (ninputs && !inputs && !sources))
        return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid destination or bindings");
    if (sources) {
        for (int i = 0; i < ninputs; i++) views[i] = sources[i].view;
        inputs = views;
    }
    return a_run(a,inputs,ninputs,rank,shape,o,NULL,0,report,error,execute,identity_map,sources,output);
}
me_artifact_status me_artifact_eval_array_into(const me_artifact *a,
    const me_array_view *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *o, const me_array_output *output,
    me_array_report *report, me_artifact_error *error) {
    return dsl_array_run_into(a,inputs,NULL,ninputs,rank,shape,o,output,report,error,true,false);
}
me_artifact_status me_artifact_eval_array_sources_into(const me_artifact *a,
    const me_array_source *inputs, int ninputs, int rank, const int64_t *shape,
    const me_array_options *o, const me_array_output *output,
    me_array_report *report, me_artifact_error *error) {
    return dsl_array_run_into(a,NULL,inputs,ninputs,rank,shape,o,output,report,error,true,false);
}
me_artifact_status me_artifact_eval_array_sources(const me_artifact *a,
    const me_array_source *sources, int ninputs, int rank, const int64_t *shape,
    const me_array_options *o, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error) {
    return a_run_sources(a,sources,ninputs,rank,shape,o,output,capacity,report,error,true,false);
}
me_artifact_status dsl_array_sources_preflight(const me_artifact *a,
    const me_array_source *sources, int ninputs, int rank, const int64_t *shape,
    const me_array_options *o, void *output, size_t capacity, me_artifact_error *error) {
    return a_run_sources(a,sources,ninputs,rank,shape,o,output,capacity,NULL,error,false,false);
}
me_artifact_status dsl_array_eval_direct_sources(const me_artifact *a,
    const me_array_source *sources, int ninputs, int rank, const int64_t *shape,
    const me_array_options *o, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error) {
    return a_run_sources(a,sources,ninputs,rank,shape,o,output,capacity,report,error,true,true);
}
me_artifact_status me_array_reshape(const me_array_view *v, int rank, const int64_t *shape,
    me_array_view *out, me_artifact_error *e) {
    size_t old_count,new_count;
    if (!v || !out || !a_valid(v) || !a_contiguous(v) || !a_product(v->rank,v->shape,&old_count) || !a_product(rank,shape,&new_count) || old_count != new_count) return a_error(e,ME_ARTIFACT_ERR_BINDING,"reshape requires equal extent and C contiguous input");
    me_array_view result = *v; result.rank = rank; uint64_t stride = a_width(v->dtype);
    for (int i = rank - 1; i >= 0; i--) {
        if (stride > INT64_MAX || (shape[i] && stride > SIZE_MAX/(uint64_t)shape[i])) return a_error(e,ME_ARTIFACT_ERR_BINDING,"reshape stride overflow");
        result.shape[i] = shape[i]; result.strides[i] = (int64_t)stride; stride *= (size_t)shape[i];
    }
    *out = result; return ME_ARTIFACT_SUCCESS;
}
me_artifact_status me_array_transpose(const me_array_view *v, const int *axes,
    me_array_view *out, me_artifact_error *e) {
    if (!v || !out || !a_valid(v) || (v->rank && !axes)) return a_error(e,ME_ARTIFACT_ERR_BINDING,"invalid transpose input");
    me_array_view result = *v; bool seen[ME_ARRAY_MAX_RANK] = {0};
    for (int i = 0; i < v->rank; i++) {
        int axis = axes[i] < 0 ? axes[i] + v->rank : axes[i];
        if (axis < 0 || axis >= v->rank || seen[axis]) return a_error(e,ME_ARTIFACT_ERR_BINDING,"invalid transpose axes");
        seen[axis] = true; result.shape[i] = v->shape[axis]; result.strides[i] = v->strides[axis];
    }
    *out = result; return ME_ARTIFACT_SUCCESS;
}
me_artifact_status me_array_slice(const me_array_view *v, int axis, int64_t start,
    int64_t count, int64_t step, me_array_view *out, me_artifact_error *e) {
    if (!v || !out || !a_valid(v)) return a_error(e,ME_ARTIFACT_ERR_BINDING,"invalid slice input");
    if (axis < 0) axis += v->rank;
    if (axis < 0 || axis >= v->rank || count < 0 || !step || step == INT64_MIN ||
        (count && (start < 0 || start >= v->shape[axis]))) return a_error(e,ME_ARTIFACT_ERR_BINDING,"invalid slice bounds");
    uint64_t magnitude = step < 0 ? (uint64_t)-step : (uint64_t)step;
    if (count && (uint64_t)(count-1) > (step < 0 ? (uint64_t)start : (uint64_t)(v->shape[axis]-1-start))/magnitude) return a_error(e,ME_ARTIFACT_ERR_BINDING,"slice exceeds extent");
    me_array_view result = *v;
    if (count && !a_add(v->offset,v->strides[axis],(uint64_t)start,&result.offset)) return a_error(e,ME_ARTIFACT_ERR_BINDING,"slice offset overflow");
    int64_t stride = v->strides[axis];
    uint64_t abs_stride = stride < 0 ? (uint64_t)(-(stride+1))+1 : (uint64_t)stride;
    if (abs_stride && magnitude > INT64_MAX/abs_stride) return a_error(e,ME_ARTIFACT_ERR_BINDING,"slice stride overflow");
    result.strides[axis] = stride*step; result.shape[axis] = count;
    *out = result; return ME_ARTIFACT_SUCCESS;
}
