/* Native logical traversal. Storage/decompression is a host concern, grouping is
 * not: each reduction visits C logical coordinates in a fixed serial order. */
#include "miniexpr_artifact.h"
#include "dsl_portable_fp.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
        default: return 0;
    }
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
static void a_load(const me_array_view *v, size_t offset, void *out) {
    size_t width = a_width(v->dtype);
    const uint8_t *p = (const uint8_t *)v->base + offset;
    if (a_native(v)) memcpy(out, p, width);
    else for (size_t i = 0; i < width; i++) ((uint8_t *)out)[i] = p[width - i - 1];
}
static bool a_axes(int rank, const me_array_options *o, bool *axes) {
    memset(axes, 0, ME_ARRAY_MAX_RANK * sizeof(bool));
    if (!o || o->version != ME_ARTIFACT_ARRAY_VERSION || o->reduction < ME_ARRAY_NONE || o->reduction > ME_ARRAY_ALL ||
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
    if (o->accumulator != ME_AUTO) return a_width(o->accumulator) && !(a_float(source) && !a_float(o->accumulator)) ? o->accumulator : ME_AUTO;
    if ((o->reduction == ME_ARRAY_SUM || o->reduction == ME_ARRAY_PROD) && !a_float(source)) return a_unsigned(source) && source != ME_BOOL ? ME_UINT64 : ME_INT64;
    return source;
}
me_artifact_status me_array_result_shape(const me_artifact *a, int rank, const int64_t *shape,
    const me_array_options *o, int *out_rank, int64_t *out_shape, me_dtype *dtype, me_artifact_error *error) {
    if (error) memset(error,0,sizeof(*error));
    bool axes[ME_ARRAY_MAX_RANK]; size_t n;
    if (!a || !out_rank || !out_shape || !dtype || !a_product(rank, shape, &n) || !a_axes(rank, o, axes)) return a_error(error, ME_ARTIFACT_ERR_BINDING, "invalid logical shape, axes or options");
    if (strcmp(me_artifact_schema_version(a), "1.1") || me_artifact_context_ndim(a) ||
        me_artifact_result_cardinality(a) != ME_ARTIFACT_ELEMENTWISE || !a_width(me_artifact_output_dtype(a))) return a_error(error, ME_ARTIFACT_ERR_UNSUPPORTED, "logical arrays require numeric elementwise rank-zero profile 1.1");
    *dtype = a_dtype(me_artifact_output_dtype(a), o);
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
static double a_real(me_dtype d, const void *p) {
    if (d == ME_FLOAT32) { float x; memcpy(&x,p,4); return x; }
    if (d == ME_FLOAT64) { double x; memcpy(&x,p,8); return x; }
    uint64_t raw = a_integer(d,p);
    if (a_unsigned(d)) return (double)raw;
    int64_t x; memcpy(&x,&raw,8); return (double)x;
}
static void a_cast(me_dtype source, const void *p, me_dtype target, void *out) {
    if (target == ME_FLOAT32) { float x = (float)a_real(source,p); memcpy(out,&x,4); }
    else if (target == ME_FLOAT64) { double x = a_real(source,p); memcpy(out,&x,8); }
    else {
        uint64_t raw = target == ME_BOOL ? a_real(source,p) != 0 : a_integer(source,p);
        size_t width = a_width(target); uint16_t little = 1;
        if (*(uint8_t *)&little) memcpy(out,&raw,width);
        else memcpy(out,(uint8_t *)&raw + 8 - width,width);
    }
}
static void a_combine(me_dtype dtype, int op, void *acc, const void *value) {
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
me_artifact_status me_artifact_eval_array(const me_artifact *a, const me_array_view *inputs, int ninputs,
    int rank, const int64_t *shape, const me_array_options *o, void *output, size_t capacity,
    me_array_report *report, me_artifact_error *error) {
    int out_rank; int64_t out_shape[ME_ARRAY_MAX_RANK]; me_dtype dtype;
    me_array_report local = {0}; if (!report) report = &local; memset(report,0,sizeof(*report));
    int rc = me_array_result_shape(a,rank,shape,o,&out_rank,out_shape,&dtype,error);
    if (rc) return rc;
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
        (out_count && (!output || (uintptr_t)output % width))) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid output capacity or bindings");
    int binding[128]; bool direct[128];
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
        if (v->dtype != me_artifact_input_dtype(a,b) || !a_broadcast(v,rank,shape) || a_overlap(v->base,v->capacity,output,capacity)) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid bounds, broadcast, dtype or overlap");
        direct[b] = suffix && a_native(v) && a_contiguous(v) && v->rank == rank &&
                    (!rank || !memcmp(v->shape,shape,(size_t)rank*sizeof(int64_t))) &&
                    !(((uintptr_t)v->base + v->offset) % a_width(v->dtype));
    }
    if (o->where && (o->where->dtype != ME_BOOL || !a_broadcast(o->where,rank,shape) || a_overlap(o->where->base,o->where->capacity,output,capacity))) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid participating mask");
    if (o->initial && a_overlap(o->initial,width,output,capacity)) return a_error(error,ME_ARTIFACT_ERR_BINDING,"initial overlaps output");
    if ((o->reduction == ME_ARRAY_MIN || o->reduction == ME_ARRAY_MAX) && !o->initial && (!red_count || o->where) && out_count) return a_error(error,ME_ARTIFACT_ERR_BINDING,"empty/masked extrema require initial");
    size_t tile = o->tile_items ? o->tile_items : 1024;
    if (tile > INT32_MAX || tile > SIZE_MAX / source_width) return a_error(error,ME_ARTIFACT_ERR_BINDING,"invalid tile size");
    if (tile > red_count && o->reduction != ME_ARRAY_NONE) tile = red_count;
    if (tile > total && o->reduction == ME_ARRAY_NONE) tile = total;
    if (!tile) tile = 1;
    void *owned[128] = {0}; me_artifact_buffer buffers[128];
    uint8_t *mask = o->where ? malloc(tile) : NULL;
    void *scratch = o->reduction != ME_ARRAY_NONE ? malloc(tile * source_width) : NULL;
    if ((o->where && !mask) || (o->reduction != ME_ARRAY_NONE && !scratch)) { rc = ME_ARTIFACT_ERR_OOM; goto cleanup; }
    report->temporary_bytes = (mask ? tile : 0) + (scratch ? tile * source_width : 0);
    for (int b = 0; b < ninputs; b++) {
        const me_array_view *v = &inputs[binding[b]]; size_t w = a_width(v->dtype);
        if (!direct[b]) {
            if (tile > SIZE_MAX / w || !(owned[b] = malloc(tile*w))) { rc = ME_ARTIFACT_ERR_OOM; goto cleanup; }
            report->temporary_bytes += tile*w;
        }
        buffers[b] = (me_artifact_buffer){v->name,v->dtype,w,NULL,0};
    }
    fenv_t saved;
    if (!dsl_portable_fp_begin(&saved)) { rc = ME_ARTIFACT_ERR_UNSUPPORTED; goto cleanup; }
    unsigned *previous = dsl_portable_status_begin(&report->fp_flags);
#ifndef __EMSCRIPTEN__
    report->fp_supported = 1;
#endif
    size_t groups = o->reduction == ME_ARRAY_NONE ? 1 : out_count;
    size_t lanes = o->reduction == ME_ARRAY_NONE ? total : red_count;
    bool gather = mask != NULL;
    for (int b = 0; b < ninputs; b++) gather |= !direct[b];
    for (size_t g = 0; g < groups && !rc; g++) {
        uint64_t acc = 0; bool initialized = false;
        if (o->initial) { memcpy(&acc,o->initial,width); initialized = true; }
        else if (o->reduction != ME_ARRAY_MIN && o->reduction != ME_ARRAY_MAX) {
            uint64_t identity = o->reduction == ME_ARRAY_PROD || o->reduction == ME_ARRAY_ALL;
            a_cast(ME_UINT64,&identity,dtype,&acc); initialized = true;
        }
        for (size_t begin = 0; begin < lanes && !rc;) {
            size_t count = lanes - begin < tile ? lanes - begin : tile;
            for (size_t lane = 0; gather && lane < count; lane++) {
                size_t outer = g, inner = begin + lane; int64_t coordinates[ME_ARRAY_MAX_RANK];
                for (int i = rank - 1; i >= 0; i--) {
                    size_t *index = o->reduction != ME_ARRAY_NONE && !axes[i] ? &outer : &inner;
                    coordinates[i] = (int64_t)(*index % (size_t)shape[i]); *index /= (size_t)shape[i];
                }
                if (mask) { a_load(o->where,a_offset(o->where,rank,coordinates),&mask[lane]); if (mask[lane] > 1) { rc = ME_ARTIFACT_ERR_BINDING; break; } report->gathered_bytes++; }
                for (int b = 0; b < ninputs; b++) if (!direct[b]) {
                    const me_array_view *v = &inputs[binding[b]];
                    a_load(v,a_offset(v,rank,coordinates),(uint8_t *)owned[b] + lane*buffers[b].itemsize);
                    report->gathered_bytes += buffers[b].itemsize;
                }
            }
            if (rc) break;
            for (int b = 0; b < ninputs; b++) {
                const me_array_view *v = &inputs[binding[b]];
                buffers[b].data = direct[b] ? (const uint8_t *)v->base + v->offset + (g*lanes + begin)*buffers[b].itemsize : owned[b];
                buffers[b].capacity = count*buffers[b].itemsize;
                if (direct[b]) report->zero_copy_tiles++;
            }
            void *destination = scratch ? scratch : (uint8_t *)output + begin*source_width;
            me_artifact_eval_descriptor descriptor = {sizeof(descriptor),ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,count,count*source_width,mask,mask ? count : 0,0,NULL,NULL,NULL};
            rc = me_artifact_eval_ex(a,buffers,ninputs,destination,&descriptor,error);
            report->evaluated_tiles++;
            if (!rc && scratch) for (size_t lane = 0; lane < count; lane++) {
                if (mask && !mask[lane]) continue;
                uint64_t value = 0; a_cast(me_artifact_output_dtype(a),(uint8_t *)scratch + lane*source_width,dtype,&value);
                if (!initialized) { memcpy(&acc,&value,width); initialized = true; }
                else a_combine(dtype,o->reduction,&acc,&value);
            }
            begin += count;
        }
        if (!rc && o->reduction != ME_ARRAY_NONE) memcpy((uint8_t *)output + g*width,&acc,width);
    }
    if (!dsl_portable_fp_end(&saved) && !rc) rc = a_error(error,ME_ARTIFACT_ERR_UNSUPPORTED,"cannot restore floating environment");
    dsl_portable_status_end(previous);
cleanup:
    for (int b = 0; b < ninputs; b++) free(owned[b]);
    free(mask); free(scratch);
    if (rc && (!error || !error->message[0])) return a_error(error,rc,"native array traversal failed");
    return rc;
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
