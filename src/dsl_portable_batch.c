#include "dsl_portable_batch.h"

/* Each loop has exactly one total floating operation, with all lanes active.
 * No recursive dispatch, reductions, masks, speculative alternatives or FMA.
 * Keeping these noinline primitives in their own translation unit permits
 * ordinary SIMD loop vectorization while the scalar evaluator keeps its strict
 * fenv-access model for checked/short-circuit operations. IEEE flags are still
 * produced by the real arithmetic and collected by the caller. No fast-math,
 * reassociation, approximation or process-wide FP-mode change is used. */
#if defined(_MSC_VER)
#define ME_BATCH_NOINLINE __declspec(noinline)
#else
#define ME_BATCH_NOINLINE __attribute__((noinline))
#endif
#define ME_BATCH_KERNEL(OP,T,S,VALUE) \
ME_BATCH_NOINLINE void dsl_portable_batch_##OP##_##S(const T *a, const T *b, T *out, int count) { \
    (void)b; \
    for (int i=0; i<count; i++) out[i]=(VALUE); \
}
#define ME_BATCH_TYPE(T,S) \
    ME_BATCH_KERNEL(add,T,S,a[i]+b[i]) \
    ME_BATCH_KERNEL(sub,T,S,a[i]-b[i]) \
    ME_BATCH_KERNEL(mul,T,S,a[i]*b[i]) \
    ME_BATCH_KERNEL(div,T,S,a[i]/b[i]) \
    ME_BATCH_KERNEL(neg,T,S,-a[i])
ME_BATCH_TYPE(float,f32)
ME_BATCH_TYPE(double,f64)
#define ME_BATCH_INTEGER(T,S) \
    ME_BATCH_KERNEL(add,T,S,(T)((uint64_t)a[i]+(uint64_t)b[i])) \
    ME_BATCH_KERNEL(sub,T,S,(T)((uint64_t)a[i]-(uint64_t)b[i])) \
    ME_BATCH_KERNEL(mul,T,S,(T)((uint64_t)a[i]*(uint64_t)b[i])) \
    ME_BATCH_KERNEL(neg,T,S,(T)(UINT64_C(0)-(uint64_t)a[i]))
ME_BATCH_INTEGER(uint8_t,u8)
ME_BATCH_INTEGER(uint16_t,u16)
ME_BATCH_INTEGER(uint32_t,u32)
ME_BATCH_INTEGER(uint64_t,u64)
#undef ME_BATCH_INTEGER
#define ME_BATCH_CAST(T,S) \
ME_BATCH_NOINLINE void dsl_portable_batch_cast_##S##_f32(const T *a, float *out, int count) { \
    for (int i=0; i<count; i++) out[i]=(float)a[i]; \
} \
ME_BATCH_NOINLINE void dsl_portable_batch_cast_##S##_f64(const T *a, double *out, int count) { \
    for (int i=0; i<count; i++) out[i]=(double)a[i]; \
}
ME_BATCH_CAST(int8_t,i8) ME_BATCH_CAST(uint8_t,u8)
ME_BATCH_CAST(int16_t,i16) ME_BATCH_CAST(uint16_t,u16)
ME_BATCH_CAST(int32_t,i32) ME_BATCH_CAST(uint32_t,u32)
ME_BATCH_CAST(int64_t,i64) ME_BATCH_CAST(uint64_t,u64)
ME_BATCH_CAST(double,f64)
ME_BATCH_NOINLINE void dsl_portable_batch_cast_f32_f64(const float *a, double *out, int count) {
    for (int i=0; i<count; i++) out[i]=(double)a[i];
}
#undef ME_BATCH_CAST
#undef ME_BATCH_TYPE
#undef ME_BATCH_KERNEL
#undef ME_BATCH_NOINLINE
