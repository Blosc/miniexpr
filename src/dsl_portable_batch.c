#include "dsl_portable_batch.h"
#include <float.h>

#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "portable batch arithmetic bridges require FLT_EVAL_METHOD == 0"
#endif

#if defined(_MSC_VER)
#define ME_BATCH_NOINLINE __declspec(noinline)
#else
#define ME_BATCH_NOINLINE __attribute__((noinline))
#endif

#define ME_BATCH_KERNEL_VV(OP,T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_portable_batch_##OP##_##S(const T *a, const T *b, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_KERNEL_VS(OP,T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_portable_batch_##OP##_vs_##S(const T *a, T b, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_KERNEL_SV(OP,T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_portable_batch_##OP##_sv_##S(T a, const T *b, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_KERNEL_NEG(T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_portable_batch_neg_##S(const T *a, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_KERNEL_FILL(T,S) \
ME_BATCH_NOINLINE void dsl_portable_batch_fill_##S(T a, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=a; \
}

#define ME_BATCH_FLOAT_OPS(T,S) \
    ME_BATCH_KERNEL_VV(add,T,S,a[i]+b[i]) \
    ME_BATCH_KERNEL_VS(add,T,S,a[i]+b) \
    ME_BATCH_KERNEL_SV(add,T,S,a+b[i]) \
    ME_BATCH_KERNEL_VV(sub,T,S,a[i]-b[i]) \
    ME_BATCH_KERNEL_VS(sub,T,S,a[i]-b) \
    ME_BATCH_KERNEL_SV(sub,T,S,a-b[i]) \
    ME_BATCH_KERNEL_VV(mul,T,S,a[i]*b[i]) \
    ME_BATCH_KERNEL_VS(mul,T,S,a[i]*b) \
    ME_BATCH_KERNEL_SV(mul,T,S,a*b[i]) \
    ME_BATCH_KERNEL_VV(div,T,S,a[i]/b[i]) \
    ME_BATCH_KERNEL_VS(div,T,S,a[i]/b) \
    ME_BATCH_KERNEL_SV(div,T,S,a/b[i]) \
    ME_BATCH_KERNEL_NEG(T,S,-a[i]) \
    ME_BATCH_KERNEL_FILL(T,S)

ME_BATCH_FLOAT_OPS(float,f32)
ME_BATCH_FLOAT_OPS(double,f64)
#undef ME_BATCH_FLOAT_OPS

#define ME_BATCH_INT_OPS(T,S) \
    ME_BATCH_KERNEL_VV(add,T,S,(T)((uint64_t)a[i]+(uint64_t)b[i])) \
    ME_BATCH_KERNEL_VS(add,T,S,(T)((uint64_t)a[i]+(uint64_t)b)) \
    ME_BATCH_KERNEL_SV(add,T,S,(T)((uint64_t)a+(uint64_t)b[i])) \
    ME_BATCH_KERNEL_VV(sub,T,S,(T)((uint64_t)a[i]-(uint64_t)b[i])) \
    ME_BATCH_KERNEL_VS(sub,T,S,(T)((uint64_t)a[i]-(uint64_t)b)) \
    ME_BATCH_KERNEL_SV(sub,T,S,(T)((uint64_t)a-(uint64_t)b[i])) \
    ME_BATCH_KERNEL_VV(mul,T,S,(T)((uint64_t)a[i]*(uint64_t)b[i])) \
    ME_BATCH_KERNEL_VS(mul,T,S,(T)((uint64_t)a[i]*(uint64_t)b)) \
    ME_BATCH_KERNEL_SV(mul,T,S,(T)((uint64_t)a*(uint64_t)b[i])) \
    ME_BATCH_KERNEL_NEG(T,S,(T)(UINT64_C(0)-(uint64_t)a[i])) \
    ME_BATCH_KERNEL_FILL(T,S)

ME_BATCH_INT_OPS(uint8_t,u8)
ME_BATCH_INT_OPS(uint16_t,u16)
ME_BATCH_INT_OPS(uint32_t,u32)
ME_BATCH_INT_OPS(uint64_t,u64)
#undef ME_BATCH_INT_OPS

#define ME_BATCH_CAST(T,S) \
ME_BATCH_NOINLINE void dsl_portable_batch_cast_##S##_f32(const T *a, float *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(float)a[i]; \
} \
ME_BATCH_NOINLINE void dsl_portable_batch_cast_##S##_f64(const T *a, double *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(double)a[i]; \
}
ME_BATCH_CAST(int8_t,i8) ME_BATCH_CAST(uint8_t,u8)
ME_BATCH_CAST(int16_t,i16) ME_BATCH_CAST(uint16_t,u16)
ME_BATCH_CAST(int32_t,i32) ME_BATCH_CAST(uint32_t,u32)
ME_BATCH_CAST(int64_t,i64) ME_BATCH_CAST(uint64_t,u64)
ME_BATCH_CAST(double,f64)
ME_BATCH_NOINLINE void dsl_portable_batch_cast_f32_f64(const float *a, double *restrict out, int count) {
    for (int i=0; i<count; i++) out[i]=(double)a[i];
}
#undef ME_BATCH_CAST

static const me_pj_arith_table s_arith_table = {
    .f64_vv = { dsl_portable_batch_add_f64, dsl_portable_batch_sub_f64, dsl_portable_batch_mul_f64, dsl_portable_batch_div_f64 },
    .f64_vs = { dsl_portable_batch_add_vs_f64, dsl_portable_batch_sub_vs_f64, dsl_portable_batch_mul_vs_f64, dsl_portable_batch_div_vs_f64 },
    .f64_sv = { dsl_portable_batch_add_sv_f64, dsl_portable_batch_sub_sv_f64, dsl_portable_batch_mul_sv_f64, dsl_portable_batch_div_sv_f64 },
    .f64_neg = dsl_portable_batch_neg_f64,
    .f64_fill = dsl_portable_batch_fill_f64,

    .f32_vv = { dsl_portable_batch_add_f32, dsl_portable_batch_sub_f32, dsl_portable_batch_mul_f32, dsl_portable_batch_div_f32 },
    .f32_vs = { dsl_portable_batch_add_vs_f32, dsl_portable_batch_sub_vs_f32, dsl_portable_batch_mul_vs_f32, dsl_portable_batch_div_vs_f32 },
    .f32_sv = { dsl_portable_batch_add_sv_f32, dsl_portable_batch_sub_sv_f32, dsl_portable_batch_mul_sv_f32, dsl_portable_batch_div_sv_f32 },
    .f32_neg = dsl_portable_batch_neg_f32,
    .f32_fill = dsl_portable_batch_fill_f32,

    .u8_vv = { dsl_portable_batch_add_u8, dsl_portable_batch_sub_u8, dsl_portable_batch_mul_u8 },
    .u8_vs = { dsl_portable_batch_add_vs_u8, dsl_portable_batch_sub_vs_u8, dsl_portable_batch_mul_vs_u8 },
    .u8_sv = { dsl_portable_batch_add_sv_u8, dsl_portable_batch_sub_sv_u8, dsl_portable_batch_mul_sv_u8 },
    .u8_neg = dsl_portable_batch_neg_u8,
    .u8_fill = dsl_portable_batch_fill_u8,

    .u16_vv = { dsl_portable_batch_add_u16, dsl_portable_batch_sub_u16, dsl_portable_batch_mul_u16 },
    .u16_vs = { dsl_portable_batch_add_vs_u16, dsl_portable_batch_sub_vs_u16, dsl_portable_batch_mul_vs_u16 },
    .u16_sv = { dsl_portable_batch_add_sv_u16, dsl_portable_batch_sub_sv_u16, dsl_portable_batch_mul_sv_u16 },
    .u16_neg = dsl_portable_batch_neg_u16,
    .u16_fill = dsl_portable_batch_fill_u16,

    .u32_vv = { dsl_portable_batch_add_u32, dsl_portable_batch_sub_u32, dsl_portable_batch_mul_u32 },
    .u32_vs = { dsl_portable_batch_add_vs_u32, dsl_portable_batch_sub_vs_u32, dsl_portable_batch_mul_vs_u32 },
    .u32_sv = { dsl_portable_batch_add_sv_u32, dsl_portable_batch_sub_sv_u32, dsl_portable_batch_mul_sv_u32 },
    .u32_neg = dsl_portable_batch_neg_u32,
    .u32_fill = dsl_portable_batch_fill_u32,

    .u64_vv = { dsl_portable_batch_add_u64, dsl_portable_batch_sub_u64, dsl_portable_batch_mul_u64 },
    .u64_vs = { dsl_portable_batch_add_vs_u64, dsl_portable_batch_sub_vs_u64, dsl_portable_batch_mul_vs_u64 },
    .u64_sv = { dsl_portable_batch_add_sv_u64, dsl_portable_batch_sub_sv_u64, dsl_portable_batch_mul_sv_u64 },
    .u64_neg = dsl_portable_batch_neg_u64,
    .u64_fill = dsl_portable_batch_fill_u64,
};

const me_pj_arith_table *dsl_portable_batch_arith_table(void) {
    return &s_arith_table;
}
