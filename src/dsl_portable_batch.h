#ifndef MINIEXPR_PORTABLE_BATCH_H
#define MINIEXPR_PORTABLE_BATCH_H
#include <stdint.h>

#if defined(_MSC_VER) && !defined(__clang__) && !defined(restrict)
#define restrict __restrict
#endif

#define ME_DSL_PORTABLE_JIT_ARITH_ABI_VERSION 1

typedef struct me_pj_arith_table {
    void (*f64_vv[4])(const double *, const double *, double *restrict, int);
    void (*f64_vs[4])(const double *, double, double *restrict, int);
    void (*f64_sv[4])(double, const double *, double *restrict, int);
    void (*f64_neg)(const double *, double *restrict, int);
    void (*f64_fill)(double, double *restrict, int);

    void (*f32_vv[4])(const float *, const float *, float *restrict, int);
    void (*f32_vs[4])(const float *, float, float *restrict, int);
    void (*f32_sv[4])(float, const float *, float *restrict, int);
    void (*f32_neg)(const float *, float *restrict, int);
    void (*f32_fill)(float, float *restrict, int);

    void (*u8_vv[3])(const uint8_t *, const uint8_t *, uint8_t *restrict, int);
    void (*u8_vs[3])(const uint8_t *, uint8_t, uint8_t *restrict, int);
    void (*u8_sv[3])(uint8_t, const uint8_t *, uint8_t *restrict, int);
    void (*u8_neg)(const uint8_t *, uint8_t *restrict, int);
    void (*u8_fill)(uint8_t, uint8_t *restrict, int);

    void (*u16_vv[3])(const uint16_t *, const uint16_t *, uint16_t *restrict, int);
    void (*u16_vs[3])(const uint16_t *, uint16_t, uint16_t *restrict, int);
    void (*u16_sv[3])(uint16_t, const uint16_t *, uint16_t *restrict, int);
    void (*u16_neg)(const uint16_t *, uint16_t *restrict, int);
    void (*u16_fill)(uint16_t, uint16_t *restrict, int);

    void (*u32_vv[3])(const uint32_t *, const uint32_t *, uint32_t *restrict, int);
    void (*u32_vs[3])(const uint32_t *, uint32_t, uint32_t *restrict, int);
    void (*u32_sv[3])(uint32_t, const uint32_t *, uint32_t *restrict, int);
    void (*u32_neg)(const uint32_t *, uint32_t *restrict, int);
    void (*u32_fill)(uint32_t, uint32_t *restrict, int);

    void (*u64_vv[3])(const uint64_t *, const uint64_t *, uint64_t *restrict, int);
    void (*u64_vs[3])(const uint64_t *, uint64_t, uint64_t *restrict, int);
    void (*u64_sv[3])(uint64_t, const uint64_t *, uint64_t *restrict, int);
    void (*u64_neg)(const uint64_t *, uint64_t *restrict, int);
    void (*u64_fill)(uint64_t, uint64_t *restrict, int);

    void (*ramp_i64)(int64_t, int64_t, int64_t, int64_t *restrict, int);
    void (*ramp_f32)(int64_t, float, float, float *restrict, int);
    void (*ramp_f64)(int64_t, double, double, double *restrict, int);
} me_pj_arith_table;

const me_pj_arith_table *dsl_portable_batch_arith_table(void);

/* Internal, all-participating contiguous primitives. The caller establishes
 * nearest/ties-even, nontrapping IEEE execution and collects aggregate status. */
#define ME_BATCH_DECL(OP,T,S) \
    void dsl_portable_batch_##OP##_##S(const T *a, const T *b, T *restrict out, int count); \
    void dsl_portable_batch_##OP##_vs_##S(const T *a, T b, T *restrict out, int count); \
    void dsl_portable_batch_##OP##_sv_##S(T a, const T *b, T *restrict out, int count);

#define ME_BATCH_UNARY_DECL(T,S) \
    void dsl_portable_batch_neg_##S(const T *a, T *restrict out, int count); \
    void dsl_portable_batch_fill_##S(T a, T *restrict out, int count);

#define ME_BATCH_TYPE(T,S) \
    ME_BATCH_DECL(add,T,S) ME_BATCH_DECL(sub,T,S) ME_BATCH_DECL(mul,T,S) \
    ME_BATCH_DECL(div,T,S) ME_BATCH_UNARY_DECL(T,S)

ME_BATCH_TYPE(float,f32)
ME_BATCH_TYPE(double,f64)
#undef ME_BATCH_TYPE

#define ME_BATCH_INTEGER(T,S) \
    ME_BATCH_DECL(add,T,S) ME_BATCH_DECL(sub,T,S) ME_BATCH_DECL(mul,T,S) \
    ME_BATCH_UNARY_DECL(T,S)

ME_BATCH_INTEGER(uint8_t,u8)
ME_BATCH_INTEGER(uint16_t,u16)
ME_BATCH_INTEGER(uint32_t,u32)
ME_BATCH_INTEGER(uint64_t,u64)
#undef ME_BATCH_INTEGER
#undef ME_BATCH_UNARY_DECL
#undef ME_BATCH_DECL

#define ME_BATCH_CAST(T,S) \
    void dsl_portable_batch_cast_##S##_f32(const T *a, float *restrict out, int count); \
    void dsl_portable_batch_cast_##S##_f64(const T *a, double *restrict out, int count);
ME_BATCH_CAST(int8_t,i8) ME_BATCH_CAST(uint8_t,u8)
ME_BATCH_CAST(int16_t,i16) ME_BATCH_CAST(uint16_t,u16)
ME_BATCH_CAST(int32_t,i32) ME_BATCH_CAST(uint32_t,u32)
ME_BATCH_CAST(int64_t,i64) ME_BATCH_CAST(uint64_t,u64)
ME_BATCH_CAST(double,f64)
void dsl_portable_batch_cast_f32_f64(const float *a, double *restrict out, int count);
#undef ME_BATCH_CAST

#endif
