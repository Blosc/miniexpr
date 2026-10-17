#ifndef MINIEXPR_PORTABLE_BATCH_H
#define MINIEXPR_PORTABLE_BATCH_H
#include <stdint.h>

/* Internal, all-participating contiguous primitives. The caller establishes
 * nearest/ties-even, nontrapping IEEE execution and collects aggregate status. */
#define ME_BATCH_DECL(OP,T,S) \
    void dsl_portable_batch_##OP##_##S(const T *a, const T *b, T *out, int count);
#define ME_BATCH_TYPE(T,S) \
    ME_BATCH_DECL(add,T,S) ME_BATCH_DECL(sub,T,S) ME_BATCH_DECL(mul,T,S) \
    ME_BATCH_DECL(div,T,S) ME_BATCH_DECL(neg,T,S)
ME_BATCH_TYPE(float,f32)
ME_BATCH_TYPE(double,f64)
#define ME_BATCH_INTEGER(T,S) \
    ME_BATCH_DECL(add,T,S) ME_BATCH_DECL(sub,T,S) ME_BATCH_DECL(mul,T,S) ME_BATCH_DECL(neg,T,S)
ME_BATCH_INTEGER(uint8_t,u8)
ME_BATCH_INTEGER(uint16_t,u16)
ME_BATCH_INTEGER(uint32_t,u32)
ME_BATCH_INTEGER(uint64_t,u64)
#undef ME_BATCH_INTEGER
#define ME_BATCH_CAST(T,S) \
    void dsl_portable_batch_cast_##S##_f32(const T *a, float *out, int count); \
    void dsl_portable_batch_cast_##S##_f64(const T *a, double *out, int count);
ME_BATCH_CAST(int8_t,i8) ME_BATCH_CAST(uint8_t,u8)
ME_BATCH_CAST(int16_t,i16) ME_BATCH_CAST(uint16_t,u16)
ME_BATCH_CAST(int32_t,i32) ME_BATCH_CAST(uint32_t,u32)
ME_BATCH_CAST(int64_t,i64) ME_BATCH_CAST(uint64_t,u64)
ME_BATCH_CAST(double,f64)
void dsl_portable_batch_cast_f32_f64(const float *a, double *out, int count);
#undef ME_BATCH_CAST
#undef ME_BATCH_TYPE
#undef ME_BATCH_DECL
#endif
