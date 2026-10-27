#include "dsl_menudet_batch.h"
#include <string.h>

#if defined(_MSC_VER)
#define ME_BATCH_NOINLINE __declspec(noinline)
#else
#define ME_BATCH_NOINLINE __attribute__((noinline))
#endif

/* Keep the scalar complex evaluator's host contraction policy. The real batch
 * kernels have a separate noncontracting policy, so these live in their own
 * translation unit. Match double intermediates (also for complex64), one final
 * narrowing, and component multiplication without C99 infinity recovery. */
#define ME_BATCH_COMPLEX_BODY(T,A,B,RR,RI) \
    T ap[2], bp[2], result[2]; \
    memcpy(ap, &(A), sizeof(ap)); \
    memcpy(bp, &(B), sizeof(bp)); \
    double ar=ap[0], ai=ap[1], br=bp[0], bi=bp[1]; \
    result[0]=(T)(RR); result[1]=(T)(RI); \
    memcpy(out+i, result, sizeof(result));

#define ME_BATCH_COMPLEX_OP(OP,T,S,RR,RI) \
ME_BATCH_NOINLINE void dsl_menudet_batch_##OP##_##S(const T _Complex *a, const T _Complex *b, T _Complex *restrict out, int count) { \
    for (int i=0; i<count; i++) { ME_BATCH_COMPLEX_BODY(T,a[i],b[i],RR,RI) } \
} \
ME_BATCH_NOINLINE void dsl_menudet_batch_##OP##_vs_##S(const T _Complex *a, T _Complex b, T _Complex *restrict out, int count) { \
    for (int i=0; i<count; i++) { ME_BATCH_COMPLEX_BODY(T,a[i],b,RR,RI) } \
} \
ME_BATCH_NOINLINE void dsl_menudet_batch_##OP##_sv_##S(T _Complex a, const T _Complex *b, T _Complex *restrict out, int count) { \
    for (int i=0; i<count; i++) { ME_BATCH_COMPLEX_BODY(T,a,b[i],RR,RI) } \
}

#define ME_BATCH_COMPLEX_OPS(T,S) \
    ME_BATCH_COMPLEX_OP(add,T,S,ar+br,ai+bi) \
    ME_BATCH_COMPLEX_OP(sub,T,S,ar-br,ai-bi) \
    ME_BATCH_COMPLEX_OP(mul,T,S,ar*br-ai*bi,ar*bi+ai*br) \
ME_BATCH_NOINLINE void dsl_menudet_batch_neg_##S(const T _Complex *a, T _Complex *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=-a[i]; \
} \
ME_BATCH_NOINLINE void dsl_menudet_batch_fill_##S(T _Complex a, T _Complex *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=a; \
}
ME_BATCH_COMPLEX_OPS(float,c64)
ME_BATCH_COMPLEX_OPS(double,c128)
#undef ME_BATCH_COMPLEX_OPS
#undef ME_BATCH_COMPLEX_OP
#undef ME_BATCH_COMPLEX_BODY
