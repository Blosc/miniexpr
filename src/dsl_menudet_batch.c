#include "dsl_menudet_batch.h"
#include "dsl_menudet_powi.h"
#include <float.h>
#include <limits.h>
#include <string.h>

#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "menudet batch arithmetic bridges require FLT_EVAL_METHOD == 0"
#endif

#if defined(_MSC_VER)
#define ME_BATCH_NOINLINE __declspec(noinline)
#else
#define ME_BATCH_NOINLINE __attribute__((noinline))
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)) && !defined(_M_ARM64EC)
#define ME_BATCH_X86_64 1
#include <emmintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#define ME_BATCH_HAVE_AVX2_DISPATCH 1
#define ME_BATCH_AVX2_TARGET __attribute__((target("avx2")))
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ME_BATCH_ARM64 1
#include <arm_neon.h>
#endif

#define ME_BATCH_KERNEL_VV(OP,T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_menudet_batch_##OP##_##S(const T *a, const T *b, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_POWI(T,S) \
ME_BATCH_NOINLINE void dsl_menudet_batch_powi_##S(const T *a, int64_t exponent, T *restrict out, int count) { \
    /* Keep the common cube a straight-line, vectorizable loop. */ \
    if (exponent == 3) { \
        for (int i=0; i<count; i++) out[i]=(a[i]*a[i])*a[i]; \
        return; \
    } \
    for (int i=0; i<count; i++) out[i]=dsl_menudet_powi_##S(a[i],exponent); \
}
ME_BATCH_POWI(float,f32)
ME_BATCH_POWI(double,f64)
#undef ME_BATCH_POWI

#define ME_BATCH_KERNEL_VS(OP,T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_menudet_batch_##OP##_vs_##S(const T *a, T b, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_KERNEL_SV(OP,T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_menudet_batch_##OP##_sv_##S(T a, const T *b, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_KERNEL_NEG(T,S,EXPR) \
ME_BATCH_NOINLINE void dsl_menudet_batch_neg_##S(const T *a, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(EXPR); \
}

#define ME_BATCH_KERNEL_FILL(T,S) \
ME_BATCH_NOINLINE void dsl_menudet_batch_fill_##S(T a, T *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=a; \
}

/* Contiguous ramp kernels.  On x86_64, baseline SSE2 lacks packed int64->float
 * conversions, so the plain C loops compile to scalar instructions.  Explicit
 * SSE2, AVX2, and ARM64 NEON kernels vectorize them while preserving
 * bit-for-bit IEEE-754 results (unfused mul+add, exact integer bounds so no
 * spurious FE_INEXACT is raised) and exact modulo-2^64 integer arithmetic. */
ME_BATCH_NOINLINE void dsl_menudet_batch_ramp_i64(int64_t base, int64_t start, int64_t step,
                                                    int64_t *restrict out, int count) {
    uint64_t ustart = (uint64_t)start, ustep = (uint64_t)step;
    uint64_t v0 = ustart + (uint64_t)base * ustep;
    int i = 0;
#if defined(ME_BATCH_X86_64)
    if (count >= 4) {
        /* Step two 128-bit lanes (4 int64s/iter) in exact modulo-2^64 arithmetic */
        __m128i cur0 = _mm_set_epi64x((int64_t)(v0 + ustep), (int64_t)v0);
        __m128i inc2 = _mm_set1_epi64x((int64_t)(2 * ustep));
        __m128i cur1 = _mm_add_epi64(cur0, inc2);
        __m128i inc4 = _mm_set1_epi64x((int64_t)(4 * ustep));
        for (; i + 4 <= count; i += 4) {
            _mm_storeu_si128((__m128i *)(out + i), cur0);
            _mm_storeu_si128((__m128i *)(out + i + 2), cur1);
            cur0 = _mm_add_epi64(cur0, inc4);
            cur1 = _mm_add_epi64(cur1, inc4);
        }
        if (i + 2 <= count) {
            _mm_storeu_si128((__m128i *)(out + i), cur0);
            i += 2;
        }
    }
#elif defined(ME_BATCH_ARM64)
    if (count >= 2) {
        /* Step four 128-bit lanes (8 int64s / 64B per iter) in exact modulo-2^64 arithmetic */
        uint64x2_t cur0 = vsetq_lane_u64(v0 + ustep, vdupq_n_u64(v0), 1);
        const uint64x2_t inc2 = vdupq_n_u64(2 * ustep);
        if (count >= 8) {
            uint64x2_t cur1 = vaddq_u64(cur0, inc2);
            const uint64x2_t inc4 = vdupq_n_u64(4 * ustep);
            uint64x2_t cur2 = vaddq_u64(cur0, inc4);
            uint64x2_t cur3 = vaddq_u64(cur1, inc4);
            const uint64x2_t inc8 = vdupq_n_u64(8 * ustep);
            for (; i + 8 <= count; i += 8) {
                vst1q_u64((uint64_t *)(out + i), cur0);
                vst1q_u64((uint64_t *)(out + i + 2), cur1);
                vst1q_u64((uint64_t *)(out + i + 4), cur2);
                vst1q_u64((uint64_t *)(out + i + 6), cur3);
                cur0 = vaddq_u64(cur0, inc8);
                cur1 = vaddq_u64(cur1, inc8);
                cur2 = vaddq_u64(cur2, inc8);
                cur3 = vaddq_u64(cur3, inc8);
            }
        }
        for (; i + 2 <= count; i += 2) {
            vst1q_u64((uint64_t *)(out + i), cur0);
            cur0 = vaddq_u64(cur0, inc2);
        }
    }
#endif
    for (; i < count; i++) out[i] = (int64_t)(v0 + (uint64_t)i * ustep);
}
ME_BATCH_NOINLINE void dsl_menudet_batch_ramp_f32(int64_t base, float start, float step,
                                                    float *restrict out, int count) {
    int i = 0;
#if defined(ME_BATCH_X86_64)
    if (count >= 4 && base >= 0 && base <= (int64_t)INT32_MAX - (int64_t)count - 8) {
        /* Base + i fits in signed int32, so cvtepi32_ps matches (float)(int64_t) bit-for-bit */
        __m128i idx0 = _mm_add_epi32(_mm_set1_epi32((int32_t)base), _mm_setr_epi32(0, 1, 2, 3));
        __m128i idx1 = _mm_add_epi32(idx0, _mm_set1_epi32(4));
        const __m128i inc8 = _mm_set1_epi32(8);
        const __m128 vstart = _mm_set1_ps(start);
        const __m128 vstep = _mm_set1_ps(step);
        for (; i + 8 <= count; i += 8) {
            __m128 f0 = _mm_cvtepi32_ps(idx0);
            __m128 f1 = _mm_cvtepi32_ps(idx1);
            _mm_storeu_ps(out + i, _mm_add_ps(vstart, _mm_mul_ps(f0, vstep)));
            _mm_storeu_ps(out + i + 4, _mm_add_ps(vstart, _mm_mul_ps(f1, vstep)));
            idx0 = _mm_add_epi32(idx0, inc8);
            idx1 = _mm_add_epi32(idx1, inc8);
        }
        if (i + 4 <= count) {
            __m128 f0 = _mm_cvtepi32_ps(idx0);
            _mm_storeu_ps(out + i, _mm_add_ps(vstart, _mm_mul_ps(f0, vstep)));
            i += 4;
        }
    }
#elif defined(ME_BATCH_ARM64)
    if (count >= 4 && base >= 0 && base <= (int64_t)INT32_MAX - (int64_t)count - 16) {
        /* Base + i fits in signed int32, so vcvtq_f32_s32 matches (float)(int64_t) bit-for-bit */
        static const int32_t offs[4] = {0, 1, 2, 3};
        int32x4_t idx0 = vaddq_s32(vdupq_n_s32((int32_t)base), vld1q_s32(offs));
        const int32x4_t inc4 = vdupq_n_s32(4);
        const float32x4_t vstart = vdupq_n_f32(start);
        const float32x4_t vstep = vdupq_n_f32(step);
        if (count >= 16) {
            int32x4_t idx1 = vaddq_s32(idx0, inc4);
            const int32x4_t inc8 = vdupq_n_s32(8);
            int32x4_t idx2 = vaddq_s32(idx0, inc8);
            int32x4_t idx3 = vaddq_s32(idx1, inc8);
            const int32x4_t inc16 = vdupq_n_s32(16);
            for (; i + 16 <= count; i += 16) {
                float32x4_t f0 = vcvtq_f32_s32(idx0);
                float32x4_t f1 = vcvtq_f32_s32(idx1);
                float32x4_t f2 = vcvtq_f32_s32(idx2);
                float32x4_t f3 = vcvtq_f32_s32(idx3);
                vst1q_f32(out + i, vaddq_f32(vstart, vmulq_f32(f0, vstep)));
                vst1q_f32(out + i + 4, vaddq_f32(vstart, vmulq_f32(f1, vstep)));
                vst1q_f32(out + i + 8, vaddq_f32(vstart, vmulq_f32(f2, vstep)));
                vst1q_f32(out + i + 12, vaddq_f32(vstart, vmulq_f32(f3, vstep)));
                idx0 = vaddq_s32(idx0, inc16);
                idx1 = vaddq_s32(idx1, inc16);
                idx2 = vaddq_s32(idx2, inc16);
                idx3 = vaddq_s32(idx3, inc16);
            }
        }
        for (; i + 4 <= count; i += 4) {
            float32x4_t f0 = vcvtq_f32_s32(idx0);
            vst1q_f32(out + i, vaddq_f32(vstart, vmulq_f32(f0, vstep)));
            idx0 = vaddq_s32(idx0, inc4);
        }
    }
#endif
    for (; i < count; i++) out[i] = start + (float)(base + i) * step;
}
ME_BATCH_NOINLINE void dsl_menudet_batch_ramp_f64(int64_t base, double start, double step,
                                                    double *restrict out, int count) {
    int i = 0;
#if defined(ME_BATCH_X86_64)
    if (count >= 2 && base >= 0 && base <= (INT64_C(1) << 53) - (int64_t)count - 4) {
        /* All indices (including trailing +4.0) stay <= 2^53, so double stepping is exact */
        __m128d idx0 = _mm_set_pd((double)(base + 1), (double)base);
        __m128d idx1 = _mm_add_pd(idx0, _mm_set1_pd(2.0));
        const __m128d inc4 = _mm_set1_pd(4.0);
        const __m128d vstart = _mm_set1_pd(start);
        const __m128d vstep = _mm_set1_pd(step);
        for (; i + 4 <= count; i += 4) {
            _mm_storeu_pd(out + i, _mm_add_pd(vstart, _mm_mul_pd(idx0, vstep)));
            _mm_storeu_pd(out + i + 2, _mm_add_pd(vstart, _mm_mul_pd(idx1, vstep)));
            idx0 = _mm_add_pd(idx0, inc4);
            idx1 = _mm_add_pd(idx1, inc4);
        }
        if (i + 2 <= count) {
            _mm_storeu_pd(out + i, _mm_add_pd(vstart, _mm_mul_pd(idx0, vstep)));
            i += 2;
        }
    }
#elif defined(ME_BATCH_ARM64)
    if (count >= 2 && base >= 0 && base <= (INT64_C(1) << 53) - (int64_t)count - 8) {
        /* All indices (including trailing +8.0) stay <= 2^53, so double stepping is exact */
        float64x2_t idx0 = vsetq_lane_f64((double)(base + 1), vdupq_n_f64((double)base), 1);
        const float64x2_t inc2 = vdupq_n_f64(2.0);
        const float64x2_t vstart = vdupq_n_f64(start);
        const float64x2_t vstep = vdupq_n_f64(step);
        if (count >= 8) {
            float64x2_t idx1 = vaddq_f64(idx0, inc2);
            const float64x2_t inc4 = vdupq_n_f64(4.0);
            float64x2_t idx2 = vaddq_f64(idx0, inc4);
            float64x2_t idx3 = vaddq_f64(idx1, inc4);
            const float64x2_t inc8 = vdupq_n_f64(8.0);
            for (; i + 8 <= count; i += 8) {
                vst1q_f64(out + i, vaddq_f64(vstart, vmulq_f64(idx0, vstep)));
                vst1q_f64(out + i + 2, vaddq_f64(vstart, vmulq_f64(idx1, vstep)));
                vst1q_f64(out + i + 4, vaddq_f64(vstart, vmulq_f64(idx2, vstep)));
                vst1q_f64(out + i + 6, vaddq_f64(vstart, vmulq_f64(idx3, vstep)));
                idx0 = vaddq_f64(idx0, inc8);
                idx1 = vaddq_f64(idx1, inc8);
                idx2 = vaddq_f64(idx2, inc8);
                idx3 = vaddq_f64(idx3, inc8);
            }
        }
        for (; i + 2 <= count; i += 2) {
            vst1q_f64(out + i, vaddq_f64(vstart, vmulq_f64(idx0, vstep)));
            idx0 = vaddq_f64(idx0, inc2);
        }
    }
#endif
    for (; i < count; i++) out[i] = start + (double)(base + i) * step;
}

#if defined(ME_BATCH_HAVE_AVX2_DISPATCH)
static int me_batch_cpu_supports_avx2(void) {
    static int cached = -1;
    int val = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (val >= 0) {
        return val;
    }
    int supported = 0;
#if defined(_MSC_VER)
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

static ME_BATCH_NOINLINE ME_BATCH_AVX2_TARGET void dsl_menudet_batch_ramp_i64_avx2(
        int64_t base, int64_t start, int64_t step, int64_t *restrict out, int count) {
    uint64_t ustart = (uint64_t)start, ustep = (uint64_t)step;
    uint64_t v0 = ustart + (uint64_t)base * ustep;
    int i = 0;
    if (count >= 4) {
        /* Step two 256-bit vectors (8 int64s / 64B per iter) in exact modulo-2^64 arithmetic */
        __m256i cur0 = _mm256_set_epi64x(
            (int64_t)(v0 + 3 * ustep), (int64_t)(v0 + 2 * ustep),
            (int64_t)(v0 + 1 * ustep), (int64_t)v0);
        __m256i inc4 = _mm256_set1_epi64x((int64_t)(4 * ustep));
        __m256i cur1 = _mm256_add_epi64(cur0, inc4);
        __m256i inc8 = _mm256_set1_epi64x((int64_t)(8 * ustep));
        for (; i + 8 <= count; i += 8) {
            _mm256_storeu_si256((__m256i *)(out + i), cur0);
            _mm256_storeu_si256((__m256i *)(out + i + 4), cur1);
            cur0 = _mm256_add_epi64(cur0, inc8);
            cur1 = _mm256_add_epi64(cur1, inc8);
        }
        if (i + 4 <= count) {
            _mm256_storeu_si256((__m256i *)(out + i), cur0);
            i += 4;
        }
    }
    for (; i < count; i++) out[i] = (int64_t)(v0 + (uint64_t)i * ustep);
}

static ME_BATCH_NOINLINE ME_BATCH_AVX2_TARGET void dsl_menudet_batch_ramp_f32_avx2(
        int64_t base, float start, float step, float *restrict out, int count) {
    int i = 0;
    if (count >= 8 && base >= 0 && base <= (int64_t)INT32_MAX - (int64_t)count - 16) {
        /* Convert 16 exact int32 indices per iter via cvtepi32_ps (unfused mul + add) */
        __m256i idx0 = _mm256_add_epi32(
            _mm256_set1_epi32((int32_t)base), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        __m256i idx1 = _mm256_add_epi32(idx0, _mm256_set1_epi32(8));
        const __m256i inc16 = _mm256_set1_epi32(16);
        const __m256 vstart = _mm256_set1_ps(start);
        const __m256 vstep = _mm256_set1_ps(step);
        for (; i + 16 <= count; i += 16) {
            __m256 f0 = _mm256_cvtepi32_ps(idx0);
            __m256 f1 = _mm256_cvtepi32_ps(idx1);
            _mm256_storeu_ps(out + i, _mm256_add_ps(vstart, _mm256_mul_ps(f0, vstep)));
            _mm256_storeu_ps(out + i + 8, _mm256_add_ps(vstart, _mm256_mul_ps(f1, vstep)));
            idx0 = _mm256_add_epi32(idx0, inc16);
            idx1 = _mm256_add_epi32(idx1, inc16);
        }
        if (i + 8 <= count) {
            __m256 f0 = _mm256_cvtepi32_ps(idx0);
            _mm256_storeu_ps(out + i, _mm256_add_ps(vstart, _mm256_mul_ps(f0, vstep)));
            i += 8;
        }
    }
    for (; i < count; i++) out[i] = start + (float)(base + i) * step;
}

static ME_BATCH_NOINLINE ME_BATCH_AVX2_TARGET void dsl_menudet_batch_ramp_f64_avx2(
        int64_t base, double start, double step, double *restrict out, int count) {
    int i = 0;
    if (count >= 4 && base >= 0 && base <= (INT64_C(1) << 53) - (int64_t)count - 8) {
        /* Step 8 exact double indices (64B) per iter; unfused mul + add preserves IEEE-754 */
        __m256d idx0 = _mm256_set_pd(
            (double)(base + 3), (double)(base + 2), (double)(base + 1), (double)base);
        __m256d idx1 = _mm256_add_pd(idx0, _mm256_set1_pd(4.0));
        const __m256d inc8 = _mm256_set1_pd(8.0);
        const __m256d vstart = _mm256_set1_pd(start);
        const __m256d vstep = _mm256_set1_pd(step);
        for (; i + 8 <= count; i += 8) {
            _mm256_storeu_pd(out + i, _mm256_add_pd(vstart, _mm256_mul_pd(idx0, vstep)));
            _mm256_storeu_pd(out + i + 4, _mm256_add_pd(vstart, _mm256_mul_pd(idx1, vstep)));
            idx0 = _mm256_add_pd(idx0, inc8);
            idx1 = _mm256_add_pd(idx1, inc8);
        }
        if (i + 4 <= count) {
            _mm256_storeu_pd(out + i, _mm256_add_pd(vstart, _mm256_mul_pd(idx0, vstep)));
            i += 4;
        }
    }
    for (; i < count; i++) out[i] = start + (double)(base + i) * step;
}
#endif

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

/* Bitwise lane-local kernels; boolean lanes are stored as 0/1 bytes, where
 * &, | and ^ coincide with the logical operators. */
#define ME_BATCH_BITWISE(T,S) \
    ME_BATCH_KERNEL_VV(and,T,S,(T)(a[i]&b[i])) \
    ME_BATCH_KERNEL_VS(and,T,S,(T)(a[i]&b)) \
    ME_BATCH_KERNEL_SV(and,T,S,(T)(a&b[i])) \
    ME_BATCH_KERNEL_VV(or,T,S,(T)(a[i]|b[i])) \
    ME_BATCH_KERNEL_VS(or,T,S,(T)(a[i]|b)) \
    ME_BATCH_KERNEL_SV(or,T,S,(T)(a|b[i])) \
    ME_BATCH_KERNEL_VV(xor,T,S,(T)(a[i]^b[i])) \
    ME_BATCH_KERNEL_VS(xor,T,S,(T)(a[i]^b)) \
    ME_BATCH_KERNEL_SV(xor,T,S,(T)(a^b[i]))

ME_BATCH_BITWISE(uint8_t,u8)
ME_BATCH_BITWISE(uint16_t,u16)
ME_BATCH_BITWISE(uint32_t,u32)
ME_BATCH_BITWISE(uint64_t,u64)
#undef ME_BATCH_BITWISE

/* NumPy signed remainder is total, with the divisor's sign. Guard C's
 * exceptional divisions before %, including INT32_MIN / -1. */
ME_BATCH_NOINLINE void dsl_menudet_batch_mod_i32(const int32_t *a, const int32_t *b,
                                               int32_t *restrict out, int count) {
    for (int i = 0; i < count; i++) {
        int32_t left = a[i], right = b[i];
        if (!right || (left == INT32_MIN && right == -1)) {
            out[i] = 0;
        } else {
            int32_t rem = left % right;
            out[i] = rem && ((rem < 0) != (right < 0)) ? rem + right : rem;
        }
    }
}

/* Real -> complex follows p_convert's double intermediate (including integer
 * -> complex64 double rounding); never construct values with real + I*imag. */
#define ME_BATCH_COMPLEX_CAST(T,S) \
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_##S##_c64(const T *a, float _Complex *restrict out, int count) { \
    for (int i=0; i<count; i++) { \
        double real=(double)a[i]; float parts[2]={(float)real,0.0f}; \
        memcpy(out+i,parts,sizeof(parts)); \
    } \
} \
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_##S##_c128(const T *a, double _Complex *restrict out, int count) { \
    for (int i=0; i<count; i++) { \
        double parts[2]={(double)a[i],0.0}; memcpy(out+i,parts,sizeof(parts)); \
    } \
}
ME_BATCH_COMPLEX_CAST(int8_t,i8) ME_BATCH_COMPLEX_CAST(uint8_t,u8)
ME_BATCH_COMPLEX_CAST(int16_t,i16) ME_BATCH_COMPLEX_CAST(uint16_t,u16)
ME_BATCH_COMPLEX_CAST(int32_t,i32) ME_BATCH_COMPLEX_CAST(uint32_t,u32)
ME_BATCH_COMPLEX_CAST(int64_t,i64) ME_BATCH_COMPLEX_CAST(uint64_t,u64)
ME_BATCH_COMPLEX_CAST(double,f64)
#undef ME_BATCH_COMPLEX_CAST

ME_BATCH_NOINLINE void dsl_menudet_batch_cast_f32_c64(const float *a, float _Complex *restrict out, int count) {
    for (int i=0; i<count; i++) {
        /* p_convert widens before narrowing; retain its sNaN quieting/status
         * rather than letting the compiler replace the round trip by a copy. */
        volatile double real=(double)a[i];
        float parts[2]={(float)real,0.0f};
        memcpy(out+i,parts,sizeof(parts));
    }
}
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_f32_c128(const float *a, double _Complex *restrict out, int count) {
    for (int i=0; i<count; i++) {
        double parts[2]={(double)a[i],0.0};
        memcpy(out+i,parts,sizeof(parts));
    }
}

ME_BATCH_NOINLINE void dsl_menudet_batch_cast_c64_c128(const float _Complex *a, double _Complex *restrict out, int count) {
    for (int i=0; i<count; i++) {
        float source[2]; double parts[2];
        memcpy(source,a+i,sizeof(source));
        parts[0]=(double)source[0]; parts[1]=(double)source[1];
        memcpy(out+i,parts,sizeof(parts));
    }
}
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_c128_c64(const double _Complex *a, float _Complex *restrict out, int count) {
    for (int i=0; i<count; i++) {
        double source[2]; float parts[2];
        memcpy(source,a+i,sizeof(source));
        parts[0]=(float)source[0]; parts[1]=(float)source[1];
        memcpy(out+i,parts,sizeof(parts));
    }
}

#define ME_BATCH_CAST(T,S) \
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_##S##_f32(const T *a, float *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(float)a[i]; \
} \
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_##S##_f64(const T *a, double *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(double)a[i]; \
}
ME_BATCH_CAST(int8_t,i8) ME_BATCH_CAST(uint8_t,u8)
ME_BATCH_CAST(int16_t,i16) ME_BATCH_CAST(uint16_t,u16)
ME_BATCH_CAST(int32_t,i32) ME_BATCH_CAST(uint32_t,u32)
ME_BATCH_CAST(int64_t,i64) ME_BATCH_CAST(uint64_t,u64)
ME_BATCH_CAST(double,f64)
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_f32_f64(const float *a, double *restrict out, int count) {
    for (int i=0; i<count; i++) out[i]=(double)a[i];
}
#undef ME_BATCH_CAST

/* Value-preserving integer extensions inserted by NumPy dtype promotion. */
#define ME_BATCH_INT_CAST(TO,TO_S,FROM,FROM_S) \
ME_BATCH_NOINLINE void dsl_menudet_batch_cast_##FROM_S##_##TO_S(const FROM *a, TO *restrict out, int count) { \
    for (int i=0; i<count; i++) out[i]=(TO)a[i]; \
}
ME_BATCH_INT_CAST(int16_t,i16,int8_t,i8) ME_BATCH_INT_CAST(int16_t,i16,uint8_t,u8)
ME_BATCH_INT_CAST(int32_t,i32,int8_t,i8) ME_BATCH_INT_CAST(int32_t,i32,uint8_t,u8)
ME_BATCH_INT_CAST(int32_t,i32,int16_t,i16) ME_BATCH_INT_CAST(int32_t,i32,uint16_t,u16)
ME_BATCH_INT_CAST(int64_t,i64,int8_t,i8) ME_BATCH_INT_CAST(int64_t,i64,uint8_t,u8)
ME_BATCH_INT_CAST(int64_t,i64,int16_t,i16) ME_BATCH_INT_CAST(int64_t,i64,uint16_t,u16)
ME_BATCH_INT_CAST(int64_t,i64,int32_t,i32) ME_BATCH_INT_CAST(int64_t,i64,uint32_t,u32)
ME_BATCH_INT_CAST(uint16_t,u16,uint8_t,u8)
ME_BATCH_INT_CAST(uint32_t,u32,uint8_t,u8) ME_BATCH_INT_CAST(uint32_t,u32,uint16_t,u16)
ME_BATCH_INT_CAST(uint64_t,u64,uint8_t,u8) ME_BATCH_INT_CAST(uint64_t,u64,uint16_t,u16)
ME_BATCH_INT_CAST(uint64_t,u64,uint32_t,u32)
#undef ME_BATCH_INT_CAST

#define ME_BATCH_COMMON_TABLE_FIELDS \
    .f64_vv = { dsl_menudet_batch_add_f64, dsl_menudet_batch_sub_f64, dsl_menudet_batch_mul_f64, dsl_menudet_batch_div_f64 }, \
    .f64_vs = { dsl_menudet_batch_add_vs_f64, dsl_menudet_batch_sub_vs_f64, dsl_menudet_batch_mul_vs_f64, dsl_menudet_batch_div_vs_f64 }, \
    .f64_sv = { dsl_menudet_batch_add_sv_f64, dsl_menudet_batch_sub_sv_f64, dsl_menudet_batch_mul_sv_f64, dsl_menudet_batch_div_sv_f64 }, \
    .f64_neg = dsl_menudet_batch_neg_f64, \
    .f64_fill = dsl_menudet_batch_fill_f64, \
    .f32_vv = { dsl_menudet_batch_add_f32, dsl_menudet_batch_sub_f32, dsl_menudet_batch_mul_f32, dsl_menudet_batch_div_f32 }, \
    .f32_vs = { dsl_menudet_batch_add_vs_f32, dsl_menudet_batch_sub_vs_f32, dsl_menudet_batch_mul_vs_f32, dsl_menudet_batch_div_vs_f32 }, \
    .f32_sv = { dsl_menudet_batch_add_sv_f32, dsl_menudet_batch_sub_sv_f32, dsl_menudet_batch_mul_sv_f32, dsl_menudet_batch_div_sv_f32 }, \
    .f32_neg = dsl_menudet_batch_neg_f32, \
    .f32_fill = dsl_menudet_batch_fill_f32, \
    .u8_vv = { dsl_menudet_batch_add_u8, dsl_menudet_batch_sub_u8, dsl_menudet_batch_mul_u8 }, \
    .u8_vs = { dsl_menudet_batch_add_vs_u8, dsl_menudet_batch_sub_vs_u8, dsl_menudet_batch_mul_vs_u8 }, \
    .u8_sv = { dsl_menudet_batch_add_sv_u8, dsl_menudet_batch_sub_sv_u8, dsl_menudet_batch_mul_sv_u8 }, \
    .u8_neg = dsl_menudet_batch_neg_u8, \
    .u8_fill = dsl_menudet_batch_fill_u8, \
    .u16_vv = { dsl_menudet_batch_add_u16, dsl_menudet_batch_sub_u16, dsl_menudet_batch_mul_u16 }, \
    .u16_vs = { dsl_menudet_batch_add_vs_u16, dsl_menudet_batch_sub_vs_u16, dsl_menudet_batch_mul_vs_u16 }, \
    .u16_sv = { dsl_menudet_batch_add_sv_u16, dsl_menudet_batch_sub_sv_u16, dsl_menudet_batch_mul_sv_u16 }, \
    .u16_neg = dsl_menudet_batch_neg_u16, \
    .u16_fill = dsl_menudet_batch_fill_u16, \
    .u32_vv = { dsl_menudet_batch_add_u32, dsl_menudet_batch_sub_u32, dsl_menudet_batch_mul_u32 }, \
    .u32_vs = { dsl_menudet_batch_add_vs_u32, dsl_menudet_batch_sub_vs_u32, dsl_menudet_batch_mul_vs_u32 }, \
    .u32_sv = { dsl_menudet_batch_add_sv_u32, dsl_menudet_batch_sub_sv_u32, dsl_menudet_batch_mul_sv_u32 }, \
    .u32_neg = dsl_menudet_batch_neg_u32, \
    .u32_fill = dsl_menudet_batch_fill_u32, \
    .u64_vv = { dsl_menudet_batch_add_u64, dsl_menudet_batch_sub_u64, dsl_menudet_batch_mul_u64 }, \
    .u64_vs = { dsl_menudet_batch_add_vs_u64, dsl_menudet_batch_sub_vs_u64, dsl_menudet_batch_mul_vs_u64 }, \
    .u64_sv = { dsl_menudet_batch_add_sv_u64, dsl_menudet_batch_sub_sv_u64, dsl_menudet_batch_mul_sv_u64 }, \
    .u64_neg = dsl_menudet_batch_neg_u64, \
    .u64_fill = dsl_menudet_batch_fill_u64, \
    .f32_compare = dsl_menudet_batch_compare_f32, \
    .f64_compare = dsl_menudet_batch_compare_f64

static const me_pj_arith_table s_arith_table = {
    ME_BATCH_COMMON_TABLE_FIELDS,
    .ramp_i64 = dsl_menudet_batch_ramp_i64,
    .ramp_f32 = dsl_menudet_batch_ramp_f32,
    .ramp_f64 = dsl_menudet_batch_ramp_f64,
};

#if defined(ME_BATCH_HAVE_AVX2_DISPATCH)
static const me_pj_arith_table s_arith_table_avx2 = {
    ME_BATCH_COMMON_TABLE_FIELDS,
    .ramp_i64 = dsl_menudet_batch_ramp_i64_avx2,
    .ramp_f32 = dsl_menudet_batch_ramp_f32_avx2,
    .ramp_f64 = dsl_menudet_batch_ramp_f64_avx2,
};
#endif

#undef ME_BATCH_COMMON_TABLE_FIELDS

const me_pj_arith_table *dsl_menudet_batch_arith_table(void) {
#if defined(ME_BATCH_HAVE_AVX2_DISPATCH)
    if (me_batch_cpu_supports_avx2()) {
        return &s_arith_table_avx2;
    }
#endif
    return &s_arith_table;
}
