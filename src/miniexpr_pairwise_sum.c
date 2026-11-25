/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE)
**********************************************************************/
#include "miniexpr_reduce.h"
#include "dsl_menudet_reduce.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

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

static bool pairwise_float(me_dtype d) {
    return d == ME_FLOAT32 || d == ME_FLOAT64;
}

static size_t pairwise_width(me_dtype d) {
    return d == ME_FLOAT32 ? sizeof(float) : d == ME_FLOAT64 ? sizeof(double) : 0;
}

static inline float pairwise_add_f32(float a, float b) {
    /* Preserve float32 rounding and FP status flags without widening results. */
    return (float)((double)a + (double)b);
}

static inline void pairwise_add_acc(me_dtype dtype, void *acc, uint64_t tail_bits) {
    if (dtype == ME_FLOAT64) {
        double x, y;
        memcpy(&x, acc, sizeof(x));
        memcpy(&y, &tail_bits, sizeof(y));
        x += y;
        memcpy(acc, &x, sizeof(x));
    }
    else {
        float x, y;
        memcpy(&x, acc, sizeof(x));
        memcpy(&y, &tail_bits, sizeof(y));
        x = pairwise_add_f32(x, y);
        memcpy(acc, &x, sizeof(x));
    }
}

/* A fixed 128-lane logical leaf, with eight independent accumulators. Iterator
 * and reader boundaries never reset a leaf. Completed leaves enter a binary
 * carry tree; its depth is bounded by the number of bits in size_t. */
static inline void pairwise_push_f64_level(me_pairwise_sum_state *s, double value,
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

static inline void pairwise_push_f32_level(me_pairwise_sum_state *s, float value,
    size_t level, size_t delta_leaves) {
    size_t occupied = s->occupied;
    size_t bit = (size_t)1 << level;
    while (occupied & bit) {
        float prev;
        memcpy(&prev, &s->tree[level], sizeof(prev));
        value = pairwise_add_f32(prev, value);
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

static inline double pairwise_fold8_f64(const double v[8]) {
    return ((v[0] + v[1]) + (v[2] + v[3])) +
           ((v[4] + v[5]) + (v[6] + v[7]));
}

static inline float pairwise_fold8_f32(const float v[8]) {
    float p0 = pairwise_add_f32(v[0], v[1]);
    float p1 = pairwise_add_f32(v[2], v[3]);
    float p2 = pairwise_add_f32(v[4], v[5]);
    float p3 = pairwise_add_f32(v[6], v[7]);
    return pairwise_add_f32(pairwise_add_f32(p0, p1), pairwise_add_f32(p2, p3));
}

static void pairwise_leaf(me_pairwise_sum_state *s, me_dtype dtype) {
    if (dtype == ME_FLOAT64) {
        double *v = s->lanes.f64;
        double sum = ((v[0] + v[1]) + (v[2] + v[3])) +
                     ((v[4] + v[5]) + (v[6] + v[7]));
        pairwise_push_f64_level(s, sum, 0, 1);
    }
    else {
        float *v = s->lanes.f32;
        float sum = ((v[0] + v[1]) + (v[2] + v[3])) +
                    ((v[4] + v[5]) + (v[6] + v[7]));
        pairwise_push_f32_level(s, sum, 0, 1);
    }
    memset(&s->lanes, 0, sizeof(s->lanes));
    s->used = 0;
}

#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
/* Probe CPUID/OSXSAVE/XGETBV once and cache whether AVX2 is usable at runtime. */
static int me_pairwise_cpu_supports_avx2(void) {
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

static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void pairwise_feed_octets_f64_avx2(
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

static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void pairwise_feed_octets_f32_avx2(
    float lanes[8], const uint8_t *src, size_t n_octets) {
    __m256 acc = _mm256_loadu_ps(lanes);
    for (size_t i = 0; i < n_octets; i++, src += 8 * sizeof(float)) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)src));
    }
    _mm256_storeu_ps(lanes, acc);
}

static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void pairwise_leaves_f64_avx2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    size_t leaf = 0;
    const size_t leaf_bytes = ME_PAIRWISE_SUM_LEAF_ITEMS * sizeof(double);
    for (; leaf + 2 <= nleaves; leaf += 2) {
        const uint8_t *s0 = src;
        const uint8_t *s1 = src + leaf_bytes;
        __m256d a0 = _mm256_setzero_pd();
        __m256d a1 = _mm256_setzero_pd();
        __m256d b0 = _mm256_setzero_pd();
        __m256d b1 = _mm256_setzero_pd();
#if defined(__GNUC__) && !defined(__clang__)
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
        __m256d ap = _mm256_add_pd(_mm256_unpacklo_pd(a0, a1), _mm256_unpackhi_pd(a0, a1));
        __m256d bp = _mm256_add_pd(_mm256_unpacklo_pd(b0, b1), _mm256_unpackhi_pd(b0, b1));
        __m128d aq = _mm_add_pd(_mm256_castpd256_pd128(ap), _mm256_extractf128_pd(ap, 1));
        __m128d bq = _mm_add_pd(_mm256_castpd256_pd128(bp), _mm256_extractf128_pd(bp, 1));
        __m128d sums = _mm_add_pd(_mm_unpacklo_pd(aq, bq), _mm_unpackhi_pd(aq, bq));
        _mm_storeu_pd((double *)dst, sums);
        src += 2 * leaf_bytes;
        dst += 2 * sizeof(double);
    }
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
        __m256d lo2 = _mm256_unpacklo_pd(acc0, acc1);
        __m256d hi2 = _mm256_unpackhi_pd(acc0, acc1);
        __m256d p = _mm256_add_pd(lo2, hi2);
        __m128d q = _mm_add_pd(_mm256_castpd256_pd128(p), _mm256_extractf128_pd(p, 1));
        double sum = _mm_cvtsd_f64(q) + _mm_cvtsd_f64(_mm_unpackhi_pd(q, q));
        memcpy(dst, &sum, sizeof(sum));
        dst += sizeof(sum);
    }
}

static ME_PAIRWISE_NOINLINE ME_PAIRWISE_AVX2_TARGET void pairwise_leaves_f32_avx2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    size_t leaf = 0;
    const size_t leaf_bytes = ME_PAIRWISE_SUM_LEAF_ITEMS * sizeof(float);
    for (; leaf + 4 <= nleaves; leaf += 4) {
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
    for (; leaf < nleaves; leaf++) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 16) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(src + 0)));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps((const float *)(src + 32)));
            src += 64;
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
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
#endif /* ME_PAIRWISE_HAVE_AVX2_DISPATCH */

#if defined(ME_PAIRWISE_X86_64)
static void pairwise_feed_octets_f64_sse2(
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

static void pairwise_feed_octets_f32_sse2(
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

static void pairwise_leaves_f64_sse2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    for (size_t leaf = 0; leaf < nleaves; leaf++) {
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
        __m128d p01 = _mm_add_pd(_mm_unpacklo_pd(a0, a1), _mm_unpackhi_pd(a0, a1));
        __m128d p23 = _mm_add_pd(_mm_unpacklo_pd(a2, a3), _mm_unpackhi_pd(a2, a3));
        __m128d q = _mm_add_pd(_mm_unpacklo_pd(p01, p23), _mm_unpackhi_pd(p01, p23));
        double sum = _mm_cvtsd_f64(q) + _mm_cvtsd_f64(_mm_unpackhi_pd(q, q));
        memcpy(dst, &sum, sizeof(sum));
        dst += sizeof(sum);
    }
}

static void pairwise_leaves_f32_sse2(
    const uint8_t *src, uint8_t *dst, size_t nleaves) {
    for (size_t leaf = 0; leaf < nleaves; leaf++) {
        __m128 lo = _mm_setzero_ps();
        __m128 hi = _mm_setzero_ps();
        for (size_t k = 0; k < ME_PAIRWISE_SUM_LEAF_ITEMS; k += 8) {
            lo = _mm_add_ps(lo, _mm_loadu_ps((const float *)(src + 0)));
            hi = _mm_add_ps(hi, _mm_loadu_ps((const float *)(src + 16)));
            src += 32;
        }
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

void dsl_pairwise_sum_feed(me_pairwise_sum_state *s, me_dtype dtype, const void *values,
    const uint8_t *mask, size_t count) {
    size_t width = pairwise_width(dtype), offset = 0;
    while (offset < count) {
        if (!mask && !(s->used % 8) && count - offset >= 8) {
#if defined(ME_PAIRWISE_X86_64)
            size_t end = count - offset;
            if (end > ME_PAIRWISE_SUM_LEAF_ITEMS - s->used) end = ME_PAIRWISE_SUM_LEAF_ITEMS - s->used;
            size_t n_octets = end / 8;
            const uint8_t *src = (const uint8_t *)values + offset * width;
            if (dtype == ME_FLOAT64) {
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
                if (me_pairwise_cpu_supports_avx2()) {
                    pairwise_feed_octets_f64_avx2(s->lanes.f64, src, n_octets);
                }
                else
#endif
                {
                    pairwise_feed_octets_f64_sse2(s->lanes.f64, src, n_octets);
                }
            }
            else {
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
                if (me_pairwise_cpu_supports_avx2()) {
                    pairwise_feed_octets_f32_avx2(s->lanes.f32, src, n_octets);
                }
                else
#endif
                {
                    pairwise_feed_octets_f32_sse2(s->lanes.f32, src, n_octets);
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
            if (s->used == ME_PAIRWISE_SUM_LEAF_ITEMS) pairwise_leaf(s, dtype);
            continue;
        }
        size_t n = 8 - s->used % 8;
        if (n > count - offset) n = count - offset;
        size_t first = s->used % 8;
        if (dtype == ME_FLOAT64) {
            for (size_t k = 0; k < n; k++) {
                double value = 0;
                if (!mask || mask[offset + k]) {
                    memcpy(&value, (const uint8_t *)values + (offset + k) * width, width);
                }
                s->lanes.f64[first + k] += value;
            }
        }
        else {
            for (size_t k = 0; k < n; k++) {
                float value = 0;
                if (!mask || mask[offset + k]) {
                    memcpy(&value, (const uint8_t *)values + (offset + k) * width, width);
                }
                s->lanes.f32[first + k] += value;
            }
        }
        offset += n;
        s->used += n;
        if (s->used == ME_PAIRWISE_SUM_LEAF_ITEMS) pairwise_leaf(s, dtype);
    }
}

void dsl_pairwise_sum_accumulate(me_pairwise_sum_state *s, me_dtype dtype, void *acc) {
    if (s->used) pairwise_leaf(s, dtype);
    /* Fold the newest (smallest) suffix into older, larger subtrees. */
    uint64_t tail = 0;
    bool found = false;
    for (size_t level = 0; level < sizeof(size_t) * CHAR_BIT; level++) {
        if (!(s->occupied & ((size_t)1 << level))) continue;
        if (found) pairwise_add_acc(dtype, &s->tree[level], tail);
        tail = s->tree[level];
        found = true;
    }
    if (found) pairwise_add_acc(dtype, acc, tail);
}

int me_pairwise_sum_init(me_pairwise_sum_state *s, me_dtype dtype) {
    if (!s || !pairwise_float(dtype)) return ME_PAIRWISE_SUM_ERR_INVALID_ARG;
    memset(s, 0, sizeof(*s));
    s->dtype = dtype;
    return ME_PAIRWISE_SUM_SUCCESS;
}

int me_pairwise_sum_update(me_pairwise_sum_state *s, const void *values, size_t count) {
    if (!s || !pairwise_float(s->dtype) || (count && !values) ||
        count > SIZE_MAX / pairwise_width(s->dtype)) return ME_PAIRWISE_SUM_ERR_INVALID_ARG;
    if (s->leaves > SIZE_MAX / ME_PAIRWISE_SUM_LEAF_ITEMS || s->used >= ME_PAIRWISE_SUM_LEAF_ITEMS ||
        count > SIZE_MAX - s->leaves * ME_PAIRWISE_SUM_LEAF_ITEMS - s->used) {
        return ME_PAIRWISE_SUM_ERR_INVALID_ARG;
    }
    dsl_pairwise_sum_feed(s, s->dtype, values, NULL, count);
    return ME_PAIRWISE_SUM_SUCCESS;
}

int me_pairwise_sum_append(me_pairwise_sum_state *s, const void *leaves, size_t count) {
    if (!s || !pairwise_float(s->dtype) || s->used || (count && !leaves)) {
        return ME_PAIRWISE_SUM_ERR_INVALID_ARG;
    }
    if (s->leaves > SIZE_MAX / ME_PAIRWISE_SUM_LEAF_ITEMS ||
        count > SIZE_MAX / ME_PAIRWISE_SUM_LEAF_ITEMS - s->leaves) {
        return ME_PAIRWISE_SUM_ERR_INVALID_ARG;
    }
    if (s->dtype == ME_FLOAT64) {
        const uint8_t *src = (const uint8_t *)leaves;
        size_t i = 0;
        while (i < count && (s->occupied & (size_t)7)) {
            double v;
            memcpy(&v, src + i * sizeof(double), sizeof(v));
            pairwise_push_f64_level(s, v, 0, 1);
            i++;
        }
        if ((s->occupied & (size_t)8) && count - i >= 8) {
            double v[8];
            memcpy(v, src + i * sizeof(double), sizeof(v));
            pairwise_push_f64_level(s, pairwise_fold8_f64(v), 3, 8);
            i += 8;
        }
        while (count - i >= 16) {
            double v0[8], v1[8];
            memcpy(v0, src + i * sizeof(double), sizeof(v0));
            memcpy(v1, src + (i + 8) * sizeof(double), sizeof(v1));
            pairwise_push_f64_level(s, pairwise_fold8_f64(v0) + pairwise_fold8_f64(v1), 4, 16);
            i += 16;
        }
        if (count - i >= 8) {
            double v[8];
            memcpy(v, src + i * sizeof(double), sizeof(v));
            pairwise_push_f64_level(s, pairwise_fold8_f64(v), 3, 8);
            i += 8;
        }
        while (i < count) {
            double v;
            memcpy(&v, src + i * sizeof(double), sizeof(v));
            pairwise_push_f64_level(s, v, 0, 1);
            i++;
        }
        return ME_PAIRWISE_SUM_SUCCESS;
    }
    const uint8_t *src = (const uint8_t *)leaves;
    size_t i = 0;
    while (i < count && (s->occupied & (size_t)7)) {
        float v;
        memcpy(&v, src + i * sizeof(float), sizeof(v));
        pairwise_push_f32_level(s, v, 0, 1);
        i++;
    }
    if ((s->occupied & (size_t)8) && count - i >= 8) {
        float v[8];
        memcpy(v, src + i * sizeof(float), sizeof(v));
        pairwise_push_f32_level(s, pairwise_fold8_f32(v), 3, 8);
        i += 8;
    }
    while (count - i >= 16) {
        float v0[8], v1[8];
        memcpy(v0, src + i * sizeof(float), sizeof(v0));
        memcpy(v1, src + (i + 8) * sizeof(float), sizeof(v1));
        pairwise_push_f32_level(s,
            pairwise_add_f32(pairwise_fold8_f32(v0), pairwise_fold8_f32(v1)), 4, 16);
        i += 16;
    }
    if (count - i >= 8) {
        float v[8];
        memcpy(v, src + i * sizeof(float), sizeof(v));
        pairwise_push_f32_level(s, pairwise_fold8_f32(v), 3, 8);
        i += 8;
    }
    while (i < count) {
        float v;
        memcpy(&v, src + i * sizeof(float), sizeof(v));
        pairwise_push_f32_level(s, v, 0, 1);
        i++;
    }
    return ME_PAIRWISE_SUM_SUCCESS;
}

int me_pairwise_sum_leaves(me_dtype dtype, const void *values,
    size_t count, void *leaves, size_t capacity) {
    size_t width = pairwise_width(dtype);
    if (!pairwise_float(dtype) || count % ME_PAIRWISE_SUM_LEAF_ITEMS || count > SIZE_MAX / width ||
        count / ME_PAIRWISE_SUM_LEAF_ITEMS > capacity / width ||
        (count && (!values || !leaves))) return ME_PAIRWISE_SUM_ERR_INVALID_ARG;
    size_t nleaves = count / ME_PAIRWISE_SUM_LEAF_ITEMS;
    if (dtype == ME_FLOAT64) {
        const uint8_t *src = (const uint8_t *)values;
        uint8_t *dst = (uint8_t *)leaves;
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
        if (me_pairwise_cpu_supports_avx2()) {
            pairwise_leaves_f64_avx2(src, dst, nleaves);
            return ME_PAIRWISE_SUM_SUCCESS;
        }
#endif
#if defined(ME_PAIRWISE_X86_64)
        pairwise_leaves_f64_sse2(src, dst, nleaves);
        return ME_PAIRWISE_SUM_SUCCESS;
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
        return ME_PAIRWISE_SUM_SUCCESS;
#endif
    }
    const uint8_t *src = (const uint8_t *)values;
    uint8_t *dst = (uint8_t *)leaves;
#if defined(ME_PAIRWISE_HAVE_AVX2_DISPATCH)
    if (me_pairwise_cpu_supports_avx2()) {
        pairwise_leaves_f32_avx2(src, dst, nleaves);
        return ME_PAIRWISE_SUM_SUCCESS;
    }
#endif
#if defined(ME_PAIRWISE_X86_64)
    pairwise_leaves_f32_sse2(src, dst, nleaves);
    return ME_PAIRWISE_SUM_SUCCESS;
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
    return ME_PAIRWISE_SUM_SUCCESS;
#endif
}

int me_pairwise_sum_finish(me_pairwise_sum_state *s, void *output) {
    if (!s || !pairwise_float(s->dtype) || !output) return ME_PAIRWISE_SUM_ERR_INVALID_ARG;
    memset(output, 0, pairwise_width(s->dtype));
    dsl_pairwise_sum_accumulate(s, s->dtype, output);
    return ME_PAIRWISE_SUM_SUCCESS;
}
