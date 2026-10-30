#ifndef MINIEXPR_MENUDET_POWI_H
#define MINIEXPR_MENUDET_POWI_H
#include <stdint.h>
#include <stdbool.h>

/* The caller qualifies a bounded, exactly represented integer exponent.
 * Squaring is also the strict constant-square policy; other chains require
 * explicit integer-power optimization (not global fast-math). */
#define ME_POWI(T,S) \
static inline T dsl_menudet_powi_##S(T base, int64_t exponent) { \
    if (exponent == 2) return base*base; \
    uint64_t n=(uint64_t)(exponent < 0 ? -exponent : exponent); \
    if (exponent < 0) base=(T)1/base; \
    T result=(T)1; \
    bool first=true; \
    while (n) { \
        if (n & 1) { \
            result=first ? base : result*base; \
            first=false; \
        } \
        n >>= 1; \
        if (n) base=base*base; \
    } \
    return result; \
}
ME_POWI(float,f32)
ME_POWI(double,f64)
#undef ME_POWI
#endif
