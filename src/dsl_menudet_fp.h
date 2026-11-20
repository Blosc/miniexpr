/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_MENUDET_FP_H
#define MINIEXPR_DSL_MENUDET_FP_H

#include <fenv.h>
#include <stdbool.h>
#include <stdint.h>
#include <float.h>

/* Scoped per-thread collector. Flags use the public stable 1/2/4/8 encoding,
 * never platform FE_* values. Nesting restores the previous collector. */
unsigned *dsl_menudet_status_begin(unsigned *flags);
void dsl_menudet_status_end(unsigned *previous);
void dsl_menudet_status_capture(void);
void dsl_menudet_status_record(unsigned flags);
/* Shared floating comparison bridge; operands are widened exactly as in the
 * menudet evaluator. Passed at invocation, never baked into disk-cache code. */
bool dsl_menudet_float_compare(const void *node, double x, double y);

typedef struct {
    fenv_t env;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t fpsr;
    bool fast;
#endif
} dsl_menudet_fp_state;

/* Menudet strict operations run with nearest/ties-even and nontrapping IEEE
 * exceptions. Save/restore the calling thread's environment (including flags).
 * No compiled handle or process-global policy is mutated. */
static inline bool dsl_menudet_fp_begin(dsl_menudet_fp_state *saved) {
#ifdef __EMSCRIPTEN__
    (void)saved;
    return true;
#else
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr) :: "memory");
    /* Fast path when RMode=nearest(0), FZ=0, FZ16=0, DN=0, AHP=0, and all traps disabled. */
    if ((fpcr & 0x07C89F00ULL) == 0) {
        uint64_t fpsr;
        __asm__ __volatile__("mrs %0, fpsr" : "=r"(fpsr) :: "memory");
        saved->fpsr = fpsr;
        saved->fast = true;
        if (fpsr != 0) {
            __asm__ __volatile__("msr fpsr, %0" :: "r"((uint64_t)0) : "memory");
        }
        return true;
    }
    saved->fast = false;
#endif
    if (feholdexcept(&saved->env) != 0) return false;
    if (fesetround(FE_TONEAREST) == 0 && fegetround() == FE_TONEAREST) {
        /* fenv does not portably expose flush-to-zero/denormals-are-zero.
         * Verify gradual underflow in this thread instead of silently running
         * with a non-IEEE mode or mutating an architecture-specific switch. */
        volatile float fsmall = FLT_MIN, fhalf = 0.5f;
        volatile double dsmall = DBL_MIN, dhalf = 0.5;
        volatile float fsub = fsmall * fhalf;
        volatile double dsub = dsmall * dhalf;
        volatile float fback = fsub / fhalf;
        volatile double dback = dsub / dhalf;
        if (fsub != 0.0f && dsub != 0.0 && fback == fsmall && dback == dsmall) return true;
    }
    fesetenv(&saved->env);
    return false;
#endif
}

static inline bool dsl_menudet_fp_end(const dsl_menudet_fp_state *saved) {
#ifdef __EMSCRIPTEN__
    (void)saved;
    return true;
#else
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    if (saved->fast) {
        uint64_t cur_fpsr;
        __asm__ __volatile__("mrs %0, fpsr" : "=r"(cur_fpsr) :: "memory");
        if ((cur_fpsr & 0x0FULL) != 0) {
            dsl_menudet_status_record((unsigned)(cur_fpsr & 0x0FULL));
        }
        if (cur_fpsr != saved->fpsr) {
            __asm__ __volatile__("msr fpsr, %0" :: "r"(saved->fpsr) : "memory");
        }
        return true;
    }
#endif
    dsl_menudet_status_capture();
    return fesetenv(&saved->env) == 0;
#endif
}

#endif
