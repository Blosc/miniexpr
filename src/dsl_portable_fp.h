/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_PORTABLE_FP_H
#define MINIEXPR_DSL_PORTABLE_FP_H

#include <fenv.h>
#include <stdbool.h>
#include <float.h>

/* Portable strict operations run with nearest/ties-even and nontrapping IEEE
 * exceptions. Save/restore the calling thread's environment (including flags).
 * No compiled handle or process-global policy is mutated. */
static inline bool dsl_portable_fp_begin(fenv_t *saved) {
    if (feholdexcept(saved) != 0) return false;
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
    fesetenv(saved);
    return false;
}

static inline bool dsl_portable_fp_end(const fenv_t *saved) {
    return fesetenv(saved) == 0;
}

#endif
