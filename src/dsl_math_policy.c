/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "dsl_math_policy.h"
#include "dsl_config.h"
#include "functions-simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ME_USE_SLEEF
#define ME_USE_SLEEF 0
#endif
#ifndef ME_USE_ACCELERATE
#define ME_USE_ACCELERATE 0
#endif

const char *dsl_math_backend_name(me_dsl_math_backend backend) {
    switch (backend) {
    case ME_DSL_MATH_AUTO: return "auto";
    case ME_DSL_MATH_SYSTEM: return "system";
    case ME_DSL_MATH_SLEEF: return "sleef";
    case ME_DSL_MATH_ACCELERATE: return "accelerate";
    case ME_DSL_MATH_SCALAR: return "scalar";
    default: return "unknown";
    }
}

const char *dsl_sleef_ulp_name(me_dsl_sleef_ulp ulp) {
    switch (ulp) {
    case ME_DSL_ULP_U10: return "u10";
    case ME_DSL_ULP_U35: return "u35";
    case ME_DSL_ULP_OMIT: return "";
    default: return "unknown";
    }
}

static void policy_error(char *error, size_t error_cap, const char *message) {
    if (error && error_cap) {
        snprintf(error, error_cap, "%s", message);
    }
}

bool dsl_math_backend_available(me_dsl_math_backend backend) {
    me_math_dispatch probe;
    switch (backend) {
    case ME_DSL_MATH_AUTO:
    case ME_DSL_MATH_SYSTEM:
    case ME_DSL_MATH_SCALAR:
        return true;
    case ME_DSL_MATH_SLEEF:
        if (!ME_USE_SLEEF) return false;
        return me_math_dispatch_init(&probe, backend, ME_DSL_ULP_U35);
    case ME_DSL_MATH_ACCELERATE:
#if ME_USE_ACCELERATE && defined(__APPLE__)
        return true;
#else
        return false;
#endif
    default:
        return false;
    }
}

static bool parse_fp_mode(const char *value, me_dsl_fp_mode *out) {
    if (!strcmp(value, "strict")) { *out = ME_DSL_FP_STRICT; return true; }
    if (!strcmp(value, "contract")) { *out = ME_DSL_FP_CONTRACT; return true; }
    if (!strcmp(value, "fast") || !strcmp(value, "relaxed")) { *out = ME_DSL_FP_FAST; return true; }
    return false;
}

static bool parse_math_backend(const char *value, me_dsl_math_backend *out) {
    if (!strcmp(value, "auto")) { *out = ME_DSL_MATH_AUTO; return true; }
    if (!strcmp(value, "system")) { *out = ME_DSL_MATH_SYSTEM; return true; }
    if (!strcmp(value, "sleef")) { *out = ME_DSL_MATH_SLEEF; return true; }
    if (!strcmp(value, "accelerate")) { *out = ME_DSL_MATH_ACCELERATE; return true; }
    if (!strcmp(value, "scalar")) { *out = ME_DSL_MATH_SCALAR; return true; }
    return false;
}

static bool env_fp_mode(me_dsl_fp_mode *out, bool *present, char *error, size_t error_cap) {
    const char *value = getenv("ME_DSL_FP_MODE");
    *present = value && value[0];
    if (!*present) return true;
    if (!parse_fp_mode(value, out)) {
        if (error && error_cap) {
            snprintf(error, error_cap,
                     "invalid ME_DSL_FP_MODE='%s' (expected 'strict', 'contract', or 'fast')", value);
        }
        return false;
    }
    return true;
}

static bool env_math_backend(me_dsl_math_backend *out, bool *present, char *error, size_t error_cap) {
    const char *value = me_jit_explicit_math_backend();
    if (!value) value = getenv("ME_SIMD_MATH_BACKEND");
    *present = value && value[0];
    if (!*present) return true;
    if (!parse_math_backend(value, out)) {
        if (error && error_cap) {
            snprintf(error, error_cap,
                     "invalid ME_SIMD_MATH_BACKEND='%s' "
                     "(expected 'auto', 'system', 'sleef', 'accelerate', or 'scalar')", value);
        }
        return false;
    }
    return true;
}

bool dsl_math_policy_resolve(const me_dsl_program *parsed,
                             const me_dsl_declared_policy *declared,
                             me_dsl_compiler compiler,
                             me_dsl_fp_mode *fp_out,
                             me_dsl_math_backend *math_out,
                             me_dsl_sleef_ulp *ulp_out,
                             me_math_dispatch *dispatch,
                             char *error, size_t error_cap) {
    if (!parsed || !fp_out || !math_out || !ulp_out || !dispatch) {
        policy_error(error, error_cap, "invalid math policy resolution request");
        return false;
    }
    if (error && error_cap) error[0] = '\0';

    me_dsl_fp_mode fp = ME_DSL_FP_STRICT;
    me_dsl_math_backend math = ME_DSL_MATH_AUTO;
    me_dsl_sleef_ulp ulp = ME_DSL_ULP_OMIT;
    bool fp_explicit = false, math_explicit = false, ulp_explicit = false;

    if (parsed->fp_explicit) {
        fp = parsed->fp_mode;
        fp_explicit = true;
    }
    else if (declared && declared->has_fp) {
        fp = declared->fp;
        fp_explicit = true;
    }
    else {
        bool present = false;
        if (!env_fp_mode(&fp, &present, error, error_cap)) return false;
        fp_explicit = present;
    }

    if (parsed->math_explicit) {
        math = parsed->math;
        math_explicit = true;
    }
    else if (declared && declared->has_math) {
        math = declared->math;
        math_explicit = true;
    }
    else {
        bool present = false;
        if (!env_math_backend(&math, &present, error, error_cap)) return false;
        math_explicit = present;
    }

    if (parsed->ulp_explicit) {
        ulp = parsed->ulp;
        ulp_explicit = true;
    }
    else if (declared && declared->has_ulp) {
        ulp = declared->ulp;
        ulp_explicit = true;
    }

    if (ulp_explicit && math != ME_DSL_MATH_SLEEF) {
        if (error && error_cap) {
            snprintf(error, error_cap,
                     "me:ulp is valid only with the SLEEF backend (effective math backend is '%s')",
                     dsl_math_backend_name(math));
        }
        return false;
    }
    if (math == ME_DSL_MATH_SLEEF && ulp == ME_DSL_ULP_OMIT) {
        ulp = ME_DSL_ULP_U35;
    }

    if (!dsl_math_backend_available(math)) {
        if (error && error_cap) {
            if (math == ME_DSL_MATH_SLEEF) {
                snprintf(error, error_cap,
                         "requested math backend 'sleef' is unavailable on this build/platform/CPU; "
                         "supported alternatives: system, accelerate (macOS builds), scalar");
            }
            else if (math == ME_DSL_MATH_ACCELERATE) {
                snprintf(error, error_cap,
                         "requested math backend 'accelerate' requires macOS and an Accelerate-enabled build; "
                         "supported alternatives: system, sleef (when built and supported)");
            }
            else {
                snprintf(error, error_cap,
                         "requested math backend '%s' is unavailable; supported alternatives: system",
                         dsl_math_backend_name(math));
            }
        }
        return false;
    }

    if (compiler == ME_DSL_COMPILER_LIBTCC && fp != ME_DSL_FP_STRICT) {
        policy_error(error, error_cap,
                     "the TCC JIT backend supports strict floating point only; use "
                     "'# me:compiler=cc' with me:fp=contract|fast");
        return false;
    }

    if (!me_math_dispatch_init(dispatch, math, ulp)) {
        policy_error(error, error_cap,
                     "requested math backend is unavailable for per-kernel dispatch");
        return false;
    }

    /* Keep `auto` and `system` distinct only for explicit policy reporting;
     * both execute the same system-math path and share cache identity. */
    *fp_out = fp;
    *math_out = math;
    *ulp_out = ulp;
    (void)fp_explicit;
    (void)math_explicit;
    return true;
}
