/*********************************************************************
  Copyright (c) 2026 Blosc Development Team <blosc@blosc.org>
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_MATH_POLICY_H
#define MINIEXPR_DSL_MATH_POLICY_H

#include "dsl_math_dispatch.h"
#include "dsl_parser.h"

#include <stdbool.h>
#include <stddef.h>

/* Resolve one immutable effective policy per prepared kernel.
 *
 * Precedence is: explicit source pragma, artifact declaration, environment,
 * built-in default. `auto` is the defined system-math policy and never
 * consults the environment. Errors are descriptive and occur before execution.
 */
bool dsl_math_policy_resolve(const me_dsl_program *parsed,
                             const me_dsl_declared_policy *declared,
                             me_dsl_compiler compiler,
                             me_dsl_fp_mode *fp_out,
                             me_dsl_math_backend *math_out,
                             me_dsl_sleef_ulp *ulp_out,
                             me_math_dispatch *dispatch,
                             char *error, size_t error_cap);

const char *dsl_math_backend_name(me_dsl_math_backend backend);
const char *dsl_sleef_ulp_name(me_dsl_sleef_ulp ulp);
bool dsl_math_backend_available(me_dsl_math_backend backend);

#endif
