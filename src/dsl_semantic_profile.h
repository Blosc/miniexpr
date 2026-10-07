/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_DSL_SEMANTIC_PROFILE_H
#define MINIEXPR_DSL_SEMANTIC_PROFILE_H

/* Per-compilation semantics, never a mutable process-wide switch. */
typedef enum {
    ME_DSL_PROFILE_FULL = 0,
    ME_DSL_PROFILE_PORTABLE_0_1,
    ME_DSL_PROFILE_PORTABLE_1_0
} me_dsl_semantic_profile;

#endif
