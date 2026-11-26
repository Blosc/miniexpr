/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_TYPES_H
#define MINIEXPR_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ME_VERSION_MAJOR 0
#define ME_VERSION_MINOR 4
#define ME_VERSION_PATCH 0
#define ME_VERSION_STRING "0.4.0"

/* Stable 1.x dtype numbers. Source, storage and library versions are independent.
 * Strings use native NUL-padded fixed slots; Unicode slots contain UCS4 units. */
typedef enum {
    ME_AUTO = 0,
    ME_BOOL = 1,
    ME_INT8 = 2,
    ME_INT16 = 3,
    ME_INT32 = 4,
    ME_INT64 = 5,
    ME_UINT8 = 6,
    ME_UINT16 = 7,
    ME_UINT32 = 8,
    ME_UINT64 = 9,
    ME_FLOAT32 = 10,
    ME_FLOAT64 = 11,
    ME_COMPLEX64 = 12,
    ME_COMPLEX128 = 13,
    ME_STRING = 14,
    ME_BYTES = 15
} me_dtype;

/* Independent of platform FE_* constants. */
enum { ME_FP_INVALID = 1u, ME_FP_DIVIDE = 2u, ME_FP_OVERFLOW = 4u, ME_FP_UNDERFLOW = 8u };

#endif
