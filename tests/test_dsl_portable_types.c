/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#include "dsl_portable_types.h"
#include <stdio.h>

/* Rows/columns: bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64.
 * Explicit expected matrix is independent of the implementation algorithm. */
static const me_dtype types[] = {
    ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
    ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64
};
#define I8 ME_INT8
#define I16 ME_INT16
#define I32 ME_INT32
#define I64 ME_INT64
#define U8 ME_UINT8
#define U16 ME_UINT16
#define U32 ME_UINT32
#define U64 ME_UINT64
#define F32 ME_FLOAT32
#define F64 ME_FLOAT64
#define X ME_AUTO
static const me_dtype expected[11][11] = {
    {I64,I64,I64,I64,I64,I64,I64,I64,X,F64,F64},
    {I64,I8,I16,I32,I64,I16,I32,I64,X,F32,F64},
    {I64,I16,I16,I32,I64,I16,I32,I64,X,F32,F64},
    {I64,I32,I32,I32,I64,I32,I32,I64,X,F64,F64},
    {I64,I64,I64,I64,I64,I64,I64,I64,X,F64,F64},
    {I64,I16,I16,I32,I64,U8,U16,U32,U64,F32,F64},
    {I64,I32,I32,I32,I64,U16,U16,U32,U64,F32,F64},
    {I64,I64,I64,I64,I64,U32,U32,U32,U64,F64,F64},
    {X,X,X,X,X,U64,U64,U64,U64,F64,F64},
    {F64,F32,F32,F64,F64,F32,F32,F64,F64,F32,F64},
    {F64,F64,F64,F64,F64,F64,F64,F64,F64,F64,F64}
};

int main(void) {
    for (int i = 0; i < 11; i++) {
        for (int j = 0; j < 11; j++) {
            me_dtype actual = dsl_portable_numeric_promote(types[i], types[j]);
            me_dtype division = expected[i][j];
            if (division != ME_AUTO && division != ME_FLOAT32) division = ME_FLOAT64;
            if (actual != expected[i][j] ||
                dsl_portable_division_dtype(types[i], types[j]) != division) {
                fprintf(stderr, "portable promotion failed at row %d column %d\n", i, j);
                return 1;
            }
        }
    }
    const me_dtype excluded[] = {ME_AUTO, ME_STRING, ME_BYTES, ME_COMPLEX64, ME_COMPLEX128};
    for (size_t i = 0; i < sizeof(excluded) / sizeof(excluded[0]); i++) {
        if (dsl_portable_numeric_promote(excluded[i], ME_INT64) != ME_AUTO ||
            dsl_portable_numeric_promote(ME_FLOAT32, excluded[i]) != ME_AUTO ||
            dsl_portable_division_dtype(excluded[i], ME_FLOAT64) != ME_AUTO) {
            fprintf(stderr, "excluded dtype admitted\n");
            return 1;
        }
    }
    puts("portable 1.0 promotion matrix: 121 pairs passed");
    return 0;
}
