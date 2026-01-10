# NumPy Integration for MiniExpr

This document explains how to integrate miniexpr with NumPy arrays in Python bindings.

## Overview

The file `miniexpr_numpy.h` provides conversion functions between miniexpr's dtype enum and NumPy's type number system, enabling seamless integration with NumPy arrays.

## Why Not Use NumPy Type Numbers Directly?

While it might seem convenient to use NumPy type numbers directly in miniexpr, there are good reasons not to:

1. **ME_AUTO Conflict**: miniexpr needs `ME_AUTO = 0` for automatic type inference, but NumPy uses `0` for bool
2. **Sparse Numbers**: NumPy has gaps (9, 10, 13) for unsupported types, which would waste memory
3. **Performance**: Dense enum values are faster for switch statements and array indexing
4. **Internal Efficiency**: The 13×13 type promotion table requires dense indexing

## API Functions

### `me_dtype_from_numpy(int numpy_type_num)`

Converts a NumPy `dtype.num` value to a miniexpr dtype.

```c
// In your Python C extension
PyArrayObject *array = ...; // Your NumPy array
int numpy_num = PyArray_TYPE(array);
me_dtype dtype = me_dtype_from_numpy(numpy_num);
if (dtype < 0) {
    // Unsupported type - error message already printed to stderr
    PyErr_SetString(PyExc_TypeError, "Unsupported NumPy dtype");
    return NULL;
}
```

**Returns:**
- The corresponding `me_dtype` value (0-13)
- `-1` for unsupported NumPy types (with error message to stderr)

### `me_dtype_to_numpy(me_dtype dtype)`

Converts a miniexpr dtype to a NumPy type number.

```c
me_expr *expr = NULL;
int err = 0;
if (me_compile(..., &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
int numpy_num = me_dtype_to_numpy(expr->dtype);
// Use numpy_num to create output array
```

**Returns:**
- NumPy type number (0-15)
- `-1` for `ME_AUTO`

### `me_numpy_type_supported(int numpy_type_num)`

Checks if a NumPy type is supported by miniexpr.

```c
if (!me_numpy_type_supported(PyArray_TYPE(array))) {
    PyErr_SetString(PyExc_TypeError, "Unsupported NumPy dtype");
    return NULL;
}
```

### `me_numpy_type_name(int numpy_type_num)`

Returns a human-readable name for error messages.

```c
const char *name = me_numpy_type_name(numpy_num);
fprintf(stderr, "Unsupported type: %s\n", name);
```

## Type Mapping

| NumPy Type   | dtype.num | ME dtype        | Supported |
|--------------|-----------|-----------------|-----------|
| bool         | 0         | ME_BOOL         | ✓         |
| int8         | 1         | ME_INT8         | ✓         |
| uint8        | 2         | ME_UINT8        | ✓         |
| int16        | 3         | ME_INT16        | ✓         |
| uint16       | 4         | ME_UINT16       | ✓         |
| int32        | 5         | ME_INT32        | ✓         |
| uint32       | 6         | ME_UINT32       | ✓         |
| int64        | 7         | ME_INT64        | ✓         |
| uint64       | 8         | ME_UINT64       | ✓         |
| float16      | 9         | -               | ✗         |
| longdouble   | 10        | -               | ✗         |
| float32      | 11        | ME_FLOAT32      | ✓         |
| float64      | 12        | ME_FLOAT64      | ✓         |
| clongdouble  | 13        | -               | ✗         |
| complex64    | 14        | ME_COMPLEX64    | ✓         |
| complex128   | 15        | ME_COMPLEX128   | ✓         |

## Usage Examples

### Using Cython

```cython
# mymodule.pyx
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "miniexpr_numpy.h":
    int me_dtype_from_numpy(int numpy_type_num)
    int me_dtype_to_numpy(int me_dtype)

cdef extern from "miniexpr.h":
    ctypedef struct me_expr:
        pass

    int me_compile(const char *expr, ..., int *error, me_expr **out)
    int me_eval(me_expr *expr, ...)
    void me_free(me_expr *expr)

def evaluate_expression(str expression, np.ndarray[double] a, np.ndarray[double] b):
    cdef np.ndarray[double] result = np.zeros_like(a)

    # Convert NumPy dtype to miniexpr dtype
    cdef int me_dtype = me_dtype_from_numpy(np.NPY_FLOAT64)

    # Setup and compile (simplified)
    cdef me_expr *expr = NULL
    if me_compile(expression.encode(), ..., &expr) != 0:
        # handle error
        return result

    # Evaluate
    me_eval(expr)  // returns ME_EVAL_SUCCESS on success

    # Cleanup
    me_free(expr)

    return result
```

### Using ctypes

```python
import ctypes
import numpy as np

# Load library
miniexpr = ctypes.CDLL('./libminiexpr.so')

# Setup function signatures
miniexpr.me_dtype_from_numpy.argtypes = [ctypes.c_int]
miniexpr.me_dtype_from_numpy.restype = ctypes.c_int

# Use it
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
me_dtype = miniexpr.me_dtype_from_numpy(a.dtype.num)
```

### Pure Python (for reference)

If you can't use C extensions, you can implement the mapping in pure Python:

```python
NUMPY_TO_ME = {
    0: 1,   # bool -> ME_BOOL
    1: 2,   # int8 -> ME_INT8
    7: 5,   # int64 -> ME_INT64
    12: 11, # float64 -> ME_FLOAT64
    # ... etc
}

me_dtype = NUMPY_TO_ME[array.dtype.num]
```

## Performance Notes

- Conversion overhead: ~2 CPU cycles (array lookup)
- Negligible compared to expression evaluation
- No runtime penalty for using conversion functions

## Testing

Run the test suite to verify conversions:

```bash
mkdir -p build
cd build
cmake ..
make -j
ctest -R test_numpy_conversion
```

All 27 tests should pass.

## See Also

- `doc/numpy_integration_example.py` - Complete example code
- `tests/test_numpy_conversion.c` - Comprehensive test suite
- NumPy dtype documentation: https://numpy.org/doc/stable/reference/arrays.dtypes.html
