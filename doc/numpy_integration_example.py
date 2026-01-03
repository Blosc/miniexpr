"""
Example Python bindings for miniexpr using NumPy conversion functions

This demonstrates how to use miniexpr_numpy.h in Python bindings
to seamlessly integrate with NumPy arrays.
"""

# Example using ctypes (simple approach)
# For production, use Cython, pybind11, or cffi

import ctypes
import numpy as np
from ctypes import c_void_p, c_int, c_char_p, POINTER

# Load the miniexpr library
# miniexpr = ctypes.CDLL('./libminiexpr.so')

# Example type conversion in Python
NUMPY_TO_ME_DTYPE = {
    0:  1,   # bool -> ME_BOOL
    1:  2,   # int8 -> ME_INT8
    2:  6,   # uint8 -> ME_UINT8
    3:  3,   # int16 -> ME_INT16
    4:  7,   # uint16 -> ME_UINT16
    5:  4,   # int32 -> ME_INT32
    6:  8,   # uint32 -> ME_UINT32
    7:  5,   # int64 -> ME_INT64
    8:  9,   # uint64 -> ME_UINT64
    11: 10,  # float32 -> ME_FLOAT32
    12: 11,  # float64 -> ME_FLOAT64
    14: 12,  # complex64 -> ME_COMPLEX64
    15: 13,  # complex128 -> ME_COMPLEX128
}

ME_DTYPE_TO_NUMPY = {v: k for k, v in NUMPY_TO_ME_DTYPE.items()}

def numpy_dtype_to_me(dtype):
    """Convert NumPy dtype to miniexpr dtype code"""
    numpy_num = dtype.num
    if numpy_num not in NUMPY_TO_ME_DTYPE:
        raise ValueError(f"Unsupported NumPy dtype: {dtype}")
    return NUMPY_TO_ME_DTYPE[numpy_num]

def me_dtype_to_numpy(me_dtype_code):
    """Convert miniexpr dtype code to NumPy dtype"""
    if me_dtype_code not in ME_DTYPE_TO_NUMPY:
        raise ValueError(f"Invalid miniexpr dtype: {me_dtype_code}")
    numpy_num = ME_DTYPE_TO_NUMPY[me_dtype_code]
    return np.dtype(f'i{numpy_num}')  # This is simplified

# Example usage with NumPy arrays
def example_usage():
    """
    Example of how miniexpr would be used with NumPy arrays
    """
    # Create NumPy arrays
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    result = np.zeros(5, dtype=np.float64)
    
    print("NumPy Integration Example")
    print("=" * 50)
    print(f"Array a: dtype={a.dtype}, dtype.num={a.dtype.num}")
    print(f"Array b: dtype={b.dtype}, dtype.num={b.dtype.num}")
    print(f"Result:  dtype={result.dtype}, dtype.num={result.dtype.num}")
    print()
    
    # Convert dtypes
    a_me_dtype = numpy_dtype_to_me(a.dtype)
    b_me_dtype = numpy_dtype_to_me(b.dtype)
    result_me_dtype = numpy_dtype_to_me(result.dtype)
    
    print(f"Converted to miniexpr:")
    print(f"  a: ME dtype = {a_me_dtype}")
    print(f"  b: ME dtype = {b_me_dtype}")
    print(f"  result: ME dtype = {result_me_dtype}")
    print()
    
    # In real bindings, you would call:
    # expr = miniexpr.me_compile(
    #     "sqrt(a*a + b*b)",
    #     variables=[
    #         {"name": "a", "dtype": a_me_dtype, "address": a.ctypes.data},
    #         {"name": "b", "dtype": b_me_dtype, "address": b.ctypes.data}
    #     ],
    #     output=result.ctypes.data,
    #     nitems=len(result),
    #     dtype=result_me_dtype
    # )
    # miniexpr.me_eval(expr)
    
    print("In real usage:")
    print("  expr = NULL")
    print("  me_compile('sqrt(a*a + b*b)', variables, ..., &err, &expr)")
    print("  me_eval(expr)")
    print("  # result array now contains computed values")

if __name__ == "__main__":
    example_usage()
    
    print()
    print("=" * 50)
    print("Python Binding Patterns:")
    print("=" * 50)
    print()
    print("1. Using C conversion functions (recommended):")
    print("   - Include miniexpr_numpy.h in your C extension")
    print("   - Use me_dtype_from_numpy(array.dtype.num)")
    print("   - Use me_dtype_to_numpy(expr->dtype)")
    print()
    print("2. Using Python dict mapping (shown above):")
    print("   - Pure Python, no C compilation needed")
    print("   - Good for prototyping")
    print()
    print("3. For Cython:")
    print("   cdef extern from 'miniexpr_numpy.h':")
    print("       int me_dtype_from_numpy(int)")
    print("       int me_dtype_to_numpy(int)")
