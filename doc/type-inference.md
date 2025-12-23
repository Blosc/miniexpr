# Type Inference and Constants

## Important: How ME_AUTO Works with Constants

When you use `ME_AUTO` as the output dtype in `me_compile()`, miniexpr automatically infers the result type. **For constants in expressions, the type is inferred from your variables.**

### The Rule

**Constants inherit the type of the first variable when output dtype is ME_AUTO.**

This ensures type consistency and prevents unexpected type promotions.

## Example: FLOAT32 Variables with Constants

```c
// Input: float32 array
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};

// Variable with explicit dtype
me_variable vars[] = {{"x", ME_FLOAT32}};

// Expression with constant
me_expr *expr = me_compile("x + 3.0", vars, 1, ME_AUTO, &err);

// Result: Constants inferred as FLOAT32
// me_get_dtype(expr) returns ME_FLOAT32 ✓
```

### What Happens

1. You specify variable `x` as `ME_FLOAT32`
2. You use `ME_AUTO` for output
3. miniexpr infers constant `3.0` as `FLOAT32` (matching variable type)
4. Result is `FLOAT32` (no unexpected promotion to FLOAT64)

## When This Matters

This is especially important for:

### 1. **Memory Efficiency with FLOAT32**

```c
// Working with single-precision arrays
float positions[1000000];  // 4MB instead of 8MB

me_variable vars[] = {{"pos", ME_FLOAT32}};
me_expr *expr = me_compile("pos * 2.5 + 1.0", vars, 1, ME_AUTO, &err);

// Constants 2.5 and 1.0 are FLOAT32
// Result is FLOAT32 → saves memory!
```

### 2. **NumPy Compatibility**

```python
# Python/NumPy code
import numpy as np
data = np.array([1, 2, 3], dtype=np.float32)  # float32 array

# miniexpr with ME_AUTO matches this dtype
# Constants in "data + 3.0" are treated as float32
```

### 3. **GPU/SIMD Optimization**

Many hardware accelerators work best with consistent types:

```c
// All FLOAT32 → can use SIMD instructions
float a[N], result[N];
me_variable vars[] = {{"a", ME_FLOAT32}};
me_expr *expr = me_compile("sqrt(a*a + 2.5)", vars, 1, ME_AUTO, &err);
// Entire computation in FLOAT32 → faster on GPU/SIMD
```

## Explicit Type Control

If you want different behavior, specify the output dtype explicitly. There are two modes:

### Mode 1: All Variables ME_AUTO + Explicit Output

When all variables are `ME_AUTO` and you specify an output dtype, all variables use that type:

```c
// All variables use FLOAT64 (homogeneous)
me_variable vars[] = {{"x"}, {"y"}};  // Both ME_AUTO
me_expr *expr = me_compile("x + y", vars, 2, ME_FLOAT64, &err);
// Result: FLOAT64 (all variables treated as FLOAT64)
```

### Mode 2: Explicit Variable Types + Explicit Output

When variables have explicit types and you specify an output dtype, variables keep their types during computation, and the result is cast to the output type:

```c
// Variables keep their types, result is cast to FLOAT64
me_variable vars[] = {{"x", ME_FLOAT32}, {"y", ME_FLOAT32}};
me_expr *expr = me_compile("x + 3.0", vars, 2, ME_FLOAT64, &err);
// Computation: FLOAT32 + FLOAT32 → FLOAT32
// Output: Cast to FLOAT64
// Result: FLOAT64 (cast from FLOAT32 computation)
```

This is useful for:
- **Memory efficiency**: Compute in FLOAT32, output in FLOAT64 when needed
- **Heterogeneous inputs**: Mixed types (INT32 + FLOAT64) with specific output requirements
- **Type safety**: Explicit control over both input and output types

See `examples/08_explicit_output_dtype.c` for complete examples.

## Best Practices

### ✅ Recommended: Use ME_AUTO with Explicit Variable Dtypes

```c
me_variable vars[] = {
    {"temperature", ME_FLOAT32},
    {"pressure", ME_FLOAT32}
};
me_expr *expr = me_compile("temperature * 1.8 + 32.0", vars, 1, ME_AUTO, &err);
// Constants match variable type → consistent FLOAT32
```

### ✅ For Mixed Types, Still Use ME_AUTO

```c
me_variable vars[] = {
    {"count", ME_INT32},
    {"price", ME_FLOAT64}
};
me_expr *expr = me_compile("count * price * 1.08", vars, 2, ME_AUTO, &err);
// Constants infer from first variable (INT32)
// But expression promotes to FLOAT64 due to mixed types
// Result: FLOAT64 ✓
```

### ❌ Avoid: Mixing Variable Types Without Explicit Dtypes

```c
// DON'T DO THIS:
me_variable vars[] = {{"x"}};  // No dtype specified!
me_expr *expr = me_compile("x + 3.0", vars, 1, ME_AUTO, &err);
// Ambiguous: what type is x? what type is 3.0?
```

## Summary Table

| Scenario | Variable Dtype | Output Dtype | Constant Type | Result Type |
|----------|---------------|--------------|---------------|-------------|
| Float32 + const | `ME_FLOAT32` | `ME_AUTO` | `FLOAT32` | `FLOAT32` ✓ |
| Float64 + const | `ME_FLOAT64` | `ME_AUTO` | `FLOAT64` | `FLOAT64` ✓ |
| Int32 + const | `ME_INT32` | `ME_AUTO` | `INT32` | `INT32` ✓ |
| Float32 + const | `ME_FLOAT32` | `ME_FLOAT64` | `FLOAT64` | `FLOAT64` ✓ |
| Mixed types | Both explicit | `ME_AUTO` | Matches 1st var | Promoted as needed ✓ |
| All ME_AUTO | All `ME_AUTO` | `ME_FLOAT64` | `FLOAT64` | `FLOAT64` ✓ (homogeneous) |
| Explicit vars | Both explicit | `ME_FLOAT32` | Matches vars | `FLOAT32` ✓ (cast) |

## See Also

- `examples/03_mixed_types.c` - Complete example with ME_AUTO
- `examples/08_explicit_output_dtype.c` - Explicit variable types + explicit output
- `doc/data-types.md` - Full type system documentation
- `tests/test_constant_type_inference.c` - Test validating this behavior
