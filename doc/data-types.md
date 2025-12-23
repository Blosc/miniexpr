# Different Data Types Tutorial

This tutorial shows how to work with various numeric data types in miniexpr, from integers to floating-point numbers.

## Type System Overview

miniexpr automatically promotes types when necessary. The general promotion hierarchy is:
- Integer types → Float types
- Lower precision → Higher precision
- Real types → Complex types (when complex is involved)

## Example 1: Integer Operations

Let's compute the area of rectangles using 32-bit integers:

```c
#include <stdio.h>
#include <stdint.h>
#include "miniexpr.h"

int main() {
    // Input arrays (32-bit integers)
    int32_t width[] = {5, 10, 15, 20};
    int32_t height[] = {3, 7, 12, 8};
    int n = 4;

    // Output array (also 32-bit integers)
    int32_t area[4];

    // Define variables (names only for uniform type)
    me_variable vars[] = {{"width"}, {"height"}};

    // Compile the expression
    int error;
    me_expr *expr = me_compile("width * height", vars, 2, ME_INT32, &error);

    if (!expr) {
        printf("Parse error at position %d\n", error);
        return 1;
    }

    // Evaluate
    const void *var_ptrs[] = {width, height};
    me_eval(expr, var_ptrs, 2, area, n);

    // Print results
    printf("Rectangle Areas (INT32):\n");
    for (int i = 0; i < n; i++) {
        printf("Width=%d, Height=%d -> Area=%d\n",
               width[i], height[i], area[i]);
    }

    me_free(expr);
    return 0;
}
```

### Expected Output

```
Rectangle Areas (INT32):
Width=5, Height=3 -> Area=15
Width=10, Height=7 -> Area=70
Width=15, Height=12 -> Area=180
Width=20, Height=8 -> Area=160
```

## Example 2: Single-Precision Floating Point

For applications where memory is constrained or when double precision isn't needed:

```c
#include <stdio.h>
#include "miniexpr.h"

int main() {
    // Using float (32-bit) instead of double (64-bit)
    float radius[] = {1.0f, 2.5f, 5.0f, 10.0f};
    int n = 4;

    float area[4];

    me_variable vars[] = {{"r"}};

    // Circle area: π * r²
    // Using an approximation of π
    int error;
    me_expr *expr = me_compile("3.14159265 * r * r", vars, 1, ME_FLOAT32, &error);

    if (!expr) {
        printf("Parse error at position %d\n", error);
        return 1;
    }

    const void *var_ptrs[] = {radius};
    me_eval(expr, var_ptrs, 1, area, n);

    printf("Circle Areas (FLOAT32):\n");
    for (int i = 0; i < n; i++) {
        printf("Radius=%.2f -> Area=%.2f\n", radius[i], area[i]);
    }

    me_free(expr);
    return 0;
}
```

### Expected Output

```
Circle Areas (FLOAT32):
Radius=1.00 -> Area=3.14
Radius=2.50 -> Area=19.63
Radius=5.00 -> Area=78.54
Radius=10.00 -> Area=314.16
```

## Example 3: Mixed Type Operations

When you mix types, miniexpr automatically promotes to the appropriate type:

```c
#include <stdio.h>
#include <stdint.h>
#include "miniexpr.h"

int main() {
    // Integer counts
    int32_t items[] = {10, 25, 50, 100};

    // Floating-point prices
    double price[] = {2.99, 15.50, 7.25, 0.99};

    int n = 4;

    // Result will be double (promoted from int32)
    double total[4];

    // For mixed types, specify explicit dtypes and use ME_AUTO for output
    me_variable vars[] = {
        {"items", ME_INT32},
        {"price", ME_FLOAT64}
    };

    // Calculate total cost with 8% tax
    // Using ME_AUTO lets the library infer the result type (ME_FLOAT64)
    int error;
    me_expr *expr = me_compile("items * price * 1.08", vars, 2, ME_AUTO, &error);

    if (!expr) {
        printf("Parse error at position %d\n", error);
        return 1;
    }

    const void *var_ptrs[] = {items, price};
    me_eval(expr, var_ptrs, 2, total, n);

    printf("Shopping Cart (Mixed Types):\n");
    for (int i = 0; i < n; i++) {
        printf("Items=%d × $%.2f = $%.2f (with tax)\n",
               items[i], price[i], total[i]);
    }

    me_free(expr);
    return 0;
}
```

### Expected Output

```
Shopping Cart (Mixed Types):
Items=10 × $2.99 = $32.29 (with tax)
Items=25 × $15.50 = $418.50 (with tax)
Items=50 × $7.25 = $391.50 (with tax)
Items=100 × $0.99 = $106.92 (with tax)
```

## Example 4: Unsigned Integers

Working with unsigned types for values that are always positive:

```c
#include <stdio.h>
#include <stdint.h>
#include "miniexpr.h"

int main() {
    // Pixel values (0-255)
    uint8_t red[] = {255, 128, 64, 32};
    uint8_t green[] = {0, 64, 128, 192};
    uint8_t blue[] = {0, 0, 64, 128};
    int n = 4;

    // Grayscale conversion: 0.299*R + 0.587*G + 0.114*B
    uint8_t gray[4];

    me_variable vars[] = {{"r"}, {"g"}, {"b"}};

    int error;
    me_expr *expr = me_compile("0.299*r + 0.587*g + 0.114*b", vars, 3, ME_UINT8, &error);

    if (!expr) {
        printf("Parse error at position %d\n", error);
        return 1;
    }

    const void *var_ptrs[] = {red, green, blue};
    me_eval(expr, var_ptrs, 3, gray, n);

    printf("RGB to Grayscale (UINT8):\n");
    for (int i = 0; i < n; i++) {
        printf("RGB(%3d,%3d,%3d) -> Gray=%3d\n",
               red[i], green[i], blue[i], gray[i]);
    }

    me_free(expr);
    return 0;
}
```

### Expected Output

```
RGB to Grayscale (UINT8):
RGB(255,  0,  0) -> Gray= 76
RGB(128, 64,  0) -> Gray= 75
RGB( 64,128, 64) -> Gray=100
RGB( 32,192,128) -> Gray=136
```

## Example 5: Boolean Output from Comparisons

Comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) can output boolean arrays. This is useful for filtering, masking, or conditional operations.

```c
#include <stdio.h>
#include <stdbool.h>
#include "miniexpr.h"

int main() {
    // Sample data where a² = a + b for all elements
    double a[] = {2.0, 3.0, 4.0, 5.0, 6.0};
    double b[] = {2.0, 6.0, 12.0, 20.0, 30.0};
    int n = 5;

    // Boolean output array
    bool is_equal[5];

    // Method 1: Explicit variable dtypes with ME_BOOL output
    // Use this when input types differ from output type
    me_variable vars[] = {
        {"a", ME_FLOAT64},
        {"b", ME_FLOAT64}
    };

    int error;
    me_expr *expr = me_compile("a ** 2 == (a + b)", vars, 2, ME_BOOL, &error);

    if (!expr) {
        printf("Parse error at position %d\n", error);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    me_eval(expr, var_ptrs, 2, is_equal, n);

    printf("Comparison Results (BOOL):\n");
    for (int i = 0; i < n; i++) {
        printf("a=%.1f: a² (%.1f) == a+b (%.1f) -> %s\n",
               a[i], a[i]*a[i], a[i]+b[i],
               is_equal[i] ? "true" : "false");
    }

    me_free(expr);
    return 0;
}
```

### Expected Output

```
Comparison Results (BOOL):
a=2.0: a² (4.0) == a+b (4.0) -> true
a=3.0: a² (9.0) == a+b (9.0) -> true
a=4.0: a² (16.0) == a+b (16.0) -> true
a=5.0: a² (25.0) == a+b (25.0) -> true
a=6.0: a² (36.0) == a+b (36.0) -> true
```

### Two Ways to Get Boolean Output

**Method 1: Explicit variable dtypes with ME_BOOL output**
```c
me_variable vars[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};
me_expr *expr = me_compile("x < y", vars, 2, ME_BOOL, &error);
```

**Method 2: ME_AUTO output (auto-infers ME_BOOL for comparisons)**
```c
me_variable vars[] = {{"x", ME_FLOAT64}, {"y", ME_FLOAT64}};
me_expr *expr = me_compile("x < y", vars, 2, ME_AUTO, &error);
// me_get_dtype(expr) == ME_BOOL  (automatically inferred)
```

Both methods require explicit variable dtypes when the computation type differs from the output type.

## Example 6: Explicit Variable Types with Explicit Output Dtype

You can specify both explicit variable types AND an explicit output dtype. This is useful when you want variables to keep their types during computation, but cast the final result to a specific output type.

```c
#include <stdio.h>
#include <stdint.h>
#include "miniexpr.h"

int main() {
    // Mixed input types
    int32_t a[] = {10, 20, 30, 40, 50};
    double b[] = {1.5, 2.5, 3.5, 4.5, 5.5};

    // Output as FLOAT32 (even though computation uses FLOAT64)
    float result[5];

    me_variable vars[] = {
        {"a", ME_INT32},
        {"b", ME_FLOAT64}
    };

    // Explicit variable types + explicit output dtype
    int error;
    me_expr *expr = me_compile("a + b", vars, 2, ME_FLOAT32, &error);

    if (!expr) {
        printf("Parse error at position %d\n", error);
        return 1;
    }

    const void *var_ptrs[] = {a, b};
    me_eval(expr, var_ptrs, 2, result, 5);

    printf("Mixed Types with FLOAT32 Output:\n");
    for (int i = 0; i < 5; i++) {
        printf("a=%d + b=%.1f = %.2f (FLOAT32)\n", a[i], b[i], result[i]);
    }

    me_free(expr);
    return 0;
}
```

### When to Use This Pattern

**Use explicit variable types + explicit output dtype when:**
- You have heterogeneous input types but need a specific output type
- You want to compute in one precision but output in another (e.g., FLOAT32 computation, FLOAT64 output)
- You need explicit control over type casting for memory efficiency or compatibility

**Key behavior:**
- Variables keep their explicit types during computation
- Type promotion happens as needed (e.g., INT32 + FLOAT64 → FLOAT64)
- Final result is cast to your specified output dtype

**Comparison with ME_AUTO:**
- `ME_AUTO`: Output type is inferred from the expression computation
- Explicit output: Output type is forced to your specification (with casting)

## Choosing the Right Data Type

| Type | Use When | Memory per Element |
|------|----------|-------------------|
| `ME_BOOL` | Comparison results, flags, masks | 1 byte |
| `ME_INT8` / `ME_UINT8` | Small integers, flags, pixel values | 1 byte |
| `ME_INT16` / `ME_UINT16` | Medium-range integers | 2 bytes |
| `ME_INT32` / `ME_UINT32` | Standard integers | 4 bytes |
| `ME_INT64` / `ME_UINT64` | Large integers, timestamps | 8 bytes |
| `ME_FLOAT32` | Moderate precision, memory-constrained | 4 bytes |
| `ME_FLOAT64` | High precision (default) | 8 bytes |

**Tips:**
- Use the smallest type that fits your data range to save memory
- Use unsigned types when values are always non-negative
- Use FLOAT64 for scientific computing requiring high precision
- Use FLOAT32 for graphics or when processing large arrays
- Use ME_BOOL for comparison expressions to get proper boolean output
