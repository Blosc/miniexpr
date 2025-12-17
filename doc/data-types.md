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

## Choosing the Right Data Type

| Type | Use When | Memory per Element |
|------|----------|-------------------|
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
