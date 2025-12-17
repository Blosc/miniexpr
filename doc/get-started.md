# Getting Started with miniexpr

This guide walks you through a simple example of using miniexpr to evaluate mathematical expressions.

## A Simple Example: Computing Distance

Let's create a program that computes `sqrt(a*a + b*b)` - the Euclidean distance formula - for arrays of values.

```c
#include <stdio.h>
#include <math.h>
#include "miniexpr.h"

int main() {
    // Input arrays
    double a[] = {3.0, 5.0, 8.0, 1.0};
    double b[] = {4.0, 12.0, 15.0, 1.0};
    int n = 4;

    // Output array
    double result[4];

    // Define variables (just names needed for simple case)
    me_variable vars[] = {{"a"}, {"b"}};

    // Compile the expression
    int error;
    me_expr *expr = me_compile("sqrt(a*a + b*b)", vars, 2, ME_FLOAT64, &error);

    if (!expr) {
        printf("Parse error at position %d\n", error);
        return 1;
    }

    // Prepare variable pointers
    const void *var_ptrs[] = {a, b};

    // Evaluate the expression (thread-safe)
    me_eval(expr, var_ptrs, 2, result, n);

    // Print results
    printf("Computing sqrt(a*a + b*b):\n");
    for (int i = 0; i < n; i++) {
        printf("a=%.1f, b=%.1f -> distance=%.2f\n",
               a[i], b[i], result[i]);
    }

    // Clean up
    me_free(expr);

    return 0;
}
```

### Expected Output

```
Computing sqrt(a*a + b*b):
a=3.0, b=4.0 -> distance=5.00
a=5.0, b=12.0 -> distance=13.00
a=8.0, b=15.0 -> distance=17.00
a=1.0, b=1.0 -> distance=1.41
```

## How It Works

1. **Define your data**: Create arrays for input variables (`a` and `b`) and output (`result`).

2. **Set up variables**: Create an array of `me_variable` structures. The simplest form only requires the variable name:
   - `name`: The identifier used in the expression

   For mixed types, also specify the dtype (see data-types.md).

3. **Compile the expression**: Call `me_compile()` with:
   - The expression string
   - The variables array (names only)
   - Number of variables
   - Output data type (or `ME_AUTO` to infer from mixed types)
   - Error position pointer

4. **Evaluate**: Call `me_eval()` to compute results. This function is thread-safe and can be used for both serial and parallel processing. You provide:
   - The compiled expression
   - Array of pointers to variable data
   - Number of variables
   - Output buffer pointer
   - Number of items to process

5. **Clean up**: Always call `me_free()` to release the compiled expression.

## Compiling the Example

If you have the miniexpr source files in your project:

```bash
gcc -o distance example.c miniexpr.c -lm
./distance
```

The `-lm` flag links the math library for functions like `sqrt()`.

## Next Steps

- Explore more complex expressions with multiple operations
- Try different data types (`ME_FLOAT32`, `ME_INT32`, etc.)
- Process large datasets by calling `me_eval()` multiple times with different chunks
- Use parallel processing by calling `me_eval()` from multiple threads (see parallel-processing.md)
