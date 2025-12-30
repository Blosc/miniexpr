# Complex Expressions Tutorial

This tutorial demonstrates how to work with more complex mathematical expressions involving multiple operations and functions.

## Example: Computing a Physics Formula

Let's create a program that evaluates a physics formula combining trigonometry, exponents, and arithmetic operations. We'll compute the trajectory of a projectile:

**Formula**: `y = h + x*tan(angle) - (g*x*x)/(2*v*v*cos(angle)*cos(angle))`

Where:
- `h` = initial height
- `x` = horizontal distance
- `angle` = launch angle in radians
- `g` = gravitational acceleration (9.81 m/s²)
- `v` = initial velocity

```c
#include <stdio.h>
#include <math.h>
#include "miniexpr.h"

int main() {
    // Input parameters (arrays for vectorized evaluation)
    double h[] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double x[] = {0.0, 10.0, 20.0, 30.0, 40.0};
    double angle[] = {0.785, 0.785, 0.785, 0.785, 0.785}; // 45 degrees in radians
    double g[] = {9.81, 9.81, 9.81, 9.81, 9.81};
    double v[] = {20.0, 20.0, 20.0, 20.0, 20.0};
    int n = 5;

    // Output array
    double y[5];

    // Define variables
    me_variable vars[] = {
        {"h"},
        {"x"},
        {"angle"},
        {"g"},
        {"v"}
    };

    // Complex expression for projectile motion
    const char *expression =
        "h + x*tan(angle) - (g*x*x)/(2*v*v*cos(angle)*cos(angle))";

    // Compile the expression
    int error;
    me_expr *expr = NULL;
    if (me_compile(expression, vars, 5, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }

    // Prepare variable pointers
    const void *var_ptrs[] = {h, x, angle, g, v};

    // Evaluate the expression (thread-safe)
    if (me_eval(expr, var_ptrs, 5, y, n) != ME_EVAL_SUCCESS) { /* handle error */ }
    // Print results
    printf("Projectile Trajectory (v=20 m/s, angle=45°):\n");
    printf("Distance (m) | Height (m)\n");
    printf("-------------|------------\n");
    for (int i = 0; i < n; i++) {
        printf("%12.1f | %10.2f\n", x[i], y[i]);
    }

    // Clean up
    me_free(expr);

    return 0;
}
```

### Expected Output

```
Projectile Trajectory (v=20 m/s, angle=45°):
Distance (m) | Height (m)
-------------|------------
         0.0 |       0.00
        10.0 |       7.57
        20.0 |      10.20
        30.0 |       7.89
        40.0 |       0.65
```

## Breaking Down the Expression

The expression combines several mathematical operations:

1. **Trigonometric functions**: `tan(angle)` and `cos(angle)`
2. **Multiplication and division**: Multiple terms combined
3. **Exponentiation**: `x*x` for squaring
4. **Arithmetic**: Addition and subtraction

miniexpr handles operator precedence automatically, evaluating:
- Function calls first
- Multiplication and division (left to right)
- Addition and subtraction (left to right)

## More Complex Examples

### Example 2: Statistical Formula

Computing the standard score (z-score): `z = (x - mean) / stddev`

```c
const char *zscore = "(x - mean) / sqrt(variance)";
```

### Example 3: Financial Compound Interest

Formula: `A = P * (1 + r/n)^(n*t)`

```c
const char *compound = "P * pow(1 + r/n, n*t)";
```

### Example 4: Signal Processing

A damped sine wave: `y = A * exp(-decay*t) * sin(2*pi*f*t)`

```c
const char *damped_sine = "A * exp(-decay*t) * sin(2*3.14159265359*f*t)";
```

## Tips for Complex Expressions

1. **Use parentheses** to make precedence explicit and improve readability
2. **Break down** very complex formulas by computing intermediate results
3. **Test incrementally** - start with simpler versions and add complexity
4. **Check for errors** - always verify the `error` parameter after compilation
5. **Use constants** - define π, e, and other constants as variables if needed

## Available Functions

miniexpr supports many standard mathematical functions:
- Trigonometric: `sin`, `cos`, `tan`, `asin`/`arcsin`, `acos`/`arccos`, `atan`/`arctan`, `atan2`/`arctan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`/`arcsinh`, `acosh`/`arccosh`, `atanh`/`arctanh`
- Exponential/Logarithmic: `exp`, `log`, `log10`, `pow`
- Power/Root: `sqrt`, `pow`, `abs`
- Rounding: `floor`, `ceil`
- And more...

These can be combined freely to create sophisticated mathematical expressions.
