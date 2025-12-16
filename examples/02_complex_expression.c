/*
 * Example 2: Complex Expression
 *
 * Demonstrates complex mathematical expressions with multiple operations
 * and built-in functions.
 *
 * Physics formula: Distance traveled by projectile
 * d = v * t * cos(angle) - 0.5 * g * t * t
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/miniexpr.h"

int main() {
    printf("=== Complex Expression Example ===\n");
    printf("Projectile distance formula:\n");
    printf("d = v*t*cos(angle) - 0.5*g*t*t\n\n");

    // Simulation parameters
    const int n = 6;
    double v[] = {20.0, 20.0, 20.0, 20.0, 20.0, 20.0};  // velocity (m/s)
    double t[] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};        // time (seconds)
    double angle[] = {0.785, 0.785, 0.785, 0.785, 0.785, 0.785};  // 45° in radians
    double g[] = {9.81, 9.81, 9.81, 9.81, 9.81, 9.81};  // gravity (m/s²)
    double distance[6];

    // Define variables
    me_variable vars[] = {{"v"}, {"t"}, {"angle"}, {"g"}};

    // Compile complex expression
    int error;
    me_expr *expr = me_compile_chunk(
        "v*t*cos(angle) - 0.5*g*t*t",
        vars, 4, ME_FLOAT64, &error
    );

    if (!expr) {
        printf("ERROR: Failed to compile at position %d\n", error);
        return 1;
    }

    // Prepare variable pointers
    const void *var_ptrs[] = {v, t, angle, g};

    // Evaluate
    me_eval_chunk_threadsafe(expr, var_ptrs, 4, distance, n);

    // Display results
    printf("Projectile motion (v=20 m/s, angle=45°):\n");
    printf("  Time (s)  Distance (m)\n");
    printf("  --------  ------------\n");
    for (int i = 0; i < n; i++) {
        printf("    %4.1f      %8.2f\n", t[i], distance[i]);
    }

    // Cleanup
    me_free(expr);

    printf("\n✅ Complex expression with trigonometry complete!\n");
    return 0;
}
