/*
 * Example 12: Mandelbrot Set Computation
 *
 * Computes the Mandelbrot set using miniexpr for the core iteration.
 * Demonstrates complex number arithmetic via real/imaginary components.
 *
 * The Mandelbrot set is defined as the set of complex numbers c for which
 * the iteration z(n+1) = z(n)^2 + c does not diverge to infinity.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"

/* ASCII art characters for different escape times (dark to light) */
static const char *CHARS = " .-:=+*#%@";
static const int NCHARS = 10;

/* Mandelbrot parameters */
#define WIDTH  78
#define HEIGHT 32
#define MAX_ITER 100

/* Complex plane bounds - classic Mandelbrot view */
#define X_MIN -2.0
#define X_MAX  0.6
#define Y_MIN -1.1
#define Y_MAX  1.1

int main() {
    printf("=== Mandelbrot Set Example ===\n\n");

    int n = WIDTH * HEIGHT;

    /* Allocate arrays for the complex plane grid */
    double *cr = malloc(n * sizeof(double));      /* Real part of c */
    double *ci = malloc(n * sizeof(double));      /* Imaginary part of c */
    double *zr = malloc(n * sizeof(double));      /* Real part of z */
    double *zi = malloc(n * sizeof(double));      /* Imaginary part of z */
    double *zr_new = malloc(n * sizeof(double));  /* New zr after iteration */
    double *zi_new = malloc(n * sizeof(double));  /* New zi after iteration */
    double *escape = malloc(n * sizeof(double));  /* |z|^2 */
    int *iterations = malloc(n * sizeof(int));

    if (!cr || !ci || !zr || !zi || !zr_new || !zi_new || !escape || !iterations) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    /* Initialize the complex plane grid */
    printf("Initializing %dx%d grid (%d points)...\n", WIDTH, HEIGHT, n);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            cr[idx] = X_MIN + (X_MAX - X_MIN) * x / (WIDTH - 1);
            ci[idx] = Y_MAX - (Y_MAX - Y_MIN) * y / (HEIGHT - 1);  /* Flip Y */
            zr[idx] = 0.0;
            zi[idx] = 0.0;
            iterations[idx] = MAX_ITER;  /* Assume in set until proven otherwise */
        }
    }

    /* Compile expressions using miniexpr
     *
     * The iteration is: z = z^2 + c
     * In components:
     *   zr_new = zr^2 - zi^2 + cr
     *   zi_new = 2*zr*zi + ci
     *   |z|^2  = zr^2 + zi^2
     *
     * Note: Each expression must be compiled with exactly the variables it uses.
     * miniexpr validates that n_vars matches actual variable count.
     */

    me_variable vars_iter[] = {
        {"zr"}, {"zi"}, {"cr"}, {"ci"}
    };
    me_variable vars_escape[] = {
        {"zr"}, {"zi"}
    };

    int err;
    me_expr *expr_zr_new = NULL;   /* zr^2 - zi^2 + cr (uses 3 vars: zr, zi, cr) */
    me_expr *expr_zi_new = NULL;   /* 2*zr*zi + ci (uses 3 vars: zr, zi, ci) */
    me_expr *expr_escape = NULL;   /* zr^2 + zi^2 (uses 2 vars: zr, zi) */

    /* For zr_new: need zr, zi, cr - but we'll add a +0*ci to use all 4 */
    if (me_compile("zr*zr - zi*zi + cr + 0*ci", vars_iter, 4, ME_FLOAT64, &err, &expr_zr_new) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile zr_new expression (err=%d)\n", err);
        return 1;
    }
    /* For zi_new: need zr, zi, ci - but we'll add +0*cr to use all 4 */
    if (me_compile("2*zr*zi + ci + 0*cr", vars_iter, 4, ME_FLOAT64, &err, &expr_zi_new) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile zi_new expression (err=%d)\n", err);
        return 1;
    }
    /* For escape: only needs zr, zi */
    if (me_compile("zr*zr + zi*zi", vars_escape, 2, ME_FLOAT64, &err, &expr_escape) != ME_COMPILE_SUCCESS) {
        printf("Failed to compile escape expression (err=%d)\n", err);
        return 1;
    }

    printf("Running up to %d iterations per point...\n", MAX_ITER);

    /* Perform the Mandelbrot iteration */
    for (int iter = 0; iter < MAX_ITER; iter++) {
        const void *ptrs_iter[] = {zr, zi, cr, ci};
        const void *ptrs_escape[] = {zr, zi};

        /* Check escape condition first */
        me_eval(expr_escape, ptrs_escape, 2, escape, n, NULL);

        /* Mark escaped points */
        for (int i = 0; i < n; i++) {
            if (iterations[i] == MAX_ITER && escape[i] > 4.0) {
                iterations[i] = iter;
            }
        }

        /* Compute next iteration values for all points
         * IMPORTANT: Compute both zr_new and zi_new BEFORE updating either,
         * since they depend on current zr and zi values */
        me_eval(expr_zr_new, ptrs_iter, 4, zr_new, n, NULL);
        me_eval(expr_zi_new, ptrs_iter, 4, zi_new, n, NULL);

        /* Copy new values for non-escaped points */
        for (int i = 0; i < n; i++) {
            if (iterations[i] == MAX_ITER) {
                zr[i] = zr_new[i];
                zi[i] = zi_new[i];
            }
        }
    }

    /* Render the result as ASCII art */
    printf("\nMandelbrot Set (x: [%.1f, %.1f], y: [%.1f, %.1f]):\n\n",
           X_MIN, X_MAX, Y_MIN, Y_MAX);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            int iter = iterations[idx];
            char ch;
            if (iter == MAX_ITER) {
                ch = '@';  /* Point is in the set */
            } else {
                /* Map iteration count to character */
                int char_idx = (iter * (NCHARS - 1)) / MAX_ITER;
                ch = CHARS[char_idx];
            }
            putchar(ch);
        }
        putchar('\n');
    }

    /* Statistics */
    int in_set = 0;
    for (int i = 0; i < n; i++) {
        if (iterations[i] == MAX_ITER) in_set++;
    }
    printf("\nPoints in set: %d / %d (%.1f%%)\n",
           in_set, n, 100.0 * in_set / n);

    /* Cleanup */
    me_free(expr_zr_new);
    me_free(expr_zi_new);
    me_free(expr_escape);
    free(cr);
    free(ci);
    free(zr);
    free(zi);
    free(zr_new);
    free(zi_new);
    free(escape);
    free(iterations);

    printf("\n✅ Mandelbrot computation complete!\n");
    return 0;
}
