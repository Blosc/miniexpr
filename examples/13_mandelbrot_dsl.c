/*
 * Example 13: Mandelbrot Set with DSL Kernel
 *
 * Demonstrates the expressiveness of DSL kernels by computing the
 * Mandelbrot set using a single multi-statement program with loops,
 * conditionals, and temporary variables.
 *
 * The DSL kernel performs the complete iteration:
 *   for iter in range(max_iter):
 *       if escaped:
 *           break
 *       z = z^2 + c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"
#include "../src/dsl_parser.h"

/* ASCII art characters for different escape times */
static const char *CHARS = " .-:=+*#%@";
static const int NCHARS = 10;

/* Mandelbrot parameters */
#define WIDTH  78
#define HEIGHT 32
#define MAX_ITER 100

/* Complex plane bounds */
#define X_MIN -2.0
#define X_MAX  0.6
#define Y_MIN -1.1
#define Y_MAX  1.1

int main() {
    printf("=== Mandelbrot Set with DSL Kernel ===\n\n");

    /*
     * DSL Kernel for Mandelbrot iteration
     *
     * This kernel demonstrates:
     * - Temporary variables (zr, zi, zr2, zi2, mag2)
     * - For loop with iteration counter (Python-style syntax)
     * - Conditional break for block-level early escape
     * - Complex arithmetic decomposed into real/imaginary parts
     *
     * The kernel computes iteration count until escape (|z|² > 4)
     * for each point c = (cr, ci) in the complex plane.
     */
    const char *mandelbrot_dsl =
        "# Mandelbrot iteration: z(n+1) = z(n)^2 + c\n"
        "# Initialize z = 0\n"
        "zr = 0.0\n"
        "zi = 0.0\n"
        "escape_iter = 100.0\n"
        "\n"
        "# Main iteration loop (Python-style syntax)\n"
        "for iter in range(100):\n"
        "    # Compute |z|^2 for escape test\n"
        "    zr2 = zr * zr\n"
        "    zi2 = zi * zi\n"
        "    mag2 = zr2 + zi2\n"
        "    \n"
        "    # Record iteration count on first escape (per element)\n"
        "    just_escaped = mag2 > 4.0 and escape_iter == 100.0\n"
        "    escape_iter = where(just_escaped, iter, escape_iter)\n"
        "    \n"
        "    # Early exit when all points escaped (block-level)\n"
        "    if all(escape_iter != 100.0):\n"
        "        break\n"
        "    \n"
        "    # Compute z = z^2 + c\n"
        "    # Real: zr_new = zr^2 - zi^2 + cr\n"
        "    # Imag: zi_new = 2*zr*zi + ci\n"
        "    zr_new = zr2 - zi2 + cr\n"
        "    zi_new = 2.0 * zr * zi + ci\n"
        "    zr = zr_new\n"
        "    zi = zi_new\n"
        "\n"
        "# Output is the iteration count at escape\n"
        "result = escape_iter";

    printf("DSL Kernel:\n");
    printf("─────────────────────────────────────────────\n");
    printf("%s\n", mandelbrot_dsl);
    printf("─────────────────────────────────────────────\n\n");

    /* Parse the DSL program */
    me_dsl_error error;
    me_dsl_program *prog = me_dsl_parse(mandelbrot_dsl, &error);
    if (!prog) {
        printf("❌ Parse error at line %d, col %d: %s\n",
               error.line, error.column, error.message);
        return 1;
    }

    printf("✓ Parsed DSL program: %d top-level statements\n", prog->block.nstmts);

    /* Count statements including loop body */
    int total_stmts = 0;
    for (int i = 0; i < prog->block.nstmts; i++) {
        total_stmts++;
        if (prog->block.stmts[i]->kind == ME_DSL_STMT_FOR) {
            total_stmts += prog->block.stmts[i]->as.for_loop.body.nstmts;
        }
    }
    printf("✓ Total statements (including loop body): %d\n\n", total_stmts);

    int n = WIDTH * HEIGHT;

    /* Allocate arrays */
    double *cr = malloc(n * sizeof(double));
    double *ci = malloc(n * sizeof(double));
    double *iterations = malloc(n * sizeof(double));

    if (!cr || !ci || !iterations) {
        printf("Memory allocation failed!\n");
        me_dsl_program_free(prog);
        return 1;
    }

    /* Initialize the complex plane grid */
    printf("Initializing %dx%d grid (%d points)...\n", WIDTH, HEIGHT, n);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            cr[idx] = X_MIN + (X_MAX - X_MIN) * x / (WIDTH - 1);
            ci[idx] = Y_MAX - (Y_MAX - Y_MIN) * y / (HEIGHT - 1);
        }
    }

    /* Compile the DSL program */
    me_variable vars[] = {
        {"cr", ME_FLOAT64},
        {"ci", ME_FLOAT64}
    };
    me_expr *expr = NULL;
    int err = 0;
    if (me_compile(mandelbrot_dsl, vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) {
        printf("❌ Compile error at position %d\n", err);
        me_dsl_program_free(prog);
        free(cr);
        free(ci);
        free(iterations);
        return 1;
    }

    printf("Executing Mandelbrot computation (DSL eval)...\n");
    const void *inputs[] = {cr, ci};
    if (me_eval(expr, inputs, 2, iterations, n, NULL) != ME_EVAL_SUCCESS) {
        printf("❌ DSL evaluation failed\n");
        me_free(expr);
        me_dsl_program_free(prog);
        free(cr);
        free(ci);
        free(iterations);
        return 1;
    }

    /* Render the result as ASCII art */
    printf("\nMandelbrot Set (x: [%.1f, %.1f], y: [%.1f, %.1f]):\n\n",
           X_MIN, X_MAX, Y_MIN, Y_MAX);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            int iter = (int)iterations[idx];
            char ch;
            if (iter == MAX_ITER) {
                ch = '@';  /* Point is in the Mandelbrot set */
            } else {
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
        if ((int)iterations[i] == MAX_ITER) in_set++;
    }
    printf("\nPoints in set: %d / %d (%.1f%%)\n",
           in_set, n, 100.0 * in_set / n);

    /* Show DSL features demonstrated */
    printf("\n");
    printf("DSL Features Demonstrated:\n");
    printf("  ✓ Temporary variables: zr, zi, zr2, zi2, mag2, zr_new, zi_new\n");
    printf("  ✓ For loop: for iter in range(100)\n");
    printf("  ✓ Conditional break: if all(escape_iter != 100.0): break\n");
    printf("  ✓ Where conditional: where(just_escaped, iter, escape_iter)\n");
    printf("  ✓ Comments: # style comments\n");
    printf("  ✓ Multi-line program structure\n");

    /* Cleanup */
    me_free(expr);
    me_dsl_program_free(prog);
    free(cr);
    free(ci);
    free(iterations);

    printf("\n✅ DSL Mandelbrot example complete!\n");
    return 0;
}
