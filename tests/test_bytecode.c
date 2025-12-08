/* Test bytecode compiler and fused evaluator */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "miniexpr.h"

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void test_correctness() {
    printf("=== Correctness Test ===\n");
    
    const int n = 10;
    double a[10], b[10], result_tree[10], result_fused[10];
    int i;
    
    for (i = 0; i < n; i++) {
        a[i] = i;
        b[i] = n - i;
    }
    
    me_variable vars[] = {{"a", a}, {"b", b}};
    int err;
    
    // Test sqrt(a*a + b*b)
    printf("\nExpression: sqrt(a*a + b*b)\n");
    me_expr *expr = me_compile("sqrt(a*a+b*b)", vars, 2, result_tree, n, ME_FLOAT64, &err);
    
    if (expr) {
        // Evaluate with tree-based method
        me_eval(expr);
        
        // Evaluate with fused bytecode
        expr->output = result_fused;
        me_eval_fused(expr);
        
        printf("Results comparison:\n");
        printf("  i   Tree     Fused    Expected  Match?\n");
        for (i = 0; i < n; i++) {
            double expected = sqrt(a[i]*a[i] + b[i]*b[i]);
            int match = (fabs(result_tree[i] - result_fused[i]) < 1e-10);
            printf("%3d  %7.2f  %7.2f  %7.2f   %s\n", 
                   i, result_tree[i], result_fused[i], expected,
                   match ? "✓" : "✗");
        }
        
        me_free(expr);
    }
    
    // Test a+5
    printf("\nExpression: a+5\n");
    expr = me_compile("a+5", vars, 1, result_tree, n, ME_FLOAT64, &err);
    
    if (expr) {
        me_eval(expr);
        expr->output = result_fused;
        me_eval_fused(expr);
        
        printf("Results comparison:\n");
        printf("  i   Tree     Fused    Expected  Match?\n");
        for (i = 0; i < n; i++) {
            double expected = a[i] + 5;
            int match = (fabs(result_tree[i] - result_fused[i]) < 1e-10);
            printf("%3d  %7.2f  %7.2f  %7.2f   %s\n",
                   i, result_tree[i], result_fused[i], expected,
                   match ? "✓" : "✗");
        }
        
        me_free(expr);
    }
}

void benchmark_comparison() {
    printf("\n\n=== Performance Benchmark (Comparing All Methods) ===\n");
    
    const int sizes[] = {1000, 10000, 100000, 1000000};
    int size_count = 4;
    
    for (int s = 0; s < size_count; s++) {
        int n = sizes[s];
        const int iterations = (n < 100000) ? 1000 : 100;
        
        double *a = malloc(n * sizeof(double));
        double *b = malloc(n * sizeof(double));
        double *result = malloc(n * sizeof(double));
        
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.1;
            b[i] = (n - i) * 0.1;
        }
        
        me_variable vars[] = {{"a", a}, {"b", b}};
        int err;
        
        printf("\n--- Vector size: %d, iterations: %d ---\n", n, iterations);
        printf("Expression: sqrt(a*a+b*b)\n\n");
        
        me_expr *expr = me_compile("sqrt(a*a+b*b)", vars, 2, result, n, ME_FLOAT64, &err);
        
        if (expr) {
            // Benchmark native C
            double start = get_time();
            for (int iter = 0; iter < iterations; iter++) {
                for (int i = 0; i < n; i++) {
                    result[i] = sqrt(a[i]*a[i] + b[i]*b[i]);
                }
            }
            double native_time = get_time() - start;
            
            // Benchmark tree-based evaluation
            start = get_time();
            for (int iter = 0; iter < iterations; iter++) {
                me_eval(expr);
            }
            double tree_time = get_time() - start;
            
            // Benchmark fused bytecode evaluation
            start = get_time();
            for (int iter = 0; iter < iterations; iter++) {
                me_eval_fused(expr);
            }
            double fused_time = get_time() - start;
            
            long long ops = (long long)iterations * n * 6; // 2 muls, 1 add, 1 sqrt
            
            printf("%-20s %.4f s  (%.2f GFLOPS)  [baseline]\n",
                   "Native C:", native_time, (ops / native_time) / 1e9);
            printf("%-20s %.4f s  (%.2f GFLOPS)  %.2fx vs C\n",
                   "Tree eval:", tree_time, (ops / tree_time) / 1e9, tree_time / native_time);
            printf("%-20s %.4f s  (%.2f GFLOPS)  %.2fx vs C\n",
                   "Fused bytecode:", fused_time, (ops / fused_time) / 1e9, fused_time / native_time);
            printf("\nSpeedups:\n");
            if (fused_time < tree_time) {
                printf("  Bytecode is %.2fx FASTER than tree eval ✓\n", tree_time / fused_time);
            } else {
                printf("  Bytecode is %.2fx slower than tree eval (needs work)\n", fused_time / tree_time);
            }
            
            me_free(expr);
        }
        
        free(a);
        free(b);
        free(result);
    }
}

int main() {
    printf("MiniExpr Bytecode Compiler Test\n");
    printf("================================\n\n");
    
    test_correctness();
    benchmark_comparison();
    
    printf("\n\nTest complete!\n");
    return 0;
}
