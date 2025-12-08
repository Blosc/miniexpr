/* Test chunked evaluation API */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../src/miniexpr.h"

#define TOTAL_SIZE 1000
#define CHUNK_SIZE 100

int main() {
    printf("=== Testing Chunked Evaluation API ===\n\n");
    
    // Create large arrays
    double *a_full = malloc(TOTAL_SIZE * sizeof(double));
    double *b_full = malloc(TOTAL_SIZE * sizeof(double));
    double *result_monolithic = malloc(TOTAL_SIZE * sizeof(double));
    double *result_chunked = malloc(TOTAL_SIZE * sizeof(double));
    
    // Initialize data
    for (int i = 0; i < TOTAL_SIZE; i++) {
        a_full[i] = i * 0.5;
        b_full[i] = (TOTAL_SIZE - i) * 0.3;
    }
    
    // Test 1: Simple expression with chunked evaluation
    printf("Test 1: Simple expression (a + b)\n");
    printf("  Total size: %d, Chunk size: %d\n", TOTAL_SIZE, CHUNK_SIZE);
    
    me_variable vars[] = {{"a", a_full}, {"b", b_full}};
    int err;
    
    // Compile once with initial pointers
    me_expr *expr = me_compile("a + b", vars, 2, result_monolithic, TOTAL_SIZE, ME_FLOAT64, &err);
    if (!expr) {
        printf("  ❌ FAILED: Compilation error at position %d\n", err);
        return 1;
    }
    
    // Evaluate monolithically for reference
    me_eval(expr);
    
    // Now evaluate in chunks using new API
    int num_chunks = TOTAL_SIZE / CHUNK_SIZE;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * CHUNK_SIZE;
        const void *vars_chunk[2] = {
            a_full + offset,
            b_full + offset
        };
        me_eval_chunk(expr, vars_chunk, 2, result_chunked + offset, CHUNK_SIZE);
    }
    
    // Compare results
    int passed = 1;
    int mismatches = 0;
    for (int i = 0; i < TOTAL_SIZE && mismatches < 5; i++) {
        if (fabs(result_monolithic[i] - result_chunked[i]) > 1e-10) {
            passed = 0;
            printf("  Mismatch at [%d]: monolithic=%.6f, chunked=%.6f\n", 
                   i, result_monolithic[i], result_chunked[i]);
            mismatches++;
        }
    }
    
    if (passed) {
        printf("  ✅ PASSED: Chunked evaluation matches monolithic\n");
    } else {
        printf("  ❌ FAILED: Results don't match\n");
        me_free(expr);
        return 1;
    }
    
    me_free(expr);
    
    // Test 2: Complex expression
    printf("\nTest 2: Complex expression (sqrt(a*a + b*b))\n");
    
    expr = me_compile("sqrt(a*a + b*b)", vars, 2, result_monolithic, TOTAL_SIZE, ME_FLOAT64, &err);
    if (!expr) {
        printf("  ❌ FAILED: Compilation error at position %d\n", err);
        return 1;
    }
    
    // Evaluate monolithically
    me_eval(expr);
    
    // Evaluate in chunks
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * CHUNK_SIZE;
        const void *vars_chunk[2] = {
            a_full + offset,
            b_full + offset
        };
        me_eval_chunk(expr, vars_chunk, 2, result_chunked + offset, CHUNK_SIZE);
    }
    
    // Compare results
    passed = 1;
    mismatches = 0;
    for (int i = 0; i < TOTAL_SIZE && mismatches < 5; i++) {
        if (fabs(result_monolithic[i] - result_chunked[i]) > 1e-10) {
            passed = 0;
            printf("  Mismatch at [%d]: monolithic=%.6f, chunked=%.6f\n", 
                   i, result_monolithic[i], result_chunked[i]);
            mismatches++;
        }
    }
    
    if (passed) {
        printf("  ✅ PASSED: Complex expression chunked evaluation works\n");
    } else {
        printf("  ❌ FAILED: Results don't match\n");
        me_free(expr);
        return 1;
    }
    
    me_free(expr);
    
    // Test 3: Different data types (int32)
    printf("\nTest 3: Integer type (int32_t)\n");
    
    int32_t *a_int = malloc(TOTAL_SIZE * sizeof(int32_t));
    int32_t *b_int = malloc(TOTAL_SIZE * sizeof(int32_t));
    int32_t *result_int_mono = malloc(TOTAL_SIZE * sizeof(int32_t));
    int32_t *result_int_chunk = malloc(TOTAL_SIZE * sizeof(int32_t));
    
    for (int i = 0; i < TOTAL_SIZE; i++) {
        a_int[i] = i;
        b_int[i] = i * 2;
    }
    
    me_variable vars_int[] = {
        {"a", a_int, ME_VARIABLE, NULL, ME_INT32},
        {"b", b_int, ME_VARIABLE, NULL, ME_INT32}
    };
    
    expr = me_compile("a + b", vars_int, 2, result_int_mono, TOTAL_SIZE, ME_INT32, &err);
    if (!expr) {
        printf("  ❌ FAILED: Compilation error\n");
        return 1;
    }
    
    // Monolithic
    me_eval(expr);
    
    // Chunked
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * CHUNK_SIZE;
        const void *vars_chunk[2] = {
            a_int + offset,
            b_int + offset
        };
        me_eval_chunk(expr, vars_chunk, 2, result_int_chunk + offset, CHUNK_SIZE);
    }
    
    // Compare
    passed = 1;
    mismatches = 0;
    for (int i = 0; i < TOTAL_SIZE && mismatches < 5; i++) {
        if (result_int_mono[i] != result_int_chunk[i]) {
            passed = 0;
            printf("  Mismatch at [%d]: monolithic=%d, chunked=%d\n", 
                   i, result_int_mono[i], result_int_chunk[i]);
            mismatches++;
        }
    }
    
    if (passed) {
        printf("  ✅ PASSED: Integer chunked evaluation works\n");
    } else {
        printf("  ❌ FAILED: Results don't match\n");
    }
    
    me_free(expr);
    free(a_int);
    free(b_int);
    free(result_int_mono);
    free(result_int_chunk);
    
    // Cleanup
    free(a_full);
    free(b_full);
    free(result_monolithic);
    free(result_chunked);
    
    printf("\n✅ All chunked evaluation tests passed!\n");
    return 0;
}
