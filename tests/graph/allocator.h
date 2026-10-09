/* Force-include before applying allocator aliases. Renaming declarations inside
 * the Windows CRT headers creates imported allocator symbols with the wrong
 * linkage. The test callbacks are ordinary functions defined by the executable. */
#ifndef GRAPH_TEST_ALLOCATOR_H
#define GRAPH_TEST_ALLOCATOR_H
#include <stdlib.h>
void *graph_test_malloc(size_t n);
void *graph_test_calloc(size_t n, size_t width);
void *graph_test_realloc(void *p, size_t n);
void graph_test_free(void *p);
#define malloc graph_test_malloc
#define calloc graph_test_calloc
#define realloc graph_test_realloc
#define free graph_test_free
#endif
