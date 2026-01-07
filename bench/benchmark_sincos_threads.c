/* Benchmark: sin^2 + cos^2 with multi-threaded evaluation */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include "miniexpr.h"
#include "minctest.h"

#define MAX_THREADS 12

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef struct {
    const char *name;
    me_dtype dtype;
    size_t elem_size;
} dtype_info_t;

typedef struct {
    const me_expr *expr;
    const void *data;
    void *output;
    size_t elem_size;
    int start_idx;
    int count;
} thread_args_t;

typedef struct {
    const void *data;
    void *output;
    const dtype_info_t *info;
    int start_idx;
    int count;
} thread_args_c_t;

static void fill_data(void *data, const dtype_info_t *info, int nitems) {
    if (info->dtype == ME_FLOAT32) {
        float *f = (float *)data;
        for (int i = 0; i < nitems; i++) {
            f[i] = (float)(i % 1024) * 0.001f + 0.1f;
        }
    } else {
        double *d = (double *)data;
        for (int i = 0; i < nitems; i++) {
            d[i] = (double)(i % 1024) * 0.001 + 0.1;
        }
    }
}

static void *eval_worker(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    const unsigned char *base = (const unsigned char *)args->data;
    const void *vars_chunk[1] = {
        base + (size_t)args->start_idx * args->elem_size
    };
    unsigned char *out_base = (unsigned char *)args->output;

    ME_EVAL_CHECK(args->expr, vars_chunk, 1,
                  out_base + (size_t)args->start_idx * args->elem_size,
                  args->count);
    return NULL;
}

static void *eval_worker_c(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    const dtype_info_t *info = args->info;
    const int start = args->start_idx;
    const int count = args->count;

    if (info->dtype == ME_FLOAT32) {
        const float *a = (const float *)args->data;
        float *o = (float *)args->output;
        for (int i = 0; i < count; i++) {
            int idx = start + i;
            float s = sinf(a[idx]);
            float c = cosf(a[idx]);
            o[idx] = s * s + c * c;
        }
    } else {
        const double *a = (const double *)args->data;
        double *o = (double *)args->output;
        for (int i = 0; i < count; i++) {
            int idx = start + i;
            double s = sin(a[idx]);
            double c = cos(a[idx]);
            o[idx] = s * s + c * c;
        }
    }

    return NULL;
}

static void run_threads_me(const me_expr *expr, const void *data, void *out,
                           size_t elem_size, int total_elems, int num_threads) {
    pthread_t threads[MAX_THREADS];
    thread_args_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].expr = expr;
        thread_args[t].data = data;
        thread_args[t].output = out;
        thread_args[t].elem_size = elem_size;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        offset += count;

        pthread_create(&threads[t], NULL, eval_worker, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

static void run_threads_c(const void *data, void *out, const dtype_info_t *info,
                          int total_elems, int num_threads) {
    pthread_t threads[MAX_THREADS];
    thread_args_c_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].data = data;
        thread_args[t].output = out;
        thread_args[t].info = info;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        offset += count;

        pthread_create(&threads[t], NULL, eval_worker_c, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

static double run_benchmark_me(const me_expr *expr, const void *data, void *out,
                               size_t elem_size, int total_elems,
                               int num_threads, int iterations) {
    run_threads_me(expr, data, out, elem_size, total_elems, num_threads);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads_me(expr, data, out, elem_size, total_elems, num_threads);
    }
    return (get_time() - start) / iterations;
}

static double run_benchmark_c(const void *data, void *out, const dtype_info_t *info,
                              int total_elems, int num_threads, int iterations) {
    run_threads_c(data, out, info, total_elems, num_threads);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads_c(data, out, info, total_elems, num_threads);
    }
    return (get_time() - start) / iterations;
}

static void benchmark_dtype(const dtype_info_t *info, int total_elems) {
    void *data = malloc((size_t)total_elems * info->elem_size);
    void *out = malloc((size_t)total_elems * info->elem_size);
    if (!data || !out) {
        printf("Allocation failed for %s\n", info->name);
        free(data);
        free(out);
        return;
    }

    fill_data(data, info, total_elems);

    me_variable vars[] = {{"a", info->dtype, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("sin(a) * sin(a) + cos(a) * cos(a)",
                             vars, 1, info->dtype, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile sin/cos expression for %s (err=%d)\n", info->name, err);
        free(data);
        free(out);
        return;
    }

    double data_gb = (double)(total_elems * info->elem_size * 2ULL) / 1e9;

    printf("\n========================================\n");
    printf("sin^2 + cos^2 (%s, GB/s)\n", info->name);
    printf("========================================\n");
    printf("Threads   ME_U10    ME_U35  ME_SCAL       C\n");
    printf("Backend U10: %s\n", me_get_sincos_backend());
    me_set_sincos_ulp(35);
    printf("Backend U35: %s\n", me_get_sincos_backend());
    me_set_sincos_ulp(10);

    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        me_set_sincos_simd(1);
        me_set_sincos_ulp(10);
        double me_time_u10 = run_benchmark_me(expr, data, out, info->elem_size,
                                              total_elems, num_threads, 5);
        me_set_sincos_ulp(35);
        double me_time_u35 = run_benchmark_me(expr, data, out, info->elem_size,
                                              total_elems, num_threads, 5);
        me_set_sincos_simd(0);
        double me_scalar_time = run_benchmark_me(expr, data, out, info->elem_size,
                                                 total_elems, num_threads, 5);
        double c_time = run_benchmark_c(data, out, info, total_elems,
                                        num_threads, 5);
        printf("%7d  %7.2f  %7.2f  %7.2f  %7.2f\n",
               num_threads,
               data_gb / me_time_u10,
               data_gb / me_time_u35,
               data_gb / me_scalar_time,
               data_gb / c_time);
    }

    me_set_sincos_simd(1);
    me_set_sincos_ulp(10);
    me_free(expr);
    free(data);
    free(out);
}

int main(void) {
    const dtype_info_t infos[] = {
        {"float32", ME_FLOAT32, sizeof(float)},
        {"float64", ME_FLOAT64, sizeof(double)}
    };
    const int total_elems = 8 * 1024 * 1024;

    printf("========================================\n");
    printf("MiniExpr sin/cos Benchmark (Threads)\n");
    printf("========================================\n");
    printf("Expression: sin(a)^2 + cos(a)^2\n");
    printf("Total elements: %d\n", total_elems);

    for (size_t i = 0; i < sizeof(infos) / sizeof(infos[0]); i++) {
        benchmark_dtype(&infos[i], total_elems);
    }

    printf("\n========================================\n");
    printf("Benchmark complete\n");
    printf("========================================\n");

    return 0;
}
