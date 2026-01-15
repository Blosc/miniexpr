/* Benchmark: 2 * exp(x) with multi-threaded evaluation */

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "miniexpr.h"
#include "functions-simd.h"

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
    const me_eval_params *params;
} thread_args_t;

typedef struct {
    const void *data;
    void *output;
    const dtype_info_t *info;
    int start_idx;
    int count;
} thread_args_c_t;

static void fill_data(void *data, const dtype_info_t *info, int nitems) {
    const double min = -5.0;
    const double max = 5.0;
    const double step = (max - min) / (double)(nitems ? nitems : 1);

    if (info->dtype == ME_FLOAT32) {
        float *f = (float *)data;
        for (int i = 0; i < nitems; i++) {
            f[i] = (float)(min + step * (double)i);
        }
    } else {
        double *d = (double *)data;
        for (int i = 0; i < nitems; i++) {
            d[i] = min + step * (double)i;
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

    int rc = me_eval(args->expr, vars_chunk, 1,
                     out_base + (size_t)args->start_idx * args->elem_size,
                     args->count, args->params);
    if (rc != ME_EVAL_SUCCESS) {
        fprintf(stderr, "me_eval failed: %d\n", rc);
        exit(1);
    }
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
            o[idx] = 2.0f * expf(a[idx]);
        }
    } else {
        const double *a = (const double *)args->data;
        double *o = (double *)args->output;
        for (int i = 0; i < count; i++) {
            int idx = start + i;
            o[idx] = 2.0 * exp(a[idx]);
        }
    }

    return NULL;
}

static void run_threads_me(const me_expr *expr, const void *data, void *out,
                           size_t elem_size, int total_elems, int num_threads,
                           const me_eval_params *params) {
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
        thread_args[t].params = params;
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
                               int num_threads, int iterations,
                               const me_eval_params *params) {
    run_threads_me(expr, data, out, elem_size, total_elems, num_threads, params);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads_me(expr, data, out, elem_size, total_elems, num_threads, params);
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

    me_variable vars[] = {{"x", info->dtype, data}};
    int err = 0;
    me_expr *expr = NULL;
    int rc_expr = me_compile("2 * exp(x)", vars, 1, info->dtype, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile exp expression for %s (err=%d)\n", info->name, err);
        free(data);
        free(out);
        return;
    }

    double data_gb = (double)(total_elems * info->elem_size) / 1e9;

    printf("\n========================================\n");
    printf("2 * exp(x) (%s, GB/s)\n", info->name);
    printf("========================================\n");
    printf("Threads   ME_U10    ME_U35  ME_SCAL       C\n");

    me_eval_params params_u10 = ME_EVAL_PARAMS_DEFAULTS;
    params_u10.simd_ulp_mode = ME_SIMD_ULP_1;
    me_eval_params params_u35 = ME_EVAL_PARAMS_DEFAULTS;
    params_u35.simd_ulp_mode = ME_SIMD_ULP_3_5;
    me_eval_params params_scalar = ME_EVAL_PARAMS_DEFAULTS;
    params_scalar.disable_simd = true;

    me_simd_params_state simd_state;
    me_simd_params_push(&params_u10, &simd_state);
    printf("Backend U10: %s (mode=%s)\n",
           me_simd_backend_label(),
           me_simd_use_u35_flag() ? "u35" : "u10");
    me_simd_params_pop(&simd_state);

    me_simd_params_push(&params_u35, &simd_state);
    printf("Backend U35: %s (mode=%s)\n",
           me_simd_backend_label(),
           me_simd_use_u35_flag() ? "u35" : "u10");
    me_simd_params_pop(&simd_state);

    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        double me_time_u10 = run_benchmark_me(expr, data, out, info->elem_size,
                                              total_elems, num_threads, 5, &params_u10);
        double me_time_u35 = run_benchmark_me(expr, data, out, info->elem_size,
                                              total_elems, num_threads, 5, &params_u35);
        double me_scalar_time = run_benchmark_me(expr, data, out, info->elem_size,
                                                 total_elems, num_threads, 5, &params_scalar);
        double c_time = run_benchmark_c(data, out, info, total_elems,
                                        num_threads, 5);
        printf("%7d  %7.2f  %7.2f  %7.2f  %7.2f\n",
               num_threads,
               data_gb / me_time_u10,
               data_gb / me_time_u35,
               data_gb / me_scalar_time,
               data_gb / c_time);
    }

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
    printf("MiniExpr exp Benchmark (Threads)\n");
    printf("========================================\n");
    printf("Expression: 2 * exp(x)\n");
    printf("Total elements: %d\n", total_elems);

    for (size_t i = 0; i < sizeof(infos) / sizeof(infos[0]); i++) {
        benchmark_dtype(&infos[i], total_elems);
    }

    printf("\n========================================\n");
    printf("Benchmark complete\n");
    printf("========================================\n");
    return 0;
}
