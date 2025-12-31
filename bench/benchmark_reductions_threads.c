/* Benchmark: Multi-threaded sum reductions */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "miniexpr.h"
#include "minctest.h"



#define MAX_THREADS 12

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

typedef enum {
    RED_SUM,
    RED_PROD,
    RED_MIN,
    RED_MAX
} reduction_kind_t;

typedef struct {
    const char *name;
    me_dtype dtype;
    size_t elem_size;
    bool is_float;
    bool is_signed;
} dtype_info_t;

typedef struct {
    const me_expr *expr;
    const void *data;
    size_t elem_size;
    size_t output_stride;
    int start_idx;
    int count;
    void *output;
} thread_args_t;

typedef struct {
    const void *data;
    int start_idx;
    int count;
    void *output;
    const dtype_info_t *info;
    reduction_kind_t kind;
} thread_args_c_t;

static bool parse_dtype(const char *name, dtype_info_t *info) {
    if (strcmp(name, "int8") == 0) {
        *info = (dtype_info_t){name, ME_INT8, sizeof(int8_t), false, true};
    } else if (strcmp(name, "int16") == 0) {
        *info = (dtype_info_t){name, ME_INT16, sizeof(int16_t), false, true};
    } else if (strcmp(name, "int32") == 0) {
        *info = (dtype_info_t){name, ME_INT32, sizeof(int32_t), false, true};
    } else if (strcmp(name, "int64") == 0) {
        *info = (dtype_info_t){name, ME_INT64, sizeof(int64_t), false, true};
    } else if (strcmp(name, "uint8") == 0) {
        *info = (dtype_info_t){name, ME_UINT8, sizeof(uint8_t), false, false};
    } else if (strcmp(name, "uint16") == 0) {
        *info = (dtype_info_t){name, ME_UINT16, sizeof(uint16_t), false, false};
    } else if (strcmp(name, "uint32") == 0) {
        *info = (dtype_info_t){name, ME_UINT32, sizeof(uint32_t), false, false};
    } else if (strcmp(name, "uint64") == 0) {
        *info = (dtype_info_t){name, ME_UINT64, sizeof(uint64_t), false, false};
    } else if (strcmp(name, "float32") == 0) {
        *info = (dtype_info_t){name, ME_FLOAT32, sizeof(float), true, true};
    } else if (strcmp(name, "float64") == 0) {
        *info = (dtype_info_t){name, ME_FLOAT64, sizeof(double), true, true};
    } else {
        return false;
    }
    return true;
}

static me_dtype output_dtype_for_kind(const dtype_info_t *info, reduction_kind_t kind) {
    if (kind == RED_MIN || kind == RED_MAX) {
        return info->dtype;
    }
    if (info->is_float) {
        return info->dtype;
    }
    return info->is_signed ? ME_INT64 : ME_UINT64;
}

static size_t dtype_size_local(me_dtype dtype) {
    switch (dtype) {
        case ME_BOOL: return sizeof(bool);
        case ME_INT8: return sizeof(int8_t);
        case ME_INT16: return sizeof(int16_t);
        case ME_INT32: return sizeof(int32_t);
        case ME_INT64: return sizeof(int64_t);
        case ME_UINT8: return sizeof(uint8_t);
        case ME_UINT16: return sizeof(uint16_t);
        case ME_UINT32: return sizeof(uint32_t);
        case ME_UINT64: return sizeof(uint64_t);
        case ME_FLOAT32: return sizeof(float);
        case ME_FLOAT64: return sizeof(double);
        case ME_COMPLEX64: return sizeof(float _Complex);
        case ME_COMPLEX128: return sizeof(double _Complex);
        default: return 0;
    }
}

static void *sum_worker(void *arg) {
    thread_args_t *args = (thread_args_t *)arg;
    const void *vars_chunk[1] = {
        (const unsigned char *)args->data + (size_t)args->start_idx * args->elem_size
    };

    ME_EVAL_CHECK(args->expr, vars_chunk, 1, args->output, args->count);
    return NULL;
}

static void *reduce_worker_c(void *arg) {
    thread_args_c_t *args = (thread_args_c_t *)arg;
    const dtype_info_t *info = args->info;
    reduction_kind_t kind = args->kind;

    if (info->is_float) {
        if (info->dtype == ME_FLOAT32) {
            const float *data = (const float *)args->data + args->start_idx;
            float acc = 0.0f;
            if (kind == RED_SUM) {
                for (int i = 0; i < args->count; i++) acc += data[i];
            } else if (kind == RED_PROD) {
                acc = 1.0f;
                for (int i = 0; i < args->count; i++) acc *= data[i];
            } else if (kind == RED_MIN) {
                acc = INFINITY;
                for (int i = 0; i < args->count; i++) if (data[i] < acc) acc = data[i];
            } else {
                acc = -INFINITY;
                for (int i = 0; i < args->count; i++) if (data[i] > acc) acc = data[i];
            }
            ((float *)args->output)[0] = acc;
        } else {
            const double *data = (const double *)args->data + args->start_idx;
            double acc = 0.0;
            if (kind == RED_SUM) {
                for (int i = 0; i < args->count; i++) acc += data[i];
            } else if (kind == RED_PROD) {
                acc = 1.0;
                for (int i = 0; i < args->count; i++) acc *= data[i];
            } else if (kind == RED_MIN) {
                acc = INFINITY;
                for (int i = 0; i < args->count; i++) if (data[i] < acc) acc = data[i];
            } else {
                acc = -INFINITY;
                for (int i = 0; i < args->count; i++) if (data[i] > acc) acc = data[i];
            }
            ((double *)args->output)[0] = acc;
        }
    } else if (info->is_signed) {
        if (kind == RED_SUM || kind == RED_PROD) {
            int64_t acc = kind == RED_PROD ? 1 : 0;
            if (info->dtype == ME_INT8) {
                const int8_t *data = (const int8_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            } else if (info->dtype == ME_INT16) {
                const int16_t *data = (const int16_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            } else if (info->dtype == ME_INT32) {
                const int32_t *data = (const int32_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            } else {
                const int64_t *data = (const int64_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            }
            ((int64_t *)args->output)[0] = acc;
        } else {
            if (info->dtype == ME_INT8) {
                const int8_t *data = (const int8_t *)args->data + args->start_idx;
                int8_t acc = kind == RED_MIN ? INT8_MAX : INT8_MIN;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((int8_t *)args->output)[0] = acc;
            } else if (info->dtype == ME_INT16) {
                const int16_t *data = (const int16_t *)args->data + args->start_idx;
                int16_t acc = kind == RED_MIN ? INT16_MAX : INT16_MIN;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((int16_t *)args->output)[0] = acc;
            } else if (info->dtype == ME_INT32) {
                const int32_t *data = (const int32_t *)args->data + args->start_idx;
                int32_t acc = kind == RED_MIN ? INT32_MAX : INT32_MIN;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((int32_t *)args->output)[0] = acc;
            } else {
                const int64_t *data = (const int64_t *)args->data + args->start_idx;
                int64_t acc = kind == RED_MIN ? INT64_MAX : INT64_MIN;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((int64_t *)args->output)[0] = acc;
            }
        }
    } else {
        if (kind == RED_SUM || kind == RED_PROD) {
            uint64_t acc = kind == RED_PROD ? 1 : 0;
            if (info->dtype == ME_UINT8) {
                const uint8_t *data = (const uint8_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            } else if (info->dtype == ME_UINT16) {
                const uint16_t *data = (const uint16_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            } else if (info->dtype == ME_UINT32) {
                const uint32_t *data = (const uint32_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            } else {
                const uint64_t *data = (const uint64_t *)args->data + args->start_idx;
                for (int i = 0; i < args->count; i++) acc = kind == RED_PROD ? acc * data[i] : acc + data[i];
            }
            ((uint64_t *)args->output)[0] = acc;
        } else {
            if (info->dtype == ME_UINT8) {
                const uint8_t *data = (const uint8_t *)args->data + args->start_idx;
                uint8_t acc = kind == RED_MIN ? UINT8_MAX : 0;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((uint8_t *)args->output)[0] = acc;
            } else if (info->dtype == ME_UINT16) {
                const uint16_t *data = (const uint16_t *)args->data + args->start_idx;
                uint16_t acc = kind == RED_MIN ? UINT16_MAX : 0;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((uint16_t *)args->output)[0] = acc;
            } else if (info->dtype == ME_UINT32) {
                const uint32_t *data = (const uint32_t *)args->data + args->start_idx;
                uint32_t acc = kind == RED_MIN ? UINT32_MAX : 0;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((uint32_t *)args->output)[0] = acc;
            } else {
                const uint64_t *data = (const uint64_t *)args->data + args->start_idx;
                uint64_t acc = kind == RED_MIN ? UINT64_MAX : 0;
                for (int i = 0; i < args->count; i++) acc = (kind == RED_MIN) ? (data[i] < acc ? data[i] : acc) : (data[i] > acc ? data[i] : acc);
                ((uint64_t *)args->output)[0] = acc;
            }
        }
    }

    return NULL;
}

static int run_threads(const me_expr *expr, const void *data, size_t elem_size,
                       size_t output_stride, int total_elems, int num_threads,
                       void *partials) {
    pthread_t threads[MAX_THREADS];
    thread_args_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].expr = expr;
        thread_args[t].data = data;
        thread_args[t].elem_size = elem_size;
        thread_args[t].output_stride = output_stride;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        thread_args[t].output = (unsigned char *)partials + (size_t)t * output_stride;
        offset += count;

        pthread_create(&threads[t], NULL, sum_worker, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    return 0;
}

static int run_threads_c(const void *data, int total_elems, int num_threads,
                         void *partials, size_t output_stride,
                         const dtype_info_t *info, reduction_kind_t kind) {
    pthread_t threads[MAX_THREADS];
    thread_args_c_t thread_args[MAX_THREADS];

    int base = total_elems / num_threads;
    int rem = total_elems % num_threads;
    int offset = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = base + (t < rem);
        thread_args[t].data = data;
        thread_args[t].start_idx = offset;
        thread_args[t].count = count;
        thread_args[t].output = (unsigned char *)partials + (size_t)t * output_stride;
        thread_args[t].info = info;
        thread_args[t].kind = kind;
        offset += count;

        pthread_create(&threads[t], NULL, reduce_worker_c, &thread_args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    return 0;
}

static double run_benchmark(const me_expr *expr, const void *data, size_t elem_size,
                            size_t output_stride, int total_elems, int num_threads,
                            int iterations, void *partials) {
    run_threads(expr, data, elem_size, output_stride, total_elems, num_threads, partials);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads(expr, data, elem_size, output_stride, total_elems, num_threads, partials);
    }
    return (get_time() - start) / iterations;
}

static double run_benchmark_c(const void *data, int total_elems, int num_threads,
                              int iterations, void *partials, size_t output_stride,
                              const dtype_info_t *info, reduction_kind_t kind) {
    run_threads_c(data, total_elems, num_threads, partials, output_stride, info, kind);

    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        run_threads_c(data, total_elems, num_threads, partials, output_stride, info, kind);
    }
    return (get_time() - start) / iterations;
}

int main(int argc, char **argv) {
    printf("===================================================\n");
    printf("MiniExpr Reduction Benchmark (Multi-threaded)\n");
    printf("===================================================\n");

    const char *op = "sum";
    const char *type_name = "int32";
    if (argc > 1) {
        op = argv[1];
    }
    if (argc > 2) {
        type_name = argv[2];
    }
    if (strcmp(op, "sum") != 0 && strcmp(op, "prod") != 0 &&
        strcmp(op, "min") != 0 && strcmp(op, "max") != 0) {
        printf("Usage: %s [sum|prod|min|max] [dtype]\n", argv[0]);
        printf("Dtypes: int8 int16 int32 int64 uint8 uint16 uint32 uint64 float32 float64\n");
        return 1;
    }
    reduction_kind_t kind = RED_SUM;
    if (strcmp(op, "prod") == 0) kind = RED_PROD;
    else if (strcmp(op, "min") == 0) kind = RED_MIN;
    else if (strcmp(op, "max") == 0) kind = RED_MAX;

    dtype_info_t info;
    if (!parse_dtype(type_name, &info)) {
        printf("Unknown dtype: %s\n", type_name);
        printf("Dtypes: int8 int16 int32 int64 uint8 uint16 uint32 uint64 float32 float64\n");
        return 1;
    }

    const size_t total_elems = 16ULL * 1024ULL * 1024ULL;
    const int iterations = 4;

    if (total_elems > (size_t)INT_MAX) {
        printf("ERROR: Dataset too large for int-sized nitems\n");
        return 1;
    }

    printf("Total elements per run: %zu\n", total_elems);
    printf("Iterations: %d\n", iterations);

    void *data = malloc(total_elems * info.elem_size);
    if (!data) {
        printf("Allocation failed for data arrays\n");
        return 1;
    }

    for (size_t i = 0; i < total_elems; i++) {
        if (info.is_float) {
            if (info.dtype == ME_FLOAT32) {
                ((float *)data)[i] = (float)(i % 97) * 0.25f;
            } else {
                ((double *)data)[i] = (double)(i % 97) * 0.25;
            }
        } else if (info.is_signed) {
            switch (info.dtype) {
            case ME_INT8: ((int8_t *)data)[i] = (int8_t)(i % 97); break;
            case ME_INT16: ((int16_t *)data)[i] = (int16_t)(i % 97); break;
            case ME_INT32: ((int32_t *)data)[i] = (int32_t)(i % 97); break;
            case ME_INT64: ((int64_t *)data)[i] = (int64_t)(i % 97); break;
            default: break;
            }
        } else {
            switch (info.dtype) {
            case ME_UINT8: ((uint8_t *)data)[i] = (uint8_t)(i % 97); break;
            case ME_UINT16: ((uint16_t *)data)[i] = (uint16_t)(i % 97); break;
            case ME_UINT32: ((uint32_t *)data)[i] = (uint32_t)(i % 97); break;
            case ME_UINT64: ((uint64_t *)data)[i] = (uint64_t)(i % 97); break;
            default: break;
            }
        }
    }

    me_variable vars[] = {{"x", info.dtype, data}};

    int err = 0;
    me_expr *expr = NULL;
    char expr_buf[16];
    snprintf(expr_buf, sizeof(expr_buf), "%s(x)", op);
    int rc_expr = me_compile(expr_buf, vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s(x) for %s (err=%d)\n", op, info.name, err);
        free(data);
        return 1;
    }

    me_dtype out_dtype = me_get_dtype(expr);
    size_t output_stride = dtype_size_local(out_dtype);
    me_dtype expected_dtype = output_dtype_for_kind(&info, kind);
    if (out_dtype != expected_dtype) {
        printf("Unexpected output dtype for reductions: got=%d expected=%d\n",
               out_dtype, expected_dtype);
        me_free(expr);
        free(data);
        return 1;
    }

    void *partials_me = malloc(MAX_THREADS * output_stride);
    void *partials_c = malloc(MAX_THREADS * output_stride);
    if (!partials_me || !partials_c) {
        printf("Allocation failed for partials\n");
        free(partials_me);
        free(partials_c);
        me_free(expr);
        free(data);
        return 1;
    }

    double data_gb = (double)(total_elems * info.elem_size) / 1e9;

    printf("\n========================================\n");
    printf("Summary (%s, %s, GB/s)\n", op, info.name);
    printf("========================================\n");
    printf("Threads     ME       C\n");

    for (int num_threads = 1; num_threads <= MAX_THREADS; num_threads++) {
        double me_time = run_benchmark(expr, data, info.elem_size,
                                       output_stride, (int)total_elems,
                                       num_threads, iterations, partials_me);
        double c_time = run_benchmark_c(data, (int)total_elems,
                                        num_threads, iterations, partials_c,
                                        output_stride, &info, kind);

        printf("%7d  %7.2f  %7.2f\n",
               num_threads,
               data_gb / me_time,
               data_gb / c_time);
    }

    printf("========================================\n");
    printf("Benchmark complete!\n");
    printf("========================================\n");

    free(partials_me);
    free(partials_c);
    me_free(expr);
    free(data);

    return 0;
}
