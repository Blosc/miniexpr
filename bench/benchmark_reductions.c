/*
 * Benchmark reductions (sum) for int32 and float32.
 * Compares MiniExpr sum(x) against a pure C loop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <stdbool.h>
#include <string.h>
#include "miniexpr.h"
#include "minctest.h"



typedef struct {
    double me_time;
    double c_time;
    double me_gbps;
    double c_gbps;
} bench_result_t;

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

static bench_result_t benchmark_reduce(const char *op, reduction_kind_t kind,
                                       const dtype_info_t *info,
                                       size_t total_elems, int iterations) {
    printf("\n=== %s(%s) ===\n", op, info->name);

    void *data = malloc(total_elems * info->elem_size);
    if (!data) {
        printf("Allocation failed for %s data\n", info->name);
        bench_result_t empty = {0};
        return empty;
    }

    for (size_t i = 0; i < total_elems; i++) {
        if (info->is_float) {
            if (info->dtype == ME_FLOAT32) {
                ((float *)data)[i] = (float)(i % 97) * 0.25f;
            } else {
                ((double *)data)[i] = (double)(i % 97) * 0.25;
            }
        } else if (info->is_signed) {
            switch (info->dtype) {
            case ME_INT8: ((int8_t *)data)[i] = (int8_t)(i % 97); break;
            case ME_INT16: ((int16_t *)data)[i] = (int16_t)(i % 97); break;
            case ME_INT32: ((int32_t *)data)[i] = (int32_t)(i % 97); break;
            case ME_INT64: ((int64_t *)data)[i] = (int64_t)(i % 97); break;
            default: break;
            }
        } else {
            switch (info->dtype) {
            case ME_UINT8: ((uint8_t *)data)[i] = (uint8_t)(i % 97); break;
            case ME_UINT16: ((uint16_t *)data)[i] = (uint16_t)(i % 97); break;
            case ME_UINT32: ((uint32_t *)data)[i] = (uint32_t)(i % 97); break;
            case ME_UINT64: ((uint64_t *)data)[i] = (uint64_t)(i % 97); break;
            default: break;
            }
        }
    }

    me_variable vars[] = {{"x", info->dtype, data}};
    int err = 0;
    me_expr *expr = NULL;
    char expr_buf[16];
    snprintf(expr_buf, sizeof(expr_buf), "%s(x)", op);
    int rc_expr = me_compile(expr_buf, vars, 1, ME_AUTO, &err, &expr);
    if (rc_expr != ME_COMPILE_SUCCESS) {
        printf("Failed to compile %s(x) for %s (err=%d)\n", op, info->name, err);
        free(data);
        bench_result_t empty = {0};
        return empty;
    }

    const void *var_ptrs[] = {data};
    union {
        int8_t i8;
        int16_t i16;
        int32_t i32;
        int64_t i64;
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        float f32;
        double f64;
    } output;

    me_dtype out_dtype = output_dtype_for_kind(info, kind);
    ME_EVAL_CHECK(expr, var_ptrs, 1, &output, (int)total_elems);

    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        ME_EVAL_CHECK(expr, var_ptrs, 1, &output, (int)total_elems);
    }
    double me_time = (get_time() - start) / iterations;

    union {
        int64_t i64;
        uint64_t u64;
        int8_t i8;
        int16_t i16;
        int32_t i32;
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        float f32;
        double f64;
    } sink;
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        if (info->is_float) {
            if (info->dtype == ME_FLOAT32) {
                float *f = (float *)data;
                float acc = 0.0f;
                if (kind == RED_SUM) {
                    for (size_t i = 0; i < total_elems; i++) acc += f[i];
                } else if (kind == RED_PROD) {
                    acc = 1.0f;
                    for (size_t i = 0; i < total_elems; i++) acc *= f[i];
                } else if (kind == RED_MIN) {
                    acc = INFINITY;
                    for (size_t i = 0; i < total_elems; i++) if (f[i] < acc) acc = f[i];
                } else {
                    acc = -INFINITY;
                    for (size_t i = 0; i < total_elems; i++) if (f[i] > acc) acc = f[i];
                }
                sink.f32 = acc;
            } else {
                double *d = (double *)data;
                double acc = 0.0;
                if (kind == RED_SUM) {
                    for (size_t i = 0; i < total_elems; i++) acc += d[i];
                } else if (kind == RED_PROD) {
                    acc = 1.0;
                    for (size_t i = 0; i < total_elems; i++) acc *= d[i];
                } else if (kind == RED_MIN) {
                    acc = INFINITY;
                    for (size_t i = 0; i < total_elems; i++) if (d[i] < acc) acc = d[i];
                } else {
                    acc = -INFINITY;
                    for (size_t i = 0; i < total_elems; i++) if (d[i] > acc) acc = d[i];
                }
                sink.f64 = acc;
            }
        } else if (info->is_signed) {
            if (kind == RED_SUM || kind == RED_PROD) {
                int64_t acc = kind == RED_PROD ? 1 : 0;
                if (info->dtype == ME_INT8) {
                    int8_t *v = (int8_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                } else if (info->dtype == ME_INT16) {
                    int16_t *v = (int16_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                } else if (info->dtype == ME_INT32) {
                    int32_t *v = (int32_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                } else {
                    int64_t *v = (int64_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                }
                sink.i64 = acc;
            } else {
                if (info->dtype == ME_INT8) {
                    int8_t *v = (int8_t *)data;
                    int8_t acc = kind == RED_MIN ? INT8_MAX : INT8_MIN;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.i8 = acc;
                } else if (info->dtype == ME_INT16) {
                    int16_t *v = (int16_t *)data;
                    int16_t acc = kind == RED_MIN ? INT16_MAX : INT16_MIN;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.i16 = acc;
                } else if (info->dtype == ME_INT32) {
                    int32_t *v = (int32_t *)data;
                    int32_t acc = kind == RED_MIN ? INT32_MAX : INT32_MIN;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.i32 = acc;
                } else {
                    int64_t *v = (int64_t *)data;
                    int64_t acc = kind == RED_MIN ? INT64_MAX : INT64_MIN;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.i64 = acc;
                }
            }
        } else {
            if (kind == RED_SUM || kind == RED_PROD) {
                uint64_t acc = kind == RED_PROD ? 1 : 0;
                if (info->dtype == ME_UINT8) {
                    uint8_t *v = (uint8_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                } else if (info->dtype == ME_UINT16) {
                    uint16_t *v = (uint16_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                } else if (info->dtype == ME_UINT32) {
                    uint32_t *v = (uint32_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                } else {
                    uint64_t *v = (uint64_t *)data;
                    for (size_t i = 0; i < total_elems; i++) acc = kind == RED_PROD ? acc * v[i] : acc + v[i];
                }
                sink.u64 = acc;
            } else {
                if (info->dtype == ME_UINT8) {
                    uint8_t *v = (uint8_t *)data;
                    uint8_t acc = kind == RED_MIN ? UINT8_MAX : 0;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.u8 = acc;
                } else if (info->dtype == ME_UINT16) {
                    uint16_t *v = (uint16_t *)data;
                    uint16_t acc = kind == RED_MIN ? UINT16_MAX : 0;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.u16 = acc;
                } else if (info->dtype == ME_UINT32) {
                    uint32_t *v = (uint32_t *)data;
                    uint32_t acc = kind == RED_MIN ? UINT32_MAX : 0;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.u32 = acc;
                } else {
                    uint64_t *v = (uint64_t *)data;
                    uint64_t acc = kind == RED_MIN ? UINT64_MAX : 0;
                    for (size_t i = 0; i < total_elems; i++) acc = (kind == RED_MIN) ? (v[i] < acc ? v[i] : acc) : (v[i] > acc ? v[i] : acc);
                    sink.u64 = acc;
                }
            }
        }
    }
    double c_time = (get_time() - start) / iterations;

    double gb = (double)(total_elems * sizeof(int32_t)) / 1e9;
    printf("MiniExpr: %.4f s (%.2f GB/s)\n", me_time, gb / me_time);
    printf("Pure C : %.4f s (%.2f GB/s)\n", c_time, gb / c_time);
    if (out_dtype == ME_INT64) {
        printf("Result check (MiniExpr): %lld\n", (long long)output.i64);
        printf("Result check (C):        %lld\n", (long long)sink.i64);
    } else if (out_dtype == ME_UINT64) {
        printf("Result check (MiniExpr): %llu\n", (unsigned long long)output.u64);
        printf("Result check (C):        %llu\n", (unsigned long long)sink.u64);
    } else if (out_dtype == ME_INT32) {
        printf("Result check (MiniExpr): %d\n", output.i32);
        printf("Result check (C):        %d\n", sink.i32);
    } else if (out_dtype == ME_INT16) {
        printf("Result check (MiniExpr): %d\n", (int)output.i16);
        printf("Result check (C):        %d\n", (int)sink.i16);
    } else if (out_dtype == ME_INT8) {
        printf("Result check (MiniExpr): %d\n", (int)output.i8);
        printf("Result check (C):        %d\n", (int)sink.i8);
    } else if (out_dtype == ME_UINT32) {
        printf("Result check (MiniExpr): %u\n", output.u32);
        printf("Result check (C):        %u\n", sink.u32);
    } else if (out_dtype == ME_UINT16) {
        printf("Result check (MiniExpr): %u\n", (unsigned)output.u16);
        printf("Result check (C):        %u\n", (unsigned)sink.u16);
    } else if (out_dtype == ME_UINT8) {
        printf("Result check (MiniExpr): %u\n", (unsigned)output.u8);
        printf("Result check (C):        %u\n", (unsigned)sink.u8);
    } else if (out_dtype == ME_FLOAT32) {
        printf("Result check (MiniExpr): %.6f\n", output.f32);
        printf("Result check (C):        %.6f\n", sink.f32);
    } else {
        printf("Result check (MiniExpr): %.6f\n", output.f64);
        printf("Result check (C):        %.6f\n", sink.f64);
    }

    me_free(expr);
    free(data);
    bench_result_t result = {
        .me_time = me_time,
        .c_time = c_time,
        .me_gbps = gb / me_time,
        .c_gbps = gb / c_time
    };
    return result;
}

int main(int argc, char **argv) {
    printf("==========================================\n");
    printf("MiniExpr Reduction Benchmark\n");
    printf("==========================================\n");

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

    const size_t sizes_mb[] = {1, 2, 4, 8, 16};
    const int iterations = 4;
    const size_t num_sizes = sizeof(sizes_mb) / sizeof(sizes_mb[0]);

    bench_result_t results[10];

    printf("Iterations: %d\n", iterations);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t bytes = sizes_mb[i] * 1024 * 1024;
        size_t total_elems = bytes / info.elem_size;

        printf("\n--- Working set: %zu MB (%zu elements) ---\n", sizes_mb[i], total_elems);
        results[i] = benchmark_reduce(op, kind, &info, total_elems, iterations);
    }

    printf("\n==========================================\n");
    printf("Summary (%s, %s, GB/s)\n", op, info.name);
    printf("==========================================\n");
    printf("Size(MB)     ME       C\n");
    for (size_t i = 0; i < num_sizes; i++) {
        printf("%7zu  %7.2f  %7.2f\n",
               sizes_mb[i],
               results[i].me_gbps,
               results[i].c_gbps);
    }

    printf("==========================================\n");
    printf("Benchmark complete!\n");
    printf("==========================================\n");

    return 0;
}
