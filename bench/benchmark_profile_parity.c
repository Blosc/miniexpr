/* Paired full-DSL/Menudet worker. See profile_parity.py for the matrix driver.
 * Numeric qualification belongs in tests; this worker verifies a small native
 * interpreter control before timing, and explicitly reports semantic differences.
 * Usage: artifact.json legacy.dsl backend sizes repeats distribution mask dump-dir
 */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "miniexpr_artifact.h"
#include "dsl_compile_internal.h"
#include "dsl_jit_test.h"
#include "miniexpr_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

static char *read_text(const char *path) {
    FILE *f = fopen(path,"rb");
    if (!f) return NULL;
    if (fseek(f,0,SEEK_END)) { fclose(f); return NULL; }
    long length = ftell(f);
    if (length < 0 || length > 262144 || fseek(f,0,SEEK_SET)) { fclose(f); return NULL; }
    char *text = calloc((size_t)length+1,1);
    if (!text || fread(text,1,(size_t)length,f) != (size_t)length) { free(text); fclose(f); return NULL; }
    fclose(f);
    return text;
}

static void json_string(const char *s) {
    putchar('"');
    for (; s && *s; s++) {
        if (*s == '\n') fputs("\\n",stdout);
        else if (*s == '\r') fputs("\\r",stdout);
        else if (*s == '\t') fputs("\\t",stdout);
        else { if (*s == '"' || *s == '\\') putchar('\\'); putchar(*s); }
    }
    putchar('"');
}

static void failure(const char *stage, const char *detail) {
    printf("{\"status\":\"error\",\"stage\":"); json_string(stage);
    printf(",\"detail\":"); json_string(detail); puts("}");
}

static void dump_source(const char *dir, const char *name, const char *source) {
    if (!source) return;
    char path[4096];
    if (snprintf(path,sizeof(path),"%s/%s",dir,name) >= (int)sizeof(path)) return;
    FILE *f = fopen(path,"wb");
    if (f) { fputs(source,f); fclose(f); }
}

static void fill(void *data, me_dtype dtype, int count, int operand, const char *distribution) {
    bool exceptional = !strcmp(distribution,"exceptional");
    bool positive = !strcmp(distribution,"positive");
    bool greater_one = !strcmp(distribution,"greater-one");
    bool boundary = !strcmp(distribution,"boundary");
    bool combinatoric = !strcmp(distribution,"combinatoric");
    double special[] = {0.0,-0.0,INFINITY,-INFINITY,NAN,0x1p-149,-0x1p-149,0.5,-1.0,1.0};
#define FILL_FLOAT(TYPE) for (int i=0; i<count; i++) { \
    double v = (double)((i+operand*73)%1024)/1024.0; \
    ((TYPE *)data)[i] = (TYPE)(exceptional ? special[(i+operand*3)%10] : \
        greater_one ? 1.25+v*0.5 : (positive || operand) ? 0.25+v*0.5 : v*1.5-0.75); }
#define FILL_INT(TYPE,MINIMUM,MAXIMUM) for (int i=0; i<count; i++) { \
    ((TYPE *)data)[i] = boundary ? (i%4 == 0 ? (MINIMUM) : i%4 == 1 ? (MAXIMUM) : (TYPE)(i%2)) : \
        (TYPE)((combinatoric && !operand ? 4 : 1)+(i+operand)%3); }
    switch (dtype) {
        case ME_FLOAT32: FILL_FLOAT(float) break;
        case ME_FLOAT64: FILL_FLOAT(double) break;
        case ME_BOOL: for (int i=0; i<count; i++) ((bool *)data)[i] = (i+operand)%3 != 0; break;
        case ME_INT8: FILL_INT(int8_t,INT8_MIN,INT8_MAX) break;
        case ME_INT16: FILL_INT(int16_t,INT16_MIN,INT16_MAX) break;
        case ME_INT32: FILL_INT(int32_t,INT32_MIN,INT32_MAX) break;
        case ME_INT64: FILL_INT(int64_t,INT64_MIN,INT64_MAX) break;
        case ME_UINT8: FILL_INT(uint8_t,0,UINT8_MAX) break;
        case ME_UINT16: FILL_INT(uint16_t,0,UINT16_MAX) break;
        case ME_UINT32: FILL_INT(uint32_t,0,UINT32_MAX) break;
        case ME_UINT64: FILL_INT(uint64_t,0,UINT64_MAX) break;
        default: break;
    }
#undef FILL_FLOAT
#undef FILL_INT
}

static bool equal_value(const void *a, const void *b, me_dtype dtype, bool exact) {
    if (!memcmp(a,b,dtype_size(dtype))) return true;
    me_scalar x, y;
    read_scalar(a,dtype,&x); read_scalar(b,dtype,&y);
    if (dtype == ME_FLOAT32 || dtype == ME_FLOAT64) {
        double av = dtype == ME_FLOAT32 ? x.f32 : x.f64;
        double bv = dtype == ME_FLOAT32 ? y.f32 : y.f64;
        if (isnan(av) && isnan(bv)) return true;
        if (!exact && isfinite(av) && isfinite(bv)) return fabs(av-bv) <=
            (dtype == ME_FLOAT32 ? 2e-5 : 2e-12)*(1+fabs(bv));
    }
    return false;
}

static void samples(const char *name, const double *values, int n) {
    printf(",\"%s\":[",name);
    for (int i=0; i<n; i++) printf("%s%.9g",i ? "," : "",values[i]);
    putchar(']');
}

int main(int argc, char **argv) {
    if (argc != 9) { fprintf(stderr,"artifact legacy backend sizes repeats distribution mask dump-dir\n"); return 2; }
    int sizes[16], nsizes=0, maximum=0, repeats=atoi(argv[5]);
    char *sizes_text = strdup(argv[4]);
    for (char *v=strtok(sizes_text,","); v && nsizes<16; v=strtok(NULL,",")) {
        int size=atoi(v);
        if (size<1 || size>100000000) { free(sizes_text); return 2; }
        sizes[nsizes++]=size; if (size>maximum) maximum=size;
    }
    free(sizes_text);
    if (!nsizes || repeats<3 || repeats>31) return 2;
    char *json=read_text(argv[1]), *source=read_text(argv[2]);
    if (!json || !source) { free(json); free(source); return 2; }
    me_jit_mode mode=!strcmp(argv[3],"interpreter") ? ME_JIT_OFF : ME_JIT_ON;
    me_artifact *portable=NULL, *oracle=NULL;
    me_artifact_error error;
    double start=now();
    int rc=me_artifact_load(json,strlen(json),mode,&portable,&error);
    double portable_compile=now()-start;
    if (rc) { failure("portable_compile",error.message); free(json); free(source); return 0; }
    rc=me_artifact_load(json,strlen(json),ME_JIT_OFF,&oracle,&error);
    if (rc) { failure("oracle_compile",error.message); me_artifact_free(portable); free(json); free(source); return 0; }
    int ninputs=me_artifact_ninputs(portable), ndim=me_artifact_context_ndim(portable);
    if (ninputs>8 || ndim>1) { failure("binding","at most eight numeric inputs and rank one supported"); return 2; }
    me_dtype output_dtype=me_artifact_output_dtype(portable);
    size_t width=dtype_size(output_dtype);
    bool scalar=me_artifact_result_cardinality(portable)==ME_ARTIFACT_BLOCK_SCALAR;
    me_variable variables[8]={0};
    me_artifact_buffer bindings[8]={0};
    const void *inputs[8]={0};
    void *owned[8]={0};
    for (int i=0; i<ninputs; i++) {
        me_dtype dtype=me_artifact_input_dtype(portable,i);
        size_t bytes=(size_t)maximum*dtype_size(dtype);
        owned[i]=malloc(bytes);
        if (!owned[i]) return 2;
        fill(owned[i],dtype,maximum,i,argv[6]);
        inputs[i]=owned[i];
        const char *name=me_artifact_input_name(portable,i);
        variables[i]=(me_variable){.name=name,.dtype=dtype};
        bindings[i]=(me_artifact_buffer){name,dtype,dtype_size(dtype),owned[i],bytes};
    }
    size_t capacity=(size_t)maximum*width;
    void *modern=malloc(capacity), *old=malloc(capacity), *reference=malloc(capacity);
    uint8_t *mask=NULL;
    if (!modern || !old || !reference) return 2;
    if (strcmp(argv[7],"none")) {
        mask=malloc((size_t)maximum);
        if (!mask) return 2;
        for (int i=0; i<maximum; i++) mask[i]=!strcmp(argv[7],"all") || (!strcmp(argv[7],"sparse") && i%8==0);
    }
    int64_t logical_shape=maximum, origin=0, extent=maximum;
    int32_t block_shape=maximum;
    me_expr *legacy=NULL;
    int position=0;
    start=now();
    int legacy_compile_rc=ndim ? me_compile_nd_jit(source,variables,ninputs,output_dtype,ndim,
        &logical_shape,&block_shape,&block_shape,mode,&position,&legacy) :
        me_compile(source,variables,ninputs,output_dtype,&position,&legacy);
    double legacy_compile=now()-start;
    char legacy_error[512];
    const char *message=legacy_compile_rc ? me_get_last_error_message() : "";
    snprintf(legacy_error,sizeof(legacy_error),"%s",message ? message : "legacy compilation failed");
    me_dsl_compiled_program *program=dsl_artifact_program_for_tests(portable);
    me_dsl_compiled_program *old_program=legacy ? legacy->dsl_program : NULL;
    dump_source(argv[8],"portable.c",program ? program->jit_c_source : NULL);
    dump_source(argv[8],"legacy.c",old_program ? old_program->jit_c_source : NULL);
    int direct=0, direct2=0;
    if (program) {
        for (int i=0; i<program->portable_jit_nmath; i++) direct+=program->portable_jit_math_fn[i]!=NULL;
        for (int i=0; i<program->portable_jit_nmath2; i++) direct2+=program->portable_jit_math2_fn[i]!=NULL;
    }
    for (int size_index=0; size_index<nsizes; size_index++) {
        int n=sizes[size_index];
        if (ndim && n!=maximum) continue;
        if (n>65536 && (legacy_compile_rc || !me_artifact_has_jit(portable) ||
            !me_expr_has_jit_kernel(legacy))) {
            printf("{\"status\":\"large_fallback_skipped\",\"nitems\":%d,\"portable_jit\":%s,\"legacy_jit\":%s}\n",
                n,me_artifact_has_jit(portable) ? "true" : "false",
                legacy && me_expr_has_jit_kernel(legacy) ? "true" : "false");
            continue;
        }
        extent=n;
        size_t output_bytes=scalar ? width : (size_t)n*width;
        me_artifact_eval_descriptor descriptor={.struct_size=sizeof(descriptor),
            .version=ME_ARTIFACT_EVAL_DESCRIPTOR_VERSION,.nitems=n,.output_capacity=capacity,
            .valid_mask=mask,.valid_mask_capacity=mask ? (size_t)n : 0,.ndim=ndim,
            .logical_shape=ndim ? &logical_shape : NULL,.block_origin=ndim ? &origin : NULL,
            .block_extent=ndim ? &extent : NULL};
        memset(modern,0xa5,capacity); memset(old,0xa5,capacity);
        start=now();
        int modern_rc=me_artifact_eval_ex(portable,bindings,ninputs,modern,&descriptor,&error);
        double first_modern=now()-start;
        start=now();
        int old_rc=legacy_compile_rc ? legacy_compile_rc : ndim ?
            me_eval_nd(legacy,inputs,ninputs,old,n,0,0,NULL) : me_eval(legacy,inputs,ninputs,old,n,NULL);
        double first_old=now()-start;
        int verify_n=n<1024 ? n : 1024;
        me_artifact_eval_descriptor verify=descriptor;
        verify.nitems=verify_n;
        int64_t verify_extent=verify_n;
        if (ndim) verify.block_extent=&verify_extent;
        me_artifact_fp_status actual_status={0}, expected_status={0};
        memset(reference,0xa5,capacity);
        int oracle_rc=me_artifact_eval_status(oracle,bindings,ninputs,reference,&verify,0,&expected_status,&error);
        int actual_rc=me_artifact_eval_status(portable,bindings,ninputs,modern,&verify,0,&actual_status,&error);
        bool verified=oracle_rc==actual_rc && actual_status.flags==expected_status.flags;
        if (!actual_rc) for (int i=0; i<(scalar ? 1 : verify_n); i++) {
            if (mask && !scalar && !mask[i]) continue;
            if (!equal_value((char *)modern+(size_t)i*width,(char *)reference+(size_t)i*width,output_dtype,true)) verified=false;
        }
        double modern_times[31]={0}, old_times[31]={0}, modern_alloc[31]={0}, old_alloc[31]={0};
        if (!modern_rc && verified) {
            for (int repeat=0; repeat<repeats; repeat++) {
                for (int order=0; order<2; order++) {
                    int engine=(repeat+order)%2;
                    if (engine && old_rc) continue;
                    start=now();
                    int result=engine ? (ndim ? me_eval_nd(legacy,inputs,ninputs,old,n,0,0,NULL) :
                        me_eval(legacy,inputs,ninputs,old,n,NULL)) :
                        me_artifact_eval_ex(portable,bindings,ninputs,modern,&descriptor,&error);
                    double elapsed=now()-start;
                    (engine ? old_times : modern_times)[repeat]=elapsed;
                    if (result) { failure("warm_execution","unexpected failure after successful first call"); return 1; }
                }
            }
            for (int repeat=0; repeat<repeats; repeat++) for (int order=0; order<2; order++) {
                int engine=(repeat+order)%2;
                if (engine && old_rc) continue;
                start=now();
                /* Legacy scalar reductions may use n output slots internally. */
                void *fresh=malloc(capacity);
                if (!fresh) return 2;
                int result=engine ? (ndim ? me_eval_nd(legacy,inputs,ninputs,fresh,n,0,0,NULL) :
                    me_eval(legacy,inputs,ninputs,fresh,n,NULL)) :
                    me_artifact_eval_ex(portable,bindings,ninputs,fresh,&descriptor,&error);
                double elapsed=now()-start;
                (engine ? old_alloc : modern_alloc)[repeat]=elapsed;
                free(fresh);
                if (result) return 2;
            }
        }
        int compared=0, differences=0;
        if (!modern_rc && !old_rc && verified) {
            int count=scalar ? 1 : n;
            int stride=count>1024 ? count/1024 : 1;
            for (int lane=0; lane<count; lane+=stride) {
                if (mask && !scalar && !mask[lane]) continue;
                compared++;
                if (!equal_value((char *)modern+(size_t)lane*width,(char *)old+(size_t)lane*width,output_dtype,false)) differences++;
            }
        }
        printf("{\"status\":\"%s\",\"nitems\":%d,\"portable_jit\":%s,\"legacy_jit\":%s,",
            verified ? "ok" : "validation_error",n,me_artifact_has_jit(portable) ? "true" : "false",
            legacy && me_expr_has_jit_kernel(legacy) ? "true" : "false");
        printf("\"portable_rc\":%d,\"legacy_rc\":%d,\"oracle_rc\":%d,\"oracle_items\":%d,",modern_rc,old_rc,oracle_rc,verify_n);
        printf("\"portable_compile_s\":%.9g,\"legacy_compile_s\":%.9g,\"portable_first_s\":%.9g,\"legacy_first_s\":%.9g,",
            portable_compile,legacy_compile,first_modern,first_old);
        printf("\"compared\":%d,\"differences\":%d,\"output_bytes\":%zu,\"fp_flags\":%u,\"fp_supported\":%s,",
            compared,differences,output_bytes,actual_status.flags,actual_status.supported ? "true" : "false");
        printf("\"direct_math\":%d,\"direct_math2\":%d,\"generic_math\":%d,\"generic_math2\":%d,\"checked_bridges\":%d,\"integer_bridges\":%d,\"predicate_bridges\":%d,\"comparison_bridges\":%d,\"legacy_error\":",
            direct,direct2,program ? program->portable_jit_nmath-direct : 0,
            program ? program->portable_jit_nmath2-direct2 : 0,program ? program->portable_jit_nchecked : 0,
            program ? program->portable_jit_niops : 0,program ? program->portable_jit_npreds : 0,
            program ? program->portable_jit_ncomparisons : 0);
        json_string(legacy_error);
        samples("portable_s",modern_times,repeats); samples("legacy_s",old_times,repeats);
        samples("portable_alloc_s",modern_alloc,repeats); samples("legacy_alloc_s",old_alloc,repeats);
        puts("}"); fflush(stdout);
    }
    me_free(legacy); me_artifact_free(oracle); me_artifact_free(portable);
    for (int i=0; i<ninputs; i++) free(owned[i]);
    free(mask); free(modern); free(old); free(reference); free(json); free(source);
    return 0;
}
