#include "dsl_compile_internal.h"
#include "dsl_eval_internal.h"
#include "dsl_menudet_expr.h"
#include "dsl_menudet_fp.h"
#include "dsl_jit_test.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static me_dsl_compiled_program *compile_power_policy(const char *expression, me_dtype dtype,
    const char *backend, const char *fp, int jit, bool integer_powers) {
    char source[512],reason[512];
    snprintf(source,sizeof(source),"# me:compiler=cc\n# me:math=%s\n# me:fp=%s\ndef k(x, y):\n    return %s\n",
        backend,fp,expression);
    me_variable vars[]={{.name="x",.dtype=dtype},{.name="y",.dtype=dtype}};
    int error;
    bool is_dsl;
    me_dsl_declared_policy policy={.integer_powers=integer_powers};
    me_dsl_compiled_program *p=dsl_compile_program_profile_declared(source,vars,2,dtype,1,jit,
        ME_DSL_PROFILE_MENUDET_0_1,&policy,&error,&is_dsl,reason,sizeof(reason));
    if (!p && strcmp(backend,"system") && strstr(reason,"requested math backend") &&
        strstr(reason,"supported alternatives")) return NULL; /* Optional backend. */
    if (!p) fprintf(stderr,"%s: %s\n",expression,reason);
    assert(p && is_dsl);
    return p;
}

static me_dsl_compiled_program *compile_power(const char *expression, me_dtype dtype,
    const char *backend, const char *fp, int jit) {
    return compile_power_policy(expression,dtype,backend,fp,jit,false);
}

static int evaluate(me_dsl_compiled_program *p, const void **inputs, void *out,
    int count, const uint8_t *mask) {
    int64_t shape[]={count},origin[]={0};
    me_dsl_menudet_eval_descriptor d={count,mask,(size_t)count*dtype_size(p->output_dtype),
        1,shape,origin,shape};
    return dsl_eval_program_menudet(p,inputs,2,out,&d);
}

static void strict_power(const char *expression, me_dtype dtype, const char *backend) {
    me_dsl_compiled_program *p=compile_power(expression,dtype,backend,"strict",ME_JIT_OFF);
    if (!p) return;
    double seeds[]={-0.0,0.0,-2.0,2.0,0.5,-0.5,INFINITY,-INFINITY,NAN,DBL_MAX,DBL_MIN,1.0001};
    double powers[]={2.0,3.0,0.0,-1.0,-3.0,0.5,1.25,INFINITY,NAN};
    double x[263],y[263],out[265];
    float xf[263],yf[263],outf[265];
    for (int i=0; i<263; i++) {
        x[i]=seeds[i%12]; y[i]=powers[(i/12)%9];
        xf[i]=(float)x[i]; yf[i]=(float)y[i];
    }
    uint64_t snan64=UINT64_C(0x7ff0000000000001);
    uint32_t snan32=UINT32_C(0x7f800001);
    memcpy(&x[261],&snan64,sizeof(snan64));
    memcpy(&xf[261],&snan32,sizeof(snan32));
    const void *inputs[]={dtype == ME_FLOAT32 ? (void *)xf : (void *)x,
        dtype == ME_FLOAT32 ? (void *)yf : (void *)y};
    int counts[]={32,255,256,257,263};
    for (int trial=0; trial<5; trial++) {
        int count=counts[trial];
        for (int i=0; i<265; i++) {
            out[i]=123.0; outf[i]=123.0f;
        }
        size_t before=dsl_menudet_batch_calls_for_tests();
#ifndef __EMSCRIPTEN__
        feclearexcept(FE_ALL_EXCEPT);
        unsigned batch_flags=0,scalar_flags=0;
        unsigned *previous=dsl_menudet_status_begin(&batch_flags);
#endif
        assert(evaluate(p,inputs,dtype == ME_FLOAT32 ? (void *)outf : (void *)out,count,NULL)==0);
#ifndef __EMSCRIPTEN__
        dsl_menudet_status_end(previous);
#endif
        assert(dsl_menudet_batch_calls_for_tests()==before+1);
        assert(out[count]==123.0 && outf[count]==123.0f);
        for (int i=0; i<count; i++) {
            const void *lane[]={dtype == ME_FLOAT32 ? (void *)&xf[i] : (void *)&x[i],
                dtype == ME_FLOAT32 ? (void *)&yf[i] : (void *)&y[i]};
            double reference=0;
            float referencef=0;
#ifndef __EMSCRIPTEN__
            feclearexcept(FE_ALL_EXCEPT);
            unsigned lane_flags=0;
            previous=dsl_menudet_status_begin(&lane_flags);
#endif
            assert(evaluate(p,lane,dtype == ME_FLOAT32 ? (void *)&referencef : (void *)&reference,1,NULL)==0);
#ifndef __EMSCRIPTEN__
            dsl_menudet_status_end(previous);
            scalar_flags |= lane_flags;
#endif
            double actual=dtype == ME_FLOAT32 ? outf[i] : out[i];
            double expected=dtype == ME_FLOAT32 ? referencef : reference;
            if (isnan(expected)) assert(isnan(actual));
            else if (!strcmp(backend,"system") || !isfinite(expected) || expected==0) {
                assert(actual==expected && signbit(actual)==signbit(expected));
            } else {
                double eps=dtype == ME_FLOAT32 ? FLT_EPSILON : DBL_EPSILON;
                assert(fabs(actual-expected)<=8*eps*fabs(expected));
            }
        }
#ifndef __EMSCRIPTEN__
        if (!strcmp(backend,"system")) assert(batch_flags==scalar_flags);
#endif
        uint8_t mask[263];
        memset(mask,1,sizeof(mask)); mask[0]=0;
        before=dsl_menudet_batch_calls_for_tests();
        assert(evaluate(p,inputs,dtype == ME_FLOAT32 ? (void *)outf : (void *)out,count,mask)==0);
        assert(dsl_menudet_batch_calls_for_tests()==before);
    }
    dsl_compiled_program_free(p);
}

static void integer_power_policy(void) {
    int exponents[]={0,1,2,3,4,7,16,31,-1,-2,-3,-7};
    for (int precision=0; precision<2; precision++) for (int spelling=0; spelling<2; spelling++) {
        me_dtype dtype=precision ? ME_FLOAT64 : ME_FLOAT32;
        for (int n=0; n<12; n++) {
            char expression[128];
            if (spelling) snprintf(expression,sizeof(expression),"pow(x + 1, %d) - 1",exponents[n]);
            else snprintf(expression,sizeof(expression),"(x + 1)**%d - 1",exponents[n]);
            me_dsl_compiled_program *p=compile_power_policy(expression,dtype,"system","strict",ME_JIT_OFF,true);
            assert(p->fp_mode==ME_DSL_FP_STRICT && p->integer_powers);
            double x[33],y[33]={0},out[33];
            float xf[33],yf[33]={0},outf[33];
            for (int i=0; i<33; i++) {
                x[i]=(i & 1) ? -2.0-i/64.0 : i/64.0;
                xf[i]=(float)x[i];
            }
            const void *inputs[]={precision ? (void *)x : (void *)xf,precision ? (void *)y : (void *)yf};
            size_t before=dsl_menudet_batch_calls_for_tests();
            assert(evaluate(p,inputs,precision ? (void *)out : (void *)outf,33,NULL)==0);
            assert(dsl_menudet_batch_calls_for_tests()==before+1);
            for (int i=0; i<33; i++) {
                double expected=pow(x[i]+1,exponents[n])-1;
                double actual=precision ? out[i] : outf[i];
                double eps=precision ? DBL_EPSILON : FLT_EPSILON;
                assert(fabs(actual-expected)<=128*eps*(1+fabs(expected)));
                const void *lane[]={precision ? (void *)&x[i] : (void *)&xf[i],precision ? (void *)&y[i] : (void *)&yf[i]};
                double scalar=0; float scalarf=0;
                assert(evaluate(p,lane,precision ? (void *)&scalar : (void *)&scalarf,1,NULL)==0);
                assert(actual==(precision ? scalar : scalarf));
            }
            dsl_compiled_program_free(p);
        }
    }
}

#ifndef __EMSCRIPTEN__
static void power_base_status(void) {
    me_dsl_compiled_program *p=compile_power_policy("(x / 0)**0",ME_FLOAT64,"system","strict",ME_JIT_ON,true);
    double x[33],y[33]={0},out[33];
    for (int i=0; i<33; i++) x[i]=1;
    const void *inputs[]={x,y};
    feclearexcept(FE_ALL_EXCEPT);
    unsigned flags=0;
    unsigned *previous=dsl_menudet_status_begin(&flags);
    assert(evaluate(p,inputs,out,33,NULL)==0);
    dsl_menudet_status_end(previous);
    assert(flags & 2u);
    for (int i=0; i<33; i++) assert(out[i]==1);
    dsl_compiled_program_free(p);
}

static void fast_power(void) {
    int exponents[]={0,1,2,3,4,7,16,31,-1,-2,-3,-7};
    for (int precision=0; precision<2; precision++) {
        me_dtype dtype=precision ? ME_FLOAT64 : ME_FLOAT32;
        for (int spelling=0; spelling<2; spelling++) for (int n=0; n<12; n++) {
            char expression[128];
            if (spelling) snprintf(expression,sizeof(expression),"pow(x + 1, %d) - 1",exponents[n]);
            else snprintf(expression,sizeof(expression),"(x + 1)**%d - 1",exponents[n]);
            me_dsl_compiled_program *p=compile_power(expression,dtype,"system","fast",ME_JIT_ON);
            assert(p->jit_c_source && strstr(p->jit_c_source,"pj_power_base_"));
            double x[33],y[33]={0},out[33];
            float xf[33],yf[33]={0},outf[33];
            for (int i=0; i<33; i++) {
                x[i]=(i & 1) ? -2.0-i/64.0 : i/64.0;
                xf[i]=(float)x[i];
            }
            const void *inputs[]={precision ? (void *)x : (void *)xf,precision ? (void *)y : (void *)yf};
            assert(evaluate(p,inputs,precision ? (void *)out : (void *)outf,33,NULL)==0);
            for (int i=0; i<33; i++) {
                double expected=pow(x[i]+1,exponents[n])-1;
                double actual=precision ? out[i] : outf[i];
                double eps=precision ? DBL_EPSILON : FLT_EPSILON;
                assert(fabs(actual-expected)<=128*eps*(1+fabs(expected)));
            }
            dsl_compiled_program_free(p);
        }
        const char *fallbacks[]={"x**2.5", "pow(x, y)", "x**2147483648"};
        for (int n=0; n<3; n++) {
            me_dsl_compiled_program *p=compile_power(fallbacks[n],dtype,"system","fast",ME_JIT_ON);
            assert(p->jit_c_source && !strstr(p->jit_c_source,"pj_power_base_"));
            dsl_compiled_program_free(p);
        }
        me_dsl_compiled_program *p=compile_power("x**3",dtype,"system","strict",ME_JIT_ON);
        assert(p->jit_c_source && !strstr(p->jit_c_source,"pj_power_base_"));
        dsl_compiled_program_free(p);
        p=compile_power("pow(x, 2)",dtype,"system","contract",ME_JIT_ON);
        assert(p->jit_c_source && strstr(p->jit_c_source,"pj_power_base_"));
        dsl_compiled_program_free(p);
    }
}
#endif

int main(void) {
    const char *expressions[]={"x**y", "pow(x, y)", "x**2 - 1", "pow(x, 3) - 1",
        "pow(x + 1, y) + x**2"};
    const char *backends[]={"system","accelerate","sleef"};
    for (int b=0; b<3; b++) for (int e=0; e<5; e++) {
        strict_power(expressions[e],ME_FLOAT32,backends[b]);
        strict_power(expressions[e],ME_FLOAT64,backends[b]);
    }
    strict_power("pow(x, 2.0)",ME_FLOAT32,"system");
    strict_power("x**2.0",ME_FLOAT64,"system");
    integer_power_policy();
#ifndef __EMSCRIPTEN__
    fast_power();
    power_base_status();
#endif
    puts("Menudet batched strict power and fast integer powers: passed");
    return 0;
}
