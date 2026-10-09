#include "miniexpr_artifact.h"
#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_extended(const char *source, me_dtype dtype, const void *x, const void *y, size_t width) {
    char json[4096];
    const char *name = dtype == ME_INT32 ? "int32" : "float64";
    snprintf(json,sizeof(json),
        "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\",\"control-flow\"],\"source\":\"%s\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"%s\"},{\"name\":\"y\",\"dtype\":\"%s\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"%s\",\"contract\":\"elementwise\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}",
        source,name,name,name);
    me_artifact *reference = NULL, *accelerated = NULL;
    me_artifact_error error = {0};
    assert(me_artifact_load(json,strlen(json),ME_JIT_OFF,&reference,&error)==0);
    assert(me_artifact_load(json,strlen(json),ME_JIT_ON,&accelerated,&error)==0);
    if (getenv("MENUDET_REQUIRE_JIT")) assert(me_artifact_has_jit(accelerated));
    me_artifact_buffer inputs[] = {{"x",dtype,width,x,5*width},{"y",dtype,width,y,5*width}};
    unsigned char mask[] = {0,1,1,0,1};
    me_artifact_eval_descriptor descriptor = {.struct_size=sizeof(descriptor),.version=1,
        .nitems=5,.output_capacity=5*width};
    /* Aligned output storage supports both fixture types. */
    double actual[5] = {0}, expected[5] = {0};
    for (int masked = 0; masked < 2; masked++) {
        descriptor.valid_mask = masked ? mask : NULL;
        descriptor.valid_mask_capacity = masked ? 5 : 0;
        me_artifact_fp_status a,b;
        assert(me_artifact_eval_status(reference,inputs,2,expected,&descriptor,0,&a,&error)==0);
        assert(me_artifact_eval_status(accelerated,inputs,2,actual,&descriptor,0,&b,&error)==0);
        assert(a.flags==b.flags && a.supported==b.supported);
        for (int i = 0; i < 5; i++) {
            if (masked && !mask[i]) continue;
            if (dtype == ME_FLOAT64 && isnan(expected[i])) assert(isnan(actual[i]));
            else assert(!memcmp((char *)actual+i*width,(char *)expected+i*width,width));
        }
    }
    me_artifact_free(reference);
    me_artifact_free(accelerated);
}

int main(void) {
    const char *json = "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\"],\"source\":\"def k(x, y):\\n    return where(x != 0, y / x, y)\\n\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"float64\"},{\"name\":\"y\",\"dtype\":\"float64\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"float64\",\"contract\":\"elementwise\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}";
    me_artifact *reference = NULL, *accelerated = NULL; me_artifact_error error = {0};
    assert(me_artifact_load(json,strlen(json),ME_JIT_OFF,&reference,&error)==0);
    assert(me_artifact_load(json,strlen(json),ME_JIT_ON,&accelerated,&error)==0);
    if (getenv("MENUDET_REQUIRE_JIT")) assert(me_artifact_has_jit(accelerated));
    double x[] = {0,-0.0,2,NAN,1}, y[] = {1,1,4,1,2}, expected[5], actual[5];
    uint64_t signaling_low_bits = UINT64_C(0x3ff000007f800001);
    memcpy(&x[4],&signaling_low_bits,sizeof(double));
    me_artifact_buffer inputs[] = {
        {"x",ME_FLOAT64,8,x,sizeof(x)}, {"y",ME_FLOAT64,8,y,sizeof(y)}
    };
    me_artifact_eval_descriptor descriptor = {.struct_size=sizeof(descriptor),.version=1,
        .nitems=5,.output_capacity=sizeof(actual)};
    me_artifact_fp_status a,b;
    assert(me_artifact_eval_status(reference,inputs,2,expected,&descriptor,0,&a,&error)==0);
    assert(me_artifact_eval_status(accelerated,inputs,2,actual,&descriptor,0,&b,&error)==0);
    assert(a.flags==b.flags && a.supported==b.supported);
    for (int i=0; i<5; i++) assert(isnan(expected[i]) ? isnan(actual[i]) : actual[i]==expected[i]);
    assert(actual[0]==1 && actual[1]==1 && actual[2]==2);
    unsigned char mask[]={1,1,1,0,1}; descriptor.valid_mask=mask; descriptor.valid_mask_capacity=5;
    assert(me_artifact_eval_status(accelerated,inputs,2,actual,&descriptor,0,&b,&error)==0);
    assert(b.flags==0);
    printf("portable JIT values/lazy mask/status: passed (%s)\n",me_artifact_has_jit(accelerated)?"compiled":"fallback");
    me_artifact_free(reference); me_artifact_free(accelerated);
    double fx[] = {0,-2,3,NAN,1}, fy[] = {1,4,5,6,2};
    check_extended("def k(x, y):\\n    z = y / x\\n    return x\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    if x > 0:\\n        return y / x\\n    return -x\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    if x > 0:\\n        z = y / x\\n    else:\\n        z = y\\n    return z + 1\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    return sin(x) + cos(x)\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    return where(x > 0, log(x), y)\\n",ME_FLOAT64,fx,fy,8);
    int32_t ix[] = {INT32_MIN,INT32_MAX,0,1,-1}, iy[] = {1,1,INT32_MAX,INT32_MIN,INT32_MIN};
    check_extended("def k(x, y):\\n    z = x + y\\n    return z * y - x\\n",ME_INT32,ix,iy,4);
    /* Bridge-lowered float operators, binary math, predicates and integer ops. */
    check_extended("def k(x, y):\\n    return x // y\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    return x % y\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    return x ** y\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    return hypot(x, y)\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    return atan2(x, y)\\n",ME_FLOAT64,fx,fy,8);
    check_extended("def k(x, y):\\n    return x % y\\n",ME_INT32,ix,iy,4);
    check_extended("def k(x, y):\\n    return x // y\\n",ME_INT32,ix,iy,4);
    check_extended("def k(x, y):\\n    return x << y\\n",ME_INT32,ix,iy,4);
    check_extended("def k(x, y):\\n    return x & y\\n",ME_INT32,ix,iy,4);
    printf("portable JIT locals/branches/math/modular integers/operators: passed\n");
    return 0;
}
