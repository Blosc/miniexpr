#include "miniexpr_artifact.h"
#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    return 0;
}
