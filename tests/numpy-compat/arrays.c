#include "miniexpr_artifact.h"
#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* A standalone C host: broadcast and axis reduction, no Python or storage ABI. */
int main(void) {
    const char *json = "{\"schema_version\":\"1.1\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.1\"},"
        "\"requires\":[\"numeric\"],\"source\":\"def k(x, y):\\n    return x + y\\n\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"int64\"},{\"name\":\"y\",\"dtype\":\"int64\"}],\"constants\":[],"
        "\"output\":{\"dtype\":\"int64\",\"contract\":\"elementwise\"},\"context\":{\"ndim\":0},"
        "\"semantics\":{\"fp\":\"strict\",\"numeric\":\"numpy-2.5\",\"casting\":\"unsafe\"},\"metadata\":{}}";
    me_artifact *artifact = NULL; me_artifact_error error = {0};
    assert(me_artifact_load(json,strlen(json),ME_JIT_OFF,&artifact,&error) == 0);
    int64_t x[] = {1,2,3,4,5,6}, y[] = {10,20,30}, output[6] = {0};
    me_array_view inputs[2] = {
        {.name="x",.dtype=ME_INT64,.base=x,.capacity=sizeof(x),.rank=2,.shape={2,3},.strides={24,8}},
        {.name="y",.dtype=ME_INT64,.base=y,.capacity=sizeof(y),.rank=1,.shape={3},.strides={8}}
    };
    int64_t shape[] = {2,3};
    me_array_options options = {.version=1,.reduction=ME_ARRAY_NONE,.naxes=0,.accumulator=ME_AUTO,.tile_items=2};
    me_array_report report;
    assert(me_artifact_eval_array(artifact,inputs,2,2,shape,&options,output,sizeof(output),&report,&error) == 0);
    int64_t expected[] = {11,22,33,14,25,36}; assert(!memcmp(output,expected,sizeof(output)));
    assert(report.temporary_bytes == 16 && report.gathered_bytes == 48 && report.zero_copy_tiles == 3);
    options.reduction=ME_ARRAY_SUM; options.naxes=1; options.axes[0]=-1;
    for (size_t tile=1; tile<=7; tile++) {
        options.tile_items=tile;
        assert(me_artifact_eval_array(artifact,inputs,2,2,shape,&options,output,16,&report,&error) == 0);
        assert(output[0] == 66 && output[1] == 75);
    }
    options.axes[0]=0;
    assert(me_artifact_eval_array(artifact,inputs,2,2,shape,&options,output,24,&report,&error) == 0);
    assert(output[0]==25 && output[1]==47 && output[2]==69);
    /* Native metadata-only basic views; reversal keeps its checked allocation. */
    me_array_view sliced, transposed, reshaped;
    assert(me_array_slice(&inputs[0],1,2,3,-1,&sliced,&error)==0);
    assert(sliced.offset==16 && sliced.strides[1]==-8);
    int axes[]={1,0}; assert(me_array_transpose(&sliced,axes,&transposed,&error)==0);
    assert(transposed.shape[0]==3 && transposed.shape[1]==2);
    int64_t flat[]={6}; assert(me_array_reshape(&inputs[0],1,flat,&reshaped,&error)==0);
    assert(me_array_reshape(&sliced,1,flat,&reshaped,&error)==ME_ARTIFACT_ERR_BINDING);
    inputs[0]=sliced; options.axes[0]=1;
    assert(me_artifact_eval_array(artifact,inputs,2,2,shape,&options,output,16,&report,&error)==0);
    assert(output[0]==66 && output[1]==75);
    /* Every invalid descriptor is rejected before output writes. */
    output[0]=12345; inputs[0].capacity=8;
    assert(me_artifact_eval_array(artifact,inputs,2,2,shape,&options,output,16,&report,&error)==ME_ARTIFACT_ERR_BINDING);
    assert(output[0]==12345); inputs[0]=sliced;
    assert(me_artifact_eval_array(artifact,inputs,2,2,shape,&options,x,sizeof(x),&report,&error)==ME_ARTIFACT_ERR_BINDING);
    options.axes[0]=3;
    assert(me_artifact_eval_array(artifact,inputs,2,2,shape,&options,output,sizeof(output),&report,&error)==ME_ARTIFACT_ERR_BINDING);
    assert(me_array_slice(&sliced,1,0,4,1,&reshaped,&error)==ME_ARTIFACT_ERR_BINDING);
    /* Modular combine stage, including crossing a tile boundary. */
    x[0]=INT64_MAX; x[1]=1; x[2]=0; y[0]=y[1]=y[2]=0;
    inputs[0]=(me_array_view){.name="x",.dtype=ME_INT64,.base=x,.capacity=24,.rank=1,.shape={3},.strides={8}};
    options.naxes=-1; options.tile_items=1; int64_t three[]={3};
    assert(me_artifact_eval_array(artifact,inputs,2,1,three,&options,output,8,&report,&error)==0);
    assert(output[0]==INT64_MIN);
    me_artifact_free(artifact);
    puts("native broadcast/axes/views/bounds/modular combination: passed");
    return 0;
}
