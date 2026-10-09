/* Small C deployment example: explicit typed buffers, no Python runtime. */
#include "miniexpr_artifact.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *json = "{\"schema_version\":\"1.0\",\"language\":{\"name\":\"miniexpr\",\"version\":\"1.0\"},"
        "\"requires\":[\"numeric\"],\"source\":\"def k(x):\\n    return x * 2 + 1\\n\",\"entry_point\":\"k\","
        "\"inputs\":[{\"name\":\"x\",\"dtype\":\"int64\"}],\"constants\":[],\"output\":{\"dtype\":\"int64\","
        "\"contract\":\"elementwise\"},\"semantics\":{\"fp\":\"strict\"},\"context\":{\"ndim\":0},\"metadata\":{}}";
    int64_t x[] = {0, 1, 2}, out[3];
    me_artifact *artifact = NULL;
    me_artifact_error error;
    if (me_artifact_load(json, strlen(json), ME_JIT_OFF, &artifact, &error)) return 1;
    me_artifact_input input = {"x", ME_INT64, x, 3};
    int rc = me_artifact_eval(artifact, &input, 1, out, 3, &error);
    me_artifact_free(artifact);
    if (rc || out[0] != 1 || out[1] != 3 || out[2] != 5) return 1;
    printf("1 3 5 (native interpreter)\n");
    return 0;
}
