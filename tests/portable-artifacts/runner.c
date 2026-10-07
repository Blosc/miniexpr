/* Standalone affine artifact demonstration; no libpython or Python preprocessing. */
#include "miniexpr_artifact.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s affine.json off|default|on\n", argv[0]);
        return 2;
    }
    me_jit_mode mode;
    if (!strcmp(argv[2], "off")) mode = ME_JIT_OFF;
    else if (!strcmp(argv[2], "default")) mode = ME_JIT_DEFAULT;
    else if (!strcmp(argv[2], "on")) mode = ME_JIT_ON;
    else return 2;
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        perror(argv[1]);
        return 2;
    }
    char *json = malloc(ME_ARTIFACT_MAX_BYTES + 1);
    if (!json) {
        fclose(file);
        return 2;
    }
    size_t length = fread(json, 1, ME_ARTIFACT_MAX_BYTES + 1, file);
    bool failed = ferror(file) != 0;
    fclose(file);
    me_artifact *artifact = NULL;
    me_artifact_error error;
    int rc = failed ? ME_ARTIFACT_ERR_FORMAT : me_artifact_load(json, length, mode, &artifact, &error);
    free(json);
    if (rc != ME_ARTIFACT_SUCCESS) {
        fprintf(stderr, "load error %d: %s\n", rc, failed ? "file read failed" : error.message);
        return 1;
    }
    if (me_artifact_ninputs(artifact) != 1 ||
        strcmp(me_artifact_input_name(artifact, 0), "x") ||
        me_artifact_input_dtype(artifact, 0) != ME_FLOAT64 ||
        me_artifact_output_dtype(artifact) != ME_FLOAT64 ||
        me_artifact_has_jit(artifact)) {
        fprintf(stderr, "expected affine float64 signature and requested backend\n");
        me_artifact_free(artifact);
        return 1;
    }
    double x[] = {0, 1, 2, 3};
    double output[4];
    me_artifact_input inputs[] = {{"x", ME_FLOAT64, x, 4}};
    rc = me_artifact_eval(artifact, inputs, 1, output, 4, &error);
    if (rc == ME_ARTIFACT_SUCCESS) {
        printf("jit=%d\n", (int)me_artifact_has_jit(artifact));
        for (int i = 0; i < 4; i++) {
            printf("%.17g\n", output[i]);
            if (output[i] != 2 * x[i] - 1) rc = ME_ARTIFACT_ERR_EVAL;
        }
    } else {
        fprintf(stderr, "eval error %d (native %d): %s\n", rc, error.native_status, error.message);
    }
    me_artifact_free(artifact);
    return rc == ME_ARTIFACT_SUCCESS ? 0 : 1;
}
