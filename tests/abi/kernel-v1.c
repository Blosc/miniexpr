/* Frozen descriptor caller: deliberately does not include current API headers.
 * Keep these layouts unchanged when testing future append-only extensions.
 * These are the int-backed enum and object-pointer slots of the qualified ABI;
 * the receiver checks byte extents before importing the representation. */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    size_t struct_size;
    const void *inputs;
    size_t ninputs;
    const void *captures;
    size_t ncaptures;
    int output_dtype;
    size_t output_itemsize;
    int cardinality;
    int context_ndim;
    int casting;
    bool integer_powers;
    uint8_t reserved[3];
} signature_v1;

typedef struct {
    size_t struct_size;
    int execution;
    bool inherit_environment;
    uint8_t reserved[3];
} prepare_options_v1;

typedef struct {
    size_t struct_size;
    unsigned raise_mask;
    void *fp_status;
    int execution;
    uint8_t reserved[4];
} run_options_v1;

void me_test_kernel_v1_caller(void (*receive)(const void *, const void *, const void *)) {
    signature_v1 signature;
    prepare_options_v1 options;
    run_options_v1 run_options;
    /* Poison unnamed padding, then initialize every declared field. This
     * catches a future reader mistaking old padding for an appended option. */
    memset(&signature, 0xa5, sizeof(signature));
    signature.struct_size = sizeof(signature);
    signature.inputs = signature.captures = NULL;
    signature.ninputs = signature.ncaptures = 0;
    signature.output_dtype = 0; /* ME_AUTO */
    signature.output_itemsize = 0;
    signature.cardinality = signature.context_ndim = signature.casting = 0;
    signature.integer_powers = false;
    memset(signature.reserved, 0, sizeof(signature.reserved));
    memset(&options, 0xa5, sizeof(options));
    options.struct_size = sizeof(options);
    options.execution = 0; /* ME_EXEC_INTERPRET */
    options.inherit_environment = false;
    memset(options.reserved, 0, sizeof(options.reserved));
    memset(&run_options, 0xa5, sizeof(run_options));
    run_options.struct_size = sizeof(run_options);
    run_options.raise_mask = 0;
    run_options.fp_status = NULL;
    run_options.execution = 0; /* ME_RUN_PREPARED */
    memset(run_options.reserved, 0, sizeof(run_options.reserved));
    receive(&signature, &options, &run_options);
}
