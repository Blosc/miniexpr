/* Arrow-style varlen output for string expressions: int64 offsets + a tight
 * byte blob (large_string for 'U', large_binary for 'S').
 *
 * The evaluator keeps its fixed-width internal buffers; this packs the finished
 * block down on the way out.  That deletes all three width-inflation factors
 * from the *stored* result -- UCS4, the case-mapping width bound, and the
 * fixed-width padding -- without touching the buffer model that every
 * DEFINE_ME_EVAL instantiation and the DSL interpreter share.
 *
 * ponytail: pack-after-eval, not varlen intermediates.  The extra pass is over
 * a block that is already in cache; the intermediates are per-block scratch
 * that never reaches storage, so widening them buys nothing measurable.
 * Upgrade path, if this pass ever shows up on a profile: teach
 * eval_string_expr() to write offsets directly and drop the scratch block.
 *
 * Built only on the public API, so it stays decoupled from functions.c.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "miniexpr.h"

/* Encode one codepoint; unpaired surrogates and out-of-range values become
 * U+FFFD so the blob is always well-formed UTF-8. */
static size_t utf8_encode(uint32_t cp, uint8_t out[4]) {
    if (cp >= 0x110000u || (cp >= 0xD800u && cp <= 0xDFFFu)) cp = 0xFFFDu;
    if (cp < 0x80u) {
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800u) {
        out[0] = (uint8_t)(0xC0u | (cp >> 6));
        out[1] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        out[0] = (uint8_t)(0xE0u | (cp >> 12));
        out[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    out[0] = (uint8_t)(0xF0u | (cp >> 18));
    out[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 4;
}

size_t me_varlen_data_bound(const me_expr *expr, int block_nitems) {
    if (!expr || block_nitems <= 0) return 0;
    const me_dtype dtype = me_get_dtype(expr);
    if (dtype != ME_STRING && dtype != ME_BYTES) return 0;
    const size_t itemsize = me_get_itemsize(expr);
    if (itemsize == 0) return 0;
    /* A UCS4 slot spends 4 bytes per codepoint and UTF-8 never needs more, so
     * the fixed-width block is its own worst case; 'S' copies verbatim. */
    return (size_t)block_nitems * itemsize;
}

int me_eval_varlen(const me_expr *expr, const void **vars_block, int n_vars,
                   int block_nitems, int64_t *offsets,
                   void *data, size_t data_capacity, size_t *data_used,
                   const me_eval_params *params) {
    if (!expr) return ME_EVAL_ERR_NULL_EXPR;
    if (!offsets || !data_used || block_nitems <= 0) return ME_EVAL_ERR_INVALID_ARG;
    if (!data && data_capacity != 0) return ME_EVAL_ERR_INVALID_ARG;

    const me_dtype dtype = me_get_dtype(expr);
    if (dtype != ME_STRING && dtype != ME_BYTES) return ME_EVAL_ERR_INVALID_ARG;
    const size_t itemsize = me_get_itemsize(expr);
    const size_t unit = (dtype == ME_STRING) ? sizeof(uint32_t) : 1;
    if (itemsize == 0 || (itemsize % unit) != 0) return ME_EVAL_ERR_INVALID_ARG;
    const size_t slot_units = itemsize / unit;

    unsigned char *scratch = malloc((size_t)block_nitems * itemsize);
    if (!scratch) return ME_EVAL_ERR_OOM;

    const int rc = me_eval(expr, vars_block, n_vars, scratch, block_nitems, params);
    if (rc != ME_EVAL_SUCCESS) {
        free(scratch);
        return rc;
    }

    uint8_t *out = (uint8_t *)data;
    size_t used = 0;
    offsets[0] = 0;
    for (int i = 0; i < block_nitems; i++) {
        const unsigned char *slot = scratch + (size_t)i * itemsize;
        /* Same length rule as string_view_at(): a value runs to the first NUL,
         * and a slot-filling value carries no terminator at all. */
        if (unit == 1) {
            size_t len = 0;
            while (len < slot_units && slot[len] != 0) len++;
            if (len > data_capacity - used) {
                free(scratch);
                return ME_EVAL_ERR_INVALID_ARG;
            }
            memcpy(out + used, slot, len);
            used += len;
        }
        else {
            const uint32_t *cps = (const uint32_t *)(const void *)slot;
            for (size_t u = 0; u < slot_units && cps[u] != 0; u++) {
                uint8_t enc[4];
                const size_t n = utf8_encode(cps[u], enc);
                if (n > data_capacity - used) {
                    free(scratch);
                    return ME_EVAL_ERR_INVALID_ARG;
                }
                memcpy(out + used, enc, n);
                used += n;
            }
        }
        offsets[i + 1] = (int64_t)used;
    }

    free(scratch);
    *data_used = used;
    return ME_EVAL_SUCCESS;
}
