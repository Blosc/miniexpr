# Execution Migration Review 4 — M5/M6 Follow-ups, First M7 Slice & Ordered-SUM Checkpoint

**Context:** Review of the working-tree follow-ups and first native M7 slice over miniexpr `0e58ed5`, C-Blosc2 `fdbf3a8d`, and Python-Blosc2 `c1acd238`, as recorded in [`plans/execution-migration-status.md`](execution-migration-status.md).

## Overall Assessment

The recent increments are clean, well-scoped, and carefully validated:

- **M5/M6 hardening:** Zero-initializing `has_frame_mutex`, checking POSIX `blosc2_pthread_mutex_init` failure with cleanup/fault injection in `b2nd_exec_execute_arrays_ex`, rejecting oversized source ranks before copying into native fixed-size arrays in `blosc2_ext.pyx`, and ensuring late reader `NotImplementedError` propagates without backend replay are all real improvements.
- **First native M7 reduction-to-map slice (`blosc/b2nd-exec-stages.h`, `blosc/b2nd-exec-stages.c`, `tests/installed-miniexpr/exec-stages.c`):** Cleanly reuses `b2nd_exec_prepare`, `b2nd_exec_specialize`, and `b2nd_exec_execute_sources` via direct node/slot remapping without JSON serialization, validates intermediate budgets and per-stage bounds before callbacks, frees producer storage on all return paths, preserves exact stage root diagnostics, and enforces per-stage `raise_mask` parity with `miniexpr_graph`.
- **Ordered-SUM benchmark (`bench/compressed_reduction_plan.py`):** Produces an accurate, reproducible baseline isolating the 1-chunk cache decode amplification (`8008` decodes across `64` physical chunks = ~`125x` on `(1001, 1003)` float64).

---

## Findings & Suggested Improvements

### 1. Missing `blosc2_pthread_mutex_init` error check in `b2nd_exec_execute_to_array`

- **Location:** `c-blosc2/blosc/b2nd-exec-storage.c` (lines 1179–1188)
- **Observation:** The unchecked `blosc2_pthread_mutex_init` loop was fixed in `b2nd_exec_execute_arrays_ex` (with matching fault injection in `tests/exec-storage/faults.c`), but the nearly identical loop inside `b2nd_exec_execute_to_array` still ignores the return value:
  ```c
  blosc2_pthread_mutex_t frame_mutexes[B2ND_EXEC_MAX_INPUTS];
  bool has_frame_mutex[B2ND_EXEC_MAX_INPUTS];
  memset(has_frame_mutex, 0, sizeof(has_frame_mutex));
  for (size_t i = 0; i < ninputs; i++) {
      if (nworkers > 1 && sink.needed[i] && sink.is_reader[i] && arrays[i]->sc->frame) {
          blosc2_pthread_mutex_init(&frame_mutexes[i], NULL);
          has_frame_mutex[i] = true;
          sink.frame_mutex[i] = &frame_mutexes[i];
      }
  }
  ```
- **Recommendation:** Extract a small shared helper for initializing/cleaning up `frame_mutexes` and cover `b2nd_exec_execute_to_array` in `exec_storage_faults` with `mutex_fail_at`.

---

### 2. Removing `except NotImplementedError` in `MenudetLazyArray._cblosc2_partition_values` can break preflight fallback

- **Location:** `python-blosc2/src/blosc2/menudet_lazy.py` (lines 182–186)
- **Observation:** Removing `try ... except NotImplementedError: return None` prevents late reader `NotImplementedError` from triggering fallback (which is right!). However, `b2nd_exec_execute_partitions` also returns `ME_ERR_UNSUPPORTED` (mapped to `NotImplementedError`) during **metadata-only preflight** for valid inputs not yet admitted by C storage/partition execution—for example, compressed `NDArray` inputs using postfilters, plugin codecs (`compcode > BLOSC_ZSTD`, such as ZFP/NDLZ/OpenZL), plugin filters (`> BLOSC_DELTA`, such as `BYTEDELTA`), or non-native complex/string views. Previously these returned `None` and ran on Python's logical partition loop; now they will raise `NotImplementedError` on opt-in builds.
- **Recommendation:** Separate preflight admission rejection from callback execution failures—either by checking those `NDArray`/view constraints in Python metadata preflight before calling `evaluate_partitions`, or by only re-raising `NotImplementedError` from `cb_state.error` while allowing preflight-only `ME_ERR_UNSUPPORTED` (when zero callbacks/reads ran) to fall back.

---

### 3. False-positive aliasing check on zero-capacity buffers in `b2nd_exec_staged_execute_sources`

- **Location:** `c-blosc2/blosc/b2nd-exec-stages.c` (lines 282–286)
- **Observation:**
  ```c
  uintptr_t begin = (uintptr_t)views[i].base, end = begin + views[i].capacity;
  uintptr_t out_begin = (uintptr_t)output.data, out_end = out_begin + output.capacity;
  if (out_begin < end && begin < out_end) {
  ```
  If `views[i].capacity == 0` with a non-NULL `base` inside `output` (or `output.capacity == 0` with a non-NULL `data` inside `views[i]`), `[begin, begin)` strictly satisfies `out_begin < begin && begin < out_end` and rejects a valid non-overlapping empty view. Compare `exec_plan_bindings` in `blosc/b2nd-exec.c:1554-1557`, which guards with `inputs[i].capacity && output_bytes && output->capacity`.
- **Minor alignment:** `b2nd-exec.c` reports `ME_ERR_UNSUPPORTED` for aliased output, whereas `b2nd-exec-stages.c` currently returns `ME_ERR_INVALID_ARG`.

---

### 4. Premature intermediate allocation & `intermediate_bytes` on preflight failure

- **Location:** `c-blosc2/blosc/b2nd-exec-stages.c` (lines 300–327)
- **Observation:** `malloc(bytes)` for the producer intermediate runs **before** `stage_bind` validates stage 0 and stage 1 metadata, output extents, and gather budgets. In addition, `local.intermediate_bytes = bytes;` is assigned outside `if (status == ME_SUCCESS)`, so a failing `stage_bind` (`entered_stages == 0`) still reports non-zero `intermediate_bytes`.
- **Recommendation:** Run `stage_bind` preflight for both stages before allocating the intermediate buffer (passing the planned `strides` / NULL base or checking views/budgets first), or set `local.intermediate_bytes = bytes;` only after allocation/preflight succeeds and execution begins.

---

### 5. `fp.supported` merge uses OR instead of AND across observed stages

- **Location:** `c-blosc2/blosc/b2nd-exec-stages.c` (lines 335–338)
- **Observation:**
  ```c
  me_fp_status fp = local.stages[stage].execution.fp;
  local.fp.flags |= fp.flags;
  /* No observation (e.g. identity integer reduction) is not a failed FP scope. */
  if (fp.supported) local.fp.supported = true;
  ```
  If stage 0 opens an FP scope and observes `supported == true`, while stage 1 attempts to observe FP and reports `supported == false`, `if (fp.supported) local.fp.supported = true;` leaves `local.fp.supported == true` instead of combining conservatively.
- **Recommendation:** Track `bool fp_seen = false;` for stages that actually observed an FP scope (or had a non-identity/floating kernel) and combine `local.fp.supported = fp_seen ? (local.fp.supported && fp.supported) : fp.supported;` only over observing stages (similar to `exec_combine` in `b2nd-exec.c`).

---

### 6. Stack footprint and shared-input test coverage before generalizing M7

- **Stack footprint (`c-blosc2/blosc/b2nd-exec-stages.c:312-313`):**
  `b2nd_exec_source child_sources[2][B2ND_EXEC_MAX_INPUTS]` and `b2nd_exec_view child_views[2][B2ND_EXEC_MAX_INPUTS]` occupy ~88 KiB on the stack. This is fine for 2 stages on Linux/macOS main threads today, but when generalizing beyond 2 stages (`MAX_STAGES`) or calling from worker threads (512 KiB default stack on macOS), preparing/binding per stage or heap-allocating `plan->counts[stage]` entries will avoid stack-overflow risk.
- **Canonical shared-input test (`c-blosc2/tests/installed-miniexpr/exec-stages.c`):**
  The current tests cover disjoint inputs (`sum(x, axis=0) + y`) and shared producer output (`sum(x * c) + sum(x * c)`). Adding an explicit `x - sum(x)` case (especially with `b2nd_exec_source` callbacks) will directly verify shared input reads across both stage 0 and stage 1.

---

## Performance Note — M5 Ordered `SUM` Decode Cache

The observation in [`compressed_reduction_plan.py`](../../python-blosc2.codeberg/bench/compressed_reduction_plan.py) points cleanly to the next storage optimization: `b2nd_exec_execute_arrays_ex` currently holds a single chunk slot (`cached_chunk`) in `storage_input`, regardless of `max_decode_bytes`. Extending `storage_input` with a bounded **multi-chunk LRU cache** sized dynamically within `max_decode_bytes` (matching the pattern used by `cache_slots` / `STORAGE_BLOCK_CACHE_SLOTS` in `b2nd_exec_execute_to_array`)—for instance, holding one chunk-row of 8 chunks (1 MiB) on `(1001, 1003)` float64 with `(128, 128)` chunks—should bring `decodes` down from `8008` (`~125x`) to `64` (`1.00x`) while keeping exact serial logical reduction order and scratch/report accounting unchanged.
