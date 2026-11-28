# Execution-plan migration review 3

Review of the in-progress M5 storage slice:
- `plans/execution-migration-contracts.md` ("Resident gaps and first storage gate")
- `../c-blosc2.codeberg/blosc/b2nd-exec-storage.h` and `blosc/b2nd-exec-storage.c`
- `../c-blosc2.codeberg/blosc/b2nd-exec.c` (worker threading vs. C-Blosc2 thread pool / prefilters)

---

## Summary assessment

**Verdict: Finish landing the current serial M5 slice untouched, then adopt the two architectural recommendations below for compressed sinks and parallel execution.**

The uncommitted M5 storage slice (`b2nd_exec_execute_arrays` over `b2nd_exec_reader`) is cleanly scoped and does not conflict with either recommendation:
- It keeps numerical scheduling, reducers, and FP scopes inside `b2nd-exec` while isolating chunk decode and geometry validation in `b2nd-exec-storage.c` (single codec thread `params.nthreads = 1`, zero `schunk->current_nchunk` mutation, preflight before compressed reads).
- Compressed output sinks and parallel storage execution are both explicitly deferred from this first gate, so finishing and committing this serial slice first avoids mid-flight rework.

---

## Architectural recommendations for the next M5 increments

### 1. Keep prefilters out of general resident/read parallelism — **use them only for compressed output sinks**

- **Do not** bring back dummy compression contexts (`resident_prefilter`) for resident maps, reductions, strided/broadcast views, or staged DAGs. Direct scheduling in `b2nd-exec.c` handles deterministic logical-order reductions, N-D strided/reversed outputs, transactional alias staging, and strict per-task FP flag unions far more cleanly than wrapping them in fake compression passes.
- **Do** use Blosc2 prefilters for **M5 compressed output sinks** (`plans/execution-plan-migration.md`, M5): when materializing directly into a compressed `b2nd_array_t` destination, running kernel evaluation inside the prefilter computes each block in cache right before shuffle and compression with zero intermediate chunk copy, preserving the speed advantage that made prefilters effective in original `python-blosc2`.

### 2. Move `b2nd-exec.c` off raw per-call `pthread_create` onto C-Blosc2's persistent thread pool

Original `python-blosc2` prefilters achieved low dispatch overhead primarily because they ran on C-Blosc2's **persistent thread pool** (`blosc_shared_pool` in `../c-blosc2.codeberg/blosc/blosc2.c`), not because of the prefilter API itself.

Currently, `../c-blosc2.codeberg/blosc/b2nd-exec.c` uses raw per-call `pthread_create`/`pthread_join` on POSIX and forces `#ifdef _WIN32 ntasks = 1`:
1. **Dispatch latency:** Spawning and joining OS threads on every `b2nd_exec_execute*` call adds significant overhead on small/medium resident arrays compared to waking persistent pool workers.
2. **Windows stays serial:** C-Blosc2 already provides cross-platform threading (`../c-blosc2.codeberg/blosc/win32/threading.c`, `blosc2_run_parallel` in `blosc/blosc2.c`), whereas `b2nd-exec.c` currently disables Windows partitioning.
3. **Ecosystem integration and M5 oversubscription:** Raw `pthread_create` bypasses `blosc2_set_threads_callback` and `blosc_shared_pool` coordination, which matters once M5 schedules parallel outer tasks over compressed reads/writes alongside inner codec threads.

**Recommended design:**
- Route `b2nd-exec.c`'s parallel job launch through a small internal launcher/dispatch hook backed by C-Blosc2's persistent thread pool (`blosc_shared_pool`) / `blosc2_set_threads_callback` and `blosc2_pthread_*` (unlocking Windows too).
- Keep the standalone fallback / test-overridable hook intact in `b2nd-exec.c` so the isolated ASan fixture (`tests/exec-plan`, `faults.c`) continues building without linking full `blosc2` internals and keeps deterministic thread-start failure injection coverage.

### 3. Follow-up performance note for `b2nd-exec-storage.c` (`storage_read`)

- The current `storage_read` implementation in `../c-blosc2.codeberg/blosc/b2nd-exec-storage.c` computes N-D chunk/block/item indices per element via integer division/modulo on every lane (`for (size_t lane = 0; lane < count; lane++)`).
- That is completely appropriate for proving exact padded/chunked correctness at the first M5 gate, including arbitrary trailing/edge chunks.
- Before benchmarking M5 throughput (`bench/`), replace the inner scalar loop with contiguous inner-block span copies (`memcpy` of maximal contiguous runs along the fastest dimension inside the active chunk/block) while keeping the general N-D step for partial edge tiles.
