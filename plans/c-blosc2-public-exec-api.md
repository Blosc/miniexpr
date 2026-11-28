# C-Blosc2 Public Compute API Plan (`include/b2nd_exec.h`)

**Status:** C1–C3 implemented and locally qualified on macOS ARM64 (`c-blosc2` + `python-blosc2`); C4 / cross-platform CI remains open before ABI freeze per [`plans/execution-plan-migration.md`](execution-plan-migration.md) and [`plans/execution-migration-status.md`](execution-migration-status.md).

**Goal:** Expose a minimal, ABI-stable, public C-Blosc2 array compute interface (`include/b2nd_exec.h`, plus optional `include/b2nd_exec_kernel.h`) over the existing internal executor (`blosc/b2nd-exec*.c`), usable directly by pure-C/FFI applications and as the sole backend for `python-blosc2`—replacing direct inclusion of private `blosc/b2nd-exec*.h` headers.

---

## 1. Architectural & Packaging Decisions

1. **Always-installed self-contained public header (`include/b2nd_exec.h`):**
   - Installed in all C-Blosc2 builds (including compression-only builds with MiniExpr disabled).
   - Zero `#include "miniexpr*.h"` dependency. Defines its own self-contained status codes, error struct, dtype enum, operand/output descriptors, and `struct_size`-versioned option/info/report structs.
   - In compression-only C-Blosc2 builds, `b2nd_exec_available()` returns `false` and all entry points fail cleanly with `B2ND_EXEC_ERR_UNSUPPORTED` stubs.
2. **Optional companion header for direct `me_kernel` callers (`include/b2nd_exec_kernel.h`):**
   - Installed only when C-Blosc2 is built with MiniExpr support.
   - Exposes `b2nd_plan_from_kernel(...)` so callers that already link MiniExpr (such as `python-blosc2` for authored Menudet handles) can pass prepared `me_kernel` handles without bundling MiniExpr types into `b2nd_exec.h`.
3. **Target callers from v1:**
   - Serves **both** pure-C/C++/Rust applications (`b2nd_eval`, `b2nd_eval_into`, `b2nd_plan_from_expr`) **and** `python-blosc2` (`b2nd_plan_from_graph`, `b2nd_plan_compose`, `b2nd_plan_from_kernel`, `b2nd_schedule_*`, `b2nd_execute`), collapsing the 36 internal `blosc/b2nd-exec*.h` functions into a clean public surface.
   - Internal headers (`blosc/b2nd-exec.h`, `blosc/b2nd-exec-storage.h`, `blosc/b2nd-exec-stages.h`) remain non-installed implementation details.
4. **Expression / language scope in v1 (zero-bloat split):**
   - `b2nd_plan_from_expr` / `b2nd_eval` / `b2nd_eval_into` support **elementwise formulas** (routed to `me_prepare_formula`) and **Menudet 0.1/0.2 UDF sources** (routed to `me_prepare_menudet`, selected explicitly via `b2nd_exec_lang_t` or auto-detected), plus optional **struct-based reduction parameters** (`sum`, `prod`, `min`, `max`, `any`, `all`, `argmin`, `argmax`, `var`, `std`, with `axes`, `keepdims`, `initial`, `ddof`, `where` mask slot, and explicit accumulator dtype).
   - Full multi-stage expressions and shared-subexpression DAGs remain on `b2nd_plan_from_graph` and `b2nd_plan_compose` (no duplicate C string-to-DAG parser required in v1; a C string parser can be added behind `b2nd_plan_from_expr` later with no API break).
5. **Reference-counted plan lifetimes:**
   - `b2nd_plan_t` carries an atomic reference count. Every `b2nd_schedule_t` and composed multi-stage `b2nd_plan_t` retains a reference to its parent/child plans and releases on `*_free`, so freeing a `b2nd_plan_t` before its `b2nd_schedule_t` (or composed parent) is always memory-safe across C and FFI callers.
6. **Optional output context inheritance in `b2nd_eval`:**
   - Passing `out_ctx == NULL` derives output shape, dtype, chunk/block geometry, and compression defaults automatically from inputs and creates a new `b2nd_array_t`.
   - Passing a non-`NULL` `out_ctx` lets the caller control destination storage/urlpath/cparams/chunkshape/blockshape.
   - `b2nd_eval_into` evaluates directly into any caller-owned `b2nd_output_t` (pre-allocated `b2nd_array_t`, resident view, or write callback).
7. **Append-only `struct_size` ABI versioning:**
   - Every public non-opaque struct starts with `size_t struct_size`, paired with `*_INITIALIZER` and `*_V1_SIZE` macros.
   - New fields in minor releases are appended strictly after previous fields and guarded by `struct_size >= ..._VN_SIZE`, preventing `_ex` suffix proliferation while preserving binary compatibility across dynamically linked upgrades.

---

## 2. Proposed Public API Surface

### 2.1 Status Codes, Dtypes & Tagged Bindings (`include/b2nd_exec.h`)

- **Status (`b2nd_exec_status_t`):**
  - `B2ND_EXEC_SUCCESS = 0`
  - `B2ND_EXEC_ERR_INVALID_ARG = -1`
  - `B2ND_EXEC_ERR_UNSUPPORTED = -2`
  - `B2ND_EXEC_ERR_SOURCE = -3`
  - `B2ND_EXEC_ERR_SIGNATURE = -4`
  - `B2ND_EXEC_ERR_EVAL = -5`
  - `B2ND_EXEC_ERR_OOM = -6`
- **Diagnostics (`b2nd_exec_error_t`):** `struct_size`, `status`, `line`, `column`, `stage` (`SIZE_MAX` if N/A), `node` (`UINT32_MAX` if N/A), `message[256]`.
- **Dtypes (`b2nd_exec_dtype_t`):** `AUTO`, `BOOL`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`, `FLOAT32`, `FLOAT64`, `COMPLEX64`, `COMPLEX128`, `STRING`, `BYTES` (with automatic resolution from `b2nd_array_t.dtype` for compressed array inputs).
- **Operands & Outputs:**
  - `b2nd_operand_t` (`kind`: `B2ND_OPERAND_ARRAY`, `B2ND_OPERAND_VIEW`, `B2ND_OPERAND_SOURCE`, union payload carrying `const b2nd_array_t *`, strided/endian resident view + allocation bounds, or metadata + read span callback).
  - `b2nd_output_t` (`kind`: `B2ND_OUTPUT_ARRAY`, `B2ND_OUTPUT_VIEW`, `B2ND_OUTPUT_SINK`, union payload carrying `b2nd_array_t *` compressed sink, contiguous/strided writable view with optional `staging_budget` for transactional alias safety, or metadata + write span callback).
- **Options, Metadata & Reports (all `struct_size`-versioned):**
  - `b2nd_exec_options_t`: compile/preparation policy (`execution`, `compiler`, `math_backend`, `fp_mode`, `math_accuracy`, `casting`, `integer_powers`, `inherit_environment`).
  - `b2nd_expr_options_t`: expression language (`AUTO`, `FORMULA`, `MENUDET_0_1`, `MENUDET_0_2`), output dtype constraint, cardinality/context rank, plus optional reduction (`reduction`, `axes`, `naxes`, `keepdims`, `initial`, `ddof`, `has_where_slot`, `accumulator`).
  - `b2nd_schedule_options_t`: `tile_items`, `max_mapped_bytes`, `max_intermediate_bytes`, `pairwise_sum`, optional logical `partition_spec`.
  - `b2nd_run_options_t`: `workers`, `max_worker_scratch`, `max_decode_bytes`, `max_gather_bytes`, `max_output_bytes`, `raise_mask`.
  - `b2nd_plan_info_t`, `b2nd_schedule_info_t`, `b2nd_stage_info_t`, `b2nd_exec_report_t`.

### 2.2 Public Function Summary

#### Core Surface (`include/b2nd_exec.h`) — 13 `BLOSC_EXPORT` functions

| Function | One-line docstring |
| --- | --- |
| `b2nd_exec_available` | Return whether this C-Blosc2 build includes numerical execution support (false in compression-only builds). |
| `b2nd_plan_from_expr` | Compile a formula or Menudet kernel source (with optional struct-based reduction options) into an immutable `b2nd_plan_t`. |
| `b2nd_plan_from_graph` | Compile a typed multi-op DAG descriptor (auto-splitting reductions and shared nodes into stages) into a `b2nd_plan_t`. |
| `b2nd_plan_compose` | Explicitly compose prepared child `b2nd_plan_t` stages with external and producer slot bindings into one multi-stage `b2nd_plan_t`. |
| `b2nd_plan_describe` | Query immutable output metadata, reduction properties, stage count, partitionability, JIT status, and effective FP/math policy. |
| `b2nd_plan_describe_stage` | Query a child stage's immutable plan info, source slot bindings, and origin node ID within a single-stage or composed/staged plan. |
| `b2nd_plan_free` | Release a caller reference to a `b2nd_plan_t`, freeing its resources once no schedules or composed plans retain it. |
| `b2nd_schedule_create` | Specialize a `b2nd_plan_t` against input dtypes/shapes/partitions and memory budgets without reading any data. |
| `b2nd_schedule_describe` | Query resolved output shape/dtype, scratch bounds, peak intermediate bytes, and optional per-stage schedule metadata. |
| `b2nd_schedule_admit` | Perform metadata-only admission to check whether concrete arrays, views, or callbacks are supported without decoding data. |
| `b2nd_schedule_free` | Destroy a `b2nd_schedule_t` and release its retained reference to the parent `b2nd_plan_t`. |
| `b2nd_execute` | Execute a `b2nd_schedule_t` over any mix of compressed arrays, resident views, or callbacks into a `b2nd_output_t` destination. |
| `b2nd_eval` | One-shot compile and evaluate an expression string over named `b2nd_array_t` inputs, allocating a new `b2nd_array_t` (with optional `out_ctx`). |
| `b2nd_eval_into` | One-shot compile and evaluate an expression string over `b2nd_operand_t` inputs directly into a caller-supplied `b2nd_output_t`. |

#### Optional Kernel Companion (`include/b2nd_exec_kernel.h`, installed only when built with MiniExpr)

| Function | One-line docstring |
| --- | --- |
| `b2nd_plan_from_kernel` | Wrap an already-prepared MiniExpr `me_kernel` (borrowing or taking ownership) with optional reduction, `where` mask, and accumulator cast into a `b2nd_plan_t`. |

### 2.3 Pure-C Example (`b2nd_eval`)

```c
const char *names[] = {"x", "y"};
const b2nd_array_t *inputs[] = {x, y};
b2nd_array_t *out = NULL;
b2nd_exec_error_t err = B2ND_EXEC_ERROR_INITIALIZER;

b2nd_exec_status_t st = b2nd_eval(
    "sin(x) * 2.5 + y", names, inputs, 2,
    NULL, NULL, NULL, NULL, NULL, &out, NULL, &err);
```

---

## 3. Milestones & Rollout Plan

**Prerequisite:** M8 ordinary-caller handoffs and pilot retirement are complete, so C1–C3 can roll out ahead of cross-platform CI while C4 remains gated on M9's cross-platform/pinning/default-switch gates.

| Milestone | Scope | Key Deliverables & Exit Criteria |
| --- | --- | --- |
| **C1 — Public Headers & Compression-Only Stubs** | C-Blosc2 headers, exports, CMake | Add `include/b2nd_exec.h` (always installed, zero MiniExpr includes) and optional `include/b2nd_exec_kernel.h` (installed only with MiniExpr). Implement `BLOSC_EXPORT` stub library behavior when `BLOSC2_USE_MINIEXPR=OFF` (`b2nd_exec_available() == false`). Verify headers compile cleanly under strict C99/C11/C++ warnings (`-Wall -Wextra -Werror`) in both configurations. |
| **C2 — Unified Facade, Refcounting & `b2nd_eval`** | `blosc/b2nd-exec-api.c` + C tests | Implement opaque refcounted `b2nd_plan_t` and `b2nd_schedule_t` unifying single-stage, multi-stage, partitioned, aliased, array, view, and callback dispatch. Implement `b2nd_plan_from_expr` (`me_prepare_formula` / `me_prepare_menudet` + reductions), `b2nd_eval` (optional `out_ctx` with input inheritance), and `b2nd_eval_into`. Add pure-C tests covering free-order safety (plan freed before schedule), `V1_SIZE` truncated struct ABI checks, ASan/UBSan clean runs, and standalone `b2nd_eval` C examples. |
| **C3 — `python-blosc2` Cutover to Public Headers** | `python-blosc2` Cython/C bridge | Migrate `src/blosc2/b2nd_exec_bridge.h`, `b2nd_exec_stages_bridge.h`, and `b2nd_exec.pxi` off private `c-blosc2/blosc/b2nd-exec*.h` onto installed `b2nd_exec.h` + `b2nd_exec_kernel.h`. Verify `python-blosc2` can build cleanly against an **installed** C-Blosc2 prefix (not only a source tree) with zero regressions across the full M8/M9 test matrix. |
| **C4 — Installed Consumer & Cross-Platform Gate** | Packaging, ASan/UBSan, Release | Extend `tests/installed-miniexpr` with pure-C public-header consumers (using only `include/b2nd_exec.h`, no MiniExpr headers/linkage) plus compression-only link/runtime stub verification. Qualify Release & sanitizer runs across macOS ARM64, Linux x86-64, Windows/MSVC, and WASM before freezing the ABI in a minor/major C-Blosc2 release. |
