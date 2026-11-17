# Shared extrema primitives

`miniexpr_reduce.h` exposes `ME_EXTREMA_REDUCE_API` (version 1) for embedding
schedulers. It is a header-only, allocation-free interface; input buffers may be
unaligned. Callers validate buffer capacities and establish their floating-point
environment. Compile without fast-math.

* `me_extrema_float()` reduces float32/float64 spans to raw value bits. It keeps
  the first NaN payload, quiets signaling NaNs and prefers negative zero for min
  and positive zero for max. The internal direct-input reducer delegates here,
  including for NaN-containing spans.
* `me_argextrema_update()` scans a typed span into `me_argextrema_state`, carrying
  value bits and an absolute int64 logical position. Supported dtypes are bool,
  signed/unsigned 8/16/32/64-bit integers, float32 and float64.
* `me_argextrema_merge()` combines states by value and position. Finite ties,
  including signed zeros, choose the lowest position. NaNs outrank non-NaNs;
  among NaNs the lowest position wins. Integer values never pass through
  floating-point conversions.

Initialize a state with `{0}`, then supply each partial's logical origin.
Merging can happen in any completion order. Keep dtype and min/max selection
consistent across updates and merges. Empty updates leave the state unchanged;
unsupported types and out-of-range origins return false.

Index reductions preserve selected value bits as auxiliary state; their public
result is the index, not an extrema value with the extrema signed-zero rule.
The native tests own dtype, NaN, tie-breaking and partition-merge semantics.
