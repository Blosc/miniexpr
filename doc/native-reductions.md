# Shared reduction primitives

`miniexpr_reduce.h` exposes `ME_EXTREMA_REDUCE_API` (version 1) and
`ME_MOMENTS_REDUCE_API` (version 1) for embedding schedulers. They are
header-only, allocation-free interfaces; input buffers may be unaligned. Callers
validate buffer capacities and establish their floating-point environment.
Compile without fast-math.

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

## Moments

`me_moments_span()` reduces a float32/float64 span into four doubles: count,
origin, mean offset and centered sum of squares M2. Float32 inputs widen to
double. Each span is scanned twice, so large offsets keep their low bits without
the cancellation of `E[x*x] - E[x]*E[x]` or Welford's per-lane division.

`me_moments_merge()` applies Chan's parallel update in logical order; the origin
stays separate from the mean offset so rounding cannot discard small variation.
`me_moments_finish()` applies `ddof` and optionally the square root. Counts must
be exactly representable, and merging rejects non-integral or overflowing counts.

Non-finite input yields a NaN variance, matching a centered two-pass reducer.
Callers cast the final result to their public dtype inside their own floating
environment, since narrowing can overflow.
