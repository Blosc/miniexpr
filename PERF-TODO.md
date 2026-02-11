# Performance TODO

## Open Issue: `cc` Backend Underperforms `tcc` on Math Kernels

- Observation:
  - In `bench/benchmark_dsl_jit_math_kernels`, `# me:compiler=cc` can be significantly slower than `# me:compiler=tcc` on warm JIT eval for kernels such as `sin`, `exp`, `pow`, `sinpi`, `cospi`.
  - This occurs even when trace logs show `jit codegen: runtime math bridge enabled`.
- Goal:
  - Identify and remove `cc`-path overhead so warm eval performance is at least on par with `tcc` for bridge-lowered kernels.
- Initial investigation checklist:
  - Verify per-kernel lowering parity between `tcc` and `cc` generated code paths.
  - Inspect `cc` codegen/ABI details (`-fPIC`, shared-object call boundaries, symbol resolution) for avoidable overhead.
  - Add per-kernel diagnostics in benchmark output (e.g. lowered/not-lowered markers) to correlate performance with lowering decisions.
  - Re-benchmark on fixed env (`CC`, `CFLAGS`, same CPU governor) and record deltas.
