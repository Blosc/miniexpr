#!/usr/bin/env python3
"""
Optional Numba baseline for Mandelbrot-style kernel timing.

Usage:
  python bench/benchmark_mandelbrot_numba.py [widthxheight | width height] [repeats] [max_iter]
"""

import sys
import time

try:
    import numpy as np
except Exception as exc:
    print(f"numpy unavailable: {exc}")
    sys.exit(1)

try:
    import numba as nb
except Exception as exc:
    print(f"numba unavailable: {exc}")
    sys.exit(1)


def parse_dims_arg(arg: str) -> tuple[int, int]:
    if "x" in arg.lower():
        parts = arg.lower().split("x", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("invalid widthxheight format")
        width = int(parts[0])
        height = int(parts[1])
    else:
        raise ValueError("expected widthxheight")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return width, height


@nb.njit(cache=True)
def mandelbrot_kernel(cr, ci, max_iter, out):
    n = cr.shape[0]
    for idx in range(n):
        zr = np.float32(0.0)
        zi = np.float32(0.0)
        escape_iter = np.float32(max_iter)
        for it in range(max_iter):
            zr2 = zr * zr
            zi2 = zi * zi
            if zr2 + zi2 > np.float32(4.0):
                escape_iter = np.float32(it)
                break
            zr_new = zr2 - zi2 + cr[idx]
            zi = np.float32(2.0) * zr * zi + ci[idx]
            zr = zr_new
        out[idx] = escape_iter


def main():
    width = 1200
    height = 800
    repeats = 6
    max_iter = 200

    argi = 1
    if len(sys.argv) > 1:
        if "x" in sys.argv[1].lower():
            try:
                width, height = parse_dims_arg(sys.argv[1])
            except Exception as exc:
                print(f"invalid size arg '{sys.argv[1]}': {exc} (use widthxheight)")
                return 1
            argi = 2
        else:
            if len(sys.argv) < 3:
                print("invalid size args: expected width height or widthxheight")
                return 1
            try:
                width = int(sys.argv[1])
                height = int(sys.argv[2])
            except Exception:
                print("invalid size args: expected width height or widthxheight")
                return 1
            argi = 3

    nitems = width * height
    if len(sys.argv) > argi:
        repeats = int(sys.argv[argi])
    if len(sys.argv) > argi + 1:
        max_iter = int(sys.argv[argi + 1])
    if len(sys.argv) > argi + 2:
        print("too many args")
        return 1

    if nitems <= 0 or repeats <= 0 or max_iter <= 0:
        print(
            f"invalid args: width={width} height={height} repeats={repeats} max_iter={max_iter} "
            "(size via widthxheight or width height; max_iter > 0)"
        )
        return 1

    x = np.linspace(-2.0, 0.6, width, dtype=np.float32)
    y = np.linspace(1.1, -1.1, height, dtype=np.float32)
    cr = np.tile(x, height)
    ci = np.repeat(y, width)
    out = np.empty_like(cr)

    t0 = time.perf_counter_ns()
    mandelbrot_kernel(cr, ci, max_iter, out)
    t1 = time.perf_counter_ns()

    best_ns = None
    for _ in range(repeats):
        run_start = time.perf_counter_ns()
        mandelbrot_kernel(cr, ci, max_iter, out)
        run_end = time.perf_counter_ns()
        run_ns = run_end - run_start
        if best_ns is None or run_ns < best_ns:
            best_ns = run_ns

    stride = max(1, nitems // 17)
    checksum = float(out[::stride].sum())
    compile_ms = (t1 - t0) / 1.0e6
    eval_ms = float(best_ns) / 1.0e6
    ns_per_elem = float(best_ns) / float(nitems)

    print("benchmark_mandelbrot_numba")
    print(f"width={width} height={height} repeats={repeats} max_iter={max_iter}")
    print(f"{'mode':12s} {'compile_ms':>12s} {'eval_ms_best':>14s} {'ns_per_elem':>12s} {'checksum':>12s}")
    print(f"{'numba-cold':12s} {compile_ms:12.3f} {compile_ms:14.3f} {(t1 - t0) / float(nitems):12.3f} {checksum:12.3f}")
    print(f"{'numba-warm':12s} {0.0:12.3f} {eval_ms:14.3f} {ns_per_elem:12.3f} {checksum:12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
