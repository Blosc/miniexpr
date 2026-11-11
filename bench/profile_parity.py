"""Native Menudet/full-DSL performance qualification and ranked gap reports.

The full matrix is numerical, not a claim to enumerate every possible program.
Unsupported signatures, data-dependent errors, semantic differences, masked work
and JIT availability are recorded, never silently folded into a parity ranking.
Runs are serial; each worker alternates engine order and retains generated C.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

DTYPES = ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64")
FLOATS = ("float32", "float64")
INTEGERS = DTYPES[1:9]


@dataclasses.dataclass
class Recipe:
    name: str
    family: str
    body: str
    x: str
    y: str
    output: str
    distribution: str = "ordinary"
    mask: str = "none"
    scalar: bool = False
    ndim: int = 0
    capture: str | None = None

    def artifact(self):
        parameters = "x, y, c" if self.capture else "x, y"
        constants = [] if not self.capture else [
            {"name": "c", "dtype": "int64", "category": self.capture, "encoding": "decimal", "value": "2"}
        ]
        return {
            "schema_version": "0.1", "language": {"name": "menudet", "version": "0.1"},
            "requires": ["numeric", "control-flow", "block-reductions"],
            "source": f"def k({parameters}):\n    {self.body}\n", "entry_point": "k",
            "inputs": [{"name": "x", "dtype": self.x}, {"name": "y", "dtype": self.y}],
            "constants": constants,
            "output": {"dtype": self.output, "contract": "block_scalar" if self.scalar else "elementwise"},
            "context": {"ndim": self.ndim},
            "semantics": {"fp": "strict", "numeric": "numpy-2.5", "casting": "unsafe"},
        }

    def legacy_source(self):
        # A captured integer is inlined only for the legacy control. Its scalar
        # strength is deliberately not claimed to match Menudet's contract.
        body = re.sub(r"\bc\b", "2", self.body) if self.capture else self.body
        return f"# me:fp=strict\ndef k(x, y):\n    {body}\n"


def matrix():  # noqa: C901 -- declarative recipe coverage
    recipes = []

    def add(name, family, expression, dtype, *, output=None, other=None, body=False, **kwargs):
        recipes.append(Recipe(name, family, expression if body else f"return {expression}",
                              dtype, other or dtype, output or dtype, **kwargs))

    for dtype in DTYPES:
        for name, expr in (("add", "x + y"), ("subtract", "x - y"), ("multiply", "x * y"),
                           ("divide", "x / y"), ("floordiv", "x // y"), ("modulo", "x % y"),
                           ("power", "x ** y"), ("literal", "x + 1"), ("chain", "(x + y) * (x - y) + x"),
                           ("deep_chain", "((x + y) * 2 - y) * 3 + x * 2 - y")):
            add(name, "arithmetic", expr, dtype, output="float64" if name == "divide" and dtype not in FLOATS else None)
        for op, name in (("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge"), ("==", "eq"), ("!=", "ne")):
            add(name, "comparison", f"x {op} y", dtype, output="bool")
        for name, expr in (("and", "x and y"), ("or", "x or y"), ("not", "not x")):
            add(name, "logical", expr, dtype, output="bool")
        if dtype not in FLOATS:
            for name, expr in (("bitand", "x & y"), ("bitor", "x | y"), ("bitxor", "x ^ y"),
                               ("invert", "~x"), ("lshift", "x << y"), ("rshift", "x >> y")):
                add(name, "bitwise", expr, dtype)
        for name, expr in (("negate", "-x"), ("positive", "+x"), ("abs", "abs(x)"),
                           ("square", "square(x)"), ("sign", "sign(x)"), ("real", "real(x)"),
                           ("imag", "imag(x)"), ("conj", "conj(x)"), ("ceil", "ceil(x)"),
                           ("floor", "floor(x)"), ("trunc", "trunc(x)"), ("round", "round(x)"), ("rint", "rint(x)")):
            add(name, "unary", expr, dtype)
        for target in DTYPES:
            add(f"cast_{target}", "cast", f"{target}(x)", dtype, output=target)
        for other in DTYPES:
            # The explicit output is part of the contract, not an inference claim.
            output = "float64" if "float64" in (dtype, other) or (dtype in FLOATS) != (other in FLOATS) else dtype
            add(f"add_{other}", "mixed", "x + y", dtype, other=other, output=output)
        for name, body in (
            ("where", "return where(x > y, x - y, y - x)"),
            ("branch", "if x > y:\n        return x - y\n    return y - x"),
            ("locals", "a = x + y\n    b = a * 2\n    return b - a"),
            ("for", "s = x\n    for i in range(4):\n        s = s + y\n    return s"),
            ("while", "s = x\n    i = 0\n    while i < 4:\n        s = s + y\n        i = i + 1\n    return s"),
            ("break", "s = x\n    for i in range(8):\n        if s > 4:\n            break\n        s = s + y\n    return s"),
            ("continue", "s = x\n    for i in range(4):\n        if i == 1:\n            continue\n        s = s + y\n    return s"),
            ("early_return", "if x > y:\n        return x\n    return y"),
        ):
            add(name, "control", body, dtype, body=True)
        for reduction in ("sum", "prod", "min", "max", "mean", "any", "all"):
            output = "bool" if reduction in ("any", "all") else "float64" if reduction == "mean" else (
                "uint64" if dtype.startswith("uint") else "int64") if reduction in ("sum", "prod") and dtype not in FLOATS else dtype
            # Product over many values is intentionally an overflow/error case.
            add(reduction, "reduction", f"block_{reduction}(x)", dtype, output=output, scalar=True)
        for name, capture in (("weak_capture", "weak"), ("typed_capture", "typed_scalar")):
            add(name, "scalar", "x * c + c", dtype, capture=capture)
        add("weak_subtree", "scalar", "x + (2 * 3)", dtype)

    unary = ("acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "cos", "cosh", "erf", "erfc",
             "exp", "exp2", "expm1", "lgamma", "log", "log10", "log1p", "log2", "sin", "sinh", "sqrt",
             "tan", "tanh", "tgamma", "exp10", "sinpi", "cospi")
    binary = ("atan2", "copysign", "fdim", "hypot", "nextafter", "remainder", "fmod", "fmin", "fmax",
              "minimum", "maximum", "logaddexp", "pow")
    for dtype in FLOATS:
        for name in unary:
            add(name, "math", f"{name}(x)", dtype, distribution="greater-one" if name == "acosh" else "positive")
        for name in binary:
            add(name, "math", f"{name}(x, y)", dtype, distribution="positive")
        add("ldexp", "math", "ldexp(x, y)", dtype, other="int64", distribution="positive")
        add("fma", "math", "fma(x, y, 1.0)", dtype)
        for predicate in ("isfinite", "isinf", "isnan", "signbit"):
            add(predicate, "predicate", f"{predicate}(x)", dtype, output="bool")
        add("trig_sum", "composition", "sin(x) + cos(x) + 1", dtype)
        add("logistic", "composition", "1 / (1 + exp(-x))", dtype)
        add("math_chain", "composition", "sqrt(x) + log1p(x) * exp(-y)", dtype, distribution="positive")
        add("math_branch", "composition", "where(x > 0, log(x), sqrt(-x))", dtype)
        add("coordinate", "context", "x + _i0 + _n0 + _flat_idx + _ndim", dtype, ndim=1)
    for dtype in INTEGERS:
        for name in ("fac", "ncr", "npr"):
            add(name, "combinatoric", f"{name}(x)" if name == "fac" else f"{name}(x, y)", dtype,
                output="int64", distribution="combinatoric")
    # Audit exceptional and participation costs independently of normal parity.
    ordinary = list(recipes)
    for recipe in ordinary:
        if recipe.family in ("math", "predicate"):
            recipes.append(dataclasses.replace(recipe, distribution="exceptional"))
        if recipe.family in ("arithmetic", "bitwise", "comparison") and recipe.x in ("int8", "int64", "uint64"):
            recipes.append(dataclasses.replace(recipe, distribution="boundary"))
        if recipe.x in ("float32", "float64", "int8", "int64") and recipe.name in ("add", "where", "sin", "isfinite", "bitxor"):
            for mask in ("all", "sparse", "off"):
                recipes.append(dataclasses.replace(recipe, mask=mask))
    return recipes


def summarize(row):
    if row.get("status") != "ok" or row.get("menudet_rc") or row.get("legacy_rc"):
        row["classification"] = "validation_error" if row.get("status") == "validation_error" else "unsupported_or_error"
        return
    for profile in ("portable", "legacy"):
        values = row[f"{profile}_s"]
        row[f"{profile}_median_s"] = statistics.median(values)
        row[f"{profile}_spread"] = (max(values) - min(values)) / max(statistics.median(values), 1e-12)
        row[f"{profile}_alloc_median_s"] = statistics.median(row[f"{profile}_alloc_s"])
    row["ratio"] = row["menudet_median_s"] / max(row["legacy_median_s"], 1e-12)
    row["classification"] = (
        "different_participation" if row["mask"] != "none" else
        "different_results" if row["differences"] else
        "different_backend" if row["menudet_jit"] != row["legacy_jit"] else
        "exceptional" if row["distribution"] in ("exceptional", "boundary") else
        "different_scalar_policy" if row["capture"] else "comparable_on_domain"
    )
    row["actionable_gap"] = row["classification"] == "comparable_on_domain" and row["ratio"] > 1.25 and (
        row["menudet_median_s"] - row["legacy_median_s"] > 5e-6
    )


def report(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts = {}
    for row in rows:
        counts[row.get("classification", row.get("status"))] = counts.get(row.get("classification", row.get("status")), 0) + 1
    gaps = sorted((r for r in rows if r.get("actionable_gap")), key=lambda r: r["menudet_median_s"]-r["legacy_median_s"], reverse=True)
    summary = {"rows": len(rows), "classifications": counts, "gaps": gaps}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "classifications": counts, "gaps": len(gaps)}, indent=2))
    for row in gaps[:35]:
        print(f"{row['backend']:11} {row['family']:12} {row['name']:20} {row['x']:8} {row['y']:8} n={row['nitems']:8} "
              f"{row['ratio']:8.2f}x {row['menudet_median_s']*1000:10.3f}/{row['legacy_median_s']*1000:10.3f} ms {row['job']}")
    return rows


def main():  # noqa: C901 -- serial qualification driver
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 65536, 1048576, 10000000])
    parser.add_argument("--backends", nargs="+", default=["tcc", "clang", "gcc", "interpreter"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--family", nargs="+")
    parser.add_argument("--name", nargs="+")
    parser.add_argument("--dtype", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report(args.output / "results.jsonl")
        return
    if not args.runner or not args.runner.is_file():
        parser.error("--runner is required")
    if args.repeats < 3 or args.repeats > 31 or args.trials < 1 or any(n < 1 or n > 100000000 for n in args.sizes):
        parser.error("positive sizes/trials and 3..31 repeats required")
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / "results.jsonl"
    if results.exists() and not args.resume:
        parser.error("output already exists; use --resume or a new directory")
    completed = set()
    if results.exists():
        completed = {json.loads(line)["job"] for line in results.read_text().splitlines()}
    recipes = [r for r in matrix() if (not args.family or r.family in args.family) and
               (not args.name or r.name in args.name) and (not args.dtype or r.x in args.dtype)]
    priority = {name: i for i, name in enumerate(("arithmetic", "comparison", "bitwise", "math", "predicate",
                                                "unary", "cast", "mixed", "scalar", "control", "composition", "context", "reduction"))}
    recipes.sort(key=lambda r: (priority.get(r.family, 99), r.distribution != "ordinary", r.mask != "none",
                               ("float64", "int64", "int8", "float32", *DTYPES).index(r.x)))
    compilers = {"tcc": None, "clang": shutil.which("clang"), "gcc": shutil.which("gcc-16"), "interpreter": None}
    metadata = {"platform": platform.platform(), "python": sys.version, "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "arguments": vars(args) | {"runner": str(args.runner), "output": str(args.output)}, "recipes": len(recipes),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip(),
                "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=Path(__file__).parent)).hexdigest(),
                "compiler_versions": {name: subprocess.check_output([exe, "--version"], text=True).splitlines()[0]
                                      for name, exe in compilers.items() if exe},
                "limitations": ["One host; native serial execution; no universal numerical equivalence between profiles.",
                                "Interpreter controls are capped at 65536 items; large unavailable-JIT pairs are reported without timing.",
                                "Correctness/FP status uses a 1024-lane Menudet interpreter control, not exhaustive input-value coverage.",
                                "Masks are a Menudet-only contract and excluded from parity rankings.",
                                "First evaluation always Menudet then legacy; warm engine order alternates."]}
    metadata_path = args.output / "metadata.json"
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["runner_sha256"] != metadata["runner_sha256"]:
            parser.error("cannot resume with a different native binary; use a new output directory")
        for key in ("sizes", "backends", "repeats", "trials", "family", "name", "dtype"):
            if previous["arguments"][key] != metadata["arguments"][key]:
                parser.error(f"cannot resume with different --{key}; use a new output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    start = time.monotonic()
    with results.open("a") as stream:
        for trial in range(args.trials):
            order = recipes if trial % 2 == 0 else list(reversed(recipes))
            for index, recipe in enumerate(order):
                for backend in args.backends:
                    key = hashlib.sha256(json.dumps(dataclasses.asdict(recipe), sort_keys=True).encode()).hexdigest()[:16]
                    job = f"{trial}-{key}-{backend}"
                    if job in completed:
                        continue
                    directory = args.output / "jobs" / job
                    directory.mkdir(parents=True, exist_ok=True)
                    artifact = directory / "artifact.json"
                    source = directory / "legacy.dsl"
                    artifact.write_text(json.dumps(recipe.artifact()) + "\n")
                    source.write_text(recipe.legacy_source())
                    (directory / "cache").mkdir(exist_ok=True)
                    sizes = [n for n in args.sizes if n <= 65536] if backend == "interpreter" or recipe.distribution in ("boundary", "exceptional") else args.sizes
                    if not sizes:
                        sizes = [min(65536, min(args.sizes))]
                    if recipe.ndim:
                        sizes = [65536]
                    environment = {k: v for k, v in os.environ.items() if k not in ("CC", "CFLAGS") and not k.startswith("ME_DSL_")}
                    environment.update(ME_DSL_JIT_COMPILER="tcc" if backend in ("tcc", "interpreter") else "cc",
                                       ME_DSL_JIT_CACHE_DIR=str(directory / "cache"),
                                       ME_DSL_JIT="0" if backend == "interpreter" else "1")
                    if compilers.get(backend):
                        environment["CC"] = compilers[backend]
                    command = [str(args.runner.resolve()), str(artifact.resolve()), str(source.resolve()), backend,
                               ",".join(map(str, sizes)), str(args.repeats), recipe.distribution, recipe.mask, str(directory.resolve())]
                    if backend not in compilers or (backend in ("gcc", "clang") and not compilers[backend]):
                        observations = [{"status": "compiler_unavailable"}]
                    else:
                        try:
                            proc = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=180)
                            (directory / "stdout.txt").write_text(proc.stdout)
                            (directory / "stderr.txt").write_text(proc.stderr)
                            observations = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
                            if proc.returncode or not observations:
                                observations.append({"status": "worker_error", "returncode": proc.returncode})
                        except subprocess.TimeoutExpired:
                            observations = [{"status": "timeout"}]
                    for observation in observations:
                        row = dataclasses.asdict(recipe) | {"job": job, "trial": trial, "backend": backend,
                                                           "reproduce": command} | observation
                        summarize(row)
                        stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    completed.add(job)
                if index % 20 == 0:
                    print(f"trial={trial} recipes={index+1}/{len(order)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    report(results)


if __name__ == "__main__":
    main()
