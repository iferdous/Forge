"""Time the warm Python call boundary with matched output allocation policies."""

import argparse
import json
import platform
import random
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

import forge

SIZES = (9, 1024, 65536, 1048576)
TWO = np.float32(2)
THREE = np.float32(3)
WORKLOADS = ("x + 2", "sqrt((x + 2) * 3)")


def elapsed(call, iterations):
    start = perf_counter_ns()
    for _ in range(iterations):
        call()
    return perf_counter_ns() - start


def measure(calls, repetitions, random_order):
    iterations = {}
    samples = {name: [] for name in calls}
    for name, call in calls.items():
        for _ in range(3):
            call()
        count = 1
        while elapsed(call, count) < 1_000_000:
            count *= 2
        iterations[name] = count
    order = list(calls)
    for _ in range(repetitions):
        random_order.shuffle(order)
        for name in order:
            count = iterations[name]
            samples[name].append(elapsed(calls[name], count) / count)
    return {
        name: {"iterations_per_sample": iterations[name],
               "samples_ns_per_call": values,
               "median_ns_per_call": median(values)}
        for name, values in samples.items()
    }


def numpy_cases(expression, x):
    output = np.empty_like(x)
    if expression == WORKLOADS[0]:
        return {
            "numpy/allocating": lambda: np.add(x, TWO),
            "numpy/preallocated": lambda: np.add(x, TWO, out=output),
        }

    add_result, multiply_result = np.empty_like(x), np.empty_like(x)

    def preallocated():
        np.add(x, TWO, out=add_result)
        np.multiply(add_result, THREE, out=multiply_result)
        np.sqrt(multiply_result, out=output)

    return {
        "numpy/allocating": lambda: np.sqrt((x + TWO) * THREE),
        "numpy/preallocated": preallocated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=9)
    options = parser.parse_args()
    if options.repetitions < 1:
        parser.error("--repetitions must be positive")

    modes = ("scalar", "avx2") if forge.has_avx2() else ("scalar",)
    results = []
    random_order = random.Random(20261122)
    targets = {}
    for expression in WORKLOADS:
        program = forge.compile(expression, inputs=("x",))
        for size in SIZES:
            x = np.arange(size, dtype=np.float32) % np.float32(81)
            cases = numpy_cases(expression, x)
            expected = cases["numpy/allocating"]()
            for mode in modes:
                for fusion in (False, True):
                    label = "fused" if fusion else "unfused"
                    kernel = program.jit(x, target=mode, fusion=fusion)
                    forge_out = np.empty_like(x)
                    targets[mode] = {
                        key: getattr(kernel.info, key) for key in ("triple", "cpu", "features")
                    }
                    np.testing.assert_array_equal(kernel(x).view(np.uint32), expected.view(np.uint32))
                    kernel.run_into(forge_out, x)
                    np.testing.assert_array_equal(forge_out.view(np.uint32), expected.view(np.uint32))
                    prefix = f"forge_{mode}_{label}"
                    cases[f"{prefix}/allocating"] = lambda kernel=kernel: kernel(x)
                    cases[f"{prefix}/preallocated"] = (
                        lambda kernel=kernel, output=forge_out: kernel.run_into(output, x)
                    )
            results.append({
                "expression": expression,
                "elements": size,
                "measurements": measure(cases, options.repetitions, random_order),
            })
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps({
        "dtype": "float32",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host_has_avx2": forge.has_avx2(),
        "targets": targets,
        "timing": "wall clock, warm Python calls; each sample is a batch mean; compilation excluded",
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
