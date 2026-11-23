"""Measure prepared reduction and matrix calls and check their numerical error."""

import argparse
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import forge
from python_baseline import measure

SEED = 620261122
LENGTHS = (33, 1024, 65536, 1048576)
MATRICES = ((16, 16, 16), (31, 63, 17), (64, 64, 64), (256, 256, 256), (512, 512, 512))
REDUCTIONS = {"sum": np.sum, "max": np.max, "mean": np.mean}


def gamma(operations):
    rounded = operations * 2.0**-24
    return rounded / (1 - rounded)


def error_report(actual, expected, bound):
    error = np.abs(actual.astype(np.float64) - expected)
    if not np.all(error <= bound):
        index = np.unravel_index(np.argmax(error - bound), error.shape)
        raise AssertionError(f"Error {error[index]} at {index} exceeds the bound {np.asarray(bound)[index]}")
    fractions = np.divide(error, bound, out=np.zeros_like(error), where=bound != 0)
    return {"max_abs_error": float(np.max(error)), "max_abs_bound": float(np.max(bound)),
            "max_fraction_of_bound": float(np.max(fractions))}


def reduction_reference(name, x):
    if name == "max":
        return np.float64(np.max(x)), np.float64(0)
    expected = math.fsum(x)
    scale = math.fsum(np.abs(x))
    if name == "mean":
        return np.float64(expected / x.size), np.float64(gamma(x.size + 1) * scale / x.size)
    return np.float64(expected), np.float64(gamma(x.size) * scale)


def matrix_reference(a, b):
    left, right = a.astype(np.float64), b.astype(np.float64)
    return left @ right, gamma(a.shape[1] + 1) * (np.abs(left) @ np.abs(right))


def require_same_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def capture_codegen(out, label, target, kernel, objdump):
    if label != "matmul" and re.search(r"<\d+ x float>", kernel.info.ir):
        raise AssertionError(f"The ordered {label} kernel vectorized")
    prefix = out / f"{label}-{target}"
    prefix.with_suffix(".ll").write_text(kernel.info.ir, encoding="utf-8")
    prefix.with_suffix(".o").write_bytes(kernel.object)
    prefix.with_suffix(".plan").write_text(kernel.plan.dump(), encoding="utf-8")
    if objdump is None:
        return
    asm = subprocess.run(
        [str(objdump), "--disassemble", "--x86-asm-syntax=att", str(prefix.with_suffix(".o"))],
        check=True, capture_output=True, text=True,
    ).stdout
    prefix.with_suffix(".s").write_text(asm, encoding="utf-8")
    if re.search(r"\bvfn?m(?:add|sub)\w*", asm):
        raise AssertionError(f"Floating-point contraction in {label}")
    if target == "avx2" and label == "matmul":
        for opcode in ("vmulps", "vaddps"):
            if not re.search(rf"\b{opcode}\b[^\n]*\bymm[0-9]+\b", asm):
                raise AssertionError(f"Missing packed {opcode} in tiled matmul")


def benchmark_case(program, inputs, cases, expected, bound, options, order, label, capture=False):
    contract = program.reference(*inputs)
    errors = {}
    for target in ("scalar", "avx2") if forge.has_avx2() else ("scalar",):
        kernel = program.jit(*inputs, target=target)
        buffer = np.empty_like(contract)
        key = f"forge_{target}"
        cases[key] = lambda kernel=kernel, buffer=buffer: kernel.run_into(buffer, *inputs)
        cases[key]()
        require_same_bits(buffer, contract)
        errors[key] = error_report(buffer, expected, bound)
        if capture:
            capture_codegen(options.out.parent, label, target, kernel, options.objdump)
    errors["numpy"] = error_report(cases["numpy"](), expected, bound)
    return {"accuracy": errors, "measurements": measure(cases, options.repetitions, order)}


def reductions(options, order, rng):
    for size in LENGTHS:
        x = rng.standard_normal(size, dtype=np.float32)
        for name, numpy_reduce in REDUCTIONS.items():
            program = forge.compile(f"{name}(x)", inputs=("x",))
            expected, bound = reduction_reference(name, x)
            out = np.empty((), dtype=np.float32)
            cases = {"numpy": lambda: numpy_reduce(x, out=out)}
            result = benchmark_case(program, (x,), cases, expected, bound, options, order,
                                    name, capture=size == LENGTHS[0])
            yield {"operation": name, "elements": size, **result}


def matrices(options, order, rng):
    program = forge.compile("a @ b", inputs=("a", "b"))
    for m, k, n in MATRICES:
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        expected, bound = matrix_reference(a, b)
        out = np.empty((m, n), dtype=np.float32)
        cases = {"numpy": lambda: np.matmul(a, b, out=out)}
        result = benchmark_case(program, (a, b), cases, expected, bound, options, order,
                                "matmul", capture=m == 64)
        yield {"operation": "matmul", "dimensions_mkn": [m, k, n], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, help="Save native JIT disassembly and check packed matmul.")
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--require-single-thread", action="store_true")
    options = parser.parse_args()
    if options.repetitions < 1:
        parser.error("--repetitions must be positive")
    options.out.parent.mkdir(parents=True, exist_ok=True)

    with threadpool_limits(limits=1, user_api="blas"):
        configured = threadpool_info()
        blas = [library for library in configured if library["user_api"] == "blas"]
        verified = bool(blas) and all(library["num_threads"] == 1 for library in blas)
        if options.require_single_thread and not verified:
            raise RuntimeError(f"Could not verify single-thread NumPy BLAS: {configured}")
        rng = np.random.default_rng(SEED)
        order = random.Random(SEED)
        results = [*reductions(options, order, rng), *matrices(options, order, rng)]

    evidence = {
        "seed": SEED, "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "numpy_blas_build": np.__config__.CONFIG["Build Dependencies"]["blas"],
        "threadpools": configured, "blas_single_thread_verified": verified,
        "host_has_avx2": forge.has_avx2(),
        "timing": "warm Python wall clock, preallocated output, compilation excluded",
        "accuracy": "float32 fixed-order reference, float64 accuracy, gamma bounds",
        "results": results,
    }
    options.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(results)} cases against the fixed-order reference and numerical bounds.")
    print(f"NumPy BLAS single-thread setting verified: {verified}")


if __name__ == "__main__":
    main()
