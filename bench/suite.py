"""Run Forge's ten-kernel grid with matched Python calls and native attribution."""

import argparse
from contextlib import redirect_stdout
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import forge
from aggregates import error_report, matrix_reference, reduction_reference
from python_baseline import measure
from suite_cases import WORKLOADS, dimensions

ROOT = Path(__file__).resolve().parents[1]
SEED = 620261122
ONE, TWO, THREE, FOUR = map(np.float32, (1, 2, 3, 4))
REDUCTIONS = {"R1": ("sum", np.sum), "R2": ("max", np.max), "R3": ("mean", np.mean)}


def fixture(shape, stream, nonnegative=False):
    seed = (SEED + stream * 0x9E3779B9) & 0xFFFFFFFF
    bits = np.arange(math.prod(shape), dtype=np.uint32)
    bits += np.uint32(seed)
    for shift, multiplier in ((16, 0x7FEB352D), (15, 0x846CA68B)):
        bits ^= bits >> shift
        bits *= np.uint32(multiplier)
    bits ^= bits >> 16
    bits >>= 16
    output = bits.astype(np.float32)
    output *= np.float32(2.0**-16)
    if not nonnegative:
        output -= np.float32(0.5)
    return output.reshape(shape)


def numpy_calls(key, args, out):
    x = args[0]
    if key == "E1":
        y = args[1]
        return lambda: x + y, lambda: np.add(x, y, out=out)
    if key == "E2":
        return lambda: x * THREE, lambda: np.multiply(x, THREE, out=out)
    if key == "E3":
        return lambda: np.sqrt(x), lambda: np.sqrt(x, out=out)
    if key in REDUCTIONS:
        _, reduce = REDUCTIONS[key]
        return lambda: reduce(x), lambda: reduce(x, out=out)
    if key == "M1":
        b = args[1]
        return lambda: x @ b, lambda: np.matmul(x, b, out=out)

    scratch = np.empty_like(x)
    if key == "E4":
        _, y, z = args

        def prepared():
            np.multiply(x, y, out=scratch)
            return np.add(scratch, z, out=out)

        return lambda: x * y + z, prepared
    if key == "E5":
        def prepared():
            np.add(x, TWO, out=scratch)
            np.multiply(scratch, THREE, out=scratch)
            return np.sqrt(scratch, out=out)

        return lambda: np.sqrt((x + TWO) * THREE), prepared

    def prepared():
        np.add(x, ONE, out=scratch)
        np.multiply(scratch, TWO, out=scratch)
        np.add(scratch, THREE, out=scratch)
        return np.multiply(scratch, FOUR, out=out)

    return lambda: ((x + ONE) * TWO + THREE) * FOUR, prepared


def check_bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected.view(np.uint32))


def python_case(workload, dims, target, repetitions, order):
    if workload.key == "M1":
        m, k, n = dims
        shapes = ((m, k), (k, n))
    else:
        shapes = (dims,) * len(workload.inputs)
    args = tuple(fixture(shape, i, workload.key == "E3") for i, shape in enumerate(shapes))
    program = forge.compile(workload.expression, inputs=workload.inputs)
    contract = program.reference(*args)
    kernel = program.jit(*args, target=target)
    forge_out, numpy_out = np.empty_like(contract), np.empty_like(contract)
    allocating, preallocated = numpy_calls(workload.key, args, numpy_out)
    check_bits(kernel(*args), contract)
    kernel.run_into(forge_out, *args)
    check_bits(forge_out, contract)
    numpy_result = allocating()
    if workload.key in REDUCTIONS:
        name, _ = REDUCTIONS[workload.key]
        high, bound = reduction_reference(name, args[0])
    elif workload.key == "M1":
        high, bound = matrix_reference(*args)
    else:
        high, bound = None, None

    if high is None:
        check_bits(numpy_result, contract)
        check_bits(preallocated(), contract)
        accuracy = {"forge_bitwise_reference": True, "numpy_bitwise_reference": True}
    else:
        accuracy = {"forge_bitwise_reference": True, "forge": error_report(forge_out, high, bound),
                    "numpy_allocating": error_report(numpy_result, high, bound),
                    "numpy_preallocated": error_report(preallocated(), high, bound)}
    del contract, numpy_result, high, bound
    return {
        "id": workload.key, "expression": workload.expression, "dimensions": dims,
        "minimum_io_bytes": sum(array.nbytes for array in args) + forge_out.nbytes,
        "accuracy": accuracy,
        "measurements": measure({
            "forge_allocating": lambda: kernel(*args),
            "forge_preallocated": lambda: kernel.run_into(forge_out, *args),
            "numpy_allocating": allocating,
            "numpy_preallocated": preallocated,
        }, repetitions, order),
    }


_STARTUP_PROBE = """\
from time import perf_counter_ns
start = perf_counter_ns()
import json, sys
import numpy as np
import forge
import_done = perf_counter_ns()
x = np.ones(1024, dtype=np.float32)
compile_start = perf_counter_ns()
kernel = forge.compile("sqrt((x + 2) * 3)", inputs=("x",)).jit(x, target=sys.argv[1])
compiled = perf_counter_ns()
result = kernel(x)
executed = perf_counter_ns()
assert result.shape == x.shape and np.all(result == np.float32(3))
print(json.dumps({"imports_ns": import_done - start, "cold_compile_ns": compiled - compile_start,
                  "first_call_ns": executed - compiled}))
"""


def latency_groups(target, repetitions, startup_repetitions):
    x = np.ones(1024, dtype=np.float32)
    source = WORKLOADS[4].expression
    program = forge.compile(source, inputs=("x",))
    program.jit(x, target=target)(x)  # Initialize LLVM in this process.
    pairs = {"cold_parse_and_compile_ns": [], "first_allocating_call_ns": [], "cache_hit_new_handle_ns": []}
    for _ in range(repetitions):
        start = perf_counter_ns()
        kernel = forge.compile(source, inputs=("x",)).jit(x, target=target)
        compiled = perf_counter_ns()
        result = kernel(x)
        executed = perf_counter_ns()
        assert np.all(result == np.float32(3))
        pairs["cold_parse_and_compile_ns"].append(compiled - start)
        pairs["first_allocating_call_ns"].append(executed - compiled)

    forge.clear_cache()
    program.jit(x, target=target, cache=True)
    for _ in range(repetitions):
        start = perf_counter_ns()
        handle = program.jit(x, target=target, cache=True)
        elapsed = perf_counter_ns() - start
        pairs["cache_hit_new_handle_ns"].append(elapsed)
        del handle
    assert forge.cache_info().hits == repetitions
    forge.clear_cache()

    fresh = []
    for _ in range(startup_repetitions):
        start = perf_counter_ns()
        child = subprocess.run([sys.executable, "-c", _STARTUP_PROBE, target],
                               check=True, capture_output=True, text=True)
        elapsed = perf_counter_ns() - start
        fresh.append({"process_end_to_end_ns": elapsed, **json.loads(child.stdout)})
    return {"workload": "E5", "elements": x.size, "initialized_process_ns": pairs,
            "fresh_process_ns": fresh}


def command_output(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def environment(profile, configured_blas, target, toolchain):
    runtime = io.StringIO()
    with redirect_stdout(runtime):
        np.show_runtime()
    cpu = {"model": platform.processor(), "logical_cpus": os.cpu_count(),
           "machine": platform.machine(), "numpy_runtime": runtime.getvalue()}
    if sys.platform.startswith("linux"):
        first = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
        values = dict(line.split(":", 1) for line in first.splitlines() if ":" in line)
        values = {key.strip(): value.strip() for key, value in values.items()}
        cpu.update(model=values.get("model name"), flags=values.get("flags", "").split(),
                   affinity=sorted(os.sched_getaffinity(0)))
    elif sys.platform == "win32":
        cpu["identifier"] = os.environ.get("PROCESSOR_IDENTIFIER")
        cpu["model"] = command_output("pwsh", "-NoProfile", "-Command",
                                      "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()")
    if hasattr(os, "getloadavg"):
        cpu["load_average_at_start"] = os.getloadavg()
    if not cpu["model"] and sys.platform == "darwin":
        cpu["model"] = command_output("sysctl", "-n", "machdep.cpu.brand_string")

    return {
        "profile": profile, "commit": command_output("git", "rev-parse", "HEAD"),
        "tracked_worktree_modified": bool(command_output("git", "status", "--porcelain", "--untracked-files=no")),
        "os": platform.platform(), "cpu": cpu, "toolchain": toolchain,
        "python": platform.python_version(), "python_executable": sys.executable,
        "numpy": np.__version__, "threadpoolctl": version("threadpoolctl"),
        "threadpools": configured_blas, "forge_target": target,
        "warm_boundary": "Python wall clock, batch means, warmed and randomized, compilation excluded",
        "numpy_reduction_allocating_result": "NumPy scalar; Forge returns an owned rank-zero array",
    }


def native_run(executable, output, target, repetitions, quick):
    regex = r"(E[1-6]|R[1-3])/m:32/|M1/m:16/" if quick else r"(E[1-6]|R[1-3]|M1)/"
    if target == "scalar":
        regex = f"({regex}).*avx2:0/"
    command = [str(executable.resolve()), f"--benchmark_filter={regex}",
               "--benchmark_min_time=0.001s", "--benchmark_min_warmup_time=0.001",
               f"--benchmark_repetitions={repetitions}", "--benchmark_enable_random_interleaving=true",
               f"--benchmark_out={output.resolve()}", "--benchmark_out_format=json"]
    log = output.with_suffix(".log")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Native benchmark failed. See {log}")
    return json.loads(output.read_text(encoding="utf-8"))["context"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True, help="Built forge_native_suite executable.")
    parser.add_argument("--objdump", type=Path, help="llvm-objdump for exact LLJIT object evidence.")
    parser.add_argument("--configuration", default="Release")
    parser.add_argument("--profile", choices=("ci-smoke", "local-check", "controlled-x64"), default="local-check")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--startup-repetitions", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="One size per case for harness development.")
    options = parser.parse_args()
    if options.repetitions < (1 if options.quick else 20) or options.startup_repetitions < 1:
        parser.error("Use at least 20 repetitions for the full grid and at least one startup repetition.")
    target = "avx2" if forge.has_avx2() else "scalar"
    if target != "avx2" and options.profile != "local-check":
        parser.error("The selected profile requires a native AVX2 host.")
    if options.profile == "controlled-x64" and options.quick:
        parser.error("The controlled x64 profile requires the full grid.")
    options.out.mkdir(parents=True, exist_ok=True)
    toolchain_path = options.native.with_name(f"toolchain-{options.configuration}.json")
    toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))

    with threadpool_limits(limits=1, user_api="blas"):
        configured = threadpool_info()
        blas = [item for item in configured if item["user_api"] == "blas"]
        verified = bool(blas) and all(item["num_threads"] == 1 for item in blas)
        if options.profile != "local-check" and not verified:
            raise RuntimeError(f"Could not verify single-thread NumPy BLAS: {configured}")
        env = environment(options.profile, configured, target, toolchain)
        env["blas_single_thread_verified"] = verified
        order = random.Random(SEED)
        rows = []
        for workload in WORKLOADS:
            for dims in dimensions(workload, options.quick):
                print(f"Timing {workload.key} {dims}", flush=True)
                rows.append(python_case(workload, dims, target, options.repetitions, order))
        latency = latency_groups(target, options.repetitions, options.startup_repetitions)
    report = {"metadata": env, "seed": SEED, "repetitions": options.repetitions,
              "quick": options.quick, "latency": latency, "cases": rows}
    (options.out / "python.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    native = native_run(options.native, options.out / "native.json", target, options.repetitions, options.quick)
    print(f"Native benchmark completed with {len(native.get('caches', []))} reported cache entries.", flush=True)

    if options.objdump:
        args = [sys.executable, str(ROOT / "bench/inspect_simd.py"), "--objdump", str(options.objdump),
                "--out", str(options.out / "demo")]
        if target == "avx2":
            args.append("--require-avx2")
        subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
