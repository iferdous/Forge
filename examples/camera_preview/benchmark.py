"""Time the same real frame batch with separate kernel and file-to-file boundaries."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
from PIL import __version__ as pillow_version

from .__main__ import SAMPLE
from .pipeline import DECODE_SRGB, PixelEngine, Preset, find_frames, load_frame, preview_batch


def rotated(backends, index):
    offset = index % len(backends)
    return backends[offset:] + backends[:offset]


def cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            return match[1]
    if platform.system() == "Darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor() or platform.machine()


def output_hashes(destination, report):
    return [sha256((destination / record["output"]).read_bytes()).hexdigest()
            for record in report["frames"]]


def benchmark(frames, preset, out, warm, full, include_numexpr, require_avx2, objdump):
    backends = ["numpy", "unfused", "forge"]
    if include_numexpr:
        backends.append("numexpr")

    linear = [DECODE_SRGB[load_frame(frame)] for frame in frames]
    engines = {name: PixelEngine(name, preset) for name in backends}
    target = engines["forge"].target
    if require_avx2 and target != "avx2":
        raise RuntimeError("This run requires native AVX2 execution")

    calls = {name: [engine.prepare(frame) for frame in linear] for name, engine in engines.items()}
    for i in range(len(frames)):
        reference = calls["numpy"][i]().view(np.uint32).copy()
        for name in backends[1:]:
            np.testing.assert_array_equal(calls[name][i]().view(np.uint32), reference,
                                          err_msg=f"{name}: float32 mismatch for {frames[i].name}")

    samples = {name: {"warm_batch_ns": [], "file_to_file_ns": [], "file_stages_ns": []}
               for name in backends}
    for run in range(warm):
        for name in rotated(backends, run):
            start = perf_counter_ns()
            for call in calls[name]:
                call()
            samples[name]["warm_batch_ns"].append(perf_counter_ns() - start)

    out.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(dir=out) as temporary:
        scratch = Path(temporary)
        expected_hashes = None
        for run in range(full):
            for name in rotated(backends, run):
                destination = scratch / f"{name}-{run}"
                start = perf_counter_ns()
                report = preview_batch(frames, destination, preset, name)
                elapsed = perf_counter_ns() - start
                hashes = output_hashes(destination, report)
                if expected_hashes is None:
                    expected_hashes = hashes
                if hashes != expected_hashes:
                    raise ValueError(f"{name}: exported PNGs differ in run {run}")
                samples[name]["file_to_file_ns"].append(elapsed)
                samples[name]["file_stages_ns"].append({
                    "compile_ns": report["compile_ns"],
                    **{key: sum(record[key] for record in report["frames"])
                       for key in ("decode_and_linearize_ns", "pixel_ns", "encode_and_write_ns")},
                })
                # Reuse filesystem space and rotate backend order. Timed work
                # ends before the checksums and cleanup.
                for file in destination.iterdir():
                    file.unlink()
                destination.rmdir()

    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "host": {"system": platform.system(), "architecture": platform.machine(), "cpu": cpu_name()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pillow": pillow_version},
        "target": target, "preset": {"stops": preset.stops, "lift": preset.lift},
        "frame_count": len(frames),
        "frame_shapes": [list(frame.shape) for frame in linear],
        "inputs": [{"file": frame.name, "sha256": sha256(frame.read_bytes()).hexdigest()} for frame in frames],
        "outputs": expected_hashes,
        "verification": "all backends match NumPy float32 bits and produce identical PNG files",
        "backends": samples,
    }
    if include_numexpr:
        report["versions"]["numexpr"] = engines["numexpr"].numexpr.__version__
        report["numexpr_threads"] = engines["numexpr"].numexpr.get_num_threads()
    for name in ("unfused", "forge"):
        for size, plan in engines[name].plans.items():
            (out / f"{name}-{size}.plan.txt").write_text(plan, encoding="utf-8")
        if objdump:
            kernel = engines[name].compiled_kernel(linear[0].reshape(-1))
            object_file = out / f"{name}.o"
            object_file.write_bytes(kernel.object())
            with (out / f"{name}.asm").open("w", encoding="utf-8") as assembly:
                subprocess.run([objdump.resolve(), "-d", object_file.name], cwd=out,
                               stdout=assembly, check=True)
    preview_batch(frames, out / "forge-previews", preset)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(frames)} frames verified on {target}; {warm} warm and {full} file-to-file samples per backend")
    for name, timings in samples.items():
        print(f"{name:8} {median(timings['warm_batch_ns']) / 1e6:7.2f} ms pixels  "
              f"{median(timings['file_to_file_ns']) / 1e6:7.2f} ms file to file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=SAMPLE)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warm", type=int, default=21, help="Warm batch samples per backend")
    parser.add_argument("--full", type=int, default=9, help="File-to-file batch samples per backend")
    parser.add_argument("--numexpr", action="store_true", help="Compare NumExpr at one thread")
    parser.add_argument("--require-avx2", action="store_true")
    parser.add_argument("--objdump", type=Path, help="Disassemble the real JIT object for the first frame")
    args = parser.parse_args()
    if args.warm < 1 or args.full < 1:
        parser.error("--warm and --full must be positive")
    try:
        benchmark(find_frames(args.source), Preset(), args.out, args.warm, args.full,
                  args.numexpr, args.require_avx2, args.objdump)
    except (AssertionError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Camera benchmark: {error}\n")


if __name__ == "__main__":
    main()
