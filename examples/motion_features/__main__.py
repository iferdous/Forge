"""Turn three recorded acceleration axes into one feature per sample window."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import forge
import numpy as np


DATA = Path(__file__).with_name("data")
RECORDINGS = ("desk_closeups", "office_walkthrough", "pioneer_turn")
WINDOW = 128
AXES = ("ax", "ay", "az")
MAGNITUDE = "sqrt((ax * ax + ay * ay) + az * az)"
FEATURE = f"mean({MAGNITUDE})"


def analyze(paths, target):
    example = tuple(np.zeros(WINDOW, dtype=np.float32) for _ in AXES)
    feature_program = forge.compile(FEATURE, inputs=AXES)
    magnitude = forge.compile(MAGNITUDE, inputs=AXES).jit(*example, target=target)
    fused = feature_program.jit(*example, target=target)
    unfused = feature_program.jit(*example, target=target, fusion=False)
    recordings = []
    for path in paths:
        samples = np.loadtxt(path, ndmin=2, dtype=np.float64)
        if samples.shape[1] != 4 or not np.all(np.isfinite(samples)):
            raise ValueError(f"{path}: expected finite timestamp, ax, ay, az columns")
        if len(samples) < WINDOW or np.any(np.diff(samples[:, 0]) <= 0):
            raise ValueError(f"{path}: expected at least {WINDOW} samples with increasing timestamps")
        times = samples[:, 0] - samples[0, 0]
        window_count, remainder = divmod(len(samples), WINDOW)
        windows, peak = [], None
        for index in range(window_count):
            start = index * WINDOW
            end = start + WINDOW
            axes = samples[start:end, 1:]
            with np.errstate(over="ignore", invalid="ignore"):
                centered = np.ascontiguousarray((axes - axes.mean(axis=0)).T, dtype=np.float32)
                ax, ay, az = centered
                reference = np.sqrt((ax * ax + ay * ay) + az * az)
            if not np.isfinite(reference).all():
                raise ValueError(f"{path}: magnitude arithmetic exceeds the float32 range")
            trace = magnitude(*centered)
            np.testing.assert_array_equal(trace.view(np.uint32), reference.view(np.uint32))
            # The compiler's mean has an ordered float32 sum with an initial zero.
            total = np.add.accumulate(np.insert(reference, 0, np.float32(0)))[-1]
            expected = (total / np.float32(WINDOW)).view(np.uint32)
            value = fused(*centered)
            np.testing.assert_array_equal(value.view(np.uint32), expected)
            np.testing.assert_array_equal(unfused(*centered).view(np.uint32), expected)
            entry = {"start_s": float(times[start]), "end_s": float(times[end - 1]),
                     "mean_magnitude": float(value)}
            windows.append(entry)
            if peak is None or entry["mean_magnitude"] > peak["mean_magnitude"]:
                peak = {**entry, "elapsed_s": (times[start:end] - times[start]).tolist(),
                        "centered_axes": centered.tolist(), "magnitude": trace.tolist()}
        recordings.append({"id": path.stem, "source_sha256": sha256(path.read_bytes()).hexdigest(),
                           "unused_samples": remainder, "windows": windows, "peak_window": peak})
    plans = {}
    for name, kernel in (("unfused", unfused), ("fused", fused)):
        plan = kernel.plan
        plans[name] = {"loops": plan.loop_count, "temporary_arrays": plan.temporary_arrays,
                       "plan": plan.dump()}
    return {"target": target, "expression": FEATURE, "window_samples": WINDOW,
            "plans": plans, "recordings": recordings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, help="Timestamp, ax, ay, az text logs")
    parser.add_argument("--target", choices=("auto", "scalar", "avx2"), default="auto")
    parser.add_argument("--explain", action="store_true", help="Print both compiled loop plans")
    parser.add_argument("--json", type=Path, help="Save the measured traces and plan counts to a new file")
    args = parser.parse_args()
    target = args.target
    if target == "auto":
        target = "avx2" if forge.has_avx2() else "scalar"
    paths = args.sources or [DATA / f"{name}.txt" for name in RECORDINGS]
    try:
        report = analyze(paths, target)
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Motion features: {error}\n")
    print(f"{FEATURE}  [{target}, {WINDOW} samples per window]")
    for name, plan in report["plans"].items():
        print(f"{name}: loops={plan['loops']} temporary_arrays={plan['temporary_arrays']}")
        if args.explain:
            print(plan["plan"].rstrip())
    for recording in report["recordings"]:
        peak = recording["peak_window"]
        print(f"{recording['id']}: {len(recording['windows'])} windows, "
              f"peak mean {peak['mean_magnitude']:.4f} at {peak['start_s']:.2f}s, "
              f"{recording['unused_samples']} trailing samples skipped")
    print("All magnitude traces and both mean plans match the ordered float32 NumPy reference bit for bit.")


if __name__ == "__main__":
    main()
