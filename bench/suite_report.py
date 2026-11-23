"""Validate the Stage 07 raw reports and render a table and speedup chart."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
from statistics import median

from suite_cases import WORKLOADS, dimensions

_NATIVE_NAME = re.compile(r"(E[1-6]|R[1-3]|M1)/m:(\d+)/k:(\d+)/n:(\d+)/avx2:(0|1)/fusion:(0|1)(?:/|$)")
_NATIVE_UNITS = {"ns": 1, "us": 1000, "ms": 1000000, "s": 1000000000}
_WARM = ("forge_allocating", "numpy_allocating", "forge_preallocated", "numpy_preallocated")
_SPEEDUPS = {
    "python_allocating_speedup": "Python allocating",
    "python_preallocated_speedup": "Python preallocated",
    "native_avx2_over_scalar": "Native AVX2 / scalar",
    "native_fusion_speedup": "Native fused / unfused",
}


def stats(values, repetitions):
    if len(values) != repetitions or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"Expected {repetitions} positive finite timing samples, got {values}")
    p95 = sorted(values)[math.ceil(len(values) * 0.95) - 1]
    return median(values), p95


def gather_native(raw):
    groups = {}
    for sample in raw["benchmarks"]:
        if sample.get("error_occurred") or sample.get("skipped"):
            raise ValueError(f"Native benchmark skipped or failed: {sample}")
        if sample["run_type"] != "iteration":
            continue
        match = _NATIVE_NAME.match(sample["run_name"])
        if not match:
            raise ValueError(f"Unrecognized native row {sample['run_name']}")
        key, m, k, n, avx2, fusion = match.groups()
        dims = (int(m), int(k), int(n)) if key == "M1" else (int(m),)
        kernel = ("avx2" if avx2 == "1" else "scalar") + ("_fused" if fusion == "1" else "_unfused")
        group = groups.setdefault((key, dims, kernel), [])
        group.append((sample["real_time"] * _NATIVE_UNITS[sample["time_unit"]], sample))
    return groups


def gather_rows(raw, native):
    timings = gather_native(native)
    caches = native["context"].get("caches", [])
    llc = max((cache["size"] for cache in caches if cache["level"] >= 3), default=0)
    target = raw["metadata"]["forge_target"]
    count = raw["repetitions"]
    expected = {(case.key, dims) for case in WORKLOADS for dims in dimensions(case, raw["quick"])}
    actual = {(entry["id"], tuple(entry["dimensions"])) for entry in raw["cases"]}
    if actual != expected or len(raw["cases"]) != len(expected):
        raise ValueError(f"Incorrect Python coverage: missing {expected - actual}, extra {actual - expected}")

    rows = []
    for case in raw["cases"]:
        key = case["id"]
        dims = tuple(case["dimensions"])
        row = {
            "id": key, "expression": case["expression"], "dimensions": " × ".join(map(str, dims)),
            "minimum_io_bytes": case["minimum_io_bytes"], "reported_llc_bytes": llc or None,
            "working_set": ("exceeds reported LLC" if case["minimum_io_bytes"] > llc
                            else "fits reported LLC") if llc else "LLC unavailable",
        }
        for name in _WARM:
            arm = case["measurements"][name]
            med, p95 = stats(arm["samples_ns_per_call"], count)
            if med != arm["median_ns_per_call"]:
                raise ValueError(f"Stored median does not match raw samples for {key} {dims} {name}")
            row.update({name + "_median_ns": med, name + "_p95_batch_mean_ns": p95})
        for policy in ("allocating", "preallocated"):
            row["python_" + policy + "_speedup"] = (
                row["numpy_" + policy + "_median_ns"] / row["forge_" + policy + "_median_ns"]
            )

        targets = ("scalar", "avx2") if target == "avx2" else ("scalar",)
        for isa in targets:
            for fusion in ("fused", "unfused") if key.startswith("E") else ("fused",):
                mode = isa + "_" + fusion
                identity = (key, dims, mode)
                if identity not in timings:
                    raise ValueError(f"No native samples for {identity}")
                samples = timings.pop(identity)
                med, p95 = stats([time for time, _ in samples], count)
                prefix = "native_" + mode
                row.update({prefix + "_median_ns": med, prefix + "_p95_sample_ns": p95})
                for counter in ("loops", "temporary_arrays", "modeled_bytes_per_element"):
                    if counter not in samples[0][1]:
                        continue
                    value = samples[0][1][counter]
                    if any(meta[counter] != value for _, meta in samples):
                        raise ValueError(f"The planner counter changed for {identity}")
                    row[prefix + "_" + counter] = value
        if target == "avx2":
            row["native_avx2_over_scalar"] = row["native_scalar_fused_median_ns"] / row["native_avx2_fused_median_ns"]
        if key.startswith("E"):
            row["native_fusion_speedup"] = (row[f"native_{target}_unfused_median_ns"] /
                                             row[f"native_{target}_fused_median_ns"])
        rows.append(row)
    if timings:
        raise ValueError(f"Unexpected native configurations: {list(timings)}")
    return rows


def chart(rows, directory, metadata):
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(14, 15), layout="constrained")
    markers = ("o", "s", "^", "D")
    for ax, case in zip(axes.flat, WORKLOADS, strict=True):
        subset = [row for row in rows if row["id"] == case.key]
        labels = [row["dimensions"].replace(" × ", "×") for row in subset]
        for (key, name), marker in zip(_SPEEDUPS.items(), markers, strict=True):
            if key in subset[0]:
                ax.plot(range(len(labels)), [row[key] for row in subset], marker=marker, label=name)
        ax.axhline(1, color="#77756c", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)), labels, rotation=24, ha="right", fontsize=9)
        ax.set_ylabel("Speedup (×)")
        ax.set_title(f"{case.key}  {case.expression}", loc="left")
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"Forge benchmark grid · {metadata['profile']} · {metadata['commit'][:7]}",
                 fontsize=18, fontweight="bold")
    fig.savefig(directory / "speedups.png", dpi=160)
    plt.close(fig)


def write_report(raw, native, directory):
    rows = gather_rows(raw, native)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata = raw["metadata"]
    lines = [f"# Forge benchmark grid · {metadata['profile']}", "",
             f"Commit `{metadata['commit']}` on {metadata['cpu']['model']} ({metadata['os']}).",
             f"NumPy `{metadata['numpy']}` and Forge target `{metadata['forge_target']}`.",
             "", "Ratios above 1 favor Forge, AVX2, or fusion, depending on the column.",
             "The CSV includes per-boundary medians in ns and p95 of Python batch means or native samples.",
             "", "| Case | Size or M × K × N | Python allocating | Python preallocated | Native AVX2 / scalar | Native fusion |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        ratios = [f"{row[key]:.2f}×" if key in row else "—" for key in _SPEEDUPS]
        lines.append("| " + " | ".join((row["id"], row["dimensions"], *ratios)) + " |")
    lines.extend(("", "## First use of E5 at 1,024 elements", "", "| Group | Median (µs) | p95 (µs) |",
                  "| --- | ---: | ---: |"))
    latency = raw["latency"]
    fresh = latency["fresh_process_ns"]
    groups = latency["initialized_process_ns"]
    all_groups = dict(groups)
    for key in fresh[0]:
        all_groups[f"fresh_{key}"] = [run[key] for run in fresh]
    for label, values in all_groups.items():
        med, p95 = stats(values, len(values))
        lines.append(f"| {label.replace('_', ' ').removesuffix(' ns')} | {med / 1000:.2f} | {p95 / 1000:.2f} |")
    for row in rows:
        if row["id"] == latency["workload"] and row["dimensions"] == str(latency["elements"]):
            warm = row["forge_allocating_median_ns"]
            saving = row["numpy_allocating_median_ns"] - warm
            if saving > 0:
                cost = median(groups["cold_parse_and_compile_ns"]) + median(groups["first_allocating_call_ns"]) - warm
                calls = max(1, math.ceil(cost / saving))
                lines.extend(("", f"Estimated allocating break-even is **{calls:,} calls** using these medians."))
            break
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    chart(rows, directory, metadata)
    print(f"Verified {len(rows)} cases with complete Python and native samples.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    raw = json.loads((args.directory / "python.json").read_text(encoding="utf-8"))
    native = json.loads((args.directory / "native.json").read_text(encoding="utf-8"))
    write_report(raw, native, args.directory)


if __name__ == "__main__":
    main()
