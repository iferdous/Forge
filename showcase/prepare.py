"""Build the public benchmark data and figures from a verified Forge evidence ZIP."""

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

HOSTS = {"linux-x64": "Ubuntu x64", "windows-x64": "Windows x64"}
METRICS = {
    "alloc": ("forge_allocating_median_ns", "numpy_allocating_median_ns", "Forge", "NumPy"),
    "prealloc": ("forge_preallocated_median_ns", "numpy_preallocated_median_ns", "Forge", "NumPy"),
    "fusion": ("native_avx2_fused_median_ns", "native_avx2_unfused_median_ns", "Fused", "Unfused"),
    "simd": ("native_avx2_fused_median_ns", "native_scalar_fused_median_ns", "AVX2", "Scalar"),
}
PAPER, INK, GREEN, ORANGE, GRAY = "#f6f4ea", "#172923", "#116754", "#ad4f2e", "#53665b"


def read_evidence(path):
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name, digest in manifest["sha256"].items():
            if sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch for {name}")
        hosts = []
        for key, label in HOSTS.items():
            csv_text = archive.read(f"{key}/summary.csv").decode("utf-8").replace("\r\n", "\n")
            raw = json.loads(archive.read(f"{key}/python.json"))
            if raw["metadata"]["commit"] != manifest["commit"]:
                raise ValueError(f"{key} commit does not match the archive manifest")
            rows = []
            for line in csv.DictReader(io.StringIO(csv_text)):
                item = {k: line[k] for k in ("id", "expression", "dimensions", "working_set")}
                item["io_bytes"] = int(line["minimum_io_bytes"])
                item["comparisons"] = {}
                for name, (forge_key, base_key, _, _) in METRICS.items():
                    if line[forge_key] and line[base_key]:
                        item["comparisons"][name] = [
                            float(line[forge_key]), float(line[base_key]),
                            float(line[forge_key.replace("_median_ns", "_p95_batch_mean_ns"
                                                        if name in ("alloc", "prealloc") else "_p95_sample_ns")]),
                            float(line[base_key.replace("_median_ns", "_p95_batch_mean_ns"
                                                       if name in ("alloc", "prealloc") else "_p95_sample_ns")]),
                        ]
                if line["native_avx2_fused_loops"]:
                    item["plan"] = [int(float(line[field])) if line[field] else None for field in (
                        "native_avx2_unfused_loops", "native_avx2_fused_loops",
                        "native_avx2_unfused_temporary_arrays", "native_avx2_fused_temporary_arrays",
                        "native_avx2_unfused_modeled_bytes_per_element",
                        "native_avx2_fused_modeled_bytes_per_element")]
                rows.append(item)
            startup = raw["latency"]["initialized_process_ns"]
            latency = {name.removesuffix("_ns"): median(series) for name, series in startup.items()}
            latency["fresh_process"] = median(row["process_end_to_end_ns"]
                                              for row in raw["latency"]["fresh_process_ns"])
            hosts.append({"id": key, "label": label, "cpu": raw["metadata"]["cpu"]["model"],
                          "profile": raw["metadata"]["profile"], "repetitions": raw["repetitions"],
                          "rows": rows, "latency_ns": latency, "csv": csv_text})
        cov = json.loads(archive.read("linux-hardening/coverage.json"))[0]["totals"]
        hardening = json.loads(archive.read("linux-hardening/hardening.json"))
    return {"commit": manifest["commit"], "ci": manifest["ci_run"],
            "hosts": hosts, "coverage": cov, "hardening": hardening}


def save(fig, folder, name):
    fig.savefig(folder / name, dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=.35)


def charts(data, folder, captures):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": "#b3beb0", "xtick.color": GRAY, "ytick.color": GRAY,
                         "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.titleweight": "bold",
                         "grid.color": "#ccd4c9", "axes.spines.top": False, "axes.spines.right": False})
    timing_source = f"shared CI · {data['commit'][:7]}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, layout="constrained")
    for ax, host in zip(axes, data["hosts"], strict=True):
        rows = [row for row in host["rows"] if row["id"] == "E5"]
        dimensions = [int(row["dimensions"]) for row in rows]
        for index, color, label in ((1, ORANGE, "Unfused"), (0, GREEN, "Fused")):
            ax.plot(dimensions, [row["comparisons"]["fusion"][index] / 1e3 for row in rows],
                    color=color, marker="o", ms=4.5, lw=2.3, label=label)
        ax.set(xscale="log", yscale="log", title=host["label"], xlabel="float32 elements")
        ax.grid(True, axis="y", alpha=.75)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Warm native AVX2 latency (µs)")
    fig.suptitle("One loop instead of three", x=.025, ha="left", fontsize=22)
    fig.supxlabel(f"E5 · identical preallocated AVX2 calls · medians · {timing_source}", color=GRAY, fontsize=10)
    save(fig, folder, "fusion-latency.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("forge", ["#963b23", PAPER, "#126454"])
    nrm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=4)
    for ax, host in zip(axes, data["hosts"], strict=True):
        keys = sorted({row["id"] for row in host["rows"]})
        columns = ["32", "1,024", "65,536", "1,048,576", "8,388,608"]
        grid = np.array([[next(row["comparisons"]["alloc"][1] / row["comparisons"]["alloc"][0]
                              for row in host["rows"] if row["id"] == key and row["dimensions"] == size)
                          for size in ("32", "1024", "65536", "1048576", "8388608")]
                         for key in keys if key != "M1"])
        scaled = np.log2(grid)
        ax.imshow(scaled, cmap=cmap, norm=nrm, aspect="auto")
        ax.set_xticks(range(len(columns)), columns, rotation=23, ha="right")
        ax.set_yticks(range(len(keys) - 1), [key for key in keys if key != "M1"])
        ax.set_title(host["label"], loc="left")
        ax.tick_params(length=0, pad=7)
        for (y, x), value in np.ndenumerate(grid):
            ax.text(x, y, f"{value:.2f}×", ha="center", va="center", fontweight="bold",
                    color="white" if abs(scaled[y, x]) > 2.5 else INK)
    fig.suptitle("Which vector workloads benefit?", x=.025, ha="left", fontsize=22)
    fig.supxlabel(f"Warm Python, allocating · NumPy median / Forge median · above 1× favors Forge · {timing_source}",
                  color=GRAY, fontsize=10)
    save(fig, folder, "workload-atlas.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, host in zip(axes, data["hosts"], strict=True):
        rows = [r for r in host["rows"] if r["id"] == "M1"]
        labels = [r["dimensions"].replace(" × ", "×") for r in rows]
        for index, color, name in ((0, GREEN, "Forge"), (1, ORANGE, "NumPy")):
            ax.plot(labels, [r["comparisons"]["alloc"][index] / 1e3 for r in rows],
                    marker="o", lw=2.3, color=color, label=name)
        ax.set(yscale="log", title=host["label"], xlabel="M × K × N")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=.75)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Warm allocating latency (µs)")
    fig.suptitle("Tiled matmul and single-threaded BLAS", x=.025, ha="left", fontsize=22)
    fig.supxlabel(f"Python API · log scale · {timing_source}", color=GRAY, fontsize=10)
    save(fig, folder, "matmul-latency.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    keys = [("cache_hit_new_handle", "Cache hit"), ("first_allocating_call", "First allocating call"),
            ("cold_parse_and_compile", "Cold parse + compile"), ("fresh_process", "Fresh process")]
    for y, (key, _) in enumerate(keys):
        for host, offset, color, marker in zip(data["hosts"], (-.13, .13), (GREEN, ORANGE), ("o", "s"), strict=True):
            value = host["latency_ns"][key] / 1e3
            ax.scatter(value, y + offset, color=color, s=62, marker=marker,
                       label=host["label"] if y == 0 else None)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:,.0f}"))
    ax.set_yticks(range(4), [label for _, label in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Median duration (µs, log scale)")
    ax.grid(True, axis="x", alpha=.8)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("What the first call costs", loc="left", fontsize=22, pad=19)
    fig.supxlabel(f"E5, 1,024 elements · fresh process includes interpreter startup and shutdown · {timing_source}",
                  color=GRAY, fontsize=10)
    save(fig, folder, "first-use.png")
    plt.close(fig)

    if captures and captures["rss_samples"]:
        fig, ax = plt.subplots(figsize=(11, 4.7), layout="constrained")
        stages = [("before_compile", "Input/output touched"), ("after_compile", "After LLJIT compile"),
                  ("after_first_call", "After first run_into")]
        for label, color in (("unfused", ORANGE), ("fused", GREEN)):
            samples = captures["rss_samples"][label]
            for sample in samples:
                ax.plot([s[1] for s in stages], [sample[s[0]] / 2**20 for s in stages],
                        color=color, alpha=.24, lw=1.2)
            ax.plot([s[1] for s in stages], [median(sample[s[0]] for sample in samples) / 2**20 for s in stages],
                    color=color, lw=2.8, marker="o", label=label.capitalize())
        ax.set_title("Where the temporary buffers show up", loc="left", fontsize=22, pad=19)
        ax.set_ylabel("Process resident set (MiB)")
        ax.grid(True, axis="y", alpha=.75)
        ax.legend(frameon=False)
        fig.supxlabel("Linux /proc/self/statm · three fresh processes per arm · 8,388,608 float32 elements",
                      color=GRAY, fontsize=10)
        save(fig, folder, "resident-memory.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--captures", type=Path)
    options = parser.parse_args()
    data = read_evidence(options.evidence)
    folder = options.out / "data"
    folder.mkdir(parents=True, exist_ok=True)
    assets = options.out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for host in data["hosts"]:
        (folder / f'{host["id"]}.csv').write_text(host.pop("csv"), encoding="utf-8", newline="\n")
    (folder / "benchmarks.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n",
                                           encoding="utf-8")
    captures = json.loads(options.captures.read_text(encoding="utf-8")) if options.captures else None
    if captures:
        (folder / "captures.json").write_text(json.dumps(captures, separators=(",", ":"), allow_nan=False) + "\n",
                                             encoding="utf-8")
    charts(data, assets, captures)


if __name__ == "__main__":
    main()
