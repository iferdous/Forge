"""Render Forge's measured tables and recorded traces in the GitHub README."""

import argparse
from collections import defaultdict
import json
from math import ceil, floor, log10
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "showcase" / "data"
BOUNDARIES = {
    "alloc": ("Python call with output allocation", "Forge", "NumPy"),
    "prealloc": ("Python call with preallocated output", "Forge", "NumPy"),
    "fusion": ("Native AVX2, effect of fusion", "Fused", "Unfused"),
    "simd": ("Native fused plan, effect of SIMD", "AVX2", "Scalar"),
}
WORKLOADS = {
    "E1": "Add two arrays",
    "E2": "Multiply by a scalar",
    "E3": "Square root",
    "E4": "Multiply and add",
    "E5": "Three-op chain",
    "E6": "Four-op chain",
    "R1": "Ordered sum",
    "R2": "NaN-aware max",
    "R3": "Ordered mean",
    "M1": "Tiled matrix multiplication",
}


def detail(title, content):
    return f"<details>\n<summary>{title}</summary>\n\n{content.rstrip()}\n\n</details>"


def table(headers, rows):
    lines = ["| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows]]
    return "\n".join(lines)


def size_label(size):
    return f"{int(size):,}" if size.isdecimal() else size


def micros(ns):
    value = ns / 1000
    places = max(0, 3 - floor(log10(value)))
    return f"{value:,.{places}f}"


def latency(median_ns, p95_ns):
    return f"{micros(median_ns)} / {micros(p95_ns)}"


def measurement(host, workload, size, boundary):
    row = next(row for row in host["rows"] if row["id"] == workload and row["dimensions"] == size)
    test, baseline, _, _ = row["comparisons"][boundary]
    return test, baseline


def python_winner(forge, numpy):
    if forge < numpy:
        return f"**Forge {numpy / forge:.2f}× faster**"
    if numpy < forge:
        return f"**NumPy {forge / numpy:.2f}× faster**"
    return "**1.00×**"


def headline_table(data):
    headers = ["Measured workload", *(host["label"] for host in data["hosts"])]
    rows = [
        ["1,024-element add → multiply → square root, allocating Python call against NumPy"],
        ["65,536-element same chain, native AVX2 unfused → fused"],
        ["65,536-element same chain, native fused scalar → AVX2"],
        ["512 × 512 matrix multiply, allocating Python call"],
    ]
    for host in data["hosts"]:
        forge, numpy = measurement(host, "E5", "1024", "alloc")
        rows[0].append(python_winner(forge, numpy))
        fused, unfused = measurement(host, "E5", "65536", "fusion")
        rows[1].append(f"**{unfused / fused:.2f}×** · {unfused / 1000:.2f} → {fused / 1000:.2f} µs")
        avx2, scalar = measurement(host, "E5", "65536", "simd")
        rows[2].append(f"**{scalar / avx2:.2f}×**")
        forge, numpy = measurement(host, "M1", "512 × 512 × 512", "alloc")
        rows[3].append(python_winner(forge, numpy))
    source = (f"These warm medians come from shared x64 CI at {data['commit'][:7]}. "
              f"The native rows compare matched preallocated calls. [Source run]({data['ci']})")
    return table(headers, rows) + "\n\n" + source


def startup_table(data):
    headers = ["First-use timing for the 1,024-element chain", *(host["label"] for host in data["hosts"])]
    metrics = (
        ("Cold parse and compile in an initialized process", "cold_parse_and_compile", 1e6, "ms"),
        ("First allocating call", "first_allocating_call", 1e3, "µs"),
        ("Cache hit on the parsed graph, including the new handle", "cache_hit_new_handle", 1e3, "µs"),
        ("Fresh process, including interpreter launch and exit", "fresh_process", 1e6, "ms"),
    )
    rows = [[label, *(f"{host['latency_ns'][key] / divisor:.2f} {unit}" for host in data["hosts"])]
            for label, key, divisor, unit in metrics]
    breakeven = ["Estimated warm calls to earn back compilation against NumPy"]
    for host in data["hosts"]:
        forge, numpy = measurement(host, "E5", "1024", "alloc")
        latency_ns = host["latency_ns"]
        cost = latency_ns["cold_parse_and_compile"] + latency_ns["first_allocating_call"] - forge
        breakeven.append(f"{max(1, ceil(cost / (numpy - forge))):,}" if numpy > forge else "None")
    return table(headers, [*rows, breakeven])


def benchmark_hardware(data):
    runners = " and ".join(f"{host['label']} ({host['cpu']})" for host in data["hosts"])
    first = data["hosts"][0]
    count = len(first["rows"])
    kernels = len({row["id"] for row in first["rows"]})
    return (f"The grid ran on {runners} with one OpenBLAS thread per runner. "
            f"It covers **{kernels} kernels and {count} size cases per runner** with "
            f"**{first['repetitions']} interleaved warm samples per configuration**. "
            "Python p95 summarizes batch means. Native p95 summarizes iteration batches. "
            "The downloadable grids include full precision and the 512 MiB streaming working set.")


def benchmark_tables(data):
    hosts = {host["id"]: host for host in data["hosts"]}
    groups = defaultdict(list)
    for row in hosts["linux-x64"]["rows"]:
        groups[row["id"]].append(row)
    windows = {(row["id"], row["dimensions"]): row for row in hosts["windows-x64"]["rows"]}
    sections = []

    for workload, rows in groups.items():
        case_sections = []
        for key, (name, primary, baseline) in BOUNDARIES.items():
            headers = [
                "Size", f"Ubuntu {primary}", f"Ubuntu {baseline}", "Ubuntu ×",
                f"Windows {primary}", f"Windows {baseline}", "Windows ×",
            ]
            measured = []
            for row in rows:
                if key not in row["comparisons"]:
                    continue
                windows_row = windows[workload, row["dimensions"]]
                values = [size_label(row["dimensions"])]
                for case in (row, windows_row):
                    test, base, test_p95, base_p95 = case["comparisons"][key]
                    values.extend([latency(test, test_p95), latency(base, base_p95), f"{base / test:.2f}×"])
                measured.append(values)
            if measured:
                case_sections.append(detail(name, table(headers, measured)))

        expression = rows[0]["expression"]
        title = f"{workload} · {WORKLOADS[workload]}"
        info = f"Expression {chr(96)}{expression}{chr(96)}. Each time cell is median / p95 in µs."
        if "fusion" not in rows[0]["comparisons"]:
            info += " This standalone kernel has a single loop plan."
        sections.append(detail(title, info + "\n\n" + "\n\n".join(case_sections)))

    return "\n\n".join(sections)


def block(kind, source):
    return f"~~~{kind}\n{source.strip()}\n~~~"


def shape_label(shape):
    return " × ".join(map(str, shape)) or "scalar"


def trace_sections(data):
    sections = []
    for case in data["cases"]:
        inputs = "\n".join(f"{name} ({shape_label(arg['shape'])}) = {json.dumps(arg['sample'])}"
                           for name, arg in case["inputs"].items())
        example = block("sh", case["command"])
        if case["sampled"]:
            inputs = "Input values are a sample from the full capture.\n\n" + block("text", inputs)
        else:
            inputs = block("text", inputs)
        categories = [detail("Input and rerun command", example + "\n\n" + inputs),
                      detail("Typed graph", block("text", case["graph"]))]
        plans = case["plans"]
        overview = table(["Plan", "Loops", "Temporary arrays"],
                         [[name, str(plans[name]["loops"]), str(plans[name]["temporaries"])]
                          for name in ("unfused", "fused")])
        for heading, field in (("Loop plan", "plan"), ("Optimized LLVM IR", "ir"), ("LLJIT disassembly", "assembly")):
            both = "\n\n".join(f"**{name.capitalize()}**\n\n" + block("text", plans[name][field])
                               for name in ("unfused", "fused"))
            categories.append(detail(heading, both))
        output = f"Verified {shape_label(case['output_shape'])} output"
        if case["sampled"]:
            output += " (sample)"
        categories.append(detail("Result", output + "\n\n" + block("json", json.dumps(case["output"]))))
        sections.append(detail(case["title"], overview + "\n\n" + "\n\n".join(categories)))
    return "\n\n".join(sections)


def read_data(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def replace_section(readme, name, body):
    start = f"<!-- BEGIN GENERATED {name} -->"
    end = f"<!-- END GENERATED {name} -->"
    before, _, tail = readme.partition(start)
    _, found, after = tail.partition(end)
    if not found:
        raise ValueError(f"README section {name} is missing its generation markers")
    return f"{before}{start}\n\n{body}\n\n{end}{after}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the generated README sections are stale")
    args = parser.parse_args()
    path = ROOT / "README.md"
    current = path.read_text(encoding="utf-8")
    benchmarks = read_data("benchmarks")
    updated = replace_section(current, "HEADLINES", headline_table(benchmarks))
    updated = replace_section(updated, "STARTUP", startup_table(benchmarks))
    updated = replace_section(updated, "BENCHMARK HARDWARE", benchmark_hardware(benchmarks))
    updated = replace_section(updated, "BENCHMARKS", benchmark_tables(benchmarks))
    updated = replace_section(updated, "TRACES", trace_sections(read_data("captures")))
    if args.check:
        if current != updated:
            parser.exit(1, "Run python3 showcase/readme.py to update README.md\n")
    else:
        path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
