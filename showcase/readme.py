"""Render Forge's measured tables and recorded traces in the GitHub README."""

import argparse
from collections import defaultdict
import json
from math import floor, log10
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
    updated = replace_section(current, "BENCHMARKS", benchmark_tables(read_data("benchmarks")))
    updated = replace_section(updated, "TRACES", trace_sections(read_data("captures")))
    if args.check:
        if current != updated:
            parser.exit(1, "Run python3 showcase/readme.py to update README.md\n")
    else:
        path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
