"""Package reproducible release evidence while omitting runner names and absolute paths."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import zipfile

BENCHMARK_FILES = ("summary.csv", "summary.md", "speedups.png")
GUARD_TESTS = (
    "Avx2Jit.GuardsBothInputAndOutputAtTheScalarRemainder",
    "GeneratedCodeBounds.GuardsOrderedReductionsAtBothEnds",
    "GeneratedCodeBounds.GuardsOddMatmulTilesAndEmptyInnerDimension",
)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def timing_evidence(root, commit):
    python = read_json(root / "python.json")
    native = read_json(root / "native.json")
    meta = python["metadata"]
    context = native["context"]
    if (meta["commit"] != commit or meta["tracked_worktree_modified"] or
            meta["forge_target"] != "avx2" or not meta["blas_single_thread_verified"] or
            context["host_has_avx2"] != "yes" or python["quick"] or python["repetitions"] < 20):
        raise ValueError(f"Benchmark evidence is incomplete or from a different build: {root}")
    with (root / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 61 or len(python["cases"]) != 61:
        raise ValueError(f"Expected the 61-case release grid: {root}")
    raw_names = [row for row in native["benchmarks"] if row["run_type"] == "iteration"]
    if len(raw_names) != 198 * python["repetitions"]:
        raise ValueError(f"Native timing repetitions are incomplete: {root}")

    meta["python_executable"] = PureWindowsPath(meta["python_executable"]).name
    for pool in meta["threadpools"]:
        if "filepath" in pool:
            pool["filepath"] = PureWindowsPath(pool["filepath"]).name
    runtime = meta["cpu"].pop("numpy_runtime")
    simd = re.search(r"'simd_extensions':\s*(\{.*?\})", runtime, flags=re.S)
    if simd:
        meta["cpu"]["numpy_simd_extensions"] = ast.literal_eval(simd.group(1))
    context.pop("host_name", None)
    context["executable"] = PureWindowsPath(context["executable"]).name
    yield "python.json", json_bytes(python)
    yield "native.json", json_bytes(native)
    for name in BENCHMARK_FILES:
        yield name, (root / name).read_bytes()
    yield "demo/targets.json", (root / "demo/targets.json").read_bytes()
    for variant in ("scalar-unfused", "scalar-fused", "avx2-unfused", "avx2-fused"):
        for suffix in (".o", ".ll", ".s", "-plan.txt", "-remarks.txt"):
            name = variant + suffix
            contents = (root / "demo" / name).read_bytes()
            if suffix == ".s":
                text = contents.decode("utf-8")
                text = re.sub(r"^.*[\\/]([^\\/\s]+\.o):", r"\1:", text, flags=re.M)
                contents = text.encode("utf-8")
            yield "demo/" + name, contents


def hardening_evidence(root, commit):
    if (root / "commit.txt").read_text(encoding="utf-8").strip() != commit:
        raise ValueError("The sanitizer reports must match the release commit")
    tests = (root / "ctest.log").read_text(encoding="utf-8")
    passed = re.search(r"100% tests passed[^\n]*out of (\d+)", tests)
    if passed is None or any(re.search(rf"{re.escape(name)}[^\n]*Passed", tests) is None
                             for name in GUARD_TESTS):
        raise ValueError("Expected a passing instrumented run and all generated-code guard tests")
    rows = read_json(root / "coverage.json")["data"]
    coverage = []
    for row in rows:
        files = [{"filename": "src/" + PureWindowsPath(entry["filename"]).name,
                  "summary": entry["summary"]} for entry in row["files"]]
        coverage.append({"files": files, "totals": row["totals"]})
    compiler = (root / "compiler.txt").read_text(encoding="utf-8").splitlines()[0]
    summary = {"commit": commit, "compiler": compiler, "sanitizers": ["address", "undefined"],
               "ctest_entries_passed": int(passed.group(1)), "guard_tests": GUARD_TESTS,
               "coverage_scope": "first-party C++ files under src/"}
    return (("hardening.json", json_bytes(summary)), ("coverage.json", json_bytes(coverage)))


def package(args):
    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    for label, root in (("linux-x64", args.linux), ("windows-x64", args.windows)):
        for name, contents in timing_evidence(root, args.commit):
            entries[f"{label}/{name}"] = contents
    for name, contents in hardening_evidence(args.hardening, args.commit):
        entries[f"linux-hardening/{name}"] = contents
    manifest = {"commit": args.commit, "ci_run": args.run,
                "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json_bytes(manifest))
    print(f"Packaged {len(entries)} sanitized evidence files at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--hardening", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    package(parser.parse_args())


if __name__ == "__main__":
    main()
