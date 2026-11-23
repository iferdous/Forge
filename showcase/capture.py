"""Capture real Forge compiler stages and an isolated Linux RSS probe."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile

import numpy as np
import forge

CHAIN = "sqrt((x + 2) * 3)"
RSS_ELEMENTS = 8_388_608


def array(values):
    return np.asarray(values, dtype=np.float32)


def examples():
    x = array([1, 10, 25, 46])
    for length in (8, 9, 17):
        yield f"chain-{length}", f"Pointwise chain · {length} elements", CHAIN, {"x": np.resize(x, length)}
    yield "shared", "Shared producer", "sqrt((x + 2) * (x + 2))", {"x": np.resize(x, 9)}
    yield "reduction", "Ordered sum", "sum(x)", {"x": array([16777216, 1, -16777216, 4])}
    yield "matmul-small", "Small matrix product", "a @ b", {
        "a": array([[1, 2, 3], [4, 5, 6]]),
        "b": array([[1, 2], [3, 4], [5, 6]]),
    }
    yield "matmul-edge", "Matrix tile edges", "a @ b", {
        "a": np.arange(17 * 33, dtype=np.float32).reshape(17, 33) % 7 - 3,
        "b": np.arange(33 * 35, dtype=np.float32).reshape(33, 35) % 5 - 2,
    }


def assembly_for(program, args, kernel, target, fusion, objdump, folder):
    if target == "avx2" and objdump:
        path = folder / "kernel.o"
        path.write_bytes(kernel.object)
        text = subprocess.run([objdump, "--disassemble", "--no-show-raw-insn", str(path)],
                              check=True, capture_output=True, text=True).stdout
        return re.sub(r"^.*kernel\.o:", "kernel.o:", text, flags=re.M).strip(), "LLJIT object"
    crossed = program.emit_x86(*args, target="avx2", fusion=fusion)
    return crossed.text.strip(), "Cross-emitted x86-64 assembly"


def excerpt(assembly):
    mnemonics = r"\b(?:v(?:add|mul|sqrt)(?:ps|ss)|vmovups|vmaxss|addss|mulss)\b"
    lines = []
    for line in assembly.splitlines():
        match = re.search(mnemonics, line)
        if match:
            lines.append(" ".join(line[match.start():].split("#", 1)[0].split()))
    return lines[:22]


def capture_case(spec, target, objdump, folder):
    key, title, expression, inputs = spec
    args = tuple(inputs.values())
    program = forge.compile(expression, inputs=tuple(inputs))
    reference = program.reference(*args)
    result = {}
    for label in ("unfused", "fused"):
        fusion = label == "fused"
        kernel = program.jit(*args, target=target, fusion=fusion)
        output = kernel(*args)
        np.testing.assert_array_equal(output.view(np.uint32), reference.view(np.uint32))
        asm, source = assembly_for(program, args, kernel, target, fusion, objdump, folder)
        result[label] = {
            "plan": kernel.plan.dump().strip(), "loops": kernel.plan.loop_count,
            "temporaries": kernel.plan.temporary_arrays, "ir": kernel.info.ir, "assembly": asm,
            "assembly_source": source, "instructions": excerpt(asm), "target": kernel.info.triple,
        }
    small = sum(value.size for value in inputs.values()) < 100
    command = ["python", "-m", "forge", expression]
    if small:
        for name, data in inputs.items():
            command.extend(("--array", f"{name}={json.dumps(data.tolist(), separators=(',', ':'))}"))
    return {
        "id": key, "title": title, "expression": expression,
        "command": shlex.join(command) if small else "python showcase/capture.py --out captures.json",
        "inputs": {name: {"shape": data.shape, "sample": data.tolist() if small else data[0, :8].tolist()}
                   for name, data in inputs.items()},
        "output": reference.tolist() if small else reference[:2, :8].tolist(),
        "output_shape": reference.shape, "sampled": not small,
        "graph": program.graph(*args).dump(), "plans": result, "verified": True,
    }


def resident_bytes():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def rss_worker(mode):
    x = np.full(RSS_ELEMENTS, 10, dtype=np.float32)
    out = np.full_like(x, -1)
    program = forge.compile(CHAIN, inputs=("x",))
    before = resident_bytes()
    target = "avx2" if forge.has_avx2() else "scalar"
    kernel = program.jit(x, target=target, fusion=mode == "fused")
    compiled = resident_bytes()
    kernel.run_into(out, x)
    executed = resident_bytes()
    np.testing.assert_array_equal(out, np.float32(6))
    return {"target": target, "before_compile": before, "after_compile": compiled,
            "after_first_call": executed, "temporaries": kernel.plan.temporary_arrays,
            "scratch_bytes": kernel.plan.temporary_arrays * x.nbytes}


def capture_cache(target):
    x = array([1, 10, 25, 46])
    program = forge.compile(CHAIN, inputs=("x",))
    forge.clear_cache()
    snapshots = [forge.cache_info()._asdict()]
    for length in (4, 4, 9):
        values = np.resize(x, length)
        kernel = program.jit(values, target=target, cache=True)
        np.testing.assert_array_equal(kernel(values), program.reference(values))
        snapshots.append(forge.cache_info()._asdict())
    forge.clear_cache()
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--objdump", type=Path)
    parser.add_argument("--require-avx2", action="store_true")
    parser.add_argument("--rss-worker", choices=("unfused", "fused"), help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.rss_worker:
        print(json.dumps(rss_worker(options.rss_worker)))
        return
    if not options.out:
        parser.error("--out is required")
    target = "avx2" if forge.has_avx2() else "scalar"
    if options.require_avx2 and (target != "avx2" or not options.objdump):
        parser.error("Native AVX2 and llvm-objdump are required for the public capture")
    options.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=options.out.parent) as temporary:
        captured = [capture_case(spec, target, options.objdump, Path(temporary)) for spec in examples()]
    memory = {}
    if Path("/proc/self/statm").exists():
        for mode in ("unfused", "fused"):
            command = [sys.executable, __file__, "--rss-worker", mode]
            memory[mode] = [json.loads(subprocess.run(command, check=True, capture_output=True,
                                                      text=True).stdout) for _ in range(3)]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    result = {"schema": 1, "host": platform.system() + " " + platform.machine(), "commit": commit,
              "execution_target": target, "cases": captured, "cache": capture_cache(target),
              "rss_elements": RSS_ELEMENTS, "rss_samples": memory}
    options.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Captured {len(captured)} verified Forge executions on {result['host']} ({target}).")


if __name__ == "__main__":
    main()
