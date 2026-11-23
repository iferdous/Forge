"""Verify and save Stage 05 plans and the actual objects emitted by LLJIT."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

import forge


def verify_assembly(assembly):
    for label in ("scalar-unfused", "scalar-fused"):
        if re.search(r"\bymm[0-9]+\b", assembly[label]):
            raise AssertionError(f"Unexpected YMM register in {label}")
    if "avx2-fused" not in assembly:
        return

    for label in ("avx2-unfused", "avx2-fused"):
        for opcode in ("vaddps", "vmulps", "vsqrtps"):
            if not re.search(rf"\b{opcode}\b[^\n]*\bymm[0-9]+\b", assembly[label]):
                raise AssertionError(f"Expected 256-bit {opcode} in {label}")
        for opcode in ("vaddss", "vmulss", "vsqrtss"):
            if not re.search(rf"\b{opcode}\b", assembly[label]):
                raise AssertionError(f"Expected {opcode} in the {label} scalar remainder")
    fused = assembly["avx2-fused"]
    if re.search(r"\bvfn?m(?:add|sub)\w*", fused):
        raise AssertionError("Unexpected floating-point contraction in the fused loop")
    stores = re.findall(r"\bvmov[au]ps\s+%ymm[0-9]+\s*,\s*[^%\s][^\n]*", fused)
    if len(stores) != 1:
        raise AssertionError(f"Expected one packed output store in the fused object, found {len(stores)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-avx2", action="store_true")
    options = parser.parse_args()
    if options.require_avx2 and not forge.has_avx2():
        parser.error("The evidence run requires native AVX2")
    options.out.mkdir(parents=True, exist_ok=True)

    program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
    array = np.resize(np.array([1, 10, 25, 46], dtype=np.float32), 9)
    expected = program.reference(array)
    metadata = {"host_has_avx2": forge.has_avx2(), "kernels": {}, "reference_output": expected.tolist()}
    modes = ("scalar", "avx2") if metadata["host_has_avx2"] else ("scalar",)
    assembly = {}
    for mode in modes:
        for fusion in (False, True):
            label = f"{mode}-{'fused' if fusion else 'unfused'}"
            kernel = program.jit(array, target=mode, fusion=fusion)
            assert (kernel.plan.loop_count, kernel.plan.temporary_arrays) == (
                (1, 0) if fusion else (3, 2)
            )
            np.testing.assert_array_equal(kernel(array).view(np.uint32), expected.view(np.uint32))
            object_file = options.out / f"{label}.o"
            object_file.write_bytes(kernel.object)
            (options.out / f"{label}.ll").write_text(kernel.info.ir, encoding="utf-8")
            (options.out / f"{label}-plan.txt").write_text(kernel.plan.dump(), encoding="utf-8")
            (options.out / f"{label}-remarks.txt").write_text(
                "\n".join(kernel.info.remarks) + "\n", encoding="utf-8"
            )
            assembly[label] = subprocess.run(
                [str(options.objdump), "--disassemble", "--x86-asm-syntax=att", str(object_file)],
                capture_output=True, text=True, check=True,
            ).stdout
            (options.out / f"{label}.s").write_text(assembly[label], encoding="utf-8")
            metadata["kernels"][label] = {
                **{key: getattr(kernel.info, key) for key in ("triple", "data_layout", "cpu", "features")},
                "loops": kernel.plan.loop_count,
                "temporary_arrays": kernel.plan.temporary_arrays,
            }

    (options.out / "targets.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    verify_assembly(assembly)
    print("Verified three unfused loops and one fused loop with bit-identical demo results.")
    if metadata["host_has_avx2"]:
        print("Verified packed operations, scalar tails, and one packed output store in fused JIT code.")
    else:
        print("AVX2 is unavailable on this host. Saved the native scalar evidence.")


if __name__ == "__main__":
    main()
