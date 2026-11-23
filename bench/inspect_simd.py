"""Save and verify disassembly from the scalar and AVX2 objects emitted by LLJIT."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

import forge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)

    program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
    array = np.arange(9, dtype=np.float32)
    metadata = {"host_has_avx2": forge.has_avx2(), "kernels": {}}
    modes = ("scalar", "avx2") if metadata["host_has_avx2"] else ("scalar",)
    assembly = {}
    for mode in modes:
        kernel = program.jit(array, target=mode)
        object_file = options.out / f"{mode}.o"
        object_file.write_bytes(kernel.object)
        (options.out / f"{mode}.ll").write_text(kernel.info.ir, encoding="utf-8")
        (options.out / f"{mode}-remarks.txt").write_text(
            "\n".join(kernel.info.remarks) + "\n", encoding="utf-8"
        )
        assembly[mode] = subprocess.run(
            [str(options.objdump), "--disassemble", "--x86-asm-syntax=att", str(object_file)],
            capture_output=True, text=True, check=True,
        ).stdout
        (options.out / f"{mode}.s").write_text(assembly[mode], encoding="utf-8")
        metadata["kernels"][mode] = {
            key: getattr(kernel.info, key) for key in ("triple", "data_layout", "cpu", "features")
        }
        expected = program.reference(array)
        np.testing.assert_array_equal(kernel(array).view(np.uint32), expected.view(np.uint32))

    (options.out / "targets.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if re.search(r"\bymm[0-9]+\b", assembly["scalar"]):
        raise AssertionError("Unexpected YMM register in scalar object")
    if "avx2" in modes:
        for opcode in ("vaddps", "vmulps", "vsqrtps"):
            if not re.search(rf"\b{opcode}\b[^\n]*\bymm[0-9]+\b", assembly["avx2"]):
                raise AssertionError(f"Expected 256-bit {opcode} in AVX2 object")
        for opcode in ("vaddss", "vmulss", "vsqrtss"):
            if not re.search(rf"\b{opcode}\b", assembly["avx2"]):
                raise AssertionError(f"Expected {opcode} in AVX2 scalar remainder")
        print("Verified packed add, multiply, and sqrt plus their scalar tails in the JIT object.")
    else:
        print("AVX2 is unavailable on this host. Saved the scalar object and skipped AVX2 inspection.")


if __name__ == "__main__":
    main()
