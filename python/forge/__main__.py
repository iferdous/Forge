"""Try Forge expressions and compare the generated loop plans."""

import argparse
import json
import re

import numpy as np

from . import _forge_ir, compile, has_avx2

_DEFAULT_ARRAYS = ["x=[1,10,25,46,1,10,25,46,1]"]
_PACKED_OR_SCALAR = re.compile(r"\b(?:vadd|vmul|vsqrt)(?:ps|ss)\b")


def parse_arrays(specs):
    arrays = {}
    for spec in specs:
        name, separator, value = spec.partition("=")
        if not separator or not name or name in arrays:
            raise ValueError("Provide each input once as --array NAME=JSON")
        array = np.array(json.loads(value), dtype=np.float32)
        if array.ndim not in (1, 2):
            raise ValueError(f"{name!r} must be a one- or two-dimensional array")
        arrays[name] = array
    return arrays


def matches_reference(actual, expected):
    nan = np.isnan(expected)
    return (np.array_equal(np.isnan(actual), nan) and
            np.array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan]))


def show(expression, arrays, inspect):
    program = compile(expression, inputs=tuple(arrays))
    inputs = tuple(arrays.values())
    reference = program.reference(*inputs)
    print(f"Expression: {expression}")
    for name, array in arrays.items():
        print(f"{name} (float32{list(array.shape)}) = {array.tolist()}")

    if not hasattr(_forge_ir, "CompiledKernel"):
        for label, fusion in (("Unfused", False), ("Fused", True)):
            print(f"{label} plan\n{program.plan(*inputs, fusion=fusion).dump()}", end="")
        print(f"NumPy-based reference result: {reference.tolist()}")
        print("Build with FORGE_ENABLE_JIT=ON to run Forge's generated kernel.")
        return True

    target = "avx2" if has_avx2() else "scalar"
    unfused = program.jit(*inputs, target=target, fusion=False)
    fused = program.jit(*inputs, target=target)
    print(f"Unfused plan\n{unfused.plan.dump()}Fused plan\n{fused.plan.dump()}", end="")
    result = fused(*inputs)
    both_match = matches_reference(result, reference) and matches_reference(unfused(*inputs), reference)
    print(f"Forge result ({target}): {result.tolist()}")
    print("Both plans match the NumPy-based reference (same float32 bits outside NaNs)." if both_match
          else "At least one Forge result differs from the NumPy-based reference.")

    if inspect == "ir":
        print(f"Fused LLVM IR ({fused.info.triple})\n{fused.info.ir}", end="")
    else:
        assembly = program.emit_x86(*inputs, target="avx2")
        if inspect == "assembly":
            print(f"Fused Linux x86-64 AVX2 assembly\n{assembly.text}", end="")
        else:
            instructions = list(dict.fromkeys(_PACKED_OR_SCALAR.findall(assembly.text)))
            if instructions:
                print(f"Instructions in generated x86-64 AVX2 code: {', '.join(instructions)}")
    return both_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expression", nargs="?", help="Enter expressions at a prompt if omitted.")
    parser.add_argument("--array", action="append", metavar="NAME=JSON",
                        help="Override sample x with one or more float32 inputs.")
    parser.add_argument("--inspect", choices=("ir", "assembly"), help="Print the full fused code.")
    args = parser.parse_args()
    if args.inspect and not hasattr(_forge_ir, "CompiledKernel"):
        parser.error("--inspect requires a build with FORGE_ENABLE_JIT=ON")
    try:
        arrays = parse_arrays(args.array or _DEFAULT_ARRAYS)
        if args.expression is not None:
            return 0 if show(args.expression, arrays, args.inspect) else 1
    except (OverflowError, SyntaxError, TypeError, ValueError) as error:
        parser.error(str(error))

    print("Enter +, *, @, sqrt(), sum(), max(), or mean(). Try mean((x + 2) * 3) with the sample x.")
    print(f"Available inputs: {', '.join(arrays)}. Enter a blank line to quit.")
    while True:
        try:
            expression = input("forge> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not expression:
            return 0
        try:
            if not show(expression, arrays, args.inspect):
                return 1
        except (OverflowError, SyntaxError, TypeError, ValueError) as error:
            print(error)


if __name__ == "__main__":
    raise SystemExit(main())
