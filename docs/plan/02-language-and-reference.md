# 02 · Define the language and reference behavior

[Previous: toolchain and CI](01-toolchain-and-ci.md) · [Plan](../../README.md) · [Next: scalar JIT](03-scalar-jit.md)

**Outcome.** One input expression produces a deterministic, typed graph of tensor operations and a reference result before code generation enters the picture.

## Language and API

1. Start with a Python-facing expression such as `forge.compile("sqrt((x + 2) * 3)", inputs=("x",))`. Treat the snippet as a proposed public API. Define inputs by name and positional order, `+`, `*`, parenthesization, finite numeric literals, and a single-argument `sqrt` call. Expose `sum`, `max`, `mean`, and `@` as Step 06 arrives. Use a restricted expression AST with an explicit node whitelist. Bound input length, depth, and node count. Reject statements, attribute access, subscripts, arbitrary calls, non-numeric literals, and unknown names before lowering.
2. Keep syntax errors separate from argument/type/shape errors. Give every node a source location and base its ID on stable ordering. Normalize scalar literals into binary32 once and explicitly represent scalar splats. Build a typed directed acyclic graph with operation, ordered operands, and shape. Capture use counts after construction. Add a deterministic text dump for fixtures and the demo.
3. Accept dense, C-contiguous, native-endian `float32` arrays of rank one or two for elementwise work. Tensor operands must have equal shapes. Scalar literal broadcast is supported. Accept zero-length dimensions and define products with checked multiplication. Make outputs owned and separate from inputs initially. Defer implicit casts, copying, general broadcasting, arbitrary strides, and in-place writes.
4. Keep execution semantics separate from NumPy's broader API. Define binary32 rounding at every pointwise op, round-to-nearest ties-to-even, gradual underflow, and preserved operation order and signed zero. Compare non-NaN outputs by bit pattern under this contract. Compare NaNs by classification, and document Forge's handling of invalid square roots. Numerical result comparison is the contract. NumPy warnings, NaN payloads, and caller-modified floating-point flags carry no equivalence guarantee.

## Reference and tests

- Implement a simple reference walker with explicit dtype and one materialized result per operation. In Python, use `np.float32` constants and same-shape ufuncs. Add fixed-bit fixtures that exercise the ordering independently of LLVM and NumPy's warning behavior.
- Test repeated inputs and shared subexpressions, shape mismatch, empty and singleton arrays, overflow in dimension or byte-size products, invalid syntax, and acceptance of only supported constructs.
- Compare bits per result within the pointwise contract. Include `-0.0`, subnormals, overflow to infinity, NaN classification, and negative square root. Use deterministic random seeds for property comparisons.
- Keep enough cases in GoogleTest to exercise the C++ graph API directly. Cover parser, reference, and CLI or Python boundary failures in the corresponding frontend tests.

## Try the frontend on the M3

Open a terminal in the Forge checkout. Use Python 3.12 or newer and CMake 3.25 or newer. The x64 CI jobs use Python 3.13. The frontend preset builds the C++ graph and Python module without loading LLVM.

~~~sh
mkdir -p .local/tmp .local/cache
export TMPDIR="$PWD/.local/tmp" XDG_CACHE_HOME="$PWD/.local/cache"
export PIP_CACHE_DIR="$PWD/.local/cache/pip"
python3 -m venv .local/venv
.local/venv/bin/python -m pip install -r requirements.txt
cmake --preset frontend "-DPython_EXECUTABLE=$PWD/.local/venv/bin/python"
cmake --build --preset frontend --parallel
ctest --preset frontend
PYTHONPATH="$PWD/.local/build/frontend/python" .local/venv/bin/python -i
~~~

~~~python
import numpy as np
import forge

x = np.array([1, 10, 25, 46], dtype=np.float32)
program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
print(program.reference(x).tolist())  # [3.0, 6.0, 9.0, 12.0]
print(program.graph(x).dump())
~~~

The graph lists operations, shapes, source locations, and use counts. Change the expression and call `forge.compile` again to explore. This stage evaluates the NumPy reference. Stage 03 adds LLVM code generation.

## Stage 02 expression contract

- Expressions use ASCII input names, signed numeric literals, `+`, `*`, parentheses, and `sqrt(expression)`. Multiplication binds more tightly than addition. A sign applies to a literal. The parser accepts up to 4,096 source characters, 32 levels of expression nesting, and 256 visited expression nodes.
- Input names define positional order. Values are ordinary NumPy `ndarray` objects with native-endian `float32` dtype, rank one or two, and C-contiguous storage. Tensor operands must have equal shapes. Scalar expressions broadcast explicitly when combined with an array. An expression's output is an independently owned array.
- Integer literals round directly to binary32 with ties to even. Floating literals first take the binary64 value parsed by Python, then round to binary32. Literals must stay finite. The default floating-point environment provides ties-to-even rounding and gradual underflow. Each operation rounds independently. Invalid square roots return NaN. Tests compare non-NaN bits and classify NaNs.
- Repeated inputs and structurally identical subexpressions share graph nodes and keep their first source location. Ordered operands stay in source order. Array byte-size arithmetic is checked in C++, including shapes with zero-sized dimensions.
- Syntax and unknown-name failures raise `SyntaxError`. Array or argument type failures raise `TypeError`. Shape, layout, and result-rank failures raise `ValueError`.

Stage 06 extends this contract with full reductions, rank-zero results, and the `@` operator. [Aggregate expression contract](06-reductions-and-matmul.md#expression-and-numerical-contract)

## Exit evidence

A short language spec, fixture-based graph dumps, and passing reference tests on both CI platforms. The same expression and shapes yield the same typed graph and reference results before and after wiring the parser to C++.

The M3 frontend passed all four C++ graph tests and ten Python tests with Apple Clang 16, Python 3.13.7, pybind11 3.1.0, and NumPy 2.5.5. The [Ubuntu and Windows CI workflow](../../.github/workflows/ci.yml) runs those checks alongside the Stage 01 LLJIT smoke test.
