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

## Exit evidence

A short language spec, fixture-based graph dumps, and passing reference tests on both CI platforms. The same expression and shapes yield the same typed graph and reference results before and after wiring the parser to C++.
