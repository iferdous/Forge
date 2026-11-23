# 06 · Add reductions and tiled matmul

[Previous: operator fusion](05-operator-fusion.md) · [Plan](../../README.md) · [Next: Python runtime and benchmarks](07-python-runtime-and-benchmarks.md)

**Outcome.** Forge supports full-array sum, max, and mean plus rank-two matrix multiplication, with explicit accuracy expectations and honest NumPy comparisons.

## Reductions

1. Define the first contract for `float32` full-array reductions. Specify accumulator initialization and order. For empty input, return positive zero for sum, quiet NaN for mean, and a clear error for max. Define NaN propagation and a deterministic tie policy for signed zeros in max. Write reference tests for these rules first.
2. Implement scalar, fixed-order accumulation. Keep sum and mean in the declared dtype and evaluate mean using one final division. Treat reductions as fusion boundaries. LLVM can vectorize certain ordered reductions on some targets. General packed reduction trees change the accumulation order, so preserve the scalar order for the initial x86 implementation.
3. Compare against the fixed-order reference. Compare separately with NumPy and a higher-precision reference where useful. Include cancellation, very large and small magnitudes, positives, negatives, empty inputs, and non-multiple lengths. For finite sums where the fixture contract excludes overflow, decide the error tolerance before benchmarking and scale it to the sum of input magnitudes and the accumulation length. Preserve the actual worst error and publish the accumulation order. [LLVM reduction notes](https://llvm.org/docs/Vectorizers.html#reductions) · [NumPy sum semantics](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)

## Matrix multiplication

1. Support two row-major `float32` matrices shaped `M × K` and `K × N` with a fresh output. Check dimension and byte-size arithmetic. Define empty output dimensions and zero-fill a nonempty result if `K` is zero.
2. Implement a straightforward C++ scalar reference. Lower one tiled loop nest in LLVM IR and preserve accumulation order per output element. Try one tile configuration first, keep contraction disabled, and include edge tiles. Compare small square, rectangular, and odd-dimension cases. Test the declared float behavior separately from throughput.
3. Record absolute and scale-aware error against a higher-precision reference for finite cases. Include a matched, single-thread NumPy BLAS baseline using the same data and output allocation policy. Record the backend and the observed thread settings. Large matrix timings are part of the evidence even if Forge loses. [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) · [NumPy runtime introspection](https://numpy.org/doc/stable/reference/generated/numpy.show_runtime.html)

## Exit evidence

Passing reduction and matmul fixtures plus a short numerical report defining the order and error bounds. Include time, output error, and the NumPy backend on a range of shapes. If the aggregate kernels threaten the elementwise release, ship the fusion milestone at its documented scope and track the remaining kernels as a follow-on.
