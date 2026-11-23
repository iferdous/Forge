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

## Expression and numerical contract

`sum(x)`, `max(x)`, and `mean(x)` consume every element of a rank-one or rank-two input in C order. An aggregate result is an owned, rank-zero float32 NumPy array. A reduction can participate in scalar arithmetic or broadcast into a later array expression. `run_into` accepts a writable rank-zero float32 output for a scalar result.

- Sum starts at positive zero and adds each value with float32 rounding. Mean uses that sum and divides once by the count converted to float32. Empty sum returns positive zero and empty mean returns a quiet NaN.
- Max starts at negative infinity, propagates NaNs, and selects positive zero on a mixed-sign zero tie. All-negative zero candidates retain negative zero. `max` on an empty array raises `ValueError` when Forge binds the input shapes.
- `a @ b` accepts C-contiguous float32 matrices of shape `M × K` and `K × N`, producing `M × N`. An empty output keeps its shape. When only `K` is zero, Forge fills the output with positive zeros. Each cell starts at positive zero and accumulates one rounded product at a time in increasing `K` order.
- LLVM emits a tiled matmul nest with tile extents `M = 16`, `N = 32`, and `K = 32`. It clamps all three edges. The AVX2 loop packs independent output columns. Reductions stay scalar and retain their operation order. Both operations are boundaries for pointwise fusion.

Forge guarantees the declared float32 order in the default floating-point environment. Correctness tests compare bits except for NaN payloads, which are checked by classification. The ordered reference is separate from NumPy's timing baseline, where sum may use pairwise accumulation and matmul can delegate to BLAS. [LLVM vectorization](https://llvm.org/docs/Vectorizers.html#reductions) · [NumPy sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) · [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

## Try it

Set `PYTHONPATH` to the LLVM-enabled build's Python directory. The playground shows both plans and verifies the result. `--inspect ir` prints optimized LLVM IR and `--inspect assembly` prints x86-64 AVX2 assembly. [Build and playground setup](05-operator-fusion.md#try-an-expression)

~~~sh
python -m forge 'mean((x + 2) * 3)'
python -m forge 'a @ b' --array 'a=[[1,2,3],[4,5,6]]' --array 'b=[[1,2],[3,4],[5,6]]'
~~~

## How the evidence is measured

The harness uses seed `620261122`, four reduction lengths from 33 to 1,048,576, and five matrix cases from a 16-square multiply to a 512-square multiply. It times warm Python calls with preallocated output for NumPy and Forge, using nine interleaved samples and excluding compilation. For the BLAS comparison it sets one thread and verifies the live library's thread count. [Benchmark code](../../bench/aggregates.py) · [Thread-pool introspection](https://github.com/joblib/threadpoolctl)

The harness checks Forge's output bitwise against a reference that rounds each step in float32 and measures absolute error independently against a higher-precision result. For finite fixtures, let `u = 2^-24` and `γ(n) = n·u/(1−n·u)`. Sum uses `γ(L) · Σ|x_i|` for length `L`. Mean uses `γ(L+1) · Σ|x_i| / L` and the sampled counts are exactly representable. Max uses zero error. Matmul uses `γ(K+1) · Σ|a_ik·b_kj|` per output cell. The reports include both the largest absolute error and the worst fraction of the applicable bound. Separate correctness fixtures cover cancellation, overflow, subnormals, NaNs, signed zeros, and empty shapes.

## Verified results

Ubuntu and Windows each passed all 20 CTest entries. Their native LLJIT objects show packed AVX2 multiply and add in the matmul column loop. The reduction IR stays scalar under both target settings. Both runners passed all 17 benchmark accuracy checks and reported OpenBLAS 0.3.34.237.2 with one active thread. The M3 also passed the native scalar checks and emitted the x86-64 code for offline inspection. [CI run and raw evidence](https://github.com/iferdous/Forge/actions/runs/45104892251)

Warm wall-clock medians in microseconds for `b74bbd2` are shown below. Each cell lists Forge with the AVX2 target followed by NumPy. The reductions still use scalar arithmetic. The Ubuntu runner used a Xeon 6975P-C and the Windows runner used an EPYC 9V45.

| Workload | Ubuntu Forge / NumPy (µs) | Windows Forge / NumPy (µs) |
| --- | ---: | ---: |
| Sum, 33 elements | 0.332 / 1.418 | 0.314 / 1.525 |
| Max, 33 elements | 0.371 / 1.449 | 0.323 / 1.542 |
| Mean, 33 elements | 0.331 / 3.353 | 0.315 / 3.803 |
| Sum, 1,048,576 elements | 551.3 / 198.3 | 525.3 / 197.5 |
| Max, 1,048,576 elements | 2,663 / 148.7 | 1,598 / 41.64 |
| Mean, 1,048,576 elements | 550.9 / 203 | 537 / 207.7 |
| Matmul, 16 × 16 × 16 | 0.680 / 0.702 | 0.757 / 1.076 |
| Matmul, 31 × 63 × 17 | 3.899 / 1.307 | 4.750 / 1.528 |
| Matmul, 64 × 64 × 64 | 25.53 / 3.611 | 27.5 / 3.239 |
| Matmul, 256 × 256 × 256 | 1,209 / 192.3 | 1,357 / 157.9 |
| Matmul, 512 × 512 × 512 | 14,710 / 1,609 | 10,500 / 1,295 |

Forge's largest absolute errors in this sweep were `9.47e-3` for sum, zero for max, `1.26e-8` for mean, and `9.08e-5` for matmul. Its worst fractions of the applicable bounds were 0.092% for sum, 0.082% for mean, and 14.4% for matmul. The raw reports keep each shape and both backends' errors separately.

The 33-element reductions benefited from the prepared kernel call. NumPy led at 1,048,576 elements and for every tested matmul above the smallest square case on both runners. At 512-square, the NumPy lead ranged from 5.6× to 9.1× across two shared CI runs. The raw timing samples, numerical results, JIT disassembly, and demo transcripts are archived on SSK under `.local/evidence/stage-06/`. [First benchmark run](https://github.com/iferdous/Forge/actions/runs/45104323567)
