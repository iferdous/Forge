# Forge

Forge is a small tensor compiler for CPU kernels. It uses C++20 and LLVM ORC LLJIT, with a Python frontend and an x86-64 AVX2 release target.

| Benchmark evidence | Status |
| --- | --- |
| Unfused three-op demo, 65,536 float32 elements | 100.1 µs scalar, 13.4 µs AVX2 median wall time on one Ubuntu CI runner |
| Native and NumPy baselines | [Raw samples and JIT disassembly](https://github.com/iferdous/Forge/actions/runs/45101169543) |

## What Forge compiles

The first complete workload is `sqrt((x + 2) * 3)` on a contiguous `float32` array. Forge validates the expression, represents it as a graph of tensor operations, and lowers each operation to a loop. The AVX2 path runs eight single-precision lanes at a time and uses a scalar tail for the remaining elements. The next milestone combines the three loops into one.

~~~text
Python expression → validated AST → typed tensor IR → kernel plan
                                                        ↓
                          NumPy result ← tests ← LLVM IR → LLJIT → output array
~~~

LLVM supplies instruction selection, vectorization, and in-process linking. Forge owns the language, shape rules, kernel planning, fusion legality, and the experiments that isolate each optimization. The full scope includes sum, max, mean, and tiled matrix multiplication.

## Implementation plan

| Step | Deliverable |
| --- | --- |
| [01 · Toolchain and CI](docs/plan/01-toolchain-and-ci.md) | Pin LLVM, qualify the environments, and prove the LLVM libraries link |
| [02 · Language and reference](docs/plan/02-language-and-reference.md) | Define the expression syntax, shapes, and floating-point contract |
| [03 · Scalar JIT](docs/plan/03-scalar-jit.md) | Compile and run a correct elementwise expression end to end |
| [04 · SIMD and baselines](docs/plan/04-simd-and-baselines.md) | Show packed instructions, test scalar tails, and capture initial timings |
| [05 · Operator fusion](docs/plan/05-operator-fusion.md) | Fuse a legal chain and isolate its effect with a control |
| [06 · Reductions and matmul](docs/plan/06-reductions-and-matmul.md) | Add aggregate and matrix kernels with separate accuracy checks |
| [07 · Python runtime and benchmarks](docs/plan/07-python-runtime-and-benchmarks.md) | Validate the Python API, compilation cache, and ten-kernel comparison |
| [08 · Release and handoff](docs/plan/08-release-and-handoff.md) | Publish a reproducible benchmark, a short demo, and evidence-backed résumé bullets |

[Research and constraints](docs/research.md) records the platform checks, numerical limits, and measurement rules behind the plan.

## Development status

[Step 03](docs/plan/03-scalar-jit.md) added the scalar JIT. [Step 04](docs/plan/04-simd-and-baselines.md) added native AVX2 and the benchmark harness. On Apple Silicon you can run the scalar kernel and [inspect the cross-emitted x86 assembly](docs/plan/04-simd-and-baselines.md#run-and-inspect-the-avx2-kernel). The timing above is a CI smoke measurement for the native call boundary. [Step 05](docs/plan/05-operator-fusion.md) adds operator fusion.

The HRT engineering write-up inspired Forge's focus on measuring repeated overhead and intermediate data movement. Forge applies that idea to arithmetic on arrays already in memory. [HRT intern projects](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-software-engineering-summer-projects/)
