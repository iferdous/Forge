# Forge

Forge is a small tensor compiler that fuses array expressions into CPU kernels. It uses C++20 and LLVM ORC LLJIT, with a Python frontend and an x86-64 AVX2 release target.

| Benchmark evidence | Status |
| --- | --- |
| NumPy comparison, cold compilation, and fusion traffic | Awaiting the compiler and benchmark harness |

## What Forge compiles

The first complete workload is `sqrt((x + 2) * 3)` on a contiguous `float32` array. Forge validates the expression, represents it as a graph of tensor operations, and lowers it to a loop. The fusion pass keeps the add and multiply results within that loop. LLVM can then run the loop across eight single-precision lanes at a time. A scalar tail handles the remaining elements.

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

The plan is ready. Step 01 is next. The target build matrix is Ubuntu x64 and Windows x64. Apple Silicon is useful for development and cross-emitting assembly. A qualified native x86-64 host provides release execution and timing evidence.

The HRT engineering write-up inspired Forge's focus on measuring repeated overhead and intermediate data movement. Forge applies that idea to arithmetic on arrays already in memory. [HRT intern projects](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-software-engineering-summer-projects/)
