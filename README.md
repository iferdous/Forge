# Forge

Forge is a small tensor compiler for CPU kernels. It uses C++20 and LLVM ORC LLJIT, with a Python frontend and an x86-64 AVX2 release target.

| Benchmark evidence | Status |
| --- | --- |
| Three-op demo, 65,536 float32 elements, native AVX2 | Ubuntu 23.25 → 12.83 µs and Windows 25.81 → 14.31 µs with fusion (~1.8× on both) |
| Benchmark scope | Warm medians on shared CI, with preallocated buffers and compilation excluded |
| Fusion comparison | [Timings and method](docs/plan/05-operator-fusion.md#verified-results) · [Raw samples and JIT disassembly](https://github.com/iferdous/Forge/actions/runs/45102680498) |
| Reductions and matmul | [Numerical checks and single-thread NumPy comparison](docs/plan/06-reductions-and-matmul.md#verified-results) |

## What Forge compiles

The first complete workload is `sqrt((x + 2) * 3)` on a contiguous `float32` array. Forge validates the expression, represents it as a graph of tensor operations, and fuses eligible operations into one loop. The example keeps the intermediate values in registers and writes only the result array. The AVX2 path runs eight single-precision lanes at a time and uses a scalar tail for the remaining elements.

~~~text
Python expression → validated AST → typed tensor IR → kernel plan
                                                        ↓
                          NumPy result ← tests ← LLVM IR → LLJIT → output array
~~~

LLVM supplies instruction selection, vectorization, and in-process linking. Forge owns the language, shape rules, kernel planning, fusion legality, and the experiments that isolate each optimization. Forge also compiles full-array `sum`, `max`, and `mean` and tiled rank-two matrix multiplication.

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

The terminal playground runs both plans and checks their results against the reference. Omit the expression to open the interactive prompt. On Apple Silicon the playground executes the scalar kernel and can show generated x86-64 assembly. [Build and run Forge](docs/plan/05-operator-fusion.md#try-an-expression)

~~~sh
python -m forge 'sqrt((x + 2) * 3)'
python -m forge 'mean((x + 2) * 3)'
~~~

The HRT engineering write-up inspired Forge's focus on measuring repeated overhead and intermediate data movement. Forge applies that idea to arithmetic on arrays already in memory. [HRT intern projects](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-software-engineering-summer-projects/)
