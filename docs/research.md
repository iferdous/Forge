# Forge research and constraints

Reviewed on November 22, 2026. The goal is to demonstrate when a fused tensor expression wins, identify the source of the improvement, and publish a reproducible comparison.

## Platform and toolchain

| Finding | Effect on Forge |
| --- | --- |
| The development Mac reported Apple M3, `arm64`, and macOS 26.6.2. The SSK volume is writable APFS. | Keep the Mac checkout, handoffs, development dependencies, build output, and research files under `/Volumes/SSK/Forge`. |
| Apple documents AVX and AVX2 translation in Rosetta, including support for most JIT applications. Translation applies to the entire process. | An optional translated smoke run needs x86-64 Python and x86-64 dependencies. Native ARM Python can build expressions and cross-emit assembly. An x86-64 machine supplies AVX2 performance evidence. [Apple Rosetta documentation](https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment?changes=_7) |
| GitHub lists x64 standard runners for Ubuntu and Windows. Runner labels describe the architecture, while the job must probe usable AVX2 before executing those kernels. | Use CI for functional checks and timing smoke tests. Use a stable x64 machine for public latency comparisons. A Codespace is an optional development environment after checking its architecture, CPU flags, and the account's compute and storage quota. [Runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) · [Codespaces billing](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces) |
| LLVM 23.1.6 was released November 17. The upstream release provides Linux x64 and Windows x64 archives and LLVM source. Windows has a separate developer archive with libraries. | Pin `llvmorg-23.1.6`. Validate a C++ LLVM package and link on each OS before writing the compiler. Keep the source build available if a binary package fails ABI or SDK checks. [LLVM 23.1.6 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.6) · [LLVM CMake guide](https://llvm.org/docs/CMake.html) |
| ORC LLJIT links relocatable objects in-process. Symbol lookup normally materializes the function. | Measure lookup inside cold compilation. Preserve the JIT owner while its function pointers are callable. Handle link and lookup failures as errors. [LLVM ORC guide](https://llvm.org/docs/ORCv2.html) |

## Cross-emission check performed on the M3

Apple Clang 16 cross-compiled a C loop equivalent to `sqrt((x + 2) * 3)` for an x86-64 AVX2 target. The compiler reported vectorization width eight. The assembly contains packed `vaddps`, `vmulps`, and `vsqrtps` using YMM registers. This validates the local assembly inspection route. The probe files reside on SSK under `.local/research/`. Forge now [cross-emits the same operation chain on the M3](plan/04-simd-and-baselines.md#run-and-inspect-the-avx2-kernel) and checks its own native JIT objects in x64 CI.

~~~c
void forge_probe(const float *input, float *restrict output, unsigned long count) {
    for (unsigned long i = 0; i < count; ++i) {
        output[i] = __builtin_sqrtf((input[i] + 2.0f) * 3.0f);
    }
}
~~~

~~~sh
clang --target=x86_64-unknown-linux-gnu -march=x86-64 -mavx2 -mno-fma \
  -O3 -fno-fast-math -ffp-contract=off -fno-math-errno \
  -Rpass=loop-vectorize -S probe.c -o probe.s
~~~

The packed floating-point instructions also exist in AVX. Record both the selected AVX2 target features and the emitted packed operations as evidence. LLVM can reject a loop for legal or cost reasons. Inspect optimization remarks alongside IR and disassembly. [LLVM vectorization guide](https://llvm.org/docs/Vectorizers.html)

## Correctness and performance boundaries

- **Fusion location.** The recommended pass runs on Forge's typed tensor IR, where producer use counts, shapes, and operation boundaries are visible. LLVM then optimizes the fused loop. An LLVM IR loop-fusion pass would need to recover legality from already-lowered loops, aliasing, and memory effects. Use LLVM's loop vectorizer first. Keep explicit vector IR as a fallback for a documented missed case. LLVM's loop-fusion documentation describes the dependence analysis required at the lower level. MLIR's Linalg dialect provides a precedent for fusing at the tensor-operation level. [LLVM Loop Fusion](https://llvm.org/docs/LoopFusion.html) · [MLIR elementwise fusion](https://mlir.llvm.org/docs/Passes/#-linalg-fuse-elementwise-ops)
- **Floating point.** Preserve elementwise op order and round each result to `float32`. Disable reassociation and FMA contraction. Use bit-level comparison for non-NaN elementwise outputs under the default floating-point environment and check NaNs by classification. Define reduction and matrix accuracy separately. NumPy can use partial pairwise summation, and BLAS may accumulate in a different order. [LLVM language reference](https://llvm.org/docs/LangRef.html#floating-point-environment) · [NumPy sum semantics](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)
- **Meaning of memory traffic.** For a single-input `float32` chain of `k` full-array passes with one read and write per pass, the byte model is `8k` bytes per element unfused versus `8` fused. For three passes the modeled total reduction is `66.7%`. For four it is `75%`. Both eliminate the modeled intermediate-array reads and writes. These figures describe logical array accesses, and cache behavior, new inputs, and allocator traffic can change actual hardware observations. Name any hardware event and its scope when publishing counters. [Google Benchmark counter documentation](https://google.github.io/benchmark/user_guide.html#user-requested-performance-counters)
- **Sources of speedup.** A Python API comparison measures dispatch, output allocation, and kernel execution together. A preallocated fused-versus-unfused Forge comparison isolates the kernel schedule at a matched call boundary and ISA. Include cold compilation separately and publish winning and losing sizes. Large fused chains can still benefit from less memory traffic. A tuned BLAS is a strong baseline for large matmul. [NumPy runtime introspection](https://numpy.org/doc/stable/reference/generated/numpy.show_runtime.html)
- **HRT connection.** Alex Yi amortized request overhead with a long-lived gRPC stream. Alex Burton measured the effect of going directly to storage and networking hardware. Forge applies the measurement approach to repeated array operations and intermediate data. Its compiled kernel is an ordinary userspace function. [HRT intern projects](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-software-engineering-summer-projects/)

## Delivery priorities

The first résumé-ready milestone is a validated fused expression, working x86-64 execution, a fair baseline, and a short demo. Protect that milestone from expansion into a general array framework. The handoff's three-to-four-week estimate depends on early Linux and Windows qualification and time available for testing. Reductions, matmul, and the ten-kernel suite complete the full plan. A performance goal can guide investigation while the measurements determine the claim.
