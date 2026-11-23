# 04 · Qualify SIMD and establish baselines

[Previous: scalar JIT](03-scalar-jit.md) · [Plan](../../README.md) · [Next: operator fusion](05-operator-fusion.md)

**Outcome.** LLVM's loop vectorizer produces packed operations for the unfused elementwise kernels, the scalar tail is correct, and the team can attribute changes against a working timing baseline.

## SIMD implementation

1. Derive target configuration once for the compiler pipeline and the object emitter. Use an explicit x86-64 AVX2 feature set and disable FMA contraction. Keep a separately compiled scalar path safe on CPUs without AVX2. Gate the AVX2 path on the executor architecture and usable hardware and OS state. Avoid applying an AVX2 flag globally across Forge. Capture the chosen target from the actual run.
2. Run LLVM's optimization pipeline with loop vectorization enabled. Capture passed and missed vectorization remarks through LLVM's diagnostic interface. Prefer simple unit-stride memory access and a fresh output. Add LLVM alias metadata only for facts enforced by the runtime. Inputs may alias each other when read-only. Natural `float32` alignment is sufficient for unaligned packed instructions. Preserve the scalar remainder.
3. Capture optimized IR, actual JIT object disassembly where the ORC layer permits it, and selected target features. An equivalent emitted object is useful when it shares the verified IR, layout, target settings and codegen pipeline. Mark how it was obtained. Look for packed arithmetic using YMM registers and confirm an independent scalar tail. If LLVM declines, examine diagnostics and simplify the IR. Explicit vector IR becomes a fallback for a demonstrated blocker.
4. Compare scalar and SIMD output bit patterns under the pointwise contract. Test sizes `0, 1, 7, 8, 9, 15, 16, 17, 1023, 1024, 1025`, contiguous inputs misaligned relative to 32 bytes, two input arrays, and special floats. Add guard or sanitizer checks around tail accesses. Check that the host selects scalar or emits a clear error when AVX2 is unavailable.

## Initial timing

- In Google Benchmark, compare scalar and AVX2 kernels using the same buffers, preallocated output, threading, input generation, and warmup. Capture raw JSON. Keep compilation and allocation out of the native timed loop.
- Add a Python harness for NumPy and Forge on one-op cases. Use matched warm call boundaries and `float32` data. Report allocations separately and preserve one fixture that deliberately uses the typical allocating API.
- Use CI for execution and benchmark plumbing. Record the host before interpreting timings. Keep timing thresholds off shared hosted CI.

## Exit evidence

One verified disassembly with packed arithmetic and a correct scalar tail. Record which x64 hosts actually ran AVX2, which used scalar, and why. Save initial raw native and Python timing artifacts with enough metadata to reproduce them. [LLVM loop vectorizer diagnostics](https://llvm.org/docs/Vectorizers.html#diagnostics) · [Google Benchmark guide](https://google.github.io/benchmark/user_guide.html)
