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

## Run and inspect the AVX2 kernel

Use the same `program` and `x` as the [scalar example](03-scalar-jit.md#run-and-inspect-the-scalar-kernel). The AVX2 option checks the current process architecture and LLVM's OS-aware AVX2 feature probe before compiling for native execution.

~~~python
print(program.emit_x86(x, target="avx2", fusion=False).text)
if forge.has_avx2():
    kernel = program.jit(x, target="avx2", fusion=False)
    print(kernel(x).tolist())
    print("\n".join(kernel.info.remarks))
~~~

Forge asks LLVM's loop vectorizer for eight `float32` values per iteration and one vector per loop step. It emits separate scalar cleanup loops and keeps the unfused operator boundaries. The per-loop output can carry LLVM's `noalias` attribute because the launcher keeps it disjoint from the read buffers. Read inputs may share storage. LLVM loads and stores use alignment one so contiguous NumPy arrays with byte offsets are supported.

## Reproduce the baseline

CI enables `FORGE_BUILD_BENCHMARKS` with Google Benchmark 1.10.0 and uploads one evidence archive per operating system. It includes the commit and host model, scalar and AVX2 LLVM IR, the actual objects produced by LLJIT, their disassembly, vectorization remarks, and raw JSON timing samples. The inspection checks for YMM arithmetic and scalar remainder instructions in the actual AVX2 object.

The native benchmark compares preallocated scalar and AVX2 calls for `x + 2` and the three-operation demo. The Python harness compares `x + 2` across Forge and NumPy for both allocating and preallocated APIs. Both harnesses use wall-clock timing. The native harness also records CPU-time fields for inspection. Compilation, fixture generation, and warmup are outside the recorded samples. Python percentiles can be derived from the raw batch means. Durations from shared CI are smoke measurements. The release benchmark will use qualified timing hardware.

The [Stage 04 workflow](https://github.com/iferdous/Forge/blob/dd421e3/.github/workflows/ci.yml) shows the exact commands used for this baseline and its disassembly check.

## Exit evidence

One verified disassembly with packed arithmetic and a correct scalar tail. Record which x64 hosts actually ran AVX2, which used scalar, and why. Save initial raw native and Python timing artifacts with enough metadata to reproduce them. [LLVM loop vectorizer diagnostics](https://llvm.org/docs/Vectorizers.html#diagnostics) · [Google Benchmark guide](https://google.github.io/benchmark/user_guide.html)


## Verified results

The [Stage 04 run](https://github.com/iferdous/Forge/actions/runs/45101169543) passed on both x64 runners at `dd421e3`. Both hosts executed AVX2 and passed the guard-page and numerical tests. Actual LLJIT object disassembly shows YMM arithmetic and scalar cleanup for add, multiply, and square root. The M3 executed the scalar tests and emitted x86-64 AVX2 assembly for offline inspection.

Warm wall-clock medians for **65,536 float32 elements** are shown below. The native demo measures all three operations with a preallocated workspace. The Python columns measure the separate `x + 2` workload through preallocated Python APIs.

| CI runner | Native demo scalar | Native demo AVX2 | Python NumPy add | Python Forge AVX2 add |
| --- | ---: | ---: | ---: | ---: |
| Ubuntu x64 · AMD EPYC 9V45 | 100.1 µs | 13.4 µs | 3.17 µs | 2.65 µs |
| Windows x64 · Intel(R) Xeon(R) Platinum 8370C | 111.1 µs | 25.2 µs | 6.59 µs | 8.78 µs |

The native harness uses a wall clock on both systems. The initial Windows CPU-time samples showed coarse timer steps and were retained separately from this baseline. MSVC reports C4702 warnings in the LLVM SDK headers; the test suites and artifact checks pass. The verified raw reports, LLVM IR, and object files are saved on SSK under `.local/evidence/stage-04/baseline/`.
