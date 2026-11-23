# 05 · Fuse elementwise operations

[Previous: SIMD and baselines](04-simd-and-baselines.md) · [Plan](../../README.md) · [Next: reductions and matmul](06-reductions-and-matmul.md)

**Outcome.** Forge owns a compiler pass that combines a legal operation chain into one loop. The same graph can also run unfused, enabling a controlled comparison.

## Design and legality

1. Run fusion on the typed tensor IR. Start at an output and collect consecutive pure, elementwise nodes with compatible shape and dtype. Inline single-use producers into the consumer kernel while preserving operand ordering. Leave multi-use producers materialized initially. Treat reductions, matmul, and changes of iteration space as boundaries.
2. Give the kernel plan a stable textual form. Show intermediate allocations and loop count in the unfused plan, followed by the fused plan with one loop and no temporary arrays for eligible expressions. Keep a switch that disables fusion while preserving target, dtype, and floating-point settings. Allow both plans to enter through the same public ABI.
3. Lower the fused region to one LLVM loop with scalar SSA values for intermediate arithmetic. Use the same LLVM pipeline as the unfused version. Verify that the optimized body retains separable multiply and add where the source requires them and that the assembly has no unintended FMA or temporary-array stores for the demo kernel.
4. Test the decision pass and the results independently. Include chains of length one to four, repeated reads, shared producers, two-array operations, non-fusible boundaries, non-vector-width lengths, empty arrays, and special floats. Compare fused and unfused scalar and SIMD results with the reference. When a property check fails, record the seed and minimal expression.

## Try an expression

Build with the pinned LLVM SDK and set `PYTHONPATH` to the configured build's `python` directory. The [scalar JIT guide](03-scalar-jit.md#run-and-inspect-the-scalar-kernel) gives the build locations for the x64 preset and the M3 checkout. The terminal playground runs both plans, compares their output with the NumPy reference, and lists packed and scalar x86 instructions from the fused assembly.

~~~sh
python -m forge 'sqrt((x + 2) * 3)'
python -m forge 'sqrt((x + 2) * 3)' --inspect assembly
python -m forge 'sqrt((a + b) * 3)' --array 'a=[1,2,3]' --array 'b=[2,3,4]'
python -m forge
~~~

The default sample has nine elements, so on an AVX2 host the demo covers one packed iteration and a scalar remainder. On the M3, the result comes from the native scalar JIT and the assembly comes from the Linux x86-64 cross emitter. Use `--inspect ir` to show optimized LLVM IR for the host kernel. The expression language supports `+`, `*`, and `sqrt()` with float32 scalars and equal-shape rank-one or rank-two arrays.

The Python API exposes `program.plan(x)` without requiring LLVM and `program.jit(x, fusion=False)` for the unfused control. Forge counts uses among live operations. For example, `(x + 2) * (x + 2)` feeds one addition into both operands of the multiply and retains one intermediate array. Inside a fused loop, values stay in SSA form until the final store. The lowering preserves operation order and separate float32 rounding without fast math or FMA.

## Measure the effect

- A preallocated fused-versus-unfused Forge run isolates loop and intermediate traffic changes. Match data, target features, outer call boundary, thread count, and allocation policy. Include the initial example and a four-pass single-input chain.
- Record which temporaries exist, model logical reads and writes, and publish the formulas. Three full passes to one changes the modeled total from `24` to `8` bytes per element. Four changes it from `32` to `8`. Report actual cache or memory events only when the runner exposes a suitable event and its measurement scope is recorded. See the [traffic definition](../research.md#correctness-and-performance-boundaries).
- Preserve the Python-to-Python NumPy comparison as a separate view. For small arrays it includes dispatch and allocation overhead. Show sizes where the gain shrinks or reverses.

## Exit evidence

The CI evidence includes the terminal demo transcript, scalar and AVX2 plans, optimized LLVM IR, vectorization remarks, actual LLJIT objects and disassembly, plus raw samples and host metadata. Use `demo.txt` for the 30-second walkthrough and `avx2-fused.s` to inspect the packed body and scalar tail. The release stage defines the conditions for publishing résumé numbers. [Release criteria](08-release-and-handoff.md)

## Verified results

Ubuntu and Windows x64 each passed all 17 CTest entries, including native AVX2 guard-page checks. Both fused LLJIT objects show eight-lane `vaddps`, `vmulps`, and `vsqrtps`, one packed output store, and a scalar remainder. The M3 ran the scalar JIT tests and emitted Linux x86-64 AVX2 assembly for inspection. [CI run and artifacts for 344b5ff](https://github.com/iferdous/Forge/actions/runs/45102680498)

The three-operation demo drops from three loops and two temporary arrays to one loop and zero temporary arrays. Each materialized pass in these single-input workloads models a 4-byte read and a 4-byte write per element. That puts the demo at 24 → 8 modeled bytes per element (67% lower) and the four-operation chain at 32 → 8 (75% lower). The model counts array accesses at the loop level and excludes hardware effects such as cache-line fills and write allocation.

### Native AVX2

Each cell shows the unfused and fused warm wall-clock medians in microseconds with the same target, preallocated output and workspace, and five interleaved measurements. Ubuntu ran on a Xeon 6975P-C and Windows ran on a Xeon Platinum 8370C. The demo is `sqrt((x + 2) * 3)`. The four-operation chain is `((x + 2) * 3 + 4) * 5`.

| Float32 elements | Ubuntu demo | Windows demo | Ubuntu four ops | Windows four ops |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.011 → 0.007 (1.45×) | 0.017 → 0.010 (1.75×) | 0.014 → 0.007 (1.88×) | 0.026 → 0.011 (2.40×) |
| 1,024 | 0.288 → 0.201 (1.43×) | 0.413 → 0.227 (1.82×) | 0.203 → 0.067 (3.03×) | 0.235 → 0.103 (2.28×) |
| 65,536 | 23.25 → 12.83 (1.81×) | 25.81 → 14.31 (1.80×) | 23.36 → 5.27 (4.43×) | 24.78 → 5.82 (4.26×) |
| 1,048,576 | 860.16 → 290.00 (2.97×) | 1,147.05 → 286.55 (4.00×) | 1,146.64 → 288.43 (3.98×) | 1,875.95 → 304.36 (6.16×) |

The `x + 2` control has one loop in both modes and shows no consistent gain. The raw reports include all scalar and AVX2 results. Ratios use full-precision medians.

### Python and NumPy

The Python harness measures warm calls in a separate experiment. In the allocating API, the three-op demo took **0.599 µs in Forge vs 1.742 µs in NumPy** on Ubuntu at 1,024 elements (2.91×). Windows measured 1.090 µs vs 3.296 µs (3.03×). At 65,536 elements the same comparison narrowed to 2.10× and 2.04×. For a single preallocated add at that size on Windows, NumPy was slightly faster (7.86 µs vs 8.28 µs).

Both harnesses exclude compilation. Shared runner load can affect timing, especially for short calls. MSVC emitted the existing C4702 warnings in LLVM SDK headers. The raw reports, plans, disassembly and transcripts are archived on SSK under `.local/evidence/stage-05/baseline/`. [Exact benchmark commands](../../.github/workflows/ci.yml)
