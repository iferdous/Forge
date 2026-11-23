# 05 · Fuse elementwise operations

[Previous: SIMD and baselines](04-simd-and-baselines.md) · [Plan](../../README.md) · [Next: reductions and matmul](06-reductions-and-matmul.md)

**Outcome.** Forge owns a compiler pass that combines a legal operation chain into one loop. The same graph can also run unfused, enabling a controlled comparison.

## Design and legality

1. Run fusion on the typed tensor IR. Start at an output and collect consecutive pure, elementwise nodes with compatible shape and dtype. Inline single-use producers into the consumer kernel while preserving operand ordering. Leave multi-use producers materialized initially. Treat reductions, matmul, and changes of iteration space as boundaries.
2. Give the kernel plan a stable textual form. Show intermediate allocations and loop count in the unfused plan, followed by the fused plan with one loop and no temporary arrays for eligible expressions. Keep a switch that disables fusion while preserving target, dtype, and floating-point settings. Allow both plans to enter through the same public ABI.
3. Lower the fused region to one LLVM loop with scalar SSA values for intermediate arithmetic. Use the same LLVM pipeline as the unfused version. Verify that the optimized body retains separable multiply and add where the source requires them and that the assembly has no unintended FMA or temporary-array stores for the demo kernel.
4. Test the decision pass and the results independently. Include chains of length one to four, repeated reads, shared producers, two-array operations, non-fusible boundaries, non-vector-width lengths, empty arrays, and special floats. Compare fused and unfused scalar and SIMD results with the reference. When a property check fails, record the seed and minimal expression.

## Measure the effect

- A preallocated fused-versus-unfused Forge run isolates loop and intermediate traffic changes. Match data, target features, outer call boundary, thread count, and allocation policy. Include the initial example and a four-pass single-input chain.
- Record which temporaries exist, model logical reads and writes, and publish the formulas. Three full passes to one changes the modeled total from `24` to `8` bytes per element. Four changes it from `32` to `8`. Report actual cache or memory events only when the runner exposes a suitable event and its measurement scope is recorded. See the [traffic definition](../research.md#correctness-and-performance-boundaries).
- Preserve the Python-to-Python NumPy comparison as a separate view. For small arrays it includes dispatch and allocation overhead. Show sizes where the gain shrinks or reverses.

## Exit evidence

A 30-second proof of concept showing the expression, two plans, packed instructions, and an actual correctness check. Keep raw benchmark runs and the target setup alongside it. At this point Forge can support a first résumé entry once the reproducible evidence and demo meet its [release criteria](08-release-and-handoff.md).
