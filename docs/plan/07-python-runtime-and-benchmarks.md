# 07 · Finish the Python runtime and benchmark suite

[Previous: reductions and matmul](06-reductions-and-matmul.md) · [Plan](../../README.md) · [Next: release and handoff](08-release-and-handoff.md)

**Outcome.** A reproducible Python API and a ten-kernel harness show where Forge wins, where it loses, and what first compilation costs.

## Runtime

1. Finish the pybind11 wrapper. Validate dtype, native byte order, rank, shape, layout, and alignment at the boundary. Borrow input arrays across synchronous execution and return a separately owned array. Check requirements explicitly with a `py::array` or equivalent no-conversion binding. pybind11's default `array_t` force-cast can hide copies. Start with the GIL held and measure the wrapper before changing thread behavior. [pybind11 NumPy bindings](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
2. Keep a compiled kernel alive through a Python-owned handle. Reuse the handle across calls, then add a small bounded in-process compilation cache if repeated compile requests merit it. Key on normalized operation graph, dtype, rank and specialized extents if any, target configuration, and fusion mode. Store only successful compilations. Ensure an evicted entry stays executable as long as a caller holds its handle. Test repeat calls, key isolation, eviction and error recovery.
3. Expose warm call, cold compile and first call separately. Cold compilation includes graph work, lowering, optimization, LLJIT materialization and lookup. Label fresh-process startup separately from cold compilation within an already initialized process. Cache-hit time carries its own label.

## Cases and size grid

| ID | Expression | Purpose |
| --- | --- | --- |
| E1 | `x + y` | One-op, two-input dispatch baseline |
| E2 | `x * 3` | One-op scalar literal baseline |
| E3 | `sqrt(x)` | Packed square root on nonnegative inputs |
| E4 | `x * y + z` | Two ops with three inputs and contraction prohibited |
| E5 | `sqrt((x + 2) * 3)` | Three-pass demo and fusion attribution |
| E6 | `((x + 1) * 2 + 3) * 4` | Four-pass logical traffic case |
| R1 | `sum(x)` | Ordered accumulation |
| R2 | `max(x)` | Comparison reduction |
| R3 | `mean(x)` | Accumulation plus final division |
| M1 | `A @ B` | Tiled matrix multiplication against NumPy BLAS |

Use pointwise and reduction lengths `32`, `1,000`, `1,024`, `65,536`, `1,048,576` and `8,388,608`. Add or label a larger streaming working set if the minimum working set at the last size still fits in the host's last-level cache. Include tail sizes in correctness and add a timed odd length if relevant. For matmul use square sizes `16, 64, 256, 512` and at least one rectangular edge-tile case, such as `31 × 63 × 17`. Seed inputs and keep the release timing inputs finite. Run special-value and cancellation fixtures separately.

## Timing and evidence protocol

- On a pinned commit and stable native x64 AVX2 host, record CPU model and flags, cache sizes, OS, compiler and dependency versions, Python environment, NumPy runtime and BLAS thread count. Control competing load and use a single thread for a matched comparison. Treat general hosted CI timings as smoke checks.
- Keep three comparison modes. **Python API** uses the same input arrays and includes one normal call and output allocation for each implementation. **Preallocated** supplies equivalent legal output and scratch buffers where appropriate. **Native attribution** compares Forge unfused versus fused and scalar versus SIMD with matching allocations and a common call boundary. Precompute and validate fixtures outside timed loops. Preserve a separate cold compilation group.
- Warm up both implementations and interleave their repetitions to reduce time-order bias. Start with at least 20 timed repetitions per small-case configuration. Calibrate batch lengths so timer overhead is a small fraction of each sample. Use process-level repeats for first-use timing. Report median and spread or p95, and label a percentile of batch means as such.
- Report every predefined size or explain an unsupported case or timeout. Publish raw JSON or CSV, a command to rerun, the artifact's commit, and assembly for the demo kernel. Calculate speedup with matching samples and units. Distinguish Python API speedup from SIMD/fusion attribution, modeled logical bytes, and hardware counters. Publish cold cost and the rough call-count break-even only if the warm gain is positive.

## Exit evidence

One reproducible table and chart covering the complete grid. A qualified x64 run supplies the numerical evidence. Preserve individual rows so a reader can audit crossover points, compare allocation policies, and verify the README summary. [Google Benchmark JSON and timing](https://google.github.io/benchmark/user_guide.html) · [NumPy runtime introspection](https://numpy.org/doc/stable/reference/generated/numpy.show_runtime.html)
