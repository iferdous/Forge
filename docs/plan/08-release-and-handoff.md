# 08 · Harden, demonstrate, and release

[Previous: Python runtime and benchmarks](07-python-runtime-and-benchmarks.md) · [Plan](../../README.md)

**Outcome.** A reader can understand the compiler in 30 seconds, reproduce a claimed result from the tagged source, and inspect the tests that support it.

## First fusion release

The first release can follow Step 05 after the focused checks below. Complete Steps 06 and 07 for the full kernel suite and final release.

1. Make a clean checkout build and test on Ubuntu and Windows with the documented SDK paths. Show where native AVX2 executed. Keep a short compiler diagram and reproduce the featured result from a qualified x64 run with raw data, target and dependency metadata, and a named baseline.
2. Record a 30-second demo. Show `sqrt((x + 2) * 3)` and a small numeric result, the unfused and fused kernel plans, a YMM instruction and the scalar remainder, then the benchmark for that exact workload. Identify the input length and timed boundary in the caption.
3. Put a measured summary and the platform before the architecture section in the README. Link the command, evidence and commit. Phrase any initial résumé bullet around the implementation and its verified evidence. Leave additional operations out of the release claims until they ship.

## Full release

1. Add differential and property tests for valid graphs and negative tests for malformed or excessive input. Fuzz the parser and validator with a reproducible seed and bounded input size. Run a Linux sanitizer job for native C++ and document which generated-code accesses it covers. Use separate guard tests for generated-code bounds. Include Python binding lifetime, aliasing and cache tests. Capture actual coverage from a supported Linux job and keep the Ubuntu and Windows test matrix visible.
2. Reproduce the complete Step 07 suite from a fresh build. Inspect the latency grid, numerical report, selected ISA and disassembly. Include failures and regressions with enough context to interpret them. Publish sanitized evidence files with machine and dependency metadata and a clear build/runner reference.
3. Update the demo and README with shipped operations. Publish an architecture diagram, API example, reproduction commands, results table, numerical contract, supported platforms and CI status. Add a tagged release linked to the evidence. Settle the source license before advertising redistribution.
4. Draft two résumé bullets. One should explain the operation chain and operator fusion. The other should give one or two measured findings with workload, baseline, and scope. Record the raw result and exact source commit for every number used.

## Stage and interview handoff

Each implementation step records its commit, build commands, platform results, passing and skipped tests, and the relevant raw data and supported claims. Record open architecture decisions and their trade-offs. For failures, capture the command, error, and smallest reproducing test before advancing.

Use the real implementation to answer five questions.

- Where does Forge decide a producer can be fused, and why does a shared producer stay materialized?
- Which evidence shows SIMD execution, and how does a non-multiple input reach its scalar tail?
- Which floating-point rules keep the fused and unfused pointwise paths equivalent?
- How much of the measured win comes from one call, avoiding intermediate arrays, and SIMD?
- Where does NumPy win, and how many calls would a measured warm win need to amortize compilation?

## Release preview

Forge is licensed under MIT. The preview bundles the 30-second demo and caption, the 61-case benchmark grid from fresh Ubuntu and Windows x64 builds, equivalent raw samples with runner names and absolute paths removed, original JIT objects and disassembly, and Linux source coverage. A controlled benchmark run on dedicated native AVX2 hardware is still needed for release-grade timing claims. [Preview, verified commit, and evidence bundle](https://github.com/iferdous/Forge/releases/tag/v0.1.0-preview.1) · [License](../../LICENSE)

Rebuild from the release tag with the pinned SDK and Release presets. Use the full benchmark reproduction command and `--profile controlled-x64` on an idle native AVX2 host. For a shared runner, use `--profile ci-smoke`. `release/record_demo.py` reads the generated loop plans, object disassembly, and native E5 timing row and reruns the interactive expression for the recording. `release/package_evidence.py` requires the benchmark and sanitizer evidence to match the tagged commit and writes file digests into the bundle. [Build and benchmark reproduction](07-python-runtime-and-benchmarks.md#reproduce-the-suite) · [CI and instrumentation commands](../../.github/workflows/ci.yml)

## Hardening scope

- The seeded suite uses `0xF047E` for 48 generated pointwise expressions, 20 matmul–reduction compositions, 96 malformed expressions, and 32 invalid input fixture groups. It compares reference and JIT results with independent ordered NumPy operations. The x64 jobs exercise both scalar and AVX2 in both fusion modes. The frontend-only build runs the parser, shape, and reference portions.
- The instrumented Ubuntu job uses Clang ASan, UBSan, and source-based coverage for Forge's C++ graph, planning, lowering, and runtime code. The pinned LLVM SDK supplies the compiler libraries. Coverage describes the first-party C++ translation units under `src/`. ASan and UBSan instrument the C++ harness and runtime. Generated machine code executes without compiler-inserted sanitizer checks. [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) · [Source-based coverage](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html)
- Separate guard tests place protected pages beside JIT buffers. AVX2 pointwise tests guard the tail, including zero-length inputs. Reduction and tiled matmul tests put guards at both ends in separate calls and include a zero inner dimension. These guard pages detect accesses that cross the tested outer buffer edges. Numerical tests cover shapes, tile results, and signed-zero and NaN behavior. The Python suite covers borrowing, disjoint output, cache isolation and eviction, and handle lifetime.
- The first instrumented build failed at the LLVM smoke-test link with missing RTTI symbols. UBSan had emitted typeinfo references for the prebuilt RTTI-disabled SDK. The fix applies the SDK's RTTI mode through the shared LLVM CMake target to both the JIT library and the smoke test. [Original failure and reproduction command](https://github.com/iferdous/Forge/actions/runs/45120044458/job/137511488108)

## Interview notes

1. The kernel planner counts uses among live nodes. A single-use pointwise producer of the same shape joins its consumer's loop. A shared producer keeps its buffer so later consumers can reuse the result without recomputation. [Planner](../../src/kernel_plan.cpp)
2. The release disassembly shows `vaddps`, `vmulps`, and `vsqrtps` on YMM registers followed by one packed output store. The scalar cleanup uses XMM instructions for elements beyond the last full eight-element group. [SIMD evidence](04-simd-and-baselines.md)
3. Fusion keeps the operation order, rounds each float32 step, and leaves contraction disabled. Tests compare exact bits outside NaNs and check NaNs by classification. Ordered reductions and matmul follow the separate aggregate contract. [Numerical contract](06-reductions-and-matmul.md#expression-and-numerical-contract)
4. On the Ubuntu Stage 07 run at 65,536 elements, fusion alone gave 1.94× with both kernels using AVX2 and the same preallocated launch. The SIMD comparison on the fused plan gave 7.91× over scalar. The separate allocating Python-to-Python measurement gave 3.08× over NumPy. Each ratio has its own baseline. [Full benchmark grid](07-python-runtime-and-benchmarks.md#verified-results)
5. NumPy led the larger ordered sums and matrix multiplications in the Stage 07 grid. For 1,024-element E5, the allocating break-even was about 2,529 calls on Ubuntu and 1,820 on Windows from the measured cold and warm medians. [First-use timings and method](07-python-runtime-and-benchmarks.md#verified-results)

## Résumé drafts

- Built a C++20 and LLVM ORC tensor compiler with a Python frontend that fuses eligible float32 expressions into a single AVX2 loop and handles non-multiple lengths with a scalar tail.
- Measured a **1.94× speedup** for a 65,536-element float32 expression on Ubuntu x64 by fusing add, multiply, and square root, reducing three loops and two temporary arrays to one loop with zero temporary arrays.

The second bullet uses `sqrt((x + 2) * 3)` on the EPYC 9V74 runner at commit `4ea42a38e44c60b933f89202b173e92d066393df`. The warm preallocated native AVX2 medians in `summary.csv` were 22,183.23 ns unfused and 11,412.14 ns fused. These are shared CI measurements. Keep the workload, baseline, and scope when citing the number. [Raw Stage 07 Ubuntu evidence](https://github.com/iferdous/Forge/actions/runs/45118321641)
