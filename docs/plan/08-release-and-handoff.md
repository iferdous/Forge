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
