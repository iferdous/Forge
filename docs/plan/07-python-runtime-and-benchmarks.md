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

## Python API

Forge borrows C-contiguous, native-endian NumPy float32 arrays through a synchronous call. Inputs can be read-only and can start at a byte offset. Generated loads and stores require one-byte alignment, including on AVX2. Allocating calls return owned arrays. `run_into` requires a writable, matching output disjoint from all live inputs. The Python binding holds the GIL across execution and initializes a handle's reusable scratch space on its first call.

~~~python
import forge
import numpy as np

x = np.arange(1024, dtype=np.float32)
program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
target = "avx2" if forge.has_avx2() else "scalar"
kernel = program.jit(x, target=target, cache=True)
result = kernel(x)
kernel.run_into(result, x)
print(forge.cache_info())
~~~

`cache=True` opts into a 16-entry least-recently-used cache. The key records operation and operand order, exact float32 literal bits, argument order, dtype, ranks and extents, target, and fusion mode. Source locations and argument spellings are excluded. Each request returns its own Python handle and workspace while sharing immutable compiled code. Eviction and `forge.clear_cache()` release the cache's references. Live handles remain executable. Failed compilation requests never enter the cache. The default `program.jit(x)` compiles directly.

## Reproduce the suite

Use the Release configuration with the pinned LLVM SDK, `FORGE_BUILD_PYTHON=ON`, and `FORGE_BUILD_BENCHMARKS=ON`. The workflow records matching C++ compiler, LLVM, CMake, and Python build settings for Ubuntu and Windows. [SDK setup](01-toolchain-and-ci.md#reproduce-the-smoke-test) · [CI commands](../../.github/workflows/ci.yml)

On an AVX2 Linux machine with the configured Release build, run from the checkout with its Python virtual environment active. Use an idle, controlled host for the `controlled-x64` profile. Keep the same thread and CPU affinity policy across both measurements. The runner records the affinity and load it sees, while Google Benchmark records cache sizes.

~~~sh
python -m pip install --only-binary=:all: -r requirements.txt threadpoolctl==3.6.0 matplotlib==3.10.8
cmake --build --preset release --parallel
export PYTHONPATH="$PWD/.local/build/release/python"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
llvm_dir=$(cat .local/llvm-dir-linux-x64.txt)
python bench/suite.py --profile controlled-x64 --out .local/evidence/stage-07 \
  --native .local/build/release/bench/forge_native_suite \
  --objdump "$llvm_dir/../../../bin/llvm-objdump"
python bench/suite_report.py .local/evidence/stage-07
~~~

Windows uses the same commands with `forge_native_suite.exe` and the Python path and LLVM binary paths set as shown in CI. On Apple Silicon, use the local LLVM-enabled build, `--profile local-check --quick`, and its native benchmark binary. The quick grid is a harness check. It runs each kernel at one size with a scalar target when native AVX2 is unavailable.

## Read the artifacts

- `python.json` stores raw warm batch means, accuracy checks, host and dependency metadata, and separate compilation, first-call, cache-hit, and fresh-process samples.
- `native.json` stores C++ wall-clock measurements for scalar and AVX2 using preallocated output and workspace. Pointwise cases include both fusion modes. Reduction and matmul cases use their standalone plans. The native program checks the first, middle, and last outputs against its reference outside the timed loop and records logical array traffic for pointwise loops.
- `summary.csv` gives the median and p95 for every measured arm, speedup ratios, and the minimum input and output working-set size against the reported LLC. `summary.md` and `speedups.png` give the complete grid at a glance.
- `demo/` contains actual LLJIT objects, disassembly, optimized IR, vectorization remarks, and plans. The demo check verifies packed arithmetic, a scalar remainder, and one packed store in the fused code.

The full grid has 61 shape cases. It adds a 1,009-element E5 tail and a 67,108,864-element E5 case with 512 MiB of minimum input and output storage. The minimum excludes scratch arrays and other overhead. An I/O footprint that exceeds the reported LLC gives a useful streaming case, while a footprint that fits is recorded as such. These are byte counts derived from the buffers and plan.

Each warm arm has 20 interleaved samples by default. The Python high-percentile entries describe batch means. Native percentiles summarize reported iteration batches. NumPy's allocating reductions return a NumPy scalar, while Forge returns an owned zero-dimensional array. Both preallocated arms write into their own matching float32 output. Finite fixtures are checked against Forge's exact ordered reference and, for reductions and matmul, against the Stage 06 high-precision bounds.

The report measures cold parse and compilation in an initialized process, then the first allocating call. A cache hit measures a request on a parsed program and includes construction of a new execution handle. Fresh-process samples include interpreter launch and shutdown as a separate end-to-end field. The allocating break-even estimate uses the median compilation cost plus the first-call premium divided by a positive warm saving at 1,024 elements.

## Verified results

Ubuntu and Windows x64 passed all 20 CTest entries and all 61 size cases at `4ea42a3`. Each runner recorded 198 native configurations with 20 timing samples per configuration. Both used OpenBLAS 0.3.34.237.2 with one verified thread. Ubuntu ran on an EPYC 9V74 and Windows ran on an EPYC 7763. Both reported that the 512 MiB E5 input and output footprint exceeded their LLC. [CI and both complete artifact sets](https://github.com/iferdous/Forge/actions/runs/45118321641)

The table gives selected ratios from the full grid. Each Python ratio is NumPy time divided by Forge time for allocating calls. The fusion row compares native AVX2 with identical caller buffers.

| Workload and boundary | Ubuntu | Windows |
| --- | ---: | ---: |
| 32-element add, warm Python | 0.81× | 0.91× |
| 1,024-element E5, warm Python | 3.48× | 3.40× |
| 65,536-element E5, warm Python | 3.08× | 3.38× |
| 65,536-element E5, native AVX2 fusion | 1.94× | 2.36× |
| 67,108,864-element E5, native AVX2 fusion | 2.88× | 3.18× |
| 8,388,608-element sum, warm Python | 0.20× | 0.92× |
| 512-square matmul, warm Python | 0.20× | 0.18× |

For 1,024-element E5, initialized-process cold parse and compilation took **4.15 ms on Ubuntu and 4.80 ms on Windows**. The first allocating call took 11.36 µs and 8.45 µs. A cache hit on the parsed program took **5.98 µs and 8.70 µs**. The estimated allocating break-even was 2,529 and 1,820 calls. The median fresh-process end-to-end time was 100 ms and 157 ms.

The full artifacts are archived on SSK under `.local/evidence/stage-07/baseline/`. The Ubuntu CSV and Markdown summaries reproduced byte for byte from the raw inputs. The M3 passed the scalar and frontend tests and ran the quick grid. The measurements above are shared-runner smoke results. Publication-grade timing still needs a dedicated native x64 host and the `controlled-x64` run.
