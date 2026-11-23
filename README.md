# Forge

Forge is a C++20 and LLVM compiler for a small Python tensor language. It compiles float32 array expressions into native CPU code. Its main optimization, **operator fusion**, combines several operations into one loop so intermediate arrays never need to be written and read back.

[Performance at a glance](#performance-at-a-glance) · [Video walkthroughs](#video-walkthroughs) · [Complete benchmark tables](#complete-benchmark-tables) · [Compiler traces](#compiler-traces) · [Try it](#try-it)

## Performance at a glance

| Measured workload | Ubuntu x64 | Windows x64 |
| --- | ---: | ---: |
| 1,024-element add → multiply → square root, allocating Python call against NumPy | **3.21× faster** | **3.65× faster** |
| 65,536-element same chain, native AVX2 with and without fusion | **1.82× faster** · 23.40 → 12.89 µs | **2.40× faster** · 31.31 → 13.07 µs |
| 65,536-element same chain, fused native scalar versus fused AVX2 | **4.00× faster** | **7.95× faster** |
| 512 × 512 matrix multiply, allocating Python call | **NumPy 7.37× faster** | **NumPy 5.51× faster** |

![Warm native AVX2 timings for the fused and unfused three-op chain at every tested size on Ubuntu and Windows](showcase/assets/fusion-latency.png)

The three-op chain goes from **three loops and two temporary arrays to one loop and zero temporary arrays**. Both plans produce identical float32 output for the checked inputs. Under the same allocating Python boundary, the 32-element mean is Forge’s strongest result (9.55× Ubuntu, 8.88× Windows). Max is its weakest on both hosts (NumPy leads by 43.52× at 65,536 elements on Ubuntu and by 15.42× at 1,048,576 on Windows).

## Video walkthroughs

### Why fusion saves time

The problem is moving data between operations. Watch the same nine inputs go through a three-loop plan and then a fused plan. Follow the two scratch arrays as they disappear and check that the answers stay the same. The final screen shows where the difference turns into measured time and resident memory on larger inputs.

[Watch or download the fusion demo](showcase/assets/fusion.mp4)

### How the compiler handles an input that doesn't fit a vector

AVX2 can work on eight float32 values at once. Watch what happens as the input grows from 8 to 9 to 17 values. The highlighted groups match the machine code for the wide loop and the scalar cleanup path. That last path lets the same kernel handle lengths that aren't divisible by eight.

[Watch or download the lanes demo](showcase/assets/lanes.mp4)

### Where Forge wins, and why it sometimes loses

The demo starts with a sum where floating-point rounding changes the answer, then shows how Forge handles partial blocks in matrix multiplication. Watch for the crossover from an array expression Forge wins to a large matrix where NumPy’s tuned BLAS wins. The final scene shows the cache reusing machine code for the same input shape.

[Watch or download the tradeoffs demo](showcase/assets/tradeoffs.mp4)

Narration by Liam via [ElevenLabs](https://elevenlabs.io/). The videos use captured Forge outputs, compiled instructions, and measured timings. [Transcripts and reproduction](docs/showcase.md#regenerate-the-demos)

## Memory, startup, and coverage

| Metric | Result |
| --- | --- |
| Three-op chain array traffic | 24 → 8 bytes per float32 element (**66.7% lower**, modeled) |
| Four-op chain array traffic | 32 → 8 bytes per float32 element (**75% lower**, modeled) |
| Intermediate workspace at 8,388,608 elements | **64 MiB → 0** |
| Median first-call Linux resident memory increase at the same size | **64.008 MiB → 0.004 MiB**, three fresh processes per arm |
| Native C++ coverage | **94.56% lines** (869/919), **85.09% branches** (291/342) |
| Verification | **23 tests and 61 benchmark cases per x64 runner**, plus **20 Linux sanitizer test entries** |

The byte counts model logical buffer accesses per element. The resident-memory experiment measures pages held by the whole process after compilation and after its first call.

| First-use timing for the 1,024-element chain | Ubuntu x64 | Windows x64 |
| --- | ---: | ---: |
| Cold parse and compile in an initialized process | 4.28 ms | 4.82 ms |
| First allocating call | 11.56 µs | 8.95 µs |
| Cache hit on the parsed graph, including the new handle | 4.91 µs | 9.00 µs |
| Fresh process, including interpreter launch and exit | 82.80 ms | 165.80 ms |
| Estimated warm calls to earn back compilation against NumPy | 3,615 | 1,723 |

The 16-entry cache specializes code by graph, shape, and target. These counts come from the same native capture used in the demos.

| Recorded request | Hits | Misses | Entries |
| --- | ---: | ---: | ---: |
| Empty cache | 0 | 0 | 0 |
| Compile for length 4 | 0 | 1 | 1 |
| Repeat length 4 | 1 | 1 | 1 |
| Compile for length 9 | 1 | 2 | 2 |

<details>
<summary>Resident memory and startup charts</summary>

![Measured resident set before compilation, after compilation, and after the first call](showcase/assets/resident-memory.png)

![Cold compile, first call, cache hit, and fresh process latency](showcase/assets/first-use.png)

</details>

## Complete benchmark tables

Expand a workload, then a call boundary. Every case shows both runners, **median and p95 in microseconds**, and **speedup as baseline time ÷ contender time**. Above 1× favors Forge, fused AVX2, or AVX2 depending on the comparison.

| Comparison | Held fixed |
| --- | --- |
| Python, allocating | Both functions allocate the output. Compilation is excluded. |
| Python, preallocated | Both functions write into provided output. Compilation is excluded. |
| Native fusion | Both plans run in AVX2 with preallocated buffers. |
| Native SIMD | The same fused plan runs scalar and AVX2 with preallocated buffers. |

These are shared-CI timings from Ubuntu x64 on Intel Xeon 6975P-C and Windows x64 on AMD EPYC 7763. NumPy used one OpenBLAS thread on both. The grid covers **10 kernels and 61 size cases per runner**, with **20 interleaved warm samples per configuration**. Python p95 summarizes batch means. Native p95 summarizes iteration batches. Full precision and the 512 MiB streaming working set are documented in the downloadable grids.

<details>
<summary>Performance by operation and matrix size</summary>

![Allocating Python speedup by workload and size on each runner](showcase/assets/workload-atlas.png)

![Warm tiled matrix multiplication against single-threaded NumPy BLAS on both runners](showcase/assets/matmul-latency.png)

</details>

<!-- BEGIN GENERATED BENCHMARKS -->

<details>
<summary>E1 · Add two arrays</summary>

Expression `x + y`. Each time cell is median / p95 in µs.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3786 / 0.3824 | 0.3256 / 0.3280 | 0.86× | 0.7664 / 0.7872 | 0.6910 / 0.7267 | 0.90× |
| 1,000 | 0.4438 / 0.4539 | 0.4277 / 0.4319 | 0.96× | 0.9744 / 0.9942 | 0.9698 / 0.9824 | 1.00× |
| 1,024 | 0.4419 / 0.4454 | 0.4276 / 0.4331 | 0.97× | 0.9625 / 1.606 | 0.9550 / 1.608 | 0.99× |
| 65,536 | 8.262 / 8.324 | 12.54 / 12.77 | 1.52× | 14.96 / 15.78 | 14.89 / 15.61 | 1.00× |
| 1,048,576 | 603.1 / 611.5 | 449.4 / 456.6 | 0.75× | 870.4 / 902.1 | 868.9 / 880.0 | 1.00× |
| 8,388,608 | 5,329 / 5,675 | 5,456 / 6,022 | 1.02× | 7,525 / 7,742 | 7,451 / 7,690 | 0.99× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3327 / 0.3353 | 0.3024 / 0.3042 | 0.91× | 0.6110 / 0.6208 | 0.6274 / 0.6621 | 1.03× |
| 1,000 | 0.3747 / 0.3767 | 0.3886 / 0.3941 | 1.04× | 0.6726 / 0.6800 | 0.8221 / 0.8370 | 1.22× |
| 1,024 | 0.3804 / 0.3838 | 0.3893 / 0.3936 | 1.02× | 0.6761 / 1.209 | 0.8084 / 0.9386 | 1.20× |
| 65,536 | 6.356 / 6.389 | 12.80 / 12.91 | 2.01× | 13.97 / 14.75 | 14.19 / 14.92 | 1.02× |
| 1,048,576 | 429.8 / 431.2 | 428.2 / 429.6 | 1.00× | 232.3 / 237.4 | 230.7 / 246.8 | 0.99× |
| 8,388,608 | 3,568 / 3,606 | 3,642 / 3,738 | 1.02× | 3,023 / 3,331 | 2,959 / 3,166 | 0.98× |

</details>

<details>
<summary>Native AVX2, effect of fusion</summary>

| Size | Ubuntu Fused | Ubuntu Unfused | Ubuntu × | Windows Fused | Windows Unfused | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.009848 / 0.01040 | 0.009987 / 0.01032 | 1.01× | 0.01257 / 0.01322 | 0.01251 / 0.01389 | 1.00× |
| 1,000 | 0.05791 / 0.06950 | 0.05730 / 0.06624 | 0.99× | 0.05930 / 0.06192 | 0.05916 / 0.06293 | 1.00× |
| 1,024 | 0.05424 / 0.06533 | 0.05646 / 0.07017 | 1.04× | 0.06003 / 0.06243 | 0.06042 / 0.09991 | 1.01× |
| 65,536 | 7.858 / 8.588 | 8.146 / 8.800 | 1.04× | 16.02 / 18.87 | 16.01 / 17.35 | 1.00× |
| 1,048,576 | 426.6 / 432.0 | 423.8 / 432.0 | 0.99× | 267.4 / 302.3 | 272.0 / 301.9 | 1.02× |
| 8,388,608 | 3,441 / 3,710 | 3,395 / 3,578 | 0.99× | 3,236 / 3,555 | 3,145 / 3,410 | 0.97× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.009848 / 0.01040 | 0.02807 / 0.02847 | 2.85× | 0.01257 / 0.01322 | 0.03147 / 0.03223 | 2.50× |
| 1,000 | 0.05791 / 0.06950 | 0.2511 / 0.2535 | 4.34× | 0.05930 / 0.06192 | 0.6574 / 0.8219 | 11.08× |
| 1,024 | 0.05424 / 0.06533 | 0.2537 / 0.2576 | 4.68× | 0.06003 / 0.06243 | 0.6715 / 0.6800 | 11.19× |
| 65,536 | 7.858 / 8.588 | 19.57 / 20.47 | 2.49× | 16.02 / 18.87 | 42.68 / 50.41 | 2.66× |
| 1,048,576 | 426.6 / 432.0 | 427.7 / 437.7 | 1.00× | 267.4 / 302.3 | 664.6 / 681.8 | 2.49× |
| 8,388,608 | 3,441 / 3,710 | 4,628 / 5,143 | 1.34× | 3,236 / 3,555 | 5,481 / 5,814 | 1.69× |

</details>

</details>

<details>
<summary>E2 · Multiply by a scalar</summary>

Expression `x * 3`. Each time cell is median / p95 in µs.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3494 / 0.3544 | 0.4874 / 0.4885 | 1.39× | 0.7253 / 0.7595 | 1.121 / 1.298 | 1.55× |
| 1,000 | 0.4276 / 0.4322 | 0.5973 / 0.6008 | 1.40× | 0.9090 / 1.527 | 1.351 / 2.243 | 1.49× |
| 1,024 | 0.4291 / 0.4312 | 0.5964 / 0.6051 | 1.39× | 0.9165 / 0.9458 | 1.348 / 1.361 | 1.47× |
| 65,536 | 6.441 / 6.816 | 6.540 / 6.752 | 1.02× | 7.746 / 8.178 | 8.379 / 8.645 | 1.08× |
| 1,048,576 | 285.8 / 289.5 | 288.6 / 292.3 | 1.01× | 836.3 / 844.5 | 832.5 / 858.6 | 1.00× |
| 8,388,608 | 3,937 / 4,158 | 3,887 / 4,129 | 0.99× | 6,798 / 7,118 | 6,771 / 7,063 | 1.00× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3007 / 0.3039 | 0.4677 / 0.4727 | 1.56× | 0.5614 / 0.6015 | 1.069 / 1.147 | 1.91× |
| 1,000 | 0.3429 / 0.3442 | 0.5389 / 0.5429 | 1.57× | 0.6270 / 1.132 | 1.206 / 2.099 | 1.92× |
| 1,024 | 0.3423 / 0.3459 | 0.5395 / 0.5436 | 1.58× | 0.6036 / 0.6257 | 1.205 / 1.220 | 2.00× |
| 65,536 | 7.316 / 7.895 | 6.925 / 7.164 | 0.95× | 7.618 / 7.835 | 7.832 / 8.146 | 1.03× |
| 1,048,576 | 284.5 / 287.1 | 285.8 / 289.1 | 1.00× | 112.0 / 118.9 | 115.7 / 122.8 | 1.03× |
| 8,388,608 | 2,376 / 2,460 | 2,330 / 2,358 | 0.98× | 2,239 / 2,522 | 2,200 / 2,374 | 0.98× |

</details>

<details>
<summary>Native AVX2, effect of fusion</summary>

| Size | Ubuntu Fused | Ubuntu Unfused | Ubuntu × | Windows Fused | Windows Unfused | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008130 / 0.008353 | 0.008153 / 0.008414 | 1.00× | 0.01047 / 0.01063 | 0.01065 / 0.01174 | 1.02× |
| 1,000 | 0.04670 / 0.05776 | 0.04653 / 0.06396 | 1.00× | 0.05586 / 0.07734 | 0.05612 / 0.05723 | 1.00× |
| 1,024 | 0.04652 / 0.05912 | 0.04725 / 0.05992 | 1.02× | 0.05696 / 0.06744 | 0.05708 / 0.05952 | 1.00× |
| 65,536 | 6.090 / 7.132 | 5.689 / 7.117 | 0.93× | 6.221 / 7.664 | 6.057 / 6.631 | 0.97× |
| 1,048,576 | 284.4 / 288.6 | 284.3 / 286.1 | 1.00× | 186.2 / 216.5 | 182.0 / 213.2 | 0.98× |
| 8,388,608 | 2,280 / 2,349 | 2,259 / 2,321 | 0.99× | 2,222 / 2,541 | 2,274 / 2,395 | 1.02× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008130 / 0.008353 | 0.02708 / 0.02886 | 3.33× | 0.01047 / 0.01063 | 0.01993 / 0.02069 | 1.90× |
| 1,000 | 0.04670 / 0.05776 | 0.2492 / 0.2517 | 5.34× | 0.05586 / 0.07734 | 0.3359 / 0.3377 | 6.01× |
| 1,024 | 0.04652 / 0.05912 | 0.2537 / 0.2630 | 5.45× | 0.05696 / 0.06744 | 0.3443 / 0.3535 | 6.05× |
| 65,536 | 6.090 / 7.132 | 15.08 / 15.57 | 2.48× | 6.221 / 7.664 | 21.05 / 22.21 | 3.38× |
| 1,048,576 | 284.4 / 288.6 | 291.7 / 294.1 | 1.03× | 186.2 / 216.5 | 336.7 / 343.7 | 1.81× |
| 8,388,608 | 2,280 / 2,349 | 2,517 / 3,806 | 1.10× | 2,222 / 2,541 | 2,987 / 3,523 | 1.34× |

</details>

</details>

<details>
<summary>E3 · Square root</summary>

Expression `sqrt(x)`. Each time cell is median / p95 in µs.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3506 / 0.3538 | 0.3064 / 0.3113 | 0.87× | 0.7264 / 0.7370 | 0.6599 / 0.6668 | 0.91× |
| 1,000 | 0.5476 / 0.5525 | 0.5564 / 0.5631 | 1.02× | 1.041 / 1.062 | 1.222 / 1.236 | 1.17× |
| 1,024 | 0.5361 / 0.5422 | 0.5447 / 0.5486 | 1.02× | 1.018 / 1.087 | 1.229 / 1.271 | 1.21× |
| 65,536 | 13.13 / 13.21 | 13.18 / 13.22 | 1.00× | 14.58 / 14.71 | 27.34 / 27.52 | 1.87× |
| 1,048,576 | 286.7 / 290.0 | 288.7 / 290.8 | 1.01× | 854.7 / 871.0 | 952.3 / 993.4 | 1.11× |
| 8,388,608 | 3,976 / 4,394 | 4,007 / 4,431 | 1.01× | 6,949 / 7,014 | 7,555 / 7,651 | 1.09× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3004 / 0.3041 | 0.2573 / 0.2600 | 0.86× | 0.5565 / 0.5701 | 0.5367 / 0.5450 | 0.96× |
| 1,000 | 0.4663 / 0.4873 | 0.4741 / 0.4788 | 1.02× | 0.7287 / 0.7387 | 0.9898 / 1.006 | 1.36× |
| 1,024 | 0.4733 / 0.4779 | 0.4771 / 0.4797 | 1.01× | 0.7340 / 0.7782 | 1.011 / 1.017 | 1.38× |
| 65,536 | 13.05 / 13.11 | 13.09 / 13.20 | 1.00× | 13.99 / 14.24 | 26.88 / 27.33 | 1.92× |
| 1,048,576 | 284.6 / 288.7 | 285.9 / 287.2 | 1.00× | 209.1 / 218.8 | 413.4 / 417.4 | 1.98× |
| 8,388,608 | 2,436 / 2,572 | 2,373 / 2,486 | 0.97× | 2,203 / 2,329 | 3,365 / 3,402 | 1.53× |

</details>

<details>
<summary>Native AVX2, effect of fusion</summary>

| Size | Ubuntu Fused | Ubuntu Unfused | Ubuntu × | Windows Fused | Windows Unfused | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008099 / 0.008318 | 0.008111 / 0.008495 | 1.00× | 0.01059 / 0.01103 | 0.01057 / 0.01073 | 1.00× |
| 1,000 | 0.1966 / 0.1995 | 0.1960 / 0.1982 | 1.00× | 0.1986 / 0.2102 | 0.1981 / 0.1987 | 1.00× |
| 1,024 | 0.2019 / 0.2041 | 0.2024 / 0.2042 | 1.00× | 0.2029 / 0.2040 | 0.2032 / 0.2050 | 1.00× |
| 65,536 | 12.94 / 13.03 | 12.91 / 13.56 | 1.00× | 13.01 / 13.27 | 12.99 / 13.27 | 1.00× |
| 1,048,576 | 286.8 / 291.3 | 284.7 / 289.6 | 0.99× | 208.9 / 217.0 | 208.2 / 221.0 | 1.00× |
| 8,388,608 | 2,265 / 2,334 | 2,272 / 2,347 | 1.00× | 2,309 / 2,454 | 2,126 / 2,371 | 0.92× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008099 / 0.008318 | 0.02648 / 0.02737 | 3.27× | 0.01059 / 0.01103 | 0.05099 / 0.05290 | 4.81× |
| 1,000 | 0.1966 / 0.1995 | 0.7819 / 0.7998 | 3.98× | 0.1986 / 0.2102 | 1.592 / 1.643 | 8.01× |
| 1,024 | 0.2019 / 0.2041 | 0.8056 / 0.8125 | 3.99× | 0.2029 / 0.2040 | 1.605 / 1.654 | 7.91× |
| 65,536 | 12.94 / 13.03 | 51.41 / 52.02 | 3.97× | 13.01 / 13.27 | 103.4 / 106.8 | 7.95× |
| 1,048,576 | 286.8 / 291.3 | 833.3 / 839.1 | 2.91× | 208.9 / 217.0 | 1,647 / 1,675 | 7.88× |
| 8,388,608 | 2,265 / 2,334 | 6,767 / 7,539 | 2.99× | 2,309 / 2,454 | 13,143 / 13,238 | 5.69× |

</details>

</details>

<details>
<summary>E4 · Multiply and add</summary>

Expression `x * y + z`. Each time cell is median / p95 in µs.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.4066 / 0.4105 | 0.6574 / 0.6638 | 1.62× | 0.8094 / 0.8179 | 1.389 / 1.439 | 1.72× |
| 1,000 | 0.4967 / 0.4994 | 0.8717 / 0.8783 | 1.75× | 1.107 / 1.136 | 1.950 / 2.029 | 1.76× |
| 1,024 | 0.5081 / 0.5150 | 0.8995 / 0.9058 | 1.77× | 1.073 / 1.098 | 1.969 / 2.071 | 1.84× |
| 65,536 | 12.03 / 12.20 | 32.12 / 32.38 | 2.67× | 20.87 / 23.45 | 25.88 / 30.74 | 1.24× |
| 1,048,576 | 738.7 / 745.4 | 938.8 / 950.1 | 1.27× | 886.1 / 938.6 | 1,774 / 1,915 | 2.00× |
| 8,388,608 | 7,519 / 8,113 | 13,568 / 14,313 | 1.80× | 8,434 / 8,651 | 15,122 / 15,337 | 1.79× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3628 / 0.3645 | 0.5824 / 0.5874 | 1.61× | 0.6581 / 0.6642 | 1.240 / 1.277 | 1.88× |
| 1,000 | 0.4303 / 0.4398 | 0.7678 / 0.7770 | 1.78× | 0.7839 / 0.7899 | 1.634 / 1.673 | 2.09× |
| 1,024 | 0.4321 / 0.4351 | 0.7777 / 0.7809 | 1.80× | 0.7617 / 0.7733 | 1.631 / 1.666 | 2.14× |
| 65,536 | 10.59 / 10.75 | 21.93 / 22.01 | 2.07× | 15.02 / 20.23 | 23.58 / 29.37 | 1.57× |
| 1,048,576 | 775.9 / 783.5 | 897.1 / 908.8 | 1.16× | 225.0 / 234.9 | 380.7 / 418.2 | 1.69× |
| 8,388,608 | 5,286 / 5,527 | 8,578 / 8,715 | 1.62× | 4,025 / 4,410 | 6,399 / 6,653 | 1.59× |

</details>

<details>
<summary>Native AVX2, effect of fusion</summary>

| Size | Ubuntu Fused | Ubuntu Unfused | Ubuntu × | Windows Fused | Windows Unfused | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.01100 / 0.01163 | 0.01566 / 0.01609 | 1.42× | 0.01565 / 0.01781 | 0.01838 / 0.03464 | 1.17× |
| 1,000 | 0.07648 / 0.08534 | 0.1067 / 0.1311 | 1.40× | 0.09897 / 0.1015 | 0.1109 / 0.1144 | 1.12× |
| 1,024 | 0.07610 / 0.08017 | 0.1032 / 0.1255 | 1.36× | 0.09711 / 0.1023 | 0.1138 / 0.1193 | 1.17× |
| 65,536 | 9.416 / 9.865 | 15.35 / 17.63 | 1.63× | 23.61 / 27.89 | 33.09 / 39.48 | 1.40× |
| 1,048,576 | 569.7 / 575.2 | 851.4 / 856.4 | 1.49× | 364.3 / 409.3 | 484.6 / 559.3 | 1.33× |
| 8,388,608 | 4,864 / 5,422 | 8,176 / 9,144 | 1.68× | 4,325 / 4,502 | 6,615 / 6,930 | 1.53× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.01100 / 0.01163 | 0.03157 / 0.03274 | 2.87× | 0.01565 / 0.01781 | 0.02905 / 0.03044 | 1.86× |
| 1,000 | 0.07648 / 0.08534 | 0.3236 / 0.3294 | 4.23× | 0.09897 / 0.1015 | 0.4992 / 0.5091 | 5.04× |
| 1,024 | 0.07610 / 0.08017 | 0.3315 / 0.3370 | 4.36× | 0.09711 / 0.1023 | 0.5118 / 0.5172 | 5.27× |
| 65,536 | 9.416 / 9.865 | 23.91 / 26.12 | 2.54× | 23.61 / 27.89 | 33.14 / 34.61 | 1.40× |
| 1,048,576 | 569.7 / 575.2 | 581.9 / 592.0 | 1.02× | 364.3 / 409.3 | 525.2 / 540.4 | 1.44× |
| 8,388,608 | 4,864 / 5,422 | 6,523 / 6,958 | 1.34× | 4,325 / 4,502 | 4,643 / 5,013 | 1.07× |

</details>

</details>

<details>
<summary>E5 · Three-op chain</summary>

Expression `sqrt((x + 2) * 3)`. Each time cell is median / p95 in µs.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3498 / 0.3535 | 1.249 / 1.257 | 3.57× | 0.7281 / 0.7375 | 2.779 / 2.829 | 3.82× |
| 1,000 | 0.5526 / 0.5627 | 1.717 / 1.742 | 3.11× | 1.068 / 1.093 | 3.755 / 3.804 | 3.52× |
| 1,009 | 0.5556 / 0.5611 | 1.745 / 1.760 | 3.14× | 1.063 / 1.089 | 3.847 / 3.914 | 3.62× |
| 1,024 | 0.5359 / 0.5380 | 1.722 / 1.732 | 3.21× | 1.059 / 1.096 | 3.864 / 4.052 | 3.65× |
| 65,536 | 13.14 / 13.20 | 27.62 / 27.90 | 2.10× | 14.69 / 14.88 | 47.52 / 51.63 | 3.24× |
| 1,048,576 | 285.5 / 289.9 | 866.4 / 871.8 | 3.04× | 835.0 / 860.5 | 2,653 / 2,728 | 3.18× |
| 8,388,608 | 4,400 / 5,660 | 14,101 / 16,184 | 3.20× | 6,944 / 8,084 | 21,492 / 22,115 | 3.10× |
| 67,108,864 | 45,424 / 47,155 | 137,677 / 140,748 | 3.03× | 55,356 / 56,548 | 173,339 / 176,966 | 3.13× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3002 / 0.3017 | 1.154 / 1.162 | 3.84× | 0.5576 / 0.5732 | 2.524 / 2.562 | 4.53× |
| 1,000 | 0.4687 / 0.4751 | 1.508 / 1.520 | 3.22× | 0.7516 / 0.7730 | 3.317 / 3.339 | 4.41× |
| 1,009 | 0.4706 / 0.4741 | 1.516 / 1.534 | 3.22× | 0.7751 / 0.7817 | 3.314 / 3.360 | 4.28× |
| 1,024 | 0.4717 / 0.4741 | 1.518 / 1.529 | 3.22× | 0.7505 / 0.7562 | 3.327 / 3.365 | 4.43× |
| 65,536 | 14.03 / 14.10 | 25.19 / 25.36 | 1.80× | 14.27 / 14.44 | 42.38 / 44.46 | 2.97× |
| 1,048,576 | 285.5 / 288.1 | 789.3 / 798.4 | 2.76× | 210.4 / 215.6 | 686.8 / 723.5 | 3.26× |
| 8,388,608 | 3,060 / 3,200 | 7,428 / 7,891 | 2.43× | 2,237 / 2,355 | 7,060 / 7,261 | 3.16× |
| 67,108,864 | 27,551 / 27,868 | 67,656 / 67,873 | 2.46× | 18,830 / 19,839 | 58,598 / 61,606 | 3.11× |

</details>

<details>
<summary>Native AVX2, effect of fusion</summary>

| Size | Ubuntu Fused | Ubuntu Unfused | Ubuntu × | Windows Fused | Windows Unfused | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008078 / 0.008372 | 0.01533 / 0.01597 | 1.90× | 0.01247 / 0.01256 | 0.01984 / 0.02393 | 1.59× |
| 1,000 | 0.1961 / 0.1983 | 0.2804 / 0.2906 | 1.43× | 0.2018 / 0.2058 | 0.3376 / 0.4639 | 1.67× |
| 1,009 | 0.1988 / 0.2025 | 0.2876 / 0.2954 | 1.45× | 0.2038 / 0.2101 | 0.3428 / 0.3445 | 1.68× |
| 1,024 | 0.2005 / 0.2033 | 0.2845 / 0.3002 | 1.42× | 0.2062 / 0.2290 | 0.3434 / 0.3463 | 1.67× |
| 65,536 | 12.89 / 12.99 | 23.40 / 25.89 | 1.82× | 13.07 / 13.31 | 31.31 / 36.14 | 2.40× |
| 1,048,576 | 287.8 / 290.8 | 854.6 / 859.6 | 2.97× | 211.0 / 219.3 | 558.4 / 580.8 | 2.65× |
| 8,388,608 | 2,292 / 2,553 | 6,921 / 7,354 | 3.02× | 2,172 / 2,433 | 6,754 / 7,137 | 3.11× |
| 67,108,864 | 27,700 / 28,723 | 80,190 / 81,295 | 2.89× | 18,480 / 19,582 | 59,144 / 61,064 | 3.20× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008078 / 0.008372 | 0.02515 / 0.02543 | 3.11× | 0.01247 / 0.01256 | 0.05071 / 0.05129 | 4.07× |
| 1,000 | 0.1961 / 0.1983 | 0.7847 / 0.7903 | 4.00× | 0.2018 / 0.2058 | 1.588 / 1.618 | 7.87× |
| 1,009 | 0.1988 / 0.2025 | 0.7979 / 0.8041 | 4.01× | 0.2038 / 0.2101 | 1.595 / 1.617 | 7.83× |
| 1,024 | 0.2005 / 0.2033 | 0.8023 / 0.8143 | 4.00× | 0.2062 / 0.2290 | 1.622 / 1.649 | 7.87× |
| 65,536 | 12.89 / 12.99 | 51.60 / 52.21 | 4.00× | 13.07 / 13.31 | 103.9 / 104.9 | 7.95× |
| 1,048,576 | 287.8 / 290.8 | 834.2 / 844.3 | 2.90× | 211.0 / 219.3 | 1,640 / 1,648 | 7.77× |
| 8,388,608 | 2,292 / 2,553 | 6,861 / 8,722 | 2.99× | 2,172 / 2,433 | 13,167 / 13,217 | 6.06× |
| 67,108,864 | 27,700 / 28,723 | 64,314 / 65,373 | 2.32× | 18,480 / 19,582 | 105,402 / 106,203 | 5.70× |

</details>

</details>

<details>
<summary>E6 · Four-op chain</summary>

Expression `((x + 1) * 2 + 3) * 4`. Each time cell is median / p95 in µs.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3527 / 0.3570 | 1.967 / 1.982 | 5.58× | 0.7340 / 0.7465 | 4.321 / 4.809 | 5.89× |
| 1,000 | 0.4544 / 0.4571 | 2.487 / 2.534 | 5.47× | 0.9444 / 0.9551 | 5.250 / 6.099 | 5.56× |
| 1,024 | 0.4383 / 0.4542 | 2.400 / 2.459 | 5.48× | 0.9683 / 0.9911 | 5.242 / 5.302 | 5.41× |
| 65,536 | 8.356 / 8.439 | 30.43 / 30.66 | 3.64× | 9.435 / 9.759 | 39.84 / 42.23 | 4.22× |
| 1,048,576 | 704.6 / 778.6 | 1,373 / 1,519 | 1.95× | 870.0 / 919.5 | 3,533 / 3,866 | 4.06× |
| 8,388,608 | 4,612 / 5,492 | 18,504 / 20,153 | 4.01× | 7,045 / 7,168 | 28,689 / 29,157 | 4.07× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3043 / 0.3124 | 1.818 / 1.839 | 5.98× | 0.5647 / 0.5716 | 3.993 / 4.186 | 7.07× |
| 1,000 | 0.3679 / 0.3735 | 2.129 / 2.148 | 5.79× | 0.6491 / 0.6565 | 4.675 / 5.504 | 7.20× |
| 1,024 | 0.3627 / 0.3668 | 2.151 / 2.171 | 5.93× | 0.6570 / 0.6746 | 4.695 / 4.798 | 7.15× |
| 65,536 | 7.755 / 7.983 | 23.58 / 23.73 | 3.04× | 7.720 / 9.876 | 26.01 / 27.41 | 3.37× |
| 1,048,576 | 694.9 / 777.5 | 949.0 / 1,039 | 1.37× | 124.8 / 188.4 | 462.3 / 534.4 | 3.70× |
| 8,388,608 | 3,174 / 3,258 | 8,809 / 9,017 | 2.77× | 2,388 / 2,491 | 7,682 / 7,877 | 3.22× |

</details>

<details>
<summary>Native AVX2, effect of fusion</summary>

| Size | Ubuntu Fused | Ubuntu Unfused | Ubuntu × | Windows Fused | Windows Unfused | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008198 / 0.01577 | 0.01928 / 0.01970 | 2.35× | 0.01346 / 0.01369 | 0.02403 / 0.02503 | 1.78× |
| 1,000 | 0.06581 / 0.06907 | 0.2103 / 0.2304 | 3.20× | 0.09697 / 0.09817 | 0.2444 / 0.2518 | 2.52× |
| 1,024 | 0.06633 / 0.07098 | 0.1881 / 0.2310 | 2.84× | 0.09868 / 0.09990 | 0.2497 / 0.3005 | 2.53× |
| 65,536 | 5.626 / 8.303 | 24.22 / 28.21 | 4.31× | 6.267 / 7.113 | 37.07 / 41.60 | 5.92× |
| 1,048,576 | 286.1 / 289.0 | 1,132 / 1,142 | 3.96× | 169.2 / 197.7 | 654.7 / 765.5 | 3.87× |
| 8,388,608 | 2,350 / 2,659 | 9,791 / 10,387 | 4.17× | 2,018 / 2,286 | 9,050 / 9,542 | 4.48× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.008198 / 0.01577 | 0.02151 / 0.02184 | 2.62× | 0.01346 / 0.01369 | 0.03137 / 0.03541 | 2.33× |
| 1,000 | 0.06581 / 0.06907 | 0.3720 / 0.3752 | 5.65× | 0.09697 / 0.09817 | 0.6562 / 0.6744 | 6.77× |
| 1,024 | 0.06633 / 0.07098 | 0.3785 / 0.3833 | 5.71× | 0.09868 / 0.09990 | 0.6493 / 0.7231 | 6.58× |
| 65,536 | 5.626 / 8.303 | 23.43 / 24.22 | 4.16× | 6.267 / 7.113 | 42.02 / 45.32 | 6.71× |
| 1,048,576 | 286.1 / 289.0 | 426.9 / 436.8 | 1.49× | 169.2 / 197.7 | 670.7 / 729.1 | 3.97× |
| 8,388,608 | 2,350 / 2,659 | 4,189 / 7,538 | 1.78× | 2,018 / 2,286 | 5,450 / 6,000 | 2.70× |

</details>

</details>

<details>
<summary>R1 · Ordered sum</summary>

Expression `sum(x)`. Each time cell is median / p95 in µs. This standalone kernel has a single loop plan.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3359 / 0.3410 | 1.203 / 1.225 | 3.58× | 0.7127 / 1.118 | 2.077 / 2.526 | 2.91× |
| 1,000 | 0.8163 / 0.8242 | 1.403 / 1.417 | 1.72× | 1.627 / 1.644 | 2.401 / 2.475 | 1.48× |
| 1,024 | 0.8251 / 0.8291 | 1.403 / 1.429 | 1.70× | 1.647 / 1.657 | 2.400 / 2.436 | 1.46× |
| 65,536 | 34.42 / 34.73 | 11.86 / 12.02 | 0.34× | 62.50 / 63.58 | 19.19 / 20.60 | 0.31× |
| 1,048,576 | 552.1 / 561.1 | 178.2 / 181.0 | 0.32× | 1,003 / 1,011 | 356.8 / 381.6 | 0.36× |
| 8,388,608 | 4,386 / 4,399 | 1,401 / 1,418 | 0.32× | 7,934 / 8,179 | 8,649 / 9,168 | 1.09× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3087 / 0.3132 | 1.413 / 1.454 | 4.58× | 0.5637 / 1.039 | 2.486 / 4.245 | 4.41× |
| 1,000 | 0.7947 / 0.8021 | 1.612 / 1.645 | 2.03× | 1.471 / 1.489 | 2.826 / 2.874 | 1.92× |
| 1,024 | 0.8086 / 0.8191 | 1.621 / 1.672 | 2.00× | 1.487 / 1.529 | 2.830 / 2.854 | 1.90× |
| 65,536 | 34.32 / 34.71 | 12.16 / 12.26 | 0.35× | 62.33 / 63.49 | 20.50 / 21.65 | 0.33× |
| 1,048,576 | 550.8 / 558.4 | 179.5 / 182.1 | 0.33× | 989.9 / 1,002 | 355.7 / 368.5 | 0.36× |
| 8,388,608 | 4,394 / 4,401 | 1,391 / 1,415 | 0.32× | 7,950 / 8,052 | 8,512 / 8,937 | 1.07× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.01515 / 0.01772 | 0.01654 / 0.01861 | 1.09× | 0.02218 / 0.02280 | 0.02200 / 0.02278 | 0.99× |
| 1,000 | 0.4943 / 0.4994 | 0.4963 / 0.5003 | 1.00× | 0.9307 / 0.9413 | 0.9323 / 0.9361 | 1.00× |
| 1,024 | 0.5069 / 0.5110 | 0.5077 / 0.5130 | 1.00× | 0.9528 / 0.9663 | 0.9555 / 0.9858 | 1.00× |
| 65,536 | 34.38 / 34.74 | 34.26 / 34.76 | 1.00× | 62.27 / 63.09 | 62.28 / 63.38 | 1.00× |
| 1,048,576 | 549.0 / 558.6 | 549.6 / 554.0 | 1.00× | 990.5 / 1,007 | 980.7 / 1,019 | 0.99× |
| 8,388,608 | 4,423 / 4,488 | 4,406 / 4,521 | 1.00× | 7,911 / 7,943 | 7,927 / 8,158 | 1.00× |

</details>

</details>

<details>
<summary>R2 · NaN-aware max</summary>

Expression `max(x)`. Each time cell is median / p95 in µs. This standalone kernel has a single loop plan.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3717 / 0.3787 | 1.182 / 1.194 | 3.18× | 0.7146 / 0.7253 | 1.988 / 2.055 | 2.78× |
| 1,000 | 2.839 / 2.875 | 1.244 / 1.283 | 0.44× | 1.692 / 1.711 | 2.178 / 2.242 | 1.29× |
| 1,024 | 2.897 / 2.935 | 1.269 / 1.278 | 0.44× | 1.703 / 1.816 | 2.159 / 2.193 | 1.27× |
| 65,536 | 165.5 / 166.7 | 3.802 / 3.853 | 0.02× | 64.16 / 65.38 | 5.552 / 5.852 | 0.09× |
| 1,048,576 | 2,656 / 2,664 | 145.7 / 147.3 | 0.05× | 1,013 / 1,022 | 65.68 / 74.72 | 0.06× |
| 8,388,608 | 21,153 / 21,236 | 1,146 / 1,157 | 0.05× | 8,176 / 8,221 | 1,428 / 1,514 | 0.17× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3513 / 0.3543 | 1.469 / 1.502 | 4.18× | 0.5698 / 0.5873 | 2.573 / 2.606 | 4.52× |
| 1,000 | 2.816 / 2.832 | 1.530 / 1.544 | 0.54× | 1.541 / 1.562 | 2.736 / 2.802 | 1.78× |
| 1,024 | 2.879 / 2.906 | 1.550 / 1.581 | 0.54× | 1.552 / 1.558 | 2.700 / 2.785 | 1.74× |
| 65,536 | 164.9 / 166.6 | 4.082 / 4.123 | 0.02× | 64.17 / 65.82 | 6.350 / 6.588 | 0.10× |
| 1,048,576 | 2,654 / 2,662 | 146.6 / 149.4 | 0.06× | 1,006 / 1,039 | 68.50 / 85.78 | 0.07× |
| 8,388,608 | 21,150 / 21,214 | 1,143 / 1,160 | 0.05× | 8,173 / 8,249 | 1,408 / 1,581 | 0.17× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.06094 / 0.06158 | 0.05942 / 0.06057 | 0.98× | 0.02866 / 0.02888 | 0.04593 / 0.04718 | 1.60× |
| 1,000 | 2.520 / 2.544 | 2.445 / 2.471 | 0.97× | 0.9796 / 1.006 | 1.528 / 1.536 | 1.56× |
| 1,024 | 2.571 / 2.607 | 2.489 / 2.529 | 0.97× | 1.012 / 1.020 | 1.563 / 1.576 | 1.54× |
| 65,536 | 165.7 / 166.8 | 160.4 / 163.1 | 0.97× | 64.00 / 67.63 | 100.1 / 100.4 | 1.56× |
| 1,048,576 | 2,650 / 2,661 | 2,573 / 2,579 | 0.97× | 1,012 / 1,047 | 1,584 / 1,592 | 1.57× |
| 8,388,608 | 21,162 / 21,340 | 20,489 / 20,779 | 0.97× | 8,151 / 8,210 | 12,664 / 13,042 | 1.55× |

</details>

</details>

<details>
<summary>R3 · Ordered mean</summary>

Expression `mean(x)`. Each time cell is median / p95 in µs. This standalone kernel has a single loop plan.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3347 / 0.3367 | 3.197 / 3.268 | 9.55× | 0.7259 / 0.7364 | 6.443 / 6.803 | 8.88× |
| 1,000 | 0.8179 / 0.8310 | 3.462 / 3.511 | 4.23× | 1.628 / 1.642 | 6.924 / 7.217 | 4.25× |
| 1,024 | 0.8304 / 0.8631 | 3.512 / 3.628 | 4.23× | 1.641 / 1.695 | 6.960 / 7.180 | 4.24× |
| 65,536 | 34.43 / 34.63 | 14.00 / 14.15 | 0.41× | 62.60 / 63.19 | 25.42 / 28.30 | 0.41× |
| 1,048,576 | 550.9 / 562.6 | 183.7 / 186.6 | 0.33× | 985.7 / 999.9 | 354.9 / 363.9 | 0.36× |
| 8,388,608 | 4,389 / 4,438 | 1,417 / 1,443 | 0.32× | 7,929 / 7,977 | 8,310 / 8,793 | 1.05× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.3096 / 0.3121 | 3.241 / 3.264 | 10.47× | 0.5745 / 0.5873 | 6.623 / 7.673 | 11.53× |
| 1,000 | 0.7950 / 0.8132 | 3.533 / 3.614 | 4.44× | 1.473 / 1.513 | 7.052 / 7.226 | 4.79× |
| 1,024 | 0.8082 / 0.8261 | 3.549 / 3.695 | 4.39× | 1.492 / 1.535 | 7.058 / 7.139 | 4.73× |
| 65,536 | 34.35 / 34.67 | 14.08 / 14.25 | 0.41× | 62.26 / 62.79 | 25.87 / 27.11 | 0.42× |
| 1,048,576 | 550.4 / 556.9 | 184.8 / 188.4 | 0.34× | 987.0 / 995.3 | 354.9 / 364.8 | 0.36× |
| 8,388,608 | 4,387 / 4,417 | 1,407 / 1,451 | 0.32× | 7,936 / 8,171 | 8,309 / 8,985 | 1.05× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.01568 / 0.01775 | 0.01613 / 0.02048 | 1.03× | 0.02295 / 0.02307 | 0.02295 / 0.02425 | 1.00× |
| 1,000 | 0.4932 / 0.5009 | 0.4960 / 0.5030 | 1.01× | 0.9309 / 0.9334 | 0.9276 / 0.9739 | 1.00× |
| 1,024 | 0.5065 / 0.5135 | 0.5078 / 0.5239 | 1.00× | 0.9536 / 0.9617 | 0.9555 / 0.9614 | 1.00× |
| 65,536 | 34.42 / 34.91 | 34.29 / 34.90 | 1.00× | 62.18 / 62.57 | 62.23 / 63.34 | 1.00× |
| 1,048,576 | 550.2 / 553.8 | 547.1 / 555.5 | 0.99× | 985.4 / 997.6 | 985.2 / 997.6 | 1.00× |
| 8,388,608 | 4,399 / 4,534 | 4,394 / 4,503 | 1.00× | 7,927 / 7,968 | 7,934 / 7,982 | 1.00× |

</details>

</details>

<details>
<summary>M1 · Tiled matrix multiplication</summary>

Expression `A @ B`. Each time cell is median / p95 in µs. This standalone kernel has a single loop plan.

<details>
<summary>Python call with output allocation</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 16 × 16 × 16 | 0.7496 / 0.7553 | 0.7356 / 0.7465 | 0.98× | 1.539 / 1.560 | 2.156 / 2.190 | 1.40× |
| 31 × 63 × 17 | 4.639 / 4.684 | 1.374 / 1.385 | 0.30× | 8.237 / 10.09 | 3.979 / 6.704 | 0.48× |
| 64 × 64 × 64 | 25.09 / 25.29 | 3.645 / 3.702 | 0.15× | 38.01 / 38.47 | 10.56 / 10.82 | 0.28× |
| 256 × 256 × 256 | 1,171 / 1,183 | 184.7 / 187.1 | 0.16× | 1,985 / 2,026 | 393.6 / 399.9 | 0.20× |
| 512 × 512 × 512 | 10,602 / 10,659 | 1,438 / 1,456 | 0.14× | 17,839 / 18,178 | 3,238 / 3,331 | 0.18× |

</details>

<details>
<summary>Python call with preallocated output</summary>

| Size | Ubuntu Forge | Ubuntu NumPy | Ubuntu × | Windows Forge | Windows NumPy | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 16 × 16 × 16 | 0.6873 / 0.6930 | 0.6935 / 0.7028 | 1.01× | 1.356 / 1.386 | 1.987 / 2.022 | 1.47× |
| 31 × 63 × 17 | 3.898 / 3.913 | 1.299 / 1.319 | 0.33× | 7.687 / 9.862 | 3.812 / 6.467 | 0.50× |
| 64 × 64 × 64 | 19.18 / 19.73 | 3.573 / 3.621 | 0.19× | 37.16 / 38.86 | 10.37 / 10.89 | 0.28× |
| 256 × 256 × 256 | 1,615 / 1,649 | 184.1 / 186.9 | 0.11× | 2,162 / 2,189 | 392.6 / 409.3 | 0.18× |
| 512 × 512 × 512 | 13,406 / 13,486 | 1,436 / 1,460 | 0.11× | 17,421 / 17,939 | 2,977 / 3,088 | 0.17× |

</details>

<details>
<summary>Native fused plan, effect of SIMD</summary>

| Size | Ubuntu AVX2 | Ubuntu Scalar | Ubuntu × | Windows AVX2 | Windows Scalar | Windows × |
| --- | --- | --- | --- | --- | --- | --- |
| 16 × 16 × 16 | 0.3221 / 0.5257 | 1.469 / 1.500 | 4.56× | 0.6521 / 0.6585 | 2.737 / 2.785 | 4.20× |
| 31 × 63 × 17 | 3.938 / 4.298 | 10.85 / 11.20 | 2.76× | 7.038 / 7.787 | 21.78 / 22.12 | 3.09× |
| 64 × 64 × 64 | 22.09 / 24.98 | 178.0 / 182.1 | 8.06× | 30.75 / 31.35 | 96.41 / 98.35 | 3.13× |
| 256 × 256 × 256 | 1,417 / 1,617 | 11,295 / 11,377 | 7.97× | 2,101 / 2,141 | 6,094 / 6,172 | 2.90× |
| 512 × 512 × 512 | 13,972 / 14,440 | 90,258 / 90,718 | 6.46× | 18,427 / 19,227 | 50,437 / 56,839 | 2.74× |

</details>

</details>

<!-- END GENERATED BENCHMARKS -->

[Ubuntu CSV](showcase/data/linux-x64.csv) · [Windows CSV](showcase/data/windows-x64.csv) · [Raw release samples and SHA-256 manifest](https://github.com/iferdous/Forge/releases/tag/v0.1.0-preview.1) · [Measurement protocol](docs/results.md)

## Compiler traces

Follow actual Linux x64 captures through input binding, a typed graph, loop planning, optimized LLVM IR, native disassembly, and the checked output. The 8-, 9-, and 17-element cases exercise the vector-body and scalar-tail paths. The shared producer stays materialized because two consumers need it. The matrix examples cover both a full small result and a shape that crosses all three tile boundaries.

<!-- BEGIN GENERATED TRACES -->

<details>
<summary>Pointwise chain · 8 elements</summary>

| Plan | Loops | Temporary arrays |
| --- | --- | --- |
| unfused | 3 | 2 |
| fused | 1 | 0 |

<details>
<summary>Input and rerun command</summary>

~~~sh
python -m forge 'sqrt((x + 2) * 3)' --array 'x=[1.0,10.0,25.0,46.0,1.0,10.0,25.0,46.0]'
~~~

~~~text
x (8) = [1.0, 10.0, 25.0, 46.0, 1.0, 10.0, 25.0, 46.0]
~~~

</details>

<details>
<summary>Typed graph</summary>

~~~text
%0 = input(0) : f32[8] @1:7 uses=1
%1 = literal(0x40000000) : f32 @1:11 uses=1
%2 = splat(%1) : f32[8] @1:11 uses=1
%3 = add(%0, %2) : f32[8] @1:7 uses=1
%4 = literal(0x40400000) : f32 @1:16 uses=1
%5 = splat(%4) : f32[8] @1:16 uses=1
%6 = multiply(%3, %5) : f32[8] @1:6 uses=1
%7 = sqrt(%6) : f32[8] @1:1 uses=0
return %7
~~~

</details>

<details>
<summary>Loop plan</summary>

**Unfused**

~~~text
loops=3 temporary_arrays=2
loop %3 -> scratch[1]
  %3 = add(%0, %2)
loop %6 -> scratch[2]
  %6 = multiply(%3, %5)
loop %7 -> output
  %7 = sqrt(%6)
~~~

**Fused**

~~~text
loops=1 temporary_arrays=0
loop %7 -> output
  %3 = add(%0, %2)
  %6 = multiply(%3, %5)
  %7 = sqrt(%6)
~~~

</details>

<details>
<summary>Optimized LLVM IR</summary>

**Unfused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %buffers, i64 8
  %2 = load ptr, ptr %1, align 8
  tail call void @forge_add_3(ptr %2, i64 %count, ptr %0, float 2.000000e+00)
  %3 = getelementptr inbounds nuw i8, ptr %buffers, i64 16
  %4 = load ptr, ptr %3, align 8
  tail call void @forge_multiply_6(ptr %4, i64 %count, ptr %2, float 3.000000e+00)
  %5 = load ptr, ptr %buffers, align 8
  tail call void @forge_sqrt_7(ptr %5, i64 %count, ptr %4)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_add_3(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %sum = fadd float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %sum, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_multiply_6(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fmul <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %product = fmul float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %product, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !7

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_sqrt_7(ptr noalias %output, i64 %count, ptr readonly %operand0) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %wide.load)
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !8

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %7 = tail call float @llvm.sqrt.f32(float %6)
  %8 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %7, ptr %8, align 1
  %next = add nuw i64 %index, 1
  %9 = icmp eq i64 %next, %count
  br i1 %9, label %exit, label %loop, !llvm.loop !9

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
!6 = distinct !{!6, !1, !2, !3}
!7 = distinct !{!7, !3, !1, !5}
!8 = distinct !{!8, !1, !2, !3}
!9 = distinct !{!9, !3, !1, !5}
~~~

**Fused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = load ptr, ptr %buffers, align 8
  tail call void @forge_fused_7(ptr %1, i64 %count, ptr %0, float 2.000000e+00, float 3.000000e+00)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_fused_7(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1, float nofpclass(nan inf zero sub nnorm) %operand2) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader4, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert1 = insertelement <8 x float> poison, float %operand2, i64 0
  %broadcast.splat2 = shufflevector <8 x float> %broadcast.splatinsert1, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index3 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index3
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = fmul <8 x float> %broadcast.splat2, %2
  %4 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %3)
  %5 = getelementptr inbounds [4 x i8], ptr %output, i64 %index3
  store <8 x float> %4, ptr %5, align 1
  %index.next = add nuw i64 %index3, 8
  %6 = icmp eq i64 %index.next, %n.vec
  br i1 %6, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader4

loop.preheader4:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader4, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader4 ]
  %7 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %8 = load float, ptr %7, align 1
  %sum = fadd float %operand1, %8
  %product = fmul float %operand2, %sum
  %9 = tail call float @llvm.sqrt.f32(float %product)
  %10 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %9, ptr %10, align 1
  %next = add nuw i64 %index, 1
  %11 = icmp eq i64 %next, %count
  br i1 %11, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
~~~

</details>

<details>
<summary>LLJIT disassembly</summary>

**Unfused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	pushq	%r15
       2:      	pushq	%r14
       4:      	pushq	%r12
       6:      	pushq	%rbx
       7:      	pushq	%rax
       8:      	movq	%rdx, %rbx
       b:      	movq	%rdi, %r14
       e:      	movq	(%rsi), %rdx
      11:      	movq	0x8(%rbx), %r15
      15:      	vmovss	(%rip), %xmm0           # 0x1d <forge_kernel+0x1d>
      1d:      	movq	%r15, %rdi
      20:      	movq	%r14, %rsi
      23:      	callq	0x60 <forge_add_3>
      28:      	movq	0x10(%rbx), %r12
      2c:      	vmovss	(%rip), %xmm0           # 0x34 <forge_kernel+0x34>
      34:      	movq	%r12, %rdi
      37:      	movq	%r14, %rsi
      3a:      	movq	%r15, %rdx
      3d:      	callq	0xb0 <forge_multiply_6>
      42:      	movq	(%rbx), %rdi
      45:      	movq	%r14, %rsi
      48:      	movq	%r12, %rdx
      4b:      	addq	$0x8, %rsp
      4f:      	popq	%rbx
      50:      	popq	%r12
      52:      	popq	%r14
      54:      	popq	%r15
      56:      	jmp	0x100 <forge_sqrt_7>
      5b:      	nopl	(%rax,%rax)

0000000000000060 <forge_add_3>:
      60:      	testq	%rsi, %rsi
      63:      	je	0xa7 <forge_add_3+0x47>
      65:      	cmpq	$0x8, %rsi
      69:      	jae	0x6f <forge_add_3+0xf>
      6b:      	xorl	%eax, %eax
      6d:      	jmp	0x95 <forge_add_3+0x35>
      6f:      	movq	%rsi, %rax
      72:      	andq	$-0x8, %rax
      76:      	vbroadcastss	%xmm0, %ymm1
      7b:      	xorl	%ecx, %ecx
      7d:      	nopl	(%rax)
      80:      	vaddps	(%rdx,%rcx,4), %ymm1, %ymm2
      85:      	vmovups	%ymm2, (%rdi,%rcx,4)
      8a:      	addq	$0x8, %rcx
      8e:      	cmpq	%rcx, %rax
      91:      	jne	0x80 <forge_add_3+0x20>
      93:      	jmp	0xa2 <forge_add_3+0x42>
      95:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm1
      9a:      	vmovss	%xmm1, (%rdi,%rax,4)
      9f:      	incq	%rax
      a2:      	cmpq	%rax, %rsi
      a5:      	jne	0x95 <forge_add_3+0x35>
      a7:      	vzeroupper
      aa:      	retq
      ab:      	nopl	(%rax,%rax)

00000000000000b0 <forge_multiply_6>:
      b0:      	testq	%rsi, %rsi
      b3:      	je	0xf7 <forge_multiply_6+0x47>
      b5:      	cmpq	$0x8, %rsi
      b9:      	jae	0xbf <forge_multiply_6+0xf>
      bb:      	xorl	%eax, %eax
      bd:      	jmp	0xe5 <forge_multiply_6+0x35>
      bf:      	movq	%rsi, %rax
      c2:      	andq	$-0x8, %rax
      c6:      	vbroadcastss	%xmm0, %ymm1
      cb:      	xorl	%ecx, %ecx
      cd:      	nopl	(%rax)
      d0:      	vmulps	(%rdx,%rcx,4), %ymm1, %ymm2
      d5:      	vmovups	%ymm2, (%rdi,%rcx,4)
      da:      	addq	$0x8, %rcx
      de:      	cmpq	%rcx, %rax
      e1:      	jne	0xd0 <forge_multiply_6+0x20>
      e3:      	jmp	0xf2 <forge_multiply_6+0x42>
      e5:      	vmulss	(%rdx,%rax,4), %xmm0, %xmm1
      ea:      	vmovss	%xmm1, (%rdi,%rax,4)
      ef:      	incq	%rax
      f2:      	cmpq	%rax, %rsi
      f5:      	jne	0xe5 <forge_multiply_6+0x35>
      f7:      	vzeroupper
      fa:      	retq
      fb:      	nopl	(%rax,%rax)

0000000000000100 <forge_sqrt_7>:
     100:      	testq	%rsi, %rsi
     103:      	je	0x14b <forge_sqrt_7+0x4b>
     105:      	cmpq	$0x8, %rsi
     109:      	jae	0x10f <forge_sqrt_7+0xf>
     10b:      	xorl	%eax, %eax
     10d:      	jmp	0x135 <forge_sqrt_7+0x35>
     10f:      	movq	%rsi, %rax
     112:      	andq	$-0x8, %rax
     116:      	xorl	%ecx, %ecx
     118:      	nopl	(%rax,%rax)
     120:      	vsqrtps	(%rdx,%rcx,4), %ymm0
     125:      	vmovups	%ymm0, (%rdi,%rcx,4)
     12a:      	addq	$0x8, %rcx
     12e:      	cmpq	%rcx, %rax
     131:      	jne	0x120 <forge_sqrt_7+0x20>
     133:      	jmp	0x146 <forge_sqrt_7+0x46>
     135:      	vmovss	(%rdx,%rax,4), %xmm0
     13a:      	vsqrtss	%xmm0, %xmm0, %xmm0
     13e:      	vmovss	%xmm0, (%rdi,%rax,4)
     143:      	incq	%rax
     146:      	cmpq	%rax, %rsi
     149:      	jne	0x135 <forge_sqrt_7+0x35>
     14b:      	vzeroupper
     14e:      	retq
~~~

**Fused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	movq	%rdi, %rax
       3:      	movq	(%rsi), %rcx
       6:      	movq	(%rdx), %rdi
       9:      	vmovss	(%rip), %xmm0           # 0x11 <forge_kernel+0x11>
      11:      	vmovss	(%rip), %xmm1           # 0x19 <forge_kernel+0x19>
      19:      	movq	%rax, %rsi
      1c:      	movq	%rcx, %rdx
      1f:      	jmp	0x30 <forge_fused_7>
      21:      	nopw	%cs:(%rax,%rax)

0000000000000030 <forge_fused_7>:
      30:      	testq	%rsi, %rsi
      33:      	je	0x97 <forge_fused_7+0x67>
      35:      	cmpq	$0x8, %rsi
      39:      	jae	0x3f <forge_fused_7+0xf>
      3b:      	xorl	%eax, %eax
      3d:      	jmp	0x7d <forge_fused_7+0x4d>
      3f:      	movq	%rsi, %rax
      42:      	andq	$-0x8, %rax
      46:      	vbroadcastss	%xmm0, %ymm2
      4b:      	vbroadcastss	%xmm1, %ymm3
      50:      	xorl	%ecx, %ecx
      52:      	nopw	%cs:(%rax,%rax)
      60:      	vaddps	(%rdx,%rcx,4), %ymm2, %ymm4
      65:      	vmulps	%ymm4, %ymm3, %ymm4
      69:      	vsqrtps	%ymm4, %ymm4
      6d:      	vmovups	%ymm4, (%rdi,%rcx,4)
      72:      	addq	$0x8, %rcx
      76:      	cmpq	%rcx, %rax
      79:      	jne	0x60 <forge_fused_7+0x30>
      7b:      	jmp	0x92 <forge_fused_7+0x62>
      7d:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm2
      82:      	vmulss	%xmm2, %xmm1, %xmm2
      86:      	vsqrtss	%xmm2, %xmm2, %xmm2
      8a:      	vmovss	%xmm2, (%rdi,%rax,4)
      8f:      	incq	%rax
      92:      	cmpq	%rax, %rsi
      95:      	jne	0x7d <forge_fused_7+0x4d>
      97:      	vzeroupper
      9a:      	retq
~~~

</details>

<details>
<summary>Result</summary>

Verified 8 output

~~~json
[3.0, 6.0, 9.0, 12.0, 3.0, 6.0, 9.0, 12.0]
~~~

</details>

</details>

<details>
<summary>Pointwise chain · 9 elements</summary>

| Plan | Loops | Temporary arrays |
| --- | --- | --- |
| unfused | 3 | 2 |
| fused | 1 | 0 |

<details>
<summary>Input and rerun command</summary>

~~~sh
python -m forge 'sqrt((x + 2) * 3)' --array 'x=[1.0,10.0,25.0,46.0,1.0,10.0,25.0,46.0,1.0]'
~~~

~~~text
x (9) = [1.0, 10.0, 25.0, 46.0, 1.0, 10.0, 25.0, 46.0, 1.0]
~~~

</details>

<details>
<summary>Typed graph</summary>

~~~text
%0 = input(0) : f32[9] @1:7 uses=1
%1 = literal(0x40000000) : f32 @1:11 uses=1
%2 = splat(%1) : f32[9] @1:11 uses=1
%3 = add(%0, %2) : f32[9] @1:7 uses=1
%4 = literal(0x40400000) : f32 @1:16 uses=1
%5 = splat(%4) : f32[9] @1:16 uses=1
%6 = multiply(%3, %5) : f32[9] @1:6 uses=1
%7 = sqrt(%6) : f32[9] @1:1 uses=0
return %7
~~~

</details>

<details>
<summary>Loop plan</summary>

**Unfused**

~~~text
loops=3 temporary_arrays=2
loop %3 -> scratch[1]
  %3 = add(%0, %2)
loop %6 -> scratch[2]
  %6 = multiply(%3, %5)
loop %7 -> output
  %7 = sqrt(%6)
~~~

**Fused**

~~~text
loops=1 temporary_arrays=0
loop %7 -> output
  %3 = add(%0, %2)
  %6 = multiply(%3, %5)
  %7 = sqrt(%6)
~~~

</details>

<details>
<summary>Optimized LLVM IR</summary>

**Unfused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %buffers, i64 8
  %2 = load ptr, ptr %1, align 8
  tail call void @forge_add_3(ptr %2, i64 %count, ptr %0, float 2.000000e+00)
  %3 = getelementptr inbounds nuw i8, ptr %buffers, i64 16
  %4 = load ptr, ptr %3, align 8
  tail call void @forge_multiply_6(ptr %4, i64 %count, ptr %2, float 3.000000e+00)
  %5 = load ptr, ptr %buffers, align 8
  tail call void @forge_sqrt_7(ptr %5, i64 %count, ptr %4)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_add_3(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %sum = fadd float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %sum, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_multiply_6(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fmul <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %product = fmul float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %product, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !7

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_sqrt_7(ptr noalias %output, i64 %count, ptr readonly %operand0) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %wide.load)
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !8

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %7 = tail call float @llvm.sqrt.f32(float %6)
  %8 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %7, ptr %8, align 1
  %next = add nuw i64 %index, 1
  %9 = icmp eq i64 %next, %count
  br i1 %9, label %exit, label %loop, !llvm.loop !9

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
!6 = distinct !{!6, !1, !2, !3}
!7 = distinct !{!7, !3, !1, !5}
!8 = distinct !{!8, !1, !2, !3}
!9 = distinct !{!9, !3, !1, !5}
~~~

**Fused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = load ptr, ptr %buffers, align 8
  tail call void @forge_fused_7(ptr %1, i64 %count, ptr %0, float 2.000000e+00, float 3.000000e+00)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_fused_7(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1, float nofpclass(nan inf zero sub nnorm) %operand2) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader4, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert1 = insertelement <8 x float> poison, float %operand2, i64 0
  %broadcast.splat2 = shufflevector <8 x float> %broadcast.splatinsert1, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index3 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index3
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = fmul <8 x float> %broadcast.splat2, %2
  %4 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %3)
  %5 = getelementptr inbounds [4 x i8], ptr %output, i64 %index3
  store <8 x float> %4, ptr %5, align 1
  %index.next = add nuw i64 %index3, 8
  %6 = icmp eq i64 %index.next, %n.vec
  br i1 %6, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader4

loop.preheader4:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader4, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader4 ]
  %7 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %8 = load float, ptr %7, align 1
  %sum = fadd float %operand1, %8
  %product = fmul float %operand2, %sum
  %9 = tail call float @llvm.sqrt.f32(float %product)
  %10 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %9, ptr %10, align 1
  %next = add nuw i64 %index, 1
  %11 = icmp eq i64 %next, %count
  br i1 %11, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
~~~

</details>

<details>
<summary>LLJIT disassembly</summary>

**Unfused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	pushq	%r15
       2:      	pushq	%r14
       4:      	pushq	%r12
       6:      	pushq	%rbx
       7:      	pushq	%rax
       8:      	movq	%rdx, %rbx
       b:      	movq	%rdi, %r14
       e:      	movq	(%rsi), %rdx
      11:      	movq	0x8(%rbx), %r15
      15:      	vmovss	(%rip), %xmm0           # 0x1d <forge_kernel+0x1d>
      1d:      	movq	%r15, %rdi
      20:      	movq	%r14, %rsi
      23:      	callq	0x60 <forge_add_3>
      28:      	movq	0x10(%rbx), %r12
      2c:      	vmovss	(%rip), %xmm0           # 0x34 <forge_kernel+0x34>
      34:      	movq	%r12, %rdi
      37:      	movq	%r14, %rsi
      3a:      	movq	%r15, %rdx
      3d:      	callq	0xb0 <forge_multiply_6>
      42:      	movq	(%rbx), %rdi
      45:      	movq	%r14, %rsi
      48:      	movq	%r12, %rdx
      4b:      	addq	$0x8, %rsp
      4f:      	popq	%rbx
      50:      	popq	%r12
      52:      	popq	%r14
      54:      	popq	%r15
      56:      	jmp	0x100 <forge_sqrt_7>
      5b:      	nopl	(%rax,%rax)

0000000000000060 <forge_add_3>:
      60:      	testq	%rsi, %rsi
      63:      	je	0xa7 <forge_add_3+0x47>
      65:      	cmpq	$0x8, %rsi
      69:      	jae	0x6f <forge_add_3+0xf>
      6b:      	xorl	%eax, %eax
      6d:      	jmp	0x95 <forge_add_3+0x35>
      6f:      	movq	%rsi, %rax
      72:      	andq	$-0x8, %rax
      76:      	vbroadcastss	%xmm0, %ymm1
      7b:      	xorl	%ecx, %ecx
      7d:      	nopl	(%rax)
      80:      	vaddps	(%rdx,%rcx,4), %ymm1, %ymm2
      85:      	vmovups	%ymm2, (%rdi,%rcx,4)
      8a:      	addq	$0x8, %rcx
      8e:      	cmpq	%rcx, %rax
      91:      	jne	0x80 <forge_add_3+0x20>
      93:      	jmp	0xa2 <forge_add_3+0x42>
      95:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm1
      9a:      	vmovss	%xmm1, (%rdi,%rax,4)
      9f:      	incq	%rax
      a2:      	cmpq	%rax, %rsi
      a5:      	jne	0x95 <forge_add_3+0x35>
      a7:      	vzeroupper
      aa:      	retq
      ab:      	nopl	(%rax,%rax)

00000000000000b0 <forge_multiply_6>:
      b0:      	testq	%rsi, %rsi
      b3:      	je	0xf7 <forge_multiply_6+0x47>
      b5:      	cmpq	$0x8, %rsi
      b9:      	jae	0xbf <forge_multiply_6+0xf>
      bb:      	xorl	%eax, %eax
      bd:      	jmp	0xe5 <forge_multiply_6+0x35>
      bf:      	movq	%rsi, %rax
      c2:      	andq	$-0x8, %rax
      c6:      	vbroadcastss	%xmm0, %ymm1
      cb:      	xorl	%ecx, %ecx
      cd:      	nopl	(%rax)
      d0:      	vmulps	(%rdx,%rcx,4), %ymm1, %ymm2
      d5:      	vmovups	%ymm2, (%rdi,%rcx,4)
      da:      	addq	$0x8, %rcx
      de:      	cmpq	%rcx, %rax
      e1:      	jne	0xd0 <forge_multiply_6+0x20>
      e3:      	jmp	0xf2 <forge_multiply_6+0x42>
      e5:      	vmulss	(%rdx,%rax,4), %xmm0, %xmm1
      ea:      	vmovss	%xmm1, (%rdi,%rax,4)
      ef:      	incq	%rax
      f2:      	cmpq	%rax, %rsi
      f5:      	jne	0xe5 <forge_multiply_6+0x35>
      f7:      	vzeroupper
      fa:      	retq
      fb:      	nopl	(%rax,%rax)

0000000000000100 <forge_sqrt_7>:
     100:      	testq	%rsi, %rsi
     103:      	je	0x14b <forge_sqrt_7+0x4b>
     105:      	cmpq	$0x8, %rsi
     109:      	jae	0x10f <forge_sqrt_7+0xf>
     10b:      	xorl	%eax, %eax
     10d:      	jmp	0x135 <forge_sqrt_7+0x35>
     10f:      	movq	%rsi, %rax
     112:      	andq	$-0x8, %rax
     116:      	xorl	%ecx, %ecx
     118:      	nopl	(%rax,%rax)
     120:      	vsqrtps	(%rdx,%rcx,4), %ymm0
     125:      	vmovups	%ymm0, (%rdi,%rcx,4)
     12a:      	addq	$0x8, %rcx
     12e:      	cmpq	%rcx, %rax
     131:      	jne	0x120 <forge_sqrt_7+0x20>
     133:      	jmp	0x146 <forge_sqrt_7+0x46>
     135:      	vmovss	(%rdx,%rax,4), %xmm0
     13a:      	vsqrtss	%xmm0, %xmm0, %xmm0
     13e:      	vmovss	%xmm0, (%rdi,%rax,4)
     143:      	incq	%rax
     146:      	cmpq	%rax, %rsi
     149:      	jne	0x135 <forge_sqrt_7+0x35>
     14b:      	vzeroupper
     14e:      	retq
~~~

**Fused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	movq	%rdi, %rax
       3:      	movq	(%rsi), %rcx
       6:      	movq	(%rdx), %rdi
       9:      	vmovss	(%rip), %xmm0           # 0x11 <forge_kernel+0x11>
      11:      	vmovss	(%rip), %xmm1           # 0x19 <forge_kernel+0x19>
      19:      	movq	%rax, %rsi
      1c:      	movq	%rcx, %rdx
      1f:      	jmp	0x30 <forge_fused_7>
      21:      	nopw	%cs:(%rax,%rax)

0000000000000030 <forge_fused_7>:
      30:      	testq	%rsi, %rsi
      33:      	je	0x97 <forge_fused_7+0x67>
      35:      	cmpq	$0x8, %rsi
      39:      	jae	0x3f <forge_fused_7+0xf>
      3b:      	xorl	%eax, %eax
      3d:      	jmp	0x7d <forge_fused_7+0x4d>
      3f:      	movq	%rsi, %rax
      42:      	andq	$-0x8, %rax
      46:      	vbroadcastss	%xmm0, %ymm2
      4b:      	vbroadcastss	%xmm1, %ymm3
      50:      	xorl	%ecx, %ecx
      52:      	nopw	%cs:(%rax,%rax)
      60:      	vaddps	(%rdx,%rcx,4), %ymm2, %ymm4
      65:      	vmulps	%ymm4, %ymm3, %ymm4
      69:      	vsqrtps	%ymm4, %ymm4
      6d:      	vmovups	%ymm4, (%rdi,%rcx,4)
      72:      	addq	$0x8, %rcx
      76:      	cmpq	%rcx, %rax
      79:      	jne	0x60 <forge_fused_7+0x30>
      7b:      	jmp	0x92 <forge_fused_7+0x62>
      7d:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm2
      82:      	vmulss	%xmm2, %xmm1, %xmm2
      86:      	vsqrtss	%xmm2, %xmm2, %xmm2
      8a:      	vmovss	%xmm2, (%rdi,%rax,4)
      8f:      	incq	%rax
      92:      	cmpq	%rax, %rsi
      95:      	jne	0x7d <forge_fused_7+0x4d>
      97:      	vzeroupper
      9a:      	retq
~~~

</details>

<details>
<summary>Result</summary>

Verified 9 output

~~~json
[3.0, 6.0, 9.0, 12.0, 3.0, 6.0, 9.0, 12.0, 3.0]
~~~

</details>

</details>

<details>
<summary>Pointwise chain · 17 elements</summary>

| Plan | Loops | Temporary arrays |
| --- | --- | --- |
| unfused | 3 | 2 |
| fused | 1 | 0 |

<details>
<summary>Input and rerun command</summary>

~~~sh
python -m forge 'sqrt((x + 2) * 3)' --array 'x=[1.0,10.0,25.0,46.0,1.0,10.0,25.0,46.0,1.0,10.0,25.0,46.0,1.0,10.0,25.0,46.0,1.0]'
~~~

~~~text
x (17) = [1.0, 10.0, 25.0, 46.0, 1.0, 10.0, 25.0, 46.0, 1.0, 10.0, 25.0, 46.0, 1.0, 10.0, 25.0, 46.0, 1.0]
~~~

</details>

<details>
<summary>Typed graph</summary>

~~~text
%0 = input(0) : f32[17] @1:7 uses=1
%1 = literal(0x40000000) : f32 @1:11 uses=1
%2 = splat(%1) : f32[17] @1:11 uses=1
%3 = add(%0, %2) : f32[17] @1:7 uses=1
%4 = literal(0x40400000) : f32 @1:16 uses=1
%5 = splat(%4) : f32[17] @1:16 uses=1
%6 = multiply(%3, %5) : f32[17] @1:6 uses=1
%7 = sqrt(%6) : f32[17] @1:1 uses=0
return %7
~~~

</details>

<details>
<summary>Loop plan</summary>

**Unfused**

~~~text
loops=3 temporary_arrays=2
loop %3 -> scratch[1]
  %3 = add(%0, %2)
loop %6 -> scratch[2]
  %6 = multiply(%3, %5)
loop %7 -> output
  %7 = sqrt(%6)
~~~

**Fused**

~~~text
loops=1 temporary_arrays=0
loop %7 -> output
  %3 = add(%0, %2)
  %6 = multiply(%3, %5)
  %7 = sqrt(%6)
~~~

</details>

<details>
<summary>Optimized LLVM IR</summary>

**Unfused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %buffers, i64 8
  %2 = load ptr, ptr %1, align 8
  tail call void @forge_add_3(ptr %2, i64 %count, ptr %0, float 2.000000e+00)
  %3 = getelementptr inbounds nuw i8, ptr %buffers, i64 16
  %4 = load ptr, ptr %3, align 8
  tail call void @forge_multiply_6(ptr %4, i64 %count, ptr %2, float 3.000000e+00)
  %5 = load ptr, ptr %buffers, align 8
  tail call void @forge_sqrt_7(ptr %5, i64 %count, ptr %4)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_add_3(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %sum = fadd float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %sum, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_multiply_6(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fmul <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %product = fmul float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %product, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !7

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_sqrt_7(ptr noalias %output, i64 %count, ptr readonly %operand0) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %wide.load)
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !8

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %7 = tail call float @llvm.sqrt.f32(float %6)
  %8 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %7, ptr %8, align 1
  %next = add nuw i64 %index, 1
  %9 = icmp eq i64 %next, %count
  br i1 %9, label %exit, label %loop, !llvm.loop !9

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
!6 = distinct !{!6, !1, !2, !3}
!7 = distinct !{!7, !3, !1, !5}
!8 = distinct !{!8, !1, !2, !3}
!9 = distinct !{!9, !3, !1, !5}
~~~

**Fused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = load ptr, ptr %buffers, align 8
  tail call void @forge_fused_7(ptr %1, i64 %count, ptr %0, float 2.000000e+00, float 3.000000e+00)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_fused_7(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1, float nofpclass(nan inf zero sub nnorm) %operand2) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader4, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert1 = insertelement <8 x float> poison, float %operand2, i64 0
  %broadcast.splat2 = shufflevector <8 x float> %broadcast.splatinsert1, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index3 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index3
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = fmul <8 x float> %broadcast.splat2, %2
  %4 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %3)
  %5 = getelementptr inbounds [4 x i8], ptr %output, i64 %index3
  store <8 x float> %4, ptr %5, align 1
  %index.next = add nuw i64 %index3, 8
  %6 = icmp eq i64 %index.next, %n.vec
  br i1 %6, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader4

loop.preheader4:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader4, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader4 ]
  %7 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %8 = load float, ptr %7, align 1
  %sum = fadd float %operand1, %8
  %product = fmul float %operand2, %sum
  %9 = tail call float @llvm.sqrt.f32(float %product)
  %10 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %9, ptr %10, align 1
  %next = add nuw i64 %index, 1
  %11 = icmp eq i64 %next, %count
  br i1 %11, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
~~~

</details>

<details>
<summary>LLJIT disassembly</summary>

**Unfused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	pushq	%r15
       2:      	pushq	%r14
       4:      	pushq	%r12
       6:      	pushq	%rbx
       7:      	pushq	%rax
       8:      	movq	%rdx, %rbx
       b:      	movq	%rdi, %r14
       e:      	movq	(%rsi), %rdx
      11:      	movq	0x8(%rbx), %r15
      15:      	vmovss	(%rip), %xmm0           # 0x1d <forge_kernel+0x1d>
      1d:      	movq	%r15, %rdi
      20:      	movq	%r14, %rsi
      23:      	callq	0x60 <forge_add_3>
      28:      	movq	0x10(%rbx), %r12
      2c:      	vmovss	(%rip), %xmm0           # 0x34 <forge_kernel+0x34>
      34:      	movq	%r12, %rdi
      37:      	movq	%r14, %rsi
      3a:      	movq	%r15, %rdx
      3d:      	callq	0xb0 <forge_multiply_6>
      42:      	movq	(%rbx), %rdi
      45:      	movq	%r14, %rsi
      48:      	movq	%r12, %rdx
      4b:      	addq	$0x8, %rsp
      4f:      	popq	%rbx
      50:      	popq	%r12
      52:      	popq	%r14
      54:      	popq	%r15
      56:      	jmp	0x100 <forge_sqrt_7>
      5b:      	nopl	(%rax,%rax)

0000000000000060 <forge_add_3>:
      60:      	testq	%rsi, %rsi
      63:      	je	0xa7 <forge_add_3+0x47>
      65:      	cmpq	$0x8, %rsi
      69:      	jae	0x6f <forge_add_3+0xf>
      6b:      	xorl	%eax, %eax
      6d:      	jmp	0x95 <forge_add_3+0x35>
      6f:      	movq	%rsi, %rax
      72:      	andq	$-0x8, %rax
      76:      	vbroadcastss	%xmm0, %ymm1
      7b:      	xorl	%ecx, %ecx
      7d:      	nopl	(%rax)
      80:      	vaddps	(%rdx,%rcx,4), %ymm1, %ymm2
      85:      	vmovups	%ymm2, (%rdi,%rcx,4)
      8a:      	addq	$0x8, %rcx
      8e:      	cmpq	%rcx, %rax
      91:      	jne	0x80 <forge_add_3+0x20>
      93:      	jmp	0xa2 <forge_add_3+0x42>
      95:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm1
      9a:      	vmovss	%xmm1, (%rdi,%rax,4)
      9f:      	incq	%rax
      a2:      	cmpq	%rax, %rsi
      a5:      	jne	0x95 <forge_add_3+0x35>
      a7:      	vzeroupper
      aa:      	retq
      ab:      	nopl	(%rax,%rax)

00000000000000b0 <forge_multiply_6>:
      b0:      	testq	%rsi, %rsi
      b3:      	je	0xf7 <forge_multiply_6+0x47>
      b5:      	cmpq	$0x8, %rsi
      b9:      	jae	0xbf <forge_multiply_6+0xf>
      bb:      	xorl	%eax, %eax
      bd:      	jmp	0xe5 <forge_multiply_6+0x35>
      bf:      	movq	%rsi, %rax
      c2:      	andq	$-0x8, %rax
      c6:      	vbroadcastss	%xmm0, %ymm1
      cb:      	xorl	%ecx, %ecx
      cd:      	nopl	(%rax)
      d0:      	vmulps	(%rdx,%rcx,4), %ymm1, %ymm2
      d5:      	vmovups	%ymm2, (%rdi,%rcx,4)
      da:      	addq	$0x8, %rcx
      de:      	cmpq	%rcx, %rax
      e1:      	jne	0xd0 <forge_multiply_6+0x20>
      e3:      	jmp	0xf2 <forge_multiply_6+0x42>
      e5:      	vmulss	(%rdx,%rax,4), %xmm0, %xmm1
      ea:      	vmovss	%xmm1, (%rdi,%rax,4)
      ef:      	incq	%rax
      f2:      	cmpq	%rax, %rsi
      f5:      	jne	0xe5 <forge_multiply_6+0x35>
      f7:      	vzeroupper
      fa:      	retq
      fb:      	nopl	(%rax,%rax)

0000000000000100 <forge_sqrt_7>:
     100:      	testq	%rsi, %rsi
     103:      	je	0x14b <forge_sqrt_7+0x4b>
     105:      	cmpq	$0x8, %rsi
     109:      	jae	0x10f <forge_sqrt_7+0xf>
     10b:      	xorl	%eax, %eax
     10d:      	jmp	0x135 <forge_sqrt_7+0x35>
     10f:      	movq	%rsi, %rax
     112:      	andq	$-0x8, %rax
     116:      	xorl	%ecx, %ecx
     118:      	nopl	(%rax,%rax)
     120:      	vsqrtps	(%rdx,%rcx,4), %ymm0
     125:      	vmovups	%ymm0, (%rdi,%rcx,4)
     12a:      	addq	$0x8, %rcx
     12e:      	cmpq	%rcx, %rax
     131:      	jne	0x120 <forge_sqrt_7+0x20>
     133:      	jmp	0x146 <forge_sqrt_7+0x46>
     135:      	vmovss	(%rdx,%rax,4), %xmm0
     13a:      	vsqrtss	%xmm0, %xmm0, %xmm0
     13e:      	vmovss	%xmm0, (%rdi,%rax,4)
     143:      	incq	%rax
     146:      	cmpq	%rax, %rsi
     149:      	jne	0x135 <forge_sqrt_7+0x35>
     14b:      	vzeroupper
     14e:      	retq
~~~

**Fused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	movq	%rdi, %rax
       3:      	movq	(%rsi), %rcx
       6:      	movq	(%rdx), %rdi
       9:      	vmovss	(%rip), %xmm0           # 0x11 <forge_kernel+0x11>
      11:      	vmovss	(%rip), %xmm1           # 0x19 <forge_kernel+0x19>
      19:      	movq	%rax, %rsi
      1c:      	movq	%rcx, %rdx
      1f:      	jmp	0x30 <forge_fused_7>
      21:      	nopw	%cs:(%rax,%rax)

0000000000000030 <forge_fused_7>:
      30:      	testq	%rsi, %rsi
      33:      	je	0x97 <forge_fused_7+0x67>
      35:      	cmpq	$0x8, %rsi
      39:      	jae	0x3f <forge_fused_7+0xf>
      3b:      	xorl	%eax, %eax
      3d:      	jmp	0x7d <forge_fused_7+0x4d>
      3f:      	movq	%rsi, %rax
      42:      	andq	$-0x8, %rax
      46:      	vbroadcastss	%xmm0, %ymm2
      4b:      	vbroadcastss	%xmm1, %ymm3
      50:      	xorl	%ecx, %ecx
      52:      	nopw	%cs:(%rax,%rax)
      60:      	vaddps	(%rdx,%rcx,4), %ymm2, %ymm4
      65:      	vmulps	%ymm4, %ymm3, %ymm4
      69:      	vsqrtps	%ymm4, %ymm4
      6d:      	vmovups	%ymm4, (%rdi,%rcx,4)
      72:      	addq	$0x8, %rcx
      76:      	cmpq	%rcx, %rax
      79:      	jne	0x60 <forge_fused_7+0x30>
      7b:      	jmp	0x92 <forge_fused_7+0x62>
      7d:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm2
      82:      	vmulss	%xmm2, %xmm1, %xmm2
      86:      	vsqrtss	%xmm2, %xmm2, %xmm2
      8a:      	vmovss	%xmm2, (%rdi,%rax,4)
      8f:      	incq	%rax
      92:      	cmpq	%rax, %rsi
      95:      	jne	0x7d <forge_fused_7+0x4d>
      97:      	vzeroupper
      9a:      	retq
~~~

</details>

<details>
<summary>Result</summary>

Verified 17 output

~~~json
[3.0, 6.0, 9.0, 12.0, 3.0, 6.0, 9.0, 12.0, 3.0, 6.0, 9.0, 12.0, 3.0, 6.0, 9.0, 12.0, 3.0]
~~~

</details>

</details>

<details>
<summary>Shared producer</summary>

| Plan | Loops | Temporary arrays |
| --- | --- | --- |
| unfused | 3 | 2 |
| fused | 2 | 1 |

<details>
<summary>Input and rerun command</summary>

~~~sh
python -m forge 'sqrt((x + 2) * (x + 2))' --array 'x=[1.0,10.0,25.0,46.0,1.0,10.0,25.0,46.0,1.0]'
~~~

~~~text
x (9) = [1.0, 10.0, 25.0, 46.0, 1.0, 10.0, 25.0, 46.0, 1.0]
~~~

</details>

<details>
<summary>Typed graph</summary>

~~~text
%0 = input(0) : f32[9] @1:7 uses=1
%1 = literal(0x40000000) : f32 @1:11 uses=1
%2 = splat(%1) : f32[9] @1:11 uses=1
%3 = add(%0, %2) : f32[9] @1:7 uses=2
%4 = multiply(%3, %3) : f32[9] @1:6 uses=1
%5 = sqrt(%4) : f32[9] @1:1 uses=0
return %5
~~~

</details>

<details>
<summary>Loop plan</summary>

**Unfused**

~~~text
loops=3 temporary_arrays=2
loop %3 -> scratch[1]
  %3 = add(%0, %2)
loop %4 -> scratch[2]
  %4 = multiply(%3, %3)
loop %5 -> output
  %5 = sqrt(%4)
~~~

**Fused**

~~~text
loops=2 temporary_arrays=1
loop %3 -> scratch[1]
  %3 = add(%0, %2)
loop %5 -> output
  %4 = multiply(%3, %3)
  %5 = sqrt(%4)
~~~

</details>

<details>
<summary>Optimized LLVM IR</summary>

**Unfused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %buffers, i64 8
  %2 = load ptr, ptr %1, align 8
  tail call void @forge_add_3(ptr %2, i64 %count, ptr %0, float 2.000000e+00)
  %3 = getelementptr inbounds nuw i8, ptr %buffers, i64 16
  %4 = load ptr, ptr %3, align 8
  tail call void @forge_multiply_4(ptr %4, i64 %count, ptr %2)
  %5 = load ptr, ptr %buffers, align 8
  tail call void @forge_sqrt_5(ptr %5, i64 %count, ptr %4)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_add_3(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %sum = fadd float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %sum, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_multiply_4(ptr noalias %output, i64 %count, ptr readonly %operand0) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fmul <8 x float> %wide.load, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %product = fmul float %6, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %product, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !7

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_sqrt_5(ptr noalias %output, i64 %count, ptr readonly %operand0) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %wide.load)
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !8

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %7 = tail call float @llvm.sqrt.f32(float %6)
  %8 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %7, ptr %8, align 1
  %next = add nuw i64 %index, 1
  %9 = icmp eq i64 %next, %count
  br i1 %9, label %exit, label %loop, !llvm.loop !9

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
!6 = distinct !{!6, !1, !2, !3}
!7 = distinct !{!7, !3, !1, !5}
!8 = distinct !{!8, !1, !2, !3}
!9 = distinct !{!9, !3, !1, !5}
~~~

**Fused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %buffers, i64 8
  %2 = load ptr, ptr %1, align 8
  tail call void @forge_add_3(ptr %2, i64 %count, ptr %0, float 2.000000e+00)
  %3 = load ptr, ptr %buffers, align 8
  tail call void @forge_fused_5(ptr %3, i64 %count, ptr %2)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_add_3(ptr noalias %output, i64 %count, ptr readonly %operand0, float nofpclass(nan inf zero sub nnorm) %operand1) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  %broadcast.splatinsert = insertelement <8 x float> poison, float %operand1, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fadd <8 x float> %broadcast.splat, %wide.load
  %3 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %2, ptr %3, align 1
  %index.next = add nuw i64 %index1, 8
  %4 = icmp eq i64 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %5 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %6 = load float, ptr %5, align 1
  %sum = fadd float %operand1, %6
  %7 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %sum, ptr %7, align 1
  %next = add nuw i64 %index, 1
  %8 = icmp eq i64 %next, %count
  br i1 %8, label %exit, label %loop, !llvm.loop !4

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_fused_5(ptr noalias %output, i64 %count, ptr readonly %operand0) unnamed_addr #0 {
entry:
  %0 = icmp eq i64 %count, 0
  br i1 %0, label %exit, label %loop.preheader

loop.preheader:                                   ; preds = %entry
  %min.iters.check = icmp ult i64 %count, 8
  br i1 %min.iters.check, label %loop.preheader2, label %vector.ph

vector.ph:                                        ; preds = %loop.preheader
  %n.vec = and i64 %count, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index1 = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %1 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index1
  %wide.load = load <8 x float>, ptr %1, align 1
  %2 = fmul <8 x float> %wide.load, %wide.load
  %3 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %2)
  %4 = getelementptr inbounds [4 x i8], ptr %output, i64 %index1
  store <8 x float> %3, ptr %4, align 1
  %index.next = add nuw i64 %index1, 8
  %5 = icmp eq i64 %index.next, %n.vec
  br i1 %5, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %count, %n.vec
  br i1 %cmp.n, label %exit, label %loop.preheader2

loop.preheader2:                                  ; preds = %loop.preheader, %middle.block
  %index.ph = phi i64 [ 0, %loop.preheader ], [ %n.vec, %middle.block ]
  br label %loop

loop:                                             ; preds = %loop.preheader2, %loop
  %index = phi i64 [ %next, %loop ], [ %index.ph, %loop.preheader2 ]
  %6 = getelementptr inbounds [4 x i8], ptr %operand0, i64 %index
  %7 = load float, ptr %6, align 1
  %product = fmul float %7, %7
  %8 = tail call float @llvm.sqrt.f32(float %product)
  %9 = getelementptr inbounds [4 x i8], ptr %output, i64 %index
  store float %8, ptr %9, align 1
  %next = add nuw i64 %index, 1
  %10 = icmp eq i64 %next, %count
  br i1 %10, label %exit, label %loop, !llvm.loop !7

exit:                                             ; preds = %loop, %middle.block, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
!6 = distinct !{!6, !1, !2, !3}
!7 = distinct !{!7, !3, !1, !5}
~~~

</details>

<details>
<summary>LLJIT disassembly</summary>

**Unfused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	pushq	%r15
       2:      	pushq	%r14
       4:      	pushq	%r12
       6:      	pushq	%rbx
       7:      	pushq	%rax
       8:      	movq	%rdx, %rbx
       b:      	movq	%rdi, %r14
       e:      	movq	(%rsi), %rdx
      11:      	movq	0x8(%rbx), %r15
      15:      	vmovss	(%rip), %xmm0           # 0x1d <forge_kernel+0x1d>
      1d:      	movq	%r15, %rdi
      20:      	movq	%r14, %rsi
      23:      	callq	0x60 <forge_add_3>
      28:      	movq	0x10(%rbx), %r12
      2c:      	movq	%r12, %rdi
      2f:      	movq	%r14, %rsi
      32:      	movq	%r15, %rdx
      35:      	callq	0xb0 <forge_multiply_4>
      3a:      	movq	(%rbx), %rdi
      3d:      	movq	%r14, %rsi
      40:      	movq	%r12, %rdx
      43:      	addq	$0x8, %rsp
      47:      	popq	%rbx
      48:      	popq	%r12
      4a:      	popq	%r14
      4c:      	popq	%r15
      4e:      	jmp	0x110 <forge_sqrt_5>
      53:      	nopw	%cs:(%rax,%rax)

0000000000000060 <forge_add_3>:
      60:      	testq	%rsi, %rsi
      63:      	je	0xa7 <forge_add_3+0x47>
      65:      	cmpq	$0x8, %rsi
      69:      	jae	0x6f <forge_add_3+0xf>
      6b:      	xorl	%eax, %eax
      6d:      	jmp	0x95 <forge_add_3+0x35>
      6f:      	movq	%rsi, %rax
      72:      	andq	$-0x8, %rax
      76:      	vbroadcastss	%xmm0, %ymm1
      7b:      	xorl	%ecx, %ecx
      7d:      	nopl	(%rax)
      80:      	vaddps	(%rdx,%rcx,4), %ymm1, %ymm2
      85:      	vmovups	%ymm2, (%rdi,%rcx,4)
      8a:      	addq	$0x8, %rcx
      8e:      	cmpq	%rcx, %rax
      91:      	jne	0x80 <forge_add_3+0x20>
      93:      	jmp	0xa2 <forge_add_3+0x42>
      95:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm1
      9a:      	vmovss	%xmm1, (%rdi,%rax,4)
      9f:      	incq	%rax
      a2:      	cmpq	%rax, %rsi
      a5:      	jne	0x95 <forge_add_3+0x35>
      a7:      	vzeroupper
      aa:      	retq
      ab:      	nopl	(%rax,%rax)

00000000000000b0 <forge_multiply_4>:
      b0:      	testq	%rsi, %rsi
      b3:      	je	0xff <forge_multiply_4+0x4f>
      b5:      	cmpq	$0x8, %rsi
      b9:      	jae	0xbf <forge_multiply_4+0xf>
      bb:      	xorl	%eax, %eax
      bd:      	jmp	0xe9 <forge_multiply_4+0x39>
      bf:      	movq	%rsi, %rax
      c2:      	andq	$-0x8, %rax
      c6:      	xorl	%ecx, %ecx
      c8:      	nopl	(%rax,%rax)
      d0:      	vmovups	(%rdx,%rcx,4), %ymm0
      d5:      	vmulps	%ymm0, %ymm0, %ymm0
      d9:      	vmovups	%ymm0, (%rdi,%rcx,4)
      de:      	addq	$0x8, %rcx
      e2:      	cmpq	%rcx, %rax
      e5:      	jne	0xd0 <forge_multiply_4+0x20>
      e7:      	jmp	0xfa <forge_multiply_4+0x4a>
      e9:      	vmovss	(%rdx,%rax,4), %xmm0
      ee:      	vmulss	%xmm0, %xmm0, %xmm0
      f2:      	vmovss	%xmm0, (%rdi,%rax,4)
      f7:      	incq	%rax
      fa:      	cmpq	%rax, %rsi
      fd:      	jne	0xe9 <forge_multiply_4+0x39>
      ff:      	vzeroupper
     102:      	retq
     103:      	nopw	%cs:(%rax,%rax)

0000000000000110 <forge_sqrt_5>:
     110:      	testq	%rsi, %rsi
     113:      	je	0x15b <forge_sqrt_5+0x4b>
     115:      	cmpq	$0x8, %rsi
     119:      	jae	0x11f <forge_sqrt_5+0xf>
     11b:      	xorl	%eax, %eax
     11d:      	jmp	0x145 <forge_sqrt_5+0x35>
     11f:      	movq	%rsi, %rax
     122:      	andq	$-0x8, %rax
     126:      	xorl	%ecx, %ecx
     128:      	nopl	(%rax,%rax)
     130:      	vsqrtps	(%rdx,%rcx,4), %ymm0
     135:      	vmovups	%ymm0, (%rdi,%rcx,4)
     13a:      	addq	$0x8, %rcx
     13e:      	cmpq	%rcx, %rax
     141:      	jne	0x130 <forge_sqrt_5+0x20>
     143:      	jmp	0x156 <forge_sqrt_5+0x46>
     145:      	vmovss	(%rdx,%rax,4), %xmm0
     14a:      	vsqrtss	%xmm0, %xmm0, %xmm0
     14e:      	vmovss	%xmm0, (%rdi,%rax,4)
     153:      	incq	%rax
     156:      	cmpq	%rax, %rsi
     159:      	jne	0x145 <forge_sqrt_5+0x35>
     15b:      	vzeroupper
     15e:      	retq
~~~

**Fused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	pushq	%r15
       2:      	pushq	%r14
       4:      	pushq	%rbx
       5:      	movq	%rdx, %rbx
       8:      	movq	%rdi, %r14
       b:      	movq	(%rsi), %rdx
       e:      	movq	0x8(%rbx), %r15
      12:      	vmovss	(%rip), %xmm0           # 0x1a <forge_kernel+0x1a>
      1a:      	movq	%r15, %rdi
      1d:      	movq	%r14, %rsi
      20:      	callq	0x40 <forge_add_3>
      25:      	movq	(%rbx), %rdi
      28:      	movq	%r14, %rsi
      2b:      	movq	%r15, %rdx
      2e:      	popq	%rbx
      2f:      	popq	%r14
      31:      	popq	%r15
      33:      	jmp	0x90 <forge_fused_5>
      35:      	nopw	%cs:(%rax,%rax)

0000000000000040 <forge_add_3>:
      40:      	testq	%rsi, %rsi
      43:      	je	0x87 <forge_add_3+0x47>
      45:      	cmpq	$0x8, %rsi
      49:      	jae	0x4f <forge_add_3+0xf>
      4b:      	xorl	%eax, %eax
      4d:      	jmp	0x75 <forge_add_3+0x35>
      4f:      	movq	%rsi, %rax
      52:      	andq	$-0x8, %rax
      56:      	vbroadcastss	%xmm0, %ymm1
      5b:      	xorl	%ecx, %ecx
      5d:      	nopl	(%rax)
      60:      	vaddps	(%rdx,%rcx,4), %ymm1, %ymm2
      65:      	vmovups	%ymm2, (%rdi,%rcx,4)
      6a:      	addq	$0x8, %rcx
      6e:      	cmpq	%rcx, %rax
      71:      	jne	0x60 <forge_add_3+0x20>
      73:      	jmp	0x82 <forge_add_3+0x42>
      75:      	vaddss	(%rdx,%rax,4), %xmm0, %xmm1
      7a:      	vmovss	%xmm1, (%rdi,%rax,4)
      7f:      	incq	%rax
      82:      	cmpq	%rax, %rsi
      85:      	jne	0x75 <forge_add_3+0x35>
      87:      	vzeroupper
      8a:      	retq
      8b:      	nopl	(%rax,%rax)

0000000000000090 <forge_fused_5>:
      90:      	testq	%rsi, %rsi
      93:      	je	0xe7 <forge_fused_5+0x57>
      95:      	cmpq	$0x8, %rsi
      99:      	jae	0x9f <forge_fused_5+0xf>
      9b:      	xorl	%eax, %eax
      9d:      	jmp	0xcd <forge_fused_5+0x3d>
      9f:      	movq	%rsi, %rax
      a2:      	andq	$-0x8, %rax
      a6:      	xorl	%ecx, %ecx
      a8:      	nopl	(%rax,%rax)
      b0:      	vmovups	(%rdx,%rcx,4), %ymm0
      b5:      	vmulps	%ymm0, %ymm0, %ymm0
      b9:      	vsqrtps	%ymm0, %ymm0
      bd:      	vmovups	%ymm0, (%rdi,%rcx,4)
      c2:      	addq	$0x8, %rcx
      c6:      	cmpq	%rcx, %rax
      c9:      	jne	0xb0 <forge_fused_5+0x20>
      cb:      	jmp	0xe2 <forge_fused_5+0x52>
      cd:      	vmovss	(%rdx,%rax,4), %xmm0
      d2:      	vmulss	%xmm0, %xmm0, %xmm0
      d6:      	vsqrtss	%xmm0, %xmm0, %xmm0
      da:      	vmovss	%xmm0, (%rdi,%rax,4)
      df:      	incq	%rax
      e2:      	cmpq	%rax, %rsi
      e5:      	jne	0xcd <forge_fused_5+0x3d>
      e7:      	vzeroupper
      ea:      	retq
~~~

</details>

<details>
<summary>Result</summary>

Verified 9 output

~~~json
[3.0, 12.0, 27.0, 48.0, 3.0, 12.0, 27.0, 48.0, 3.0]
~~~

</details>

</details>

<details>
<summary>Ordered sum</summary>

| Plan | Loops | Temporary arrays |
| --- | --- | --- |
| unfused | 1 | 0 |
| fused | 1 | 0 |

<details>
<summary>Input and rerun command</summary>

~~~sh
python -m forge 'sum(x)' --array 'x=[16777216.0,1.0,-16777216.0,4.0]'
~~~

~~~text
x (4) = [16777216.0, 1.0, -16777216.0, 4.0]
~~~

</details>

<details>
<summary>Typed graph</summary>

~~~text
%0 = input(0) : f32[4] @1:5 uses=1
%1 = sum(%0) : f32 @1:1 uses=0
return %1
~~~

</details>

<details>
<summary>Loop plan</summary>

**Unfused**

~~~text
loops=1 temporary_arrays=0
loop %1 -> output
  %1 = sum(%0)
~~~

**Fused**

~~~text
loops=1 temporary_arrays=0
loop %1 -> output
  %1 = sum(%0)
~~~

</details>

<details>
<summary>Optimized LLVM IR</summary>

**Unfused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none)
define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = tail call fastcc float @forge_reduce_sum_1(ptr %0)
  %2 = load ptr, ptr %buffers, align 8
  store float %1, ptr %2, align 1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: read)
define internal fastcc float @forge_reduce_sum_1(ptr readonly captures(none) %0) unnamed_addr #1 {
entry:
  br label %ordered

ordered:                                          ; preds = %ordered, %entry
  %index = phi i64 [ 0, %entry ], [ %next, %ordered ]
  %accumulator = phi float [ 0.000000e+00, %entry ], [ %total, %ordered ]
  %1 = getelementptr inbounds nuw [4 x i8], ptr %0, i64 %index
  %2 = load float, ptr %1, align 1
  %total = fadd float %accumulator, %2
  %next = add nuw nsw i64 %index, 1
  %3 = icmp eq i64 %next, 4
  br i1 %3, label %exit, label %ordered, !llvm.loop !0

exit:                                             ; preds = %ordered
  ret float %total
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none) }
attributes #1 = { nofree noinline norecurse nosync nounwind memory(argmem: read) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 false}
~~~

**Fused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none)
define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = tail call fastcc float @forge_reduce_sum_1(ptr %0)
  %2 = load ptr, ptr %buffers, align 8
  store float %1, ptr %2, align 1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: read)
define internal fastcc float @forge_reduce_sum_1(ptr readonly captures(none) %0) unnamed_addr #1 {
entry:
  br label %ordered

ordered:                                          ; preds = %ordered, %entry
  %index = phi i64 [ 0, %entry ], [ %next, %ordered ]
  %accumulator = phi float [ 0.000000e+00, %entry ], [ %total, %ordered ]
  %1 = getelementptr inbounds nuw [4 x i8], ptr %0, i64 %index
  %2 = load float, ptr %1, align 1
  %total = fadd float %accumulator, %2
  %next = add nuw nsw i64 %index, 1
  %3 = icmp eq i64 %next, 4
  br i1 %3, label %exit, label %ordered, !llvm.loop !0

exit:                                             ; preds = %ordered
  ret float %total
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none) }
attributes #1 = { nofree noinline norecurse nosync nounwind memory(argmem: read) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 false}
~~~

</details>

<details>
<summary>LLJIT disassembly</summary>

**Unfused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	pushq	%rbx
       1:      	movq	%rdx, %rbx
       4:      	movq	(%rsi), %rdi
       7:      	callq	0x20 <forge_reduce_sum_1>
       c:      	movq	(%rbx), %rax
       f:      	vmovss	%xmm0, (%rax)
      13:      	popq	%rbx
      14:      	retq
      15:      	nopw	%cs:(%rax,%rax)

0000000000000020 <forge_reduce_sum_1>:
      20:      	vxorps	%xmm0, %xmm0, %xmm0
      24:      	xorl	%eax, %eax
      26:      	nopw	%cs:(%rax,%rax)
      30:      	vaddss	(%rdi,%rax,4), %xmm0, %xmm0
      35:      	incq	%rax
      38:      	cmpq	$0x4, %rax
      3c:      	jne	0x30 <forge_reduce_sum_1+0x10>
      3e:      	retq
~~~

**Fused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	pushq	%rbx
       1:      	movq	%rdx, %rbx
       4:      	movq	(%rsi), %rdi
       7:      	callq	0x20 <forge_reduce_sum_1>
       c:      	movq	(%rbx), %rax
       f:      	vmovss	%xmm0, (%rax)
      13:      	popq	%rbx
      14:      	retq
      15:      	nopw	%cs:(%rax,%rax)

0000000000000020 <forge_reduce_sum_1>:
      20:      	vxorps	%xmm0, %xmm0, %xmm0
      24:      	xorl	%eax, %eax
      26:      	nopw	%cs:(%rax,%rax)
      30:      	vaddss	(%rdi,%rax,4), %xmm0, %xmm0
      35:      	incq	%rax
      38:      	cmpq	$0x4, %rax
      3c:      	jne	0x30 <forge_reduce_sum_1+0x10>
      3e:      	retq
~~~

</details>

<details>
<summary>Result</summary>

Verified scalar output

~~~json
4.0
~~~

</details>

</details>

<details>
<summary>Small matrix product</summary>

| Plan | Loops | Temporary arrays |
| --- | --- | --- |
| unfused | 1 | 0 |
| fused | 1 | 0 |

<details>
<summary>Input and rerun command</summary>

~~~sh
python -m forge 'a @ b' --array 'a=[[1.0,2.0,3.0],[4.0,5.0,6.0]]' --array 'b=[[1.0,2.0],[3.0,4.0],[5.0,6.0]]'
~~~

~~~text
a (2 × 3) = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
b (3 × 2) = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
~~~

</details>

<details>
<summary>Typed graph</summary>

~~~text
%0 = input(0) : f32[2, 3] @1:1 uses=1
%1 = input(1) : f32[3, 2] @1:5 uses=1
%2 = matmul(%0, %1) : f32[2, 2] @1:1 uses=0
return %2
~~~

</details>

<details>
<summary>Loop plan</summary>

**Unfused**

~~~text
loops=1 temporary_arrays=0
loop %2 -> output
  %2 = matmul(%0, %1)
~~~

**Fused**

~~~text
loops=1 temporary_arrays=0
loop %2 -> output
  %2 = matmul(%0, %1)
~~~

</details>

<details>
<summary>Optimized LLVM IR</summary>

**Unfused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %inputs, i64 8
  %2 = load ptr, ptr %1, align 8
  %3 = load ptr, ptr %buffers, align 8
  tail call void @forge_matmul_2(ptr %3, ptr %0, ptr %2)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_matmul_2(ptr noalias %0, ptr readonly %1, ptr readonly %2) unnamed_addr #0 {
entry:
  store float 0.000000e+00, ptr %0, align 1
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store float 0.000000e+00, ptr %3, align 1
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store float 0.000000e+00, ptr %4, align 1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 12
  store float 0.000000e+00, ptr %5, align 1
  br label %row.body

row.body:                                         ; preds = %entry, %inner.exit
  %6 = phi i1 [ false, %inner.exit ], [ true, %entry ]
  %row = phi i64 [ 1, %inner.exit ], [ 0, %entry ]
  %.idx = mul nuw nsw i64 %row, 12
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 %.idx
  %.idx4 = shl nuw nsw i64 %row, 3
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 %.idx4
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 4
  %.promoted = load float, ptr %8, align 1
  %.promoted6 = load float, ptr %9, align 1
  br label %inner.body

row.exit:                                         ; preds = %inner.exit
  ret void

inner.body:                                       ; preds = %row.body, %inner.body
  %10 = phi float [ %.promoted6, %row.body ], [ %sum.c, %inner.body ]
  %sum5 = phi float [ %.promoted, %row.body ], [ %sum, %inner.body ]
  %inner = phi i64 [ 0, %row.body ], [ %17, %inner.body ]
  %11 = getelementptr inbounds nuw [4 x i8], ptr %7, i64 %inner
  %12 = load float, ptr %11, align 1
  %.idx3 = shl nuw nsw i64 %inner, 3
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 %.idx3
  %14 = load float, ptr %13, align 1
  %product = fmul float %12, %14
  %sum = fadd float %sum5, %product
  %15 = getelementptr inbounds nuw i8, ptr %13, i64 4
  %16 = load float, ptr %15, align 1
  %product.c = fmul float %12, %16
  %sum.c = fadd float %10, %product.c
  %17 = add nuw nsw i64 %inner, 1
  %exitcond.not = icmp eq i64 %17, 3
  br i1 %exitcond.not, label %inner.exit, label %inner.body

inner.exit:                                       ; preds = %inner.body
  store float %sum, ptr %8, align 1
  store float %sum.c, ptr %9, align 1
  br i1 %6, label %row.body, label %row.exit
}

attributes #0 = { nobuiltin noinline }
~~~

**Fused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %inputs, i64 8
  %2 = load ptr, ptr %1, align 8
  %3 = load ptr, ptr %buffers, align 8
  tail call void @forge_matmul_2(ptr %3, ptr %0, ptr %2)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_matmul_2(ptr noalias %0, ptr readonly %1, ptr readonly %2) unnamed_addr #0 {
entry:
  store float 0.000000e+00, ptr %0, align 1
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store float 0.000000e+00, ptr %3, align 1
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store float 0.000000e+00, ptr %4, align 1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 12
  store float 0.000000e+00, ptr %5, align 1
  br label %row.body

row.body:                                         ; preds = %entry, %inner.exit
  %6 = phi i1 [ false, %inner.exit ], [ true, %entry ]
  %row = phi i64 [ 1, %inner.exit ], [ 0, %entry ]
  %.idx = mul nuw nsw i64 %row, 12
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 %.idx
  %.idx4 = shl nuw nsw i64 %row, 3
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 %.idx4
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 4
  %.promoted = load float, ptr %8, align 1
  %.promoted6 = load float, ptr %9, align 1
  br label %inner.body

row.exit:                                         ; preds = %inner.exit
  ret void

inner.body:                                       ; preds = %row.body, %inner.body
  %10 = phi float [ %.promoted6, %row.body ], [ %sum.c, %inner.body ]
  %sum5 = phi float [ %.promoted, %row.body ], [ %sum, %inner.body ]
  %inner = phi i64 [ 0, %row.body ], [ %17, %inner.body ]
  %11 = getelementptr inbounds nuw [4 x i8], ptr %7, i64 %inner
  %12 = load float, ptr %11, align 1
  %.idx3 = shl nuw nsw i64 %inner, 3
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 %.idx3
  %14 = load float, ptr %13, align 1
  %product = fmul float %12, %14
  %sum = fadd float %sum5, %product
  %15 = getelementptr inbounds nuw i8, ptr %13, i64 4
  %16 = load float, ptr %15, align 1
  %product.c = fmul float %12, %16
  %sum.c = fadd float %10, %product.c
  %17 = add nuw nsw i64 %inner, 1
  %exitcond.not = icmp eq i64 %17, 3
  br i1 %exitcond.not, label %inner.exit, label %inner.body

inner.exit:                                       ; preds = %inner.body
  store float %sum, ptr %8, align 1
  store float %sum.c, ptr %9, align 1
  br i1 %6, label %row.body, label %row.exit
}

attributes #0 = { nobuiltin noinline }
~~~

</details>

<details>
<summary>LLJIT disassembly</summary>

**Unfused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	movq	(%rsi), %rax
       3:      	movq	0x8(%rsi), %rcx
       7:      	movq	(%rdx), %rdi
       a:      	movq	%rax, %rsi
       d:      	movq	%rcx, %rdx
      10:      	jmp	0x20 <forge_matmul_2>
      12:      	nopw	%cs:(%rax,%rax)

0000000000000020 <forge_matmul_2>:
      20:      	vxorps	%xmm0, %xmm0, %xmm0
      24:      	vmovups	%xmm0, (%rdi)
      28:      	movb	$0x1, %al
      2a:      	xorl	%ecx, %ecx
      2c:      	nopl	(%rax)
      30:      	leaq	(%rcx,%rcx,2), %r8
      34:      	leaq	(%rsi,%r8,4), %r8
      38:      	vmovss	(%rdi,%rcx,8), %xmm1
      3d:      	vmovss	0x4(%rdi,%rcx,8), %xmm0
      43:      	xorl	%r9d, %r9d
      46:      	nopw	%cs:(%rax,%rax)
      50:      	vmovss	(%r8,%r9,4), %xmm2
      56:      	vmulss	(%rdx,%r9,8), %xmm2, %xmm3
      5c:      	vmulss	0x4(%rdx,%r9,8), %xmm2, %xmm2
      63:      	vaddss	%xmm3, %xmm1, %xmm1
      67:      	vaddss	%xmm2, %xmm0, %xmm0
      6b:      	incq	%r9
      6e:      	cmpq	$0x3, %r9
      72:      	jne	0x50 <forge_matmul_2+0x30>
      74:      	vmovss	%xmm1, (%rdi,%rcx,8)
      79:      	vmovss	%xmm0, 0x4(%rdi,%rcx,8)
      7f:      	movl	$0x1, %ecx
      84:      	testb	$0x1, %al
      86:      	movl	$0x0, %eax
      8b:      	jne	0x30 <forge_matmul_2+0x10>
      8d:      	retq
~~~

**Fused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	movq	(%rsi), %rax
       3:      	movq	0x8(%rsi), %rcx
       7:      	movq	(%rdx), %rdi
       a:      	movq	%rax, %rsi
       d:      	movq	%rcx, %rdx
      10:      	jmp	0x20 <forge_matmul_2>
      12:      	nopw	%cs:(%rax,%rax)

0000000000000020 <forge_matmul_2>:
      20:      	vxorps	%xmm0, %xmm0, %xmm0
      24:      	vmovups	%xmm0, (%rdi)
      28:      	movb	$0x1, %al
      2a:      	xorl	%ecx, %ecx
      2c:      	nopl	(%rax)
      30:      	leaq	(%rcx,%rcx,2), %r8
      34:      	leaq	(%rsi,%r8,4), %r8
      38:      	vmovss	(%rdi,%rcx,8), %xmm1
      3d:      	vmovss	0x4(%rdi,%rcx,8), %xmm0
      43:      	xorl	%r9d, %r9d
      46:      	nopw	%cs:(%rax,%rax)
      50:      	vmovss	(%r8,%r9,4), %xmm2
      56:      	vmulss	(%rdx,%r9,8), %xmm2, %xmm3
      5c:      	vmulss	0x4(%rdx,%r9,8), %xmm2, %xmm2
      63:      	vaddss	%xmm3, %xmm1, %xmm1
      67:      	vaddss	%xmm2, %xmm0, %xmm0
      6b:      	incq	%r9
      6e:      	cmpq	$0x3, %r9
      72:      	jne	0x50 <forge_matmul_2+0x30>
      74:      	vmovss	%xmm1, (%rdi,%rcx,8)
      79:      	vmovss	%xmm0, 0x4(%rdi,%rcx,8)
      7f:      	movl	$0x1, %ecx
      84:      	testb	$0x1, %al
      86:      	movl	$0x0, %eax
      8b:      	jne	0x30 <forge_matmul_2+0x10>
      8d:      	retq
~~~

</details>

<details>
<summary>Result</summary>

Verified 2 × 2 output

~~~json
[[22.0, 28.0], [49.0, 64.0]]
~~~

</details>

</details>

<details>
<summary>Matrix tile edges</summary>

| Plan | Loops | Temporary arrays |
| --- | --- | --- |
| unfused | 1 | 0 |
| fused | 1 | 0 |

<details>
<summary>Input and rerun command</summary>

~~~sh
python showcase/capture.py --out captures.json
~~~

Input values are a sample from the full capture.

~~~text
a (17 × 33) = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0]
b (33 × 35) = [-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0]
~~~

</details>

<details>
<summary>Typed graph</summary>

~~~text
%0 = input(0) : f32[17, 33] @1:1 uses=1
%1 = input(1) : f32[33, 35] @1:5 uses=1
%2 = matmul(%0, %1) : f32[17, 35] @1:1 uses=0
return %2
~~~

</details>

<details>
<summary>Loop plan</summary>

**Unfused**

~~~text
loops=1 temporary_arrays=0
loop %2 -> output
  %2 = matmul(%0, %1)
~~~

**Fused**

~~~text
loops=1 temporary_arrays=0
loop %2 -> output
  %2 = matmul(%0, %1)
~~~

</details>

<details>
<summary>Optimized LLVM IR</summary>

**Unfused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %inputs, i64 8
  %2 = load ptr, ptr %1, align 8
  %3 = load ptr, ptr %buffers, align 8
  tail call void @forge_matmul_2(ptr %3, ptr %0, ptr %2)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_matmul_2(ptr noalias %0, ptr readonly %1, ptr readonly %2) unnamed_addr #0 {
entry:
  br label %row.tile.body

row.tile.body:                                    ; preds = %entry, %col.tile.exit
  %3 = phi i64 [ 16, %entry ], [ 17, %col.tile.exit ]
  %4 = phi i1 [ true, %entry ], [ false, %col.tile.exit ]
  %row.tile = phi i64 [ 0, %entry ], [ 16, %col.tile.exit ]
  %5 = mul nuw nsw i64 %row.tile, 140
  %6 = getelementptr i8, ptr %0, i64 %5
  br label %col.tile.body

row.tile.exit:                                    ; preds = %col.tile.exit
  ret void

col.tile.body:                                    ; preds = %row.tile.body, %inner.tile.exit
  %7 = phi i1 [ true, %row.tile.body ], [ false, %inner.tile.exit ]
  %col.tile = phi i64 [ 0, %row.tile.body ], [ 32, %inner.tile.exit ]
  %8 = tail call i64 @llvm.umin.i64(i64 %col.tile, i64 3)
  %9 = or disjoint i64 %8, 32
  %10 = shl nuw nsw i64 %col.tile, 2
  %11 = or disjoint i64 %col.tile, 1
  %umax = tail call i64 @llvm.umax.i64(i64 %9, i64 %11)
  %12 = sub nuw nsw i64 %umax, %col.tile
  %13 = shl nuw nsw i64 %12, 2
  %14 = getelementptr i8, ptr %6, i64 %10
  br label %zero.row.body

col.tile.exit:                                    ; preds = %inner.tile.exit
  br i1 %4, label %row.tile.body, label %row.tile.exit

zero.row.body:                                    ; preds = %col.tile.body, %zero.row.body
  %indvar = phi i64 [ 0, %col.tile.body ], [ %indvar.next, %zero.row.body ]
  %zero.row = phi i64 [ %row.tile, %col.tile.body ], [ %16, %zero.row.body ]
  %15 = mul nuw nsw i64 %indvar, 140
  %scevgep = getelementptr i8, ptr %14, i64 %15
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep, i8 0, i64 %13, i1 false)
  %16 = add nuw nsw i64 %zero.row, 1
  %17 = icmp samesign ult i64 %16, %3
  %indvar.next = add nuw nsw i64 %indvar, 1
  br i1 %17, label %zero.row.body, label %inner.tile.body.preheader

inner.tile.body.preheader:                        ; preds = %zero.row.body
  %18 = or disjoint i64 %8, 32
  %19 = or disjoint i64 %col.tile, 1
  %20 = tail call i64 @llvm.umax.i64(i64 %18, i64 %19)
  %21 = sub nuw nsw i64 %20, %col.tile
  %min.iters.check = icmp samesign ult i64 %21, 8
  %n.mod.vf = and i64 %20, 3
  %n.vec = sub nuw nsw i64 %21, %n.mod.vf
  %22 = add nuw nsw i64 %col.tile, %n.vec
  %cmp.n = icmp eq i64 %n.mod.vf, 0
  br label %inner.tile.body

inner.tile.body:                                  ; preds = %inner.tile.body.preheader, %row.exit
  %23 = phi i64 [ 33, %row.exit ], [ 32, %inner.tile.body.preheader ]
  %24 = phi i1 [ false, %row.exit ], [ true, %inner.tile.body.preheader ]
  %inner.tile = phi i64 [ 32, %row.exit ], [ 0, %inner.tile.body.preheader ]
  br label %row.body

inner.tile.exit:                                  ; preds = %row.exit
  br i1 %7, label %col.tile.body, label %col.tile.exit

row.body:                                         ; preds = %inner.tile.body, %inner.exit
  %row = phi i64 [ %row.tile, %inner.tile.body ], [ %36, %inner.exit ]
  %.idx6 = mul nuw nsw i64 %row, 132
  %25 = getelementptr i8, ptr %1, i64 %.idx6
  %.idx8 = mul nuw nsw i64 %row, 140
  %26 = getelementptr i8, ptr %0, i64 %.idx8
  br label %inner.body

row.exit:                                         ; preds = %inner.exit
  br i1 %24, label %inner.tile.body, label %inner.tile.exit

inner.body:                                       ; preds = %row.body, %col.exit
  %inner = phi i64 [ %inner.tile, %row.body ], [ %43, %col.exit ]
  %27 = getelementptr [4 x i8], ptr %25, i64 %inner
  %28 = load float, ptr %27, align 1
  %.idx7 = mul nuw nsw i64 %inner, 140
  %29 = getelementptr i8, ptr %2, i64 %.idx7
  br i1 %min.iters.check, label %col.body.preheader, label %vector.ph

vector.ph:                                        ; preds = %inner.body
  %broadcast.splatinsert = insertelement <8 x float> poison, float %28, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %30 = add i64 %col.tile, %index
  %31 = getelementptr [4 x i8], ptr %29, i64 %30
  %wide.load = load <8 x float>, ptr %31, align 1
  %32 = getelementptr [4 x i8], ptr %26, i64 %30
  %33 = fmul <8 x float> %broadcast.splat, %wide.load
  %wide.load12 = load <8 x float>, ptr %32, align 1
  %34 = fadd <8 x float> %wide.load12, %33
  store <8 x float> %34, ptr %32, align 1
  %index.next = add nuw i64 %index, 8
  %35 = icmp eq i64 %index.next, %n.vec
  br i1 %35, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %col.exit, label %col.body.preheader

col.body.preheader:                               ; preds = %inner.body, %middle.block
  %col.ph = phi i64 [ %col.tile, %inner.body ], [ %22, %middle.block ]
  br label %col.body

inner.exit:                                       ; preds = %col.exit
  %36 = add nuw nsw i64 %row, 1
  %exitcond9.not = icmp eq i64 %36, %3
  br i1 %exitcond9.not, label %row.exit, label %row.body

col.body:                                         ; preds = %col.body.preheader, %col.body
  %col = phi i64 [ %41, %col.body ], [ %col.ph, %col.body.preheader ]
  %37 = getelementptr [4 x i8], ptr %29, i64 %col
  %38 = load float, ptr %37, align 1
  %39 = getelementptr [4 x i8], ptr %26, i64 %col
  %product = fmul float %28, %38
  %40 = load float, ptr %39, align 1
  %sum = fadd float %40, %product
  store float %sum, ptr %39, align 1
  %41 = add nuw nsw i64 %col, 1
  %42 = icmp samesign ult i64 %41, %9
  br i1 %42, label %col.body, label %col.exit, !llvm.loop !4

col.exit:                                         ; preds = %col.body, %middle.block
  %43 = add nuw nsw i64 %inner, 1
  %exitcond.not = icmp eq i64 %43, %23
  br i1 %exitcond.not, label %inner.exit, label %inner.body
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
~~~

**Fused**

~~~text
; ModuleID = 'forge'
source_filename = "forge"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @forge_kernel(i64 %count, ptr nofree readonly captures(none) %inputs, ptr nofree readonly captures(none) %buffers) local_unnamed_addr {
entry:
  %0 = load ptr, ptr %inputs, align 8
  %1 = getelementptr inbounds nuw i8, ptr %inputs, i64 8
  %2 = load ptr, ptr %1, align 8
  %3 = load ptr, ptr %buffers, align 8
  tail call void @forge_matmul_2(ptr %3, ptr %0, ptr %2)
  ret void
}

; Function Attrs: nobuiltin noinline
define internal void @forge_matmul_2(ptr noalias %0, ptr readonly %1, ptr readonly %2) unnamed_addr #0 {
entry:
  br label %row.tile.body

row.tile.body:                                    ; preds = %entry, %col.tile.exit
  %3 = phi i64 [ 16, %entry ], [ 17, %col.tile.exit ]
  %4 = phi i1 [ true, %entry ], [ false, %col.tile.exit ]
  %row.tile = phi i64 [ 0, %entry ], [ 16, %col.tile.exit ]
  %5 = mul nuw nsw i64 %row.tile, 140
  %6 = getelementptr i8, ptr %0, i64 %5
  br label %col.tile.body

row.tile.exit:                                    ; preds = %col.tile.exit
  ret void

col.tile.body:                                    ; preds = %row.tile.body, %inner.tile.exit
  %7 = phi i1 [ true, %row.tile.body ], [ false, %inner.tile.exit ]
  %col.tile = phi i64 [ 0, %row.tile.body ], [ 32, %inner.tile.exit ]
  %8 = tail call i64 @llvm.umin.i64(i64 %col.tile, i64 3)
  %9 = or disjoint i64 %8, 32
  %10 = shl nuw nsw i64 %col.tile, 2
  %11 = or disjoint i64 %col.tile, 1
  %umax = tail call i64 @llvm.umax.i64(i64 %9, i64 %11)
  %12 = sub nuw nsw i64 %umax, %col.tile
  %13 = shl nuw nsw i64 %12, 2
  %14 = getelementptr i8, ptr %6, i64 %10
  br label %zero.row.body

col.tile.exit:                                    ; preds = %inner.tile.exit
  br i1 %4, label %row.tile.body, label %row.tile.exit

zero.row.body:                                    ; preds = %col.tile.body, %zero.row.body
  %indvar = phi i64 [ 0, %col.tile.body ], [ %indvar.next, %zero.row.body ]
  %zero.row = phi i64 [ %row.tile, %col.tile.body ], [ %16, %zero.row.body ]
  %15 = mul nuw nsw i64 %indvar, 140
  %scevgep = getelementptr i8, ptr %14, i64 %15
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep, i8 0, i64 %13, i1 false)
  %16 = add nuw nsw i64 %zero.row, 1
  %17 = icmp samesign ult i64 %16, %3
  %indvar.next = add nuw nsw i64 %indvar, 1
  br i1 %17, label %zero.row.body, label %inner.tile.body.preheader

inner.tile.body.preheader:                        ; preds = %zero.row.body
  %18 = or disjoint i64 %8, 32
  %19 = or disjoint i64 %col.tile, 1
  %20 = tail call i64 @llvm.umax.i64(i64 %18, i64 %19)
  %21 = sub nuw nsw i64 %20, %col.tile
  %min.iters.check = icmp samesign ult i64 %21, 8
  %n.mod.vf = and i64 %20, 3
  %n.vec = sub nuw nsw i64 %21, %n.mod.vf
  %22 = add nuw nsw i64 %col.tile, %n.vec
  %cmp.n = icmp eq i64 %n.mod.vf, 0
  br label %inner.tile.body

inner.tile.body:                                  ; preds = %inner.tile.body.preheader, %row.exit
  %23 = phi i64 [ 33, %row.exit ], [ 32, %inner.tile.body.preheader ]
  %24 = phi i1 [ false, %row.exit ], [ true, %inner.tile.body.preheader ]
  %inner.tile = phi i64 [ 32, %row.exit ], [ 0, %inner.tile.body.preheader ]
  br label %row.body

inner.tile.exit:                                  ; preds = %row.exit
  br i1 %7, label %col.tile.body, label %col.tile.exit

row.body:                                         ; preds = %inner.tile.body, %inner.exit
  %row = phi i64 [ %row.tile, %inner.tile.body ], [ %36, %inner.exit ]
  %.idx6 = mul nuw nsw i64 %row, 132
  %25 = getelementptr i8, ptr %1, i64 %.idx6
  %.idx8 = mul nuw nsw i64 %row, 140
  %26 = getelementptr i8, ptr %0, i64 %.idx8
  br label %inner.body

row.exit:                                         ; preds = %inner.exit
  br i1 %24, label %inner.tile.body, label %inner.tile.exit

inner.body:                                       ; preds = %row.body, %col.exit
  %inner = phi i64 [ %inner.tile, %row.body ], [ %43, %col.exit ]
  %27 = getelementptr [4 x i8], ptr %25, i64 %inner
  %28 = load float, ptr %27, align 1
  %.idx7 = mul nuw nsw i64 %inner, 140
  %29 = getelementptr i8, ptr %2, i64 %.idx7
  br i1 %min.iters.check, label %col.body.preheader, label %vector.ph

vector.ph:                                        ; preds = %inner.body
  %broadcast.splatinsert = insertelement <8 x float> poison, float %28, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %30 = add i64 %col.tile, %index
  %31 = getelementptr [4 x i8], ptr %29, i64 %30
  %wide.load = load <8 x float>, ptr %31, align 1
  %32 = getelementptr [4 x i8], ptr %26, i64 %30
  %33 = fmul <8 x float> %broadcast.splat, %wide.load
  %wide.load12 = load <8 x float>, ptr %32, align 1
  %34 = fadd <8 x float> %wide.load12, %33
  store <8 x float> %34, ptr %32, align 1
  %index.next = add nuw i64 %index, 8
  %35 = icmp eq i64 %index.next, %n.vec
  br i1 %35, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %col.exit, label %col.body.preheader

col.body.preheader:                               ; preds = %inner.body, %middle.block
  %col.ph = phi i64 [ %col.tile, %inner.body ], [ %22, %middle.block ]
  br label %col.body

inner.exit:                                       ; preds = %col.exit
  %36 = add nuw nsw i64 %row, 1
  %exitcond9.not = icmp eq i64 %36, %3
  br i1 %exitcond9.not, label %row.exit, label %row.body

col.body:                                         ; preds = %col.body.preheader, %col.body
  %col = phi i64 [ %41, %col.body ], [ %col.ph, %col.body.preheader ]
  %37 = getelementptr [4 x i8], ptr %29, i64 %col
  %38 = load float, ptr %37, align 1
  %39 = getelementptr [4 x i8], ptr %26, i64 %col
  %product = fmul float %28, %38
  %40 = load float, ptr %39, align 1
  %sum = fadd float %40, %product
  store float %sum, ptr %39, align 1
  %41 = add nuw nsw i64 %col, 1
  %42 = icmp samesign ult i64 %41, %9
  br i1 %42, label %col.body, label %col.exit, !llvm.loop !4

col.exit:                                         ; preds = %col.body, %middle.block
  %43 = add nuw nsw i64 %inner, 1
  %exitcond.not = icmp eq i64 %43, %23
  br i1 %exitcond.not, label %inner.exit, label %inner.body
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

attributes #0 = { nobuiltin noinline }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.body", i32 1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1, !5}
!5 = !{!"llvm.loop.vectorize.epilogue", i32 1}
~~~

</details>

<details>
<summary>LLJIT disassembly</summary>

**Unfused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	movq	(%rsi), %rax
       3:      	movq	0x8(%rsi), %rcx
       7:      	movq	(%rdx), %rdi
       a:      	movq	%rax, %rsi
       d:      	movq	%rcx, %rdx
      10:      	jmp	0x20 <forge_matmul_2>
      12:      	nopw	%cs:(%rax,%rax)

0000000000000020 <forge_matmul_2>:
      20:      	pushq	%rbp
      21:      	pushq	%r15
      23:      	pushq	%r14
      25:      	pushq	%r13
      27:      	pushq	%r12
      29:      	pushq	%rbx
      2a:      	subq	$0x38, %rsp
      2e:      	movq	%rdx, 0x28(%rsp)
      33:      	movq	%rsi, 0x30(%rsp)
      38:      	movq	%rdi, 0x20(%rsp)
      3d:      	movb	$0x1, %al
      3f:      	movl	$0x10, %r15d
      45:      	xorl	%ebp, %ebp
      47:      	jmp	0x66 <forge_matmul_2+0x46>
      49:      	nopl	(%rax)
      50:      	movl	$0x10, %ebp
      55:      	movl	$0x11, %r15d
      5b:      	testb	$0x1, 0x8(%rsp)
      60:      	je	0x21d <forge_matmul_2+0x1fd>
      66:      	movl	%eax, 0x8(%rsp)
      6a:      	imulq	$0x8c, %rbp, %rax
      71:      	addq	0x20(%rsp), %rax
      76:      	movq	%rax, 0x10(%rsp)
      7b:      	movb	$0x1, %al
      7d:      	xorl	%r14d, %r14d
      80:      	movq	%rbp, 0x18(%rsp)
      85:      	jmp	0xa7 <forge_matmul_2+0x87>
      87:      	nopw	(%rax,%rax)
      90:      	movl	$0x20, %r14d
      96:      	testb	$0x1, 0xc(%rsp)
      9b:      	movl	$0x0, %eax
      a0:      	movq	0x18(%rsp), %rbp
      a5:      	je	0x50 <forge_matmul_2+0x30>
      a7:      	movl	%eax, 0xc(%rsp)
      ab:      	cmpq	$0x3, %r14
      af:      	movl	$0x3, %ebx
      b4:      	cmovbq	%r14, %rbx
      b8:      	orq	$0x20, %rbx
      bc:      	leaq	0x1(%r14), %r12
      c0:      	cmpq	%r12, %rbx
      c3:      	movq	%r12, %rax
      c6:      	cmovaq	%rbx, %rax
      ca:      	subq	%r14, %rax
      cd:      	shlq	$0x2, %rax
      d1:      	movq	%rax, (%rsp)
      d5:      	movq	0x10(%rsp), %rax
      da:      	leaq	(%rax,%r14,4), %r13
      de:      	nop
      e0:      	movq	%r13, %rdi
      e3:      	xorl	%esi, %esi
      e5:      	movq	(%rsp), %rdx
      e9:      	vzeroupper
      ec:      	callq	0xf1 <forge_matmul_2+0xd1>
      f1:      	incq	%rbp
      f4:      	addq	$0x8c, %r13
      fb:      	cmpq	%r15, %rbp
      fe:      	jb	0xe0 <forge_matmul_2+0xc0>
     100:      	cmpq	%r12, %rbx
     103:      	cmovaq	%rbx, %r12
     107:      	movq	%r12, %rax
     10a:      	subq	%r14, %rax
     10d:      	movl	%r12d, %ecx
     110:      	andl	$0x3, %ecx
     113:      	andq	$-0x4, %r12
     117:      	movb	$0x1, %dil
     11a:      	movl	$0x20, %esi
     11f:      	xorl	%edx, %edx
     121:      	jmp	0x149 <forge_matmul_2+0x129>
     123:      	nopw	%cs:(%rax,%rax)
     130:      	movl	$0x20, %edx
     135:      	movl	$0x21, %esi
     13a:      	testb	$0x1, (%rsp)
     13e:      	movl	$0x0, %edi
     143:      	je	0x90 <forge_matmul_2+0x70>
     149:      	movl	%edi, (%rsp)
     14c:      	imulq	$0x8c, %rdx, %r8
     153:      	addq	0x28(%rsp), %r8
     158:      	movq	0x10(%rsp), %r9
     15d:      	movq	0x18(%rsp), %r10
     162:      	jmp	0x17f <forge_matmul_2+0x15f>
     164:      	nopw	%cs:(%rax,%rax)
     170:      	incq	%r10
     173:      	addq	$0x8c, %r9
     17a:      	cmpq	%r15, %r10
     17d:      	je	0x130 <forge_matmul_2+0x110>
     17f:      	movq	%r10, %rdi
     182:      	shlq	$0x7, %rdi
     186:      	leaq	(%rdi,%r10,4), %r11
     18a:      	addq	0x30(%rsp), %r11
     18f:      	movq	%r8, %r13
     192:      	movq	%rdx, %rdi
     195:      	jmp	0x1af <forge_matmul_2+0x18f>
     197:      	nopw	(%rax,%rax)
     1a0:      	incq	%rdi
     1a3:      	addq	$0x8c, %r13
     1aa:      	cmpq	%rsi, %rdi
     1ad:      	je	0x170 <forge_matmul_2+0x150>
     1af:      	vmovss	(%r11,%rdi,4), %xmm0
     1b5:      	movq	%r14, %rbp
     1b8:      	cmpq	$0x8, %rax
     1bc:      	jb	0x200 <forge_matmul_2+0x1e0>
     1be:      	vbroadcastss	%xmm0, %ymm1
     1c3:      	movq	%r14, %rbp
     1c6:      	nopw	%cs:(%rax,%rax)
     1d0:      	vmulps	(%r13,%rbp,4), %ymm1, %ymm2
     1d7:      	vaddps	(%r9,%rbp,4), %ymm2, %ymm2
     1dd:      	vmovups	%ymm2, (%r9,%rbp,4)
     1e3:      	addq	$0x8, %rbp
     1e7:      	cmpq	%rbp, %r12
     1ea:      	jne	0x1d0 <forge_matmul_2+0x1b0>
     1ec:      	movq	%r12, %rbp
     1ef:      	testq	%rcx, %rcx
     1f2:      	je	0x1a0 <forge_matmul_2+0x180>
     1f4:      	nopw	%cs:(%rax,%rax)
     200:      	vmulss	(%r13,%rbp,4), %xmm0, %xmm1
     207:      	vaddss	(%r9,%rbp,4), %xmm1, %xmm1
     20d:      	vmovss	%xmm1, (%r9,%rbp,4)
     213:      	incq	%rbp
     216:      	cmpq	%rbx, %rbp
     219:      	jb	0x200 <forge_matmul_2+0x1e0>
     21b:      	jmp	0x1a0 <forge_matmul_2+0x180>
     21d:      	addq	$0x38, %rsp
     221:      	popq	%rbx
     222:      	popq	%r12
     224:      	popq	%r13
     226:      	popq	%r14
     228:      	popq	%r15
     22a:      	popq	%rbp
     22b:      	vzeroupper
     22e:      	retq
~~~

**Fused**

~~~text
kernel.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <forge_kernel>:
       0:      	movq	(%rsi), %rax
       3:      	movq	0x8(%rsi), %rcx
       7:      	movq	(%rdx), %rdi
       a:      	movq	%rax, %rsi
       d:      	movq	%rcx, %rdx
      10:      	jmp	0x20 <forge_matmul_2>
      12:      	nopw	%cs:(%rax,%rax)

0000000000000020 <forge_matmul_2>:
      20:      	pushq	%rbp
      21:      	pushq	%r15
      23:      	pushq	%r14
      25:      	pushq	%r13
      27:      	pushq	%r12
      29:      	pushq	%rbx
      2a:      	subq	$0x38, %rsp
      2e:      	movq	%rdx, 0x28(%rsp)
      33:      	movq	%rsi, 0x30(%rsp)
      38:      	movq	%rdi, 0x20(%rsp)
      3d:      	movb	$0x1, %al
      3f:      	movl	$0x10, %r15d
      45:      	xorl	%ebp, %ebp
      47:      	jmp	0x66 <forge_matmul_2+0x46>
      49:      	nopl	(%rax)
      50:      	movl	$0x10, %ebp
      55:      	movl	$0x11, %r15d
      5b:      	testb	$0x1, 0x8(%rsp)
      60:      	je	0x21d <forge_matmul_2+0x1fd>
      66:      	movl	%eax, 0x8(%rsp)
      6a:      	imulq	$0x8c, %rbp, %rax
      71:      	addq	0x20(%rsp), %rax
      76:      	movq	%rax, 0x10(%rsp)
      7b:      	movb	$0x1, %al
      7d:      	xorl	%r14d, %r14d
      80:      	movq	%rbp, 0x18(%rsp)
      85:      	jmp	0xa7 <forge_matmul_2+0x87>
      87:      	nopw	(%rax,%rax)
      90:      	movl	$0x20, %r14d
      96:      	testb	$0x1, 0xc(%rsp)
      9b:      	movl	$0x0, %eax
      a0:      	movq	0x18(%rsp), %rbp
      a5:      	je	0x50 <forge_matmul_2+0x30>
      a7:      	movl	%eax, 0xc(%rsp)
      ab:      	cmpq	$0x3, %r14
      af:      	movl	$0x3, %ebx
      b4:      	cmovbq	%r14, %rbx
      b8:      	orq	$0x20, %rbx
      bc:      	leaq	0x1(%r14), %r12
      c0:      	cmpq	%r12, %rbx
      c3:      	movq	%r12, %rax
      c6:      	cmovaq	%rbx, %rax
      ca:      	subq	%r14, %rax
      cd:      	shlq	$0x2, %rax
      d1:      	movq	%rax, (%rsp)
      d5:      	movq	0x10(%rsp), %rax
      da:      	leaq	(%rax,%r14,4), %r13
      de:      	nop
      e0:      	movq	%r13, %rdi
      e3:      	xorl	%esi, %esi
      e5:      	movq	(%rsp), %rdx
      e9:      	vzeroupper
      ec:      	callq	0xf1 <forge_matmul_2+0xd1>
      f1:      	incq	%rbp
      f4:      	addq	$0x8c, %r13
      fb:      	cmpq	%r15, %rbp
      fe:      	jb	0xe0 <forge_matmul_2+0xc0>
     100:      	cmpq	%r12, %rbx
     103:      	cmovaq	%rbx, %r12
     107:      	movq	%r12, %rax
     10a:      	subq	%r14, %rax
     10d:      	movl	%r12d, %ecx
     110:      	andl	$0x3, %ecx
     113:      	andq	$-0x4, %r12
     117:      	movb	$0x1, %dil
     11a:      	movl	$0x20, %esi
     11f:      	xorl	%edx, %edx
     121:      	jmp	0x149 <forge_matmul_2+0x129>
     123:      	nopw	%cs:(%rax,%rax)
     130:      	movl	$0x20, %edx
     135:      	movl	$0x21, %esi
     13a:      	testb	$0x1, (%rsp)
     13e:      	movl	$0x0, %edi
     143:      	je	0x90 <forge_matmul_2+0x70>
     149:      	movl	%edi, (%rsp)
     14c:      	imulq	$0x8c, %rdx, %r8
     153:      	addq	0x28(%rsp), %r8
     158:      	movq	0x10(%rsp), %r9
     15d:      	movq	0x18(%rsp), %r10
     162:      	jmp	0x17f <forge_matmul_2+0x15f>
     164:      	nopw	%cs:(%rax,%rax)
     170:      	incq	%r10
     173:      	addq	$0x8c, %r9
     17a:      	cmpq	%r15, %r10
     17d:      	je	0x130 <forge_matmul_2+0x110>
     17f:      	movq	%r10, %rdi
     182:      	shlq	$0x7, %rdi
     186:      	leaq	(%rdi,%r10,4), %r11
     18a:      	addq	0x30(%rsp), %r11
     18f:      	movq	%r8, %r13
     192:      	movq	%rdx, %rdi
     195:      	jmp	0x1af <forge_matmul_2+0x18f>
     197:      	nopw	(%rax,%rax)
     1a0:      	incq	%rdi
     1a3:      	addq	$0x8c, %r13
     1aa:      	cmpq	%rsi, %rdi
     1ad:      	je	0x170 <forge_matmul_2+0x150>
     1af:      	vmovss	(%r11,%rdi,4), %xmm0
     1b5:      	movq	%r14, %rbp
     1b8:      	cmpq	$0x8, %rax
     1bc:      	jb	0x200 <forge_matmul_2+0x1e0>
     1be:      	vbroadcastss	%xmm0, %ymm1
     1c3:      	movq	%r14, %rbp
     1c6:      	nopw	%cs:(%rax,%rax)
     1d0:      	vmulps	(%r13,%rbp,4), %ymm1, %ymm2
     1d7:      	vaddps	(%r9,%rbp,4), %ymm2, %ymm2
     1dd:      	vmovups	%ymm2, (%r9,%rbp,4)
     1e3:      	addq	$0x8, %rbp
     1e7:      	cmpq	%rbp, %r12
     1ea:      	jne	0x1d0 <forge_matmul_2+0x1b0>
     1ec:      	movq	%r12, %rbp
     1ef:      	testq	%rcx, %rcx
     1f2:      	je	0x1a0 <forge_matmul_2+0x180>
     1f4:      	nopw	%cs:(%rax,%rax)
     200:      	vmulss	(%r13,%rbp,4), %xmm0, %xmm1
     207:      	vaddss	(%r9,%rbp,4), %xmm1, %xmm1
     20d:      	vmovss	%xmm1, (%r9,%rbp,4)
     213:      	incq	%rbp
     216:      	cmpq	%rbx, %rbp
     219:      	jb	0x200 <forge_matmul_2+0x1e0>
     21b:      	jmp	0x1a0 <forge_matmul_2+0x180>
     21d:      	addq	$0x38, %rsp
     221:      	popq	%rbx
     222:      	popq	%r12
     224:      	popq	%r13
     226:      	popq	%r14
     228:      	popq	%r15
     22a:      	popq	%rbp
     22b:      	vzeroupper
     22e:      	retq
~~~

</details>

<details>
<summary>Result</summary>

Verified 17 × 35 output (sample)

~~~json
[[10.0, 5.0, 0.0, -5.0, -10.0, 10.0, 5.0, 0.0], [2.0, 1.0, 0.0, -1.0, -2.0, 2.0, 1.0, 0.0]]
~~~

</details>

</details>

<!-- END GENERATED TRACES -->

[Raw traces and cache capture](showcase/data/captures.json) · [Native trace CI](https://github.com/iferdous/Forge/actions/runs/45124189095)

## Try it

Once the LLVM-enabled build is on `PYTHONPATH`, run `python -m forge 'sqrt((x + 2) * 3)'` for both loop plans, the result, and the generated x86 instructions. Omit the expression for the interactive prompt. Use `--inspect ir` or `--inspect assembly` for the full output. [Build setup](docs/plan/01-toolchain-and-ci.md#reproduce-the-smoke-test) · [Playground setup](docs/plan/05-operator-fusion.md#try-an-expression)

~~~python
import forge
import numpy as np

x = np.array([1, 10, 25, 46], dtype=np.float32)
program = forge.compile("sqrt((x + 2) * 3)", inputs=("x",))
target = "avx2" if forge.has_avx2() else "scalar"
kernel = program.jit(x, target=target, cache=True)
print(kernel(x))  # [ 3.  6.  9. 12.]
~~~

## Architecture

~~~mermaid
flowchart LR
    A["Python expression"] --> B["Bounded parser"]
    B --> C["Typed tensor graph"]
    C --> D["Kernel plan and fusion"]
    D --> E["LLVM IR and vectorization"]
    E --> F["ORC LLJIT"]
    F --> G["Owned float32 result"]
~~~

Forge owns expression validation, shape inference, fusion decisions, loop scheduling, and LLVM IR lowering. LLVM supplies instruction selection, vectorization, and in-process linking. The verified AVX2 demo processes eight float32 elements per packed iteration and uses a scalar remainder. [Inspect recorded compiler stages](#compiler-traces) · [Source map](docs/showcase.md#follow-one-expression-through-the-compiler)

## Numerical contract and platforms

- Inputs are native-endian, C-contiguous rank-one or rank-two NumPy float32 arrays. Pointwise operators accept equal shapes or a scalar operand.
- Pointwise steps round separately in float32. Sum, mean, and matmul preserve their documented accumulation order. Max propagates NaNs and has a deterministic signed-zero tie rule. [Reduction and matmul contract](docs/plan/06-reductions-and-matmul.md#expression-and-numerical-contract)
- Ubuntu and Windows x64 run the native AVX2 checks. Apple Silicon runs the scalar JIT and can emit Linux x86-64 assembly for inspection. [Release handoff and test scope](docs/plan/08-release-and-handoff.md)

## Implementation plan

Stages 1–8 shipped, and the latest Ubuntu, Windows, and sanitizer jobs passed. Release-grade timings on a dedicated AVX2 host remain open. [Stage audit](docs/stage-audit.md) · [CI run](https://github.com/iferdous/Forge/actions/runs/45126633544) · [MIT license](LICENSE)

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

The HRT engineering write-up inspired Forge's focus on measuring repeated overhead and intermediate data movement. [Research and constraints](docs/research.md) · [HRT intern projects](https://www.hudsonrivertrading.com/hrtbeat/intern-spotlight-software-engineering-summer-projects/)
