# Forge Camera Preview

A camera test can produce a folder of frames that are hard to review in the shadows. This example applies a single exposure and tone preset to a folder and writes brighter PNG previews and a run report. The bundled set contains 16 different frames from a recorded TUM RGB-D camera sequence. [Source and attribution](../examples/camera_preview/data/README.md)

## Run the bundled example

From the repository root, use the Python environment and LLVM-enabled Forge module from the [Forge build instructions](../README.md#try-it).

~~~sh
.local/venv/bin/python -m pip install -r examples/camera_preview/requirements.txt
PYTHONPATH="$PWD/.local/build/release/python" .local/venv/bin/python \
  -m examples.camera_preview --out .local/camera-preview/demo --explain
~~~

On Apple Silicon, use the existing Forge build's Python directory. The Forge target defaults to the best native target available. Check the batch's actual target, frame count, compiler timings, per-frame clipping counts, and loop plans in `report.json`.

~~~sh
PYTHONPATH="$PWD/.local/build/macos-llvm/python" .local/venv/bin/python \
  -m examples.camera_preview --out .local/camera-preview/mac-demo --explain
~~~

### Use your own frames

Pass a single PNG/JPEG or a folder of such files. Names are processed in sorted order. Each output ends in `.preview.png`, and the source files remain available for comparison. Use a new output folder each time.

~~~sh
PYTHONPATH="$PWD/.local/build/release/python" .local/venv/bin/python \
  -m examples.camera_preview ./recorded-frames \
  --out .local/camera-preview/my-previews --stops 0.6 --lift 0.025 --explain
~~~

The same command accepts `--backend unfused` or `--backend numpy` for comparisons. Higher `--stops` puts more highlights at the output ceiling. `--lift` lifts dark values and changes the preview tone curve. The defaults are 0.35 stops and a lift of 0.015. The application reads ordinary 8-bit grayscale and RGB stills. Pillow applies EXIF orientation and converts supported embedded ICC profiles to sRGB. Frames without a declared color profile are treated as sRGB. Transparency, animated PNGs, higher bit depths, and unsupported color metadata produce an error. A failed batch leaves no published preview folder.

## Follow one frame

1. **Read and linearize.** Pillow decodes the image. A 256-entry lookup maps the 8-bit sRGB channel values to linear float32. For these 8-bit inputs, the lookup evaluates the same sRGB transfer function as a per-pixel formula. [Pillow formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) · [sRGB transfer function](https://www.w3.org/TR/css-color-4/#color-conversion-code)
2. **Run the pixel kernel.** Flatten the contiguous image into a view. Forge sees the expression `sqrt((x * gain + lift) * scale)`, where `gain = 2**stops` and `scale = 1 / (1 + lift)`. This is a preview tone curve. The compiled handle is reused for frames with the same number of channel values. Each batch keeps up to four compiled handles and reuses an output buffer when adjacent frame sizes match.
3. **Export.** Count channel values above the output ceiling, clip to the display range, re-encode to sRGB, and write 8-bit PNGs. The report preserves the timing boundaries and the exact formula. [PNG sRGB metadata](https://www.w3.org/TR/png-3/#11sRGB)

The source photos and output previews depict the same recorded frames. Fused and unfused Forge follow the same float32 operation order. For a 640 × 480 RGB frame, the unfused plan materializes three arrays of 921,600 float32 channel values. The fused plan performs all four arithmetic operations in one pass. This eliminates about 10.55 MiB of Forge scratch for that shape. The figure comes from the kernel workspace plan. Pillow and sRGB conversion have their own allocations.

The README gallery uses the default preset for all six frame pairs. The worked floor pixel is column 30, row 420 in `1305031467.759907.png`, counting from zero in the full 640 × 480 source. The image pipeline measured the exact source and exported RGB bytes there. The intermediate column values were recomputed in float32 and checked against the exported bytes. [Pixel data](../showcase/data/camera-preview/pixel-example.json)

## Reproduce the image study

The gallery and findings cover 48 images. Each of three TUM RGB-D recordings contributes 16 evenly spaced RGB frames, including its first and last. The featured pairs are selected for readable scene details. The graph uses every sampled frame. The desk benchmark below uses its original 16-frame workload. [Recordings, manifests, and license](../examples/camera_preview/data/README.md)

For each source and exported PNG, the study decodes RGB to linear light and calculates relative luminance at each pixel with `Y = 0.2126 R + 0.7152 G + 0.0722 B`. It reports the median of those pixel values as a percentage of reference white. The separate clipping series divides the app's count of intermediate RGB channel values above 1.0 by the total number of channel values. Clipping measures how often the preview hits the display limit, including channels already at white in the source. The measurements describe the tone preset on these recordings. A reference exposure or a task-specific quality metric would be needed to assess image enhancement quality. [sRGB and luminance calculation](https://www.w3.org/TR/css-color-4/#color-conversion-code)

~~~sh
.local/venv/bin/python -m pip install -r examples/camera_preview/requirements.txt matplotlib==3.10.8
PYTHONPATH="$PWD/.local/build/macos-llvm/python" \
  MPLCONFIGDIR="$PWD/.local/camera-preview/matplotlib" \
  .local/venv/bin/python -m showcase.camera_study --out .local/camera-preview/study
~~~

On Linux, use `.local/build/release/python` for `PYTHONPATH`. The study verifies the source hashes, runs Forge on all selected frames, and saves the full preview PNGs and `report.json` in `forge-previews/`. It also writes the gallery, graph, measured pixel, and per-frame CSV. Use a new output directory for each run. [Published per-frame data](../showcase/data/camera-preview/camera-study.csv)

## Reproduce the timing

Use native Linux x64 with AVX2 for measurements of Forge's native vector target. The tool also supports scalar measurements on the M3. The command verifies float32 bits and encoded output hashes before reporting comparative timings.

~~~sh
.local/venv/bin/python -m pip install -r examples/camera_preview/requirements.txt numexpr==2.14.2
llvm_dir=$(cat .local/llvm-dir-linux-x64.txt)
PYTHONPATH="$PWD/.local/build/release/python" \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .local/venv/bin/python -m examples.camera_preview.benchmark \
  --numexpr --require-avx2 --objdump "$llvm_dir/../../../bin/llvm-objdump" \
  --out .local/camera-preview/benchmark
~~~

The warm boundary processes the same 16 already decoded frames with reusable output buffers and compiled handles. The file-to-file boundary creates a new engine, reads the input files through the host filesystem, does the image and pixel transformations, and writes PNGs and the report. Each repetition rotates backend order. Timed file I/O uses the OS file cache. Cleanup and output-hash verification occur after the timed call. NumPy applies its four operations into one preallocated output. NumExpr uses one thread and moderate optimization with float32 constants. [NumExpr API](https://numexpr.readthedocs.io/en/latest/user_guide.html)

The benchmark output includes every timed sample, input hashes, the full file-to-file stage timings, and the fused and unfused plans. When given llvm-objdump, it also includes disassembly of the kernels that executed the frame math.

On Apple Silicon, point `PYTHONPATH` at the `macos-llvm` build and omit `--require-avx2`. Use `--objdump /usr/bin/objdump` to collect the native ARM64 scalar instructions.

### Native AVX2 measurement

The shared Ubuntu runner reported an Intel Xeon 6975P-C. Each backend ran on the same 16 recorded frames, with 21 warm and 9 file-to-file samples. These are medians in milliseconds per batch. The four backends matched NumPy's float32 bits and each produced the same PNG files. [Raw samples and hashes](../showcase/data/camera-preview/avx2.json) · [GitHub Actions run](https://github.com/iferdous/Forge/actions/runs/45511313327)

| Backend | Warm pixel math | Complete file-to-file job |
| --- | ---: | ---: |
| Forge fused AVX2 | **4.78 ms** | 530.07 ms |
| Forge unfused AVX2 | 16.45 ms | 553.77 ms |
| NumPy, preallocated | 11.41 ms | 532.64 ms |
| NumExpr, one thread | 31.24 ms | 550.23 ms |

Fusion made the warm AVX2 pixel batch **3.44× faster** than the unfused kernel and **2.39× faster** than preallocated NumPy on this run. In a complete fused job the median JIT time was 6.13 ms. The median stage for converting back to sRGB, clipping, encoding, and writing the PNGs was 384.53 ms. File-to-file performance against NumPy was roughly even on this run.

The executed kernel disassembles to the following AVX2 instructions in the fused loop. Each `ymm` register holds eight float32 channel values. [Fused plan](../showcase/data/camera-preview/fused.plan.txt) · [Fused assembly](../showcase/data/camera-preview/forge-avx2.asm) · [Unfused plan](../showcase/data/camera-preview/unfused.plan.txt) · [Unfused assembly](../showcase/data/camera-preview/unfused-avx2.asm)

```asm
vmulps  (%rdx,%rcx,4), %ymm3, %ymm6
vaddps  %ymm6, %ymm4, %ymm6
vmulps  %ymm6, %ymm5, %ymm6
vsqrtps %ymm6, %ymm6
```

### Apple M3 measurements

Each of two local runs used 21 warm and 9 complete batch samples per backend. The warm fused Forge batch took 8.21–8.22 ms, the unfused batch took 20.68–20.92 ms, and NumPy took 5.55–5.80 ms. The file-to-file order changed between runs. Encoding and writing was the largest timed stage for fused Forge in both, at roughly 324–325 ms per batch. [First run](../showcase/data/camera-preview/m3-run-1.json) · [Second run](../showcase/data/camera-preview/m3-run-2.json)

### Rebuild the benchmark figures

Use the capture directory from the bundled-frame benchmark. The renderer checks the source and preview hashes before composing the before-and-after image.

~~~sh
.local/venv/bin/python -m pip install matplotlib==3.10.8
MPLCONFIGDIR="$PWD/.local/camera-preview/matplotlib" .local/venv/bin/python \
  showcase/camera_figures.py .local/camera-preview/benchmark --out .local/camera-preview/figures
~~~
