# Accelerometer motion features

Use this example to turn three channels from an accelerometer log into a single measure of variation per window. Large values can flag windows for closer inspection. The bundled files contain 36 real windows sampled across the handheld desk and office recordings and the robot-mounted Pioneer recording used by the camera study. [Recorded inputs, selection, and license](../examples/motion_features/data/README.md)

## From a reading to a feature

1. Read `timestamp ax ay az` rows and group every 128 successive samples. Compute each axis's own mean over the window, subtract it from that axis, then give Forge three contiguous float32 arrays. Timestamps remain float64 and supply the plot's time axis.
2. Evaluate `sqrt((ax * ax + ay * ay) + az * az)` at each sample. This is the Euclidean distance from that sample's acceleration vector to the average vector of the window. The camera expression also maps each sample independently, but its input is a single array and its output stays an array for PNG export.
3. Apply `mean(...)` to the 128 distances to get one scalar for the entire window. The feature measures deviations from the window's average acceleration vector. Changes in device orientation, vibration, and acceleration can all affect the result. The source timestamps are unevenly spaced, so a 128-sample window has a variable duration and the mean weights all 128 readings equally.

## What the compiler does

The unfused plan uses six passes to build the magnitude and one pass to take the mean. The fused plan does all six pointwise steps in its first pass, writes one intermediate array, and reads that array in a separate ordered float32 reduction. The resulting plan needs **512 bytes** of array scratch for a 128-sample window, down from **3,072 bytes** in the unfused plan. These counts describe the compiler's intermediate arrays.

The example separately compiles the magnitude trace for plotting, then checks both means against an independent NumPy calculation that adds in the specified float32 order. Each kernel is compiled once and reused for every window. The included scalar capture gives the measured features, source hashes, and both loop plans. [Recorded analysis](../showcase/data/motion-features.json) · [Native AVX2 checks](https://github.com/iferdous/Forge/actions/runs/45551662474)

## Run it

From the checkout, use the LLVM-enabled Python build described in the README. The best available native target is selected automatically. Use `.local/build/release/python` as the Python module path on Linux. [Forge build instructions](../README.md#try-it)

```sh
PYTHONPATH="$PWD/.local/build/macos-llvm/python" .local/venv/bin/python \
  -m examples.motion_features --explain --json .local/motion-features/report.json
```

Pass one or more paths before the flags to analyze your own whitespace-delimited logs. Each line contains a timestamp in seconds and three acceleration readings. `#` introduces a comment. Timestamps must increase within a file. The command reports any trailing samples after the last complete window. Omit `--json` for terminal output only, or choose a fresh file name for each report.

To rebuild the README chart, use the JSON report from the bundled inputs.

```sh
.local/venv/bin/python -m pip install matplotlib==3.10.8
MPLCONFIGDIR="$PWD/.local/motion-features/matplotlib" .local/venv/bin/python \
  -m showcase.motion_figure .local/motion-features/report.json \
  --out .local/motion-features/motion-features.png
```
