# Recorded camera frames

These 16 original RGB frames come from the **Freiburg 1 desk** sequence of the TUM RGB-D SLAM Dataset. They are attributed to Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems* (IROS, 2012). The source frames and their processed derivatives are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Forge's source code has its separate MIT license. [Dataset and license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license) · [Sequence download](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)

The original PNGs are unchanged. The sample includes both endpoints and 14 evenly spaced indices from the 613 RGB images in timestamp order. [The manifest](manifest.json) pins the source archive, the selected members and their checksums. Forge's README comparison is derived from this sample and labels the applied brightness preset.

TUM documents the frames as 640 × 480, 8-bit RGB. The example treats frames without a color profile as sRGB for preview generation. [TUM file formats](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
