# Recorded camera frames

The camera showcase uses 48 original RGB frames from three TUM RGB-D SLAM Dataset recordings. They are attributed to Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems* (IROS, 2012). The source frames and their processed derivatives are distributed under CC BY 4.0. Forge's source code has its separate MIT license. [Dataset and license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license) · [Sequence download](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Recording | Sample | Provenance |
| --- | ---: | --- |
| Freiburg 1 desk, handheld close-ups | 16 frames | [PNG files](frames) · [Manifest](manifest.json) |
| Freiburg 1 room, handheld office walkthrough | 16 frames | [PNG files](study/room) · [Manifest](study/manifest.json) |
| Freiburg 2 Pioneer 360, robot-mounted camera | 16 frames | [PNG files](study/pioneer) · [Manifest](study/manifest.json) |

Each set includes the first and last RGB frames and 14 evenly spaced indices in timestamp order. The included PNGs are original archive members. The manifests pin the source archive, member names, and checksums. The gallery resizes the full images on both sides, labels the measurements, and marks the sampled pixel on the first pair. The timing benchmark and default CLI demo use the 16 desk frames. The gallery shows six pairs, and the image findings measure all 48 frames.

TUM documents the frames as 640 × 480, 8-bit RGB. The example treats frames without a color profile as sRGB for preview generation. [TUM file formats](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
