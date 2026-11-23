"""Record a 30-second Forge walkthrough from executed kernels and benchmark evidence."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import forge

EXPRESSION = "sqrt((x + 2) * 3)"
ELEMENTS = 65536
WIDTH, HEIGHT = 1280, 720
DURATIONS = (6, 6, 7, 7, 4)
INK = "#f3f4f0"
MUTED = "#a8b7bb"
ACCENT = "#84e2b3"
BACKGROUND = "#131a20"
PANEL = "#1d282f"


def capture(evidence):
    command = [sys.executable, "-m", "forge", EXPRESSION]
    transcript = subprocess.run(command, check=True, capture_output=True, encoding="utf-8").stdout
    assert "Both plans match the NumPy-based reference" in transcript
    demo = json.loads((evidence / "demo/targets.json").read_text(encoding="utf-8"))
    assert demo["host_has_avx2"]
    target = "avx2" if forge.has_avx2() else "scalar"
    result_prefix = f"Forge result ({target}): "
    result = next(line.removeprefix(result_prefix) for line in transcript.splitlines()
                  if line.startswith(result_prefix))
    assert json.loads(result) == demo["reference_output"]

    plans = []
    for variant in ("unfused", "fused"):
        path = evidence / "demo" / f"avx2-{variant}-plan.txt"
        plans.append(path.read_text(encoding="utf-8").splitlines())
    assert plans[0][0] == "loops=3 temporary_arrays=2"
    assert plans[1][0] == "loops=1 temporary_arrays=0"
    assembly = (evidence / "demo/avx2-fused.s").read_text(encoding="utf-8")
    instructions = []
    for opcode in ("vaddps", "vmulps", "vsqrtps", "vmovups", "vsqrtss", "vmovss"):
        register = "%ymm" if opcode.endswith("ps") else "%xmm"
        candidates = re.findall(rf"\b{opcode}\b[ \t]+([^\n#]+)", assembly)
        operand = next(re.sub(r"\s+", " ", value.strip()) for value in candidates
                       if register in value and (opcode != "vmovss" or value.lstrip().startswith("%xmm")))
        instructions.append(f"{opcode:<10}{operand}")

    with (evidence / "summary.csv").open(encoding="utf-8", newline="") as stream:
        row = next(row for row in csv.DictReader(stream)
                   if (row["id"], row["dimensions"]) == ("E5", str(ELEMENTS)))
    metadata = json.loads((evidence / "python.json").read_text(encoding="utf-8"))["metadata"]

    x = np.resize(np.array([1, 10, 25, 46], dtype=np.float32), 9)
    outputs = []
    for name in ("sum", "max", "mean"):
        program = forge.compile(f"{name}(x)", inputs=("x",))
        value = program.jit(x, target=target)(x)
        np.testing.assert_array_equal(value, program.reference(x))
        outputs.append((f"{name}(x)", value.item()))
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    program = forge.compile("a @ b", inputs=("a", "b"))
    product = program.jit(a, b, target=target)(a, b)
    np.testing.assert_array_equal(product, program.reference(a, b))
    outputs.append(("a @ b", product.tolist()))
    return transcript, target, result, plans, instructions, row, metadata, outputs


class Frames:
    def __init__(self, work):
        os.environ.setdefault("MPLCONFIGDIR", str(work / "fonts"))
        from matplotlib import get_data_path

        directory = Path(get_data_path()) / "fonts" / "ttf"
        self.title = ImageFont.truetype(str(directory / "DejaVuSans-Bold.ttf"), 42)
        self.code = ImageFont.truetype(str(directory / "DejaVuSansMono.ttf"), 22)
        self.caption = ImageFont.truetype(str(directory / "DejaVuSans.ttf"), 19)
        self.work = work
        self.index = 0

    def slide(self, title, foot):
        self.index += 1
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.text((64, 49), "FORGE  /  CPU TENSOR COMPILER", fill=ACCENT, font=self.caption)
        draw.text((64, 91), title, fill=INK, font=self.title)
        draw.text((64, 654), foot, fill=MUTED, font=self.caption)
        draw.line((64, 626, 1216, 626), fill="#34444b", width=2)
        return image, draw

    def panel(self, draw, lines, left=64, right=1216, top=185, spacing=42):
        bottom = top + 52 + spacing * len(lines)
        assert bottom <= 610
        draw.rounded_rectangle((left, top, right, bottom), 12, fill=PANEL)
        for i, line in enumerate(lines):
            if draw.textlength(line, font=self.code) > right - left - 48:
                raise ValueError(f"Text exceeds the demo panel width: {line!r}")
            draw.text((left + 24, top + 26 + spacing * i), line,
                      fill=ACCENT if i == 0 else INK, font=self.code)

    def save(self, image):
        image.save(self.work / f"frame-{self.index}.png")


def render(out, evidence):
    transcript, target, result, plans, asm, row, meta, ops = capture(evidence)
    unfused = float(row["native_avx2_unfused_median_ns"]) / 1000
    fused = float(row["native_avx2_fused_median_ns"]) / 1000
    speedup = unfused / fused
    host = meta["cpu"]["model"]
    short_host = re.sub(r"\s+\d+-Core Processor$", "", host)
    short_host = short_host.replace("Intel(R)", "Intel").replace("Xeon(R)", "Xeon")
    profile = "Shared CI timing" if meta["profile"] == "ci-smoke" else meta["profile"]
    caption = (
        f"Demo sample: nine float32 values, executed with Forge's {target} target. "
        f"Assembly: the fused native AVX2 LLJIT object from {host}.\n\n"
        f"Benchmark: `{EXPRESSION}` at {ELEMENTS:,} float32 elements on {host}. "
        f"Warm preallocated native AVX2 median, unfused {unfused:.2f} µs and fused {fused:.2f} µs "
        f"({speedup:.2f}×). The timed call excludes compilation and allocation. "
        f"Timing profile: {meta['profile']}. Evidence commit: `{meta['commit']}`.\n"
    )

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forge-demo-", dir=out) as temporary:
        work = Path(temporary).resolve()
        frames = Frames(work)
        image, draw = frames.slide("Run an array expression", f"Native {target} execution  |  Nine float32 input values")
        frames.panel(draw, [f"$ python -m forge '{EXPRESSION}'", "",
                            "x = [1, 10, 25, 46, 1, 10, 25, 46, 1]", f"result = {result}", "",
                            "Both plans match the ordered float32 reference"])
        frames.save(image)

        image, draw = frames.slide("Three passes become one",
                                   "Intermediate values in the fused loop flow directly into the next operation")
        frames.panel(draw, ["UNFUSED", *plans[0]], left=64, right=622, spacing=34)
        frames.panel(draw, ["FUSED", *plans[1]], left=650, right=1216, spacing=34)
        frames.save(image)

        image, draw = frames.slide("Eight lanes, then a scalar tail",
                                   "Disassembled from the actual fused x86-64 LLJIT object")
        frames.panel(draw, ["PACKED BODY  /  YMM", *asm[:4], "", "SCALAR REMAINDER  /  XMM", *asm[4:]],
                     spacing=36)
        frames.save(image)

        image, draw = frames.slide("Measure fusion in isolation",
                                   f"{short_host}  |  {profile}  |  Commit {meta['commit'][:7]}")
        frames.panel(draw, [f"{ELEMENTS:,} float32 elements  /  warm native AVX2", "",
                            f"Unfused    {unfused:>8.2f} µs     3 loops",
                            f"Fused      {fused:>8.2f} µs     1 loop", "",
                            f"{speedup:.2f}× faster in this run",
                            "Same preallocated call boundary  /  compile time excluded"], spacing=40)
        frames.save(image)

        image, draw = frames.slide("Reductions and tiled matrix multiplication",
                                   "Full results and reproduction commands in the GitHub release")
        frames.panel(draw, [f"{expr:<9} → {value}" for expr, value in ops], spacing=48)
        frames.save(image)

        timeline = work / "timeline.txt"
        timeline.write_text("".join(f"file 'frame-{i}.png'\nduration {seconds}\n"
                                    for i, seconds in enumerate(DURATIONS, 1)) + "file 'frame-5.png'\n",
                            encoding="utf-8")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        video = (out / "forge-demo.mp4").resolve()
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", timeline.name, "-vf", "fps=30", "-t", str(sum(DURATIONS)), "-c:v", "libx264",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                       cwd=work, check=True)
        info = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video), "-f", "null", "-"],
                              capture_output=True, encoding="utf-8", check=True).stderr
        if "Duration: 00:00:30.00" not in info:
            raise RuntimeError("Expected exactly 30 seconds of video")

    (out / "forge-demo-caption.md").write_text(caption, encoding="utf-8")
    (out / "forge-demo-transcript.txt").write_text(transcript + "\n" + "\n".join(
        f"{expr} → {value}" for expr, value in ops
    ) + "\n", encoding="utf-8")
    print(f"Recorded the 30-second walkthrough at {video}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="A run directory containing the Python and native benchmark evidence.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.out, args.evidence)


if __name__ == "__main__":
    main()
