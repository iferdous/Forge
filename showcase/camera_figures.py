"""Render the README camera figures from a bundled-sample benchmark capture."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import matplotlib
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FRAME = "1305031467.759907.png"
BACKGROUND, TEXT, MUTED = "#101a17", "#eff6f2", "#afc2ba"
BACKENDS = {
    "forge": ("Forge fused", "#79dca8"),
    "unfused": ("Forge unfused", "#789387"),
    "numpy": ("NumPy", "#a2c5e7"),
    "numexpr": ("NumExpr", "#989dbb"),
}


def render_preview(capture, data, out):
    index = next(i for i, record in enumerate(data["inputs"]) if record["file"] == FRAME)
    original = ROOT / "examples/camera_preview/data/frames" / FRAME
    preview = capture / "forge-previews" / f"{FRAME}.preview.png"
    font_file = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    card = Image.new("RGB", (1376, 636), BACKGROUND)
    draw = ImageDraw.Draw(card)
    draw.text((40, 25), "Review camera frames in one batch", fill=TEXT,
              font=ImageFont.truetype(font_file, 38))
    draw.rectangle((702, 131, 1345, 614), outline=BACKENDS["forge"][1], width=2)
    label_font = ImageFont.truetype(font_file, 28)
    panels = (
        (40, original, data["inputs"][index]["sha256"], "Recorded frame", TEXT),
        (704, preview, data["outputs"][index], "Forge preview", BACKENDS["forge"][1]),
    )
    for x, path, digest, label, color in panels:
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frame hash differs from the benchmark: {path}")
        draw.text((x, 87), label, fill=color, font=label_font)
        with Image.open(path) as frame:
            card.paste(frame.convert("RGB"), (x, 133))
    card.save(out / "camera-preview.png", optimize=True)


def render_timings(data, out):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backends = [name for name in BACKENDS if name in data["backends"]]
    labels, colors = zip(*(BACKENDS[name] for name in backends), strict=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.75), dpi=120, facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.68, bottom=0.23, wspace=0.49)
    fig.text(0.043, 0.92, "What fusion changes in the camera batch", color=TEXT,
             fontsize=21, weight="bold", va="top")
    host = data["host"]
    target = "AVX2" if data["target"] == "avx2" else data["target"]
    fig.text(0.044, 0.79, f"{host['system']} {host['architecture']} · {target} Forge  /  "
             f"{data['frame_count']} recorded frames  /  median batch time", color=MUTED, fontsize=11)
    timings = (("warm_batch_ns", "Warm pixel math"),
               ("file_to_file_ns", "Complete file-to-file job"))
    for ax, (key, title) in zip(axes, timings, strict=True):
        values = [median(data["backends"][name][key]) / 1e6 for name in backends]
        ax.set_facecolor(BACKGROUND)
        ax.set_title(title, loc="left", pad=12, fontsize=13, weight="bold", color=TEXT)
        ax.barh(labels, values, color=colors, height=0.57, zorder=3)
        ax.tick_params(axis="both", length=0, labelsize=10, colors=MUTED, pad=7)
        ax.set_xlim(0, 1.26 * max(values))
        ax.invert_yaxis()
        ax.xaxis.grid(color="#34443c", linewidth=0.6)
        for border in ax.spines.values():
            border.set_visible(False)
        for row, value in enumerate(values):
            ax.text(value + 0.025 * max(values), row, f"{value:.2f}", va="center",
                    color=TEXT, fontsize=10)
        ax.set_xlabel("Milliseconds per batch", color=MUTED, fontsize=10, labelpad=6)
    samples = data["backends"]["forge"]
    warm, complete = len(samples["warm_batch_ns"]), len(samples["file_to_file_ns"])
    fig.text(0.044, 0.052, f"{warm} warm and {complete} complete samples per backend. "
             "Identical PNGs across all backends.", color=MUTED, fontsize=10)
    fig.savefig(out / "camera-performance.png", facecolor=BACKGROUND)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Output directory from the bundled-frame benchmark")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.capture / "benchmark.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    render_preview(args.capture, data, args.out)
    render_timings(data, args.out)


if __name__ == "__main__":
    main()
