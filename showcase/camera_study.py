"""Run the camera study and render the README gallery and batch findings."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.camera_preview.pipeline import DECODE_SRGB, Preset, encode_srgb, load_frame, preview_batch
from showcase.camera_figures import BACKGROUND, FRAME, MUTED, ROOT, TEXT


DATA = ROOT / "examples/camera_preview/data"
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
PREVIEW_COLOR = "#86e4ad"
SOURCE_COLOR = "#a1bfdf"
CLIP_COLOR = "#ffb58e"
PIXEL = (30, 420)

# Indices within the evenly spaced samples. Each row compares the same frame.
FEATURED = (11, 5, 3)
EXTRA = (6, 10, 10)


def read_recordings():
    desk = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    extra = json.loads((DATA / "study/manifest.json").read_text(encoding="utf-8"))
    recordings = [{"scene": "Desk close-ups", "folder": "frames", "frames": desk["frames"]}, *extra]
    for recording in recordings:
        for frame in recording["frames"]:
            path = DATA / recording["folder"] / frame["file"]
            if sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
                raise ValueError(f"Source frame differs from the manifest: {path}")
    return recordings


def measure(recordings, previews, report):
    results = {entry["input"]: entry for entry in report["frames"]}
    groups = []
    for recording in recordings:
        rows = []
        for frame in recording["frames"]:
            name = frame["file"]
            original = DATA / recording["folder"] / name
            result = results[name]
            preview = previews / result["output"]
            before = DECODE_SRGB[load_frame(original)] @ LUMINANCE_WEIGHTS
            after = DECODE_SRGB[load_frame(preview)] @ LUMINANCE_WEIGHTS
            rows.append({
                "scene": recording["scene"], "frame": name, "source_index": frame["index"],
                "source_sha256": frame["sha256"], "preview_sha256": sha256(preview.read_bytes()).hexdigest(),
                "before_median_y": float(np.median(before)), "after_median_y": float(np.median(after)),
                "channel_count": result["samples"], "clipped_channels": result["highlight_clips"],
            })
        groups.append(rows)
    return groups


def pixel_example(previews, preset):
    column, row = PIXEL
    rgb = load_frame(DATA / "frames" / FRAME)[row, column]
    preview = load_frame(previews / f"{FRAME}.preview.png")[row, column]
    gain, lift, scale = preset.coefficients
    linear = DECODE_SRGB[rgb]
    exposed = linear * gain
    lifted = exposed + lift
    scaled = lifted * scale
    tone = np.sqrt(scaled)
    np.testing.assert_array_equal(encode_srgb(tone), preview)
    return {
        "frame": FRAME, "column": column, "row": row, "source_rgb": rgb.tolist(),
        "linear_rgb": linear.tolist(), "after_gain": exposed.tolist(),
        "after_lift": lifted.tolist(), "after_scale": scaled.tolist(),
        "after_sqrt": tone.tolist(), "preview_rgb": preview.tolist(),
    }


def render_gallery(recordings, groups, previews, indices, destination):
    card = Image.new("RGB", (1320, 940), BACKGROUND)
    draw = ImageDraw.Draw(card)
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    bold = ImageFont.truetype(fonts / "DejaVuSans-Bold.ttf", 27)
    caption = ImageFont.truetype(fonts / "DejaVuSans.ttf", 22)
    figure = ImageFont.truetype(fonts / "DejaVuSans-Bold.ttf", 33)
    draw.text((28, 24), "Recorded RGB frame", fill=TEXT, font=bold)
    draw.text((424, 24), "Forge preview", fill=PREVIEW_COLOR, font=bold)
    for index, (recording, group, sample) in enumerate(zip(recordings, groups, indices, strict=True)):
        row = group[sample]
        y = 69 + 286 * index
        source = DATA / recording["folder"] / row["frame"]
        preview = previews / f"{row['frame']}.preview.png"
        for x, image_path in ((28, source), (424, preview)):
            with Image.open(image_path) as frame:
                panel = frame.convert("RGB").resize((352, 264), Image.Resampling.LANCZOS)
                mark_x = x + PIXEL[0] * panel.width / frame.width
                mark_y = y + PIXEL[1] * panel.height / frame.height
                card.paste(panel, (x, y))
            if row["frame"] == FRAME:
                draw.ellipse((mark_x - 7, mark_y - 7, mark_x + 7, mark_y + 7),
                             outline=TEXT, width=3)
        draw.text((820, y + 26), row["scene"], fill=TEXT, font=bold)
        draw.text((820, y + 83), "Median linear luminance", fill=MUTED, font=caption)
        before, after = row["before_median_y"], row["after_median_y"]
        draw.text((820, y + 119), f"{before:.1%} → {after:.1%}", fill=PREVIEW_COLOR, font=figure)
        clipped = row["clipped_channels"] / row["channel_count"]
        draw.text((820, y + 179), f"{clipped:.1%} of RGB channels clipped", fill=MUTED, font=caption)
        if row["frame"] == FRAME:
            draw.text((820, y + 219), "Circle: worked pixel example", fill=MUTED, font=caption)
    card.save(destination, quality=93, optimize=True)


def render_findings(groups, path):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, (brightness, clipping) = plt.subplots(2, 1, sharex=True, figsize=(12, 6.6), dpi=120,
                                               facecolor=BACKGROUND)
    fig.subplots_adjust(left=.09, right=.975, top=.78, bottom=.145, hspace=.55)
    count = sum(len(group) for group in groups)
    fig.text(.06, .925, f"What changed across {count} recorded frames", color=TEXT,
             fontsize=23, weight="bold", va="top")
    fig.text(.06, .85, "Same preview preset on three recordings · each marker is one frame",
             color=MUTED, fontsize=12)
    for axis in (brightness, clipping):
        axis.set_facecolor(BACKGROUND)
        axis.tick_params(colors=MUTED, labelsize=11, length=0, pad=8)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.grid(axis="y", color="#34443c", linewidth=.7, zorder=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlim(-1, count)
    brightness.set_title("Median linear luminance", loc="left", color=TEXT, fontsize=14, weight="bold")
    clipping.set_title("RGB channels clipped at export", loc="left", color=TEXT, fontsize=14, weight="bold")
    offset, centers, scene_labels = 0, [], []
    for group in groups:
        samples = np.arange(offset, offset + len(group))
        initial = offset == 0
        for field, label, color in (("before_median_y", "Recorded", SOURCE_COLOR),
                                    ("after_median_y", "Forge preview", PREVIEW_COLOR)):
            brightness.plot(samples, [row[field] for row in group], color=color,
                            marker="o", markersize=3, label=label if initial else None)
        percentages = [row["clipped_channels"] / row["channel_count"] for row in group]
        clipping.bar(samples, percentages, color=CLIP_COLOR, width=.62, zorder=2)
        if not initial:
            for axis in (brightness, clipping):
                axis.axvline(offset - .5, color="#607569", linewidth=.8)
        centers.append(float(samples.mean()))
        scene_labels.append(group[0]["scene"])
        offset += len(group)
    clipping.set_xticks(centers, scene_labels)
    brightness.legend(loc="upper left", bbox_to_anchor=(.53, 1.33), ncols=2,
                      frameon=False, labelcolor=TEXT, fontsize=11)
    for axis in (brightness, clipping):
        axis.set_ylim(bottom=0)
    fig.text(.06, .035, "Sixteen evenly spaced frames per recording, in timestamp order. "
             "Clipping counts channels above 1.0 before the PNG is written.", color=MUTED, fontsize=10)
    fig.savefig(path, facecolor=BACKGROUND)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New directory for previews, data, and figures")
    args = parser.parse_args()
    recordings = read_recordings()
    paths = [DATA / recording["folder"] / frame["file"]
             for recording in recordings for frame in recording["frames"]]
    args.out.mkdir(parents=True, exist_ok=False)
    previews = args.out / "forge-previews"
    preset = Preset()
    report = preview_batch(paths, previews, preset)
    groups = measure(recordings, previews, report)
    with (args.out / "camera-study.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(groups[0][0]), lineterminator="\n")
        writer.writeheader()
        for group in groups:
            writer.writerows(group)
    pixel = pixel_example(previews, preset)
    (args.out / "pixel-example.json").write_text(json.dumps(pixel, indent=2) + "\n", encoding="utf-8")
    render_gallery(recordings, groups, previews, FEATURED, args.out / "camera-study-gallery.jpg")
    render_gallery(recordings, groups, previews, EXTRA, args.out / "camera-study-more.jpg")
    render_findings(groups, args.out / "camera-study-findings.png")
    print(f"Measured {len(paths)} frames with {report['target']} Forge. Figures and data in {args.out}.")


if __name__ == "__main__":
    main()
