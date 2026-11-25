"""Draw the README accelerometer figure from a motion-features JSON report."""

import argparse
import json
from pathlib import Path

import matplotlib


RECORDING_NAMES = {"desk_closeups": "Desk", "office_walkthrough": "Office", "pioneer_turn": "Robot"}
BACKGROUND, INK, MUTED = "#101a17", "#eef5f1", "#b4c4bd"
MAGNITUDE_COLOR = "#86e4ad"


def render(report, output):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recordings = report["recordings"]
    office = next(recording for recording in recordings if recording["id"] == "office_walkthrough")
    example = office["peak_window"]
    count = sum(len(recording["windows"]) for recording in recordings)
    fig, (trace, windows) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=120, facecolor=BACKGROUND)
    fig.subplots_adjust(left=.085, right=.976, bottom=.20, top=.70, wspace=.37)
    fig.text(.043, .93, "From accelerometer traces to one feature", color=INK,
             fontsize=23, weight="bold", va="top")
    fig.text(.043, .82, f"{count} windows across three TUM recordings · acceleration in recorded units",
             color=MUTED, fontsize=11)
    for ax in (trace, windows):
        ax.set_facecolor(BACKGROUND)
        ax.tick_params(colors=MUTED, labelsize=10, length=0, pad=7)
        ax.yaxis.grid(color="#34443c", linewidth=.6)
        for spine in ax.spines.values():
            spine.set_visible(False)

    time_ms = [seconds * 1e3 for seconds in example["elapsed_s"]]
    axes = zip(example["centered_axes"], ("ax", "ay", "az"), ("#a7c7ed", "#e8ad94", "#c6a6d9"), strict=True)
    for values, label, color in axes:
        trace.plot(time_ms, values, color=color, lw=1, alpha=.8, label=label)
    trace.plot(time_ms, example["magnitude"], color=MAGNITUDE_COLOR, linewidth=2.5, label="magnitude")
    trace.set_title("Three axes → magnitude per sample", loc="left", color=INK, weight="bold", pad=12)
    trace.set_xlabel("Time within window (ms)", color=MUTED, fontsize=10)
    trace.legend(loc="upper left", ncols=4, fontsize=8.5, frameon=False, labelcolor=INK)
    trace.margins(y=.21)

    for index, recording in enumerate(recordings):
        measured = [row["mean_magnitude"] for row in recording["windows"]]
        offsets = [(i / (len(measured) - 1) - .5) * .46 for i in range(len(measured))]
        windows.scatter([index + offset for offset in offsets], measured, color=MAGNITUDE_COLOR,
                        s=25, zorder=3)
        if recording is office:
            peak_index = measured.index(example["mean_magnitude"])
            windows.scatter(index + offsets[peak_index], measured[peak_index],
                            edgecolor=INK, linewidth=1.5, facecolor="none", s=110, zorder=4)
    windows.set_xticks(range(len(recordings)), [RECORDING_NAMES[r["id"]] for r in recordings])
    windows.set_ylim(bottom=0)
    windows.set_title("128 magnitudes → mean per window", loc="left", color=INK, weight="bold", pad=12)
    windows.set_xlabel("One dot per recorded window", color=MUTED, fontsize=10)
    fig.text(.043, .057,
             f"Left: office window starting at {example['start_s']:.2f} s. Its mean is circled on the right.",
             color=MUTED, fontsize=10)
    fig.savefig(output, facecolor=BACKGROUND)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    render(json.loads(args.report.read_text(encoding="utf-8")), args.out)


if __name__ == "__main__":
    main()
