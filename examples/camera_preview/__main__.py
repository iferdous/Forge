"""Make brighter PNG review previews from a folder of camera frames."""

import argparse
from pathlib import Path

from .pipeline import BACKENDS, Preset, find_frames, preview_batch


SAMPLE = Path(__file__).with_name("data") / "frames"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=SAMPLE,
                        help="PNG/JPEG frame or folder (default: bundled TUM sample)")
    parser.add_argument("--out", type=Path, required=True, help="New output folder")
    parser.add_argument("--backend", choices=BACKENDS, default="forge")
    parser.add_argument("--target", choices=("auto", "scalar", "avx2"), default="auto")
    parser.add_argument("--stops", type=float, default=Preset.stops, help="Exposure adjustment in stops")
    parser.add_argument("--lift", type=float, default=Preset.lift, help="Shadow lift between 0 and 1")
    parser.add_argument("--explain", action="store_true", help="Show the pixel formula and Forge loop plans")
    args = parser.parse_args()
    try:
        report = preview_batch(find_frames(args.source), args.out, Preset(args.stops, args.lift),
                               args.backend, args.target)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Camera preview: {error}\n")
    print(f"{len(report['frames'])} previews saved to {args.out}")
    target = report["target"] or report["backend"]
    print(f"Pixel backend {report['backend']} ({target}), {report['compilations']} JIT compilations")
    print(f"Batch time {report['total_ns'] / 1e6:.1f} ms")
    if args.explain:
        print(f"\nLinear pixel expression\n{report['expression']}")
        for size, plan in report["plans"].items():
            print(f"\n{size:,} channel samples\n{plan.rstrip()}")
    print(f"\nRun details in {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
