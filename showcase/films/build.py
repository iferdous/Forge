"""Generate deterministic HyperFrames compositions from Forge evidence."""

import argparse
import html
import json
from pathlib import Path
from statistics import median
import subprocess
import sys


def esc(value):
    return html.escape(str(value))


def panel(title, lines, css=""):
    items = "".join(f"<li>{esc(line)}</li>" for line in lines)
    return f'<div class="card {css}"><h3>{esc(title)}</h3><ul>{items}</ul></div>'


def lanes(n):
    tags = "".join(f'<span class="{"scalar" if i >= n // 8 * 8 else "packed"}">{i+1}</span>'
                   for i in range(n))
    return f'<div class="cells">{tags}</div>'


def execution_flow(fused):
    if fused:
        rows = [("CPU registers", ("read x", "add 2", "multiply by 3", "square root", "write output"))]
    else:
        rows = [
            ("Pass 1", ("read x", "add 2", "write scratch[1]")),
            ("Pass 2", ("read scratch[1]", "multiply by 3", "write scratch[2]")),
            ("Pass 3", ("read scratch[2]", "square root", "write output")),
        ]
    pieces = []
    for label, stages in rows:
        steps = "".join(f'<span class="flow-node{" buffer" if "scratch" in step else ""}">{esc(step)}</span>'
                        for step in stages)
        pieces.append(f'<div class="flow-row"><b>{esc(label)}</b>{steps}</div>')
    return f'<div class="execution-flow">{"".join(pieces)}</div>'


def memory_bars(rss):
    largest = max(rss.values())
    bars = "".join(f'<div class="memory-row"><span>{mode.capitalize()}</span>'
                   f'<div class="memory-track"><i style="width:{rss[mode] / largest:.3%}"></i></div>'
                   f'<b>{rss[mode]:.3f} MiB</b></div>' for mode in ("unfused", "fused"))
    return f'<div class="memory-bars">{bars}</div>'


def point(data, key, dimensions):
    return next(r for r in data["hosts"][0]["rows"] if r["id"] == key and r["dimensions"] == dimensions)


def performances(trace, data):
    cases = {row["id"]: row for row in trace["cases"]}
    chain = cases["chain-9"]
    med = point(data, "E5", "65536")["comparisons"]["fusion"]
    mat = point(data, "M1", "512 × 512 × 512")["comparisons"]["alloc"]
    py = point(data, "E5", "1024")["comparisons"]["alloc"]
    samples = trace["rss_samples"]
    rss = {mode: median(p["after_first_call"] - p["after_compile"] for p in samples[mode]) / 2**20
           for mode in ("unfused", "fused")}
    cache = trace["cache"]
    packs = [line for line in chain["plans"]["fused"]["instructions"] if "ymm" in line][:4]
    scalars = [line for line in chain["plans"]["fused"]["instructions"] if "vsqrtss" in line][:1]
    if not packs or not scalars:
        raise ValueError("The films require captured packed and scalar instructions")
    return {
        "fusion": [
            ("Nine values go in", "Real native AVX2 execution",
             panel("Terminal", ["$ python -m forge 'sqrt((x + 2) * 3)'",
                               "x = [1, 10, 25, 46, 1, 10, 25, 46, 1]",
                               "→ [3, 6, 9, 12, 3, 6, 9, 12, 3]",
                               "Both plans passed the float32 bit check"])),
            ("Three complete passes", "The intermediate arrays are real",
             execution_flow(False) +
             panel("Loop planner", ["3 loops · 2 temporary arrays", "24 modeled bytes per float32 element"], "old")),
            ("One fused pass", "The result matches, with zero scratch arrays",
             execution_flow(True) +
             panel("Loop planner", ["1 loop · 0 temporary arrays", "8 modeled bytes per float32 element"])),
            ("Measure the difference", "Separate probes for memory and latency",
             '<div class="card"><h3>First-call Linux RSS increase · 8,388,608 elements</h3>' +
             memory_bars(rss) + "</div>" +
             panel("Warm native AVX2 · 65,536 elements",
                   [f"{med[1]/1e3:.2f} → {med[0]/1e3:.2f} µs  ·  {med[1]/med[0]:.2f}×",
                    "Preallocated · shared Ubuntu CI"])),
        ],
        "lanes": [
            ("Fill one vector", "8 elements · 1 packed iteration",
             lanes(8) + panel("Disassembled LLJIT packed body", packs)),
            ("Now add one value", "9 elements · 1 packed iteration + 1 scalar",
             lanes(9) + panel("Scalar cleanup", scalars + ["Output tail → 3.0", "Exact float32 reference check passed"])),
            ("Run the vector loop twice", "17 elements · 2 packed iterations + 1 scalar",
             lanes(17) + panel("Captured execution",
              [f"Output tail → {cases['chain-17']['output'][-1]:.1f}",
               "Eight floats per packed group",
               "The tail uses scalar arithmetic"])),
        ],
        "tradeoffs": [
            ("Addition has an order", "The accumulator remains float32",
             panel("Captured sum", ["sum([16777216, 1, -16777216, 4])", f"Forge → {cases['reduction']['output']:.1f}",
                                    "Adding 1 after 2²⁴ rounds away"])),
            ("Cross the tile edges", "17 × 33 × 35 crosses all three 16 × 32 × 32 tile bounds",
             panel("Captured matrix product", [f"Output shape → {cases['matmul-edge']['output_shape']}",
                                               f"First row → {cases['matmul-edge']['output'][0][:5]} …",
                                               "The full matrix passed the reference check"])),
            ("The workload decides", "Warm allocating Python · Ubuntu shared CI",
             panel("NumPy median / Forge median", [f"Pointwise chain · 1,024 floats   {py[1]/py[0]:.2f}×",
                                                     f"512-square matrix multiply       {mat[1]/mat[0]:.2f}×",
                                                     "Above 1× favors Forge"])),
            ("Reuse the machine code", "The cache key includes the array shape",
             panel("Recorded cache requests", [f"Length 4 → miss {cache[1]['misses']}",
                                               f"Length 4 → hit {cache[2]['hits']}",
                                               f"Length 9 → miss {cache[3]['misses']}"])),
        ],
    }


def read_scripts(path):
    return [[line[3:] for line in section.splitlines() if line[:3] in ("1. ", "2. ", "3. ", "4. ")]
            for section in path.read_text(encoding="utf-8").split("\n## ")[1:]]


def call(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def voice_audio(directory, index, text, voice):
    audio = directory / f"voice-{index}.wav"
    aiff = audio.with_suffix(".aiff")
    call("say", "-v", voice, "-r", "184", "-o", str(aiff), text)
    call("ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "48000", str(audio))
    aiff.unlink()
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(audio)], capture_output=True, text=True, check=True)
    return audio.name, float(probe.stdout.strip())


def timestamp(seconds):
    millis = round(seconds * 1000)
    return f"{millis // 3_600_000:02}:{millis // 60_000 % 60:02}:{millis // 1000 % 60:02}.{millis % 1000:03}"


def build(directory, key, beats, scripts, voice, commit):
    directory.mkdir(parents=True, exist_ok=True)
    clips, audios, cues, vtt = [], [], [], ["WEBVTT", ""]
    start = 0.0
    for index, ((title, kicker, card), narration) in enumerate(zip(beats, scripts, strict=True)):
        audio, spoken = voice_audio(directory, index, narration, voice) if voice else (None, 5.7)
        duration = max(5.6, spoken + .8)
        overlap = .45 if index + 1 < len(beats) else 0
        clips.append(f"""<section id="beat-{index}" class="clip beat"
            data-start="{start:.4f}" data-duration="{duration + overlap:.4f}"
            data-track-index="{1 + index % 2}" style="z-index:{index}">
            <div class="scene-content"><div class="frame-top"><span>FORGE / CPU TENSOR COMPILER</span>
            <span>{esc(key.upper())}  ·  {index + 1} / {len(beats)}</span></div>
            <div class="headline"><h1>{esc(title)}</h1><p>{esc(kicker)}</p></div>
            <div class="cards">{card}</div>
            <div class="frame-bottom">Actual Forge evidence · {esc(commit)}</div></div></section>""")
        cues.append({"id": f"#beat-{index}", "start": round(start, 4), "duration": round(duration, 4)})
        if audio:
            audios.append(f'<audio id="voice-{index}" class="clip" data-start="{start + .20:.4f}" '
                          f'data-duration="{spoken:.4f}" data-track-index="4" data-volume="1" src="{audio}"></audio>')
        vtt.extend([f"{timestamp(start + .20)} --> {timestamp(start + .20 + spoken)}", narration, ""])
        start += duration
    src = Path(__file__).parent
    for name in ("film.css", "film.js"):
        (directory / name).write_bytes((src / name).read_bytes())
    xml = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <title>Forge / {esc(key)}</title><link rel="stylesheet" href="film.css">
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script></head>
    <body><main id="forge" data-composition-id="forge" data-start="0" data-duration="{start:.4f}"
    data-width="1280" data-height="720" data-track-index="0">
    {"".join(clips)}{"".join(audios)}</main>
    <script>window.forgeBeats={json.dumps(cues)};</script><script src="film.js"></script></body></html>"""
    (directory / "index.html").write_text(xml, encoding="utf-8")
    (directory / f"{key}.vtt").write_text("\n".join(vtt), encoding="utf-8")
    print(f"{key}: {start:.2f}s, {len(beats)} scenes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--voice", choices=("Daniel", "Samantha", "silent"), default="Daniel")
    options = parser.parse_args()
    if options.voice != "silent" and sys.platform != "darwin":
        parser.error("Use --voice silent outside macOS or supply narration using an equivalent local TTS setup")
    trace = json.loads((options.data / "captures.json").read_text(encoding="utf-8"))
    evidence = json.loads((options.data / "benchmarks.json").read_text(encoding="utf-8"))
    if trace["execution_target"] != "avx2" or not trace["rss_samples"]:
        parser.error("The films require the Linux AVX2 capture with RSS samples")
    scripts = read_scripts(Path(__file__).with_name("SCRIPT.md"))
    for (key, beats), narration in zip(performances(trace, evidence).items(), scripts, strict=True):
        origins = f"trace {trace['commit'][:7]}  ·  timing {evidence['commit'][:7]}"
        build(options.work / key, key, beats, narration,
              None if options.voice == "silent" else options.voice, origins)


if __name__ == "__main__":
    main()
