# Reproduce the Forge showcase

[GitHub README](../README.md) · [Audit and acceptance criteria](stage-audit.md) · [Benchmark report](results.md)

## Follow one expression through the compiler

| Step | What you can inspect | Implementation |
| --- | --- | --- |
| Expression and typed graph | Inputs, shape, graph node types, intermediate use counts, and float32 literal bits | [Python frontend](../python/forge) · [C++ graph validation](../src/graph.cpp) |
| Kernel plan | Fusion control, loop boundaries and materialized intermediates | [Kernel planner](../src/kernel_plan.cpp) |
| Native lowering | Optimized LLVM IR, packed instructions, scalar cleanup, and emitted object disassembly | [LLVM lowering](../src/llvm_lowering.cpp) · [ORC JIT](../src/jit.cpp) |
| Runtime | Numeric output and shape, owned or preallocated output, cold and cached compilation | [Python JIT bindings](../python/jit_bindings.cpp) · [Capture script](../showcase/capture.py) |
| Benchmark | Measured warm medians and p95s under four matched call boundaries | [Timing driver](../bench/suite.py) · [Report generator](../bench/suite_report.py) |

Open the 8-, 9-, and 17-element traces to compare the full eight-lane groups and scalar tails. Compare both plans on the nine-element example to see the output stay fixed. The shared-producer trace shows which temporary survives fusion. The lane and data-movement diagrams interpret the plan and input shape. The output, IR, and disassembly come from native execution or the corresponding optimized kernel build.

## Watch the films

| Film | What it shows |
| --- | --- |
| [One loop instead of three](../README.md#why-fusion-saves-time) | Replays the real nine-element input, its three-loop and fused plans, and the separate 65,536-element native timings. |
| [Eight lanes and a tail](../README.md#how-the-compiler-handles-an-input-that-doesnt-fit-a-vector) | Changes the input from 8 to 9 to 17 elements and matches the lane model to packed and scalar instructions in the JIT object. |
| [Where Forge wins and loses](../README.md#where-forge-wins-and-why-it-sometimes-loses) | Inspects an ordered sum and an edge-tiled matrix product and compares their allocating performance. |

[Original native terminal recording](https://github.com/iferdous/Forge/releases/tag/v0.1.0-preview.1#assets) · [Captions and narration script](../showcase/films/SCRIPT.md) · [Storyboard and shot sources](../showcase/films/STORYBOARD.md)

## Run a new expression

Use the [pinned LLVM build instructions](plan/01-toolchain-and-ci.md#reproduce-the-smoke-test). Run the following from the repository root after building the Python extension. Use no expression argument to open the local interactive prompt. The M3 executes the scalar path and can emit x86-64 assembly for inspection. Linux or Windows x64 with AVX2 can execute the vector kernel.

~~~sh
export PYTHONPATH="$PWD/.local/build/release/python"
python -m forge 'sqrt((x + 2) * 3)' --array 'x=[1,10,25,46,1,10,25,46,1]' --inspect assembly
python -m forge 'sum(x)' --array 'x=[16777216,1,-16777216,4]'
python -m forge --help
~~~

## Regenerate the figures and README tables

Install the [build prerequisites](plan/01-toolchain-and-ci.md), enable LLVM and Python bindings, then follow the [full timing protocol](plan/07-python-runtime-and-benchmarks.md#reproduce-the-suite) to run a new benchmark. To recreate the published charts from their original inputs, use the tagged evidence ZIP and the [existing capture](../showcase/data/captures.json). Keep generated work under the ignored SSK checkout directory .local.

~~~sh
mkdir -p .local/showcase
gh release download v0.1.0-preview.1 --repo iferdous/Forge \
  --pattern forge-evidence.zip --dir .local/showcase
python -m pip install matplotlib==3.10.8
cp showcase/data/captures.json .local/showcase/showcase-captures.json
MPLCONFIGDIR="$PWD/.local/showcase/matplotlib" python showcase/prepare.py \
  --evidence .local/showcase/forge-evidence.zip \
  --captures .local/showcase/showcase-captures.json --out showcase
python3 showcase/readme.py
python3 showcase/readme.py --check
~~~

To collect fresh traces, run on native Linux x64 with the Release Python extension and the LLVM 23.1.6 SDK. The RSS phase needs up to roughly 140 MiB of caller buffers and workspace plus the Python and LLVM processes. Linux also collects RSS automatically via /proc/self/statm. Omit both the objdump and require-avx2 flags for local scalar inspection using emitted assembly.

~~~sh
export PYTHONPATH="$PWD/.local/build/release/python"
llvm_dir=$(cat .local/llvm-dir-linux-x64.txt)
python showcase/capture.py --out .local/showcase/new-captures.json \
  --objdump "$llvm_dir/../../../bin/llvm-objdump" --require-avx2
~~~

## Regenerate the demos

Use Node 22 or later, Chromium, FFmpeg with libass support, and FFprobe. The source includes the ElevenLabs voice takes. Run with the silent flag to generate a caption-only composition. Set absolute paths in HYPERFRAMES_FFMPEG_PATH and HYPERFRAMES_FFPROBE_PATH if the renderer needs them.

~~~sh
python3 showcase/films/build.py --data showcase/data --work .local/showcase/productions
for film in fusion lanes tradeoffs; do
  dir=".local/showcase/productions/$film"
  npx --yes hyperframes@0.9.84 lint "$dir"
  npx --yes hyperframes@0.9.84 validate "$dir"
  npx --yes hyperframes@0.9.84 inspect "$dir"
  npx --yes hyperframes@0.9.84 render "$dir" \
    --output "$PWD/$dir/$film.mp4" --fps 24 --workers 1 --quality looks
  ffmpeg -y -i "$dir/$film.mp4" \
    -vf "subtitles=$dir/$film.vtt:force_style='FontName=Arial,FontSize=12,MarginV=30,BorderStyle=3,Outline=3,Shadow=0'" \
    -c:v libx264 -preset fast -crf 18 -c:a copy -movflags +faststart "showcase/assets/$film.mp4"
  cp "$dir/$film.vtt" "showcase/assets/$film.vtt"
done
~~~

[Download the MP4s and WebVTT captions](../showcase/assets) · [Read the scripts](../showcase/films/SCRIPT.md)

The benchmark presentation uses MindMesh as a reference for placing graphs near their claims and exposing evidence in a few clicks. Forge's plots and films use its own measurements, code, and design. [MindMesh reference](https://github.com/DhruvKithany/MindMesh)
