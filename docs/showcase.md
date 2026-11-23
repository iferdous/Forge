# Reproduce the Forge showcase

[Open the interactive showcase](https://iferdous.github.io/Forge/) · [Audit and acceptance criteria](stage-audit.md) · [Benchmark report](results.md)

## Follow one expression through the compiler

| Step | What you can inspect | Implementation |
| --- | --- | --- |
| Expression and typed graph | Inputs, shape, graph node types, intermediate use counts, and float32 literal bits | [Python frontend](../python/forge) · [C++ graph validation](../src/graph.cpp) |
| Kernel plan | Fusion control, loop boundaries and materialized intermediates | [Kernel planner](../src/kernel_plan.cpp) |
| Native lowering | Optimized LLVM IR, packed instructions, scalar cleanup, and emitted object disassembly | [LLVM lowering](../src/llvm_lowering.cpp) · [ORC JIT](../src/jit.cpp) |
| Runtime | Numeric output and shape, owned or preallocated output, cold and cached compilation | [Python JIT bindings](../python/jit_bindings.cpp) · [Capture script](../showcase/capture.py) |
| Benchmark | Measured warm medians and p95s under four matched call boundaries | [Timing driver](../bench/suite.py) · [Report generator](../bench/suite_report.py) |

Switch between 8, 9 and 17 elements to see zero or one tail element and a different number of full eight-lane groups. Toggle fusion on the nine-element example to see the output stay fixed. Switch to the shared producer to see which temporary must survive. The lane and data-movement diagrams interpret the plan and input shape. The output, IR, and disassembly come from native execution or the corresponding optimized kernel build.

## Watch the films

| Film | What it shows |
| --- | --- |
| [One loop instead of three](https://iferdous.github.io/Forge/assets/fusion.mp4) | Replays the real nine-element input, its three-loop and fused plans, and the separate 65,536-element native timings. |
| [Eight lanes and a tail](https://iferdous.github.io/Forge/assets/lanes.mp4) | Changes the input from 8 to 9 to 17 elements and matches the lane model to packed and scalar instructions in the JIT object. |
| [Where Forge wins and loses](https://iferdous.github.io/Forge/assets/tradeoffs.mp4) | Inspects an ordered sum and an edge-tiled matrix product and compares their allocating performance. |

[Original native terminal recording](https://github.com/iferdous/Forge/releases/tag/v0.1.0-preview.1#assets) · [Captions and narration script](../showcase/films/SCRIPT.md) · [Storyboard and shot sources](../showcase/films/STORYBOARD.md)

## Run a new expression

Use the [pinned LLVM build instructions](plan/01-toolchain-and-ci.md#reproduce-the-smoke-test). Run the following from the repository root after building the Python extension. Use no expression argument to open the local interactive prompt. The M3 executes the scalar path and can emit x86-64 assembly for inspection. Linux or Windows x64 with AVX2 can execute the vector kernel.

~~~sh
export PYTHONPATH="$PWD/.local/build/release/python"
python -m forge 'sqrt((x + 2) * 3)' --array 'x=[1,10,25,46,1,10,25,46,1]' --inspect assembly
python -m forge 'sum(x)' --array 'x=[16777216,1,-16777216,4]'
python -m forge --help
~~~

## Serve the recorded showcase

The browser needs only the committed static files. GitHub Pages uses the same directory.

~~~sh
python3 -m http.server 8175 --bind 127.0.0.1 --directory showcase/site
~~~

Open [localhost port 8175](http://127.0.0.1:8175/).

## Regenerate data, charts, and films

Install the [build prerequisites](plan/01-toolchain-and-ci.md), enable LLVM and Python bindings, then follow the [full timing protocol](plan/07-python-runtime-and-benchmarks.md#reproduce-the-suite) to run a new benchmark. To recreate the published charts from their original inputs, use the tagged evidence ZIP and the [existing capture](../showcase/site/data/captures.json). Keep generated work under the ignored SSK checkout directory .local.

~~~sh
mkdir -p .local/showcase
gh release download v0.1.0-preview.1 --repo iferdous/Forge \
  --pattern forge-evidence.zip --dir .local/showcase
python -m pip install matplotlib==3.10.8
cp showcase/site/data/captures.json .local/showcase/showcase-captures.json
MPLCONFIGDIR="$PWD/.local/showcase/matplotlib" python showcase/prepare.py \
  --evidence .local/showcase/forge-evidence.zip \
  --captures .local/showcase/showcase-captures.json --site showcase/site
~~~

To collect fresh traces, run on native Linux x64 with the Release Python extension and the LLVM 23.1.6 SDK. The RSS phase needs up to roughly 140 MiB of caller buffers and workspace plus the Python and LLVM processes. Linux also collects RSS automatically via /proc/self/statm. Omit both the objdump and require-avx2 flags for local scalar inspection using emitted assembly.

~~~sh
export PYTHONPATH="$PWD/.local/build/release/python"
llvm_dir=$(cat .local/llvm-dir-linux-x64.txt)
python showcase/capture.py --out .local/showcase/new-captures.json \
  --objdump "$llvm_dir/../../../bin/llvm-objdump" --require-avx2
~~~

The films require Node, Chromium, FFmpeg and FFprobe. Their current narration uses the macOS Daniel voice. The silent switch generates the same presentation and readable captions on other systems. If the renderer needs explicit binary locations, set absolute paths in HYPERFRAMES_FFMPEG_PATH and HYPERFRAMES_FFPROBE_PATH.

~~~sh
python3 showcase/films/build.py --data showcase/site/data \
  --work .local/showcase/productions
for film in fusion lanes tradeoffs; do
  dir=".local/showcase/productions/$film"
  npx --yes hyperframes@0.9.84 lint "$dir"
  npx --yes hyperframes@0.9.84 validate "$dir"
  npx --yes hyperframes@0.9.84 inspect "$dir"
  npx --yes hyperframes@0.9.84 render "$dir" \
    --output "$PWD/showcase/site/assets/$film.mp4" --fps 24 --workers 1 --quality looks
  ffmpeg -y -ss 2 -i "showcase/site/assets/$film.mp4" -frames:v 1 "showcase/site/assets/$film.jpg"
  cp "$dir/$film.vtt" "showcase/site/assets/$film.vtt"
done
~~~

The benchmark presentation uses MindMesh as a reference for placing graphs near their claims and exposing evidence in a few clicks. Forge's plots and films use its own measurements, code, and design. [MindMesh reference](https://github.com/DhruvKithany/MindMesh)
