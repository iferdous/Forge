const $ = (id) => document.getElementById(id);
const stages = ["Input", "Typed graph", "Loop plan", "LLVM IR", "Assembly", "Result"];
const boundaries = {
  alloc: ["Allocating Python", "Forge", "NumPy", "Both calls allocate the output. Compilation is excluded."],
  prealloc: ["Preallocated Python", "Forge", "NumPy", "Both calls reuse output storage. Compilation is excluded."],
  fusion: ["Native fusion", "Fused", "Unfused", "Both calls run native AVX2 with preallocated buffers."],
  simd: ["Native SIMD", "AVX2", "Scalar", "Both calls use the fused plan with preallocated buffers."],
};
const colors = ["#116754", "#ad4f2e"];
const chartNamespace = "http://www.w3.org/2000/svg";

function element(tag, text, cls) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (cls) item.className = cls;
  return item;
}

function selectOptions(input, options) {
  input.replaceChildren(...options.map(([value, label]) => {
    const option = element("option", label);
    option.value = value;
    return option;
  }));
}

function code(value) {
  return element("pre", value);
}

function details(label, value) {
  const item = element("details");
  item.append(element("summary", label), code(value));
  return item;
}

function describe(section, title, paragraph) {
  section.append(element("h3", title), element("p", paragraph));
}

function laneMap(section, count) {
  const lanes = element("div", undefined, "lanes");
  const bound = Math.floor(count / 8) * 8;
  lanes.setAttribute("aria-label", String(bound) + " packed elements plus " + String(count - bound) + " scalar elements");
  for (let i = 0; i < count; i += 1) {
    const scalar = i >= bound;
    const lane = element("span", i + 1, scalar ? "lane tail" : "lane");
    lane.title = scalar ? "Scalar cleanup" : "Packed group " + (1 + Math.floor(i / 8));
    lanes.append(lane);
  }
  section.append(lanes, element("p", bound / 8 + " packed iteration(s) and " + (count - bound) + " scalar iteration(s)."));
}

function showCapture(data) {
  const cases = new Map(data.cases.map((demo) => [demo.id, demo]));
  selectOptions($("case"), data.cases.map((demo) => [demo.id, demo.title]));
  $("case").value = "chain-9";
  let step = 1;
  let mode = "fused";
  const buttons = stages.map((label, index) => {
    const button = element("button", (1 + index) + " · " + label);
    button.type = "button";
    button.onclick = () => { step = index; render(); };
    return button;
  });
  $("pipeline").append(...buttons);

  function render() {
    const demo = cases.get($("case").value);
    const plan = demo.plans[mode];
    const stage = $("stage");
    stage.replaceChildren();
    buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === step)));
    $("fusion").querySelectorAll("button").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
    $("case-title").textContent = demo.title;
    $("expression").textContent = demo.expression;
    $("trace-meta").textContent = data.host + " · " + data.execution_target + " execution · " + data.commit.slice(0, 7);
    $("command").textContent = demo.command;
    $("case-stats").replaceChildren(...[[plan.loops, " loops"], [plan.temporaries, " temporaries"]].map(([value, label]) => {
      const item = element("span");
      item.append(element("strong", value), document.createTextNode(label));
      return item;
    }));

    if (step === 0) {
      describe(stage, "Bind the inputs", "Forge specializes the typed graph to the captured float32 input shapes.");
      stage.append(code(Object.entries(demo.inputs).map(([name, input]) =>
        name + " " + input.shape.join(" × ") + (demo.sampled ? " sample" : "") + " → " + JSON.stringify(input.sample)).join("\n")));
    }
    if (step === 1) {
      describe(stage, "Follow the typed values", demo.id === "shared"
        ? "The addition has two uses. The planner preserves the shared value in a temporary."
        : "Operation IDs, input shapes, and uses are printed by the typed graph.");
      stage.append(code(demo.graph));
    }
    if (step === 2) {
      describe(stage, "Schedule loops and buffers", "The planner log shows which operations share a pass and which values need storage.");
      if (demo.id.startsWith("chain")) {
        const flow = element("ol", undefined, "flow");
        const passes = mode === "fused"
          ? [["Read x", "add → multiply → sqrt → write output"]]
          : [["Read x", "add → write scratch[1]"], ["Read scratch[1]", "multiply → write scratch[2]"],
             ["Read scratch[2]", "sqrt → write output"]];
        for (const [input, output] of passes) {
          const item = element("li");
          item.append(element("strong", input), document.createTextNode(" → " + output));
          flow.append(item);
        }
        stage.append(flow);
      }
      stage.append(code(plan.plan));
    }
    if (step === 3) {
      describe(stage, "Inspect the lowered IR", "The compiled kernel exposes its optimized LLVM IR for the selected schedule.");
      stage.append(code(plan.ir));
    }
    if (step === 4) {
      describe(stage, "Read the machine instructions", plan.assembly_source + " · x86-64 AVX2");
      if (demo.id.startsWith("chain") && data.execution_target === "avx2") {
        laneMap(stage, demo.output_shape[0]);
      }
      if (plan.instructions.length) stage.append(code(plan.instructions.slice(0, 12).join("\n")));
      stage.append(details("Full assembly", plan.assembly));
    }
    if (step === 5) {
      describe(stage, "Check the result", "Fused and unfused executions matched the captured float32 reference bit for bit.");
      stage.append(code((demo.sampled ? "Output corner · " : "Output · ") +
        (demo.output_shape.join(" × ") || "scalar") + "\n" + JSON.stringify(demo.output)));
      if (demo.id === "reduction") {
        stage.append(element("p", "The float32 accumulator loses the +1 after 2²⁴. A wider exact accumulator would produce 5."));
      }
      if (demo.id === "matmul-edge") {
        stage.append(element("p", "Dimensions 17 × 33 × 35 cross all three 16 × 32 × 32 tile bounds. The displayed corner comes from the verified full result."));
      }
    }
  }

  $("case").onchange = render;
  $("fusion").onclick = (event) => {
    const button = event.target.closest("button[data-mode]");
    if (button) { mode = button.dataset.mode; render(); }
  };
  render();
}

function svg(tag, attrs, parent, text) {
  const item = document.createElementNS(chartNamespace, tag);
  Object.entries(attrs).forEach(([key, value]) => item.setAttribute(key, value));
  if (text !== undefined) item.textContent = text;
  parent.append(item);
  return item;
}

function micro(ns) {
  const us = ns / 1000;
  return us.toLocaleString("en-US", {maximumSignificantDigits: 4});
}
function sizeLabel(size) {
  return size.includes("×") ? size.replaceAll(" ", "") : Number(size).toLocaleString("en-US");
}
function multiplier(point) {
  return (point[1] / point[0]).toFixed(2) + "×";
}

function table(rows, details) {
  const result = element("table");
  const headings = ["Size", details[1] + " median", details[2] + " median",
    details[1] + " p95", details[2] + " p95", "Speedup"];
  const head = result.createTHead().insertRow();
  head.append(...headings.map((label) => element("th", label)));
  const body = result.createTBody();
  for (const row of rows) {
    const point = row.comparisons[$("boundary").value];
    body.insertRow().append(...[sizeLabel(row.dimensions), ...point.map((value) => micro(value) + " µs"),
      multiplier(point)].map((value) => element("td", value)));
  }
  return result;
}

function drawChart(rows, details, ratio) {
  const container = $("chart");
  container.replaceChildren();
  const width = Math.max(330, Math.min(1100, container.clientWidth - 26));
  const height = 310;
  const left = 57, right = 23, top = 23, bottom = 53;
  const chart = svg("svg", {viewBox: "0 0 " + width + " " + height, role: "group",
    "aria-label": ratio ? "Measured speedup by size" : "Measured latency by size"}, container);
  const values = rows.map((row) => row.comparisons[$("boundary").value]);
  const rendered = ratio ? [values.map((value) => value[1] / value[0])] :
    [values.map((value) => value[0] / 1000), values.map((value) => value[1] / 1000)];
  const flat = rendered.flat();
  const lower = ratio ? Math.min(1, ...flat) : Math.min(...flat);
  const upper = ratio ? Math.max(1, ...flat) : Math.max(...flat);
  const lo = Math.log10(lower) - .18, hi = Math.log10(upper) + .18;
  const x = (i) => rows.length === 1 ? width / 2 : left + i * (width - left - right) / (rows.length - 1);
  const y = (value) => top + (hi - Math.log10(value)) * (height - top - bottom) / (hi - lo);
  let ticks = Array.from({length: Math.floor(hi) - Math.ceil(lo) + 1}, (_, i) => 10 ** (i + Math.ceil(lo)));
  if (ticks.length < 3) ticks = [.1, .25, .5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    .filter((value) => Math.log10(value) >= lo && Math.log10(value) <= hi);
  for (const tick of ticks) {
    svg("line", {x1: left, x2: width - right, y1: y(tick), y2: y(tick),
      stroke: ratio && tick === 1 ? "#627d60" : "#d1d9cd", "stroke-dasharray": ratio && tick === 1 ? "6 4" : ""}, chart);
    svg("text", {x: left - 8, y: y(tick) + 4, "text-anchor": "end"}, chart,
      micro(tick * 1000) + (ratio ? "×" : ""));
  }
  rows.forEach((row, i) => {
    const display = sizeLabel(row.dimensions);
    if (width > 580 || i % 2 === 0 || i === rows.length - 1) {
      svg("text", {x: x(i), y: height - bottom + 14, transform: "rotate(-28 " + x(i) + " " +
        (height - bottom + 14) + ")", "text-anchor": "end"}, chart, display);
    }
  });
  function show(i) {
    const point = values[i];
    $("chart-summary").textContent = sizeLabel(rows[i].dimensions) + " · " + details[1] + " " +
      micro(point[0]) + " µs · " + details[2] + " " + micro(point[1]) + " µs · " + multiplier(point);
  }
  rendered.forEach((series, index) => {
    const points = series.map((value, i) => x(i) + "," + y(value)).join(" ");
    svg("polyline", {points, fill: "none", stroke: colors[index], "stroke-width": 3,
      "stroke-linejoin": "round", "stroke-dasharray": index ? "6 4" : ""}, chart);
    series.forEach((value, i) => {
      const label = sizeLabel(rows[i].dimensions) + " · " + (ratio ? multiplier(values[i]) :
        details[index + 1] + " " + micro(value * 1000) + " microseconds");
      const dot = svg("circle", {cx: x(i), cy: y(value), r: 5, fill: colors[index],
        tabindex: 0, role: "graphics-symbol", "aria-label": label}, chart);
      svg("title", {}, dot, label);
      for (const event of ["mouseover", "focus", "click"]) dot.addEventListener(event, () => show(i));
    });
  });
  const first = rows.findIndex((row) => row.dimensions === "65536");
  show(first < 0 ? Math.floor(rows.length / 2) : first);
}

function showBenchmarks(data) {
  const hosts = new Map(data.hosts.map((host) => [host.id, host]));
  selectOptions($("host"), data.hosts.map((host) => [host.id, host.label]));
  const workloadOptions = [...new Map(data.hosts[0].rows.map((row) =>
    [row.id, row.id + " · " + row.expression]))];
  selectOptions($("workload"), workloadOptions);
  $("workload").value = "E5";
  selectOptions($("boundary"), Object.entries(boundaries).map(([key, details]) => [key, details[0]]));
  $("hero-coverage").textContent = data.coverage.lines.percent.toFixed(1) + "%";
  $("hero-count").textContent = data.hosts[0].rows.length;
  const sample = data.hosts[0].rows.find((r) => r.id === "E5" && r.dimensions === "65536");
  $("hero-fusion").textContent = multiplier(sample.comparisons.fusion);

  function render() {
    const host = hosts.get($("host").value);
    const workload = $("workload").value;
    const rows = host.rows.filter((row) => row.id === workload);
    for (const option of $("boundary").options) {
      option.disabled = rows.some((row) => !row.comparisons[option.value]);
    }
    if ($("boundary").selectedOptions[0].disabled) $("boundary").value = "alloc";
    const details = boundaries[$("boundary").value];
    const ratio = $("axis").value === "ratio";
    $("chart-note").textContent = details[3] + " " + (ratio ? "Above 1× favors " + details[1] :
      details[1] + " in green · " + details[2] + " in rust") + " · sizes spaced by case";
    $("benchmark-meta").textContent = host.label + " · " + host.cpu + " · " + host.repetitions +
      " interleaved samples per case · shared CI · commit " + data.commit.slice(0, 7) +
      ". Python p95 uses batch means. Native p95 uses iteration samples.";
    $("download-csv").href = "data/" + host.id + ".csv";
    $("bench-table").replaceChildren(table(rows, details));
    drawChart(rows, details, ratio);
  }
  for (const id of ["host", "workload", "boundary", "axis"]) $(id).onchange = render;
  window.addEventListener("resize", render);
  render();
}

function showCache(data) {
  const steps = ["Empty cache", "Compile n=4", "Repeat n=4", "Compile n=9"];
  const messages = ["Cleared the process cache.", "Compiled the first shape.",
    "Reused compiled code for the same shape.", "Compiled a new shape."];
  const buttons = steps.map((label, i) => {
    const button = element("button", label);
    button.type = "button";
    button.onclick = () => render(i);
    return button;
  });
  $("cache-steps").append(...buttons);

  function render(i) {
    const stats = data.cache[i];
    buttons.forEach((button, j) => button.setAttribute("aria-pressed", String(i === j)));
    $("cache-state").textContent = messages[i] + " " + stats.hits + " hits · " +
      stats.misses + " misses · " + stats.currsize + "/" + stats.maxsize + " entries";
  }
  render(0);
}

async function load(name) {
  const response = await fetch("data/" + name + ".json");
  if (!response.ok) throw new Error(name + " returned " + response.status);
  return response.json();
}

load("captures").then((data) => {
  showCapture(data);
  showCache(data);
}).catch(() => {
  $("stage").replaceChildren(element("p", "Serve the showcase directory over HTTP to load the captures."));
});
load("benchmarks").then(showBenchmarks).catch(() => {
  $("chart-summary").textContent = "Serve the showcase directory over HTTP to load the benchmark data.";
});
