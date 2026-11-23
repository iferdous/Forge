window.__timelines = window.__timelines || {};
const timeline = gsap.timeline({paused: true});
for (const [index, beat] of window.forgeBeats.entries()) {
  const t = beat.start;
  if (index > 0) {
    timeline.fromTo(beat.id, {opacity: 0}, {opacity: 1, duration: .45, ease: "sine.inOut"}, t);
  }
  timeline.from(beat.id + " .frame-top", {opacity: 0, duration: .25, ease: "sine.out"}, t + .10);
  timeline.from(beat.id + " .headline h1", {x: -21, opacity: 0, duration: .48, ease: "power3.out"}, t + .16);
  timeline.from(beat.id + " .headline p", {opacity: 0, duration: .37, ease: "power2.out"}, t + .32);
  timeline.from(beat.id + " .card", {y: 15, opacity: 0, duration: .46, stagger: .12, ease: "expo.out"}, t + .44);
  const cells = document.querySelectorAll(beat.id + " .cells span");
  if (cells.length) timeline.from(cells, {scale: .89, opacity: 0, duration: .31, stagger: .024, ease: "power2.out"}, t + .40);
  const flowRows = document.querySelectorAll(beat.id + " .flow-row");
  for (const [row, flow] of flowRows.entries()) {
    const when = t + .72 + row * 1.28;
    const nodes = flow.querySelectorAll(".flow-node");
    timeline.from(nodes, {opacity: .25, duration: .38, stagger: .24, ease: "sine.out"}, when);
    timeline.from(flow.querySelector("b"), {opacity: .25, duration: .60}, when);
  }
  for (const [group, lane] of Array.from(cells).entries()) {
    const start = t + 1.6 + Math.floor(group / 8) * .82;
    timeline.to(lane, {borderColor: "#f3f4e8", duration: .28, yoyo: true, repeat: 1}, start);
  }
  const bars = document.querySelectorAll(beat.id + " .memory-track i");
  if (bars.length) timeline.from(bars, {scaleX: 0, transformOrigin: "left", duration: 1.25, stagger: .18}, t + .87);
  timeline.from(beat.id + " .frame-bottom", {opacity: 0, duration: .36, ease: "sine.inOut"}, t + .61);
}
window.__timelines.forge = timeline;
