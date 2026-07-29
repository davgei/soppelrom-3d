/* The room page: owns the state, drives whichever view is on screen, talks to the server.
 *
 * The split is deliberate. view3d.js and view2d.js only DRAW and let you point at things; they share
 * one bins array and never fetch anything. This file owns that array, decides when the layout is worth
 * asking Python about, and renders the panels. So the two views cannot drift apart, and adding a third
 * one later means implementing an interface, not copying the logic.
 *
 * The authority sits in Python. Everything this file computes locally is provisional feedback for your
 * hand; POST /api/validate is the answer, because "can this bin still be wheeled in from the entrance"
 * is a graph search over the same grid, running the same placement.route_corridor that produced the
 * proposals in the first place.
 */

import { createView3D } from "./view3d.js";
import { createView2D } from "./view2d.js";
import { loadMasks, localProblems } from "./geom.js";

const STEM = decodeURIComponent(location.pathname.split("/").pop());

const $ = (id) => document.getElementById(id);
const esc = (text) => {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
};

let plan = null;
let masks = null;
let bins = [];
let selected = -1;
let lastCheck = null;
let dirty = false;             // has anything changed since the last save?
const views = {};
let active = "3d";

/* ----------------------------------------------------------------- state */

const ctx = {
  get plan() { return plan; },
  get masks() { return masks; },
  get bins() { return bins; },
  selected: () => selected,
  select(index) {
    if (index === selected) return;
    selected = index;
    renderBinList();
    for (const view of Object.values(views)) view.refresh();
  },
  /** A view changed the layout. live=true means the pointer is still down: redraw, but do not ask the
   *  server on every mouse move. */
  changed({ live } = {}) {
    dirty = true;
    renderBinList();
    renderRoomStats();
    if (!live) scheduleCheck();
  },
  onStatus(message, kind = "") {
    const box = $("stagestatus");
    box.innerHTML = message ? `<span class="${kind}">${esc(message)}</span>` : "";
    box.hidden = !message;
  },
  onProgress(fraction, label) {
    const box = $("stageload");
    if (fraction >= 1) { box.hidden = true; return; }
    box.hidden = false;
    $("stageloadlabel").textContent = label || "laster …";
    $("stageloadbar").style.width = `${Math.round(fraction * 100)}%`;
  },
};

/* ------------------------------------------------------------------ load */

async function load() {
  let planRes, maskRes;
  try {
    [planRes, maskRes] = await Promise.all([
      fetch(`/api/plan/${encodeURIComponent(STEM)}`),
      fetch(`/api/masks/${encodeURIComponent(STEM)}`),
    ]);
  } catch (error) {
    return fail("Fikk ikke kontakt med serveren", String(error));
  }
  if (!planRes.ok) return fail("Kunne ikke åpne rommet", await planRes.text());
  plan = await planRes.json();
  if (maskRes.ok) {
    masks = await loadMasks(await maskRes.blob(), plan.grid, plan.mask_bits);
  }

  $("addr").textContent = plan.address || plan.scan;
  document.title = `${plan.address || plan.scan} — Søppelrom 3D`;
  $("scanid").textContent = plan.scan;

  // A saved draft wins over our proposals: somebody was in the middle of something.
  const saved = plan.saved_proposal;
  const source = (saved && Array.isArray(saved.bins) && saved.bins.length) ? saved.bins : plan.bins;
  bins = source.map((bin) => ({ ...bin, center: [...bin.center] }));
  if (saved) {
    $("by").value = saved.by || "";
    $("note").value = saved.note || "";
    setSaveMessage("lagret forslag lastet inn", "saved");
  }

  buildTypeSelect();
  renderRoomStats();
  renderBinList();
  renderLegend();

  views["3d"] = createView3D($("pane3d"), ctx);
  views["2d"] = createView2D($("pane2d"), ctx);
  // Only the 3D view starts on load; the plan view sizes itself the first time its tab is shown,
  // because a hidden element has no width to fit the room into.
  views["3d"].start();
  if (!plan.mesh) {
    // No mesh means 3D has nothing to show. Open the plan instead of an empty grey box.
    setTab("2d");
    ctx.onStatus("Skannet mangler 3D-modell — viser planen. Kjør «Generer bilder» på nytt for å lage den.", "err");
  }
  views["3d"].setRoute(plan.path || null);
  views["2d"].setRoute(plan.path || null);
  runCheck();
}

function fail(heading, detail) {
  $("addr").textContent = heading;
  $("stagemsg").hidden = false;
  $("stagemsgtitle").textContent = heading;
  $("stagemsgbody").textContent = detail || "";
  $("stageload").hidden = true;
}

/* --------------------------------------------------------------- panels */

function buildTypeSelect() {
  const select = $("bintype");
  select.textContent = "";
  for (const name of Object.keys(plan.bin_types || {})) {
    const spec = plan.bin_types[name];
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name} · ${spec.length_m} × ${spec.width_m} m`;
    select.appendChild(option);
  }
  select.value = "4-hjuls container" in (plan.bin_types || {})
    ? "4-hjuls container"
    : (select.options[0]?.value || "");
}

function renderRoomStats() {
  const room = plan.room;
  $("s-size").textContent = `${room.length_m} × ${room.width_m} m`;
  $("s-area").textContent = `${room.area_m2} m²`;
  $("s-kind").textContent = room.indoor ? "innendørs" : "utendørs";
  $("s-bins").textContent = `${bins.length}`;
  // Left alone until the server answers: the free area depends on where the bins are, and a number
  // this page computed itself would disagree with the one that decides.
  if (lastCheck) $("s-free").textContent = `${lastCheck.free_area_m2} m²`;
}

function renderBinList() {
  const host = $("bins");
  host.textContent = "";
  if (!bins.length) {
    host.innerHTML = `<div class="hint">Ingen kasser. Legg til en over.</div>`;
    return;
  }
  bins.forEach((bin, index) => {
    // The server's verdict when we have one for this bin, our own instant check while dragging.
    const server = lastCheck?.bins?.[index];
    const problems = server ? server.problems : localProblems(plan, masks, bins, index);
    const row = document.createElement("div");
    row.className = "binrow" + (index === selected ? " sel" : "");
    row.innerHTML =
      `<span class="dot" style="background:var(${bin.source === "existing" ? "--bin-existing" : "--bin-new"})"></span>` +
      `<span class="name">${esc(bin.type || "kasse")}</span>` +
      `<span class="size">${bin.length_m.toFixed(2)}×${bin.width_m.toFixed(2)}</span>` +
      (problems.length ? `<span class="bad" title="${esc(problems.join("; "))}">!</span>` : "");
    row.addEventListener("click", () => ctx.select(index));
    host.appendChild(row);
  });
}

function renderLegend() {
  const rows = [
    ["--free", "ledig gulv"],
    ["--occupied", "opptatt"],
    ["--unknown", "ikke skannet"],
    ["--path", "trillevei"],
    ["--bin-existing", "kasse i dag"],
    ["--bin-new", "foreslått kasse"],
    ["--entrance", "inngang"],
    ["--outline", "romkant"],
  ];
  $("legend").innerHTML = rows
    .map(([token, label]) => `<div><span class="sw" style="background:var(${token})"></span>${label}</div>`)
    .join("");
}

function setSaveMessage(text, kind) {
  const box = $("savemsg");
  box.textContent = text;
  box.className = kind || "sub";
}

/* ------------------------------------------------------------ the check */

let checkTimer = null;
function scheduleCheck() {
  clearTimeout(checkTimer);
  // Debounced: rotate and add both land here, and a burst of clicks should be one request.
  checkTimer = setTimeout(runCheck, 130);
}

async function runCheck() {
  const verdict = $("verdict");
  verdict.className = "verdict busy";
  $("verdicttext").textContent = "sjekker …";
  let data;
  try {
    const res = await fetch(`/api/validate/${encodeURIComponent(STEM)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bins }),
    });
    if (!res.ok) throw new Error((await res.text()) || res.statusText);
    data = await res.json();
  } catch (error) {
    verdict.className = "verdict bad";
    $("verdicttext").innerHTML = `<span class="err">Kunne ikke sjekke: ${esc(error.message)}</span>`;
    return;
  }
  lastCheck = data;
  verdict.className = `verdict ${data.ok ? "good" : "bad"}`;
  $("verdicttext").innerHTML = data.ok
    ? `<strong>Plasseringen holder.</strong> ${esc(data.corridor_note)}`
    : `<strong>Noe stemmer ikke.</strong> ${esc(data.corridor_note)}`;

  const list = $("problems");
  list.textContent = "";
  data.bins.forEach((bin, index) => {
    for (const problem of bin.problems) {
      const item = document.createElement("li");
      item.textContent = `Kasse ${index + 1} (${bin.type || "?"}): ${problem}`;
      item.addEventListener("click", () => ctx.select(index));
      list.appendChild(item);
    }
  });
  if (data.ok) {
    const item = document.createElement("li");
    item.className = "ok";
    item.textContent = `Ledig gulv etter dette: ${data.free_area_m2} m²`;
    list.appendChild(item);
  }

  const route = data.route && data.route.length ? data.route : null;
  for (const view of Object.values(views)) view.setRoute(route);
  renderRoomStats();
  renderBinList();
}

/* ------------------------------------------------------------- commands */

function addBin() {
  const type = $("bintype").value;
  const spec = plan.bin_types?.[type];
  if (!spec) return;
  // Dropped in the middle of the room: immediately visible, immediately draggable, and the check says
  // straight away whether it works there.
  const cs = plan.room.corners;
  const cx = cs.reduce((sum, p) => sum + p[0], 0) / cs.length;
  const cz = cs.reduce((sum, p) => sum + p[1], 0) / cs.length;
  bins.push({
    center: [cx, cz],
    length_m: spec.length_m,
    width_m: spec.width_m,
    yaw_deg: plan.room.angle_deg || 0,
    type,
    source: "proposed",
  });
  selected = bins.length - 1;
  dirty = true;
  renderBinList();
  for (const view of Object.values(views)) view.refresh();
  scheduleCheck();
}

function removeSelected() {
  if (selected < 0) return;
  bins.splice(selected, 1);
  selected = -1;
  dirty = true;
  lastCheck = null;          // the per-bin verdicts are indexed by position and just shifted
  renderBinList();
  for (const view of Object.values(views)) view.refresh();
  scheduleCheck();
}

function resetBins() {
  bins = plan.bins.map((bin) => ({ ...bin, center: [...bin.center] }));
  selected = -1;
  lastCheck = null;
  dirty = false;
  renderBinList();
  for (const view of Object.values(views)) { view.refresh(); view.setRoute(plan.path || null); }
  scheduleCheck();
}

function rotateSelected() {
  if (selected < 0) return;
  views[active].rotateSelected();
  // The other view holds the same bins array; it only needs to redraw.
  for (const [name, view] of Object.entries(views)) if (name !== active) view.refresh();
  dirty = true;
  renderBinList();
  scheduleCheck();
}

/* ----------------------------------------------------------------- tabs */

function setTab(name) {
  if (!views[name] && name !== active) return;
  active = name;
  $("pane3d").hidden = name !== "3d";
  $("pane2d").hidden = name !== "2d";
  for (const button of document.querySelectorAll("#tabs button")) {
    button.classList.toggle("on", button.dataset.tab === name);
  }
  const view = views[name];
  if (view) {
    // A pane that was display:none had no size to lay out against; it gets one now.
    if (name === "2d" && !view.started) { view.start(); view.started = true; }
    view.resize();
    view.refresh();
  }
  $("only3d").hidden = name !== "3d";
}

/* ------------------------------------------------------------- wiring */

$("add").addEventListener("click", addBin);
$("rotate").addEventListener("click", rotateSelected);
$("remove").addEventListener("click", removeSelected);
$("reset").addEventListener("click", resetBins);
$("check").addEventListener("click", runCheck);
$("frame").addEventListener("click", () => views[active].frameRoom());
$("top").addEventListener("click", () => views[active].topDown());

$("togglescan").addEventListener("click", (event) => {
  const on = views["3d"].toggleScan();
  event.currentTarget.classList.toggle("on", on);
  event.currentTarget.textContent = on ? "Skjul skann" : "Vis skann";
});
$("togglefloor").addEventListener("click", (event) => {
  const on = views[active].toggleFloor();
  event.currentTarget.classList.toggle("on", on);
  event.currentTarget.textContent = on ? "Skjul gulvlag" : "Vis gulvlag";
});

for (const button of document.querySelectorAll("#tabs button")) {
  button.addEventListener("click", () => setTab(button.dataset.tab));
}

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.key === "r" || event.key === "R") { rotateSelected(); event.preventDefault(); }
  if (event.key === "Delete" || event.key === "Backspace") { removeSelected(); event.preventDefault(); }
  if (event.key === "Escape") ctx.select(-1);
  if (event.key === "3") setTab("3d");
  if (event.key === "2") setTab("2d");
  if (event.key === "f" || event.key === "F") views[active].frameRoom();
});

$("save").addEventListener("click", async () => {
  setSaveMessage("lagrer …", "sub");
  try {
    const res = await fetch(`/api/proposal/${encodeURIComponent(STEM)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bins, by: $("by").value, note: $("note").value }),
    });
    if (!res.ok) throw new Error((await res.text()) || res.statusText);
    const data = await res.json();
    lastCheck = data.validation;
    dirty = false;
    setSaveMessage("lagret som forslag", "saved");
    renderBinList();
  } catch (error) {
    setSaveMessage(`feilet: ${error.message}`, "err");
  }
});

// Leaving with unsaved changes is almost always a mistake -- the layout only exists in this tab.
window.addEventListener("beforeunload", (event) => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

window.addEventListener("resize", () => {
  for (const view of Object.values(views)) view.resize();
});

// Every colour in both views is read from CSS, so a light/dark switch is a rebuild, not a reload.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  for (const view of Object.values(views)) view.retheme();
});

load();
