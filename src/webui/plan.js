"use strict";
/* The plan view.
 *
 * Draws the room from the exported grids and geometry rather than showing the rendered PNG: the PNG
 * is a finished sheet with its own title block and legend, it cannot be aligned to world coordinates,
 * and it is 700 KB against 2 KB of masks. Drawing it here also means the picture updates while a bin
 * is being dragged.
 *
 * Division of labour, on purpose: this file answers "is this corner on scanned floor" and "does this
 * overlap another bin" instantly, from the mask bitmap. It never decides whether a layout is legal.
 * The question that matters -- can each bin still be wheeled in from the entrance -- is a graph
 * search over the same grid, and it runs in Python (POST /api/validate) so there is exactly one
 * implementation of the placement rules.
 */

const STEM = decodeURIComponent(location.pathname.split("/").pop());
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");

let PLAN = null;
let BITS = null;
let MASK = null;          // {data, cols, rows} raw bytes of masks.png
let bins = [];            // working copy; each {center:[x,z], length_m, width_m, yaw_deg, type, source}
let selected = -1;
let view = { x0: 0, z0: 0, scale: 1 };   // world -> css pixels
let drag = null;
let route = null;         // push path from the last server check
let lastCheck = null;

const COL = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------------------------------------------------------------- load */

async function load() {
  const [planRes, maskRes] = await Promise.all([
    fetch(`/api/plan/${encodeURIComponent(STEM)}`),
    fetch(`/api/masks/${encodeURIComponent(STEM)}`),
  ]);
  if (!planRes.ok) {
    document.getElementById("addr").textContent = "Kunne ikke åpne rommet";
    document.getElementById("scanid").textContent = await planRes.text();
    return;
  }
  PLAN = await planRes.json();
  BITS = PLAN.mask_bits;
  MASK = await readMask(await maskRes.blob());

  document.getElementById("addr").textContent = PLAN.address || PLAN.scan;
  document.getElementById("scanid").textContent = PLAN.scan;
  document.getElementById("s-size").textContent = `${PLAN.room.length_m} × ${PLAN.room.width_m} m`;
  document.getElementById("s-area").textContent = `${PLAN.room.area_m2} m²`;
  document.getElementById("s-free").textContent = `${PLAN.room.free_area_m2} m²`;
  document.getElementById("s-kind").textContent = PLAN.room.indoor ? "innendørs" : "utendørs/åpent";

  const sel = document.getElementById("bintype");
  for (const name of Object.keys(PLAN.bin_types)) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.appendChild(o);
  }
  sel.value = "4-hjuls container" in PLAN.bin_types ? "4-hjuls container" : sel.options[0].value;

  // A saved draft wins over our proposals: someone was in the middle of something.
  const saved = PLAN.saved_proposal;
  bins = (saved && Array.isArray(saved.bins) && saved.bins.length ? saved.bins : PLAN.bins)
    .map((b) => ({ ...b, center: [...b.center] }));
  if (saved) {
    document.getElementById("by").value = saved.by || "";
    document.getElementById("note").value = saved.note || "";
    document.getElementById("savemsg").textContent = "lagret forslag lastet inn";
    document.getElementById("savemsg").className = "saved";
  }
  route = PLAN.path || null;

  resize();
  renderBinList();
  check();
}

/** masks.png -> raw bytes. Drawn onto a canvas at 1:1 and read back; the PNG is a single opaque
 *  8-bit channel precisely so this read-back is lossless. */
async function readMask(blob) {
  const bmp = await createImageBitmap(blob);
  const off = new OffscreenCanvas(bmp.width, bmp.height);
  const octx = off.getContext("2d", { willReadFrequently: true });
  octx.drawImage(bmp, 0, 0);
  const img = octx.getImageData(0, 0, bmp.width, bmp.height);
  const out = new Uint8Array(bmp.width * bmp.height);
  for (let i = 0; i < out.length; i++) out[i] = img.data[i * 4];   // R channel carries the flags
  return { data: out, cols: bmp.width, rows: bmp.height };
}

/* ------------------------------------------------------- coordinates */

function worldToPx(x, z) {
  return [(x - view.x0) * view.scale, (z - view.z0) * view.scale];
}
function pxToWorld(px, pz) {
  return [px / view.scale + view.x0, pz / view.scale + view.z0];
}

function resize() {
  const stage = canvas.parentElement;
  const cssW = stage.clientWidth, cssH = stage.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Fit the exported view box, which already covers the room outline as well as the grid -- the
  // rotated room rect sticks out past the scanned area, so fitting the grid alone clips the outline.
  const v = PLAN.view;
  const w = v.max[0] - v.min[0], h = v.max[1] - v.min[1];
  view.scale = Math.min(cssW / w, cssH / h);
  view.x0 = v.min[0] - (cssW / view.scale - w) / 2;
  view.z0 = v.min[1] - (cssH / view.scale - h) / 2;
  draw();
}

/* --------------------------------------------------------- mask reads */

function cellAt(x, z) {
  const g = PLAN.grid;
  const c = Math.floor((x - g.origin[0]) / g.cell);
  const r = Math.floor((z - g.origin[1]) / g.cell);
  if (r < 0 || c < 0 || r >= MASK.rows || c >= MASK.cols) return 0;
  return MASK.data[r * MASK.cols + c];
}

/** Fraction of a bin's footprint that is on scanned floor.
 *  NB: not the centre cell. A depth scan never sees through a container, so where a bin already
 *  stands its own middle reads as unscanned -- testing the centre would reject real bins. */
function onScannedFloor(b) {
  const g = PLAN.grid;
  const step = g.cell;
  const cos = Math.cos((b.yaw_deg * Math.PI) / 180), sin = Math.sin((b.yaw_deg * Math.PI) / 180);
  let total = 0, seen = 0;
  for (let a = -b.length_m / 2; a <= b.length_m / 2; a += step) {
    for (let d = -b.width_m / 2; d <= b.width_m / 2; d += step) {
      const x = b.center[0] + a * cos - d * sin;
      const z = b.center[1] + a * sin + d * cos;
      total++;
      if (cellAt(x, z) & BITS.floor_observed) seen++;
    }
  }
  return total ? seen / total : 0;
}

function corners(b) {
  const cos = Math.cos((b.yaw_deg * Math.PI) / 180), sin = Math.sin((b.yaw_deg * Math.PI) / 180);
  const hl = b.length_m / 2, hw = b.width_m / 2;
  return [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]].map(([a, d]) => [
    b.center[0] + a * cos - d * sin,
    b.center[1] + a * sin + d * cos,
  ]);
}

/** Separating-axis overlap test for two oriented rectangles. */
function overlaps(a, b) {
  const pa = corners(a), pb = corners(b);
  for (const poly of [pa, pb]) {
    for (let i = 0; i < 4; i++) {
      const [x1, z1] = poly[i], [x2, z2] = poly[(i + 1) % 4];
      const ax = -(z2 - z1), az = x2 - x1;              // edge normal
      let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
      for (const [x, z] of pa) { const d = x * ax + z * az; aMin = Math.min(aMin, d); aMax = Math.max(aMax, d); }
      for (const [x, z] of pb) { const d = x * ax + z * az; bMin = Math.min(bMin, d); bMax = Math.max(bMax, d); }
      if (aMax <= bMin || bMax <= aMin) return false;
    }
  }
  return true;
}

/** Has this bin been moved away from where the plan says it stands? Only changed bins are judged --
 *  the server applies the same rule (web.validate_layout), and if the two disagreed a bin would flash
 *  red under the pointer and then be declared fine on drop. */
function isJudged(b) {
  if (b.source !== "existing") return true;
  return !PLAN.bins.some(
    (o) => o.source === "existing" &&
      Math.abs(o.center[0] - b.center[0]) <= 0.02 && Math.abs(o.center[1] - b.center[1]) <= 0.02 &&
      Math.abs(o.length_m - b.length_m) <= 0.02 && Math.abs(o.width_m - b.width_m) <= 0.02 &&
      Math.abs(o.yaw_deg - b.yaw_deg) <= 1.0
  );
}

function localProblems(index) {
  const b = bins[index];
  if (!isJudged(b)) return [];       // a bin that is simply there is not on trial
  const out = [];
  const frac = onScannedFloor(b);
  if (frac < 0.85) out.push(`${Math.round(frac * 100)}% på skannet gulv`);
  for (let j = 0; j < bins.length; j++) if (j !== index && overlaps(b, bins[j])) { out.push("overlapper"); break; }
  return out;
}

/* -------------------------------------------------------------- draw */

function draw() {
  if (!PLAN) return;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  ctx.fillStyle = COL("--panel-bg");
  ctx.fillRect(0, 0, cssW, cssH);

  drawFloor();
  drawPath();
  drawRoom();
  for (let i = 0; i < bins.length; i++) drawBin(i);
  drawEntrances();
  drawScale();
}

/** The three floor states as cells. One fillRect per cell is fine at 5 cm: the biggest grid here is
 *  ~200x200, and it only redraws on resize or drag. */
function drawFloor() {
  const g = PLAN.grid;
  const size = g.cell * view.scale;
  const px = Math.ceil(size) + 1;                 // +1 so neighbours never leave hairline seams
  const free = COL("--free-floor"), occ = COL("--occupied-floor"), unk = COL("--unknown-floor");
  for (let r = 0; r < MASK.rows; r++) {
    for (let c = 0; c < MASK.cols; c++) {
      const v = MASK.data[r * MASK.cols + c];
      if (!(v & BITS.floor_observed)) continue;   // never scanned: leave the panel colour
      const colour = (v & BITS.free) ? free : ((v & BITS.occupied) ? occ : unk);
      const [x, y] = worldToPx(g.origin[0] + c * g.cell, g.origin[1] + r * g.cell);
      ctx.fillStyle = colour;
      ctx.fillRect(x, y, px, px);
    }
  }
}

function drawPath() {
  if (!route || !route.length) return;
  const g = PLAN.grid;
  const px = Math.ceil(g.cell * view.scale) + 1;
  ctx.fillStyle = COL("--path");
  ctx.globalAlpha = 0.75;
  for (const [x, z] of route) {
    const [sx, sy] = worldToPx(x - g.cell / 2, z - g.cell / 2);
    ctx.fillRect(sx, sy, px, px);
  }
  ctx.globalAlpha = 1;
}

function drawRoom() {
  const pts = PLAN.room.corners.map(([x, z]) => worldToPx(x, z));
  ctx.strokeStyle = COL("--room-outline");
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
}

function drawBin(index) {
  const b = bins[index];
  const pts = corners(b).map(([x, z]) => worldToPx(x, z));
  const bad = localProblems(index).length > 0;
  const isNew = b.source !== "existing";
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = isNew ? COL("--new-bin") : COL("--existing-bin");
  ctx.globalAlpha = bad ? 0.32 : 0.62;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.lineWidth = index === selected ? 3 : 1.5;
  ctx.strokeStyle = bad ? COL("--danger") : (index === selected ? COL("--accent-hover") : "#0b0e11");
  ctx.stroke();

  // a tick on the SHORT side, so which way the bin faces is visible without reading a number
  const [c0, c1, c2] = corners(b);
  const midShort = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2];
  const [mx, my] = worldToPx(midShort[0], midShort[1]);
  const [cx, cy] = worldToPx(b.center[0], b.center[1]);
  ctx.strokeStyle = "#0b0e11";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(mx, my);
  ctx.stroke();
}

function drawEntrances() {
  for (const [x, z] of PLAN.entrances || []) {
    const [px, py] = worldToPx(x, z);
    ctx.fillStyle = COL("--entrance");
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#0b0e11";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function drawScale() {
  const metres = 2;
  const len = metres * view.scale;
  const x = 16, y = canvas.clientHeight - 20;
  ctx.strokeStyle = COL("--text-muted");
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x + len, y);
  ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
  ctx.moveTo(x + len, y - 4); ctx.lineTo(x + len, y + 4);
  ctx.stroke();
  ctx.fillStyle = COL("--text-muted");
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.fillText(`${metres} m`, x + len + 8, y + 4);
}

/* ------------------------------------------------------- interaction */

function hitTest(px, pz) {
  const [x, z] = pxToWorld(px, pz);
  // topmost first, so a bin drawn over another is the one you grab
  for (let i = bins.length - 1; i >= 0; i--) {
    const pts = corners(bins[i]);
    let inside = false;
    for (let a = 0, b = 3; a < 4; b = a++) {
      const [xa, za] = pts[a], [xb, zb] = pts[b];
      if ((za > z) !== (zb > z) && x < ((xb - xa) * (z - za)) / (zb - za) + xa) inside = !inside;
    }
    if (inside) return i;
  }
  return -1;
}

canvas.addEventListener("pointerdown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = e.clientX - rect.left, pz = e.clientY - rect.top;
  const index = hitTest(px, pz);
  select(index);
  if (index >= 0) {
    const [wx, wz] = pxToWorld(px, pz);
    drag = { index, dx: bins[index].center[0] - wx, dz: bins[index].center[1] - wz };
    canvas.classList.add("dragging");
    canvas.setPointerCapture(e.pointerId);
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const rect = canvas.getBoundingClientRect();
  const [wx, wz] = pxToWorld(e.clientX - rect.left, e.clientY - rect.top);
  bins[drag.index].center = [wx + drag.dx, wz + drag.dz];
  draw();
  renderBinList();
});

canvas.addEventListener("pointerup", (e) => {
  if (!drag) return;
  drag = null;
  canvas.classList.remove("dragging");
  canvas.releasePointerCapture(e.pointerId);
  check();                 // the authoritative check runs on drop, not on every mouse move
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select")) return;
  if (e.key === "r" || e.key === "R") { rotate(); e.preventDefault(); }
  if (e.key === "Delete" || e.key === "Backspace") { removeSelected(); e.preventDefault(); }
});

function select(index) {
  selected = index;
  draw();
  renderBinList();
}

function rotate() {
  if (selected < 0) return;
  const b = bins[selected];
  b.yaw_deg = (b.yaw_deg + 90) % 360;
  draw();
  check();
}

function removeSelected() {
  if (selected < 0) return;
  bins.splice(selected, 1);
  selected = -1;
  draw();
  renderBinList();
  check();
}

document.getElementById("add").addEventListener("click", () => {
  const type = document.getElementById("bintype").value;
  const spec = PLAN.bin_types[type];
  // Drop it in the middle of the room rather than at a corner: it is immediately visible and
  // immediately draggable, and the check will say straight away whether it works there.
  const cs = PLAN.room.corners;
  const cx = cs.reduce((s, p) => s + p[0], 0) / cs.length;
  const cz = cs.reduce((s, p) => s + p[1], 0) / cs.length;
  bins.push({
    center: [cx, cz], length_m: spec.length_m, width_m: spec.width_m,
    yaw_deg: PLAN.room.angle_deg, type, source: "proposed",
  });
  select(bins.length - 1);
  check();
});

document.getElementById("rotate").addEventListener("click", rotate);
document.getElementById("remove").addEventListener("click", removeSelected);
document.getElementById("reset").addEventListener("click", () => {
  bins = PLAN.bins.map((b) => ({ ...b, center: [...b.center] }));
  selected = -1;
  route = PLAN.path || null;
  draw();
  renderBinList();
  check();
});

/* ------------------------------------------------------------ panels */

function renderBinList() {
  const host = document.getElementById("bins");
  host.textContent = "";
  bins.forEach((b, i) => {
    const row = document.createElement("div");
    row.className = "binrow" + (i === selected ? " sel" : "");
    const bad = localProblems(i);
    const server = lastCheck && lastCheck.bins && lastCheck.bins[i];
    const problems = server ? server.problems : bad;
    row.innerHTML =
      `<span class="dot" style="background:${b.source === "existing" ? "var(--existing-bin)" : "var(--new-bin)"}"></span>` +
      `<span class="name">${b.type || "kasse"}</span>` +
      (problems && problems.length ? `<span class="bad">!</span>` : "");
    row.addEventListener("click", () => select(i));
    host.appendChild(row);
  });
  if (!bins.length) host.innerHTML = `<div class="hint">Ingen kasser. Legg til en over.</div>`;
}

let checkTimer = null;
function check() {
  clearTimeout(checkTimer);
  // Debounced: rotate() and add() both call this, and a burst of clicks should be one request.
  checkTimer = setTimeout(runCheck, 120);
}

async function runCheck() {
  const verdict = document.getElementById("verdict");
  const list = document.getElementById("problems");
  verdict.textContent = "sjekker …";
  list.textContent = "";
  let data;
  try {
    const res = await fetch(`/api/validate/${encodeURIComponent(STEM)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bins }),
    });
    if (!res.ok) throw new Error(await res.text());
    data = await res.json();
  } catch (err) {
    verdict.innerHTML = `<span class="err">kunne ikke sjekke: ${err.message}</span>`;
    return;
  }
  lastCheck = data;
  route = data.route && data.route.length ? data.route : route;
  document.getElementById("s-free").textContent = `${data.free_area_m2} m²`;
  verdict.innerHTML = data.ok
    ? `<span class="saved">Plasseringen holder.</span> ${esc(data.corridor_note)}`
    : `<span class="err">Noe stemmer ikke.</span> ${esc(data.corridor_note)}`;
  data.bins.forEach((b, i) => {
    for (const p of b.problems) {
      const li = document.createElement("li");
      li.textContent = `Kasse ${i + 1} (${b.type || "?"}): ${p}`;
      list.appendChild(li);
    }
  });
  if (data.ok) {
    const li = document.createElement("li");
    li.className = "ok";
    li.textContent = `Ledig gulv etter dette: ${data.free_area_m2} m²`;
    list.appendChild(li);
  }
  draw();
  renderBinList();
}

document.getElementById("check").addEventListener("click", runCheck);

document.getElementById("save").addEventListener("click", async () => {
  const msg = document.getElementById("savemsg");
  msg.textContent = "lagrer …";
  msg.className = "sub";
  try {
    const res = await fetch(`/api/proposal/${encodeURIComponent(STEM)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bins,
        by: document.getElementById("by").value,
        note: document.getElementById("note").value,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    lastCheck = data.validation;
    msg.textContent = "lagret som forslag";
    msg.className = "saved";
    renderBinList();
  } catch (err) {
    msg.textContent = `feilet: ${err.message}`;
    msg.className = "err";
  }
});

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : s;
  return d.innerHTML;
}

window.addEventListener("resize", () => { if (PLAN) resize(); });
load();
