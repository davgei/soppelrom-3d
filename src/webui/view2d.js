/* The plan view: the room straight from above, on a 2D canvas.
 *
 * Kept alongside the 3D view because the two answer different questions. 3D tells you what a room IS --
 * how high the pile of pallets in the corner is, whether that dark shape is a bin or a doorway. The plan
 * tells you whether a layout WORKS: distances are true everywhere, nothing hides behind anything, and
 * a bin lands where you put it instead of where the perspective suggests.
 *
 * Drawn from the exported grids rather than from the rendered PNG. The PNG is a finished sheet with its
 * own title block, it cannot be aligned to world coordinates, and it is 700 KB against 2 KB of masks --
 * and drawing it here is what lets the picture follow a bin while it is being dragged.
 *
 * Same interface as view3d.js (start/refresh/setRoute/...), so room.js drives either one without
 * knowing which is on screen.
 */

import { corners, containsPoint, localProblems, snapYaw } from "./geom.js";

const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function createView2D(host, ctx) {
  const canvas = document.createElement("canvas");
  canvas.className = "view2d";
  host.appendChild(canvas);
  const g = canvas.getContext("2d");

  // world metres -> css pixels
  const view = { x0: 0, z0: 0, scale: 1 };
  const state = { drag: null, routePoints: null, showFloor: true };

  const toPixels = (x, z) => [(x - view.x0) * view.scale, (z - view.z0) * view.scale];
  const toWorld = (px, pz) => [px / view.scale + view.x0, pz / view.scale + view.z0];

  function resize() {
    const cssW = host.clientWidth, cssH = host.clientHeight;
    if (!cssW || !cssH) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Fit the exported view box, which already covers the room outline as well as the grid: the room
    // rect is the rotated min-area rect of the observed floor, so its bounding box sticks out past the
    // scanned cells and fitting the grid alone clips the outline by up to two metres.
    const box = ctx.plan.view;
    const w = box.max[0] - box.min[0], h = box.max[1] - box.min[1];
    view.scale = Math.min(cssW / w, cssH / h);
    view.x0 = box.min[0] - (cssW / view.scale - w) / 2;
    view.z0 = box.min[1] - (cssH / view.scale - h) / 2;
    draw();
  }

  /* ---------------------------------------------------------------- draw */

  function draw() {
    const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
    g.fillStyle = cssVar("--surface");
    g.fillRect(0, 0, cssW, cssH);
    if (state.showFloor) drawFloor();
    drawRoute();
    drawRoom();
    ctx.bins.forEach((_, index) => drawBin(index));
    drawEntrances();
    drawScaleBar();
  }

  /** The three floor states, one filled square per grid cell. Fine at 5 cm: the biggest grid here is
   *  about 200x200 cells and it only redraws on resize or while dragging. */
  function drawFloor() {
    const masks = ctx.masks;
    if (!masks) return;
    const { bits } = masks;
    const size = Math.ceil(masks.cell * view.scale) + 1;   // +1 so neighbours leave no hairline seams
    const free = cssVar("--free"), occupied = cssVar("--occupied"), unknown = cssVar("--unknown");
    g.globalAlpha = 0.85;
    for (let r = 0; r < masks.rows; r++) {
      for (let c = 0; c < masks.cols; c++) {
        const v = masks.data[r * masks.cols + c];
        if (!(v & bits.floor_observed)) continue;      // never scanned: leave the paper colour
        g.fillStyle = (v & bits.free) ? free : ((v & bits.occupied) ? occupied : unknown);
        const [x, y] = toPixels(masks.origin[0] + c * masks.cell, masks.origin[1] + r * masks.cell);
        g.fillRect(x, y, size, size);
      }
    }
    g.globalAlpha = 1;
  }

  function drawRoute() {
    const points = state.routePoints;
    if (!points || !points.length) return;
    const cell = ctx.plan.grid.cell;
    const size = Math.ceil(cell * view.scale) + 1;
    g.fillStyle = cssVar("--path");
    g.globalAlpha = 0.7;
    for (const [x, z] of points) {
      const [px, py] = toPixels(x - cell / 2, z - cell / 2);
      g.fillRect(px, py, size, size);
    }
    g.globalAlpha = 1;
  }

  function drawRoom() {
    const points = (ctx.plan.room?.corners || []).map(([x, z]) => toPixels(x, z));
    if (!points.length) return;
    g.strokeStyle = cssVar("--outline");
    g.lineWidth = 1.5;
    g.setLineDash([7, 5]);
    g.beginPath();
    points.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath();
    g.stroke();
    g.setLineDash([]);
  }

  function drawBin(index) {
    const bin = ctx.bins[index];
    const points = corners(bin).map(([x, z]) => toPixels(x, z));
    const bad = localProblems(ctx.plan, ctx.masks, ctx.bins, index).length > 0;
    const isNew = bin.source !== "existing";
    const selected = index === ctx.selected();
    const base = cssVar(isNew ? "--bin-new" : "--bin-existing");

    g.beginPath();
    points.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath();
    g.fillStyle = bad ? cssVar("--danger") : base;
    g.globalAlpha = bad ? 0.3 : (isNew ? 0.45 : 0.6);
    g.fill();
    g.globalAlpha = 1;
    g.lineWidth = selected ? 2.5 : 1.5;
    g.strokeStyle = selected ? cssVar("--ink") : (bad ? cssVar("--danger") : base);
    g.stroke();

    // A tick from the centre out through the short side, so which way the bin faces is visible without
    // reading a number off the panel.
    const [, c1, c2] = corners(bin);
    const [mx, my] = toPixels((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2);
    const [cx, cy] = toPixels(bin.center[0], bin.center[1]);
    g.strokeStyle = selected ? cssVar("--ink") : base;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(cx, cy);
    g.lineTo(mx, my);
    g.stroke();
  }

  function drawEntrances() {
    for (const [x, z] of ctx.plan.entrances || []) {
      const [px, py] = toPixels(x, z);
      g.fillStyle = cssVar("--entrance");
      g.beginPath();
      g.arc(px, py, 7, 0, Math.PI * 2);
      g.fill();
      g.strokeStyle = cssVar("--surface");
      g.lineWidth = 2;
      g.stroke();
    }
  }

  /** A two-metre bar. Without it nothing on screen says how big the room is. */
  function drawScaleBar() {
    const metres = 2;
    const length = metres * view.scale;
    const x = 18, y = canvas.clientHeight - 22;
    g.strokeStyle = cssVar("--ink2");
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(x, y); g.lineTo(x + length, y);
    g.moveTo(x, y - 4); g.lineTo(x, y + 4);
    g.moveTo(x + length, y - 4); g.lineTo(x + length, y + 4);
    g.stroke();
    g.fillStyle = cssVar("--ink2");
    g.font = "12px system-ui, sans-serif";
    g.fillText(`${metres} m`, x + length + 8, y + 4);
  }

  /* -------------------------------------------------------- interaction */

  function pointerWorld(event) {
    const rect = canvas.getBoundingClientRect();
    return toWorld(event.clientX - rect.left, event.clientY - rect.top);
  }

  function binAt(x, z) {
    for (let i = ctx.bins.length - 1; i >= 0; i--) {
      if (containsPoint(ctx.bins[i], x, z)) return i;
    }
    return -1;
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const [x, z] = pointerWorld(event);
    const index = binAt(x, z);
    ctx.select(index);
    if (index < 0) return;
    state.drag = { index, dx: ctx.bins[index].center[0] - x, dz: ctx.bins[index].center[1] - z,
                   moved: false };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("grabbing");
  });

  canvas.addEventListener("pointermove", (event) => {
    const [x, z] = pointerWorld(event);
    if (!state.drag) {
      canvas.classList.toggle("overbin", binAt(x, z) >= 0);
      return;
    }
    ctx.bins[state.drag.index].center = [x + state.drag.dx, z + state.drag.dz];
    state.drag.moved = true;
    ctx.changed({ live: true });
    draw();
  });

  function endDrag(event) {
    if (!state.drag) return;
    const moved = state.drag.moved;
    state.drag = null;
    canvas.classList.remove("grabbing");
    if (event && canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (moved) ctx.changed({ live: false });
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  /* ------------------------------------------------------------- public */

  return {
    start() { resize(); },
    refresh() { draw(); },
    setRoute(points) { state.routePoints = points; draw(); },
    rotateSelected() {
      const index = ctx.selected();
      if (index < 0) return;
      const bin = ctx.bins[index];
      bin.yaw_deg = snapYaw(bin.yaw_deg + 90, ctx.plan.room?.angle_deg || 0);
      draw();
    },
    frameRoom() { resize(); },
    topDown() { resize(); },        // already top-down; refit so the button does something sensible
    toggleFloor(on) {
      state.showFloor = on === undefined ? !state.showFloor : on;
      draw();
      return state.showFloor;
    },
    toggleScan() { return false; },     // no scan backdrop in the plan view
    hasScan: () => false,
    retheme() { draw(); },
    resize,
    dispose() { canvas.remove(); },
  };
}
