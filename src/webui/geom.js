/* Geometry and mask lookups shared by the 2D plan and the 3D view.
 *
 * Pure functions and one small class -- no DOM, no three.js, no fetch. Both views ask the SAME
 * questions ("is this bin off the scanned floor", "does it overlap that one"), and when the 2D and 3D
 * copies of that answer disagreed, a bin flashed red in one tab and green in the other.
 *
 * These are the INSTANT checks, the ones that have to keep up with the pointer. They are not the
 * verdict. Whether a layout is legal -- can every bin still be wheeled in from the entrance -- is a
 * graph search that runs in Python (POST /api/validate), so there is exactly one implementation of the
 * placement rules and it is the one that produced the proposals.
 */

/** Fraction of a footprint that must sit on scanned floor before we stop complaining. Matches
 *  web.validate_layout's min_on_floor: if the two differed, dragging would show one answer and
 *  dropping would report another. */
export const MIN_ON_FLOOR = 0.85;

/** How close a submitted bin has to be to a stored one to count as "not moved". Same numbers as
 *  web._unchanged_existing. */
const TOL_M = 0.02;
const TOL_DEG = 1.0;

/** The four (x, z) corners of a bin, in annotations.BinBox's convention: length along
 *  (cos yaw, sin yaw), width along (-sin yaw, cos yaw). web_export.corners_from_yaw derives the same
 *  four points server-side; any disagreement here draws every bin rotated. */
export function corners(bin) {
  const yaw = (bin.yaw_deg * Math.PI) / 180;
  const cos = Math.cos(yaw), sin = Math.sin(yaw);
  const hl = bin.length_m / 2, hw = bin.width_m / 2;
  return [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]].map(([along, across]) => [
    bin.center[0] + along * cos - across * sin,
    bin.center[1] + along * sin + across * cos,
  ]);
}

/** Separating-axis test for two oriented rectangles. Touching does not count as overlapping. */
export function overlaps(a, b) {
  const pa = corners(a), pb = corners(b);
  for (const poly of [pa, pb]) {
    for (let i = 0; i < 4; i++) {
      const [x1, z1] = poly[i], [x2, z2] = poly[(i + 1) % 4];
      const nx = -(z2 - z1), nz = x2 - x1;                 // edge normal
      let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
      for (const [x, z] of pa) { const d = x * nx + z * nz; aMin = Math.min(aMin, d); aMax = Math.max(aMax, d); }
      for (const [x, z] of pb) { const d = x * nx + z * nz; bMin = Math.min(bMin, d); bMax = Math.max(bMax, d); }
      if (aMax <= bMin || bMax <= aMin) return false;      // a gap on this axis: they are apart
    }
  }
  return true;
}

/** Is (x, z) inside this bin's footprint? Even-odd ray crossing -- used to pick a bin by clicking. */
export function containsPoint(bin, x, z) {
  const pts = corners(bin);
  let inside = false;
  for (let a = 0, b = 3; a < 4; b = a++) {
    const [xa, za] = pts[a], [xb, zb] = pts[b];
    if ((za > z) !== (zb > z) && x < ((xb - xa) * (z - za)) / (zb - za) + xa) inside = !inside;
  }
  return inside;
}

/** Has this bin been changed from what the plan says stands in the room?
 *
 *  Only changed bins are judged. An existing bin sitting where it has always sat is a fact about the
 *  room, not a suggestion: it does not have to be wheelable in (it is already in), and its own
 *  footprint reads as partly unscanned because a depth scan never sees through a container. The server
 *  applies this same rule; if the two disagreed a bin would flash red under the pointer and then be
 *  declared fine on drop. */
export function isJudged(plan, bin) {
  if (bin.source !== "existing") return true;
  const same = (plan.bins || []).some((o) =>
    o.source === "existing" &&
    Math.abs(o.center[0] - bin.center[0]) <= TOL_M &&
    Math.abs(o.center[1] - bin.center[1]) <= TOL_M &&
    Math.abs(o.length_m - bin.length_m) <= TOL_M &&
    Math.abs(o.width_m - bin.width_m) <= TOL_M &&
    Math.abs(o.yaw_deg - bin.yaw_deg) <= TOL_DEG);
  return !same;
}

/* ------------------------------------------------------------------ masks */

/** The free-space grids, as read back from masks.png.
 *
 *  One 8-bit channel holding five grids as bit flags (see web_export.MASK_BITS). The bit positions
 *  arrive in plan.json rather than being hard-coded, so the numbers cannot drift from the Python table
 *  that wrote them.
 */
export class Masks {
  constructor(bytes, cols, rows, grid, bits) {
    this.data = bytes;
    this.cols = cols;
    this.rows = rows;
    this.cell = grid.cell;
    this.origin = grid.origin;
    this.bits = bits;
  }

  /** Flags at a world point, or 0 outside the grid (which reads as "never scanned"). */
  at(x, z) {
    const c = Math.floor((x - this.origin[0]) / this.cell);
    const r = Math.floor((z - this.origin[1]) / this.cell);
    if (r < 0 || c < 0 || r >= this.rows || c >= this.cols) return 0;
    return this.data[r * this.cols + c];
  }

  /** World centre of cell (r, c) -- the half-cell offset matters when drawing. */
  cellCenter(r, c) {
    return [this.origin[0] + (c + 0.5) * this.cell, this.origin[1] + (r + 0.5) * this.cell];
  }

  /** Fraction of a bin's footprint whose cells have `bit` set.
   *
   *  Deliberately samples the whole footprint, NOT the centre cell. On Frydenlundgata 4B not one of
   *  the five existing bins has an occupied centre and three have an unobserved one, because a depth
   *  scan of a closed container registers its sides and never sees through its middle. Testing the
   *  centre rejects bins that are unquestionably there.
   */
  fractionOn(bin, bit) {
    const yaw = (bin.yaw_deg * Math.PI) / 180;
    const cos = Math.cos(yaw), sin = Math.sin(yaw);
    const step = this.cell;
    let total = 0, hit = 0;
    for (let a = -bin.length_m / 2; a <= bin.length_m / 2; a += step) {
      for (let d = -bin.width_m / 2; d <= bin.width_m / 2; d += step) {
        const x = bin.center[0] + a * cos - d * sin;
        const z = bin.center[1] + a * sin + d * cos;
        total++;
        if (this.at(x, z) & bit) hit++;
      }
    }
    return total ? hit / total : 0;
  }
}

/** masks.png -> a Masks object.
 *
 *  Drawn on an offscreen canvas at 1:1 and read back. The PNG is a single OPAQUE 8-bit channel
 *  precisely so this survives: a browser may premultiply an RGBA image, and then the value of a fully
 *  transparent pixel is not guaranteed to come back out of getImageData -- which is exactly the pixel
 *  a mask needs to read.
 */
export async function loadMasks(blob, grid, bits) {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const image = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const bytes = new Uint8Array(bitmap.width * bitmap.height);
  for (let i = 0; i < bytes.length; i++) bytes[i] = image.data[i * 4];   // R carries the flags
  return new Masks(bytes, bitmap.width, bitmap.height, grid, bits);
}

/* -------------------------------------------------------------- verdicts */

/** The instant, local complaints about one bin: off the scanned floor, or on top of another bin.
 *  Deliberately does NOT include reachability -- that is the server's answer. */
export function localProblems(plan, masks, bins, index) {
  const bin = bins[index];
  if (!isJudged(plan, bin)) return [];
  const out = [];
  if (masks) {
    const frac = masks.fractionOn(bin, masks.bits.floor_observed);
    if (frac < MIN_ON_FLOOR) out.push(`${Math.round(frac * 100)} % på skannet gulv`);
  }
  for (let j = 0; j < bins.length; j++) {
    if (j !== index && overlaps(bin, bins[j])) { out.push("overlapper en annen kasse"); break; }
  }
  return out;
}

/** Snap an angle to the room's own axis, so bins stand parallel to the walls like the proposals do.
 *  Free rotation is available by holding a modifier; the default should be the tidy answer. */
export function snapYaw(yaw, roomAngleDeg) {
  let best = yaw, bestGap = Infinity;
  for (let k = 0; k < 4; k++) {
    const candidate = roomAngleDeg + k * 90;
    // compare on the circle, so 179 and -179 are two degrees apart and not 358
    const gap = Math.abs(((yaw - candidate + 540) % 360) - 180);
    if (gap < bestGap) { bestGap = gap; best = candidate; }
  }
  return ((best + 180) % 360) - 180;
}
