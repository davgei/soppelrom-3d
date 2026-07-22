"""Where can a NEW bin go? Pure geometry, no training.

A bin fits where its whole footprint (plus a clearance margin) lands on free floor — tested by
eroding the free mask with the bin rectangle, rotated to the wall direction so candidates line
up with the walls. Real bins stand AGAINST a wall (leaving the middle open to walk), so we rank
wall-hugging spots first. The camera trajectory (where the scanner walked) gives accessibility
and the entrance (its start), which is kept clear. No labels, fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .freespace import FreeSpaceResult


@dataclass
class Candidate:
    center_xz: tuple[float, float]
    rect: tuple            # cv2.minAreaRect-style ((cx,cz),(L,W),angle) in aligned X/Z
    length_m: float
    width_m: float
    clearance_m: float     # distance from the bin centre to the nearest wall/obstacle
    bin_type: str = ""     # which bin type this slot is sized for (mixed packing)


@dataclass
class PlacementResult:
    cell: float
    origin: np.ndarray
    clearance: np.ndarray
    walkway: np.ndarray
    accessible: np.ndarray
    candidates: list[Candidate]
    entrances: list[tuple[float, float]]
    bin_type: str
    existing_bins: list[tuple[float, float, float, float, float]]
    reachable: np.ndarray | None = None  # push-path corridor (blue): entrance -> around existing bins
    route: np.ndarray | None = None      # thin near-straight route skeleton the corridor is grown from


def _to_cells(points_xz: np.ndarray, origin: np.ndarray, cell: float, shape: tuple[int, int]):
    cols = np.floor((points_xz[:, 0] - origin[0]) / cell).astype(int)
    rows = np.floor((points_xz[:, 1] - origin[1]) / cell).astype(int)
    inside = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    return rows[inside], cols[inside]


def build_wall_mask(
    fs: FreeSpaceResult,
    points: np.ndarray,
    floor_height: float,
    existing_bins: list[tuple[float, float, float, float, float]] | None = None,
    wall_height: float = 1.0,
    min_wall_extent: float = 1.2,
) -> np.ndarray:
    """Grid cells that are WALL: tall structure (a point >wall_height above the floor) that is
    not an existing bin, keeping only long connected runs (real walls, not scattered vegetation
    or clutter). Prefer feeding the watertight Poisson vertices so unscanned holes don't read
    as gaps."""
    cell, origin = fs.cell, fs.origin
    rows, cols = fs.free.shape
    height_map = np.zeros((rows, cols))
    height = points[:, 1] - floor_height
    col = np.floor((points[:, 0] - origin[0]) / cell).astype(int)
    row = np.floor((points[:, 2] - origin[1]) / cell).astype(int)
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows) & (height > 0.3)
    np.maximum.at(height_map, (row[inside], col[inside]), height[inside])
    wall = height_map > wall_height

    bins_mask = np.zeros((rows, cols), np.uint8)
    for bx, bz, bl, bw, byaw in existing_bins or []:
        box = cv2.boxPoints(((bx, bz), (bl + 0.25, bw + 0.25), byaw))
        pts = np.stack([(box[:, 0] - origin[0]) / cell, (box[:, 1] - origin[1]) / cell], axis=1)
        cv2.fillPoly(bins_mask, [pts.astype(np.int32)], 1)
    wall = wall & (bins_mask == 0)

    labels, n = label(wall)  # keep only long runs = real walls
    for i in range(1, n + 1):
        cells = np.argwhere(labels == i)
        extent = (cells.max(axis=0) - cells.min(axis=0) + 1) * cell
        if max(extent) < min_wall_extent:
            wall[labels == i] = False
    return wall


def detect_entrances(
    fs: FreeSpaceResult,
    footprint,
    wall_mask: np.ndarray,
    camera_xz: np.ndarray,
    min_gap_m: float = 0.5,
) -> list[tuple[float, float]]:
    """Auto-find doorways: gaps in the wall ring around the room where the floor leaks out and
    the scanner actually walked (that last part rejects ragged scan edges). Best-effort; the
    manual click overrides it."""
    cell, origin = fs.cell, fs.origin
    rows, cols = fs.free.shape
    floor_region = footprint.mask.astype(bool)
    wall = wall_mask

    ring = max(1, int(0.4 / cell))
    outer = binary_dilation(floor_region, iterations=ring) & ~floor_region
    wall_near = binary_dilation(wall, iterations=max(1, int(0.35 / cell)))
    opening = outer & ~wall_near

    if len(camera_xz):  # a real doorway is where the scanner went, not a ragged scan edge
        walked = np.zeros((rows, cols), dtype=bool)
        r_idx, c_idx = _to_cells(camera_xz, origin, cell, fs.free.shape)
        walked[r_idx, c_idx] = True
        opening = opening & binary_dilation(walked, iterations=max(1, int(0.9 / cell)))

    labels, n = label(opening)
    entrances: list[tuple[float, float]] = []
    for i in range(1, n + 1):
        cells = np.argwhere(labels == i)
        extent = (cells.max(axis=0) - cells.min(axis=0) + 1) * cell
        if max(extent) < min_gap_m:
            continue
        cr, cc = cells.mean(axis=0)
        entrances.append((float(origin[0] + (cc + 0.5) * cell), float(origin[1] + (cr + 0.5) * cell)))

    if not entrances and len(camera_xz):  # fall back to the scan-start door
        start = camera_xz[: min(10, len(camera_xz))].mean(axis=0)
        entrances = [(float(start[0]), float(start[1]))]
    return entrances


def reachable_from_entrance(
    fs: FreeSpaceResult,
    rollable: np.ndarray,
    entrances: list[tuple[float, float]],
    passage_width: float,
    margin: float = 0.05,
    seed_radius: float = 1.5,
) -> np.ndarray:
    """Floor cells a bin `passage_width` wide can actually be WHEELED to from an entrance.

    `rollable` is the free, accessible floor (walls/obstacles already excluded). We keep only the
    cells whose distance to the nearest blocked cell is >= passage_width/2 (so the bin fits — a
    corridor at least that wide), then flood-fill from each entrance: a spot counts only if it is
    in the same connected corridor as a door. This enforces a clear path the WHOLE way, not just
    room at the exit. With no entrance (e.g. a sealed room) nothing is reachable."""
    cell, origin = fs.cell, fs.origin
    reachable = np.zeros(rollable.shape, dtype=bool)
    if not entrances or not rollable.any():
        return reachable
    # bridge small clutter holes first ("push it aside") so scattered debris doesn't fragment the
    # corridor and make a genuinely walkable room look impassable
    walkable = binary_closing(rollable, iterations=max(1, int(0.15 / cell)))
    clearance = distance_transform_edt(walkable) * cell
    corridor = walkable & (clearance >= passage_width / 2 + margin)
    if not corridor.any():
        return reachable
    labels, _ = label(corridor)
    ys, xs = np.where(corridor)
    cx = origin[0] + (xs + 0.5) * cell
    cz = origin[1] + (ys + 0.5) * cell
    for ex, ez in entrances:
        dist = np.hypot(cx - ex, cz - ez)
        nearest = int(np.argmin(dist))
        if dist[nearest] <= seed_radius:  # the corridor actually reaches this door
            reachable |= labels == labels[ys[nearest], xs[nearest]]
    return reachable


def _box_mask(rect: tuple, origin: np.ndarray, cell: float, shape: tuple[int, int], grow: float = 0.0) -> np.ndarray:
    """Boolean grid of the footprint of one oriented box (optionally grown by `grow` metres)."""
    (cx, cz), (length, width), angle = rect
    box = cv2.boxPoints(((cx, cz), (length + grow, width + grow), angle))
    mask = np.zeros(shape, np.uint8)
    pts = np.stack([(box[:, 0] - origin[0]) / cell, (box[:, 1] - origin[1]) / cell], axis=1)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def _boxes_mask(bins, origin: np.ndarray, cell: float, shape: tuple[int, int], grow: float = 0.0) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for bx, bz, bl, bw, byaw in bins:
        mask |= _box_mask(((bx, bz), (bl, bw), byaw), origin, cell, shape, grow)
    return mask


def _nearest_true(mask: np.ndarray, row: int, col: int) -> tuple[int, int] | None:
    """The cell in `mask` closest to (row, col); None if the mask is empty."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    k = int(np.argmin((ys - row) ** 2 + (xs - col) ** 2))
    return int(ys[k]), int(xs[k])


def _grid_graph(passable: np.ndarray, node_cost: np.ndarray):
    """8-connected shortest-path graph over the passable cells. Edge weight = step length x the
    mean traversal cost of its two endpoints, so a route prefers wide, open floor and only squeezes
    through narrow gaps when it must. Returns (csr graph, node-id grid, row coords, col coords)."""
    rows, cols = passable.shape
    node_id = -np.ones((rows, cols), dtype=np.int64)
    ys, xs = np.where(passable)
    node_id[ys, xs] = np.arange(ys.size)
    src, dst, wgt = [], [], []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        step = float(np.hypot(dr, dc))
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        a = node_id[r0:r1, c0:c1]
        b = node_id[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        ca = node_cost[r0:r1, c0:c1]
        cb = node_cost[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        m = (a >= 0) & (b >= 0)
        src.append(a[m]); dst.append(b[m]); wgt.append(step * 0.5 * (ca[m] + cb[m]))
    n = ys.size
    graph = coo_matrix((np.concatenate(wgt), (np.concatenate(src), np.concatenate(dst))),
                       shape=(n, n)).tocsr()
    return graph, node_id, ys, xs


def route_corridor(
    fs: FreeSpaceResult,
    rollable: np.ndarray,
    entrances: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    passage_width: float,
    margin: float = 0.05,
    closing_m: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The push-path: a NEAR-STRAIGHT route from an entrance to (and around) every existing bin,
    plus a corridor grown to the width of the largest bin.

    Returns (route, corridor, region):
      route    — the thin shortest-path skeleton (entrance -> each target), no big detours;
      corridor — that route widened to passage_width and clipped to real floor (the sacred path);
      region   — ALL floor reachable from an entrance at all (used to test if a new bin can be
                 wheeled in). Uses a cost-based shortest path (narrow gaps cost more but are not
                 blocked), so clutter no longer shatters the path into holes the way a hard
                 minimum-width mask does. With no entrance nothing is reachable."""
    cell = fs.cell
    origin = fs.origin
    shape = fs.free.shape
    empty = np.zeros(shape, dtype=bool)
    # bridge small clutter gaps ("push it aside") so a route can thread through a messy room
    passable = rollable | (binary_closing(rollable, iterations=max(1, int(closing_m / cell))) & fs.floor_observed)
    if not passable.any() or not entrances:
        return empty, empty, empty

    clearance = distance_transform_edt(passable) * cell
    short = np.maximum(0.0, passage_width / 2 + margin - clearance)  # metres the bin is too wide by
    node_cost = 1.0 + 2.5 * (short / cell)                          # narrow => costly, open => cheap
    graph, node_id, ys, xs = _grid_graph(passable, node_cost)
    if ys.size == 0:
        return empty, empty, empty

    ent_nodes = []
    for ex, ez in entrances:
        c = int(round((ex - origin[0]) / cell))
        r = int(round((ez - origin[1]) / cell))
        near = _nearest_true(passable, r, c)
        if near is not None:
            ent_nodes.append(int(node_id[near]))
    ent_nodes = sorted(set(e for e in ent_nodes if e >= 0))
    if not ent_nodes:
        return empty, empty, empty

    dist, pred, _ = dijkstra(graph, directed=False, indices=ent_nodes,
                             return_predecessors=True, min_only=True)
    finite = np.isfinite(dist)
    region = empty.copy()
    region[ys[finite], xs[finite]] = True

    route = empty.copy()
    for tx, tz in targets:
        c = int(round((tx - origin[0]) / cell))
        r = int(round((tz - origin[1]) / cell))
        near = _nearest_true(region, r, c)  # snap the bin to the nearest reachable cell
        if near is None:
            continue
        node = int(node_id[near])
        guard = 0
        while node >= 0 and guard < ys.size + 5:
            route[ys[node], xs[node]] = True
            node = int(pred[node])  # -9999 at an entrance (source)
            guard += 1

    corridor = empty.copy()
    if route.any():
        corridor = binary_dilation(route, iterations=max(1, int(passage_width / 2 / cell))) & passable
    return route, corridor, region


def pack_placements(
    fs: FreeSpaceResult,
    wall_mask: np.ndarray | None,
    accessible: np.ndarray,
    entrances: list[tuple[float, float]],
    existing_bins: list,
    bin_specs: list[tuple[str, float, float]],
    passage_width: float,
    wall_angle_deg: float = 0.0,
    margin: float = 0.20,
    spacing: float = 0.15,
    max_bins: int = 40,
    wall_weight: float = 3.0,
    near_weight: float = 1.0,
) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    """Fill free floor with a MIX of bin types (bin_specs = (name, length, width), largest first).

    Order of priority, matching how the user reasons about it:
      1. lay the push-path FIRST — a near-straight route from a door to every existing bin — and
         treat it as SACRED: no new bin may sit on it (so the existing bins can always be wheeled
         out), EXCEPT a small apron right around each existing bin, which is reclaimed as usable
         floor so new bins can line up next to the old ones;
      2. then fill the remaining free floor, hugging the walls/edges and clustering near the
         existing bins. Bins never overlap; each must be reachable from a door.
    Returns (candidates, corridor, route)."""
    cell, origin = fs.cell, fs.origin
    shape = fs.free.shape
    free_acc = fs.free & accessible
    existing_centers = [(b[0], b[1]) for b in existing_bins]
    occ = _boxes_mask(existing_bins, origin, cell, shape, grow=0.10)

    # 1) the push-path, computed once and then frozen
    route, corridor, region = route_corridor(fs, free_acc & ~occ, entrances, existing_centers, passage_width)
    apron = binary_dilation(_boxes_mask(existing_bins, origin, cell, shape, grow=0.0),
                            iterations=max(1, int(0.55 / cell)))  # reclaimable ring around each bin
    protected = corridor & ~apron
    placeable_region = region | binary_dilation(corridor, iterations=1)  # floor reachable from a door

    # wall/edge preference: a real bin stands against a WALL — the room's perimeter or a detected
    # tall wall — not marooned in the open middle and not clinging to a stray clutter island in the
    # centre. So "wall" = the outer rim of the observed floor plus any tall wall, and `wall_dist`
    # measures how far a spot is from it: ~0 against a wall, large out in the open.
    perimeter = fs.floor_observed & ~binary_erosion(fs.floor_observed, iterations=2)
    walls = perimeter | wall_mask if wall_mask is not None else perimeter
    wall_dist_map = distance_transform_edt(~walls) * cell if walls.any() else np.full(shape, 5.0)

    from . import place_prior  # local import: keeps placement importable without torch present
    prior = place_prior.cached_prior()
    dist_wall, clearance = (place_prior.scene_feature_maps(fs, wall_mask)
                            if prior is not None else (None, None))
    placed_centers: list[tuple[float, float]] = []
    placed: list[Candidate] = []

    def _wall_m(xz: tuple[float, float]) -> float:
        c = int(np.clip((xz[0] - origin[0]) / cell, 0, shape[1] - 1))
        r = int(np.clip((xz[1] - origin[1]) / cell, 0, shape[0] - 1))
        return float(wall_dist_map[r, c])

    def _try(spot: Candidate, name: str, length: float, width: float) -> None:
        nonlocal occ
        cand_mask = _box_mask(spot.rect, origin, cell, shape, grow=spacing)
        if (cand_mask & occ).any():
            return  # overlaps an existing/placed bin
        if (cand_mask & protected).any():
            return  # sits on the sacred push-path
        if not (cand_mask & placeable_region).any():
            return  # cannot be wheeled here from a door
        placed.append(Candidate(spot.center_xz, spot.rect, length, width, spot.clearance_m, bin_type=name))
        occ = occ | cand_mask
        placed_centers.append(spot.center_xz)

    def _rank(spots: list[Candidate]) -> list[Candidate]:
        if not spots:
            return spots
        wall_bonus = np.array([np.clip(1.0 - _wall_m(c.center_xz) / 1.5, 0.0, 1.0) for c in spots])
        if existing_centers:
            ex = np.array(existing_centers)
            near = np.array([float(np.min(np.hypot(ex[:, 0] - c.center_xz[0], ex[:, 1] - c.center_xz[1])))
                             for c in spots])
            near_bonus = np.clip(1.0 - near / 3.0, 0.0, 1.0)
        else:
            near_bonus = np.zeros(len(spots))
        if prior is not None:  # learned prior: how much the spot looks like where bins really stand
            others = existing_centers + placed_centers
            feats = [place_prior.features_at(c.center_xz, fs, dist_wall, clearance, others, entrances)
                     for c in spots]
            base = np.array(prior.score(feats))
        else:
            base = np.zeros(len(spots))
        score = base + wall_weight * wall_bonus + near_weight * near_bonus
        return [s for _, s in sorted(zip(score, spots), key=lambda pair: -float(pair[0]))]

    pool = max(4 * max_bins, 30)  # a generous pool so ranking can actually choose the best spots
    for name, length, width in bin_specs:  # largest first
        # bins hug the edge of the free floor; open middle only if the room has no wall-adjacent room
        spots = _wall_candidates(free_acc & ~occ, walls, length, width, origin, cell, spacing, pool)
        if not spots:
            spots = _open_floor_candidates(free_acc & ~occ, wall_angle_deg, length, width, margin,
                                           origin, cell, existing_bins, spacing, pool)
        for spot in _rank(spots):
            if len(placed) >= max_bins:
                break
            _try(spot, name, length, width)
        if len(placed) >= max_bins:
            break
    return placed, corridor, route


def _box_corners(center, wall_dir, inward, along, into) -> np.ndarray:
    return np.array(
        [
            center - wall_dir * along / 2 - inward * into / 2,
            center - wall_dir * along / 2 + inward * into / 2,
            center + wall_dir * along / 2 + inward * into / 2,
            center + wall_dir * along / 2 - inward * into / 2,
        ]
    )


def _box_fits(allowed: np.ndarray, corners: np.ndarray, origin: np.ndarray, cell: float,
              tol: float = 0.0) -> bool:
    """True if the box footprint lands on `allowed` floor. `tol` is the fraction of the footprint
    permitted to fall outside — a little slack so a bin can still hug a jagged, real-world edge
    instead of being pushed metres into the open just to clear the ragged scan boundary."""
    mask = np.zeros(allowed.shape, np.uint8)
    pts = np.stack([(corners[:, 0] - origin[0]) / cell, (corners[:, 1] - origin[1]) / cell], axis=1)
    cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], 1)
    covered = mask.astype(bool)
    total = int(covered.sum())
    if total == 0:
        return False
    outside = int((covered & ~allowed).sum())
    return outside <= tol * total


def _wall_candidates(
    free_acc: np.ndarray,
    wall_mask: np.ndarray,
    length: float,
    width: float,
    origin: np.ndarray,
    cell: float,
    spacing: float,
    max_candidates: int,
    wall_gap: float = 0.10,
) -> list[Candidate]:
    """Place bins hugging the walls / edge of the free floor. A bin parks with its back a small gap
    from the wall and its LONG side along the wall (short side into the room — how bins actually
    stand), oriented by the local wall normal. Candidates are generated densely along every edge
    and a little slack is allowed at ragged edges; the caller ranks and de-conflicts them."""
    if wall_mask is None or not wall_mask.any():
        return []
    span = max(length, width)   # runs along the wall
    depth = min(length, width)  # sticks into the room
    distance, (row_idx, col_idx) = distance_transform_edt(~wall_mask, return_indices=True)
    distance_m = distance * cell
    rows, cols = free_acc.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    dir_col = xx - col_idx  # world X points along columns
    dir_row = yy - row_idx  # world Z points along rows

    target = depth / 2 + wall_gap
    band = free_acc & (distance_m >= target - cell) & (distance_m <= target + cell * 8)
    ys, xs = np.where(band)
    if not len(xs):
        return []

    taken = np.zeros(free_acc.shape, dtype=bool)
    dilate = max(1, int(spacing / cell))
    candidates: list[Candidate] = []
    for k in np.argsort(np.abs(distance_m[ys, xs] - target)):
        r0, c0 = int(ys[k]), int(xs[k])
        if taken[r0, c0]:
            continue
        inward = np.array([dir_col[r0, c0], dir_row[r0, c0]], dtype=float)  # (x, z), wall -> room
        norm = np.linalg.norm(inward)
        if norm < 1e-6:
            continue
        inward /= norm
        wall_dir = np.array([-inward[1], inward[0]])
        ctr = np.array([origin[0] + (c0 + 0.5) * cell, origin[1] + (r0 + 0.5) * cell])
        box = _box_corners(ctr, wall_dir, inward, span, depth)
        if _box_fits(free_acc & ~taken, box, origin, cell, tol=0.12):
            rect = cv2.minAreaRect(box.astype(np.float32))
            candidates.append(Candidate((float(ctr[0]), float(ctr[1])), rect, float(length), float(width), float(distance_m[r0, c0])))
            mask = np.zeros(free_acc.shape, np.uint8)
            bpts = np.stack([(box[:, 0] - origin[0]) / cell, (box[:, 1] - origin[1]) / cell], axis=1)
            cv2.fillPoly(mask, [bpts.astype(np.int32).reshape(-1, 1, 2)], 1)
            taken |= binary_dilation(mask.astype(bool), iterations=dilate)
            if len(candidates) >= max_candidates:
                break
    return candidates


def _open_floor_candidates(
    free_acc: np.ndarray,
    wall_angle_deg: float,
    length: float,
    width: float,
    margin: float,
    origin: np.ndarray,
    cell: float,
    existing_bins: list,
    spacing: float,
    max_candidates: int,
) -> list[Candidate]:
    """Fallback when no wall spots exist: erode the free floor by the footprint and pick the
    most wall-hugging fits."""
    rows, cols = free_acc.shape
    rotation = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), wall_angle_deg, 1.0)
    rotated = cv2.warpAffine((free_acc.astype(np.uint8)) * 255, rotation, (cols, rows), flags=cv2.INTER_NEAREST)
    kx = max(1, int(round((length + 2 * margin) / cell)))
    ky = max(1, int(round((width + 2 * margin) / cell)))
    fits = cv2.erode(rotated, np.ones((ky, kx), np.uint8))
    clearance_rot = distance_transform_edt(rotated > 0) * cell
    inverse = cv2.invertAffineTransform(rotation)

    ys, xs = np.where(fits > 0)
    candidates: list[Candidate] = []
    if not len(xs):
        return candidates
    world_x = origin[0] + (inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2] + 0.5) * cell
    world_z = origin[1] + (inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2] + 0.5) * cell
    score = clearance_rot[ys, xs].copy()
    if existing_bins:
        ex = np.array([[b[0], b[1]] for b in existing_bins])
        nearest = np.min(np.hypot(world_x[:, None] - ex[:, 0], world_z[:, None] - ex[:, 1]), axis=1)
        score = clearance_rot[ys, xs] + 0.3 * nearest
    taken = np.zeros_like(fits, dtype=bool)
    exclusion = max(1, int(round((max(length, width) + spacing) / cell)))
    for k in np.argsort(score):
        r0, c0 = int(ys[k]), int(xs[k])
        if taken[r0, c0]:
            continue
        cx, cz = float(world_x[k]), float(world_z[k])
        candidates.append(Candidate((cx, cz), ((cx, cz), (float(length), float(width)), float(wall_angle_deg)),
                                     float(length), float(width), float(clearance_rot[r0, c0])))
        taken[max(0, r0 - exclusion):r0 + exclusion, max(0, c0 - exclusion):c0 + exclusion] = True
        if len(candidates) >= max_candidates:
            break
    return candidates


def find_placements(
    fs: FreeSpaceResult,
    camera_xz: np.ndarray,
    footprint_lw: tuple[float, float],
    bin_type: str,
    wall_mask: np.ndarray | None = None,
    wall_angle_deg: float = 0.0,
    margin: float = 0.20,
    existing_bins: list[tuple[float, float, float, float, float]] | None = None,
    entrance_override: list[tuple[float, float]] | None = None,
    entrance_clear_radius: float = 1.0,
    pull_out_lane: float = 1.0,
    spacing: float = 0.15,
    max_candidates: int = 8,
    passage_width: float | None = None,
    bin_specs: list[tuple[str, float, float]] | None = None,
) -> PlacementResult:
    """existing_bins: (cx, cz, length, width, yaw_deg) per already-present bin, in the aligned frame.
    New bins line up NEXT TO them (ranked by proximity) and never sit in their pull-out lane.

    passage_width: only keep spots a bin this wide can be wheeled to a door (clear corridor the
    whole way). Defaults to the placed bin's short side; pass the largest bin's short side to
    require a path wide enough for the biggest bin in the room. entrance_override=[] (not None)
    means "no entrances" (e.g. a sealed room) -> nothing is reachable -> no candidates."""
    existing_bins = existing_bins or []
    cell, origin = fs.cell, fs.origin
    free = fs.free.copy()
    rows, cols = free.shape
    length, width = footprint_lw

    yy, xx = np.mgrid[0:rows, 0:cols]
    wx = origin[0] + (xx + 0.5) * cell
    wz = origin[1] + (yy + 0.5) * cell

    # accessibility: free floor connected to where the scanner walked (else the largest region)
    walkway = np.zeros_like(free, dtype=bool)
    if len(camera_xz):
        r_idx, c_idx = _to_cells(camera_xz, origin, cell, free.shape)
        walkway[r_idx, c_idx] = True
        walkway = binary_dilation(walkway, iterations=max(1, int(0.3 / cell))) & free
    labels, n_labels = label(free)
    if walkway.any():
        touched = set(np.unique(labels[walkway & (labels > 0)]))
        accessible = np.isin(labels, list(touched))
    elif n_labels:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        accessible = labels == int(sizes.argmax())
    else:
        accessible = free
    free_acc = free & accessible
    rollable = free_acc.copy()  # floor a bin can roll over (before placement-only exclusions)

    entrances: list[tuple[float, float]] = []
    if entrance_override is not None:  # [] means "no entrances" (sealed room); None = auto-detect
        entrances = [(float(x), float(z)) for x, z in entrance_override]
    elif len(camera_xz):
        start = camera_xz[: min(10, len(camera_xz))].mean(axis=0)
        entrances = [(float(start[0]), float(start[1]))]
    for ex, ez in entrances:  # keep a clear zone in each doorway
        free_acc = free_acc & (np.hypot(wx - ex, wz - ez) >= entrance_clear_radius)

    # keep existing bins' footprints and their pull-out lane (toward the NEAREST door) clear
    if existing_bins:
        entrance_arr = np.array(entrances) if entrances else np.array([[wx.mean(), wz.mean()]])
        occupied = np.zeros((rows, cols), np.uint8)
        apron = np.zeros((rows, cols), dtype=bool)
        for bx, bz, bl, bw, byaw in existing_bins:
            box = cv2.boxPoints(((bx, bz), (bl + 0.15, bw + 0.15), byaw))
            pts = np.stack([(box[:, 0] - origin[0]) / cell, (box[:, 1] - origin[1]) / cell], axis=1)
            cv2.fillPoly(occupied, [pts.astype(np.int32)], 1)
            nearest = entrance_arr[np.argmin(np.hypot(entrance_arr[:, 0] - bx, entrance_arr[:, 1] - bz))]
            direction = nearest - np.array([bx, bz])
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction /= norm
            along = (wx - bx) * direction[0] + (wz - bz) * direction[1]
            perp = np.abs(-(wx - bx) * direction[1] + (wz - bz) * direction[0])
            apron |= (along > -0.1) & (along < pull_out_lane) & (perp <= max(bw, width) / 2 + 0.1)
        free_acc = free_acc & (occupied == 0) & (~apron)

    passage = passage_width if passage_width is not None else min(length, width)
    route = None
    if bin_specs is not None:
        # multi-type: lay the push-path first (sacred), then fill the remaining free floor with a
        # MIX of bin types hugging the walls and clustering near the existing bins, never overlapping
        candidates, reachable, route = pack_placements(
            fs, wall_mask, accessible, entrances, existing_bins, bin_specs, passage,
            wall_angle_deg=wall_angle_deg, margin=margin, spacing=spacing, max_bins=max_candidates,
        )
    else:
        # legacy single-type: wall-hugging spots (open-floor fallback), then keep only those a bin
        # can be wheeled to a door
        candidates = _wall_candidates(free_acc, wall_mask, length, width, origin, cell, spacing, max_candidates)
        if not candidates:
            candidates = _open_floor_candidates(
                free_acc, wall_angle_deg, length, width, margin, origin, cell,
                existing_bins, spacing, max_candidates,
            )
        reachable = reachable_from_entrance(fs, rollable, entrances, passage)

        def _reachable(center_xz: tuple[float, float]) -> bool:
            col = int((center_xz[0] - origin[0]) / cell)
            row = int((center_xz[1] - origin[1]) / cell)
            return 0 <= row < rows and 0 <= col < cols and bool(reachable[row, col])

        candidates = [c for c in candidates if _reachable(c.center_xz)]

    return PlacementResult(
        cell=cell,
        origin=origin,
        clearance=distance_transform_edt(fs.free) * cell,
        walkway=walkway,
        accessible=free_acc,
        candidates=candidates,
        entrances=entrances,
        bin_type=bin_type,
        existing_bins=existing_bins,
        reachable=reachable,
        route=route,
    )
