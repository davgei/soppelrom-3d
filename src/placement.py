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
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from .freespace import FreeSpaceResult


@dataclass
class Candidate:
    center_xz: tuple[float, float]
    rect: tuple            # cv2.minAreaRect-style ((cx,cz),(L,W),angle) in aligned X/Z
    length_m: float
    width_m: float
    clearance_m: float     # distance from the bin centre to the nearest wall/obstacle


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


def _box_corners(center, wall_dir, inward, along, into) -> np.ndarray:
    return np.array(
        [
            center - wall_dir * along / 2 - inward * into / 2,
            center - wall_dir * along / 2 + inward * into / 2,
            center + wall_dir * along / 2 + inward * into / 2,
            center + wall_dir * along / 2 - inward * into / 2,
        ]
    )


def _box_fits(allowed: np.ndarray, corners: np.ndarray, origin: np.ndarray, cell: float) -> bool:
    mask = np.zeros(allowed.shape, np.uint8)
    pts = np.stack([(corners[:, 0] - origin[0]) / cell, (corners[:, 1] - origin[1]) / cell], axis=1)
    cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], 1)
    covered = mask.astype(bool)
    return bool(covered.any()) and not bool(np.any(covered & ~allowed))


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
    """Place bins hugging the ACTUAL walls: a bin sits where the distance to the nearest wall
    equals length/2 + gap (its short side then rests against the wall), oriented by the wall
    normal so it sticks `length` into the room and spans `width` along the wall. Greedy along
    each wall so they sit next to each other with a small gap."""
    if wall_mask is None or not wall_mask.any():
        return []
    distance, (row_idx, col_idx) = distance_transform_edt(~wall_mask, return_indices=True)
    distance_m = distance * cell
    rows, cols = free_acc.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    dir_col = xx - col_idx  # world X points along columns
    dir_row = yy - row_idx  # world Z points along rows

    target = length / 2 + wall_gap
    band = free_acc & (np.abs(distance_m - target) < cell * 1.5)
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
        box = _box_corners(ctr, wall_dir, inward, width, length)
        if _box_fits(free_acc & ~taken, box, origin, cell):
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
    max_candidates: int = 12,
) -> PlacementResult:
    """existing_bins: (cx, cz, length, width, yaw_deg) per already-present bin, in the aligned frame.
    New bins line up NEXT TO them (ranked by proximity) and never sit in their pull-out lane."""
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

    entrances: list[tuple[float, float]] = []
    if entrance_override:
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

    # 1) preferred: line the bins up along the walls, short side to the wall
    candidates = _wall_candidates(
        free_acc, wall_mask, length, width, origin, cell, spacing, max_candidates
    )
    # 2) fallback (no wall spots): most wall-hugging open-floor fits
    if not candidates:
        candidates = _open_floor_candidates(
            free_acc, wall_angle_deg, length, width, margin, origin, cell,
            existing_bins, spacing, max_candidates,
        )

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
    )
