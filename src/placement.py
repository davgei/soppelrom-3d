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
    entrance_xz: tuple[float, float] | None
    bin_type: str
    existing_bins: list[tuple[float, float, float, float, float]]


def _to_cells(points_xz: np.ndarray, origin: np.ndarray, cell: float, shape: tuple[int, int]):
    cols = np.floor((points_xz[:, 0] - origin[0]) / cell).astype(int)
    rows = np.floor((points_xz[:, 1] - origin[1]) / cell).astype(int)
    inside = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    return rows[inside], cols[inside]


def find_placements(
    fs: FreeSpaceResult,
    camera_xz: np.ndarray,
    footprint_lw: tuple[float, float],
    bin_type: str,
    wall_angle_deg: float = 0.0,
    margin: float = 0.20,
    existing_bins: list[tuple[float, float, float, float, float]] | None = None,
    entrance_override: tuple[float, float] | None = None,
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

    entrance_xz: tuple[float, float] | None = None
    if entrance_override is not None:
        entrance_xz = (float(entrance_override[0]), float(entrance_override[1]))
    elif len(camera_xz):
        start = camera_xz[: min(10, len(camera_xz))].mean(axis=0)
        entrance_xz = (float(start[0]), float(start[1]))
    if entrance_xz is not None:
        free_acc = free_acc & (np.hypot(wx - entrance_xz[0], wz - entrance_xz[1]) >= entrance_clear_radius)

    # keep existing bins' footprints and their pull-out lane (toward the exit) clear
    if existing_bins:
        occupied = np.zeros((rows, cols), np.uint8)
        target = np.array(entrance_xz) if entrance_xz is not None else np.array([wx.mean(), wz.mean()])
        apron = np.zeros((rows, cols), dtype=bool)
        for bx, bz, bl, bw, byaw in existing_bins:
            box = cv2.boxPoints(((bx, bz), (bl + 0.15, bw + 0.15), byaw))
            pts = np.stack([(box[:, 0] - origin[0]) / cell, (box[:, 1] - origin[1]) / cell], axis=1)
            cv2.fillPoly(occupied, [pts.astype(np.int32)], 1)
            direction = target - np.array([bx, bz])
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction /= norm
            along = (wx - bx) * direction[0] + (wz - bz) * direction[1]
            perp = np.abs(-(wx - bx) * direction[1] + (wz - bz) * direction[0])
            apron |= (along > -0.1) & (along < pull_out_lane) & (perp <= max(bw, width) / 2 + 0.1)
        free_acc = free_acc & (occupied == 0) & (~apron)

    # rotate to wall-aligned frame, then a rectangle erosion = "the footprint+margin fits here"
    rotation = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), wall_angle_deg, 1.0)
    rotated = cv2.warpAffine((free_acc.astype(np.uint8)) * 255, rotation, (cols, rows), flags=cv2.INTER_NEAREST)
    kx = max(1, int(round((length + 2 * margin) / cell)))
    ky = max(1, int(round((width + 2 * margin) / cell)))
    fits = cv2.erode(rotated, np.ones((ky, kx), np.uint8))
    clearance_rot = distance_transform_edt(rotated > 0) * cell
    inverse = cv2.invertAffineTransform(rotation)

    ys, xs = np.where(fits > 0)
    candidates: list[Candidate] = []
    if len(xs):
        world_x = origin[0] + (inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2] + 0.5) * cell
        world_z = origin[1] + (inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2] + 0.5) * cell
        if existing_bins:
            ex = np.array([[b[0], b[1]] for b in existing_bins])
            nearest = np.min(np.hypot(world_x[:, None] - ex[:, 0], world_z[:, None] - ex[:, 1]), axis=1)
            order = np.argsort(nearest)  # snap next to existing bins first (extend the row)
        else:
            order = np.argsort(clearance_rot[ys, xs])  # else hug walls, keep the middle open
        taken = np.zeros_like(fits, dtype=bool)
        exclusion = max(1, int(round((max(length, width) + spacing) / cell)))
        for k in order:
            r0, c0 = int(ys[k]), int(xs[k])
            if taken[r0, c0]:
                continue
            cx, cz = float(world_x[k]), float(world_z[k])
            candidates.append(
                Candidate(
                    center_xz=(cx, cz),
                    rect=((cx, cz), (float(length), float(width)), float(wall_angle_deg)),
                    length_m=float(length),
                    width_m=float(width),
                    clearance_m=float(clearance_rot[r0, c0]),
                )
            )
            taken[max(0, r0 - exclusion):r0 + exclusion, max(0, c0 - exclusion):c0 + exclusion] = True
            if len(candidates) >= max_candidates:
                break

    return PlacementResult(
        cell=cell,
        origin=origin,
        clearance=distance_transform_edt(fs.free) * cell,
        walkway=walkway,
        accessible=free_acc,
        candidates=candidates,
        entrance_xz=entrance_xz,
        bin_type=bin_type,
        existing_bins=existing_bins,
    )
