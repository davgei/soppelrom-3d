"""Free-floor-space analysis: how much flat floor is available (not covered by obstacles).

Reuses the backbone footprint (largest connected flat-floor region on a 5 cm grid). A cell is
FREE if flat floor was observed there and nothing stands above it; OCCUPIED if an obstacle
(bin, wall, clutter) projects onto it; UNKNOWN otherwise (occluded / unscanned). Bins occlude
the floor beneath them, so their footprints are already holes in the floor mask — correctly
excluded from free area.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.ndimage import label, median_filter

from .backbone import Footprint


@dataclass
class FreeSpaceResult:
    cell: float
    origin: np.ndarray            # (X, Z) world coord of grid cell (0, 0)
    floor_observed: np.ndarray    # bool grid [Z, X]
    occupied: np.ndarray          # bool grid [Z, X]
    free: np.ndarray              # bool grid [Z, X]
    free_area_m2: float
    observed_floor_area_m2: float
    occupied_on_floor_m2: float
    gross_area_m2: float


def compute_free_space(
    aligned_pcd: o3d.geometry.PointCloud,
    floor_height: float,
    footprint: Footprint,
    obstacle_min_height: float = 0.12,
    obstacle_max_height: float = 2.0,
    min_obstacle_area_m2: float = 0.06,
) -> FreeSpaceResult:
    cell = footprint.cell
    origin = footprint.origin
    floor_observed = footprint.mask.astype(bool)
    rows, cols = floor_observed.shape

    points = np.asarray(aligned_pcd.points)
    col_all = np.floor((points[:, 0] - origin[0]) / cell).astype(int)  # X -> column
    row_all = np.floor((points[:, 2] - origin[1]) / cell).astype(int)  # Z -> row
    inside = (col_all >= 0) & (col_all < cols) & (row_all >= 0) & (row_all < rows)
    col_all, row_all, y = col_all[inside], row_all[inside], points[inside, 1]

    # Local ground height per cell instead of one global floor plane. Outdoor yards slope and are
    # uneven, so measuring obstacle height from a single plane wrongly flags the far, empty end of
    # the floor as "occupied" (it drifts a few cm off the plane). Taking the lowest scanned point in
    # each cell as that cell's ground lets the free-space test follow the actual slope. A light
    # median smooth removes single-cell wells (stray low points) without spanning real obstacles.
    # Ground = from the local low point up to obstacle_min_height (~12 cm): terrain always varies a
    # little and that would never stop you rolling a bin there, so it must count as floor — this is
    # also what lets a bin reach right up to a wall, where the floor otherwise reads as red.
    ground = np.full((rows, cols), np.inf)
    np.minimum.at(ground, (row_all, col_all), y)
    ground[~np.isfinite(ground)] = floor_height
    ground = median_filter(ground, size=3)

    above = y - ground[row_all, col_all]
    obs = (above > obstacle_min_height) & (above < obstacle_max_height)
    occupied = np.zeros((rows, cols), dtype=bool)
    occupied[row_all[obs], col_all[obs]] = True

    # tiny occupied specks (a shoe, scan noise) can be pushed aside — treat them as free floor, not
    # as obstacles, so the free area is not eaten by red slivers. Real obstacles (bins, walls,
    # clutter) are large connected regions and survive.
    if occupied.any():
        labels, n = label(occupied)
        if n:
            sizes = np.bincount(labels.ravel())
            min_cells = max(1, int(min_obstacle_area_m2 / (cell * cell)))
            tiny = np.isin(labels, np.flatnonzero(sizes < min_cells)) & (labels > 0)
            occupied = occupied & ~tiny

    free = floor_observed & ~occupied
    area_per_cell = cell * cell
    return FreeSpaceResult(
        cell=cell,
        origin=origin,
        floor_observed=floor_observed,
        occupied=occupied,
        free=free,
        free_area_m2=float(free.sum() * area_per_cell),
        observed_floor_area_m2=float(floor_observed.sum() * area_per_cell),
        occupied_on_floor_m2=float((floor_observed & occupied).sum() * area_per_cell),
        gross_area_m2=float(footprint.area_m2),
    )
