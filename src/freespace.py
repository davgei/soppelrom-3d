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
    obstacle_min_height: float = 0.08,
    obstacle_max_height: float = 2.0,
) -> FreeSpaceResult:
    cell = footprint.cell
    origin = footprint.origin
    floor_observed = footprint.mask.astype(bool)
    rows, cols = floor_observed.shape

    points = np.asarray(aligned_pcd.points)
    height = points[:, 1] - floor_height
    obstacles = points[(height > obstacle_min_height) & (height < obstacle_max_height)]

    occupied = np.zeros((rows, cols), dtype=bool)
    if len(obstacles):
        col_idx = np.floor((obstacles[:, 0] - origin[0]) / cell).astype(int)  # X -> column
        row_idx = np.floor((obstacles[:, 2] - origin[1]) / cell).astype(int)  # Z -> row
        inside = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
        occupied[row_idx[inside], col_idx[inside]] = True

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
