"""Geometric backbone: gravity alignment, floor/ceiling detection, room dimensions, indoor/outdoor.

Assumes the point cloud is roughly Y-up (ARKit world convention). No training data required.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes

UP = np.array([0.0, 1.0, 0.0])


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / (np.linalg.norm(vector) + 1e-12)


@dataclass
class Footprint:
    center_xz: tuple[float, float]
    length_m: float
    width_m: float
    angle_deg: float
    area_m2: float
    rect: tuple  # cv2.minAreaRect result, in (X, Z) metres
    mask: np.ndarray  # occupancy grid of the main floor region (rows = Z, cols = X)
    origin: np.ndarray  # (X, Z) world coordinate of grid cell (0, 0)
    cell: float


def points_in_footprint(points_xz: np.ndarray, footprint: Footprint) -> np.ndarray:
    cells = np.floor((points_xz - footprint.origin) / footprint.cell).astype(int)
    height, width = footprint.mask.shape
    inside = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < width)
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < height)
    )
    result = np.zeros(len(points_xz), dtype=bool)
    valid = cells[inside]
    result[inside] = footprint.mask[valid[:, 1], valid[:, 0]] > 0
    return result


@dataclass
class RoomGeometry:
    point_count: int
    floor_height_m: float
    ceiling_height_m: float | None
    room_height_m: float
    footprint: Footprint
    is_indoor: bool
    ceiling_coverage: float
    reason: str
    rotation: np.ndarray | None = None  # gravity-align rotation (original -> aligned frame)


def _find_horizontal_planes(
    pcd: o3d.geometry.PointCloud, dist_thresh: float, max_planes: int, min_frac: float
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    work = o3d.geometry.PointCloud(pcd)
    n_total = len(pcd.points)
    planes: list[tuple[float, np.ndarray, np.ndarray]] = []
    for _ in range(max_planes):
        if len(work.points) < max(1000, int(min_frac * n_total)):
            break
        model, inliers = work.segment_plane(dist_thresh, 3, 500)
        points = np.asarray(work.points)[inliers]
        normal = _unit(model[:3])
        if len(inliers) >= min_frac * n_total and abs(normal @ UP) > 0.85:
            planes.append((float(points[:, 1].mean()), normal, points))
        work = work.select_by_index(inliers, invert=True)
    return planes


def _gravity_align(
    pcd: o3d.geometry.PointCloud, floor_normal: np.ndarray
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    normal = _unit(floor_normal)
    if normal @ UP < 0:
        normal = -normal
    axis = np.cross(normal, UP)
    sin_angle = np.linalg.norm(axis)
    aligned = o3d.geometry.PointCloud(pcd)
    if sin_angle < 1e-8:
        return aligned, np.eye(3)
    rotation_vector = (axis / sin_angle) * np.arccos(np.clip(normal @ UP, -1.0, 1.0))
    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
    aligned.rotate(rotation, center=(0.0, 0.0, 0.0))
    return aligned, rotation


def _footprint(floor_xz: np.ndarray, cell: float = 0.05, min_count: int = 3) -> Footprint:
    origin = floor_xz.min(axis=0)
    cells = np.floor((floor_xz - origin) / cell).astype(int)
    grid_w = int(cells[:, 0].max()) + 1
    grid_h = int(cells[:, 1].max()) + 1
    counts = np.zeros((grid_h, grid_w), dtype=np.int32)
    np.add.at(counts, (cells[:, 1], cells[:, 0]), 1)
    occupied = (counts >= min_count).astype(np.uint8)

    # Close first to bridge gaps where the floor is occluded (objects standing on it), so the room
    # stays ONE region; then open to drop thin stray streaks. Without the close the largest
    # connected component can collapse to a narrow strip and the footprint comes out far too thin.
    occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # ~0.35 m
    occupied = cv2.morphologyEx(occupied, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(occupied, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        occupied = (labels == largest).astype(np.uint8)
    occupied = binary_fill_holes(occupied).astype(np.uint8)  # count floor hidden under objects too

    rows, cols = np.where(occupied > 0)
    centers = np.stack([cols, rows], axis=1).astype(np.float64) * cell + origin + cell / 2
    rect = cv2.minAreaRect(centers.astype(np.float32))
    (cx, cz), (side_a, side_b), angle = rect
    length, width = (side_a, side_b) if side_a >= side_b else (side_b, side_a)
    return Footprint(
        center_xz=(float(cx), float(cz)),
        length_m=float(length),
        width_m=float(width),
        angle_deg=float(angle),
        area_m2=float(length * width),
        rect=rect,
        mask=occupied,
        origin=origin,
        cell=cell,
    )


def _ceiling_coverage(floor_xz: np.ndarray, ceiling_xz: np.ndarray, cell: float = 0.1) -> float:
    origin = floor_xz.min(axis=0)
    floor_keys = set(map(tuple, np.floor((floor_xz - origin) / cell).astype(int)))
    ceiling_keys = set(map(tuple, np.floor((ceiling_xz - origin) / cell).astype(int)))
    if not floor_keys:
        return 0.0
    return len(floor_keys & ceiling_keys) / len(floor_keys)


def analyze(
    pcd: o3d.geometry.PointCloud,
    dist_thresh: float = 0.03,
    max_planes: int = 8,
    min_plane_frac: float = 0.03,
    min_ceiling_height_m: float = 1.2,
    indoor_coverage_threshold: float = 0.5,
    seed: int = 42,
) -> tuple[RoomGeometry, o3d.geometry.PointCloud]:
    o3d.utility.random.seed(seed)  # RANSAC is randomized; seed it so results are reproducible
    planes = _find_horizontal_planes(pcd, dist_thresh, max_planes, min_plane_frac)
    if not planes:
        raise ValueError("no horizontal floor plane found")

    # Floor = the LOWEST horizontal plane. The ceiling is always above the floor, so choosing by
    # size is what mislabels an enclosed/cluttered room's ceiling as the floor: there the ceiling
    # is the biggest plane, and if the floor is occluded a size filter drops it, leaving only the
    # ceiling. Every candidate already passed the min-fraction + horizontal tests, and nothing sits
    # below the real floor, so the lowest one is the floor. planes[i][0] = mean height (~Y-up).
    _, floor_normal, floor_points = min(planes, key=lambda plane: plane[0])

    aligned, rotation = _gravity_align(pcd, floor_normal)
    aligned_points = np.asarray(aligned.points)
    floor_points_aligned = floor_points @ rotation.T
    floor_height = float(np.median(floor_points_aligned[:, 1]))
    floor_xz = floor_points_aligned[:, [0, 2]]

    footprint = _footprint(floor_xz)

    # Ceiling = the largest horizontal plane sitting clearly above the floor, if any.
    ceiling_height: float | None = None
    coverage = 0.0
    above_floor = [
        (pts, float((pts @ rotation.T)[:, 1].mean()))
        for (_, _, pts) in planes
    ]
    above_floor = [(pts, h) for (pts, h) in above_floor if h - floor_height > min_ceiling_height_m]
    if above_floor:
        ceiling_points, ceiling_h = max(above_floor, key=lambda c: len(c[0]))
        coverage = _ceiling_coverage(floor_xz, (ceiling_points @ rotation.T)[:, [0, 2]])
        ceiling_height = ceiling_h

    is_indoor = coverage >= indoor_coverage_threshold
    if ceiling_height is not None and is_indoor:
        room_height = ceiling_height - floor_height
    else:
        in_region = points_in_footprint(aligned_points[:, [0, 2]], footprint)
        heights = aligned_points[in_region, 1] - floor_height
        room_height = float(np.percentile(heights, 98)) if len(heights) else 0.0

    reason = (
        f"ceiling covers {coverage * 100:.0f}% of the floor -> indoor"
        if is_indoor
        else f"no ceiling over the floor (coverage {coverage * 100:.0f}%) -> outdoor/open"
    )

    geometry = RoomGeometry(
        point_count=len(pcd.points),
        floor_height_m=floor_height,
        ceiling_height_m=ceiling_height,
        room_height_m=room_height,
        footprint=footprint,
        is_indoor=is_indoor,
        ceiling_coverage=coverage,
        reason=reason,
        rotation=rotation,
    )
    return geometry, aligned
