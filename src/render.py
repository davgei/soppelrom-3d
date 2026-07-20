"""Headless orthographic previews of a point cloud, for verifying orientation without a GUI."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def _rasterize(
    points: np.ndarray,
    colors: np.ndarray,
    axis_u: int,
    axis_v: int,
    axis_depth: int,
    px_per_m: int,
    flip_v: bool,
) -> np.ndarray:
    u = points[:, axis_u]
    v = points[:, axis_v]
    depth = points[:, axis_depth]

    width = max(int((u.max() - u.min()) * px_per_m) + 1, 1)
    height = max(int((v.max() - v.min()) * px_per_m) + 1, 1)
    px = np.clip(((u - u.min()) * px_per_m).astype(int), 0, width - 1)
    py = np.clip(((v - v.min()) * px_per_m).astype(int), 0, height - 1)

    order = np.argsort(depth)  # draw far points first, nearer points overwrite
    image = np.zeros((height, width, 3), np.uint8)
    image[py[order], px[order]] = (colors[order] * 255).astype(np.uint8)
    if flip_v:
        image = image[::-1]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ortho_previews(pcd: o3d.geometry.PointCloud, out_dir: str | Path, px_per_m: int = 100) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # ARKit world has Y up: top-down looks along Y (u=X, v=Z); front looks along Z (u=X, v=Y).
    cv2.imwrite(str(out / "topdown_xz.png"), _rasterize(points, colors, 0, 2, 1, px_per_m, flip_v=True))
    cv2.imwrite(str(out / "front_xy.png"), _rasterize(points, colors, 0, 1, 2, px_per_m, flip_v=True))


def annotated_topdown(
    aligned_pcd: o3d.geometry.PointCloud,
    footprint,
    out_path: str | Path,
    px_per_m: int = 100,
) -> None:
    """Top-down view of the gravity-aligned cloud with the room footprint rectangle drawn on top."""
    points = np.asarray(aligned_pcd.points)
    colors = np.asarray(aligned_pcd.colors)
    u = points[:, 0]
    v = points[:, 2]
    depth = points[:, 1]

    u_min, v_min = u.min(), v.min()
    width = max(int((u.max() - u_min) * px_per_m) + 1, 1)
    height = max(int((v.max() - v_min) * px_per_m) + 1, 1)
    px = np.clip(((u - u_min) * px_per_m).astype(int), 0, width - 1)
    py = np.clip(((v - v_min) * px_per_m).astype(int), 0, height - 1)

    order = np.argsort(depth)
    image = np.zeros((height, width, 3), np.uint8)
    image[py[order], px[order]] = (colors[order] * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    box = cv2.boxPoints(footprint.rect)
    box_px = np.clip(((box[:, 0] - u_min) * px_per_m).astype(int), 0, width - 1)
    box_py = np.clip(((box[:, 1] - v_min) * px_per_m).astype(int), 0, height - 1)
    polygon = np.stack([box_px, box_py], axis=1).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    label = f"{footprint.length_m:.2f} x {footprint.width_m:.2f} m"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(str(Path(out_path)), image)


def freespace_topdown(result, out_path: str | Path, px_per_m: int = 100) -> None:
    """Top-down free-space map: green = free floor, red = occupied, gray = other observed floor."""
    rows, cols = result.free.shape
    image = np.zeros((rows, cols, 3), np.uint8)
    image[result.floor_observed] = (90, 90, 90)   # observed floor (gray, BGR)
    image[result.occupied] = (40, 40, 200)        # occupied (red)
    image[result.free] = (40, 180, 40)            # free (green)

    scale = max(int(px_per_m * result.cell), 1)
    image = cv2.resize(image, (cols * scale, rows * scale), interpolation=cv2.INTER_NEAREST)
    image = np.ascontiguousarray(image[::-1])  # flip Z (up = +Z); contiguous so cv2 can draw on it
    cv2.putText(
        image, f"Ledig gulv: {result.free_area_m2:.1f} m2", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 40), 2,
    )
    cv2.imwrite(str(Path(out_path)), image)


def detections_topdown(
    pcd: o3d.geometry.PointCloud,
    instances,
    out_path: str | Path,
    px_per_m: int = 100,
) -> None:
    """Top-down view with each detected bin drawn as a red footprint rectangle."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    u = points[:, 0]
    v = points[:, 2]

    u_min, v_min = u.min(), v.min()
    width = max(int((u.max() - u_min) * px_per_m) + 1, 1)
    height = max(int((v.max() - v_min) * px_per_m) + 1, 1)
    px = np.clip(((u - u_min) * px_per_m).astype(int), 0, width - 1)
    py = np.clip(((v - v_min) * px_per_m).astype(int), 0, height - 1)

    order = np.argsort(points[:, 1])
    image = np.zeros((height, width, 3), np.uint8)
    image[py[order], px[order]] = (colors[order] * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for index, inst in enumerate(instances):
        box = cv2.boxPoints(inst.rect)
        box_px = np.clip(((box[:, 0] - u_min) * px_per_m).astype(int), 0, width - 1)
        box_py = np.clip(((box[:, 1] - v_min) * px_per_m).astype(int), 0, height - 1)
        polygon = np.stack([box_px, box_py], axis=1).reshape(-1, 1, 2)
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        cx = int((inst.center[0] - u_min) * px_per_m)
        cy = int((inst.center[2] - v_min) * px_per_m)
        cv2.putText(
            image, f"#{index + 1}", (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
    cv2.imwrite(str(Path(out_path)), image)
