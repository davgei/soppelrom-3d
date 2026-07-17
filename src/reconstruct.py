"""Reconstruct a colored, metric point cloud from the raw RGB-D keyframes and their poses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .scan_io import Keyframe, ScanArchive

# ARKit/Polycam pose is camera-to-world in an OpenGL-style camera frame (x right, y up,
# looking down -z). Depth is unprojected in the OpenCV frame (x right, y down, +z forward),
# so we flip y and z before applying the pose. "opencv" keeps the raw frame (fallback).
_CAMERA_FLIP = {
    "arkit": np.diag([1.0, -1.0, -1.0]),
    "opencv": np.eye(3),
}


@dataclass
class ReconstructionConfig:
    min_confidence: int = 1  # drop depth pixels below this (0 = ARKit low/invalid)
    min_depth_m: float = 0.1
    max_depth_m: float = 8.0
    voxel_size_m: float = 0.02
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    camera_convention: str = "arkit"


def _unproject_frame(
    archive: ScanArchive, keyframe: Keyframe, config: ReconstructionConfig
) -> tuple[np.ndarray, np.ndarray]:
    depth = archive.depth_m(keyframe.timestamp)
    confidence = archive.confidence(keyframe.timestamp)
    rgb = archive.rgb(keyframe.timestamp)

    depth_h, depth_w = depth.shape
    rgb_h, rgb_w = rgb.shape[:2]
    scale_x = depth_w / keyframe.rgb_width
    scale_y = depth_h / keyframe.rgb_height
    fx = keyframe.intrinsics[0, 0] * scale_x
    fy = keyframe.intrinsics[1, 1] * scale_y
    cx = keyframe.intrinsics[0, 2] * scale_x
    cy = keyframe.intrinsics[1, 2] * scale_y

    us, vs = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    valid = (
        (confidence >= config.min_confidence)
        & (depth > config.min_depth_m)
        & (depth < config.max_depth_m)
    )
    u = us[valid]
    v = vs[valid]
    z = depth[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack([x, y, z], axis=1) @ _CAMERA_FLIP[config.camera_convention].T

    rotation = keyframe.pose_cam_to_world[:3, :3]
    translation = keyframe.pose_cam_to_world[:3, 3]
    points_world = points_camera @ rotation.T + translation

    u_rgb = np.clip((u / scale_x).astype(int), 0, rgb_w - 1)
    v_rgb = np.clip((v / scale_y).astype(int), 0, rgb_h - 1)
    colors = rgb[v_rgb, u_rgb].astype(np.float64) / 255.0
    return points_world, colors


def reconstruct(archive: ScanArchive, config: ReconstructionConfig) -> o3d.geometry.PointCloud:
    point_batches: list[np.ndarray] = []
    color_batches: list[np.ndarray] = []
    for timestamp in archive.timestamps:
        keyframe = archive.keyframe(timestamp)
        points, colors = _unproject_frame(archive, keyframe, config)
        if len(points):
            point_batches.append(points)
            color_batches.append(colors)

    if not point_batches:
        raise ValueError("no valid depth points found in any keyframe")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(point_batches))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(color_batches))
    pcd = pcd.voxel_down_sample(config.voxel_size_m)
    if config.outlier_neighbors > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            config.outlier_neighbors, config.outlier_std_ratio
        )
    return pcd
