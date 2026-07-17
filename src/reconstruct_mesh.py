"""Reconstruct a colored surface mesh from the raw RGB-D keyframes via TSDF fusion.

A meshed surface is far easier to read than a raw point cloud, so this is the preferred
visualization layer. Uses Open3D's ScalableTSDFVolume with the posed depth + color frames.

The depth maps are fully dense (every pixel has a value), so holes in the mesh come only from
confidence filtering. We therefore use a RADIAL confidence threshold: keep even low-confidence
depth near the image centre (where LiDAR is reliable and TSDF fusion averages out noise), and
demand high confidence toward the edges (where depth is noisiest). This fills central holes
while keeping the borders clean.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from .scan_io import ScanArchive

# Depth is unprojected by Open3D in the OpenCV camera frame (x right, y down, +z forward),
# while the stored pose is camera-to-world in an OpenGL-style frame (y up, looking down -z).
# The extrinsic (world -> OpenCV camera) is therefore FLIP @ inv(pose), FLIP = diag(1,-1,-1,1).
_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class MeshConfig:
    voxel_length: float = 0.03
    sdf_trunc: float = 0.09
    depth_trunc: float = 5.0
    center_confidence: int = 0    # required confidence at the image centre (0 = keep everything)
    edge_confidence: int = 255    # required confidence at the image edge (255 = high only)
    center_radius: float = 0.4    # normalized radius (0..1) below which center_confidence applies
    edge_radius: float = 0.85     # normalized radius above which edge_confidence applies
    min_cluster_triangles: int = 200  # drop disconnected mesh fragments smaller than this
    poisson_depth: int = 9        # Poisson octree depth (higher = finer, slower)
    density_quantile: float = 0.05  # trim this lowest fraction of Poisson vertices (anti-balloon)


def _radial_confidence_threshold(shape: tuple[int, int], config: MeshConfig) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
    radius /= np.sqrt(2.0)  # normalize so a corner is 1.0
    ramp = np.clip(
        (radius - config.center_radius) / max(config.edge_radius - config.center_radius, 1e-6), 0, 1
    )
    return config.center_confidence + (config.edge_confidence - config.center_confidence) * ramp


def _remove_small_clusters(mesh: o3d.geometry.TriangleMesh, min_triangles: int) -> None:
    cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_sizes = np.asarray(cluster_sizes)
    too_small = cluster_sizes[cluster_ids] < min_triangles
    mesh.remove_triangles_by_mask(too_small)
    mesh.remove_unreferenced_vertices()


def _integrate(archive: ScanArchive, config: MeshConfig) -> o3d.pipelines.integration.ScalableTSDFVolume:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_length,
        sdf_trunc=config.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    threshold: np.ndarray | None = None
    for timestamp in archive.timestamps:
        keyframe = archive.keyframe(timestamp)
        depth = archive.depth_m(timestamp).copy()
        confidence = archive.confidence(timestamp)
        rgb = archive.rgb(timestamp)

        depth_h, depth_w = depth.shape
        if threshold is None:
            threshold = _radial_confidence_threshold((depth_h, depth_w), config)
        depth[confidence < threshold] = 0.0
        depth[depth > config.depth_trunc] = 0.0

        color = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0,
            depth_trunc=config.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        scale_x = depth_w / keyframe.rgb_width
        scale_y = depth_h / keyframe.rgb_height
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth_w,
            depth_h,
            keyframe.intrinsics[0, 0] * scale_x,
            keyframe.intrinsics[1, 1] * scale_y,
            keyframe.intrinsics[0, 2] * scale_x,
            keyframe.intrinsics[1, 2] * scale_y,
        )
        extrinsic = _FLIP @ np.linalg.inv(keyframe.pose_cam_to_world)
        volume.integrate(rgbd, intrinsic, extrinsic)

    return volume


def reconstruct_mesh(archive: ScanArchive, config: MeshConfig) -> o3d.geometry.TriangleMesh:
    """TSDF surface mesh: only surfaces actually observed, so it can have holes."""
    mesh = _integrate(archive, config).extract_triangle_mesh()
    if config.min_cluster_triangles > 0:
        _remove_small_clusters(mesh, config.min_cluster_triangles)
    mesh.compute_vertex_normals()
    return mesh


def reconstruct_mesh_poisson(archive: ScanArchive, config: MeshConfig) -> o3d.geometry.TriangleMesh:
    """Poisson surface mesh: a watertight surface that interpolates gaps from the point normals
    and colors (fills holes), then trimmed by density and cropped to the data to curb ballooning."""
    pcd = _integrate(archive, config).extract_point_cloud()
    if not pcd.has_normals():
        pcd.estimate_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=config.poisson_depth
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, config.density_quantile))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    if config.min_cluster_triangles > 0:
        _remove_small_clusters(mesh, config.min_cluster_triangles)
    mesh.compute_vertex_normals()
    return mesh
