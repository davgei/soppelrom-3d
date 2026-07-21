"""Interactive 3D visualization of a reconstructed room in the Rerun viewer."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import open3d as o3d
import rerun as rr

from .scan_io import ScanArchive


def _ensure_viewer_on_path() -> None:
    """The bundled Rerun viewer lives in the venv's bin/Scripts dir, which is not on PATH
    when running via `python -m` without activating the venv. Add it so spawn() can find it."""
    bindir = os.path.join(sys.prefix, "Scripts" if os.name == "nt" else "bin")
    if os.path.isdir(bindir):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


def show(
    pcd: o3d.geometry.PointCloud,
    archive: ScanArchive | None = None,
    app_id: str = "soppelrom-3d",
) -> None:
    _ensure_viewer_on_path()
    rr.init(app_id, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    rr.log("world/room", rr.Points3D(points, colors=colors, radii=0.01))

    if archive is not None:
        camera_positions = np.array(
            [archive.keyframe(ts).pose_cam_to_world[:3, 3] for ts in archive.timestamps]
        )
        rr.log(
            "world/camera_path",
            rr.Points3D(camera_positions, colors=[0, 160, 255], radii=0.03),
        )


def _room_edges(footprint, floor_y: float, ceiling_y: float) -> list[np.ndarray]:
    corners_xz = cv2.boxPoints(footprint.rect)  # 4 x (X, Z)
    bottom = [np.array([x, floor_y, z]) for x, z in corners_xz]
    top = [np.array([x, ceiling_y, z]) for x, z in corners_xz]
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges.append(np.stack([bottom[i], bottom[j]]))
        edges.append(np.stack([top[i], top[j]]))
        edges.append(np.stack([bottom[i], top[i]]))
    return edges


def show_room(
    aligned_pcd: o3d.geometry.PointCloud,
    geometry,
    app_id: str = "soppelrom-3d",
) -> None:
    _ensure_viewer_on_path()
    rr.init(app_id, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    points = np.asarray(aligned_pcd.points)
    colors = (np.asarray(aligned_pcd.colors) * 255).astype(np.uint8)
    rr.log("world/room", rr.Points3D(points, colors=colors, radii=0.008))

    ceiling_y = geometry.floor_height_m + geometry.room_height_m
    edges = _room_edges(geometry.footprint, geometry.floor_height_m, ceiling_y)
    rr.log("world/room_box", rr.LineStrips3D(edges, colors=[255, 255, 0], radii=0.01))


def show_mesh(
    mesh: o3d.geometry.TriangleMesh,
    app_id: str = "soppelrom-3d",
) -> None:
    _ensure_viewer_on_path()
    rr.init(app_id, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = (
        (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        if mesh.has_vertex_colors()
        else None
    )
    vertex_normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
    rr.log(
        "world/mesh",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=triangles,
            vertex_colors=vertex_colors,
            vertex_normals=vertex_normals,
        ),
    )


def show_mesh_o3d(
    mesh: o3d.geometry.TriangleMesh,
    window_name: str = "soppelrom-3d mesh",
) -> None:
    """Open Open3D's in-process 3D window (no separate executable, so it works where a
    locked-down machine blocks launching the Rerun viewer). Drag to orbit, scroll to zoom."""
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh], window_name=window_name, mesh_show_back_face=True
    )


def show_pointcloud_o3d(
    pcd: o3d.geometry.PointCloud,
    window_name: str = "soppelrom-3d point cloud",
) -> None:
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def bin_box_lineset(
    rect: tuple, y_min: float, y_max: float, color: tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> o3d.geometry.LineSet:
    """Wireframe box for a detected bin: footprint rect (cv2.minAreaRect in X,Z) x height range."""
    corners_xz = cv2.boxPoints(rect)
    corners = [[x, y_min, z] for x, z in corners_xz] + [[x, y_max, z] for x, z in corners_xz]
    edges = (
        [[i, (i + 1) % 4] for i in range(4)]
        + [[4 + i, 4 + (i + 1) % 4] for i in range(4)]
        + [[i, 4 + i] for i in range(4)]
    )
    lineset = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.array(corners, dtype=float)),
        o3d.utility.Vector2iVector(np.array(edges)),
    )
    lineset.paint_uniform_color(color)
    return lineset


def show_scene_o3d(
    geometries: list, window_name: str = "soppelrom-3d"
) -> None:
    o3d.visualization.draw_geometries(
        geometries, window_name=window_name, mesh_show_back_face=True
    )


def _tinted_floor_cloud(
    aligned_pcd: o3d.geometry.PointCloud, fs, floor_height: float
) -> o3d.geometry.PointCloud:
    """Copy of the cloud with floor points painted green (free) / red (occupied) — the 'carpet'."""
    points = np.asarray(aligned_pcd.points)
    colors = np.asarray(aligned_pcd.colors).copy()
    cell, origin = fs.cell, fs.origin
    rows, cols = fs.free.shape

    near_floor = np.abs(points[:, 1] - floor_height) < 0.12
    col_idx = np.floor((points[:, 0] - origin[0]) / cell).astype(int)
    row_idx = np.floor((points[:, 2] - origin[1]) / cell).astype(int)
    valid = np.where(
        near_floor & (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
    )[0]
    colors[valid[fs.free[row_idx[valid], col_idx[valid]]]] = [0.1, 0.8, 0.1]
    colors[valid[fs.occupied[row_idx[valid], col_idx[valid]]]] = [0.85, 0.1, 0.1]

    tinted = o3d.geometry.PointCloud(aligned_pcd)
    tinted.colors = o3d.utility.Vector3dVector(colors)
    return tinted


def show_freespace_o3d(
    aligned_pcd: o3d.geometry.PointCloud,
    fs,
    floor_height: float,
    window_name: str = "Ledig gulv (gronn = ledig, rod = opptatt)",
) -> None:
    """Orbit the real scene with the free/occupied 'carpet' painted onto the floor."""
    o3d.visualization.draw_geometries([_tinted_floor_cloud(aligned_pcd, fs, floor_height)], window_name=window_name)


def show_placements_o3d(
    aligned_pcd: o3d.geometry.PointCloud,
    fs,
    result,
    floor_height: float,
    bin_height: float = 1.15,
    window_name: str = "Plass til ny kasse (gronn teppe = ledig, bla boks = kandidat)",
) -> None:
    """Free/occupied carpet on the floor PLUS candidate boxes (blue, to stand out from the carpet)."""
    geometries: list = [_tinted_floor_cloud(aligned_pcd, fs, floor_height)]
    for bx, bz, bl, bw, byaw in result.existing_bins:
        rect = ((bx, bz), (bl, bw), byaw)
        geometries.append(bin_box_lineset(rect, floor_height, floor_height + bin_height, color=(1.0, 0.1, 0.1)))
    for cand in result.candidates:
        box = bin_box_lineset(cand.rect, floor_height, floor_height + bin_height, color=(0.1, 0.4, 1.0))
        geometries.append(box)
    o3d.visualization.draw_geometries(geometries, window_name=window_name, mesh_show_back_face=True)
