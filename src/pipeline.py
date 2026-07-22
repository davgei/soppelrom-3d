"""Shared analysis + preview-render pipeline for one prepared scan.

Used by the dashboard GUI (and reusable by CLIs). Produces, per scan, a set of preview PNGs
and a stats.json under outputs/previews/<stem>/. Deliberately does NOT import prepare_scan
(which pulls in ultralytics) so the GUI starts fast.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from . import backbone, doors, freespace, placement, render, set_entrance
from .annotations import BIN_TYPES, load_annotations
from .loader import load_point_cloud
from .reconstruct import ReconstructionConfig

from .paths import ANNOTATION_DIR, CACHE_ROOT, PREVIEW_ROOT, PROJECT_ROOT, RAW_DIR


def list_scans() -> list[str]:
    return [p.stem for p in sorted(RAW_DIR.glob("*.zip"))]


def is_prepared(stem: str) -> bool:
    return (CACHE_ROOT / stem / "done.flag").exists()


def is_annotated(stem: str) -> bool:
    return (ANNOTATION_DIR / f"{stem}.json").exists()


def preview_dir(stem: str) -> Path:
    return PREVIEW_ROOT / stem


def existing_bin_count(stem: str) -> int:
    path = ANNOTATION_DIR / f"{stem}.json"
    if not path.exists():
        path = CACHE_ROOT / stem / "proposals.json"
    if not path.exists():
        return 0
    _, boxes = load_annotations(path)
    return len(boxes)


def load_existing_bins(stem: str, rotation: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    annotated = ANNOTATION_DIR / f"{stem}.json"
    proposals = CACHE_ROOT / stem / "proposals.json"
    path = annotated if annotated.exists() else (proposals if proposals.exists() else None)
    if path is None:
        return []
    _, boxes = load_annotations(path)
    result = []
    for box in boxes:
        center = rotation @ np.asarray(box.center)
        length = max(box.extent[0], box.extent[2])
        width = min(box.extent[0], box.extent[2])
        result.append((float(center[0]), float(center[2]), float(length), float(width), float(box.yaw_deg)))
    return result


def _address(archive) -> str | None:
    try:
        location = archive.gps(archive.timestamps[0])
        place = location.get("placemark", {}) if location else {}
        parts = [
            f"{place.get('thoroughfare', '')} {place.get('subThoroughfare', '')}".strip(),
            f"{place.get('postalCode', '')} {place.get('locality', '')}".strip(),
        ]
        joined = ", ".join(p for p in parts if p)
        return joined or None
    except Exception:
        return None


@dataclass
class Scene:
    """Everything computed for one prepared scan — shared by the preview renderer and 3D viewer."""
    stem: str
    bin_type: str
    aligned: o3d.geometry.PointCloud
    scene_vis: o3d.geometry.PointCloud            # dense mesh-sampled backdrop (readable)
    mesh: o3d.geometry.TriangleMesh | None        # gravity-aligned Poisson mesh (dollhouse)
    geometry: object
    footprint: object
    fs: object
    wall_mask: np.ndarray
    result: "placement.PlacementResult"
    existing: list
    entrances: list
    enclosed: bool
    clicked: bool
    address: str | None
    floor_height: float
    rotation: np.ndarray


def compute_scene(stem: str, bin_type: str) -> Scene:
    """Load/reconstruct the scan and compute room geometry, free space, entrances and bin
    placement. Shared by analyze_and_render (writes PNGs) and place3d (the 3D viewer)."""
    zip_path = RAW_DIR / f"{stem}.zip"
    cache_cloud = CACHE_ROOT / stem / "cloud.ply"
    ply = str(cache_cloud) if cache_cloud.exists() else None
    pcd, archive, _ = load_point_cloud(zip_path, ply, ReconstructionConfig(min_confidence=255, max_depth_m=5.0))

    geometry, aligned = backbone.analyze(pcd)
    footprint = geometry.footprint
    fs = freespace.compute_free_space(aligned, geometry.floor_height_m, footprint)
    rotation = geometry.rotation if geometry.rotation is not None else np.eye(3)

    mesh = None
    poisson = CACHE_ROOT / stem / "mesh_poisson.ply"
    if poisson.exists():
        loaded = o3d.io.read_triangle_mesh(str(poisson))
        if loaded.has_triangles():
            loaded.rotate(rotation, center=(0.0, 0.0, 0.0))  # gravity-align like the cloud
            mesh = loaded
    if mesh is not None and mesh.has_vertex_colors():
        scene_vis = mesh.sample_points_uniformly(number_of_points=1_000_000)  # readable backdrop
    else:
        scene_vis = aligned

    existing = load_existing_bins(stem, rotation)
    wall_points = np.asarray(mesh.vertices) if mesh is not None else np.asarray(aligned.points)
    wall_mask = placement.build_wall_mask(fs, wall_points, geometry.floor_height_m, existing)
    camera_world = np.array([archive.keyframe(ts).pose_cam_to_world[:3, 3] for ts in archive.timestamps])
    camera_xz = (camera_world @ rotation.T)[:, [0, 2]]

    # a room scanned with the door shut is a sealed box (only scan holes) — no way in, so skip it
    enclosed = doors.is_enclosed(fs, footprint, wall_mask)
    clicked = set_entrance.load_entrances(stem)  # stored in the original frame (like the boxes)
    if enclosed:
        entrances: list = []  # [] (not None) tells find_placements there is no entrance
    elif clicked:
        clicked3d = np.array([[x, 0.0, z] for x, z in clicked]) @ rotation.T
        entrances = [(float(p[0]), float(p[2])) for p in clicked3d]
    else:
        entrances = doors.find_doors(fs, footprint, wall_mask, camera_xz)

    length, _, width = BIN_TYPES[bin_type]
    # fill the room with a MIX of the real bin types, largest first; the push-corridor need only be
    # as wide as the SHORTEST side of the LARGEST bin
    real = sorted(("4-hjuls container", "2-hjuls dunk"),
                  key=lambda t: BIN_TYPES[t][0] * BIN_TYPES[t][2], reverse=True)
    bin_specs = [(t, BIN_TYPES[t][0], BIN_TYPES[t][2]) for t in real]
    passage_width = min(BIN_TYPES[real[0]][0], BIN_TYPES[real[0]][2])
    result = placement.find_placements(
        fs, camera_xz, (length, width), bin_type, wall_mask=wall_mask,
        wall_angle_deg=footprint.angle_deg, existing_bins=existing, entrance_override=entrances,
        passage_width=passage_width, bin_specs=bin_specs,
    )
    address = _address(archive)
    archive.close()
    return Scene(stem, bin_type, aligned, scene_vis, mesh, geometry, footprint, fs, wall_mask, result,
                 existing, entrances, enclosed, bool(clicked), address, geometry.floor_height_m, rotation)


def analyze_and_render(stem: str, bin_type: str) -> dict:
    """Compute one scan and render all preview PNGs; returns (and writes) a stats dict."""
    scene = compute_scene(stem, bin_type)
    out = preview_dir(stem)
    out.mkdir(parents=True, exist_ok=True)
    render.annotated_topdown(scene.scene_vis, scene.footprint, out / "room_topdown.png")
    render.freespace_over_scene(scene.scene_vis, scene.fs, out / "freespace_over_scene.png")
    render.placements_over_scene(scene.scene_vis, scene.result, out / "placements.png")

    stats = {
        "scan": stem,
        "bin_type": bin_type,
        "length_m": round(scene.footprint.length_m, 2),
        "width_m": round(scene.footprint.width_m, 2),
        "area_m2": round(scene.footprint.area_m2, 1),
        "indoor": bool(scene.geometry.is_indoor),
        "room_height_m": round(scene.geometry.room_height_m, 2),
        "n_existing": len(scene.existing),
        "free_area_m2": round(scene.fs.free_area_m2, 1),
        "n_candidates": len(scene.result.candidates),
        "n_entrances": len(scene.result.entrances),
        "closed_room": bool(scene.enclosed),
        "entrance_source": "innesperret" if scene.enclosed else ("klikket" if scene.clicked else "auto"),
        "address": scene.address,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
