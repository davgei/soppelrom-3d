"""Shared analysis + preview-render pipeline for one prepared scan.

Used by the dashboard GUI (and reusable by CLIs). Produces, per scan, a set of preview PNGs
and a stats.json under outputs/previews/<stem>/. Deliberately does NOT import prepare_scan
(which pulls in ultralytics) so the GUI starts fast.
"""
from __future__ import annotations

import json
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


def analyze_and_render(stem: str, bin_type: str) -> dict:
    """Compute room geometry, free space and bin placement for one prepared scan, render all
    preview PNGs, and return a stats dict (also written to stats.json)."""
    zip_path = RAW_DIR / f"{stem}.zip"
    cache_cloud = CACHE_ROOT / stem / "cloud.ply"
    ply = str(cache_cloud) if cache_cloud.exists() else None
    config = ReconstructionConfig(min_confidence=255, max_depth_m=5.0)
    pcd, archive, _ = load_point_cloud(zip_path, ply, config)

    geometry, aligned = backbone.analyze(pcd)
    footprint = geometry.footprint
    fs = freespace.compute_free_space(aligned, geometry.floor_height_m, footprint)

    out = preview_dir(stem)
    out.mkdir(parents=True, exist_ok=True)
    render.annotated_topdown(aligned, footprint, out / "room_topdown.png")
    render.freespace_over_scene(aligned, fs, out / "freespace_over_scene.png")

    rotation = geometry.rotation if geometry.rotation is not None else np.eye(3)
    existing = load_existing_bins(stem, rotation)
    poisson = CACHE_ROOT / stem / "mesh_poisson.ply"
    if poisson.exists():
        wall_points = np.asarray(o3d.io.read_triangle_mesh(str(poisson)).vertices) @ rotation.T
    else:
        wall_points = np.asarray(aligned.points)
    wall_mask = placement.build_wall_mask(fs, wall_points, geometry.floor_height_m, existing)
    camera_world = np.array([archive.keyframe(ts).pose_cam_to_world[:3, 3] for ts in archive.timestamps])
    camera_xz = (camera_world @ rotation.T)[:, [0, 2]]
    clicked = set_entrance.load_entrances(stem)  # stored in the original frame (like the boxes)
    if clicked:
        clicked3d = np.array([[x, 0.0, z] for x, z in clicked]) @ rotation.T
        entrances = [(float(p[0]), float(p[2])) for p in clicked3d]
    else:
        entrances = doors.find_doors(fs, footprint, wall_mask, camera_xz)
    length, _, width = BIN_TYPES[bin_type]
    result = placement.find_placements(
        fs, camera_xz, (length, width), bin_type, wall_mask=wall_mask,
        wall_angle_deg=footprint.angle_deg, existing_bins=existing, entrance_override=entrances,
    )
    render.placements_over_scene(aligned, result, out / "placements.png")

    stats = {
        "scan": stem,
        "bin_type": bin_type,
        "length_m": round(footprint.length_m, 2),
        "width_m": round(footprint.width_m, 2),
        "area_m2": round(footprint.area_m2, 1),
        "indoor": bool(geometry.is_indoor),
        "room_height_m": round(geometry.room_height_m, 2),
        "n_existing": len(existing),
        "free_area_m2": round(fs.free_area_m2, 1),
        "n_candidates": len(result.candidates),
        "n_entrances": len(result.entrances),
        "entrance_source": "klikket" if clicked else "auto",
        "address": _address(archive),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    archive.close()
    return stats
