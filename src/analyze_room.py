"""Analyze a scan's geometric backbone: room dimensions and indoor/outdoor, with previews."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import backbone, freespace, placement, render
from .annotations import BIN_TYPES, load_annotations
from .loader import load_point_cloud
from .reconstruct import ReconstructionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_ROOT / "outputs" / "annotations"
CACHE_ROOT = PROJECT_ROOT / "outputs" / "cache"


def _load_existing_bins(
    scan_stem: str, rotation: np.ndarray
) -> list[tuple[float, float, float, float, float]]:
    """Approved annotations if present, else the cached auto-proposals, transformed to the
    gravity-aligned frame as (cx, cz, length, width, yaw)."""
    annotated = ANNOTATION_DIR / f"{scan_stem}.json"
    proposals = CACHE_ROOT / scan_stem / "proposals.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure room dimensions and classify indoor/outdoor from a scan."
    )
    parser.add_argument("--scan", required=True, help="path to the raw capture .zip")
    parser.add_argument("--ply", default=None, help="optional exported/saved .ply to load instead")
    parser.add_argument("--render-dir", default=None, help="write an annotated top-down preview here")
    parser.add_argument("--view", action="store_true", help="open the Rerun 3D viewer")
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--place", default=None, choices=list(BIN_TYPES),
                        help="find spots for a NEW bin of this type")
    parser.add_argument("--margin", type=float, default=0.20, help="clearance margin around a new bin (m)")
    args = parser.parse_args()

    config = ReconstructionConfig(
        min_confidence=args.min_confidence, max_depth_m=args.max_depth, voxel_size_m=args.voxel
    )
    pcd, archive, source = load_point_cloud(args.scan, args.ply, config)
    print(f"scan: {Path(args.scan).name}   source: {source}   points: {len(pcd.points)}")

    geometry, aligned = backbone.analyze(pcd)
    footprint = geometry.footprint

    print("\n=== Room geometry ===")
    print(f"footprint (length x width): {footprint.length_m:.2f} x {footprint.width_m:.2f} m")
    print(f"floor area:                 {footprint.area_m2:.1f} m^2")
    if geometry.is_indoor:
        print(f"room height (floor->ceil):  {geometry.room_height_m:.2f} m")
    else:
        print(f"content height in area:     {geometry.room_height_m:.2f} m  (98pct; open, no ceiling)")
    print(f"indoor/outdoor:             {'INDOOR' if geometry.is_indoor else 'OUTDOOR / open'}")
    print(f"reason:                     {geometry.reason}")

    fs = freespace.compute_free_space(aligned, geometry.floor_height_m, footprint)
    print("\n=== Free floor space ===")
    print(f"observed flat floor:        {fs.observed_floor_area_m2:.1f} m^2")
    print(f"occupied on floor:          {fs.occupied_on_floor_m2:.1f} m^2")
    print(f"FREE floor area:            {fs.free_area_m2:.1f} m^2")

    placement_result = None
    if args.place:
        length, _, width = BIN_TYPES[args.place]
        rotation = geometry.rotation if geometry.rotation is not None else np.eye(3)
        camera_world = np.array(
            [archive.keyframe(ts).pose_cam_to_world[:3, 3] for ts in archive.timestamps]
        )
        camera_xz = (camera_world @ rotation.T)[:, [0, 2]]
        existing_bins = _load_existing_bins(Path(args.scan).stem, rotation)
        placement_result = placement.find_placements(
            fs, camera_xz, (length, width), args.place,
            wall_angle_deg=footprint.angle_deg, margin=args.margin, existing_bins=existing_bins,
        )
        print(f"tar hensyn til {len(existing_bins)} eksisterende kasse(r)")
        print(f"\n=== Plass til ny '{args.place}' ({length:.2f} x {width:.2f} m + {args.margin:.2f} m margin) ===")
        print(f"mulige plasseringer: {len(placement_result.candidates)}")
        for index, cand in enumerate(placement_result.candidates, start=1):
            print(f"  #{index}: klaring {cand.clearance_m:.2f} m  @ ({cand.center_xz[0]:.2f}, {cand.center_xz[1]:.2f})")

    if args.render_dir:
        out = Path(args.render_dir) / "room_topdown.png"
        render.annotated_topdown(aligned, footprint, out)
        print(f"\nannotated preview -> {out}")
        fs_out = Path(args.render_dir) / "freespace_topdown.png"
        render.freespace_topdown(fs, fs_out)
        print(f"free-space preview -> {fs_out}")
        scene_out = Path(args.render_dir) / "freespace_over_scene.png"
        render.freespace_over_scene(aligned, fs, scene_out)
        print(f"free-space over real scene -> {scene_out}")
        if placement_result is not None:
            place_out = Path(args.render_dir) / "placements_over_scene.png"
            render.placements_over_scene(aligned, placement_result, place_out)
            print(f"placement preview -> {place_out}")

    if args.view:
        from . import visualize

        if placement_result is not None:
            visualize.show_placements_o3d(
                aligned, fs, placement_result, geometry.floor_height_m, BIN_TYPES[args.place][1]
            )
        else:
            visualize.show_freespace_o3d(aligned, fs, geometry.floor_height_m)

    archive.close()


if __name__ == "__main__":
    main()
