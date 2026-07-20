"""Analyze a scan's geometric backbone: room dimensions and indoor/outdoor, with previews."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import backbone, freespace, render
from .loader import load_point_cloud
from .reconstruct import ReconstructionConfig


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

    if args.view:
        from . import visualize

        visualize.show_freespace_o3d(aligned, fs, geometry.floor_height_m)

    archive.close()


if __name__ == "__main__":
    main()
