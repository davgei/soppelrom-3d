"""Build a colored metric point cloud for a scan: use an exported PLY if present, else reconstruct."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from . import render
from .loader import resolve_ply
from .reconstruct import ReconstructionConfig, reconstruct
from .scan_io import ScanArchive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a colored metric point cloud from a Polycam scan "
        "(uses an exported PLY if one exists, otherwise reconstructs from raw RGB-D + poses)."
    )
    parser.add_argument("--scan", required=True, help="path to the raw capture .zip")
    parser.add_argument("--ply", default=None, help="optional exported .ply (overrides reconstruction)")
    parser.add_argument("--save", default=None, help="path to write the resulting .ply")
    parser.add_argument("--render-dir", default=None, help="write orthographic preview PNGs here")
    parser.add_argument("--view", action="store_true", help="open the Rerun 3D viewer")
    parser.add_argument("--voxel", type=float, default=0.02, help="voxel downsample size (m)")
    parser.add_argument("--min-confidence", type=int, default=1, help="min depth confidence to keep")
    parser.add_argument("--max-depth", type=float, default=8.0, help="max depth to keep (m)")
    parser.add_argument("--convention", choices=["arkit", "opencv"], default="arkit")
    args = parser.parse_args()

    zip_path = Path(args.scan)
    archive = ScanArchive(zip_path)
    print(f"scan: {zip_path.name}   paired keyframes: {len(archive.timestamps)}")

    ply_path = resolve_ply(zip_path, args.ply)
    if ply_path is not None:
        print(f"using existing PLY: {ply_path}")
        pcd = o3d.io.read_point_cloud(str(ply_path))
    else:
        print("no PLY found -> reconstructing from raw RGB-D + poses")
        config = ReconstructionConfig(
            min_confidence=args.min_confidence,
            max_depth_m=args.max_depth,
            voxel_size_m=args.voxel,
            camera_convention=args.convention,
        )
        pcd = reconstruct(archive, config)

    points = np.asarray(pcd.points)
    print(f"point cloud: {len(points)} points")
    if len(points):
        extent = points.max(axis=0) - points.min(axis=0)
        print(f"bbox extent (m): x={extent[0]:.2f}  y={extent[1]:.2f}  z={extent[2]:.2f}")
        plane, inliers = pcd.segment_plane(0.03, 3, 500)
        normal = np.array(plane[:3])
        normal /= np.linalg.norm(normal)
        print(
            f"largest plane normal: [{normal[0]:.2f} {normal[1]:.2f} {normal[2]:.2f}]  "
            f"inliers: {len(inliers)}/{len(points)}"
        )

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"saved point cloud -> {args.save}")

    if args.render_dir:
        render.ortho_previews(pcd, args.render_dir)
        print(f"orthographic previews -> {args.render_dir}")

    if args.view:
        from . import visualize

        visualize.show(pcd, archive)

    archive.close()


if __name__ == "__main__":
    main()
