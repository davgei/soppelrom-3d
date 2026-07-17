"""Reconstruct a colored surface mesh from a scan (TSDF fusion) and optionally view it in Rerun."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from .reconstruct_mesh import MeshConfig, reconstruct_mesh, reconstruct_mesh_poisson
from .scan_io import ScanArchive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a colored surface mesh from a Polycam scan (TSDF fusion of RGB-D)."
    )
    parser.add_argument("--scan", required=True, help="path to the raw capture .zip")
    parser.add_argument("--save", default=None, help="path to write the mesh (.ply / .obj)")
    parser.add_argument("--view", action="store_true", help="open an interactive 3D viewer")
    parser.add_argument(
        "--viewer",
        choices=["open3d", "rerun"],
        default="open3d",
        help="which viewer to use (open3d runs in-process; rerun needs the bundled viewer exe)",
    )
    parser.add_argument("--voxel", type=float, default=0.03, help="TSDF voxel length (m)")
    parser.add_argument("--sdf-trunc", type=float, default=0.09, help="TSDF truncation distance (m)")
    parser.add_argument("--max-depth", type=float, default=5.0, help="max depth to integrate (m)")
    parser.add_argument("--center-confidence", type=int, default=0, help="required confidence at image centre (0 = keep all)")
    parser.add_argument("--edge-confidence", type=int, default=255, help="required confidence at image edge")
    parser.add_argument(
        "--method",
        choices=["tsdf", "poisson"],
        default="tsdf",
        help="tsdf = observed surfaces only (can have holes); poisson = watertight, interpolates gaps",
    )
    parser.add_argument("--poisson-depth", type=int, default=9, help="Poisson octree depth (higher = finer)")
    parser.add_argument("--density-quantile", type=float, default=0.05, help="trim lowest fraction of Poisson vertices")
    args = parser.parse_args()

    archive = ScanArchive(args.scan)
    print(f"scan: {Path(args.scan).name}   keyframes: {len(archive.timestamps)}")

    config = MeshConfig(
        voxel_length=args.voxel,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.max_depth,
        center_confidence=args.center_confidence,
        edge_confidence=args.edge_confidence,
        poisson_depth=args.poisson_depth,
        density_quantile=args.density_quantile,
    )
    print(f"method: {args.method}")
    if args.method == "poisson":
        mesh = reconstruct_mesh_poisson(archive, config)
    else:
        mesh = reconstruct_mesh(archive, config)
    print(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    if len(mesh.vertices):
        extent = mesh.get_axis_aligned_bounding_box().get_extent()
        print(f"bbox extent (m): x={extent[0]:.2f}  y={extent[1]:.2f}  z={extent[2]:.2f}")
        triangles = np.asarray(mesh.triangles)
        edges = np.sort(
            np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
            axis=1,
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        print(f"boundary edges (hole indicator): {int((counts == 1).sum())}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(args.save, mesh)
        print(f"saved mesh -> {args.save}")

    if args.view:
        from . import visualize

        if args.viewer == "rerun":
            visualize.show_mesh(mesh)
        else:
            visualize.show_mesh_o3d(mesh)

    archive.close()


if __name__ == "__main__":
    main()
