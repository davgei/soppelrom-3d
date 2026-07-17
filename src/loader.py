"""Shared entry point: get a colored metric point cloud for a scan (PLY if present, else reconstruct)."""
from __future__ import annotations

from pathlib import Path

import open3d as o3d

from .reconstruct import ReconstructionConfig, reconstruct
from .scan_io import ScanArchive


def resolve_ply(zip_path: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    sibling = Path(zip_path).with_suffix(".ply")
    return sibling if sibling.exists() else None


def load_point_cloud(
    zip_path: str | Path,
    ply: str | None = None,
    config: ReconstructionConfig | None = None,
) -> tuple[o3d.geometry.PointCloud, ScanArchive, str]:
    archive = ScanArchive(zip_path)
    ply_path = resolve_ply(Path(zip_path), ply)
    if ply_path is not None:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        source = f"ply:{ply_path.name}"
    else:
        pcd = reconstruct(archive, config or ReconstructionConfig())
        source = "reconstructed"
    return pcd, archive, source
