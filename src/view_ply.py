"""View Polycam's own .ply export for a scan, corrected to the pipeline's orientation.

Polycam exports are Z-up while everything here assumes Y-up (ARKit), so a raw view shows the room
lying on its side. They also live in PLY_DIR rather than next to the scan zip, precisely so that
loader.resolve_ply() cannot pick them up and silently feed a 90-degree-rotated cloud into the
analysis. This module is the deliberate way to look at them.

The up axis is found from the data (the dominant plane is the floor/ground) rather than assumed, so
an export in a different convention still comes out upright.

    .venv\\Scripts\\python.exe -m src.view_ply --list
    .venv\\Scripts\\python.exe -m src.view_ply --scan 80623_20260709T1108_KSL
    .venv\\Scripts\\python.exe -m src.view_ply --scan 80623_20260709T1108_KSL --compare
    .venv\\Scripts\\python.exe -m src.view_ply --scan 80623 --snapshot ut.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from .paths import CACHE_ROOT, PLY_DIR, RAW_DIR

UP = np.array([0.0, 1.0, 0.0])


def available() -> list[Path]:
    return sorted(PLY_DIR.glob("*.ply"))


def find_ply(scan: str) -> Path | None:
    """Exact stem match first, then a unique prefix match so a short scan number is enough."""
    exact = PLY_DIR / f"{scan}.ply"
    if exact.exists():
        return exact
    hits = [p for p in available() if p.stem.startswith(scan)]
    return hits[0] if len(hits) == 1 else None


def up_axis_rotation(points: np.ndarray, seed: int = 42) -> np.ndarray:
    """Rotation that brings the cloud's up axis onto +Y.

    The floor/ground is the largest plane in these scans, so its normal is 'up'. That is measured
    rather than assumed: Polycam happens to export Z-up today, but relying on a hardcoded axis swap
    would break silently the day that changes. Falls back to the shortest bounding-box axis (rooms
    are wider than they are tall) when no dominant plane is found."""
    if len(points) < 1000:
        return np.eye(3)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.utility.random.seed(seed)
    try:
        model, inliers = cloud.segment_plane(0.05, 3, 300)
        normal = np.asarray(model[:3], dtype=float)
        strong = len(inliers) >= 0.05 * len(points)
    except Exception:
        normal, strong = np.zeros(3), False
    if not strong or np.linalg.norm(normal) < 1e-9:
        normal = np.zeros(3)
        normal[int(np.argmin(np.ptp(points, axis=0)))] = 1.0
    normal = normal / np.linalg.norm(normal)
    if normal @ UP < 0:
        normal = -normal
    axis = np.cross(normal, UP)
    sin_a = float(np.linalg.norm(axis))
    if sin_a < 1e-8:
        return np.eye(3)                                    # already upright
    angle = float(np.arccos(np.clip(normal @ UP, -1.0, 1.0)))
    return o3d.geometry.get_rotation_matrix_from_axis_angle((axis / sin_a) * angle)


def load_upright(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if not len(points):
        raise SystemExit(f"tom punktsky: {path}")
    cloud.rotate(up_axis_rotation(points), center=(0.0, 0.0, 0.0))
    return cloud


def our_cloud(stem: str) -> o3d.geometry.PointCloud | None:
    """Our own reconstruction for the same scan, for comparison."""
    path = CACHE_ROOT / stem / "cloud.ply"
    return o3d.io.read_point_cloud(str(path)) if path.exists() else None


def _snapshot(geometries: list, out_path: Path, width: int = 1600, height: int = 1000) -> None:
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(visible=False, width=width, height=height)
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.09, 0.09, 0.11])
    viewer.poll_events()
    viewer.update_renderer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viewer.capture_screen_image(str(out_path), do_render=True)
    viewer.destroy_window()
    print(f"lagret {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vis Polycams egen .ply for et skann (rettet opp).")
    parser.add_argument("--scan", default=None, help="skann-stamme (eller entydig start av den)")
    parser.add_argument("--list", action="store_true", help="list .ply-filene som finnes")
    parser.add_argument("--compare", action="store_true",
                        help="vis vaar egen rekonstruksjon ved siden av (forskjoevet)")
    parser.add_argument("--snapshot", default=None, help="skriv PNG i stedet for aa aapne vindu")
    args = parser.parse_args()

    files = available()
    if args.list or not args.scan:
        print(f"{len(files)} .ply i {PLY_DIR}")
        for path in files:
            has_scan = (RAW_DIR / f"{path.stem}.zip").exists()
            size = path.stat().st_size / 1e6
            print(f"  {path.stem:38} {size:6.0f} MB  {'skann finnes' if has_scan else '(ingen zip)'}")
        if not args.scan:
            return

    path = find_ply(args.scan)
    if path is None:
        raise SystemExit(f"fant ingen entydig .ply for '{args.scan}' — bruk --list")

    cloud = load_upright(path)
    points = np.asarray(cloud.points)
    print(f"{path.name}: {len(points):,} punkter, farger: {cloud.has_colors()}")
    print(f"  utstrekning etter oppretting (X,Y,Z) = {np.ptp(points, axis=0).round(2)}  (Y = hoeyde)")

    geometries = [cloud]
    if args.compare:
        ours = our_cloud(path.stem)
        if ours is None:
            print("  (ingen egen rekonstruksjon aa sammenligne med)")
        else:
            our_points = np.asarray(ours.points)
            print(f"  vaar egen: {len(our_points):,} punkter "
                  f"({len(points) / max(len(our_points), 1):.2f}x saa mange i Polycams)")
            shift = float(np.ptp(points[:, 0])) * 1.15    # place them side by side
            ours.translate((shift, 0.0, 0.0))
            geometries.append(ours)

    if args.snapshot:
        _snapshot(geometries, Path(args.snapshot))
    else:
        o3d.visualization.draw_geometries(geometries, window_name=f"Polycam .ply — {path.stem}",
                                          width=1500, height=950)


if __name__ == "__main__":
    main()
