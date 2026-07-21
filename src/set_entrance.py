"""Let the user click the entrance/door in a scan, so bin placement uses it instead of the
auto-guess (start of the scan). The picked point is stored in the gravity-aligned frame that
placement uses, keyed by scan.

Usage:  .venv\\Scripts\\python.exe -m src.set_entrance --scan data\\raw\\<scan>.zip --ply outputs\\cache\\<scan>\\cloud.ply
Then SHIFT+CLICK the door/entrance in the window and close it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from .loader import load_point_cloud
from .reconstruct import ReconstructionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRANCE_DIR = PROJECT_ROOT / "outputs" / "entrances"


def load_entrances(scan_stem: str) -> list[tuple[float, float]]:
    path = ENTRANCE_DIR / f"{scan_stem}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if "entrances_xz" in data:
        return [(float(x), float(z)) for x, z in data["entrances_xz"]]
    if "entrance_xz" in data:  # backward compatibility with the single-point format
        return [(float(data["entrance_xz"][0]), float(data["entrance_xz"][1]))]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Klikk paa inngangen/doren i skannet.")
    parser.add_argument("--scan", required=True, help="path to the raw capture .zip")
    parser.add_argument("--ply", default=None, help="saved/cached .ply point cloud")
    args = parser.parse_args()

    pcd, archive, _ = load_point_cloud(
        args.scan, args.ply, ReconstructionConfig(min_confidence=255, max_depth_m=5.0)
    )

    print("SHIFT+KLIKK paa hver inngang/dor i vinduet (flere er ok), lukk vinduet naar du er ferdig.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Shift+klikk paa inngangene")
    vis.add_geometry(pcd)  # original frame, same as the annotation tool + saved boxes
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if not picked:
        print("ingen punkt valgt - inngang uendret")
        archive.close()
        return

    points_xyz = np.asarray(pcd.points)[picked]
    entrances = [[float(p[0]), float(p[2])] for p in points_xyz]
    ENTRANCE_DIR.mkdir(parents=True, exist_ok=True)
    out = ENTRANCE_DIR / f"{Path(args.scan).stem}.json"
    out.write_text(json.dumps({"entrances_xz": entrances}, indent=2), encoding="utf-8")
    print(f"{len(entrances)} inngang(er) lagret -> {out}")
    archive.close()


if __name__ == "__main__":
    main()
