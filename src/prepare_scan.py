"""Precompute everything the annotation tool needs for a scan, cached on disk.

Also runs as the background worker: `--pending` prepares every unprepared scan in data/raw
sequentially, so the next scan is ready while the user annotates the current one.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import open3d as o3d

from . import annotations, backproject, detection
from .detect_bins3d import estimate_floor_height
from .reconstruct import ReconstructionConfig, reconstruct
from .reconstruct_mesh import MeshConfig, reconstruct_mesh_poisson
from .scan_io import ScanArchive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "outputs" / "cache"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANNOTATION_DIR = PROJECT_ROOT / "outputs" / "annotations"


def is_prepared(zip_path: Path, cache_root: Path = CACHE_ROOT) -> bool:
    return (cache_root / zip_path.stem / "done.flag").exists()


def is_annotated(zip_path: Path) -> bool:
    return (ANNOTATION_DIR / f"{zip_path.stem}.json").exists()


def prepare(
    zip_path: Path,
    cache_root: Path = CACHE_ROOT,
    weights: str | None = None,
    conf: float = 0.05,
    min_views: int = 2,
    force: bool = False,
) -> Path:
    cache = cache_root / zip_path.stem
    if is_prepared(zip_path, cache_root) and not force:
        return cache
    cache.mkdir(parents=True, exist_ok=True)

    archive = ScanArchive(zip_path)
    print(f"[prepare] {zip_path.name}: point cloud ...", flush=True)
    cloud = reconstruct(archive, ReconstructionConfig(min_confidence=255, max_depth_m=5.0))
    o3d.io.write_point_cloud(str(cache / "cloud.ply"), cloud)

    print(f"[prepare] {zip_path.name}: poisson mesh ...", flush=True)
    mesh = reconstruct_mesh_poisson(archive, MeshConfig())
    o3d.io.write_triangle_mesh(str(cache / "mesh_poisson.ply"), mesh)

    floor_height = estimate_floor_height(cloud)

    print(f"[prepare] {zip_path.name}: bin detection ...", flush=True)
    model = detection.load_model(weights)
    per_frame = detection.detect_scan(archive, model, conf=conf)
    instances = backproject.merge_detections(
        archive, per_frame, floor_height=floor_height, min_views=min_views
    )

    boxes: list[annotations.BinBox] = []
    for inst in instances:
        y_min = float(inst.center[1] - inst.size[1] / 2)
        y_max = float(inst.center[1] + inst.size[1] / 2)
        box = annotations.BinBox.from_min_area_rect(
            inst.rect, y_min, y_max, n_views=inst.n_views, confidence=inst.mean_confidence
        )
        majority = max(inst.labels, key=inst.labels.get) if inst.labels else None
        if majority in annotations.BIN_TYPES:  # fine-tuned model predicts our types directly
            box.bin_type = majority
        else:
            box.bin_type = annotations.guess_bin_type(box.extent)
        boxes.append(box)

    annotations.save_annotations(cache / "proposals.json", zip_path.name, floor_height, boxes)
    (cache / "done.flag").touch()
    print(f"[prepare] {zip_path.name}: done ({len(boxes)} proposals)", flush=True)
    archive.close()
    return cache


def watch(
    max_ready: int,
    weights: str | None = None,
    conf: float = 0.05,
    min_views: int = 2,
    poll_seconds: float = 5.0,
) -> None:
    """Keep at most `max_ready` prepared-but-unannotated scans ahead of the user.
    Never exits: keeps polling so zips dropped into data/raw later are picked up too."""
    while True:
        zips = sorted(RAW_DIR.glob("*.zip"))
        unprepared = [z for z in zips if not is_prepared(z)]
        ready_unannotated = [z for z in zips if is_prepared(z) and not is_annotated(z)]
        if not unprepared or len(ready_unannotated) >= max_ready:
            time.sleep(poll_seconds)
            continue
        prepare(unprepared[0], weights=weights, conf=conf, min_views=min_views)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare scan(s) for the annotation tool.")
    parser.add_argument("--scan", default=None, help="prepare a single scan zip")
    parser.add_argument("--pending", action="store_true", help="prepare all unprepared scans in data/raw")
    parser.add_argument("--watch", action="store_true",
                        help="keep a buffer of prepared scans ahead of annotation progress")
    parser.add_argument("--max-ready", type=int, default=5,
                        help="buffer size for --watch: prepared-but-unannotated scans")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    parser.add_argument("--weights", default=None,
                        help="detector weights (default: outputs/models/bins_latest.pt if trained, else yolov8s-worldv2)")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--min-views", type=int, default=2)
    args = parser.parse_args()

    if args.scan:
        prepare(Path(args.scan), weights=args.weights, conf=args.conf,
                min_views=args.min_views, force=args.force)
        return

    if args.watch:
        watch(args.max_ready, weights=args.weights, conf=args.conf, min_views=args.min_views)
        return

    if args.pending:
        zips = sorted(RAW_DIR.glob("*.zip"))
        todo = [z for z in zips if args.force or not is_prepared(z)]
        print(f"[prepare] {len(todo)} of {len(zips)} scans to prepare", flush=True)
        for zip_path in todo:
            prepare(zip_path, weights=args.weights, conf=args.conf,
                    min_views=args.min_views, force=args.force)
        return

    parser.error("use --scan <zip>, --pending or --watch")


if __name__ == "__main__":
    main()
