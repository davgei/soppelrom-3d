"""Precompute everything the annotation tool needs for a scan, cached on disk.

Also runs as the background worker: `--pending` prepares every unprepared scan in data/raw
sequentially, so the next scan is ready while the user annotates the current one.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from . import annotations, backproject, binfit, detection, verify_bins
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
        verdict = binfit.score_candidate(inst.size, inst.mean_confidence, inst.n_views)
        if not verdict.keep:  # size+appearance fusion rejects noise (slivers, blobs, structure)
            continue
        y_min = float(inst.center[1] - inst.size[1] / 2)
        y_max = float(inst.center[1] + inst.size[1] / 2)
        box = annotations.BinBox.from_min_area_rect(
            inst.rect, y_min, y_max, n_views=inst.n_views, confidence=verdict.score
        )
        box.bin_type = verdict.bin_type
        box.status = annotations.STATUS_PROPOSED  # proposals are NEVER auto-approved — only the user approves
        boxes.append(box)

    # PointNet++ verification: look at the actual 3D points in each surviving box and drop the
    # confident non-bins the size gate let through. It only removes proposals, never approves them.
    # No-op when no verifier has been trained yet (models/verifier_latest.pt).
    verifier = verify_bins.load_verifier()
    if verifier is not None and boxes:
        cloud_points = np.asarray(cloud.points)
        probs = verifier.score_boxes(boxes, cloud_points, floor_height)
        kept: list[annotations.BinBox] = []
        for box, prob in zip(boxes, probs):
            if prob < verify_bins.DROP_BELOW:
                continue  # confident non-bin; the verifier only drops — it never approves
            kept.append(box)
        print(f"[prepare] {zip_path.name}: verifier kept {len(kept)}/{len(boxes)} proposals", flush=True)
        boxes = kept

    # 2-/4-wheel bins have a fixed real-world size, so use the exact dimensions instead of the
    # noisy measured footprint (position and orientation are kept), then drop overlapping boxes so
    # a small bin can't end up nested inside a larger one.
    for box in boxes:
        annotations.snap_box_to_type(box, floor_height)
    boxes = annotations.remove_overlapping_boxes(boxes)

    annotations.save_annotations(cache / "proposals.json", zip_path.name, floor_height, boxes)
    (cache / "done.flag").touch()
    print(f"[prepare] {zip_path.name}: done ({len(boxes)} proposals)", flush=True)
    archive.close()
    return cache


def refilter(cache_root: Path = CACHE_ROOT, weights: str | None = None) -> None:
    """Re-score already-cached proposals against the current verifier and rewrite the cleaned lists.

    Cheap backlog cleanup for scans prepared before the verifier existed (or before a retrain):
    reuses each scan's cloud.ply + proposals.json (no detection re-run), applies the same
    drop/downgrade rules as prepare(), and only rewrites scans that actually change. Skips
    annotated scans (their annotations override proposals). Note it can only *narrow* an existing
    list — a previously dropped box is gone; use --force to regenerate proposals from scratch.
    """
    verifier = verify_bins.load_verifier(weights)
    if verifier is None:
        print("[refilter] no verifier trained (models/verifier_latest.pt) — nothing to do", flush=True)
        return

    scans = [p.parent for p in sorted(cache_root.glob("*/proposals.json"))
             if not is_annotated(Path(p.parent.name))]
    total = d_type = d_verify = d_overlap = downgraded = changed = 0
    for cache in scans:
        floor_height, boxes = annotations.load_annotations(cache / "proposals.json")
        n_in = len(boxes)
        if n_in == 0:
            continue
        total += n_in
        changed_here = False

        # 1) drop types that are no longer generated (molok / annet)
        boxes = [b for b in boxes if b.bin_type in binfit.SCORE_TYPES]
        d_type += n_in - len(boxes)

        # 2) re-run the verifier against the current model + threshold
        cloud_file = cache / "cloud.ply"
        if verifier is not None and boxes and cloud_file.exists():
            cloud_points = np.asarray(o3d.io.read_point_cloud(str(cloud_file)).points)
            probs = verifier.score_boxes(boxes, cloud_points, floor_height)
            kept: list[annotations.BinBox] = []
            for box, prob in zip(boxes, probs):
                if prob < verify_bins.DROP_BELOW:
                    d_verify += 1
                    continue
                kept.append(box)
            boxes = kept

        # 3) snap to canonical size and drop overlaps
        for box in boxes:
            annotations.snap_box_to_type(box, floor_height)
        before_nms = len(boxes)
        boxes = annotations.remove_overlapping_boxes(boxes)
        d_overlap += before_nms - len(boxes)

        # 4) proposals must never carry an "approved" status — only the user approves
        for box in boxes:
            if box.status != annotations.STATUS_PROPOSED:
                box.status = annotations.STATUS_PROPOSED
                downgraded += 1
                changed_here = True

        if len(boxes) != n_in or changed_here:
            annotations.save_annotations(cache / "proposals.json", f"{cache.name}.zip", floor_height, boxes)
            changed += 1
    print(f"[refilter] {len(scans)} scan(s), {total} proposals -> dropped {d_type} (molok/annet), "
          f"{d_verify} (verifier), {d_overlap} (overlap); downgraded {downgraded}; "
          f"rewrote {changed} scan(s)", flush=True)


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
    parser.add_argument("--refilter", action="store_true",
                        help="re-score cached proposals against the current verifier and rewrite the "
                             "cleaned lists (no detection re-run)")
    parser.add_argument("--watch", action="store_true",
                        help="keep a buffer of prepared scans ahead of annotation progress")
    parser.add_argument("--max-ready", type=int, default=5,
                        help="buffer size for --watch: prepared-but-unannotated scans")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    parser.add_argument("--skip-annotated", action="store_true",
                        help="don't reprocess scans that already have saved annotations (protects your work)")
    parser.add_argument("--weights", default=None,
                        help="detector weights (default: outputs/models/bins_latest.pt if trained, else yolov8s-worldv2)")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--min-views", type=int, default=2)
    args = parser.parse_args()

    if args.scan:
        prepare(Path(args.scan), weights=args.weights, conf=args.conf,
                min_views=args.min_views, force=args.force)
        return

    if args.refilter:
        refilter(weights=args.weights)
        return

    if args.watch:
        watch(args.max_ready, weights=args.weights, conf=args.conf, min_views=args.min_views)
        return

    if args.pending:
        zips = sorted(RAW_DIR.glob("*.zip"))
        todo = [z for z in zips if args.force or not is_prepared(z)]
        if args.skip_annotated:
            skipped = [z for z in todo if is_annotated(z)]
            todo = [z for z in todo if not is_annotated(z)]
            print(f"[prepare] skipping {len(skipped)} annotated scan(s) — annotations untouched", flush=True)
        print(f"[prepare] {len(todo)} of {len(zips)} scans to prepare", flush=True)
        for zip_path in todo:
            prepare(zip_path, weights=args.weights, conf=args.conf,
                    min_views=args.min_views, force=args.force)
        return

    parser.error("use --scan <zip>, --pending, --refilter or --watch")


if __name__ == "__main__":
    main()
