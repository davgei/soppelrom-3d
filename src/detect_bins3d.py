"""Detect bins in 3D: zero-shot 2D detection on all frames, back-projection, and merging
into distinct bin instances with position and size. These instances are also the proposal
boxes for 3D annotation review later."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from . import backproject, detection, render
from .loader import load_point_cloud
from .reconstruct import ReconstructionConfig


def estimate_floor_height(pcd: o3d.geometry.PointCloud, seed: int = 42) -> float | None:
    o3d.utility.random.seed(seed)
    model, inliers = pcd.segment_plane(0.03, 3, 500)
    normal = np.asarray(model[:3])
    normal /= np.linalg.norm(normal)
    if abs(normal[1]) < 0.85:
        return None
    return float(np.median(np.asarray(pcd.points)[inliers, 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and localize bins in 3D from a scan.")
    parser.add_argument("--scan", required=True, help="path to the raw capture .zip")
    parser.add_argument("--ply", default=None, help="optional saved .ply point cloud to reuse")
    parser.add_argument("--weights", default="yolov8s-worldv2.pt", help="YOLO-World weights")
    parser.add_argument("--conf", type=float, default=0.05, help="2D confidence threshold")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--eps", type=float, default=0.4, help="merge radius between detections (m)")
    parser.add_argument("--min-views", type=int, default=2, help="min frames a bin must be seen in")
    parser.add_argument("--prompts", nargs="*", default=None, help="override the text prompts")
    parser.add_argument("--save-json", default=None, help="write instances to this JSON file")
    parser.add_argument("--render-dir", default=None, help="write a top-down preview here")
    parser.add_argument("--view", action="store_true", help="open Open3D viewer (mesh/cloud + red boxes)")
    args = parser.parse_args()

    config = ReconstructionConfig(min_confidence=255, max_depth_m=5.0)
    pcd, archive, source = load_point_cloud(args.scan, args.ply, config)
    print(f"scan: {Path(args.scan).name}   cloud: {source} ({len(pcd.points)} points)")

    floor_height = estimate_floor_height(pcd)
    print(f"floor height: {floor_height:.3f} m" if floor_height is not None else "floor height: not found")

    model = detection.load_model(args.weights, args.prompts)
    print(f"running {args.weights} on every {args.stride} frame(s), conf >= {args.conf} ...")
    per_frame = detection.detect_scan(archive, model, conf=args.conf, stride=args.stride)
    n_detections = sum(len(d) for d in per_frame.values())
    n_frames_hit = sum(1 for d in per_frame.values() if d)
    print(f"2D detections: {n_detections} across {n_frames_hit} frames")

    instances = backproject.merge_detections(
        archive, per_frame, floor_height=floor_height, eps=args.eps, min_views=args.min_views
    )

    print(f"\n=== {len(instances)} bin instance(s) ===")
    for index, inst in enumerate(instances):
        length, height, width = inst.size
        print(
            f"#{index + 1}: footprint {length:.2f} x {width:.2f} m, height {height:.2f} m, "
            f"seen in {inst.n_views} frames, mean conf {inst.mean_confidence:.2f}, labels {inst.labels}"
        )

    if args.save_json:
        payload = [
            {
                "center": inst.center.tolist(),
                "size_lhw": inst.size.tolist(),
                "yaw_deg": inst.yaw_deg,
                "n_views": inst.n_views,
                "mean_confidence": inst.mean_confidence,
                "labels": inst.labels,
            }
            for inst in instances
        ]
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(payload, indent=2))
        print(f"instances -> {args.save_json}")

    if args.render_dir:
        out = Path(args.render_dir) / "bins_topdown.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        render.detections_topdown(pcd, instances, out)
        print(f"top-down preview -> {out}")

    if args.view:
        from . import visualize

        geometries: list = [pcd]
        for inst in instances:
            y_min = inst.center[1] - inst.size[1] / 2
            y_max = inst.center[1] + inst.size[1] / 2
            geometries.append(visualize.bin_box_lineset(inst.rect, y_min, y_max))
        visualize.show_scene_o3d(geometries)

    archive.close()


if __name__ == "__main__":
    main()
