"""Run zero-shot 2D bin detection over a scan's RGB frames and save annotated previews."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from . import detection
from .scan_io import ScanArchive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot bin detection on the RGB frames of a scan (YOLO-World)."
    )
    parser.add_argument("--scan", required=True, help="path to the raw capture .zip")
    parser.add_argument("--weights", default="yolov8s-worldv2.pt", help="YOLO-World weights")
    parser.add_argument("--conf", type=float, default=0.05, help="confidence threshold")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--out-dir", default="outputs/detections", help="where to save annotated frames")
    parser.add_argument("--prompts", nargs="*", default=None, help="override the text prompts")
    args = parser.parse_args()

    archive = ScanArchive(args.scan)
    model = detection.load_model(args.weights, args.prompts)
    print(f"scan: {Path(args.scan).name}   frames: {len(archive.timestamps)}   prompts: {model.names}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    frames_with_detections = 0
    total_detections = 0
    best_confidence = 0.0
    label_counts: dict[str, int] = {}

    for index, timestamp in enumerate(archive.timestamps):
        if index % args.stride:
            continue
        processed += 1
        detections, result = detection.detect_frame(model, archive.rgb(timestamp), args.conf)
        if not detections:
            continue
        frames_with_detections += 1
        total_detections += len(detections)
        for _, score, label in detections:
            best_confidence = max(best_confidence, score)
            label_counts[label] = label_counts.get(label, 0) + 1
        cv2.imwrite(str(out_dir / f"{timestamp}.jpg"), result.plot())

    print(f"\nframes processed:        {processed}")
    print(f"frames with detections:  {frames_with_detections}")
    print(f"total detections:        {total_detections}")
    print(f"best confidence:         {best_confidence:.3f}")
    print(f"detections per label:    {label_counts}")
    print(f"annotated frames ->      {out_dir}")
    archive.close()


if __name__ == "__main__":
    main()
