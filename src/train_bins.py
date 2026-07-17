"""Fine-tune a YOLO detector on the exported bin dataset and publish it as bins_latest.pt.

After training, prepare_scan (and thus the annotation tool's background worker)
automatically picks up outputs/models/bins_latest.pt for better proposals.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from .detection import LATEST_WEIGHTS
from .export_labels import DATASET_DIR
from .prepare_scan import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the exported bin labels.")
    parser.add_argument("--base", default="yolov8s.pt", help="base model to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data", default=str(DATASET_DIR / "dataset.yaml"))
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"dataset not found: {args.data} — run src.export_labels first")

    model = YOLO(args.base)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_ROOT / "outputs" / "models"),
        name="bins",
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    LATEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, LATEST_WEIGHTS)
    print(f"\nbest weights -> {best}")
    print(f"published    -> {LATEST_WEIGHTS} (used automatically by prepare_scan/annotation tool)")


if __name__ == "__main__":
    main()
