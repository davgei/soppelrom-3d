"""2D bin detection on RGB frames: zero-shot YOLO-World, or our fine-tuned model when available."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .scan_io import ScanArchive

DEFAULT_PROMPTS = [
    "trash bin",
    "garbage bin",
    "waste container",
    "dustbin",
    "wheelie bin",
    "recycling bin",
    "dumpster",
]

LATEST_WEIGHTS = Path(__file__).resolve().parents[1] / "outputs" / "models" / "bins_latest.pt"


def default_weights() -> str:
    """Use the fine-tuned model once one has been trained, else the zero-shot world model."""
    return str(LATEST_WEIGHTS) if LATEST_WEIGHTS.exists() else "yolov8s-worldv2.pt"


@dataclass
class Detection2D:
    timestamp: int
    xyxy: np.ndarray  # [x1, y1, x2, y2] in RGB pixel coordinates
    confidence: float
    label: str


def load_model(weights: str | None = None, prompts: list[str] | None = None) -> YOLO:
    weights = weights or default_weights()
    model = YOLO(weights)
    if "world" in Path(weights).name.lower():
        model.set_classes(prompts or DEFAULT_PROMPTS)
    return model


def detect_frame(model: YOLO, rgb: np.ndarray, conf: float):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = model.predict(bgr, conf=conf, verbose=False)[0]
    detections: list[tuple[np.ndarray, float, str]] = []
    for box, score, cls in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
    ):
        detections.append((box, float(score), result.names[int(cls)]))
    return detections, result


def detect_scan(
    archive: ScanArchive, model: YOLO, conf: float = 0.05, stride: int = 1
) -> dict[int, list[Detection2D]]:
    per_frame: dict[int, list[Detection2D]] = {}
    for index, timestamp in enumerate(archive.timestamps):
        if index % stride:
            continue
        detections, _ = detect_frame(model, archive.rgb(timestamp), conf)
        per_frame[timestamp] = [
            Detection2D(timestamp, box, score, label) for box, score, label in detections
        ]
    return per_frame
