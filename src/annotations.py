"""Data model and storage for 3D bin-box annotations."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

BIN_TYPES: dict[str, tuple[float, float, float]] = {
    # default (length, height, width) in metres, from REG / EN 840 dimensions
    "2-hjuls dunk": (0.60, 1.15, 0.75),
    "4-hjuls container": (1.37, 1.25, 0.78),
    "molok": (1.30, 1.20, 1.30),
    "annet": (0.50, 1.00, 0.50),
}

STATUS_PROPOSED = "forslag"
STATUS_APPROVED = "godkjent"

BOX_EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]


@dataclass
class BinBox:
    center: list[float]        # (x, y, z) world, metres
    extent: list[float]        # (ex along local x, ey up, ez along local z)
    yaw_deg: float
    bin_type: str = "2-hjuls dunk"
    status: str = STATUS_PROPOSED
    source: str = "auto"
    n_views: int = 0
    confidence: float = 0.0

    def local_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw = math.radians(self.yaw_deg)
        ux = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
        uy = np.array([0.0, 1.0, 0.0])
        uz = np.array([-math.sin(yaw), 0.0, math.cos(yaw)])
        return ux, uy, uz

    def corners(self) -> np.ndarray:
        ux, uy, uz = self.local_axes()
        center = np.asarray(self.center)
        ex, ey, ez = self.extent
        points = []
        for a in (-1, 1):
            for b in (-1, 1):
                for d in (-1, 1):
                    points.append(center + a * ux * ex / 2 + b * uy * ey / 2 + d * uz * ez / 2)
        return np.array(points)

    def rotation_matrix(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "BinBox":
        return BinBox(**data)

    @staticmethod
    def from_min_area_rect(
        rect: tuple,
        y_min: float,
        y_max: float,
        bin_type: str = "2-hjuls dunk",
        n_views: int = 0,
        confidence: float = 0.0,
    ) -> "BinBox":
        points = cv2.boxPoints(rect)
        edge0 = points[1] - points[0]
        edge1 = points[2] - points[1]
        ex = float(np.linalg.norm(edge0))
        ez = float(np.linalg.norm(edge1))
        yaw = math.degrees(math.atan2(float(edge0[1]), float(edge0[0])))
        cx, cz = points.mean(axis=0)
        return BinBox(
            center=[float(cx), (y_min + y_max) / 2, float(cz)],
            extent=[ex, y_max - y_min, ez],
            yaw_deg=yaw,
            bin_type=bin_type,
            n_views=n_views,
            confidence=confidence,
        )


def guess_bin_type(extent: list[float]) -> str:
    length = max(extent[0], extent[2])
    width = min(extent[0], extent[2])
    if length > 1.1:
        return "4-hjuls container"
    if length > 0.9 and width > 0.9:
        return "molok"
    return "2-hjuls dunk"


def save_annotations(path: str | Path, scan_name: str, floor_height: float | None, boxes: list[BinBox]) -> None:
    payload = {
        "scan": scan_name,
        "floor_height": floor_height,
        "boxes": [box.to_dict() for box in boxes],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_annotations(path: str | Path) -> tuple[float | None, list[BinBox]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("floor_height"), [BinBox.from_dict(b) for b in data["boxes"]]
