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


# Types whose real-world size is fixed and known — every bin of the type is identical, so a
# proposal can be snapped to the exact dimensions instead of the noisy measured footprint.
# "annet" is the unknown/other catch-all, so it is left at its measured size.
FIXED_SIZE_TYPES = tuple(name for name in BIN_TYPES if name != "annet")


def snap_box_to_type(box: "BinBox", floor_height: float | None) -> None:
    """Resize a box to its bin type's exact dimensions, keeping the footprint centre and yaw and
    re-seating the base on the floor. The canonical length goes on whichever footprint side was
    already longer, so the orientation is preserved. No-op for 'annet' (size unknown)."""
    if box.bin_type not in FIXED_SIZE_TYPES:
        return
    length, height, width = BIN_TYPES[box.bin_type]
    base = floor_height if floor_height is not None else (box.center[1] - box.extent[1] / 2)
    ex, _, ez = box.extent
    box.extent = [length, height, width] if ex >= ez else [width, height, length]
    box.center[1] = base + height / 2


MAX_OVERLAP = 0.15  # two kept boxes may overlap at most this fraction of the smaller footprint


def _footprint_corners_xz(box: "BinBox") -> np.ndarray:
    ux, _, uz = box.local_axes()
    center = np.asarray(box.center)
    ex, _, ez = box.extent
    corners = [center + a * ux * ex / 2 + d * uz * ez / 2 for a in (-1, 1) for d in (-1, 1)]
    return np.array([[c[0], c[2]] for c in corners], dtype=np.float32)


def footprint_overlap_ratio(a: "BinBox", b: "BinBox") -> float:
    """Overlap of two oriented floor footprints as a fraction of the SMALLER one (so a small bin
    fully inside a big one scores ~1.0, which pure IoU would understate)."""
    _, region = cv2.rotatedRectangleIntersection(
        cv2.minAreaRect(_footprint_corners_xz(a)), cv2.minAreaRect(_footprint_corners_xz(b))
    )
    if region is None:
        return 0.0
    inter = float(cv2.contourArea(region))
    area_a = a.extent[0] * a.extent[2]
    area_b = b.extent[0] * b.extent[2]
    return inter / max(min(area_a, area_b), 1e-6)


def remove_overlapping_boxes(boxes: list["BinBox"], max_overlap: float = MAX_OVERLAP) -> list["BinBox"]:
    """Greedy non-max suppression on floor footprints: keep boxes by descending confidence, drop
    any that overlap an already-kept box by more than `max_overlap`. Stops a 2-wheel bin from
    sitting inside a 4-wheel one (which can never really happen)."""
    kept: list[BinBox] = []
    for box in sorted(boxes, key=lambda b: -b.confidence):
        if all(footprint_overlap_ratio(box, other) <= max_overlap for other in kept):
            kept.append(box)
    return kept


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
